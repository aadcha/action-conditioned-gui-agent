"""Phase 9 — build the frozen episode/goal-disjoint AITW split manifest.

Consumes the ordered filtered-stream index produced by
`modal run modal_app.py::scan_aitw_stream` (pulled to
results/phase8_reanalysis/stream_index_train_n*.json) and emits the frozen
manifest `data/manifests/aitw_frozen_v1.json`, which is committed to the repo
and shipped into every Modal container.

Split design (prespecified; see results/phase9_rerun/PRESPEC.md):

- POOL: the first 12,000 steps of the all_with_coords-filtered stream
  (labels tap / swipe_* / type), pinned to mirror revision
  5c0dc7139aaf714e69f8d8e4bd2ea2bc5a41700e.
- GROUPS: steps grouped by normalized goal string (lowercased,
  whitespace-collapsed, trailing punctuation stripped). Every episode has one
  goal, so goal groups are supersets of episodes; grouping at the goal level
  also removes the shared-task-description dependence that episode-level
  splitting alone misses.
- UNTOUCHED TEST: only goal groups whose EVERY step has filtered-stream index
  >= 5500 are test-eligible. Historical runs sliced the stream up to index
  5,250 (n_train=5000 + n_val=250), so the region beyond 5,500 was never
  loaded, trained on, evaluated on, or looked at. Test-eligible groups are
  assigned to test by salted hash; every step of a test group is quarantined
  (never trainable), whatever its label.
- TRAIN / VAL: remaining groups split by salted hash ~88/12. Training subsets
  are the first n_train steps of the train split in stream order (nested:
  n=300 is a prefix of n=500, etc). Val eval sets are the first n steps of
  the val split under each data_mix filter. All eval sets are FROZEN and
  identical across every training size and variant.

Run:  uv run python scripts/p9_build_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "manifests"
SCAN_DIR = ROOT / "results" / "phase8_reanalysis"

VERSION = "aitw_frozen_v1"
POOL_N = 12000                # all_with_coords-filtered steps in the pool
UNTOUCHED_FROM = 5500         # min filtered-stream index for test-eligible groups
TEST_FRAC_OF_ELIGIBLE = 45    # % of eligible goal groups (by salted hash) -> test
VAL_PCT = 12                  # % of remaining groups -> val
AWC_LABELS = {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"}


def normalize_goal(g: str) -> str:
    return " ".join(g.lower().split()).rstrip(".!?, ")


def salted_bucket(key: str, salt: str, mod: int = 100) -> int:
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest(), 16) % mod


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(len(ta | tb), 1)


def main() -> None:
    cands = sorted(SCAN_DIR.glob("stream_index_train_n*.json"))
    if not cands:
        raise SystemExit("stream index missing — run scan_aitw_stream first")
    scan = json.loads(cands[-1].read_text())
    records = scan["records"]

    awc = [r for r in records if r["label"] in AWC_LABELS]
    if len(awc) < POOL_N:
        raise SystemExit(f"scan too small: {len(awc)} awc steps < POOL_N={POOL_N}")
    pool = awc[:POOL_N]
    # awc-sequence position is the leakage-relevant index (historical slices
    # were taken in this coordinate system).
    for j, r in enumerate(pool):
        r["awc_index"] = j

    # ---- goal groups ----
    groups: dict[str, list[dict]] = {}
    for r in pool:
        groups.setdefault(normalize_goal(r["goal"]), []).append(r)

    # Consistency: an episode must never span two goal groups.
    ep_goal: dict[str, str] = {}
    for gk, rs in groups.items():
        for r in rs:
            if r["ep_id"] in ep_goal and ep_goal[r["ep_id"]] != gk:
                raise SystemExit(f"episode {r['ep_id']} spans two goal groups")
            ep_goal[r["ep_id"]] = gk

    # ---- assignment ----
    assign: dict[str, str] = {}
    n_eligible = 0
    for gk, rs in groups.items():
        min_idx = min(r["awc_index"] for r in rs)
        if min_idx >= UNTOUCHED_FROM:
            n_eligible += 1
            if salted_bucket(gk, f"{VERSION}:test") < TEST_FRAC_OF_ELIGIBLE:
                assign[gk] = "test"
                continue
        assign[gk] = "val" if salted_bucket(gk, f"{VERSION}:val") < VAL_PCT else "train"

    entries = []
    for r in pool:
        gk = normalize_goal(r["goal"])
        entries.append({
            "awc_index": r["awc_index"],
            "ep_id": r["ep_id"],
            "step_id": r["step_id"],
            "goal_key": gk,
            "label": r["label"],
            "cid": r["cid"],
            "touch_yx": r["touch_yx"],
            "lift_yx": r["lift_yx"],
            "split": assign[gk],
        })

    # ---- disjointness + duplicate checks ----
    split_eps = {s: set() for s in ("train", "val", "test")}
    split_goals = {s: set() for s in ("train", "val", "test")}
    for e in entries:
        split_eps[e["split"]].add(e["ep_id"])
        split_goals[e["split"]].add(e["goal_key"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (split_eps[a] & split_eps[b]), f"episode overlap {a}/{b}"
        assert not (split_goals[a] & split_goals[b]), f"goal overlap {a}/{b}"
    test_min = min(e["awc_index"] for e in entries if e["split"] == "test")
    assert test_min >= UNTOUCHED_FROM, "test group dips into the historically-touched region"

    # Near-duplicate goal report across train/test (token Jaccard >= 0.8).
    tr_goals, te_goals = sorted(split_goals["train"]), sorted(split_goals["test"])
    near_dups = []
    for tg in te_goals:
        for rg in tr_goals:
            if token_jaccard(tg, rg) >= 0.8:
                near_dups.append({"test_goal": tg, "train_goal": rg,
                                  "jaccard": round(token_jaccard(tg, rg), 3)})
    print(f"[p9] near-dup (Jaccard>=0.8) train<->test goal pairs: {len(near_dups)}")

    # ---- summary ----
    counts = {s: Counter(e["label"] for e in entries if e["split"] == s)
              for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        n = sum(counts[s].values())
        print(f"[p9] {s}: {n} steps, {len(split_goals[s])} goals, "
              f"{len(split_eps[s])} episodes, labels={dict(counts[s])}")

    body = {
        "version": VERSION,
        "repo": scan["repo"],
        "revision": scan["revision"],
        "aitw_split": scan["split"],
        "filter_labels": sorted(AWC_LABELS),
        "pool_n": POOL_N,
        "untouched_test_min_awc_index": UNTOUCHED_FROM,
        "assignment_rule": {
            "test": f"goal groups fully in awc_index>=[{UNTOUCHED_FROM}] with "
                    f"sha256('{VERSION}:test:'+goal_key)%100 < {TEST_FRAC_OF_ELIGIBLE}",
            "val": f"remaining groups with sha256('{VERSION}:val:'+goal_key)%100 < {VAL_PCT}",
            "train": "all other groups",
        },
        "counts": {s: {"steps": sum(counts[s].values()),
                       "goals": len(split_goals[s]),
                       "episodes": len(split_eps[s]),
                       "labels": dict(counts[s])} for s in ("train", "val", "test")},
        "near_dup_train_test_goal_pairs": near_dups,
        "entries": entries,
    }
    canonical = json.dumps(body["entries"], sort_keys=True).encode()
    body["entries_sha256"] = hashlib.sha256(canonical).hexdigest()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{VERSION}.json"
    out.write_text(json.dumps(body, indent=1))
    print(f"[p9] wrote {out}  entries_sha256={body['entries_sha256'][:16]}...")


if __name__ == "__main__":
    main()
