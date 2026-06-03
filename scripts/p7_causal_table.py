"""Phase 7.3 — aggregate the D-token embedding causal-use test.

Reads every Dtoken *_causal.json run (the --causal-eval runs), which each store a
`causal_eval` block with gold / wrong / zero evaluations on the SAME val set.

  gold  : true action id at the slot (normal inference)
  wrong : each example fed a *different* valid action id (cyclic permutation)
  zero  : action-embedding table zeroed for the eval

Since gold/wrong/zero are evaluated on identical val examples, their
per_example_dist arrays are paired -> a paired bootstrap over pooled distances
tests whether feeding the wrong / zeroed embedding significantly changes
grounding. gold >> wrong/zero => the learned embedding is causally used at
inference; gold ~= wrong ~= zero => the model ignores the slot.

Usage:
    uv run python scripts/p7_causal_table.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.bootstrap import paired_bootstrap, pool_distances_across_seeds  # noqa: E402

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"
SEEDS = [42, 43, 44]
FILE_TEMPLATE = "Dtoken_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02_causal.json"
CONDS = ["gold", "wrong", "zero"]
HITS = ["hit_at_005", "hit_at_010", "hit_at_025"]


def main() -> None:
    files = [PHASE4 / FILE_TEMPLATE.format(s=s) for s in SEEDS]
    missing = [p for p in files if not p.exists()]
    if missing:
        rel = ", ".join(str(p.relative_to(PHASE4.parent.parent)) for p in missing)
        raise FileNotFoundError(f"causal-use table is missing expected result file(s): {rel}")
    print(f"found {len(files)} causal run(s): {[Path(f).name for f in files]}\n")

    agg: dict[str, dict[str, list[float]]] = {c: {m: [] for m in HITS + ["mean_normalized_l2"]} for c in CONDS}
    dists: dict[str, list[np.ndarray]] = {c: [] for c in CONDS}
    emb_norms: list[float] = []
    for f in files:
        ce = json.loads(Path(f).read_text()).get("causal_eval")
        if not ce:
            continue
        emb_norms.append(ce.get("action_emb_norm", float("nan")))
        for c in CONDS:
            block = ce.get(c)
            if not block:
                continue
            for m in HITS + ["mean_normalized_l2"]:
                agg[c][m].append(block[m])
            dists[c].append(np.asarray(block["per_example_dist"], dtype=np.float64))

    mean = lambda xs: st.mean(xs) if xs else float("nan")
    sd = lambda xs: st.pstdev(xs) if len(xs) > 1 else 0.0

    print("=" * 70)
    print(f"D-TOKEN CAUSAL-USE TEST (mean over {len(files)} seeds; action_emb_norm={mean(emb_norms):.2f})")
    print("=" * 70)
    print(f"  {'cond':<7} {'hit@0.05':<11} {'hit@0.10':<11} {'hit@0.25':<11} {'mean L2':<9}")
    for c in CONDS:
        print(f"  {c:<7} "
              f"{mean(agg[c]['hit_at_005']):<11.3f} {mean(agg[c]['hit_at_010']):<11.3f} "
              f"{mean(agg[c]['hit_at_025']):<11.3f} {mean(agg[c]['mean_normalized_l2']):<9.3f}")

    # ---- paired bootstrap over pooled per-example distances ----
    print("\nPaired bootstrap on pooled per-example distances (higher hit = embedding used):")
    pooled = {c: pool_distances_across_seeds(dists[c]) for c in CONDS if dists[c]}
    out: dict = {"n_seeds": len(files), "action_emb_norm_mean": mean(emb_norms),
                 "means": {c: {m: mean(agg[c][m]) for m in HITS + ["mean_normalized_l2"]} for c in CONDS},
                 "paired": {}}
    if "gold" in pooled:
        for other in ["wrong", "zero"]:
            if other not in pooled or pooled[other].shape != pooled["gold"].shape:
                continue
            for metric in ["hit_at_010", "hit_at_025"]:
                # dist_a=other, dist_b=gold -> delta = hit(gold) - hit(other)
                r = paired_bootstrap(pooled[other], pooled["gold"], metric=metric, seed=0)
                sig = "ns" if (r.ci_low <= 0 <= r.ci_high) else "***" if r.p_value < 0.01 else "**" if r.p_value < 0.05 else "*"
                print(f"  gold − {other:<5} {metric}: {r.delta:+.3f}  "
                      f"95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  p={r.p_value:.3f}  {sig}")
                out["paired"][f"gold_minus_{other}_{metric}"] = {
                    "delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": r.p_value}

    (PHASE4 / "causal_use_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {PHASE4 / 'causal_use_summary.json'}")
    print("\nINTERPRETATION:")
    print("  gold >> wrong/zero (significant) => embedding IS causally used (refutation = 'used, not superior')")
    print("  gold ~= wrong ~= zero (ns)        => model IGNORES the slot      (refutation = 'path is inert')")


if __name__ == "__main__":
    main()
