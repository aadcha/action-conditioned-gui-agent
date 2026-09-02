"""Phase 8 — how much of the headline conditioning gain is "not emitting the
type sentinel on clicks"?

AITW stores non-touch (type) actions with touch point (-1, -1); the coordinate
serializer clamps that to the string "(0, 0)", so every model learns to emit
the screen origin for type events and their distance to the raw target is
exactly sqrt(2). A flat model that cannot tell clicks from type events may emit
the origin for a click too. The headline validation slice is deterministic, so
the gold targets logged by the qualitative job (results/phase8/qualitative_v2/
render.json, same n_train=1200 / n_val=250 slice) align index-for-index with
the per-example distances in every headline run. An exact "(0, 0)" prediction
on a click produces a distance equal to the target's norm, which identifies it.

For every headline run (A/B/C/D-hook/D-token x seeds 42-46) this script reports,
on click examples: hit@0.10, the fraction of predictions that are the sentinel,
and hit@0.10 among non-sentinel predictions. If the conditioning gain on clicks
survives after excluding sentinel emissions, it is not just noise partitioning.

Usage:
    uv run python scripts/p8_sentinel_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.p8_consolidate import HEADLINE_SEEDS, NAMES, load_run  # noqa: E402

P8 = ROOT / "results" / "phase8"
TOL = 2e-3  # distances are computed from integer-grid coordinates; exact origin gives |d - ||t||| == 0


def main() -> None:
    meta = json.loads((P8 / "qualitative_v2" / "render.json").read_text())
    rows = meta["all_rows"]
    assert len(rows) == 250, len(rows)
    gold = np.array([r["gold"] for r in rows], dtype=float)
    action = np.array([r["action"] for r in rows])
    norm = np.hypot(gold[:, 0], gold[:, 1])
    is_click = action == "click"
    is_scroll = action == "scroll"
    out = {"n_click": int(is_click.sum()), "n_scroll": int(is_scroll.sum()), "variants": {}}
    print(f"validation slice: {is_click.sum()} clicks, {is_scroll.sum()} scrolls, {(action=='type').sum()} type")
    print(f"{'variant':<20} {'seed':>4} {'click hit':>9} {'sentinel%':>9} {'hit|non-sent':>12} {'scroll sentinel%':>16}")
    for v in ("A", "B", "C", "Dhook", "Dtoken"):
        per_seed = []
        for s in HEADLINE_SEEDS:
            r = load_run(v, s, 1200, "all_with_coords")
            if r is None or r.dist is None or len(r.dist) != 250:
                continue
            d = r.dist
            sent = np.abs(d - norm) < TOL
            c_hit = float((d[is_click] <= 0.10).mean())
            c_sent = float(sent[is_click].mean())
            ns = is_click & ~sent
            c_hit_ns = float((d[ns] <= 0.10).mean()) if ns.any() else float("nan")
            s_sent = float(sent[is_scroll].mean())
            per_seed.append({"seed": s, "click_hit": c_hit, "click_sentinel_frac": c_sent,
                             "click_hit_non_sentinel": c_hit_ns, "scroll_sentinel_frac": s_sent})
            print(f"{NAMES[v]:<20} {s:>4} {c_hit:>9.3f} {c_sent:>9.1%} {c_hit_ns:>12.3f} {s_sent:>16.1%}")
        if per_seed:
            agg = {k: (float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)
                   for k in ("click_hit", "click_sentinel_frac", "click_hit_non_sentinel", "scroll_sentinel_frac")}
            out["variants"][v] = {"per_seed": per_seed, "mean_std": agg}
            print(f"{NAMES[v]:<20} {'mean':>4} {agg['click_hit'][0]:>9.3f} {agg['click_sentinel_frac'][0]:>9.1%} "
                  f"{agg['click_hit_non_sentinel'][0]:>12.3f} {agg['scroll_sentinel_frac'][0]:>16.1%}")
    (P8 / "sentinel_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {P8 / 'sentinel_analysis.json'}")


if __name__ == "__main__":
    main()
