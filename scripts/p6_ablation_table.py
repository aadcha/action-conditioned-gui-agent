"""Phase 6 — full A/B/C/D ablation table + pairwise paired bootstrap.

Loads per-example normalized-L2 distances for all four variants on a given
data mix, pools across seeds (the AITW val set is identical across runs), and:
  - prints the 4-variant mean table (hit@r + mean L2)
  - runs paired bootstrap for each comparison the project plan's rubric needs:
      D-hook vs A   (does decoupling help?)
      B      vs A   (does the training signal alone help?)
      C      vs A   (does hard routing help?)
      D-hook vs B   (does inference-time conditioning beat training signal?)
      D-hook vs C   (does the learned embedding beat hard routing?)
  - writes results/phase4/ablation_ABCD_<mix>.json

Run after B and C land (A and D-hook per-example data already present).

Usage: uv run python scripts/p6_ablation_table.py --mix all_with_coords
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.bootstrap import paired_bootstrap, pool_distances_across_seeds

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"

# glob patterns per variant (per-example logging required)
VARIANTS = {
    "A": "variantA_seed*_mix-{mix}.json",
    "B": "variantB_seed*_mix-{mix}_aux*.json",
    "C": "variantC_seed*_mix-{mix}.json",
    "D-hook": "Dhook_seed*_mix-{mix}_init0.0.json",
}
METRICS = ["hit_at_005", "hit_at_010", "hit_at_025", "mean_normalized_l2"]


def load_pooled(glob_pat: str) -> np.ndarray | None:
    arrays = []
    for p in sorted(PHASE4.glob(glob_pat)):
        d = json.loads(p.read_text())
        ped = (d.get("final_val_metrics") or {}).get("per_example_dist")
        if ped is None:
            continue
        arrays.append(np.asarray(ped, dtype=np.float64))
    if not arrays:
        return None
    val_len = min(len(a) for a in arrays)
    arrays = [a[:val_len] for a in arrays]
    return pool_distances_across_seeds(arrays)


def metric_mean(pool: np.ndarray, metric: str) -> float:
    if metric == "mean_normalized_l2":
        return float(pool.mean())
    r = {"hit_at_005": 0.05, "hit_at_010": 0.10, "hit_at_025": 0.25}[metric]
    return float((pool <= r).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="all_with_coords")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    pools: dict[str, np.ndarray] = {}
    for name, pat in VARIANTS.items():
        pool = load_pooled(pat.format(mix=args.mix))
        if pool is None:
            print(f"[p6] {name}: NO per-example data yet (skipping)")
        else:
            pools[name] = pool
            print(f"[p6] {name}: pooled {pool.shape[0]} units")

    if "A" not in pools:
        print("[p6] need variant A per-example data; abort")
        return

    # Align all pools to a common length (same seeds x val_len ideally).
    common = min(p.shape[0] for p in pools.values())
    pools = {k: v[:common] for k, v in pools.items()}
    print(f"[p6] aligned all variants to {common} pooled units\n")

    # Mean table
    print(f"{'variant':<8} " + "  ".join(f"{m:>18}" for m in METRICS))
    print("-" * 84)
    table = {}
    for name, pool in pools.items():
        row = {m: metric_mean(pool, m) for m in METRICS}
        table[name] = row
        print(f"{name:<8} " + "  ".join(f"{row[m]:>18.3f}" for m in METRICS))

    # Pairwise bootstraps per the rubric
    comparisons = [("D-hook", "A"), ("B", "A"), ("C", "A"), ("D-hook", "B"), ("D-hook", "C")]
    out = {"mix": args.mix, "n_units": common, "means": table, "comparisons": {}}
    print("\npaired bootstrap (B_var - A_var); + = first better for hit@r, - = first better for L2")
    for b_var, a_var in comparisons:
        if b_var not in pools or a_var not in pools:
            continue
        key = f"{b_var}_vs_{a_var}"
        out["comparisons"][key] = {}
        print(f"\n  {b_var} vs {a_var}:")
        for metric in METRICS:
            res = paired_bootstrap(pools[a_var], pools[b_var], metric=metric, n_boot=args.n_boot, seed=0)
            sig = "***" if res.p_value < 0.01 else "** " if res.p_value < 0.05 else "   "
            print(f"    {metric:<20} delta={res.delta:>+.4f}  95%CI[{res.ci_low:>+.4f},{res.ci_high:>+.4f}]  p={res.p_value:.4f} {sig}")
            out["comparisons"][key][metric] = {
                "mean_a": res.mean_a, "mean_b": res.mean_b, "delta": res.delta,
                "ci95": [res.ci_low, res.ci_high], "p": res.p_value,
            }

    out_path = PHASE4 / f"ablation_ABCD_{args.mix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[p6] wrote {out_path}")


if __name__ == "__main__":
    main()
