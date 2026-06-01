"""Phase 5 — paired bootstrap A vs D-hook from per-example distances.

Reads Stage 2 run JSONs that contain `final_val_metrics.per_example_dist`
(added to the eval pipeline in src/train/stage2.py), pools the per-example
normalized-L2 distances across seeds for each variant, and runs a paired
bootstrap + permutation test for each grounding metric.

The val set is identical across seeds and across variants for a given
(data_mix, n_train, n_val) — the AITW stream is deterministic — so example i
refers to the same screenshot in every run. Pooling across seeds therefore
gives (seed, example) units that are paired between A and D-hook.

Usage:
    uv run python scripts/p5_paired_bootstrap.py --mix all_with_coords

Requires runs produced AFTER per_example_dist logging was added. Run JSONs
without it are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.bootstrap import paired_bootstrap, permutation_test, pool_distances_across_seeds

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"


def _load_per_example(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    fvm = d.get("final_val_metrics") or {}
    return fvm.get("per_example_dist")


def collect(prefix_glob: str) -> list[np.ndarray]:
    """Collect per-example distance arrays from all matching run JSONs."""
    arrays = []
    for p in sorted(PHASE4.glob(prefix_glob)):
        ped = _load_per_example(p)
        if ped is None:
            print(f"  [skip] {p.name} has no per_example_dist (pre-logging run)")
            continue
        arrays.append(np.asarray(ped, dtype=np.float64))
        print(f"  [ok]   {p.name}  n={len(ped)}")
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="all_with_coords")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    print(f"[bootstrap] variant A runs (mix={args.mix}):")
    a_arrays = collect(f"variantA_seed*_mix-{args.mix}.json")
    print(f"[bootstrap] variant D-hook runs (mix={args.mix}):")
    d_arrays = collect(f"Dhook_seed*_mix-{args.mix}_init0.0.json")

    if not a_arrays or not d_arrays:
        print("\n[bootstrap] insufficient per-example data. Re-run A and D-hook "
              "with the updated eval (per_example_dist logging) first.")
        return

    # Align: use the common val length (all should match for same mix/n).
    val_len = min(min(len(x) for x in a_arrays), min(len(x) for x in d_arrays))
    a_arrays = [x[:val_len] for x in a_arrays]
    d_arrays = [x[:val_len] for x in d_arrays]
    # For a paired test we need equal numbers of pooled units. Use the same
    # number of seeds on each side (pair seed-for-seed by pooling equal counts).
    n_seeds = min(len(a_arrays), len(d_arrays))
    a_pool = pool_distances_across_seeds(a_arrays[:n_seeds])
    d_pool = pool_distances_across_seeds(d_arrays[:n_seeds])
    print(f"\n[bootstrap] pooled {n_seeds} seeds x {val_len} examples = "
          f"{a_pool.shape[0]} paired units per variant")

    out = {"mix": args.mix, "n_seeds": n_seeds, "val_len": val_len,
           "n_paired_units": int(a_pool.shape[0]), "metrics": {}}

    print(f"\n{'metric':<20} {'A':>8} {'D-hook':>8} {'delta':>9} {'95% CI':>20} {'p(boot)':>9} {'p(perm)':>9}")
    print("-" * 90)
    for metric in ["hit_at_005", "hit_at_010", "hit_at_025", "mean_normalized_l2"]:
        res = paired_bootstrap(a_pool, d_pool, metric=metric, n_boot=args.n_boot, seed=0)
        p_perm = permutation_test(a_pool, d_pool, metric=metric, n_perm=args.n_boot, seed=0)
        better = "(lower=better)" if not res.higher_is_better else ""
        print(f"{metric:<20} {res.mean_a:>8.3f} {res.mean_b:>8.3f} {res.delta:>+9.4f} "
              f"[{res.ci_low:>+.4f},{res.ci_high:>+.4f}] {res.p_value:>9.4f} {p_perm:>9.4f} {better}")
        out["metrics"][metric] = {
            "mean_A": res.mean_a, "mean_Dhook": res.mean_b, "delta": res.delta,
            "ci95": [res.ci_low, res.ci_high], "p_bootstrap": res.p_value,
            "p_permutation": p_perm, "higher_is_better": res.higher_is_better,
        }

    out_path = PHASE4 / f"paired_bootstrap_{args.mix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[bootstrap] wrote {out_path}")


if __name__ == "__main__":
    main()
