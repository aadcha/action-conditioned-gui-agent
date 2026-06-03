"""Phase 7 — compounding-error / per-action-type grounding analysis.

The project's motivating claim: a flat agent that does not separate *what
action* from *where to act* will ground poorly precisely on the action types
where the action identity is spatially decisive. This script tests that claim
directly by breaking the Stage-2 grounding metrics down BY ACTION TYPE for
every variant on the all_with_coords mix, averaged across all available seeds.

It reads the per-class breakdown already persisted in each run JSON
(`final_val_metrics.per_class`) — no new compute — and reports, per action
type, hit@0.10 / hit@0.25 / mean-L2 for A vs the conditioned variants, plus the
per-class conditioning advantage (B-A, Dhook-A, Dtoken-A).

Usage:
    uv run python scripts/p7_compounding_error.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"

# variant -> (filename glob without seed) ; {seed} is substituted
VARIANTS = {
    "A": "variantA_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords.json",
    "B": "variantB_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_aux1.0.json",
    "C": "variantC_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords.json",
    "Dhook": "Dhook_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.0.json",
    "Dtoken": "Dtoken_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02.json",
}
SEEDS = [42, 43, 44, 45, 46]
CLASSES = ["click", "scroll", "type"]
METRICS = ["hit_at_010", "hit_at_025", "mean_normalized_l2"]


def load_per_class(variant: str) -> dict[str, dict[str, list[float]]]:
    """Return {class: {metric: [values across seeds]}} for one variant."""
    acc: dict[str, dict[str, list[float]]] = {c: {m: [] for m in METRICS} for c in CLASSES}
    n_runs = 0
    for s in SEEDS:
        p = PHASE4 / VARIANTS[variant].format(s=s)
        if not p.exists():
            continue
        n_runs += 1
        pc = json.loads(p.read_text())["final_val_metrics"]["per_class"]
        for c in CLASSES:
            if c in pc:
                for m in METRICS:
                    acc[c][m].append(pc[c][m])
    acc["_n_runs"] = n_runs  # type: ignore[assignment]
    return acc


def mean(xs: list[float]) -> float | None:
    return st.mean(xs) if xs else None


def main() -> None:
    data = {v: load_per_class(v) for v in VARIANTS}

    print("=" * 78)
    print("PER-ACTION-TYPE GROUNDING on all_with_coords (n_train=1200), mean over seeds")
    print("=" * 78)
    for v in VARIANTS:
        n = data[v]["_n_runs"]
        print(f"\n[{v}]  ({n} seeds)")
        print(f"  {'class':<8} {'hit@0.10':<12} {'hit@0.25':<12} {'mean L2 (lower=better)':<14}")
        for c in CLASSES:
            h10 = mean(data[v][c]["hit_at_010"])
            h25 = mean(data[v][c]["hit_at_025"])
            l2 = mean(data[v][c]["mean_normalized_l2"])
            if h10 is None:
                print(f"  {c:<8} (absent)")
            else:
                print(f"  {c:<8} {h10:<12.3f} {h25:<12.3f} {l2:<14.3f}")

    # ---- per-class conditioning advantage vs A ----
    print("\n" + "=" * 78)
    print("CONDITIONING ADVANTAGE vs flat A, hit@0.10, BY action type")
    print("=" * 78)
    print(f"  {'class':<8} {'A':<9} {'B-A':<9} {'Dhook-A':<9} {'Dtoken-A':<9} {'C-A':<9}")
    summary: dict = {"per_class_hit010": {}, "advantage_vs_A_hit010": {}}
    for c in CLASSES:
        a = mean(data["A"][c]["hit_at_010"])
        row = {"A": a}
        cells = [f"{a:<9.3f}" if a is not None else f"{'--':<9}"]
        for v in ["B", "Dhook", "Dtoken", "C"]:
            mv = mean(data[v][c]["hit_at_010"])
            if a is not None and mv is not None:
                row[f"{v}-A"] = mv - a
                cells.append(f"{mv - a:<+9.3f}")
            else:
                cells.append(f"{'--':<9}")
        summary["advantage_vs_A_hit010"][c] = row
        print(f"  {c:<8} " + " ".join(cells))

    for v in VARIANTS:
        summary["per_class_hit010"][v] = {
            c: mean(data[v][c]["hit_at_010"]) for c in CLASSES
        }

    out = PHASE4 / "compounding_error_per_class.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")

    # ---- interpretation guardrail ----
    print("\n" + "-" * 78)
    a_type = mean(data["A"]["type"]["hit_at_010"]) or 0.0
    a_type_l2 = mean(data["A"]["type"]["mean_normalized_l2"]) or 0.0
    print(f"NOTE: flat-A 'type' hit@0.10 = {a_type:.3f}, mean L2 = {a_type_l2:.3f}. "
          "If L2 ~= 1.41 for ALL variants, 'type' targets are degenerate "
          "(ungroundable) and the real action-type signal is click-vs-scroll.")


if __name__ == "__main__":
    main()
