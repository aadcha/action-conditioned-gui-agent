"""Plot the data-scaling curve: (B-A) and (D-A) deltas vs n_train."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"
# Phase 7 extends the curve into the low-data regime (300/500/800, 3 seeds).
SEEDS = {300: [42, 43, 44], 500: [42, 43, 44], 800: [42, 43, 44],
         1200: [42, 43, 44], 2500: [42], 5000: [42]}


def metric(prefix: str, n: int, suffix: str, key: str) -> float | None:
    xs = []
    for s in SEEDS[n]:
        p = PHASE4 / f"{prefix}_seed{s}_n{n}_ep2_lr2e-05_mix-all_with_coords{suffix}.json"
        if p.exists():
            xs.append(json.loads(p.read_text())["final_val_metrics"][key])
    return st.mean(xs) if xs else None


def main() -> None:
    ns = [300, 500, 800, 1200, 2500, 5000]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, label in [(axes[0], "hit_at_010", "hit@0.10"), (axes[1], "hit_at_025", "hit@0.25")]:
        A = [metric("variantA", n, "", key) for n in ns]
        B = [metric("variantB", n, "_aux1.0", key) for n in ns]
        D = [metric("Dhook", n, "_init0.0", key) for n in ns]
        bda = [b - a for a, b in zip(A, B)]
        dda = [d - a for a, d in zip(A, D)]
        ax.axhline(0, color="#888", lw=0.8, ls=":")
        ax.plot(ns, bda, "o-", color="#2a9d4a", label="B − A (aux loss)")
        ax.plot(ns, dda, "s-", color="#9d4edd", label="D − A (embedding)")
        for x, y in zip(ns, bda):
            ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, color="#2a9d4a")
        for x, y in zip(ns, dda):
            ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points", xytext=(0, -14), fontsize=8, color="#9d4edd")
        ax.set_xscale("log")
        ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("n_train (log scale)")
        ax.set_ylabel(f"conditioning advantage Δ {label}")
        ax.set_title(f"Δ {label} vs training size")
        ax.legend()
    fig.suptitle("Data-scaling: action-type conditioning advantage vs training size\n"
                 "(AITW all_with_coords; 300-1200 are 3-seed means, 2500/5000 single seed)", y=1.02)
    fig.tight_layout()
    out = PHASE4 / "scaling_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
