"""Plot the corrected Phase 6 A/B/C/D ablation from ablation_ABCD_<mix>.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":19,"axes.titlesize":23,"axes.labelsize":21,"xtick.labelsize":17,"ytick.labelsize":17,"legend.fontsize":17,"legend.title_fontsize":18,"figure.titlesize":24})
import numpy as np

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"


def main(mix: str = "all_with_coords") -> None:
    data = json.loads((PHASE4 / f"ablation_ABCD_{mix}.json").read_text())
    means = data["means"]
    order = ["A", "B", "C", "D-hook"]
    labels = {"A": "A\nflat", "B": "B\naux loss", "C": "C\nhard route", "D-hook": "D\nembedding (ours)"}
    colors = {"A": "#888888", "B": "#2a9d4a", "C": "#c0852f", "D-hook": "#9d4edd"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))

    # Left: hit@r grouped bars
    ax = axes[0]
    metrics = ["hit_at_005", "hit_at_010", "hit_at_025"]
    mlabels = ["hit@0.05", "hit@0.10", "hit@0.25"]
    x = np.arange(len(metrics))
    w = 0.2
    for i, v in enumerate(order):
        vals = [means[v][m] for m in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, label=labels[v].replace("\n", " "), color=colors[v])
    ax.set_xticks(x); ax.set_xticklabels(mlabels)
    ax.set_ylabel("hit@r (higher better)")
    ax.set_title("Grounding hit@r by variant")
    ax.legend(fontsize=17)
    ax.axhline(np.pi * 0.10**2, ls=":", color="#444", lw=0.8)

    # Right: mean normalized L2 (lower better)
    ax = axes[1]
    vals = [means[v]["mean_normalized_l2"] for v in order]
    bars = ax.bar([labels[v] for v in order], vals, color=[colors[v] for v in order])
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", fontsize=17)
    ax.set_ylabel("mean normalized L2 (lower better)")
    ax.set_title("Mean grounding error")
    ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle("Stage 2 ablation (AITW all_with_coords): B ≈ D-hook > C > A",
                 fontsize=22, y=1.02)
    fig.tight_layout()
    out = PHASE4 / f"ablation_ABCD_{mix}.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
