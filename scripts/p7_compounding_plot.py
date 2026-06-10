"""Phase 7 — per-action-type grounding figure (compounding-error localization).

Two panels:
  (left)  per-class hit@0.10 for A / B / D-hook / D-token on all_with_coords
          (shows the conditioning gain is concentrated in `click`; `type` is
          degenerate at 0.000 for every variant).
  (right) per-class conditioning advantage vs flat A (B-A, Dhook-A) — the gain
          lives in click, scroll is ~neutral, type contributes nothing.

Usage:
    uv run python scripts/p7_compounding_plot.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":19,"axes.titlesize":23,"axes.labelsize":21,"xtick.labelsize":17,"ytick.labelsize":17,"legend.fontsize":17,"legend.title_fontsize":18,"figure.titlesize":24})
import numpy as np

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"
PATTERNS = {
    "A": "variantA_seed*_n1200_ep2_lr2e-05_mix-all_with_coords.json",
    "B": "variantB_seed*_n1200_ep2_lr2e-05_mix-all_with_coords_aux1.0.json",
    "D-hook": "Dhook_seed*_n1200_ep2_lr2e-05_mix-all_with_coords_init0.0.json",
    "D-token": "Dtoken_seed*_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02.json",
}
CLASSES = ["click", "scroll", "type"]
COLORS = {"A": "#888888", "B": "#2a9d4a", "D-hook": "#9d4edd", "D-token": "#e07a1f"}


def per_class_mean(pattern: str, cls: str, metric: str = "hit_at_010"):
    xs = []
    for f in glob.glob(str(PHASE4 / pattern)):
        pc = json.loads(Path(f).read_text())["final_val_metrics"].get("per_class", {})
        if cls in pc:
            xs.append(pc[cls][metric])
    return (st.mean(xs) if xs else 0.0), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    means = {v: {c: per_class_mean(p, c)[0] for c in CLASSES} for v, p in PATTERNS.items()}
    stds = {v: {c: per_class_mean(p, c)[1] for c in CLASSES} for v, p in PATTERNS.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    x = np.arange(len(CLASSES))
    w = 0.2
    order = ["A", "B", "D-hook", "D-token"]
    for i, v in enumerate(order):
        vals = [means[v][c] for c in CLASSES]
        errs = [stds[v][c] for c in CLASSES]
        ax1.bar(x + (i - 1.5) * w, vals, w, label=v, color=COLORS[v], yerr=errs, capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n" for c in CLASSES])
    ax1.set_ylabel("hit@0.10")
    ax1.set_title("Grounding accuracy by action type (all_with_coords)")
    ax1.legend(title="variant")
    ax1.annotate("type targets degenerate\n(0.000 for every variant)",
                 xy=(2, 0.02), xytext=(1.45, 0.22), fontsize=16,
                 arrowprops=dict(arrowstyle="->", color="#b00"))

    # right: advantage vs A
    a = means["A"]
    for i, v in enumerate(["B", "D-hook", "D-token"]):
        adv = [means[v][c] - a[c] for c in CLASSES]
        ax2.bar(x + (i - 1) * w, adv, w, label=f"{v} − A", color=COLORS[v])
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES)
    ax2.set_ylabel("Δ hit@0.10 vs flat A")
    ax2.set_title("Conditioning advantage is concentrated in `click`")
    ax2.legend()
    for i, v in enumerate(["B", "D-hook", "D-token"]):
        for j, c in enumerate(CLASSES):
            d = means[v][c] - a[c]
            ax2.annotate(f"{d:+.2f}", (j + (i - 1) * w, d),
                         textcoords="offset points", xytext=(0, 3 if d >= 0 else -11),
                         ha="center", fontsize=15)

    fig.suptitle("Per-class decomposition "
                 "(AITW all_with_coords, n=1200)", y=1.02, fontsize=22)
    fig.tight_layout()
    out = PHASE4 / "compounding_error_per_class.png"
    fig.savefig(out, dpi=350, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
