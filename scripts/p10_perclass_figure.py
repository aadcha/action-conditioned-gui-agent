"""Regenerate the exploratory per-class figure on the paper's conventions.

The historical figure (`scripts/p7_compounding_plot.py`) averaged over ALL
available seeds — 5 for A/D-hook but only 3 for B/C/D-token — and read the
run JSONs' `per_class` block, whose denominators count parsed predictions
only. Both disagree with the paper: it reports matched seeds 42--44 and
scores parse failures at the sqrt(2) sentinel as misses. The old figure
therefore printed a click delta ~50% larger than the text beside it.

This script rebuilds the figure from per-example distances, grouping by the
class labels reconstructed from the pinned stream index, so the figure and
the prose agree by construction.

Run:  uv run python scripts/p10_perclass_figure.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P4 = ROOT / "results" / "phase4"
SCAN = ROOT / "results" / "phase8_reanalysis"
OUT = ROOT / "results" / "phase4" / "compounding_error_per_class.png"

SEEDS = (42, 43, 44)                       # matched across every variant
AWC = {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"}
CLS = {"tap": "click", "type": "type", "swipe_up": "scroll",
       "swipe_down": "scroll", "swipe_left": "scroll", "swipe_right": "scroll"}
PATTERNS = {
    "A (flat)": "variantA_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords.json",
    "B (aux loss)": "variantB_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_aux1.0.json",
    "C (hard routing)": "variantC_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords.json",
    "D-hook": "Dhook_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.0.json",
    "D-token": "Dtoken_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02.json",
}


def val_classes() -> list[str]:
    cands = sorted(SCAN.glob("stream_index_train_n*.json"))
    if not cands:
        raise SystemExit("stream index missing; run scan_aitw_stream first")
    recs = json.loads(cands[-1].read_text())["records"]
    awc = [r for r in recs if r["label"] in AWC]
    return [CLS[r["label"]] for r in awc[1200:1450]]


def per_class(pattern: str, cls: list[str]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for s in SEEDS:
        f = P4 / pattern.format(s=s)
        if not f.exists():
            continue
        d = np.asarray(json.loads(f.read_text())["final_val_metrics"]["per_example_dist"],
                       dtype=np.float64)
        if d.shape[0] != len(cls):
            continue
        for c in sorted(set(cls)):
            m = np.array([x == c for x in cls])
            acc.setdefault(c, []).append(float((d[m] <= 0.10).mean()))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def main() -> None:
    cls = val_classes()
    data = {nm: per_class(pat, cls) for nm, pat in PATTERNS.items()}
    classes = ["click", "scroll", "type"]
    counts = {c: sum(1 for x in cls if x == c) for c in classes}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(classes))
    w = 0.16
    for i, (nm, vals) in enumerate(data.items()):
        axes[0].bar(x + (i - 2) * w, [vals.get(c, 0.0) for c in classes], w, label=nm)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{c}\n(n={counts[c]})" for c in classes])
    axes[0].set_ylabel("hit@0.10")
    axes[0].set_title("Per-class grounding (exploratory splits)")
    axes[0].legend(fontsize=8)

    base = data["A (flat)"]
    for i, (nm, vals) in enumerate([(k, v) for k, v in data.items() if k != "A (flat)"]):
        axes[1].bar(x + (i - 1.5) * w, [vals.get(c, 0.0) - base.get(c, 0.0) for c in classes],
                    w, label=nm)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].set_ylabel(r"$\Delta$ hit@0.10 vs. A")
    axes[1].set_title("Conditioning advantage by class")
    axes[1].legend(fontsize=8)

    fig.suptitle("Exploratory per-class decomposition "
                 "(seeds 42-44, parse failures counted as misses)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"[p10-fig] wrote {OUT}")
    for nm, vals in data.items():
        print(f"  {nm:18s} " + "  ".join(f"{c}={vals.get(c, float('nan')):.3f}" for c in classes))


if __name__ == "__main__":
    main()
