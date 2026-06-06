"""Render the qualitative grounding figure LOCALLY from the Modal raw dump.

The Modal job (stage2_qualitative) dumps each selected panel's raw screenshot +
coordinates to the stage1-cache Volume under qualitative/data + qualitative/
render.json. This script renders the final figure from that dump, so styling can
be iterated locally with no GPU/retrain.

Pull the dump first:
    modal volume get stage1-cache qualitative/data       results/phase4/qual_data
    modal volume get stage1-cache qualitative/render.json results/phase4/qual_render.json

Then:
    uv run python scripts/p7_qual_render.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PHASE4 = Path(__file__).resolve().parent.parent / "results" / "phase4"
DATA = PHASE4 / "qual_data"
if (DATA / "data").is_dir():  # `modal volume get` can nest the folder one level
    DATA = DATA / "data"
RENDER = PHASE4 / "qual_render.json"
OUT = PHASE4 / "qualitative_grounding.png"


def clamp(p, W, H):
    return None if p is None else (min(max(p[0], 0.0), 1.0) * W, min(max(p[1], 0.0), 1.0) * H)


def main() -> None:
    meta = json.loads(RENDER.read_text())
    # Keep only the informative panels (drop 'both correct' and 'type degenerate'),
    # laid out in a single wide row so the figure fits the page width.
    KEEP = [0, 1, 3, 4]
    NCOLS = 4
    allp = meta["panels"]
    panels = [allp[i] for i in KEEP if i < len(allp)]
    print(f"click hit: A={meta.get('click_hit_A'):.3f}  D-hook={meta.get('click_hit_D'):.3f}")
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    cols = NCOLS
    rowsn = max(1, (len(panels) + cols - 1) // cols)
    fig, axes = plt.subplots(rowsn, cols, figsize=(cols * 2.5, rowsn * 5.8))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, panels):
        img = np.array(Image.open(DATA / p["file"]).convert("RGB"))
        H, W = img.shape[:2]
        ax.imshow(img)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # lock axes to image extent (no whitespace blowup)
        ax.scatter([p["gold"][0] * W], [p["gold"][1] * H], s=320, marker="o", facecolors="none",
                   edgecolors="#11cc11", linewidths=3.0, path_effects=halo, zorder=6, label="ground truth")
        a = clamp(p["predA"], W, H)
        d = clamp(p["predD"], W, H)
        if a:
            ax.scatter([a[0]], [a[1]], s=240, marker="X", c="#ec1c1c", edgecolors="white",
                       linewidths=1.4, zorder=7, label="flat A")
        if d:
            ax.scatter([d[0]], [d[1]], s=240, marker="X", c="#1f6fe0", edgecolors="white",
                       linewidths=1.4, zorder=8, label="D-hook (conditioned)")
        ax.set_anchor("N")  # top-align image in its cell so titles never collide
        ax.set_title(f"{p['title']}\nA dist={p['distA']:.2f}   D dist={p['distD']:.2f}",
                     fontsize=9, pad=6)
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.105,
                        wspace=0.05, hspace=0.40)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=3, fontsize=11, frameon=False)
    fig.suptitle("Qualitative grounding: predicted vs. ground-truth point "
                 "(AITW all_with_coords, n_train=800)", y=0.965, fontsize=12)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
