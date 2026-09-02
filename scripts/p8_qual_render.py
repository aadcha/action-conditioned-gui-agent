"""Render the Phase 8 qualitative grounding figure locally from the Modal dump
(results/phase8/qualitative_v2/{render.json,data/}) so styling can be iterated
without a GPU.

Unlike scripts/p7_qual_render.py this keeps EVERY selected panel -- the panel
set is a fixed outcome spread (2 rescues, 1 hurt, 1 both-correct, 1 scroll,
1 degenerate type) chosen by modal_app.py::_stage2_qualitative_remote, and the
outcome base rates are printed in the title so the figure cannot read as a
highlight reel. Two layouts are written:
  qualitative_v2_1x6.png  all six panels (appendix)
  qualitative_v2_1x4.png  rescue / hurt / both-correct / scroll (main text)

Usage:
    uv run python scripts/p8_qual_render.py
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

ROOT = Path(__file__).resolve().parent.parent
QDIR = ROOT / "results" / "phase8" / "qualitative_v2"
DATA = QDIR / "data"
FIGS = ROOT / "neurips2026" / "figures"


def clamp(p, W, H):
    return None if p is None else (min(max(p[0], 0.0), 1.0) * W, min(max(p[1], 0.0), 1.0) * H)


def render(panels: list[dict], meta: dict, out: Path, title: str) -> None:
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    n = len(panels)
    imgs = [np.array(Image.open(DATA / p["file"]).convert("RGB")) for p in panels]
    aspect = max(im.shape[0] / im.shape[1] for im in imgs)  # H/W, portrait ~1.78
    pw = 2.1
    fig, axes = plt.subplots(1, n, figsize=(pw * n, pw * aspect + 0.9))
    axes = np.atleast_1d(axes)
    for ax, p, img in zip(axes, panels, imgs):
        H, W = img.shape[:2]
        ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.scatter([p["gold"][0] * W], [p["gold"][1] * H], s=300, marker="o", facecolors="none",
                   edgecolors="#11cc11", linewidths=3.0, path_effects=halo, zorder=6, label="ground truth")
        a = clamp(p["predA"], W, H); d = clamp(p["predD"], W, H)
        if a:
            ax.scatter([a[0]], [a[1]], s=220, marker="X", c="#ec1c1c", edgecolors="white", linewidths=1.4,
                       zorder=7, label="flat A" + (" (off-screen, clamped)" if p["distA"] > 1.0 else ""))
        if d:
            ax.scatter([d[0]], [d[1]], s=220, marker="X", c="#1f6fe0", edgecolors="white", linewidths=1.4,
                       zorder=8, label="D-hook (conditioned)")
        ax.set_title(f"{p['title']}\nA {p['distA']:.2f}  D {p['distD']:.2f}", fontsize=9, pad=4)
        ax.axis("off")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.075, wspace=0.04)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=9, frameon=False)
    fig.suptitle(title, y=0.985, fontsize=10)
    fig.savefig(out, dpi=350, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    meta = json.loads((QDIR / "render.json").read_text())
    panels = meta["panels"]
    # Exhaustive outcome partition of the clicks at r = 0.10 (the panel-selection
    # thresholds in modal_app.py are stricter and do not partition).
    clicks = [r for r in meta["all_rows"] if r["action"] == "click"]
    both = sum(r["distA"] <= 0.10 and r["distD"] <= 0.10 for r in clicks)
    rescue = sum(r["distA"] > 0.10 and r["distD"] <= 0.10 for r in clicks)
    hurt = sum(r["distA"] <= 0.10 and r["distD"] > 0.10 for r in clicks)
    miss = len(clicks) - both - rescue - hurt
    print(f"clicks={len(clicks)} both={both} rescue={rescue} hurt={hurt} both_miss={miss}")
    base = (f"AITW all_with_coords, n_train={meta['n_train']}, seed {meta.get('seed', 42)}: click hit@0.10 "
            f"A={meta['click_hit_A']:.2f} vs D-hook={meta['click_hit_D']:.2f}; of {len(clicks)} clicks, "
            f"{rescue} rescued (A miss, D hit), {hurt} hurt (A hit, D miss), {both} both correct, "
            f"{miss} both missed at r=0.10 (distances are normalized L2)")
    FIGS.mkdir(parents=True, exist_ok=True)
    render(panels, meta, FIGS / "qualitative_v2_1x6.png", base)
    want = ["click: conditioning rescues", "click: conditioning hurts", "click: both correct", "scroll: conditioning helps"]
    sel = []
    for w in want:
        for p in panels:
            if p["title"] == w and p not in sel:
                sel.append(p); break
    render(sel, meta, FIGS / "qualitative_v2_1x4.png", base)


if __name__ == "__main__":
    main()
