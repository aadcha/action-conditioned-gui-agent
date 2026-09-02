"""Phase 8 — summarize the aggregate attention analysis (E5).

Reads results/phase8/attn_aggregate/<variant>/attn_aggregate_*.json (pull with
`modal run modal_app.py::pull_attn_aggregate`; a `_v2` file supersedes the
v1 file for the same variant) and writes a markdown report, a LaTeX table and
a figure comparing conditioning conditions (gold / wrong / wrong2 / zero) on
  target_frac@r   share of a probe token's image attention within r of the
                  gold point ("does conditioning move evidence to the target")
  image_mass      total image attention from that probe token
  hit@0.10        greedy-decoded grounding accuracy under the same conditioning

Probe positions (v2 files): last_prompt (token that predicts "("), pre_x (token
that predicts the first x digit), pre_y (token that predicts the first y digit).
v1 files only have last_prompt.

Usage:
    uv run python scripts/p8_attn_summary.py [--pos pre_x] [--layer 3q]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P8 = ROOT / "results" / "phase8"
ADIR = P8 / "attn_aggregate"
CCOL = {"gold": "#2a9d4a", "wrong": "#c9184a", "wrong2": "#e07b39", "zero": "#888888"}
CDESC = {"gold": "gold type", "wrong": "wrong: cyclic (click→type)", "wrong2": "wrong: click↔scroll", "zero": "zeroed embedding"}
TEXDESC = {"gold": "gold type", "wrong": "wrong (cyclic)", "wrong2": "wrong (click$\\leftrightarrow$scroll)", "zero": "zeroed"}
VNAME = {"Dhook": "D-hook (additive)", "Dtoken": "D-token (prepended)"}


def stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def load() -> dict[str, dict]:
    out = {}
    for p in sorted(ADIR.glob("*/attn_aggregate_*.json")):  # ..._v2.json sorts after the v1 file
        d = json.loads(p.read_text())
        d["_file"] = p.name
        out[d["summary"]["variant"]] = d
    return out


def pos_block(s: dict, pos: str, layer: str) -> dict:
    if "by_position" in s and pos in s["by_position"]:
        return s["by_position"][pos][layer]
    return s["by_layer"][layer]  # v1: last_prompt only


def fmt_d(r: dict | None) -> str:
    if not r:
        return "--"
    return f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {stars(r['p_boot'])}"


def tex_d(r: dict | None) -> str:
    if not r:
        return "--"
    s = stars(r["p_boot"]); s = "" if s in ("ns", "") else f"$^{{{s}}}$"
    return f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]{s}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="pre_y", help="probe position for the headline table/figure "
                    "(pre_y = token whose next prediction is the y coordinate; the most localized position)")
    ap.add_argument("--layer", default="3q")
    a = ap.parse_args()
    data = load()
    if not data:
        raise SystemExit(f"no attn_aggregate JSONs under {ADIR}; pull first")
    md = ["# Aggregate attention analysis (E5)", "",
          "Head-averaged attention from a probe token to the image tokens. `target_frac@r` = share of that token's image "
          "attention within r of the gold point; chance = share of image tokens within r. Paired bootstrap over probe "
          "examples, 10k resamples. Conditions: " + "; ".join(f"`{k}` = {v}" for k, v in CDESC.items()) + ".", ""]
    pos_name = {"pre_y": "y-predicting", "pre_x": "x-predicting", "last_prompt": "last prompt"}.get(a.pos, a.pos)
    tex = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           f"\\caption{{Where the model looks, and whether it matters. Share of the {pos_name} token's image attention "
           "within 0.10 of the gold target (3/4-depth layer; chance is 0.026) and greedy hit@0.10 under gold, wrong and zeroed action "
           "conditioning on the first 120 validation examples, one seed per model. $\\Delta$ = gold $-$ condition with 95\\% paired-bootstrap CI.}",
           "\\label{tab:attn}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llcccc}", "\\toprule",
           "Model & Conditioning & target attn.\\ & $\\Delta$ vs.\\ gold & hit@0.10 & $\\Delta$ vs.\\ gold \\\\", "\\midrule"]
    for v, d in data.items():
        s = d["summary"]; recs = d["records"]
        conds = s.get("conditions", ["gold", "wrong", "zero"])
        positions = s.get("probe_positions", ["last_prompt"])
        area = float(np.mean([r["conditions"]["gold"]["layers"][a.layer]["target_area_frac_0.10"] for r in recs]))
        md += [f"## {VNAME.get(v, v)} — seed {s['seed']}, n_train={s['n_train']}, {s['n_probe']} probe examples, "
               f"embedding norm {s['action_emb_norm']:.2f}, file `{d['_file']}`", "",
               f"wrong map {s.get('wrong_map')}; wrong2 map {s.get('wrong2_map', 'n/a')}. "
               f"Chance target_frac@0.10 (area share) = {area:.3f}.", ""]
        for pos in positions:
            bl = pos_block(s, pos, a.layer)
            md += [f"**Probe position `{pos}`, layer {a.layer}**", "",
                   "| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy |", "|---|---|---|---|---|"]
            for c in conds:
                md.append(f"| {c} | {bl['target_frac_0.10']['mean'][c]:.3f} | {bl['target_frac_0.25']['mean'][c]:.3f} | "
                          f"{bl['image_mass']['mean'][c]:.3f} | {bl['entropy']['mean'][c]:.2f} |")
            md += ["", "| contrast | Δ target_frac@0.10 | Δ target_frac@0.25 | Δ image_mass |", "|---|---|---|---|"]
            for c in conds:
                if c == "gold":
                    continue
                k = f"gold_minus_{c}"
                md.append(f"| gold − {c} | {fmt_d(bl['target_frac_0.10'].get(k))} | {fmt_d(bl['target_frac_0.25'].get(k))} | "
                          f"{fmt_d(bl['image_mass'].get(k))} |")
            md.append("")
        h = s["hits"]
        md += ["**Greedy decoding under each conditioning**", "", "| condition | hit@0.10 | hit@0.25 | mean dist | Δ hit@0.10 vs gold |", "|---|---|---|---|---|"]
        for c in conds:
            md.append(f"| {c} | {h['hit_010']['mean'][c]:.3f} | {h['hit_025']['mean'][c]:.3f} | {h['dist']['mean'][c]:.3f} | "
                      f"{'--' if c == 'gold' else fmt_d(h['hit_010'].get(f'gold_minus_{c}'))} |")
        md += ["", "Per gold class (target_frac@0.10 at " + a.pos + " if available else last_prompt / hit@0.10):", "",
               "| class | n | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
        for cls, blk in s["by_class"].items():
            key = "target_frac_0.10_pre_x" if (a.pos == "pre_x" and "target_frac_0.10_pre_x" in blk[conds[0]]) else "target_frac_0.10"
            md.append(f"| {cls} | {blk['n']} | " + " | ".join(f"{blk[c][key]:.3f} / {blk[c]['hit_010']:.3f}" for c in conds) + " |")
        md.append("")
        bl = pos_block(s, a.pos, a.layer)
        for i, c in enumerate(conds):
            k = f"gold_minus_{c}"
            tex.append(f"{VNAME.get(v, v) if i == 0 else ''} & {TEXDESC[c]} & {bl['target_frac_0.10']['mean'][c]:.3f} & "
                       f"{'' if c == 'gold' else tex_d(bl['target_frac_0.10'].get(k))} & {h['hit_010']['mean'][c]:.3f} & "
                       f"{'' if c == 'gold' else tex_d(h['hit_010'].get(k))} \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex += ["\\end{tabular}}", "\\end{table}"]
    (P8 / "ATTN_AGGREGATE.md").write_text("\n".join(md) + "\n")
    (P8 / "tables").mkdir(exist_ok=True)
    (P8 / "tables" / "attn.tex").write_text("\n".join(tex) + "\n")
    (ROOT / "neurips2026" / "tables").mkdir(exist_ok=True)
    (ROOT / "neurips2026" / "tables" / "attn.tex").write_text("\n".join(tex) + "\n")

    # figure: target_frac@0.10 (probe pos) and hit@0.10 per condition per variant
    vs = list(data)
    all_conds = [c for c in ("gold", "wrong", "wrong2", "zero") if any(c in data[v]["summary"].get("conditions", ["gold", "wrong", "zero"]) for v in vs)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(vs)); w = 0.8 / len(all_conds)
    for ax, key, title in ((axes[0], "target_frac_0.10", f"Image attention within r=0.10 of target ({a.pos} token, layer {a.layer})"),
                           (axes[1], "hit_010", "Greedy hit@0.10 under the same conditioning")):
        for j, c in enumerate(all_conds):
            means, errs = [], []
            for v in vs:
                s = data[v]["summary"]
                src = pos_block(s, a.pos, a.layer)[key] if key.startswith("target") else s["hits"][key]
                if c not in src["mean"]:
                    means.append(np.nan); errs.append(0); continue
                r = src.get(f"gold_minus_{c}") if c != "gold" else None
                means.append(src["mean"][c]); errs.append((r["ci95"][1] - r["ci95"][0]) / 2 if r else 0.0)
            ax.bar(x + (j - (len(all_conds) - 1) / 2) * w, means, w, yerr=errs, capsize=3, color=CCOL[c], label=CDESC[c], alpha=0.9)
        if key.startswith("target"):
            area = np.mean([np.mean([r["conditions"]["gold"]["layers"][a.layer]["target_area_frac_0.10"] for r in data[v]["records"]]) for v in vs])
            ax.axhline(area, ls=":", color="#444", label=f"chance (area share) {area:.3f}")
        ax.set_xticks(x); ax.set_xticklabels([VNAME.get(v, v) for v in vs]); ax.set_title(title, fontsize=11); ax.legend(fontsize=8)
    fig.suptitle("Attention to the target region vs. grounding accuracy under gold / wrong / zeroed action conditioning", fontsize=12)
    fig.tight_layout(); fig.savefig(P8 / "attn_aggregate.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    (ROOT / "neurips2026" / "figures").mkdir(exist_ok=True)
    import shutil; shutil.copy(P8 / "attn_aggregate.png", ROOT / "neurips2026" / "figures" / "attn_aggregate.png")
    print("\n".join(md)); print(f"wrote {P8}/ATTN_AGGREGATE.md, tables/attn.tex, attn_aggregate.png (+ mirrors in neurips2026/)")


if __name__ == "__main__":
    main()
