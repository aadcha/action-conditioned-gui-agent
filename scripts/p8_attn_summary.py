"""Phase 8 — summarize the aggregate attention analysis (E5).

Reads results/phase8/attn_aggregate/<variant>/attn_aggregate_*.json (pull with
`modal run modal_app.py::pull_attn_aggregate`) and writes a markdown table, a
LaTeX table and a figure comparing gold / wrong / zero conditioning on
  target_frac@r   share of last-prompt-token image attention within r of the
                  gold point (the "does conditioning move evidence toward the
                  target" measure)
  image_mass      total image attention from that token
  hit@0.10        greedy-decoded grounding accuracy under the same conditioning

Usage:
    uv run python scripts/p8_attn_summary.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P8 = ROOT / "results" / "phase8"
ADIR = P8 / "attn_aggregate"
CONDS = ("gold", "wrong", "zero")
CCOL = {"gold": "#2a9d4a", "wrong": "#c9184a", "zero": "#888888"}
VNAME = {"Dhook": "D-hook (additive)", "Dtoken": "D-token (prepended)"}
LAYER = "3q"


def stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def load() -> dict[str, dict]:
    out = {}
    for p in sorted(ADIR.glob("*/attn_aggregate_*.json")):
        d = json.loads(p.read_text())
        out[d["summary"]["variant"]] = d
    return out


def ci_of(rec: dict, m: str, layer: str = LAYER) -> dict:
    return rec["summary"]["by_layer"][layer][m]


def main() -> None:
    data = load()
    if not data:
        raise SystemExit(f"no attn_aggregate JSONs under {ADIR}; pull first")
    md = ["# Aggregate attention analysis (E5)", "",
          f"Attention from the last prompt token to image tokens, head-averaged at the {LAYER} layer "
          "(3/4 depth; 1/2-depth and last-layer numbers are in the JSON). `target_frac@r` = share of image attention "
          "within r of the gold point; `area@r` = share of image tokens within r (chance level). "
          "Paired bootstrap over probe examples, 10k resamples.", ""]
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{Where does the model look? Share of last-prompt-token image attention that falls within 0.10 "
           "of the gold target (3/4-depth layer, head-averaged) and greedy hit@0.10 under gold, wrong and zeroed "
           "action conditioning, over the first 120 validation examples. $\\Delta$ = gold $-$ condition with 95\\% paired-bootstrap CI.}",
           "\\label{tab:attn}", "\\begin{tabular}{llcccc}", "\\toprule",
           "Model & Cond. & target frac@0.10 & image mass & hit@0.10 & $\\Delta$ target frac (gold $-$ cond.) \\\\", "\\midrule"]
    for v, d in data.items():
        s = d["summary"]; recs = d["records"]
        area = float(np.mean([r["conditions"]["gold"]["layers"][LAYER]["target_area_frac_0.10"] for r in recs]))
        md += [f"## {VNAME.get(v, v)} — seed {s['seed']}, n_train={s['n_train']}, {s['n_probe']} probe examples, "
               f"embedding norm {s['action_emb_norm']:.2f}, wrong map {s['wrong_map']}", "",
               f"Chance level (area share within r=0.10 of the target): {area:.3f}", "",
               "| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy | hit@0.10 | hit@0.25 | mean dist |",
               "|---|---|---|---|---|---|---|---|"]
        for c in CONDS:
            bl = s["by_layer"][LAYER]; h = s["hits"]
            md.append(f"| {c} | {bl['target_frac_0.10']['mean'][c]:.3f} | {bl['target_frac_0.25']['mean'][c]:.3f} | "
                      f"{bl['image_mass']['mean'][c]:.3f} | {bl['entropy']['mean'][c]:.2f} | {h['hit_010']['mean'][c]:.3f} | "
                      f"{h['hit_025']['mean'][c]:.3f} | {h['dist']['mean'][c]:.3f} |")
        md += ["", "| contrast | Δ target_frac@0.10 [CI] | Δ target_frac@0.25 [CI] | Δ image_mass [CI] | Δ hit@0.10 [CI] |", "|---|---|---|---|---|"]
        for k in ("gold_minus_wrong", "gold_minus_zero"):
            def f(m, src):
                r = src[m][k]
                return f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {stars(r['p_boot'])}" if r else "--"
            md.append(f"| {k.replace('_', ' ')} | {f('target_frac_0.10', s['by_layer'][LAYER])} | "
                      f"{f('target_frac_0.25', s['by_layer'][LAYER])} | {f('image_mass', s['by_layer'][LAYER])} | {f('hit_010', s['hits'])} |")
        md += ["", "Per gold class (target_frac@0.10 / hit@0.10):", "", "| class | n | gold | wrong | zero |", "|---|---|---|---|---|"]
        for cls, blk in s["by_class"].items():
            md.append(f"| {cls} | {blk['n']} | " + " | ".join(
                f"{blk[c]['target_frac_0.10']:.3f} / {blk[c]['hit_010']:.3f}" for c in CONDS) + " |")
        md.append("")
        for c in CONDS:
            bl = s["by_layer"][LAYER]; h = s["hits"]
            dk = "--" if c == "gold" else (lambda r: f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]" +
                                          ("" if stars(r["p_boot"]) in ("ns", "") else f"$^{{{stars(r['p_boot'])}}}$"))(
                bl["target_frac_0.10"][f"gold_minus_{c}"])
            tex.append(f"{VNAME.get(v, v) if c == 'gold' else ''} & {c} & {bl['target_frac_0.10']['mean'][c]:.3f} & "
                       f"{bl['image_mass']['mean'][c]:.3f} & {h['hit_010']['mean'][c]:.3f} & {dk} \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex += ["\\end{tabular}", "\\end{table}"]
    (P8 / "ATTN_AGGREGATE.md").write_text("\n".join(md) + "\n")
    (P8 / "tables").mkdir(exist_ok=True)
    (P8 / "tables" / "attn.tex").write_text("\n".join(tex) + "\n")

    # figure: target_frac@0.10 and hit@0.10, gold/wrong/zero, per variant
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    vs = list(data)
    x = np.arange(len(vs)); w = 0.26
    for ax, key, title in ((axes[0], "target_frac_0.10", "Image attention within r=0.10 of target"),
                           (axes[1], "hit_010", "Greedy hit@0.10")):
        for j, c in enumerate(CONDS):
            means, errs = [], []
            for v in vs:
                s = data[v]["summary"]
                src = s["by_layer"][LAYER][key] if key.startswith("target") else s["hits"][key]
                m = src["mean"][c]
                if c == "gold":
                    e = 0.0
                else:
                    r = src[f"gold_minus_{c}"]; e = (r["ci95"][1] - r["ci95"][0]) / 2 if r else 0.0
                means.append(m); errs.append(e)
            ax.bar(x + (j - 1) * w, means, w, yerr=errs, capsize=3, color=CCOL[c], label=c, alpha=0.9)
        if key.startswith("target"):
            area = np.mean([np.mean([r["conditions"]["gold"]["layers"][LAYER]["target_area_frac_0.10"] for r in data[v]["records"]]) for v in vs])
            ax.axhline(area, ls=":", color="#444", label=f"chance (area share) {area:.2f}")
        ax.set_xticks(x); ax.set_xticklabels([VNAME.get(v, v) for v in vs]); ax.set_title(title); ax.legend(fontsize=9)
    fig.suptitle("Conditioning moves evidence: attention to the target region and grounding accuracy under gold / wrong / zeroed action embeddings")
    fig.tight_layout(); fig.savefig(P8 / "attn_aggregate.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("\n".join(md)); print(f"wrote {P8}/ATTN_AGGREGATE.md, tables/attn.tex, attn_aggregate.png")


if __name__ == "__main__":
    main()
