"""Phase 8 — consolidate every Stage 2 run into the tables/figures the VLM4RWD
paper needs. Reads results/phase4/*.json (pull first with
`modal run modal_app.py::list_stage2_runs`) and writes results/phase8/.

Sections
  1. Headline ablation   all_with_coords, n=1200, seeds 42-46, A/B/C/D-hook/D-token
  2. Control             taps_and_swipes, n=1000, seeds 42-44, A/B/C/D-hook
  3. Scaling curve       n in {300..5000}, seeds 42-44, A/B/D-hook
  4. Mind2Web control    n=1000, seeds 42-44, A vs D-hook
  5. D-token causal use  gold / wrong / zero, every *_causal.json seed
  6. Per-class hit@0.10  headline cells, all seeds

Every hit@r / mean-L2 number is recomputed from `per_example_dist` when the run
logged it (parse failures count as misses via the sqrt(2) sentinel), so the
paired bootstrap and the table share one definition. Runs without per-example
logging fall back to `final_val_metrics` and are flagged in the output.

Degrades gracefully while runs are still landing: each cell reports how many
seeds it found. Idempotent -- re-run after every pull.

Usage:
    uv run python scripts/p8_consolidate.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.bootstrap import paired_bootstrap, permutation_test, pool_distances_across_seeds  # noqa: E402

P4 = ROOT / "results" / "phase4"
P8 = ROOT / "results" / "phase8"
TABLES = P8 / "tables"

TEMPLATES = {
    "A": "variantA_seed{s}_n{n}_ep2_lr2e-05_mix-{mix}.json",
    "B": "variantB_seed{s}_n{n}_ep2_lr2e-05_mix-{mix}_aux1.0.json",
    "C": "variantC_seed{s}_n{n}_ep2_lr2e-05_mix-{mix}.json",
    "Dhook": "Dhook_seed{s}_n{n}_ep2_lr2e-05_mix-{mix}_init0.0.json",
    "Dtoken": "Dtoken_seed{s}_n{n}_ep2_lr2e-05_mix-{mix}_init0.02.json",
}
CAUSAL_TEMPLATE = "Dtoken_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02_causal.json"
M2W_TEMPLATE = "m2w_{v}_seed{s}_n1000_ep2.json"
NAMES = {"A": "A (flat)", "B": "B (aux loss)", "C": "C (hard routing)",
         "Dhook": "D-hook (additive)", "Dtoken": "D-token (prepended)"}
TEX_NAMES = {"A": "A: flat", "B": "B: aux.\\ loss", "C": "C: hard routing",
             "Dhook": "D-hook: additive", "Dtoken": "D-token: prepended"}
HEADLINE_SEEDS = [42, 43, 44, 45, 46]
CURVE_SEEDS = [42, 43, 44]
CURVE_NS = [300, 500, 800, 1200, 2500, 5000]
METRICS = ["hit_at_005", "hit_at_010", "hit_at_025", "mean_normalized_l2"]
RADII = {"hit_at_005": 0.05, "hit_at_010": 0.10, "hit_at_025": 0.25}
N_BOOT = 10000
COLORS = {"A": "#3a6ea5", "B": "#2a9d4a", "C": "#e07b39", "Dhook": "#9d4edd", "Dtoken": "#c9184a"}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})


@dataclass
class Run:
    path: Path
    variant: str
    seed: int
    n_train: int
    mix: str
    fvm: dict
    dist: np.ndarray | None
    per_class: dict = field(default_factory=dict)

    @property
    def parse_rate(self) -> float:
        return float(self.fvm.get("parse_rate", 1.0))

    def metric(self, m: str) -> float:
        if self.dist is not None:
            if m == "mean_normalized_l2":
                return float(self.dist.mean())
            return float((self.dist <= RADII[m]).mean())
        return float(self.fvm[m])


def load_run(variant: str, seed: int, n: int, mix: str) -> Run | None:
    p = P4 / TEMPLATES[variant].format(s=seed, n=n, mix=mix)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    fvm = d.get("final_val_metrics") or {}
    if not fvm:
        return None
    ped = fvm.get("per_example_dist")
    return Run(p, variant, seed, int(d["n_train"]), mix, fvm,
               np.asarray(ped, dtype=np.float64) if ped else None, fvm.get("per_class") or {})


def cell(runs: list[Run], m: str) -> dict:
    vals = [r.metric(m) for r in runs]
    return {"mean": float(np.mean(vals)) if vals else math.nan,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals), "seeds": [r.seed for r in runs], "values": vals}


def paired(base: list[Run], other: list[Run], m: str) -> dict | None:
    """other - base over pooled (seed, example) units; seeds matched by id."""
    bs = {r.seed: r for r in base if r.dist is not None}
    os_ = {r.seed: r for r in other if r.dist is not None}
    seeds = sorted(set(bs) & set(os_))
    if not seeds:
        return None
    L = min(min(len(bs[s].dist) for s in seeds), min(len(os_[s].dist) for s in seeds))
    a = pool_distances_across_seeds([bs[s].dist[:L] for s in seeds])
    b = pool_distances_across_seeds([os_[s].dist[:L] for s in seeds])
    r = paired_bootstrap(a, b, metric=m, n_boot=N_BOOT, seed=0)
    p_perm = permutation_test(a, b, metric=m, n_perm=N_BOOT, seed=0)
    return {"delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_boot": r.p_value,
            "p_perm": p_perm, "n_units": int(a.shape[0]), "seeds": seeds,
            "higher_is_better": r.higher_is_better}


def stars(p: float | None) -> str:
    if p is None or math.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fmt_ms(c: dict, digits: int = 3) -> str:
    if math.isnan(c["mean"]):
        return "--"
    return f"{c['mean']:.{digits}f} ± {c['std']:.{digits}f}" if c["n_seeds"] > 1 else f"{c['mean']:.{digits}f}"


def tex_ms(c: dict, digits: int = 3) -> str:
    if math.isnan(c["mean"]):
        return "--"
    return f"{c['mean']:.{digits}f} $\\pm$ {c['std']:.{digits}f}" if c["n_seeds"] > 1 else f"{c['mean']:.{digits}f}"


def fmt_delta(d: dict | None) -> str:
    if d is None:
        return "--"
    return f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] {stars(d['p_boot'])}"


def tex_delta(d: dict | None) -> str:
    if d is None:
        return "--"
    s = stars(d["p_boot"])
    s = "" if s == "ns" else f"$^{{{s.replace('*', '*')}}}$"
    return f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{s}"


# ---------------------------------------------------------------------------
def section_variants(mix: str, n: int, seeds: list[int], variants: list[str], title: str) -> dict:
    runs = {v: [r for s in seeds if (r := load_run(v, s, n, mix))] for v in variants}
    out = {"mix": mix, "n_train": n, "seeds_requested": seeds, "cells": {}, "deltas_vs_A": {}, "deltas_vs_B": {},
           "parse_rates": {}, "no_per_example": []}
    for v in variants:
        out["cells"][v] = {m: cell(runs[v], m) for m in METRICS}
        out["parse_rates"][v] = [r.parse_rate for r in runs[v]]
        out["no_per_example"] += [r.path.name for r in runs[v] if r.dist is None]
        if v != "A":
            out["deltas_vs_A"][v] = {m: paired(runs["A"], runs[v], m) for m in METRICS}
        if v not in ("A", "B") and "B" in runs:
            out["deltas_vs_B"][v] = {m: paired(runs["B"], runs[v], m) for m in METRICS}
    out["per_class"] = {}
    for v in variants:
        pc = defaultdict(list)
        for r in runs[v]:
            for cls, blk in r.per_class.items():
                pc[cls].append(blk.get("hit_at_010", math.nan))
        out["per_class"][v] = {cls: {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                                     "n_seeds": len(x)} for cls, x in pc.items()}
    out["title"] = title
    out["_runs"] = runs
    return out


def md_variants(sec: dict) -> str:
    L = [f"### {sec['title']}", "",
         f"Data mix `{sec['mix']}`, n_train={sec['n_train']}, seeds requested {sec['seeds_requested']}. "
         "Mean ± sample std over seeds; every metric recomputed from per-example distances "
         "(parse failures = miss). Paired bootstrap pools (seed, example) units with seeds matched by id; "
         f"{N_BOOT} resamples; 95% percentile CI; two-sided bootstrap p (permutation p agrees unless noted).", "",
         "| variant | seeds | hit@0.05 | hit@0.10 | hit@0.25 | mean L2 ↓ | parse |", "|---|---|---|---|---|---|---|"]
    for v, cells in sec["cells"].items():
        pr = sec["parse_rates"][v]
        L.append(f"| {NAMES[v]} | {cells['hit_at_010']['n_seeds']} | {fmt_ms(cells['hit_at_005'])} | "
                 f"{fmt_ms(cells['hit_at_010'])} | {fmt_ms(cells['hit_at_025'])} | {fmt_ms(cells['mean_normalized_l2'])} | "
                 f"{min(pr):.3f}–{max(pr):.3f} |" if pr else f"| {NAMES[v]} | 0 | -- | -- | -- | -- | -- |")
    L += ["", "Paired deltas vs A (Δ [95% CI] sig):", "",
          "| variant | hit@0.05 | hit@0.10 | hit@0.25 | mean L2 (lower better) | units |", "|---|---|---|---|---|---|"]
    for v, ds in sec["deltas_vs_A"].items():
        u = next((d["n_units"] for d in ds.values() if d), "--")
        L.append(f"| {NAMES[v]} − A | " + " | ".join(fmt_delta(ds[m]) for m in METRICS) + f" | {u} |")
    if sec["deltas_vs_B"]:
        L += ["", "Paired deltas vs B:", "", "| variant | hit@0.05 | hit@0.10 | hit@0.25 | mean L2 | units |", "|---|---|---|---|---|---|"]
        for v, ds in sec["deltas_vs_B"].items():
            u = next((d["n_units"] for d in ds.values() if d), "--")
            L.append(f"| {NAMES[v]} − B | " + " | ".join(fmt_delta(ds[m]) for m in METRICS) + f" | {u} |")
    classes = sorted({c for pcs in sec["per_class"].values() for c in pcs})
    if classes:
        L += ["", "Per-class hit@0.10 (mean ± std over seeds):", "",
              "| variant | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for v, pcs in sec["per_class"].items():
            L.append(f"| {NAMES[v]} | " + " | ".join(fmt_ms(pcs[c]) if c in pcs else "--" for c in classes) + " |")
    if sec["no_per_example"]:
        L += ["", "Runs WITHOUT per-example logging (fell back to summary metrics): " + ", ".join(sec["no_per_example"])]
    return "\n".join(L) + "\n"


def tex_variants(sec: dict, label: str, caption: str) -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}", "\\toprule",
         "Variant & hit@0.05 & hit@0.10 & hit@0.25 & mean L2 $\\downarrow$ & $\\Delta$hit@0.10 vs.\\ A \\\\", "\\midrule"]
    for v, cells in sec["cells"].items():
        d = sec["deltas_vs_A"].get(v, {}).get("hit_at_010") if v != "A" else None
        L.append(f"{TEX_NAMES[v]} & {tex_ms(cells['hit_at_005'])} & {tex_ms(cells['hit_at_010'])} & "
                 f"{tex_ms(cells['hit_at_025'])} & {tex_ms(cells['mean_normalized_l2'])} & "
                 f"{tex_delta(d) if d else ''} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def section_scaling() -> dict:
    out = {"ns": CURVE_NS, "seeds": CURVE_SEEDS, "cells": {}, "deltas": {}}
    for n in CURVE_NS:
        runs = {v: [r for s in CURVE_SEEDS if (r := load_run(v, s, n, "all_with_coords"))] for v in ("A", "B", "Dhook")}
        out["cells"][n] = {v: {m: cell(runs[v], m) for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2")} for v in runs}
        out["deltas"][n] = {v: {m: paired(runs["A"], runs[v], m) for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2")}
                            for v in ("B", "Dhook")}
    return out


def md_scaling(sec: dict) -> str:
    L = ["### Data-scaling curve (all_with_coords, seeds 42–44 at every n)", "",
         "| n_train | A hit@0.10 | B hit@0.10 | D-hook hit@0.10 | B − A [CI] | D-hook − A [CI] | A hit@0.25 | B hit@0.25 | D-hook hit@0.25 | B − A @0.25 | D-hook − A @0.25 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in sec["ns"]:
        c = sec["cells"][n]; d = sec["deltas"][n]
        L.append(f"| {n} | {fmt_ms(c['A']['hit_at_010'])} ({c['A']['hit_at_010']['n_seeds']}s) | {fmt_ms(c['B']['hit_at_010'])} | "
                 f"{fmt_ms(c['Dhook']['hit_at_010'])} | {fmt_delta(d['B']['hit_at_010'])} | {fmt_delta(d['Dhook']['hit_at_010'])} | "
                 f"{fmt_ms(c['A']['hit_at_025'])} | {fmt_ms(c['B']['hit_at_025'])} | {fmt_ms(c['Dhook']['hit_at_025'])} | "
                 f"{fmt_delta(d['B']['hit_at_025'])} | {fmt_delta(d['Dhook']['hit_at_025'])} |")
    L += ["", "Note: the val slice is `examples[n_train:n_train+250]`, so absolute numbers are NOT comparable across n; "
          "only the within-n deltas are. Seeds (s) shown for A; B and D-hook have the same seed count unless a run is missing."]
    return "\n".join(L) + "\n"


def tex_scaling(sec: dict) -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Conditioning advantage vs.\\ training-set size (AITW \\texttt{all\\_with\\_coords}, 3 seeds per cell, "
         "paired bootstrap over pooled (seed, example) units). Absolute values are not comparable across rows because the "
         "validation slice moves with $n$.}", "\\label{tab:scaling}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{rccccc}", "\\toprule",
         "$n_{\\text{train}}$ & A hit@0.10 & B $-$ A & D-hook $-$ A & B $-$ A (hit@0.25) & D-hook $-$ A (hit@0.25) \\\\", "\\midrule"]
    for n in sec["ns"]:
        c = sec["cells"][n]; d = sec["deltas"][n]
        L.append(f"{n} & {tex_ms(c['A']['hit_at_010'])} & {tex_delta(d['B']['hit_at_010'])} & {tex_delta(d['Dhook']['hit_at_010'])} & "
                 f"{tex_delta(d['B']['hit_at_025'])} & {tex_delta(d['Dhook']['hit_at_025'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


def plot_scaling(sec: dict, out: Path) -> None:
    ns = [n for n in sec["ns"] if sec["cells"][n]["A"]["hit_at_010"]["n_seeds"] > 0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for v in ("A", "B", "Dhook"):
        m = [sec["cells"][n][v]["hit_at_010"]["mean"] for n in ns]
        s = [sec["cells"][n][v]["hit_at_010"]["std"] for n in ns]
        ax.errorbar(ns, m, yerr=s, marker="o", capsize=4, color=COLORS[v], label=NAMES[v])
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n_train (log scale)"); ax.set_ylabel("hit@0.10 (mean ± std, 3 seeds)")
    ax.set_title("Absolute grounding accuracy"); ax.legend()
    for ax, m, lab in ((axes[1], "hit_at_010", "hit@0.10"), (axes[2], "hit_at_025", "hit@0.25")):
        ax.axhline(0, color="#888", lw=0.8, ls=":")
        for v, off in (("B", -0.03), ("Dhook", 0.03)):
            xs, ys, lo, hi = [], [], [], []
            for n in ns:
                d = sec["deltas"][n][v][m]
                if d is None:
                    continue
                xs.append(n * (1 + off)); ys.append(d["delta"]); lo.append(d["delta"] - d["ci_low"]); hi.append(d["ci_high"] - d["delta"])
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="s" if v == "Dhook" else "o", capsize=4, ls="-",
                        color=COLORS[v], label=f"{NAMES[v]} − A")
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("n_train (log scale)"); ax.set_ylabel(f"Δ {lab} vs. A (95% paired-bootstrap CI)")
        ax.set_title(f"Conditioning advantage, {lab}"); ax.legend()
    counts = sorted({sec["cells"][n][v]["hit_at_010"]["n_seeds"] for n in ns for v in ("A", "B", "Dhook")})
    seeds_txt = f"{counts[0]} seeds at every n" if len(counts) == 1 else f"{counts[0]}–{counts[-1]} seeds per cell"
    fig.suptitle(f"Data scaling on AITW all_with_coords ({seeds_txt})", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_headline_vs_control(head: dict, ctrl: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [5, 4]})
    for ax, sec, title in ((axes[0], head, "all_with_coords (action type spatially informative)"),
                           (axes[1], ctrl, "taps_and_swipes (control)")):
        vs = [v for v in sec["cells"] if sec["cells"][v]["hit_at_010"]["n_seeds"] > 0]
        x = np.arange(len(vs))
        means = [sec["cells"][v]["hit_at_010"]["mean"] for v in vs]
        stds = [sec["cells"][v]["hit_at_010"]["std"] for v in vs]
        ax.bar(x, means, 0.62, yerr=stds, capsize=5, color=[COLORS[v] for v in vs], alpha=0.85)
        for i, v in enumerate(vs):
            vals = sec["cells"][v]["hit_at_010"]["values"]
            ax.scatter(np.full(len(vals), x[i]) + np.linspace(-0.12, 0.12, len(vals)), vals, color="k", s=14, zorder=3)
            d = sec["deltas_vs_A"].get(v, {}).get("hit_at_010")
            if d:
                ax.text(x[i], means[i] + stds[i] + 0.015, f"{d['delta']:+.3f}\n{stars(d['p_boot'])}", ha="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([NAMES[v].replace(" (", "\n(") for v in vs])
        ax.set_ylabel("hit@0.10 (mean ± std; dots = seeds)")
        ax.set_title(f"{title}\nn_train={sec['n_train']}, seeds={sec['cells'][vs[0]]['hit_at_010']['n_seeds']}")
        ax.set_ylim(0, max(means) + max(stds) + 0.12)
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
def section_m2w() -> dict:
    out = {"cells": {}, "delta": {}}
    for v in ("A", "Dhook"):
        vals = defaultdict(list); seeds = []
        for s in (42, 43, 44):
            p = P4 / M2W_TEMPLATE.format(v=v, s=s)
            if not p.exists():
                continue
            fvm = json.loads(p.read_text())["final_val_metrics"]; seeds.append(s)
            for m in ("hit_at_010", "hit_at_025", "hit_at_bbox", "mean_normalized_l2"):
                vals[m].append(float(fvm[m]))
        out["cells"][v] = {m: {"mean": float(np.mean(x)) if x else math.nan,
                               "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, "n_seeds": len(x), "values": x}
                           for m, x in vals.items()}
        out["cells"][v]["seeds"] = seeds
    for m in ("hit_at_010", "hit_at_025", "hit_at_bbox", "mean_normalized_l2"):
        a = out["cells"]["A"].get(m, {}).get("mean", math.nan); d = out["cells"]["Dhook"].get(m, {}).get("mean", math.nan)
        out["delta"][m] = d - a
    return out


def md_m2w(sec: dict) -> str:
    L = ["### Mind2Web grounding control (Multimodal-Mind2Web, n_train=1000, n_val=250)", "",
         "| variant | seeds | hit@0.10 | hit@0.25 | hit@bbox | mean L2 ↓ |", "|---|---|---|---|---|---|"]
    for v in ("A", "Dhook"):
        c = sec["cells"][v]
        L.append(f"| {NAMES[v]} | {c.get('seeds')} | {fmt_ms(c['hit_at_010'])} | {fmt_ms(c['hit_at_025'])} | "
                 f"{fmt_ms(c['hit_at_bbox'])} | {fmt_ms(c['mean_normalized_l2'])} |")
    L.append(f"| D-hook − A | | {sec['delta']['hit_at_010']:+.3f} | {sec['delta']['hit_at_025']:+.3f} | "
             f"{sec['delta']['hit_at_bbox']:+.3f} | {sec['delta']['mean_normalized_l2']:+.3f} |")
    L.append("\nSeed-level only (this job does not log per-example distances).")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def section_causal() -> dict:
    conds = ("gold", "wrong", "zero")
    agg = {c: defaultdict(list) for c in conds}; dists = {c: [] for c in conds}; seeds = []; norms = []
    for s in HEADLINE_SEEDS:
        p = P4 / CAUSAL_TEMPLATE.format(s=s)
        if not p.exists():
            continue
        ce = json.loads(p.read_text()).get("causal_eval")
        if not ce:
            continue
        seeds.append(s); norms.append(ce.get("action_emb_norm", math.nan))
        for c in conds:
            blk = ce[c]
            d = np.asarray(blk["per_example_dist"], dtype=np.float64)
            dists[c].append(d)
            for m in METRICS:
                agg[c][m].append(float(d.mean()) if m == "mean_normalized_l2" else float((d <= RADII[m]).mean()))
    out = {"seeds": seeds, "emb_norm": float(np.mean(norms)) if norms else math.nan,
           "cells": {c: {m: {"mean": float(np.mean(agg[c][m])) if agg[c][m] else math.nan,
                             "std": float(np.std(agg[c][m], ddof=1)) if len(agg[c][m]) > 1 else 0.0,
                             "n_seeds": len(agg[c][m]), "values": agg[c][m]} for m in METRICS} for c in conds},
           "paired": {}}
    if dists["gold"]:
        g = pool_distances_across_seeds(dists["gold"])
        for other in ("wrong", "zero"):
            o = pool_distances_across_seeds(dists[other])
            for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2"):
                r = paired_bootstrap(o, g, metric=m, n_boot=N_BOOT, seed=0)
                out["paired"][f"gold_minus_{other}"] = out["paired"].get(f"gold_minus_{other}", {})
                out["paired"][f"gold_minus_{other}"][m] = {"delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high,
                                                           "p_boot": r.p_value, "n_units": int(g.shape[0])}
    return out


def md_causal(sec: dict) -> str:
    L = [f"### D-token causal-use test (seeds {sec['seeds']}, learned embedding norm {sec['emb_norm']:.2f})", "",
         "| condition | hit@0.05 | hit@0.10 | hit@0.25 | mean L2 ↓ |", "|---|---|---|---|---|"]
    for c in ("gold", "wrong", "zero"):
        cc = sec["cells"][c]
        L.append(f"| {c} | " + " | ".join(fmt_ms(cc[m]) for m in METRICS) + " |")
    L += ["", "| contrast | hit@0.10 | hit@0.25 | mean L2 | units |", "|---|---|---|---|---|"]
    for k, ds in sec["paired"].items():
        L.append(f"| {k.replace('_', ' ')} | {fmt_delta(ds['hit_at_010'])} | {fmt_delta(ds['hit_at_025'])} | "
                 f"{fmt_delta(ds['mean_normalized_l2'])} | {ds['hit_at_010']['n_units']} |")
    return "\n".join(L) + "\n"


def tex_causal(sec: dict) -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         f"\\caption{{D-token causal-use test: the same trained model evaluated with the gold action embedding, a wrong "
         f"(cyclically permuted) one, and a zeroed one ({len(sec['seeds'])} seeds, paired bootstrap over pooled examples).}}",
         "\\label{tab:causal}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccc}", "\\toprule",
         "Conditioning & hit@0.05 & hit@0.10 & hit@0.25 & mean L2 $\\downarrow$ \\\\", "\\midrule"]
    for c in ("gold", "wrong", "zero"):
        cc = sec["cells"][c]
        L.append(f"{c} & " + " & ".join(tex_ms(cc[m]) for m in METRICS) + " \\\\")
    L.append("\\midrule")
    for k, ds in sec["paired"].items():
        L.append(f"{k.replace('_minus_', ' $-$ ')} & & {tex_delta(ds['hit_at_010'])} & {tex_delta(ds['hit_at_025'])} & "
                 f"{tex_delta(ds['mean_normalized_l2'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def strip_runs(obj):
    if isinstance(obj, dict):
        return {k: strip_runs(v) for k, v in obj.items() if k != "_runs"}
    if isinstance(obj, list):
        return [strip_runs(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main() -> None:
    P8.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    head = section_variants("all_with_coords", 1200, HEADLINE_SEEDS, ["A", "B", "C", "Dhook", "Dtoken"],
                            "Headline ablation (all_with_coords, n_train=1200, n_val=250)")
    ctrl = section_variants("taps_and_swipes", 1000, CURVE_SEEDS, ["A", "B", "C", "Dhook"],
                            "Control (taps_and_swipes, n_train=1000, n_val=200)")
    scal = section_scaling()
    m2w = section_m2w()
    caus = section_causal()

    md = ["# Phase 8 results — pre-submission consolidation", "",
          "Generated by `scripts/p8_consolidate.py` from `results/phase4/*.json`. "
          "See `neurips2026/SUBMISSION_NOTES.md` for the experiment plan (E1–E6).", "",
          md_variants(head), md_variants(ctrl), md_scaling(scal), md_m2w(m2w), md_causal(caus)]
    (P8 / "PHASE8_RESULTS.md").write_text("\n".join(md))
    (P8 / "phase8_summary.json").write_text(json.dumps(strip_runs(
        {"headline": head, "control": ctrl, "scaling": scal, "m2w": m2w, "causal": caus}), indent=2))
    (TABLES / "headline.tex").write_text(tex_variants(
        head, "tab:headline", "Stage 2 grounding on AITW \\texttt{all\\_with\\_coords} (n$_{\\text{train}}$=1200, n$_{\\text{val}}$=250, "
        f"{head['cells']['A']['hit_at_010']['n_seeds']} seeds). Mean $\\pm$ std over seeds; $\\Delta$ with 95\\% paired-bootstrap CI over pooled "
        "(seed, example) units; $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$."))
    (TABLES / "control.tex").write_text(tex_variants(
        ctrl, "tab:control", "Control: AITW \\texttt{taps\\_and\\_swipes} (n$_{\\text{train}}$=1000, n$_{\\text{val}}$=200, "
        f"{ctrl['cells']['A']['hit_at_010']['n_seeds']} seeds), where action type is not spatially informative."))
    (TABLES / "scaling.tex").write_text(tex_scaling(scal))
    (TABLES / "causal.tex").write_text(tex_causal(caus))
    plot_headline_vs_control(head, ctrl, P8 / "ablation_headline_vs_control.png")
    plot_scaling(scal, P8 / "scaling_curve_multiseed.png")
    # mirror paper-facing outputs into the submission directory
    import shutil
    paper_tables = ROOT / "neurips2026" / "tables"; paper_tables.mkdir(parents=True, exist_ok=True)
    for t in TABLES.glob("*.tex"):
        shutil.copy(t, paper_tables / t.name)
    for f in ("ablation_headline_vs_control.png", "scaling_curve_multiseed.png"):
        shutil.copy(P8 / f, ROOT / "neurips2026" / "figures" / f)
    print((P8 / "PHASE8_RESULTS.md").read_text())
    print(f"wrote {P8}/PHASE8_RESULTS.md, phase8_summary.json, tables/*.tex, 2 figures (mirrored to neurips2026/)")


if __name__ == "__main__":
    main()
