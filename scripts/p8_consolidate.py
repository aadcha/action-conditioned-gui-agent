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
    "Dtext": "Dtext_seed{s}_n{n}_ep2_lr2e-05_mix-{mix}.json",
}
CAUSAL_TEMPLATE = "Dtoken_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02_causal.json"
INTERV_TEMPLATE = "interv_{v}_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords.json"
M2W_TEMPLATE = "m2w_{v}_seed{s}_n1000_ep2.json"
NAMES = {"A": "A (flat)", "B": "B (aux loss)", "C": "C (hard routing)",
         "Dhook": "D-hook (additive)", "Dtoken": "D-token (prepended)", "Dtext": "D-text (action word in prompt)"}
TEX_NAMES = {"A": "A: flat", "B": "B: aux.\\ loss", "C": "C: hard routing",
             "Dhook": "D-hook: additive", "Dtoken": "D-token: prepended", "Dtext": "D-text: word in prompt"}
HEADLINE_SEEDS = [42, 43, 44, 45, 46]
CURVE_SEEDS = [42, 43, 44]
CURVE_NS = [300, 500, 800, 1200, 2500, 5000]
METRICS = ["hit_at_005", "hit_at_010", "hit_at_025", "mean_normalized_l2"]
RADII = {"hit_at_005": 0.05, "hit_at_010": 0.10, "hit_at_014": 0.14, "hit_at_025": 0.25}
N_BOOT = 10000
COLORS = {"A": "#3a6ea5", "B": "#2a9d4a", "C": "#e07b39", "Dhook": "#9d4edd", "Dtoken": "#c9184a", "Dtext": "#6c757d"}

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


def load_run(variant: str, seed: int, n: int, mix: str, suffix: str = "") -> Run | None:
    name = TEMPLATES[variant].format(s=seed, n=n, mix=mix)
    if suffix:
        name = name[:-5] + suffix + ".json"
    p = P4 / name
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


_EPISODES_CACHE: dict = {}


def slice_episodes(mix: str, n: int) -> tuple[list[str], str | None] | None:
    """Episode id per validation index for a (mix, n_train) slice, plus the id of the
    training-side boundary episode (None if the slice starts a fresh episode)."""
    p = P8 / "val_episodes.json"
    if not p.exists():
        return None
    if not _EPISODES_CACHE:
        _EPISODES_CACHE.update(json.loads(p.read_text()))
    key = f"{mix}_n{n}"
    if key not in _EPISODES_CACHE:
        return None
    v = _EPISODES_CACHE[key]
    eps = [x["ep_id"] for x in v["val"]]
    return eps, (v.get("train_last_ep") if eps and eps[0] == v.get("train_last_ep") else None)


def _episode_bootstrap(diff_ex: np.ndarray, eps: list[str], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Cluster bootstrap over EPISODES of a per-example difference vector."""
    ids = {e: i for i, e in enumerate(dict.fromkeys(eps))}
    idx = np.array([ids[e] for e in eps]); E = len(ids)
    sums = np.bincount(idx, weights=diff_ex, minlength=E); cnts = np.bincount(idx, minlength=E).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, E, size=(n_boot, E))
    boots = sums[pick].sum(1) / cnts[pick].sum(1)
    obs = float(diff_ex.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(min(1.0, 2 * ((boots <= 0).mean() if obs >= 0 else (boots >= 0).mean())))
    return {"ci_low": float(lo), "ci_high": float(hi), "p": p, "n_episodes": int(E)}


def paired(base: list[Run], other: list[Run], m: str, mix: str | None = None, n: int | None = None,
           eps: list[str] | None = None) -> dict | None:
    """other - base over pooled (seed, example) units; seeds matched by id.
    `mix`/`n` name the validation slice when a run's own n_train is not the slice key (the _notype
    runs record the post-filter count); `eps` gives the episode id of every retained example for
    subset contrasts (e.g. clicks only), so the episode bootstrap clusters the right steps."""
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
    # Cluster bootstrap: the (seed, example) units sharing an example are correlated, so
    # resample EXAMPLES (keeping every seed of a sampled example) instead of units.
    va = np.stack([_metric_vals(bs[s].dist[:L], m) for s in seeds])   # [S, L]
    vb = np.stack([_metric_vals(os_[s].dist[:L], m) for s in seeds])
    diff_ex = (vb - va).mean(0)                                        # per-example mean over seeds
    rng = np.random.default_rng(0)
    idx = rng.integers(0, L, size=(N_BOOT, L))
    boots = diff_ex[idx].mean(1)
    cl_lo, cl_hi = np.percentile(boots, [2.5, 97.5])
    obs = float(diff_ex.mean())
    cl_p = float(min(1.0, 2 * ((boots <= 0).mean() if obs >= 0 else (boots >= 0).mean())))
    # Seed-level paired test on per-seed means (n = number of shared seeds).
    sa, sb = va.mean(1), vb.mean(1)
    seed_p = float("nan")
    if len(seeds) >= 3:
        from scipy import stats
        seed_p = float(stats.ttest_rel(sb, sa).pvalue)
    out = {"delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_boot": r.p_value,
           "p_perm": p_perm, "n_units": int(a.shape[0]), "seeds": seeds,
           "higher_is_better": r.higher_is_better,
           "cluster_ci_low": float(cl_lo), "cluster_ci_high": float(cl_hi), "cluster_p": cl_p,
           "seed_t_p": seed_p, "n_examples": int(L)}
    # Episode-cluster bootstrap (steps of one episode are correlated) + boundary-episode check.
    ref = next(iter(bs.values()))
    ep = slice_episodes(mix or ref.mix, n or ref.n_train)
    if eps is not None:
        assert len(eps) == L, (len(eps), L)
        ep = (list(eps), ep[1] if ep is not None else None)
    if ep is not None and len(ep[0]) >= L:
        eps, boundary = ep[0][:L], ep[1]
        eb = _episode_bootstrap(diff_ex, eps)
        out.update({"ep_ci_low": eb["ci_low"], "ep_ci_high": eb["ci_high"], "ep_p": eb["p"], "n_episodes": eb["n_episodes"]})
        if boundary is not None:
            keep = np.array([e != boundary for e in eps])
            if keep.sum() > 10:
                eb2 = _episode_bootstrap(diff_ex[keep], [e for e, k in zip(eps, keep) if k])
                out.update({"nb_delta": float(diff_ex[keep].mean()), "nb_ci_low": eb2["ci_low"], "nb_ci_high": eb2["ci_high"],
                            "nb_p": eb2["p"], "n_boundary_steps": int((~keep).sum())})
    return out


def _metric_vals(dist: np.ndarray, m: str) -> np.ndarray:
    return dist if m == "mean_normalized_l2" else (dist <= RADII[m]).astype(float)


def slice_labels(mix: str, n: int) -> list[str] | None:
    """Per-example action labels of a deterministic validation slice, when a job logged them.
    The qualitative job logs gold coordinates and actions for the headline slice."""
    if mix == "all_with_coords" and n == 1200:
        p = P8 / "qualitative_v2" / "render.json"
        if p.exists():
            rows = json.loads(p.read_text()).get("all_rows") or []
            if len(rows) == 250:
                return [r["action"] for r in rows]
    return None


def section_e2e() -> dict:
    """Predicted-type pipeline vs oracle vs flat A, per seed and paired over examples."""
    out = {"seeds": [], "per_seed": [], "paired": {}}
    a_d, o_d, p_d = [], [], []
    for s in CURVE_SEEDS:
        pe = P4 / f"e2e_seed{s}_n1200_ep2_lr2e-05_mix-all_with_coords.json"
        ra = load_run("A", s, 1200, "all_with_coords")
        if not pe.exists() or ra is None or ra.dist is None:
            continue
        e = json.loads(pe.read_text())
        od = np.asarray(e["oracle_per_example_dist"], float); pd_ = np.asarray(e["predicted_per_example_dist"], float)
        L = min(len(od), len(pd_), len(ra.dist))
        a_d.append(ra.dist[:L]); o_d.append(od[:L]); p_d.append(pd_[:L]); out["seeds"].append(s)
        out["per_seed"].append({"seed": s, "stage1_acc": e.get("stage1_val_acc"), "stage1_macro_f1": e.get("stage1_val_macro_f1"),
                                "A_hit010": float((ra.dist[:L] <= 0.10).mean()), "oracle_hit010": float((od[:L] <= 0.10).mean()),
                                "predicted_hit010": float((pd_[:L] <= 0.10).mean())})
    if not a_d:
        return out
    def _pair(x, y, m):
        rx = [Run(Path("x"), "x", s, 1200, "all_with_coords", {}, d) for s, d in zip(out["seeds"], x)]
        ry = [Run(Path("y"), "y", s, 1200, "all_with_coords", {}, d) for s, d in zip(out["seeds"], y)]
        return paired(rx, ry, m)
    for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2"):
        out["paired"][m] = {"predicted_minus_A": _pair(a_d, p_d, m), "oracle_minus_A": _pair(a_d, o_d, m),
                            "oracle_minus_predicted": _pair(p_d, o_d, m)}
    out["means"] = {k: float(np.mean([p[k] for p in out["per_seed"]])) for k in ("stage1_acc", "stage1_macro_f1", "A_hit010", "oracle_hit010", "predicted_hit010")}
    return out


def md_e2e(sec: dict) -> str:
    if not sec.get("per_seed"):
        return "### End to end\n\n(no e2e runs found)\n"
    L = [f"### End to end: Stage-1 predicted types vs oracle vs flat A (seeds {sec['seeds']})", "",
         "| seed | Stage-1 acc | Stage-1 macro-F1 | A hit@0.10 | oracle hit@0.10 | predicted hit@0.10 |", "|---|---|---|---|---|---|"]
    for p in sec["per_seed"]:
        L.append(f"| {p['seed']} | {p['stage1_acc']:.3f} | {p['stage1_macro_f1']:.3f} | {p['A_hit010']:.3f} | {p['oracle_hit010']:.3f} | {p['predicted_hit010']:.3f} |")
    m = sec["means"]
    L += [f"| mean | {m['stage1_acc']:.3f} | {m['stage1_macro_f1']:.3f} | {m['A_hit010']:.3f} | {m['oracle_hit010']:.3f} | {m['predicted_hit010']:.3f} |", "",
          "| contrast | hit@0.10 | hit@0.25 | mean L2 |", "|---|---|---|---|"]
    for k in ("predicted_minus_A", "oracle_minus_A", "oracle_minus_predicted"):
        L.append(f"| {k.replace('_', ' ')} | " + " | ".join(fmt_delta(sec["paired"][mm][k]) for mm in ("hit_at_010", "hit_at_025", "mean_normalized_l2")) + " |")
    return "\n".join(L) + "\n"


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
    s = f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] {stars(d['p_boot'])}"
    if "cluster_ci_low" in d:
        s += f" / ex-cluster [{d['cluster_ci_low']:+.3f}, {d['cluster_ci_high']:+.3f}] {stars(d['cluster_p'])}"
    if "ep_ci_low" in d:
        s += f" / EPISODE [{d['ep_ci_low']:+.3f}, {d['ep_ci_high']:+.3f}] {stars(d['ep_p'])} (E={d['n_episodes']})"
    if "nb_delta" in d:
        s += f" / no-boundary {d['nb_delta']:+.3f} [{d['nb_ci_low']:+.3f}, {d['nb_ci_high']:+.3f}] {stars(d['nb_p'])} (-{d['n_boundary_steps']} steps)"
    if not math.isnan(d.get("seed_t_p", float("nan"))):
        s += f" / seed-t p={d['seed_t_p']:.3f}"
    return s


def tex_delta(d: dict | None) -> str:
    """Delta with the cluster-bootstrap interval (examples resampled, seeds kept together) when available."""
    if d is None:
        return "--"
    if "cluster_ci_low" in d:
        return tex_delta_cluster(d)
    s = stars(d["p_boot"])
    s = "" if s == "ns" else f"$^{{{s}}}$"
    return f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{s}"


def tex_delta_cluster(d: dict | None) -> str:
    """Delta with the EPISODE-cluster interval when episode ids are known, else the example-cluster one."""
    if d is None:
        return "--"
    if "ep_ci_low" in d:
        s = stars(d["ep_p"]); s = "" if s == "ns" else f"$^{{{s}}}$"
        return f"{d['delta']:+.3f} [{d['ep_ci_low']:+.3f}, {d['ep_ci_high']:+.3f}]{s}"
    s = stars(d["cluster_p"])
    s = "" if s == "ns" else f"$^{{{s}}}$"
    return f"{d['delta']:+.3f} [{d['cluster_ci_low']:+.3f}, {d['cluster_ci_high']:+.3f}]{s}"


def tex_seed_p(d: dict | None) -> str:
    if d is None or math.isnan(d.get("seed_t_p", float("nan"))):
        return "--"
    return f"{d['seed_t_p']:.3f}"


# ---------------------------------------------------------------------------
def section_variants(mix: str, n: int, seeds: list[int], variants: list[str], title: str) -> dict:
    runs = {v: [r for s in seeds if (r := load_run(v, s, n, mix))] for v in variants}
    out = {"mix": mix, "n_train": n, "seeds_requested": seeds, "cells": {}, "deltas_vs_A": {}, "deltas_vs_B": {},
           "parse_rates": {}, "no_per_example": []}
    for v in variants:
        out["cells"][v] = {m: cell(runs[v], m) for m in METRICS + ["hit_at_014"]}
        out["parse_rates"][v] = [r.parse_rate for r in runs[v]]
        # parsed-only rerun of the hit rates (parse failures carry the sqrt(2) sentinel)
        parsed = [Run(r.path, r.variant, r.seed, r.n_train, r.mix, r.fvm, r.dist[~np.isclose(r.dist, 2 ** 0.5)])
                  for r in runs[v] if r.dist is not None]
        out.setdefault("cells_parsed_only", {})[v] = {m: cell(parsed, m) for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2")}
        out["no_per_example"] += [r.path.name for r in runs[v] if r.dist is None]
        if v != "A":
            out["deltas_vs_A"][v] = {m: paired(runs["A"], runs[v], m) for m in METRICS + ["hit_at_014"]}
        if v not in ("A", "B") and "B" in runs:
            out["deltas_vs_B"][v] = {m: paired(runs["B"], runs[v], m) for m in METRICS}
    out["per_class"] = {}
    labels = slice_labels(mix, n)
    out["per_class_basis"] = ("per-example distances, parse failures = miss" if labels is not None
                              else "run-logged per_class blocks (parsed outputs only)")
    for v in variants:
        pc = defaultdict(list)
        for r in runs[v]:
            if labels is not None and r.dist is not None and len(r.dist) == len(labels):
                for cls in sorted(set(labels)):
                    msk = np.array([l == cls for l in labels])
                    pc[cls].append(float((r.dist[msk] <= 0.10).mean()))
            else:
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
        L += ["", f"Per-class hit@0.10 (mean ± std over seeds; basis: {sec.get('per_class_basis')}):", "",
              "| variant | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for v, pcs in sec["per_class"].items():
            L.append(f"| {NAMES[v]} | " + " | ".join(fmt_ms(pcs[c]) if c in pcs else "--" for c in classes) + " |")
    L += ["", "hit@0.14 (official AITW tap threshold) per variant and delta vs A:", ""]
    for v, cells in sec["cells"].items():
        d = sec["deltas_vs_A"].get(v, {}).get("hit_at_014")
        L.append(f"- {NAMES[v]}: {fmt_ms(cells['hit_at_014'])}" + (f"; vs A {fmt_delta(d)}" if d else ""))
    if sec.get("cells_parsed_only"):
        L += ["", "Parsed outputs only (parse failures removed rather than scored as misses):", ""]
        for v, c in sec["cells_parsed_only"].items():
            L.append(f"- {NAMES[v]}: hit@0.10 {fmt_ms(c['hit_at_010'])}, hit@0.25 {fmt_ms(c['hit_at_025'])}, mean L2 {fmt_ms(c['mean_normalized_l2'])}")
    if sec["no_per_example"]:
        L += ["", "Runs WITHOUT per-example logging (fell back to summary metrics): " + ", ".join(sec["no_per_example"])]
    return "\n".join(L) + "\n"


def tex_variants(sec: dict, label: str, caption: str) -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccccc}", "\\toprule",
         "Variant & hit@0.05 & hit@0.10 & hit@0.25 & mean L2 $\\downarrow$ & $\\Delta$hit@0.10 vs.\\ A [episode CI] & seed-level $p$ \\\\", "\\midrule"]
    for v, cells in sec["cells"].items():
        if cells["hit_at_010"]["n_seeds"] == 0:
            continue
        d = sec["deltas_vs_A"].get(v, {}).get("hit_at_010") if v != "A" else None
        L.append(f"{TEX_NAMES[v]} & {tex_ms(cells['hit_at_005'])} & {tex_ms(cells['hit_at_010'])} & "
                 f"{tex_ms(cells['hit_at_025'])} & {tex_ms(cells['mean_normalized_l2'])} & "
                 f"{tex_delta_cluster(d) if d else ''} & {tex_seed_p(d) if d else ''} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def section_scaling() -> dict:
    out = {"ns": CURVE_NS, "seeds": CURVE_SEEDS, "cells": {}, "deltas": {}, "val_mix": {}}
    for n in CURVE_NS:
        runs = {v: [r for s in CURVE_SEEDS if (r := load_run(v, s, n, "all_with_coords"))] for v in ("A", "B", "Dhook")}
        ra = runs["A"][0] if runs["A"] else None
        if ra is not None:
            raw = json.loads(ra.path.read_text()).get("val_action_distribution") or {}
            out["val_mix"][n] = {{"0": "click", "3": "scroll", "2": "type"}.get(k, k): v for k, v in raw.items()}
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
          "only the within-n deltas are. Seeds (s) shown for A; B and D-hook have the same seed count unless a run is missing.", "",
          "Validation class mix per n: " + "; ".join(f"n={n}: " + ", ".join(f"{k} {v}" for k, v in mx.items()) for n, mx in sec["val_mix"].items())]
    return "\n".join(L) + "\n"


def tex_scaling(sec: dict) -> str:
    L = ["\\begin{table}[h]", "\\centering", "\\small",
         "\\caption{Conditioning advantage vs.\\ training-set size (AITW \\texttt{all\\_with\\_coords}, 3 seeds per cell). "
         "Intervals are 95\\% episode-cluster bootstraps (every step and seed of a sampled episode kept together); "
         "stars from the same bootstrap. Absolute values are not comparable across rows because the validation slice moves "
         "with $n$; its click/scroll/type counts are given per row. The study is descriptive: 24 contrasts, no multiplicity correction.}",
         "\\label{tab:scaling}",
         "\\resizebox{\\linewidth}{!}{\\begin{tabular}{rlccccc}", "\\toprule",
         "$n_{\\text{train}}$ & val mix (c/s/t) & A hit@0.10 & B $-$ A & D-hook $-$ A & B $-$ A (hit@0.25) & D-hook $-$ A (hit@0.25) \\\\", "\\midrule"]
    for n in sec["ns"]:
        c = sec["cells"][n]; d = sec["deltas"][n]; mx = sec["val_mix"].get(n, {})
        mix_txt = "/".join(str(mx.get(k, "?")) for k in ("click", "scroll", "type"))
        L.append(f"{n} & {mix_txt} & {tex_ms(c['A']['hit_at_010'])} & {tex_delta_cluster(d['B']['hit_at_010'])} & {tex_delta_cluster(d['Dhook']['hit_at_010'])} & "
                 f"{tex_delta_cluster(d['B']['hit_at_025'])} & {tex_delta_cluster(d['Dhook']['hit_at_025'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


def plot_scaling(sec: dict, out: Path) -> None:
    """Two panels (deltas only; absolute accuracy is in the appendix table) sized for a
    single NeurIPS column, with cluster-bootstrap intervals."""
    ns = [n for n in sec["ns"] if sec["cells"][n]["A"]["hit_at_010"]["n_seeds"] > 0]
    with plt.rc_context({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
        for ax, m, lab in ((axes[0], "hit_at_010", "hit@0.10"), (axes[1], "hit_at_025", "hit@0.25")):
            ax.axhline(0, color="#888", lw=0.9, ls=":")
            for v, off in (("B", -0.04), ("Dhook", 0.04)):
                xs, ys, lo, hi = [], [], [], []
                for n in ns:
                    d = sec["deltas"][n][v][m]
                    if d is None:
                        continue
                    cl = d.get("ep_ci_low", d.get("cluster_ci_low", d["ci_low"])); ch = d.get("ep_ci_high", d.get("cluster_ci_high", d["ci_high"]))
                    xs.append(n * (1 + off)); ys.append(d["delta"]); lo.append(d["delta"] - cl); hi.append(ch - d["delta"])
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="s" if v == "Dhook" else "o", ms=8, capsize=5, lw=2, ls="-",
                            color=COLORS[v], label=f"{NAMES[v]} − A")
            ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
            ax.set_xlabel("training examples (log scale)"); ax.set_ylabel(f"Δ {lab} vs. flat A")
            ax.set_title(f"Conditioning advantage, {lab}"); ax.legend(loc="best")
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
        mk = lambda ds: [Run(Path("c"), "c", s, 1200, "all_with_coords", {}, d) for s, d in zip(seeds, ds)]
        for other in ("wrong", "zero"):
            out["paired"][f"gold_minus_{other}"] = {m: paired(mk(dists[other]), mk(dists["gold"]), m)
                                                   for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2")}
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
    L = ["\\begin{table}[h]", "\\centering", "\\small",
         f"\\caption{{D-token causal-use test: the same trained model evaluated with the gold action embedding, a wrong "
         f"(cyclically permuted) one, and a zeroed one ({len(sec['seeds'])} seeds; paired episode-cluster bootstrap intervals).}}",
         "\\label{tab:causal}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccc}", "\\toprule",
         "Conditioning & hit@0.05 & hit@0.10 & hit@0.25 & mean L2 $\\downarrow$ \\\\", "\\midrule"]
    for c in ("gold", "wrong", "zero"):
        cc = sec["cells"][c]
        L.append(f"{c} & " + " & ".join(tex_ms(cc[m]) for m in METRICS) + " \\\\")
    L.append("\\midrule")
    for k, ds in sec["paired"].items():
        L.append(f"{k.replace('_minus_', ' $-$ ')} & & {tex_delta_cluster(ds['hit_at_010'])} & {tex_delta_cluster(ds['hit_at_025'])} & "
                 f"{tex_delta_cluster(ds['mean_normalized_l2'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def _cls_masks(labels: list[str]) -> dict[str, np.ndarray]:
    return {c: np.array([l == c for l in labels]) for c in sorted(set(labels))}


NOTYPE_VARIANTS = ("A", "B", "C", "Dhook", "Dtoken", "Dtext")


def section_notype() -> dict:
    """Same training slice with type events removed from TRAINING only, scored on the
    identical headline validation slice. Tests whether the conditioning gain needs the
    degenerate-target class to be present in training. All six mechanisms, three seeds."""
    labels = slice_labels("all_with_coords", 1200)
    masks = _cls_masks(labels) if labels else {}
    ep = slice_episodes("all_with_coords", 1200)
    eps_all = ep[0] if ep else None
    kw = {"mix": "all_with_coords", "n": 1200}
    out = {"seeds": CURVE_SEEDS, "variants": [], "cells": {}, "deltas_vs_A": {}, "drop_effect": {}, "per_class": {}}
    runs = {}
    for v in NOTYPE_VARIANTS:
        runs[(v, "with")] = [r for s in CURVE_SEEDS if (r := load_run(v, s, 1200, "all_with_coords"))]
        runs[(v, "without")] = [r for s in CURVE_SEEDS if (r := load_run(v, s, 1200, "all_with_coords", "_notype"))]
        if runs[(v, "without")]:
            out["variants"].append(v)
    for (v, cond), rs in runs.items():
        out["cells"][f"{v}_{cond}"] = {m: cell(rs, m) for m in METRICS}
        pc = {}
        for c, msk in masks.items():
            vals = [float((r.dist[msk] <= 0.10).mean()) for r in rs if r.dist is not None and len(r.dist) == len(labels)]
            pc[c] = {"mean": float(np.mean(vals)) if vals else math.nan, "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}
        out["per_class"][f"{v}_{cond}"] = pc
    for v in out["variants"]:
        if v != "A":
            out["deltas_vs_A"][f"{v}_without"] = {m: paired(runs[("A", "without")], runs[(v, "without")], m, **kw) for m in METRICS}
        out["drop_effect"][v] = {m: paired(runs[(v, "with")], runs[(v, "without")], m, **kw) for m in METRICS}
    # click-only paired contrasts (the class the gain lives in), episode ids restricted to the clicks
    if masks:
        out["click_deltas"] = {}
        ck = masks["click"]
        eps_click = [e for e, k in zip(eps_all, ck) if k] if eps_all is not None else None
        def _sub(rs, msk):
            return [Run(r.path, r.variant, r.seed, r.n_train, r.mix, r.fvm, r.dist[msk]) for r in rs if r.dist is not None and len(r.dist) == len(labels)]
        for v in out["variants"]:
            if v == "A":
                continue
            for cond in ("with", "without"):
                out["click_deltas"][f"{v}_{cond}"] = paired(_sub(runs[("A", cond)], ck), _sub(runs[(v, cond)], ck), "hit_at_010", eps=eps_click, **kw)
    return out


def md_notype(sec: dict) -> str:
    vs = sec["variants"]
    L = ["### Type events removed from training (same 250-example validation slice, seeds 42–44)", "",
         "| variant | training stream | hit@0.10 | hit@0.25 | mean L2 | click hit@0.10 | scroll hit@0.10 |", "|---|---|---|---|---|---|---|"]
    nanc = {"mean": math.nan, "std": 0, "n_seeds": 0}
    for v in vs:
        for cond in ("with", "without"):
            c = sec["cells"][f"{v}_{cond}"]; pc = sec["per_class"].get(f"{v}_{cond}", {})
            L.append(f"| {NAMES[v]} | {'with type' if cond == 'with' else 'type removed'} | {fmt_ms(c['hit_at_010'])} | {fmt_ms(c['hit_at_025'])} | "
                     f"{fmt_ms(c['mean_normalized_l2'])} | {fmt_ms(pc.get('click', nanc))} | {fmt_ms(pc.get('scroll', nanc))} |")
    L += ["", "Paired deltas within the type-removed condition (vs A trained without type):", "", "| contrast | hit@0.10 | hit@0.25 | mean L2 |", "|---|---|---|---|"]
    for v in vs:
        if v == "A":
            continue
        ds = sec["deltas_vs_A"][f"{v}_without"]
        L.append(f"| {NAMES[v]} − A (both without type) | " + " | ".join(fmt_delta(ds[m]) for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2")) + " |")
    L += ["", "Effect of removing type events from training (without − with), per variant:", "", "| variant | hit@0.10 | hit@0.25 | mean L2 |", "|---|---|---|---|"]
    for v in vs:
        ds = sec["drop_effect"][v]
        L.append(f"| {NAMES[v]} | " + " | ".join(fmt_delta(ds[m]) for m in ("hit_at_010", "hit_at_025", "mean_normalized_l2")) + " |")
    if sec.get("click_deltas"):
        L += ["", "Click-only paired deltas vs A (hit@0.10; episode bootstrap over the episodes that contain clicks):", "", "| contrast | with type | type removed |", "|---|---|---|"]
        for v in vs:
            if v == "A":
                continue
            L.append(f"| {NAMES[v]} − A | {fmt_delta(sec['click_deltas'][f'{v}_with'])} | {fmt_delta(sec['click_deltas'][f'{v}_without'])} |")
    return "\n".join(L) + "\n"


def tex_notype(sec: dict) -> str:
    nanc = {"mean": math.nan, "std": 0, "n_seeds": 0}
    L = ["\\begin{table}[h]", "\\centering", "\\small",
         "\\caption{Removing \\emph{type} events from the training slice only, scored on the identical headline validation "
         "slice (three seeds). Deltas are paired episode-cluster bootstraps (every step and seed of a sampled episode kept "
         "together); seed-level $p$ in parentheses.}",
         "\\label{tab:notype}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llcccc}", "\\toprule",
         "Variant & Training stream & hit@0.10 & click hit@0.10 & scroll hit@0.10 & $\\Delta$hit@0.10 vs.\\ A (same stream) \\\\", "\\midrule"]
    for v in sec["variants"]:
        for cond in ("with", "without"):
            c = sec["cells"][f"{v}_{cond}"]; pc = sec["per_class"].get(f"{v}_{cond}", {})
            if c["hit_at_010"]["n_seeds"] == 0:
                continue
            d = None
            if v != "A":
                d = (sec["deltas_vs_A"].get(f"{v}_without", {}).get("hit_at_010") if cond == "without" else None)
            L.append(f"{TEX_NAMES[v]} & {'with type' if cond == 'with' else 'type removed'} & {tex_ms(c['hit_at_010'])} & "
                     f"{tex_ms(pc.get('click', nanc))} & {tex_ms(pc.get('scroll', nanc))} & "
                     f"{(tex_delta_cluster(d) + ' (' + tex_seed_p(d) + ')') if d else ''} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


def section_interventions() -> dict:
    """Full-power gold/wrong/wrong2/zero/class-mean interventions for D-hook and D-token
    (five seeds x 250 examples), with flat A on the same examples as the reference row."""
    conds = ("gold", "wrong", "wrong2", "zero", "classmean")
    out = {"variants": {}, "conds": list(conds)}
    labels = slice_labels("all_with_coords", 1200)
    masks = _cls_masks(labels) if labels else {}
    a_runs = [r for s in HEADLINE_SEEDS if (r := load_run("A", s, 1200, "all_with_coords"))]
    for v in ("Dhook", "Dtoken"):
        per_cond_runs = {c: [] for c in conds}
        norms, tok_norms, seeds = [], [], []
        for s in HEADLINE_SEEDS:
            p = P4 / INTERV_TEMPLATE.format(v=v, s=s)
            if not p.exists():
                continue
            d = json.loads(p.read_text()); seeds.append(s)
            norms.append(d["action_embedding_norms"]); tok_norms.append(d["token_embedding_mean_norm"])
            for c in conds:
                dist = np.asarray(d["conditions"][c]["per_example_dist"], float)
                per_cond_runs[c].append(Run(p, f"{v}:{c}", s, 1200, "all_with_coords", d["conditions"][c], dist))
        if not seeds:
            continue
        vs = {"seeds": seeds, "cells": {}, "per_class": {}, "gold_minus": {}, "A": {}, "norms": {}}
        for c in conds:
            vs["cells"][c] = {m: cell(per_cond_runs[c], m) for m in METRICS}
            vs["per_class"][c] = {}
            for cls, msk in masks.items():
                vals = [float((r.dist[msk] <= 0.10).mean()) for r in per_cond_runs[c] if len(r.dist) == len(labels)]
                vs["per_class"][c][cls] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}
            if c != "gold":
                vs["gold_minus"][c] = {m: paired(per_cond_runs[c], per_cond_runs["gold"], m) for m in METRICS}
        vs["A"]["cells"] = {m: cell(a_runs, m) for m in METRICS}
        vs["A"]["gold_minus_A"] = {m: paired(a_runs, per_cond_runs["gold"], m) for m in METRICS}
        vs["A"]["zero_minus_A"] = {m: paired(a_runs, per_cond_runs["zero"], m) for m in METRICS}
        vs["A"]["per_class"] = {cls: {"mean": float(np.mean([float((r.dist[msk] <= 0.10).mean()) for r in a_runs if r.dist is not None]))}
                                for cls, msk in masks.items()}
        keys = sorted({k for n_ in norms for k in n_})
        vs["norms"] = {k: float(np.mean([n_[k] for n_ in norms])) for k in keys}
        vs["token_embedding_mean_norm"] = float(np.mean(tok_norms))
        out["variants"][v] = vs
    return out


def md_interv(sec: dict) -> str:
    if not sec["variants"]:
        return "### Interventions (full power)\n\n(no interv_* runs found)\n"
    L = ["### Full-power interventions (same trained model, five conditionings; seeds 42–46, 250 examples)", ""]
    for v, vs in sec["variants"].items():
        L += [f"**{NAMES[v]}** seeds {vs['seeds']}; action-embedding row norms: " +
              ", ".join(f"{k} {val:.3f}" for k, val in vs["norms"].items() if k in ("click", "scroll", "type")) +
              f"; backbone token-embedding mean norm {vs['token_embedding_mean_norm']:.3f}", "",
              "| condition | hit@0.05 | hit@0.10 | hit@0.25 | mean L2 | click hit@0.10 | scroll hit@0.10 | gold − condition (hit@0.10) |", "|---|---|---|---|---|---|---|---|"]
        for c in sec["conds"]:
            cc = vs["cells"][c]; pc = vs["per_class"][c]
            L.append(f"| {c} | {fmt_ms(cc['hit_at_005'])} | {fmt_ms(cc['hit_at_010'])} | {fmt_ms(cc['hit_at_025'])} | {fmt_ms(cc['mean_normalized_l2'])} | "
                     f"{fmt_ms(pc.get('click', {'mean': math.nan, 'std': 0, 'n_seeds': 0}))} | {fmt_ms(pc.get('scroll', {'mean': math.nan, 'std': 0, 'n_seeds': 0}))} | "
                     f"{fmt_delta(vs['gold_minus'][c]['hit_at_010']) if c != 'gold' else '--'} |")
        A = vs["A"]
        L.append(f"| flat A (same examples) | {fmt_ms(A['cells']['hit_at_005'])} | {fmt_ms(A['cells']['hit_at_010'])} | {fmt_ms(A['cells']['hit_at_025'])} | "
                 f"{fmt_ms(A['cells']['mean_normalized_l2'])} | {A['per_class'].get('click', {}).get('mean', math.nan):.3f} | {A['per_class'].get('scroll', {}).get('mean', math.nan):.3f} | "
                 f"gold − A: {fmt_delta(A['gold_minus_A']['hit_at_010'])}; zero − A: {fmt_delta(A['zero_minus_A']['hit_at_010'])} |")
        L.append("")
    return "\n".join(L) + "\n"


def tex_interv(sec: dict) -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Interventions: the same trained model decoded with the gold action embedding, a wrong one (cyclic; "
         "click$\\to$type), a wrong one swapping only clicks and scrolls, the table zeroed, and every row replaced by the class mean. "
         "Five seeds $\\times$ 250 examples; $\\Delta$ = gold $-$ condition, 95\\% episode-cluster CI, seed-level paired $p$; "
         "flat A on the same examples is the reference row.}",
         "\\label{tab:interv}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llccccc}", "\\toprule",
         "Model & Conditioning & hit@0.10 & click hit@0.10 & scroll hit@0.10 & $\\Delta$ gold $-$ cond.\\ (hit@0.10) & seed $p$ \\\\", "\\midrule"]
    cond_name = {"gold": "gold type", "wrong": "wrong (cyclic)", "wrong2": "wrong (click$\\leftrightarrow$scroll)", "zero": "zeroed", "classmean": "class mean"}
    for v, vs in sec["variants"].items():
        for i, c in enumerate(sec["conds"]):
            cc = vs["cells"][c]; pc = vs["per_class"][c]; d = vs["gold_minus"].get(c, {}).get("hit_at_010")
            L.append(f"{TEX_NAMES[v] if i == 0 else ''} & {cond_name[c]} & {tex_ms(cc['hit_at_010'])} & "
                     f"{tex_ms(pc.get('click', {'mean': math.nan, 'std': 0, 'n_seeds': 0}))} & {tex_ms(pc.get('scroll', {'mean': math.nan, 'std': 0, 'n_seeds': 0}))} & "
                     f"{tex_delta_cluster(d) if d else ''} & {tex_seed_p(d) if d else ''} \\\\")
        A = vs["A"]; d = A["gold_minus_A"]["hit_at_010"]
        L.append(f" & flat A, same examples & {tex_ms(A['cells']['hit_at_010'])} & {A['per_class'].get('click', {}).get('mean', math.nan):.3f} & "
                 f"{A['per_class'].get('scroll', {}).get('mean', math.nan):.3f} & {tex_delta_cluster(d)} & {tex_seed_p(d)} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"



DTOKEN_LR = {"base": ("", "shared ($2\\times10^{-5}$)"), "alr0.0002": ("_alr0.0002", "$2\\times10^{-4}$ (10$\\times$)"),
             "alr0.002": ("_alr0.002", "$2\\times10^{-3}$ (100$\\times$)")}


def _row_stats(path: Path, rows: tuple[str, ...]) -> dict | None:
    d = json.loads(path.read_text())
    if "action_row_displacement" not in d:
        return None
    return {"disp": float(np.mean([d["action_row_displacement"][k] for k in rows])),
            "init": float(np.mean([d["action_row_norms_init"][k] for k in rows])),
            "trained": float(np.mean([d["action_row_norms_trained"][k] for k in rows]))}


def section_dtoken_lr() -> dict:
    """D-token with the action-embedding table trained at 1x / 10x / 100x the base learning rate
    on the headline slice, plus how far the rows of the classes present moved from initialization."""
    labels = slice_labels("all_with_coords", 1200); masks = _cls_masks(labels) if labels else {}
    a_runs = [r for s in HEADLINE_SEEDS if (r := load_run("A", s, 1200, "all_with_coords"))]
    b_runs = [r for s in HEADLINE_SEEDS if (r := load_run("B", s, 1200, "all_with_coords"))]
    out = {"configs": {}, "seeds": {}, "cells": {}, "per_class": {}, "deltas_vs_A": {}, "deltas_vs_B": {}, "rows": {}}
    for key, (suffix, desc) in DTOKEN_LR.items():
        rs = [r for s in HEADLINE_SEEDS if (r := load_run("Dtoken", s, 1200, "all_with_coords", suffix))]
        if not rs:
            continue
        out["configs"][key] = desc; out["seeds"][key] = [r.seed for r in rs]
        out["cells"][key] = {m: cell(rs, m) for m in METRICS}
        out["per_class"][key] = {c: cell([Run(r.path, r.variant, r.seed, r.n_train, r.mix, r.fvm, r.dist[msk])
                                          for r in rs if r.dist is not None and len(r.dist) == len(labels)], "hit_at_010")
                                 for c, msk in masks.items()}
        out["deltas_vs_A"][key] = {m: paired(a_runs, rs, m) for m in ("hit_at_010", "hit_at_025")}
        out["deltas_vs_B"][key] = {m: paired(b_runs, rs, m) for m in ("hit_at_010", "hit_at_025")}
        st = [x for r in rs if (x := _row_stats(r.path, ("0", "2", "3")))]     # click, type, scroll rows
        if st:
            out["rows"][key] = {k: float(np.mean([x[k] for x in st])) for k in ("disp", "init", "trained")} | {"n_runs": len(st)}
    # the headline D-token runs predate the displacement logging; the _notype runs share code path and LR
    st = [x for s in CURVE_SEEDS if (r := load_run("Dtoken", s, 1200, "all_with_coords", "_notype")) and (x := _row_stats(r.path, ("0", "3")))]
    if st:
        out["rows"]["base_proxy_notype"] = {k: float(np.mean([x[k] for x in st])) for k in ("disp", "init", "trained")} | {"n_runs": len(st)}
    return out


def md_dtoken_lr(sec: dict) -> str:
    if not sec["cells"]:
        return "### D-token learning-rate sweep\n\n(no runs)\n"
    L = ["### D-token action-table learning-rate sweep (all_with_coords, n_train=1200)", "",
         "| table LR | seeds | hit@0.10 | hit@0.25 | click hit@0.10 | scroll hit@0.10 | Δ vs A (hit@0.10) | Δ vs B (hit@0.10) | row displacement / init norm / trained norm |",
         "|---|---|---|---|---|---|---|---|---|"]
    nanc = {"mean": math.nan, "std": 0, "n_seeds": 0}
    for key, desc in sec["configs"].items():
        c = sec["cells"][key]; pc = sec["per_class"].get(key, {}); rw = sec["rows"].get(key)
        rtxt = f"{rw['disp']:.3f} / {rw['init']:.3f} / {rw['trained']:.3f} ({rw['n_runs']} runs)" if rw else "--"
        L.append(f"| {desc} | {sec['seeds'][key]} | {fmt_ms(c['hit_at_010'])} | {fmt_ms(c['hit_at_025'])} | {fmt_ms(pc.get('click', nanc))} | "
                 f"{fmt_ms(pc.get('scroll', nanc))} | {fmt_delta(sec['deltas_vs_A'][key]['hit_at_010'])} | {fmt_delta(sec['deltas_vs_B'][key]['hit_at_010'])} | {rtxt} |")
    if "base_proxy_notype" in sec["rows"]:
        rw = sec["rows"]["base_proxy_notype"]
        L.append(f"\nBase-LR displacement measured on the _notype D-token runs (click and scroll rows, same LR and code path): "
                 f"{rw['disp']:.3f} from init norm {rw['init']:.3f} ({rw['n_runs']} runs).")
    return "\n".join(L) + "\n"


def tex_dtoken_lr(sec: dict) -> str:
    nanc = {"mean": math.nan, "std": 0, "n_seeds": 0}
    L = ["\\begin{table}[h]", "\\centering", "\\small",
         "\\caption{D-token with the action-embedding table trained at 1$\\times$, 10$\\times$, and 100$\\times$ the base learning rate "
         "on the headline slice. Displacement is the mean distance the rows of the classes present (click, type, scroll) moved from "
         "their $\\mathcal{N}(0, 0.02^2)$ initialization, whose norm is 0.78; the 1$\\times$ value comes from the type-removed runs, "
         "which share the code path and learning rate. $\\Delta$ vs.\\ A uses the shared seeds with an episode-cluster interval.}",
         "\\label{tab:dtoken_lr}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccccc}", "\\toprule",
         "Table LR & seeds & hit@0.10 & hit@0.25 & click hit@0.10 & $\\Delta$hit@0.10 vs.\\ A & seed $p$ & row displacement \\\\", "\\midrule"]
    for key, desc in sec["configs"].items():
        c = sec["cells"][key]; pc = sec["per_class"].get(key, {}); d = sec["deltas_vs_A"][key]["hit_at_010"]
        rw = sec["rows"].get(key) or (sec["rows"].get("base_proxy_notype") if key == "base" else None)
        L.append(f"{desc} & {len(sec['seeds'][key])} & {tex_ms(c['hit_at_010'])} & {tex_ms(c['hit_at_025'])} & {tex_ms(pc.get('click', nanc))} & "
                 f"{tex_delta_cluster(d)} & {tex_seed_p(d)} & {(f'{rw[chr(100)+chr(105)+chr(115)+chr(112)]:.3f}') if rw else '--'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


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
    head = section_variants("all_with_coords", 1200, HEADLINE_SEEDS, ["A", "B", "C", "Dhook", "Dtoken", "Dtext"],
                            "Headline ablation (all_with_coords, n_train=1200, n_val=250)")
    ctrl = section_variants("taps_and_swipes", 1000, CURVE_SEEDS, ["A", "B", "C", "Dhook", "Dtext"],
                            "Control (taps_and_swipes, n_train=1000, n_val=200)")
    scal = section_scaling()
    m2w = section_m2w()
    caus = section_causal()
    e2e = section_e2e()
    notype = section_notype()
    interv = section_interventions()
    dlr = section_dtoken_lr()

    md = ["# Phase 8 results — pre-submission consolidation", "",
          "Generated by `scripts/p8_consolidate.py` from `results/phase4/*.json`. "
          "See `neurips2026/SUBMISSION_NOTES.md` for the experiment plan (E1–E6). "
          "Each delta shows: pooled-unit bootstrap CI and stars / cluster bootstrap over examples (seeds kept together) / EPISODE cluster bootstrap (primary; steps and seeds of an episode kept together) / boundary-episode-excluded / "
          "seed-level paired t-test p (n = shared seeds).", "",
          md_variants(head), md_variants(ctrl), md_notype(notype), md_dtoken_lr(dlr), md_scaling(scal), md_e2e(e2e), md_m2w(m2w), md_causal(caus), md_interv(interv)]
    (P8 / "PHASE8_RESULTS.md").write_text("\n".join(md))
    (P8 / "phase8_summary.json").write_text(json.dumps(strip_runs(
        {"headline": head, "control": ctrl, "notype": notype, "scaling": scal, "e2e": e2e, "m2w": m2w, "causal": caus,
         "interventions": interv, "dtoken_lr": dlr}), indent=2))
    if dlr["cells"]:
        (TABLES / "dtoken_lr.tex").write_text(tex_dtoken_lr(dlr))
    (TABLES / "notype.tex").write_text(tex_notype(notype))
    if interv["variants"]:
        (TABLES / "interv.tex").write_text(tex_interv(interv))
    (TABLES / "headline.tex").write_text(tex_variants(
        head, "tab:headline", "Stage 2 grounding on AITW \\texttt{all\\_with\\_coords} (n$_{\\text{train}}$=1200, n$_{\\text{val}}$=250, "
        f"{head['cells']['A']['hit_at_010']['n_seeds']} seeds). Mean $\\pm$ std over seeds; $\\Delta$ vs.\\ A with a 95\\% episode-cluster bootstrap CI "
        "(stars from the same bootstrap, $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$) and the seed-level paired $t$-test $p$."))
    (TABLES / "control.tex").write_text(tex_variants(
        ctrl, "tab:control", "Control: AITW \\texttt{taps\\_and\\_swipes} (n$_{\\text{train}}$=1000, n$_{\\text{val}}$=200, "
        f"{ctrl['cells']['A']['hit_at_010']['n_seeds']} seeds), a training stream without the degenerate class. Same statistics as Table~\\ref{{tab:headline}}."))
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
