"""Phase 9: consolidate the large-validation-slice runs.

Reads results/phase9/bigval/*.json (pulled from the Modal volume with
`modal volume get stage1-cache bigval results/phase9/`), and writes

    results/phase9/PHASE9_RESULTS.md     every contrast, human-readable
    results/phase9/phase9_summary.json   the same, machine-readable
    aaai27/tables/*.tex                  the tables the paper inputs

Statistics match Phase 8: for each contrast the per-example metric is averaged
over the seeds the two conditions share, then a cluster bootstrap resamples
validation EPISODES (10,000 draws, 95% percentile interval, two-sided p), and a
paired t-test runs on the per-seed means. A difference is "established" only if
both agree (p < .05).

Primary metrics are computed on the click and scroll steps only ("spatial"
steps). A type event's raw target is the off-screen point (-1, -1), so no model
can hit it, and where a model happens to point on a type step only adds noise to
mean distance. Every metric is also reported over all steps.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P9 = ROOT / "results" / "phase9"
RUNS = P9 / "bigval"
TABLES = ROOT / "aaai27" / "tables"
N_BOOT = 10000
RADII = {"hit_at_005": 0.05, "hit_at_010": 0.10, "hit_at_014": 0.14, "hit_at_025": 0.25}
METRICS = list(RADII) + ["mean_l2"]
CLICK, TYPE, SCROLL = 0, 2, 3          # canonical action ids (src/data/taxonomy.py)
NAMES = {"A": "A: flat", "B": "B: aux.\\ loss", "Bprompt": "B$'$: aux.\\ loss, prompt-pooled",
         "C": "C: hard routing", "Dhook": "D-hook: additive", "Dhook10x": "D-hook, table LR $\\times$10",
         "Dtoken10x": "D-token: prepended, table LR $\\times$10", "Dtext": "D-text: prompt word"}
PLAIN = {k: v.replace("\\ ", " ").replace("$'$", "'").replace("$\\times$", "x") for k, v in NAMES.items()}


# ---------------------------------------------------------------- loading
def key_of(r: dict) -> str:
    v = r["variant"]
    if r.get("action_lr") and r["action_lr"] > 0:
        v += "10x"
    return v


def load() -> dict:
    """{stream: {variant_key: {seed: run}}}, stream in {'mixed', 'clean'}."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for p in sorted(RUNS.glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("data_mix") != "all_with_coords":
            continue
        stream = "clean" if r.get("drop_type_from_train") else "mixed"
        out[stream][key_of(r)][r["seed"]] = r
    return out


def check_alignment(runs: dict) -> dict:
    """Every run must be scored on the identical validation slice, in order."""
    keys = None
    for stream in runs.values():
        for by_seed in stream.values():
            for r in by_seed.values():
                k = [(m["ep_id"], m["step_id"]) for m in r["val_meta"]]
                if keys is None:
                    keys, ref = k, r
                assert k == keys, f"{r['run_name']} is scored on a different slice than {ref['run_name']}"
    meta = ref["val_meta"]
    labels = np.array([m["action_id"] for m in meta])
    eps = [m["ep_id"] for m in meta]
    return {"n": len(meta), "episodes": eps, "labels": labels, "n_episodes": len(set(eps)),
            "n_click": int((labels == CLICK).sum()), "n_scroll": int((labels == SCROLL).sum()),
            "n_type": int((labels == TYPE).sum())}


def per_example(r: dict, metric: str, which: str = "gold") -> np.ndarray:
    d = np.asarray(r[which]["per_example_dist"], dtype=float)
    return d if metric == "mean_l2" else (d <= RADII[metric]).astype(float)


# ---------------------------------------------------------------- statistics
def episode_bootstrap(diff: np.ndarray, eps: list, seed: int = 0) -> dict:
    ids = {e: i for i, e in enumerate(dict.fromkeys(eps))}
    idx = np.array([ids[e] for e in eps]); E = len(ids)
    sums = np.bincount(idx, weights=diff, minlength=E); cnts = np.bincount(idx, minlength=E).astype(float)
    pick = np.random.default_rng(seed).integers(0, E, size=(N_BOOT, E))
    boots = sums[pick].sum(1) / cnts[pick].sum(1)
    obs = float(diff.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(min(1.0, 2 * ((boots <= 0).mean() if obs >= 0 else (boots >= 0).mean())))
    return {"ci_low": float(lo), "ci_high": float(hi), "p": p, "n_episodes": E}


def contrast(base: dict, other: dict, metric: str, mask: np.ndarray, eps: list,
             base_which: str = "gold", other_which: str = "gold") -> dict | None:
    """other - base on the seeds both have, restricted to `mask`."""
    seeds = sorted(set(base) & set(other))
    seeds = [s for s in seeds if base[s].get(base_which) and other[s].get(other_which)]
    if not seeds:
        return None
    a = np.stack([per_example(base[s], metric, base_which)[mask] for s in seeds])
    b = np.stack([per_example(other[s], metric, other_which)[mask] for s in seeds])
    diff = (b - a).mean(0)
    sub_eps = [e for e, k in zip(eps, mask) if k]
    out = {"delta": float(diff.mean()), "seeds": seeds, **episode_bootstrap(diff, sub_eps)}
    if len(seeds) >= 3:
        from scipy import stats
        out["seed_p"] = float(stats.ttest_rel(b.mean(1), a.mean(1)).pvalue)
    else:
        out["seed_p"] = float("nan")
    out["established"] = bool(out["p"] < 0.05 and out["seed_p"] < 0.05)
    return out


def cell(by_seed: dict, metric: str, mask: np.ndarray, which: str = "gold") -> dict:
    vals = [float(per_example(r, metric, which)[mask].mean()) for r in by_seed.values() if r.get(which)]
    return {"mean": float(np.mean(vals)) if vals else math.nan,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}


# ---------------------------------------------------------------- formatting
def stars(d: dict | None) -> str:
    if not d:
        return ""
    p = d["p"]
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def tex_ms(c: dict, digits: int = 3) -> str:
    if c["n"] == 0 or math.isnan(c["mean"]):
        return "--"
    return f"{c['mean']:.{digits}f} $\\pm$ {c['std']:.{digits}f}" if c["n"] > 1 else f"{c['mean']:.{digits}f}"


def tex_d(d: dict | None, digits: int = 3) -> str:
    if not d:
        return "--"
    s = stars(d)
    s = f"$^{{{s}}}$" if s else ""
    return f"{d['delta']:+.{digits}f} [{d['ci_low']:+.{digits}f}, {d['ci_high']:+.{digits}f}]{s}"


def tex_p(d: dict | None) -> str:
    if not d or math.isnan(d.get("seed_p", math.nan)):
        return "--"
    return "$<$0.001" if d["seed_p"] < 0.001 else f"{d['seed_p']:.3f}"


def md_d(d: dict | None) -> str:
    if not d:
        return "--"
    est = " **established**" if d["established"] else ""
    return (f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] {stars(d) or 'ns'} "
            f"(E={d['n_episodes']}, seeds={len(d['seeds'])}, seed p={d['seed_p']:.3f}){est}")


# ---------------------------------------------------------------- sections
def main() -> None:
    runs = load()
    if not runs:
        raise SystemExit(f"no runs in {RUNS}")
    sl = check_alignment(runs)
    labels, eps = sl["labels"], sl["episodes"]
    spatial = (labels == CLICK) | (labels == SCROLL)
    allm = np.ones_like(spatial, dtype=bool)
    click, scroll = labels == CLICK, labels == SCROLL
    mixed, clean = runs.get("mixed", {}), runs.get("clean", {})
    S: dict = {"slice": {k: v for k, v in sl.items() if k not in ("episodes", "labels")}}
    md = ["# Phase 9: large held-out validation slice", "",
          f"Slice: {sl['n']} steps from {sl['n_episodes']} episodes ({sl['n_click']} clicks, "
          f"{sl['n_scroll']} scrolls, {sl['n_type']} type events); no episode is shared with training.", "",
          "Primary metrics use the click and scroll steps only; type targets are off-screen and unhittable.", ""]

    # 1. mechanism ranking on the mixed stream
    order = [v for v in ("A", "B", "Bprompt", "C", "Dhook", "Dhook10x", "Dtoken10x", "Dtext") if v in mixed]
    S["mixed"] = {}
    md += ["## 1. Mixed stream (type events in training)", "",
           "| variant | seeds | hit@0.10 | hit@0.25 | mean L2 | Δhit@0.10 vs A | Δhit@0.14 vs A | ΔL2 vs A | Δhit@0.10 vs A (all steps) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for v in order:
        rec = {m: cell(mixed[v], m, spatial) for m in METRICS}
        rec["all"] = {m: cell(mixed[v], m, allm) for m in METRICS}
        rec["click"] = cell(mixed[v], "hit_at_010", click)
        rec["scroll"] = cell(mixed[v], "hit_at_010", scroll)
        if v != "A" and "A" in mixed:
            rec["vsA"] = {m: contrast(mixed["A"], mixed[v], m, spatial, eps) for m in METRICS}
            rec["vsA_all"] = {m: contrast(mixed["A"], mixed[v], m, allm, eps) for m in METRICS}
            rec["vsA_click"] = contrast(mixed["A"], mixed[v], "hit_at_010", click, eps)
            rec["vsA_scroll"] = contrast(mixed["A"], mixed[v], "hit_at_010", scroll, eps)
        if v not in ("A", "B") and "B" in mixed:
            rec["vsB"] = {m: contrast(mixed["B"], mixed[v], m, spatial, eps) for m in ("hit_at_010", "mean_l2")}
        S["mixed"][v] = rec
        va = rec.get("vsA", {})
        md.append(f"| {PLAIN[v]} | {rec['hit_at_010']['n']} | {rec['hit_at_010']['mean']:.3f} | {rec['hit_at_025']['mean']:.3f} | "
                  f"{rec['mean_l2']['mean']:.3f} | {md_d(va.get('hit_at_010'))} | {md_d(va.get('hit_at_014'))} | "
                  f"{md_d(va.get('mean_l2'))} | {md_d(rec.get('vsA_all', {}).get('hit_at_010'))} |")
    md += ["", "Per class, hit@0.10:", ""]
    for v in order:
        r = S["mixed"][v]
        md.append(f"- {PLAIN[v]}: click {r['click']['mean']:.3f}, scroll {r['scroll']['mean']:.3f}"
                  + (f"; click vs A {md_d(r.get('vsA_click'))}; scroll vs A {md_d(r.get('vsA_scroll'))}" if v != "A" else ""))
    for v in ("Bprompt", "Dhook10x", "Dtoken10x", "Dtext", "Dhook", "C"):
        if v in S["mixed"] and "vsB" in S["mixed"][v]:
            md.append(f"- {PLAIN[v]} − B: {md_d(S['mixed'][v]['vsB']['hit_at_010'])}")
    if "Bprompt" in mixed and "B" in mixed:
        S["Bprompt_minus_B"] = {m: contrast(mixed["B"], mixed["Bprompt"], m, spatial, eps) for m in METRICS}
        md.append(f"- B' (prompt-pooled) − B, mean L2: {md_d(S['Bprompt_minus_B']['mean_l2'])}")
    if "Dhook10x" in mixed and "Dhook" in mixed:
        S["Dhook10x_minus_Dhook"] = {m: contrast(mixed["Dhook"], mixed["Dhook10x"], m, spatial, eps) for m in METRICS}
        md.append(f"- D-hook 10x − D-hook: {md_d(S['Dhook10x_minus_Dhook']['hit_at_010'])}")

    # 2. clean stream
    S["clean"] = {}
    if clean:
        md += ["", "## 2. Clean stream (type events removed from training only; same validation slice)", "",
               "| variant | seeds | hit@0.10 | mean L2 | Δhit@0.10 vs A (clean) | ΔL2 vs A (clean) | Δhit@0.25 vs A (clean) | removal effect hit@0.10 | removal effect L2 |",
               "|---|---|---|---|---|---|---|---|---|"]
        for v in [x for x in ("A", "B", "Dhook", "Dtext", "C", "Dtoken10x") if x in clean]:
            rec = {m: cell(clean[v], m, spatial) for m in METRICS}
            rec["click"] = cell(clean[v], "hit_at_010", click)
            rec["scroll"] = cell(clean[v], "hit_at_010", scroll)
            if v != "A" and "A" in clean:
                rec["vsA"] = {m: contrast(clean["A"], clean[v], m, spatial, eps) for m in METRICS}
                rec["vsA_click"] = contrast(clean["A"], clean[v], "hit_at_010", click, eps)
            if v in mixed:
                rec["removal"] = {m: contrast(mixed[v], clean[v], m, spatial, eps) for m in METRICS}
            S["clean"][v] = rec
            va, rm = rec.get("vsA", {}), rec.get("removal", {})
            md.append(f"| {PLAIN[v]} | {rec['hit_at_010']['n']} | {rec['hit_at_010']['mean']:.3f} | {rec['mean_l2']['mean']:.3f} | "
                      f"{md_d(va.get('hit_at_010'))} | {md_d(va.get('mean_l2'))} | {md_d(va.get('hit_at_025'))} | "
                      f"{md_d(rm.get('hit_at_010'))} | {md_d(rm.get('mean_l2'))} |")

    # 3. end to end
    S["e2e"] = {}
    md += ["", "## 3. End to end: Stage-1 predicted types", ""]
    s1acc = [r["stage1"]["accuracy"] for by in mixed.values() for r in by.values() if r.get("stage1")]
    if s1acc:
        maj = [r["stage1"]["majority_baseline"] for by in mixed.values() for r in by.values() if r.get("stage1")]
        S["stage1_accuracy"] = {"mean": float(np.mean(s1acc)), "majority": float(np.mean(maj))}
        md.append(f"Stage-1 accuracy {np.mean(s1acc):.3f} (majority class {np.mean(maj):.3f}).")
        md += ["", "| variant | gold hit@0.10 | predicted hit@0.10 | predicted − A | gold − predicted | predicted − A (L2) |", "|---|---|---|---|---|---|"]
    for v in ("C", "Dhook", "Dtoken10x", "Dtext"):
        if v not in mixed or not any(r.get("e2e") for r in mixed[v].values()):
            continue
        rec = {"gold": cell(mixed[v], "hit_at_010", spatial), "pred": cell(mixed[v], "hit_at_010", spatial, "e2e"),
               "pred_minus_A": contrast(mixed["A"], mixed[v], "hit_at_010", spatial, eps, other_which="e2e"),
               "pred_minus_A_l2": contrast(mixed["A"], mixed[v], "mean_l2", spatial, eps, other_which="e2e"),
               "gold_minus_pred": contrast({s: r for s, r in mixed[v].items() if r.get("e2e")}, mixed[v], "hit_at_010", spatial, eps, base_which="e2e")}
        S["e2e"][v] = rec
        md.append(f"| {PLAIN[v]} | {rec['gold']['mean']:.3f} | {rec['pred']['mean']:.3f} | {md_d(rec['pred_minus_A'])} | "
                  f"{md_d(rec['gold_minus_pred'])} | {md_d(rec['pred_minus_A_l2'])} |")
    if "B" in S["mixed"]:
        md.append(f"| {PLAIN['B']} (needs no type at inference) | {S['mixed']['B']['hit_at_010']['mean']:.3f} | same | "
                  f"{md_d(S['mixed']['B'].get('vsA', {}).get('hit_at_010'))} | 0 | -- |")

    # 4. action-table displacement
    S["rows"] = {}
    for v in ("Dhook", "Dhook10x", "Dtoken10x"):
        if v in mixed:
            disp = [np.mean([x["displacement"] for k, x in r["action_rows"].items() if isinstance(x, dict)])
                    for r in mixed[v].values() if r.get("action_rows")]
            if disp:
                S["rows"][v] = float(np.mean(disp))
    if S["rows"]:
        md += ["", "Action-table row displacement from initialization: " +
               ", ".join(f"{PLAIN[k]} {v:.3f}" for k, v in S["rows"].items())]

    # 5. origin bias (does the flat model pull clicks toward the clamped (0,0)?)
    S["origin_bias"] = {}
    for v in ("A", "B", "Dhook", "Dtext"):
        if v not in mixed:
            continue
        near = []
        for r in mixed[v].values():
            pr = r["gold"]["pred_xy"]
            for i in np.where(click)[0]:
                p = pr[i]
                if p is None:
                    continue
                near.append(math.hypot(p[0], p[1]) <= 0.10)
        S["origin_bias"][v] = {"frac_click_preds_within_0.10_of_origin": float(np.mean(near)) if near else math.nan}
    if S["origin_bias"]:
        md += ["", "Share of click predictions within 0.10 of the origin: " +
               ", ".join(f"{PLAIN[k]} {v['frac_click_preds_within_0.10_of_origin']:.3%}" for k, v in S["origin_bias"].items())]

    # 6. batched-decoding check
    bc = [r["batch_check"] for st in runs.values() for by in st.values() for r in by.values() if r.get("batch_check")]
    if bc:
        S["batch_check"] = {"runs": len(bc), "frac_identical_min": min(b["frac_identical"] for b in bc),
                            "frac_identical_mean": float(np.mean([b["frac_identical"] for b in bc])),
                            "max_hit010_gap": max(abs(b["hit010_batched"] - b["hit010_unbatched"]) for b in bc)}
        md += ["", f"Batched vs one-at-a-time decoding on the first {bc[0]['k']} steps of {len(bc)} runs: "
               f"{S['batch_check']['frac_identical_mean']:.1%} identical outputs on average "
               f"(minimum {S['batch_check']['frac_identical_min']:.1%}), largest hit@0.10 gap {S['batch_check']['max_hit010_gap']:.3f}."]

    # 7. claims table
    claims = []
    def claim(text, d, tex=None):
        if d:
            claims.append({"claim": text, "tex": tex or text, "delta": d["delta"], "ci": [d["ci_low"], d["ci_high"]],
                           "boot_p": d["p"], "seed_p": d["seed_p"], "established": d["established"]})
    for v in ("B", "Dhook", "Dtext", "C", "Dtoken10x", "Bprompt", "Dhook10x"):
        if v in S["mixed"]:
            claim(f"{PLAIN[v]} beats A on the mixed stream (hit@0.10)", S["mixed"][v].get("vsA", {}).get("hit_at_010"),
                  f"{NAMES[v]} beats A, mixed stream")
    if "A" in S["clean"]:
        claim("Removing type events from training improves A (hit@0.10)", S["clean"]["A"].get("removal", {}).get("hit_at_010"),
              "Removing type events improves A")
    for v in ("B", "Dhook", "Dtext"):
        if v in S["clean"]:
            claim(f"{PLAIN[v]} beats A on the clean stream (hit@0.10)", S["clean"][v].get("vsA", {}).get("hit_at_010"),
                  f"{NAMES[v]} beats A, clean stream")
            claim(f"{PLAIN[v]} beats A on the clean stream (mean L2, negative is better)", S["clean"][v].get("vsA", {}).get("mean_l2"),
                  f"{NAMES[v]} shortens misses, clean stream (L2)")
    for v, rec in S["e2e"].items():
        claim(f"{PLAIN[v]} with predicted types beats A (hit@0.10)", rec["pred_minus_A"], f"{NAMES[v]}, predicted types, beats A")
    if "Bprompt_minus_B" in S:
        claim("Prompt-pooled aux loss differs from answer-pooled (hit@0.10)", S["Bprompt_minus_B"]["hit_at_010"],
              "Prompt-pooled B$'$ differs from B")
    S["claims"] = claims
    md += ["", "## Claims", "", "| claim | Δ | 95% CI | boot p | seed p | established |", "|---|---|---|---|---|---|"]
    for c in claims:
        md.append(f"| {c['claim']} | {c['delta']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['boot_p']:.3f} | {c['seed_p']:.3f} | {'yes' if c['established'] else 'no'} |")

    P9.mkdir(parents=True, exist_ok=True)
    (P9 / "PHASE9_RESULTS.md").write_text("\n".join(md) + "\n")
    (P9 / "phase9_summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_tables(S, sl)
    print("\n".join(md))


# ---------------------------------------------------------------- LaTeX tables (two-column AAAI)
def write_tables(S: dict, sl: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    n_note = (f"{sl['n']} held-out steps from {sl['n_episodes']} episodes; metrics on the "
              f"{sl['n_click'] + sl['n_scroll']} click and scroll steps")
    # Table: mechanism ranking
    L = ["\\begin{table*}[t]", "\\centering", "\\small",
         "\\caption{Mechanism ranking on the mixed stream (" + n_note + "). Mean $\\pm$ std over five seeds; "
         "$\\Delta$ vs.\\ A with a 95\\% episode-cluster bootstrap interval (stars from the same bootstrap, "
         "$^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$) and the seed-level paired $t$-test $p$.}",
         "\\label{tab:headline}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccccccc}", "\\toprule",
         "Variant & \\hitat{0.10} & \\hitat{0.25} & mean L2 $\\downarrow$ & $\\Delta$\\hitat{0.10} vs.\\ A & seed $p$ & "
         "$\\Delta$ mean L2 vs.\\ A & seed $p$ \\\\", "\\midrule"]
    for v, r in S["mixed"].items():
        a, l2 = r.get("vsA", {}).get("hit_at_010"), r.get("vsA", {}).get("mean_l2")
        L.append(f"{NAMES[v]} & {tex_ms(r['hit_at_010'])} & {tex_ms(r['hit_at_025'])} & {tex_ms(r['mean_l2'])} & "
                 f"{tex_d(a)} & {tex_p(a)} & {tex_d(l2)} & {tex_p(l2)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    (TABLES / "headline.tex").write_text("\n".join(L) + "\n")

    # Table: clean stream
    if S["clean"]:
        L = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\caption{Type events removed from training only, scored on the identical held-out slice (" + n_note + "). "
             "Left: each variant against the baseline trained the same way. Right: the effect of removing the class on each "
             "variant. Same statistics as Table~\\ref{tab:headline}.}",
             "\\label{tab:clean}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}", "\\toprule",
             "Variant & \\hitat{0.10} & mean L2 $\\downarrow$ & $\\Delta$\\hitat{0.10} vs.\\ A & $\\Delta$ mean L2 vs.\\ A & "
             "removal effect, \\hitat{0.10} & removal effect, mean L2 \\\\", "\\midrule"]
        for v, r in S["clean"].items():
            va, rm = r.get("vsA", {}), r.get("removal", {})
            L.append(f"{NAMES[v]} & {tex_ms(r['hit_at_010'])} & {tex_ms(r['mean_l2'])} & {tex_d(va.get('hit_at_010'))} & "
                     f"{tex_d(va.get('mean_l2'))} & {tex_d(rm.get('hit_at_010'))} & {tex_d(rm.get('mean_l2'))} \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
        (TABLES / "clean.tex").write_text("\n".join(L) + "\n")

    # Table: end to end (single column)
    if S["e2e"]:
        acc = S.get("stage1_accuracy", {})
        L = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\caption{{End to end. Each conditioned model scored with Stage-1 predicted types (accuracy "
             f"{acc.get('mean', float('nan')):.3f}, majority class {acc.get('majority', float('nan')):.3f}) instead of gold; "
             "\\hitat{0.10} on click and scroll steps. B needs no type at inference, so its gold and deployed numbers coincide.}",
             "\\label{tab:e2e}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lccc}", "\\toprule",
             "Variant & gold & predicted & predicted $-$ A \\\\", "\\midrule"]
        for v, r in S["e2e"].items():
            L.append(f"{NAMES[v]} & {r['gold']['mean']:.3f} & {r['pred']['mean']:.3f} & {tex_d(r['pred_minus_A'])} \\\\")
        if "B" in S["mixed"]:
            L.append(f"{NAMES['B']} & {S['mixed']['B']['hit_at_010']['mean']:.3f} & same & "
                     f"{tex_d(S['mixed']['B'].get('vsA', {}).get('hit_at_010'))} \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
        (TABLES / "e2e.tex").write_text("\n".join(L) + "\n")

    # Table: claims (single column)
    if S["claims"]:
        L = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Every quantitative claim, with whether it is established (episode bootstrap and seed-level test both "
             "$p<.05$).}", "\\label{tab:claims}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcc}", "\\toprule",
             "Claim & $\\Delta$ [95\\% CI] & established \\\\", "\\midrule"]
        for c in S["claims"]:
            L.append(f"{c['tex']} & {c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] & {'yes' if c['established'] else 'no'} \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
        (TABLES / "claims.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
