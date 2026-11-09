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
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P9 = ROOT / "results" / "phase9"
RUNS = Path(os.environ.get("P9_RUNS", P9 / "bigval"))          # override only to dry-run on smoke output
CTRL = P9 / "bigval_control"
LEGACY = ROOT / "results" / "phase4"
TABLES = Path(os.environ.get("P9_TABLES", ROOT / "aaai27" / "tables"))
# taps_and_swipes control (n_train=1000, n_val=200, seeds 42-44): the Phase-8 runs, decoded one at a time
CTRL_SEEDS = (42, 43, 44)
CTRL_LEGACY = {"A": "variantA_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes.json",
               "B": "variantB_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes_aux1.0.json",
               "C": "variantC_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes.json",
               "Dhook": "Dhook_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes_init0.0.json",
               "Dtext": "Dtext_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes.json"}
# the retracted D-slot runs: the same prepended-token variant, injected through inputs_embeds (no per-example log)
DSLOT = "train_seed{s}_n1000_ep2_lr2e-05_mix-taps_and_swipes.json"
N_BOOT = 10000
RADII = {"hit_at_005": 0.05, "hit_at_010": 0.10, "hit_at_014": 0.14, "hit_at_025": 0.25}
METRICS = list(RADII) + ["mean_l2"]
CLICK, TYPE, SCROLL = 0, 2, 3          # canonical action ids (src/data/taxonomy.py)
NAMES = {"A": "A: flat", "B": "B: aux.\\ loss", "Bprompt": "B$'$: aux.\\ loss, prompt-pooled",
         "C": "C: hard routing", "Dhook": "D-hook: additive", "Dhook10x": "D-hook, table LR $\\times$10",
         "Dtoken": "D-token: prepended", "Dtoken10x": "D-token: prepended, table LR $\\times$10",
         "Dtext": "D-text: prompt word", "Dslot": "D-slot: prepended, via \\texttt{inputs\\_embeds}"}
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
        if "variant" not in r or r.get("data_mix") != "all_with_coords":   # stage1_preds_* files
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
    prev = json.loads((ROOT / "results" / "phase8" / "val_episodes.json").read_text())["all_with_coords_n1200"]["val"]
    prev_keys = {(x["ep_id"], x["step_id"]) for x in prev}
    fresh = np.array([(m["ep_id"], m["step_id"]) not in prev_keys for m in meta])
    return {"n": len(meta), "episodes": eps, "labels": labels, "n_episodes": len(set(eps)),
            "n_click": int((labels == CLICK).sum()), "n_scroll": int((labels == SCROLL).sum()),
            "n_type": int((labels == TYPE).sum()), "fresh": fresh,
            "n_fresh": int(fresh.sum()), "n_overlap": int((~fresh).sum()),
            "n_fresh_episodes": len({e for e, f in zip(eps, fresh) if f})}


def per_example(r: dict, metric: str, which: str = "gold") -> np.ndarray:
    d = np.asarray(r[which]["per_example_dist"], dtype=float)
    # a prediction off the 0-999 grid (one in the whole matrix: "(532157, 707)") is malformed;
    # score it like an unparseable output, a miss at sqrt 2, instead of letting it dominate mean L2
    xy = r[which].get("pred_xy")
    if xy is not None:
        off = np.array([p is not None and (max(p) > 1.0 + 1e-6 or min(p) < -1e-6) for p in xy])
        if off.any():
            d = d.copy(); d[off] = np.sqrt(2)
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


def dd_contrast(am: dict, vm: dict, ac: dict, vc: dict, metric: str, mask: np.ndarray, eps: list) -> dict | None:
    """(V - A on the mixed stream) - (V - A on the clean stream): how much of the mixed-stream gain removing
    the type events takes away. Same episode bootstrap; seed level is a one-sample t on per-seed values."""
    seeds = sorted(set(am) & set(vm) & set(ac) & set(vc))
    if not seeds:
        return None
    d = np.stack([((per_example(vm[s], metric) - per_example(am[s], metric))
                   - (per_example(vc[s], metric) - per_example(ac[s], metric)))[mask] for s in seeds])
    diff = d.mean(0)
    out = {"delta": float(diff.mean()), "seeds": seeds, **episode_bootstrap(diff, [e for e, k in zip(eps, mask) if k])}
    from scipy import stats
    out["seed_p"] = float(stats.ttest_1samp(d.mean(1), 0.0).pvalue) if len(seeds) >= 3 else float("nan")
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
    s = f"^{{{s}}}" if s else ""
    return f"${d['delta']:+.{digits}f}$ $[{d['ci_low']:+.{digits}f}, {d['ci_high']:+.{digits}f}]{s}$"


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
    fresh = spatial & sl["fresh"]          # spatial steps that were never in the earlier 250-step slice
    oldm = spatial & ~sl["fresh"]          # spatial steps of the earlier slice
    cv = json.loads((ROOT / "results" / "phase8" / "val_episodes.json").read_text())["taps_and_swipes_n1000"]["val"]
    ck = {(x["ep_id"], x["step_id"]) for x in cv}
    ref_meta = next(iter(next(iter(runs.values())).values()))
    ref_meta = next(iter(ref_meta.values()))["val_meta"]
    ctl = spatial & np.array([(m["ep_id"], m["step_id"]) in ck for m in ref_meta])   # control's held-out steps
    S: dict = {"slice": {k: v for k, v in sl.items() if k not in ("episodes", "labels", "fresh")}}
    S["slice"]["n_fresh_spatial"] = int(fresh.sum())
    S["slice"]["n_old_spatial"] = int(oldm.sum())
    S["slice"]["n_old_episodes"] = len({e for e, k in zip(eps, oldm) if k})
    S["slice"]["n_control_in_slice"] = int(ctl.sum())
    md = ["# Phase 9: large held-out validation slice", "",
          f"Slice: {sl['n']} steps from {sl['n_episodes']} episodes ({sl['n_click']} clicks, "
          f"{sl['n_scroll']} scrolls, {sl['n_type']} type events); no episode is shared with training.", "",
          "Primary metrics use the click and scroll steps only; type targets are off-screen and unhittable.",
          f"{sl['n_overlap']} steps are the earlier 250-step slice (minus its boundary episode); "
          f"{int(fresh.sum())} spatial steps from {sl['n_fresh_episodes']} episodes are new ('fresh' columns).", ""]

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
            rec["vsA_fresh"] = {m: contrast(mixed["A"], mixed[v], m, fresh, eps) for m in ("hit_at_010", "mean_l2")}
            rec["vsA_old"] = {m: contrast(mixed["A"], mixed[v], m, oldm, eps) for m in ("hit_at_010", "mean_l2")}
        if v not in ("A", "B") and "B" in mixed:
            rec["vsB"] = {m: contrast(mixed["B"], mixed[v], m, spatial, eps) for m in ("hit_at_010", "mean_l2")}
        S["mixed"][v] = rec
        va = rec.get("vsA", {})
        md.append(f"| {PLAIN[v]} | {rec['hit_at_010']['n']} | {rec['hit_at_010']['mean']:.3f} | {rec['hit_at_025']['mean']:.3f} | "
                  f"{rec['mean_l2']['mean']:.3f} | {md_d(va.get('hit_at_010'))} | {md_d(va.get('hit_at_014'))} | "
                  f"{md_d(va.get('mean_l2'))} | {md_d(rec.get('vsA_all', {}).get('hit_at_010'))} |")
    md += ["", "Fresh steps only (never in the earlier slice) and earlier-slice steps only, Δhit@0.10 vs A:", ""]
    for v in order:
        if "vsA_fresh" in S["mixed"][v]:
            md.append(f"- {PLAIN[v]}: fresh {md_d(S['mixed'][v]['vsA_fresh']['hit_at_010'])}; "
                      f"earlier slice {md_d(S['mixed'][v]['vsA_old']['hit_at_010'])}")
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
    if "Dhook10x" in mixed and "Dtoken10x" in mixed:   # both tables at the same 10x rate
        S["Dhook10x_minus_Dtoken10x"] = {m: contrast(mixed["Dtoken10x"], mixed["Dhook10x"], m, spatial, eps) for m in METRICS}
        md.append(f"- D-hook 10x − D-token 10x: {md_d(S['Dhook10x_minus_Dtoken10x']['hit_at_010'])}")

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
                rec["vsA_fresh"] = {m: contrast(clean["A"], clean[v], m, fresh, eps) for m in ("hit_at_010", "mean_l2")}
                rec["vsA_old"] = {m: contrast(clean["A"], clean[v], m, oldm, eps) for m in ("hit_at_010", "mean_l2")}
                rec["vsA_ctl"] = {m: contrast(clean["A"], clean[v], m, ctl, eps) for m in ("hit_at_010", "mean_l2")}
                if v in mixed and "A" in mixed:
                    rec["shrink"] = {m: dd_contrast(mixed["A"], mixed[v], clean["A"], clean[v], m, spatial, eps)
                                     for m in ("hit_at_010", "mean_l2")}
            if v in mixed:
                rec["removal"] = {m: contrast(mixed[v], clean[v], m, spatial, eps) for m in METRICS}
                rec["removal_fresh"] = {m: contrast(mixed[v], clean[v], m, fresh, eps) for m in ("hit_at_010", "mean_l2")}
            S["clean"][v] = rec
            va, rm = rec.get("vsA", {}), rec.get("removal", {})
            md.append(f"| {PLAIN[v]} | {rec['hit_at_010']['n']} | {rec['hit_at_010']['mean']:.3f} | {rec['mean_l2']['mean']:.3f} | "
                      f"{md_d(va.get('hit_at_010'))} | {md_d(va.get('mean_l2'))} | {md_d(va.get('hit_at_025'))} | "
                      f"{md_d(rm.get('hit_at_010'))} | {md_d(rm.get('mean_l2'))} |")
        md += ["", f"Clean stream by region of the slice, Δhit@0.10 vs A (earlier slice: {int(oldm.sum())} spatial steps; "
               f"the taps-and-swipes control's held-out steps that fall in this slice: {int(ctl.sum())}):", ""]
        for v, rec in S["clean"].items():
            if "vsA_old" in rec:
                md.append(f"- {PLAIN[v]}: fresh {md_d(rec['vsA_fresh']['hit_at_010'])}; earlier slice "
                          f"{md_d(rec['vsA_old']['hit_at_010'])}; control steps {md_d(rec['vsA_ctl']['hit_at_010'])}")
        md += ["", "Share of the mixed-stream gain that removing type events takes away, (V-A mixed) - (V-A clean):", ""]
        for v, rec in S["clean"].items():
            if rec.get("shrink"):
                g = S["mixed"][v]["vsA"]["hit_at_010"]["delta"]
                md.append(f"- {PLAIN[v]}: hit@0.10 {md_d(rec['shrink']['hit_at_010'])} of a mixed gain of {g:+.3f} "
                          f"({rec['shrink']['hit_at_010']['delta'] / g:.0%}); mean L2 {md_d(rec['shrink']['mean_l2'])}")

    # 3. end to end
    S["e2e"] = {}
    md += ["", "## 3. End to end: Stage-1 predicted types", ""]
    s1acc = [r["stage1"]["accuracy"] for by in mixed.values() for r in by.values() if r.get("stage1")]
    if s1acc:
        maj = [r["stage1"]["majority_baseline"] for by in mixed.values() for r in by.values() if r.get("stage1")]
        S["stage1_accuracy"] = {"mean": float(np.mean(s1acc)), "majority": float(np.mean(maj))}
        md.append(f"Stage-1 accuracy {np.mean(s1acc):.3f} (majority class {np.mean(maj):.3f}).")
        md += ["", "| variant | gold hit@0.10 | predicted hit@0.10 | predicted − A | gold − predicted | predicted − A (L2) | predicted − B |", "|---|---|---|---|---|---|---|"]
    for v in ("C", "Dhook", "Dtoken10x", "Dtext"):
        if v not in mixed or not any(r.get("e2e") for r in mixed[v].values()):
            continue
        rec = {"gold": cell(mixed[v], "hit_at_010", spatial), "pred": cell(mixed[v], "hit_at_010", spatial, "e2e"),
               "pred_minus_A": contrast(mixed["A"], mixed[v], "hit_at_010", spatial, eps, other_which="e2e"),
               "pred_minus_A_l2": contrast(mixed["A"], mixed[v], "mean_l2", spatial, eps, other_which="e2e"),
               "pred_minus_B": contrast(mixed["B"], mixed[v], "hit_at_010", spatial, eps, other_which="e2e") if "B" in mixed else None,
               "gold_minus_pred": contrast({s: r for s, r in mixed[v].items() if r.get("e2e")}, mixed[v], "hit_at_010", spatial, eps, base_which="e2e")}
        S["e2e"][v] = rec
        md.append(f"| {PLAIN[v]} | {rec['gold']['mean']:.3f} | {rec['pred']['mean']:.3f} | {md_d(rec['pred_minus_A'])} | "
                  f"{md_d(rec['gold_minus_pred'])} | {md_d(rec['pred_minus_A_l2'])} | {md_d(rec['pred_minus_B'])} |")
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

    # 5. origin bias (does the flat model pull clicks toward the clamped (0,0)?), on both streams
    S["origin_bias"], S["origin_bias_clean"] = {}, {}
    for stream, block, key in (("mixed", mixed, "origin_bias"), ("clean", clean, "origin_bias_clean")):
        for v in [x for x in ("A", "B", "Bprompt", "C", "Dhook", "Dtoken10x", "Dtext") if x in block]:
            near = []
            for r in block[v].values():
                pr = r["gold"]["pred_xy"]
                for i in np.where(click)[0]:
                    p = pr[i]
                    if p is None:
                        continue
                    near.append(math.hypot(p[0], p[1]) <= 0.10)
            S[key][v] = {"frac_click_preds_within_0.10_of_origin": float(np.mean(near)) if near else math.nan}
        if S[key]:
            md += ["", f"Share of click predictions within 0.10 of the origin ({stream} stream): " +
                   ", ".join(f"{PLAIN[k]} {v['frac_click_preds_within_0.10_of_origin']:.1%}" for k, v in S[key].items())]

    # 5b. unparseable outputs (scored as misses at sqrt 2); only hard routing is expected to produce them
    S["unparsed"] = {}
    for v, by in mixed.items():
        u = {c: float(np.mean([np.mean([r["gold"]["pred_xy"][i] is None for i in np.where(m)[0]]) for r in by.values()]))
             for c, m in (("click", click), ("scroll", scroll))}
        if u["click"] > 0 or u["scroll"] > 0:
            S["unparsed"][v] = u
            parsed = [float(np.mean([d <= 0.10 for i, d in enumerate(r["gold"]["per_example_dist"])
                                     if m[i] and r["gold"]["pred_xy"][i] is not None])) for r in by.values()
                      for m in (scroll,)]
            S["unparsed"][v]["scroll_hit010_parsed"] = float(np.mean(parsed))
    if S["unparsed"]:
        md += ["", "Unparseable outputs: " + ", ".join(
            f"{PLAIN[k]} click {x['click']:.1%} / scroll {x['scroll']:.1%} (scroll hit@0.10 on parsed {x['scroll_hit010_parsed']:.3f})"
            for k, x in S["unparsed"].items())]

    # 5c. Stage-1 errors on the spatial steps, and where the end-to-end loss comes from
    s1 = {}
    for p in sorted(RUNS.glob("stage1_preds_seed*_n1200_v2000_mix-all_with_coords.json")):
        r = json.loads(p.read_text())
        assert [tuple(k) for k in r["val_keys"]] == [(m["ep_id"], m["step_id"]) for m in next(iter(mixed["A"].values()))["val_meta"]], p.name
        s1[r["seed"]] = np.asarray(r["pred_action_ids"])
    if s1:
        P = np.stack([s1[s] for s in sorted(s1)])
        S["stage1_err"] = {"macro_f1": float(np.mean([json.loads(p.read_text())["macro_f1"] for p in RUNS.glob("stage1_preds_seed*.json")])),
                           "spatial_err": float((P[:, spatial] != labels[spatial][None]).mean()),
                           "spatial_to_type": float((P[:, spatial] == TYPE).mean()),
                           "click_to_scroll": float((P[:, click] == SCROLL).mean()),
                           "scroll_to_click": float((P[:, scroll] == CLICK).mean()),
                           "type_recall": float((P[:, labels == TYPE] == TYPE).mean())}
        for v, rec in S["e2e"].items():
            right, wrong, wrong_gold = [], [], []
            for s, r in mixed[v].items():
                if not r.get("e2e") or s not in s1:
                    continue
                ok = s1[s] == labels
                de, dg = np.asarray(r["e2e"]["per_example_dist"]), np.asarray(r["gold"]["per_example_dist"])
                right.append(float((de[spatial & ok] <= 0.10).mean()))
                wrong.append(float((de[spatial & ~ok] <= 0.10).mean()))
                wrong_gold.append(float((dg[spatial & ~ok] <= 0.10).mean()))
            rec["on_misclassified"] = {"pred": float(np.mean(wrong)), "gold": float(np.mean(wrong_gold)),
                                       "pred_on_correct": float(np.mean(right))}
        e = S["stage1_err"]
        md += ["", f"Stage-1 on click and scroll steps: {e['spatial_err']:.1%} misclassified, {e['spatial_to_type']:.1%} sent to "
               f"type; clicks called scroll {e['click_to_scroll']:.1%}, scrolls called click {e['scroll_to_click']:.1%}; "
               f"type recall {e['type_recall']:.1%}; macro-F1 {e['macro_f1']:.3f}."]
        for v, rec in S["e2e"].items():
            o = rec["on_misclassified"]
            md.append(f"- {PLAIN[v]}: hit@0.10 on misclassified spatial steps {o['pred']:.3f} with the predicted type vs "
                      f"{o['gold']:.3f} with gold; on correctly classified steps {o['pred_on_correct']:.3f}")

    # 6. batched-decoding check
    bc = [r["batch_check"] for st in runs.values() for by in st.values() for r in by.values() if r.get("batch_check")]
    if bc:
        import re
        def _xy(raw):
            m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", raw or "")
            return (int(m.group(1)) / 1000, int(m.group(2)) / 1000) if m else None
        gaps = []   # distance between the batched and the one-at-a-time prediction, per example
        for b in bc:
            for x, y in zip(b.get("raws_batched", []), b.get("raws_unbatched", [])):
                px, py = _xy(x), _xy(y)
                if px and py:
                    gaps.append(math.hypot(px[0] - py[0], px[1] - py[1]))
        gaps = np.array(gaps)
        S["batch_check"] = {"runs": len(bc), "k": bc[0]["k"], "frac_identical_min": min(b["frac_identical"] for b in bc),
                            "frac_identical_mean": float(np.mean([b["frac_identical"] for b in bc])),
                            "max_hit010_gap": max(abs(b["hit010_batched"] - b["hit010_unbatched"]) for b in bc),
                            "mean_hit010_gap": float(np.mean([b["hit010_batched"] - b["hit010_unbatched"] for b in bc])),
                            "frac_pred_within_0.01": float((gaps <= 0.01).mean()) if gaps.size else math.nan,
                            "frac_pred_within_0.05": float((gaps <= 0.05).mean()) if gaps.size else math.nan,
                            "mean_abs_dist_diff": float(np.mean([b["mean_abs_dist_diff"] for b in bc if b.get("mean_abs_dist_diff") is not None]))}
        b = S["batch_check"]
        md += ["", f"Batched vs one-at-a-time decoding on the first {b['k']} steps of {len(bc)} runs: "
               f"{b['frac_identical_mean']:.1%} identical strings on average (minimum {b['frac_identical_min']:.1%}); "
               f"{b['frac_pred_within_0.01']:.1%} of prediction pairs within 0.01 of each other, {b['frac_pred_within_0.05']:.1%} within 0.05; "
               f"mean |Δ distance to target| {b['mean_abs_dist_diff']:.3f}; hit@0.10 batched minus unbatched {b['mean_hit010_gap']:+.3f} on average, "
               f"largest gap {b['max_hit010_gap']:.3f}."]

    # 7. claims table: every contrast as "first - second", so the sign carries the direction and
    # "established" means the difference (either way) passes both tests
    claims = []
    SH = {"A": "A", "B": "B", "Bprompt": "B$'$", "C": "C", "Dhook": "D-hook", "Dhook10x": "D-hook$\\times$10",
          "Dtoken": "D-token", "Dtoken10x": "D-token$\\times$10", "Dtext": "D-text"}
    SHP = {k: v.replace("$'$", "'").replace("$\\times$", " x") for k, v in SH.items()}
    def claim(group, a, b, d, metric="hit@0.10"):
        if d:
            claims.append({"group": group, "claim": f"{SHP.get(a, a)} - {SHP.get(b, b)}, {group} ({metric})",
                           "tex": f"{SH.get(a, a)} $-$ {SH.get(b, b)}" + (" (L2)" if metric != "hit@0.10" else ""),
                           "delta": d["delta"], "ci": [d["ci_low"], d["ci_high"]],
                           "boot_p": d["p"], "seed_p": d["seed_p"], "established": d["established"]})
    G = "mixed stream"
    for v in ("B", "Bprompt", "C", "Dhook", "Dhook10x", "Dtoken10x", "Dtext"):
        if v in S["mixed"]:
            claim(G, v, "A", S["mixed"][v].get("vsA", {}).get("hit_at_010"))
    for v in ("Dhook", "Dtext", "Dtoken10x", "Bprompt"):
        if v in S["mixed"] and S["mixed"][v].get("vsB"):
            claim(G, v, "B", S["mixed"][v]["vsB"]["hit_at_010"])
    if "Dhook10x_minus_Dhook" in S:
        claim(G, "Dhook10x", "Dhook", S["Dhook10x_minus_Dhook"]["hit_at_010"])
    if "Dhook10x_minus_Dtoken10x" in S:
        claim(G, "Dhook10x", "Dtoken10x", S["Dhook10x_minus_Dtoken10x"]["hit_at_010"])
    G = "mixed stream, new steps only"
    for v in ("B", "Dhook", "Dtoken10x", "Dtext"):
        if v in S["mixed"] and S["mixed"][v].get("vsA_fresh"):
            claim(G, v, "A", S["mixed"][v]["vsA_fresh"]["hit_at_010"])
    G = "clean stream"
    if "A" in S["clean"]:
        d = S["clean"]["A"].get("removal", {}).get("hit_at_010")
        if d:
            claims.append({"group": G, "claim": "A clean - A mixed (hit@0.10)", "tex": "A clean $-$ A mixed",
                           "delta": d["delta"], "ci": [d["ci_low"], d["ci_high"]], "boot_p": d["p"],
                           "seed_p": d["seed_p"], "established": d["established"]})
    for v in ("B", "Dhook", "Dtext"):
        if v in S["clean"]:
            claim(G, v, "A", S["clean"][v].get("vsA", {}).get("hit_at_010"))
    for v in ("B", "Dhook", "Dtext"):
        if v in S["clean"]:
            claim(G, v, "A", S["clean"][v].get("vsA", {}).get("mean_l2"), metric="mean L2")
    G = "clean stream, new steps only"
    for v in ("B", "Dhook", "Dtext"):
        if v in S["clean"] and S["clean"][v].get("vsA_fresh"):
            claim(G, v, "A", S["clean"][v]["vsA_fresh"]["hit_at_010"])
    G = "predicted types"
    for v, rec in S["e2e"].items():
        claim(G, v, "A", rec["pred_minus_A"])
    for v, rec in S["e2e"].items():
        claim(G, v, "B", rec.get("pred_minus_B"))
    # 8. control stream: D-token on the fixed path, paired with the legacy control runs
    md += control_section(S)
    G = "taps-and-swipes control"
    for v in ("B", "C", "Dhook", "Dtoken", "Dtext"):
        if v in S.get("control", {}):
            claim(G, v, "A", S["control"][v]["vsA"]["hit_at_010"])

    S["claims"] = claims
    md += ["", "## Claims", "", "| claim | Δ | 95% CI | boot p | seed p | established |", "|---|---|---|---|---|---|"]
    for c in claims:
        md.append(f"| {c['claim']} | {c['delta']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['boot_p']:.3f} | {c['seed_p']:.3f} | {'yes' if c['established'] else 'no'} |")

    md += ["", "## Reproduction", "",
           "```bash",
           "uv run modal deploy modal_app.py",
           "uv run python scripts/p9_launch.py stage1",
           "uv run python scripts/p9_launch.py full          # spawns every run on the deployed app",
           "uv run python scripts/p9_status.py               # per-call status",
           "cd results/phase9 && uv run modal volume get --force stage1-cache bigval . && "
           "uv run modal volume get --force stage1-cache bigval_control . && cd ../..",
           "uv run python scripts/p9_consolidate.py          # this file, phase9_summary.json, aaai27/tables/*.tex",
           "```", "",
           "Files: `bigval/<variant>[_alr0.0002][_notype]_seed<s>_n1200_v2000_mix-all_with_coords.json` hold per-example "
           "distances, predictions and raw outputs for gold (and, where run, Stage-1 predicted) types; "
           "`bigval/stage1_preds_seed<s>_*.json` the Stage-1 predictions; `bigval_control/` the taps-and-swipes D-token runs."]
    P9.mkdir(parents=True, exist_ok=True)
    (P9 / "PHASE9_RESULTS.md").write_text("\n".join(md) + "\n")
    (P9 / "phase9_summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_tables(S, sl)
    write_macros(S, sl)
    print("\n".join(md))


# ---------------------------------------------------------------- control stream
def control_section(S: dict) -> list[str]:
    """taps_and_swipes control. The legacy A/B/C/D-hook/D-text runs and the new fixed-path D-token runs are
    scored on the same 200 steps (asserted), so contrasts are paired per example; the retracted D-slot runs
    logged only aggregates, so the same-variant comparison with them is seed-level."""
    v_ep = json.loads((ROOT / "results" / "phase8" / "val_episodes.json").read_text())["taps_and_swipes_n1000"]["val"]
    eps = [x["ep_id"] for x in v_ep]
    keys = [(x["ep_id"], x["step_id"]) for x in v_ep]
    runs: dict = {}
    for v, t in CTRL_LEGACY.items():
        by = {}
        for s in CTRL_SEEDS:
            p = LEGACY / t.format(s=s)
            if p.exists():
                fvm = json.loads(p.read_text()).get("final_val_metrics") or {}
                if fvm.get("per_example_dist"):
                    assert len(fvm["per_example_dist"]) == len(keys), p.name
                    by[s] = {"gold": {"per_example_dist": fvm["per_example_dist"]}}
        if by:
            runs[v] = by
    new = {}
    for p in sorted(CTRL.glob("Dtoken_seed*_n1000_v200_mix-taps_and_swipes*.json")):
        r = json.loads(p.read_text())
        if r.get("action_lr"):
            continue
        k = [(m["ep_id"], m["step_id"]) for m in r["val_meta"]]
        assert k == keys, f"{p.name}: control slice does not match the legacy control slice"
        new[r["seed"]] = r
    if new:
        runs["Dtoken"] = new
    if "A" not in runs:
        return []
    allm = np.ones(len(keys), dtype=bool)
    S["control"] = {}
    md = ["", "## Control: taps_and_swipes (n_train=1000, n_val=200, seeds 42-44)", "",
          "| variant | seeds | hit@0.10 | mean L2 | Δhit@0.10 vs A | ΔL2 vs A |", "|---|---|---|---|---|---|"]
    for v in [x for x in ("A", "B", "C", "Dhook", "Dtoken", "Dtext") if x in runs]:
        rec = {m: cell(runs[v], m, allm) for m in METRICS}
        if v != "A":
            rec["vsA"] = {m: contrast(runs["A"], runs[v], m, allm, eps) for m in METRICS}
        S["control"][v] = rec
        va = rec.get("vsA", {})
        md.append(f"| {PLAIN.get(v, v)} | {rec['hit_at_010']['n']} | {rec['hit_at_010']['mean']:.3f} | {rec['mean_l2']['mean']:.3f} | "
                  f"{md_d(va.get('hit_at_010'))} | {md_d(va.get('mean_l2'))} |")
    # same variant, two injection paths
    slot = {}
    for s in CTRL_SEEDS:
        p = LEGACY / DSLOT.format(s=s)
        if p.exists():
            fvm = json.loads(p.read_text())["final_val_metrics"]
            slot[s] = {"hit_at_010": fvm["hit_at_010"], "mean_l2": fvm["mean_normalized_l2"]}
    if slot:
        vals = [x["hit_at_010"] for x in slot.values()]
        S["control_dslot"] = {"hit_at_010": {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "n": len(vals)},
                              "mean_l2": {"mean": float(np.mean([x["mean_l2"] for x in slot.values()]))}}
    if slot and new:
        from scipy import stats
        seeds = sorted(set(slot) & set(new))
        out = {}
        for m in ("hit_at_010", "mean_l2"):
            fixed = [float(per_example(new[s], m).mean()) for s in seeds]
            old = [slot[s][m] for s in seeds]
            out[m] = {"delta": float(np.mean(fixed) - np.mean(old)), "seeds": seeds,
                      "per_seed_fixed": fixed, "per_seed_slot": old,
                      "seed_p": float(stats.ttest_rel(fixed, old).pvalue) if len(seeds) >= 2 else math.nan}
        S["control_fixed_minus_slot"] = out
        d = out["hit_at_010"]
        md.append(f"- Same prepended-token variant, input_ids path minus inputs_embeds path: {d['delta']:+.3f} hit@0.10 "
                  f"(seed-level paired t p={d['seed_p']:.3f}; per seed fixed {[round(x, 3) for x in d['per_seed_fixed']]}, "
                  f"slot {[round(x, 3) for x in d['per_seed_slot']]})")
    return md


# ---------------------------------------------------------------- macros for the paper text
def _mname(*parts: str) -> str:
    s = "".join(parts).replace("10x", "Ten").replace("Bprompt", "Bp")
    s = "".join(ch for ch in s if ch.isalpha())
    return "\\n" + s


def _fmt(x: float, digits: int = 3, sign: bool = False) -> str:
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _pfmt(p: float) -> str:
    if p is None or math.isnan(p):
        return "p = \\text{n/a}"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


MCODE = {"hit_at_005": "Hfive", "hit_at_010": "Hten", "hit_at_014": "Hfourteen", "hit_at_025": "Hquarter", "mean_l2": "Ltwo"}


def write_macros(S: dict, sl: dict) -> None:
    """aaai27/tables/numbers.tex: one \\newcommand per number the text cites, so prose and tables cannot drift.
    For a contrast X: \\nX (delta), \\nXci (interval), \\nXp (seed-level p, as a relation), \\nXbp (bootstrap p)."""
    L = ["% generated by scripts/p9_consolidate.py; do not edit"]

    def put(name: str, val: str) -> None:
        L.append(f"\\newcommand{{{name}}}{{\\ensuremath{{{val}}}}}")

    def put_contrast(name: str, d: dict | None) -> None:
        if not d:
            return
        put(name, _fmt(d["delta"], sign=True))
        put(name + "ci", f"[{_fmt(d['ci_low'], sign=True)}, {_fmt(d['ci_high'], sign=True)}]")
        put(name + "p", _pfmt(d.get("seed_p", math.nan)))
        put(name + "bp", _pfmt(d.get("p", math.nan)))

    def put_cell(name: str, c: dict) -> None:
        if c and c.get("n"):
            put(name, _fmt(c["mean"]))
            put(name + "sd", f"{_fmt(c['mean'])} \\pm {_fmt(c['std'])}")

    n = lambda x: f"{x:,}".replace(",", "{,}")
    sl2 = S["slice"]
    put("\\nSliceN", n(sl2["n"])); put("\\nSliceEps", n(sl2["n_episodes"]))
    put("\\nSliceClick", n(sl2["n_click"])); put("\\nSliceScroll", n(sl2["n_scroll"]))
    put("\\nSliceType", n(sl2["n_type"])); put("\\nSliceSpatial", n(sl2["n_click"] + sl2["n_scroll"]))
    put("\\nSliceOverlap", n(sl2["n_overlap"])); put("\\nSliceFresh", n(sl2["n_fresh_spatial"]))
    put("\\nSliceFreshEps", n(sl2["n_fresh_episodes"]))
    put("\\nSliceOldSpatial", n(sl2["n_old_spatial"])); put("\\nSliceOldEps", n(sl2["n_old_episodes"]))
    put("\\nCtlInSlice", n(sl2["n_control_in_slice"]))
    for prefix, block in (("", S.get("mixed", {})), ("c", S.get("clean", {})), ("k", S.get("control", {}))):
        for v, r in block.items():
            for m, code in MCODE.items():
                put_cell(_mname(prefix, v, code), r.get(m))
                put_contrast(_mname(prefix, v, "vA", code), r.get("vsA", {}).get(m))
            for cls in ("click", "scroll"):
                put_cell(_mname(prefix, v, cls), r.get(cls))
                put_contrast(_mname(prefix, v, "vA", cls), r.get(f"vsA_{cls}"))
            for m in ("hit_at_010", "mean_l2"):
                put_contrast(_mname(prefix, v, "vB", MCODE[m]), r.get("vsB", {}).get(m))
            for m, code in MCODE.items():
                put_contrast(_mname(prefix, v, "removal", code), r.get("removal", {}).get(m))
            for m in ("hit_at_010", "mean_l2"):
                put_contrast(_mname(prefix, v, "vAold", MCODE[m]), r.get("vsA_old", {}).get(m))
                put_contrast(_mname(prefix, v, "vActl", MCODE[m]), r.get("vsA_ctl", {}).get(m))
                put_contrast(_mname(prefix, v, "shrink", MCODE[m]), (r.get("shrink") or {}).get(m))
            if r.get("shrink") and r["shrink"].get("hit_at_010") and v in S.get("mixed", {}):
                put(_mname(prefix, v, "shrinkpct"), f"{100 * r['shrink']['hit_at_010']['delta'] / S['mixed'][v]['vsA']['hit_at_010']['delta']:.0f}\\%")
            for m in ("hit_at_010", "mean_l2"):
                put_contrast(_mname(prefix, v, "vAfresh", MCODE[m]), r.get("vsA_fresh", {}).get(m))
                put_contrast(_mname(prefix, v, "removalfresh", MCODE[m]), r.get("removal_fresh", {}).get(m))
    for key, nm in (("Bprompt_minus_B", "BpminusB"), ("Dhook10x_minus_Dhook", "DhookTenminusDhook"),
                     ("Dhook10x_minus_Dtoken10x", "DhookTenminusDtokenTen")):
        for m, code in MCODE.items():
            put_contrast(_mname(nm, code), S.get(key, {}).get(m))
    for v, r in S.get("e2e", {}).items():
        put_cell(_mname("e", v, "gold"), r["gold"]); put_cell(_mname("e", v, "pred"), r["pred"])
        put_contrast(_mname("e", v, "vA"), r["pred_minus_A"]); put_contrast(_mname("e", v, "vAL"), r["pred_minus_A_l2"])
        put_contrast(_mname("e", v, "gap"), r["gold_minus_pred"]); put_contrast(_mname("e", v, "vB"), r.get("pred_minus_B"))
        if r.get("pred_minus_B"):
            put(_mname("e", v, "vBabs"), _fmt(abs(r["pred_minus_B"]["delta"])))
    if "stage1_accuracy" in S:
        put("\\nSacc", _fmt(S["stage1_accuracy"]["mean"])); put("\\nSmaj", _fmt(S["stage1_accuracy"]["majority"]))
    for v, x in S.get("rows", {}).items():
        put(_mname("Rows", v), _fmt(x))
    for v, x in S.get("origin_bias", {}).items():
        put(_mname("Orig", v), f"{100 * x['frac_click_preds_within_0.10_of_origin']:.1f}\\%")
    for v, x in S.get("origin_bias_clean", {}).items():
        put(_mname("cOrig", v), f"{100 * x['frac_click_preds_within_0.10_of_origin']:.1f}\\%")
    for v, x in S.get("unparsed", {}).items():
        put(_mname("Unparsed", v, "click"), f"{100 * x['click']:.1f}\\%"); put(_mname("Unparsed", v, "scroll"), f"{100 * x['scroll']:.1f}\\%")
        put(_mname("Unparsed", v, "scrollparsed"), _fmt(x["scroll_hit010_parsed"]))
    if "stage1_err" in S:
        e = S["stage1_err"]
        put("\\nSfone", _fmt(e["macro_f1"])); put("\\nSerr", f"{100 * e['spatial_err']:.1f}\\%")
        put("\\nStotype", f"{100 * e['spatial_to_type']:.1f}\\%"); put("\\nStyperecall", f"{100 * e['type_recall']:.0f}\\%")
        put("\\nSclicktoscroll", f"{100 * e['click_to_scroll']:.1f}\\%"); put("\\nSscrolltoclick", f"{100 * e['scroll_to_click']:.1f}\\%")
    for v, r in S.get("e2e", {}).items():
        if "on_misclassified" in r:
            o = r["on_misclassified"]
            put(_mname("e", v, "wrongpred"), _fmt(o["pred"])); put(_mname("e", v, "wronggold"), _fmt(o["gold"]))
            put(_mname("e", v, "rightpred"), _fmt(o["pred_on_correct"]))
    if "batch_check" in S:
        b = S["batch_check"]
        put("\\nBatchRuns", str(b["runs"])); put("\\nBatchIdent", f"{100 * b['frac_identical_mean']:.0f}\\%")
        put("\\nBatchIdentMin", f"{100 * b['frac_identical_min']:.0f}\\%"); put("\\nBatchGap", _fmt(b["max_hit010_gap"]))
        put("\\nBatchK", str(b["k"])); put("\\nBatchMeanGap", _fmt(b["mean_hit010_gap"], sign=True))
        put("\\nBatchNearOne", f"{100 * b['frac_pred_within_0.01']:.0f}\\%"); put("\\nBatchNearFive", f"{100 * b['frac_pred_within_0.05']:.0f}\\%")
        put("\\nBatchDist", _fmt(b["mean_abs_dist_diff"]))
    if "control_dslot" in S:
        put("\\nkDslotHten", _fmt(S["control_dslot"]["hit_at_010"]["mean"]))
        put("\\nkDslotHtensd", f"{_fmt(S['control_dslot']['hit_at_010']['mean'])} \\pm {_fmt(S['control_dslot']['hit_at_010']['std'])}")
    if "control_fixed_minus_slot" in S:
        d = S["control_fixed_minus_slot"]["hit_at_010"]
        put("\\nkFixedMinusSlot", _fmt(d["delta"], sign=True)); put("\\nkFixedMinusSlotp", _pfmt(d["seed_p"]))
    # duplicate names would be a LaTeX error; fail here instead
    names = [l.split("}{")[0] for l in L[1:]]
    dup = sorted({x for x in names if names.count(x) > 1})
    assert not dup, f"duplicate macro names: {dup[:10]}"
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "numbers.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- LaTeX tables (two-column AAAI)
def write_tables(S: dict, sl: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    c = lambda x: f"{x:,}".replace(",", "{,}")
    n_note = (f"{c(sl['n'])} held-out steps from {sl['n_episodes']} episodes; metrics on the "
              f"{c(sl['n_click'] + sl['n_scroll'])} click and scroll steps")
    # Table: mechanism ranking
    L = ["\\begin{table*}[t]", "\\centering", "\\small",
         "\\caption{Mechanism ranking on the mixed stream (" + n_note + "). Mean $\\pm$ std over five seeds; "
         "$\\Delta$ vs.\\ A with a 95\\% episode-cluster bootstrap interval (stars from the same bootstrap, "
         "$^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$) and the seed-level paired $t$-test $p$. \\emph{New steps}: the "
         + f"{S['slice']['n_fresh_spatial']:,}".replace(",", "{,}") + " click and scroll steps that were never in the earlier 250-step slice.}",
         "\\label{tab:headline}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccccc}", "\\toprule",
         "Variant & \\hitat{0.10} & \\hitat{0.25} & mean L2 $\\downarrow$ & $\\Delta$\\hitat{0.10} vs.\\ A & seed $p$ & "
         "$\\Delta$\\hitat{0.10}, new steps & $\\Delta$ mean L2 vs.\\ A & seed $p$ \\\\", "\\midrule"]
    for v, r in S["mixed"].items():
        a, l2 = r.get("vsA", {}).get("hit_at_010"), r.get("vsA", {}).get("mean_l2")
        fr = r.get("vsA_fresh", {}).get("hit_at_010")
        L.append(f"{NAMES[v]} & {tex_ms(r['hit_at_010'])} & {tex_ms(r['hit_at_025'])} & {tex_ms(r['mean_l2'])} & "
                 f"{tex_d(a)} & {tex_p(a)} & {tex_d(fr)} & {tex_d(l2)} & {tex_p(l2)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    (TABLES / "headline.tex").write_text("\n".join(L) + "\n")

    # Appendix table: the other radii, and every metric over all steps including the unhittable type events
    L = ["\\begin{table*}[t]", "\\centering", "\\small",
         "\\caption{The mixed stream on every metric. Left: click and scroll steps, at the two radii not in "
         "Table~\\ref{tab:headline} (0.14 is the tap threshold of AITW's action matcher). Right: all "
         + f"{sl['n']:,}".replace(",", "{,}") + " held-out steps, type events included; no model can hit a type "
         "event's off-screen target, so these rates are lower for every model. Same statistics as Table~\\ref{tab:headline}.}",
         "\\label{tab:allsteps}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}", "\\toprule",
         "& \\multicolumn{3}{c}{click and scroll steps} & \\multicolumn{3}{c}{all steps} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Variant & \\hitat{0.05} & \\hitat{0.14} & $\\Delta$\\hitat{0.14} vs.\\ A & \\hitat{0.10} & mean L2 $\\downarrow$ & "
         "$\\Delta$\\hitat{0.10} vs.\\ A \\\\", "\\midrule"]
    for v, r in S["mixed"].items():
        L.append(f"{NAMES[v]} & {tex_ms(r['hit_at_005'])} & {tex_ms(r['hit_at_014'])} & {tex_d(r.get('vsA', {}).get('hit_at_014'))} & "
                 f"{tex_ms(r['all']['hit_at_010'])} & {tex_ms(r['all']['mean_l2'])} & {tex_d(r.get('vsA_all', {}).get('hit_at_010'))} \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    (TABLES / "allsteps.tex").write_text("\n".join(L) + "\n")

    # Table: clean stream
    if S["clean"]:
        L = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\caption{Type events removed from training only, scored on the identical held-out slice (" + n_note + "). "
             "Each variant against the baseline trained the same way, on all click and scroll steps, on the new steps, and on the "
             + c(S["slice"]["n_old_spatial"]) + " steps of the earlier slice; last column, the effect on each variant of removing "
             "the class (clean minus mixed). Same statistics as Table~\\ref{tab:headline}.}",
             "\\label{tab:clean}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccccccc}", "\\toprule",
             "& & & \\multicolumn{3}{c}{$\\Delta$\\hitat{0.10} vs.\\ A} & & \\\\", "\\cmidrule(lr){4-6}",
             "Variant & \\hitat{0.10} & mean L2 $\\downarrow$ & all & new steps & earlier steps & $\\Delta$ mean L2 vs.\\ A & "
             "removal effect, \\hitat{0.10} \\\\", "\\midrule"]
        for v, r in S["clean"].items():
            va, rm = r.get("vsA", {}), r.get("removal", {})
            L.append(f"{NAMES[v]} & {tex_ms(r['hit_at_010'])} & {tex_ms(r['mean_l2'])} & {tex_d(va.get('hit_at_010'))} & "
                     f"{tex_d(r.get('vsA_fresh', {}).get('hit_at_010'))} & {tex_d(r.get('vsA_old', {}).get('hit_at_010'))} & "
                     f"{tex_d(va.get('mean_l2'))} & {tex_d(rm.get('hit_at_010'))} \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
        (TABLES / "clean.tex").write_text("\n".join(L) + "\n")

    # Table: end to end (single column)
    if S["e2e"]:
        acc = S.get("stage1_accuracy", {})
        L = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\caption{{End to end. Each conditioned model scored with Stage-1 predicted types (accuracy "
             f"{acc.get('mean', float('nan')):.3f}, majority class {acc.get('majority', float('nan')):.3f}) instead of gold; "
             "\\hitat{0.10} on click and scroll steps. B needs no type at inference, so its gold and deployed numbers coincide.}",
             "\\label{tab:e2e}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcccc}", "\\toprule",
             "Variant & gold & predicted & predicted $-$ A & predicted $-$ B \\\\", "\\midrule"]
        for v, r in S["e2e"].items():
            L.append(f"{NAMES[v]} & {r['gold']['mean']:.3f} & {r['pred']['mean']:.3f} & {tex_d(r['pred_minus_A'])} & "
                     f"{tex_d(r.get('pred_minus_B'))} \\\\")
        if "B" in S["mixed"]:
            L.append(f"{NAMES['B']} & {S['mixed']['B']['hit_at_010']['mean']:.3f} & same & "
                     f"{tex_d(S['mixed']['B'].get('vsA', {}).get('hit_at_010'))} & -- \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
        (TABLES / "e2e.tex").write_text("\n".join(L) + "\n")

    # Table: per class on the mixed stream (single column)
    if S["mixed"]:
        L = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Where the mixed-stream gain lives: \\hitat{0.10} by class, and the change against A with its "
             "episode-cluster interval. Every model scores zero on the type events, whose raw target is off-screen.}",
             "\\label{tab:perclass}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcccc}", "\\toprule",
             f"Variant & click ({sl['n_click']:,}) & scroll ({sl['n_scroll']:,}) & $\\Delta$ click & $\\Delta$ scroll \\\\".replace(",", "{,}"),
             "\\midrule"]
        for v, r in S["mixed"].items():
            L.append(f"{NAMES[v]} & {r['click']['mean']:.3f} & {r['scroll']['mean']:.3f} & "
                     f"{tex_d(r.get('vsA_click'))} & {tex_d(r.get('vsA_scroll'))} \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
        (TABLES / "perclass.tex").write_text("\n".join(L) + "\n")

    # Table: control stream with the fixed-path D-token row
    if S.get("control"):
        L = ["\\begin{table*}[t]", "\\centering", "\\small",
             "\\caption{Control: AITW \\mix{taps\\_and\\_swipes}, a training stream with no degenerate class (1,000 training "
             "and 200 held-out steps, three seeds, every model decoded one example at a time). D-token is new here and "
             "uses the shared learning rate; the D-slot row is the same variant with its embedding injected through "
             "\\texttt{inputs\\_embeds} (Section~\\ref{sec:mrope}), which logged only aggregates. Same statistics as "
             "Table~\\ref{tab:headline}.}",
             "\\label{tab:control}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}", "\\toprule",
             "Variant & \\hitat{0.05} & \\hitat{0.10} & \\hitat{0.25} & mean L2 $\\downarrow$ & "
             "$\\Delta$\\hitat{0.10} vs.\\ A & seed $p$ \\\\", "\\midrule"]
        for v, r in S["control"].items():
            a = r.get("vsA", {}).get("hit_at_010")
            L.append(f"{NAMES[v]} & {tex_ms(r['hit_at_005'])} & {tex_ms(r['hit_at_010'])} & {tex_ms(r['hit_at_025'])} & "
                     f"{tex_ms(r['mean_l2'])} & {tex_d(a)} & {tex_p(a)} \\\\")
        if "control_dslot" in S:
            c = S["control_dslot"]["hit_at_010"]
            L += ["\\midrule", f"{NAMES['Dslot']} & -- & {tex_ms(c)} & -- & {S['control_dslot']['mean_l2']['mean']:.3f} & -- & -- \\\\"]
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
        (TABLES / "control.tex").write_text("\n".join(L) + "\n")

    # Table: claims (single column), grouped by stream
    if S["claims"]:
        L = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Every contrast the paper reports, \\hitat{0.10} unless marked L2 (mean distance, lower is better). "
             "$\\Delta$ is the first model minus the second, with the episode-cluster interval and the seed-level $p$; "
             "\\emph{established} means both tests give $p<.05$, in either direction. Predicted-type rows use Stage-1 types "
             "for the first model only.}", "\\label{tab:claims}",
             "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lccc}", "\\toprule",
             "Contrast & $\\Delta$ [95\\% CI] & seed $p$ & established \\\\"]
        group = None
        for c in S["claims"]:
            if c["group"] != group:
                group = c["group"]
                L += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\emph{{{group[0].upper() + group[1:]}}}}} \\\\"]
            sp = "$<$0.001" if c["seed_p"] < 0.001 else f"{c['seed_p']:.3f}"
            L.append(f"\\quad {c['tex']} & ${c['delta']:+.3f}$ $[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]$ & {sp} & "
                     f"{'yes' if c['established'] else 'no'} \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
        (TABLES / "claims.tex").write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
