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
    S: dict = {"slice": {k: v for k, v in sl.items() if k not in ("episodes", "labels", "fresh")}}
    S["slice"]["n_fresh_spatial"] = int(fresh.sum())
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
        if v not in ("A", "B") and "B" in mixed:
            rec["vsB"] = {m: contrast(mixed["B"], mixed[v], m, spatial, eps) for m in ("hit_at_010", "mean_l2")}
        S["mixed"][v] = rec
        va = rec.get("vsA", {})
        md.append(f"| {PLAIN[v]} | {rec['hit_at_010']['n']} | {rec['hit_at_010']['mean']:.3f} | {rec['hit_at_025']['mean']:.3f} | "
                  f"{rec['mean_l2']['mean']:.3f} | {md_d(va.get('hit_at_010'))} | {md_d(va.get('hit_at_014'))} | "
                  f"{md_d(va.get('mean_l2'))} | {md_d(rec.get('vsA_all', {}).get('hit_at_010'))} |")
    md += ["", "Fresh steps only (never in the earlier slice), Δhit@0.10 vs A:", ""]
    for v in order:
        if "vsA_fresh" in S["mixed"][v]:
            md.append(f"- {PLAIN[v]}: {md_d(S['mixed'][v]['vsA_fresh']['hit_at_010'])}")
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
                rec["vsA_fresh"] = {m: contrast(clean["A"], clean[v], m, fresh, eps) for m in ("hit_at_010", "mean_l2")}
            if v in mixed:
                rec["removal"] = {m: contrast(mixed[v], clean[v], m, spatial, eps) for m in METRICS}
                rec["removal_fresh"] = {m: contrast(mixed[v], clean[v], m, fresh, eps) for m in ("hit_at_010", "mean_l2")}
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
    for v in ("B", "Dhook", "Dtext"):
        if v in S["mixed"] and S["mixed"][v].get("vsA_fresh"):
            claim(f"{PLAIN[v]} beats A on the mixed stream, fresh steps only (hit@0.10)", S["mixed"][v]["vsA_fresh"]["hit_at_010"],
                  f"{NAMES[v]} beats A, mixed, new steps only")
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
    # 8. control stream: D-token on the fixed path, paired with the legacy control runs
    md += control_section(S)
    if "Dtoken" in S.get("control", {}):
        claim("D-token (shared LR) beats A on taps_and_swipes (hit@0.10)", S["control"]["Dtoken"]["vsA"]["hit_at_010"],
              "D-token beats A, taps-and-swipes control")

    S["claims"] = claims
    md += ["", "## Claims", "", "| claim | Δ | 95% CI | boot p | seed p | established |", "|---|---|---|---|---|---|"]
    for c in claims:
        md.append(f"| {c['claim']} | {c['delta']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['boot_p']:.3f} | {c['seed_p']:.3f} | {'yes' if c['established'] else 'no'} |")

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
                put_contrast(_mname(prefix, v, "vAfresh", MCODE[m]), r.get("vsA_fresh", {}).get(m))
                put_contrast(_mname(prefix, v, "removalfresh", MCODE[m]), r.get("removal_fresh", {}).get(m))
    for key, nm in (("Bprompt_minus_B", "BpminusB"), ("Dhook10x_minus_Dhook", "DhookTenminusDhook")):
        for m, code in MCODE.items():
            put_contrast(_mname(nm, code), S.get(key, {}).get(m))
    for v, r in S.get("e2e", {}).items():
        put_cell(_mname("e", v, "gold"), r["gold"]); put_cell(_mname("e", v, "pred"), r["pred"])
        put_contrast(_mname("e", v, "vA"), r["pred_minus_A"]); put_contrast(_mname("e", v, "vAL"), r["pred_minus_A_l2"])
        put_contrast(_mname("e", v, "gap"), r["gold_minus_pred"])
    if "stage1_accuracy" in S:
        put("\\nSacc", _fmt(S["stage1_accuracy"]["mean"])); put("\\nSmaj", _fmt(S["stage1_accuracy"]["majority"]))
    for v, x in S.get("rows", {}).items():
        put(_mname("Rows", v), _fmt(x))
    for v, x in S.get("origin_bias", {}).items():
        put(_mname("Orig", v), f"{100 * x['frac_click_preds_within_0.10_of_origin']:.1f}\\%")
    if "batch_check" in S:
        b = S["batch_check"]
        put("\\nBatchRuns", str(b["runs"])); put("\\nBatchIdent", f"{100 * b['frac_identical_mean']:.0f}\\%")
        put("\\nBatchIdentMin", f"{100 * b['frac_identical_min']:.0f}\\%"); put("\\nBatchGap", _fmt(b["max_hit010_gap"]))
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

    # Table: claims (single column)
    if S["claims"]:
        L = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Every quantitative claim, with whether it is established (episode bootstrap and seed-level test both "
             "$p<.05$).}", "\\label{tab:claims}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcc}", "\\toprule",
             "Claim & $\\Delta$ [95\\% CI] & established \\\\", "\\midrule"]
        for c in S["claims"]:
            L.append(f"{c['tex']} & ${c['delta']:+.3f}$ $[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]$ & {'yes' if c['established'] else 'no'} \\\\")
        L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
        (TABLES / "claims.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
