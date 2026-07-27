"""Phase 9 — confirmatory analysis of the leakage-free rerun.

Written and committed BEFORE the untouched-test results existed, alongside
results/phase9_rerun/PRESPEC.md, and executed on the results as committed.
It consumes only the run JSONs the frozen-split jobs persist (pulled via
`modal run modal_app.py::list_stage2_runs`) and computes exactly the
prespecified contrasts.

Post-results edits, disclosed in full (git history is the record): after the
results existed, the Markdown p-value FORMATTER was changed to print
"<1e-4" instead of "<=9.999e-05". No estimator, contrast, metric, cluster
definition, resampling budget, or correction was touched, and no number
changed. Any future edit that would alter a computed value must be reported
as a deviation in the paper.

Primary (untouched test, all_with_coords n_train=1200, hit@0.10, Holm over 4):
  P1 B vs A         P2 D-hook vs A         P3 B vs D-token
  P4 e2e predicted-type pipeline vs A

Everything else is secondary/descriptive and labeled as such in the output.

Run:  uv run python scripts/p9_analyze_rerun.py
"""

from __future__ import annotations

import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.clustered import clustered_paired_test, seed_level_welch

ROOT = Path(__file__).resolve().parent.parent
P4DIR = ROOT / "results" / "phase4"
OUT = ROOT / "results" / "phase9_rerun"

FS = "aitw_frozen_v1"
METRICS = ["hit_at_010", "hit_at_025", "hit_at_005", "mean_normalized_l2"]
N_BOOT = 10000
N_PERM = 10000
SEEDS = "[234]"  # 42/43/44


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adj[k] = min(1.0, running)
    return adj


def load_runs(pattern: str) -> list[dict]:
    files = sorted(glob.glob(str(P4DIR / pattern)))
    return [dict(json.loads(Path(f).read_text()), _path=f) for f in files]


def eval_block(run: dict, which: str) -> tuple[np.ndarray, list[str], dict]:
    """(per-example distances, episode ids, metrics dict) for 'val' or 'test'."""
    if which == "test":
        m = run["final_test_metrics"]
        keys = run["test_step_keys"]
    else:
        m = run["final_val_metrics"]
        keys = run["val_step_keys"]
    if run["variant"] == "D_hook_e2e" and which == "test":
        raise ValueError("use e2e_eval_block for e2e test")
    return (np.asarray(m["per_example_dist"], dtype=np.float64),
            [k["ep_id"] for k in keys], m)


def e2e_test_block(run: dict, mode: str) -> np.ndarray:
    m = run["final_test_metrics"][f"{mode}_metrics"]
    return np.asarray(m["per_example_dist"], dtype=np.float64)


def contrast(runs_a, runs_b, which, name, extract_b=None) -> dict:
    dists_a, eps = [], None
    for r in runs_a:
        d, e, _ = eval_block(r, which)
        dists_a.append(d)
        if eps is None:
            eps = e
        assert e == eps, f"{name}: eval sets differ between runs ({r['_path']})"
    dists_b = []
    for r in runs_b:
        if extract_b is not None:
            d = extract_b(r)
            keys = r.get("test_step_keys" if which == "test" else "val_step_keys")
            if keys is not None:
                assert [k["ep_id"] for k in keys] == eps, \
                    f"{name}: eval sets differ ({r['_path']})"
        else:
            d, e, _ = eval_block(r, which)
            assert e == eps, f"{name}: eval sets differ ({r['_path']})"
        dists_b.append(d)
    out = {"contrast": name, "eval": which,
           "files_a": [Path(r["_path"]).name for r in runs_a],
           "files_b": [Path(r["_path"]).name for r in runs_b],
           "metrics": {}}
    for m in METRICS:
        r_ep = clustered_paired_test(dists_a, dists_b, eps, metric=m,
                                     n_boot=N_BOOT, n_perm=N_PERM, seed=0)
        out["metrics"][m] = {"episode_clustered": asdict(r_ep),
                             "seed_level_welch": seed_level_welch(dists_a, dists_b, metric=m)}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_fs-{FS}"

    A = load_runs(f"variantA_seed4{SEEDS}_n1200_*mix-all_with_coords{tag}.json")
    B = load_runs(f"variantB_seed4{SEEDS}_n1200_*aux1.0{tag}.json")
    C = load_runs(f"variantC_seed4{SEEDS}_n1200_*mix-all_with_coords{tag}.json")
    Dh = load_runs(f"Dhook_seed4{SEEDS}_n1200_*mix-all_with_coords_init0.0{tag}.json")
    Dt = load_runs(f"Dtoken_seed4{SEEDS}_n1200_*init0.02_causal{tag}.json")
    E2 = load_runs(f"e2e_seed4{SEEDS}_n1200_*mix-all_with_coords{tag}.json")
    for nm, rs in [("A", A), ("B", B), ("C", C), ("Dhook", Dh), ("Dtoken", Dt), ("e2e", E2)]:
        print(f"[p9] {nm}: {len(rs)} runs")
        if len(rs) != 3:
            print(f"[p9]   WARNING: expected 3 runs for {nm}")

    results: dict = {"prespec": "results/phase9_rerun/PRESPEC.md",
                     "frozen_split": FS, "n_boot": N_BOOT, "n_perm": N_PERM,
                     "primary": {}, "secondary": {}, "descriptive": {}}

    # ---- PRIMARY (untouched test, Holm across the four) ----
    results["primary"]["P1_B_vs_A"] = contrast(A, B, "test", "P1_B_vs_A")
    results["primary"]["P2_Dhook_vs_A"] = contrast(A, Dh, "test", "P2_Dhook_vs_A")
    results["primary"]["P3_Dtoken_vs_B"] = contrast(B, Dt, "test", "P3_Dtoken_vs_B")
    results["primary"]["P4_e2e_pred_vs_A"] = contrast(
        A, E2, "test", "P4_e2e_pred_vs_A",
        extract_b=lambda r: e2e_test_block(r, "predicted"))
    praw = {k: v["metrics"]["hit_at_010"]["episode_clustered"]["p_value"]
            for k, v in results["primary"].items()}
    padj = holm(praw)
    for k in results["primary"]:
        results["primary"][k]["holm_adjusted_p_hit010"] = padj[k]

    # ---- SECONDARY on test ----
    results["secondary"]["C_vs_A"] = contrast(A, C, "test", "C_vs_A")
    results["secondary"]["Dtoken_vs_A"] = contrast(A, Dt, "test", "Dtoken_vs_A")
    results["secondary"]["Dhook_vs_B"] = contrast(B, Dh, "test", "Dhook_vs_B")
    # pred-vs-oracle is within-model: both sides come from the e2e test block.
    ora = []
    eps = None
    for r in E2:
        ora.append(e2e_test_block(r, "oracle"))
        e = [k["ep_id"] for k in r["test_step_keys"]]
        if eps is None:
            eps = e
        assert e == eps
    pre = [e2e_test_block(r, "predicted") for r in E2]
    oc = {"contrast": "e2e_pred_vs_oracle", "eval": "test", "metrics": {}}
    for m in METRICS:
        oc["metrics"][m] = {"episode_clustered": asdict(clustered_paired_test(
            ora, pre, eps, metric=m, n_boot=N_BOOT, n_perm=N_PERM, seed=0))}
    results["secondary"]["e2e_pred_vs_oracle"] = oc

    # ---- control mix on its test set ----
    Ac = load_runs(f"variantA_seed4{SEEDS}_n1000_*mix-taps_and_swipes{tag}.json")
    Bc = load_runs(f"variantB_seed4{SEEDS}_n1000_*mix-taps_and_swipes_aux1.0{tag}.json")
    Cc = load_runs(f"variantC_seed4{SEEDS}_n1000_*mix-taps_and_swipes{tag}.json")
    Dc = load_runs(f"Dhook_seed4{SEEDS}_n1000_*mix-taps_and_swipes_init0.0{tag}.json")
    if Ac and Bc:
        results["secondary"]["control_B_vs_A"] = contrast(Ac, Bc, "test", "control_B_vs_A")
    if Ac and Dc:
        results["secondary"]["control_Dhook_vs_A"] = contrast(Ac, Dc, "test", "control_Dhook_vs_A")
    if Ac and Cc:
        results["secondary"]["control_C_vs_A"] = contrast(Ac, Cc, "test", "control_C_vs_A")

    # ---- low-data grid on the SAME frozen test set ----
    for n in (300, 500, 800):
        An = load_runs(f"variantA_seed4{SEEDS}_n{n}_*mix-all_with_coords{tag}.json")
        Bn = load_runs(f"variantB_seed4{SEEDS}_n{n}_*aux1.0{tag}.json")
        Dn = load_runs(f"Dhook_seed4{SEEDS}_n{n}_*mix-all_with_coords_init0.0{tag}.json")
        if An and Bn:
            results["secondary"][f"lowdata{n}_B_vs_A"] = contrast(An, Bn, "test", f"lowdata{n}_B_vs_A")
        if An and Dn:
            results["secondary"][f"lowdata{n}_Dhook_vs_A"] = contrast(An, Dn, "test", f"lowdata{n}_Dhook_vs_A")

    # ---- descriptive: means, per-class, parse rates, stage1, causal ----
    # AITW step label -> canonical class, for miss-scored per-class metrics.
    _CLS = {"tap": "click", "type": "type", "swipe_up": "scroll",
            "swipe_down": "scroll", "swipe_left": "scroll", "swipe_right": "scroll"}

    def _per_class_missscored(run: dict, m: dict, which: str) -> dict | None:
        """Per-class metrics with parse failures scored at the sqrt(2) sentinel
        and counted as misses, over FIXED class denominators. The run JSONs'
        own `per_class` block divides by parsed-only counts, which disagrees
        with every aggregate in this file whenever parse_rate < 1."""
        keys = run.get("test_step_keys" if which == "test" else "val_step_keys")
        dist = m.get("per_example_dist")
        if not keys or not dist or len(keys) != len(dist):
            return None
        d = np.asarray(dist, dtype=np.float64)
        labels = [_CLS[k["label"]] for k in keys]
        out = {}
        for cls in sorted(set(labels)):
            mask = np.array([l == cls for l in labels])
            sub = d[mask]
            out[cls] = {
                "n": int(mask.sum()),
                "hit_at_005": float((sub <= 0.05).mean()),
                "hit_at_010": float((sub <= 0.10).mean()),
                "hit_at_025": float((sub <= 0.25).mean()),
                "mean_normalized_l2": float(sub.mean()),
            }
        return out

    def desc(runs, which="test"):
        rows = []
        for r in runs:
            try:
                if r["variant"] == "D_hook_e2e":
                    m = r["final_test_metrics"]["predicted_metrics"] if which == "test" else r["predicted_metrics"]
                else:
                    m = r["final_test_metrics"] if which == "test" else r["final_val_metrics"]
                rows.append({"file": Path(r["_path"]).name, "seed": r["seed"],
                             "hit_at_010": m["hit_at_010"], "hit_at_025": m["hit_at_025"],
                             "hit_at_005": m["hit_at_005"],
                             "mean_normalized_l2": m["mean_normalized_l2"],
                             "parse_rate": m["parse_rate"],
                             "per_class_missscored": _per_class_missscored(r, m, which),
                             "per_class_parsed_only_DEPRECATED": m.get("per_class")})
            except (KeyError, TypeError) as e:
                rows.append({"file": Path(r["_path"]).name, "error": str(e)})
        return rows

    for nm, rs in [("A", A), ("B", B), ("C", C), ("Dhook", Dh), ("Dtoken", Dt), ("e2e_predicted", E2)]:
        results["descriptive"][f"{nm}_test"] = desc(rs)
    causal = []
    for r in Dt:
        ce = r.get("causal_eval")
        if ce:
            causal.append({"seed": r["seed"],
                           "gold_hit010": ce["gold"]["hit_at_010"],
                           "wrong_hit010": (ce["wrong"] or {}).get("hit_at_010"),
                           "zero_hit010": ce["zero"]["hit_at_010"]})
    results["descriptive"]["dtoken_causal_val"] = causal

    (OUT / "rerun_analysis.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"[p9] wrote {OUT / 'rerun_analysis.json'}")

    # ---- render ----
    L = ["# Phase 9 — confirmatory results on the untouched, goal-disjoint test set\n",
         "Prespecified in PRESPEC.md (committed before test access). Primary metric "
         "hit@0.10, episode-clustered paired test, Holm over the four primary contrasts.\n",
         "| contrast | Δ hit@0.10 (95% CI) | perm p | Holm p | per-seed means (A) | per-seed means (B) |",
         "|---|---|---|---|---|---|"]
    def pstr_of(r: dict) -> str:
        """Never report below the permutation budget's Monte Carlo resolution."""
        import math
        if r["p_value"] <= r["p_resolution"]:
            return f"<1e{int(math.ceil(math.log10(r['p_resolution'])))}"
        return f"{r['p_value']:.4g}"

    for k, v in results["primary"].items():
        r = v["metrics"]["hit_at_010"]["episode_clustered"]
        pstr = pstr_of(r)
        L.append(f"| {k} | {r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {pstr} | "
                 f"{v['holm_adjusted_p_hit010']:.4g} | "
                 f"{', '.join(f'{x:.3f}' for x in r['seed_means_a'])} | "
                 f"{', '.join(f'{x:.3f}' for x in r['seed_means_b'])} |")
    L.append("\n## Secondary contrasts (test)\n")
    L.append("| contrast | metric | Δ (95% CI) | perm p |")
    L.append("|---|---|---|---|")
    for k, v in results["secondary"].items():
        for m in METRICS:
            if m not in v["metrics"]:
                continue
            r = v["metrics"][m]["episode_clustered"]
            pstr = pstr_of(r)
            L.append(f"| {k} | {m} | {r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {pstr} |")
    L.append("\nSee rerun_analysis.json for descriptive per-run values, per-class "
             "breakdowns, parse rates, and the D-token causal sensitivity (val).\n")
    (OUT / "RERUN_RESULTS.md").write_text("\n".join(L))
    print(f"[p9] wrote {OUT / 'RERUN_RESULTS.md'}")


if __name__ == "__main__":
    main()
