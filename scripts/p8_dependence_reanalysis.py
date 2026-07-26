"""Phase 8 — dependence-aware reanalysis of every saved Stage 2 prediction.

Replaces the pooled (seed, example) paired bootstrap (src/eval/bootstrap.py
`pool_distances_across_seeds` + scripts/p5_paired_bootstrap.py) with the
episode-clustered analysis in src/eval/clustered.py, applied to the SAME saved
per-example distances. Nothing is retrained here; this answers "does the
Phase 6/7 effect survive a calibrated analysis?" before the leakage-free rerun.

Inputs
------
- results/phase4/*.json                      saved run summaries (per-example dists)
- results/phase8_reanalysis/stream_index_train_n*.json
      ordered index of the filtered AITW step stream (built by
      `modal run modal_app.py::scan_aitw_stream`), used to reconstruct which
      episode every historical train/val example belongs to.

Outputs (results/phase8_reanalysis/)
------
- reanalysis.json      every contrast: episode- and example-clustered CIs,
                       permutation p, per-seed means, legacy pooled p, leakage
                       quantification, synthetic-null validation.
- REANALYSIS.md        rendered summary.

Run:  uv run python scripts/p8_dependence_reanalysis.py
"""

from __future__ import annotations

import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.bootstrap import paired_bootstrap, pool_distances_across_seeds
from src.eval.clustered import (
    clustered_paired_test,
    seed_level_welch,
    validate_power,
    validate_type1_error,
)

ROOT = Path(__file__).resolve().parent.parent
P4 = ROOT / "results" / "phase4"
OUT = ROOT / "results" / "phase8_reanalysis"

MIX_LABELS = {
    "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
    "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
}
METRICS = ["hit_at_010", "hit_at_025", "hit_at_005", "mean_normalized_l2"]
N_BOOT = 10000
N_PERM = 10000


def load_stream_index() -> list[dict]:
    cands = sorted(OUT.glob("stream_index_train_n*.json"))
    if not cands:
        raise SystemExit(
            "stream index missing — run `modal run modal_app.py::scan_aitw_stream` "
            "and `modal volume get stage1-cache aitw_scan/<file> results/phase8_reanalysis/`"
        )
    data = json.loads(cands[-1].read_text())
    print(f"[p8] stream index: {cands[-1].name} revision={data['revision']} "
          f"n_filtered={data['n_filtered']}")
    return data["records"]


def mix_sequence(records: list[dict], mix: str) -> list[dict]:
    allowed = MIX_LABELS[mix]
    return [r for r in records if r["label"] in allowed]


def val_slice(records_mix: list[dict], n_train: int, n_val: int) -> list[dict]:
    sl = records_mix[n_train:n_train + n_val]
    if len(sl) != n_val:
        raise ValueError(f"stream index too short for n_train={n_train}, n_val={n_val}")
    return sl


def check_val_dist(sl: list[dict], recorded: dict | None, tag: str) -> None:
    """The recorded val_action_distribution (present in variantA runs) must match
    the re-derived slice exactly — this is the alignment proof."""
    if not recorded:
        return
    from collections import Counter
    got = Counter(str(r["cid"]) for r in sl)
    want = {str(k): int(v) for k, v in recorded.items()}
    if dict(got) != want:
        raise ValueError(f"[{tag}] slice/class mismatch: derived={dict(got)} recorded={want}")
    print(f"[p8] {tag}: re-derived val slice matches recorded class distribution {want}")


def normalize_goal(g: str) -> str:
    return " ".join(g.lower().split())


def leakage_stats(records_mix: list[dict], n_train: int, n_val: int) -> dict:
    train = records_mix[:n_train]
    val = records_mix[n_train:n_train + n_val]
    train_eps = {r["ep_id"] for r in train}
    train_goals = {normalize_goal(r["goal"]) for r in train}
    val_ep_overlap = sum(1 for r in val if r["ep_id"] in train_eps)
    val_goal_overlap = sum(1 for r in val if normalize_goal(r["goal"]) in train_goals)
    return {
        "n_train": n_train, "n_val": n_val,
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len({r["ep_id"] for r in val}),
        "val_steps_sharing_train_episode": val_ep_overlap,
        "val_steps_sharing_train_episode_frac": val_ep_overlap / n_val,
        "val_steps_sharing_train_goal": val_goal_overlap,
        "val_steps_sharing_train_goal_frac": val_goal_overlap / n_val,
    }


def ped(path: Path) -> np.ndarray:
    d = json.loads(path.read_text())
    arr = (d.get("final_val_metrics") or {}).get("per_example_dist")
    if arr is None:
        raise ValueError(f"{path.name} has no per_example_dist")
    return np.asarray(arr, dtype=np.float64)


def collect(pattern: str) -> tuple[list[np.ndarray], list[str]]:
    files = sorted(glob.glob(str(P4 / pattern)))
    dists, names = [], []
    for f in files:
        try:
            dists.append(ped(Path(f)))
            names.append(Path(f).name)
        except ValueError:
            print(f"[p8]   (skipping {Path(f).name}: no per-example distances)")
    return dists, names


def run_contrast(
    name: str,
    dists_a: list[np.ndarray],
    dists_b: list[np.ndarray],
    ep_ids: list[str],
    files_a: list[str],
    files_b: list[str],
) -> dict:
    out = {"contrast": name, "files_a": files_a, "files_b": files_b, "metrics": {}}
    for m in METRICS:
        r_ep = clustered_paired_test(dists_a, dists_b, ep_ids, metric=m,
                                     n_boot=N_BOOT, n_perm=N_PERM, seed=0)
        r_ex = clustered_paired_test(dists_a, dists_b, list(range(len(ep_ids))),
                                     metric=m, n_boot=N_BOOT, n_perm=N_PERM, seed=0)
        legacy = paired_bootstrap(pool_distances_across_seeds(dists_a),
                                  pool_distances_across_seeds(dists_b),
                                  metric=m, n_boot=N_BOOT, seed=0)
        welch = seed_level_welch(dists_a, dists_b, metric=m)
        out["metrics"][m] = {
            "episode_clustered": asdict(r_ep),
            "example_clustered": asdict(r_ex),
            "legacy_pooled": {"delta": legacy.delta, "ci": [legacy.ci_low, legacy.ci_high],
                              "p_value": legacy.p_value, "n_pooled_rows": legacy.n_examples},
            "seed_level_welch": welch,
        }
    return out


def fmt_ci(r: dict) -> str:
    return f"{r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"


def fmt_p(r: dict) -> str:
    return f"<={r['p_resolution']:.4g}" if r["p_value"] <= r["p_resolution"] else f"{r['p_value']:.4g}"


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, (m - rank) * p)
        adj[k] = min(1.0, running)
    return adj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_stream_index()
    awc = mix_sequence(records, "all_with_coords")
    results: dict = {"note": "exploratory reanalysis of already-seen results; "
                             "confirmatory claims require the untouched test set",
                     "n_boot": N_BOOT, "n_perm": N_PERM, "cells": {}}

    # ---- alignment proof against recorded val distributions ----
    a1200 = json.loads((P4 / "variantA_seed42_n1200_ep2_lr2e-05_mix-all_with_coords.json").read_text())
    sl1200 = val_slice(awc, 1200, 250)
    check_val_dist(sl1200, a1200.get("val_action_distribution"), "awc n1200")
    for n in (300, 500, 800):
        aN = json.loads((P4 / f"variantA_seed42_n{n}_ep2_lr2e-05_mix-all_with_coords.json").read_text())
        check_val_dist(val_slice(awc, n, 250), aN.get("val_action_distribution"), f"awc n{n}")

    # ---- leakage quantification (the headline diagnosis) ----
    results["leakage"] = {
        "awc_n1200": leakage_stats(awc, 1200, 250),
        "awc_n300": leakage_stats(awc, 300, 250),
        "awc_n500": leakage_stats(awc, 500, 250),
        "awc_n800": leakage_stats(awc, 800, 250),
        "awc_n2500": leakage_stats(awc, 2500, 250),
        "awc_n5000": leakage_stats(awc, 5000, 250),
        "ts_n1000": leakage_stats(mix_sequence(records, "taps_and_swipes"), 1000, 200),
    }

    # ---- main ablation, all_with_coords n1200 ----
    ep1200 = [r["ep_id"] for r in sl1200]
    A, A_f = collect("variantA_seed4[234]_n1200_ep2_lr2e-05_mix-all_with_coords.json")
    B, B_f = collect("variantB_seed4[234]_n1200_ep2_lr2e-05_mix-all_with_coords_aux1.0.json")
    C, C_f = collect("variantC_seed4[234]_n1200_ep2_lr2e-05_mix-all_with_coords.json")
    Dh, Dh_f = collect("Dhook_seed4[234]_n1200_ep2_lr2e-05_mix-all_with_coords_init0.0.json")
    Dt, Dt_f = collect("Dtoken_seed4[234]_n1200_ep2_lr2e-05_mix-all_with_coords_init0.02.json")
    main_contrasts = [
        ("B_vs_A", A, B, A_f, B_f),
        ("Dhook_vs_A", A, Dh, A_f, Dh_f),
        ("Dtoken_vs_A", A, Dt, A_f, Dt_f),
        ("C_vs_A", A, C, A_f, C_f),
        ("Dtoken_vs_B", B, Dt, B_f, Dt_f),
        ("Dhook_vs_B", B, Dh, B_f, Dh_f),
    ]
    cell = {}
    for nm, da, db, fa, fb in main_contrasts:
        print(f"[p8] awc n1200 contrast {nm}")
        cell[nm] = run_contrast(nm, da, db, ep1200, fa, fb)
    # Holm correction across the four vs-A contrasts on the headline metric.
    vsA = {k: cell[k]["metrics"]["hit_at_010"]["episode_clustered"]["p_value"]
           for k in ("B_vs_A", "Dhook_vs_A", "Dtoken_vs_A", "C_vs_A")}
    adj = holm(vsA)
    for k, p in adj.items():
        cell[k]["metrics"]["hit_at_010"]["holm_adjusted_p_vsA_family"] = p
    results["cells"]["awc_n1200"] = cell

    # ---- low-data cells ----
    for n in (300, 500, 800):
        sl = val_slice(awc, n, 250)
        eps = [r["ep_id"] for r in sl]
        An, An_f = collect(f"variantA_seed4[234]_n{n}_ep2_lr2e-05_mix-all_with_coords.json")
        Bn, Bn_f = collect(f"variantB_seed4[234]_n{n}_ep2_lr2e-05_mix-all_with_coords_aux1.0.json")
        Dn, Dn_f = collect(f"Dhook_seed4[234]_n{n}_ep2_lr2e-05_mix-all_with_coords_init0.0.json")
        results["cells"][f"awc_n{n}"] = {
            "B_vs_A": run_contrast("B_vs_A", An, Bn, eps, An_f, Bn_f),
            "Dhook_vs_A": run_contrast("Dhook_vs_A", An, Dn, eps, An_f, Dn_f),
        }
        print(f"[p8] low-data n{n} done")

    # ---- single-seed scale cells (descriptive; conditional on one checkpoint pair) ----
    for n in (2500, 5000):
        sl = val_slice(awc, n, 250)
        eps = [r["ep_id"] for r in sl]
        An, An_f = collect(f"variantA_seed42_n{n}_ep2_lr2e-05_mix-all_with_coords.json")
        Bn, Bn_f = collect(f"variantB_seed42_n{n}_ep2_lr2e-05_mix-all_with_coords_aux1.0.json")
        Dn, Dn_f = collect(f"Dhook_seed42_n{n}_ep2_lr2e-05_mix-all_with_coords_init0.0.json")
        results["cells"][f"awc_n{n}_single_seed"] = {
            "B_vs_A": run_contrast("B_vs_A", An, Bn, eps, An_f, Bn_f),
            "Dhook_vs_A": run_contrast("Dhook_vs_A", An, Dn, eps, An_f, Dn_f),
        }

    # ---- e2e: predicted vs oracle vs flat A (same n1200 val slice) ----
    e2e_files = sorted(glob.glob(str(P4 / "e2e_seed4[234]_n1200_ep2_lr2e-05_mix-all_with_coords.json")))
    pred, orac = [], []
    for f in e2e_files:
        d = json.loads(Path(f).read_text())
        pred.append(np.asarray(d["predicted_per_example_dist"], dtype=np.float64))
        orac.append(np.asarray(d["oracle_per_example_dist"], dtype=np.float64))
    if pred:
        results["cells"]["e2e_n1200"] = {
            "predicted_vs_flatA": run_contrast("predicted_vs_flatA", A, pred, ep1200,
                                               A_f, [Path(f).name for f in e2e_files]),
            "oracle_vs_flatA": run_contrast("oracle_vs_flatA", A, orac, ep1200,
                                            A_f, [Path(f).name for f in e2e_files]),
            "predicted_vs_oracle": run_contrast("predicted_vs_oracle", orac, pred, ep1200,
                                                [Path(f).name for f in e2e_files],
                                                [Path(f).name for f in e2e_files]),
        }

    # ---- D-token causal-use sensitivity (gold vs wrong / zero) ----
    causal_files = sorted(glob.glob(str(P4 / "Dtoken_seed4[234]_*_causal.json")))
    gold, wrong, zero = [], [], []
    for f in causal_files:
        ce = json.loads(Path(f).read_text())["causal_eval"]
        gold.append(np.asarray(ce["gold"]["per_example_dist"], dtype=np.float64))
        wrong.append(np.asarray(ce["wrong"]["per_example_dist"], dtype=np.float64))
        zero.append(np.asarray(ce["zero"]["per_example_dist"], dtype=np.float64))
    if gold:
        nmz = [Path(f).name for f in causal_files]
        results["cells"]["dtoken_causal_n1200"] = {
            "wrong_vs_gold": run_contrast("wrong_vs_gold", gold, wrong, ep1200, nmz, nmz),
            "zero_vs_gold": run_contrast("zero_vs_gold", gold, zero, ep1200, nmz, nmz),
        }

    # ---- synthetic validation of the machinery ----
    print("[p8] validating type-I error on synthetic clustered nulls...")
    results["validation"] = {
        "fixed_checkpoint_null": validate_type1_error(
            n_sims=1000, n_perm=400, seed=42, n_episodes=90, steps_per_episode=3,
            n_seeds=3, checkpoint_sd=0.0),
        "procedure_null_with_run_offsets": validate_type1_error(
            n_sims=400, n_perm=400, seed=43, n_episodes=90, steps_per_episode=3,
            n_seeds=3, checkpoint_sd=0.05),
        "power_at_observed_effect": validate_power(
            true_delta=0.03, n_sims=400, seed=44, n_episodes=90,
            steps_per_episode=3, n_seeds=3),
        "note": ("fixed_checkpoint_null: population-mean null with example/episode "
                 "effect heterogeneity shared across seeds — the estimand our "
                 "episode-clustered test targets; its rejection rate should be ~alpha "
                 "while the legacy pooled test is inflated. "
                 "procedure_null_with_run_offsets: adds run-level (seed) offsets; "
                 "ANY fixed-checkpoint test over-rejects here, which is why per-seed "
                 "effects are reported alongside and procedure-level claims are avoided."),
    }

    (OUT / "reanalysis.json").write_text(json.dumps(results, indent=2))
    print(f"[p8] wrote {OUT / 'reanalysis.json'}")
    render_md(results)


def render_md(results: dict) -> None:
    L: list[str] = []
    L.append("# Phase 8 — dependence-aware reanalysis (saved predictions, no retraining)\n")
    L.append("**Status: exploratory.** These are the Phase 6/7 runs re-analyzed with an "
             "episode-clustered paired test (seed-averaged per-example deltas; cluster "
             "bootstrap CI over episodes; episode-level sign-flip permutation p). The "
             "train/val split leakage documented below is NOT fixed by reanalysis — that "
             "requires the Phase 9 episode-disjoint rerun. Confirmatory numbers come only "
             "from the untouched test set.\n")

    lk = results["leakage"]
    L.append("## Leakage in the historical split (quantified)\n")
    L.append("| cell | val steps | val steps sharing a train episode | sharing a train goal |")
    L.append("|---|---|---|---|")
    for k, v in lk.items():
        L.append(f"| {k} | {v['n_val']} | {v['val_steps_sharing_train_episode']} "
                 f"({100*v['val_steps_sharing_train_episode_frac']:.0f}%) | "
                 f"{v['val_steps_sharing_train_goal']} "
                 f"({100*v['val_steps_sharing_train_goal_frac']:.0f}%) |")
    L.append("")

    def table(cell: dict, title: str) -> None:
        L.append(f"## {title}\n")
        L.append("| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |")
        L.append("|---|---|---|---|---|---|")
        for nm, c in cell.items():
            for m in METRICS:
                r = c["metrics"][m]["episode_clustered"]
                lp = c["metrics"][m]["legacy_pooled"]["p_value"]
                sm_a = c["metrics"][m]["episode_clustered"]["seed_means_a"]
                sm_b = c["metrics"][m]["episode_clustered"]["seed_means_b"]
                seed_note = f"A:{','.join(f'{x:.3f}' for x in sm_a)} B:{','.join(f'{x:.3f}' for x in sm_b)}"
                holm_p = c["metrics"][m].get("holm_adjusted_p_vsA_family")
                extra = f" (Holm {holm_p:.3g})" if holm_p is not None and m == "hit_at_010" else ""
                L.append(f"| {nm} | {m} | {fmt_ci(r)} | {fmt_p(r)}{extra} | {lp:.4g} | {seed_note} |")
        L.append("")

    for key, title in [
        ("awc_n1200", "Main ablation — all_with_coords, n_train=1200 (3 seeds)"),
        ("awc_n300", "Low data — n_train=300"),
        ("awc_n500", "Low data — n_train=500"),
        ("awc_n800", "Low data — n_train=800"),
        ("awc_n2500_single_seed", "Scale — n_train=2500 (single seed; descriptive)"),
        ("awc_n5000_single_seed", "Scale — n_train=5000 (single seed; descriptive)"),
        ("e2e_n1200", "End-to-end pipeline — n_train=1200"),
        ("dtoken_causal_n1200", "D-token causal-use sensitivity — n1200"),
    ]:
        if key in results["cells"]:
            table(results["cells"][key], title)

    v = results["validation"]
    L.append("## Synthetic-null validation of the test itself\n")
    L.append(f"- Fixed-checkpoint population null (the estimand we test): clustered reject rate "
             f"**{v['fixed_checkpoint_null']['clustered_reject_rate']:.3f}** at α=0.05 "
             f"(MC SE {v['fixed_checkpoint_null']['mc_se']:.3f}); legacy pooled test rejects "
             f"**{v['fixed_checkpoint_null']['pooled_reject_rate']:.3f}** — the pooled routine is "
             f"anti-conservative exactly as suspected.")
    L.append(f"- Procedure-level null with run-to-run offsets: fixed-checkpoint tests (ours included) "
             f"reject at {v['procedure_null_with_run_offsets']['clustered_reject_rate']:.3f} — why we "
             f"report per-seed effects and make no training-procedure significance claim from 3 seeds.")
    L.append(f"- Power at a Δ=0.03 (mean-L2) true effect: "
             f"{v['power_at_observed_effect']['clustered_power']:.3f}.\n")
    L.append("Numbers here supersede every previously quoted pooled p-value "
             "(including the retracted p=0.0002 and any p=0).\n")

    (OUT / "REANALYSIS.md").write_text("\n".join(L))
    print(f"[p8] wrote {OUT / 'REANALYSIS.md'}")


if __name__ == "__main__":
    main()
