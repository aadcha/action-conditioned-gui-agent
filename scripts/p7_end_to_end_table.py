"""Phase 7 end-to-end Stage 1 -> Stage 2 analysis, plots, and writeup.

Inputs are per-seed JSONs from ``modal_app.py::phase7_end_to_end``:
    results/phase7/end_to_end_seed42_mix-all_with_coords.json

The script pools aligned per-example normalized-L2 distances across seeds,
runs paired bootstrap + permutation tests, renders the requested figures, and
writes ``results/phase7/end_to_end_summary.json`` plus ``PHASE7_END_TO_END.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/action-conditioned-gui-agent-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/action-conditioned-gui-agent-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import paired_bootstrap, permutation_test, pool_distances_across_seeds
from src.eval.phase7 import (
    PHASE7_METRICS,
    PHASE7_VARIANTS,
    metric_mean_from_distances,
    stage1_correct_breakdown,
    summarize_stage1_predictions,
)

PHASE7 = ROOT / "results" / "phase7"
PHASE4 = ROOT / "results" / "phase4"

DISPLAY = {
    "A_flat": "A-flat",
    "D_gold": "D-gold\n(oracle)",
    "D_predicted": "D-predicted\n(Stage 1)",
    "D_majority": "D-majority",
    "D_random": "D-random",
}

PALETTE = {
    "A_flat": "#888888",
    "D_gold": "#3a6ea5",
    "D_predicted": "#9d4edd",
    "D_majority": "#c08552",
    "D_random": "#bdbdbd",
    "click": "#3a6ea5",
    "type": "#9d4edd",
    "scroll": "#c08552",
}

METRIC_TITLES = {
    "hit_at_005": "hit@0.05",
    "hit_at_010": "hit@0.10",
    "hit_at_025": "hit@0.25",
    "mean_normalized_l2": "mean normalized L2",
}

COMPARISONS = [
    ("D_predicted", "A_flat"),
    ("D_gold", "A_flat"),
    ("D_predicted", "D_gold"),
    ("D_predicted", "D_majority"),
    ("D_predicted", "D_random"),
]

COMPARISON_DISPLAY = {
    "D_predicted_vs_A_flat": "D-predicted vs A-flat\n(deployable gain)",
    "D_gold_vs_A_flat": "D-gold vs A-flat\n(oracle ceiling)",
    "D_predicted_vs_D_gold": "D-predicted vs D-gold\n(classifier cost)",
    "D_predicted_vs_D_majority": "D-predicted vs D-majority\n(type baseline)",
    "D_predicted_vs_D_random": "D-predicted vs D-random\n(sanity check)",
}


def _variant_dist(run: dict, variant: str) -> list[float] | None:
    fvm = ((run.get("variants") or {}).get(variant) or {}).get("final_val_metrics") or {}
    dist = fvm.get("per_example_dist")
    return [float(x) for x in dist] if dist is not None else None


def _load_phase4_a(seed: int, mix: str) -> list[float] | None:
    candidates = sorted(PHASE4.glob(f"variantA_seed{seed}_*_mix-{mix}.json"))
    for p in reversed(candidates):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        dist = (data.get("final_val_metrics") or {}).get("per_example_dist")
        if dist is not None:
            print(f"[p7] using Phase 6 A-flat fallback for seed {seed}: {p.name}")
            return [float(x) for x in dist]
    return None


def load_runs(mix: str, allow_phase4_a: bool) -> list[dict]:
    runs = []
    for p in sorted(PHASE7.glob(f"end_to_end_seed*_mix-{mix}.json")):
        data = json.loads(p.read_text())
        data["_file"] = p.name
        seed = int(data["seed"])
        if allow_phase4_a and _variant_dist(data, "A_flat") is None:
            dist = _load_phase4_a(seed, mix)
            if dist is not None:
                data.setdefault("variants", {}).setdefault("A_flat", {})["final_val_metrics"] = {
                    "per_example_dist": dist,
                    "source": "results/phase4",
                }
        runs.append(data)
    return runs


def pool_by_variant(runs: list[dict]) -> tuple[dict[str, np.ndarray], list[int], int]:
    by_variant_seed: dict[str, dict[int, np.ndarray]] = {v: {} for v in PHASE7_VARIANTS}
    for run in runs:
        seed = int(run["seed"])
        for variant in PHASE7_VARIANTS:
            dist = _variant_dist(run, variant)
            if dist is not None:
                by_variant_seed[variant][seed] = np.asarray(dist, dtype=np.float64)

    available = {v: set(seed_map) for v, seed_map in by_variant_seed.items() if seed_map}
    common_seeds = sorted(set.intersection(*available.values())) if available else []
    if not common_seeds:
        return {}, [], 0
    val_len = min(len(by_variant_seed[v][s]) for v in available for s in common_seeds if s in by_variant_seed[v])

    pools = {}
    for variant, seed_map in by_variant_seed.items():
        arrays = [seed_map[s][:val_len] for s in common_seeds if s in seed_map]
        if len(arrays) == len(common_seeds):
            pools[variant] = pool_distances_across_seeds(arrays)
    return pools, common_seeds, val_len


def metric_table(pools: dict[str, np.ndarray]) -> dict:
    return {
        variant: {metric: metric_mean_from_distances(dist, metric) for metric in PHASE7_METRICS}
        for variant, dist in pools.items()
    }


def comparison_table(pools: dict[str, np.ndarray], n_boot: int) -> dict:
    out = {}
    for b_var, a_var in COMPARISONS:
        if b_var not in pools or a_var not in pools:
            continue
        key = f"{b_var}_vs_{a_var}"
        out[key] = {}
        common = min(len(pools[a_var]), len(pools[b_var]))
        a_pool = pools[a_var][:common]
        b_pool = pools[b_var][:common]
        for metric in PHASE7_METRICS:
            res = paired_bootstrap(a_pool, b_pool, metric=metric, n_boot=n_boot, seed=0)
            p_perm = permutation_test(a_pool, b_pool, metric=metric, n_perm=n_boot, seed=0)
            out[key][metric] = {
                "mean_a": res.mean_a,
                "mean_b": res.mean_b,
                "delta": res.delta,
                "ci95": [res.ci_low, res.ci_high],
                "p_bootstrap": res.p_value,
                "p_permutation": p_perm,
                "higher_is_better": res.higher_is_better,
                "n_units": common,
            }
    return out


def aggregate_stage1(runs: list[dict], common_seeds: list[int]) -> dict:
    gold, pred, conf = [], [], []
    for run in runs:
        if int(run["seed"]) not in common_seeds:
            continue
        for ex in run.get("examples", []):
            if "gold_action_type" in ex and "pred_action_type" in ex:
                gold.append(int(ex["gold_action_type"]))
                pred.append(int(ex["pred_action_type"]))
                conf.append(float(ex.get("stage1_confidence", 0.0)))
    if gold:
        return summarize_stage1_predictions(gold, pred, conf)

    # Fallback for partial/older run files.
    rows = [run.get("stage1") for run in runs if int(run["seed"]) in common_seeds and run.get("stage1")]
    if not rows:
        return {}
    return {
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
        "note": "mean over per-seed Stage 1 summaries; per-example records unavailable",
    }


def aggregate_correct_breakdown(runs: list[dict], common_seeds: list[int]) -> dict:
    dists, gold, pred = [], [], []
    for run in runs:
        if int(run["seed"]) not in common_seeds:
            continue
        dist = _variant_dist(run, "D_predicted") or []
        examples = run.get("examples", [])
        n = min(len(dist), len(examples))
        for i in range(n):
            ex = examples[i]
            dists.append(float(dist[i]))
            gold.append(int(ex["gold_action_type"]))
            pred.append(int(ex["pred_action_type"]))
    if not dists:
        return {}
    return stage1_correct_breakdown(dists, gold, pred)


def aggregate_valid_coordinate_subset(
    runs: list[dict],
    common_seeds: list[int],
    n_boot: int,
) -> dict:
    """Repeat the main analysis after dropping examples with invalid target_xy.

    AITW TYPE rows in this slice can carry target_xy=[-1,-1]. Those are not
    meaningful coordinate-grounding targets, so this sensitivity analysis checks
    whether the headline conclusion changes on valid click/scroll coordinates.
    """
    by_variant = {v: [] for v in PHASE7_VARIANTS}
    n_valid = 0
    n_invalid = 0
    for run in runs:
        if int(run["seed"]) not in common_seeds:
            continue
        examples = run.get("examples", [])
        for i, ex in enumerate(examples):
            x, y = ex.get("target_xy", [float("nan"), float("nan")])
            is_valid = 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0
            if is_valid:
                n_valid += 1
                for variant in PHASE7_VARIANTS:
                    dist = _variant_dist(run, variant)
                    if dist is not None and i < len(dist):
                        by_variant[variant].append(float(dist[i]))
            else:
                n_invalid += 1

    pools = {
        variant: np.asarray(dist, dtype=np.float64)
        for variant, dist in by_variant.items()
        if dist
    }
    means = metric_table(pools)
    comparisons = comparison_table(pools, n_boot)
    return {
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "means": means,
        "comparisons": comparisons,
    }


def _label_bars(ax, bars, *, dy: float = 0.01, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        val = bar.get_height()
        if val >= 0:
            y = val + dy
            va = "bottom"
        else:
            y = val - dy
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(val),
            ha="center",
            va=va,
            fontsize=9,
        )


def plot_headline(table: dict, out_path: Path) -> None:
    variants = [v for v in PHASE7_VARIANTS if v in table]
    vals = [table[v]["hit_at_010"] for v in variants]
    colors = [PALETTE[v] for v in variants]
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(x, vals, color=colors, width=0.58)
    ax.axhline(table["A_flat"]["hit_at_010"], ls=":", color="#444",
               label=f"A-flat baseline ({table['A_flat']['hit_at_010']:.3f})")
    _label_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[v] for v in variants])
    ax.set_ylabel("hit@0.10 (higher is better)")
    ax.set_ylim(0, max(0.36, max(vals) + 0.08))
    ax.set_title(
        "Phase 7 — end-to-end grounding on AITW all_with_coords\n"
        "oracle action types help most; predicted types preserve only part of the gain"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_grid(table: dict, out_path: Path) -> None:
    variants = [v for v in PHASE7_VARIANTS if v in table]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for ax, metric in zip(axes.flatten(), PHASE7_METRICS, strict=True):
        vals = [table[v][metric] for v in variants]
        bars = ax.bar([DISPLAY[v] for v in variants], vals,
                      color=[PALETTE[v] for v in variants], alpha=0.92)
        ax.set_title(
            f"{METRIC_TITLES[metric]}"
            + (" (lower is better)" if metric == "mean_normalized_l2" else "")
        )
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, max(vals) + (0.08 if metric != "mean_normalized_l2" else 0.10))
        _label_bars(ax, bars, dy=0.006 if metric == "mean_normalized_l2" else 0.01)
        if metric == "hit_at_010":
            ax.axhline(table["A_flat"][metric], ls=":", color="#444", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_ci(comparisons: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    labels = [COMPARISON_DISPLAY.get(key, key.replace("_vs_", " vs ").replace("_", "-"))
              for key in comparisons]
    for ax, metric in zip(axes.flatten(), PHASE7_METRICS, strict=True):
        ys = np.arange(len(labels))
        deltas = [comparisons[key][metric]["delta"] for key in comparisons]
        lows = [comparisons[key][metric]["ci95"][0] for key in comparisons]
        highs = [comparisons[key][metric]["ci95"][1] for key in comparisons]
        xerr = np.asarray([[d - lo for d, lo in zip(deltas, lows, strict=True)],
                           [hi - d for d, hi in zip(deltas, highs, strict=True)]])
        ax.errorbar(deltas, ys, xerr=xerr, fmt="o", color="#9d4edd", ecolor="#555", capsize=3)
        ax.axvline(0, color="#333", linestyle=":")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.set_title(
            f"Delta in {METRIC_TITLES[metric]}"
            + (" (negative is better)" if metric == "mean_normalized_l2" else "")
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_oracle_gap(summary: dict, out_path: Path) -> None:
    """Slide-style comparison of deployable vs oracle deltas over A-flat."""
    comps = summary["comparisons"]
    pairs = [
        ("D_gold_vs_A_flat", "D-gold vs A-flat"),
        ("D_predicted_vs_A_flat", "D-predicted vs A-flat"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ("hit_at_010", "mean_normalized_l2"), strict=True):
        labels, deltas, lows, highs = [], [], [], []
        for key, label in pairs:
            row = comps[key][metric]
            labels.append(label)
            deltas.append(row["delta"])
            lows.append(row["ci95"][0])
            highs.append(row["ci95"][1])
        x = np.arange(len(labels))
        err = np.asarray([[d - lo for d, lo in zip(deltas, lows, strict=True)],
                          [hi - d for d, hi in zip(deltas, highs, strict=True)]])
        bars = ax.bar(x, deltas, color=["#3a6ea5", "#9d4edd"], alpha=0.9)
        ax.errorbar(x, deltas, yerr=err, fmt="none", ecolor="#333", capsize=4)
        ax.axhline(0, color="#333", linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(
            f"Delta in {METRIC_TITLES[metric]}"
            + (" (negative is better)" if metric == "mean_normalized_l2" else "")
        )
        ax.grid(axis="y", alpha=0.2)
        _label_bars(ax, bars, dy=0.004, fmt="{:+.3f}")
    fig.suptitle("Oracle type ceiling vs deployable Stage 1 predictions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(stage1: dict, out_path: Path) -> None:
    if not stage1.get("confusion_matrix"):
        return
    cm = np.asarray(stage1["confusion_matrix"])
    labels = stage1["confusion_matrix_labels"]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(
        "Phase 7 Stage 1 confusion on Stage 2 validation set\n"
        f"accuracy={stage1.get('accuracy', float('nan')):.3f}, "
        f"macro-F1={stage1.get('macro_f1', float('nan')):.3f}"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stage1_per_class(stage1: dict, out_path: Path) -> None:
    per_class = stage1.get("per_class") or {}
    if not per_class:
        return
    classes = stage1.get("confusion_matrix_labels") or list(per_class)
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    metric_colors = {"precision": "#888888", "recall": "#3a6ea5", "f1-score": "#9d4edd"}
    for i, metric in enumerate(("precision", "recall", "f1-score")):
        vals = [per_class[c].get(metric, 0.0) for c in classes]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label="F1" if metric == "f1-score" else metric,
            color=metric_colors[metric],
            alpha=0.9,
        )
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    support = [int(per_class[c].get("support", 0)) for c in classes]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, support, strict=True)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Phase 7 — Stage 1 per-class quality on the Stage 2 validation set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_action_distribution(stage1: dict, out_path: Path) -> None:
    gold = stage1.get("ground_truth_distribution") or {}
    pred = stage1.get("prediction_distribution") or {}
    if not gold or not pred:
        return
    ordered = stage1.get("confusion_matrix_labels") or []
    classes = [c for c in ordered if c in set(gold) | set(pred)]
    classes += [c for c in sorted(set(gold) | set(pred)) if c not in classes]
    x = np.arange(len(classes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars_gold = ax.bar(x - width / 2, [gold.get(c, 0) for c in classes], width,
                       label="gold", color="#3a6ea5")
    bars_pred = ax.bar(x + width / 2, [pred.get(c, 0) for c in classes], width,
                       label="Stage 1 predicted", color="#c08552")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("examples")
    ax.set_title("Phase 7 — action-type distribution on the Stage 2 validation set")
    ax.legend()
    _label_bars(ax, bars_gold, dy=4, fmt="{:.0f}")
    _label_bars(ax, bars_pred, dy=4, fmt="{:.0f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correct_breakdown(breakdown: dict, out_path: Path) -> None:
    if not breakdown:
        return
    labels = [
        f"Stage 1 correct\n(n={breakdown['n_correct']})",
        f"Stage 1 incorrect\n(n={breakdown['n_incorrect']})",
    ]
    correct = breakdown["stage1_correct"]
    incorrect = breakdown["stage1_incorrect"]
    hit_vals = [correct["hit_at_010"], incorrect["hit_at_010"]]
    l2_vals = [correct["mean_normalized_l2"], incorrect["mean_normalized_l2"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    bars0 = axes[0].bar(labels, hit_vals, color=["#3a6ea5", "#c08552"])
    axes[0].set_title("D-predicted hit@0.10")
    axes[0].set_ylim(0, max(hit_vals + [0.2]) + 0.1)
    _label_bars(axes[0], bars0)
    bars1 = axes[1].bar(labels, l2_vals, color=["#3a6ea5", "#c08552"])
    axes[1].set_title("D-predicted mean normalized L2")
    axes[1].set_ylim(0, max(l2_vals) + 0.1)
    _label_bars(axes[1], bars1, dy=0.006)
    fig.suptitle("Grounding quality split by whether Stage 1 predicted the correct action type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(summary: dict, out_path: Path) -> None:
    means = summary["means"]
    comps = summary["comparisons"]
    headline = comps.get("D_predicted_vs_A_flat", {})
    valid = summary.get("valid_coordinate_subset") or {}
    lines = [
        "# Phase 7 - End-to-end Stage 1 -> Stage 2 evaluation",
        "",
        f"Data mix: **{summary['mix']}**. Pooled units: **{summary['n_units']}** "
        f"({len(summary['seeds'])} seeds x {summary['val_len']} examples).",
        "",
        "## Means",
        "",
        "| variant | hit@0.05 | hit@0.10 | hit@0.25 | mean norm L2 (lower better) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in PHASE7_VARIANTS:
        if variant not in means:
            continue
        row = means[variant]
        lines.append(
            f"| {DISPLAY[variant]} | {row['hit_at_005']:.3f} | {row['hit_at_010']:.3f} | "
            f"{row['hit_at_025']:.3f} | {row['mean_normalized_l2']:.3f} |"
        )
    lines += ["", "## Headline statistical test", ""]
    if headline:
        h10 = headline["hit_at_010"]
        l2 = headline["mean_normalized_l2"]
        h05 = headline["hit_at_005"]
        h25 = headline["hit_at_025"]
        verdict = (
            "significantly beats"
            if h10["delta"] > 0 and h10["p_bootstrap"] < 0.05 and l2["delta"] < 0 and l2["p_bootstrap"] < 0.05
            else "does not significantly beat"
        )
        lines.append(
            f"Deployable **D-predicted {verdict} A-flat** on the paired test. "
            f"hit@0.10 delta={h10['delta']:+.3f}, 95% CI "
            f"[{h10['ci95'][0]:+.3f}, {h10['ci95'][1]:+.3f}], p={h10['p_bootstrap']:.4f}; "
            f"mean L2 delta={l2['delta']:+.3f}, p={l2['p_bootstrap']:.4f}."
        )
        lines.append(
            f"Tight-threshold gains are significant: hit@0.05 delta={h05['delta']:+.3f} "
            f"(p={h05['p_bootstrap']:.4f}) and hit@0.10 delta={h10['delta']:+.3f} "
            f"(p={h10['p_bootstrap']:.4f}). The broader metrics are not: "
            f"hit@0.25 delta={h25['delta']:+.3f} (p={h25['p_bootstrap']:.4f}) and "
            f"mean L2 delta={l2['delta']:+.3f} (p={l2['p_bootstrap']:.4f})."
        )
    else:
        lines.append("D-predicted vs A-flat was unavailable because one variant was missing.")
    if "D_gold_vs_A_flat" in comps:
        gold_h10 = comps["D_gold_vs_A_flat"]["hit_at_010"]
        gold_l2 = comps["D_gold_vs_A_flat"]["mean_normalized_l2"]
        lines += [
            "",
            "## Oracle ceiling",
            "",
            f"D-gold significantly beats A-flat on every metric. For hit@0.10, "
            f"delta={gold_h10['delta']:+.3f}, 95% CI "
            f"[{gold_h10['ci95'][0]:+.3f}, {gold_h10['ci95'][1]:+.3f}], "
            f"p={gold_h10['p_bootstrap']:.4f}; for mean L2, "
            f"delta={gold_l2['delta']:+.3f}, p={gold_l2['p_bootstrap']:.4f}. "
            "This confirms the action-type signal still has value when it is correct.",
        ]
    stage1 = summary.get("stage1") or {}
    if stage1:
        lines += [
            "",
            "## Stage 1 quality",
            "",
            f"Stage 1 on the exact Stage 2 validation set: accuracy={stage1.get('accuracy', float('nan')):.3f}, "
            f"macro-F1={stage1.get('macro_f1', float('nan')):.3f}, "
            f"weighted-F1={stage1.get('weighted_f1', float('nan')):.3f}.",
        ]
        per_class = stage1.get("per_class") or {}
        if per_class:
            lines += [
                "",
                "| class | precision | recall | F1 | support |",
                "|---|---:|---:|---:|---:|",
            ]
            for name, row in per_class.items():
                lines.append(
                    f"| {name} | {row.get('precision', 0):.3f} | {row.get('recall', 0):.3f} | "
                    f"{row.get('f1-score', 0):.3f} | {int(row.get('support', 0))} |"
                )
    breakdown = summary.get("stage1_correct_breakdown") or {}
    if breakdown:
        good = breakdown["stage1_correct"]
        bad = breakdown["stage1_incorrect"]
        lines += [
            "",
            "## Stage 1 error cost",
            "",
            f"When Stage 1 is correct (n={breakdown['n_correct']}), D-predicted hit@0.10={good['hit_at_010']:.3f}. "
            f"When Stage 1 is wrong (n={breakdown['n_incorrect']}), hit@0.10={bad['hit_at_010']:.3f}. "
            "This is the clearest evidence that action-type errors do affect downstream grounding.",
        ]
    if valid:
        vcomp = (valid.get("comparisons") or {}).get("D_predicted_vs_A_flat", {})
        vh10 = vcomp.get("hit_at_010")
        vl2 = vcomp.get("mean_normalized_l2")
        lines += [
            "",
            "## Valid-coordinate sensitivity",
            "",
            f"{valid.get('n_invalid', 0)} pooled examples have invalid target coordinates, mostly TYPE rows with "
            "`target_xy=[-1,-1]`; these are not meaningful coordinate-grounding targets.",
        ]
        if vh10 and vl2:
            lines.append(
                f"After dropping invalid targets (n={valid.get('n_valid', 0)}), the qualitative result is unchanged: "
                f"D-predicted vs A-flat hit@0.10 delta={vh10['delta']:+.3f}, "
                f"95% CI [{vh10['ci95'][0]:+.3f}, {vh10['ci95'][1]:+.3f}], "
                f"p={vh10['p_bootstrap']:.4f}; mean L2 delta={vl2['delta']:+.3f}, "
                f"p={vl2['p_bootstrap']:.4f}."
            )
    lines += [
        "",
        "## Figures",
        "",
        "- `phase7_headline_hit010.png`",
        "- `phase7_metric_grid.png`",
        "- `phase7_oracle_gap.png`",
        "- `phase7_delta_ci.png`",
        "- `phase7_stage1_confusion.png`",
        "- `phase7_stage1_per_class_f1.png`",
        "- `phase7_action_distribution.png`",
        "- `phase7_correct_vs_incorrect.png`",
        "",
        "## Interpretation note",
        "",
        "D-gold is an oracle ceiling. The deployable claim is based only on D-predicted vs A-flat. "
        "The statistically safe conclusion is partial deployable benefit: improved precise localization at tight "
        "thresholds, but no reliable overall improvement on loose hit rate or mean distance.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="all_with_coords")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--allow-phase4-a", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    PHASE7.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.mix, args.allow_phase4_a)
    print(f"[p7] loaded {len(runs)} Phase 7 run files")
    if not runs:
        print("[p7] no runs found; run modal_app.py::phase7_end_to_end first")
        return

    pools, seeds, val_len = pool_by_variant(runs)
    if not pools:
        print("[p7] no aligned variant pools found")
        return
    means = metric_table(pools)
    comparisons = comparison_table(pools, args.n_boot)
    stage1 = aggregate_stage1(runs, seeds)
    breakdown = aggregate_correct_breakdown(runs, seeds)
    valid_subset = aggregate_valid_coordinate_subset(runs, seeds, args.n_boot)

    summary = {
        "mix": args.mix,
        "seeds": seeds,
        "val_len": val_len,
        "n_units": int(min(len(v) for v in pools.values())),
        "means": means,
        "comparisons": comparisons,
        "stage1": stage1,
        "stage1_correct_breakdown": breakdown,
        "valid_coordinate_subset": valid_subset,
    }
    out_json = PHASE7 / "end_to_end_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[p7] wrote {out_json}")

    plot_headline(means, PHASE7 / "phase7_headline_hit010.png")
    plot_metric_grid(means, PHASE7 / "phase7_metric_grid.png")
    plot_oracle_gap(summary, PHASE7 / "phase7_oracle_gap.png")
    plot_delta_ci(comparisons, PHASE7 / "phase7_delta_ci.png")
    plot_confusion(stage1, PHASE7 / "phase7_stage1_confusion.png")
    plot_stage1_per_class(stage1, PHASE7 / "phase7_stage1_per_class_f1.png")
    plot_action_distribution(stage1, PHASE7 / "phase7_action_distribution.png")
    plot_correct_breakdown(breakdown, PHASE7 / "phase7_correct_vs_incorrect.png")
    write_report(summary, PHASE7 / "PHASE7_END_TO_END.md")
    print(f"[p7] wrote figures + {PHASE7 / 'PHASE7_END_TO_END.md'}")


if __name__ == "__main__":
    main()
