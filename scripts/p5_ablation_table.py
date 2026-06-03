"""Phase 5 — build the A-vs-D ablation table + figures from persisted run JSONs.

Reads every `results/phase4/*.json` that looks like a Stage 2 training result
(must have `variant` and `final_val_metrics`), groups by (variant, data_mix),
and emits a multi-seed comparison table + matplotlib figures.

Idempotent — re-run after each new training finishes.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE4_DIR = Path(__file__).resolve().parent.parent / "results" / "phase4"


@dataclass
class RunRow:
    file_name: str
    variant: str        # "A" or "D"
    data_mix: str
    seed: int
    n_train: int
    epochs: int
    final_loss: float
    hit_at_005: float
    hit_at_010: float
    hit_at_025: float
    mean_norm_l2: float
    parse_rate: float


def _canon_variant(v: str, file_name: str = "") -> str:
    v_l = v.lower()
    if v_l.startswith("a_flat") or v_l == "a":
        return "A"
    if "action_conditioned" in v_l or v_l == "d":
        return "D"
    # Fall back to file naming convention
    fn_l = file_name.lower()
    if fn_l.startswith("variant"):
        return "A" if "variantA" in file_name else "?"
    if fn_l.startswith("train_seed") or fn_l.startswith("stage2_n"):
        return "D"
    return "?"


def load_runs(d: Path = PHASE4_DIR) -> list[RunRow]:
    rows: list[RunRow] = []
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        # Must look like a training-run JSON
        if "final_val_metrics" not in data or "n_train" not in data:
            continue
        fvm = data.get("final_val_metrics") or {}
        if not fvm:
            continue

        variant = _canon_variant(data.get("variant", ""), p.name)
        if variant == "?":
            # Skip files we can't classify (e.g. exploratory one-off runs)
            continue

        data_mix = data.get("data_mix") or ("taps_only" if data.get("only_taps", True) else "all_with_coords")

        rows.append(RunRow(
            file_name=p.name,
            variant=variant,
            data_mix=data_mix,
            seed=int(data.get("seed", -1)),
            n_train=int(data.get("n_train", -1)),
            epochs=int(data.get("epochs", -1)),
            final_loss=float(fvm.get("train_loss", math.nan)),
            hit_at_005=float(fvm.get("hit_at_005", math.nan)),
            hit_at_010=float(fvm.get("hit_at_010", math.nan)),
            hit_at_025=float(fvm.get("hit_at_025", math.nan)),
            mean_norm_l2=float(fvm.get("mean_normalized_l2", math.nan)),
            parse_rate=float(fvm.get("parse_rate", 1.0)),
        ))
    return rows


def aggregate(rows: list[RunRow]) -> dict:
    """Group by (variant, data_mix); compute mean/std on each numeric column."""
    groups: dict[tuple[str, str], list[RunRow]] = defaultdict(list)
    for r in rows:
        groups[(r.variant, r.data_mix)].append(r)

    summary: dict = {}
    for (variant, mix), runs in groups.items():
        if not runs:
            continue
        seeds = sorted({r.seed for r in runs})
        # If multiple files share the same seed, keep the latest one only (re-run wins)
        by_seed: dict[int, RunRow] = {}
        for r in runs:
            by_seed[r.seed] = r
        runs_canonical = list(by_seed.values())

        def col(name: str) -> list[float]:
            return [getattr(r, name) for r in runs_canonical if not math.isnan(getattr(r, name))]

        def mean_std(name: str) -> tuple[float, float]:
            vals = col(name)
            if not vals:
                return math.nan, 0.0
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return m, s

        h05_mean, h05_std = mean_std("hit_at_005")
        h10_mean, h10_std = mean_std("hit_at_010")
        h25_mean, h25_std = mean_std("hit_at_025")
        l2_mean, l2_std = mean_std("mean_norm_l2")
        loss_mean, loss_std = mean_std("final_loss")

        key = f"{variant}__{mix}"
        summary[key] = {
            "variant": variant,
            "data_mix": mix,
            "n_seeds": len(runs_canonical),
            "seeds": seeds,
            "n_train": runs_canonical[0].n_train if runs_canonical else None,
            "epochs": runs_canonical[0].epochs if runs_canonical else None,
            "hit_at_005": {"mean": h05_mean, "std": h05_std, "seeds": col("hit_at_005")},
            "hit_at_010": {"mean": h10_mean, "std": h10_std, "seeds": col("hit_at_010")},
            "hit_at_025": {"mean": h25_mean, "std": h25_std, "seeds": col("hit_at_025")},
            "mean_norm_l2": {"mean": l2_mean, "std": l2_std},
            "final_loss": {"mean": loss_mean, "std": loss_std},
            "files": [r.file_name for r in runs_canonical],
        }
    return summary


def delta(a: dict, b: dict, metric: str) -> dict:
    """Compute mean / pooled std for B - A (variant D minus variant A)."""
    am, as_ = a[metric]["mean"], a[metric]["std"]
    bm, bs = b[metric]["mean"], b[metric]["std"]
    delta = bm - am
    pooled = math.sqrt(as_ ** 2 + bs ** 2)
    return {"delta": delta, "pooled_std": pooled,
            "n_sigma": abs(delta) / pooled if pooled > 0 else float("inf")}


def print_table(summary: dict) -> None:
    mixes = sorted({s["data_mix"] for s in summary.values()})
    print(f"\n{'mix':<22} {'variant':<8} {'n':<4} {'loss':>8} {'hit@.05':>14} {'hit@.10':>14} {'hit@.25':>14} {'norm L2':>14}")
    print("-" * 110)
    for mix in mixes:
        for variant in ("A", "D"):
            key = f"{variant}__{mix}"
            if key not in summary:
                continue
            s = summary[key]
            def fmt(metric: str) -> str:
                m = s[metric]["mean"]
                std = s[metric]["std"]
                if math.isnan(m):
                    return "n/a".rjust(14)
                if s["n_seeds"] > 1:
                    return f"{m:>5.3f} ± {std:>5.3f}"
                return f"{m:>5.3f}        "
            print(f"{mix:<22} {variant:<8} {s['n_seeds']:<4} "
                  f"{s['final_loss']['mean']:>8.3f} "
                  f"{fmt('hit_at_005')}  "
                  f"{fmt('hit_at_010')}  "
                  f"{fmt('hit_at_025')}  "
                  f"{fmt('mean_norm_l2')}")
        # Delta within this mix
        a_key, d_key = f"A__{mix}", f"D__{mix}"
        if a_key in summary and d_key in summary:
            for metric, label in (("hit_at_010", "hit@.10"), ("mean_norm_l2", "norm L2")):
                d = delta(summary[a_key], summary[d_key], metric)
                sign = "+" if d["delta"] >= 0 else ""
                print(f"  -> {mix} (D - A) {label}: {sign}{d['delta']:.4f}  pooled_std={d['pooled_std']:.4f}  n_sigma={d['n_sigma']:.2f}")


def plot_ablation(summary: dict, out_path: Path) -> None:
    mixes = sorted({s["data_mix"] for s in summary.values()})
    x_labels: list[str] = []
    a_means, a_stds, d_means, d_stds = [], [], [], []
    for mix in mixes:
        a_key = f"A__{mix}"
        d_key = f"D__{mix}"
        a = summary.get(a_key)
        d = summary.get(d_key)
        if not (a and d):
            continue
        x_labels.append(f"{mix}\n(n_train={a.get('n_train','?')}, seeds={a.get('n_seeds','?')})")
        a_means.append(a["hit_at_010"]["mean"])
        a_stds.append(a["hit_at_010"]["std"])
        d_means.append(d["hit_at_010"]["mean"])
        d_stds.append(d["hit_at_010"]["std"])

    if not x_labels:
        print("[p5] no paired A/D groups to plot")
        return

    x = np.arange(len(x_labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(2 + 3 * len(x_labels), 5))
    ax.bar(x - w / 2, a_means, w, yerr=a_stds, capsize=5, label="A (flat baseline)", color="#3a6ea5")
    ax.bar(x + w / 2, d_means, w, yerr=d_stds, capsize=5, label="D (action-conditioned)", color="#9d4edd")
    for i, (am, ds_) in enumerate(zip(a_means, d_means)):
        ax.text(x[i] - w / 2, am + 0.02, f"{am:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + w / 2, ds_ + 0.02, f"{ds_:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("hit@0.10 (mean ± std)")
    ax.set_ylim(0, max(max(a_means + d_means) + 0.15, 0.7))
    ax.axhline(math.pi * 0.10 ** 2, ls=":", color="#444",
               label=f"uniform-prior hit@0.10 ({math.pi * 0.01:.3f})")
    ax.set_title("Phase 5 ablation: Variant A vs Variant D on AITW grounding")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = load_runs()
    print(f"[p5] loaded {len(rows)} training run JSONs from {PHASE4_DIR}")
    if not rows:
        return
    summary = aggregate(rows)
    (PHASE4_DIR / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[p5] wrote {PHASE4_DIR / 'ablation_summary.json'}")

    plot_ablation(summary, PHASE4_DIR / "ablation_A_vs_D.png")
    print(f"[p5] wrote {PHASE4_DIR / 'ablation_A_vs_D.png'}")

    print_table(summary)


if __name__ == "__main__":
    main()
