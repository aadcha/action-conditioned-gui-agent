"""Aggregate every Milestone 3 result into one numbers.json + headline figures.

Run after every individual m3_* script. Idempotent — re-run safely after the
Modal zero-shot job finishes to fold those numbers in.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.mind2web import load_mind2web_text
from src.data.taxonomy import ID_TO_ACTION

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "milestone3"


def load_optional(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def dataset_stats() -> dict:
    train, val = load_mind2web_text()
    train_dist = Counter(s.canonical_action_id for s in train)
    val_dist = Counter(s.canonical_action_id for s in val)
    return {
        "source": "osunlp/Mind2Web (HF, unauth)",
        "split_strategy": "task-level held-out from train (val_frac=0.15, seed=42)",
        "train_tasks": len({s.annotation_id for s in train}),
        "val_tasks": len({s.annotation_id for s in val}),
        "train_steps": len(train),
        "val_steps": len(val),
        "raw_action_counts": {
            "CLICK": sum(1 for s in train + val if s.raw_op == "CLICK"),
            "TYPE": sum(1 for s in train + val if s.raw_op == "TYPE"),
            "SELECT": sum(1 for s in train + val if s.raw_op == "SELECT"),
        },
        "canonical_action_distribution_train": {
            ID_TO_ACTION[k]: v for k, v in sorted(train_dist.items())
        },
        "canonical_action_distribution_val": {
            ID_TO_ACTION[k]: v for k, v in sorted(val_dist.items())
        },
        "majority_class_baseline_accuracy": max(val_dist.values()) / sum(val_dist.values()),
    }


def plot_per_class_f1(baselines: list[dict], out_path: Path) -> None:
    """Per-class F1 across the headline models. Highlights how the type class
    is the hard one and how class-balanced training helps it."""
    headline = [
        b for b in baselines
        if b["name"] in {
            "majority_class",
            "tfidf_logreg/instr_only/cw=None",
            "tfidf_logreg/instr_only/cw=balanced",
            "tfidf_logreg/instr_plus_history/cw=balanced",
            "tfidf_logreg/instr_plus_history_plus_html/cw=balanced",
        }
    ]
    classes = ["click", "type"]
    x = np.arange(len(classes))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = ["#888888", "#5b8def", "#3a6ea5", "#c08552", "#7a5195"]
    for i, b in enumerate(headline):
        f1s = [b["per_class"].get(c, {}).get("f1-score", 0.0) for c in classes]
        ax.bar(x + (i - len(headline) / 2) * width, f1s, width,
               label=b["name"].replace("tfidf_logreg/", ""), color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 (val, task-held-out)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class F1 across text-only Stage 1 baselines (Mind2Web)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_bar(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["macro_f1"])
    names = [r["name"] for r in rows]
    macro = [r["macro_f1"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - 0.2, macro, 0.4, label="macro-F1", color="#3a6ea5")
    ax.barh(y + 0.2, acc, 0.4, label="accuracy", color="#c08552")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.472, ls=":", color="#888", label="majority macro-F1 floor")
    ax.set_xlabel("score")
    ax.set_title("Stage 1 text-only baselines — Mind2Web val (1199 steps, task-held-out)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    stats = dataset_stats()
    text_baselines = load_optional(RESULTS_DIR / "text_baselines.json") or []
    zero_shot = load_optional(RESULTS_DIR / "zero_shot_qwen2vl.json")

    consolidated = {
        "milestone": 3,
        "date": "2026-05-29",
        "dataset": stats,
        "text_baselines": text_baselines,
        "zero_shot_qwen2vl": zero_shot,
        "headline_numbers": {
            "majority_baseline_macro_f1": 0.472,
            "best_text_only_macro_f1": max(
                (b["macro_f1"] for b in text_baselines), default=None
            ),
            "best_text_only_model": max(
                text_baselines, key=lambda b: b["macro_f1"]
            )["name"] if text_baselines else None,
            "zero_shot_qwen2vl_macro_f1": zero_shot["macro_f1"] if zero_shot else None,
        },
    }

    out = RESULTS_DIR / "numbers.json"
    out.write_text(json.dumps(consolidated, indent=2))
    print(f"[m3] wrote {out}")

    if text_baselines:
        plot_per_class_f1(text_baselines, RESULTS_DIR / "per_class_f1.png")
        plot_summary_bar(text_baselines, RESULTS_DIR / "summary_bar.png")
        print(f"[m3] wrote per_class_f1.png + summary_bar.png")

    print("\n=== Headline ===")
    print(f"majority baseline macro-F1: 0.472")
    print(f"best text-only baseline:    {consolidated['headline_numbers']['best_text_only_model']}")
    print(f"  macro-F1: {consolidated['headline_numbers']['best_text_only_macro_f1']:.3f}")
    if zero_shot:
        print(f"zero-shot Qwen2-VL-2B:      macro-F1 {zero_shot['macro_f1']:.3f}, acc {zero_shot['accuracy']:.3f}")
    else:
        print("zero-shot Qwen2-VL-2B:      not yet run (modal run modal_app.py::zero_shot_m3)")


if __name__ == "__main__":
    main()
