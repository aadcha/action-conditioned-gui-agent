"""Phase 2 — turn `results/phase2/stage1_results.json` into figures + a writeup.

Run after `modal run modal_app.py::train_stage1`. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHASE2_DIR = Path(__file__).resolve().parent.parent / "results" / "phase2"
MILESTONE3_DIR = Path(__file__).resolve().parent.parent / "results" / "milestone3"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def plot_variant_comparison(results: dict, out_path: Path) -> None:
    variants = ["vision_zeroed", "text_only", "vision_text"]
    macro_f1 = [results["variants"][v]["best_val_macro_f1"] for v in variants]
    acc = [results["variants"][v]["best_val_metrics"]["accuracy"] for v in variants]

    # Reference lines from milestone 3
    m3 = _load(MILESTONE3_DIR / "numbers.json") or {}
    majority = m3.get("headline_numbers", {}).get("majority_baseline_macro_f1", 0.472)
    tfidf_best = m3.get("headline_numbers", {}).get("best_text_only_baseline_macro_f1", 0.622)

    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, macro_f1, width=0.55, color=["#bdbdbd", "#3a6ea5", "#9d4edd"])
    for i, v in enumerate(macro_f1):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["vision_zeroed\n(sanity check)", "text_only", "vision+text"])
    ax.set_ylabel("val macro-F1 (Multimodal-Mind2Web test_task)")
    ax.set_ylim(0, max(1.0, max(macro_f1) + 0.1))
    ax.axhline(majority, ls=":", color="#888", label=f"majority floor ({majority:.3f})")
    ax.axhline(tfidf_best, ls="--", color="#c08552", label=f"TF-IDF best ({tfidf_best:.3f})")
    ax.set_title("Phase 2 — Stage 1 MLP on cached Qwen2-VL-2B features")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"text_only": "#3a6ea5", "vision_text": "#9d4edd", "vision_zeroed": "#bdbdbd"}
    for variant in ("vision_zeroed", "text_only", "vision_text"):
        hist = results["variants"][variant]["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], color=colors[variant], label=variant)
        axes[1].plot(epochs, [h["val_macro_f1"] for h in hist], color=colors[variant], label=variant)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val macro-F1")
    axes[0].set_title("train loss")
    axes[1].set_title("val macro-F1 across epochs")
    axes[0].legend(loc="upper right"); axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_per_variant(results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    variants = ["text_only", "vision_text", "vision_zeroed"]
    for ax, variant in zip(axes, variants):
        m = results["variants"][variant]["best_val_metrics"]
        cm = np.array(m["confusion_matrix"])
        labels = m["confusion_matrix_labels"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(f"{variant}\nmacro-F1={m['macro_f1']:.3f}, acc={m['accuracy']:.3f}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate_seeds(seed_files: list[Path]) -> dict:
    """Mean / std across seeds for the three variants."""
    if not seed_files:
        return {}
    seed_results = [json.loads(p.read_text()) for p in seed_files]

    variants = ["text_only", "vision_text", "vision_zeroed"]
    per_variant: dict[str, dict] = {}
    for v in variants:
        macros = [r["variants"][v]["best_val_macro_f1"] for r in seed_results]
        accs = [r["variants"][v]["best_val_metrics"]["accuracy"] for r in seed_results]
        type_f1s = [r["variants"][v]["best_val_metrics"]["per_class"].get("type", {}).get("f1-score", 0.0) for r in seed_results]
        click_f1s = [r["variants"][v]["best_val_metrics"]["per_class"].get("click", {}).get("f1-score", 0.0) for r in seed_results]
        per_variant[v] = {
            "macro_f1_mean": float(np.mean(macros)),
            "macro_f1_std": float(np.std(macros, ddof=1)) if len(macros) > 1 else 0.0,
            "macro_f1_seeds": macros,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "click_f1_mean": float(np.mean(click_f1s)),
            "type_f1_mean": float(np.mean(type_f1s)),
        }
    deltas = [r["delta_vision_minus_text_only"] for r in seed_results]
    return {
        "per_variant": per_variant,
        "n_seeds": len(seed_results),
        "seeds": [r["variants"]["text_only"]["train_cfg"]["seed"] for r in seed_results],
        "vision_delta_mean": float(np.mean(deltas)),
        "vision_delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
    }


def plot_variant_comparison_multiseed(agg: dict, headline_path: Path, out_path: Path) -> None:
    variants = ["vision_zeroed", "text_only", "vision_text"]
    means = [agg["per_variant"][v]["macro_f1_mean"] for v in variants]
    stds = [agg["per_variant"][v]["macro_f1_std"] for v in variants]

    m3 = _load(MILESTONE3_DIR / "numbers.json") or {}
    majority = m3.get("headline_numbers", {}).get("majority_baseline_macro_f1", 0.472)
    tfidf_best = m3.get("headline_numbers", {}).get("best_text_only_baseline_macro_f1", 0.622)

    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.55,
                  color=["#bdbdbd", "#3a6ea5", "#9d4edd"])
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.012, f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["vision_zeroed\n(sanity check)", "text_only", "vision+text"])
    ax.set_ylabel("val macro-F1 (Multimodal-Mind2Web test_task)")
    ax.set_ylim(0, max(1.0, max(means) + max(stds) + 0.1))
    ax.axhline(majority, ls=":", color="#888", label=f"majority floor ({majority:.3f})")
    ax.axhline(tfidf_best, ls="--", color="#c08552", label=f"TF-IDF best ({tfidf_best:.3f})")
    ax.set_title(f"Phase 2 — Stage 1 MLP on Qwen2-VL-2B features  (n={agg['n_seeds']} seeds)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    # Single-seed (headline) file used for confusion matrices + curves
    src = PHASE2_DIR / "stage1_results.json"
    if not src.exists():
        print(f"[p2-consolidate] {src} not found — run `modal run modal_app.py::train_stage1` first")
        return
    results = json.loads(src.read_text())

    plot_variant_comparison(results, PHASE2_DIR / "stage1_variants.png")
    plot_training_curves(results, PHASE2_DIR / "stage1_training_curves.png")
    plot_confusion_per_variant(results, PHASE2_DIR / "stage1_confusion.png")
    print(f"[p2-consolidate] wrote 3 single-seed figures")

    # Multi-seed aggregation
    seed_files = sorted(PHASE2_DIR.glob("stage1_results*.json"))
    # Filter to the canonical-config seeds (skip v2 / sweep files)
    canonical_seed_files = [p for p in seed_files if p.name in {
        "stage1_results.json",
        "stage1_results_seed43.json",
        "stage1_results_seed44.json",
    }]
    if canonical_seed_files:
        agg = aggregate_seeds(canonical_seed_files)
        (PHASE2_DIR / "stage1_multiseed.json").write_text(json.dumps(agg, indent=2))
        plot_variant_comparison_multiseed(agg, src, PHASE2_DIR / "stage1_variants_multiseed.png")
        print(f"[p2-consolidate] wrote multi-seed aggregation across {agg['n_seeds']} seeds")

        print("\n=== Phase 2 multi-seed headline ===")
        for v, stats in agg["per_variant"].items():
            print(f"  {v:<14}  macro-F1 = {stats['macro_f1_mean']:.4f} ± {stats['macro_f1_std']:.4f}  "
                  f"(click_F1 {stats['click_f1_mean']:.3f}, type_F1 {stats['type_f1_mean']:.3f})")
        print(f"\n  vision_delta = {agg['vision_delta_mean']:+.4f} ± {agg['vision_delta_std']:.4f}")
    else:
        print("[p2-consolidate] no canonical seed files found — skipping multi-seed agg")


if __name__ == "__main__":
    main()
