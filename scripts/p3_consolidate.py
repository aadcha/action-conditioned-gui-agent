"""Phase 3 — figures + writeup from AITW Stage 1 + Mind2Web comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.taxonomy import ID_TO_ACTION

PHASE2_DIR = Path(__file__).resolve().parent.parent / "results" / "phase2"
PHASE3_DIR = Path(__file__).resolve().parent.parent / "results" / "phase3"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def plot_variant_comparison(results: dict, out_path: Path) -> None:
    variants = ["vision_zeroed", "text_only", "vision_text"]
    macro_f1 = [results["variants"][v]["best_val_macro_f1"] for v in variants]
    acc = [results["variants"][v]["best_val_metrics"]["accuracy"] for v in variants]

    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, macro_f1, width=0.55, color=["#bdbdbd", "#3a6ea5", "#9d4edd"])
    for i, v in enumerate(macro_f1):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["vision_zeroed", "text_only", "vision+text"])
    ax.set_ylabel(f"val macro-F1 (AITW test, n={results['n_val']})")
    ax.set_ylim(0, max(1.0, max(macro_f1) + 0.1))
    # Naive 1/k floor for 5-class problem
    ax.axhline(1.0 / 5, ls=":", color="#888", label="uniform-prior macro-F1 (5 classes)")
    ax.set_title(
        f"Phase 3.5 — Stage 1 MLP on AITW (cjfcsjt/AITW_General, n_train={results['n_train']})"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_per_variant(results: dict, out_path: Path) -> None:
    variants = ["text_only", "vision_text", "vision_zeroed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(results: dict, out_path: Path) -> None:
    variants = ["text_only", "vision_text"]
    # union of class names across variants
    class_names: list[str] = []
    for v in variants:
        for c in results["variants"][v]["best_val_metrics"]["per_class"].keys():
            if c not in class_names:
                class_names.append(c)
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3a6ea5", "#9d4edd"]
    for i, v in enumerate(variants):
        per_class = results["variants"][v]["best_val_metrics"]["per_class"]
        f1s = [per_class.get(c, {}).get("f1-score", 0.0) for c in class_names]
        ax.bar(x + (i - 0.5) * width, f1s, width, label=v, color=colors[i])
        for j, f in enumerate(f1s):
            ax.text(x[j] + (i - 0.5) * width, f + 0.01, f"{f:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("per-class F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase 3.5 — Per-class F1 on AITW val")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mind2web_vs_aitw(out_path: Path) -> None:
    """Side-by-side comparison: Stage 1 macro-F1 on Mind2Web (2-class) vs AITW (5-class)."""
    m2w = _load(PHASE2_DIR / "stage1_multiseed.json")
    aitw = _load(PHASE3_DIR / "stage1_aitw_results.json")
    if not (m2w and aitw):
        print("[p3-consolidate] skipping mind2web_vs_aitw — both numbers not yet available")
        return

    variants = ["text_only", "vision_text", "vision_zeroed"]
    m2w_values = [m2w["per_variant"][v]["macro_f1_mean"] for v in variants]
    m2w_err = [m2w["per_variant"][v]["macro_f1_std"] for v in variants]
    aitw_values = [aitw["variants"][v]["best_val_macro_f1"] for v in variants]

    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, m2w_values, width, yerr=m2w_err, capsize=4,
           label="Mind2Web (2-class, 3 seeds)", color="#3a6ea5")
    ax.bar(x + width / 2, aitw_values, width,
           label="AITW (5-class, 1 seed)", color="#c08552")
    for i, v in enumerate(m2w_values):
        ax.text(x[i] - width / 2, v + m2w_err[i] + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(aitw_values):
        ax.text(x[i] + width / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("val macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Stage 1 MLP: Mind2Web (2-class) vs AITW (5-class)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate_aitw_seeds(seed_files: list[Path]) -> dict:
    seed_results = [json.loads(p.read_text()) for p in seed_files]
    per_variant: dict[str, dict] = {}
    for v in ("text_only", "vision_text", "vision_zeroed"):
        macros = [r["variants"][v]["best_val_macro_f1"] for r in seed_results]
        accs = [r["variants"][v]["best_val_metrics"]["accuracy"] for r in seed_results]
        per_variant[v] = {
            "macro_f1_mean": float(np.mean(macros)),
            "macro_f1_std": float(np.std(macros, ddof=1)) if len(macros) > 1 else 0.0,
            "macro_f1_seeds": macros,
            "accuracy_mean": float(np.mean(accs)),
        }
    deltas_vt = [r["delta_vision_minus_text_only"] for r in seed_results]
    deltas_vz = [r["delta_vision_minus_zeroed"] for r in seed_results]
    return {
        "n_seeds": len(seed_results),
        "per_variant": per_variant,
        "vision_minus_text_only": {
            "mean": float(np.mean(deltas_vt)),
            "std": float(np.std(deltas_vt, ddof=1)) if len(deltas_vt) > 1 else 0.0,
            "seeds": deltas_vt,
        },
        "vision_minus_zeroed": {
            "mean": float(np.mean(deltas_vz)),
            "std": float(np.std(deltas_vz, ddof=1)) if len(deltas_vz) > 1 else 0.0,
        },
    }


def plot_aitw_multiseed_bar(agg: dict, tfidf: list[dict] | None, out_path: Path) -> None:
    variants = ["vision_zeroed", "text_only", "vision_text"]
    means = [agg["per_variant"][v]["macro_f1_mean"] for v in variants]
    stds = [agg["per_variant"][v]["macro_f1_std"] for v in variants]
    labels = ["MLP\nvision_zeroed", "MLP\ntext_only", "MLP\nvision+text"]

    tfidf_lookup: dict[str, float] = {}
    if tfidf:
        for r in tfidf:
            tfidf_lookup[r["name"]] = r["macro_f1"]

    extra_labels: list[str] = []
    extra_values: list[float] = []
    for k, label in (
        ("majority_class", "majority\n(text baseline)"),
        ("tfidf_logreg/goal_only/cw=balanced", "TF-IDF\ngoal_only"),
        ("tfidf_logreg/goal_plus_typed/cw=balanced", "TF-IDF\ngoal+typed\n(leaky)"),
    ):
        if k in tfidf_lookup:
            extra_labels.append(label)
            extra_values.append(tfidf_lookup[k])

    all_labels = extra_labels + labels
    all_values = extra_values + means
    all_errs = [0.0] * len(extra_values) + stds

    x = np.arange(len(all_labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#888"] * len(extra_values) + ["#bdbdbd", "#3a6ea5", "#9d4edd"]
    ax.bar(x, all_values, yerr=all_errs, capsize=4, color=colors)
    for i, (v, e) in enumerate(zip(all_values, all_errs)):
        ax.text(i, v + (e or 0) + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylabel(f"val macro-F1 on AITW test (n=1000)")
    ax.set_ylim(0, 1.0)
    ax.axhline(1.0 / 5, ls=":", color="#444", label="uniform-prior macro-F1 (5 classes = 0.2)")
    ax.set_title(
        f"AITW Stage 1 — text baselines vs Qwen2-VL-2B MLP\n"
        f"({agg['n_seeds']}-seed mean ± std for MLPs)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    src = PHASE3_DIR / "stage1_aitw_results.json"
    if not src.exists():
        print(f"[p3-consolidate] {src} not found — run train_stage1_aitw first")
        return
    results = json.loads(src.read_text())

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    plot_variant_comparison(results, PHASE3_DIR / "aitw_stage1_variants.png")
    plot_confusion_per_variant(results, PHASE3_DIR / "aitw_stage1_confusion.png")
    plot_per_class_f1(results, PHASE3_DIR / "aitw_stage1_per_class_f1.png")
    plot_mind2web_vs_aitw(PHASE3_DIR / "stage1_mind2web_vs_aitw.png")
    print(f"[p3-consolidate] single-seed figures written")

    # Multi-seed aggregation
    canonical_seed_files = [
        PHASE3_DIR / "stage1_aitw_results.json",
        PHASE3_DIR / "stage1_aitw_results_seed43.json",
        PHASE3_DIR / "stage1_aitw_results_seed44.json",
    ]
    existing = [p for p in canonical_seed_files if p.exists()]
    if len(existing) > 1:
        agg = aggregate_aitw_seeds(existing)
        (PHASE3_DIR / "stage1_aitw_multiseed.json").write_text(json.dumps(agg, indent=2))
        tfidf = _load(PHASE3_DIR / "aitw_text_baselines.json")
        plot_aitw_multiseed_bar(agg, tfidf, PHASE3_DIR / "aitw_stage1_headline.png")
        print(f"[p3-consolidate] multi-seed aggregation written (n={agg['n_seeds']})")

        print("\n=== AITW Stage 1 multi-seed headline ===")
        for v, stats in agg["per_variant"].items():
            print(f"  {v:<14}  macro-F1 = {stats['macro_f1_mean']:.4f} ± {stats['macro_f1_std']:.4f}")
        print(f"\n  vision_text - text_only:    {agg['vision_minus_text_only']['mean']:+.4f} ± {agg['vision_minus_text_only']['std']:.4f}")

    print("\n(single-seed seed=42 raw)")
    for v, mf1 in results["headline_macro_f1"].items():
        print(f"  {v:<14}  macro-F1 = {mf1:.4f}")


if __name__ == "__main__":
    main()
