"""Milestone 3 — Stage 1 action-type classifier: text-only baselines on Mind2Web.

Produces real, quantitative preliminary results without any GPU. These baselines
directly measure the "text leakage" risk flagged as risk #1 in the project
overview: if text-only classifiers already saturate the metric, the vision
encoder isn't doing much work on Mind2Web and the project framing has to shift.

Outputs all metrics to results/milestone3/text_baselines.json plus a confusion
matrix figure per model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.data.mind2web import load_mind2web_text, step_to_instruction_text
from src.data.taxonomy import CANONICAL_ACTIONS, ID_TO_ACTION

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "milestone3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def featurize(steps, *, with_history: bool, with_html: bool) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    for step in steps:
        parts = [step.confirmed_task]
        if with_history and step.history:
            parts.append("History: " + " | ".join(step.history))
        if with_html:
            parts.append(f"Target HTML: {step.target_html}")
        texts.append("\n".join(parts))
    labels = np.array([s.canonical_action_id for s in steps])
    return texts, labels


def eval_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    classes_present = sorted(set(y_true.tolist() + y_pred.tolist()))
    labels_for_report = sorted(set(y_true.tolist()))  # only score on classes seen in val
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_for_report, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels_for_report, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    per_class = classification_report(
        y_true,
        y_pred,
        labels=labels_for_report,
        target_names=[ID_TO_ACTION[i] for i in labels_for_report],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report).tolist()
    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            ID_TO_ACTION[c]: per_class[ID_TO_ACTION[c]]
            for c in labels_for_report
        },
        "confusion_matrix": cm,
        "confusion_matrix_labels": [ID_TO_ACTION[c] for c in labels_for_report],
        "classes_present_in_predictions": [ID_TO_ACTION[c] for c in classes_present],
    }


def plot_confusion(result: dict, out_path: Path) -> None:
    cm = np.array(result["confusion_matrix"])
    labels = result["confusion_matrix_labels"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{result['name']}\nmacro-F1 = {result['macro_f1']:.3f}, acc = {result['accuracy']:.3f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(train_steps, val_steps, out_path: Path) -> None:
    from collections import Counter
    train_c = Counter(s.canonical_action_id for s in train_steps)
    val_c = Counter(s.canonical_action_id for s in val_steps)
    classes = sorted(CANONICAL_ACTIONS.values())
    labels = [ID_TO_ACTION[c] for c in classes]
    train_counts = [train_c.get(c, 0) for c in classes]
    val_counts = [val_c.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, train_counts, w, label=f"train (n={sum(train_counts)})", color="#3a6ea5")
    ax.bar(x + w / 2, val_counts, w, label=f"val (n={sum(val_counts)})", color="#c08552")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("step count")
    ax.set_title("Mind2Web canonical action distribution\n(SELECT/HOVER collapsed into click)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("[m3] loading Mind2Web...")
    train_steps, val_steps = load_mind2web_text()
    print(f"[m3] train: {len(train_steps)} steps, val: {len(val_steps)} steps")

    plot_class_distribution(train_steps, val_steps, RESULTS_DIR / "class_distribution.png")

    feature_sets = {
        "instr_only": dict(with_history=False, with_html=False),
        "instr_plus_history": dict(with_history=True, with_html=False),
        "instr_plus_html": dict(with_history=False, with_html=True),
        "instr_plus_history_plus_html": dict(with_history=True, with_html=True),
    }

    results: list[dict] = []

    # ------- baselines that don't need text -------
    y_train = np.array([s.canonical_action_id for s in train_steps])
    y_val = np.array([s.canonical_action_id for s in val_steps])

    # majority class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_val), 1)))
    results.append(eval_model("majority_class", y_val, y_pred))

    # stratified random
    rng = np.random.default_rng(0)
    from collections import Counter
    train_dist = Counter(y_train.tolist())
    classes = list(train_dist.keys())
    probs = np.array([train_dist[c] for c in classes], dtype=float)
    probs /= probs.sum()
    y_pred = rng.choice(classes, size=len(y_val), p=probs)
    results.append(eval_model("stratified_random", y_val, y_pred))

    # ------- TF-IDF + LogReg under different feature sets -------
    for fs_name, fs_kw in feature_sets.items():
        for class_weight in (None, "balanced"):
            X_train, _ = featurize(train_steps, **fs_kw)
            X_val, _ = featurize(val_steps, **fs_kw)
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(min_df=2, max_features=50_000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight)),
            ])
            print(f"[m3] training tfidf_logreg/{fs_name}/cw={class_weight}...")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            tag = f"tfidf_logreg/{fs_name}/cw={class_weight}"
            results.append(eval_model(tag, y_val, y_pred))

    # ------- save results -------
    out_json = RESULTS_DIR / "text_baselines.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[m3] wrote {out_json}")

    # confusion matrices for the headline baselines
    for r in results:
        safe = r["name"].replace("/", "__").replace("=", "-")
        plot_confusion(r, RESULTS_DIR / f"confusion_{safe}.png")

    # quick console summary
    print("\n=== Stage 1 text-only baselines (Mind2Web val, task-held-out) ===")
    print(f"{'model':<50} {'acc':>6} {'macroF1':>8}  {'weightedF1':>10}")
    for r in results:
        print(f"{r['name']:<50} {r['accuracy']:>6.3f} {r['macro_f1']:>8.3f}  {r['weighted_f1']:>10.3f}")


if __name__ == "__main__":
    main()
