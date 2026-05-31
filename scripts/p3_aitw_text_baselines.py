"""Phase 3.4 — TF-IDF + LogReg baselines on AITW (text features only).

Tests whether bag-of-words on the goal_info string alone can predict the
canonical action type. Milestone 3 on Mind2Web hit macro-F1 0.622 with this
recipe; the Phase 3.5 Stage 1 result on AITW shows that with screenshots,
the MLP hits 0.515 — but with TEXT only, the Stage 1 MLP only got 0.159.

This script confirms whether TF-IDF + LogReg can do better than a Stage 1
text-only MLP. If TF-IDF also collapses to a low number, that's strong
evidence the AITW action-type problem is genuinely vision-decidable and
text-undecidable.

Outputs:
  results/phase3/aitw_text_baselines.json    — every baseline's metrics
  results/phase3/aitw_text_confusion_*.png   — one confusion matrix per
"""

from __future__ import annotations

import json
from collections import Counter
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

from src.data.aitw import load_aitw_steps
from src.data.taxonomy import ID_TO_ACTION

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "phase3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def featurize(steps, *, with_typed_text: bool) -> list[str]:
    out = []
    for s in steps:
        parts = [f"Goal: {s.goal_info}"]
        if with_typed_text and s.typed_text:
            parts.append(f"Typed text: {s.typed_text}")
        out.append("\n".join(parts))
    return out


def eval_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(set(y_true.tolist()))
    target_names = [ID_TO_ACTION[i] for i in labels]
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cls_rep = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {target_names[i]: cls_rep[target_names[i]] for i in range(len(target_names))},
        "confusion_matrix": cm,
        "confusion_matrix_labels": target_names,
        "prediction_distribution": {
            ID_TO_ACTION[k]: int(v) for k, v in Counter(y_pred.tolist()).items()
        },
    }


def plot_confusion(result: dict, out_path: Path) -> None:
    cm = np.array(result["confusion_matrix"])
    labels = result["confusion_matrix_labels"]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{result['name']}\nmacro-F1={result['macro_f1']:.3f}, acc={result['accuracy']:.3f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    n_train, n_val = 5000, 1000
    print(f"[p3-text] loading AITW {n_train} train + {n_val} val (text only)...")
    train_steps = load_aitw_steps(split="train", n_max=n_train, include_images=False)
    val_steps = load_aitw_steps(split="test", n_max=n_val, include_images=False)
    print(f"[p3-text] loaded train={len(train_steps)} val={len(val_steps)}")

    y_train = np.array([s.canonical_action_id for s in train_steps])
    y_val = np.array([s.canonical_action_id for s in val_steps])

    feature_sets = {
        "goal_only": dict(with_typed_text=False),
        "goal_plus_typed": dict(with_typed_text=True),
    }

    results: list[dict] = []

    # majority class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_val), 1)))
    results.append(eval_model("majority_class", y_val, y_pred))

    # stratified random
    rng = np.random.default_rng(0)
    train_dist = Counter(y_train.tolist())
    classes = list(train_dist.keys())
    probs = np.array([train_dist[c] for c in classes], dtype=float)
    probs /= probs.sum()
    y_pred = rng.choice(classes, size=len(y_val), p=probs)
    results.append(eval_model("stratified_random", y_val, y_pred))

    # TF-IDF + LogReg
    for fs_name, fs_kw in feature_sets.items():
        for class_weight in (None, "balanced"):
            X_train = featurize(train_steps, **fs_kw)
            X_val = featurize(val_steps, **fs_kw)
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(min_df=2, max_features=50_000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight)),
            ])
            print(f"[p3-text] training tfidf_logreg/{fs_name}/cw={class_weight}...")
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            results.append(eval_model(f"tfidf_logreg/{fs_name}/cw={class_weight}", y_val, y_pred))

    # save + plot
    out_json = RESULTS_DIR / "aitw_text_baselines.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[p3-text] wrote {out_json}")

    for r in results:
        safe = r["name"].replace("/", "__").replace("=", "-")
        plot_confusion(r, RESULTS_DIR / f"aitw_text_confusion_{safe}.png")

    print("\n=== AITW text-only baselines (n_train=5000, n_val=1000) ===")
    print(f"{'model':<55} {'acc':>6} {'macroF1':>8}  {'weightedF1':>10}")
    for r in results:
        print(f"{r['name']:<55} {r['accuracy']:>6.3f} {r['macro_f1']:>8.3f}  {r['weighted_f1']:>10.3f}")


if __name__ == "__main__":
    main()
