"""Phase 7 helpers for end-to-end Stage 1 -> Stage 2 evaluation.

The GPU-heavy training/evaluation lives in ``modal_app.py``. This module keeps
the pure bookkeeping pieces testable locally: deterministic AITW filtering,
prediction summaries, metric aggregation from per-example distances, and
Stage-1-correct vs incorrect grounding slices.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.taxonomy import ID_TO_ACTION, NUM_ACTIONS

PHASE7_VARIANTS: tuple[str, ...] = (
    "A_flat",
    "D_gold",
    "D_predicted",
    "D_majority",
    "D_random",
)

PHASE7_METRICS: tuple[str, ...] = (
    "hit_at_005",
    "hit_at_010",
    "hit_at_025",
    "mean_normalized_l2",
)

DATA_MIX_LABELS: dict[str, set[str]] = {
    "taps_only": {"tap"},
    "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
    "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
}


class AITWLikeStep(Protocol):
    string_label: str
    canonical_action_id: int


@dataclass(frozen=True)
class Phase7Split:
    train_steps: list[AITWLikeStep]
    val_steps: list[AITWLikeStep]
    allowed_labels: set[str]


def select_phase7_steps(
    steps: Iterable[AITWLikeStep],
    *,
    n_train: int,
    n_val: int,
    data_mix: str = "all_with_coords",
) -> Phase7Split:
    """Filter AITW steps exactly like Phase 6 Stage 2 and split by stream order."""
    if data_mix not in DATA_MIX_LABELS:
        raise ValueError(f"unknown data_mix {data_mix!r}; known={sorted(DATA_MIX_LABELS)}")
    allowed = DATA_MIX_LABELS[data_mix]
    needed = n_train + n_val
    chosen: list[AITWLikeStep] = []
    for step in steps:
        if step.string_label not in allowed:
            continue
        chosen.append(step)
        if len(chosen) >= needed:
            break
    if len(chosen) < needed:
        raise ValueError(
            f"only found {len(chosen)} eligible steps for data_mix={data_mix!r}; "
            f"need {needed} ({n_train} train + {n_val} val)"
        )
    return Phase7Split(
        train_steps=chosen[:n_train],
        val_steps=chosen[n_train:needed],
        allowed_labels=set(allowed),
    )


def validate_action_predictions(
    gold_ids: Iterable[int],
    pred_ids: Iterable[int],
) -> tuple[list[int], list[int]]:
    """Return validated gold/pred action IDs, raising on length/range mismatch."""
    gold = [int(x) for x in gold_ids]
    pred = [int(x) for x in pred_ids]
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    bad = [x for x in pred if x < 0 or x >= NUM_ACTIONS]
    if bad:
        raise ValueError(f"predicted action IDs out of range [0,{NUM_ACTIONS - 1}]: {bad[:10]}")
    return gold, pred


def summarize_stage1_predictions(
    gold_ids: Iterable[int],
    pred_ids: Iterable[int],
    confidences: Iterable[float] | None = None,
) -> dict:
    """Classification metrics for Stage 1 on the exact Stage 2 validation set."""
    gold, pred = validate_action_predictions(gold_ids, pred_ids)
    labels = sorted(set(gold))
    target_names = [ID_TO_ACTION[i] for i in labels]
    report = classification_report(
        gold,
        pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    out = {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(gold, pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {name: report[name] for name in target_names},
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
        "confusion_matrix_labels": target_names,
        "labels_present": labels,
        "prediction_distribution": {ID_TO_ACTION[k]: int(v) for k, v in Counter(pred).items()},
        "ground_truth_distribution": {ID_TO_ACTION[k]: int(v) for k, v in Counter(gold).items()},
    }
    if confidences is not None:
        conf = np.asarray(list(confidences), dtype=np.float64)
        if conf.shape[0] != len(gold):
            raise ValueError(f"confidence length mismatch: {conf.shape[0]} vs {len(gold)}")
        out["confidence"] = {
            "mean": float(conf.mean()) if conf.size else float("nan"),
            "std": float(conf.std(ddof=1)) if conf.size > 1 else 0.0,
            "min": float(conf.min()) if conf.size else float("nan"),
            "max": float(conf.max()) if conf.size else float("nan"),
        }
    return out


def metrics_from_distances(distances: Iterable[float]) -> dict:
    """Compute grounding metrics from aligned per-example normalized L2 distances."""
    dist = np.asarray(list(distances), dtype=np.float64)
    if dist.ndim != 1:
        raise ValueError(f"distances must be 1D, got shape={dist.shape}")
    if dist.size == 0:
        return {
            "n_total": 0,
            "n_parsed": 0,
            "parse_rate": 0.0,
            "mean_normalized_l2": float("nan"),
            "hit_at_005": 0.0,
            "hit_at_010": 0.0,
            "hit_at_025": 0.0,
            "per_example_dist": [],
        }
    finite = np.isfinite(dist)
    parsed = dist[finite]
    return {
        "n_total": int(dist.size),
        "n_parsed": int(parsed.size),
        "parse_rate": float(parsed.size / dist.size),
        "mean_normalized_l2": float(parsed.mean()) if parsed.size else float("nan"),
        "hit_at_005": float((dist <= 0.05).mean()),
        "hit_at_010": float((dist <= 0.10).mean()),
        "hit_at_025": float((dist <= 0.25).mean()),
        "per_example_dist": dist.tolist(),
    }


def metric_mean_from_distances(distances: Iterable[float], metric: str) -> float:
    dist = np.asarray(list(distances), dtype=np.float64)
    if metric == "mean_normalized_l2":
        return float(dist.mean())
    radii = {"hit_at_005": 0.05, "hit_at_010": 0.10, "hit_at_025": 0.25}
    if metric not in radii:
        raise ValueError(f"unknown metric {metric!r}")
    return float((dist <= radii[metric]).mean())


def stage1_correct_breakdown(
    distances: Iterable[float],
    gold_ids: Iterable[int],
    pred_ids: Iterable[int],
) -> dict:
    """Grounding metrics split by whether Stage 1 predicted the action type correctly."""
    gold, pred = validate_action_predictions(gold_ids, pred_ids)
    dist = np.asarray(list(distances), dtype=np.float64)
    if dist.shape[0] != len(gold):
        raise ValueError(f"distance/action length mismatch: {dist.shape[0]} vs {len(gold)}")
    mask = np.asarray([g == p for g, p in zip(gold, pred, strict=True)], dtype=bool)
    return {
        "stage1_correct": metrics_from_distances(dist[mask].tolist()),
        "stage1_incorrect": metrics_from_distances(dist[~mask].tolist()),
        "n_correct": int(mask.sum()),
        "n_incorrect": int((~mask).sum()),
    }


def make_control_action_ids(
    train_action_ids: Iterable[int],
    gold_val_ids: Iterable[int],
    *,
    seed: int,
) -> dict[str, list[int] | int]:
    """Deterministic majority and random action controls for Phase 7."""
    train = [int(x) for x in train_action_ids]
    gold = [int(x) for x in gold_val_ids]
    if not train:
        raise ValueError("train_action_ids must be non-empty")
    counts = Counter(train)
    majority_id = counts.most_common(1)[0][0]
    present = sorted(counts)
    rng = np.random.default_rng(seed + 7000)
    random_ids = rng.choice(present, size=len(gold), replace=True).astype(int).tolist()
    return {
        "majority_id": int(majority_id),
        "majority_ids": [int(majority_id)] * len(gold),
        "random_ids": random_ids,
        "random_pool": present,
    }


def distance_or_sentinel(pred_xy: tuple[float, float] | None, target_xy: tuple[float, float]) -> float:
    """Normalized L2 distance, with sqrt(2) sentinel for parse failures."""
    if pred_xy is None:
        return 2.0**0.5
    return math.hypot(pred_xy[0] - target_xy[0], pred_xy[1] - target_xy[1])
