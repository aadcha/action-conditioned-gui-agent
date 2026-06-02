from dataclasses import dataclass

import pytest

from src.eval.phase7 import (
    make_control_action_ids,
    metric_mean_from_distances,
    metrics_from_distances,
    select_phase7_steps,
    stage1_correct_breakdown,
    summarize_stage1_predictions,
    validate_action_predictions,
)


@dataclass(frozen=True)
class _Step:
    string_label: str
    canonical_action_id: int
    idx: int


def test_select_phase7_steps_matches_phase6_stream_order():
    steps = [
        _Step("press_back", 5, 0),  # filtered out for all_with_coords
        _Step("tap", 0, 1),
        _Step("swipe_up", 3, 2),
        _Step("type", 2, 3),
        _Step("wait", 6, 4),        # filtered out
        _Step("swipe_left", 3, 5),
        _Step("tap", 0, 6),
    ]

    split = select_phase7_steps(steps, n_train=3, n_val=2, data_mix="all_with_coords")

    assert [s.idx for s in split.train_steps] == [1, 2, 3]
    assert [s.idx for s in split.val_steps] == [5, 6]
    assert split.allowed_labels == {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"}


def test_select_phase7_steps_errors_when_not_enough_examples():
    steps = [_Step("tap", 0, 0)]
    with pytest.raises(ValueError, match="only found"):
        select_phase7_steps(steps, n_train=1, n_val=1)


def test_validate_action_predictions_checks_lengths_and_ranges():
    assert validate_action_predictions([0, 2], [0, 3]) == ([0, 2], [0, 3])
    with pytest.raises(ValueError, match="length mismatch"):
        validate_action_predictions([0, 2], [0])
    with pytest.raises(ValueError, match="out of range"):
        validate_action_predictions([0], [99])


def test_stage1_summary_has_confusion_and_confidence_stats():
    summary = summarize_stage1_predictions(
        [0, 2, 2, 3],
        [0, 0, 2, 3],
        confidences=[0.9, 0.4, 0.8, 0.7],
    )

    assert summary["accuracy"] == 0.75
    assert summary["confusion_matrix_labels"] == ["click", "type", "scroll"]
    assert summary["confidence"]["mean"] == pytest.approx(0.7)


def test_metrics_from_distances_and_metric_mean_direction():
    dists = [0.01, 0.09, 0.20, 2**0.5]
    metrics = metrics_from_distances(dists)

    assert metrics["hit_at_005"] == 0.25
    assert metrics["hit_at_010"] == 0.5
    assert metrics["hit_at_025"] == 0.75
    assert metric_mean_from_distances(dists, "hit_at_010") == 0.5
    assert metric_mean_from_distances(dists, "mean_normalized_l2") == pytest.approx(sum(dists) / 4)


def test_stage1_correct_breakdown_aligns_by_example():
    dists = [0.01, 0.50, 0.03, 0.40]
    gold = [0, 2, 3, 3]
    pred = [0, 0, 3, 2]

    out = stage1_correct_breakdown(dists, gold, pred)

    assert out["n_correct"] == 2
    assert out["n_incorrect"] == 2
    assert out["stage1_correct"]["hit_at_010"] == 1.0
    assert out["stage1_incorrect"]["hit_at_010"] == 0.0


def test_control_action_ids_are_deterministic_and_valid():
    a = make_control_action_ids([0, 0, 2, 3], [0, 2, 3, 0], seed=42)
    b = make_control_action_ids([0, 0, 2, 3], [0, 2, 3, 0], seed=42)

    assert a == b
    assert a["majority_id"] == 0
    assert a["majority_ids"] == [0, 0, 0, 0]
    assert set(a["random_ids"]).issubset({0, 2, 3})
