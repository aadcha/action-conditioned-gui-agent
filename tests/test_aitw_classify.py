"""Unit tests for AITW row → canonical-action conversion.

These don't touch the network; they exercise pure logic over synthetic rows.
"""

from src.data.aitw import (
    AITW_ACTION_TYPE_ENUM,
    DEFAULT_TAP_THRESHOLD,
    _row_to_step,
    classify_dual_point,
    row_to_string_label,
)
from src.data.taxonomy import CANONICAL_ACTIONS


def _row(action_type: int, touch=(0.5, 0.5), lift=(0.5, 0.5), goal="g", typed=None):
    return {
        "ep_id": "ep1",
        "step_id": 0,
        "goal_info": goal,
        "image": None,
        "results_action_type": action_type,
        "results_yx_touch": list(touch),
        "results_yx_lift": list(lift),
        "results_type_action": [typed] if typed is not None else [],
    }


def test_classify_dual_point_tap_when_no_motion():
    assert classify_dual_point((0.5, 0.5), (0.5, 0.5)) == "tap"


def test_classify_dual_point_small_motion_is_tap():
    eps = DEFAULT_TAP_THRESHOLD / 2
    assert classify_dual_point((0.5, 0.5), (0.5 + eps, 0.5 + eps)) == "tap"


def test_classify_dual_point_swipe_directions():
    # y grows downward
    assert classify_dual_point((0.8, 0.5), (0.2, 0.5)) == "swipe_up"
    assert classify_dual_point((0.2, 0.5), (0.8, 0.5)) == "swipe_down"
    assert classify_dual_point((0.5, 0.8), (0.5, 0.2)) == "swipe_left"
    assert classify_dual_point((0.5, 0.2), (0.5, 0.8)) == "swipe_right"


def test_row_to_string_label_covers_all_known_types():
    # DUAL_POINT routes to tap (no motion) or swipe (motion)
    assert row_to_string_label(_row(4)) == "tap"
    assert row_to_string_label(_row(4, touch=(0.8, 0.5), lift=(0.2, 0.5))) == "swipe_up"
    # Other action types map directly
    assert row_to_string_label(_row(3, typed="hello")) == "type"
    assert row_to_string_label(_row(5)) == "press_back"
    assert row_to_string_label(_row(6)) == "press_home"
    assert row_to_string_label(_row(7)) == "press_enter"
    assert row_to_string_label(_row(10)) == "status_task_complete"
    assert row_to_string_label(_row(11)) == "status_task_impossible"


def test_row_to_string_label_returns_none_for_unknown_int():
    assert row_to_string_label(_row(99)) is None


def test_row_to_step_attaches_canonical_id():
    step = _row_to_step(_row(4, goal="open chrome"), include_images=False)
    assert step is not None
    assert step.string_label == "tap"
    assert step.canonical_action_id == CANONICAL_ACTIONS["click"]
    assert step.goal_info == "open chrome"
    assert step.image_bytes is None


def test_row_to_step_captures_typed_text():
    step = _row_to_step(_row(3, typed="hello world"), include_images=False)
    assert step is not None
    assert step.string_label == "type"
    assert step.canonical_action_id == CANONICAL_ACTIONS["type"]
    assert step.typed_text == "hello world"


def test_guess_aitw_shape_recognizes_observed_lengths():
    """The byte lengths empirically observed in cjfcsjt/AITW_General/standard
    must all resolve to a plausible (w, h) — otherwise feature extraction
    crashes on those rows."""
    from src.data.aitw import _guess_aitw_shape
    observed_lengths = [
        1_749_600,  # 540x1080
        1_846_800,  # 540x1140
        1_895_400,  # 540x1170
        904_752,    # 412x732
        3_283_200,  # 720x1520
        3_110_400,  # 720x1440
        486_000,    # 270x600
    ]
    for L in observed_lengths:
        shape = _guess_aitw_shape(L)
        assert shape is not None, f"len {L} must resolve to a shape"
        w, h = shape
        assert w * h * 3 == L, f"shape {w}x{h} doesn't account for {L} bytes"
        assert 1.2 * w <= h <= 3.0 * w, f"shape {w}x{h} is not portrait"


def test_action_type_enum_is_complete():
    # Sanity: every name in the enum must be representable in our taxonomy
    # via either DUAL_POINT classification or _NON_DUAL_POINT_TO_STRING.
    for name in AITW_ACTION_TYPE_ENUM.values():
        if name == "DUAL_POINT":
            continue
        # Round-trip: synthesize a row with this raw type, get string label
        raw_int = next(k for k, v in AITW_ACTION_TYPE_ENUM.items() if v == name)
        label = row_to_string_label(_row(raw_int))
        assert label is not None, f"AITW name {name!r} has no taxonomy mapping"
