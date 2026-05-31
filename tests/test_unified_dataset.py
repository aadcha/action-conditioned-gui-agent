"""Unit tests for the unified dataset abstraction.

Uses synthetic Mind2Web/AITW step inputs — no network calls.
"""

from src.data.aitw import AITWStep
from src.data.mind2web import Mind2WebStep
from src.data.taxonomy import CANONICAL_ACTIONS, ID_TO_ACTION
from src.data.unified import (
    UnifiedStep,
    _from_aitw,
    _from_mind2web,
    class_balanced_indices,
)


def _mind2web_step(label_id: int) -> Mind2WebStep:
    return Mind2WebStep(
        annotation_id="task1",
        domain="d",
        website="w",
        confirmed_task="click submit",
        step_index=0,
        history=(),
        raw_op="CLICK",
        canonical_action_id=label_id,
        target_html="<button>OK</button>",
    )


def _aitw_step(label_id: int, raw_label: str = "tap", typed: str = "") -> AITWStep:
    return AITWStep(
        ep_id="ep1",
        step_id=0,
        goal_info="open chrome",
        string_label=raw_label,
        canonical_action_id=label_id,
        raw_action_type_int=4,
        touch_yx=(0.5, 0.5),
        lift_yx=(0.5, 0.5),
        typed_text=typed,
        image_bytes=None,
    )


def test_from_mind2web_preserves_canonical_action():
    cid = CANONICAL_ACTIONS["click"]
    out = _from_mind2web(_mind2web_step(cid))
    assert out.source == "mind2web"
    assert out.canonical_action_id == cid
    assert out.canonical_action_name == ID_TO_ACTION[cid]
    assert out.raw_label == "CLICK"
    assert out.touch_xy is None
    assert out.image is None


def test_from_aitw_flips_yx_to_xy():
    out = _from_aitw(_aitw_step(CANONICAL_ACTIONS["click"]), include_images=False)
    assert out.source == "aitw"
    # input was touch_yx=(0.5, 0.5), output should be touch_xy=(0.5, 0.5) (same here)
    assert out.touch_xy == (0.5, 0.5)
    # ensure typed_text propagates
    typed_step = _aitw_step(CANONICAL_ACTIONS["type"], raw_label="type", typed="hello")
    out2 = _from_aitw(typed_step, include_images=False)
    assert out2.typed_text == "hello"


def test_class_balanced_indices_yields_balanced_sample():
    # 100 click, 5 type, 0 scroll → request 20 per class for the two present
    steps = []
    for _ in range(100):
        out = _from_mind2web(_mind2web_step(CANONICAL_ACTIONS["click"]))
        steps.append(out)
    for _ in range(5):
        out = _from_mind2web(_mind2web_step(CANONICAL_ACTIONS["type"]))
        steps.append(out)
    idxs = class_balanced_indices(steps, n_per_class=20, seed=0)
    chosen_classes = [steps[i].canonical_action_id for i in idxs]
    click_n = chosen_classes.count(CANONICAL_ACTIONS["click"])
    type_n = chosen_classes.count(CANONICAL_ACTIONS["type"])
    assert click_n == 20
    assert type_n == 20  # bootstrapped from 5 with replacement
    assert len(idxs) == 40


def test_class_balanced_indices_is_deterministic():
    steps = [_from_mind2web(_mind2web_step(CANONICAL_ACTIONS["click"])) for _ in range(50)]
    steps += [_from_mind2web(_mind2web_step(CANONICAL_ACTIONS["type"])) for _ in range(10)]
    a = class_balanced_indices(steps, n_per_class=8, seed=42)
    b = class_balanced_indices(steps, n_per_class=8, seed=42)
    assert a == b
    c = class_balanced_indices(steps, n_per_class=8, seed=43)
    assert a != c
