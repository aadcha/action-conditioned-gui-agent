"""Canonical 8-class action taxonomy + per-dataset mappings.

The canonical space is dataset-agnostic. Most public web/Android/desktop
benchmarks only populate a subset of it — Mind2Web's web data, for instance,
only contains CLICK / SELECT / TYPE in the unauth-accessible release.

Each judgment call is documented inline. Teammates should be able to read this
file and reproduce every label that ends up in the unified dataset.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical action space
# ---------------------------------------------------------------------------

CANONICAL_ACTIONS: dict[str, int] = {
    "click": 0,
    "double_click": 1,
    "type": 2,
    "scroll": 3,
    "drag": 4,
    "hotkey": 5,
    "wait": 6,
    "finished": 7,
}
ID_TO_ACTION: dict[int, str] = {v: k for k, v in CANONICAL_ACTIONS.items()}
NUM_ACTIONS: int = len(CANONICAL_ACTIONS)


# ---------------------------------------------------------------------------
# Mind2Web — web data (CLICK / SELECT / TYPE only in the canonical release)
# ---------------------------------------------------------------------------
#
# Judgment calls:
#   SELECT -> click   : Mind2Web SELECT means "pick a dropdown option". From the
#                       user's perspective this is a discrete click on an
#                       option element. Mapping it to a separate class would
#                       create a near-singleton (4.2% of steps) that the
#                       classifier cannot learn well.
#   HOVER  -> click   : Not observed in the unauth release but kept here for
#                       robustness. HOVER targets the same affordance class
#                       (interactable element) as click.
#
MIND2WEB_TO_CANONICAL: dict[str, str] = {
    "CLICK": "click",
    "TYPE": "type",
    "SELECT": "click",
    "HOVER": "click",
}


# ---------------------------------------------------------------------------
# Android in the Wild — mobile data (richer non-click coverage)
# ---------------------------------------------------------------------------
#
# Judgment calls:
#   swipe_*    -> scroll  : All directional swipes collapse to one class. We
#                           drop direction; recovering it would require a
#                           directional argument head which is Phase 4 work.
#   press_back / press_home / press_enter -> hotkey
#                           : System-level navigation actions all go to the
#                           hotkey bucket. Mind2Web has no equivalent so this
#                           class will be AITW-only initially.
#
AITW_TO_CANONICAL: dict[str, str] = {
    "tap": "click",
    "double_tap": "double_click",
    "long_press": "click",      # judgment: closer to click than drag
    "swipe_up": "scroll",
    "swipe_down": "scroll",
    "swipe_left": "scroll",
    "swipe_right": "scroll",
    "type": "type",
    "press_back": "hotkey",
    "press_home": "hotkey",
    "press_enter": "hotkey",
    "wait": "wait",
    "status_task_complete": "finished",
    "status_task_impossible": "finished",
}


# ---------------------------------------------------------------------------
# AndroidControl — same source ecosystem as AITW, similar mapping
# ---------------------------------------------------------------------------

ANDROIDCONTROL_TO_CANONICAL: dict[str, str] = {
    "click": "click",
    "long_press": "click",
    "input_text": "type",
    "scroll": "scroll",
    "drag": "drag",
    "navigate_back": "hotkey",
    "navigate_home": "hotkey",
    "open_app": "hotkey",
    "wait": "wait",
    "complete": "finished",
}


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------

_DATASET_MAPPINGS: dict[str, dict[str, str]] = {
    "mind2web": MIND2WEB_TO_CANONICAL,
    "aitw": AITW_TO_CANONICAL,
    "androidcontrol": ANDROIDCONTROL_TO_CANONICAL,
}


def unify_action(source_label: str, dataset: str) -> int:
    """Map a dataset-specific action label to a canonical class id.

    Raises KeyError on unmapped labels so silent label loss shows up in tests.
    """
    if dataset not in _DATASET_MAPPINGS:
        raise KeyError(f"Unknown dataset: {dataset!r}. Known: {sorted(_DATASET_MAPPINGS)}")
    mapping = _DATASET_MAPPINGS[dataset]
    if source_label not in mapping:
        raise KeyError(
            f"Unmapped label {source_label!r} for dataset {dataset!r}. "
            f"Add it to {dataset.upper()}_TO_CANONICAL with a documented judgment."
        )
    canonical = mapping[source_label]
    return CANONICAL_ACTIONS[canonical]
