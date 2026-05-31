"""Android in the Wild (AITW) loader.

Mirrors `src/data/mind2web.py` in spirit: produce a flat list of typed steps
that can be canonicalized into our 8-class action space and fed to the same
Stage 1 / Stage 2 training paths as Mind2Web.

Source: https://huggingface.co/datasets/cjfcsjt/AITW_General  (parquet mirror of
google-research/android_in_the_wild). The "standard" config exposes train/test
parquet shards with one row per step.

Schema fields we use:
- ep_id:               str  — episode id (groups steps from the same task)
- step_id:             int  — position within the episode
- goal_info:           str  — natural-language task ("Open a new Chrome private window")
- image:               bytes — encoded screenshot
- results_action_type: int  — AITW enum (see AITW_ACTION_TYPE_ENUM below)
- results_yx_touch:    [y, x] in [0,1] — start position of a DUAL_POINT
- results_yx_lift:     [y, x] in [0,1] — end position of a DUAL_POINT
- results_type_action: list[str] — typed text for TYPE actions
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Iterable, Iterator

from datasets import load_dataset
from PIL import Image

from src.data.taxonomy import unify_action

# AITW int enum from google-research/android_in_the_wild
AITW_ACTION_TYPE_ENUM: dict[int, str] = {
    3: "TYPE",
    4: "DUAL_POINT",         # taps and swipes
    5: "PRESS_BACK",
    6: "PRESS_HOME",
    7: "PRESS_ENTER",
    10: "STATUS_TASK_COMPLETE",
    11: "STATUS_TASK_IMPOSSIBLE",
}

# Conversion from the raw enum name to the string keys our taxonomy uses.
# DUAL_POINT requires a touch/lift comparison to split into tap vs swipe_*.
_NON_DUAL_POINT_TO_STRING: dict[str, str] = {
    "TYPE": "type",
    "PRESS_BACK": "press_back",
    "PRESS_HOME": "press_home",
    "PRESS_ENTER": "press_enter",
    "STATUS_TASK_COMPLETE": "status_task_complete",
    "STATUS_TASK_IMPOSSIBLE": "status_task_impossible",
}

# Threshold for treating a DUAL_POINT as a tap (vs a swipe).
# Distance is measured in normalized image space (Euclidean over [0,1]^2).
DEFAULT_TAP_THRESHOLD = 0.04


# AITW HF mirror (cjfcsjt/AITW_General) stores screenshots as raw RGB pixel
# bytes with no header, packed as W*H*3 uint8s. Empirically the General config
# uses 540x1080 (half-resolution Pixel 3 portrait): 540*1080*3 = 1,749,600 bytes.
# If a future row deviates, _decode_aitw_image_bytes inspects byte length and
# tries the standard candidates.
_AITW_KNOWN_SHAPES: tuple[tuple[int, int], ...] = (
    (540, 1080),   # 1,749,600 bytes — the General config
    (1080, 540),   # 1,749,600 bytes — same product, in case orientation flipped
    (1080, 2160),  # 6,998,400 bytes — full Pixel 3 portrait
    (480, 800),    # 1,152,000 bytes — alternate device
)


def _decode_aitw_image_bytes(buf: bytes) -> Image.Image:
    """Decode AITW raw-RGB image bytes into a PIL.Image.

    Tries the known shapes first; falls back to attempting `Image.open` in case
    a future variant of the dataset stores encoded images instead.
    """
    n = len(buf)
    for w, h in _AITW_KNOWN_SHAPES:
        if w * h * 3 == n:
            return Image.frombytes("RGB", (w, h), buf)
    # Fallback: maybe it's a real encoded image after all.
    return Image.open(io.BytesIO(buf)).convert("RGB")


@dataclass(frozen=True)
class AITWStep:
    ep_id: str
    step_id: int
    goal_info: str
    string_label: str          # tap / swipe_up / type / press_back / ...
    canonical_action_id: int   # in [0, 7]
    raw_action_type_int: int   # the AITW enum value
    touch_yx: tuple[float, float]
    lift_yx: tuple[float, float]
    typed_text: str            # "" unless string_label == "type"
    image_bytes: bytes | None  # raw bytes; decode lazily via .open_image()

    def open_image(self) -> Image.Image:
        if self.image_bytes is None:
            raise ValueError("AITWStep has no image bytes (load with include_images=True)")
        return _decode_aitw_image_bytes(self.image_bytes)


def classify_dual_point(
    touch_yx: tuple[float, float],
    lift_yx: tuple[float, float],
    tap_threshold: float = DEFAULT_TAP_THRESHOLD,
) -> str:
    """Decide tap / swipe_{up,down,left,right} from touch and lift positions.

    Coordinates are normalized to [0, 1]. y grows downward (image convention).
    """
    ty, tx = touch_yx
    ly, lx = lift_yx
    dy = ly - ty
    dx = lx - tx
    if math.hypot(dx, dy) < tap_threshold:
        return "tap"
    if abs(dy) > abs(dx):
        return "swipe_up" if dy < 0 else "swipe_down"
    return "swipe_left" if dx < 0 else "swipe_right"


def row_to_string_label(row: dict) -> str | None:
    at = row["results_action_type"]
    name = AITW_ACTION_TYPE_ENUM.get(at)
    if name is None:
        return None
    if name == "DUAL_POINT":
        return classify_dual_point(row["results_yx_touch"], row["results_yx_lift"])
    return _NON_DUAL_POINT_TO_STRING.get(name)


def _row_to_step(row: dict, include_images: bool) -> AITWStep | None:
    label = row_to_string_label(row)
    if label is None:
        return None
    try:
        cid = unify_action(label, "aitw")
    except KeyError:
        return None
    typed = ""
    rta = row.get("results_type_action") or []
    if rta and isinstance(rta, list) and isinstance(rta[0], str):
        typed = rta[0]
    return AITWStep(
        ep_id=str(row["ep_id"]),
        step_id=int(row["step_id"]),
        goal_info=row.get("goal_info", "") or "",
        string_label=label,
        canonical_action_id=cid,
        raw_action_type_int=int(row["results_action_type"]),
        touch_yx=tuple(row["results_yx_touch"]),  # type: ignore[arg-type]
        lift_yx=tuple(row["results_yx_lift"]),    # type: ignore[arg-type]
        typed_text=typed,
        image_bytes=row.get("image") if include_images else None,
    )


def iter_aitw_steps(
    *,
    config: str = "standard",
    split: str = "train",
    repo: str = "cjfcsjt/AITW_General",
    n_max: int = 0,             # 0 = all
    include_images: bool = False,
    streaming: bool = True,
) -> Iterator[AITWStep]:
    """Stream AITW rows, convert each into an AITWStep, skip unmappable rows."""
    ds = load_dataset(repo, name=config, split=split, streaming=streaming)
    n_yielded = 0
    for row in ds:
        step = _row_to_step(row, include_images=include_images)
        if step is None:
            continue
        yield step
        n_yielded += 1
        if n_max and n_yielded >= n_max:
            return


def load_aitw_steps(
    *,
    config: str = "standard",
    split: str = "train",
    repo: str = "cjfcsjt/AITW_General",
    n_max: int = 0,
    include_images: bool = False,
) -> list[AITWStep]:
    """Materialize a list of AITWStep — convenient when you'll iterate multiple times."""
    return list(
        iter_aitw_steps(
            config=config,
            split=split,
            repo=repo,
            n_max=n_max,
            include_images=include_images,
        )
    )


def episode_level_split(
    steps: Iterable[AITWStep],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[AITWStep], list[AITWStep]]:
    """Episode-level held-out split — every step from the same episode falls into
    the same partition so the model can't see steps from a task it was trained on.
    """
    steps = list(steps)
    ep_ids = sorted({s.ep_id for s in steps})
    rng = random.Random(seed)
    rng.shuffle(ep_ids)
    n_val = max(1, int(len(ep_ids) * val_frac))
    val_eps = set(ep_ids[:n_val])
    train, val = [], []
    for s in steps:
        (val if s.ep_id in val_eps else train).append(s)
    return train, val


def step_to_instruction_text(step: AITWStep) -> str:
    """Canonical text representation parallel to `src.data.mind2web.step_to_instruction_text`."""
    parts = [f"Goal: {step.goal_info}"]
    if step.typed_text:
        parts.append(f"Typed text: {step.typed_text}")
    parts.append("Action type:")
    return "\n".join(parts)
