"""Unified Mind2Web + AITW dataset.

Both sources resolve to the same canonical 8-class action space (see
`src.data.taxonomy`). This module exposes a single iterator / dataset
that yields a consistent schema for downstream Stage 1 / Stage 2 code.

What's NOT in this module:
- Feature pre-computation. Stage 1 trains on features cached on the Modal
  Volume, not directly on these dataclasses. The unified dataset is a
  bookkeeping abstraction for raw items, not a torch.utils.data.Dataset of
  tensors.
- Image loading for AITW unless you ask for it (`include_images=True`).
  Mind2Web screenshots come from Multimodal-Mind2Web; AITW screenshots
  come from the raw-RGB byte path documented in `src.data.aitw`.

Schema (the `UnifiedStep` dataclass below):
- source:        "mind2web" or "aitw"
- task_id:       Mind2Web annotation_id / AITW ep_id
- step_index:    position within the task / episode (0-based)
- instruction:   human-language description of the user's goal
- canonical_action_id:  int in [0, 7]
- canonical_action_name: string label from CANONICAL_ACTIONS
- raw_label:     "CLICK" / "TYPE" / "SELECT" / "tap" / "swipe_up" / ...
- typed_text:    "" unless canonical action is "type"
- target_html:   Mind2Web cleaned_html (first 1000 chars), or "" for AITW
- touch_xy:      [x, y] for AITW DUAL_POINT actions in [0,1]; None for Mind2Web
- lift_xy:       [x, y] for AITW DUAL_POINT actions in [0,1]; None for Mind2Web
- image:         PIL.Image if include_images else None
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator

from PIL.Image import Image as PILImage

from src.data.aitw import AITWStep, iter_aitw_steps
from src.data.mind2web import Mind2WebStep, load_mind2web_text
from src.data.taxonomy import ID_TO_ACTION


@dataclass(frozen=True)
class UnifiedStep:
    source: str            # "mind2web" | "aitw"
    task_id: str
    step_index: int
    instruction: str
    canonical_action_id: int
    canonical_action_name: str
    raw_label: str
    typed_text: str
    target_html: str
    touch_xy: tuple[float, float] | None  # (x, y) normalized — AITW only
    lift_xy: tuple[float, float] | None
    image: PILImage | None


def _from_mind2web(step: Mind2WebStep) -> UnifiedStep:
    return UnifiedStep(
        source="mind2web",
        task_id=step.annotation_id,
        step_index=step.step_index,
        instruction=step.confirmed_task,
        canonical_action_id=step.canonical_action_id,
        canonical_action_name=ID_TO_ACTION[step.canonical_action_id],
        raw_label=step.raw_op,
        typed_text="",
        target_html=step.target_html,
        touch_xy=None,
        lift_xy=None,
        image=None,  # Mind2Web text-only loader has no screenshots
    )


def _from_aitw(step: AITWStep, include_images: bool) -> UnifiedStep:
    # AITW reports (y, x); flip to (x, y) for consistency with Mind2Web convention
    ty, tx = step.touch_yx
    ly, lx = step.lift_yx
    img = step.open_image() if include_images and step.image_bytes is not None else None
    return UnifiedStep(
        source="aitw",
        task_id=step.ep_id,
        step_index=step.step_id,
        instruction=step.goal_info,
        canonical_action_id=step.canonical_action_id,
        canonical_action_name=ID_TO_ACTION[step.canonical_action_id],
        raw_label=step.string_label,
        typed_text=step.typed_text,
        target_html="",
        touch_xy=(tx, ty),
        lift_xy=(lx, ly),
        image=img,
    )


def iter_unified(
    *,
    sources: tuple[str, ...] = ("mind2web", "aitw"),
    aitw_n_max: int = 0,
    aitw_split: str = "train",
    aitw_config: str = "standard",
    aitw_include_images: bool = False,
    m2w_val_frac: float = 0.15,
    m2w_seed: int = 42,
    m2w_split: str = "train",  # "train" or "val" (val = held-out tasks)
) -> Iterator[UnifiedStep]:
    """Yield UnifiedStep items from each requested source.

    Mind2Web steps come from the task-level held-out split of the unauth
    release. AITW steps are streamed from cjfcsjt/AITW_General.
    """
    if "mind2web" in sources:
        train, val = load_mind2web_text(val_frac=m2w_val_frac, seed=m2w_seed)
        chosen = train if m2w_split == "train" else val
        for s in chosen:
            yield _from_mind2web(s)

    if "aitw" in sources:
        for s in iter_aitw_steps(
            config=aitw_config, split=aitw_split, n_max=aitw_n_max,
            include_images=aitw_include_images,
        ):
            yield _from_aitw(s, include_images=aitw_include_images)


def load_unified(
    *,
    sources: tuple[str, ...] = ("mind2web", "aitw"),
    aitw_n_max: int = 0,
    aitw_split: str = "train",
    m2w_split: str = "train",
    **kwargs,
) -> list[UnifiedStep]:
    return list(iter_unified(
        sources=sources, aitw_n_max=aitw_n_max,
        aitw_split=aitw_split, m2w_split=m2w_split, **kwargs,
    ))


def class_balanced_indices(
    steps: Iterable[UnifiedStep],
    n_per_class: int,
    seed: int = 42,
) -> list[int]:
    """Return indices that sample `n_per_class` items per canonical action.

    Use as a sampler: random.sample with replacement when n_per_class exceeds
    the size of a class, without replacement otherwise. This is the simplest
    implementation of the "class-balanced sampler" the roadmap calls for and
    avoids the overhead of WeightedRandomSampler when training on cached
    features (we resample once and shuffle).
    """
    steps_list = list(steps)
    by_class: dict[int, list[int]] = {}
    for i, s in enumerate(steps_list):
        by_class.setdefault(s.canonical_action_id, []).append(i)

    rng = random.Random(seed)
    out: list[int] = []
    for cls_id, indices in by_class.items():
        if len(indices) >= n_per_class:
            out.extend(rng.sample(indices, n_per_class))
        else:
            # bootstrap: sample with replacement to reach the target
            out.extend(rng.choices(indices, k=n_per_class))
    rng.shuffle(out)
    return out
