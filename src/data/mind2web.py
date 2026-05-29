"""Mind2Web text-data loader.

The unauth-accessible HF release contains the train split only (1009 tasks,
~7775 steps). Test splits (Cross-Task / Cross-Website / Cross-Domain) are
gated; we hold out a task-level slice of train for validation instead so that
val tasks are entirely unseen during training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset

from src.data.taxonomy import unify_action


@dataclass(frozen=True)
class Mind2WebStep:
    """One (instruction, history, action_type) tuple suitable for Stage 1."""

    annotation_id: str           # task id (groups steps)
    domain: str
    website: str
    confirmed_task: str          # high-level user instruction
    step_index: int              # position within the task
    history: tuple[str, ...]     # action_reprs of all prior steps in this task
    raw_op: str                  # CLICK / TYPE / SELECT
    canonical_action_id: int     # in [0, 7]
    target_html: str             # cleaned_html of the target element


def _iter_steps(ds_split) -> Iterator[Mind2WebStep]:
    for ex in ds_split:
        history: list[str] = []
        for i, act in enumerate(ex["actions"]):
            raw_op = act["operation"]["op"]
            try:
                cid = unify_action(raw_op, "mind2web")
            except KeyError:
                # skip unmapped ops rather than crash a whole task
                history.append(ex["action_reprs"][i] if i < len(ex["action_reprs"]) else "")
                continue
            yield Mind2WebStep(
                annotation_id=ex["annotation_id"],
                domain=ex["domain"],
                website=ex["website"],
                confirmed_task=ex["confirmed_task"],
                step_index=i,
                history=tuple(history),
                raw_op=raw_op,
                canonical_action_id=cid,
                target_html=act["cleaned_html"][:1000],  # truncate; raw_html is huge
            )
            history.append(ex["action_reprs"][i] if i < len(ex["action_reprs"]) else "")


def load_mind2web_text(
    split: str = "train",
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Mind2WebStep], list[Mind2WebStep]]:
    """Load Mind2Web and return (train_steps, val_steps) with a TASK-LEVEL split.

    Task-level splitting matters: if we split at the step level, the same
    `confirmed_task` instruction appears in both train and val and the model
    can memorize the action sequence.
    """
    ds = load_dataset("osunlp/Mind2Web", split=split)
    task_ids = [ex["annotation_id"] for ex in ds]
    unique_tasks = sorted(set(task_ids))

    rng = random.Random(seed)
    rng.shuffle(unique_tasks)
    n_val = max(1, int(len(unique_tasks) * val_frac))
    val_tasks = set(unique_tasks[:n_val])

    train_steps: list[Mind2WebStep] = []
    val_steps: list[Mind2WebStep] = []
    for step in _iter_steps(ds):
        (val_steps if step.annotation_id in val_tasks else train_steps).append(step)

    return train_steps, val_steps


def step_to_instruction_text(step: Mind2WebStep, with_history: bool = True) -> str:
    """Canonical text representation used by the text-only baseline."""
    parts = [step.confirmed_task]
    if with_history and step.history:
        parts.append("History: " + " | ".join(step.history))
    parts.append(f"Next action target HTML: {step.target_html}")
    return "\n".join(parts)
