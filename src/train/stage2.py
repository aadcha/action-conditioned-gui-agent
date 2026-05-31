"""Stage 2 — supervised grounding training on AITW DUAL_POINT actions.

Trains the action-type-conditioned VLM (`src.models.stage2_grounding`) to
output coordinate strings for the next tap. Loss is standard LM cross-entropy
over the answer tokens (input + image tokens are masked out of the loss).

Coordinate format: `(x, y)` where x, y are integers in [0, 999]. We translate
AITW's normalized [0,1] touch coordinates by multiplying by 1000 and
rounding. This keeps the output a fixed-length numeric string in plain ASCII,
which Qwen2-VL's tokenizer handles cleanly without special tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from qwen_vl_utils import process_vision_info

from src.models.stage2_grounding import (
    ACTION_SLOT_TOKEN,
    Stage2ConditionedGrounding,
    make_stage2_prompt,
)


def coord_to_string(xy: tuple[float, float], scale: int = 1000) -> str:
    """Normalized (x, y) in [0, 1] → '(int_x, int_y)' with int in [0, scale-1]."""
    x = max(0, min(scale - 1, int(round(xy[0] * scale))))
    y = max(0, min(scale - 1, int(round(xy[1] * scale))))
    return f"({x}, {y})"


def string_to_coord(s: str, scale: int = 1000) -> tuple[float, float] | None:
    """Inverse of coord_to_string. Returns None if the string can't be parsed."""
    import re

    m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if m is None:
        return None
    x = int(m.group(1)) / scale
    y = int(m.group(2)) / scale
    return (x, y)


@dataclass
class Stage2Example:
    """Pure-Python container for one training/eval example."""

    image: object  # PIL.Image
    goal_info: str
    action_type_id: int
    target_xy: tuple[float, float]  # normalized [0, 1]


def build_batch(
    model: Stage2ConditionedGrounding,
    examples: list[Stage2Example],
    device: torch.device,
    coord_scale: int = 1000,
    include_labels: bool = True,
):
    """Tokenize a batch of Stage2Examples into Qwen2-VL inputs.

    Output dict keys:
      input_ids, attention_mask, pixel_values, image_grid_thw, action_type_id
      (and `labels` if include_labels=True)
    """
    processor = model.processor

    # Build messages per example
    messages_batch = []
    answers = []
    for ex in examples:
        prompt = make_stage2_prompt(ex.goal_info)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ex.image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if include_labels:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}],
            })
        messages_batch.append(messages)
        answers.append(coord_to_string(ex.target_xy, coord_scale))

    # Apply chat template. Two modes:
    #   - include_labels=True (training): the assistant turn is in messages_batch
    #     and the chat template renders the full conversation; add_generation_prompt
    #     would add a stray empty assistant header on top of the existing turn.
    #   - include_labels=False (eval/inference): no assistant turn in the messages,
    #     so we MUST add_generation_prompt=True or the model has no `<|im_start|>
    #     assistant\n` cue and tries to invent the chat structure itself
    #     (this manifested as Variant D's generated outputs starting with
    #     "user\n(x, y)" — the model predicting a new user turn instead of
    #     emitting an assistant answer).
    texts = [
        processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=(not include_labels)
        )
        for m in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    action_type_id = torch.tensor(
        [ex.action_type_id for ex in examples], dtype=torch.long, device=device
    )

    out = dict(inputs)
    out["action_type_id"] = action_type_id

    if include_labels:
        # Mask everything before the answer span with -100 so loss only fires on
        # the predicted coordinate tokens. We locate the answer by finding the
        # last occurrence of the assistant-turn marker; the answer starts there.
        labels = out["input_ids"].clone()
        # Mark image tokens with -100 (no loss on those positions either).
        # We approximate "loss on the answer only" by training on the assistant
        # turn alone. The chat template wraps assistant content with
        # <|im_start|>assistant\n ... <|im_end|>; everything before the assistant
        # marker is the prompt and should be masked.
        im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")

        # For each row, find the last <|im_start|>assistant marker. Mask
        # everything up to and including that marker pair.
        for b in range(labels.shape[0]):
            row = out["input_ids"][b]
            # find indices where row == im_start
            starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
            cut = 0
            for idx in reversed(starts):
                # the token right after im_start may or may not be 'assistant'
                if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                    cut = idx + 2  # mask up through the role token
                    # also mask the newline after the role marker if present
                    if cut < row.shape[0]:
                        cut += 1
                    break
            labels[b, :cut] = -100
        out["labels"] = labels

    return out


def train_one_epoch(
    model: Stage2ConditionedGrounding,
    batches: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 10,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(batches):
        # Some Qwen2-VL processor outputs include extra keys; pass via **batch.
        out = model(**batch)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n += 1
        if log_every and (i + 1) % log_every == 0:
            print(f"[stage2-train] step {i + 1}  loss={loss.item():.4f}")
    return total_loss / max(n, 1)


def _metrics_from_predictions(
    targets: list[tuple[float, float]],
    preds: list[tuple[float, float] | None],
    action_type_ids: list[int] | None = None,
) -> dict:
    """Compute hit@r + per-class breakdowns over already-collected predictions."""
    import math
    from collections import defaultdict
    from src.data.taxonomy import ID_TO_ACTION

    parsed_pairs = [(p, t, (action_type_ids[i] if action_type_ids else None))
                    for i, (p, t) in enumerate(zip(preds, targets)) if p is not None]
    n_total = len(targets)
    n_parsed = len(parsed_pairs)
    if n_parsed == 0:
        return {
            "n_total": n_total, "n_parsed": 0,
            "parse_rate": 0.0,
            "mean_normalized_l2": float("nan"),
            "hit_at_005": 0.0, "hit_at_010": 0.0, "hit_at_025": 0.0,
            "per_class": {},
        }

    dists = [math.hypot(p[0] - t[0], p[1] - t[1]) for p, t, _ in parsed_pairs]
    hit = lambda r: sum(1 for d in dists if d <= r) / n_parsed

    per_class: dict[str, dict] = {}
    if action_type_ids is not None:
        by_class: dict[int, list[float]] = defaultdict(list)
        for d, (_, _, cid) in zip(dists, parsed_pairs):
            if cid is not None:
                by_class[cid].append(d)
        for cid, dlist in sorted(by_class.items()):
            per_class[ID_TO_ACTION[cid]] = {
                "n": len(dlist),
                "mean_normalized_l2": float(sum(dlist) / len(dlist)),
                "hit_at_005": float(sum(1 for d in dlist if d <= 0.05) / len(dlist)),
                "hit_at_010": float(sum(1 for d in dlist if d <= 0.10) / len(dlist)),
                "hit_at_025": float(sum(1 for d in dlist if d <= 0.25) / len(dlist)),
            }

    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / n_total,
        "mean_normalized_l2": float(sum(dists) / n_parsed),
        "hit_at_005": float(hit(0.05)),
        "hit_at_010": float(hit(0.10)),
        "hit_at_025": float(hit(0.25)),
        "per_class": per_class,
    }


def evaluate_grounding(
    model: Stage2ConditionedGrounding,
    examples: list[Stage2Example],
    device: torch.device,
    coord_scale: int = 1000,
    max_new_tokens: int = 16,
    batch_size: int = 1,
    image_resolution_hint: tuple[int, int] | None = None,
) -> dict:
    """Generate coords for each example, compute hit@r overall and per canonical class."""
    model.eval()
    preds: list[tuple[float, float] | None] = []
    targets: list[tuple[float, float]] = []
    action_ids: list[int] = []
    raw_outputs: list[str] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        inputs = build_batch(model, batch, device, coord_scale=coord_scale, include_labels=False)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                action_type_id=inputs["action_type_id"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model.processor.tokenizer.eos_token_id,
            )
        # When the underlying generate() is given input_ids alongside our
        # inputs_embeds, it returns the full sequence (prompt prepended).
        # When it's given only inputs_embeds, it returns just new tokens.
        # Detect at runtime and trim if needed so the parser only sees the
        # actual answer span.
        if generated.shape[1] > prompt_len:
            new_tokens = generated[:, prompt_len:]
        else:
            new_tokens = generated
        decoded = model.processor.batch_decode(new_tokens, skip_special_tokens=True)
        for ex, raw in zip(batch, decoded):
            raw_outputs.append(raw)
            parsed = string_to_coord(raw, scale=coord_scale)
            preds.append(parsed)
            targets.append(ex.target_xy)
            action_ids.append(ex.action_type_id)

    metrics = _metrics_from_predictions(targets, preds, action_ids)
    metrics["raw_outputs"] = raw_outputs[:20]
    return metrics
