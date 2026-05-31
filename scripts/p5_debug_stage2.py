"""Diagnostic — verify Variant D's mechanism is actually wired up correctly.

Checks, in order:
  1. Does `<|action_slot|>` survive the chat template as exactly one token?
  2. After build_batch, is the slot token present at exactly one position
     in every row's input_ids?
  3. Does `_embed_with_action` actually overwrite that position's embedding?
  4. Does changing `action_type_id` (with everything else fixed) change the
     model's output logits / generated text?
  5. Does the loss backward populate `action_embeddings.weight.grad`?
  6. Does the gradient direction differ per action_type_id?

If any of these fail, the training was learning around a broken pipeline.

Designed to run as a standalone Modal job (loads Qwen2-VL-2B). ~$0.05.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.taxonomy import CANONICAL_ACTIONS, ID_TO_ACTION
from src.models.stage2_grounding import (
    ACTION_SLOT_TOKEN,
    Stage2Config,
    Stage2ConditionedGrounding,
    make_stage2_prompt,
)
from src.train.stage2 import Stage2Example, build_batch


def make_synthetic_pil_image(w: int = 512, h: int = 512):
    """Phone-screenshot-ish synthetic image, sized so Qwen2-VL's vision encoder is happy.
    256x256 triggers a split_with_sizes assertion in Qwen2-VL; 512x512 works.
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), color=(200, 200, 220))
    d = ImageDraw.Draw(img)
    d.rectangle((100, 200, 400, 280), fill=(70, 130, 200))
    d.text((200, 230), "Submit", fill=(255, 255, 255))
    d.rectangle((100, 100, 400, 150), outline=(80, 80, 80), width=2)
    d.text((110, 115), "Email:", fill=(60, 60, 80))
    return img


def main() -> dict:
    print("[debug] loading Stage2 model on whatever device is available...")
    cfg = Stage2Config(device_map="auto" if torch.cuda.is_available() else "cpu")
    model = Stage2ConditionedGrounding(cfg)
    model.eval()
    proc = model.processor

    results: dict = {}

    # ---- 1. Chat template preserves slot token ---------------------------
    msg = [
        {"role": "user", "content": [
            {"type": "image", "image": make_synthetic_pil_image()},
            {"type": "text", "text": make_stage2_prompt("open chrome")},
        ]}
    ]
    rendered = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    slot_in_text = ACTION_SLOT_TOKEN in rendered
    n_slot_in_text = rendered.count(ACTION_SLOT_TOKEN)
    print(f"[debug-1] rendered prompt contains slot token: {slot_in_text} (count={n_slot_in_text})")
    print(f"[debug-1] rendered (first 300 chars):\n  {rendered[:300]!r}")
    results["1_slot_in_rendered_text"] = {"present": slot_in_text, "count": n_slot_in_text}
    assert slot_in_text and n_slot_in_text == 1, "chat template lost the slot!"

    # ---- 2. After build_batch, exactly one slot token per row ------------
    examples = [
        Stage2Example(image=make_synthetic_pil_image(), goal_info="open chrome",
                      action_type_id=CANONICAL_ACTIONS["click"], target_xy=(0.5, 0.4)),
        Stage2Example(image=make_synthetic_pil_image(), goal_info="scroll up",
                      action_type_id=CANONICAL_ACTIONS["scroll"], target_xy=(0.3, 0.7)),
    ]
    device = next(model.parameters()).device
    batch = build_batch(model, examples, device, include_labels=True)

    slot_id = model.action_slot_token_id
    slot_counts = (batch["input_ids"] == slot_id).sum(dim=1).cpu().tolist()
    print(f"[debug-2] slot token id = {slot_id}; per-row counts in batch = {slot_counts}")
    results["2_slot_in_input_ids"] = {"slot_token_id": slot_id, "per_row_counts": slot_counts}
    assert all(c == 1 for c in slot_counts), "build_batch didn't preserve exactly one slot per row!"

    # ---- 3. _embed_with_action actually overwrites the slot embedding ----
    with torch.inference_mode():
        embed = model.vlm.get_input_embeddings()
        normal_embeds = embed(batch["input_ids"])
        rewritten = model._embed_with_action(batch["input_ids"], batch["action_type_id"])
    diff = (rewritten - normal_embeds).abs()
    # The diff is zero everywhere EXCEPT at slot positions
    nonzero_positions = diff.sum(dim=-1) > 1e-6
    print(f"[debug-3] positions where rewritten != normal embed (per row):")
    for b in range(diff.shape[0]):
        idx = nonzero_positions[b].nonzero(as_tuple=True)[0].cpu().tolist()
        slot_idx = (batch["input_ids"][b] == slot_id).nonzero(as_tuple=True)[0].cpu().tolist()
        print(f"  row {b}: changed at {idx}  (slot at {slot_idx})")
        assert idx == slot_idx, f"row {b}: rewrite touched non-slot positions {set(idx) - set(slot_idx)}"
    results["3_embed_replacement_localized"] = True

    # Also check: the value at the slot position equals action_embeddings[action_type_id]
    for b in range(2):
        slot_pos = slot_counts.index(1)  # always 1 — find the slot position
        slot_pos = (batch["input_ids"][b] == slot_id).nonzero(as_tuple=True)[0].item()
        rewritten_val = rewritten[b, slot_pos]
        expected_val = model.action_embeddings(
            batch["action_type_id"][b:b+1]
        )[0].to(rewritten_val.dtype)
        assert torch.allclose(rewritten_val.float(), expected_val.float(), atol=1e-3), \
            f"row {b}: slot value != action_embeddings[{batch['action_type_id'][b].item()}]"
    results["3b_slot_value_equals_action_embedding"] = True
    print("[debug-3b] slot position value == action_embeddings[action_type_id]: OK")

    # ---- 4. Changing action_type_id changes the OUTPUT (text-only path) --
    # Build a TEXT-ONLY batch (no image) so we can compare logits without
    # hitting Qwen2-VL's vision encoder. This validates the slot replacement
    # influences the LM's logits — which is the entire point of variant D.
    text_only_msg = [
        [{"role": "user", "content": [
            {"type": "text", "text": make_stage2_prompt("open chrome")},
        ]}]
    ]
    rendered_txt = proc.apply_chat_template(text_only_msg[0], tokenize=False, add_generation_prompt=True)
    txt_inputs = proc(text=[rendered_txt], return_tensors="pt", padding=True).to(device)
    # Try with click vs scroll
    a_click = torch.tensor([CANONICAL_ACTIONS["click"]], device=device)
    a_scroll = torch.tensor([CANONICAL_ACTIONS["scroll"]], device=device)
    with torch.inference_mode():
        o1 = model(action_type_id=a_click, input_ids=txt_inputs["input_ids"],
                   attention_mask=txt_inputs["attention_mask"], return_dict=True)
        o2 = model(action_type_id=a_scroll, input_ids=txt_inputs["input_ids"],
                   attention_mask=txt_inputs["attention_mask"], return_dict=True)
    last_logits_1 = o1.logits[0, -1].float()
    last_logits_2 = o2.logits[0, -1].float()
    delta = (last_logits_1 - last_logits_2).abs()
    max_delta = float(delta.max().item())
    mean_delta = float(delta.mean().item())
    cosine = float(torch.nn.functional.cosine_similarity(
        last_logits_1.unsqueeze(0), last_logits_2.unsqueeze(0)
    ).item())
    top1_changed = int(last_logits_1.argmax().item() != last_logits_2.argmax().item())
    top5_1 = torch.topk(last_logits_1, 5).indices.cpu().tolist()
    top5_2 = torch.topk(last_logits_2, 5).indices.cpu().tolist()
    print(f"[debug-4] (text-only) click vs scroll on same prompt:")
    print(f"  last-token logits |delta| max={max_delta:.5f}  mean={mean_delta:.7f}  cosine={cosine:.6f}")
    print(f"  top-1 token id differs: {bool(top1_changed)}")
    print(f"  top-5 token ids: click={top5_1}, scroll={top5_2}")
    results["4_action_changes_logits"] = {
        "max_abs_delta": max_delta,
        "mean_abs_delta": mean_delta,
        "cosine_similarity": cosine,
        "top1_token_differs_at_random_init": bool(top1_changed),
        "top5_click": top5_1, "top5_scroll": top5_2,
    }

    # ---- 5. Backward pass populates action_embeddings.weight.grad --------
    # Use a text-only labeled batch so we don't trip Qwen2-VL's vision attn quirk.
    text_train_input = proc(text=[rendered_txt + "(500, 400)<|im_end|>"],
                            return_tensors="pt", padding=True).to(device)
    labels = text_train_input["input_ids"].clone()
    # Mask everything except the last 10 tokens (rough "answer span" proxy)
    if labels.shape[1] > 10:
        labels[:, :-10] = -100
    model.train()
    out = model(
        action_type_id=a_click,
        input_ids=text_train_input["input_ids"],
        attention_mask=text_train_input["attention_mask"],
        labels=labels,
        return_dict=True,
    )
    loss = out.loss
    print(f"[debug-5] forward loss = {loss.item():.4f}")
    loss.backward()
    ae_grad = model.action_embeddings.weight.grad
    if ae_grad is None:
        results["5_grad_populated"] = False
        print("[debug-5] action_embeddings.weight.grad is None — BUG")
    else:
        per_row_norms = ae_grad.norm(dim=-1).cpu().tolist()
        print(f"[debug-5] action_embeddings.weight.grad norm per class id:")
        for i, n in enumerate(per_row_norms):
            print(f"  {i} ({ID_TO_ACTION[i]:>14}): {n:.6f}")
        results["5_grad_populated"] = True
        results["5_grad_norms_per_class"] = {ID_TO_ACTION[i]: per_row_norms[i] for i in range(len(per_row_norms))}

    # ---- 6. Gradient is localized to the actually-used class -------------
    # We only used a_click above, so only the "click" row should receive grad.
    if ae_grad is not None:
        per_row_norms_t = ae_grad.norm(dim=-1)
        click_id = CANONICAL_ACTIONS["click"]
        active_norms = per_row_norms_t[click_id].item()
        inactive_max = per_row_norms_t[
            torch.tensor([i for i in range(per_row_norms_t.shape[0]) if i != click_id], device=ae_grad.device)
        ].max().item()
        print(f"[debug-6] click-row grad norm = {active_norms:.6f}  (must be > 0)")
        print(f"[debug-6] max non-click grad norm = {inactive_max:.6f}  (must be ~0)")
        results["6_grad_localized_to_active_classes"] = {
            "click_grad_norm": active_norms,
            "max_inactive_grad_norm": inactive_max,
            "ok": active_norms > 1e-8 and inactive_max < 1e-8,
        }

    # ---- 6b. Label masking sanity ----------------------------------------
    # Verify build_batch's label-masking actually masks user-turn tokens.
    print("\n[debug-6b] label masking sanity check")
    txt_input_with_assistant = proc(
        text=[rendered_txt + "(500, 400)<|im_end|>"],
        return_tensors="pt", padding=True,
    ).to(device)
    # Run the SAME masking logic used in build_batch
    tokenizer = proc.tokenizer
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    print(f"  <|im_start|> token id: {im_start_id}")
    print(f"  'assistant' token id (raw): {assistant_id}")
    row = txt_input_with_assistant["input_ids"][0]
    starts = (row == im_start_id).nonzero(as_tuple=True)[0].tolist()
    print(f"  <|im_start|> positions: {starts}")
    cut = 0
    for idx in reversed(starts):
        if idx + 1 < row.shape[0]:
            next_tok = row[idx + 1].item()
            print(f"  at position {idx}: next token id = {next_tok} (assistant? {next_tok == assistant_id})")
            if next_tok == assistant_id:
                cut = idx + 2
                if cut < row.shape[0]:
                    cut += 1
                break
    print(f"  computed cut: {cut}")
    if cut == 0:
        print(f"  WARNING: no cut -> labels include entire sequence (user turn would be trained on)")
    # Decode the un-masked portion (what loss actually fires on)
    if cut > 0:
        print(f"  masked tokens [:cut]:    {tokenizer.decode(row[:cut])!r}")
        print(f"  trained tokens [cut:]:   {tokenizer.decode(row[cut:])!r}")
    else:
        print(f"  full sequence (no mask): {tokenizer.decode(row)!r}")
    results["6b_label_masking"] = {
        "im_start_id": int(im_start_id),
        "assistant_id_via_convert": int(assistant_id) if assistant_id is not None else None,
        "im_start_positions": starts,
        "computed_cut": int(cut),
        "trained_tokens_text": tokenizer.decode(row[cut:]) if cut > 0 else tokenizer.decode(row),
    }

    # ---- 8. SUSPECT BUG: Qwen2-VL needs input_ids to merge vision features
    # at <|image_pad|> positions. D's forward currently passes inputs_embeds
    # but drops input_ids. Test: compare D's logits (zero action embedding)
    # against A's pathway (input_ids, no replacement) on the SAME image+text.
    # If outputs differ off the slot position, image fusion is being mishandled.
    print("\n[debug-8] image-fusion-pathway comparison (D vs A code paths)")
    # Build a real image+text batch
    img_examples = [
        Stage2Example(image=make_synthetic_pil_image(), goal_info="open chrome",
                      action_type_id=CANONICAL_ACTIONS["click"], target_xy=(0.5, 0.4)),
    ]
    img_batch = build_batch(model, img_examples, device, include_labels=False)
    # Zero out the action embedding so the only diff is the code path
    with torch.no_grad():
        saved_ae = model.action_embeddings.weight.data.clone()
        model.action_embeddings.weight.data.zero_()

    # D's pathway (inputs_embeds, NO input_ids)
    inputs_embeds_d = model._embed_with_action(img_batch["input_ids"], img_batch["action_type_id"])
    with torch.inference_mode():
        out_d = model.vlm(
            inputs_embeds=inputs_embeds_d,
            attention_mask=img_batch["attention_mask"],
            pixel_values=img_batch.get("pixel_values"),
            image_grid_thw=img_batch.get("image_grid_thw"),
            return_dict=True,
        )

    # A's pathway (input_ids, no slot replacement). For fairness, use the same
    # batch (slot token still in input_ids, but the model uses the table's
    # current row for it, which we just zeroed — so the slot is also a zero
    # vector in both paths).
    with torch.inference_mode():
        out_a = model.vlm(
            input_ids=img_batch["input_ids"],
            attention_mask=img_batch["attention_mask"],
            pixel_values=img_batch.get("pixel_values"),
            image_grid_thw=img_batch.get("image_grid_thw"),
            return_dict=True,
        )

    # Restore action embeddings
    model.action_embeddings.weight.data.copy_(saved_ae)

    # Compare logits position-by-position
    logits_d = out_d.logits.float()
    logits_a = out_a.logits.float()
    print(f"  D logits shape: {tuple(logits_d.shape)}")
    print(f"  A logits shape: {tuple(logits_a.shape)}")
    if logits_d.shape == logits_a.shape:
        per_pos_max = (logits_d - logits_a).abs().max(dim=-1).values  # [B, T]
        for b in range(logits_d.shape[0]):
            # Find slot position for this row
            slot_pos = (img_batch["input_ids"][b] == model.action_slot_token_id).nonzero(as_tuple=True)[0]
            slot_pos = slot_pos.item() if len(slot_pos) else -1
            non_slot_max = per_pos_max[b].clone()
            if slot_pos >= 0:
                non_slot_max[slot_pos] = 0.0
            print(f"  row {b}: max |D-A| at slot position {slot_pos}: {per_pos_max[b, slot_pos].item() if slot_pos>=0 else 'n/a':.5f}")
            print(f"  row {b}: max |D-A| at any OTHER position: {non_slot_max.max().item():.5f}")
            results["8_image_fusion_check"] = {
                "logits_shape_d": list(logits_d.shape),
                "logits_shape_a": list(logits_a.shape),
                "slot_position": int(slot_pos),
                "max_abs_delta_at_slot": float(per_pos_max[b, slot_pos].item() if slot_pos >= 0 else float("nan")),
                "max_abs_delta_off_slot": float(non_slot_max.max().item()),
            }
    else:
        print(f"  WARNING: logits shapes differ — image fusion clearly different")
        results["8_image_fusion_check"] = {
            "logits_shape_d": list(logits_d.shape),
            "logits_shape_a": list(logits_a.shape),
            "shapes_differ": True,
        }

    # ---- 7. Generation shape check ---------------------------------------
    # When using inputs_embeds via Stage2ConditionedGrounding.generate,
    # does the returned tensor include the prompt tokens or only new ones?
    # If it's the full sequence, evaluate_grounding's `batch_decode(generated, ...)`
    # is leaking prompt content into the parser.
    print("\n[debug-7] generation shape check (text-only):")
    model.eval()
    txt_inputs = proc(text=[rendered_txt], return_tensors="pt", padding=True).to(device)
    a_click = torch.tensor([CANONICAL_ACTIONS["click"]], device=device)
    prompt_len = txt_inputs["input_ids"].shape[1]
    with torch.inference_mode():
        gen = model.generate(
            action_type_id=a_click,
            input_ids=txt_inputs["input_ids"],
            attention_mask=txt_inputs["attention_mask"],
            max_new_tokens=12, do_sample=False,
            pad_token_id=proc.tokenizer.eos_token_id,
        )
    print(f"  prompt input_ids length: {prompt_len}")
    print(f"  generated.shape: {tuple(gen.shape)}")
    print(f"  returned >= prompt: {gen.shape[1] >= prompt_len}")
    decoded_full = proc.batch_decode(gen, skip_special_tokens=True)[0]
    print(f"  decoded full: {decoded_full!r}")
    if gen.shape[1] >= prompt_len:
        decoded_trimmed = proc.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0]
        print(f"  decoded trimmed-by-prompt-len: {decoded_trimmed!r}")
        results["7_generation"] = {
            "prompt_input_ids_len": int(prompt_len),
            "generated_shape": list(gen.shape),
            "decoded_full": decoded_full,
            "decoded_trimmed": decoded_trimmed,
        }
    else:
        results["7_generation"] = {
            "prompt_input_ids_len": int(prompt_len),
            "generated_shape": list(gen.shape),
            "decoded_full": decoded_full,
            "decoded_trimmed": "(generated shorter than prompt — not the full-sequence return mode)",
        }

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Stage 2 / Variant D — mechanism verification")
    print("=" * 70)
    out = main()
    print()
    print("=" * 70)
    print(json.dumps(out, indent=2, default=str))
    out_path = Path("results/phase4/stage2_mechanism_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")
