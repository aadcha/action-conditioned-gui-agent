"""Stage 2 — action-type-conditioned grounding model.

The project's central architectural intervention: take Qwen2-VL-2B, attach a
learned embedding table `E ∈ ℝ^(num_action_types × d)`, and route the predicted
action type's row into the input sequence as a discrete conditioning signal
before LoRA fine-tuning the model to produce coordinates.

Concretely:
  1. Tokenizer is extended with one new sentinel token, `<|action_slot|>`.
  2. Stage 2 prompts always include this sentinel — once — at the start of the
     instruction text (after the image placeholder).
  3. At forward time, the embedding at the sentinel's position is REPLACED by
     `action_embeddings(action_type_id)`. The rest of the model sees a
     standard embedding sequence, so all of Qwen2-VL's image-token / cross-
     attention / generation machinery works unmodified.
  4. LoRA on (q,k,v,o) projections + the action embedding table are the only
     trainable parameters; the base VLM weights stay frozen.

Why marker-replacement instead of `inputs_embeds`-concat:
  Qwen2-VL fuses image features into the sequence via image placeholder
  tokens that the processor inserts in `input_ids`. Bypassing input_ids
  entirely (i.e. supplying `inputs_embeds` directly) would require us to
  re-implement that fusion ourselves. By keeping `input_ids` intact and only
  swapping a single embedding row in `inputs_embeds`, we get Qwen2-VL's image
  pipeline for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.data.taxonomy import NUM_ACTIONS

ACTION_SLOT_TOKEN = "<|action_slot|>"
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass
class Stage2Config:
    model_id: str = DEFAULT_MODEL_ID
    num_action_types: int = NUM_ACTIONS
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    dtype: torch.dtype = torch.bfloat16
    device_map: str | dict = "auto"
    # Diagnostic flag for Phase 5: when False, the action embedding table
    # stays at random init throughout training (requires_grad=False). The
    # model still sees the slot at the same position; only the per-class
    # specialization is removed. Comparing trainable=True (variant D) vs
    # trainable=False isolates "slot disturbs the prompt" from "embedding
    # not learning fast enough".
    action_embeddings_trainable: bool = True


class Stage2ConditionedGrounding(nn.Module):
    """Qwen2-VL-2B + LoRA + learned action-type embedding table.

    Inputs to `forward` and `generate` are the standard Qwen2-VL kwargs
    (input_ids, attention_mask, pixel_values, image_grid_thw, ...) plus a
    new required keyword: `action_type_id` of shape [B].
    """

    def __init__(self, cfg: Stage2Config | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = Stage2Config()
        self.cfg = cfg

        # --- 1. Load base model + processor ---
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(cfg.model_id)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            device_map=cfg.device_map,
        )

        # --- 2. Extend tokenizer with the action-slot sentinel ---
        # Use add_special_tokens so it doesn't get split or stripped during decoding.
        num_added = self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [ACTION_SLOT_TOKEN]}
        )
        if num_added:
            base.resize_token_embeddings(len(self.processor.tokenizer))
        self.action_slot_token_id: int = self.processor.tokenizer.convert_tokens_to_ids(
            ACTION_SLOT_TOKEN
        )
        assert isinstance(self.action_slot_token_id, int) and self.action_slot_token_id >= 0

        # --- 3. Attach LoRA ---
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.target_modules),
            lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.vlm: PeftModel = get_peft_model(base, lora_config)

        # --- 4. Action-type embedding table ---
        hidden_dim = self._infer_hidden_dim()
        self.action_embeddings = nn.Embedding(cfg.num_action_types, hidden_dim)
        nn.init.normal_(self.action_embeddings.weight, mean=0.0, std=0.02)
        # Match the LM's device/dtype — device_map="auto" puts the VLM on
        # cuda:0 (or wherever), and action_embeddings must live alongside it
        # since their outputs slot directly into inputs_embeds.
        first_vlm_param = next(self.vlm.parameters())
        self.action_embeddings.to(device=first_vlm_param.device, dtype=cfg.dtype)
        if not cfg.action_embeddings_trainable:
            # Diagnostic mode: embedding stays at random init throughout training.
            for p in self.action_embeddings.parameters():
                p.requires_grad_(False)

    # ---- helpers --------------------------------------------------------

    def _infer_hidden_dim(self) -> int:
        """Hidden size of the LM stack (used for the action embedding table)."""
        cfg = self.vlm.base_model.model.config
        for path in ("text_config.hidden_size", "hidden_size"):
            cur = cfg
            ok = True
            for part in path.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok:
                return int(cur)
        # Last resort: probe the embedding layer
        return int(self.vlm.get_input_embeddings().embedding_dim)

    def _embed_with_action(
        self,
        input_ids: torch.Tensor,
        action_type_id: torch.Tensor,
    ) -> torch.Tensor:
        """Look up input embeddings and overwrite the action-slot position with
        the learned action embedding for each sample.
        """
        embed_layer = self.vlm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B, T, H]

        # Locate the action slot per row. There should be exactly one per sample.
        is_slot = input_ids == self.action_slot_token_id  # [B, T]
        # We assume one slot per row (asserted below); if absent, this is a bug.
        slot_count = is_slot.sum(dim=1)
        if (slot_count != 1).any():
            bad_rows = (slot_count != 1).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"Expected exactly one action-slot token per row; rows {bad_rows} "
                f"have {slot_count[bad_rows].tolist()}. Check the prompt template."
            )

        slot_positions = is_slot.nonzero(as_tuple=False)  # [B, 2] (batch_idx, time_idx)
        # Sort by batch_idx so we can index in row order
        slot_positions = slot_positions[slot_positions[:, 0].argsort()]
        batch_idx = slot_positions[:, 0]
        time_idx = slot_positions[:, 1]

        # Look up the action embedding for each row, in the order of batch_idx
        action_embeds = self.action_embeddings(action_type_id[batch_idx]).to(inputs_embeds.dtype)

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_idx, time_idx] = action_embeds
        return inputs_embeds

    # ---- public API -----------------------------------------------------

    def forward(
        self,
        action_type_id: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **vlm_kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids required (we look up & rewrite embeddings)")
        inputs_embeds = self._embed_with_action(input_ids, action_type_id)
        return self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **vlm_kwargs,
        )

    def generate(
        self,
        action_type_id: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **vlm_kwargs,
    ):
        inputs_embeds = self._embed_with_action(input_ids, action_type_id)
        return self.vlm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **vlm_kwargs,
        )

    def print_trainable_parameters(self) -> None:
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.parameters())
        action_n = self.action_embeddings.weight.numel()
        print(
            f"Stage 2 trainable params: {n_trainable:,} || all: {n_all:,} || "
            f"trainable%: {100 * n_trainable / n_all:.4f} (incl. {action_n:,} action embedding params)"
        )


def make_stage2_prompt(instruction: str) -> str:
    """Build the text portion of the Stage 2 prompt.

    The action slot always sits immediately after the user instruction marker
    so that its embedding precedes the natural-language task description.
    """
    return f"{ACTION_SLOT_TOKEN} {instruction.strip()}\nPredict the action coordinate."
