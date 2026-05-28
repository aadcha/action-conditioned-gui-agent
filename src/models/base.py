"""Qwen2-VL loader with LoRA applied.

The roadmap calls for a single loader function — no abstract base classes.

Default base is Qwen2-VL-2B-Instruct (chosen for compute budget: every ablation
runs on 2B; 7B is a stretch run on the winning variant). Pass `model_id` to
override (see configs/smoke_test_7b.yaml for the 7B path).
"""

from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def load_qwen2vl_with_lora(
    model_id: str = DEFAULT_MODEL_ID,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
) -> tuple[PeftModel, AutoProcessor]:
    """Load Qwen2-VL-7B-Instruct with a LoRA adapter applied to attention projections.

    On a single A100 80GB, default `device_map="auto"` works. If multi-GPU split
    misbehaves with LoRA, pin to one GPU via `device_map={"": 0}`.
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, processor
