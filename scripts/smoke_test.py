"""Phase 1 smoke test: load Qwen2-VL-7B + LoRA, run one image+prompt, print output.

Self-contained — if no `--image` is given, generates a synthetic UI screenshot so
this script verifies cluster + model setup without needing any dataset on disk.

Success criteria:
- `model.print_trainable_parameters()` reports ~0.1% trainable (LoRA adapters only).
- A decoded response prints to stdout.
- Exit code 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.base import load_qwen2vl_with_lora  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def synthesize_ui_screenshot(size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Render a placeholder UI with a clearly-visible Submit button."""
    img = Image.new("RGB", size, color=(245, 245, 250))
    draw = ImageDraw.Draw(img)

    draw.rectangle((40, 40, 472, 100), outline=(180, 180, 200), width=2)
    draw.text((56, 60), "Email:", fill=(60, 60, 80))

    draw.rectangle((40, 130, 472, 190), outline=(180, 180, 200), width=2)
    draw.text((56, 150), "Password:", fill=(60, 60, 80))

    btn_box = (180, 380, 332, 432)
    draw.rectangle(btn_box, fill=(70, 130, 200), outline=(50, 100, 170), width=2)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((228, 396), "Submit", fill=(255, 255, 255), font=font)

    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2-VL + LoRA smoke test")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "smoke_test.yaml",
        help="YAML config (see configs/smoke_test.yaml)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional screenshot path. Synthesizes one if omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["seed"])

    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        print(f"[smoke] using image: {args.image}")
    else:
        image = synthesize_ui_screenshot()
        print("[smoke] using synthesized 512x512 UI screenshot")

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    print(f"[smoke] loading {model_cfg['id']} (dtype={model_cfg['dtype']})...")

    model, processor = load_qwen2vl_with_lora(
        model_id=model_cfg["id"],
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=tuple(lora_cfg["target_modules"]),
        dtype=DTYPE_MAP[model_cfg["dtype"]],
        device_map=model_cfg["device_map"],
    )
    model.print_trainable_parameters()

    gen_cfg = cfg["generation"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": gen_cfg["prompt"]},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("[smoke] generating...")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=gen_cfg["max_new_tokens"])
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    print("=" * 60)
    print(f"PROMPT:\n  {gen_cfg['prompt']}")
    print(f"RESPONSE:\n  {output}")
    print("=" * 60)
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
