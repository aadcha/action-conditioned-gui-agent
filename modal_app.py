"""Modal entry points for cloud runs.

Why Modal: serverless GPU billing — pay only for container-active seconds.
Best for spiky workloads (smoke tests, debugging, short ablation slices).
GCP spot A100 is the bulk-training plan; see COMPUTE.md.

Setup:
    uv sync --extra modal
    modal token new            # one-time auth

Run the smoke test on a Modal L4 (~$0.80/hr, ~3 min cold start + ~30 sec run):
    modal run modal_app.py::smoke

Override the config or pin a different GPU:
    modal run modal_app.py::smoke --config configs/smoke_test_7b.yaml --gpu A100-40GB
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent

# ---- Image -----------------------------------------------------------------
# Mirrors pyproject.toml deps. Kept explicit (not `uv sync` inside the image)
# so Modal's layer caching works well and cold-start stays predictable.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "torchvision>=0.19",
        "transformers>=4.45",
        "peft>=0.13",
        "accelerate>=0.34",
        "datasets>=2.21",
        "pillow>=10",
        "qwen-vl-utils>=0.0.8",
        "pyyaml>=6.0",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/repo",
        ignore=[
            "**/.venv",
            "**/__pycache__",
            "**/.git",
            "**/data/raw/*",
            "**/data/processed/*",
            "**/wandb",
            "**/runs",
            "**/outputs",
            "**/.pytest_cache",
            "**/.ruff_cache",
        ],
    )
)

# Cache HF model weights across runs so we don't re-download Qwen2-VL every cold start.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

app = modal.App("action-conditioned-gui-agent")


# ---- Functions -------------------------------------------------------------


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=900,
)
def _run_smoke(config_rel_path: str) -> str:
    """Inner function — runs on the GPU container. Returns the model's decoded output."""
    import os
    import sys

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    # Re-implement scripts/smoke_test.py main() inline so we get the output as a return
    # value rather than just stdout. Keeps the local script the single source of truth
    # for everything else (synthetic image, dtype map, etc).
    import torch
    import yaml
    from qwen_vl_utils import process_vision_info

    from scripts.smoke_test import DTYPE_MAP, synthesize_ui_screenshot
    from src.models.base import load_qwen2vl_with_lora
    from src.utils.seed import set_seed

    cfg = yaml.safe_load(Path(f"/root/repo/{config_rel_path}").read_text())
    set_seed(cfg["seed"])

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    gen_cfg = cfg["generation"]
    print(f"[modal-smoke] loading {model_cfg['id']} (dtype={model_cfg['dtype']})...")

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
    hf_cache.commit()  # persist any newly-downloaded weights for next run

    image_pil = synthesize_ui_screenshot()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": gen_cfg["prompt"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("[modal-smoke] generating...")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=gen_cfg["max_new_tokens"])
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


@app.local_entrypoint()
def smoke(
    config: str = "configs/smoke_test.yaml",
    gpu: str = "L4",
) -> None:
    """Run the smoke test on Modal. Use `--gpu A100-40GB` to switch hardware.

    Note: changing `gpu` here only affects the *next* deployment of `_run_smoke` if
    you redeploy. For ad-hoc GPU choice, edit the decorator above and rerun.
    """
    if gpu != "L4":
        print(
            f"[modal-smoke] note: requested --gpu {gpu} but the deployed function "
            "is pinned to L4. Edit @app.function(gpu=...) in modal_app.py to change."
        )
    output = _run_smoke.remote(config)
    print("=" * 60)
    print(f"RESPONSE:\n  {output}")
    print("=" * 60)
    print("[modal-smoke] OK")
