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
        "scikit-learn>=1.5",
        "tqdm>=4.66",
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


# ---- Milestone 3: zero-shot Qwen2-VL-2B action-type prediction --------------
#
# Hits Qwen2-VL-2B with a chat prompt asking it to predict the action type for
# each Mind2Web step (text-only inputs: instruction + history + target HTML;
# screenshots come later via the Multimodal-Mind2Web variant). Produces the
# "frontier model with no fine-tuning" baseline number to compare against the
# TF-IDF baselines and (later) our fine-tuned Stage 1 classifier.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=3600,
)
def _run_zero_shot_m3(
    model_id: str,
    n_val_steps: int,
    batch_size: int,
    seed: int,
) -> dict:
    """Inner: zero-shot action-type prediction on a slice of Mind2Web val."""
    import os
    import sys
    import json
    from collections import Counter

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    from src.data.mind2web import load_mind2web_text
    from src.data.taxonomy import ID_TO_ACTION

    print(f"[m3-zs] loading {model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    hf_cache.commit()

    print("[m3-zs] loading Mind2Web val...")
    _, val_steps = load_mind2web_text()
    if n_val_steps > 0:
        val_steps = val_steps[:n_val_steps]
    print(f"[m3-zs] running on {len(val_steps)} val steps")

    label_words = {"CLICK": "click", "TYPE": "type", "SELECT": "click"}

    def build_prompt(step) -> str:
        hist = " | ".join(step.history[-3:]) if step.history else "(none)"
        return (
            "You are a GUI agent. Given a user task and the target HTML element, "
            "predict the next action type. Reply with exactly one of: CLICK, TYPE, SELECT.\n"
            f"User task: {step.confirmed_task}\n"
            f"Recent history: {hist}\n"
            f"Target HTML: {step.target_html}\n"
            "Action type:"
        )

    y_true: list[int] = []
    y_pred: list[int] = []
    raw_outputs: list[str] = []

    for i in tqdm(range(0, len(val_steps), batch_size), desc="batches"):
        batch = val_steps[i : i + batch_size]
        messages_batch = [
            [{"role": "user", "content": [{"type": "text", "text": build_prompt(s)}]}]
            for s in batch
        ]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        inputs = processor(text=texts, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, max_new_tokens=8, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        outputs = processor.batch_decode(trimmed, skip_special_tokens=True)

        for step, raw in zip(batch, outputs, strict=True):
            raw_outputs.append(raw)
            raw_upper = raw.strip().upper()
            # Heuristic parse: first matching keyword wins.
            predicted_raw = "CLICK"  # default to majority class on parse failure
            for w in ("SELECT", "CLICK", "TYPE"):
                if w in raw_upper:
                    predicted_raw = w
                    break
            from src.data.taxonomy import unify_action
            try:
                y_pred.append(unify_action(predicted_raw, "mind2web"))
            except KeyError:
                y_pred.append(unify_action("CLICK", "mind2web"))
            y_true.append(step.canonical_action_id)

    labels = sorted(set(y_true))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    parse_counts = dict(Counter(o.strip().upper().split()[0] if o.strip() else "" for o in raw_outputs))

    return {
        "model_id": model_id,
        "n_eval_steps": len(val_steps),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [ID_TO_ACTION[c] for c in labels],
        "first_token_distribution": parse_counts,
        "sample_outputs": raw_outputs[:20],
    }


@app.local_entrypoint()
def zero_shot_m3(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    n_val_steps: int = 0,  # 0 = full val (~1199 steps)
    batch_size: int = 4,
    seed: int = 42,
) -> None:
    """Zero-shot Qwen2-VL-2B action-type baseline on Mind2Web val.

    Writes results to results/milestone3/zero_shot_qwen2vl.json on the local box
    so the milestone doc can pick them up.
    """
    import json
    from pathlib import Path

    result = _run_zero_shot_m3.remote(model_id, n_val_steps, batch_size, seed)
    out = Path("results/milestone3/zero_shot_qwen2vl.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[m3-zs] wrote {out}")
    print(f"[m3-zs] accuracy={result['accuracy']:.3f}  macro_f1={result['macro_f1']:.3f}")


# ---- Milestone 3: vision-ablation on Multimodal-Mind2Web --------------------
#
# Same Qwen2-VL-2B, same prompt, same eval set — once with screenshots, once
# without. Difference in macro-F1 = causal contribution of the vision encoder
# to action-type prediction. This is the project's risk-#1 measurement.

hf_secret = modal.Secret.from_name("huggingface")


@app.function(
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=[hf_secret],
    timeout=600,
)
def _probe_multimodal_schema(split: str = "test_task") -> dict:
    """Inspect the schema of Multimodal-Mind2Web to find the right field names."""
    import os
    import json

    os.environ["HF_HOME"] = HF_CACHE_PATH
    from datasets import load_dataset

    ds = load_dataset(
        "osunlp/Multimodal-Mind2Web",
        split=split,
        token=os.environ.get("HF_TOKEN"),
        streaming=True,
    )
    # streaming -> grab the first few rows without downloading whole split
    rows = []
    for i, row in enumerate(ds):
        if i >= 3:
            break
        # serialize: keep keys + type info + sample values for non-image fields
        summary = {}
        for k, v in row.items():
            t = type(v).__name__
            if t == "bytes":
                summary[k] = f"<bytes len={len(v)}>"
            elif hasattr(v, "size") and hasattr(v, "mode"):  # PIL Image
                summary[k] = f"<PIL.Image {v.mode} {v.size}>"
            elif isinstance(v, (dict, list)):
                summary[k] = {"type": t, "sample": str(v)[:200]}
            else:
                summary[k] = {"type": t, "value": str(v)[:200]}
        rows.append(summary)
    return {"rows": rows}


@app.local_entrypoint()
def probe_schema(split: str = "test_task") -> None:
    """One-shot schema probe — useful when adapting to a new dataset version."""
    import json
    out = _probe_multimodal_schema.remote(split)
    print(json.dumps(out, indent=2))


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=[hf_secret],
    timeout=7200,
)
def _run_vision_ablation_m3(
    model_id: str,
    split: str,
    n_steps: int,
    batch_size: int,
    max_image_pixels: int,
    seed: int,
    option_order: str = "CLICK,TYPE,SELECT",
) -> dict:
    """Vision-ablation: run Qwen2-VL-2B in two modes on the same Multimodal-Mind2Web slice.

    Mode A — text_only:  instruction + history + target HTML, no image.
    Mode B — vision_text: instruction + history + screenshot.
    """
    import os
    import sys
    import io
    import random
    from collections import Counter

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from datasets import load_dataset
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    from src.data.taxonomy import unify_action, ID_TO_ACTION

    print(f"[m3-vis] loading {model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    hf_cache.commit()

    print(f"[m3-vis] loading osunlp/Multimodal-Mind2Web split={split}...")
    ds = load_dataset(
        "osunlp/Multimodal-Mind2Web",
        split=split,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"[m3-vis] full split size: {len(ds)} steps")
    print(f"[m3-vis] schema keys: {list(ds[0].keys())}")

    # Sample for cost control
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if n_steps > 0:
        indices = indices[:n_steps]
    print(f"[m3-vis] evaluating on {len(indices)} sampled steps")

    # Build canonical examples. Schema (per probe):
    #   operation is a JSON-encoded string like '{"original_op": "TYPE", "value": "...", "op": "TYPE"}'
    #   screenshot is already a PIL.Image (full-page captures can be up to ~4000px tall)
    import json as _json
    examples: list[dict] = []
    skipped_unmapped = 0
    skipped_no_image = 0
    for idx in indices:
        ex = ds[idx]
        op_raw = ex.get("operation")
        if isinstance(op_raw, str):
            try:
                op = _json.loads(op_raw).get("op")
            except Exception:
                op = None
        elif isinstance(op_raw, dict):
            op = op_raw.get("op")
        else:
            op = None
        if not op:
            continue
        try:
            cid = unify_action(op, "mind2web")
        except KeyError:
            skipped_unmapped += 1
            continue
        screenshot = ex.get("screenshot")
        if screenshot is None:
            skipped_no_image += 1
            continue
        img = screenshot if hasattr(screenshot, "convert") else Image.open(io.BytesIO(screenshot)).convert("RGB")
        img = img.convert("RGB")
        if max_image_pixels > 0 and img.width * img.height > max_image_pixels:
            r = (max_image_pixels / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
        examples.append({
            "task": ex.get("confirmed_task", ""),
            "target_html": (ex.get("cleaned_html") or ex.get("raw_html") or "")[:1000],
            "raw_op": op,
            "canonical": cid,
            "image": img,
        })
    print(f"[m3-vis] kept {len(examples)} (skipped {skipped_unmapped} unmapped, {skipped_no_image} no-image)")

    option_list = [w.strip().upper() for w in option_order.split(",") if w.strip()]
    options_str = ", ".join(option_list)
    SHARED_PROMPT_TMPL = (
        "You are a GUI agent. Given a user task and the target HTML element, "
        f"predict the next action type. Reply with exactly one of: {options_str}.\n"
        "User task: {task}\n"
        "Target HTML: {html}\n"
        "Action type:"
    )

    def parse_pred(raw: str) -> int:
        u = raw.strip().upper()
        for w in ("SELECT", "CLICK", "TYPE"):
            if w in u:
                return unify_action(w, "mind2web")
        return unify_action("CLICK", "mind2web")

    def run_mode(use_image: bool) -> dict:
        mode = "vision_text" if use_image else "text_only"
        y_true: list[int] = []
        y_pred: list[int] = []
        raw_first_tokens: list[str] = []
        sample_outputs: list[str] = []
        for i in tqdm(range(0, len(examples), batch_size), desc=f"{mode} batches"):
            batch = examples[i : i + batch_size]
            messages_batch = []
            for ex in batch:
                prompt = SHARED_PROMPT_TMPL.format(task=ex["task"], html=ex["target_html"])
                content = []
                if use_image:
                    content.append({"type": "image", "image": ex["image"]})
                content.append({"type": "text", "text": prompt})
                messages_batch.append([{"role": "user", "content": content}])
            texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_batch
            ]
            if use_image:
                image_inputs, video_inputs = process_vision_info(messages_batch)
            else:
                image_inputs, video_inputs = None, None
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
            ]
            outputs = processor.batch_decode(trimmed, skip_special_tokens=True)
            for ex, raw in zip(batch, outputs, strict=True):
                if len(sample_outputs) < 20:
                    sample_outputs.append(raw)
                y_true.append(ex["canonical"])
                y_pred.append(parse_pred(raw))
                raw_first_tokens.append(raw.strip().upper().split()[0] if raw.strip() else "")

        labels = sorted(set(y_true))
        return {
            "mode": mode,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "confusion_matrix_labels": [ID_TO_ACTION[c] for c in labels],
            "first_token_distribution": dict(Counter(raw_first_tokens)),
            "sample_outputs": sample_outputs,
        }

    print("[m3-vis] running text_only mode...")
    text_only = run_mode(use_image=False)
    print("[m3-vis] running vision_text mode...")
    vision_text = run_mode(use_image=True)

    return {
        "model_id": model_id,
        "dataset": "osunlp/Multimodal-Mind2Web",
        "split": split,
        "n_eval_steps": len(examples),
        "skipped_unmapped": skipped_unmapped,
        "skipped_no_image": skipped_no_image,
        "max_image_pixels": max_image_pixels,
        "seed": seed,
        "option_order": option_list,
        "results": {
            "text_only": text_only,
            "vision_text": vision_text,
        },
        "vision_delta": {
            "macro_f1": vision_text["macro_f1"] - text_only["macro_f1"],
            "accuracy": vision_text["accuracy"] - text_only["accuracy"],
        },
    }


@app.local_entrypoint()
def vision_ablation_m3(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    split: str = "test_task",
    n_steps: int = 500,
    batch_size: int = 2,
    max_image_pixels: int = 1_000_000,  # ~1MP cap on screenshot resolution
    seed: int = 42,
    option_order: str = "CLICK,TYPE,SELECT",
    out_name: str = "vision_ablation_qwen2vl",
) -> None:
    """Vision-ablation: zero-shot Qwen2-VL-2B with vs without screenshots on Multimodal-Mind2Web.

    Use `--option-order` to control the order the labels are listed in the prompt
    (e.g. `--option-order SELECT,TYPE,CLICK` for the position-bias control).
    """
    import json
    from pathlib import Path

    result = _run_vision_ablation_m3.remote(
        model_id, split, n_steps, batch_size, max_image_pixels, seed, option_order
    )
    out = Path(f"results/milestone3/{out_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[m3-vis] wrote {out}")
    txt = result["results"]["text_only"]
    vis = result["results"]["vision_text"]
    print(f"[m3-vis] option_order: {option_order}")
    print(f"[m3-vis] text_only:    macro_f1={txt['macro_f1']:.3f}  acc={txt['accuracy']:.3f}")
    print(f"[m3-vis] vision_text:  macro_f1={vis['macro_f1']:.3f}  acc={vis['accuracy']:.3f}")
    d = result["vision_delta"]
    print(f"[m3-vis] DELTA:        macro_f1={d['macro_f1']:+.3f}  acc={d['accuracy']:+.3f}")
