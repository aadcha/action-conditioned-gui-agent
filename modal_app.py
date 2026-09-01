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
        "matplotlib>=3.9",
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

# Cache Qwen2-VL pooled features + small artifacts across runs. The MLP itself
# is tiny so we can retrain it many times once features are cached.
stage1_cache = modal.Volume.from_name("stage1-cache", create_if_missing=True)
STAGE1_CACHE_PATH = "/root/cache/stage1"

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


# ---- Phase 2: Stage 1 real classifier on cached Qwen2-VL features -----------
#
# Step 1: Extract pooled features from a frozen Qwen2-VL-2B forward pass over
#         Multimodal-Mind2Web (one cache per (split, mode) pair).
# Step 2: Train the MLP head (src/models/stage1_classifier.py +
#         src/train/stage1.py) on cached features. Cheap and re-runnable.
#
# This is the real Stage 1 the project proposes — replaces the TF-IDF baseline
# headline from Milestone 3.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=21600,
)
def _extract_features(
    model_id: str,
    split: str,
    mode: str,                # "text_only" | "vision_text"
    n_steps: int,             # 0 = full split
    batch_size: int,
    max_image_pixels: int,
    seed: int,
    force_recompute: bool,
) -> dict:
    """Forward-pass Qwen2-VL-2B over a Multimodal-Mind2Web split, save pooled
    features to the stage1-cache Modal Volume.

    The pooling is mean-of-last-hidden-state across all tokens (visual + text
    fused after the LM's cross-attention). Single vector per example.
    """
    import os
    import sys
    import json as _json
    import random
    from collections import Counter
    from pathlib import Path

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from datasets import load_dataset
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from src.data.taxonomy import unify_action

    assert mode in ("text_only", "vision_text"), f"bad mode: {mode}"

    cache_dir = Path(STAGE1_CACHE_PATH)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"features_{split}_{mode}_n{n_steps}_seed{seed}.pt"
    meta_file = cache_dir / f"features_{split}_{mode}_n{n_steps}_seed{seed}.meta.json"

    if cache_file.exists() and not force_recompute:
        print(f"[extract] cached features exist at {cache_file}; skipping (use force_recompute=True to re-run)")
        meta = _json.loads(meta_file.read_text()) if meta_file.exists() else {}
        meta["cache_file"] = str(cache_file)
        meta["from_cache"] = True
        return meta

    print(f"[extract] loading {model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    hf_cache.commit()

    # In transformers v5 Qwen2VLConfig nests hidden_size under text_config. Probe
    # both locations; fall back to a dynamic read after the first forward pass.
    hidden_size = None
    for attr_path in ("text_config.hidden_size", "hidden_size"):
        cur = model.config
        try:
            for part in attr_path.split("."):
                cur = getattr(cur, part)
            hidden_size = int(cur)
            break
        except AttributeError:
            continue
    print(f"[extract] hidden_size (from config) = {hidden_size}")

    print(f"[extract] loading osunlp/Multimodal-Mind2Web split={split}...")
    ds = load_dataset(
        "osunlp/Multimodal-Mind2Web", split=split, token=os.environ.get("HF_TOKEN")
    )
    print(f"[extract] full split size: {len(ds)} steps")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if n_steps > 0:
        indices = indices[:n_steps]
    print(f"[extract] will process {len(indices)} steps in mode={mode}")

    # Pre-build examples (parse operation, decode screenshot, etc.)
    PROMPT_TMPL = (
        "Predict the next action type for the user task and target HTML element.\n"
        "User task: {task}\n"
        "Target HTML: {html}\n"
        "Action type:"
    )

    examples = []
    skipped_unmapped = skipped_no_image = 0
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
        img = screenshot.convert("RGB") if hasattr(screenshot, "convert") else screenshot
        if max_image_pixels > 0 and img.width * img.height > max_image_pixels:
            r = (max_image_pixels / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
        text_prompt = PROMPT_TMPL.format(
            task=ex.get("confirmed_task", "") or "",
            html=(ex.get("cleaned_html") or ex.get("raw_html") or "")[:1000],
        )
        examples.append({"text": text_prompt, "image": img, "label": cid})

    print(f"[extract] kept {len(examples)} (skipped {skipped_unmapped} unmapped, {skipped_no_image} no-image)")

    # Forward pass + pool. Process in batches.
    all_features: list[torch.Tensor] = []
    all_labels: list[int] = []
    pbar = tqdm(range(0, len(examples), batch_size), desc=f"extract({mode})")
    for start in pbar:
        batch = examples[start : start + batch_size]
        messages_batch = []
        for ex in batch:
            content = []
            if mode == "vision_text":
                content.append({"type": "image", "image": ex["image"]})
            content.append({"type": "text", "text": ex["text"]})
            messages_batch.append([{"role": "user", "content": content}])
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        image_inputs, video_inputs = (
            process_vision_info(messages_batch) if mode == "vision_text" else (None, None)
        )
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1]  # [B, T, H]
            if hidden_size is None:
                hidden_size = int(last_hidden.shape[-1])
                print(f"[extract] hidden_size (from forward) = {hidden_size}")
            # Masked mean-pool over real tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            pooled = pooled.to(torch.float32).cpu()

        all_features.append(pooled)
        all_labels.extend([ex["label"] for ex in batch])

    features = torch.cat(all_features, dim=0)  # [N, H]
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"[extract] features shape: {features.shape}  labels shape: {labels.shape}")
    print(f"[extract] feature norm mean/std: {features.norm(dim=1).mean():.3f} / {features.norm(dim=1).std():.3f}")

    torch.save({"features": features, "labels": labels, "hidden_size": hidden_size}, cache_file)
    label_dist = {int(k): int(v) for k, v in Counter(all_labels).items()}
    meta = {
        "model_id": model_id,
        "split": split,
        "mode": mode,
        "n_steps_requested": n_steps,
        "n_steps_used": len(examples),
        "skipped_unmapped": skipped_unmapped,
        "skipped_no_image": skipped_no_image,
        "seed": seed,
        "batch_size": batch_size,
        "max_image_pixels": max_image_pixels,
        "hidden_size": hidden_size,
        "features_shape": list(features.shape),
        "label_distribution_by_id": label_dist,
        "feature_norm_mean": float(features.norm(dim=1).mean()),
        "feature_norm_std": float(features.norm(dim=1).std()),
        "cache_file": str(cache_file),
        "from_cache": False,
    }
    meta_file.write_text(_json.dumps(meta, indent=2))
    stage1_cache.commit()
    return meta


@app.local_entrypoint()
def extract_features(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    split: str = "train",
    mode: str = "vision_text",
    n_steps: int = 0,
    batch_size: int = 2,
    max_image_pixels: int = 1_000_000,
    seed: int = 42,
    force_recompute: bool = False,
) -> None:
    """One (split, mode) pair → cached features on the stage1-cache Volume.

    Typical usage to get everything we need for Phase 2:

        modal run modal_app.py::extract_features --split train --mode vision_text
        modal run modal_app.py::extract_features --split train --mode text_only
        modal run modal_app.py::extract_features --split test_task --mode vision_text
        modal run modal_app.py::extract_features --split test_task --mode text_only
    """
    import json
    from pathlib import Path

    meta = _extract_features.remote(
        model_id, split, mode, n_steps, batch_size, max_image_pixels, seed, force_recompute
    )
    out = Path(f"results/phase2/feature_meta_{split}_{mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    print(f"[extract] meta saved to {out}")
    print(f"[extract] features cached at: {meta.get('cache_file')}")
    print(f"[extract] n_steps_used = {meta.get('n_steps_used')}, label dist by id = {meta.get('label_distribution_by_id')}")


@app.function(
    image=image,
    gpu="L4",  # Tiny MLP — could be CPU. Keeping L4 for parity with the extract image; only ~$0.03 per run.
    volumes={STAGE1_CACHE_PATH: stage1_cache},
    timeout=1800,
)
def _train_stage1_remote(
    train_split: str,
    val_split: str,
    n_train: int,
    n_val: int,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    class_weighted: bool,
    batch_size: int,
    train_seed: int,
) -> dict:
    """Load cached features, train three MLP variants (text_only, vision_text, vision_zeroed).

    The vision_zeroed variant is the Phase 2.4 day-1 sanity check: take the
    vision+text features and replace them with zeros so the MLP cannot use any
    of the information the VLM extracted from the screenshot. If macro-F1 holds
    up vs vision_text, vision contributes nothing and the project framing is in
    trouble.
    """
    import os
    import sys
    import json as _json
    from pathlib import Path

    sys.path.insert(0, "/root/repo")
    import torch

    from src.models.stage1_classifier import Stage1Config
    from src.train.stage1 import TrainConfig, train_stage1

    cache_dir = Path(STAGE1_CACHE_PATH)

    def _load(split: str, mode: str, n: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        path = cache_dir / f"features_{split}_{mode}_n{n}_seed{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Feature cache missing: {path}")
        blob = torch.load(path, map_location="cpu")
        return blob["features"], blob["labels"], {"hidden_size": int(blob["hidden_size"]), "path": str(path)}

    print("[train] loading cached features...")
    feats_train_text, labels_train, meta_text_train = _load(train_split, "text_only", n_train)
    feats_train_vis, labels_train_vis, meta_vis_train = _load(train_split, "vision_text", n_train)
    feats_val_text, labels_val, meta_text_val = _load(val_split, "text_only", n_val)
    feats_val_vis, labels_val_vis, _meta_vis_val = _load(val_split, "vision_text", n_val)

    assert labels_train.equal(labels_train_vis), "train labels diverge between modes"
    assert labels_val.equal(labels_val_vis), "val labels diverge between modes"
    hidden = meta_text_train["hidden_size"]
    assert hidden == meta_vis_train["hidden_size"], "hidden dim mismatch between modes"

    print(f"[train] train n = {feats_train_text.shape[0]}, val n = {feats_val_text.shape[0]}")
    print(f"[train] hidden_size = {hidden}")

    def _run(variant: str, ft: torch.Tensor, fv: torch.Tensor) -> dict:
        print(f"\n[train] === variant: {variant} ===")
        return train_stage1(
            features_train=ft,
            labels_train=labels_train,
            features_val=fv,
            labels_val=labels_val,
            model_cfg=Stage1Config(
                feature_dim=hidden,
                hidden_dim=hidden_dim,
                num_classes=8,
                dropout=dropout,
            ),
            train_cfg=TrainConfig(
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                class_weighted=class_weighted,
                seed=train_seed,
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    results: dict[str, dict] = {}

    # Variant: text_only
    results["text_only"] = _run("text_only", feats_train_text, feats_val_text)

    # Variant: vision_text
    results["vision_text"] = _run("vision_text", feats_train_vis, feats_val_vis)

    # Variant: vision_zeroed (sanity check) — same shape as vision_text but zeros
    feats_train_zero = torch.zeros_like(feats_train_vis)
    feats_val_zero = torch.zeros_like(feats_val_vis)
    # train_stage1 needs SOMETHING for the model to fit; with all-zero features
    # the model can only learn a class-prior bias. That's exactly the point: if
    # macro-F1 matches text_only, the only information the model used was the
    # class-prior, not the screenshot.
    results["vision_zeroed"] = _run("vision_zeroed", feats_train_zero, feats_val_zero)

    # Strip large tensors before returning (Modal serializes the whole dict).
    def _slim(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "best_state"}

    summary = {
        "train_split": train_split,
        "val_split": val_split,
        "n_train": int(feats_train_text.shape[0]),
        "n_val": int(feats_val_text.shape[0]),
        "hidden_size": hidden,
        "variants": {k: _slim(v) for k, v in results.items()},
        "headline_macro_f1": {k: results[k]["best_val_macro_f1"] for k in results},
        "delta_vision_minus_text_only": (
            results["vision_text"]["best_val_macro_f1"] - results["text_only"]["best_val_macro_f1"]
        ),
        "delta_vision_minus_zeroed": (
            results["vision_text"]["best_val_macro_f1"] - results["vision_zeroed"]["best_val_macro_f1"]
        ),
    }
    return summary


@app.local_entrypoint()
def train_stage1(
    train_split: str = "train",
    val_split: str = "test_task",
    n_train: int = 0,
    n_val: int = 0,
    seed: int = 42,
    epochs: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    hidden_dim: int = 1024,
    dropout: float = 0.1,
    class_weighted: bool = True,
    batch_size: int = 128,
    train_seed: int = 42,
    out_name: str = "stage1_results",
) -> None:
    """Train the three MLP variants on cached features. Writes results JSON locally."""
    import json
    from pathlib import Path

    result = _train_stage1_remote.remote(
        train_split, val_split, n_train, n_val, seed,
        epochs, lr, weight_decay, hidden_dim, dropout, class_weighted,
        batch_size, train_seed,
    )
    out = Path(f"results/phase2/{out_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[train] wrote {out}")

    print("\n=== Stage 1 headline ===")
    print(f"train n={result['n_train']}  val n={result['n_val']}  hidden={result['hidden_size']}")
    for variant, macro in result["headline_macro_f1"].items():
        print(f"  {variant:<14} macro-F1 = {macro:.4f}")
    print(f"\nvision_text - text_only:    {result['delta_vision_minus_text_only']:+.4f}")
    print(f"vision_text - vision_zeroed: {result['delta_vision_minus_zeroed']:+.4f}")


# ---- Phase 3.5: AITW feature extraction + Stage 1 MLP -----------------------
#
# Same shape as the Multimodal-Mind2Web pipeline, but the source data is
# google-research/android_in_the_wild via cjfcsjt/AITW_General. The schema is
# different (DUAL_POINT splits into tap/swipe_* based on touch/lift delta), so
# we route through src.data.aitw which handles all of that.
#
# Crucially: AITW populates 5 canonical action classes (click, scroll, finished,
# type, hotkey), unlike Mind2Web's effective 2 (click, type). This is the data
# the project actually claims to evaluate on.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=21600,
)
def _extract_aitw_features(
    model_id: str,
    split: str,             # "train" or "test"
    config_name: str,       # "standard" — AITW_General config
    mode: str,              # "text_only" | "vision_text"
    n_steps: int,           # 0 = all
    batch_size: int,
    max_image_pixels: int,
    seed: int,
    force_recompute: bool,
) -> dict:
    """Forward-pass Qwen2-VL-2B over an AITW slice, cache pooled features."""
    import os
    import sys
    import json as _json
    from collections import Counter
    from pathlib import Path

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from src.data.aitw import iter_aitw_steps

    assert mode in ("text_only", "vision_text"), f"bad mode: {mode}"

    cache_dir = Path(STAGE1_CACHE_PATH)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"aitw_{config_name}_{split}_{mode}_n{n_steps}_seed{seed}"
    cache_file = cache_dir / f"features_{tag}.pt"
    meta_file = cache_dir / f"features_{tag}.meta.json"

    if cache_file.exists() and not force_recompute:
        print(f"[aitw-extract] cached at {cache_file}; skipping")
        meta = _json.loads(meta_file.read_text()) if meta_file.exists() else {}
        meta["cache_file"] = str(cache_file)
        meta["from_cache"] = True
        return meta

    print(f"[aitw-extract] loading {model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    hf_cache.commit()

    PROMPT_TMPL = (
        "Predict the next action type for this mobile UI task.\n"
        "Goal: {goal}\n"
        "Action type:"
    )

    # Stream from AITW; materialize the slice as we go (need images for vision).
    print(f"[aitw-extract] streaming AITW_General/{config_name}/{split} up to {n_steps or 'all'} steps...")
    steps = list(iter_aitw_steps(
        config=config_name, split=split, n_max=n_steps, include_images=(mode == "vision_text"),
    ))
    print(f"[aitw-extract] kept {len(steps)} steps after taxonomy mapping")

    all_features: list[torch.Tensor] = []
    all_labels: list[int] = []
    hidden_size: int | None = None

    pbar = tqdm(range(0, len(steps), batch_size), desc=f"aitw-extract({mode})")
    for start in pbar:
        batch = steps[start : start + batch_size]
        messages_batch = []
        for s in batch:
            content = []
            if mode == "vision_text":
                img = s.open_image()
                if max_image_pixels > 0 and img.width * img.height > max_image_pixels:
                    r = (max_image_pixels / (img.width * img.height)) ** 0.5
                    img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": PROMPT_TMPL.format(goal=s.goal_info)})
            messages_batch.append([{"role": "user", "content": content}])
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        image_inputs, video_inputs = (
            process_vision_info(messages_batch) if mode == "vision_text" else (None, None)
        )
        inputs = processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1]
            if hidden_size is None:
                hidden_size = int(last_hidden.shape[-1])
                print(f"[aitw-extract] hidden_size = {hidden_size}")
            attention_mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            pooled = pooled.to(torch.float32).cpu()

        all_features.append(pooled)
        all_labels.extend([s.canonical_action_id for s in batch])

    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"[aitw-extract] features shape: {features.shape}  labels shape: {labels.shape}")

    torch.save({"features": features, "labels": labels, "hidden_size": hidden_size}, cache_file)
    label_dist = {int(k): int(v) for k, v in Counter(all_labels).items()}
    meta = {
        "model_id": model_id,
        "dataset": "cjfcsjt/AITW_General",
        "config": config_name,
        "split": split,
        "mode": mode,
        "n_steps_requested": n_steps,
        "n_steps_used": len(steps),
        "seed": seed,
        "hidden_size": hidden_size,
        "features_shape": list(features.shape),
        "label_distribution_by_id": label_dist,
        "feature_norm_mean": float(features.norm(dim=1).mean()),
        "feature_norm_std": float(features.norm(dim=1).std()),
        "cache_file": str(cache_file),
        "from_cache": False,
    }
    meta_file.write_text(_json.dumps(meta, indent=2))
    stage1_cache.commit()
    return meta


@app.local_entrypoint()
def extract_aitw_features(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    split: str = "train",
    config_name: str = "standard",
    mode: str = "vision_text",
    n_steps: int = 5000,
    batch_size: int = 2,
    max_image_pixels: int = 1_000_000,
    seed: int = 42,
    force_recompute: bool = False,
) -> None:
    """Extract AITW features for one (split, mode) pair, cache to Modal Volume."""
    import json
    from pathlib import Path

    meta = _extract_aitw_features.remote(
        model_id, split, config_name, mode, n_steps, batch_size,
        max_image_pixels, seed, force_recompute,
    )
    out = Path(f"results/phase3/aitw_feature_meta_{split}_{mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    print(f"[aitw-extract] meta saved to {out}")
    print(f"[aitw-extract] cached: {meta.get('cache_file')}")
    print(f"[aitw-extract] kept {meta.get('n_steps_used')} steps, label dist by id = {meta.get('label_distribution_by_id')}")


@app.function(
    image=image,
    gpu="L4",
    volumes={STAGE1_CACHE_PATH: stage1_cache},
    timeout=1800,
)
def _train_stage1_aitw_remote(
    train_split: str,
    val_split: str,
    config_name: str,
    n_train: int,
    n_val: int,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    class_weighted: bool,
    batch_size: int,
    train_seed: int,
) -> dict:
    """Train Stage 1 MLP variants on cached AITW features."""
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/repo")
    import torch

    from src.models.stage1_classifier import Stage1Config
    from src.train.stage1 import TrainConfig, train_stage1

    cache_dir = Path(STAGE1_CACHE_PATH)

    def _load(split: str, mode: str, n: int):
        p = cache_dir / f"features_aitw_{config_name}_{split}_{mode}_n{n}_seed{seed}.pt"
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        blob = torch.load(p, map_location="cpu")
        return blob["features"], blob["labels"], int(blob["hidden_size"])

    ftt, lt, h1 = _load(train_split, "text_only", n_train)
    fvt, lt2, _ = _load(train_split, "vision_text", n_train)
    fvalt, lv, h2 = _load(val_split, "text_only", n_val)
    fvalv, lv2, _ = _load(val_split, "vision_text", n_val)
    assert torch.equal(lt, lt2) and torch.equal(lv, lv2), "labels diverge across modes"
    assert h1 == h2

    print(f"[aitw-train] train n={ftt.shape[0]}, val n={fvalt.shape[0]}, hidden={h1}")

    def _run(name: str, ft: torch.Tensor, fv: torch.Tensor) -> dict:
        print(f"\n[aitw-train] === {name} ===")
        return train_stage1(
            features_train=ft, labels_train=lt,
            features_val=fv, labels_val=lv,
            model_cfg=Stage1Config(feature_dim=h1, hidden_dim=hidden_dim, num_classes=8, dropout=dropout),
            train_cfg=TrainConfig(epochs=epochs, lr=lr, weight_decay=weight_decay,
                                  batch_size=batch_size, class_weighted=class_weighted, seed=train_seed),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    results = {
        "text_only": _run("text_only", ftt, fvalt),
        "vision_text": _run("vision_text", fvt, fvalv),
        "vision_zeroed": _run("vision_zeroed", torch.zeros_like(fvt), torch.zeros_like(fvalv)),
    }

    def _slim(r): return {k: v for k, v in r.items() if k != "best_state"}
    return {
        "dataset": "cjfcsjt/AITW_General",
        "config": config_name,
        "train_split": train_split,
        "val_split": val_split,
        "n_train": int(ftt.shape[0]),
        "n_val": int(fvalt.shape[0]),
        "hidden_size": h1,
        "variants": {k: _slim(v) for k, v in results.items()},
        "headline_macro_f1": {k: results[k]["best_val_macro_f1"] for k in results},
        "delta_vision_minus_text_only": (
            results["vision_text"]["best_val_macro_f1"] - results["text_only"]["best_val_macro_f1"]
        ),
        "delta_vision_minus_zeroed": (
            results["vision_text"]["best_val_macro_f1"] - results["vision_zeroed"]["best_val_macro_f1"]
        ),
    }


@app.local_entrypoint()
def train_stage1_aitw(
    train_split: str = "train",
    val_split: str = "test",
    config_name: str = "standard",
    n_train: int = 5000,
    n_val: int = 1000,
    seed: int = 42,
    epochs: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    hidden_dim: int = 1024,
    dropout: float = 0.1,
    class_weighted: bool = True,
    batch_size: int = 128,
    train_seed: int = 42,
    out_name: str = "stage1_aitw_results",
) -> None:
    """Train Stage 1 MLP on cached AITW features. Results to results/phase3/."""
    import json
    from pathlib import Path

    result = _train_stage1_aitw_remote.remote(
        train_split, val_split, config_name, n_train, n_val, seed,
        epochs, lr, weight_decay, hidden_dim, dropout, class_weighted,
        batch_size, train_seed,
    )
    out = Path(f"results/phase3/{out_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[aitw-train] wrote {out}")

    print("\n=== AITW Stage 1 headline ===")
    print(f"train n={result['n_train']}  val n={result['n_val']}  hidden={result['hidden_size']}")
    for v, mf1 in result["headline_macro_f1"].items():
        print(f"  {v:<14} macro-F1 = {mf1:.4f}")
    print(f"\nvision_text - text_only:    {result['delta_vision_minus_text_only']:+.4f}")
    print(f"vision_text - vision_zeroed: {result['delta_vision_minus_zeroed']:+.4f}")


# ---- Phase 4: Stage 2 action-type-conditioned grounding ---------------------
#
# Trains the action-type-conditioned VLM on AITW DUAL_POINT actions. Stage 1
# is *not* required for teacher-forced training — we use the gold canonical
# action ID. Student-forcing (passing Stage 1 predictions instead) is a
# follow-up once teacher-forced loss curves are healthy.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=[hf_secret],
    timeout=600,
)
def _stage2_smoke_remote(n_examples: int = 2) -> dict:
    """Load Stage 2 model on Modal, run one forward+backward on synthetic data."""
    import os
    import sys

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from PIL import Image as PILImage

    from src.data.aitw import iter_aitw_steps
    from src.models.stage2_grounding import Stage2ConditionedGrounding, Stage2Config
    from src.train.stage2 import Stage2Example, build_batch

    print("[s2-smoke] building Stage2ConditionedGrounding...")
    model = Stage2ConditionedGrounding(Stage2Config())
    hf_cache.commit()
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    print(f"[s2-smoke] device: {device}")

    # Pull a few real AITW DUAL_POINT examples
    print("[s2-smoke] streaming AITW examples...")
    examples: list[Stage2Example] = []
    for step in iter_aitw_steps(n_max=200, include_images=True):
        if step.string_label != "tap":
            continue
        examples.append(Stage2Example(
            image=step.open_image(),
            goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0]),  # (x, y) from (y, x)
        ))
        if len(examples) >= n_examples:
            break
    print(f"[s2-smoke] got {len(examples)} examples")

    # Build a batch and run forward + backward
    batch = build_batch(model, examples, device)
    print(f"[s2-smoke] batch keys: {list(batch.keys())}")
    print(f"[s2-smoke] input_ids shape: {batch['input_ids'].shape}")

    print("[s2-smoke] forward + backward...")
    out = model(**batch)
    loss = out.loss
    print(f"[s2-smoke] loss: {loss.item():.4f}")
    loss.backward()

    # Check that the action embedding got a gradient
    ae_grad_norm = float(model.action_embeddings.weight.grad.norm()) if model.action_embeddings.weight.grad is not None else 0.0
    print(f"[s2-smoke] action_embeddings grad norm: {ae_grad_norm:.4f}")

    # Check that *some* LoRA parameters got gradients (we don't enumerate all)
    lora_with_grad = 0
    for n, p in model.named_parameters():
        if "lora" in n.lower() and p.grad is not None and p.grad.norm().item() > 0:
            lora_with_grad += 1
    print(f"[s2-smoke] LoRA params with non-zero grad: {lora_with_grad}")

    return {
        "loss": float(loss.item()),
        "action_embedding_grad_norm": ae_grad_norm,
        "lora_params_with_grad": lora_with_grad,
        "n_examples": len(examples),
        "input_ids_shape": list(batch["input_ids"].shape),
    }


@app.local_entrypoint()
def stage2_smoke(n_examples: int = 2) -> None:
    """Forward+backward smoke test of the Stage 2 model on Modal L4."""
    result = _stage2_smoke_remote.remote(n_examples)
    print("\n=== Stage 2 smoke result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_train_remote(
    n_train: int,
    n_val: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    aitw_split: str,
    coord_scale: int,
    only_taps: bool,
    data_mix: str = "taps_only",
) -> dict:
    """Teacher-forced Stage 2 grounding training on AITW DUAL_POINT actions.

    `data_mix` controls which step types are eligible for the training/eval slice:
      - "taps_only"        : tap actions only (canonical class: click)
      - "taps_and_swipes"  : taps + 4 swipe directions (canonical: click + scroll)
      - "all_with_coords"  : taps + swipes + types (canonical: click, scroll, type)
    """
    import os
    import sys
    import random
    from collections import Counter

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from src.data.aitw import iter_aitw_steps
    from src.models.stage2_grounding import Stage2ConditionedGrounding, Stage2Config
    from src.train.stage2 import (
        Stage2Example,
        build_batch,
        evaluate_grounding,
    )

    torch.manual_seed(seed)
    random.seed(seed)

    print(f"[s2-train] streaming AITW {aitw_split} for {n_train + n_val} examples (data_mix={data_mix})...")
    examples_all: list[Stage2Example] = []
    target_needed = n_train + n_val
    # `data_mix` decides which AITW string_labels are eligible. Each level adds
    # action types that have a meaningful (x, y) touch coordinate to learn from.
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed_labels = _DATA_MIX_LABELS.get(data_mix, {"tap"})
    if only_taps:
        # Back-compat: legacy callers set only_taps=True/False without data_mix.
        allowed_labels = {"tap"}
    # Pull more rows than needed because we filter and AITW string_labels are skewed.
    for step in iter_aitw_steps(split=aitw_split, n_max=max(target_needed * 4, 1000), include_images=True):
        if step.string_label not in allowed_labels:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(),
            goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0]),  # (x, y) from (y, x)
        ))
        if len(examples_all) >= target_needed:
            break
    if len(examples_all) < target_needed:
        print(f"[s2-train] WARN: only got {len(examples_all)} examples after filtering")
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[s2-train] train n={len(train_examples)}  val n={len(val_examples)}")
    print(f"[s2-train] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    print("[s2-train] loading Stage 2 model...")
    model = Stage2ConditionedGrounding(Stage2Config())
    hf_cache.commit()
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        print(f"\n[s2-train] === epoch {epoch}/{epochs} ===")
        # Shuffle
        rng = random.Random(seed + epoch)
        order = list(range(len(train_examples)))
        rng.shuffle(order)

        model.train()
        epoch_loss = 0.0
        n_steps = 0
        for i in range(0, len(order), batch_size):
            batch_indices = order[i : i + batch_size]
            batch_examples = [train_examples[j] for j in batch_indices]
            batch = build_batch(model, batch_examples, device, coord_scale=coord_scale)
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1
            if (n_steps % 25) == 0:
                print(f"[s2-train]   step {n_steps}  loss={loss.item():.4f}")

        avg = epoch_loss / max(n_steps, 1)
        eval_metrics = evaluate_grounding(model, val_examples, device, coord_scale=coord_scale)
        history.append({
            "epoch": epoch,
            "train_loss": avg,
            **{k: v for k, v in eval_metrics.items() if k != "raw_outputs"},
        })
        print(f"[s2-train] epoch {epoch} avg loss = {avg:.4f}")
        print(f"[s2-train] val: parsed {eval_metrics['n_parsed']}/{eval_metrics['n_total']}")
        print(f"[s2-train] val: mean normalized L2 = {eval_metrics['mean_normalized_l2']:.3f}")
        print(f"[s2-train] val: hit@0.05={eval_metrics['hit_at_005']:.3f}  hit@0.10={eval_metrics['hit_at_010']:.3f}  hit@0.25={eval_metrics['hit_at_025']:.3f}")
        print(f"[s2-train] val sample outputs: {eval_metrics.get('raw_outputs', [])[:5]}")

    summary = {
        "variant": "D_action_conditioned",
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "coord_scale": coord_scale,
        "seed": seed,
        "aitw_split": aitw_split,
        "only_taps": only_taps,
        "data_mix": data_mix,
        "train_action_distribution": dict(Counter(e.action_type_id for e in train_examples)),
        "val_action_distribution": dict(Counter(e.action_type_id for e in val_examples)),
        "history": history,
        "final_val_metrics": history[-1] if history else None,
    }
    # Persist to the stage1-cache Volume so detached runs are recoverable —
    # the local entrypoint exits before the remote function returns when
    # `modal run --detach` is used.
    import json as _json
    from pathlib import Path as _Path
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"train_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[s2-train] persisted result to {out_path}")
    return summary


@app.function(
    image=image,
    volumes={STAGE1_CACHE_PATH: stage1_cache},
    timeout=120,
)
def _list_stage2_runs() -> list[dict]:
    """List persisted Stage 2 run summaries on the Volume."""
    import json as _json
    from pathlib import Path as _Path
    runs_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    if not runs_dir.exists():
        return []
    out = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            data = _json.loads(p.read_text())
            data["_file"] = str(p.name)
            out.append(data)
        except Exception as e:
            out.append({"_file": str(p.name), "_error": str(e)})
    return out


@app.local_entrypoint()
def list_stage2_runs() -> None:
    """Pull Stage 2 training results persisted on the Volume into results/phase4/."""
    runs = _list_stage2_runs.remote()
    print(f"[s2] found {len(runs)} runs on Volume:")
    import json
    from pathlib import Path
    for r in runs:
        f = r.get("final_val_metrics") or {}
        print(f"\n--- {r.get('_file')} ---")
        print(f"  n_train={r.get('n_train')}  epochs={r.get('epochs')}  lr={r.get('lr')}")
        if f:
            print(f"  final train_loss={f.get('train_loss', 'n/a')}  hit@0.10={f.get('hit_at_010', 'n/a')}  norm_L2={f.get('mean_normalized_l2', 'n/a')}")
        local = Path("results/phase4") / r.get("_file", "unknown.json")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(r, indent=2))
        print(f"  saved -> {local}")


@app.local_entrypoint()
def train_stage2(
    n_train: int = 500,
    n_val: int = 100,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 1,
    seed: int = 42,
    aitw_split: str = "train",
    coord_scale: int = 1000,
    only_taps: bool = True,
    data_mix: str = "taps_only",
    out_name: str = "stage2_train_results",
) -> None:
    import json
    from pathlib import Path

    # If caller picks a non-tap-only mix, drop the only_taps default so it
    # doesn't override the data_mix filter inside the remote function.
    if data_mix != "taps_only":
        only_taps = False

    result = _stage2_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, only_taps, data_mix,
    )
    out = Path(f"results/phase4/{out_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[s2-train] wrote {out}")
    if result.get("final_val_metrics"):
        m = result["final_val_metrics"]
        print(f"[s2-train] final: loss={m['train_loss']:.4f}  hit@0.10={m['hit_at_010']:.3f}  norm L2={m['mean_normalized_l2']:.3f}")


# ---- Phase 5.1: Variant A (flat baseline) Stage 2 grounding -----------------
#
# Identical to variant D except: no action embedding table, no <|action_slot|>
# token in the prompt, no slot replacement at forward time. Same Qwen2-VL-2B
# backbone + LoRA on (q,k,v,o), same coordinate-string output. Same data,
# same compute, same eval set. The only diff is the action-type conditioning
# signal — so any quality delta is attributable to that.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_variantA_train_remote(
    n_train: int,
    n_val: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    aitw_split: str,
    coord_scale: int,
    only_taps: bool,
    data_mix: str = "taps_only",
) -> dict:
    """Flat-baseline Stage 2 — plain Qwen2-VL-2B + LoRA, no action conditioning.

    `data_mix` controls eligible step types; mirror of `_stage2_train_remote`.
    """
    import os
    import sys
    import random
    from collections import Counter
    from pathlib import Path as _Path
    import json as _json

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from src.data.aitw import iter_aitw_steps
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord

    torch.manual_seed(seed)
    random.seed(seed)

    print(f"[A-train] streaming AITW {aitw_split} for {n_train + n_val} taps...")
    examples_all: list[Stage2Example] = []
    target_needed = n_train + n_val
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed_labels = _DATA_MIX_LABELS.get(data_mix, {"tap"})
    if only_taps:
        allowed_labels = {"tap"}
    for step in iter_aitw_steps(split=aitw_split, n_max=max(target_needed * 4, 1000), include_images=True):
        if step.string_label not in allowed_labels:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(),
            goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0]),
        ))
        if len(examples_all) >= target_needed:
            break

    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[A-train] data_mix={data_mix}  train n={len(train_examples)}  val n={len(val_examples)}")
    print(f"[A-train] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    print("[A-train] loading flat Qwen2-VL-2B + LoRA (no action conditioning)...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    hf_cache.commit()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[A-train] trainable {n_trainable:,} / {n_all:,} = {100*n_trainable/n_all:.4f}%")

    device = next(model.parameters()).device

    def build_batch(examples: list[Stage2Example], include_labels: bool):
        messages_batch = []
        for ex in examples:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex.image},
                    {"type": "text", "text": f"Goal: {ex.goal_info}\nPredict the action coordinate."},
                ],
            }]
            if include_labels:
                messages.append({"role": "assistant",
                                 "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            messages_batch.append(messages)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]
                starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
                cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]:
                            cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        return inputs

    def evaluate(examples: list[Stage2Example]) -> dict:
        from src.train.stage2 import _metrics_from_predictions
        model.eval()
        preds, targets, action_ids, raw_outs = [], [], [], []
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            inputs = build_batch(batch, include_labels=False)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                           pad_token_id=processor.tokenizer.eos_token_id)
            trimmed = [g[len(i):] for i, g in zip(inputs["input_ids"], generated)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
            for ex, raw in zip(batch, decoded):
                raw_outs.append(raw)
                preds.append(string_to_coord(raw, scale=coord_scale))
                targets.append(ex.target_xy)
                action_ids.append(ex.action_type_id)
        m = _metrics_from_predictions(targets, preds, action_ids)
        m["raw_outputs"] = raw_outs[:20]
        return m

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[A-train] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch)
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        model.train()
        epoch_loss, n_steps = 0.0, 0
        for i in range(0, len(order), batch_size):
            batch_examples = [train_examples[j] for j in order[i:i + batch_size]]
            batch = build_batch(batch_examples, include_labels=True)
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1
            if (n_steps % 25) == 0:
                print(f"[A-train]   step {n_steps}  loss={loss.item():.4f}")
        avg = epoch_loss / max(n_steps, 1)
        eval_m = evaluate(val_examples)
        history.append({"epoch": epoch, "train_loss": avg,
                        **{k: v for k, v in eval_m.items() if k != "raw_outputs"}})
        print(f"[A-train] epoch {epoch} avg loss = {avg:.4f}")
        print(f"[A-train] val: parsed {eval_m['n_parsed']}/{eval_m['n_total']}")
        print(f"[A-train] val: hit@0.05={eval_m['hit_at_005']:.3f}  hit@0.10={eval_m['hit_at_010']:.3f}  hit@0.25={eval_m['hit_at_025']:.3f}")
        print(f"[A-train] val sample outputs: {eval_m['raw_outputs'][:5]}")

    summary = {
        "variant": "A_flat_baseline",
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "coord_scale": coord_scale,
        "seed": seed,
        "aitw_split": aitw_split,
        "only_taps": only_taps,
        "data_mix": data_mix,
        "train_action_distribution": dict(Counter(e.action_type_id for e in train_examples)),
        "val_action_distribution": dict(Counter(e.action_type_id for e in val_examples)),
        "trainable_params": int(n_trainable),
        "all_params": int(n_all),
        "history": history,
        "final_val_metrics": history[-1] if history else None,
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"variantA_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[A-train] persisted result to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_variantA(
    n_train: int = 500,
    n_val: int = 100,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 1,
    seed: int = 42,
    aitw_split: str = "train",
    coord_scale: int = 1000,
    only_taps: bool = True,
    data_mix: str = "taps_only",
) -> None:
    """Train variant A (flat baseline, no action conditioning). Use `modal run --detach`."""
    if data_mix != "taps_only":
        only_taps = False
    result = _stage2_variantA_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, only_taps, data_mix,
    )
    if result is not None and result.get("final_val_metrics"):
        import json
        from pathlib import Path
        out = Path(f"results/phase4/variantA_seed{seed}_n{n_train}_ep{epochs}_lr{lr}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        m = result["final_val_metrics"]
        print(f"[A-train] final: loss={m['train_loss']:.4f}  hit@0.10={m['hit_at_010']:.3f}  hit@0.25={m['hit_at_025']:.3f}")


# ---- Phase 5 diagnostic: variant D with frozen action_embeddings ------------
#
# Same Stage2ConditionedGrounding pipeline as variant D, but the action
# embedding table is frozen at random init. The slot token is present in the
# prompt at the same position, but the model can't specialize per-class.
# This disentangles "slot disrupts the prompt" from "embedding can't learn fast
# enough" — both of which would cause D < A.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_Dfrozen_train_remote(
    n_train: int,
    n_val: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    aitw_split: str,
    coord_scale: int,
    data_mix: str,
) -> dict:
    """Variant D with action_embeddings frozen at random init. Diagnostic only."""
    import os
    import sys
    import random
    from collections import Counter
    from pathlib import Path as _Path
    import json as _json

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from src.data.aitw import iter_aitw_steps
    from src.models.stage2_grounding import Stage2ConditionedGrounding, Stage2Config
    from src.train.stage2 import Stage2Example, build_batch, evaluate_grounding

    torch.manual_seed(seed)
    random.seed(seed)

    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed_labels = _DATA_MIX_LABELS.get(data_mix, {"tap"})

    print(f"[Dfrozen] streaming AITW {aitw_split} (data_mix={data_mix})...")
    examples_all: list[Stage2Example] = []
    target_needed = n_train + n_val
    for step in iter_aitw_steps(split=aitw_split, n_max=max(target_needed * 4, 1000), include_images=True):
        if step.string_label not in allowed_labels:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(),
            goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0]),
        ))
        if len(examples_all) >= target_needed:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[Dfrozen] train n={len(train_examples)}  val n={len(val_examples)}")
    print(f"[Dfrozen] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    # Build the FROZEN-embedding variant
    print("[Dfrozen] loading Stage2 model with action_embeddings frozen...")
    model = Stage2ConditionedGrounding(Stage2Config(action_embeddings_trainable=False))
    hf_cache.commit()
    model.print_trainable_parameters()

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Dfrozen] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch)
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        model.train()
        epoch_loss, n_steps = 0.0, 0
        for i in range(0, len(order), batch_size):
            batch_examples = [train_examples[j] for j in order[i:i + batch_size]]
            batch = build_batch(model, batch_examples, device, coord_scale=coord_scale)
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1
            if (n_steps % 25) == 0:
                print(f"[Dfrozen]   step {n_steps}  loss={loss.item():.4f}")
        avg = epoch_loss / max(n_steps, 1)
        eval_m = evaluate_grounding(model, val_examples, device, coord_scale=coord_scale)
        history.append({"epoch": epoch, "train_loss": avg,
                        **{k: v for k, v in eval_m.items() if k != "raw_outputs"}})
        print(f"[Dfrozen] epoch {epoch} avg loss = {avg:.4f}")
        print(f"[Dfrozen] val: hit@0.05={eval_m['hit_at_005']:.3f}  hit@0.10={eval_m['hit_at_010']:.3f}  hit@0.25={eval_m['hit_at_025']:.3f}")
        print(f"[Dfrozen] val sample outputs: {eval_m.get('raw_outputs', [])[:5]}")

    summary = {
        "variant": "D_frozen_diagnostic",
        "n_train": len(train_examples), "n_val": len(val_examples),
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "coord_scale": coord_scale,
        "seed": seed, "aitw_split": aitw_split, "data_mix": data_mix,
        "history": history,
        "final_val_metrics": history[-1] if history else None,
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"Dfrozen_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[Dfrozen] persisted result to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_Dfrozen(
    n_train: int = 1000,
    n_val: int = 200,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 1,
    seed: int = 42,
    aitw_split: str = "train",
    coord_scale: int = 1000,
    data_mix: str = "taps_and_swipes",
) -> None:
    """Diagnostic variant D with frozen action embeddings. Use --detach."""
    _stage2_Dfrozen_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, data_mix,
    )


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=[hf_secret],
    timeout=900,
)
def _stage2_debug_remote() -> dict:
    """Run scripts/p5_debug_stage2.py inside Modal so it has GPU + HF cache."""
    import os, sys, json
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")
    # Execute the script's main() and return its results
    from scripts.p5_debug_stage2 import main
    result = main()
    return result


@app.local_entrypoint()
def stage2_debug() -> None:
    import json
    from pathlib import Path
    result = _stage2_debug_remote.remote()
    out = Path("results/phase4/stage2_mechanism_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[debug] wrote {out}")
    print(json.dumps(result, indent=2, default=str))


# ---- Phase 5 variant D-hook: additive action conditioning, TRUE superset of A
#
# The slot-replacement D used the inputs_embeds bypass, which routes around
# Qwen2-VL's input_ids-keyed M-RoPE 3D position computation — subtly degrading
# the spatial encoding of image patches (which is exactly what grounding needs).
#
# D-hook keeps the FULL input_ids path (identical M-RoPE to variant A) and
# injects a per-action-type bias via a forward hook on the embedding layer:
#     embed_out = embed_tokens(input_ids) + action_embeddings(action_id)[:, None, :]
# The action embedding is ZERO-INITIALIZED, so at step 0 D-hook == A exactly.
# It is a strict superset of A: if conditioning helps it wins; if not it ties.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_variantDhook_train_remote(
    n_train: int,
    n_val: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    aitw_split: str,
    coord_scale: int,
    data_mix: str,
    init_std: float,
    action_lr: float = 0.0,
) -> dict:
    """Variant D-hook: additive action conditioning via embedding-layer hook.

    init_std=0.0 -> zero init (exact superset of A). >0 -> small random init.
    action_lr<=0 -> action embedding shares the base lr. action_lr>0 -> the
    action embedding gets its own (higher) learning rate via a separate param
    group, so a zero-init embedding can actually develop per-class vectors
    instead of barely moving (ae_norm ~0.06 at lr=2e-5 was undertrained).
    """
    import os, sys, random, json as _json
    from collections import Counter
    from pathlib import Path as _Path

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from src.data.aitw import iter_aitw_steps
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord, _metrics_from_predictions

    torch.manual_seed(seed)
    random.seed(seed)

    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed = _DATA_MIX_LABELS.get(data_mix, {"tap"})

    print(f"[Dhook] streaming AITW {aitw_split} (data_mix={data_mix}, init_std={init_std})...")
    examples_all: list[Stage2Example] = []
    need = n_train + n_val
    for step in iter_aitw_steps(split=aitw_split, n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0]),
        ))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[Dhook] train n={len(train_examples)} val n={len(val_examples)}")
    print(f"[Dhook] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    lora_config = LoraConfig(r=16, lora_alpha=32,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_config)
    hf_cache.commit()

    device = next(model.parameters()).device
    hidden = model.get_input_embeddings().embedding_dim

    # Action embedding table + forward hook on the embedding layer.
    action_embeddings = nn.Embedding(8, hidden).to(device=device, dtype=torch.bfloat16)
    if init_std <= 0:
        nn.init.zeros_(action_embeddings.weight)
    else:
        nn.init.normal_(action_embeddings.weight, mean=0.0, std=init_std)

    # Holder for the current batch's action_type_id, read by the hook.
    _holder = {"action_id": None}

    embed_layer = model.get_input_embeddings()

    def _embed_hook(module, inputs, output):
        aid = _holder["action_id"]
        if aid is None:
            return output
        bias = action_embeddings(aid)  # [B, H]
        # output: [B, T, H]; broadcast-add the per-row action bias to all positions
        return output + bias.unsqueeze(1).to(output.dtype)

    hook_handle = embed_layer.register_forward_hook(_embed_hook)

    PROMPT = "Goal: {goal}\nPredict the action coordinate."

    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            content = [{"type": "image", "image": ex.image},
                       {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]
            m = [{"role": "user", "content": content}]
            if include_labels:
                m.append({"role": "assistant",
                          "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]
                starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
                cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]:
                            cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        action_ids = torch.tensor([ex.action_type_id for ex in examples], dtype=torch.long, device=device)
        return inputs, action_ids

    def evaluate(examples):
        model.eval()
        preds, targets, aids, raws = [], [], [], []
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            inputs, action_ids = build(chunk, include_labels=False)
            _holder["action_id"] = action_ids
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            _holder["action_id"] = None
            new = gen[:, prompt_len:] if gen.shape[1] > prompt_len else gen
            dec = processor.batch_decode(new, skip_special_tokens=True)
            for ex, raw in zip(chunk, dec):
                raws.append(raw)
                preds.append(string_to_coord(raw, scale=coord_scale))
                targets.append(ex.target_xy)
                aids.append(ex.action_type_id)
        m = _metrics_from_predictions(targets, preds, aids)
        m["raw_outputs"] = raws[:20]
        return m

    if action_lr and action_lr > 0:
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr},
            {"params": list(action_embeddings.parameters()), "lr": action_lr},
        ], weight_decay=0.01)
        print(f"[Dhook] separate param groups: base lr={lr}, action_emb lr={action_lr}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()),
            lr=lr, weight_decay=0.01)

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Dhook] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch)
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        model.train()
        ep_loss, nstep = 0.0, 0
        for i in range(0, len(order), batch_size):
            chunk = [train_examples[j] for j in order[i:i + batch_size]]
            inputs, action_ids = build(chunk, include_labels=True)
            _holder["action_id"] = action_ids
            out = model(**inputs)
            _holder["action_id"] = None
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()),
                max_norm=1.0)
            optimizer.step()
            ep_loss += float(loss.item()); nstep += 1
            if nstep % 25 == 0:
                print(f"[Dhook]   step {nstep}  loss={loss.item():.4f}  ae_norm={float(action_embeddings.weight.norm()):.4f}")
        avg = ep_loss / max(nstep, 1)
        em = evaluate(val_examples)
        history.append({"epoch": epoch, "train_loss": avg, "action_emb_norm": float(action_embeddings.weight.norm()),
                        **{k: v for k, v in em.items() if k != "raw_outputs"}})
        print(f"[Dhook] epoch {epoch} loss={avg:.4f} hit@0.10={em['hit_at_010']:.3f} hit@0.25={em['hit_at_025']:.3f} ae_norm={float(action_embeddings.weight.norm()):.4f}")
        print(f"[Dhook] val sample: {em.get('raw_outputs', [])[:5]}")

    hook_handle.remove()

    summary = {
        "variant": "D_hook_additive",
        "n_train": len(train_examples), "n_val": len(val_examples),
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "coord_scale": coord_scale,
        "seed": seed, "aitw_split": aitw_split, "data_mix": data_mix, "init_std": init_std,
        "action_lr": action_lr,
        "history": history, "final_val_metrics": history[-1] if history else None,
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _alr_tag = f"_alr{action_lr}" if action_lr and action_lr > 0 else ""
    out_path = cache_dir / f"Dhook_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}_init{init_std}{_alr_tag}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[Dhook] persisted result to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_Dhook(
    n_train: int = 1000,
    n_val: int = 200,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 1,
    seed: int = 42,
    aitw_split: str = "train",
    coord_scale: int = 1000,
    data_mix: str = "taps_and_swipes",
    init_std: float = 0.0,
    action_lr: float = 0.0,
) -> None:
    """Variant D-hook (additive conditioning, true superset of A). Use --detach.

    --action-lr 1e-3 gives the action embedding its own (higher) learning rate.
    """
    _stage2_variantDhook_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, data_mix, init_std, action_lr)


# ---- Phase 5 variant D-text: action type as natural language in the prompt -
#
# Instead of a learned embedding, state the action type as a word the frozen
# model already understands. Prompt: "Action: {word}. Goal: {goal} ...".
# This is strictly more information than A in a model-native format, so it
# should match or beat A. Different mechanism from D-hook (text vs embedding),
# so it's an independent test of "does action conditioning help grounding".


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_variantDtext_train_remote(
    n_train: int, n_val: int, epochs: int, lr: float, batch_size: int,
    seed: int, aitw_split: str, coord_scale: int, data_mix: str,
) -> dict:
    import os, sys, random, json as _json
    from collections import Counter
    from pathlib import Path as _Path
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from src.data.aitw import iter_aitw_steps
    from src.data.taxonomy import ID_TO_ACTION
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord, _metrics_from_predictions

    torch.manual_seed(seed); random.seed(seed)
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed = _DATA_MIX_LABELS.get(data_mix, {"tap"})

    print(f"[Dtext] streaming AITW {aitw_split} (data_mix={data_mix})...")
    examples_all = []
    need = n_train + n_val
    for step in iter_aitw_steps(split=aitw_split, n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[Dtext] train n={len(train_examples)} val n={len(val_examples)}")
    print(f"[Dtext] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device

    # The action word the model conditions on. ID_TO_ACTION gives canonical names.
    def prompt_for(ex):
        word = ID_TO_ACTION[ex.action_type_id]  # click / scroll / type / ...
        return f"Action: {word}. Goal: {ex.goal_info}\nPredict the action coordinate."

    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            content = [{"type": "image", "image": ex.image},
                       {"type": "text", "text": prompt_for(ex)}]
            m = [{"role": "user", "content": content}]
            if include_labels:
                m.append({"role": "assistant", "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]
                starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
                cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]: cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        return inputs

    def evaluate(examples):
        model.eval()
        preds, targets, aids, raws = [], [], [], []
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            inputs = build(chunk, include_labels=False)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            new = gen[:, prompt_len:] if gen.shape[1] > prompt_len else gen
            for ex, raw in zip(chunk, processor.batch_decode(new, skip_special_tokens=True)):
                raws.append(raw); preds.append(string_to_coord(raw, scale=coord_scale))
                targets.append(ex.target_xy); aids.append(ex.action_type_id)
        m = _metrics_from_predictions(targets, preds, aids); m["raw_outputs"] = raws[:20]
        return m

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Dtext] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
        model.train(); ep_loss, nstep = 0.0, 0
        for i in range(0, len(order), batch_size):
            chunk = [train_examples[j] for j in order[i:i + batch_size]]
            inputs = build(chunk, include_labels=True)
            out = model(**inputs); loss = out.loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); ep_loss += float(loss.item()); nstep += 1
            if nstep % 25 == 0:
                print(f"[Dtext]   step {nstep}  loss={loss.item():.4f}")
        avg = ep_loss / max(nstep, 1)
        em = evaluate(val_examples)
        history.append({"epoch": epoch, "train_loss": avg, **{k: v for k, v in em.items() if k != "raw_outputs"}})
        print(f"[Dtext] epoch {epoch} loss={avg:.4f} hit@0.10={em['hit_at_010']:.3f} hit@0.25={em['hit_at_025']:.3f}")
        print(f"[Dtext] val sample: {em.get('raw_outputs', [])[:5]}")

    summary = {
        "variant": "D_text_prompt",
        "n_train": len(train_examples), "n_val": len(val_examples),
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "coord_scale": coord_scale,
        "seed": seed, "aitw_split": aitw_split, "data_mix": data_mix,
        "history": history, "final_val_metrics": history[-1] if history else None,
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"Dtext_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[Dtext] persisted result to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_Dtext(
    n_train: int = 1000, n_val: int = 200, epochs: int = 2, lr: float = 2e-5,
    batch_size: int = 1, seed: int = 42, aitw_split: str = "train",
    coord_scale: int = 1000, data_mix: str = "taps_and_swipes",
) -> None:
    """Variant D-text (action word in prompt). Use --detach."""
    _stage2_variantDtext_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, data_mix)


# ---- Phase 6 variant B: auxiliary action-type loss --------------------------
#
# Flat Qwen2-VL-2B + LoRA (plain prompt -> coords, identical to variant A at
# inference) PLUS a small action-type classification head trained jointly:
#     loss = LM_coord_loss + lambda_aux * CE(aux_head(pooled_hidden), gold_type)
# The action-type signal is present ONLY as a training objective; there is no
# conditioning at the input or at inference. Tests the plan's question: does the
# signal-during-training alone help, without inference-time conditioning?
# Rubric: D-hook > B  =>  inference-time conditioning matters beyond the signal.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_variantB_train_remote(
    n_train: int, n_val: int, epochs: int, lr: float, batch_size: int,
    seed: int, aitw_split: str, coord_scale: int, data_mix: str, lambda_aux: float,
) -> dict:
    import os, sys, random, json as _json
    from collections import Counter
    from pathlib import Path as _Path
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from src.data.aitw import iter_aitw_steps
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord, _metrics_from_predictions

    torch.manual_seed(seed); random.seed(seed)
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed = _DATA_MIX_LABELS.get(data_mix, {"tap"})
    print(f"[B-train] streaming AITW {aitw_split} (data_mix={data_mix}, lambda_aux={lambda_aux})...")
    examples_all = []
    need = n_train + n_val
    for step in iter_aitw_steps(split=aitw_split, n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[B-train] train n={len(train_examples)} val n={len(val_examples)}")
    print(f"[B-train] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device
    hidden = model.get_input_embeddings().embedding_dim

    # Auxiliary action-type classification head (8 canonical classes).
    aux_head = nn.Linear(hidden, 8).to(device=device, dtype=torch.bfloat16)

    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            content = [{"type": "image", "image": ex.image},
                       {"type": "text", "text": f"Goal: {ex.goal_info}\nPredict the action coordinate."}]
            m = [{"role": "user", "content": content}]
            if include_labels:
                m.append({"role": "assistant", "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]
                starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
                cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]: cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        action_ids = torch.tensor([ex.action_type_id for ex in examples], dtype=torch.long, device=device)
        return inputs, action_ids

    def evaluate(examples):
        model.eval()
        preds, targets, aids, raws = [], [], [], []
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            inputs, _ = build(chunk, include_labels=False)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            new = gen[:, prompt_len:] if gen.shape[1] > prompt_len else gen
            for ex, raw in zip(chunk, processor.batch_decode(new, skip_special_tokens=True)):
                raws.append(raw); preds.append(string_to_coord(raw, scale=coord_scale))
                targets.append(ex.target_xy); aids.append(ex.action_type_id)
        m = _metrics_from_predictions(targets, preds, aids); m["raw_outputs"] = raws[:20]
        return m

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(aux_head.parameters()),
        lr=lr, weight_decay=0.01)

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[B-train] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
        model.train(); ep_lm, ep_aux, ep_acc, nstep = 0.0, 0.0, 0.0, 0
        for i in range(0, len(order), batch_size):
            chunk = [train_examples[j] for j in order[i:i + batch_size]]
            inputs, action_ids = build(chunk, include_labels=True)
            out = model(**inputs, output_hidden_states=True)
            lm_loss = out.loss
            last_hidden = out.hidden_states[-1]                       # [B,T,H]
            amask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * amask).sum(1) / amask.sum(1).clamp(min=1)  # [B,H]
            aux_logits = aux_head(pooled.to(aux_head.weight.dtype)).float()    # [B,8]
            aux_loss = F.cross_entropy(aux_logits, action_ids)
            loss = lm_loss + lambda_aux * aux_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(aux_head.parameters()), 1.0)
            optimizer.step()
            ep_lm += float(lm_loss.item()); ep_aux += float(aux_loss.item())
            ep_acc += float((aux_logits.argmax(-1) == action_ids).float().mean().item()); nstep += 1
            if nstep % 25 == 0:
                print(f"[B-train]   step {nstep}  lm={lm_loss.item():.4f} aux={aux_loss.item():.4f} aux_acc={ep_acc/nstep:.3f}")
        em = evaluate(val_examples)
        history.append({"epoch": epoch, "train_loss": ep_lm / max(nstep, 1),
                        "aux_loss": ep_aux / max(nstep, 1), "aux_train_acc": ep_acc / max(nstep, 1),
                        **{k: v for k, v in em.items() if k != "raw_outputs"}})
        print(f"[B-train] epoch {epoch} lm={ep_lm/max(nstep,1):.4f} aux={ep_aux/max(nstep,1):.4f} "
              f"aux_acc={ep_acc/max(nstep,1):.3f} hit@0.10={em['hit_at_010']:.3f} hit@0.25={em['hit_at_025']:.3f}")

    summary = {
        "variant": "B_aux_loss", "n_train": len(train_examples), "n_val": len(val_examples),
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "coord_scale": coord_scale,
        "seed": seed, "aitw_split": aitw_split, "data_mix": data_mix, "lambda_aux": lambda_aux,
        "history": history, "final_val_metrics": history[-1] if history else None,
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"variantB_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}_aux{lambda_aux}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[B-train] persisted result to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_variantB(
    n_train: int = 1200, n_val: int = 250, epochs: int = 2, lr: float = 2e-5,
    batch_size: int = 1, seed: int = 42, aitw_split: str = "train",
    coord_scale: int = 1000, data_mix: str = "all_with_coords", lambda_aux: float = 1.0,
) -> None:
    """Variant B (auxiliary action-type loss). Use --detach."""
    _stage2_variantB_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, data_mix, lambda_aux)


# ---- Phase 6 variant C: hard routing ----------------------------------------
#
# Reframed for the coordinate-grounding setup. The model is TRAINED to emit
# "{action_word} (x, y)" (action word then coordinate). At EVAL we HARD-ROUTE:
# the action word is forced to the (gold / Stage-1) action type as a decode
# prefix, and the model generates the coordinate conditioned on that forced
# word. This is the plan's "predict type, then constrain decoding to
# type-consistent tokens" intervention, expressed in the output stream.
# Rubric: D-hook > C  =>  the learned latent embedding beats hard routing.
#
# Matched to D-hook's setting: both use the GOLD action type (oracle), so the
# comparison isolates the conditioning MECHANISM (forced output token vs latent
# additive embedding), not classifier error.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_variantC_train_remote(
    n_train: int, n_val: int, epochs: int, lr: float, batch_size: int,
    seed: int, aitw_split: str, coord_scale: int, data_mix: str,
) -> dict:
    import os, sys, random, json as _json
    from collections import Counter
    from pathlib import Path as _Path
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from src.data.aitw import iter_aitw_steps
    from src.data.taxonomy import ID_TO_ACTION
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord, _metrics_from_predictions

    torch.manual_seed(seed); random.seed(seed)
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed = _DATA_MIX_LABELS.get(data_mix, {"tap"})
    print(f"[C-train] streaming AITW {aitw_split} (data_mix={data_mix})...")
    examples_all = []
    need = n_train + n_val
    for step in iter_aitw_steps(split=aitw_split, n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[C-train] train n={len(train_examples)} val n={len(val_examples)}")
    print(f"[C-train] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device

    PROMPT = "Goal: {goal}\nPredict the action type and coordinate."
    # Target during training: "{word} (x, y)". The word is supervised too.
    def answer_str(ex):
        return f"{ID_TO_ACTION[ex.action_type_id]} {coord_to_string(ex.target_xy, coord_scale)}"

    def build_train(examples):
        msgs = []
        for ex in examples:
            msgs.append([
                {"role": "user", "content": [{"type": "image", "image": ex.image},
                                             {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer_str(ex)}]},
            ])
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        labels = inputs["input_ids"].clone()
        im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
        for b in range(labels.shape[0]):
            row = inputs["input_ids"][b]
            starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
            cut = 0
            for idx in reversed(starts):
                if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                    cut = idx + 2
                    if cut < row.shape[0]: cut += 1
                    break
            labels[b, :cut] = -100
        inputs["labels"] = labels
        return inputs

    def build_eval_forced(examples):
        """Render the user turn + generation prompt, then APPEND the forced
        action word so generation is hard-routed to start after it."""
        msgs = [[{"role": "user", "content": [{"type": "image", "image": ex.image},
                                              {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]}]
                for ex in examples]
        rendered = []
        for ex, m in zip(examples, msgs):
            base_text = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            # HARD ROUTING: force the (gold) action word as the decode prefix.
            rendered.append(base_text + f"{ID_TO_ACTION[ex.action_type_id]} ")
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=rendered, images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        return inputs

    def evaluate(examples):
        model.eval()
        preds, targets, aids, raws = [], [], [], []
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            inputs = build_eval_forced(chunk)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            new = gen[:, prompt_len:] if gen.shape[1] > prompt_len else gen
            for ex, raw in zip(chunk, processor.batch_decode(new, skip_special_tokens=True)):
                raws.append(raw); preds.append(string_to_coord(raw, scale=coord_scale))
                targets.append(ex.target_xy); aids.append(ex.action_type_id)
        m = _metrics_from_predictions(targets, preds, aids); m["raw_outputs"] = raws[:20]
        return m

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[C-train] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
        model.train(); ep_loss, nstep = 0.0, 0
        for i in range(0, len(order), batch_size):
            chunk = [train_examples[j] for j in order[i:i + batch_size]]
            inputs = build_train(chunk)
            out = model(**inputs); loss = out.loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); ep_loss += float(loss.item()); nstep += 1
            if nstep % 25 == 0:
                print(f"[C-train]   step {nstep}  loss={loss.item():.4f}")
        em = evaluate(val_examples)
        history.append({"epoch": epoch, "train_loss": ep_loss / max(nstep, 1),
                        **{k: v for k, v in em.items() if k != "raw_outputs"}})
        print(f"[C-train] epoch {epoch} loss={ep_loss/max(nstep,1):.4f} hit@0.10={em['hit_at_010']:.3f} hit@0.25={em['hit_at_025']:.3f}")
        print(f"[C-train] val sample: {em.get('raw_outputs', [])[:5]}")

    summary = {
        "variant": "C_hard_routing", "n_train": len(train_examples), "n_val": len(val_examples),
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "coord_scale": coord_scale,
        "seed": seed, "aitw_split": aitw_split, "data_mix": data_mix,
        "routing": "gold_action_word_forced_at_decode",
        "history": history, "final_val_metrics": history[-1] if history else None,
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"variantC_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[C-train] persisted result to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_variantC(
    n_train: int = 1200, n_val: int = 250, epochs: int = 2, lr: float = 2e-5,
    batch_size: int = 1, seed: int = 42, aitw_split: str = "train",
    coord_scale: int = 1000, data_mix: str = "all_with_coords",
) -> None:
    """Variant C (hard routing: forced action word at decode). Use --detach."""
    _stage2_variantC_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, data_mix)


# ---- Phase 6 end-to-end: Stage 1 predicted types -> Stage 2 conditioning ----
#
# The project's central pitch is a TWO-STAGE pipeline: Stage 1 predicts the
# action type, Stage 2 conditions grounding on it. Every Stage 2 result so far
# used the GOLD action type (oracle upper bound). This job closes the loop:
#   1. train D-hook (additive action conditioning) as usual
#   2. extract frozen-backbone features (LoRA disabled, hook off) for train+val
#   3. train a small Stage 1 MLP head on (feature -> gold action type)
#   4. evaluate D-hook TWICE: conditioned on (i) gold types [oracle], and
#      (ii) Stage 1 PREDICTED types
# The oracle-vs-predicted gap isolates Stage-1 classifier error from grounding
# error (a must-have diagnostic from the plan).


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_e2e_remote(
    n_train: int, n_val: int, epochs: int, lr: float, batch_size: int,
    seed: int, aitw_split: str, coord_scale: int, data_mix: str,
) -> dict:
    import os, sys, random, json as _json
    from collections import Counter
    from pathlib import Path as _Path
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from sklearn.metrics import f1_score, accuracy_score
    from src.data.aitw import iter_aitw_steps
    from src.data.taxonomy import ID_TO_ACTION
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord, _metrics_from_predictions

    torch.manual_seed(seed); random.seed(seed)
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed = _DATA_MIX_LABELS.get(data_mix, {"tap"})
    print(f"[e2e] streaming AITW {aitw_split} (data_mix={data_mix})...")
    examples_all = []
    need = n_train + n_val
    for step in iter_aitw_steps(split=aitw_split, n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[e2e] train n={len(train_examples)} val n={len(val_examples)}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device
    hidden = model.get_input_embeddings().embedding_dim

    action_embeddings = nn.Embedding(8, hidden).to(device=device, dtype=torch.bfloat16)
    nn.init.zeros_(action_embeddings.weight)
    _holder = {"action_id": None}
    def _embed_hook(module, inputs, output):
        aid = _holder["action_id"]
        if aid is None:
            return output
        return output + action_embeddings(aid).unsqueeze(1).to(output.dtype)
    hook = model.get_input_embeddings().register_forward_hook(_embed_hook)

    PROMPT = "Goal: {goal}\nPredict the action coordinate."
    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            m = [{"role": "user", "content": [{"type": "image", "image": ex.image},
                                              {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]}]
            if include_labels:
                m.append({"role": "assistant", "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]
                starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
                cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]: cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        action_ids = torch.tensor([ex.action_type_id for ex in examples], dtype=torch.long, device=device)
        return inputs, action_ids

    # --- train D-hook ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()),
        lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        print(f"\n[e2e] === D-hook epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
        model.train(); nstep = 0
        for i in range(0, len(order), batch_size):
            chunk = [train_examples[j] for j in order[i:i + batch_size]]
            inputs, action_ids = build(chunk, include_labels=True)
            _holder["action_id"] = action_ids
            out = model(**inputs); loss = out.loss
            _holder["action_id"] = None
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()), 1.0)
            optimizer.step(); nstep += 1
            if nstep % 50 == 0:
                print(f"[e2e]   step {nstep}  loss={loss.item():.4f}")

    # --- extract frozen-backbone features (LoRA disabled, hook off) ---
    print("[e2e] extracting Stage 1 features (frozen backbone)...")
    def extract_feats(examples):
        feats = []
        model.eval()
        _holder["action_id"] = None
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            inputs, _ = build(chunk, include_labels=False)
            with torch.inference_mode(), model.disable_adapter():
                out = model(**inputs, output_hidden_states=True)
                lh = out.hidden_states[-1]
                am = inputs["attention_mask"].unsqueeze(-1).to(lh.dtype)
                pooled = (lh * am).sum(1) / am.sum(1).clamp(min=1)
                feats.append(pooled.float().cpu())
        return torch.cat(feats, 0)

    ftr = extract_feats(train_examples); fva = extract_feats(val_examples)
    ytr = torch.tensor([e.action_type_id for e in train_examples])
    yva = torch.tensor([e.action_type_id for e in val_examples])

    # --- train Stage 1 MLP head (class-weighted, full-batch) ---
    print("[e2e] training Stage 1 head...")
    s1 = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 8)).to(device)
    counts = torch.bincount(ytr, minlength=8).float().clamp(min=1)
    w = (counts.sum() / (8 * counts)).to(device)
    opt1 = torch.optim.AdamW(s1.parameters(), lr=1e-3, weight_decay=1e-4)
    ftr_d, ytr_d = ftr.to(device), ytr.to(device)
    for ep in range(200):
        s1.train(); opt1.zero_grad()
        logits = s1(ftr_d)
        loss = F.cross_entropy(logits, ytr_d, weight=w)
        loss.backward(); opt1.step()
    s1.eval()
    with torch.inference_mode():
        pred_val = s1(fva.to(device)).argmax(-1).cpu()
    s1_acc = float(accuracy_score(yva.numpy(), pred_val.numpy()))
    s1_macro = float(f1_score(yva.numpy(), pred_val.numpy(), average="macro", zero_division=0))
    print(f"[e2e] Stage1 val: acc={s1_acc:.3f} macroF1={s1_macro:.3f}")
    print(f"[e2e] Stage1 pred dist: {Counter(pred_val.tolist())}  gold dist: {Counter(yva.tolist())}")

    # --- eval D-hook with gold (oracle) vs predicted types ---
    def eval_with(types_tensor):
        model.eval()
        preds, targets, aids, raws = [], [], [], []
        for s in range(0, len(val_examples), batch_size):
            chunk = val_examples[s:s + batch_size]
            type_chunk = types_tensor[s:s + batch_size].to(device)
            inputs, _ = build(chunk, include_labels=False)
            _holder["action_id"] = type_chunk
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            _holder["action_id"] = None
            new = gen[:, prompt_len:] if gen.shape[1] > prompt_len else gen
            for ex, raw in zip(chunk, processor.batch_decode(new, skip_special_tokens=True)):
                raws.append(raw); preds.append(string_to_coord(raw, scale=coord_scale))
                targets.append(ex.target_xy); aids.append(ex.action_type_id)
        m = _metrics_from_predictions(targets, preds, aids); m["raw_outputs"] = raws[:10]
        return m

    print("[e2e] eval with GOLD types (oracle)...")
    oracle = eval_with(yva)
    print(f"[e2e]   oracle hit@0.10={oracle['hit_at_010']:.3f} hit@0.25={oracle['hit_at_025']:.3f} L2={oracle['mean_normalized_l2']:.3f}")
    print("[e2e] eval with PREDICTED types...")
    predicted = eval_with(pred_val)
    print(f"[e2e]   pred   hit@0.10={predicted['hit_at_010']:.3f} hit@0.25={predicted['hit_at_025']:.3f} L2={predicted['mean_normalized_l2']:.3f}")
    hook.remove()

    summary = {
        "variant": "D_hook_e2e", "n_train": len(train_examples), "n_val": len(val_examples),
        "epochs": epochs, "lr": lr, "seed": seed, "data_mix": data_mix, "coord_scale": coord_scale,
        "stage1_val_acc": s1_acc, "stage1_val_macro_f1": s1_macro,
        "oracle_metrics": {k: v for k, v in oracle.items() if k not in ("raw_outputs", "per_example_dist")},
        "predicted_metrics": {k: v for k, v in predicted.items() if k not in ("raw_outputs", "per_example_dist")},
        "oracle_per_example_dist": oracle.get("per_example_dist"),
        "predicted_per_example_dist": predicted.get("per_example_dist"),
        "gap_hit_at_010": oracle["hit_at_010"] - predicted["hit_at_010"],
        "gap_mean_l2": predicted["mean_normalized_l2"] - oracle["mean_normalized_l2"],
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"e2e_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[e2e] persisted to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_e2e(
    n_train: int = 1200, n_val: int = 250, epochs: int = 2, lr: float = 2e-5,
    batch_size: int = 1, seed: int = 42, aitw_split: str = "train",
    coord_scale: int = 1000, data_mix: str = "all_with_coords",
) -> None:
    """End-to-end: Stage 1 predicted types -> D-hook conditioning. oracle vs predicted gap. --detach."""
    _stage2_e2e_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, data_mix)


# ---- Phase 6 attention visualization ----------------------------------------
#
# "The figure that sells the mechanism": for a real screenshot, show where the
# answer token attends to image patches under click- vs type-conditioning.
# Trains D-hook briefly, then for a few val examples runs output_attentions
# forwards under different action embeddings and overlays the image-patch
# attention as a heatmap. Also records image-vs-text attention mass.
#
# Caveat: B ~= D in the ablation (conditioning mechanism is ~neutral), so this
# is a qualitative/mechanistic illustration, not a load-bearing result.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_attn_viz_remote(
    n_train: int, epochs: int, lr: float, seed: int, data_mix: str,
    n_examples: int, coord_scale: int,
) -> dict:
    import os, sys, random, io, base64, json as _json
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from src.data.aitw import iter_aitw_steps
    from src.data.taxonomy import ID_TO_ACTION, CANONICAL_ACTIONS
    from src.train.stage2 import Stage2Example, coord_to_string

    torch.manual_seed(seed); random.seed(seed)
    allowed = {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"}
    examples_all = []
    for step in iter_aitw_steps(split="train", n_max=max(n_train * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= n_train + 60:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:]

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")  # eager required for output_attentions (sdpa/flash return None)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device
    hidden = model.get_input_embeddings().embedding_dim
    action_embeddings = nn.Embedding(8, hidden).to(device=device, dtype=torch.bfloat16)
    nn.init.zeros_(action_embeddings.weight)
    _holder = {"action_id": None}
    def _hook(m, i, o):
        aid = _holder["action_id"]
        return o if aid is None else o + action_embeddings(aid).unsqueeze(1).to(o.dtype)
    h = model.get_input_embeddings().register_forward_hook(_hook)

    PROMPT = "Goal: {goal}\nPredict the action coordinate."
    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            m = [{"role": "user", "content": [{"type": "image", "image": ex.image},
                                              {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]}]
            if include_labels:
                m.append({"role": "assistant", "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        ii, vi = process_vision_info(msgs)
        inputs = processor(text=texts, images=ii, videos=vi, padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]; starts = (row == im_start).nonzero(as_tuple=True)[0].tolist(); cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]: cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        aids = torch.tensor([e.action_type_id for e in examples], dtype=torch.long, device=device)
        return inputs, aids

    print(f"[viz] training D-hook {epochs} epochs on {len(train_examples)} examples...")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()), lr=lr, weight_decay=0.01)
    for epoch in range(epochs):
        rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
        model.train()
        for i in range(0, len(order), 1):
            ex = [train_examples[order[i]]]
            inputs, aids = build(ex, include_labels=True)
            _holder["action_id"] = aids
            loss = model(**inputs).loss
            _holder["action_id"] = None
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()), 1.0)
            opt.step()

    # image token id
    img_tok = getattr(model.config, "image_token_id", None)
    if img_tok is None:
        img_tok = getattr(getattr(model.config, "vision_config", object()), "image_token_id", 151655)

    # pick a 'type' example and a 'click' example for the demo
    def pick(label_id):
        for e in val_examples:
            if e.action_type_id == label_id:
                return e
        return None
    demo = [e for e in [pick(CANONICAL_ACTIONS["type"]), pick(CANONICAL_ACTIONS["click"]),
                        pick(CANONICAL_ACTIONS["scroll"])] if e is not None][:n_examples]

    model.eval()
    figs_b64 = []
    mass_records = []
    conditions = [CANONICAL_ACTIONS["click"], CANONICAL_ACTIONS["type"], CANONICAL_ACTIONS["scroll"]]
    for ei, ex in enumerate(demo):
        inputs, _ = build([ex], include_labels=False)
        ids = inputs["input_ids"][0]
        img_pos = (ids == img_tok).nonzero(as_tuple=True)[0]
        if len(img_pos) == 0:
            continue
        grid = inputs["image_grid_thw"][0].tolist()  # [t,h,w] patches
        merge = 2
        gh, gw = grid[1] // merge, grid[2] // merge
        last_pos = int(inputs["attention_mask"][0].sum().item()) - 1
        heatmaps = {}
        for cond in conditions:
            _holder["action_id"] = torch.tensor([cond], device=device)
            with torch.inference_mode():
                out = model(**inputs, output_attentions=True)
            _holder["action_id"] = None
            # average attention over heads at a late layer, from last prompt token
            layer = len(out.attentions) * 3 // 4
            attn = out.attentions[layer][0].float().mean(0)  # [T,T]
            row = attn[last_pos]                              # [T]
            img_attn = row[img_pos]                           # [n_img]
            img_mass = float(img_attn.sum().item())
            heat = img_attn.detach().cpu().numpy()
            if heat.size == gh * gw:
                heat = heat.reshape(gh, gw)
            else:
                side = int(np.sqrt(heat.size)); heat = heat[:side * side].reshape(side, side)
            heatmaps[ID_TO_ACTION[cond]] = (heat, img_mass)

        # render: screenshot + click/type/scroll heatmap overlays
        img = ex.image.convert("RGB")
        fig, axes = plt.subplots(1, 1 + len(conditions), figsize=(4 * (1 + len(conditions)), 5))
        axes[0].imshow(img); axes[0].set_title(f"goal: {ex.goal_info[:30]}\ngold: {ID_TO_ACTION[ex.action_type_id]}", fontsize=8)
        axes[0].axis("off")
        for j, cond in enumerate(conditions):
            heat, mass = heatmaps[ID_TO_ACTION[cond]]
            ax = axes[j + 1]
            ax.imshow(img, alpha=0.5)
            ax.imshow(heat, cmap="jet", alpha=0.5, extent=[0, img.width, img.height, 0], aspect="auto")
            ax.set_title(f"cond={ID_TO_ACTION[cond]}\nimg-attn mass={mass:.3f}", fontsize=8)
            ax.axis("off")
            mass_records.append({"example": ei, "gold": ID_TO_ACTION[ex.action_type_id],
                                 "condition": ID_TO_ACTION[cond], "image_attention_mass": mass})
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
        figs_b64.append(base64.b64encode(buf.getvalue()).decode())

    h.remove()
    # Persist to the Volume so a --detach run is recoverable (local exits early).
    from pathlib import Path as _Path
    vdir = _Path(STAGE1_CACHE_PATH) / "attn_viz"
    vdir.mkdir(parents=True, exist_ok=True)
    for i, b64 in enumerate(figs_b64):
        (vdir / f"attn_example_{i}.png").write_bytes(base64.b64decode(b64))
    (vdir / "attention_mass.json").write_text(_json.dumps(mass_records, indent=2))
    stage1_cache.commit()
    print(f"[viz] persisted {len(figs_b64)} figures to Volume {vdir}")
    return {"n_figs": len(figs_b64), "figs_b64": figs_b64, "attention_mass": mass_records,
            "layer_used_frac": 0.75, "note": "attention from last prompt token to image patches"}


@app.local_entrypoint()
def attn_viz(
    n_train: int = 600, epochs: int = 2, lr: float = 2e-5, seed: int = 42,
    data_mix: str = "all_with_coords", n_examples: int = 3, coord_scale: int = 1000,
) -> None:
    """Attention viz: click/type/scroll-conditioned attention heatmaps. Writes PNGs locally."""
    import base64
    from pathlib import Path
    res = _stage2_attn_viz_remote.remote(n_train, epochs, lr, seed, data_mix, n_examples, coord_scale)
    outdir = Path("results/phase4/attn_viz"); outdir.mkdir(parents=True, exist_ok=True)
    for i, b64 in enumerate(res["figs_b64"]):
        (outdir / f"attn_example_{i}.png").write_bytes(base64.b64decode(b64))
    import json
    (outdir / "attention_mass.json").write_text(json.dumps(res["attention_mass"], indent=2))
    print(f"[viz] wrote {res['n_figs']} figures to {outdir}")
    for r in res["attention_mass"]:
        print(f"  ex{r['example']} gold={r['gold']:<9} cond={r['condition']:<9} img_mass={r['image_attention_mass']:.3f}")


@app.function(image=image, volumes={STAGE1_CACHE_PATH: stage1_cache}, timeout=300)
def _pull_attn_viz() -> dict:
    import base64 as _b64
    from pathlib import Path as _Path
    vdir = _Path(STAGE1_CACHE_PATH) / "attn_viz"
    out = {"pngs": {}, "mass": None}
    if not vdir.exists():
        return out
    for p in sorted(vdir.glob("*.png")):
        out["pngs"][p.name] = _b64.b64encode(p.read_bytes()).decode()
    mp = vdir / "attention_mass.json"
    if mp.exists():
        out["mass"] = mp.read_text()
    return out


@app.local_entrypoint()
def pull_attn_viz() -> None:
    """Pull attention-viz figures persisted on the Volume into results/phase4/attn_viz/."""
    import base64, json
    from pathlib import Path
    res = _pull_attn_viz.remote()
    outdir = Path("results/phase4/attn_viz"); outdir.mkdir(parents=True, exist_ok=True)
    for name, b64 in res["pngs"].items():
        (outdir / name).write_bytes(base64.b64decode(b64))
    if res["mass"]:
        (outdir / "attention_mass.json").write_text(res["mass"])
    print(f"[viz] pulled {len(res['pngs'])} figures to {outdir}")


# ---- Phase 8 aggregate attention analysis (replaces the 3-example anecdote) --
#
# Trains a conditioned variant (Dhook | Dtoken) at the headline config with
# eager attention, then over the first `n_examples` val examples measures, under
# gold / wrong (cyclic-permuted) / zero conditioning:
#   image_mass    attention from the last prompt token to all image tokens
#   target_frac_r share of that image attention within radius r of the gold
#                 point (r = 0.10, 0.25 normalized) -- "does conditioning move
#                 evidence toward the target?"
#   entropy       entropy of the image-attention distribution
#   hit@r         greedy-decoded coordinate hit under the same conditioning
# Head-averaged attention is recorded at the 1/2-depth, 3/4-depth and last
# layers (3/4 is the primary, matching attn_viz). Persists raw per-example
# records + a paired-bootstrap summary + a few high-DPI heatmaps to the Volume
# under attn_aggregate/<variant>/ so a --detach run is recoverable.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_attn_aggregate_remote(
    variant: str, n_train: int, n_val: int, epochs: int, lr: float, seed: int,
    data_mix: str, n_examples: int, coord_scale: int, n_render: int, init_std: float,
) -> dict:
    import os, sys, random, math, textwrap, json as _json
    from collections import Counter, defaultdict
    from pathlib import Path as _Path
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from src.data.aitw import iter_aitw_steps
    from src.data.taxonomy import ID_TO_ACTION
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord

    assert variant in ("Dhook", "Dtoken"), variant
    torch.manual_seed(seed); random.seed(seed)
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed = _DATA_MIX_LABELS.get(data_mix, {"tap"})
    print(f"[attn-{variant}] streaming AITW train (data_mix={data_mix})...")
    examples_all = []
    need = n_train + n_val
    for step in iter_aitw_steps(split="train", n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[attn-{variant}] train n={len(train_examples)} val n={len(val_examples)} "
          f"val dist={Counter(e.action_type_id for e in val_examples)}")

    # eager attention is required for output_attentions (sdpa/flash return None)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    SLOT = "<|action_slot|>"
    slot_id = None
    if variant == "Dtoken":
        n_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": [SLOT]})
        if n_added:
            base.resize_token_embeddings(len(processor.tokenizer))
        slot_id = processor.tokenizer.convert_tokens_to_ids(SLOT)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device
    hidden = model.get_input_embeddings().embedding_dim

    action_embeddings = nn.Embedding(8, hidden).to(device=device, dtype=torch.bfloat16)
    if variant == "Dhook":
        nn.init.zeros_(action_embeddings.weight)
    else:
        nn.init.normal_(action_embeddings.weight, mean=0.0, std=init_std)
    _holder = {"action_id": None}

    if variant == "Dhook":
        def _hook(module, args, output):
            aid = _holder["action_id"]
            return output if aid is None else output + action_embeddings(aid).unsqueeze(1).to(output.dtype)
        PROMPT = "Goal: {goal}\nPredict the action coordinate."
    else:
        def _hook(module, args, output):
            aid = _holder["action_id"]
            if aid is None:
                return output
            slot_mask = (args[0] == slot_id)
            if not bool(slot_mask.any()):
                return output
            out = output.clone()
            for bt in slot_mask.nonzero(as_tuple=False):
                b, t = int(bt[0]), int(bt[1])
                out[b, t] = action_embeddings(aid[b]).to(out.dtype)
            return out
        PROMPT = f"{SLOT} Goal: {{goal}}\nPredict the action coordinate."
    hook = model.get_input_embeddings().register_forward_hook(_hook)

    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            m = [{"role": "user", "content": [{"type": "image", "image": ex.image},
                                              {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]}]
            if include_labels:
                m.append({"role": "assistant", "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        ii, vi = process_vision_info(msgs)
        inputs = processor(text=texts, images=ii, videos=vi, padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]; starts = (row == im_start).nonzero(as_tuple=True)[0].tolist(); cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]: cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        aids = torch.tensor([e.action_type_id for e in examples], dtype=torch.long, device=device)
        return inputs, aids

    # ---- train (byte-identical loop to the per-variant remotes, batch 1) ----
    params = [p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
        model.train(); ep_loss = 0.0
        for n, i in enumerate(order):
            inputs, aids = build([train_examples[i]], True)
            _holder["action_id"] = aids
            loss = model(**inputs).loss
            _holder["action_id"] = None
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            ep_loss += float(loss.item())
            if (n + 1) % 100 == 0:
                print(f"[attn-{variant}]   epoch {epoch} step {n+1}/{len(order)} loss={loss.item():.4f}", flush=True)
        print(f"[attn-{variant}] epoch {epoch} avg loss={ep_loss/max(len(order),1):.4f} "
              f"ae_norm={float(action_embeddings.weight.norm()):.4f}", flush=True)

    # ---- attention + prediction under gold / wrong / zero conditioning ----
    img_tok = getattr(model.config, "image_token_id", None)
    if img_tok is None:
        img_tok = getattr(getattr(model.config, "vision_config", object()), "image_token_id", 151655)
    distinct = sorted({e.action_type_id for e in val_examples})
    wrong_map = {d: distinct[(i + 1) % len(distinct)] for i, d in enumerate(distinct)}
    probe = val_examples[:n_examples]
    radii = (0.10, 0.25)
    model.eval()
    records, renders = [], {}
    n_layers = None
    for k, ex in enumerate(probe):
        inputs, aids = build([ex], False)
        ids = inputs["input_ids"][0]
        img_pos = (ids == img_tok).nonzero(as_tuple=True)[0]
        if len(img_pos) == 0:
            continue
        t_, gh2, gw2 = inputs["image_grid_thw"][0].tolist()
        gh, gw = gh2 // 2, gw2 // 2
        n_img = len(img_pos)
        if gh * gw != n_img:
            side = int(math.sqrt(n_img)); gh, gw = side, max(1, n_img // side)
        rows = torch.arange(n_img, device=device) // gw
        cols = torch.arange(n_img, device=device) % gw
        cx = (cols.float() + 0.5) / gw; cy = (rows.float() + 0.5) / gh
        tx, ty = ex.target_xy
        d_tok = torch.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
        masks = {r: (d_tok <= r) for r in radii}
        last_pos = int(inputs["attention_mask"][0].sum().item()) - 1
        rec = {"i": k, "gold_action": ID_TO_ACTION[ex.action_type_id], "target_xy": [tx, ty],
               "n_img_tokens": n_img, "grid": [gh, gw], "conditions": {}}
        heat_for_render = {}
        for cond in ("gold", "wrong", "zero"):
            cid = ex.action_type_id if cond != "wrong" else wrong_map[ex.action_type_id]
            saved = None
            if cond == "zero":
                saved = action_embeddings.weight.data.clone(); action_embeddings.weight.data.zero_()
            _holder["action_id"] = torch.tensor([cid], device=device)
            with torch.inference_mode():
                out = model(**inputs, output_attentions=True)
            n_layers = len(out.attentions)
            layer_ids = {"half": n_layers // 2, "3q": n_layers * 3 // 4, "last": n_layers - 1}
            cstats = {"cond_action": ID_TO_ACTION[cid], "layers": {}}
            for lname, li in layer_ids.items():
                row = out.attentions[li][0].float().mean(0)[last_pos]
                img_attn = row[img_pos]
                mass = float(img_attn.sum().item())
                p = (img_attn / img_attn.sum().clamp(min=1e-12))
                ent = float(-(p * (p + 1e-12).log()).sum().item())
                st = {"image_mass": mass, "entropy": ent}
                for r in radii:
                    tm = float(img_attn[masks[r]].sum().item())
                    st[f"target_mass_{r:.2f}"] = tm
                    st[f"target_frac_{r:.2f}"] = tm / mass if mass > 0 else 0.0
                    st[f"target_area_frac_{r:.2f}"] = float(masks[r].float().mean().item())
                cstats["layers"][lname] = st
                if lname == "3q":
                    heat_for_render[cond] = img_attn.detach().cpu().numpy()
            del out
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            _holder["action_id"] = None
            if saved is not None:
                action_embeddings.weight.data.copy_(saved)
            new = gen[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(new, skip_special_tokens=True)[0]
            pred = string_to_coord(raw, scale=coord_scale)
            dist = math.sqrt(2.0) if pred is None else math.hypot(pred[0] - tx, pred[1] - ty)
            cstats.update({"pred_xy": list(pred) if pred else None, "dist": dist,
                           "hit_010": dist <= 0.10, "hit_025": dist <= 0.25})
            rec["conditions"][cond] = cstats
        records.append(rec)
        if len(renders) < n_render and rec["gold_action"] not in renders:
            renders[rec["gold_action"]] = (ex, rec, heat_for_render, (gh, gw))
        if (k + 1) % 20 == 0:
            g = rec["conditions"]["gold"]["layers"]["3q"]; w = rec["conditions"]["wrong"]["layers"]["3q"]
            print(f"[attn-{variant}] {k+1}/{len(probe)}  gold target_frac@0.10={g['target_frac_0.10']:.3f} "
                  f"wrong={w['target_frac_0.10']:.3f}", flush=True)
    hook.remove()

    # ---- paired summary over examples (numpy bootstrap, 10k resamples) ----
    rng = np.random.default_rng(0)
    def paired(a, b, n_boot=10000):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float); d = a - b; n = len(d)
        if n == 0:
            return None
        idx = rng.integers(0, n, size=(n_boot, n)); boots = d[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5]); p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
        return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(d.mean()),
                "ci95": [float(lo), float(hi)], "p_boot": p, "n": int(n)}
    metrics = ["image_mass", "entropy"] + [f"target_frac_{r:.2f}" for r in radii] + [f"target_mass_{r:.2f}" for r in radii]
    summary = {"variant": variant, "seed": seed, "n_train": n_train, "n_val": n_val, "n_probe": len(records),
               "n_layers": n_layers, "wrong_map": {ID_TO_ACTION[k]: ID_TO_ACTION[v] for k, v in wrong_map.items()},
               "action_emb_norm": float(action_embeddings.weight.norm()), "by_layer": {}, "hits": {}, "by_class": {}}
    for lname in ("half", "3q", "last"):
        sl = {}
        for m in metrics:
            col = {c: [r["conditions"][c]["layers"][lname][m] for r in records] for c in ("gold", "wrong", "zero")}
            sl[m] = {"mean": {c: float(np.mean(v)) for c, v in col.items()},
                     "gold_minus_wrong": paired(col["gold"], col["wrong"]),
                     "gold_minus_zero": paired(col["gold"], col["zero"])}
        summary["by_layer"][lname] = sl
    for hm in ("hit_010", "hit_025", "dist"):
        col = {c: [float(r["conditions"][c][hm]) for r in records] for c in ("gold", "wrong", "zero")}
        summary["hits"][hm] = {"mean": {c: float(np.mean(v)) for c, v in col.items()},
                               "gold_minus_wrong": paired(col["gold"], col["wrong"]),
                               "gold_minus_zero": paired(col["gold"], col["zero"])}
    by_cls = defaultdict(list)
    for r in records:
        by_cls[r["gold_action"]].append(r)
    for cls, rs in by_cls.items():
        summary["by_class"][cls] = {"n": len(rs)}
        for c in ("gold", "wrong", "zero"):
            summary["by_class"][cls][c] = {
                "target_frac_0.10": float(np.mean([r["conditions"][c]["layers"]["3q"]["target_frac_0.10"] for r in rs])),
                "image_mass": float(np.mean([r["conditions"][c]["layers"]["3q"]["image_mass"] for r in rs])),
                "hit_010": float(np.mean([r["conditions"][c]["hit_010"] for r in rs]))}
    s3 = summary["by_layer"]["3q"]
    print(f"[attn-{variant}] 3/4-layer target_frac@0.10: gold={s3['target_frac_0.10']['mean']['gold']:.3f} "
          f"wrong={s3['target_frac_0.10']['mean']['wrong']:.3f} zero={s3['target_frac_0.10']['mean']['zero']:.3f}")
    print(f"[attn-{variant}] hit@0.10: gold={summary['hits']['hit_010']['mean']['gold']:.3f} "
          f"wrong={summary['hits']['hit_010']['mean']['wrong']:.3f} zero={summary['hits']['hit_010']['mean']['zero']:.3f}")

    # ---- persist ----
    vdir = _Path(STAGE1_CACHE_PATH) / "attn_aggregate" / variant
    vdir.mkdir(parents=True, exist_ok=True)
    tag = f"{variant}_seed{seed}_n{n_train}"
    (vdir / f"attn_aggregate_{tag}.json").write_text(_json.dumps({"summary": summary, "records": records}, indent=1))
    for cls, (ex, rec, heats, (gh, gw)) in renders.items():
        img = ex.image.convert("RGB")
        fig, axes = plt.subplots(1, 4, figsize=(13, 5.2))
        goal = "\n".join(textwrap.wrap(ex.goal_info, 34)[:3])
        axes[0].imshow(img); axes[0].set_title(f"gold: {cls}\n{goal}", fontsize=8)
        axes[0].scatter([rec["target_xy"][0] * img.width], [rec["target_xy"][1] * img.height], s=200, marker="o",
                        facecolors="none", edgecolors="#11cc11", linewidths=2.5)
        axes[0].axis("off")
        for j, cond in enumerate(("gold", "wrong", "zero")):
            heat = heats[cond]
            heat = heat.reshape(gh, gw) if heat.size == gh * gw else heat[: (int(math.sqrt(heat.size)) ** 2)].reshape(int(math.sqrt(heat.size)), -1)
            ax = axes[j + 1]; ax.imshow(img, alpha=0.55)
            ax.imshow(heat, cmap="jet", alpha=0.5, extent=[0, img.width, img.height, 0], aspect="auto")
            ax.scatter([rec["target_xy"][0] * img.width], [rec["target_xy"][1] * img.height], s=200, marker="o",
                       facecolors="none", edgecolors="white", linewidths=2.0)
            st = rec["conditions"][cond]["layers"]["3q"]
            ax.set_title(f"cond={cond} ({rec['conditions'][cond]['cond_action']})\n"
                         f"img mass={st['image_mass']:.3f}  target frac@0.10={st['target_frac_0.10']:.2f}", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(vdir / f"attn_{tag}_{cls}.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    stage1_cache.commit()
    print(f"[attn-{variant}] persisted to {vdir}")
    return {"summary": summary, "n_records": len(records)}


@app.local_entrypoint()
def attn_aggregate(
    variant: str = "Dhook", n_train: int = 1200, n_val: int = 250, epochs: int = 2, lr: float = 2e-5,
    seed: int = 42, data_mix: str = "all_with_coords", n_examples: int = 120, coord_scale: int = 1000,
    n_render: int = 3, init_std: float = 0.02,
) -> None:
    """Aggregate attention analysis for Dhook | Dtoken at the headline config. Use --detach.

    init_std only applies to Dtoken (Dhook is always zero-init, matching the headline runs).
    Pull afterwards with `modal run modal_app.py::pull_attn_aggregate`.
    """
    _stage2_attn_aggregate_remote.remote(
        variant, n_train, n_val, epochs, lr, seed, data_mix, n_examples, coord_scale, n_render, init_std)


@app.function(image=image, volumes={STAGE1_CACHE_PATH: stage1_cache}, timeout=300)
def _pull_attn_aggregate() -> dict:
    import base64 as _b64
    from pathlib import Path as _Path
    vdir = _Path(STAGE1_CACHE_PATH) / "attn_aggregate"
    out = {}
    if not vdir.exists():
        return out
    for p in sorted(vdir.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(vdir))] = _b64.b64encode(p.read_bytes()).decode()
    return out


@app.local_entrypoint()
def pull_attn_aggregate() -> None:
    """Pull attn_aggregate/ from the Volume into results/phase8/attn_aggregate/."""
    import base64
    from pathlib import Path
    res = _pull_attn_aggregate.remote()
    outdir = Path("results/phase8/attn_aggregate")
    for rel, b64 in res.items():
        dst = outdir / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(b64))
        print(f"[attn] pulled {dst}")
    print(f"[attn] {len(res)} files")


# ---- Phase 6 variant D-token: M-RoPE-correct prepended action token ---------
#
# The hypothesis's ACTUAL architecture, done correctly. A dedicated
# <|action_slot|> token sits in the prompt as a REAL position in input_ids, so
# Qwen2-VL computes M-RoPE 3D positions and merges image features normally
# (everything is keyed off input_ids — unlike the broken D-slot which fed
# inputs_embeds and bypassed M-RoPE). A forward hook on the embedding layer
# REPLACES that one slot position's embedding with the learned action embedding
# (full norm, attention-routable) — unlike D-hook which adds a tiny bias to
# every position (a diffuse, non-routable smear that collapsed to ~0).
#
# This is the clean test of "does the learned-embedding conditioning beat the
# auxiliary loss (B)?" — i.e., does D > B as the hypothesis predicts.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_variantDtoken_train_remote(
    n_train: int, n_val: int, epochs: int, lr: float, batch_size: int,
    seed: int, aitw_split: str, coord_scale: int, data_mix: str, init_std: float,
    causal_eval: bool = False,
) -> dict:
    import os, sys, random, json as _json
    from collections import Counter
    from pathlib import Path as _Path
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from src.data.aitw import iter_aitw_steps
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord, _metrics_from_predictions

    SLOT = "<|action_slot|>"
    torch.manual_seed(seed); random.seed(seed)
    _DATA_MIX_LABELS = {
        "taps_only": {"tap"},
        "taps_and_swipes": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"},
        "all_with_coords": {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"},
    }
    allowed = _DATA_MIX_LABELS.get(data_mix, {"tap"})
    print(f"[Dtoken] streaming AITW {aitw_split} (data_mix={data_mix}, init_std={init_std})...")
    examples_all = []
    need = n_train + n_val
    for step in iter_aitw_steps(split=aitw_split, n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[Dtoken] train n={len(train_examples)} val n={len(val_examples)}")
    print(f"[Dtoken] train action dist: {Counter(e.action_type_id for e in train_examples)}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    # add the slot token + resize so its id is valid in input_ids
    n_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": [SLOT]})
    if n_added:
        base.resize_token_embeddings(len(processor.tokenizer))
    slot_id = processor.tokenizer.convert_tokens_to_ids(SLOT)
    print(f"[Dtoken] slot token id = {slot_id} (added={n_added})")

    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device
    hidden = model.get_input_embeddings().embedding_dim

    action_embeddings = nn.Embedding(8, hidden).to(device=device, dtype=torch.bfloat16)
    nn.init.normal_(action_embeddings.weight, mean=0.0, std=init_std)

    _holder = {"action_id": None}
    def _embed_hook(module, args, output):
        aid = _holder["action_id"]
        if aid is None:
            return output
        ids = args[0]
        slot_mask = (ids == slot_id)
        if not bool(slot_mask.any()):
            return output           # generation steps after the prompt: no slot
        out = output.clone()
        positions = slot_mask.nonzero(as_tuple=False)
        for bt in positions:
            b, t = int(bt[0]), int(bt[1])
            out[b, t] = action_embeddings(aid[b]).to(out.dtype)
        return out
    hook = model.get_input_embeddings().register_forward_hook(_embed_hook)

    # slot token sits at the START of the user text content (after the image)
    PROMPT = f"{SLOT} Goal: {{goal}}\nPredict the action coordinate."

    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            m = [{"role": "user", "content": [{"type": "image", "image": ex.image},
                                              {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]}]
            if include_labels:
                m.append({"role": "assistant", "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        # sanity: exactly one slot per row
        if include_labels is not None:
            cnt = (inputs["input_ids"] == slot_id).sum(dim=1)
            assert bool((cnt == 1).all()), f"slot count per row must be 1, got {cnt.tolist()}"
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]
                starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
                cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]: cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        action_ids = torch.tensor([ex.action_type_id for ex in examples], dtype=torch.long, device=device)
        return inputs, action_ids

    def evaluate(examples, mode="gold", wrong_map=None):
        # mode: "gold" -> true action ids (normal); "wrong" -> feed a different
        # valid action id per example (causal test: does false conditioning
        # hurt?); "zero" -> zero the embedding table for the eval (ablate the
        # conditioning vector entirely). gold>>wrong/zero => embedding is used.
        model.eval()
        saved = None
        if mode == "zero":
            saved = action_embeddings.weight.data.clone()
            action_embeddings.weight.data.zero_()
        preds, targets, aids, raws = [], [], [], []
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            inputs, action_ids = build(chunk, include_labels=False)
            if mode == "wrong" and wrong_map is not None:
                action_ids = torch.tensor([wrong_map[int(a)] for a in action_ids.tolist()],
                                          dtype=torch.long, device=device)
            _holder["action_id"] = action_ids
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            _holder["action_id"] = None
            new = gen[:, prompt_len:] if gen.shape[1] > prompt_len else gen
            for ex, raw in zip(chunk, processor.batch_decode(new, skip_special_tokens=True)):
                raws.append(raw); preds.append(string_to_coord(raw, scale=coord_scale))
                targets.append(ex.target_xy); aids.append(ex.action_type_id)
        if saved is not None:
            action_embeddings.weight.data.copy_(saved)
        m = _metrics_from_predictions(targets, preds, aids); m["raw_outputs"] = raws[:10]
        return m

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()),
        lr=lr, weight_decay=0.01)

    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[Dtoken] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
        model.train(); ep_loss, nstep = 0.0, 0
        for i in range(0, len(order), batch_size):
            chunk = [train_examples[j] for j in order[i:i + batch_size]]
            inputs, action_ids = build(chunk, include_labels=True)
            _holder["action_id"] = action_ids
            out = model(**inputs); loss = out.loss
            _holder["action_id"] = None
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(action_embeddings.parameters()), 1.0)
            optimizer.step(); ep_loss += float(loss.item()); nstep += 1
            if nstep % 50 == 0:
                print(f"[Dtoken]   step {nstep}  loss={loss.item():.4f}  ae_norm={float(action_embeddings.weight.norm()):.3f}")
        em = evaluate(val_examples)
        history.append({"epoch": epoch, "train_loss": ep_loss / max(nstep, 1),
                        "action_emb_norm": float(action_embeddings.weight.norm()),
                        **{k: v for k, v in em.items() if k != "raw_outputs"}})
        print(f"[Dtoken] epoch {epoch} loss={ep_loss/max(nstep,1):.4f} hit@0.10={em['hit_at_010']:.3f} "
              f"hit@0.25={em['hit_at_025']:.3f} ae_norm={float(action_embeddings.weight.norm()):.3f}")
        print(f"[Dtoken] val sample: {em.get('raw_outputs', [])[:5]}")

    # ---- causal-use test: is the learned embedding actually used at inference? ----
    causal = None
    if causal_eval:
        distinct = sorted({e.action_type_id for e in val_examples})
        wmap = ({d: distinct[(i + 1) % len(distinct)] for i, d in enumerate(distinct)}
                if len(distinct) > 1 else None)
        print(f"[Dtoken-causal] distinct val action ids={distinct}  wrong_map={wmap}")
        em_gold = evaluate(val_examples, mode="gold")
        em_wrong = evaluate(val_examples, mode="wrong", wrong_map=wmap) if wmap else None
        em_zero = evaluate(val_examples, mode="zero")
        _strip = lambda m: ({k: v for k, v in m.items() if k != "raw_outputs"} if m else None)
        causal = {
            "gold": _strip(em_gold), "wrong": _strip(em_wrong), "zero": _strip(em_zero),
            "distinct_action_ids": distinct, "wrong_map": wmap,
            "action_emb_norm": float(action_embeddings.weight.norm()),
        }
        gg, zz = em_gold["hit_at_010"], em_zero["hit_at_010"]
        ww = em_wrong["hit_at_010"] if em_wrong else float("nan")
        print(f"[Dtoken-causal] hit@0.10  gold={gg:.3f}  wrong={ww:.3f}  zero={zz:.3f}")
        print(f"[Dtoken-causal] gold-wrong={gg-ww:+.3f}  gold-zero={gg-zz:+.3f}  "
              f"(positive => embedding causally used at inference)")

    hook.remove()
    summary = {
        "variant": "D_token_prepended", "n_train": len(train_examples), "n_val": len(val_examples),
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "coord_scale": coord_scale,
        "seed": seed, "aitw_split": aitw_split, "data_mix": data_mix, "init_std": init_std,
        "history": history, "final_val_metrics": history[-1] if history else None,
        "causal_eval": causal,
    }
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _ctag = "_causal" if causal_eval else ""
    out_path = cache_dir / f"Dtoken_seed{seed}_n{n_train}_ep{epochs}_lr{lr}_mix-{data_mix}_init{init_std}{_ctag}.json"
    out_path.write_text(_json.dumps(summary, indent=2))
    stage1_cache.commit()
    print(f"[Dtoken] persisted to {out_path}")
    return summary


@app.local_entrypoint()
def train_stage2_Dtoken(
    n_train: int = 1200, n_val: int = 250, epochs: int = 2, lr: float = 2e-5,
    batch_size: int = 1, seed: int = 42, aitw_split: str = "train",
    coord_scale: int = 1000, data_mix: str = "all_with_coords", init_std: float = 0.02,
    causal_eval: bool = False,
) -> None:
    """Variant D-token (M-RoPE-correct prepended action token). Use --detach.

    --causal-eval adds a post-training 3-way eval (gold / wrong / zero action
    embedding) that tests whether the learned embedding is causally used.
    """
    _stage2_variantDtoken_train_remote.remote(
        n_train, n_val, epochs, lr, batch_size, seed, aitw_split, coord_scale, data_mix,
        init_std, causal_eval)


# ---- Phase 6 Mind2Web grounding benchmark (proposal-named dataset) ----------
#
# Adds a SECOND Stage 2 benchmark on Multimodal-Mind2Web (a primary dataset in
# the proposal). Grounding = predict a point on the screenshot; target = the
# center of the gold element's bounding box. Two hit metrics:
#   hit@r     : predicted point within radius r of the bbox center (normalized,
#               AITW-consistent)
#   hit@bbox  : predicted point falls INSIDE the gold element bbox (the
#               Mind2Web-native element-grounding criterion)
# Mind2Web is click/type-dominated (action type ~uninformative), so this also
# serves as a cross-dataset control: we expect D-hook ~= A here, mirroring the
# AITW taps_and_swipes control.


def _m2w_parse_target_bbox(row):
    """Return (x, y, w, h) of the gold element bbox in screenshot pixels, or None."""
    import json as _json
    pcs = row.get("pos_candidates") or []
    chosen = None
    parsed = []
    for pc in pcs:
        try:
            d = _json.loads(pc) if isinstance(pc, str) else pc
        except Exception:
            continue
        parsed.append(d)
        if d.get("is_original_target") or d.get("is_top_level_target"):
            chosen = d
            break
    if chosen is None and parsed:
        chosen = parsed[0]
    if chosen is None:
        return None
    attrs = chosen.get("attributes")
    if isinstance(attrs, str):
        try:
            attrs = _json.loads(attrs)
        except Exception:
            return None
    if not attrs:
        return None
    rect = attrs.get("bounding_box_rect")
    if not rect or rect in ("-1,-1,-1,-1",):
        return None
    try:
        x, y, w, h = [float(v) for v in rect.split(",")]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


@app.function(
    image=image, gpu="L4", volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret], timeout=900,
)
def _m2w_probe() -> dict:
    """Verify bbox parsing on a few Multimodal-Mind2Web rows."""
    import os, sys, json as _json
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")
    from datasets import load_dataset
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="test_task", token=os.environ.get("HF_TOKEN"))
    out = []
    ok = 0
    for i in range(min(40, len(ds))):
        row = ds[i]
        bbox = _m2w_parse_target_bbox(row)
        img = row.get("screenshot")
        W, H = (img.width, img.height) if hasattr(img, "width") else (None, None)
        if bbox and W:
            ok += 1
            x, y, w, h = bbox
            if len(out) < 6:
                out.append({"op": str(row.get("operation"))[:60], "bbox": bbox,
                            "img_wh": [W, H], "norm_center": [(x + w / 2) / W, (y + h / 2) / H]})
    return {"parsed_ok_of_40": ok, "samples": out}


@app.local_entrypoint()
def m2w_probe() -> None:
    import json
    print(json.dumps(_m2w_probe.remote(), indent=2))


@app.function(
    image=image, gpu="L4", volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret], timeout=14400,
)
def _stage2_m2w_grounding_remote(
    variant: str, n_train: int, n_val: int, epochs: int, lr: float, batch_size: int,
    seed: int, coord_scale: int, max_image_pixels: int,
) -> dict:
    """Stage 2 grounding on Multimodal-Mind2Web. variant in {A, Dhook}.
    Target = gold element bbox center (normalized). Metrics: hit@r + hit@bbox."""
    import os, sys, random, math, json as _json
    from collections import Counter
    from pathlib import Path as _Path
    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from src.data.taxonomy import unify_action, ID_TO_ACTION
    from src.train.stage2 import coord_to_string, string_to_coord

    torch.manual_seed(seed); random.seed(seed)

    def load_split(split, n):
        ds = load_dataset("osunlp/Multimodal-Mind2Web", split=split, token=os.environ.get("HF_TOKEN"))
        rng = random.Random(seed); idx = list(range(len(ds))); rng.shuffle(idx)
        out = []
        for i in idx:
            row = ds[i]
            bbox = _m2w_parse_target_bbox(row)
            img = row.get("screenshot")
            if bbox is None or not hasattr(img, "width"):
                continue
            op_raw = row.get("operation")
            try:
                op = _json.loads(op_raw)["op"] if isinstance(op_raw, str) else op_raw.get("op")
                cid = unify_action(op, "mind2web")
            except Exception:
                continue
            W, H = img.width, img.height
            x, y, w, h = bbox
            img2 = img.convert("RGB")
            if max_image_pixels > 0 and W * H > max_image_pixels:
                r = (max_image_pixels / (W * H)) ** 0.5
                img2 = img2.resize((max(1, int(W * r)), max(1, int(H * r))))
            out.append({
                "image": img2, "goal": row.get("confirmed_task", "") or "",
                "action_id": cid,
                "target_xy": ((x + w / 2) / W, (y + h / 2) / H),
                "bbox_norm": (x / W, y / H, w / W, h / H),
            })
            if len(out) >= n:
                break
        return out

    print(f"[m2w-{variant}] loading Multimodal-Mind2Web...")
    train_ex = load_split("train", n_train)
    val_ex = load_split("test_task", n_val)
    print(f"[m2w-{variant}] train n={len(train_ex)} val n={len(val_ex)}  "
          f"train action dist={Counter(e['action_id'] for e in train_ex)}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM"))
    hf_cache.commit()
    device = next(model.parameters()).device
    hidden = model.get_input_embeddings().embedding_dim

    use_d = (variant == "Dhook")
    action_embeddings = nn.Embedding(8, hidden).to(device=device, dtype=torch.bfloat16)
    nn.init.zeros_(action_embeddings.weight)
    _holder = {"action_id": None}
    if use_d:
        def _hook(m, i, o):
            aid = _holder["action_id"]
            return o if aid is None else o + action_embeddings(aid).unsqueeze(1).to(o.dtype)
        hook = model.get_input_embeddings().register_forward_hook(_hook)

    PROMPT = "Goal: {goal}\nPredict the action coordinate."
    def build(examples, include_labels):
        msgs = []
        for ex in examples:
            m = [{"role": "user", "content": [{"type": "image", "image": ex["image"]},
                                              {"type": "text", "text": PROMPT.format(goal=ex["goal"])}]}]
            if include_labels:
                m.append({"role": "assistant", "content": [{"type": "text", "text": coord_to_string(ex["target_xy"], coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        ii, vi = process_vision_info(msgs)
        inputs = processor(text=texts, images=ii, videos=vi, padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]; starts = (row == im_start).nonzero(as_tuple=True)[0].tolist(); cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]: cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        aids = torch.tensor([e["action_id"] for e in examples], dtype=torch.long, device=device)
        return inputs, aids

    def evaluate(examples):
        model.eval()
        n_parsed = 0; dists = []; in_box = 0
        for s in range(0, len(examples), batch_size):
            chunk = examples[s:s + batch_size]
            inputs, aids = build(chunk, include_labels=False)
            if use_d: _holder["action_id"] = aids
            plen = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            if use_d: _holder["action_id"] = None
            new = gen[:, plen:] if gen.shape[1] > plen else gen
            for ex, raw in zip(chunk, processor.batch_decode(new, skip_special_tokens=True)):
                p = string_to_coord(raw, scale=coord_scale)
                if p is None:
                    dists.append(2.0 ** 0.5); continue
                n_parsed += 1
                tx, ty = ex["target_xy"]
                dists.append(math.hypot(p[0] - tx, p[1] - ty))
                bx, by, bw, bh = ex["bbox_norm"]
                if bx <= p[0] <= bx + bw and by <= p[1] <= by + bh:
                    in_box += 1
        n = len(examples)
        hit = lambda r: sum(1 for d in dists if d <= r) / n
        return {"n_total": n, "n_parsed": n_parsed, "parse_rate": n_parsed / n,
                "mean_normalized_l2": sum(dists) / n,
                "hit_at_005": hit(0.05), "hit_at_010": hit(0.10), "hit_at_025": hit(0.25),
                "hit_at_bbox": in_box / n}

    params = [p for p in model.parameters() if p.requires_grad] + (list(action_embeddings.parameters()) if use_d else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n[m2w-{variant}] === epoch {epoch}/{epochs} ===")
        rng = random.Random(seed + epoch); order = list(range(len(train_ex))); rng.shuffle(order)
        model.train(); ep_loss, nstep = 0.0, 0
        for i in range(0, len(order), batch_size):
            chunk = [train_ex[j] for j in order[i:i + batch_size]]
            inputs, aids = build(chunk, include_labels=True)
            if use_d: _holder["action_id"] = aids
            out = model(**inputs); loss = out.loss
            if use_d: _holder["action_id"] = None
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
            ep_loss += float(loss.item()); nstep += 1
            if nstep % 50 == 0:
                print(f"[m2w-{variant}]   step {nstep}  loss={loss.item():.4f}")
        em = evaluate(val_ex)
        history.append({"epoch": epoch, "train_loss": ep_loss / max(nstep, 1), **em})
        print(f"[m2w-{variant}] epoch {epoch} loss={ep_loss/max(nstep,1):.4f} "
              f"hit@0.10={em['hit_at_010']:.3f} hit@bbox={em['hit_at_bbox']:.3f} L2={em['mean_normalized_l2']:.3f}")
    if use_d: hook.remove()

    summary = {"variant": f"m2w_{variant}", "dataset": "Multimodal-Mind2Web",
               "n_train": len(train_ex), "n_val": len(val_ex), "epochs": epochs, "lr": lr,
               "seed": seed, "coord_scale": coord_scale,
               "history": history, "final_val_metrics": history[-1] if history else None}
    cache_dir = _Path(STAGE1_CACHE_PATH) / "stage2_runs"; cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"m2w_{variant}_seed{seed}_n{n_train}_ep{epochs}.json"
    out_path.write_text(_json.dumps(summary, indent=2)); stage1_cache.commit()
    print(f"[m2w-{variant}] persisted to {out_path}")
    return summary


@app.local_entrypoint()
def train_m2w_grounding(
    variant: str = "A", n_train: int = 1000, n_val: int = 250, epochs: int = 2,
    lr: float = 2e-5, batch_size: int = 1, seed: int = 42, coord_scale: int = 1000,
    max_image_pixels: int = 1000000,
) -> None:
    """Stage 2 grounding on Multimodal-Mind2Web (variant A or Dhook). Use --detach."""
    _stage2_m2w_grounding_remote.remote(
        variant, n_train, n_val, epochs, lr, batch_size, seed, coord_scale, max_image_pixels)


# ---- Phase 7: low-data sweep (extend the scaling curve into small n) --------
#
# Tests whether the conditioning advantage (B-A, D-A) GROWS as training data
# shrinks — the direct prediction of the "low-data prior" mechanism. Runs the
# WHOLE matrix for one variant (n_list x seeds) inside a SINGLE container,
# reusing the validated per-variant remote bodies via `.local()` so the
# train/eval/persist logic is byte-identical to the points already on the curve.
# Each inner run persists its own JSON to the Volume as it finishes
# (crash-resilient). Fire ONE detached run per variant (avoids the parallel-
# detach cancellation burn from Phase 6).


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_lowdata_sweep_remote(
    variant: str,
    n_list: list,
    seeds: list,
    n_val: int,
    epochs: int,
    lr: float,
    data_mix: str,
) -> dict:
    """Run one variant across (n_list x seeds) in a single container via .local()."""
    import gc

    import torch

    done = []
    for n_train in n_list:
        for seed in seeds:
            print(f"\n[sweep-{variant}] ===== n_train={n_train} seed={seed} =====", flush=True)
            if variant == "A":
                _stage2_variantA_train_remote.local(
                    n_train, n_val, epochs, lr, 1, seed, "train", 1000, False, data_mix)
            elif variant == "B":
                _stage2_variantB_train_remote.local(
                    n_train, n_val, epochs, lr, 1, seed, "train", 1000, data_mix, 1.0)
            elif variant == "Dhook":
                _stage2_variantDhook_train_remote.local(
                    n_train, n_val, epochs, lr, 1, seed, "train", 1000, data_mix, 0.0, 0.0)
            else:
                raise ValueError(f"unknown variant {variant!r} (expected A|B|Dhook)")
            done.append({"variant": variant, "n_train": n_train, "seed": seed})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[sweep-{variant}] completed {len(done)} runs: {done}", flush=True)
    return {"done": done}


@app.local_entrypoint()
def train_stage2_lowdata_sweep(
    variant: str = "A",
    n_list: str = "300,500,800",
    seeds: str = "42,43,44",
    n_val: int = 250,
    epochs: int = 2,
    lr: float = 2e-5,
    data_mix: str = "all_with_coords",
) -> None:
    """Low-data sweep for one variant (A|B|Dhook). Fire ONE detached run per variant:

        modal run --detach modal_app.py::train_stage2_lowdata_sweep --variant A
        modal run --detach modal_app.py::train_stage2_lowdata_sweep --variant B
        modal run --detach modal_app.py::train_stage2_lowdata_sweep --variant Dhook
    """
    ns = [int(x) for x in n_list.split(",")]
    ss = [int(x) for x in seeds.split(",")]
    _stage2_lowdata_sweep_remote.remote(variant, ns, ss, n_val, epochs, lr, data_mix)


# ---- Phase 7 qualitative figure: predicted-vs-ground-truth grounding ---------
#
# Trains flat A and D-hook on the SAME all_with_coords data/val set, predicts on
# the shared val examples (whose PIL screenshots are retained), then renders a
# grid of real screenshots with ground-truth (o), flat-A (x red) and conditioned
# D-hook (x blue) points. Selects a representative spread: a click where
# conditioning rescues a flat-A miss, a both-correct click, a scroll, and the
# degenerate `type` case (both miss). Saves one paper-ready PNG to the Volume.


@app.function(
    image=image,
    gpu="L4",
    volumes={HF_CACHE_PATH: hf_cache, STAGE1_CACHE_PATH: stage1_cache},
    secrets=[hf_secret],
    timeout=14400,
)
def _stage2_qualitative_remote(
    n_train: int = 800, n_val: int = 150, epochs: int = 2, lr: float = 2e-5,
    seed: int = 42, coord_scale: int = 1000, out_subdir: str = "qualitative",
) -> dict:
    import math
    import os
    import random
    import sys
    import json as _json
    from collections import Counter
    from pathlib import Path as _Path

    os.environ["HF_HOME"] = HF_CACHE_PATH
    sys.path.insert(0, "/root/repo")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from src.data.aitw import iter_aitw_steps
    from src.data.taxonomy import ID_TO_ACTION
    from src.train.stage2 import Stage2Example, coord_to_string, string_to_coord

    allowed = {"tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right", "type"}
    print("[qual] streaming all_with_coords...")
    examples_all: list[Stage2Example] = []
    need = n_train + n_val
    for step in iter_aitw_steps(split="train", n_max=max(need * 4, 1000), include_images=True):
        if step.string_label not in allowed:
            continue
        examples_all.append(Stage2Example(
            image=step.open_image(), goal_info=step.goal_info,
            action_type_id=step.canonical_action_id,
            target_xy=(step.touch_yx[1], step.touch_yx[0])))
        if len(examples_all) >= need:
            break
    train_examples = examples_all[:n_train]
    val_examples = examples_all[n_train:n_train + n_val]
    print(f"[qual] train {len(train_examples)} val {len(val_examples)} "
          f"val_dist={Counter(e.action_type_id for e in val_examples)}")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    PROMPT = "Goal: {goal}\nPredict the action coordinate."

    def build(model, examples, include_labels, device):
        msgs = []
        for ex in examples:
            content = [{"type": "image", "image": ex.image},
                       {"type": "text", "text": PROMPT.format(goal=ex.goal_info)}]
            m = [{"role": "user", "content": content}]
            if include_labels:
                m.append({"role": "assistant",
                          "content": [{"type": "text", "text": coord_to_string(ex.target_xy, coord_scale)}]})
            msgs.append(m)
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=(not include_labels)) for m in msgs]
        img_in, vid_in = process_vision_info(msgs)
        inputs = processor(text=texts, images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(device)
        if include_labels:
            labels = inputs["input_ids"].clone()
            im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
            for b in range(labels.shape[0]):
                row = inputs["input_ids"][b]
                starts = (row == im_start).nonzero(as_tuple=True)[0].tolist()
                cut = 0
                for idx in reversed(starts):
                    if idx + 1 < row.shape[0] and row[idx + 1].item() == assistant_id:
                        cut = idx + 2
                        if cut < row.shape[0]:
                            cut += 1
                        break
                labels[b, :cut] = -100
            inputs["labels"] = labels
        return inputs

    def make_model():
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        model = get_peft_model(base, LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, task_type="CAUSAL_LM"))
        hf_cache.commit()
        return model

    def train_and_predict(use_hook: bool):
        torch.manual_seed(seed); random.seed(seed)  # identical data order for both
        model = make_model(); device = next(model.parameters()).device
        holder = {"action_id": None}; action_embeddings = None; hook = None
        params = [p for p in model.parameters() if p.requires_grad]
        if use_hook:
            hidden = model.get_input_embeddings().embedding_dim
            action_embeddings = nn.Embedding(8, hidden).to(device=device, dtype=torch.bfloat16)
            nn.init.zeros_(action_embeddings.weight)

            def _hook(mod, inp, out):
                aid = holder["action_id"]
                if aid is None:
                    return out
                return out + action_embeddings(aid).unsqueeze(1).to(out.dtype)
            hook = model.get_input_embeddings().register_forward_hook(_hook)
            params = params + list(action_embeddings.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        tag = "D-hook" if use_hook else "flat-A"
        for epoch in range(1, epochs + 1):
            rng = random.Random(seed + epoch); order = list(range(len(train_examples))); rng.shuffle(order)
            model.train()
            for n, i in enumerate(order):
                ex = train_examples[i]; inputs = build(model, [ex], True, device)
                if use_hook:
                    holder["action_id"] = torch.tensor([ex.action_type_id], device=device)
                out = model(**inputs)
                if use_hook:
                    holder["action_id"] = None
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            print(f"[qual] {tag} epoch {epoch} done", flush=True)
        model.eval(); preds = []
        for ex in val_examples:
            inputs = build(model, [ex], False, device); plen = inputs["input_ids"].shape[1]
            if use_hook:
                holder["action_id"] = torch.tensor([ex.action_type_id], device=device)
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     pad_token_id=processor.tokenizer.eos_token_id)
            if use_hook:
                holder["action_id"] = None
            new = gen[:, plen:] if gen.shape[1] > plen else gen
            raw = processor.batch_decode(new, skip_special_tokens=True)[0]
            preds.append(string_to_coord(raw, scale=coord_scale))
        if hook:
            hook.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return preds

    print("[qual] training flat A...")
    predA = train_and_predict(False)
    print("[qual] training D-hook...")
    predD = train_and_predict(True)

    def dist(p, t):
        return 1.4142 if p is None else math.hypot(p[0] - t[0], p[1] - t[1])

    rows = []
    for k, ex in enumerate(val_examples):
        t = ex.target_xy
        rows.append({"i": k, "action": ID_TO_ACTION[ex.action_type_id], "gold": t,
                     "predA": predA[k], "predD": predD[k],
                     "distA": dist(predA[k], t), "distD": dist(predD[k], t)})

    clicks = [r for r in rows if r["action"] == "click"]
    scrolls = [r for r in rows if r["action"] == "scroll"]
    types = [r for r in rows if r["action"] == "type"]
    # Fixed selection protocol (documented in the figure caption): the panel
    # set is a SPREAD of outcomes, not a highlight reel -- rescues AND hurts
    # are both shown, and the base rate of each outcome class is persisted in
    # render.json so the caption can state it.
    click_wins = [r for r in clicks if r["distD"] <= 0.10 and r["distA"] >= 0.18]
    click_hurts = [r for r in clicks if r["distA"] <= 0.10 and r["distD"] >= 0.18]
    both_ok = [r for r in clicks if r["distA"] <= 0.10 and r["distD"] <= 0.10]
    both_miss = [r for r in clicks if r["distA"] > 0.10 and r["distD"] > 0.10]
    scroll_wins = [r for r in scrolls if r["distD"] <= 0.12]
    scroll_hurts = [r for r in scrolls if r["distA"] <= 0.12 and r["distD"] > 0.12]
    # prefer 'type' examples whose predictions stay on-canvas (cleaner panel)
    types_clean = [r for r in types if r["distA"] <= 1.45 and r["distD"] <= 1.45] or types

    sel: list = []; used: set = set()

    def take(lst, title, n=1, key=None):
        c = 0
        for r in (sorted(lst, key=key) if key else lst):
            if len(sel) >= 6 or r["i"] in used:
                continue
            sel.append({"title": title, **r}); used.add(r["i"]); c += 1
            if c >= n:
                break

    take(click_wins, "click: conditioning rescues", 2, key=lambda r: -(r["distA"] - r["distD"]))
    take(click_hurts, "click: conditioning hurts", 1, key=lambda r: -(r["distD"] - r["distA"]))
    take(both_ok, "click: both correct", 1, key=lambda r: r["distA"] + r["distD"])
    take(scroll_wins, "scroll: conditioning helps", 1, key=lambda r: r["distD"])
    take(types_clean, "type: degenerate (both fail)", 1, key=lambda r: r["distD"])
    for r in sorted(rows, key=lambda r: r["distD"]):  # pad to 6 with best remaining
        if len(sel) >= 6:
            break
        if r["i"] not in used:
            sel.append({"title": r["action"], **r}); used.add(r["i"])
    sel = sel[:6]
    outcome_counts = {
        "n_click": len(clicks), "n_scroll": len(scrolls), "n_type": len(types),
        "click_rescue": len(click_wins), "click_hurt": len(click_hurts),
        "click_both_ok": len(both_ok), "click_both_miss": len(both_miss),
        "scroll_D_hit012": len(scroll_wins), "scroll_hurt": len(scroll_hurts),
    }
    print(f"[qual] outcome counts: {outcome_counts}", flush=True)

    out_dir = _Path(STAGE1_CACHE_PATH) / out_subdir; out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dump raw selected screenshots + coords (for fast local re-rendering) ----
    data_dir = out_dir / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    render_panels = []
    for k, r in enumerate(sel):
        ex = val_examples[r["i"]]; im = ex.image.convert("RGB")
        im.save(data_dir / f"panel_{k}.png")
        render_panels.append({"panel": k, "file": f"panel_{k}.png", "title": r["title"],
                              "action": r["action"], "goal": ex.goal_info,
                              "W": im.width, "H": im.height, "gold": r["gold"],
                              "predA": r["predA"], "predD": r["predD"],
                              "distA": r["distA"], "distD": r["distD"]})
    (out_dir / "render.json").write_text(_json.dumps(
        {"n_train": n_train, "n_val": len(val_examples), "seed": seed,
         "click_hit_A": sum(1 for r in clicks if r["distA"] <= 0.10) / max(len(clicks), 1),
         "click_hit_D": sum(1 for r in clicks if r["distD"] <= 0.10) / max(len(clicks), 1),
         "outcome_counts": outcome_counts,
         "all_rows": rows,
         "panels": render_panels}, indent=2, default=float))

    # ---- inline render: axes LOCKED to image, off-canvas markers CLAMPED ----
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3, foreground="white")]

    def clamp(p, W, H):
        return None if p is None else (min(max(p[0], 0.0), 1.0) * W, min(max(p[1], 0.0), 1.0) * H)

    cols = 3; rowsn = max(1, math.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(cols * 2.5, rowsn * 5.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, sel):
        ex = val_examples[r["i"]]; img = np.array(ex.image.convert("RGB")); H, W = img.shape[:2]
        ax.imshow(img)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)  # lock to image extent: no whitespace blowup
        ax.scatter([r["gold"][0] * W], [r["gold"][1] * H], s=320, marker="o", facecolors="none",
                   edgecolors="#11cc11", linewidths=3.0, path_effects=halo, zorder=6, label="ground truth")
        a = clamp(r["predA"], W, H); d = clamp(r["predD"], W, H)
        if a:
            ax.scatter([a[0]], [a[1]], s=240, marker="X", c="#ec1c1c", edgecolors="white",
                       linewidths=1.4, zorder=7, label="flat A")
        if d:
            ax.scatter([d[0]], [d[1]], s=240, marker="X", c="#1f6fe0", edgecolors="white",
                       linewidths=1.4, zorder=8, label="D-hook (conditioned)")
        ax.set_anchor("N")  # top-align image in its cell so titles never collide
        ax.set_title(f"{r['title']}\nA dist={r['distA']:.2f}   D dist={r['distD']:.2f}",
                     fontsize=9, pad=6)
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.065,
                        wspace=0.06, hspace=0.40)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=3, fontsize=11, frameon=False)
    fig.suptitle("Qualitative grounding: predicted vs. ground-truth point "
                 f"(AITW all_with_coords, n_train={n_train})", y=0.975, fontsize=12)

    figpath = out_dir / "qualitative_grounding.png"
    fig.savefig(figpath, dpi=300, bbox_inches="tight"); plt.close(fig)
    stage1_cache.commit()
    print(f"[qual] saved {figpath} with {len(sel)} panels; raw dump in {data_dir}")
    return {"figure": f"{out_subdir}/qualitative_grounding.png", "n_selected": len(sel),
            "outcome_counts": outcome_counts}


@app.local_entrypoint()
def stage2_qualitative(
    n_train: int = 800, n_val: int = 150, epochs: int = 2, lr: float = 2e-5, seed: int = 42,
    out_subdir: str = "qualitative",
) -> None:
    """Generate the predicted-vs-ground-truth qualitative figure. Use --detach.

    Pull afterwards with:
        modal volume get stage1-cache <out_subdir>/qualitative_grounding.png results/phase8/
        modal volume get stage1-cache <out_subdir>/render.json results/phase8/
    """
    _stage2_qualitative_remote.remote(n_train, n_val, epochs, lr, seed, 1000, out_subdir)
