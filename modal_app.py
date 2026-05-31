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
