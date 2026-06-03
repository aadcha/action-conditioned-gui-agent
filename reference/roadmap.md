# Action-Type-Conditioned Grounding for GUI Agents — Build Roadmap

> **Historical planning document (May 2026).** This roadmap records the original
> build plan and early compute assumptions. It is retained for provenance and
> project-direction context, not as the current empirical/result state. For the
> current status and paper framing, use `../README.md#status`,
> `../results/phase4/PHASE6_FINAL.md`, and
> `../results/phase4/PHASE7_RESULTS.md`.

**Core principle:** Start with the dumbest possible end-to-end pipeline, then make it better. The biggest failure mode for projects like this is spending three weeks on infrastructure and one week on experiments. Flip that ratio. Spend the boring days at the start so the interesting days at the end are actually informative.

Day estimates assume one person working a few hours a day. Adjust for the team.

---

## Build Progress

_As of May 28, 2026 — see [`../README.md`](../README.md#status) for the live checklist._

- ✅ **Phase 0 — repo + env** (commit `d5ec466`). Directory tree per spec, `uv` + `pyproject.toml` + `uv.lock`, `.gitignore`, `.python-version` (3.11), W&B in deps but not yet wired.
- ✅ **Phase 1 — Qwen2-VL + LoRA smoke test** (commits `d5ec466`, `d7a2f53`). `src/models/base.py:load_qwen2vl_with_lora()` + `scripts/smoke_test.py` (synthesizes a placeholder UI so it runs without any dataset). Import test passed locally; the GPU run was still pending at this May 28 snapshot and later completed. Caught `torchvision` as a missing transitive dep of `qwen_vl_utils` before the cloud did.
- ⏳ **Phase 2 — data pipeline** was the next chunk at this May 28 snapshot. Start with one Mind2Web example end-to-end before touching the taxonomy.

### Scope changes from the original roadmap (compute-budget driven)

Cloud-only setup with **$750 total credits across Modal/GCP/AWS/Azure** (no on-prem cluster). See [`../COMPUTE.md`](../COMPUTE.md) for the full plan. Key deltas:

- **Default base model: Qwen2-VL-2B-Instruct** (was 7B). All ablations run on 2B. A single 7B run on the winning variant is the stretch goal. Architecture, processor, LoRA targets, and every ablation comparison (A/B/C/D) are identical — only absolute numbers change.
- **Data scope cut to Mind2Web only** (balanced ~25k subset) for the first end-to-end ablation table. AITW / AndroidControl / Wave-UI-25K deferred — added only if the story needs them.
- **Primary cloud: Modal** for iteration ($0.80/hr L4 serverless, no idle cost). **Bulk training on GCP spot A100s** ($1.50/hr) in Phase 6. AWS/Azure held as overflow.
- **Local Mac dev path now viable**: 2B in bf16 fits in 24 GB unified memory, so smoke test and small-batch debug runs work locally on M4 Pro via MPS.

---

## High-Level: What We're Doing

We're building a GUI agent (a model that looks at a screenshot and decides where to click/type/scroll) that's better than current ones at one specific thing: not confusing *what* to do with *where* to do it.

Current models like UI-TARS take a screenshot plus an instruction and output something like `click(x=340, y=220)` or `type("hello")` or `scroll(down)` — all as one stream of text from one head. The model makes two decisions at once: which kind of action, and where on the screen. These get tangled. When the model picks the wrong action type — scrolling when it should click — the location prediction that follows is also wrong, because it's now grounding for the wrong task. This is a documented failure mode.

We separate those decisions into two explicit steps:

1. **Stage 1 — Action-Type Classifier:** a small classifier looks at the screenshot + instruction and predicts only the action type (click, double-click, type, scroll, drag, hotkey, wait, finished — 8 classes).
2. **Stage 2 — Conditioned Grounding:** knowing the action type, a second stage does spatial grounding, with a learned embedding for the predicted action type prepended to the input. That embedding hints the model: "you're doing a click task, look at buttons" or "you're doing a type task, look at input fields."

**The bet:** forcing the model to commit to an action type before grounding, and giving it a rich learned representation of that type, makes the grounding stage's attention specialize correctly. The flat baseline can't, because it juggles both decisions at once.

**Setup:** Qwen2-VL-7B base, LoRA fine-tuning, public datasets (Mind2Web, AITW, AndroidControl, Wave-UI-25K). Four ablations isolate where the gain comes from. If our version wins on non-click actions specifically (where current models are weakest), the hypothesis is validated.

---

## Phase 0 — Environment and Repo (Day 1)

Set up a clean repo before anything else; debugging environment issues mid-project taxes every later day.

```
action-type-gui-agent/
├── README.md
├── pyproject.toml          # or requirements.txt
├── configs/                # YAML configs per experiment
├── data/
│   ├── raw/                # gitignored; downloaded datasets
│   └── processed/          # gitignored; unified taxonomy outputs
├── src/
│   ├── data/               # loaders, taxonomy, sampler
│   ├── models/             # classifier, conditioned grounding
│   ├── train/              # training loops
│   ├── eval/               # eval harness
│   └── utils/              # logging, checkpointing
├── scripts/                # CLI entry points
├── notebooks/              # exploration only — no project logic
└── tests/                  # smoke tests
```

**Initial dependencies:** `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `pillow`, `wandb` (or `tensorboard`), `pytest`. Pin versions. Use `uv` or `conda` — pick one, document in README.

**Set up Weights & Biases on day 1** (free for students). Log every training run from the start. Three weeks from now you'll thank yourself when comparing 12 ablation variants.

**`.gitignore`:** `data/`, `*.pt`, `*.ckpt`, `wandb/`, `__pycache__/`, `.venv/`. Datasets are big and not ours to redistribute; checkpoints are huge.

---

## Phase 1 — One Example Through the Model End-to-End (Days 2-3)

Before writing data loaders or training anything, prove you can:

1. Load Qwen2-VL-7B with LoRA on a GPU
2. Pass one screenshot + one instruction through it
3. Get a response back

Write in a notebook first, then move working code into `src/models/base.py`:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch
from PIL import Image

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Smoke test
image = Image.open("screenshot.png")
prompt = "click the submit button"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(outputs[0]))
```

If this runs and prints something coherent, you have proof of life. If it OOMs, debug compute **now**, not the night before milestone 3. Two A100 80GB should fit Qwen2-VL-7B with rank-16 LoRA — verify with your actual cluster.

**Common gotcha:** `device_map="auto"` can split layers across GPUs in ways that interact badly with LoRA. If the model fits on one GPU, use `device_map={"": 0}` explicitly. If you need both, use `accelerate` with FSDP or DeepSpeed.

---

## Phase 2 — Data Pipeline (Days 4-7)

Most engineering time goes here. Build in this order.

### 2a. One dataset, one example
Get one Mind2Web example loaded as a dict: `screenshot` (PIL Image), `instruction` (str), `action_type` (str), `target_bbox` (xyxy), `target_coords` (xy). Run it once. Verify the screenshot opens, the bbox is on the right thing, the instruction makes sense. Mind2Web is on HuggingFace as `osunlp/Mind2Web` — screenshots are separate large downloads; read the dataset card.

### 2b. Unify the action taxonomy
`src/data/taxonomy.py` with one function: `unify_action(source_label, dataset) -> int`.

```python
CANONICAL_ACTIONS = {
    "click": 0, "double_click": 1, "type": 2, "scroll": 3,
    "drag": 4, "hotkey": 5, "wait": 6, "finished": 7,
}

MIND2WEB_TO_CANONICAL = {
    "CLICK": "click",
    "TYPE": "type",
    "SELECT": "click",   # judgment call: dropdown selection
    "HOVER": "click",    # judgment call
}

AITW_TO_CANONICAL = {
    "tap": "click",
    "swipe_up": "scroll", "swipe_down": "scroll",
    "type": "type",
    "press_back": "hotkey", "press_home": "hotkey", "press_enter": "hotkey",
}
```

Document every judgment call in comments. Teammates need to know *why* SELECT became click.

### 2c. Single unified dataset class
`src/data/unified_dataset.py` wraps all four datasets under one PyTorch `Dataset`. Each item yields the same dict schema regardless of source. This abstraction makes everything downstream simple.

### 2d. Class-balanced sampler
Click-majority is real (~80%). Write a `WeightedRandomSampler` keyed on action type. Test: draw 1000 samples, verify each class is ~12.5%.

### 2e. Smoke-test the full pipeline
```python
dataset = UnifiedGUIDataset(splits=["mind2web_train"])
loader = DataLoader(dataset, batch_size=4, sampler=BalancedSampler(dataset))
batch = next(iter(loader))
assert batch["screenshot"].shape == (4, 3, H, W)
assert batch["instruction"][0] is not None
```
Commit as a pytest test. Run before every major change.

---

## Phase 3 — Stage 1 Classifier (Days 8-10)

First real ML work; moves fast because Phases 1-2 are done.

```python
import torch.nn as nn

class ActionTypeClassifier(nn.Module):
    def __init__(self, vlm_hidden_dim=3584, num_classes=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vlm_hidden_dim * 2, 1024),  # *2 for visual + text concat
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, visual_features, text_features):
        x = torch.cat([visual_features.mean(dim=1), text_features.mean(dim=1)], dim=-1)
        return self.mlp(x)
```

**Cache features.** Pull pooled features from frozen Qwen2-VL once, save to `data/processed/features.pt`, then train the MLP on cached features. 100x faster than running the VLM every batch.

```python
for batch in tqdm(loader):
    with torch.no_grad():
        visual, text = vlm.get_features(batch["screenshot"], batch["instruction"])
        # save visual, text, labels to disk
```

Training loop then loads cached features — each epoch takes minutes. Iterate on architecture in an afternoon.

**Critical sanity check:** measure classifier accuracy with the screenshot zeroed out. If accuracy is the same as with the screenshot, the model is ignoring vision and the interesting part of the project has collapsed. This is risk #1 — measure it on day 1 of Stage 1.

**Targets:** action-type F1 in the 70-85% range on Mind2Web val. Below 60% with full features → something's broken in the data pipeline. Above 90% → suspect text leakage; run the vision-ablated check.

---

## Phase 4 — Stage 2 Conditioned Grounding (Days 11-15)

The actual research contribution. Add the action-type embedding table.

```python
class ConditionedGroundingModel(nn.Module):
    def __init__(self, base_vlm, num_action_types=8):
        super().__init__()
        self.vlm = base_vlm  # LoRA already applied
        embed_dim = base_vlm.config.hidden_size
        self.action_embeddings = nn.Embedding(num_action_types, embed_dim)

    def forward(self, screenshot, instruction_input_ids, action_type_id, **kw):
        action_embed = self.action_embeddings(action_type_id).unsqueeze(1)  # [B, 1, D]
        instruction_embeds = self.vlm.get_input_embeddings()(instruction_input_ids)
        conditioned_embeds = torch.cat([action_embed, instruction_embeds], dim=1)
        return self.vlm(inputs_embeds=conditioned_embeds, **kw)
```

**Watch out:** attention masks need updating for the prepended token. The model expects specific special tokens (image placeholder, BOS). Read Qwen2-VL's processor code to understand the exact input format before injecting your embedding.

**Training schedule:** first epochs teacher-forced (ground-truth action types), final epoch student-forced (Stage 1 predictions) so the model learns to handle classifier errors.

---

## Phase 5 — Evaluation Harness (Days 16-18)

Write once, use for all four ablations. One CLI entry point:

```bash
python scripts/eval.py --model_path runs/variant-d/best.pt \
                       --benchmark showdown_clicks \
                       --output_dir results/variant-d-showdown
```

Three benchmarks: action-type F1 on held-out Mind2Web, Showdown Clicks top-1, Mind2Web step success. Output a JSON results file + a CSV easy to compare across runs.

Run on the Stage 2 baseline (LoRA fine-tune, no conditioning) before running on your method. If the baseline number is wildly off published numbers, the eval is wrong, not the model.

---

## Phase 6 — Ablations (Days 19-22)

Four variants, matched compute, same eval:

- **A — Flat baseline:** UI-TARS-style LoRA fine-tune, no Stage 1, no conditioning.
- **B — Auxiliary loss:** flat model + classifier head, both losses trained jointly.
- **C — Hard routing:** Stage 1 prediction + constrained decoding (mask LLM logits to action-consistent tokens). Eval-time intervention; no retraining.
- **D — Type embedding (ours):** the full method.

Build the comparison table.

**Interpreting results:**
- Beat A but lose to B → signal-during-training was enough; inference-time conditioning doesn't matter.
- Beat A and B but lose to C → hard routing was the key.
- Beat all three → embedding-based conditioning is doing real work.

Each comparison rules out a different alternative explanation.

---

## Phase 7 — Analysis and Writeup (Days 23-28)

The contribution lands here, not in the architecture. With numbers in hand:

- Per-class F1 breakdowns
- Attention visualizations showing where the model looks under each condition
- Conditional accuracy: P[grounding correct | classifier correct]
- Failure-mode taxonomy: when D wins or loses to C, what kinds of examples?

The architecture is straightforward; the science is what the ablations reveal.

---

## Immediate Next Steps (Today)

1. Make the repo. Push the skeleton to GitHub.
2. Get Qwen2-VL-7B loading on your assigned GPUs with LoRA applied. Generate one output for one screenshot. Commit the smoke test.
3. Download Mind2Web. Get one example loaded as a dict with all fields.

Three things, one day, no premature optimization.

---

## Things to Avoid Early

- Don't write the ablation harness before the baseline runs.
- Don't generalize prematurely — write code for one dataset, refactor when you add the second.
- Don't tune hyperparameters before the pipeline runs end-to-end with defaults.
- Don't write abstract base classes (`BaseAgent`, `ConditionalAgent`) — write concrete classes, refactor later if needed.

The trap: ML feels rewarding (training runs, loss curves, results); engineering feels tedious (data wrangling, eval scripts, taxonomy mapping). Teams gravitate to the rewarding part and ship sloppy infrastructure. Then milestone 3 reveals the data pipeline was broken all along and the numbers mean nothing.

---

## Open Hyperparameter Decisions (Defaults, Tune Later)

| Choice | Default | Notes |
|--------|---------|-------|
| LoRA rank | 16 | Convention for 7B models; ablate {4, 8, 16, 32} if time |
| LoRA alpha | 32 | Typically 2× rank |
| LoRA targets | attn (q,k,v,o) | Consider adding FFN (gate/up/down) if underfitting |
| Stage 1 lr | 1e-4 | MLP on frozen features |
| Stage 2 lr | 2e-5 | LoRA fine-tune |
| Stage 1 epochs | 5 | Cheap; cached features |
| Stage 2 epochs | 3 | Teacher-forced then student-forced |
| Optimizer | AdamW | Standard |

None of these are principled. What separates a good project from a sloppy one isn't the initial value — it's running the ablation that shows whether the choice mattered.
