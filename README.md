# Action-Type-Conditioned Grounding for GUI Agents

CS 231N, Spring 2026 — Aadi Chauhan, Arthur Ilyasov.

Current GUI agents (UI-TARS, OS-Atlas, SeeClick, ShowUI) decode *what action* and *where to act* in one entangled stream. The documented failure mode is committing to the wrong action type (scrolling past a visible target instead of clicking) then grounding coherently but wrongly within that incorrect type. Errors compound on long-horizon tasks.

This project factors those decisions into a two-stage pipeline on Qwen2-VL:

1. **Stage 1 — Action-Type Classifier**: frozen VLM → MLP → 8-way softmax over `{click, double_click, type, scroll, drag, hotkey, wait, finished}`.
2. **Stage 2 — Conditioned Grounding**: predicted action type → learned embedding → prepended to the instruction stream; LoRA-fine-tuned to produce coordinates.

**Default base is Qwen2-VL-2B-Instruct.** Ablations run on 2B; a single 7B run on the winning variant is the stretch goal. See [COMPUTE.md](COMPUTE.md) for the compute-budget rationale.

Full motivation, hypotheses, evaluation plan, and ablations: [reference/PROJECT_OVERVIEW.md](reference/PROJECT_OVERVIEW.md). Build plan: [reference/roadmap.md](reference/roadmap.md).

---

## Setup

### Local (Mac or Linux laptop)

```bash
uv sync --extra dev
uv run pytest tests/test_imports.py
```

The 2B base is small enough (~4.4 GB in bf16) that the smoke test *can* run on a 24 GB Mac via MPS. It's slow (~minutes to load + generate) but useful for end-to-end debugging without burning credits:

```bash
uv run python scripts/smoke_test.py
```

If it OOMs on Mac, drop `dtype` to `float16` in `configs/smoke_test.yaml` or just iterate on Modal instead.

### Cloud (Modal — primary)

```bash
uv sync --extra modal
modal token new                          # one-time
modal run modal_app.py::smoke            # ~5 min, ~$0.07 on L4
```

The first run downloads Qwen2-VL-2B (~4.4 GB) into a Modal Volume; subsequent cold starts skip the download. To run the same smoke test on the 7B base:

```bash
modal run modal_app.py::smoke --config configs/smoke_test_7b.yaml
```

### Cloud (GCP — bulk training, later)

GCP spot A100s are the workhorse for the four ablation runs in Phase 6. Setup deferred until Phase 4 — see [COMPUTE.md](COMPUTE.md) for the cost plan.

---

## Smoke test (local)

Confirms Qwen2-VL loads, the LoRA adapter attaches, and a forward + generate runs end-to-end.

```bash
uv run python scripts/smoke_test.py
```

No flags needed — uses `configs/smoke_test.yaml` and synthesizes a placeholder UI screenshot. To use a real screenshot:

```bash
uv run python scripts/smoke_test.py --image path/to/screenshot.png
```

**Expected output (Qwen2-VL-2B):**

```
[smoke] using synthesized 512x512 UI screenshot
[smoke] loading Qwen/Qwen2-VL-2B-Instruct (dtype=bfloat16)...
trainable params: ~5M || all params: ~2.2B || trainable%: ~0.23
[smoke] generating...
============================================================
PROMPT:
  Where should I click to submit this form? Respond with click(x, y).
RESPONSE:
  click(256, 406)        # or any string — the synthetic image isn't a real benchmark
============================================================
[smoke] OK
```

**Multi-GPU gotcha (only relevant on real cloud GPUs):** `device_map="auto"` can split layers across GPUs in ways that interact badly with LoRA. If the model fits on a single GPU, edit `configs/smoke_test.yaml` to set `device_map: {"": 0}`. Use FSDP/DeepSpeed via `accelerate` if both GPUs are required.

---

## Repo layout

```
.
├── README.md
├── pyproject.toml          # uv-managed deps
├── configs/                # YAML per experiment
├── data/                   # raw/ and processed/ (gitignored)
├── src/
│   ├── models/             # base.py (loader), conditioned_grounding (Phase 4)
│   ├── data/               # taxonomy, unified dataset, sampler (Phase 2)
│   ├── train/              # training loops (Phases 3-4)
│   ├── eval/               # eval harness (Phase 5)
│   └── utils/              # seeding, logging
├── scripts/                # CLI entry points
├── notebooks/              # exploration only — no project logic
├── tests/                  # smoke tests, runnable without a GPU
├── modal_app.py            # Modal cloud entry points (smoke; training later)
├── COMPUTE.md              # cloud-credit budget, per-workload cost estimates
└── reference/              # spec & lit-review docs (source of truth)
```

---

## Status

_Last updated: May 28, 2026._

- [x] **Phase 0 — repo, deps, .gitignore.** Layout from roadmap §Phase 0, uv-managed (`pyproject.toml` + `uv.lock`), Python 3.11 pinned. `tests/test_imports.py` passes locally with no GPU.
- [x] **Phase 1 — Qwen2-VL + LoRA smoke test.** `src/models/base.py:load_qwen2vl_with_lora()` + `scripts/smoke_test.py`. Default base switched to **Qwen2-VL-2B-Instruct** for compute-budget reasons (see [COMPUTE.md](COMPUTE.md)); 7B path lives in `configs/smoke_test_7b.yaml`. Synthesizes a placeholder UI screenshot so it runs with zero datasets downloaded. **Modal entry point** at `modal_app.py::smoke` for cloud verification. _Still needs one successful end-to-end run (local on Mac via MPS or on Modal L4)._
- [ ] **Phase 2 — data pipeline** (Mind2Web first, balanced ~25k subset; AITW / AndroidControl / Wave-UI-25K deferred).
- [ ] **Phase 3 — Stage 1 action-type classifier** (+ vision-ablated sanity check on day 1).
- [ ] **Phase 4 — Stage 2 conditioned grounding** (action-type embedding prepended).
- [ ] **Phase 5 — eval harness** (Mind2Web type-F1 / step-SR + Showdown Clicks top-1).
- [ ] **Phase 6 — ablations A–D** (flat baseline / aux loss / hard routing / type embedding).
- [ ] **Phase 7 — analysis, attention viz, writeup.**
