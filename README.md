# Action-Type-Conditioned Grounding for GUI Agents

CS 231N, Spring 2026 — Aadi Chauhan, Arthur Ilyasov.

Current GUI agents (UI-TARS, OS-Atlas, SeeClick, ShowUI) decode *what action* and *where to act* in one entangled stream. The documented failure mode is committing to the wrong action type (scrolling past a visible target instead of clicking) then grounding coherently but wrongly within that incorrect type. Errors compound on long-horizon tasks.

This project factors those decisions into a two-stage pipeline on Qwen2-VL-7B:

1. **Stage 1 — Action-Type Classifier**: frozen VLM → MLP → 8-way softmax over `{click, double_click, type, scroll, drag, hotkey, wait, finished}`.
2. **Stage 2 — Conditioned Grounding**: predicted action type → learned embedding → prepended to the instruction stream; LoRA-fine-tuned to produce coordinates.

Full motivation, hypotheses, evaluation plan, and ablations: [reference/PROJECT_OVERVIEW.md](reference/PROJECT_OVERVIEW.md). Build plan: [reference/roadmap.md](reference/roadmap.md).

---

## Setup

### Local dev (Mac, no GPU)

```bash
uv sync
uv run pytest tests/test_imports.py
```

This installs deps and runs the import-only test. It does *not* load the model — that needs a GPU.

### Cluster (Stanford SCG / Sherlock or similar Linux + CUDA)

```bash
uv sync
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

The second line replaces the CPU/MPS torch wheel with the CUDA 12.1 build. Adjust `cu121` to match your cluster's driver if needed.

---

## Smoke test

Confirms Qwen2-VL-7B loads, the LoRA adapter attaches, and a forward + generate runs end-to-end.

```bash
python scripts/smoke_test.py
```

No flags needed — uses `configs/smoke_test.yaml` and synthesizes a placeholder UI screenshot. To use a real screenshot:

```bash
python scripts/smoke_test.py --image path/to/screenshot.png
```

**Expected output:**

```
[smoke] using synthesized 512x512 UI screenshot
[smoke] loading Qwen/Qwen2-VL-7B-Instruct (dtype=bfloat16)...
trainable params: ~10M || all params: ~7.6B || trainable%: ~0.13
[smoke] generating...
============================================================
PROMPT:
  Where should I click to submit this form? Respond with click(x, y).
RESPONSE:
  click(256, 406)        # or any string — the synthetic image isn't a real benchmark
============================================================
[smoke] OK
```

**Multi-GPU gotcha:** `device_map="auto"` can split layers across GPUs in ways that interact badly with LoRA. If the model fits on a single GPU, edit `configs/smoke_test.yaml` to set `device_map: {"": 0}` (YAML: `device_map: {"": 0}`). Use FSDP/DeepSpeed via `accelerate` if both GPUs are required.

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
└── reference/              # spec & lit-review docs (source of truth)
```

---

## Status

_Last updated: May 28, 2026._

- [x] **Phase 0 — repo, deps, .gitignore.** Layout from roadmap §Phase 0, uv-managed (`pyproject.toml` + `uv.lock`), Python 3.11 pinned. `tests/test_imports.py` passes locally with no GPU.
- [x] **Phase 1 — Qwen2-VL + LoRA smoke test.** `src/models/base.py:load_qwen2vl_with_lora()` + `scripts/smoke_test.py`. Synthesizes a placeholder UI screenshot so it runs on the cluster with zero datasets downloaded. _Still needs one successful run on the cluster GPU to confirm the 7B forward pass + LoRA attach work end-to-end (see [Smoke test](#smoke-test) for the exact invocation)._
- [ ] **Phase 2 — data pipeline** (Mind2Web first, then AITW / AndroidControl / Wave-UI-25K).
- [ ] **Phase 3 — Stage 1 action-type classifier** (+ vision-ablated sanity check on day 1).
- [ ] **Phase 4 — Stage 2 conditioned grounding** (action-type embedding prepended).
- [ ] **Phase 5 — eval harness** (Mind2Web type-F1 / step-SR + Showdown Clicks top-1).
- [ ] **Phase 6 — ablations A–D** (flat baseline / aux loss / hard routing / type embedding).
- [ ] **Phase 7 — analysis, attention viz, writeup.**
