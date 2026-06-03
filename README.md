# Action-Type-Conditioned Grounding for GUI Agents

CS 231N, Spring 2026 — Aadi Chauhan, Arthur Ilyasov.

Current GUI agents (UI-TARS, OS-Atlas, SeeClick, ShowUI) decode *what action* and *where to act* in one entangled stream. The documented failure mode is committing to the wrong action type (scrolling past a visible target instead of clicking) then grounding coherently but wrongly within that incorrect type. Errors compound on long-horizon tasks.

This project studies whether factoring those decisions helps on Qwen2-VL:

1. **Stage 1 — Action-Type Classifier**: frozen VLM → MLP → 8-way softmax over `{click, double_click, type, scroll, drag, hotkey, wait, finished}`.
2. **Stage 2 — Grounding with action-type supervision**: compare flat decoding against auxiliary-loss, hard-routing, additive-hook, and M-RoPE-correct prepended-token conditioning mechanisms.

**Default base is Qwen2-VL-2B-Instruct.** Ablations run on 2B; a single 7B run on the winning variant remains a stretch goal. See [COMPUTE.md](COMPUTE.md) for the historical compute-budget rationale.

Historical motivation, hypotheses, evaluation plan, and build plan are in [reference/PROJECT_OVERVIEW.md](reference/PROJECT_OVERVIEW.md) and [reference/roadmap.md](reference/roadmap.md). Current empirical verdict: [results/phase4/PHASE6_FINAL.md](results/phase4/PHASE6_FINAL.md) and [results/phase4/PHASE7_RESULTS.md](results/phase4/PHASE7_RESULTS.md).

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

### Cloud (GCP — optional overflow)

The completed Phase 6/7 runs used Modal. GCP spot A100s remain optional overflow for a future 7B or larger-scale run; see [COMPUTE.md](COMPUTE.md) for the historical cost plan.

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
└── reference/              # historical spec, proposal, and lit-review docs
```

---

## Status

_Last updated: June 3, 2026._

- [x] **Phase 0 — repo, deps.** uv-managed (`pyproject.toml` + `uv.lock`), Python 3.11 pinned. 29 tests passing.
- [x] **Phase 1 — Qwen2-VL + LoRA smoke test.** Default base switched to Qwen2-VL-2B-Instruct ([COMPUTE.md](COMPUTE.md)). Verified on Modal L4.
- [x] **Phase 2 — Stage 1 classifier on Mind2Web** (text + vision features). 3-seed result: vision_text macro-F1 = **0.605 ± 0.016** vs majority floor 0.472; vision delta +0.009. Sanity check (vision_zeroed) collapses to majority. [`results/phase2/PHASE2_RESULTS.md`](results/phase2/PHASE2_RESULTS.md).
- [x] **Phase 3 — Stage 1 classifier on AITW** (the multi-class story). 3-seed result: vision_text macro-F1 = **0.555 ± 0.036**, text_only 0.141 ± 0.032, vision_zeroed 0.110. **Vision delta = +0.414 ± 0.052 macro-F1** (40× larger than Mind2Web). [`results/phase3/PHASE3_RESULTS.md`](results/phase3/PHASE3_RESULTS.md).
- [x] **Phase 4 — Stage 2 conditioned grounding architecture.** `src/models/stage2_grounding.py` works end-to-end on Modal. First training (n=500 taps × 2 epochs): hit@0.10 = 0.380, hit@0.25 = 0.650 (~12× random). [`results/phase4/PHASE4_RESULTS.md`](results/phase4/PHASE4_RESULTS.md).
- [x] **Phase 5 — A vs D ablation** (see [`results/phase4/PHASE5_CORRECTED.md`](results/phase4/PHASE5_CORRECTED.md)). The first "8σ A beats D negative result" was a **M-RoPE bug** in D's `inputs_embeds` injection (found via `scripts/p5_debug_stage2.py`). Fixed with **D-hook** (additive conditioning, full `input_ids` path). Corrected result: D-hook **ties A** on tap/swipe (action type uninformative) and **significantly beats A on `all_with_coords`** (the signal is click-vs-scroll/type disambiguation; AITW `type` coordinates are degenerate). **Paired bootstrap** (750 paired units): hit@0.10 +0.045 (95% CI [+0.017,+0.073], p=0.002), hit@0.25 +0.055 (p<0.001), mean L2 −0.027 (p<0.001) — all 4 metrics significant.
- [x] **Phase 4.4 — grounding eval harness** ([`src/eval/bootstrap.py`](src/eval/bootstrap.py)): paired bootstrap + permutation test, per-example distance logging, 95% CIs. 29 tests pass.
- [x] **Phase 6 — full ablation + e2e pipeline + hypothesis verdict** ([`results/phase4/PHASE6_FINAL.md`](results/phase4/PHASE6_FINAL.md)). 5 variants (A/B/C/D-hook/D-token), 2 settings, 3 seeds. **Broad thesis supported** (action-type supervision helps grounding where action type is spatially informative; e2e pipeline with predicted types beats flat A, oracle gap ~0.02). **Specific architectural claim refuted** — the auxiliary loss (B) is the best conditioned variant; the literal hypothesized embedding (D-token, M-RoPE-correct, norm 2.2) improves over A on some secondary metrics but does not beat B/D-hook on headline grounding. Also: found+fixed a M-RoPE injection bug that caused an 8σ false-negative.
- [x] **Phase 7 — mechanism + low-data strengthening** ([`results/phase4/PHASE7_RESULTS.md`](results/phase4/PHASE7_RESULTS.md)). Low-data matrix is complete for A/B/D-hook at n_train ∈ {300,500,800}, seeds 42/43/44; strict audit passes via `scripts/p7_result_audit.py`. D-hook is the most stable low-data conditioned mechanism. D-token causal-use test is complete over 3 seeds: gold action id beats wrong by +0.192 hit@0.10 and zero by +0.093, so the learned embedding is used, just not the winning mechanism.
- [x] **Milestone 3** (May 29) — submitted. Slide-handoff doc: [`results/milestone3/MILESTONE3.md`](results/milestone3/MILESTONE3.md). Headline: zero-shot Qwen2-VL-2B has vision-delta = 0.000 on Mind2Web action-type; TF-IDF beats the 2B VLM by 15 macro-F1.
- [ ] **Writeup + stretch** (final report, poster; optional 7B run).

Cumulative Modal spend: **~$52 of $200 (26%).**
