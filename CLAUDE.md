# Project notes for the agent

Operational guide for future sessions working on this codebase. Skim this first.

---

## Hard rules

1. **NEVER add `Co-Authored-By: Claude` or any "Generated with Claude" footer** to commit messages, code comments, docs, or slides. The user is the sole author. This rule is non-negotiable — historical commits were rewritten once to scrub Claude attribution; do not reintroduce it.
2. **Do not change long-term project direction without asking.** The user wrote the multi-phase plan; execute it as written. Tactical implementation choices inside a phase are yours to make; scope cuts, deadline reinterpretations, or phase reorderings are not.
3. **Do not declare scheduling/timeline issues** unless asked. The user knows their own deadlines.
4. **Force-push to main requires explicit user authorization.** "Yes force push" or equivalent. The auto-mode classifier blocks otherwise.

---

## Project one-liner

CS 231N, Spring 2026. Goal: GUI agent that decouples *what action* (Stage 1: action-type classifier) from *where to act* (Stage 2: action-type-supervised grounding). Backbone Qwen2-VL-2B + LoRA. Final evidence supports the broad action-type supervision/factorization thesis, but refutes the literal learned prepended action-embedding claim as the winning mechanism.

Source of truth for the original spec/direction:
`reference/PROJECT_OVERVIEW.md` and `reference/roadmap.md` (both are now
explicitly marked historical). Current empirical source of truth:
`README.md#status`, `results/phase4/PHASE6_FINAL.md`, and
`results/phase4/PHASE7_RESULTS.md`.

---

## Codebase map

```
src/
  data/
    taxonomy.py            canonical 8-class action space + per-source mappings
                           (mind2web/aitw/androidcontrol → click/scroll/type/...)
    mind2web.py            text loader for osunlp/Mind2Web (unauth release)
    aitw.py                streaming loader for cjfcsjt/AITW_General.
                           CRITICAL: AITW screenshots are raw RGB bytes
                           (no PNG/JPEG header) — _decode_aitw_image_bytes
                           handles 7 known (w,h) shapes + a portrait-aspect
                           guesser. Don't use PIL.Image.open() directly.
    unified.py             UnifiedStep dataclass spanning both sources +
                           class_balanced_indices sampler
  models/
    base.py                Qwen2-VL-2B + LoRA loader (for smoke test only).
                           NOTE: in transformers v5, hidden_size is at
                           model.config.text_config.hidden_size, not
                           model.config.hidden_size — code probes both.
    stage1_classifier.py   3-layer MLP on cached VLM features
    stage2_grounding.py    Original Stage 2 slot-replacement model. Useful for
                           smoke tests and the M-RoPE bug post-mortem, but the
                           final Phase 6 variants live in modal_app.py:
                           A flat, B aux loss, C hard routing, D-hook additive
                           conditioning, D-token M-RoPE-correct prepended slot.
  train/
    stage1.py              train_stage1 + evaluate, runs on cached features
    stage2.py              build_batch (with -100 masking for assistant
                           turn), evaluate_grounding (with per-class hit@r),
                           coord_to_string ('(x,y)' on 0–999 grid)
scripts/
  smoke_test.py            local Qwen2-VL smoke test (Mac/MPS, no real GPU)
  m3_*.py                  Milestone 3 baselines + consolidation
  p2_consolidate.py        Phase 2 multi-seed aggregation + figures
  p3_*.py                  Phase 3 AITW distribution + TF-IDF + consolidate
  p5_ablation_table.py     Phase 5: aggregate every Stage 2 run JSON,
                           print A vs D table with deltas, save chart
  p5_render_writeup.py     auto-render PHASE5_HEADLINE.md from
                           ablation_summary.json
  p6_scaling_plot.py       Data-scaling / low-data curve plotter; now strict
                           about expected Phase 7 cells
  p7_result_audit.py       Strict Phase 7 completeness audit
  p7_causal_table.py       D-token gold/wrong/zero causal-use aggregation
modal_app.py               ALL Modal entry points. See "Modal patterns" below.
configs/                   YAML configs for smoke tests
tests/                     29 tests; runs in seconds on CPU
results/
  milestone3/              MILESTONE3.md handoff doc + numbers.json + figures
  phase2/                  Stage 1 Mind2Web: features metas, stage1_*.json,
                           PHASE2_RESULTS.md, figures
  phase3/                  AITW: distribution.json, feature metas,
                           stage1_aitw_*.json, PHASE3_RESULTS.md, figures,
                           aitw_text_baselines.json
  phase4/                  Stage 2 grounding: PHASE4_RESULTS.md, every
                           train_seed*.json (variant D), variantA_seed*.json,
                           ablation_summary.json, PHASE5_HEADLINE.md,
                           ABLATION_v_A_vs_D.md, figures
```

Spec files in `reference/` (untouched): PROJECT_OVERVIEW.md, roadmap.md, deep-research reviews. Treat as read-only sources.

---

## Modal patterns

The `modal_app.py` file is the SINGLE source of all cloud entrypoints. Pattern: each `_xxx_remote` function runs on Modal (decorated with `@app.function`), and a corresponding `@app.local_entrypoint()` wraps it for `modal run` invocation.

### Modal profile (Sep 2026)

`~/.modal.toml` has two profiles; the ACTIVE one (`mercor-rl`) has a dead token
("Token not found"). The project's workspace is the `sentinel` profile. Prefix
every Modal command with `MODAL_PROFILE=sentinel` rather than switching the
user's active profile:

    MODAL_PROFILE=sentinel uv run modal app list

The workspace is capped at **10 concurrent GPUs** — extra detached runs queue
(Modal emails a "reached the limit of 10 GPUs" notice; harmless).

### Volumes

- `hf-cache` (`HF_CACHE_PATH = /root/.cache/huggingface`) — caches HF model weights so cold starts skip the 4 GB Qwen2-VL download.
- `stage1-cache` (`STAGE1_CACHE_PATH = /root/cache/stage1`) — caches:
  - Pooled VLM features (`features_<split>_<mode>_n<N>_seed42.pt`) for Stage 1 MLP training
  - Persisted Stage 2 run summaries (`stage2_runs/{train,variantA}_seed*.json`)
- HF auth secret named `huggingface` (created via `modal secret create huggingface HF_TOKEN=...`)

**Gotcha**: if a remote function writes to a Volume, the `@app.function` decorator MUST include that Volume in `volumes={...}`. Missed once on `_stage2_train_remote` (only had `hf_cache`) — fixed to include both. Always double-check.

### Detached runs

For any training job longer than ~10 minutes, use:

    modal run --detach modal_app.py::<entrypoint>

Without `--detach`, the local Modal client maintains a heartbeat with the remote container; if the local process exits (e.g. shell timeout), the remote gets cancelled. With `--detach`, the local entrypoint returns ~30 seconds after submission and the remote keeps running independently.

**Gotcha**: with `--detach`, `result = _xxx_remote.remote(...)` returns `None` because the local exits before the remote returns. So the local entrypoint's `out.write_text(json.dumps(result, indent=2))` saves `"null"`. Workaround: the remote function itself writes its summary to the `stage1-cache` Volume before returning. After detached runs complete, pull them down via:

    modal run modal_app.py::list_stage2_runs

This reads `stage2_runs/*.json` from the Volume and writes each to `results/phase4/`.

**Gotcha (worse)**: if you do `kill <modal-CLI-pid>` to clean up a stuck local process, Modal *will* receive a cancellation signal and kill the remote app too — even though `--detach` is supposed to decouple them. Be careful with `kill`; let detached jobs run their course.

### Submitting many runs in parallel

Submit each `modal run --detach` as its **own** `run_in_background` Bash invocation. They run independently on Modal.

**DO NOT** batch them in a single Bash with a `for` loop — and ESPECIALLY not with `... & ... wait`. CONFIRMED BURN (Jun 2): firing the 6 data-scaling runs as `for n ...; do modal run --detach ... & sleep 10; done; wait` held all 6 detach-client subprocesses alive inside that one Bash task; when the task was reaped, SIGTERM propagated to every client and Modal cancelled ALL the remotes simultaneously (same `kill`-cancels-detached gotcha as above) — at the same timestamp, right before they persisted. Lost ~3h of compute. Even unrelated detached runs fired from other Bash tasks got swept up. One `modal run --detach` per `run_in_background` Bash, full stop.

### Checking status

    modal app list                        # shows apps from last ~24h
    modal app logs <ap-id>                # full log stream for one app
    modal volume ls stage1-cache stage2_runs   # files persisted to Volume
    modal container list                  # currently-running containers

For a quick "is the run done?", grep `modal app logs` for `[s2-train] epoch 2/2` or `[A-train] epoch 2/2` markers, or `persisted result to` (which only prints after eval finishes).

---

## How to do common things

### Run all tests (no GPU needed)

    uv run pytest tests/ -q

### Add a new Modal training entry point

1. Add `@app.function(...)` + `_run_xxx_remote(...)` worker.
2. Add `@app.local_entrypoint()` `xxx(...)` wrapper that calls `_run_xxx_remote.remote(...)`.
3. Persist any artifacts the local entrypoint needs to a Volume (because `--detach` makes the local return immediately).
4. Add a corresponding `list_*` entrypoint if results need to be pulled to local.

### Add a new ablation variant

Pattern: clone `_stage2_train_remote` or `_stage2_variantA_train_remote` and adjust the diff. Keep everything else (data, optimizer, eval) identical so the comparison is matched-compute.

### Pull all completed Stage 2 results

    modal run modal_app.py::list_stage2_runs
    uv run python scripts/p5_ablation_table.py     # aggregates + plots
    uv run python scripts/p5_render_writeup.py     # rewrites PHASE5_HEADLINE.md

### Add a new data mix

`modal_app.py` has `_DATA_MIX_LABELS` in both `_stage2_train_remote` and `_stage2_variantA_train_remote`. Add a new key, e.g. `"taps_swipes_types_hotkey": {...}`. Both training functions check that map.

---

## Current state of results (June 3, 2026)

| Phase | What | Headline | Where |
|---|---|---|---|
| Milestone 3 | Zero-shot Qwen2-VL-2B on Mind2Web | vision delta = 0.000 macro-F1 | `results/milestone3/MILESTONE3.md` |
| Phase 2 | Stage 1 MLP on Mind2Web (2-class) | vision_text 0.605 ± 0.016, +0.009 over text | `results/phase2/PHASE2_RESULTS.md` |
| Phase 3 | Stage 1 MLP on AITW (5-class) | vision_text 0.555 ± 0.036, **+0.414 over text** | `results/phase3/PHASE3_RESULTS.md` |
| Phase 4 | Stage 2 architecture works | hit@0.10 = 0.38 on taps (~12× random) | `results/phase4/PHASE4_RESULTS.md` |
| Phase 5 | A vs D bug hunt | The initial 8σ "A beats D" result was a M-RoPE bug from the `inputs_embeds` injection path. D-hook fixes it by preserving `input_ids`; tap/swipe becomes a tie, all_with_coords becomes a real positive signal. | `results/phase4/PHASE5_CORRECTED.md` |
| Phase 6 | Full ablation + e2e | **Broad thesis supported, specific embedding claim refuted.** all_with_coords: B≈D-hook > D-token≳C>A; B and D-hook beat A by paired bootstrap. taps_and_swipes control: no conditioned mechanism clearly beats A. E2E predicted types beat flat A with ~0.02 oracle gap. | `results/phase4/PHASE6_FINAL.md` |
| Phase 7 | Mechanism + low-data strengthening | Low-data matrix complete for A/B/D-hook at n_train ∈ {300,500,800}, seeds 42/43/44; strict audit passes. D-hook is the most stable low-data mechanism. D-token causal-use test shows the learned embedding is used (gold − wrong +0.192 hit@0.10; gold − zero +0.093) but still not the winning mechanism. | `results/phase4/PHASE7_RESULTS.md`, `results/phase4/phase7_result_audit.json`, `results/phase4/causal_use_summary.json` |
| Diagnostics | Attention viz + D-token + Dfrozen | Attention heatmaps exist under `results/phase4/attn_viz/`; D-token is M-RoPE-correct and still does not beat B/D-hook on headline grounding; Dfrozen confirmed the original slot path was the problem. | `results/phase4/attn_viz/`, `results/phase4/Dtoken_*`, `results/phase4/Dfrozen_*` |

Cumulative Modal spend: **~$52 of $200**.

Paper framing to use now:
- **Supported:** action-type supervision improves grounding when action type is spatially predictive (`all_with_coords`).
- **Refuted:** the literal learned prepended action embedding is not the winning mechanism; auxiliary loss B captures the benefit more simply.
- **Mechanism:** D-token is causally used at inference (wrong/zero action embeddings hurt), so the embedding-path refutation is "used but not best", not "ignored."
- **Mechanistic caution:** naive `inputs_embeds` injection bypasses Qwen2-VL's M-RoPE position computation and can create a false negative.
- **Control:** when action type is not spatially informative (`taps_and_swipes`), conditioning is neutral to mildly harmful.

---

## Gotchas worth knowing about

### AITW image decoding

Screenshots in `cjfcsjt/AITW_General` are raw RGB pixel bytes, no header. Common shapes observed in the data:
- 540×1080 (Pixel 3 half-res): 1,749,600 bytes
- 540×1140 (Pixel 4 half-res): 1,846,800 bytes
- 540×1170 (Pixel 5 half-res): 1,895,400 bytes
- 720×1520, 720×1440, 412×732, 270×600 — others

`src/data/aitw._decode_aitw_image_bytes` handles all of these. Don't use `PIL.Image.open(BytesIO(bytes))` — it'll raise `UnidentifiedImageError`.

### Qwen2-VL config in transformers v5

`config.hidden_size` is gone — it lives at `config.text_config.hidden_size` now. `src/models/base.py` and `src/models/stage2_grounding.py` probe both paths.

### Qwen2-VL M-RoPE and embedding injection

Do **not** feed Stage 2 via `inputs_embeds` unless you have explicitly preserved
Qwen2-VL's multimodal position-id path. The original D-slot replacement bypassed
`input_ids`-keyed M-RoPE and produced the false 8σ negative result. The final
working variants in `modal_app.py` keep the normal `input_ids` path and inject
conditioning through embedding-layer hooks.

### Stage 2 slot replacement

`Stage2ConditionedGrounding._embed_with_action` requires EXACTLY ONE `<|action_slot|>` token per row. That class is now mainly historical/smoke-test code; for paper results, use the Modal variants. If you change any D-token prompt template, verify the slot is still present once and that generation still uses the full `input_ids` path.

### Training instability

Val macro-F1 / hit@r swings ~15-20 points between epochs on small data (n_val ~100-200). Best-epoch checkpointing in Stage 1 hides this; Stage 2 just reports last-epoch. For multi-seed reporting, the std across seeds is the right uncertainty number, not the single-seed epoch curve.

### Modal cost ballparks (L4)

- Small Modal job (load model + ~50 examples eval): ~$0.05
- Cache features for Multimodal-Mind2Web (one mode, one split, 5000 ex): ~$0.30–$0.90 depending on resolution
- Stage 2 training (n_train=500, 2 epochs, taps only): ~$0.30
- Stage 2 training (n_train=1000, 2 epochs, taps+swipes): ~$0.50

---

## What "lock in" looks like

The user wants forward progress, not waiting. When background jobs are running:
- Don't just schedule a wakeup and sit. Actively poll Modal logs to confirm jobs are progressing as expected.
- Use the time to write the next analysis script, the next writeup, or the next ablation variant.
- Commit incremental work; the user reviews via git push.
- When status changes, surface it to the user with concrete numbers, not vague "still running".

Polling pattern that works (`bash`):

    until uv run modal volume ls stage1-cache stage2_runs 2>&1 | grep -q "<expected-filename>"; do sleep 30; done

Drop this into a `run_in_background: true` Bash and the harness notifies when it finishes.

---

## When you finish a phase

1. Drop a `PHASE<n>_RESULTS.md` in the matching `results/phase<n>/` dir with: setup, table of numbers, analysis, limitations, cost, reproduction commands, file index.
2. Update `README.md` Status block with the headline number and link.
3. Update this file (`CLAUDE.md`) if you discovered a new gotcha or changed a pattern.
4. Commit + push. No Co-Authored-By trailer.
