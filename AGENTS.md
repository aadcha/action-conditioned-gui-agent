# Project notes for the agent

Operational guide for future sessions working on this codebase. Skim this first.

---

## Hard rules

1. **NEVER add `Co-Authored-By: Codex` or any "Generated with Codex" footer** to commit messages, code comments, docs, or slides. The user is the sole author. This rule is non-negotiable — historical commits were rewritten once to scrub Codex attribution; do not reintroduce it.
2. **Do not change long-term project direction without asking.** The user wrote the multi-phase plan; execute it as written. Tactical implementation choices inside a phase are yours to make; scope cuts, deadline reinterpretations, or phase reorderings are not.
3. **Do not declare scheduling/timeline issues** unless asked. The user knows their own deadlines.
4. **Force-push to main requires explicit user authorization.** "Yes force push" or equivalent. The auto-mode classifier blocks otherwise.

---

## Project one-liner

CS 231N, Spring 2026. Goal: GUI agent that decouples *what action* (Stage 1: action-type classifier) from *where to act* (Stage 2: action-type-conditioned grounding). Backbone Qwen2-VL-2B + LoRA. The central architectural claim is that a learned action-type embedding prepended to the instruction stream is better than flat decoding.

Source of truth for the spec: `reference/PROJECT_OVERVIEW.md` + `reference/roadmap.md`. The roadmap is prescriptive — most "what should I build next" questions are answered there.

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
    stage2_grounding.py    Stage 2: tokenizer adds <|action_slot|>, then the
                           slot's embedding is REPLACED at forward time by
                           a learned action embedding. LoRA + the embedding
                           table are trainable; rest frozen.
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
modal_app.py               ALL Modal entry points. See "Modal patterns" below.
configs/                   YAML configs for smoke tests
tests/                     22 tests; runs in seconds on CPU
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

Don't use a shell `for` loop in a single Bash invocation — the loop blocks if any `modal run` hangs, and you'll end up with only the first iteration registered. Instead, submit each `modal run --detach` as its own background Bash invocation. They run independently on Modal.

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

## Current state of results (May 31, 2026)

| Phase | What | Headline | Where |
|---|---|---|---|
| Milestone 3 | Zero-shot Qwen2-VL-2B on Mind2Web | vision delta = 0.000 macro-F1 | `results/milestone3/MILESTONE3.md` |
| Phase 2 | Stage 1 MLP on Mind2Web (2-class) | vision_text 0.605 ± 0.016, +0.009 over text | `results/phase2/PHASE2_RESULTS.md` |
| Phase 3 | Stage 1 MLP on AITW (5-class) | vision_text 0.555 ± 0.036, **+0.414 over text** | `results/phase3/PHASE3_RESULTS.md` |
| Phase 4 | Stage 2 architecture works | hit@0.10 = 0.38 on taps (~12× random) | `results/phase4/PHASE4_RESULTS.md` |
| Phase 5.1 first cut | A vs D on taps-only | A 0.363 vs D 0.320 hit@0.10 — within noise; mechanism understood (no action-type variance to learn from) | `results/phase4/ABLATION_v_A_vs_D.md` |
| Phase 5.2 | A vs D on taps+swipes (n=1000, 3 seeds each) | **A 0.390 ± 0.010 vs D 0.288 ± 0.008 hit@0.10 — A wins by 8σ**. NEGATIVE RESULT for the project hypothesis | `results/phase4/NEGATIVE_RESULT.md` |
| Phase 5.3 | Dfrozen diagnostic (D with action_embeddings frozen at random init) | IN FLIGHT | — |
| Phase 5.4 | all_with_coords (3 active classes) smoke | IN FLIGHT | — |

Cumulative Modal spend: **~$8 of $200**.

The Phase 5.2 8σ negative result is the BIGGEST finding of the project so far. The user's plan explicitly anticipates `D ≤ A` as a possible outcome and prescribes "paper becomes a negative result + diagnostics" — `NEGATIVE_RESULT.md` outlines the paper framing.

The diagnostic experiments in flight (Dfrozen) decompose the negative result:
- Dfrozen ≈ D → slot disrupts the prompt regardless of training
- Dfrozen ≪ D → embedding *was* helping; D was bottlenecked by training time
- Dfrozen ≈ A → embedding training actively harmed (unlikely)

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

### Stage 2 slot replacement

`Stage2ConditionedGrounding._embed_with_action` requires EXACTLY ONE `<|action_slot|>` token per row. The chat template + `make_stage2_prompt` guarantees this. If you change the prompt template, verify the slot is still present once.

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
3. Update this file (`AGENTS.md`) if you discovered a new gotcha or changed a pattern.
4. Commit + push. No Co-Authored-By trailer.
