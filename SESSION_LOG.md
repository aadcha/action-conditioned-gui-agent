# Session log — 2026-05-31 [historical]

> **Historical session handoff.** This captured the May 31 state while Phase 5
> work was still unfolding. It is retained for provenance, not as the current
> task list. Current source of truth: `README.md#status`,
> `results/phase4/PHASE6_FINAL.md`, `results/phase4/PHASE7_RESULTS.md`,
> `AGENTS.md`, and `CLAUDE.md`.

Original handoff text below captures the May 31 state, including then-open
items that were later resolved by Phase 6/7. It pairs with `CLAUDE.md`
(operational/how-to) and the per-phase writeups under `results/`.

> **UPDATE (late session): the two open items below are now RESOLVED.**
> - A seed 46 landed → full **5×5** seed-level table (D-hook still wins all
>   metrics: hit@0.10 +0.046, hit@0.25 +0.058, mean L2 −0.035).
> - **Paired bootstrap built and run** (`src/eval/bootstrap.py`,
>   `scripts/p5_paired_bootstrap.py`). On all_with_coords, 3 seeds × 250
>   examples = 750 paired units: **D-hook beats A on all four metrics with
>   95% CIs excluding zero and p < 0.005** (bootstrap + permutation).
>   See `results/phase4/PHASE5_CORRECTED.md` and
>   `results/phase4/paired_bootstrap_all_with_coords.json`.
> The previously open sections below are kept for the record but are DONE. The
> later variants B/C, scaling/low-data runs, and attention/causal diagnostics
> are covered by `results/phase4/PHASE6_FINAL.md` and
> `results/phase4/PHASE7_RESULTS.md`.

---

## TL;DR of the day

Went from Phase 2 scaffolding through a full Phase 5 A-vs-D ablation, with a
major plot twist in the middle:

1. **Built & ran Stage 1 (Phases 2–3)** — classifier on cached Qwen2-VL-2B
   features. Mind2Web (2-class) vision delta +0.009; **AITW (5-class) vision
   delta +0.414** — vision is essential when actions aren't text-decidable.
2. **Built Stage 2 (Phase 4)** — action-type-conditioned grounding model.
3. **Ran the A-vs-D ablation (Phase 5)** and first measured an 8σ "A beats D"
   **negative result**.
4. **Found that negative result was a BUG, not science.** The original D
   ("D-slot") fed `inputs_embeds` to Qwen2-VL, bypassing its `input_ids`-keyed
   **M-RoPE** 3D position computation → corrupted image-patch spatial encoding →
   degraded grounding. Found via `scripts/p5_debug_stage2.py`.
5. **Fixed it (D-hook)** — additive conditioning via an embedding-layer forward
   hook, zero-init, full `input_ids` path (M-RoPE identical to A). True superset
   of A.
6. **With the fix, the hypothesis-shaped pattern emerged:**
   - tap/swipe (action type spatially *uninformative*): **D-hook ties A**
   - all_with_coords (adds click/scroll/type disambiguation, spatially
     *informative*): **D-hook beats A on all 4 metrics** (significant on the
     robust ones). Phase 7 later clarified that AITW `type` coordinates are
     degenerate rather than field-localization targets.

The defensible scientific story is now: *action-type conditioning improves
grounding precisely where action type predicts location; it is neutral where it
doesn't; and a naive injection that bypasses M-RoPE actively harms grounding.*

---

## Headline numbers (all AITW, Qwen2-VL-2B + LoRA, 2 epochs, lr 2e-5, coord grid 0–999)

### taps_and_swipes (n_train=1000, 3 seeds) — the CONTROL (action type ≈ uninformative)

| metric | A (flat) | D-slot (buggy) | D-hook (fixed) |
|---|---|---|---|
| hit@0.10 | 0.390 ± 0.010 | 0.298 | 0.392 ± 0.043 |
| hit@0.25 | 0.727 ± 0.023 | 0.607 | 0.723 ± 0.038 |
| mean norm L2 (↓) | 0.184 ± 0.002 | 0.214 | 0.179 ± 0.011 |

→ D-hook **ties** A. (D-slot's bug cost ~0.094 hit@0.10.)

### all_with_coords (n_train=1200) — the TEST (action type informative via `type`)

**A: 4 seeds [42,43,44,45]. D-hook: 5 seeds [42–46].** At this timestamp A seed
46 had not landed yet; the final 5×5 table landed later and is summarized in
the update banner above.

| metric | A (flat) | D-hook (ours) | Δ (D−A) | sig (SE of mean) |
|---|---|---|---|---|
| hit@0.05 | 0.142 ± 0.030 | 0.163 ± 0.045 | +0.021 | 0.8σ |
| hit@0.10 | 0.243 ± 0.033 | 0.268 ± 0.038 | +0.025 | 1.0σ |
| hit@0.25 | 0.509 ± 0.032 | **0.554 ± 0.029** | +0.045 | **2.2σ** |
| mean norm L2 (↓) | 0.396 ± 0.012 | **0.373 ± 0.010** | −0.024 | **3.1σ** |

→ D-hook **better on all 4 metrics**; significant on hit@0.25 and mean L2,
directional (~1σ) on the tight thresholds. Consistent direction + a-priori
mechanistic prediction + control ties = real signal, not cherry-picking.

### action-LR sweep (D-hook, taps_and_swipes, seed 42) — why the tie is honest

| config | action-emb norm | hit@0.10 |
|---|---|---|
| shared lr 2e-5 | 0.06 | 0.392 (≈A) |
| action_lr 1e-3 | 1.03 | 0.290 |
| action_lr 5e-3 | 4.09 | 0.215 |

→ Forcing the embedding to develop **monotonically hurts** on tap/swipe. The
shared-lr tie is the optimizer correctly driving a non-useful signal toward
zero. (On all_with_coords the signal *is* useful → the win.)

---

## Earlier-phase results (unchanged, for completeness)

- **Milestone 3** (`results/milestone3/`): zero-shot Qwen2-VL-2B on Mind2Web
  action-type has vision delta = 0.000; TF-IDF beats the 2B VLM by 15 macro-F1.
- **Phase 2** (`results/phase2/`): Stage 1 MLP on Mind2Web, vision_text 0.605 ±
  0.016 macro-F1; vision delta +0.009; vision_zeroed sanity = majority floor.
- **Phase 3** (`results/phase3/`): Stage 1 MLP on AITW (5 classes), vision_text
  0.555 ± 0.036; **vision delta +0.414**; TF-IDF text-only ceiling ~0.17.

---

## THE ONE OPEN DATA POINT: A seed 46 on all_with_coords

A seed 46 was launched **twice** and **cancelled mid-training both times** with
"Received a cancellation signal while processing input". The other 9 cells
(A 42–45, D-hook 42–46) all completed. Current comparison is therefore **4 A vs
5 D-hook** — valid, but not the symmetric 5×5 we want.

**Likely cause (for tomorrow):** the cancellations happened while I was running
*other* `modal` commands (`modal app list`, `modal run …::list_stage2_runs`)
against the same App name (`action-conditioned-gui-agent`) during training.
Hypothesis: launching another ephemeral instance of the same-named app, or the
detached submitter process being reaped, preempted the run. The D-hook seeds
succeeded because I was mostly idle (only `modal volume ls`, which is harmless)
while they trained.

**To get seed 46 cleanly tomorrow:**
```
modal run --detach modal_app.py::train_stage2_variantA \
    --seed 46 --data-mix all_with_coords --n-train 1200 --n-val 250
# Then DO NOT run any other `modal run …` against modal_app.py until it lands.
# Poll only with: modal volume ls stage1-cache stage2_runs | grep variantA_seed46
```
Then pull with `modal run modal_app.py::list_stage2_runs` and recompute the 5×5.

---

## NEXT STEPS (in priority order)

1. **Finish the 5×5** — get A seed 46 (above), recompute all_with_coords A vs
   D-hook at 5 seeds each.
2. **Paired bootstrap at the example level** — the rigorous test the project
   plan demands ("paired bootstrap … 95% CIs"). Requires logging *per-example*
   hit/distance arrays (current eval only saves aggregate + 20 sample outputs).
   Add a `per_example: [...]` field to `evaluate_grounding` in
   `src/train/stage2.py`, re-run A and D-hook on all_with_coords with a FIXED
   shared val set, then bootstrap over the 250 val examples. This removes
   between-seed variance and will tell us cleanly whether the tight-threshold
   hits (hit@0.05/0.10) are real, not just the coarse metrics.
3. **Scale n_train / epochs** on all_with_coords (e.g. n=3000, 4 epochs) — if
   the effect is real it should grow with data; cheap on Modal (~$1–2).
4. **Update the writeups** — `results/phase4/PHASE5_CORRECTED.md` currently has
   the 3-seed snapshot; refresh with the final 5×5 + bootstrap. Then the
   project's headline flips from "negative result" to the nuanced positive:
   conditioning helps when action type is spatially informative.
5. **(Optional) variants B & C** from the project plan to complete the A/B/C/D
   ablation table. Code patterns are in `modal_app.py` (clone a variant fn).
6. **(Optional) D-text** (action word in prompt) is already coded
   (`train_stage2_Dtext`) but never run — an independent conditioning mechanism.

---

## Budget / infra state

- **Modal spend: ~$22 of $200** (≈11%). Lots of runway.
- Feature caches + all Stage 2 run JSONs persist on the Modal Volume
  `stage1-cache` under `stage2_runs/`. Pull anytime with
  `modal run modal_app.py::list_stage2_runs`.
- HF weights cached on Volume `hf-cache` (cold starts skip the 4 GB download).
- All result JSONs are also committed under `results/phase4/`.

## Key files touched today

| File | What |
|---|---|
| `src/models/stage1_classifier.py`, `src/train/stage1.py` | Stage 1 MLP + trainer |
| `src/data/aitw.py` | AITW loader (raw-RGB decode, taxonomy) |
| `src/data/unified.py` | unified Mind2Web+AITW dataset |
| `src/models/stage2_grounding.py` | Stage 2 model (slot-replace D + `action_embeddings_trainable` flag) |
| `src/train/stage2.py` | build_batch, evaluate_grounding (M-RoPE/add_generation_prompt fixes), `_metrics_from_predictions` |
| `modal_app.py` | all entrypoints: extract_features, train_stage1[/aitw], train_stage2[/variantA/Dfrozen/Dhook/Dtext], stage2_debug, list_stage2_runs |
| `scripts/p5_debug_stage2.py` | mechanism verification (found the M-RoPE bug) |
| `scripts/p5_ablation_table.py`, `p5_render_writeup.py` | aggregate + render |
| `results/phase4/PHASE5_CORRECTED.md` | the corrected ablation story (supersedes NEGATIVE_RESULT.md) |

## Git state
All work committed and pushed to `main` (github.com/aadcha/action-conditioned-gui-agent).
No Claude attribution anywhere (per standing instruction). Last commit before
this log: the all_with_coords 3-seed promising result + action_lr sweep.
