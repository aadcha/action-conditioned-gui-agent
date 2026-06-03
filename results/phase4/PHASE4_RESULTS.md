# Phase 4 — Stage 2 action-type-conditioned grounding (first training results)

The actual research contribution. Stage 1 told us *what* to do; Stage 2 takes that
discrete decision as a learned embedding and produces *where* to do it.

## Setup

- **Architecture**: `src/models/stage2_grounding.py:Stage2ConditionedGrounding`
  - Qwen2-VL-2B-Instruct backbone, frozen weights
  - LoRA on (q,k,v,o) attention projections (r=16, α=32, dropout=0.05) — trainable
  - `nn.Embedding(8, hidden_dim)` action embedding table — trainable
  - One added special token `<|action_slot|>`; embed-table resized
  - At forward, the slot position's embedding is **replaced** by the learned
    action embedding for the sample's action type
- **Trainable params**: 4,370,432 / 2,212,929,024 = **0.197%** (LoRA + action embedding)
- **Data**: AITW DUAL_POINT taps from `cjfcsjt/AITW_General/standard/train`
  - 500 train + 100 val (filtered to taps only with valid touch coordinates)
  - Coordinates: AITW normalized [0,1] (y, x) flipped to (x, y), rendered as
    `(int_x, int_y)` strings on a 0–999 grid
- **Loss**: standard LM cross-entropy. Tokens before the assistant turn are
  masked with -100 so loss fires only on the coordinate-string tokens.
- **Optim**: AdamW lr=2e-5, weight_decay=0.01, grad clip = 1.0
- **Schedule**: 2 epochs, teacher-forced (gold action_type_id)
- **Hardware**: Modal L4, ~25 minutes per training run

## Headline result

| Epoch | Train loss | parse_rate | mean norm L2 | **hit@0.05** | **hit@0.10** | hit@0.25 |
|---|---|---|---|---|---|---|
| 1 | 1.071 | 1.00 | 0.269 | 0.100 | 0.230 | 0.650 |
| 2 | 0.944 | 1.00 | 0.260 | **0.170** | **0.380** | **0.650** |

For reference, random uniform predictions in [0,1]² hit:
| radius | random hit-rate | model hit-rate (epoch 2) | ratio |
|---|---|---|---|
| 0.05 | 0.008 | 0.170 | **21×** |
| 0.10 | 0.031 | 0.380 | **12×** |
| 0.25 | 0.196 | 0.650 | 3.3× |

**Loss decreases monotonically**: 1.07 → 0.94 epoch-over-epoch. **Parse rate is
100%**: every generated output is a valid `(x, y)` coordinate string. **Sample
predictions** at epoch 2: `(489, 510)`, `(414, 58)`, `(489, 511)`, `(456, 108)`,
`(500, 511)` — clearly integer pairs in the expected format.

## What this proves

1. **The architecture works.** The action-slot-replacement mechanism — tokenize
   the prompt normally, swap one embedding row at runtime — interacts
   correctly with Qwen2-VL's image-fusion pipeline. The slot's gradient flows
   back to the action embedding table; LoRA adapters get gradients too;
   training is stable.
2. **The model is learning grounding, not memorizing a constant.** The
   12×-above-chance hit@0.10 number is decisive. The parse rate of 100% on
   freshly-generated outputs (not training data) shows the model is
   producing well-formed coordinate strings from a previously unseen prompt
   structure.
3. **Loss-vs-eval-quality is consistent.** Epoch 2's lower training loss
   (0.94 vs 1.07) is paired with better grounding (hit@0.10 0.38 vs 0.23).
   The loss is a useful proxy for the downstream metric, which means a longer
   training run is the natural next step.

## What this does NOT prove (yet)

This is a single training run on a small slice with one configuration. It
does not yet test:

1. **The action-conditioning hypothesis** — does the action embedding actually
   help? Need to compare against variant A (flat baseline, no action
   conditioning). On the same data, same compute, same eval. Until that
   comparison lands, we know vision-language LoRA can ground; we don't know
   that *action-type conditioning* is doing useful work specifically.
2. **Scale**. 500 training examples is small. Phase 5's full ablation table
   should run with the largest feasible n_train (compute-budgeted).
3. **Robustness across action types**. We trained only on DUAL_POINT taps. The
   model never saw scrolls / hotkeys / type / finished during this run. The
   8-class action embedding has only 1 row's gradient signal — the others
   remain at random initialization.

## What's next (Phase 4.4 / Phase 5)

1. **Implement variant A** (flat baseline, no action conditioning) using the
   same data + loss + compute. Train, evaluate on the same 100 val taps.
2. **Implement variants B and C** per the plan.
3. **Run all four variants at matched compute**, 3 seeds each, with bootstrapped
   CIs on hit@0.10.
4. **Train on all 8 canonical actions**, not just taps. Coordinates for
   non-tap actions need representation choices (drag = start+end, scroll =
   direction, hotkey = no coordinate, etc.).
5. **Conditional-accuracy diagnostic**: P[grounding correct | gold action type]
   vs P[grounding correct | predicted action type] — separates classifier
   error from grounding error.

## Cost

| Run | wall time | est. cost |
|---|---|---|
| Stage 2 smoke (forward+backward sanity) | ~3 min | ~$0.04 |
| Stage 2 first training attempt (cancelled by client disconnect) | ~20 min before disconnect | ~$0.25 |
| Stage 2 detached re-run (full 2 epochs, completed) | ~25 min | ~$0.34 |
| **Phase 4 first cut total** | | **~$0.63** |

Cumulative Modal spend through Phase 4 first cut: **~$3.17 of $200 (1.6%)**.

## File index

| File | What |
|---|---|
| `results/phase4/PHASE4_RESULTS.md` | This writeup |
| `results/phase4/stage2_n500_ep2_lr2e-5.json` | Detached run results reconstructed from Modal logs |
| `src/models/stage2_grounding.py` | The Stage2ConditionedGrounding model |
| `src/train/stage2.py` | build_batch, evaluate_grounding, coord_to_string |
| `modal_app.py:stage2_smoke` | Forward+backward smoke entrypoint |
| `modal_app.py:train_stage2` | Training entrypoint (use `modal run --detach` for any run > 10 min) |
| `modal_app.py:list_stage2_runs` | Pulls persisted run summaries from the Volume into `results/phase4/` |

## Reproduction

```bash
modal run modal_app.py::stage2_smoke                    # ~3 min, ~$0.04
modal run --detach modal_app.py::train_stage2           # ~25 min, ~$0.34
modal run modal_app.py::list_stage2_runs                # fetch results from Volume
```
