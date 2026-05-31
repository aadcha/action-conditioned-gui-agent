# Phase 5.1 first-cut ablation — variant A vs variant D (taps only)

Matched-compute comparison: variant D (action-type-conditioned, the project's
contribution) vs variant A (flat baseline, no conditioning). Same data, same
LoRA config, same training schedule, same eval, same 3 seeds.

## Setup (both variants)

- Backbone: Qwen2-VL-2B-Instruct, frozen
- LoRA: r=16, α=32, dropout=0.05, target=(q,k,v,o)
- Data: 500 AITW `DUAL_POINT` taps from train, 100 from val
- Optim: AdamW lr=2e-5, wd=0.01, grad clip=1.0, batch_size=1
- Schedule: 2 epochs, teacher-forced
- Coordinate format: `(int_x, int_y)` on a [0, 999] grid
- Eval metric: hit@r (fraction of predictions within Euclidean radius r in
  normalized [0,1]² space), mean normalized L2, parse rate
- Seeds: 42, 43, 44

Only difference between variants:

- **A — flat baseline.** Prompt: `Goal: {goal}\nPredict the action coordinate.`
  No action embedding. Standard LoRA fine-tune.
- **D — action-conditioned (ours).** Prompt: `<|action_slot|> {goal}\nPredict the action coordinate.`
  `<|action_slot|>` is a new special token. At forward time, its embedding is
  REPLACED by `action_embeddings(action_type_id)` from a trainable
  `nn.Embedding(8, 1536)` table. The embedding table is trainable along with
  the LoRA adapters; no other base-model parameters change.

## Results

### Per-run epoch-2 numbers

| variant | seed | train_loss | hit@0.05 | **hit@0.10** | hit@0.25 | norm L2 |
|---|---|---|---|---|---|---|
| A | 42 | 0.920 | 0.160 | 0.320 | 0.650 | 0.249 |
| A | 43 | 0.918 | 0.180 | 0.400 | 0.690 | 0.229 |
| A | 44 | 0.917 | 0.160 | 0.370 | 0.660 | 0.231 |
| D | 42 | 0.944 | — | 0.350 | — | 0.271 |
| D | 43 | 0.947 | — | 0.330 | — | 0.261 |
| D | 44 | 0.939 | — | 0.280 | — | 0.279 |

### Aggregates (3 seeds, mean ± sample stdev)

| variant | hit@0.10 | mean norm L2 |
|---|---|---|
| A | **0.363 ± 0.040** | 0.236 ± 0.011 |
| D | **0.320 ± 0.036** | 0.270 ± 0.009 |

**D – A = –0.043** hit@0.10. Combined std ≈ 0.054 — the difference is
**well within one combined standard deviation**. **Treat as null.**

A single random uniform predictor in [0,1]² has hit@0.10 ≈ 0.031, so both
variants are well above chance (≈10–12× chance). The comparison is between
two competent models, not between a model and noise.

## Why D doesn't beat A here

The taps-only training set has every example's `canonical_action_id == "click"`.
That means:

1. The action embedding row for `click` is the only row that receives gradient
   from any training step. The other 7 rows stay at their N(0, 0.02)
   initialization for the entire run.
2. At every training and eval step, the model sees the SAME embedding vector
   at the slot position. There's no variance for the model to exploit — it's
   effectively a constant prefix token.
3. A constant prefix token is roughly equivalent to adjusting LoRA's bias.
   The model can learn the equivalent of "always click" through LoRA alone.

**This is the expected outcome for a degenerate-action-type training set.**
Variant D's mechanism — discrete action conditioning — only matters when the
training distribution covers multiple action types. The taps-only experiment
was a sound first sanity check of the *architecture* (the gradient flows, the
parse rate is 100%, the model learns grounding) but it cannot test the
*hypothesis* (that conditioning helps).

## What the result does tell us

- The Stage 2 pipeline (slot replacement, marker-token approach, LoRA + action
  embedding co-training, coordinate-string LM head) works end-to-end on
  Modal L4.
- Loss decreases monotonically across epochs for both variants
  (~1.04 → ~0.92 for A, ~1.07 → ~0.94 for D).
- Parse rate is 100% on freshly-generated outputs for both variants.
- Single-seed swings within a variant (e.g. A swings 0.32 → 0.40 → 0.37
  across seeds 42 → 43 → 44) are the dominant source of variance at this
  scale; n=100 val is small enough that 8 prediction flips move hit@0.10 by
  0.08.

## The right next experiment

The project's hypothesis (action conditioning improves grounding) needs
training data with multiple canonical action types. Concretely:

1. **Replace taps-only with all DUAL_POINT actions** (taps + 4 swipe
   directions) plus typing actions. This populates 3–4 of the 8 canonical
   classes during training and forces the action embedding table to encode
   class-conditional information.
2. **Per-action-type label format**:
   - tap → `tap (x, y)`
   - swipe → `swipe (start_x, start_y) → (end_x, end_y)`
   - type → `type (x, y) "{text}"`
3. **Eval per action type** so we can measure conditional accuracy:
   P[grounding correct | gold action type] vs P[grounding correct | predicted].
4. **Re-run variants A and D** under the multi-class setup. With variance in
   the action_type_id stream, variant D's conditioning should now have
   measurable work to do.

Estimated cost: ~$2 for 3 seeds × 2 variants on Modal L4.

## Cost spent

| Run | wall time | est. cost |
|---|---|---|
| Variant A × 3 seeds (detached) | ~75 min total (3 parallel L4) | ~$1.00 |
| Variant D × 3 seeds (detached, with Volume fix) | ~75 min total | ~$1.00 |
| (Plus earlier Phase 4 single-seed runs, partial runs) | | ~$0.70 |
| **This ablation block total** | | **~$2.70** |

Cumulative Modal spend: **~$4.20 of $200 (2.1%)**.

## File index

| File | What |
|---|---|
| `results/phase4/variantA_seed{42,43,44}_n500_ep2_lr2e-05.json` | Variant A per-seed |
| `results/phase4/train_seed{42,43,44}_n500_ep2_lr2e-05.json` | Variant D per-seed (Volume-persisted retry) |
| `results/phase4/stage2_n500_ep2_lr2e-5.json` | Earlier variant D seed-42 reconstructed from logs (slightly different number: 0.380; current re-run: 0.350) |

## Reproducibility

```bash
# Three seeds, each variant, detached (~25 min × parallel on Modal):
for s in 42 43 44; do
  modal run --detach modal_app.py::train_stage2 --seed $s
  modal run --detach modal_app.py::train_stage2_variantA --seed $s
done

# When all complete, pull from the Volume:
modal run modal_app.py::list_stage2_runs
```
