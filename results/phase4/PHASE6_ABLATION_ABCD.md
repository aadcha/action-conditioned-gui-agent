# Phase 6 — full A/B/C/D ablation (the table the paper hinges on)

All four variants on **AITW all_with_coords** (action types click/scroll/type,
the setting where action type is spatially informative), Qwen2-VL-2B + LoRA,
n_train=1200, 2 epochs, 3 seeds. Comparison via **paired bootstrap** over
pooled per-example normalized-L2 distances (3 seeds × 250 examples = 750 units;
identical deterministic val set across all runs). 10k resamples.

## The four variants

- **A — Flat baseline.** Plain prompt → coordinate. No action-type info.
- **B — Auxiliary loss.** Flat model + a jointly-trained action-type
  classification head: `loss = LM_coord + 1.0·CE(head(pooled_hidden), gold_type)`.
  Action-type signal present only during training; inference identical to A.
- **C — Hard routing.** Model trained to emit `{action_word} (x,y)`; at eval the
  gold action word is **forced** as the decode prefix, then it grounds.
- **D — Type embedding (ours), "D-hook".** Learned `nn.Embedding(8,d)` added to
  the instruction stream via an embedding-layer hook, zero-init, full input_ids
  path (M-RoPE intact). Conditions grounding on the gold action type.

All conditioned variants (B, C, D) use the **gold** action type, so the
comparison isolates the conditioning *mechanism*, not Stage-1 classifier error.

## Means

| variant | hit@0.05 | hit@0.10 | hit@0.25 | mean norm L2 (↓) |
|---|---|---|---|---|
| A (flat) | 0.157 | 0.255 | 0.515 | 0.392 |
| **B (aux loss)** | **0.195** | **0.305** | **0.585** | **0.362** |
| C (hard routing) | 0.180 | 0.279 | 0.543 | 0.412 |
| **D-hook (ours)** | **0.195** | 0.300 | 0.569 | 0.365 |

## Paired-bootstrap comparisons (the rubric)

| comparison | hit@0.05 | hit@0.10 | hit@0.25 | mean L2 | verdict |
|---|---|---|---|---|---|
| **D − A** | +0.037*** | +0.045*** | +0.055*** | −0.027*** | D beats A on all (p<0.002) |
| **B − A** | +0.037*** | +0.051*** | +0.071*** | −0.030*** | B beats A on all (p<0.001) |
| **C − A** | +0.023** | +0.024 (ns) | +0.028 (ns) | +0.020** (worse) | C barely helps; hurts L2 |
| **D − B** | 0.000 | −0.005 (ns) | −0.016 (ns) | +0.003 (ns) | **tie** (all p>0.27) |
| **D − C** | +0.015 (ns) | +0.021 (ns) | +0.027 (ns) | −0.046*** | D > C on precision only |

`***` p<0.01, `**` p<0.05, `ns` 95% CI brackets 0. Raw:
`results/phase4/ablation_ABCD_all_with_coords.json`.

## Interpretation (matches a pre-registered rubric outcome)

The project plan's rubric: *"Beat A but lose to B → signal-during-training was
enough; inference-time conditioning doesn't matter."* We land almost exactly
there, with B and D tied:

1. **Action-type supervision helps grounding** when action type is spatially
   informative. Both B and D significantly beat the flat baseline A on every
   metric (≈ +0.05 hit@0.10, −0.03 mean L2, all p<0.002).
2. **The conditioning mechanism does not matter much.** D (learned input
   embedding) and B (auxiliary training loss) are **statistically
   indistinguishable** (all comparisons p>0.27). Getting the action-type
   signal in *during training* is sufficient; the inference-time latent
   conditioning of D adds nothing measurable over it.
3. **Hard routing (C) is the weakest** conditioned variant — only marginally
   better than A on the tightest threshold and actually *worse* on average
   precision (forcing the action word as a decode prefix perturbs the
   coordinate decode).
4. Ranking: **B ≈ D > C > A** (on this setting).

### Honest takeaway for the writeup

This is *not* "our embedding method (D) is uniquely best." It is the more
nuanced and defensible finding the controlled ablation was designed to surface:

> Action-type information measurably improves screenshot grounding precisely
> where the action type is spatially predictive, but a simple auxiliary
> classification loss captures essentially all of that benefit — the learned
> action-type embedding and hard routing do not beat it. The contribution is
> the *controlled demonstration* (with a clean M-RoPE-bug post-mortem and a
> spatially-informative-vs-not contrast) of when and how much action-type
> conditioning helps, not a claim that one mechanism dominates.

## Control: taps_and_swipes

On the spatially-uninformative mix (tap vs swipe — both generic touch points),
D-hook ties A (see `PHASE5_CORRECTED.md`). We have not yet run B/C there, but
the prediction is that all conditioned variants tie A, because there is no
action-type→location signal to exploit. (Cheap follow-up if needed.)

## Reproduction

```bash
for v in B C; do for s in 42 43 44; do
  modal run --detach modal_app.py::train_stage2_variant$v --seed $s --data-mix all_with_coords
done; done
modal run modal_app.py::list_stage2_runs
uv run python scripts/p6_ablation_table.py --mix all_with_coords
```

Modal spend through Phase 6: **~$31 of $200**.
