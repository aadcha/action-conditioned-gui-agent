# Phase 5 — Ablation headline (Variant A vs Variant D) [superseded]

> This auto-rendered snapshot predates the M-RoPE fix and the final Phase 6
> ablation. Use `results/phase4/PHASE5_CORRECTED.md` for the corrected A-vs-D
> story and `results/phase4/PHASE6_FINAL.md` for the paper-ready verdict.

Matched-compute comparison of the project's central architectural
intervention against the flat baseline. Variants differ only in the
action-type conditioning signal — same Qwen2-VL-2B backbone, same LoRA,
same data, same hyperparameters, same eval.

- **A (flat baseline)** — plain Qwen2-VL-2B + LoRA. Goal text + image → coordinate.
- **D (action-conditioned, ours)** — adds an `<|action_slot|>` token to the
  prompt and `nn.Embedding(8, hidden_dim)` table of trainable action embeddings.
  At forward time the slot's embedding is replaced by the row for the gold
  action type.

## data_mix = `all_with_coords`

- n_train: 1200  ·  epochs: 2  ·  seeds: 1

| Variant | final train loss | hit@0.05 | hit@0.10 | hit@0.25 | mean norm L2 |
|---|---|---|---|---|---|
| **A (flat baseline)** | 0.849 | 0.160 | **0.276** | 0.540 | 0.393 |
| **D (action-conditioned)** | 0.837 | 0.109 | **0.227** | 0.486 | 0.391 |

Deltas (D − A; positive = D better for hit@r, negative = D better for L2):

- **hit@0.05**: -0.0507  ·  pooled std = 0.0000  ·  infσ (≥2σ — meaningful)
- **hit@0.10**: -0.0493  ·  pooled std = 0.0000  ·  infσ (≥2σ — meaningful)
- **hit@0.25**: -0.0542  ·  pooled std = 0.0000  ·  infσ (≥2σ — meaningful)
- **mean norm L2**: -0.0019  ·  pooled std = 0.0000  ·  infσ (≥2σ — meaningful)

## data_mix = `taps_and_swipes`

- n_train: 1000  ·  epochs: 2  ·  seeds: 3

| Variant | final train loss | hit@0.05 | hit@0.10 | hit@0.25 | mean norm L2 |
|---|---|---|---|---|---|
| **A (flat baseline)** | 0.976 ± 0.002 | 0.233 ± 0.010 | **0.390 ± 0.010** | 0.727 ± 0.023 | 0.184 ± 0.002 |
| **D (action-conditioned)** | 0.986 ± 0.003 | 0.143 ± 0.024 | **0.282 ± 0.032** | 0.597 ± 0.040 | 0.219 ± 0.005 |

Deltas (D − A; positive = D better for hit@r, negative = D better for L2):

- **hit@0.05**: -0.0900  ·  pooled std = 0.0258  ·  3.49σ (≥2σ — meaningful)
- **hit@0.10**: -0.1083  ·  pooled std = 0.0333  ·  3.25σ (≥2σ — meaningful)
- **hit@0.25**: -0.1300  ·  pooled std = 0.0465  ·  2.79σ (≥2σ — meaningful)
- **mean norm L2**: +0.0350  ·  pooled std = 0.0055  ·  6.31σ (≥2σ — meaningful)

## data_mix = `taps_only`

- n_train: 500  ·  epochs: 2  ·  seeds: 3

| Variant | final train loss | hit@0.05 | hit@0.10 | hit@0.25 | mean norm L2 |
|---|---|---|---|---|---|
| **A (flat baseline)** | 0.918 ± 0.002 | 0.167 ± 0.012 | **0.363 ± 0.040** | 0.667 ± 0.021 | 0.236 ± 0.011 |
| **D (action-conditioned)** | 0.943 ± 0.004 | 0.137 ± 0.015 | **0.320 ± 0.036** | 0.623 ± 0.021 | 0.271 ± 0.009 |

Deltas (D − A; positive = D better for hit@r, negative = D better for L2):

- **hit@0.05**: -0.0300  ·  pooled std = 0.0191  ·  1.57σ (~1σ)
- **hit@0.10**: -0.0433  ·  pooled std = 0.0542  ·  0.80σ (within noise)
- **hit@0.25**: -0.0433  ·  pooled std = 0.0294  ·  1.47σ (~1σ)
- **mean norm L2**: +0.0344  ·  pooled std = 0.0143  ·  2.40σ (≥2σ — meaningful)

## Interpretation crib sheet (from the project plan)

- **D > A** → decoupling helps. The action embedding carries useful
  signal that the flat decode can't easily recover from input alone.
- **D ≈ A** → conditioning doesn't help on this data. Most likely cause:
  the training distribution doesn't span enough action types for the
  embedding to learn class-conditional behavior.
- **D < A** → conditioning is actively harming. Either the embedding is
  consuming gradient capacity without benefit, or the training schedule
  isn't long enough to leverage the extra parameters.

Figure: `results/phase4/ablation_A_vs_D.png`
Raw aggregates: `results/phase4/ablation_summary.json`
