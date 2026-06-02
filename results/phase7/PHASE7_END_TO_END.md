# Phase 7 - End-to-end Stage 1 -> Stage 2 evaluation

Data mix: **all_with_coords**. Pooled units: **750** (3 seeds x 250 examples).

## Means

| variant | hit@0.05 | hit@0.10 | hit@0.25 | mean norm L2 (lower better) |
|---|---:|---:|---:|---:|
| A-flat | 0.157 | 0.255 | 0.515 | 0.392 |
| D-gold
(oracle) | 0.193 | 0.299 | 0.588 | 0.365 |
| D-predicted
(Stage 1) | 0.180 | 0.283 | 0.528 | 0.395 |
| D-majority | 0.184 | 0.291 | 0.583 | 0.414 |
| D-random | 0.065 | 0.112 | 0.268 | 0.650 |

## Headline statistical test

Deployable **D-predicted does not significantly beat A-flat** on the paired test. hit@0.10 delta=+0.028, 95% CI [+0.003, +0.053], p=0.0358; mean L2 delta=+0.003, p=0.6766.
Tight-threshold gains are significant: hit@0.05 delta=+0.023 (p=0.0266) and hit@0.10 delta=+0.028 (p=0.0358). The broader metrics are not: hit@0.25 delta=+0.013 (p=0.4152) and mean L2 delta=+0.003 (p=0.6766).

## Oracle ceiling

D-gold significantly beats A-flat on every metric. For hit@0.10, delta=+0.044, 95% CI [+0.019, +0.069], p=0.0004; for mean L2, delta=-0.028, p=0.0000. This confirms the action-type signal still has value when it is correct.

## Stage 1 quality

Stage 1 on the exact Stage 2 validation set: accuracy=0.735, macro-F1=0.693, weighted-F1=0.736.

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| click | 0.808 | 0.797 | 0.802 | 507 |
| type | 0.935 | 0.962 | 0.948 | 105 |
| scroll | 0.324 | 0.333 | 0.329 | 138 |

## Stage 1 error cost

When Stage 1 is correct (n=551), D-predicted hit@0.10=0.327. When Stage 1 is wrong (n=199), hit@0.10=0.161. This is the clearest evidence that action-type errors do affect downstream grounding.

## Valid-coordinate sensitivity

105 pooled examples have invalid target coordinates, mostly TYPE rows with `target_xy=[-1,-1]`; these are not meaningful coordinate-grounding targets.
After dropping invalid targets (n=645), the qualitative result is unchanged: D-predicted vs A-flat hit@0.10 delta=+0.033, 95% CI [+0.003, +0.062], p=0.0358; mean L2 delta=+0.002, p=0.7536.

## Figures

- `phase7_headline_hit010.png`
- `phase7_metric_grid.png`
- `phase7_oracle_gap.png`
- `phase7_delta_ci.png`
- `phase7_stage1_confusion.png`
- `phase7_stage1_per_class_f1.png`
- `phase7_action_distribution.png`
- `phase7_correct_vs_incorrect.png`

## Interpretation note

D-gold is an oracle ceiling. The deployable claim is based only on D-predicted vs A-flat. The statistically safe conclusion is partial deployable benefit: improved precise localization at tight thresholds, but no reliable overall improvement on loose hit rate or mean distance.
