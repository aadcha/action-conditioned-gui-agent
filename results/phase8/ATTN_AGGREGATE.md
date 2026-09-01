# Aggregate attention analysis (E5)

Attention from the last prompt token to image tokens, head-averaged at the 3q layer (3/4 depth; 1/2-depth and last-layer numbers are in the JSON). `target_frac@r` = share of image attention within r of the gold point; `area@r` = share of image tokens within r (chance level). Paired bootstrap over probe examples, 10k resamples.

## D-hook (additive) — seed 42, n_train=1200, 120 probe examples, embedding norm 0.08, wrong map {'click': 'type', 'type': 'scroll', 'scroll': 'click'}

Chance level (area share within r=0.10 of the target): 0.026

| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy | hit@0.10 | hit@0.25 | mean dist |
|---|---|---|---|---|---|---|---|
| gold | 0.040 | 0.196 | 0.124 | 4.76 | 0.358 | 0.642 | 0.355 |
| wrong | 0.036 | 0.155 | 0.087 | 4.95 | 0.083 | 0.175 | 0.861 |
| zero | 0.036 | 0.187 | 0.112 | 4.68 | 0.392 | 0.658 | 0.405 |

| contrast | Δ target_frac@0.10 [CI] | Δ target_frac@0.25 [CI] | Δ image_mass [CI] | Δ hit@0.10 [CI] |
|---|---|---|---|---|
| gold minus wrong | +0.004 [-0.001, +0.008] ns | +0.041 [+0.029, +0.054] *** | +0.037 [+0.026, +0.048] *** | +0.275 [+0.183, +0.367] *** |
| gold minus zero | +0.003 [+0.002, +0.005] *** | +0.009 [+0.006, +0.012] *** | +0.012 [+0.005, +0.019] ** | -0.033 [-0.117, +0.050] ns |

Per gold class (target_frac@0.10 / hit@0.10):

| class | n | gold | wrong | zero |
|---|---|---|---|---|
| click | 81 | 0.051 / 0.432 | 0.044 / 0.012 | 0.045 / 0.519 |
| scroll | 22 | 0.031 / 0.364 | 0.037 / 0.409 | 0.033 / 0.227 |
| type | 17 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

