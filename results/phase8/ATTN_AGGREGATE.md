# Aggregate attention analysis (E5)

Head-averaged attention from a probe token to the image tokens. `target_frac@r` = share of that token's image attention within r of the gold point; chance = share of image tokens within r. Paired bootstrap over probe examples, 10k resamples. Conditions: `gold` = gold type; `wrong` = wrong: cyclic (click→type); `wrong2` = wrong: click↔scroll; `zero` = zeroed embedding.

## D-hook (additive) — seed 42, n_train=1200, 120 probe examples, embedding norm 0.08, file `attn_aggregate_Dhook_seed42_n1200.json`

wrong map {'click': 'type', 'type': 'scroll', 'scroll': 'click'}; wrong2 map n/a. Chance target_frac@0.10 (area share) = 0.026.

**Probe position `last_prompt`, layer 3q**

| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy |
|---|---|---|---|---|
| gold | 0.040 | 0.196 | 0.124 | 4.76 |
| wrong | 0.036 | 0.155 | 0.087 | 4.95 |
| zero | 0.036 | 0.187 | 0.112 | 4.68 |

| contrast | Δ target_frac@0.10 | Δ target_frac@0.25 | Δ image_mass |
|---|---|---|---|
| gold − wrong | +0.004 [-0.001, +0.008] ns | +0.041 [+0.029, +0.054] *** | +0.037 [+0.026, +0.048] *** |
| gold − zero | +0.003 [+0.002, +0.005] *** | +0.009 [+0.006, +0.012] *** | +0.012 [+0.005, +0.019] ** |

**Greedy decoding under each conditioning**

| condition | hit@0.10 | hit@0.25 | mean dist | Δ hit@0.10 vs gold |
|---|---|---|---|---|
| gold | 0.358 | 0.642 | 0.355 | -- |
| wrong | 0.083 | 0.175 | 0.861 | +0.275 [+0.183, +0.367] *** |
| zero | 0.392 | 0.658 | 0.405 | -0.033 [-0.117, +0.050] ns |

Per gold class (target_frac@0.10 at last_prompt if available else last_prompt / hit@0.10):

| class | n | gold | wrong | zero |
|---|---|---|---|---|
| click | 81 | 0.051 / 0.432 | 0.044 / 0.012 | 0.045 / 0.519 |
| scroll | 22 | 0.031 / 0.364 | 0.037 / 0.409 | 0.033 / 0.227 |
| type | 17 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

