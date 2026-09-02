# Aggregate attention analysis (E5)

Head-averaged attention from a probe token to the image tokens. `target_frac@r` = share of that token's image attention within r of the gold point; chance = share of image tokens within r. Paired bootstrap over probe examples, 10k resamples. Conditions: `gold` = gold type; `wrong` = wrong: cyclic (click→type); `wrong2` = wrong: click↔scroll; `zero` = zeroed embedding.

## D-hook (additive) — seed 42, n_train=1200, 120 probe examples, embedding norm 0.08, file `attn_aggregate_Dhook_seed42_n1200_v2.json`

wrong map {'click': 'type', 'type': 'scroll', 'scroll': 'click'}; wrong2 map {'click': 'scroll', 'type': 'click', 'scroll': 'click'}. Chance target_frac@0.10 (area share) = 0.026.

**Probe position `last_prompt`, layer 3q**

| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy |
|---|---|---|---|---|
| gold | 0.040 | 0.196 | 0.124 | 4.76 |
| wrong | 0.036 | 0.155 | 0.087 | 4.95 |
| wrong2 | 0.035 | 0.173 | 0.101 | 4.70 |
| zero | 0.036 | 0.186 | 0.112 | 4.68 |

| contrast | Δ target_frac@0.10 | Δ target_frac@0.25 | Δ image_mass |
|---|---|---|---|
| gold − wrong | +0.003 [-0.001, +0.008] ns | +0.041 [+0.029, +0.054] *** | +0.037 [+0.026, +0.049] *** |
| gold − wrong2 | +0.005 [+0.002, +0.007] *** | +0.023 [+0.018, +0.029] *** | +0.024 [+0.010, +0.037] ** |
| gold − zero | +0.003 [+0.002, +0.005] *** | +0.010 [+0.006, +0.013] *** | +0.012 [+0.005, +0.019] *** |

**Probe position `pre_x`, layer 3q**

| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy |
|---|---|---|---|---|
| gold | 0.039 | 0.217 | 0.132 | 4.55 |
| wrong | 0.036 | 0.183 | 0.083 | 4.65 |
| wrong2 | 0.037 | 0.203 | 0.106 | 4.42 |
| zero | 0.039 | 0.212 | 0.131 | 4.51 |

| contrast | Δ target_frac@0.10 | Δ target_frac@0.25 | Δ image_mass |
|---|---|---|---|
| gold − wrong | +0.003 [-0.002, +0.009] ns | +0.033 [+0.017, +0.050] *** | +0.049 [+0.036, +0.060] *** |
| gold − wrong2 | +0.002 [-0.002, +0.006] ns | +0.014 [+0.003, +0.025] * | +0.026 [+0.011, +0.040] ** |
| gold − zero | +0.001 [-0.002, +0.003] ns | +0.004 [-0.001, +0.010] ns | +0.001 [-0.006, +0.008] ns |

**Probe position `pre_y`, layer 3q**

| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy |
|---|---|---|---|---|
| gold | 0.111 | 0.307 | 0.237 | 4.79 |
| wrong | 0.064 | 0.239 | 0.190 | 4.78 |
| wrong2 | 0.049 | 0.213 | 0.201 | 4.74 |
| zero | 0.119 | 0.320 | 0.263 | 4.75 |

| contrast | Δ target_frac@0.10 | Δ target_frac@0.25 | Δ image_mass |
|---|---|---|---|
| gold − wrong | +0.048 [+0.030, +0.065] *** | +0.067 [+0.041, +0.093] *** | +0.047 [+0.033, +0.061] *** |
| gold − wrong2 | +0.062 [+0.040, +0.086] *** | +0.094 [+0.063, +0.126] *** | +0.035 [+0.014, +0.056] ** |
| gold − zero | -0.008 [-0.017, +0.001] ns | -0.013 [-0.030, +0.002] ns | -0.027 [-0.037, -0.016] *** |

**Greedy decoding under each conditioning**

| condition | hit@0.10 | hit@0.25 | mean dist | Δ hit@0.10 vs gold |
|---|---|---|---|---|
| gold | 0.358 | 0.642 | 0.355 | -- |
| wrong | 0.083 | 0.175 | 0.861 | +0.275 [+0.183, +0.367] *** |
| wrong2 | 0.100 | 0.333 | 0.595 | +0.258 [+0.167, +0.350] *** |
| zero | 0.392 | 0.658 | 0.405 | -0.033 [-0.117, +0.042] ns |

Per gold class (target_frac@0.10 at pre_y if available else last_prompt / hit@0.10):

| class | n | gold | wrong | wrong2 | zero |
|---|---|---|---|---|---|
| click | 81 | 0.051 / 0.432 | 0.044 / 0.012 | 0.042 / 0.037 | 0.045 / 0.519 |
| scroll | 22 | 0.031 / 0.364 | 0.037 / 0.409 | 0.037 / 0.409 | 0.033 / 0.227 |
| type | 17 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

## D-token (prepended) — seed 42, n_train=1200, 120 probe examples, embedding norm 2.22, file `attn_aggregate_Dtoken_seed42_n1200.json`

wrong map {'click': 'type', 'type': 'scroll', 'scroll': 'click'}; wrong2 map n/a. Chance target_frac@0.10 (area share) = 0.026.

**Probe position `last_prompt`, layer 3q**

| condition | target_frac@0.10 | target_frac@0.25 | image_mass | entropy |
|---|---|---|---|---|
| gold | 0.037 | 0.188 | 0.145 | 4.75 |
| wrong | 0.034 | 0.174 | 0.176 | 4.86 |
| zero | 0.035 | 0.178 | 0.172 | 4.86 |

| contrast | Δ target_frac@0.10 | Δ target_frac@0.25 | Δ image_mass |
|---|---|---|---|
| gold − wrong | +0.004 [+0.001, +0.007] * | +0.014 [+0.007, +0.021] *** | -0.032 [-0.040, -0.023] *** |
| gold − zero | +0.002 [-0.000, +0.005] ns | +0.010 [+0.005, +0.016] *** | -0.028 [-0.034, -0.022] *** |

**Greedy decoding under each conditioning**

| condition | hit@0.10 | hit@0.25 | mean dist | Δ hit@0.10 vs gold |
|---|---|---|---|---|
| gold | 0.267 | 0.575 | 0.370 | -- |
| wrong | 0.117 | 0.308 | 0.625 | +0.150 [+0.058, +0.242] *** |
| zero | 0.142 | 0.400 | 0.517 | +0.125 [+0.042, +0.208] ** |

Per gold class (target_frac@0.10 at pre_y if available else last_prompt / hit@0.10):

| class | n | gold | wrong | zero |
|---|---|---|---|---|
| click | 81 | 0.047 / 0.284 | 0.041 / 0.062 | 0.043 / 0.123 |
| scroll | 22 | 0.032 / 0.409 | 0.033 / 0.409 | 0.033 / 0.318 |
| type | 17 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |

