# Phase 9: large held-out validation slice

Slice: 2000 steps from 274 episodes (1437 clicks, 316 scrolls, 247 type events); no episode is shared with training.

Primary metrics use the click and scroll steps only; type targets are off-screen and unhittable.
246 steps are the earlier 250-step slice (minus its boundary episode); 1542 spatial steps from 234 episodes are new ('fresh' columns).

## 1. Mixed stream (type events in training)

| variant | seeds | hit@0.10 | hit@0.25 | mean L2 | Δhit@0.10 vs A | Δhit@0.14 vs A | ΔL2 vs A | Δhit@0.10 vs A (all steps) |
|---|---|---|---|---|---|---|---|---|
| A: flat | 5 | 0.333 | 0.607 | 0.249 | -- | -- | -- | -- |
| B: aux. loss | 5 | 0.391 | 0.657 | 0.219 | +0.057 [+0.046, +0.069] *** (E=274, seeds=5, seed p=0.000) **established** | +0.055 [+0.042, +0.067] *** (E=274, seeds=5, seed p=0.001) **established** | -0.029 [-0.037, -0.022] *** (E=274, seeds=5, seed p=0.031) **established** | +0.050 [+0.040, +0.061] *** (E=274, seeds=5, seed p=0.000) **established** |
| B': aux. loss, prompt-pooled | 5 | 0.390 | 0.657 | 0.219 | +0.057 [+0.045, +0.069] *** (E=274, seeds=5, seed p=0.000) **established** | +0.058 [+0.046, +0.070] *** (E=274, seeds=5, seed p=0.000) **established** | -0.030 [-0.037, -0.022] *** (E=274, seeds=5, seed p=0.015) **established** | +0.050 [+0.040, +0.061] *** (E=274, seeds=5, seed p=0.000) **established** |
| C: hard routing | 5 | 0.352 | 0.598 | 0.289 | +0.018 [+0.006, +0.031] ** (E=274, seeds=5, seed p=0.368) | +0.004 [-0.011, +0.020] ns (E=274, seeds=5, seed p=0.844) | +0.040 [+0.024, +0.056] *** (E=274, seeds=5, seed p=0.009) **established** | +0.016 [+0.005, +0.027] ** (E=274, seeds=5, seed p=0.368) |
| D-hook: additive | 5 | 0.415 | 0.693 | 0.196 | +0.082 [+0.070, +0.094] *** (E=274, seeds=5, seed p=0.000) **established** | +0.087 [+0.073, +0.101] *** (E=274, seeds=5, seed p=0.000) **established** | -0.053 [-0.060, -0.046] *** (E=274, seeds=5, seed p=0.002) **established** | +0.072 [+0.062, +0.083] *** (E=274, seeds=5, seed p=0.000) **established** |
| D-hook, table LR x10 | 5 | 0.429 | 0.710 | 0.188 | +0.096 [+0.082, +0.110] *** (E=274, seeds=5, seed p=0.000) **established** | +0.104 [+0.090, +0.119] *** (E=274, seeds=5, seed p=0.000) **established** | -0.061 [-0.070, -0.053] *** (E=274, seeds=5, seed p=0.001) **established** | +0.084 [+0.072, +0.097] *** (E=274, seeds=5, seed p=0.000) **established** |
| D-token: prepended, table LR x10 | 5 | 0.412 | 0.696 | 0.195 | +0.078 [+0.067, +0.090] *** (E=274, seeds=5, seed p=0.000) **established** | +0.088 [+0.076, +0.102] *** (E=274, seeds=5, seed p=0.000) **established** | -0.054 [-0.062, -0.047] *** (E=274, seeds=5, seed p=0.003) **established** | +0.069 [+0.059, +0.079] *** (E=274, seeds=5, seed p=0.000) **established** |
| D-text: prompt word | 5 | 0.416 | 0.707 | 0.192 | +0.083 [+0.069, +0.097] *** (E=274, seeds=5, seed p=0.001) **established** | +0.096 [+0.081, +0.112] *** (E=274, seeds=5, seed p=0.001) **established** | -0.057 [-0.065, -0.049] *** (E=274, seeds=5, seed p=0.002) **established** | +0.072 [+0.061, +0.084] *** (E=274, seeds=5, seed p=0.001) **established** |

Fresh steps only (never in the earlier slice) and earlier-slice steps only, Δhit@0.10 vs A:

- B: aux. loss: fresh +0.054 [+0.042, +0.066] *** (E=234, seeds=5, seed p=0.001) **established**; earlier slice +0.082 [+0.050, +0.116] *** (E=41, seeds=5, seed p=0.004) **established**
- B': aux. loss, prompt-pooled: fresh +0.054 [+0.042, +0.067] *** (E=234, seeds=5, seed p=0.001) **established**; earlier slice +0.082 [+0.049, +0.115] *** (E=41, seeds=5, seed p=0.004) **established**
- C: hard routing: fresh +0.015 [+0.002, +0.028] * (E=234, seeds=5, seed p=0.471); earlier slice +0.043 [+0.000, +0.083] * (E=41, seeds=5, seed p=0.159)
- D-hook: additive: fresh +0.088 [+0.075, +0.101] *** (E=234, seeds=5, seed p=0.001) **established**; earlier slice +0.042 [+0.008, +0.078] * (E=41, seeds=5, seed p=0.024) **established**
- D-hook, table LR x10: fresh +0.098 [+0.084, +0.113] *** (E=234, seeds=5, seed p=0.000) **established**; earlier slice +0.081 [+0.032, +0.131] ** (E=41, seeds=5, seed p=0.009) **established**
- D-token: prepended, table LR x10: fresh +0.081 [+0.070, +0.094] *** (E=234, seeds=5, seed p=0.000) **established**; earlier slice +0.057 [+0.020, +0.096] ** (E=41, seeds=5, seed p=0.067)
- D-text: prompt word: fresh +0.083 [+0.069, +0.097] *** (E=234, seeds=5, seed p=0.001) **established**; earlier slice +0.081 [+0.034, +0.131] *** (E=41, seeds=5, seed p=0.022) **established**

Per class, hit@0.10:

- A: flat: click 0.374, scroll 0.146
- B: aux. loss: click 0.449, scroll 0.123; click vs A +0.075 [+0.062, +0.088] *** (E=272, seeds=5, seed p=0.001) **established**; scroll vs A -0.023 [-0.047, -0.001] * (E=155, seeds=5, seed p=0.237)
- B': aux. loss, prompt-pooled: click 0.450, scroll 0.121; click vs A +0.075 [+0.063, +0.089] *** (E=272, seeds=5, seed p=0.001) **established**; scroll vs A -0.025 [-0.048, -0.005] * (E=155, seeds=5, seed p=0.137)
- C: hard routing: click 0.414, scroll 0.066; click vs A +0.040 [+0.026, +0.054] *** (E=272, seeds=5, seed p=0.128); scroll vs A -0.080 [-0.114, -0.052] *** (E=155, seeds=5, seed p=0.005) **established**
- D-hook: additive: click 0.463, scroll 0.199; click vs A +0.089 [+0.076, +0.101] *** (E=272, seeds=5, seed p=0.000) **established**; scroll vs A +0.053 [+0.024, +0.084] *** (E=155, seeds=5, seed p=0.008) **established**
- D-hook, table LR x10: click 0.476, scroll 0.219; click vs A +0.101 [+0.088, +0.115] *** (E=272, seeds=5, seed p=0.000) **established**; scroll vs A +0.073 [+0.029, +0.121] ** (E=155, seeds=5, seed p=0.015) **established**
- D-token: prepended, table LR x10: click 0.459, scroll 0.196; click vs A +0.085 [+0.073, +0.097] *** (E=272, seeds=5, seed p=0.000) **established**; scroll vs A +0.049 [+0.020, +0.081] *** (E=155, seeds=5, seed p=0.001) **established**
- D-text: prompt word: click 0.461, scroll 0.213; click vs A +0.086 [+0.074, +0.099] *** (E=272, seeds=5, seed p=0.002) **established**; scroll vs A +0.067 [+0.023, +0.117] ** (E=155, seeds=5, seed p=0.008) **established**
- B': aux. loss, prompt-pooled − B: -0.000 [-0.004, +0.004] ns (E=274, seeds=5, seed p=0.961)
- D-hook, table LR x10 − B: +0.039 [+0.024, +0.054] *** (E=274, seeds=5, seed p=0.003) **established**
- D-token: prepended, table LR x10 − B: +0.021 [+0.009, +0.033] ** (E=274, seeds=5, seed p=0.047) **established**
- D-text: prompt word − B: +0.025 [+0.011, +0.040] *** (E=274, seeds=5, seed p=0.015) **established**
- D-hook: additive − B: +0.025 [+0.013, +0.037] *** (E=274, seeds=5, seed p=0.001) **established**
- C: hard routing − B: -0.039 [-0.052, -0.026] *** (E=274, seeds=5, seed p=0.073)
- B' (prompt-pooled) − B, mean L2: -0.000 [-0.002, +0.002] ns (E=274, seeds=5, seed p=0.966)
- D-hook 10x − D-hook: +0.014 [+0.005, +0.023] ** (E=274, seeds=5, seed p=0.091)
- D-hook 10x − D-token 10x: +0.018 [+0.009, +0.027] *** (E=274, seeds=5, seed p=0.003) **established**

## 2. Clean stream (type events removed from training only; same validation slice)

| variant | seeds | hit@0.10 | mean L2 | Δhit@0.10 vs A (clean) | ΔL2 vs A (clean) | Δhit@0.25 vs A (clean) | removal effect hit@0.10 | removal effect L2 |
|---|---|---|---|---|---|---|---|---|
| A: flat | 5 | 0.386 | 0.214 | -- | -- | -- | +0.053 [+0.043, +0.063] *** (E=274, seeds=5, seed p=0.008) **established** | -0.035 [-0.042, -0.029] *** (E=274, seeds=5, seed p=0.020) **established** |
| B: aux. loss | 5 | 0.413 | 0.205 | +0.026 [+0.016, +0.037] *** (E=274, seeds=5, seed p=0.021) **established** | -0.008 [-0.015, -0.002] * (E=274, seeds=5, seed p=0.134) | +0.025 [+0.012, +0.040] *** (E=274, seeds=5, seed p=0.073) | +0.022 [+0.012, +0.032] *** (E=274, seeds=5, seed p=0.089) | -0.014 [-0.019, -0.009] *** (E=274, seeds=5, seed p=0.022) **established** |
| D-hook: additive | 5 | 0.422 | 0.190 | +0.036 [+0.027, +0.046] *** (E=274, seeds=5, seed p=0.000) **established** | -0.023 [-0.030, -0.017] *** (E=274, seeds=5, seed p=0.000) **established** | +0.039 [+0.026, +0.053] *** (E=274, seeds=5, seed p=0.000) **established** | +0.007 [-0.002, +0.016] ns (E=274, seeds=5, seed p=0.608) | -0.005 [-0.010, -0.001] ** (E=274, seeds=5, seed p=0.304) |
| D-text: prompt word | 5 | 0.401 | 0.198 | +0.015 [+0.002, +0.028] * (E=274, seeds=5, seed p=0.111) | -0.016 [-0.023, -0.009] *** (E=274, seeds=5, seed p=0.018) **established** | +0.038 [+0.024, +0.054] *** (E=274, seeds=5, seed p=0.006) **established** | -0.014 [-0.023, -0.007] *** (E=274, seeds=5, seed p=0.145) | +0.006 [+0.002, +0.009] *** (E=274, seeds=5, seed p=0.239) |

Clean stream by region of the slice, Δhit@0.10 vs A (earlier slice: 211 spatial steps; the taps-and-swipes control's held-out steps that fall in this slice: 178):

- B: aux. loss: fresh +0.029 [+0.017, +0.040] *** (E=234, seeds=5, seed p=0.014) **established**; earlier slice +0.010 [-0.010, +0.031] ns (E=41, seeds=5, seed p=0.651); control steps +0.000 [-0.019, +0.019] ns (E=34, seeds=5, seed p=1.000)
- D-hook: additive: fresh +0.039 [+0.028, +0.050] *** (E=234, seeds=5, seed p=0.000) **established**; earlier slice +0.015 [-0.007, +0.037] ns (E=41, seeds=5, seed p=0.595); control steps +0.009 [-0.015, +0.034] ns (E=34, seeds=5, seed p=0.760)
- D-text: prompt word: fresh +0.021 [+0.007, +0.035] ** (E=234, seeds=5, seed p=0.081); earlier slice -0.024 [-0.071, +0.025] ns (E=41, seeds=5, seed p=0.117); control steps -0.034 [-0.086, +0.019] ns (E=34, seeds=5, seed p=0.046)

Share of the mixed-stream gain that removing type events takes away, (V-A mixed) - (V-A clean):

- B: aux. loss: hit@0.10 +0.031 [+0.019, +0.043] *** (E=274, seeds=5, seed p=0.002) **established** of a mixed gain of +0.057 (54%); mean L2 -0.021 [-0.028, -0.014] *** (E=274, seeds=5, seed p=0.050)
- D-hook: additive: hit@0.10 +0.046 [+0.034, +0.058] *** (E=274, seeds=5, seed p=0.001) **established** of a mixed gain of +0.082 (56%); mean L2 -0.030 [-0.037, -0.023] *** (E=274, seeds=5, seed p=0.015) **established**
- D-text: prompt word: hit@0.10 +0.067 [+0.055, +0.080] *** (E=274, seeds=5, seed p=0.003) **established** of a mixed gain of +0.083 (82%); mean L2 -0.041 [-0.048, -0.034] *** (E=274, seeds=5, seed p=0.007) **established**

## 3. End to end: Stage-1 predicted types

Stage-1 accuracy 0.776 (majority class 0.719).

| variant | gold hit@0.10 | predicted hit@0.10 | predicted − A | gold − predicted | predicted − A (L2) | predicted − B |
|---|---|---|---|---|---|---|
| C: hard routing | 0.352 | 0.321 | -0.012 [-0.024, +0.000] ns (E=274, seeds=5, seed p=0.486) | +0.030 [+0.023, +0.038] *** (E=274, seeds=5, seed p=0.000) **established** | +0.056 [+0.041, +0.072] *** (E=274, seeds=5, seed p=0.004) **established** | -0.069 [-0.083, -0.056] *** (E=274, seeds=5, seed p=0.008) **established** |
| D-hook: additive | 0.415 | 0.372 | +0.039 [+0.027, +0.050] *** (E=274, seeds=5, seed p=0.002) **established** | +0.044 [+0.035, +0.052] *** (E=274, seeds=5, seed p=0.000) **established** | -0.014 [-0.023, -0.006] *** (E=274, seeds=5, seed p=0.119) | -0.019 [-0.030, -0.008] ** (E=274, seeds=5, seed p=0.005) **established** |
| D-token: prepended, table LR x10 | 0.412 | 0.372 | +0.039 [+0.028, +0.050] *** (E=274, seeds=5, seed p=0.007) **established** | +0.040 [+0.031, +0.048] *** (E=274, seeds=5, seed p=0.000) **established** | -0.016 [-0.024, -0.008] *** (E=274, seeds=5, seed p=0.145) | -0.019 [-0.031, -0.006] ** (E=274, seeds=5, seed p=0.110) |
| D-text: prompt word | 0.416 | 0.372 | +0.039 [+0.026, +0.051] *** (E=274, seeds=5, seed p=0.010) **established** | +0.044 [+0.034, +0.055] *** (E=274, seeds=5, seed p=0.000) **established** | -0.020 [-0.029, -0.012] *** (E=274, seeds=5, seed p=0.089) | -0.019 [-0.032, -0.006] ** (E=274, seeds=5, seed p=0.072) |
| B: aux. loss (needs no type at inference) | 0.391 | same | +0.057 [+0.046, +0.069] *** (E=274, seeds=5, seed p=0.000) **established** | 0 | -- |

Action-table row displacement from initialization: D-hook: additive 0.044, D-hook, table LR x10 0.183, D-token: prepended, table LR x10 0.286

Share of click predictions within 0.10 of the origin (mixed stream): A: flat 8.3%, B: aux. loss 4.2%, B': aux. loss, prompt-pooled 4.3%, C: hard routing 0.1%, D-hook: additive 0.8%, D-token: prepended, table LR x10 0.3%, D-text: prompt word 0.8%

Share of click predictions within 0.10 of the origin (clean stream): A: flat 0.5%, B: aux. loss 0.5%, D-hook: additive 0.4%, D-text: prompt word 0.6%

Unparseable outputs: C: hard routing click 1.3% / scroll 25.7% (scroll hit@0.10 on parsed 0.089), D-hook, table LR x10 click 0.0% / scroll 0.4% (scroll hit@0.10 on parsed 0.220)

Stage-1 on click and scroll steps: 22.3% misclassified, 3.4% sent to type; clicks called scroll 7.6%, scrolls called click 70.4%; type recall 76.6%; macro-F1 0.652.
- C: hard routing: hit@0.10 on misclassified spatial steps 0.062 with the predicted type vs 0.198 with gold; on correctly classified steps 0.396
- D-hook: additive: hit@0.10 on misclassified spatial steps 0.072 with the predicted type vs 0.267 with gold; on correctly classified steps 0.458
- D-token: prepended, table LR x10: hit@0.10 on misclassified spatial steps 0.083 with the predicted type vs 0.260 with gold; on correctly classified steps 0.455
- D-text: prompt word: hit@0.10 on misclassified spatial steps 0.086 with the predicted type vs 0.284 with gold; on correctly classified steps 0.454

Batched vs one-at-a-time decoding on the first 64 steps of 12 runs: 38.2% identical strings on average (minimum 20.3%); 74.0% of prediction pairs within 0.01 of each other, 93.0% within 0.05; mean |Δ distance to target| 0.012; hit@0.10 batched minus unbatched +0.008 on average, largest gap 0.047.

## Control: taps_and_swipes (n_train=1000, n_val=200, seeds 42-44)

| variant | seeds | hit@0.10 | mean L2 | Δhit@0.10 vs A | ΔL2 vs A |
|---|---|---|---|---|---|
| A: flat | 3 | 0.382 | 0.185 | -- | -- |
| B: aux. loss | 3 | 0.368 | 0.173 | -0.013 [-0.042, +0.013] ns (E=37, seeds=3, seed p=0.456) | -0.012 [-0.021, -0.003] ** (E=37, seeds=3, seed p=0.291) |
| C: hard routing | 3 | 0.325 | 0.271 | -0.057 [-0.101, -0.011] * (E=37, seeds=3, seed p=0.215) | +0.086 [+0.050, +0.133] *** (E=37, seeds=3, seed p=0.167) |
| D-hook: additive | 3 | 0.363 | 0.182 | -0.018 [-0.056, +0.019] ns (E=37, seeds=3, seed p=0.334) | -0.002 [-0.012, +0.007] ns (E=37, seeds=3, seed p=0.681) |
| D-token: prepended | 3 | 0.332 | 0.200 | -0.050 [-0.083, -0.018] ** (E=37, seeds=3, seed p=0.041) **established** | +0.015 [+0.008, +0.022] *** (E=37, seeds=3, seed p=0.111) |
| D-text: prompt word | 3 | 0.335 | 0.199 | -0.047 [-0.094, +0.002] ns (E=37, seeds=3, seed p=0.088) | +0.014 [+0.001, +0.027] * (E=37, seeds=3, seed p=0.020) **established** |
- Same prepended-token variant, input_ids path minus inputs_embeds path: +0.033 hit@0.10 (seed-level paired t p=0.109; per seed fixed [0.36, 0.32, 0.315], slot [0.32, 0.27, 0.305])

## Claims

| claim | Δ | 95% CI | boot p | seed p | established |
|---|---|---|---|---|---|
| B - A, mixed stream (hit@0.10) | +0.057 | [+0.046, +0.069] | 0.000 | 0.000 | yes |
| B' - A, mixed stream (hit@0.10) | +0.057 | [+0.045, +0.069] | 0.000 | 0.000 | yes |
| C - A, mixed stream (hit@0.10) | +0.018 | [+0.006, +0.031] | 0.004 | 0.368 | no |
| D-hook - A, mixed stream (hit@0.10) | +0.082 | [+0.070, +0.094] | 0.000 | 0.000 | yes |
| D-hook x10 - A, mixed stream (hit@0.10) | +0.096 | [+0.082, +0.110] | 0.000 | 0.000 | yes |
| D-token x10 - A, mixed stream (hit@0.10) | +0.078 | [+0.067, +0.090] | 0.000 | 0.000 | yes |
| D-text - A, mixed stream (hit@0.10) | +0.083 | [+0.069, +0.097] | 0.000 | 0.001 | yes |
| D-hook - B, mixed stream (hit@0.10) | +0.025 | [+0.013, +0.037] | 0.000 | 0.001 | yes |
| D-text - B, mixed stream (hit@0.10) | +0.025 | [+0.011, +0.040] | 0.000 | 0.015 | yes |
| D-token x10 - B, mixed stream (hit@0.10) | +0.021 | [+0.009, +0.033] | 0.001 | 0.047 | yes |
| B' - B, mixed stream (hit@0.10) | -0.000 | [-0.004, +0.004] | 0.926 | 0.961 | no |
| D-hook x10 - D-hook, mixed stream (hit@0.10) | +0.014 | [+0.005, +0.023] | 0.002 | 0.091 | no |
| D-hook x10 - D-token x10, mixed stream (hit@0.10) | +0.018 | [+0.009, +0.027] | 0.000 | 0.003 | yes |
| B - A, mixed stream, new steps only (hit@0.10) | +0.054 | [+0.042, +0.066] | 0.000 | 0.001 | yes |
| D-hook - A, mixed stream, new steps only (hit@0.10) | +0.088 | [+0.075, +0.101] | 0.000 | 0.001 | yes |
| D-token x10 - A, mixed stream, new steps only (hit@0.10) | +0.081 | [+0.070, +0.094] | 0.000 | 0.000 | yes |
| D-text - A, mixed stream, new steps only (hit@0.10) | +0.083 | [+0.069, +0.097] | 0.000 | 0.001 | yes |
| A clean - A mixed (hit@0.10) | +0.053 | [+0.043, +0.063] | 0.000 | 0.008 | yes |
| B - A, clean stream (hit@0.10) | +0.026 | [+0.016, +0.037] | 0.000 | 0.021 | yes |
| D-hook - A, clean stream (hit@0.10) | +0.036 | [+0.027, +0.046] | 0.000 | 0.000 | yes |
| D-text - A, clean stream (hit@0.10) | +0.015 | [+0.002, +0.028] | 0.021 | 0.111 | no |
| B - A, clean stream (mean L2) | -0.008 | [-0.015, -0.002] | 0.014 | 0.134 | no |
| D-hook - A, clean stream (mean L2) | -0.023 | [-0.030, -0.017] | 0.000 | 0.000 | yes |
| D-text - A, clean stream (mean L2) | -0.016 | [-0.023, -0.009] | 0.000 | 0.018 | yes |
| B - A, clean stream, new steps only (hit@0.10) | +0.029 | [+0.017, +0.040] | 0.000 | 0.014 | yes |
| D-hook - A, clean stream, new steps only (hit@0.10) | +0.039 | [+0.028, +0.050] | 0.000 | 0.000 | yes |
| D-text - A, clean stream, new steps only (hit@0.10) | +0.021 | [+0.007, +0.035] | 0.003 | 0.081 | no |
| C - A, predicted types (hit@0.10) | -0.012 | [-0.024, +0.000] | 0.056 | 0.486 | no |
| D-hook - A, predicted types (hit@0.10) | +0.039 | [+0.027, +0.050] | 0.000 | 0.002 | yes |
| D-token x10 - A, predicted types (hit@0.10) | +0.039 | [+0.028, +0.050] | 0.000 | 0.007 | yes |
| D-text - A, predicted types (hit@0.10) | +0.039 | [+0.026, +0.051] | 0.000 | 0.010 | yes |
| C - B, predicted types (hit@0.10) | -0.069 | [-0.083, -0.056] | 0.000 | 0.008 | yes |
| D-hook - B, predicted types (hit@0.10) | -0.019 | [-0.030, -0.008] | 0.001 | 0.005 | yes |
| D-token x10 - B, predicted types (hit@0.10) | -0.019 | [-0.031, -0.006] | 0.004 | 0.110 | no |
| D-text - B, predicted types (hit@0.10) | -0.019 | [-0.032, -0.006] | 0.006 | 0.072 | no |
| B - A, taps-and-swipes control (hit@0.10) | -0.013 | [-0.042, +0.013] | 0.330 | 0.456 | no |
| C - A, taps-and-swipes control (hit@0.10) | -0.057 | [-0.101, -0.011] | 0.014 | 0.215 | no |
| D-hook - A, taps-and-swipes control (hit@0.10) | -0.018 | [-0.056, +0.019] | 0.345 | 0.334 | no |
| D-token - A, taps-and-swipes control (hit@0.10) | -0.050 | [-0.083, -0.018] | 0.002 | 0.041 | yes |
| D-text - A, taps-and-swipes control (hit@0.10) | -0.047 | [-0.094, +0.002] | 0.064 | 0.088 | no |

## Reproduction

```bash
uv run modal deploy modal_app.py
uv run python scripts/p9_launch.py stage1
uv run python scripts/p9_launch.py full          # spawns every run on the deployed app
uv run python scripts/p9_status.py               # per-call status
cd results/phase9 && uv run modal volume get --force stage1-cache bigval . && uv run modal volume get --force stage1-cache bigval_control . && cd ../..
uv run python scripts/p9_consolidate.py          # this file, phase9_summary.json, aaai27/tables/*.tex
```

Files: `bigval/<variant>[_alr0.0002][_notype]_seed<s>_n1200_v2000_mix-all_with_coords.json` hold per-example distances, predictions and raw outputs for gold (and, where run, Stage-1 predicted) types; `bigval/stage1_preds_seed<s>_*.json` the Stage-1 predictions; `bigval_control/` the taps-and-swipes D-token runs.
