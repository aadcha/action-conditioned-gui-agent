# Phase 8 — dependence-aware reanalysis (saved predictions, no retraining)

**Status: exploratory.** These are the Phase 6/7 runs re-analyzed with an episode-clustered paired test (seed-averaged per-example deltas; cluster bootstrap CI over episodes; episode-level sign-flip permutation p). The train/val split leakage documented below is NOT fixed by reanalysis — that requires the Phase 9 episode-disjoint rerun. Confirmatory numbers come only from the untouched test set.

## Leakage in the historical split (quantified)

| cell | val steps | val steps sharing a train episode | sharing a train goal |
|---|---|---|---|
| awc_n1200 | 250 | 4 (2%) | 61 (24%) |
| awc_n300 | 250 | 4 (2%) | 14 (6%) |
| awc_n500 | 250 | 4 (2%) | 47 (19%) |
| awc_n800 | 250 | 5 (2%) | 49 (20%) |
| awc_n2500 | 250 | 10 (4%) | 133 (53%) |
| awc_n5000 | 250 | 2 (1%) | 213 (85%) |
| ts_n1000 | 200 | 0 (0%) | 48 (24%) |

## Main ablation — all_with_coords, n_train=1200 (3 seeds)

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| B_vs_A | hit_at_010 | +0.051 [+0.020, +0.081] | 0.0043 (Holm 0.0172) | 0.0002 | A:0.280,0.256,0.228 B:0.316,0.312,0.288 |
| B_vs_A | hit_at_025 | +0.071 [+0.037, +0.103] | 0.0004 | 0 | A:0.532,0.528,0.484 B:0.600,0.596,0.560 |
| B_vs_A | hit_at_005 | +0.037 [+0.014, +0.060] | 0.005899 | 0.0006 | A:0.156,0.172,0.144 B:0.204,0.196,0.184 |
| B_vs_A | mean_normalized_l2 | -0.030 [-0.044, -0.016] | 0.0003 | 0 | A:0.391,0.381,0.404 B:0.359,0.364,0.365 |
| Dhook_vs_A | hit_at_010 | +0.045 [+0.011, +0.085] | 0.0227 (Holm 0.0681) | 0.0016 | A:0.280,0.256,0.228 B:0.316,0.260,0.324 |
| Dhook_vs_A | hit_at_025 | +0.055 [+0.019, +0.090] | 0.006899 | 0.0002 | A:0.532,0.528,0.484 B:0.580,0.548,0.580 |
| Dhook_vs_A | hit_at_005 | +0.037 [+0.012, +0.064] | 0.0116 | 0.0014 | A:0.156,0.172,0.144 B:0.200,0.180,0.204 |
| Dhook_vs_A | mean_normalized_l2 | -0.027 [-0.041, -0.013] | 0.0012 | 0 | A:0.391,0.381,0.404 B:0.363,0.374,0.359 |
| Dtoken_vs_A | hit_at_010 | +0.015 [-0.018, +0.047] | 0.4366 (Holm 0.556) | 0.2908 | A:0.280,0.256,0.228 B:0.296,0.276,0.236 |
| Dtoken_vs_A | hit_at_025 | +0.019 [-0.009, +0.047] | 0.2368 | 0.118 | A:0.532,0.528,0.484 B:0.560,0.540,0.500 |
| Dtoken_vs_A | hit_at_005 | +0.027 [+0.007, +0.049] | 0.0202 | 0.0028 | A:0.156,0.172,0.144 B:0.200,0.176,0.176 |
| Dtoken_vs_A | mean_normalized_l2 | -0.017 [-0.029, -0.005] | 0.0117 | 0.0006 | A:0.391,0.381,0.404 B:0.366,0.371,0.389 |
| C_vs_A | hit_at_010 | +0.024 [-0.016, +0.063] | 0.2779 (Holm 0.556) | 0.1082 | A:0.280,0.256,0.228 B:0.324,0.260,0.252 |
| C_vs_A | hit_at_025 | +0.028 [-0.016, +0.072] | 0.2488 | 0.1202 | A:0.532,0.528,0.484 B:0.616,0.540,0.472 |
| C_vs_A | hit_at_005 | +0.023 [-0.001, +0.045] | 0.09489 | 0.0316 | A:0.156,0.172,0.144 B:0.184,0.184,0.172 |
| C_vs_A | mean_normalized_l2 | +0.020 [-0.010, +0.055] | 0.2766 | 0.0368 | A:0.391,0.381,0.404 B:0.414,0.400,0.422 |
| Dtoken_vs_B | hit_at_010 | -0.036 [-0.077, +0.005] | 0.1136 | 0.0192 | A:0.316,0.312,0.288 B:0.296,0.276,0.236 |
| Dtoken_vs_B | hit_at_025 | -0.052 [-0.082, -0.019] | 0.0048 | 0.0006 | A:0.600,0.596,0.560 B:0.560,0.540,0.500 |
| Dtoken_vs_B | hit_at_005 | -0.011 [-0.038, +0.020] | 0.552 | 0.3666 | A:0.204,0.196,0.184 B:0.200,0.176,0.176 |
| Dtoken_vs_B | mean_normalized_l2 | +0.013 [+0.001, +0.026] | 0.0466 | 0.0104 | A:0.359,0.364,0.365 B:0.366,0.371,0.389 |
| Dhook_vs_B | hit_at_010 | -0.005 [-0.041, +0.032] | 0.8337 | 0.724 | A:0.316,0.312,0.288 B:0.316,0.260,0.324 |
| Dhook_vs_B | hit_at_025 | -0.016 [-0.044, +0.013] | 0.3255 | 0.2934 | A:0.600,0.596,0.560 B:0.580,0.548,0.580 |
| Dhook_vs_B | hit_at_005 | -0.000 [-0.024, +0.025] | 1 | 1 | A:0.204,0.196,0.184 B:0.200,0.180,0.204 |
| Dhook_vs_B | mean_normalized_l2 | +0.003 [-0.008, +0.014] | 0.612 | 0.5382 | A:0.359,0.364,0.365 B:0.363,0.374,0.359 |

## Low data — n_train=300

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| B_vs_A | hit_at_010 | +0.067 [+0.029, +0.102] | 0.0017 | 0 | A:0.108,0.172,0.136 B:0.156,0.256,0.204 |
| B_vs_A | hit_at_025 | -0.055 [-0.101, -0.011] | 0.0257 | 0 | A:0.472,0.520,0.476 B:0.428,0.432,0.444 |
| B_vs_A | hit_at_005 | +0.003 [-0.019, +0.023] | 0.9026 | 0.8338 | A:0.048,0.064,0.064 B:0.052,0.076,0.056 |
| B_vs_A | mean_normalized_l2 | +0.138 [+0.111, +0.168] | <=9.999e-05 | 0 | A:0.558,0.528,0.547 B:0.683,0.687,0.677 |
| Dhook_vs_A | hit_at_010 | +0.069 [+0.040, +0.098] | <=9.999e-05 | 0 | A:0.108,0.172,0.136 B:0.184,0.252,0.188 |
| Dhook_vs_A | hit_at_025 | +0.067 [+0.031, +0.102] | 0.0007999 | 0 | A:0.472,0.520,0.476 B:0.548,0.568,0.552 |
| Dhook_vs_A | hit_at_005 | +0.040 [+0.019, +0.062] | 0.0005 | 0 | A:0.048,0.064,0.064 B:0.076,0.148,0.072 |
| Dhook_vs_A | mean_normalized_l2 | -0.085 [-0.111, -0.061] | <=9.999e-05 | 0 | A:0.558,0.528,0.547 B:0.468,0.443,0.466 |

## Low data — n_train=500

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| B_vs_A | hit_at_010 | +0.011 [-0.020, +0.040] | 0.5549 | 0.4314 | A:0.252,0.324,0.368 B:0.252,0.352,0.372 |
| B_vs_A | hit_at_025 | +0.004 [-0.028, +0.034] | 0.8689 | 0.8046 | A:0.416,0.552,0.540 B:0.408,0.568,0.544 |
| B_vs_A | hit_at_005 | +0.028 [-0.005, +0.064] | 0.1394 | 0.0336 | A:0.160,0.228,0.232 B:0.180,0.256,0.268 |
| B_vs_A | mean_normalized_l2 | +0.027 [+0.010, +0.045] | 0.0031 | 0.0018 | A:0.519,0.461,0.471 B:0.561,0.492,0.479 |
| Dhook_vs_A | hit_at_010 | +0.001 [-0.032, +0.033] | 1 | 0.963 | A:0.252,0.324,0.368 B:0.308,0.312,0.328 |
| Dhook_vs_A | hit_at_025 | +0.059 [+0.030, +0.085] | 0.0006999 | 0 | A:0.416,0.552,0.540 B:0.540,0.572,0.572 |
| Dhook_vs_A | hit_at_005 | -0.011 [-0.038, +0.016] | 0.5095 | 0.4776 | A:0.160,0.228,0.232 B:0.200,0.188,0.200 |
| Dhook_vs_A | mean_normalized_l2 | -0.068 [-0.085, -0.049] | <=9.999e-05 | 0 | A:0.519,0.461,0.471 B:0.422,0.417,0.408 |

## Low data — n_train=800

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| B_vs_A | hit_at_010 | -0.040 [-0.077, -0.006] | 0.0367 | 0.0082 | A:0.280,0.264,0.240 B:0.256,0.188,0.220 |
| B_vs_A | hit_at_025 | -0.039 [-0.081, +0.001] | 0.09279 | 0.0056 | A:0.536,0.552,0.548 B:0.524,0.472,0.524 |
| B_vs_A | hit_at_005 | -0.040 [-0.071, -0.012] | 0.0139 | 0 | A:0.116,0.108,0.108 B:0.088,0.048,0.076 |
| B_vs_A | mean_normalized_l2 | +0.047 [+0.031, +0.064] | <=9.999e-05 | 0 | A:0.493,0.483,0.490 B:0.537,0.557,0.515 |
| Dhook_vs_A | hit_at_010 | +0.048 [+0.010, +0.085] | 0.0284 | 0.003 | A:0.280,0.264,0.240 B:0.356,0.276,0.296 |
| Dhook_vs_A | hit_at_025 | +0.064 [+0.009, +0.111] | 0.0257 | 0 | A:0.536,0.552,0.548 B:0.624,0.600,0.604 |
| Dhook_vs_A | hit_at_005 | +0.019 [-0.007, +0.042] | 0.205 | 0.147 | A:0.116,0.108,0.108 B:0.156,0.124,0.108 |
| Dhook_vs_A | mean_normalized_l2 | -0.074 [-0.105, -0.039] | 0.0002 | 0.0002 | A:0.493,0.483,0.490 B:0.393,0.421,0.431 |

## Scale — n_train=2500 (single seed; descriptive)

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| B_vs_A | hit_at_010 | +0.036 [+0.009, +0.063] | 0.0363 | 0.0304 | A:0.204 B:0.240 |
| B_vs_A | hit_at_025 | +0.072 [+0.027, +0.116] | 0.008499 | 0.0016 | A:0.436 B:0.508 |
| B_vs_A | hit_at_005 | +0.016 [-0.015, +0.047] | 0.459 | 0.352 | A:0.140 B:0.156 |
| B_vs_A | mean_normalized_l2 | -0.024 [-0.040, -0.009] | 0.0044 | 0.0092 | A:0.428 B:0.404 |
| Dhook_vs_A | hit_at_010 | +0.076 [+0.041, +0.114] | 0.0006999 | 0 | A:0.204 B:0.280 |
| Dhook_vs_A | hit_at_025 | +0.204 [+0.153, +0.253] | <=9.999e-05 | 0 | A:0.436 B:0.640 |
| Dhook_vs_A | hit_at_005 | +0.044 [+0.011, +0.079] | 0.0373 | 0.0092 | A:0.140 B:0.184 |
| Dhook_vs_A | mean_normalized_l2 | -0.082 [-0.105, -0.060] | <=9.999e-05 | 0 | A:0.428 B:0.346 |

## Scale — n_train=5000 (single seed; descriptive)

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| B_vs_A | hit_at_010 | +0.016 [-0.033, +0.064] | 0.6421 | 0.561 | A:0.368 B:0.384 |
| B_vs_A | hit_at_025 | -0.044 [-0.097, +0.004] | 0.1664 | 0.0362 | A:0.620 B:0.576 |
| B_vs_A | hit_at_005 | +0.000 [-0.038, +0.035] | 1 | 1 | A:0.228 B:0.228 |
| B_vs_A | mean_normalized_l2 | +0.016 [-0.000, +0.034] | 0.09619 | 0.0308 | A:0.372 B:0.388 |
| Dhook_vs_A | hit_at_010 | +0.028 [-0.031, +0.083] | 0.4377 | 0.2854 | A:0.368 B:0.396 |
| Dhook_vs_A | hit_at_025 | +0.024 [-0.019, +0.068] | 0.3954 | 0.257 | A:0.620 B:0.644 |
| Dhook_vs_A | hit_at_005 | +0.000 [-0.053, +0.049] | 1 | 1 | A:0.228 B:0.228 |
| Dhook_vs_A | mean_normalized_l2 | -0.024 [-0.042, -0.007] | 0.0104 | 0.0002 | A:0.372 B:0.348 |

## End-to-end pipeline — n_train=1200

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| predicted_vs_flatA | hit_at_010 | +0.016 [-0.017, +0.052] | 0.424 | 0.2348 | A:0.280,0.256,0.228 B:0.288,0.248,0.276 |
| predicted_vs_flatA | hit_at_025 | +0.031 [-0.005, +0.064] | 0.1121 | 0.042 | A:0.532,0.528,0.484 B:0.572,0.516,0.548 |
| predicted_vs_flatA | hit_at_005 | +0.017 [-0.009, +0.044] | 0.2627 | 0.1162 | A:0.156,0.172,0.144 B:0.176,0.164,0.184 |
| predicted_vs_flatA | mean_normalized_l2 | -0.012 [-0.026, +0.003] | 0.1139 | 0.0542 | A:0.391,0.381,0.404 B:0.374,0.391,0.375 |
| oracle_vs_flatA | hit_at_010 | +0.037 [+0.005, +0.076] | 0.05449 | 0.0048 | A:0.280,0.256,0.228 B:0.308,0.260,0.308 |
| oracle_vs_flatA | hit_at_025 | +0.053 [+0.020, +0.086] | 0.005 | 0.0004 | A:0.532,0.528,0.484 B:0.604,0.524,0.576 |
| oracle_vs_flatA | hit_at_005 | +0.031 [+0.007, +0.056] | 0.0283 | 0.0058 | A:0.156,0.172,0.144 B:0.192,0.172,0.200 |
| oracle_vs_flatA | mean_normalized_l2 | -0.026 [-0.041, -0.013] | 0.0007999 | 0 | A:0.391,0.381,0.404 B:0.359,0.377,0.361 |
| predicted_vs_oracle | hit_at_010 | -0.021 [-0.041, -0.003] | 0.05149 | 0.006 | A:0.308,0.260,0.308 B:0.288,0.248,0.276 |
| predicted_vs_oracle | hit_at_025 | -0.023 [-0.046, +0.000] | 0.07879 | 0.0034 | A:0.604,0.524,0.576 B:0.572,0.516,0.548 |
| predicted_vs_oracle | hit_at_005 | -0.013 [-0.028, -0.001] | 0.08639 | 0.0058 | A:0.192,0.172,0.200 B:0.176,0.164,0.184 |
| predicted_vs_oracle | mean_normalized_l2 | +0.014 [+0.004, +0.025] | 0.0105 | 0.0002 | A:0.359,0.377,0.361 B:0.374,0.391,0.375 |

## D-token causal-use sensitivity — n1200

| contrast | metric | Δ (episode-clustered, 95% CI) | perm p | legacy pooled p | per-seed Δ means |
|---|---|---|---|---|---|
| wrong_vs_gold | hit_at_010 | -0.192 [-0.233, -0.149] | <=9.999e-05 | 0 | A:0.280,0.300,0.248 B:0.100,0.084,0.068 |
| wrong_vs_gold | hit_at_025 | -0.329 [-0.372, -0.284] | <=9.999e-05 | 0 | A:0.528,0.556,0.504 B:0.244,0.176,0.180 |
| wrong_vs_gold | hit_at_005 | -0.168 [-0.193, -0.144] | <=9.999e-05 | 0 | A:0.184,0.188,0.176 B:0.012,0.016,0.016 |
| wrong_vs_gold | mean_normalized_l2 | +0.346 [+0.318, +0.371] | <=9.999e-05 | 0 | A:0.370,0.366,0.388 B:0.603,0.776,0.784 |
| zero_vs_gold | hit_at_010 | -0.093 [-0.128, -0.059] | <=9.999e-05 | 0 | A:0.280,0.300,0.248 B:0.104,0.172,0.272 |
| zero_vs_gold | hit_at_025 | -0.112 [-0.145, -0.079] | <=9.999e-05 | 0 | A:0.528,0.556,0.504 B:0.272,0.444,0.536 |
| zero_vs_gold | hit_at_005 | -0.096 [-0.117, -0.075] | <=9.999e-05 | 0 | A:0.184,0.188,0.176 B:0.016,0.092,0.152 |
| zero_vs_gold | mean_normalized_l2 | +0.118 [+0.104, +0.131] | <=9.999e-05 | 0 | A:0.370,0.366,0.388 B:0.580,0.468,0.429 |

## Synthetic-null validation of the test itself

- Fixed-checkpoint population null (the estimand we test): clustered reject rate **0.051** at α=0.05 (MC SE 0.007); legacy pooled test rejects **0.212** — the pooled routine is anti-conservative exactly as suspected.
- Procedure-level null with run-to-run offsets: fixed-checkpoint tests (ours included) reject at 0.552 — why we report per-seed effects and make no training-procedure significance claim from 3 seeds.
- Power at a Δ=0.03 (mean-L2) true effect: 0.527.

Numbers here supersede every previously quoted pooled p-value (including the retracted p=0.0002 and any p=0).
