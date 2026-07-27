# Phase 9 — confirmatory results on the untouched, goal-disjoint test set

Prespecified in PRESPEC.md (committed before test access). Primary metric hit@0.10, episode-clustered paired test, Holm over the four primary contrasts.

| contrast | Δ hit@0.10 (95% CI) | perm p | Holm p | per-seed means (A) | per-seed means (B) |
|---|---|---|---|---|---|
| P1_B_vs_A | +0.011 [-0.005, +0.029] | 0.2303 | 0.309 | 0.225, 0.195, 0.237 | 0.235, 0.230, 0.225 |
| P2_Dhook_vs_A | +0.057 [+0.035, +0.079] | <1e-4 | 0.0004 | 0.225, 0.195, 0.237 | 0.277, 0.280, 0.272 |
| P3_Dtoken_vs_B | +0.035 [+0.017, +0.054] | 0.0008999 | 0.0027 | 0.235, 0.230, 0.225 | 0.267, 0.260, 0.268 |
| P4_e2e_pred_vs_A | +0.015 [-0.004, +0.034] | 0.1545 | 0.309 | 0.225, 0.195, 0.237 | 0.228, 0.218, 0.255 |

## Secondary contrasts (test)

| contrast | metric | Δ (95% CI) | perm p |
|---|---|---|---|
| C_vs_A | hit_at_010 | +0.000 [-0.022, +0.021] | 1 |
| C_vs_A | hit_at_025 | +0.025 [-0.017, +0.067] | 0.3355 |
| C_vs_A | hit_at_005 | -0.004 [-0.021, +0.011] | 0.6369 |
| C_vs_A | mean_normalized_l2 | +0.048 [+0.002, +0.108] | 0.07749 |
| Dtoken_vs_A | hit_at_010 | +0.046 [+0.027, +0.067] | 0.0002 |
| Dtoken_vs_A | hit_at_025 | +0.094 [+0.067, +0.121] | <1e-4 |
| Dtoken_vs_A | hit_at_005 | +0.021 [+0.009, +0.034] | 0.0025 |
| Dtoken_vs_A | mean_normalized_l2 | -0.062 [-0.081, -0.044] | <1e-4 |
| Dhook_vs_B | hit_at_010 | +0.046 [+0.026, +0.066] | <1e-4 |
| Dhook_vs_B | hit_at_025 | +0.108 [+0.067, +0.149] | <1e-4 |
| Dhook_vs_B | hit_at_005 | +0.021 [+0.006, +0.036] | 0.0104 |
| Dhook_vs_B | mean_normalized_l2 | -0.069 [-0.090, -0.049] | <1e-4 |
| e2e_pred_vs_oracle | hit_at_010 | -0.048 [-0.061, -0.035] | <1e-4 |
| e2e_pred_vs_oracle | hit_at_025 | -0.079 [-0.108, -0.050] | <1e-4 |
| e2e_pred_vs_oracle | hit_at_005 | -0.036 [-0.047, -0.026] | <1e-4 |
| e2e_pred_vs_oracle | mean_normalized_l2 | +0.056 [+0.041, +0.072] | <1e-4 |
| control_B_vs_A | hit_at_010 | +0.013 [-0.013, +0.041] | 0.3639 |
| control_B_vs_A | hit_at_025 | -0.005 [-0.028, +0.021] | 0.7357 |
| control_B_vs_A | hit_at_005 | +0.026 [+0.003, +0.051] | 0.0451 |
| control_B_vs_A | mean_normalized_l2 | -0.004 [-0.014, +0.006] | 0.452 |
| control_Dhook_vs_A | hit_at_010 | +0.047 [+0.010, +0.079] | 0.0327 |
| control_Dhook_vs_A | hit_at_025 | +0.105 [+0.028, +0.194] | 0.0027 |
| control_Dhook_vs_A | hit_at_005 | +0.032 [+0.002, +0.060] | 0.06189 |
| control_Dhook_vs_A | mean_normalized_l2 | -0.055 [-0.099, -0.016] | 0.0005999 |
| control_C_vs_A | hit_at_010 | -0.067 [-0.100, -0.039] | <1e-4 |
| control_C_vs_A | hit_at_025 | -0.097 [-0.150, -0.054] | <1e-4 |
| control_C_vs_A | hit_at_005 | -0.037 [-0.063, -0.015] | 0.0016 |
| control_C_vs_A | mean_normalized_l2 | +0.187 [+0.098, +0.271] | <1e-4 |
| lowdata300_B_vs_A | hit_at_010 | +0.038 [+0.013, +0.064] | 0.0046 |
| lowdata300_B_vs_A | hit_at_025 | +0.087 [+0.031, +0.145] | 0.002 |
| lowdata300_B_vs_A | hit_at_005 | +0.002 [-0.010, +0.013] | 0.7849 |
| lowdata300_B_vs_A | mean_normalized_l2 | +0.004 [-0.031, +0.034] | 0.9259 |
| lowdata300_Dhook_vs_A | hit_at_010 | +0.049 [+0.025, +0.073] | 0.0006999 |
| lowdata300_Dhook_vs_A | hit_at_025 | +0.137 [+0.101, +0.171] | <1e-4 |
| lowdata300_Dhook_vs_A | hit_at_005 | +0.031 [+0.015, +0.048] | 0.0005999 |
| lowdata300_Dhook_vs_A | mean_normalized_l2 | -0.112 [-0.134, -0.090] | <1e-4 |
| lowdata500_B_vs_A | hit_at_010 | -0.004 [-0.027, +0.018] | 0.755 |
| lowdata500_B_vs_A | hit_at_025 | -0.007 [-0.036, +0.023] | 0.7466 |
| lowdata500_B_vs_A | hit_at_005 | -0.005 [-0.020, +0.009] | 0.5574 |
| lowdata500_B_vs_A | mean_normalized_l2 | +0.015 [-0.006, +0.034] | 0.1327 |
| lowdata500_Dhook_vs_A | hit_at_010 | +0.016 [-0.008, +0.039] | 0.2278 |
| lowdata500_Dhook_vs_A | hit_at_025 | +0.050 [+0.010, +0.091] | 0.0229 |
| lowdata500_Dhook_vs_A | hit_at_005 | +0.016 [+0.001, +0.029] | 0.06539 |
| lowdata500_Dhook_vs_A | mean_normalized_l2 | -0.064 [-0.087, -0.042] | <1e-4 |
| lowdata800_B_vs_A | hit_at_010 | +0.003 [-0.014, +0.020] | 0.787 |
| lowdata800_B_vs_A | hit_at_025 | +0.004 [-0.014, +0.022] | 0.6698 |
| lowdata800_B_vs_A | hit_at_005 | +0.018 [+0.002, +0.035] | 0.0424 |
| lowdata800_B_vs_A | mean_normalized_l2 | +0.025 [+0.015, +0.035] | <1e-4 |
| lowdata800_Dhook_vs_A | hit_at_010 | +0.036 [+0.015, +0.057] | 0.0014 |
| lowdata800_Dhook_vs_A | hit_at_025 | +0.074 [+0.049, +0.101] | <1e-4 |
| lowdata800_Dhook_vs_A | hit_at_005 | +0.034 [+0.019, +0.052] | <1e-4 |
| lowdata800_Dhook_vs_A | mean_normalized_l2 | -0.053 [-0.067, -0.039] | <1e-4 |

See rerun_analysis.json for descriptive per-run values, per-class breakdowns, parse rates, and the D-token causal sensitivity (val).
