# Phase 7 — strengthening experiments (mechanism + regime)

Three follow-ups that turn the Phase 6 "nuanced/negative" result into a sharper,
mechanistic story. None of these touch or discard prior data — they add
decompositions and new regimes. All on AITW Stage 2 grounding, Qwen2-VL-2B + LoRA.

1. **Compounding-error / per-action-type decomposition** — *where* does
   conditioning help? (DONE)
2. **Low-data sweep** — does the advantage grow as data shrinks? (DONE)
3. **Embedding causal-use test** — is the learned action embedding actually
   used at inference, or ignored? (DONE)

---

## Result 7.1 — the conditioning advantage is concentrated in `click`

Breaking the all_with_coords (n_train=1200) grounding metrics down BY ACTION
TYPE, averaged over seeds (A/D-hook 5 seeds; B/C/D-token 3 seeds). Source:
`final_val_metrics.per_class` already logged in every run. Script:
`scripts/p7_compounding_error.py`; figure: `compounding_error_per_class.png`.

**hit@0.10 by action type:**

| action | A (flat) | B (aux) | D-hook | D-token | C (hard) |
|---|---|---|---|---|---|
| **click** (68% of val) | 0.253 | **0.373** | 0.311 | 0.318 | **0.375** |
| **scroll** (18%) | 0.313 | 0.290 | 0.352 | 0.297 | 0.169 |
| **type** (14%) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Conditioning advantage vs flat A (Δ hit@0.10):**

| action | B − A | D-hook − A | D-token − A | C − A |
|---|---|---|---|---|
| click | **+0.120** | +0.058 | +0.064 | **+0.122** |
| scroll | −0.023 | +0.039 | −0.016 | **−0.144** |
| type | 0.000 | 0.000 | 0.000 | 0.000 |

### What this says

1. **The overall conditioning win is a `click` win.** Click is 68% of the val
   set, and that is where essentially all of the B/C advantage lives (+0.12
   hit@0.10). Knowing the action is a *click* — a precise target on a button /
   icon / link — rather than hedging across click/scroll/type, sharpens the
   grounded point substantially. This is a measurable signature of the
   project's motivating *compounding-error* mechanism (commit to the wrong
   action semantics → ground coherently but wrongly): resolve the action
   identity first and the precise-target class is grounded far better.

2. **`type` is degenerate, not a "type→fields" signal.** Every variant scores
   **0.000** hit@0.10 *and* 0.000 hit@0.25 on `type`, with mean normalized L2 =
   **1.414** (= √2, exact corner-to-corner). Zero hits within a quarter-diagonal
   across ~35×(seeds) examples is statistically impossible for real targets —
   AITW `type` events carry a pinned/degenerate touch coordinate, so `type` is
   *ungroundable for any model*. **This corrects the earlier "action type is
   informative via type→fields" claim** in PHASE5/PHASE6: the real signal is
   **click-vs-scroll/type disambiguation**, not field localization. `type`
   contributes only a constant 0.000 drag that lowers every variant's macro
   number equally (which is why it does not affect the *deltas*).

3. **C (hard routing) explains its own mediocrity.** It has the *largest* click
   gain (+0.122) but *destroys* scroll (−0.144): forcing the gold action word as
   a decode prefix helps the dominant precise class but derails the gesture
   class. Net: middling overall. B (aux loss) captures the click gain (+0.120)
   with only a small scroll cost (−0.023) → why B is the best overall variant.

4. **The two embedding variants spread the gain differently.** D-hook is the
   only variant that *improves scroll* (+0.039) as well as click (+0.058); the
   aux loss puts everything on click. So "the information helps" is robust, but
   *how it is distributed across classes* depends on the mechanism.

**Limitation.** This is a per-class decomposition, not a per-example
wrong-type→wrong-location trace (the run JSONs log per-example distance but not
predicted coordinates). The claim is "the grounding gain is action-type-specific
and concentrated in the precise-target class," which the per-class data supports
directly.

---

## Result 7.2 — low-data sweep

Extending the scaling curve down to n_train ∈ {300, 500, 800} (3 seeds, A/B/D-hook)
to test the *low-data prior* prediction: the conditioning advantage should be
*largest* with little data and erode as the flat baseline sees more.
All cells are complete: 3 variants × 3 train sizes × 3 seeds = 27 runs. The
strict completeness check is `scripts/p7_result_audit.py`; raw audit output is
`phase7_result_audit.json`; the replotted curve is `scaling_curve.png`.

**3-seed mean hit rates:**

| n_train | A hit@0.10 | B hit@0.10 | D-hook hit@0.10 | B − A | D-hook − A | A hit@0.25 | B hit@0.25 | D-hook hit@0.25 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.139 | 0.205 | **0.208** | +0.067 | +0.069 | 0.489 | 0.435 | **0.556** |
| 500 | 0.315 | **0.325** | 0.316 | +0.011 | +0.001 | 0.503 | 0.507 | **0.561** |
| 800 | 0.261 | 0.222 | **0.311** | −0.040 | +0.050 | 0.545 | 0.507 | **0.613** |

### What this says

1. **There is no clean monotonic low-data curve.** The largest hit@0.10 gain is
   at n=300, but n=500 is essentially a tie and n=800 swings back toward
   D-hook. Small-data optimization noise is real.
2. **D-hook is the stable low-data conditioned mechanism.** It is best on
   hit@0.25 at every low-data size and has the best mean normalized L2 at every
   low-data size (see `phase7_result_audit.json`). The additive embedding path
   is not just a high-data artifact.
3. **B is not uniformly better in the low-data regime.** The auxiliary loss has
   a strong n=300 hit@0.10 bump and wins n=500 hit@0.10 narrowly, but it loses
   hit@0.25 and mean L2 to D-hook throughout. Phase 6's "B is simplest/best" is
   true at the headline n=1200 setting, not as a universal small-data rule.

---

## Result 7.3 — embedding causal-use test

Train D-token, then evaluate three ways on the same val set:
- **gold**: true action id (normal),
- **wrong**: each example fed a *different* valid action id (cyclic permutation),
- **zero**: action embedding table zeroed for the eval.

All 3 seeds are complete. Aggregation script: `scripts/p7_causal_table.py`;
summary JSON: `causal_use_summary.json`.

| condition | hit@0.05 | hit@0.10 | hit@0.25 | mean L2 |
|---|---:|---:|---:|---:|
| **gold action id** | 0.183 | **0.276** | **0.529** | **0.375** |
| wrong action id | 0.015 | 0.084 | 0.200 | 0.721 |
| zeroed embedding | 0.087 | 0.183 | 0.417 | 0.492 |

**Paired bootstrap over pooled per-example distances:**

| contrast | metric | Δ | 95% CI | p |
|---|---|---:|---:|---:|
| gold − wrong | hit@0.10 | +0.192 | [+0.156, +0.228] | <0.001 |
| gold − wrong | hit@0.25 | +0.329 | [+0.289, +0.369] | <0.001 |
| gold − zero | hit@0.10 | +0.093 | [+0.060, +0.128] | <0.001 |
| gold − zero | hit@0.25 | +0.112 | [+0.077, +0.148] | <0.001 |

### What this says

The learned D-token embedding is **causally used at inference**. Feeding the
wrong action id badly damages grounding, and zeroing the embedding also hurts.
So the D-token refutation is precise: the path is *not inert*; it is used, but
it is not the best way to exploit action supervision compared with B/D-hook at
the headline setting.

---

## Cost
Phase 7 added roughly $6–8 of L4 time after completing the missing cells. The
extra cost came from running each missing low-data cell independently to avoid
another long-container stop, plus two D-token causal reruns.
