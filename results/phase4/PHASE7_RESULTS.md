# Phase 7 — strengthening experiments (mechanism + regime)

Three follow-ups that turn the Phase 6 "nuanced/negative" result into a sharper,
mechanistic story. None of these touch or discard prior data — they add
decompositions and new regimes. All on AITW Stage 2 grounding, Qwen2-VL-2B + LoRA.

1. **Compounding-error / per-action-type decomposition** — *where* does
   conditioning help? (DONE)
2. **Low-data sweep** — does the advantage grow as data shrinks? (IN FLIGHT)
3. **Embedding causal-use test** — is the learned action embedding actually
   used at inference, or ignored? (IN FLIGHT)

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

## Result 7.2 — low-data sweep (IN FLIGHT)

Extending the scaling curve down to n_train ∈ {300, 500, 800} (3 seeds, A/B/D-hook)
to test the *low-data prior* prediction: the conditioning advantage should be
*largest* with little data and erode as the flat baseline sees more.
Runs submitted via `train_stage2_lowdata_sweep` (one detached container per
variant, looping n×seed). Numbers + replotted `scaling_curve.png` to follow.

---

## Result 7.3 — embedding causal-use test (IN FLIGHT)

Train D-token, then evaluate three ways on the same val set:
- **gold**: true action id (normal),
- **wrong**: each example fed a *different* valid action id (cyclic permutation),
- **zero**: action embedding table zeroed for the eval.

If gold ≫ wrong/zero, the learned embedding is *causally used* at inference (the
refutation is "used but not superior to an aux loss"); if gold ≈ wrong ≈ zero,
the model *ignores* the slot (the refutation is "the conditioning path is
inert"). Submitted via `train_stage2_Dtoken --causal-eval` (3 seeds). Numbers to
follow.

---

## Cost
Phase 7 adds ~$3–4 of L4 time (low-data runs are small; causal runs are eval-only
on top of training). Cumulative project spend tracked in README.
