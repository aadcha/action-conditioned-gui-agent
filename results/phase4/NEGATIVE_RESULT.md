# Phase 5 negative result — variant D underperforms variant A

Matched-compute comparison across two training distributions. Variant D
(action-type-conditioned, the project's central architectural intervention)
**statistically and consistently underperforms** variant A (flat baseline)
on grounding hit@0.10. The negative result is robust across seeds.

## Numbers

All on AITW DUAL_POINT actions, Qwen2-VL-2B + LoRA, lr=2e-5, 2 epochs,
batch_size=1.

### taps_only (n_train=500, 3 seeds each)

|  | A (flat) | D (action-conditioned) | (D - A) | n_sigma |
|---|---|---|---|---|
| hit@0.05 | 0.167 ± 0.012 | 0.137 ± 0.015 | -0.030 | 1.57 |
| **hit@0.10** | **0.363 ± 0.040** | **0.320 ± 0.036** | **-0.043** | **0.80** |
| hit@0.25 | 0.667 ± 0.021 | 0.623 ± 0.021 | -0.043 | 1.47 |
| mean norm L2 | 0.236 ± 0.011 | 0.271 ± 0.009 | +0.034 (A wins) | 2.40 |

### taps_and_swipes (n_train=1000, 3 seeds each — FINAL)

|  | A (flat) | D (action-conditioned) | (D - A) | n_sigma |
|---|---|---|---|---|
| hit@0.05 | 0.233 ± 0.010 | 0.163 ± 0.003 | -0.070 | 6.85 |
| **hit@0.10** | **0.390 ± 0.010** | **0.288 ± 0.008** | **-0.102** | **8.08** |
| hit@0.25 | 0.727 ± 0.023 | 0.612 ± 0.020 | -0.115 | 5.50 |
| mean norm L2 | 0.184 ± 0.002 | 0.215 ± 0.006 | +0.031 (A wins) | 5.05 |

The **multi-action** result is the experimentally clean one — the one designed
to give variant D's action-conditioning signal something to do. It is
exactly there that variant A pulls ahead the most.

## Why this is a real result, not a bug or fluke

1. **Robust across seeds.** All 3 A seeds beat the best D seed on hit@0.10
   in the multi-action setting. With n=200 val examples per run, std across
   seeds is ~0.01 — meaningfully smaller than the ~0.10 gap.
2. **Direction is consistent.** A beats D on every hit@r threshold and on
   mean normalized L2, across both training distributions. There is no
   metric where D wins.
3. **Sanity check passes.** Variant A's own headline numbers move in the
   expected direction (better with more data, better with multi-action
   training) — both variants are doing something sensible. D just does it
   worse.
4. **Mechanism is plausible.** Variant D adds a learned 12,288-parameter
   embedding table that needs gradient signal to specialize. With 2 epochs
   on 1000 examples (~500 per active class), the embedding rows may not
   converge to meaningful per-class vectors. Meanwhile the slot token at
   a fixed position introduces a perturbation to the prompt structure that
   the LoRA adapter has to compensate for.
5. **The user's plan anticipated this case.** From the interpretation rubric
   in the project plan, the `D ≤ A` outcome is explicitly listed as a
   possible result. The right response is "write it up as a negative result
   with diagnostics," not "rework until D wins."

## What goes in the paper

The contribution becomes a **failure-analysis paper with diagnostics**, not
a SOTA claim. Concretely:

- **Claim**: Action-type conditioning via the marker-replacement mechanism
  proposed in this project does not improve grounding on AITW.
- **Evidence**: 6-seed ablation A vs D across two training distributions
  (one degenerate-action, one mixed). D consistently underperforms A.
- **Diagnostics** (the section that makes the paper publishable):
  1. The slot-token perturbation effect — does dropping the slot token
     from the prompt but keeping the embedding-as-prefix recover the gap?
  2. The action-embedding-norm question — does scaling action_embeddings to
     match the base Qwen2-VL embedding norm (~1.0 vs ~0.02) help?
  3. The gradient-update-count question — does a longer training schedule
     (5+ epochs, larger n_train) let the embedding specialize?
  4. The variance question — does training data with all 5+ canonical
     action classes (using the `all_with_coords` data mix) change the
     picture? (1-seed runs of all_with_coords are in flight as a smoke test.)

## What to NOT do

- **Don't keep tuning until D wins.** The negative result is real and the
  experimental design is sound. Re-running with different hyperparameters
  until a number above A appears is p-hacking.
- **Don't drop variant D from the paper.** Negative results without the
  failure case are not interesting; the failure case IS the contribution.
- **Don't blame the implementation.** Variant D's pipeline ran end-to-end:
  loss decreases monotonically, parse rate is 100%, both LoRA and the
  action embedding receive gradient. The mechanism is wired correctly.

## Open questions for the writeup

- Is the slot-replacement approach (replace embedding at runtime) materially
  different from adding the action-type as a special token in the natural
  token sequence (variant B-ish)? A direct head-to-head would isolate this.
- Would Qwen2-VL-7B (vs 2B) close the gap? Larger LMs have more capacity
  to use a learned class-conditional embedding.
- Does the negative result generalize off AITW? Mind2Web has only 2
  canonical classes (click + type) and the project's Phase 2 result
  already showed vision delta = +0.009 macro-F1 — also small. Mind2Web is
  not a great test of D either.

## What we still need

- [x] 3rd D seed for multi-action — **landed**, headline updated above.
- [ ] 1-seed `all_with_coords` smoke (3 active classes: click+scroll+type) — in flight.
- [ ] 3-seed `Dfrozen` diagnostic (action_embeddings frozen at random init) — in flight.
  - If Dfrozen ≈ D: the slot disrupts the prompt regardless of embedding training.
  - If Dfrozen ≪ D: the embedding *was* doing useful work; D just couldn't
    learn it fast enough.
  - If Dfrozen ≈ A: shouldn't happen given the slot disruption, but if it
    does, the slot at this position is benign and our embedding training
    actively harmed things.

The headline result (taps_and_swipes 8σ gap) will not change.
