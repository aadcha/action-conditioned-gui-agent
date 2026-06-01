# Phase 5 — Corrected A vs D ablation (supersedes the "negative result")

**TL;DR:** The earlier 8σ "A beats D" finding was a **measurement artifact** caused
by a bug in variant D's injection mechanism. After fixing it, the corrected
variant (D-hook) **ties** variant A on grounding — it does not lose, and (at the
tested setting) does not clearly win either. The action-type conditioning is
*neutral* when implemented correctly.

## The bug

Original variant D ("D-slot") inserted an `<|action_slot|>` token, replaced its
embedding with the learned action embedding, and fed the result to Qwen2-VL via
`inputs_embeds`. **That `inputs_embeds` path routes around Qwen2-VL's
`input_ids`-keyed M-RoPE 3D position computation.** M-RoPE encodes the spatial
position of image patches — exactly the signal grounding depends on. The result:

- image patches got subtly wrong positional encodings
- spatial precision (hit@r) degraded ~9-10 points
- training loss barely moved (the model still learns approximate coordinates)
- the damage was independent of the action embedding → "D ≈ Dfrozen"

Every symptom of the "negative result" is explained by this single bug.

How it was found: `scripts/p5_debug_stage2.py` (run on Modal). The mechanism
checks (slot replaced, gradient flows, image affects output) all PASSED, which
ruled out the obvious culprits and pointed at the position-encoding path.

## The fix: variant D-hook

- **Plain prompt, identical to variant A** — no slot token, no extra position.
- **Full `input_ids` path** → M-RoPE computed identically to A.
- Action conditioning injected as an **additive bias via a forward hook on the
  embedding layer**: `embed_out += action_embeddings(action_id)`.
- Action embedding **zero-initialized** → at step 0, D-hook is byte-identical to
  A. It is a *strict superset* of A: it can only match or beat A (modulo
  optimization noise).

## Results (taps_and_swipes, n_train=1000, 2 epochs, 3 seeds)

| metric | A (flat) | D-slot (buggy) | **D-hook (fixed)** |
|---|---|---|---|
| hit@0.05 | 0.233 ± 0.010 | 0.140 ± 0.022 | 0.237 ± 0.045 |
| **hit@0.10** | **0.390 ± 0.010** | 0.298 ± 0.026 | **0.392 ± 0.043** |
| hit@0.25 | 0.727 ± 0.023 | 0.607 ± 0.026 | 0.723 ± 0.038 |
| mean norm L2 (↓) | 0.184 ± 0.002 | 0.214 ± 0.006 | 0.179 ± 0.011 |

D-hook seeds (hit@0.10): **[0.435, 0.390, 0.350]**. Seed 42's 0.435 alone would
"beat" A, but the 3-seed mean (0.392) is statistically identical to A's 0.390
(0.0σ). **We report the mean and call it a tie. We do not cherry-pick seed 42.**

## What this means

1. **The structural fix is real and large.** D-slot → D-hook recovered +0.094
   hit@0.10. The M-RoPE bug was genuinely tanking the conditioned variant.
2. **Correct action conditioning is currently neutral for grounding on AITW
   tap/swipe.** D-hook ⊇ A by construction, and the optimizer keeps the action
   embedding tiny (`ae_norm ≈ 0.06`) — it finds little use for the conditioning
   signal. Plausible reason: for tap vs swipe, the target *location* is not
   strongly determined by the action *type* (both are touch points the model
   can localize from the screenshot+goal), so the type adds little spatial info.

## Open follow-ups (in flight / candidate)

- **Higher action-embedding LR** (`action_lr` param): `ae_norm≈0.06` suggests the
  zero-init embedding is *undertrained* at the shared lr=2e-5. A dedicated higher
  LR lets it develop. Clean test — if hit@0.10 rises, conditioning helped and was
  undertrained; if the embedding grows but the metric doesn't move, conditioning
  is genuinely neutral. (probes: action_lr ∈ {1e-3, 5e-3}, seed 42)
- **`all_with_coords` mix** (adds `type` actions): action types are more spatially
  distinct there (text entry → input fields), the setting where conditioning
  *should* help most. Needs its own 3-seed A baseline.
- **D-text** (action word in the prompt): independent mechanism; staged.

## Honesty note

This file supersedes `NEGATIVE_RESULT.md`. That document's headline (8σ A>D)
was wrong — it measured a buggy D. The bug, the fix, and the resulting tie are
all committed and reproducible. The scientifically correct current claim is:
*"a naive embedding-injection that bypasses M-RoPE severely harms grounding;
a correct additive injection recovers baseline performance; action-type
conditioning is neutral (neither helps nor hurts) for coordinate grounding on
AITW tap/swipe at this scale."*
