# Phase 6 — Final ablation, end-to-end pipeline, and the hypothesis verdict

This consolidates the full controlled study. All Stage 2 grounding on AITW,
Qwen2-VL-2B + LoRA, 2 epochs, 3 seeds. Paired-bootstrap comparisons are
available for the all_with_coords setting, where every variant has aligned
per-example distances. The taps_and_swipes control is reported as 3-seed
mean/stdev because the A and D-hook control runs predate per-example logging.
This file supersedes the partial PHASE6_ABLATION_ABCD.md (which lacked D-token).

## The variants

| | mechanism | action signal at... |
|---|---|---|
| **A** | flat baseline, plain prompt → coords | nowhere |
| **B** | + auxiliary action-type classification head (joint loss) | training only |
| **C** | emit `{word} (x,y)`, force gold word at decode | output stream (hard) |
| **D-hook** | additive action embedding broadcast to all tokens | input (diffuse) |
| **D-token** | dedicated `<|action_slot|>` token, embedding replaced, M-RoPE-correct | input (routable) — the literal hypothesis |

## Result 1 — full ablation, all_with_coords (action type spatially informative)

3 seeds each, mean ± stdev:

| variant | hit@0.10 | hit@0.25 | mean L2 ↓ |
|---|---|---|---|
| A (flat) | 0.255 ± 0.021 | 0.515 | 0.392 |
| **B (aux loss)** | **0.305 ± 0.012** | **0.585** | **0.362** |
| C (hard routing) | 0.279 ± 0.032 | 0.543 | 0.412 |
| D-hook (additive) | 0.300 ± 0.028 | 0.569 | 0.365 |
| D-token (prepended, the hypothesis) | 0.269 ± 0.025 | 0.533 | 0.375 |

Paired bootstrap (750 units) confirmed **B and D-hook each beat A** on all
metrics (p < 0.002). C metrics above count parse failures as misses via the
same per-example sentinel-distance convention used by the bootstrap. **Ranking:
B ≈ D-hook > D-token ≳ C > A**, with D-token still below B/D-hook on the
headline hit@0.10 and hit@0.25 metrics.

D-token's comparison to A is mixed rather than a clean win: it improves hit@0.05
(+0.027, p=0.003) and mean L2 (-0.017, p<0.001), but not hit@0.10 (+0.015,
ns) or hit@0.25 (+0.019, ns). Against B, it is significantly worse on
hit@0.10, hit@0.25, and mean L2.

## Result 2 — control, taps_and_swipes (action type NOT spatially informative)

| variant | hit@0.10 | hit@0.25 | mean L2 ↓ |
|---|---|---|---|
| A (flat) | 0.390 ± 0.008 | 0.727 | 0.184 |
| B (aux loss) | 0.368 ± 0.009 | 0.730 | 0.173 |
| C (hard routing) | 0.325 ± 0.032 | 0.635 | 0.271 |
| D-hook (additive) | 0.392 ± 0.035 | 0.723 | 0.179 |

Here A is already strong and no conditioned mechanism clearly beats it:
D-hook is neutral, B trades slightly lower hit@0.10 for slightly lower mean L2,
and C is worse after counting parse failures. Exactly as predicted: no
action-type→location signal to exploit when tap and swipe both land anywhere.

## Result 3 — end-to-end pipeline (Stage 1 predicted → Stage 2), all_with_coords

3 seeds. Stage 1 = frozen-backbone MLP trained in-job (3-class):

| | value |
|---|---|
| Stage 1 val accuracy | 0.843 |
| Stage 1 val macro-F1 | 0.795 |
| Oracle (gold type) hit@0.10 | 0.292 |
| **Predicted (Stage 1) hit@0.10** | **0.271** |
| oracle→predicted gap | 0.021 ± 0.008 |
| flat A baseline | 0.255 |

→ The real two-stage pipeline (predicted types) **beats flat A** (0.271 vs
0.255), and the oracle→predicted gap is small (~0.02) because Stage 1 is
accurate. Classifier error is not the bottleneck.

## Result 4 — data-scaling curve (does the advantage hold at scale?)

A/B/D-hook at n_train ∈ {1200, 2500, 5000} on all_with_coords (n=1200 is the
3-seed mean; 2500/5000 are single seed). At each n, A/B/D share a val slice, so
the **delta at each n is apples-to-apples**; absolute A varies across n because
the val set differs (a caveat — the curve is directional, not a clean monotone).

Conditioning advantage (Δ vs A), hit@0.10:

| n_train | (B − A) | (D − A) |
|---|---|---|
| 1,200 | +0.051 | +0.045 |
| 2,500 | +0.036 | +0.076 (noisy) |
| 5,000 | **+0.016** | **+0.028** |

And hit@0.25: (B − A) goes **+0.071 → +0.072 → −0.044** at n=5000.

→ The **B − A advantage shrinks monotonically with data**, and even reverses on
hit@0.25 at n=5000. The D − A line is noisy at n=2500 (single seed, harder val
set) but its 1200→5000 endpoints also shrink. This **confirms the prediction**
that action-type conditioning is a **low-data prior the flat baseline absorbs at
scale** — the benefit is biggest with little data and erodes as A sees more.
Figure: `results/phase4/scaling_curve.png`.

## Result 5 — second benchmark: Mind2Web grounding (cross-dataset control)

Stage 2 grounding on **Multimodal-Mind2Web** (a proposal-named dataset). Target =
gold element bbox center. hit@bbox = predicted point inside the gold element box
(Mind2Web-native element grounding). A vs D-hook, 2 seeds (n_train=1000):

| variant | hit@0.10 | hit@0.25 | hit@bbox | mean L2 ↓ |
|---|---|---|---|---|
| A (flat) | 0.370 ± 0.030 | 0.660 ± 0.024 | 0.086 ± 0.010 | 0.223 |
| D-hook | 0.364 ± 0.028 | 0.678 ± 0.046 | 0.086 ± 0.014 | 0.227 |
| (D − A) | −0.006 | +0.018 | 0.000 | — |

→ **D-hook ≈ A (slightly worse).** Mind2Web is click/type-dominated (action type
uninformative about location), so conditioning does not help — a **clean
cross-dataset replication of the AITW taps_and_swipes control**. (hit@bbox is low
~0.07–0.10 because element grounding on full-page screenshots is hard; the A-vs-D
comparison is the point.) This is a *second benchmark* and a *second control*.

## The hypothesis verdict

The project's hypothesis has two levels. The controlled study cleanly separates them:

**Broad thesis — SUPPORTED.** Action-type supervision improves screenshot
grounding *when action type is spatially informative*. B and D-hook beat the
flat baseline significantly on all_with_coords; the effect vanishes on the
spatially-uninformative control. The two-stage pipeline works end-to-end.

**Specific architectural claim — REFUTED.** The proposal's central bet was a
*learned action-type embedding prepended to the instruction stream* ("an
embedding carries richer per-class priors than the text token"). The data say
this mechanism is **not** what matters:
- A plain **auxiliary loss (B)** — no inference-time conditioning at all — is
  the **best** conditioned variant.
- The **literal hypothesized architecture (D-token)** — a routable prepended
  embedding, M-RoPE-correct, with the embedding fully developed (norm 2.2) —
  **does not beat B/D-hook on headline grounding.** It improves over A on
  hit@0.05 and mean L2, but not significantly on hit@0.10 or hit@0.25.
- The *stronger* the embedding signal injected at a token position, the
  less competitive it is with B/D-hook on the headline thresholds (D-token norm
  2.2 trails D-hook norm 0.06 on hit@0.10, hit@0.25, and mean L2; raising
  D-hook's action-embedding LR also hurt). A strong action token at a sequence
  position appears to *compete with* the image/goal for attention rather than
  help.

We ruled out the obvious "implementation was weak" escape: we found and fixed
a real M-RoPE bug (the original `inputs_embeds` injection), then built the
embedding three ways (additive, additive+high-LR, routable prepended token)
and none beat the auxiliary loss.

### Honest one-paragraph framing for the report
> We factor action-type prediction from spatial grounding and show, via a
> controlled five-variant ablation with paired-bootstrap significance, that
> action-type supervision improves grounding **only where the action type is
> spatially predictive of the target** — and that **the simplest mechanism, an
> auxiliary training loss, captures the full benefit**, while the learned
> action-type embedding our proposal advocated does not beat B/D-hook on
> headline grounding (and a strong embedding signal is not the best mechanism).
> We also document a M-RoPE implementation pitfall
> that silently degrades grounding (an 8σ false-negative if uncaught), and an
> end-to-end pipeline whose predicted-type degradation over the oracle is small
> (~0.02 hit@0.10) because the Stage 1 classifier is accurate. This is a
> failure-analysis contribution: *when* action conditioning helps, *how much*,
> and *which mechanism* — not a SOTA architecture claim.

## Cost
Modal spend through Phase 6: **~$45 of $200**.

## Reproduction
```bash
# full ablation (all_with_coords + taps_and_swipes)
for v in A B C; do for s in 42 43 44; do modal run --detach modal_app.py::train_stage2_variant$v --seed $s --data-mix all_with_coords; done; done
for s in 42 43 44; do modal run --detach modal_app.py::train_stage2_Dhook  --seed $s --data-mix all_with_coords; done
for s in 42 43 44; do modal run --detach modal_app.py::train_stage2_Dtoken --seed $s --data-mix all_with_coords; done
for s in 42 43 44; do modal run --detach modal_app.py::train_stage2_e2e    --seed $s --data-mix all_with_coords; done
for v in A B C; do for s in 42 43 44; do modal run --detach modal_app.py::train_stage2_variant$v --seed $s --data-mix taps_and_swipes --n-train 1000 --n-val 200; done; done
for s in 42 43 44; do modal run --detach modal_app.py::train_stage2_Dhook --seed $s --data-mix taps_and_swipes --n-train 1000 --n-val 200; done
modal run modal_app.py::list_stage2_runs
uv run python scripts/p6_ablation_table.py --mix all_with_coords
```
