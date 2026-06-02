# Phase 6 — Final ablation, end-to-end pipeline, and the hypothesis verdict

This consolidates the full controlled study. All Stage 2 grounding on AITW,
Qwen2-VL-2B + LoRA, 2 epochs, 3 seeds, paired-bootstrap-grade comparisons.
Supersedes the partial PHASE6_ABLATION_ABCD.md (which lacked D-token).

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
| C (hard routing) | 0.285 ± 0.033 | 0.555 | 0.388 |
| D-hook (additive) | 0.300 ± 0.028 | 0.569 | 0.365 |
| D-token (prepended, the hypothesis) | 0.269 ± 0.025 | 0.533 | 0.375 |

Paired bootstrap (750 units) confirmed **B and D-hook each beat A** on all
metrics (p < 0.002). **Ranking: B ≈ D-hook > C ≳ D-token ≳ A.**

## Result 2 — control, taps_and_swipes (action type NOT spatially informative)

| variant | hit@0.10 | hit@0.25 | mean L2 ↓ |
|---|---|---|---|
| A (flat) | 0.390 ± 0.008 | 0.727 | 0.184 |
| B (aux loss) | 0.368 ± 0.009 | 0.730 | 0.173 |
| C (hard routing) | 0.337 ± 0.040 | 0.659 | 0.229 |
| D-hook (additive) | 0.392 ± 0.035 | 0.723 | 0.179 |

Here A is already strong and **nothing beats it**; conditioning is neutral
(D-hook) to mildly harmful (B, C). Exactly as predicted: no action-type→
location signal to exploit when tap and swipe both land anywhere.

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
  **does not beat B; it underperforms it and ties A.**
- The *stronger* the embedding signal injected at a token position, the
  *worse* grounding gets (D-token norm 2.2 < D-hook norm 0.06 in performance;
  raising D-hook's action-embedding LR also hurt). A strong action token at a
  sequence position appears to *compete with* the image/goal for attention
  rather than help.

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
> action-type embedding our proposal advocated does not beat it (and a strong
> embedding signal hurts). We also document a M-RoPE implementation pitfall
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
modal run modal_app.py::list_stage2_runs
uv run python scripts/p6_ablation_table.py --mix all_with_coords
```
