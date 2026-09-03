**Summary**
The paper fine-tunes Qwen2-VL-2B with LoRA to emit a coordinate string on Android in the Wild and compares a flat baseline (A) with five ways of supplying the gold action type: auxiliary loss (B), hard routing (C), additive embedding (D-hook), prepended learned token (D-token), and type-as-word (D-text). On a mixed stream containing a "type" class whose target is clamped to (0,0), B, D-hook, and D-text beat A by 5-7 hit@0.10 points; D-token does not. Retraining with the type events removed raises A by four points, and a taps-and-swipes control shows no gain. Interventions on trained models show both learned embeddings are used at inference, and the paper retracts an earlier `inputs_embeds` result attributed to Qwen2-VL's M-RoPE path.

**Strengths**
- Matched-compute, doubly paired design; zero-initialized D-hook makes the A vs. D-hook contrast unusually tight (l.135-136, 155-157).
- Two-level inference with an explicit "established only when both agree" rule (l.176-178), a cluster bootstrap rather than pooled (seed, example) units, and an honest statement that no multiplicity correction is applied.
- Candid about its own history: retracts single-seed claims (l.241-243, 262-264), reports the inverted 6.2 result, reports the end-to-end null (5.3), and runs the type-removed retrain that deflates its own headline (Table 7).
- Internally consistent tables: I recomputed the class-weighted overall hit rates from the per-class rows of Tables 2, 4, and 7 and they reproduce Table 1 and Table 4 totals. The class-mean and click-swap interventions are well-chosen controls.
- The `inputs_embeds` note is accurate as far as it goes; I verified that the cited HF issue #35463 reports the `input_ids` dependence as a crash, as the paper says (l.528-530).

**Weaknesses**
1. **Wrong clustering unit; every bootstrap star is anti-conservative.** Validation slices are consecutive steps from "a few dozen episodes" (l.158-159, 487-489); steps in an episode share the app, the goal text, and near-identical screens. The bootstrap clusters on step (l.172-174), so the effective n is roughly 30-40, not 250. Under a design effect of 2-3, B's headline interval likely still excludes zero, but D-hook and D-text become marginal and most secondary stars (Table 4 class-mean, Table 7 B-click, most of Table 8) would not survive. The limitation paragraph names seed sharing (l.292-293), not episode sharing. Boundary-episode leakage into training is also admitted (l.488-489).
2. **Null results are read as equalities, and these are the headline deployment claims.** (a) "Zeroed, the additive model returns to the baseline's accuracy" (abstract l.18-19; l.265-267; l.309-311) rests on gold−zeroed = +0.027, CI [−0.006, +0.060], seed p=0.22 (l.260-261): the point estimate is half of D-hook's gain, and Appendix G calls the same quantity "a small real cost" (l.549-550), so the interval is read both ways. On seed 42, zeroed D-hook exceeds gold (Fig. 3 right). (b) "No mechanism beats it" after type removal (abstract l.15; l.53-54) contradicts l.213-214, where B keeps a click gain +0.039 [+0.004, +0.075] and hit@0.25 +0.053 [+0.024, +0.083]. (c) "Conditioning does nothing" on the control (l.306-307) from three-seed CIs admitting ±4 points (l.220-221). (d) "D-hook's advantage persists with data and B's does not" (l.241) while D-hook−B is within noise at four of six sizes (l.239-240): a difference in significance is not a significant difference. All four need an equivalence framing with a stated smallest effect of interest.
3. **The D-token "refutation" (l.43, 308) is neither fair-tuned nor powered.** One learning rate is shared by LoRA and a freshly initialized N(0, 0.02²) token (l.139, 496-497); the authors concede a faster schedule could rescue it (l.293-295). D-token−A CI [−0.015, +0.034] cannot exclude a 3-point gain. D-text, at the same prompt position with pretrained embeddings, works, which is exactly what under-training of the new token predicts.
4. **Hard routing's losses are confounded with parse failures.** 2-8% of C's outputs are unparseable and scored at √2 (l.166-167, 498-499); the promised "both ways" numbers never appear in a table. In Table 3, C's mean-L2 increase (+0.086) equals about 6% × √2, and its −0.057 hit@0.10 is the size of the unparseable fraction, so "C is worse" (l.221-222) and the 16-point scroll loss (l.199-201) may be output-format failures, not grounding effects.
5. **Scaling confounds n with validation composition.** Each size has a different slice and class mix (Table 8: scrolls range 12 to 71). B hurts scrolls and D-hook helps them (Table 2), so B−A ≤ 0 at the scroll-heavy sizes (800, 2500) is predicted by composition alone. B's sign flip at n=300 (hit@0.10 +0.067***, hit@0.25 −0.055**) signals metric fragility that is never discussed.
6. **Selective reporting.** The abstract omits that the predicted-type pipeline does not beat A (+0.016, CI [−0.013, +0.047], p=0.44; l.230-233), the only deployment-relevant number. It credits D-text with a 6-10 point click gain, yet D-text appears in no per-class table (Table 2 omits it) and in none of the controls (Tables 3, 4, 7, 8) despite being the best variant. Only the headline cell was extended from three to five seeds (l.169); l.238-239 implies D-hook's three-seed seed-level p at n=1200 exceeded 0.05, and it reached 0.028 only after extension. If that decision followed the three-seed look, the seed-level p is not interpretable at face value.
7. **Smaller.** "The only difference between this stream and the control is the type class" (l.201-202) is false: n_train, n_val, seed count, and slice all differ. "Distance is exactly √2 whatever the model predicts" (l.110) is false unless a fixed penalty is assigned. The M-RoPE cause was inferred, not verified (l.527-528), and the fix changed the visual-merge path simultaneously (l.525-527). The r=0.10 headline is non-standard for AITW, whose official matcher uses 0.14 or bbox containment; the element annotations are unused. Appendices C and D are empty headers; Table 9's caption still says "paired bootstrap over pooled examples" although l.510-511 claims every star was recomputed on the cluster basis; Fig. 4 is unreadable; over 50 uncorrected contrasts.

**Questions for the authors**
1. Episodes per validation slice, and episode-clustered intervals for Tables 1 and 4?
2. Was the five-seed extension pre-specified? What were the three-seed seed-level p values for Table 1?
3. C's parse-failure rate per class, and Tables 1-3 on parsed outputs only?
4. D-token with the embedding table at 10× and 100× the base learning rate, and the frozen-slot control (l.295-296)?
5. D-text in Tables 3, 4 (wrong word in prompt), and 7?
6. Did you dump `position_ids` to confirm the 1-D fallback rather than infer it?
7. Is A's click error a pull toward the origin (mean predicted coordinate)? The "broader degradation" (l.206-207) is asserted, not measured.

**Limitations**
Section 7 is candid for a workshop paper (untuned D-token, single backbone, descriptive scaling, single-seed attention, Mind2Web at floor). Missing: episode-level dependence; the parse-failure confound for C; that the "degenerate class" is a serialization artifact the authors introduced by clamping (−1,−1) to (0,0), so the practical lesson largely reduces to "do not train grounding on non-spatial actions with a dummy target", which is common practice; and that the best mechanism (D-text) is untested in every control.

**Ratings**
Soundness: 2. Presentation: 3. Contribution: 2. Overall: 4 (borderline reject). Confidence: 4.

**Recommendation**
Reject in the current form; a revision along the lines below would be a credible poster.

**What would change my score**
1. Re-run every interval with episodes as clusters (re-analysis only) and report the episode count per slice; state which stars survive.
2. Replace every "returns to baseline / does nothing / no mechanism beats it" with equivalence tests and a stated margin, and fix the abstract and conclusion to match l.213-214 and l.230-233.
3. Tabulate C on parsed outputs with per-class parse rates.
4. A small learning-rate sweep for D-token's table (and the frozen-slot control) before using the word "refute".
5. Add D-text to Tables 3, 4, and 7.
6. Either fix the scaling design (a common validation slice or per-class curves) or drop the persist/not-persist claim.