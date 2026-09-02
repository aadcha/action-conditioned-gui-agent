I have everything I need: both page ranges of the PDF, the text extraction for line references, and a check of the public HF `get_rope_index` code (in both a 2024 release and current `main`, the 3-D positions are built by scanning `input_ids` for the vision-start/image ids; with `input_ids=None` the code falls back to 1-D positions replicated across the three axes, or to `position_ids=None`, so image tokens lose their (h, w) layout entirely, and the caller must also scatter the visual features itself). Here is the review.

---

**Summary**

The paper asks whether telling a Qwen2-VL-2B grounding model the action type improves coordinate prediction on AITW (with Mind2Web as a control). Four conditioning mechanisms (auxiliary loss B, hard routing C, additive embedding D-hook, prepended token D-token) are compared with a flat baseline under matched data, compute and decoding, with five seeds and paired bootstraps. B and D-hook gain 5-6 hit@0.10 points on a mix containing clicks, scrolls and ungroundable type events, and nothing on a click/scroll control; D-token does not help. Interventions (wrong/zeroed embeddings) plus a single-layer attention-share measure are used to argue that D-hook's gain is a training-time effect while D-token is "used" at inference, and a false negative caused by injecting through `inputs_embeds` is documented.

**Strengths**

- Matched-compute design, deterministic evaluation slices, five seeds with paired statistics, and a control mix where conditioning should be inert. The retracted D-slot result is disclosed, not buried.
- An honest negative result on the authors' own hypothesis; per-class numbers (Table 6) and all probe positions (Table 8) are reported, not only the favourable token.
- A second wrong map to address the type-class confound (Sec. 6.2-6.3), and a candid Limitations section.
- The `inputs_embeds` pitfall is practically useful for anyone hooking Qwen2-VL.

**Weaknesses**

1. The headline attribution ("conditioning helps where the type predicts location", Sec. 5.2 title, abstract) has an untested, simpler explanation. The only difference between the headline mix and the control is the 14% of type events with degenerate targets (lines 178-181, 274-275). For flat A these are unlabelled label noise; any variant told which examples are type can partition them off. Click-vs-scroll alone (Table 2) shows no gain, and Table 6 shows the headline gain is a click gain. "Action type identifies the noise examples" fits the data at least as well as "action type is a spatial prior." A control that adds a synthetic degenerate class to `taps_and_swipes`, or trains without type events and evaluates on the same clicks/scrolls, is needed before the factorization claim is earned.

2. The zeroing intervention is not the same operation for the two architectures, so the training-time/inference-time dichotomy (lines 259-269, abstract) is not licensed. For D-hook, zero restores exactly A's input distribution; for D-token, a zero-norm token at the slot is out of distribution. D-token then collapses to the same value under every perturbation (Table 4: hit 0.117/0.142/0.142; attention 0.080/0.079/0.079), which is what brittleness to an OOD slot looks like, not evidence the type is "consulted." Conversely the paper reports D-token's norm (2.2, line 241) but never D-hook's. With zero init and lr 2e-5 (lines 465-467) it is plausibly tiny, so "zeroing costs nothing" may be near a no-op by construction, while wrong-map sensitivity reflects LoRA amplification of one direction with zero sitting on the boundary. If E[click] is near zero, "zeroed" is just "click for everything," correct for 68% of examples. Per-class norms and per-class intervention results are required.

3. The D-hook intervention is underpowered for the weight placed on it: one seed, 120 examples, no A row on those examples. gold minus zero hit@0.10 = -0.033 [-0.117, +0.042]; the interval allows zeroing to remove roughly 80% of the +0.054 headline gain. Table 3 uses 5 x 250 for D-token; D-hook deserves the same.

4. The attention measure does not support "evidence localization" (lines 88-90, 254-256). It is one layer, head-averaged (sink/register-like image tokens and heterogeneous heads averaged in), attention weight rather than information flow (no value-norm weighting, rollout, or patch ablation), and teacher-forced: the y-predicting token has the gold x in its prefix, so mass near the target is partly the supplied column. "Localization happens once the x coordinate has been emitted" (line 257) is exactly what that artifact predicts. The number is not conditioned on correctness, so "grounded ... when it is right" was not measured, and there is no flat-A row showing conditioning changed anything.

5. The M-RoPE diagnosis is inferred, not verified, and its description is probably wrong. In the public HF code the 3-D positions come from scanning `input_ids` for the vision-start/image ids; with `input_ids=None` the fallback is 1-D positions replicated across the three axes (or `position_ids=None`), so image tokens lose their (h, w) layout entirely. That is far larger than "positioned as if the slot were not there" (line 228), which under relative RoPE would be nearly harmless. The caller must also scatter visual features itself on that path, a second candidate failure. Neither position-id inspection nor the decisive test (`inputs_embeds` plus explicit `get_rope_index` positions) was run, and the fix changes path and mechanism together: Table 2 has no D-token row, so D-slot is compared with D-hook.

6. Statistics: the pooled (seed, example) bootstrap treats 1,250 correlated units as independent; the appendix admits it is "optimistic" (lines 473-475), yet stars are used at face value, including three-seed scaling cells (B "hurts" at 800, p<0.01) that drive Sec. 5.5. Seed spread is large (0.169-0.280, line 175). A cluster bootstrap over examples or per-seed paired tests should be the reported inference.

7. Causal language ("obeys", "consulted", "used"; lines 241, 245, 265) outruns the evidence; perturbation sensitivity is not semantic use. Show that clicks conditioned as scroll move toward scroll-typical locations.

Minor: Appendix C is an empty heading (line 477); "adds E[a] to every token embedding" (line 126) presumably excludes image positions overwritten by `masked_scatter`; the predicted-type-vs-A gap (+0.016, line 200) has no CI.

**Questions for the authors**

1. Per-class norms of D-hook's E after training, relative to token-embedding norms. Is E[click] near zero?
2. Per-class (click/scroll) hit and attention under each Table 4 intervention, and A on the same 120 screens.
3. Did you inspect `position_ids` under D-slot? Does `inputs_embeds` with explicit `get_rope_index` positions recover the baseline?
4. What does D-token (hook path) score on `taps_and_swipes`?
5. Does the B/D-hook gain survive removing type events from training, or appear on `taps_and_swipes` once a synthetic degenerate class is added?
6. Attention without teacher forcing, across layers, and for flat A?

**Limitations**

Scale, seed count, one layer/one seed for attention, and the ungroundable type class are acknowledged. Missing: that the type-class admission undercuts the headline attribution; that the zero intervention is not comparable across architectures; that the M-RoPE story was never checked against position ids; the teacher-forcing confound; and that admittedly optimistic intervals still supply the stars.

**Ratings**

Soundness: 2. Presentation: 3. Contribution: 2. Overall: 5. Confidence: 4.

**Recommendation**

Accept (poster), conditional on softening the mechanism claims. Not spotlight: the "grounded/faithful" framing is only partly earned; the workshop-relevant contributions are the matched ablation and the failure analysis, not the faithfulness evidence.

**What would change my score**

1. A degenerate-class control (item 1). If the gain survives in a click-vs-scroll-only setting with spatially distinct distributions, the factorization claim stands: +2.
2. D-hook interventions at 5 seeds x 250 with an A row, per-class breakdown and E norms: +1.
3. Direct M-RoPE verification and a D-token row in Table 2.
4. Attention: layer sweep, free-running variant, flat-A row, value-weighted or per-head variant, or patch ablation.
5. Cluster bootstrap or per-seed tests; replace "obeys/consulted" with "sensitive to."