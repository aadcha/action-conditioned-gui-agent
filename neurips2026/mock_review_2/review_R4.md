**Summary**
The paper LoRA-tunes Qwen2-VL-2B on Android in the Wild to emit a coordinate string and compares a flat baseline (A) with five ways of supplying the gold action type under matched data, adapters and decoding: an auxiliary type loss (B), a forced action word (C), an additive learned vector on every token (D-hook), a prepended learned token (D-token, the authors' original hypothesis), and the type as a prompt word (D-text). With five seeds and a cluster bootstrap, B, D-hook and D-text gain 5–7 hit@0.10 points on a stream containing type events clamped to (0,0); D-token does not. Removing those events raises the baseline four points, after which nothing beats it at hit@0.10, and nothing helps on a taps-and-swipes control. Interventions show both learned embeddings are consulted at inference, and an earlier `inputs_embeds` variant is retracted as an M-RoPE artifact.

**Strengths**
- Rare candor: the refuted hypothesis is reported as refuted, a retracted variant is documented, per-seed spread is shown, and Fig. 4 includes failures.
- The right controls exist (type-removed retrain, Table 7; taps_and_swipes, Table 3) and the authors let them overturn the headline.
- The Table 4 interventions cleanly separate training effects from inference-time use; a good template.
- The `inputs_embeds`/M-RoPE pitfall (Sec. 6.2, App. F) is real: I verified in current transformers source that positions fall back to a 1-D arange expanded over the three axes when `input_ids` is absent.
- Citations I checked (LiMAC, CoCo-Agent, GUI-Libra, GUI-AIMA, SeeClick, HF #35463) are characterized accurately. I found no prior matched-compute comparison of type-conditioning mechanisms for grounding, so the study itself is not preempted.

**Weaknesses**
1. The headline gain is an artifact of the authors' own reduction. Lines 108–111: AITW stores type events at (−1,−1); the serializer clamps them to (0,0) and supervises a coordinate on them. Standard pipelines (Auto-UI, which the paper cites) keep "[-1.0,-1.0]" beside the action type in one string, and native formats emit `type(content=...)` with no coordinate, so the "degenerate class" does not exist in the designs baseline A claims to stand for (line 126). Table 7 shows the cure is simply dropping those targets: A rises to 0.297, matching every conditioned model on the mixed stream, and B/D-hook then add +0.021/+0.008 (CIs include zero); Table 3 agrees. The correct summary is that type supervision does not improve grounding here; the title, the Sec. 5.1 heading, and abstract lines 10–12 lead with the artifact.
2. The D-token refutation is confounded by an untrained embedding. N(0, 0.02²) in d=1536 (line 139) gives an initial norm of 0.02·√1536 ≈ 0.78; line 256 reports trained rows of norm 0.78. At LR 2·10⁻⁵ over ~2,400 batch-1 steps the slot never left its random initialization; LoRA adapted around a fixed random per-class vector, which explains both "consulted at inference" and "no gain". Limitations concede the untuned schedule (293–295), yet the abstract states the refutation as a finding. A separate embedding learning rate is standard and cheap.
3. The deployment argument is inverted. Zeroing an embedding (abstract 17–20, line 268) is not a deployment condition; a wrong Stage-1 type is, and Table 4 shows it collapses D-hook to 0.054/0.099, far below flat A (0.229). Consistently, end-to-end with predicted types (Sec. 5.3) yields +0.016, CI [−0.013, +0.047], which the abstract omits. Only B has a deployment-real gain, and it vanishes on clean data.
4. D-text, the best variant (Table 1, +0.073) and what practitioners already do, gets no per-class row (Table 2), no control (Table 3), no type-removed run (Table 7), no scaling, no intervention, and is not drawn in Fig. 1.
5. Statistics: 250/200-step slices from "a few dozen episodes" with a boundary episode possibly shared with training (488–489); the bootstrap clusters on steps, not episodes, so intervals are optimistic; each scaling size is scored on a different slice (line 158, Table 8), leaving Fig. 2 uninterpretable when a fixed held-out set would have been trivial; 24 uncorrected contrasts; single-seed attention.
6. Likely incorrect mechanism description: D-hook "adds E[a] to every text and image token embedding" (134–135). In HF Qwen2-VL, visual features are `masked_scatter`ed over placeholders after the embedding lookup, so an embedding-module hook never touches image tokens.
7. Related work: CoCo-Agent already ablates conditional target prediction (+12 points on AITW-General), so "what has not been measured" (32–34) should be narrowed; AutoGLM (decoupled planning/grounding) and TinyClick (multitask supervision adding 4–8 points to a 0.27B grounder) are missing.
8. Unfinished presentation: Appendices C, D and H are empty headers with tables floating elsewhere; Figs. 1 and 4 are illegible in print; "correct our own description" (abstract) reads as a revision note; no anonymized code (line 302); the AITW mirror is unnamed; "6 to 10 points" mixes Table 1 overall deltas with Table 2 per-class ones; "no mechanism beats it" ignores B's significant hit@0.25 gain (line 214); Table 5 shows the Stage-1 features losing to TF-IDF on Mind2Web.

**Questions for the authors**
1. Where exactly is the D-hook hook registered, and does E[a] reach image tokens?
2. What were D-token's embedding norms at initialization versus after training? Does a 10–100× embedding learning rate change Table 1?
3. Why clamp (−1,−1) to (0,0) and supervise it, rather than masking the coordinate loss as standard formats effectively do? Does any conditioning gain survive that choice?
4. How many episodes are in the 250-step slice, and do the intervals change when clustering on episodes?
5. Why does the scaling study not use one fixed validation set?
6. What do D-text with predicted types, and B combined with D-text, score?

**Limitations**
Sec. 7 is unusually candid (one backbone, small slices, untuned D-token, no ScreenSpot or task success). Missing: that the degenerate class is a preprocessing decision rather than a property of AITW as commonly used; episode-level correlation; wrong-type fragility as the actual deployment risk; and that D-text was never analyzed.

**Ratings**
Soundness 2, Presentation 2, Contribution 2, Overall 4, Confidence 4.

**Recommendation**
Reject (in its current framing; a reframed version would be a credible poster).

**What would change my score**
1. Reframe around the negative result: on clean or masked streams no mechanism beats writing the type into the prompt; move the artifact story to a diagnosis section and retitle.
2. Rerun D-token with a proper embedding learning rate (or the frozen-slot control) before claiming refutation.
3. Put end-to-end numbers for all variants in the abstract and treat wrong-type collapse as the deployment finding.
4. Fixed validation set for scaling; episode-level clustering; D-text in Tables 2, 3 and 7.
5. Fix the hook description, fill or remove empty appendices, release anonymized code.
With items 1–3 done I would move to 5–6 and Accept (poster).

Sources consulted for the novelty and citation checks: [LiMAC](https://arxiv.org/html/2410.17883), [CoCo-Agent](https://arxiv.org/html/2402.11941), [GUI-Libra](https://arxiv.org/abs/2602.22190), [GUI-AIMA](https://arxiv.org/html/2511.00810), [SeeClick](https://arxiv.org/abs/2401.10935), [TinyClick](https://arxiv.org/html/2410.11871v2), [AutoGLM](https://arxiv.org/abs/2411.00820), [Auto-UI](https://arxiv.org/html/2309.11436), [Phi-Ground](https://arxiv.org/abs/2507.23779), [HF issue #35463](https://github.com/huggingface/transformers/issues/35463), [PR #35466](https://github.com/huggingface/transformers/pull/35466), [Qwen2-VL modeling source](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py), [AITW README](https://github.com/google-research/google-research/blob/master/android_in_the_wild/README.md).