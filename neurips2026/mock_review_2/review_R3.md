**Summary**
The paper fine-tunes Qwen2-VL-2B with LoRA to emit click/scroll coordinates on AITW and compares a flat baseline with five ways of supplying the gold action type (auxiliary loss B, hard routing C, additive embedding D-hook, prepended learned token D-token, type as a prompt word D-text) under matched data, adapters and decoding. On a stream whose type events carry a clamped sentinel target, B, D-hook and D-text gain 6-10 hit@0.10 points and D-token does not; on a stream without that class nothing helps, and dropping type events from training lifts the flat baseline to the conditioned models. Input interventions on the trained models are read as showing that both learned embeddings are consulted at inference but only D-hook degrades gracefully, and an earlier result caused by `inputs_embeds` injection bypassing M-RoPE is retracted.

**Strengths**
- Matched-compute design with a genuine negative control (taps_and_swipes), a class-removal retrain (Table 7), per-class numbers, an end-to-end run with predicted types, and paired statistics at two levels.
- The interventions are real do-operations on the input, and the click↔scroll swap sensibly excludes the "everything routed to the degenerate class" reading.
- App. G reports attention with the right caveats (one layer, head-averaged, teacher forcing, attention is not information flow), does not lean on it, and surfaces its inconvenient finding: the flat model localizes as much as the conditioned ones.
- The retraction (§6.2, App. F) is candid and its account of the library is correct: I checked that visual features are merged inside `if inputs_embeds is None` and that without `input_ids` positions collapse to a 1-D index replicated on all three axes.

**Weaknesses**
1. Confounded refutation. D-token rows are reported at norm 0.78 (l.255-256); the N(0, 0.02²) init in d=1536 has expected norm 0.02·√1536 = 0.78, and the D-hook table, trained from zero under the identical schedule, moved only 0.04. The "learned" slot is therefore ~95% its random initialization, so the frozen-slot control the paper says it lacks (l.294-295) is effectively what was run, while D-text, a prepended token with a pretrained embedding, is the best mechanism (+0.073). What is refuted is "a fresh embedding at lr 2e-5," not "a prepended token." Initializing E from the word embeddings, or a higher LR for the new rows, was not tried. This undercuts the abstract's central negative claim.
2. The switch claim is not licensed. "With its embedding zeroed, the additive model returns to the baseline's accuracy" (abstract; "therefore behaves as a switch," l.264-267) rests on gold−zero = +0.027, CI [−0.006, +0.060], and zero−A = +0.019 n.s.; that interval contains both "zeroing costs nothing" and "zeroing removes the whole gain." The training-versus-inference decomposition for D-hook is underpowered; the body hedges, the abstract and conclusion do not. Moreover the deployment-relevant perturbation is a wrong type (Stage-1 is 84% accurate, §5.3), and there D-hook is the more brittle mechanism (0.054/0.099 vs 0.079/0.155, Table 4); graceful degradation would matter only under confidence-gated zeroing, which was never run.
3. Clustering. The bootstrap clusters on validation examples, but these are consecutive steps from "a few dozen episodes" (l.157-159, 487-489); the number of independent units is tens, not 250, so every interval and star is anti-conservative. The n=5 seed-level tests (D-hook p=0.028, D-text 0.018, 24+ uncorrected contrasts) cannot carry the load alone. Cluster by episode.
4. Unverified diagnosis (contribution 4). Attributing the 9-point deficit to 1-D positions was "inferred from the code" (l.527-528), and the before/after compares different mechanisms: D-slot (prepended, buggy path) against D-hook (additive, fixed path) on taps_and_swipes (App. F); fixed D-token is absent from Table 3. Alternatives on the same path (the caller's own visual merge; mask/cache handling, and the cited issue #35463 is a mask-length crash on exactly this path) are not excluded. Running flat A through `inputs_embeds` with and without explicit 3-D position_ids would settle it in minutes.
5. "Uninformative" is not matched across architectures. For D-hook, zero is the flat architecture's forward and the training-time initial state; for D-token it is out-of-distribution. The class mean is called "in-distribution norm" (Table 4 caption), but the mean of three near-orthogonal 0.78-norm vectors has norm ≈0.45. A fresh N(0, 0.02²) draw is the proper control for D-token. Part of the D-hook/D-token contrast is an artifact of how "no signal" is operationalized.
6. The mechanism stops at a label. "Protection from the degenerate class" is supported by Table 7 but never characterized: exact (0,0) emissions explain little (Table 2), and nothing asks whether flat click errors are biased toward the origin or how logit mass on the "(0," mode moves under gold/zero/wrong conditioning. D-text, the strongest and parameter-free mechanism, appears in no intervention, control, removal or scaling table. The practical lesson also reduces to a preprocessing artifact, since dropping type events from the flat baseline recovers the whole gain (l.209-211), and the "grounded/faithful" framing is thin given the paper's own finding that conditioning adds no evidence localization.
7. Precision and presentation. l.134-135 says E[a] is added to "every text and image token embedding," but in HF's Qwen2-VL the visual features are masked_scatter'ed over placeholder embeddings after `embed_tokens`, so a hook on the embedding layer cannot reach image positions. Appendix sections C, D and H are empty headers; Figure 4 is illegible; the text cites "an earlier version of this paper" (l.241-243, 262-264).

**Questions for the authors**
1. Where exactly does the D-hook hook sit relative to the visual merge, and does E[a] reach image tokens?
2. Report the displacement of D-token rows from initialization, pairwise cosines between class rows, and the norm of the class-mean vector for both architectures.
3. Was fixed D-token trained on taps_and_swipes? Was the deficit reproduced with `input_ids` plus explicitly passed 1-D position_ids, or removed with explicit 3-D ones?
4. How many episodes per validation slice, and which stars survive an episode-level bootstrap?
5. For B, which positions are pooled into z̄; are the teacher-forced answer tokens, which reveal the type via "(0, 0)", excluded?
6. Which layer index is "3/4-depth," and how is "share of image attention" normalized?

**Limitations**
§7 is unusually thorough (untuned D-token, single slice, uncorrected contrasts, attention caveats). Missing: within-episode dependence of validation examples; that the D-token norm implies an essentially untrained slot; that the deployment claim presumes a gating scheme never run; that hit@0.10 is a coarse criterion (10% of the normalized screen); and that the headline is specific to AITW's sentinel target for type actions.

**Ratings**
Soundness 2, Presentation 2, Contribution 2, Overall 5, Confidence 4.

**Recommendation**
Accept (poster), conditional on the abstract and conclusion no longer asserting the switch behavior or the refutation of "the prepended token" beyond what the intervals and the norm arithmetic support.

**What would change my score**
1. D-token initialized from the word embeddings (or a separate LR for the new rows) plus a frozen-slot control on the fixed path. If it matches D-text, rewrite the refutation; if it still fails, the mechanism claim becomes real (would move me to 6-7).
2. Episode-clustered bootstrap for every starred contrast.
3. The position_ids test and fixed D-token on taps_and_swipes.
4. Confidence-gated end-to-end (zero E[a] when Stage-1 is unsure), turning "degrades to baseline" into a measured deployment benefit.
5. Origin-bias or logit-mass analysis of the protection effect, and interventions on D-text.