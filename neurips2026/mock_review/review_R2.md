**Summary**
The paper asks whether supplying the action type to a LoRA-adapted Qwen2-VL-2B improves GUI coordinate grounding. Four mechanisms (auxiliary loss B, hard routing C, additive embedding D-hook, prepended token D-token) are compared with a flat baseline A under matched data and optimization on AITW, with a taps-and-swipes mix and Mind2Web as controls. B and D-hook are reported to gain +0.064 and +0.054 hit@0.10 over five seeds and D-token nothing; interventions and attention probes are used to argue the benefit is a training-time effect. A false-negative M-RoPE/inputs_embeds pitfall is also documented.

**Strengths**
- Matched-compute design with a type-uninformative control mix; the right experimental logic.
- The authors refute their own hypothesis; the M-RoPE pitfall (Sec. 6.1) is a useful practitioner finding.
- Per-example distances are logged for every run, so a correct re-analysis is cheap; Table 3 uses all seeds and examples.
- Candid Limitations section; clean Stage-1 result (Table 5).

**Weaknesses**

1. Wrong inference unit (l.154-157, App. A). The bootstrap resamples 1,250 (seed, example) units as exchangeable; the abstract and l.43-44 call them "1,250 examples", but they are 250 examples x 5 runs. Units sharing a seed share a training trajectory, and this dominates: A's seed std of 0.043 (Table 1; l.174-175) gives a seed-level SE of 0.019 for A alone, larger than the whole bootstrap SE (0.011, from the +/-0.022 CI half-widths), i.e. the CIs reflect example-level noise only. App. A acknowledges same-example correlation but not same-seed correlation, the larger one. The permutation test "agrees" because it makes the identical exchangeability assumption, as App. A itself says. Unpaired seed-level t-tests from the reported stds (per-seed differences are not shown): B-A t~3.0 (p~0.03), D-hook-A t~2.0 (p~0.08), C-A t~1.3, D-token-B t~2.3, control C-A t~2.3 on 3 seeds. A five-seed sign test cannot go below p=0.06, so no *** is credible at the seed level, although Table 3's large contrasts clearly survive. Consecutive validation steps (likely a few dozen episodes) are a third ignored cluster. Survives: B>A, probably D-hook>A. Does not: C's gain, "D-token worse than B", "C harms" on the control.

2. The abstract's mechanistic claim rests on one seed and 120 examples. "Can be zeroed at test time without loss" (l.13-14, 284-286) is Table 4's Delta=-0.033, CI [-0.117, +0.042]: an underpowered null read as no effect, while the opposing D-token test (Table 3) got 5 seeds x 250 examples. The "zero" conditions are also not comparable: zero is D-hook's initialization and equals A's input, whereas a zero-vector slot token is out-of-distribution for D-token; a class-mean embedding is the fair uninformative condition for both. The probe token was chosen after seeing the last-prompt-token result (l.276-277), the layer choice is unexplained, and the uniform-attention "chance" null ignores attention sinks.

3. Scaling study confounded and over-tested (Sec. 5.5, Table 7). Each n is scored on a different validation slice (l.147-149) whose class mix is unreported although the gain is "a click gain" (Table 6); A's accuracy is non-monotone in n (0.139 to 0.363). Steps 5000-5249 lie outside every training slice, so all models could be re-scored on one common set. Roughly 90 uncorrected tests appear; "B hurts at 800" (l.48, l.207) is one cell of 24, and at n=300 B-A is +0.067*** on hit@0.10 but -0.055*** on hit@0.25. "Differ everywhere else" (l.212) infers a difference from one contrast being starred and the other not, without testing B against D-hook; at 300 and 500 they are nearly identical.

4. Equivalence from non-significance. "Statistically tied" (l.170), "change nothing" (l.10), "within noise" (l.173, 188): the control CIs [-0.038, +0.012] and [-0.047, +0.010] admit harms nearly as large as the headline benefit; the D-hook-B CI admits B leading by half the effect.

5. Numbers disagree across the paper. (a) Table 6 implies C hit@0.10 = (169x0.356+46x0.172)/250 = 0.272; Table 1 says 0.264, while the other four rows reconcile within rounding. (b) Table 3 "gold" should equal Table 1's D-token row; every column differs (0.236+/-0.058 vs 0.238+/-0.049). (c) Sec. 6.1 gives A=0.390 and D-hook=0.392 "on the same control"; Table 2 has 0.382 and 0.363. (d) Sec. 5.4: 0.021/(0.292-0.255) is 57% of the gain, not "about a third", and Table 7 implies three-seed D-hook=0.300, not 0.292. (e) l.216-217: 18+11+45+90 = 164 of 169 clicks. Different runs, or non-deterministic evaluation?

6. End to end (Sec. 5.4): three seeds, no CI for predicted-vs-flat (+0.016, inside A's std); only D-hook, which per Sec. 6.3 does not need the signal; no zeroed comparator, although Table 4 suggests zeroing may beat Stage-1 predictions. No joint type+coordinate baseline tests the introduction's motivation.

7. Protocol and parsing. One learning rate, lambda=1, two epochs, batch 1, D-token's table trained at 2e-5 from N(0, 0.02^2): the refutation is conditional on an untuned configuration (l.280-281, absent from the abstract). C's 2-8% unparseable outputs score distance sqrt(2); other variants' parse rates and results excluding failures are not reported, so C's losses may be a formatting artifact. Seeds 45-46 were added only to the headline; Tables 1 and 7 imply they average A~0.19, raising the effects from +0.051/+0.045 to +0.064/+0.054. Was five seeds pre-specified?

**Questions for the authors**
1. Per-seed paired differences for every contrast and a seed-level or two-way cluster analysis: which stars survive?
2. Explain W5; is greedy evaluation deterministic run to run?
3. Class mix of each scaling slice, and all scaling models re-scored on steps 5000-5249.
4. D-hook interventions on all five seeds and 250 examples, with a class-mean condition for both mechanisms.
5. Parse-failure rates per variant; norm of D-hook's E[a] relative to token embeddings; how was the layer chosen?
6. Was the Stage-1 MLP in Sec. 5.4 checkpoint-selected on the evaluation slice (l.107-108)? Code?

**Limitations**
Candid on scale, single-seed attention, the post hoc probe, and one learning rate. Missing: seed-level clustering (App. A names only the example level), the moving validation slice as a confound, multiplicity, episode clustering of validation steps, and that the abstract's "zeroed without loss" claim is a single-seed null result.

**Ratings**
Soundness 2, Presentation 3, Contribution 3, Overall 4, Confidence 4.

**Recommendation**
Reject (as submitted). Worth publishing in principle, but the inferential claims are stated at a confidence the data cannot support, an abstract-level claim rests on one seed, and the tables do not reconcile. Every fix is re-analysis or inference-only; a revision doing them would be a clear Accept (poster).

**What would change my score**
1. Seed-level or hierarchical inference for every contrast, with "1,250 examples" and unsupported stars removed. If B-A and D-hook-A hold, the factorization claim stands and I move to 6, Accept (poster).
2. D-hook zero/wrong interventions on all seeds and examples, or drop the zeroing claim from abstract and conclusion.
3. Re-score scaling models on the common held-out slice, report class mix, and correct for multiplicity or present the study as descriptive.
4. Reconcile W5, fix the "about a third" arithmetic, add a CI and zeroed comparator to Sec. 5.4.
5. Report parse failures and C without them; replace "tied/nothing" with CI bounds.