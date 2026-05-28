# Action-Type-Conditioned Grounding for GUI Agents
### Project Overview — CS 231N, Spring 2026

**Team:** Aadi Chauhan, Arthur Ilyasov, Nevin Kunampuram
**Track:** Models
**Base model:** Qwen2-VL-7B-Instruct
**One-line pitch:** Current GUI agents decide *what action* and *where to act* in one entangled decode; we factor those into a classifier-first, action-type-conditioned grounding pipeline and study whether the factorization reduces the wrong-action-type failures that compound across long-horizon tasks.

---

## 1. The Problem

A GUI agent looks at a screenshot plus an instruction ("add this to my cart") and must produce an action: a *type* (click, double-click, type, scroll, drag, hotkey, wait, finished) and, for spatial actions, a *location* (pixel coordinates of the target).

State-of-the-art native agents — UI-TARS, OS-Atlas, SeeClick, ShowUI — emit both as one autoregressive token stream from the same head. Type and location are entangled. The documented failure mode: the model commits to the wrong action type (scrolls past a visible target instead of clicking, types into a button instead of an input field), then grounds *coherently but wrongly* within that incorrect type. These errors compound across steps in long-horizon tasks.

This isn't a contrived modeling preference. Mind2Web's official metrics already separate **element accuracy**, **operation F1**, and **step success rate**, and define a successful step as getting *both* the element and the operation right. The benchmark structure itself reflects the what/where split we want to exploit. Recent mobile world-model work also documents that action-type accuracy and step accuracy can diverge — coordinate fixes can mask persistent type errors.

---

## 2. The Hypothesis

Explicitly decoupling action-type prediction from grounding — predicting type first via a dedicated classifier, then conditioning the grounding stage on the predicted type via a learned class embedding prepended to the instruction stream — improves both action-type reliability and downstream grounding, because the grounding stage receives a discrete, known conditioning signal that forces attention specialization: buttons for clicks, input fields for types, scrollable containers for scrolls.

The flat baseline can't do this because it doesn't commit to a type until it's already decoding the location.

---

## 3. Method

A two-stage pipeline on a frozen-then-LoRA Qwen2-VL-7B backbone.

**Stage 1 — Action-Type Classifier.**
Screenshot + instruction → frozen Qwen2-VL-7B → mean-pool visual and text token embeddings → 3-layer MLP → softmax over 8 canonical action classes. Trained with class-weighted cross-entropy. The base VLM is frozen; only the MLP head learns. Features are cached to disk so the MLP trains in minutes.

**Stage 2 — Action-Type-Conditioned Grounding.**
A learned embedding table `E ∈ ℝ^(8×d)`, one row per action class. The predicted type's embedding is prepended to the instruction tokens, and Qwen2-VL is LoRA-fine-tuned (rank 16, attention + FFN) to produce spatial coordinates conditioned on that signal. Training is teacher-forced (gold types) for early epochs, then student-forced (Stage 1 predictions) for a final epoch so the model learns to handle classifier errors.

**Output:** `(action_type, x, y)`.

### Key design choices and rationale
- **Frozen base in Stage 1:** pretrained features already encode action-relevant cues; full fine-tuning a 7B model for 8-way classification is wasteful and risks forgetting.
- **Learned embedding, not the text token "click":** an embedding carries richer per-class priors (where to look, which affordances matter) than a single English token.
- **LoRA, not full fine-tuning:** preserves grounding ability, cheap to ablate, matches how UI-TARS / SeeClick adapt their bases. Rank 16 is a convention to be ablated, not a principled choice.

---

## 4. Data

All sources provide aligned (screenshot, instruction, action-type, target) tuples. Unified under a **canonical 8-class action taxonomy** with class-balanced sampling to fix the ~80% click majority.

| Source | Scale | Domain | Role |
|---|---|---|---|
| **Mind2Web** | ~137k steps (2,350 tasks, 137 sites) | Web | Primary train + eval; official Type/Grounding/SR metrics |
| **Android in the Wild (AITW)** | ~715k episodes, 30k instructions | Android | Largest source of non-click action supervision |
| **AndroidControl** | 15,283 demos, 14,548 tasks, 833 apps | Android | Generalization; data-scale caveats |
| **Wave-UI-25K** | ~25k labeled UI elements | Web (grounding) | Supplemental grounding-only data (dataset card, not a peer-reviewed benchmark — describe transparently) |

**Open data risk flagged by lit review:** canonicalizing *non-point* actions (drag, scroll, hotkey, wait, finished) across web/Android/desktop is nontrivial — OS-Atlas and UI-TARS both emphasize how harmful action-space conflicts are when datasets are mixed naively. Every taxonomy judgment call must be documented in code. Also: AITW "episodes" contain ~5-15 steps each, so step counts differ in unit from Mind2Web's "steps" — document exact step counts after preprocessing.

---

## 5. Evaluation

Separate **diagnostic** metrics from **final trajectory** metrics.

**Primary metrics:**
- **Action-type macro-F1** on held-out Mind2Web (isolates Stage 1; report per-class confusion matrix, not just accuracy).
- **Showdown Clicks top-1** click accuracy (probes grounding gain on the click family only — 5,679 macOS left-clicks; specify public subset vs full release because small absolute gains mean different things at different sample sizes).
- **Mind2Web step success rate** (joint metric — does decoupling help end-to-end?).

**Ablations (the heart of the paper):**
- **A — Flat joint decode** (vanilla Qwen2-VL LoRA; the must-beat baseline).
- **B — Auxiliary loss** (flat decode + action-type classification loss, no conditioning — tests whether the signal alone helps).
- **C — Hard routing** (predict type, then constrain decoding to type-consistent tokens — tests whether explicit selection alone suffices).
- **D — Type embedding (ours)** (predicted type → learned embedding → prepended).

**Must-have diagnostics (from lit review — these separate a real contribution from "extra supervision in disguise"):**
- **Oracle-type conditioning** upper bound (grounding accuracy given *gold* type) vs predicted-type conditioning — the gap isolates classifier error from grounding error.
- **Vision-ablated classifier accuracy** — if zeroing the screenshot doesn't hurt Stage 1, the model is parroting instruction verbs and the project's interesting part has collapsed. **Measure this on day 1 of Stage 1.**
- **Calibration (ECE)** of the action-type classifier — hard routing with a miscalibrated classifier can be *worse* than joint decoding.
- **Conditional grounding accuracy:** P[grounding correct | type correct].
- **First-error position** on long-horizon tasks — a one-step gain after the first fatal mistake doesn't help users.
- **Action-conditioned attention/saliency visualizations** — show click-conditioned prompts focus on buttons, type-conditioned on editable fields. This is the figure that sells the mechanism.

**Statistics:** paired bootstrap or permutation tests at task/episode level, 95% CIs. Lit-review-suggested bar: ~2-3+ absolute Step SR and 3-5+ macro-F1, consistent across Mind2Web Cross-Task / Cross-Website / Cross-Domain. Treat <3 pts on a small Showdown subset as weak evidence.

---

## 6. Novelty Assessment (from two deep-research reviews)

**Honest verdict: moderately novel, not radically novel.** Strong as a CS231n project; plausibly workshop/arXiv-publishable with a clean empirical story; *not* a main-track CVPR/NeurIPS novelty claim in its current form.

**What's genuinely not in the open literature:** the exact pipeline of frozen VLM features → dedicated action-type classifier → predicted-type embedding prepended to the instruction stream → LoRA grounding fine-tune. No reviewed paper does precisely this.

**Close neighbors that bound the novelty (must cite and distinguish):**
- **UI-R1** — RL with reward decomposed into action-type + action-argument terms. Closest "this is adjacent" paper, but decomposition is in the *reward*, not the *architecture*.
- **GUI-Actor** — dedicated attention-based grounding head (coordinate-free). Closest *architectural* neighbor, but improves *how* grounding is represented, not *which type* governs it.
- **ShowUI** — joint JSON action object (type + value + position). The exact coupling we challenge; the direct "flat" baseline to reproduce conceptually.
- **OS-Atlas / UI-TARS** — unified action spaces; establish that GUI action is already a hybrid object, and that taxonomy unification matters.
- **UGround / SeeAct-V** — modular planner + grounder, but the boundary is *plan-text → grounding*, not *action-type → grounding*.
- **A 2026 diffusion grounding paper** — learns a phase predicting action type + anchor coords before conditional refinement. The strongest novelty threat conceptually; occupies the same "action-type info helps downstream spatial prediction" narrative space.

**The defensible claim** (do *not* claim "first to decouple action type and grounding"):
> Within native screenshot-only VLM GUI agents, we explicitly factor discrete action selection from spatial grounding via a dedicated classifier and an action-type-conditioned latent prompt embedding, and we show this improves action-type reliability, grounding quality, and long-horizon robustness specifically on wrong-action-type failure cases.

**Winning framing:** a **failure-analysis paper with an architectural intervention**, not an architectural revolution. The causal narrative — entangled decoding *causes* the failures, our conditioning addresses them better than simpler alternatives — is where the contribution lives.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Text leakage** — instruction verbs make Stage 1 trivial, vision ignored | Report vision-ablated classifier accuracy; measure day 1 |
| **Class imbalance** — ~80% clicks → "always predict click" | Class-balanced sampler; per-class F1, not macro accuracy |
| **Stage 1 errors cascade** into Stage 2 | Conditional accuracy given correct type; confidence-threshold fallback to flat decode; consider soft/top-k routing |
| **Taxonomy unification noise** across datasets | Document every judgment call; validate non-point actions carefully |
| **Type gains don't convert to step SR** | Report both; first-error-position analysis |
| **Gain may be small (2-5 pts)** | Frame as controlled study + diagnostics, not SOTA chase |
| **Test-time search (RegionFocus) is a competing explanation** | Acknowledge; argue we improve internal factorization, not inference compute |

---

## 8. Higher-Novelty Pivots (if the team wants more upside)

Surfaced by the reviews, in rough order of effort:
1. **Action-schema-conditioned argument heads** — route to schema-specific argument decoders (click=point, drag=start+end, type=field+text, scroll=container+direction). Closest to current idea, meaningfully fresher.
2. **Confidence-aware selective execution** — joint uncertainty over type *and* grounding; execute / search / verify / defer. Targets a real deployment gap.
3. **Uncertainty-aware type marginalization** — condition grounding on top-k type distribution rather than argmax; turns the classifier from a brittle gate into a probabilistic latent.
4. **Counterfactual one-step lookahead** with a GUI world model — score candidate types by simulating short-horizon consequences. Most ambitious; directly addresses the wrong-type problem.

Current call: **stay the course** with the failure-analysis framing and strong diagnostics. Pivots are backup if early results are flat.

---

## 9. Build Plan (high level)

Detailed in `ROADMAP.md`. The principle: **dumbest end-to-end pipeline first, then improve.** Don't spend three weeks on infrastructure and one on experiments.

0. Repo + env + W&B logging (Day 1)
1. One example through Qwen2-VL + LoRA — proof of life (Days 2-3)
2. Data pipeline: one example → taxonomy → unified dataset → balanced sampler → smoke test (Days 4-7)
3. Stage 1 classifier on cached features + vision-ablation check (Days 8-10)
4. Stage 2 conditioned grounding (Days 11-15)
5. Eval harness — write once, use for all ablations (Days 16-18)
6. Ablations A-D, matched compute (Days 19-22)
7. Analysis, diagnostics, attention viz, writeup (Days 23-28)

---

## 10. Reading List (prioritized, from lit review)

**Read thoroughly first (1-10):** Mind2Web · SeeClick · ShowUI · OS-Atlas · UI-TARS · UGround/SeeAct-V · GUI-Actor · Qwen2-VL · UI-R1 · ScreenSpot-Pro
**Background (11-16):** AITW · AndroidControl · SeeAct (GPT-4V Is a Generalist Web Agent, If Grounded) · WebVoyager · Visual Grounding for User Interfaces · Set-of-Mark
**Conceptual support (17-23):** DECOLA (concept-conditioned localization) · RegionFocus (test-time scaling) · RT-2 · SayCan · OmniParser · OSWorld-G/Jedi · Showdown Clicks artifact

Annotate each in four columns: **problem · representation of action · representation of grounding · how evaluation separates or entangles them.** That scheme makes the related-work section write itself.

---

## 11. Milestone Status

- ✅ **Proposal** (Apr 24) — submitted
- ✅ **Milestone 1** (May 15) — problem + related work
- ✅ **Milestone 2** (May 22) — technical approach
- ⏳ **Milestone 3** (May 29) — preliminary results
- ⏳ **Final report** (Jun 5) — 6-8 pg CVPR format
- ⏳ **Poster** (Jun 10)

---

*This overview consolidates the project proposal, both milestone decks, and two GPT deep-research reviews (literature landscape + novelty/publishability). It supersedes scattered notes; treat it as the single source of truth for scope, framing, and the experiments that actually matter.*
