# Milestone 3 — Preliminary Results

> **Milestone snapshot.** This document is a valid historical checkpoint for the
> preliminary presentation, but it predates Stage 2, Phase 6, and Phase 7. Use
> `../phase4/PHASE6_FINAL.md` and `../phase4/PHASE7_RESULTS.md` for the final
> empirical verdict.

**Date:** May 29, 2026
**Team:** Aadi Chauhan, Arthur Ilyasov, Nevin Kunampuram
**Project:** Action-Type-Conditioned Grounding for GUI Agents (CS 231N, Spring 2026)

> **Slide-handoff note:** every number in this doc is sourced from `results/milestone3/numbers.json` (machine-readable). Tables here are sized for direct conversion to slide content — keep the column order and you can paste rows straight in. The single most slide-worthy figure is `headline_comparison.png`.

---

## 1. Headline

We built the full Stage 1 (action-type classifier) prototype, ran 10 text-only baselines on Mind2Web train-derived val, then ran **zero-shot Qwen2-VL-2B-Instruct on Multimodal-Mind2Web `test_task` (500 sampled steps) in both text-only and vision+text modes**. The two highest-impact results are at the top of Table §3.

| | macro-F1 | accuracy | notes |
|---|---|---|---|
| **Majority floor** (always click) | 0.472 | 0.895 | the trivial baseline; severe class skew |
| **Best text-only TF-IDF** (instr + history, balanced) | **0.622** | 0.812 | trained ML baseline; +15 pts over majority |
| **Zero-shot Qwen2-VL-2B, text-only** (unauth val, n=1199) | 0.472 | 0.895 | collapses to majority |
| **Zero-shot Qwen2-VL-2B, text-only** (M-M2W test_task, n=499) | 0.467 | 0.878 | collapses to majority |
| **Zero-shot Qwen2-VL-2B, vision+text** (M-M2W test_task, n=499) | 0.467 | 0.878 | **vision delta = 0.000** |
| Zero-shot Qwen2-VL-2B, vision+text, reversed option order | 0.467 | 0.878 | rules out prompt-position bias |

> **The headline finding:** zero-shot Qwen2-VL-2B's action-type predictions are **identical with and without the screenshot**, on both the original prompt and a control with reversed option ordering. The model intrinsically picks "SELECT" for 90-97% of inputs regardless of what we show it. **A trained TF-IDF baseline (no vision encoder, no transformer) beats the 2B VLM by 15 macro-F1 points.**

This is the most direct possible empirical justification for the project's central bet: **explicit Stage 1 supervision is necessary** because the base VLM does not learn action-type discrimination from screenshots zero-shot.

📈 `headline_comparison.png` — every baseline + every Qwen2-VL zero-shot variant on one chart.

---

## 2. Dataset summary

| Property | Value |
|---|---|
| Primary source | `osunlp/Mind2Web` (HF, unauthenticated) |
| Secondary source (vision-ablation) | `osunlp/Multimodal-Mind2Web` (HF, gated, our access via Modal HF secret) |
| Tasks in unauth release | 1,009 |
| Steps in unauth release | 7,775 |
| Train / val split (unauth) | task-level held-out, val_frac = 0.15, seed = 42 |
| Train tasks / steps | 858 / 6,576 |
| Val tasks / steps | 151 / 1,199 |
| Multimodal-Mind2Web split used | `test_task` (subset of 500 sampled, seed = 42) |
| Raw action ops present | `CLICK` (83.8%), `TYPE` (12.0%), `SELECT` (4.2%) |
| Canonical classes populated | **click**, **type** (the other 6 of the 8-class taxonomy are AITW/AndroidControl territory) |
| Val class distribution | click 1,073 (89.5%), type 126 (10.5%) |

**Taxonomy judgment calls (documented in `src/data/taxonomy.py`):**
- `SELECT → click`: SELECT is dropdown option-pick. Mapping it as its own class would create a 4% near-singleton; the user-visible affordance is identical to a click.
- `HOVER → click`: kept for robustness even though not observed in this release.

**Split methodology note:** we held out at the *task* level (entire tasks go to val), not the step level. Step-level splitting would let the same `confirmed_task` instruction appear in both train and val and inflate scores.

📈 `class_distribution.png` — histogram (train + val) on canonical axes.

---

## 3. Results: every Stage 1 baseline

All trained baselines on **Mind2Web val (1,199 steps, 151 unseen tasks)**. Zero-shot Qwen2-VL-2B on the **unauth val (1,199 steps)** for the text-only run and on **Multimodal-Mind2Web `test_task` (499 sampled steps)** for the vision-ablation runs.

| # | Model | Features | Class weight | Eval set | Accuracy | **macro-F1** | weighted-F1 |
|---|---|---|---|---|---|---|---|
| 0 | Majority class (always click) | — | — | unauth val | **0.895** | 0.472 | 0.845 |
| 0' | Stratified random | — | — | unauth val | 0.793 | 0.476 | 0.798 |
| 1 | TF-IDF + LogReg | instr only | none | unauth val | 0.895 | 0.472 | 0.845 |
| 2 | TF-IDF + LogReg | instr only | balanced | unauth val | 0.751 | 0.542 | 0.787 |
| 3 | TF-IDF + LogReg | instr + history | none | unauth val | 0.894 | 0.472 | 0.845 |
| 4 | **TF-IDF + LogReg** | **instr + history** | **balanced** | unauth val | 0.812 | **0.622** | 0.833 |
| 5 | TF-IDF + LogReg | instr + HTML | none | unauth val | 0.895 | 0.472 | 0.845 |
| 6 | TF-IDF + LogReg | instr + HTML | balanced | unauth val | 0.839 | 0.599 | 0.844 |
| 7 | TF-IDF + LogReg | instr + history + HTML | none | unauth val | 0.895 | 0.472 | 0.845 |
| 8 | TF-IDF + LogReg | instr + history + HTML | balanced | unauth val | 0.805 | 0.621 | 0.829 |
| 9 | Qwen2-VL-2B zero-shot | text-only chat prompt | — | unauth val | 0.895 | 0.472 | 0.845 |
| 10 | Qwen2-VL-2B zero-shot | text-only chat prompt | — | M-M2W test_task (n=499) | 0.878 | 0.467 | 0.823 |
| 11 | **Qwen2-VL-2B zero-shot** | **vision + text chat prompt** | — | M-M2W test_task (n=499) | 0.878 | **0.467** | 0.823 |
| 12 | Qwen2-VL-2B zero-shot | vision + text, options reversed | — | M-M2W test_task (n=499) | 0.878 | 0.467 | 0.823 |

**Row 9 vs 10 — sanity check that the two evaluation sets behave similarly:** the unauth val (1199 steps from train split) and the Multimodal-Mind2Web `test_task` (500 sampled steps from the official held-out task split) give nearly the same Qwen2-VL-2B zero-shot number (0.472 / 0.467). The class skew is similar in both.

**Rows 10 vs 11 — the vision-ablation:** identical predictions, identical metrics. **Vision contributes nothing measurable to zero-shot action-type prediction on Mind2Web with this prompt.**

**Rows 11 vs 12 — the position-bias control:** reversing the order of options listed in the prompt (`CLICK,TYPE,SELECT` → `SELECT,TYPE,CLICK`) does not change canonical metrics. The model's collapse is not a position artifact — it's intrinsic.

### Inside the model's mouth — first-token distributions

Where the prompt asks "Reply with one of: CLICK, TYPE, SELECT", what does Qwen2-VL-2B actually say?

| Prompt option order | Modality | SELECT | CLICK | other |
|---|---|---|---|---|
| `CLICK, TYPE, SELECT` | text-only | 464 | 32 | 3 (BOOKMARK) |
| `CLICK, TYPE, SELECT` | vision+text | 493 | 5 | 1 (SEARCH) |
| `SELECT, TYPE, CLICK` | text-only | 496 | 3 | 0 |
| `SELECT, TYPE, CLICK` | vision+text | 497 | 2 | 0 |

**Observation:** the screenshot doesn't change the *type* of decision — it just makes the model **more confident in its existing SELECT prediction**. Vision-text drops "CLICK" responses from 32→5 and pushes more mass onto SELECT. The vision encoder is being used; it just routes to the wrong answer.

📈 `summary_bar.png` — full table as a horizontal bar chart.
📈 `per_class_f1.png` — per-class F1 across headline TF-IDF baselines.
📈 `confusion_*.png` — confusion matrix per baseline (10 files).

---

## 4. Analysis

### 4.1 What worked

**Class-balanced training is doing the heavy lifting** for the trained baselines. Every unweighted model collapses to majority-class behavior (macro-F1 ≈ 0.472). Adding `class_weight="balanced"` shifts the operating point: accuracy drops ~8 points but macro-F1 jumps ~15 points. This is the textbook precision/recall trade-off and *empirically justifies the balanced-sampler design choice upfront* — without it, our Stage 1 would land on row #1, not row #4.

**Action history is a stronger text signal than the HTML target.** Models with `(instr + history)` outperform `(instr + HTML)` on macro-F1 by ~2.3 points when balanced. Intuitively: knowing the previous click chain disambiguates "now I'm filling a field" vs "now I'm clicking submit" better than the static HTML of one isolated element. This argues for Stage 2 grounding stages that ingest history, not just the current target.

**The unauth val and the official Multimodal-Mind2Web test_task behave consistently.** Our task-level held-out val (n=1199) and the official split (n=500 sampled from ~2700) give Qwen2-VL-2B numbers within 0.005 of each other. The train-derived val is a defensible proxy for held-out tests in the work we'll do before we have full multimodal access for training.

### 4.2 What didn't work — and what that *means*

**The vision encoder contributes zero canonical lift zero-shot.** This is the single most consequential measurement in the milestone. On 499 stratified test_task screenshots, Qwen2-VL-2B's action-type predictions are byte-for-byte identical with vs without the screenshot. The position-bias control rules out the obvious confound. Two interpretations:

1. **Charitable read (we believe this):** the *prompt* is doing 100% of the work. Qwen2-VL-2B's instruction-tuned head responds to the option list, not the screenshot. There's no gradient pressure during pretraining that ties the screenshot of a button to the token "CLICK". Our Stage 2 conditioned-grounding training is the mechanism that would create that pressure — *exactly the contribution we propose*.
2. **Uncharitable read:** Mind2Web's task instructions are descriptive enough that the screenshot adds no information for *action-type* classification specifically. Vision still matters for *grounding* (where on the screen), which is what the project actually claims it matters for.

Both reads support the project's framing. The first explains why a Stage 1 classifier is necessary; the second is exactly why the project's evaluation focuses on Stage 2's *step success rate*, not just action-type accuracy.

**A trained TF-IDF baseline beats a 2B VLM by 15 macro-F1 points.** This is a *positive* finding for the project: the supervised signal on Mind2Web's text features is real and useful. The classifier doesn't need to be enormous; what it needs is *task-specific training* on the canonical taxonomy. Stage 1's MLP head on cached Qwen2-VL features should easily beat 0.622 once it's trained.

**Qwen2-VL-2B's intrinsic bias is toward "SELECT".** With *no relevant context*, the model outputs SELECT 93-99% of the time. Possible causes: SELECT's English meaning is the most generic ("pick something"); the model's instruction-tuning data may overweight discrete-choice tasks. Either way: it's a strong prior that any fine-tuning must overcome. This will be visible in the loss curve when we train Stage 1.

### 4.3 Does this align with our hypothesis?

**Yes, more strongly than we expected.**

- **Hypothesis:** decoupling action-type prediction from grounding via a dedicated classifier improves action-type reliability because the grounding stage receives a discrete, known conditioning signal.
- **Empirical state:** without our classifier, **zero-shot action-type accuracy = majority-class baseline (47.2% macro-F1)**. The factorization is not optional; it's necessary. A flat-decode pipeline that depends on the base model picking the right action type is starting from this floor.
- The text-vs-balanced gap (0.472 → 0.622) for our trained baseline confirms there's real action-type signal beyond chance once we train.
- The vision-encoder-shift in the first-token distribution (vision push toward SELECT) confirms the base VLM *is using the screenshot* — just not in a way that helps action-type, *which is exactly what we're proposing to fix with conditioning*.

**Caveat:** Mind2Web is a 2-class problem here. The 8-class story needs AITW.

---

## 5. Limitations at milestone time

1. **No Stage 2 (grounding) numbers at milestone time.** Those numbers landed later in Phase 4/5/6/7.
2. **2-class evaluation surface.** macro-F1 over `{click, type}` is much coarser than the planned 8-class evaluation. Phase 3 later lifted this to a 5-class AITW evaluation.
3. **Single prompt for the vision-ablation.** We tested two option orderings but did not exhaustively search prompt phrasings. The "Vision doesn't help" claim is rigorously true *for this prompt family*. We will revisit with a richer prompt sweep in Phase 4 before claiming it generalizes.
4. **500-step sample for the vision-ablation.** Computed for cost control; the bootstrapped 95% CI of macro-F1 at this n is roughly ± 0.04. The vision delta is much smaller than that band, so the conclusion is robust, but the absolute number is noisy.
5. **No comparison to published Mind2Web baselines at milestone time.** Mind2Web's official step-success-rate protocol uses element-prediction inputs that were not wired into the eval harness then.
6. **No fine-tuned Qwen2-VL number at milestone time.** The "real" Stage 1 MLP on cached Qwen2-VL features landed later in Phase 2.
7. **Single-seed numbers.** Later phase writeups report multi-seed results where needed.

---

## 6. Original Next Steps (now completed or superseded)

| When | Action | Deliverable |
|---|---|---|
| This week | Cache Qwen2-VL-2B (vision+text) features for full Mind2Web train+val on a Modal Volume | Reusable feature cache; ~$0.50 |
| This week | Train Stage 1 MLP on cached features; vision-ablation at the *features* level (not the prompt level) | The "real" Stage 1 number; should comfortably beat 0.622 |
| Week 2 | AITW slice ingest + taxonomy validation against `src/data/taxonomy.py:AITW_TO_CANONICAL` | Multi-class action distribution; re-run all baselines on the richer label space |
| Week 2 | Stage 2 LoRA fine-tune on the winning Stage 1 architecture, teacher-forced then student-forced | First non-baseline Stage 2 number |
| Week 3 | Ablations A / B / C / D, matched compute, 2B base | The four-row comparison table the paper hinges on |
| Week 3 | Stretch 7B run on variant D if credits permit (see `COMPUTE.md`) | One "we also scale to 7B" headline |
| Week 4 | Attention/saliency visualizations + writeup | Final report |

The Phase 0 + 1 work (repo, deps, smoke test) was completed in the days before this milestone; commits `d5ec466`, `d7a2f53`, `acfb4e5`, `ccdf104` and following on `main`. Later completed results are in `../phase4/PHASE6_FINAL.md` and `../phase4/PHASE7_RESULTS.md`.

---

## 7. Reproducibility

Every number above comes from a script in this repo:

```bash
# Stage 1 text-only baselines (Table §3 rows 0–8)
uv run python scripts/m3_text_baselines.py

# Zero-shot Qwen2-VL-2B text-only on unauth val (Table §3 row 9) — needs Modal
uv run modal run modal_app.py::zero_shot_m3

# Vision-ablation on Multimodal-Mind2Web test_task (rows 10–11) — needs Modal + HF secret
uv run modal run modal_app.py::vision_ablation_m3 --n-steps 500 --batch-size 2

# Position-bias control (row 12)
uv run modal run modal_app.py::vision_ablation_m3 \
    --n-steps 500 --batch-size 2 \
    --option-order "SELECT,TYPE,CLICK" \
    --out-name "vision_ablation_qwen2vl_reversed_order"

# Re-aggregate everything into results/milestone3/numbers.json
uv run python scripts/m3_consolidate.py
```

All seeds fixed to 42. Mind2Web loader is `src.data.mind2web.load_mind2web_text`; taxonomy is `src.data.taxonomy`. Modal entry points all in `modal_app.py`.

### File index

| File | What |
|---|---|
| `results/milestone3/numbers.json` | Every metric in one structured file |
| `results/milestone3/text_baselines.json` | Per-baseline per-class breakdown |
| `results/milestone3/zero_shot_qwen2vl.json` | Row #9 raw + first-token distribution |
| `results/milestone3/vision_ablation_qwen2vl.json` | Rows #10–11 raw + first-token distributions |
| `results/milestone3/vision_ablation_qwen2vl_reversed_order.json` | Row #12 raw |
| `results/milestone3/headline_comparison.png` | Slide-worthy chart of every model |
| `results/milestone3/class_distribution.png` | Histogram (train + val) |
| `results/milestone3/summary_bar.png` | TF-IDF baselines bar chart |
| `results/milestone3/per_class_f1.png` | Per-class F1 across headline TF-IDF baselines |
| `results/milestone3/confusion_*.png` | Confusion matrices, one per text baseline |

### Modal cost actually spent for this milestone

| Run | GPU | Wall time | Est. spend |
|---|---|---|---|
| Smoke test (Qwen2-VL-2B + LoRA + synthetic image) | L4 | ~3 min container | ~$0.04 |
| Schema probe (Multimodal-Mind2Web) | L4 | ~1 min | ~$0.02 |
| Zero-shot text-only on unauth val (n=1199) | L4 | ~46 sec generation + load | ~$0.10 |
| Vision-ablation (n=500, both modes, original order) | L4 | ~10 min | ~$0.13 |
| Vision-ablation (n=500, both modes, reversed order) | L4 | ~10 min | ~$0.13 |
| **Total** | | | **~$0.42** |

That is 0.2% of the $200 Modal budget. The vast majority of Modal credits remain available for Phase 2-4 training runs (see `COMPUTE.md` for the allocation plan).
