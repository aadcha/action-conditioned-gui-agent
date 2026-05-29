# Milestone 3 — Preliminary Results

**Date:** May 29, 2026
**Team:** Aadi Chauhan, Arthur Ilyasov, Nevin Kunampuram
**Project:** Action-Type-Conditioned Grounding for GUI Agents (CS 231N, Spring 2026)

> **Slide-handoff note:** every number in this doc is sourced from `results/milestone3/numbers.json` (machine-readable) and the per-baseline figures listed at the bottom. Tables here are sized for direct conversion to slide content — keep the column order and you can paste rows straight in.

---

## 1. Headline

We built the full Stage 1 (action-type classifier) prototype on Mind2Web text data and ran four head-to-head baselines. The intended Stage 1 fine-tune on Qwen2-VL-2B + Stage 2 conditioned grounding are scoped for the post-milestone window — see §6.

| Metric | Value | Interpretation |
|---|---|---|
| **Best text-only Stage 1 macro-F1** | **0.622** (TF-IDF + history, class-balanced) | +15 pts over the majority-class floor — the classification problem is real but partially solvable from text alone. |
| **Majority-class floor (macro-F1)** | 0.472 | "Always predict click" already gets 89.5% accuracy — class skew is severe. |
| **Mind2Web active class count** | **2** (after canonicalization) | The HF unauth release only contains CLICK / TYPE / SELECT raw ops. SELECT → click per our spec leaves 2 classes. |
| **Class skew (val)** | 89.5% click / 10.5% type | Worse than the ~80% click figure cited in the project overview. |

**Why these numbers matter:** the macro-F1 lift from 0.472 → 0.622 is *exactly* the "Stage 1 signal alone" the project hypothesizes. If we had only seen majority-class behavior, the Stage 2 conditioning argument would collapse. We didn't — there's real signal to condition on.

---

## 2. Dataset summary

| Property | Value |
|---|---|
| Source | `osunlp/Mind2Web` (HF, unauthenticated release) |
| Tasks | 1,009 |
| Steps (actions) | 7,775 |
| Train / val split | task-level held-out, val_frac = 0.15, seed = 42 |
| Train tasks / steps | 858 / 6,576 |
| Val tasks / steps | 151 / 1,199 |
| Raw action ops present | `CLICK` (83.8%), `TYPE` (12.0%), `SELECT` (4.2%) |
| Canonical classes populated | **click**, **type** (the other 6 of the 8-class taxonomy are AITW/AndroidControl territory) |
| Val class distribution | click 1,073 (89.5%), type 126 (10.5%) |

**Taxonomy judgment calls (documented in `src/data/taxonomy.py`):**
- `SELECT → click`: SELECT is dropdown option-pick. Mapping it as its own class would create a 4% near-singleton; the user-visible affordance is identical to a click.
- `HOVER → click`: kept for robustness even though not observed in this release.

**Split methodology note:** we held out at the *task* level (entire tasks go to val), not the step level. Step-level splitting would let the same `confirmed_task` instruction appear in both train and val and inflate scores.

📈 `class_distribution.png` — train vs val histogram on the canonical axes.

---

## 3. Results: Stage 1 text-only baselines

All numbers on **Mind2Web val (1,199 steps, 151 unseen tasks)**. Trained on the 6,576-step task-held-out train slice.

| # | Model | Features | Class weight | Accuracy | **macro-F1** | weighted-F1 |
|---|---|---|---|---|---|---|
| 0 | Majority class (always click) | — | — | **0.895** | 0.472 | 0.845 |
| 0' | Stratified random | — | — | 0.793 | 0.476 | 0.798 |
| 1 | TF-IDF + LogReg | instruction only | none | 0.895 | 0.472 | 0.845 |
| 2 | TF-IDF + LogReg | instruction only | balanced | 0.751 | 0.542 | 0.787 |
| 3 | TF-IDF + LogReg | instr + history | none | 0.894 | 0.472 | 0.845 |
| 4 | **TF-IDF + LogReg** | **instr + history** | **balanced** | 0.812 | **0.622** | 0.833 |
| 5 | TF-IDF + LogReg | instr + HTML | none | 0.895 | 0.472 | 0.845 |
| 6 | TF-IDF + LogReg | instr + HTML | balanced | 0.839 | 0.599 | 0.844 |
| 7 | TF-IDF + LogReg | instr + history + HTML | none | 0.895 | 0.472 | 0.845 |
| 8 | TF-IDF + LogReg | instr + history + HTML | balanced | 0.805 | 0.621 | 0.829 |
| 9 | Zero-shot Qwen2-VL-2B (Modal) | instr + history + HTML, text-only prompt | — | _pending_ | _pending_ | _pending_ |

📈 `summary_bar.png` — full table as a horizontal bar chart with the majority floor.
📈 `per_class_f1.png` — per-class F1 for the 5 headline rows above.
📈 `confusion_*.png` — confusion matrix per baseline (10 files).

---

## 4. Analysis

### 4.1 What worked

**Class-balanced training is doing the heavy lifting.** Every unweighted model collapses to majority-class behavior (macro-F1 ≈ 0.472, indistinguishable from "always click"). Adding `class_weight="balanced"` shifts the operating point: accuracy drops ~8 points but macro-F1 jumps ~15 points. This is exactly the trade-off the roadmap anticipated and *justifies the balanced sampler design upfront*.

**Action history helps more than the HTML target.** Models with `(instr + history)` outperform `(instr + HTML)` on macro-F1 by ~2.3 points when balanced. Intuitively: knowing the previous click chain disambiguates "now I'm filling a field" vs "now I'm clicking submit" better than the static HTML of one element. This argues for a Stage 2 grounding stage that ingests history, not just the current target.

**Even simple text features are non-trivially informative.** The 0.622 ceiling we hit isn't 1.0 — there's meaningful headroom for vision to fill. But it's also not 0.472, so the type/click decision *can* be made from text alone a lot of the time. This is the project's "risk #1 — text leakage" surfacing as a real measurement.

### 4.2 What didn't work (or wasn't expected)

**Mind2Web alone is a 2-class problem.** The HF unauth release contains only `CLICK`/`TYPE`/`SELECT`. After our canonicalization (SELECT → click), val has 89.5% click / 10.5% type. There is no `scroll`, no `drag`, no `hotkey` to evaluate — those classes are 0 in our train and val. This means **Mind2Web cannot, by itself, validate the multi-class action-type contribution of our paper**. The 8-class story needs AITW or AndroidControl data, both of which require additional Phase 2 work.

**HTML targets in the canonical release are noisy.** Spot-checking: target HTML for a first step like "Check for pickup restaurant…" includes elements like a "Skip to main content" link rather than the intended target. This is a Mind2Web preprocessing artifact and inflates the variance of the `(instr + HTML)` baseline. We document but don't fix yet — the fine-tuned classifier should be robust if exposure during training matches eval.

**A vision-only Stage 1 number is not in this milestone.** Two reasons:
1. The unauth Mind2Web HF release doesn't ship screenshots; the multimodal release (`osunlp/Multimodal-Mind2Web`) is HF-license-gated.
2. We are intentionally not paying Modal credits to fine-tune on Mind2Web until we've also pulled an AITW slice (so the four ablations train on the same multi-class data the paper claims to evaluate on).

### 4.3 Does this align with our hypothesis?

**Mostly yes, with one important caveat.**

- The text-vs-balanced gap (0.472 → 0.622) confirms there's *real action-type signal beyond chance* — Stage 1 is a legitimate prediction problem, not a degenerate "always click" rubber-stamping. The factorization argument therefore has something to factor.
- The 89.5% click skew confirms the project's stated motivation: a flat baseline trained on this data will over-commit to click; explicitly forcing a type decision (our Stage 1) is doing useful work.
- **Caveat:** because Mind2Web is 2-class, the most diagnostic ablations (per-class confusion on `scroll`, `drag`, etc.) require AITW. We adjusted the Phase 2 plan accordingly — see §6.

---

## 5. Limitations

1. **No Stage 2 (grounding) numbers yet.** All current results are on the action-type classifier. Stage 2 needs an LoRA fine-tune on Qwen2-VL-2B which we have scoped for Modal but haven't run.
2. **No vision-encoder contribution measured.** We can't separate "vision helps Stage 1" from "text saturates Stage 1" without (a) screenshots and (b) a frozen-VLM features baseline. Both are scoped for the next week.
3. **2-class evaluation surface.** macro-F1 over `{click, type}` is a much coarser test than the planned 8-class evaluation. AITW pulls this up to 5–6 active classes.
4. **No comparison to published Mind2Web baselines yet.** We are not running Mind2Web's official `step success rate` because that requires the screenshot release + the official element-prediction protocol. Scoped for Phase 5 (eval harness).
5. **Single-seed numbers.** All baselines run with a single seed. Variance bands are not in this milestone; will add bootstrapped 95% CIs in Phase 5 (cheap — same eval harness).
6. **Engineering-only verification of the GPU path.** `scripts/smoke_test.py` and the Modal wrapper are written but the Modal run was blocked on auth at submission time. Local import tests pass; first cloud run is scheduled immediately post-milestone.

---

## 6. Next Steps (ordered)

| When | Action | Deliverable |
|---|---|---|
| This week | `modal run modal_app.py::smoke` once auth lands — confirms the 7B/2B + LoRA stack runs end-to-end | A 1-line "trainable params: ~0.1%" log line + generated text |
| This week | `modal run modal_app.py::zero_shot_m3` — Qwen2-VL-2B zero-shot baseline (row #9 above) | Fills the open cell in Table §3; first apples-to-apples comparison to a frontier model |
| Week 2 | Multimodal-Mind2Web screenshots into a Modal Volume | Real vision+text Stage 1 features; vision-ablation answer |
| Week 2 | AITW slice ingest + taxonomy validation | Multi-class action distribution; re-run all baselines on richer label space |
| Week 2 | Stage 2 LoRA fine-tune on the winning baseline architecture | First non-baseline Stage 2 number |
| Week 3 | Ablations A / B / C / D, matched compute, 2B base | The four-row comparison table that the paper hinges on |
| Week 3 | Stretch 7B run on variant D if credits permit (see `COMPUTE.md`) | One "we also scale to 7B" headline |
| Week 4 | Attention/saliency visualizations + writeup | Final report |

The Phase 0 + 1 work (repo, deps, smoke test) was completed in the days before this milestone; commits `d5ec466`, `d7a2f53`, `acfb4e5`, `ccdf104` and following on `main`.

---

## 7. Reproducibility

Every number above comes from a script in this repo:

```bash
# Stage 1 text-only baselines (Table §3 rows 0–8)
uv run python scripts/m3_text_baselines.py

# Re-aggregate everything into results/milestone3/numbers.json
uv run python scripts/m3_consolidate.py

# Zero-shot Qwen2-VL-2B baseline (Table §3 row 9) — needs Modal auth
modal run modal_app.py::zero_shot_m3
```

All seeds fixed to 42. Mind2Web loader is `src.data.mind2web.load_mind2web_text`; taxonomy is `src.data.taxonomy`. Confusion matrices, class distribution figure, per-class F1 figure, and summary bar chart are all auto-generated alongside the JSON.

**File index:**

| File | What |
|---|---|
| `results/milestone3/numbers.json` | Every metric in one structured file |
| `results/milestone3/text_baselines.json` | Per-baseline per-class breakdown |
| `results/milestone3/class_distribution.png` | Histogram (train + val) |
| `results/milestone3/summary_bar.png` | All baselines, macro-F1 + accuracy |
| `results/milestone3/per_class_f1.png` | Per-class F1 across headline baselines |
| `results/milestone3/confusion_*.png` | Confusion matrices, one per baseline |
| `results/milestone3/zero_shot_qwen2vl.json` | Created when the Modal job lands |
