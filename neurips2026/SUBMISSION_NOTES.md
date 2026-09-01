# VLM4RWD 2026 submission — working notes

Target: 2nd Workshop on Grounded and Faithful Vision-Language Models for
Real-World Deployment (VLM4RWD), NeurIPS 2026, Sydney.
Deadline **Sep 5, 2026, 8:59 PM PDT** (extended; no second extension expected).
8 pages excluding references/appendices, NeurIPS 2026 format, double-blind,
non-archival. OpenReview: `NeurIPS.cc/2026/Workshop/VLM4RWD`.
Site: https://vlm4rwd.github.io/

## Build

- `main.tex` — skeleton, `\usepackage[dblblindworkshop]{neurips_2026}` +
  `\workshoptitle{...}`. Camera-ready: add `final` option.
- `make` builds with tectonic. `figures/` holds copies of the 350-dpi PNGs
  from `results/`. NeurIPS checklist file present but `\input` commented out
  (workshops usually don't require it; confirm against CFP before submit).

## Caliber calibration vs the 2025 edition (17 accepted papers)

- 2025 acceptance ≈ 74% (17/~23); tiers: 3 Spotlight, 10 Regular,
  2 Extended Abstract, 2 Demo. Reviews are private on OpenReview.
- Bottom quartile of accepted papers: 1–2 author, API-only prompting or
  "preliminary evaluations", single dataset, no training, no variance
  reporting (Algoverse student projects, extended abstracts).
- Top tier: lab papers (UIUC SAFFRON, Huawei VSC-RL, Fudan Seg-R1,
  U. Macau VisAidMath) — 7B–72B models or 8×A100 training, multi-benchmark.
  Even there, multi-seed variance reporting is rare (MetaTPT's 3 runs is the
  exception; Seg-R1 reports none).
- Our genre-neighbors among accepted papers: Seg-R1 (small Qwen-VL
  fine-tune, ~7k examples, ablations, zero seeds), MetaTPT (single-GPU
  controlled tuning study, 3 runs), VSC-RL (AitW device-control RL — closest
  topically; its baseline list is a ready related-work roster: DigiRL,
  AutoUI, CogAgent, AppAgent, WebRL, Set-of-Marks).
- Negative/diagnostic framing has precedent (VisAidMath was a Spotlight).
- No accepted 2025 paper does action-type-conditioned grounding; the niche
  is unoccupied at this venue. 2026 topics shifted toward
  grounding/faithfulness, which fits us better than 2025's efficiency slant.

**Verdict:** rigor (3–5 seeds, matched-compute controls, paired bootstrap,
causal intervention, honest refutation) exceeds the median accepted Regular;
scale (2B model, n_train ≤ 5k, ~$52 L4 compute) sits between the student
tier and the lab tier. Comfortably above the 2025 acceptance bar; plausible
Regular, outside shot at Spotlight with a crisp headline.

## Known weaknesses to fix before/while writing

1. **3-vs-5 seed inconsistency (must decide).** Seeds 45/46 exist for A and
   D-hook at n=1200 all_with_coords; headline table uses 3 seeds
   (A 0.255±0.021, D-hook 0.300±0.028) while 5-seed values are lower
   (A 0.229±0.037, D-hook 0.275±0.043) with the delta intact (+0.046).
   Defensible move: 5 seeds everywhere → needs seeds 45/46 for B, C, D-token.
2. **Scaling-curve tail is single-seed** (n=2500, 5000) and non-monotonic;
   paper.tex L347 overclaims a "clean low-data prior" relative to
   PHASE7's own honest text. Fix language and/or add seeds.
3. **Attention figure** (`attn_example_1.png`, 120 dpi): goal text doesn't
   match screenshot; ordering claim holds on only 2/3 logged examples (click
   conditioning attends most even when gold is scroll). Reword or replace
   with aggregate attention-mass stats over many val examples; likely drop
   the heatmap or move to appendix.
4. **Qualitative figure** is 4 selected D-hook wins from an n_train=800
   model where the aggregate click gap is 1.3 pts. Keep only as
   illustration with an honest caption, or re-render from n=1200 model.
5. **Control (taps_and_swipes) has no paired bootstrap** (runs predate
   per-example logging).
6. **Mind2Web Stage 2 is 2 seeds.**
7. **Bootstrap resample count discrepancy**: paper.tex says 1000, scripts
   and PHASE5/PHASE6 say 10k. Verify and fix.
8. **Table 5 (per-class) caption discusses variant C but the table has no C
   column.**
9. **references.bib**: 15 entries, `and others` author truncation
   throughout, `regionfocus` has literal `Anonymous` author. Needs
   expansion to ~30–40 verified entries. `reference/deep-research-report-
   literature-review.md` has the landscape (verify every number — its
   citations are search-tool artifacts).
10. **No architecture/method figure exists** — must be drawn (fig:arch
    placeholder in main.tex).
11. CS231N back-matter (Contributions, GenAI statement) not applicable;
    drop for the workshop version.

## Candidate pre-submission experiments (cheap, L4, ~25 min/run)

| # | What | Runs | ~Cost | Value |
|---|------|------|-------|-------|
| E1 | Seeds 45/46 for B, C, D-token @ n=1200 all_with_coords → uniform 5-seed headline | 6 | ~$3 | High |
| E2 | Seeds 43/44 for A/B/D-hook @ n=2500 and n=5000 → multi-seed scaling tail | 12 | ~$10–15 | Med-high |
| E3 | Re-run taps_and_swipes A/B/D-hook with per-example logging → paired bootstrap on control | 6 | ~$3 | Medium |
| E4 | Mind2Web Stage 2 seed 44 for A/D-hook | 2 | ~$1.50 | Medium |
| E5 | Aggregate attention-mass eval over ~100 val examples (replaces 3-example anecdote) | 1 | ~$1–2 | Medium |
| E6 | Re-render qualitative panels from n=1200 headline model | 1 | ~$0.50 | Low-med |

Total if all approved: ~$20–25 of the ~$148 remaining Modal budget; all
submittable in parallel (one `modal run --detach` per background Bash — see
CLAUDE.md burn warning).

## Writing plan (post-approval)

Port + tighten from `overleaf_submission/paper.tex` (byte-identical to root
`paper.tex`; compiles, ~4,040 words, 6 figures / 6 tables, honest
supported/refuted framing already in place). Reframe for the workshop:
grounding-as-evidence-localization + training-paradigms-for-grounded-VLAs +
M-RoPE pitfall as deployment failure analysis. Expand related work.
8-page budget is roomy vs the CVPR 2-column source; appendix gets per-seed
tables, attn viz, reproduction details.
