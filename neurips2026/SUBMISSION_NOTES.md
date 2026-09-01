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
10. ~~No architecture/method figure exists~~ — DONE Sep 1: TikZ figure in
    `figures/arch_diagram.tex` (`\input` from main.tex, builds).
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

## Launch record — Sep 1, 2026 ~14:20–14:45 PDT (all six approved; 35 detached jobs)

User ruled the old seed-45/46 A/D-hook runs and the old qualitative render
invalid, so E1 re-runs seeds 45/46 for ALL five variants. Modal profile:
`MODAL_PROFILE=sentinel` (see CLAUDE.md). Workspace cap 10 GPUs → jobs queue.

| Exp | Jobs | Expected Volume artifacts |
|---|---|---|
| E1 | A/B/C/Dhook/Dtoken × seeds 45,46 @ n=1200 (+ Dtoken `--causal-eval` × 2) = 12 | `stage2_runs/{variantA,variantB,variantC,Dhook,Dtoken}_seed4{5,6}_n1200_…` |
| E2 | A/B/Dhook × n∈{2500,5000} × seeds 43,44 = 12 | `stage2_runs/…_n{2500,5000}_…` |
| E3 | A, Dhook × seeds 42–44 @ taps_and_swipes n=1000 = 6 | `stage2_runs/…_mix-taps_and_swipes…` (overwrites pre-logging runs) |
| E4 | m2w A, Dhook seed 44 = 2 | `stage2_runs/m2w_{A,Dhook}_seed44_n1000_ep2.json` |
| E5 | `attn_aggregate --variant {Dhook,Dtoken}` (new entrypoint) = 2 | `attn_aggregate/<variant>/…json + png` |
| E6 | `stage2_qualitative --n-train 1200 --n-val 250 --out-subdir qualitative_v2` = 1 | `qualitative_v2/qualitative_grounding.png`, `render.json` |

Cost estimate ≈ $35 (E2's n=5000 runs dominate). Monitoring: `scripts/p8_poll.py`
(`--wave 1` = everything except E2; `--wave 2` = E2). After landing:

    MODAL_PROFILE=sentinel uv run modal run modal_app.py::list_stage2_runs
    MODAL_PROFILE=sentinel uv run modal run modal_app.py::pull_attn_aggregate
    MODAL_PROFILE=sentinel uv run modal volume get stage1-cache qualitative_v2/ results/phase8/qualitative_v2/
    uv run python scripts/p8_consolidate.py && uv run python scripts/p8_attn_summary.py

`scripts/p8_consolidate.py` reproduces the PHASE6/PHASE7 numbers exactly from
the existing JSONs and emits `results/phase8/{PHASE8_RESULTS.md, tables/*.tex,
ablation_headline_vs_control.png, scaling_curve_multiseed.png}`.

## Interim findings (Sep 1, ~16:45 PDT; E1 11/12, E3 6/6, E6 done, E5 v1 D-hook)

- **5-seed headline strengthens the story.** B − A = +0.064*** and D-hook − A =
  +0.054*** (1250 paired units), while D-token − A drops to +0.009 (ns) and
  D-token − B = −0.055***. Ranking B ≈ D-hook > C ≳ D-token ≈ A. The literal
  prepended embedding is now clearly *not* the mechanism.
- **Control now has paired bootstrap for all four variants**: B −0.015 ns,
  D-hook −0.005 ns, C −0.047* on hit@0.10 vs A. Conditioning neutral (B, D-hook)
  to harmful (C) when type is not spatially informative.
- **D-token causal test at 5 seeds**: gold − wrong +0.157***, gold − zero
  +0.069*** (1250 units).
- **New mechanism finding (E5, D-hook, seed 42, 120 probes):** zeroing the
  additive embedding at inference does NOT hurt (hit@0.10 0.392 vs 0.358 gold,
  ns) while a wrong embedding is catastrophic (0.083). D-hook's benefit is a
  *training-time* effect (like B's auxiliary loss); D-token's is partly
  inference-time (its zero condition hurts). The "wrong" collapse is almost
  entirely the click→type swap (click hit 0.432 → 0.012), i.e. conditioning
  onto `type` makes the model emit the degenerate type coordinate — a confound
  in the cyclic wrong map that also affects the D-token causal numbers. E5 v2
  (launched 16:10) adds a click↔scroll "wrong2" map and probes the tokens that
  actually predict the x/y digits (pre_x, pre_y) instead of only the last
  prompt token; v1 showed target-region attention barely above chance
  (0.040 vs 0.026 area share) and changing little with conditioning.
- **Qualitative v2 base rates (n=1200, seed 42):** click hit@0.10 A 0.343 vs
  D-hook 0.391; of 169 clicks: 18 rescued, 11 hurt, 45 both correct, 90 both
  missed. Figures: `figures/qualitative_v2_1x4.png` (main), `_1x6.png` (appendix).

## Writing plan (post-approval)

Port + tighten from `overleaf_submission/paper.tex` (byte-identical to root
`paper.tex`; compiles, ~4,040 words, 6 figures / 6 tables, honest
supported/refuted framing already in place). Reframe for the workshop:
grounding-as-evidence-localization + training-paradigms-for-grounded-VLAs +
M-RoPE pitfall as deployment failure analysis. Expand related work.
8-page budget is roomy vs the CVPR 2-column source; appendix gets per-seed
tables, attn viz, reproduction details.
