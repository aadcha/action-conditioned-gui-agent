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
- **E5 v2 (both variants, seed 42, 120 probes, y-predicting token, 3/4 layer):**
  target_frac@0.10 (chance 0.026) — D-hook: gold 0.111, wrong 0.064, wrong2
  0.049, zero 0.119 (gold−zero −0.008 ns; gold−wrong2 +0.062***). D-token:
  gold 0.095, wrong 0.080, wrong2 0.079, zero 0.079 (gold−zero +0.016***).
  Decoded hit@0.10 — D-hook: 0.358 / 0.083 / 0.100 / 0.392; D-token:
  0.267 / 0.117 / 0.142 / 0.142. So (i) localization happens at the token that
  predicts y, not at the last prompt token; (ii) the non-degenerate wrong map is
  as harmful as the cyclic one, so the "wrong hurts" result is NOT just the
  type-coordinate confound; (iii) D-hook's embedding is redundant at inference
  (zero = gold) while D-token's is used (zero < gold) — attention and decoded
  hits agree. Figure `figures/attn_aggregate.png`, table `tables/attn.tex`,
  full per-position numbers in `results/phase8/ATTN_AGGREGATE.md`.
- **Qualitative v2 base rates (n=1200, seed 42):** click hit@0.10 A 0.343 vs
  D-hook 0.391; of 169 clicks: 18 rescued, 11 hurt, 45 both correct, 90 both
  missed. Figures: `figures/qualitative_v2_1x4.png` (main), `_1x6.png` (appendix).

## Final Phase 8 status (Sep 1, 18:40 PDT) — all 37 jobs complete, ~37 GPU-h ≈ $33

- **Scaling tail at 3 seeds changes one conclusion.** B − A on hit@0.10:
  +0.067*** (300), +0.011 ns (500), −0.040** (800), +0.051*** (1200),
  −0.012 ns (2500), +0.005 ns (5000). D-hook − A: +0.069***, +0.001 ns,
  +0.048**, +0.045**, +0.037**, +0.043** (and positive/significant on hit@0.25
  at all six sizes). So the auxiliary loss's advantage is real only around
  n≈1200 and at n=300, whereas the additive hook helps consistently across
  two orders of magnitude of data. Paper framing: "B ≈ D-hook at the headline
  size; D-hook is the more robust mechanism across n" (replaces PHASE6's
  "B is simplest/best" and the old single-seed 'advantage shrinks with data').
- All other conclusions as in the interim sections above. Consolidated:
  `results/phase8/PHASE8_RESULTS.md`, `ATTN_AGGREGATE.md`, LaTeX tables in
  `neurips2026/tables/`, figures in `neurips2026/figures/` (headline-vs-control,
  scaling_curve_multiseed, attn_aggregate, qualitative_v2_1x4/1x6, arch_diagram).
- User decision (Sep 1): NO re-run of the D-token causal evals with the
  click↔scroll map — "we're working with the data we have available."
  Paper handling: lead the causal section with gold − zero (+0.069***,
  confound-free); report gold − wrong with one sentence that the cyclic map
  routes clicks onto the degenerate `type` coordinate; cite the single-seed
  click↔scroll numbers from E5 v2 (D-hook 0.358→0.100, D-token 0.267→0.142)
  as corroboration that plausible-but-wrong conditioning is also harmful.

## Draft status (Sep 1, ~19:30 PDT)

Full first draft written in `main.tex` (commit 26e34be): 8-page body ending
on page 8 with references starting there, 15 pages total with appendix.
Constraints applied: no em dashes anywhere (`---` count is 0), no stock
LLM phrasing (scanned; only "serves as" was found and removed), tables are
generated by `scripts/p8_consolidate.py` / `p8_attn_summary.py` and wrapped
in `\resizebox` so they fit the column. Figures in the body: architecture,
scaling curve, attention/causal comparison; qualitative and headline bar
figures are in the appendix along with Stage-1, per-class, full scaling,
and per-position attention tables.

Before submission: (1) author proofread, especially the intro framing and
the mechanism section; (2) confirm on the CFP whether the NeurIPS checklist
is required (currently not included); (3) the `ack` block is hidden by the
style at submission; (4) camera-ready needs the `final` option and real
author block; (5) do not mention the CS231N origin (double-blind).

## Revision after the mock panel (Sep 2)

User authorized experiments and rewriting at my discretion. Done so far:
- Analysis: cluster bootstrap over examples + seed-level paired t-tests on
  every contrast; per-class table on the Table-1 basis; sentinel-emission
  analysis (`scripts/p8_sentinel_analysis.py`); e2e paired analysis; scaling
  class mix; exhaustive qualitative categories.
- Paper rewritten (commit 46573c2): headline reframed around the mixed
  action stream with a degenerate-target class; statistics stated at two
  levels; e2e descriptive; scaling descriptive; CoCo-Agent + LiMAC cited and
  positioned; GUI-Libra description corrected; M-RoPE account corrected
  (1-D fallback) and demoted, transformers issue #35463 cited; D-text variant
  added; numbers reconciled (Sec 6.4 vs Table 2; Table 1 vs Table 3;
  qualitative categories partition; "nine points").
- 27 follow-up jobs launched ~14:55 PDT: interventions (Dhook/Dtoken x 5
  seeds, gold/wrong/wrong2/zero/classmean + embedding norms), no-type
  (A/B/Dhook x 3 seeds, type removed from training only, same val slice),
  D-text x 5 seeds, attention v3 (A/Dhook/Dtoken; flat-A row + free-running
  probe). `\pending{}` slots in main.tex mark where they go.

## Revision complete (Sep 2, ~16:30 PDT; commit edbb1ce)

All 27 follow-up runs landed (~$25). What they showed and how the paper
changed:
- Interventions at 5 seeds (both mechanisms): D-hook wrong 0.275→0.054,
  wrong2 →0.099, class-mean →0.206 (+0.070***), zero →0.248 (+0.027, cluster
  CI [−0.006,+0.060], ns); zeroed D-hook ≈ flat A (+0.019 ns). D-token: every
  perturbation collapses it incl. class-mean; zeroed D-token is BELOW A
  (−0.067***). D-hook rows have norm 0.04 vs token-embedding norm 0.58. The
  single-seed "zeroing costs nothing" claim was dropped; paper now says both
  embeddings are consulted at inference and differ in what remains when the
  signal is removed.
- No-type (type events removed from training, same val slice, 3 seeds):
  flat A rises 0.255→0.297 (+0.043**); on the type-free stream D-hook − A =
  +0.008 (ns), B − A = +0.021 (ns) overall (B keeps click +0.039* and
  hit@0.25 +0.053***). Paper now says the gain is protection from the
  degenerate class, not a general spatial prior; consistent with the control.
- D-text (type as a word in the prompt, 5 seeds): +0.057*** over A, within
  noise of B. Added as a sixth row; the refutation is now specific to the
  learned prepended token.
- Attention v3: free-running probe keeps the localization (D-hook 0.092,
  D-token 0.085 vs 0.111/0.095 teacher-forced). Flat-A row: first run crashed
  on a progress print (fixed), re-fired ~16:35; add one appendix sentence when
  it lands.
- Body ends on page 8; refs on page 9; no em dashes; stock-phrase scan clean.

## Writing plan (post-approval)

Port + tighten from `overleaf_submission/paper.tex` (byte-identical to root
`paper.tex`; compiles, ~4,040 words, 6 figures / 6 tables, honest
supported/refuted framing already in place). Reframe for the workshop:
grounding-as-evidence-localization + training-paradigms-for-grounded-VLAs +
M-RoPE pitfall as deployment failure analysis. Expand related work.
8-page budget is roomy vs the CVPR 2-column source; appendix gets per-seed
tables, attn viz, reproduction details.
