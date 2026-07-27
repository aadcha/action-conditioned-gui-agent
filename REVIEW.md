# REVIEW — repaired CS231N paper: what changed, what the numbers are now, what remains for a human

_Prepared 2026-07-26. The repair is complete: reanalysis, leakage-free rerun,
paper rewrite, and both builds are done. What remains is human-only._

## TL;DR

The paper was repaired end-to-end per `publication-gate.md` and the integrity
memo. The headline outcome is that **the repair changed the science, not just
the error bars**: on an untouched, goal-disjoint test set, every mechanistic
conclusion of the course-era paper reversed. The paper now reports that
honestly and is framed around it.

**What the untouched test set says (prespecified, Holm-corrected):**

| contrast | Δ hit@0.10 [95% CI] | Holm p | verdict |
|---|---|---|---|
| P1 · B (aux loss) vs A | +0.011 [−0.005, +0.029] | 0.309 | **null** — the exploratory winner does not replicate |
| P2 · D-hook vs A | +0.057 [+0.035, +0.079] | ≤0.0004 | **significant, but oracle** (gold action type at test) |
| P3 · D-token vs B | +0.035 [+0.017, +0.054] | 0.0027 | **reverses in our favour** — the proposal's embedding beats the aux loss, but this is an oracle-vs-deployable contrast (D-token reads the gold type, B reads nothing) |
| P4 · e2e predicted-type vs A | +0.015 [−0.004, +0.034] | 0.309 | **null** — the deployable claim fails |

Supporting facts: Stage-1 action-type accuracy falls from 0.795 on the
frozen validation split to 0.708 on the untouched test region (both are
goal-disjoint from training, so this is an observed generalization gap, not
a diagnosed goal-novelty effect). The predicted-vs-oracle gap (−0.048)
absorbs **76%** of the e2e oracle arm's +0.063 advantage over A, leaving a
residual indistinguishable from zero. Per-class, the *relative* gain is
largest on **scroll** (0.064 → 0.162, 2.5×) versus click (0.312 → 0.366,
1.2×) — but because click is ~3× more frequent it still supplies **61%** of
the absolute gain, so what moved is where the effect concentrates, not which
class contributes most. The **control condition failed**: D-hook gains
+0.047 (p=0.033) on the mix where action type was assumed uninformative, so
we no longer have a valid matched control and say so. Variant C's
confirmatory numbers are partly a decode-format artifact — its parse rate is
91.7% on the test set (reported in the tables); all metrics count parse
failures as misses.

Five distinct reversals versus the exploratory analysis are enumerated in
the paper (§"What the broken evaluation actually cost"): which variant wins,
whether our own hypothesis was refuted, where the gain lives, whether the
pipeline is deployable, and whether the control holds.

## What changed (map to the release gate)

### Statistics (gate: dependence-aware analysis, never quote p=0.0002)
- New `src/eval/clustered.py`: seed-averaged per-example deltas,
  episode-cluster bootstrap CIs (10k), episode-level sign-flip permutation
  p (10k; never below Monte Carlo resolution, never p=0), per-seed means,
  Welch seed-level sensitivity.
- Calibration measured on synthetic clustered nulls
  (`results/phase8_reanalysis/reanalysis.json → validation`): legacy pooled
  test rejects a true null **21.2%** of the time at nominal 5% (1000 sims);
  clustered test **5.1%**. A procedure-level null with run offsets drives
  any fixed-checkpoint test to ~55% → estimand fixed as "these checkpoints",
  per-seed effects always reported, no procedure-level significance claims.
- `scripts/p5_paired_bootstrap.py` hard-deprecated (exits unless explicitly
  overridden), so the invalid pooled p-values cannot be regenerated.
- Full reanalysis of the historical results:
  `results/phase8_reanalysis/REANALYSIS.md`.

### Data split (gate: group-disjoint splits, untouched test, duplicate checks)
- Leakage quantified from a pinned-revision stream reconstruction
  (alignment **proven**: re-derived val slices match every run's recorded
  class distribution exactly): episode overlap ~2% of val steps, but shared
  task goals 24% (headline) → 85% (n=5000). The dominant channel was
  repeated goal strings, invisible to an episode-level check.
- `data/manifests/aitw_frozen_v1.json` (committed, sha-checked): 12,000
  all_with_coords steps of mirror revision 5c0dc713, grouped by normalized
  goal, split goal-disjointly: train 9,511 / val 1,660 / test 829. Test
  groups lie entirely beyond stream index 5,500 — past the furthest index
  any historical run loaded (5,250). 17 near-duplicate train↔test goal pairs
  (Jaccard ≥ 0.8) disclosed in the manifest and the paper's limitations.
  Disjointness enforced by `tests/test_manifest.py`.

### Rerun (decision + protocol)
- **Decision: full rerun** (57 runs = main ablation 5 variants + e2e × 3
  seeds; control mix A/B/C/D-hook × 3 seeds; low-data A/B/D-hook ×
  {300,500,800} × 3 seeds), justified because the effect survived
  reanalysis and the budget allowed. Nothing was scaled down.
- `results/phase9_rerun/PRESPEC.md` committed **before** any test
  evaluation; `scripts/p9_analyze_rerun.py` written before test data existed
  and run on the results as committed. One post-results edit is disclosed in
  the script docstring and below: the Markdown p-value **formatter** only
  (`<1e-4` instead of `<=9.999e-05`). No estimator, contrast, metric,
  cluster definition, resampling budget, or correction changed; no computed
  value changed.
- Frozen data path validated by 6 tiny smoke runs before the batch
  (eval-set identity across variants confirmed).
- **Failed-run log:** the batch was disrupted twice by Modal
  `--detach` clients being killed with the harness's shell timeouts, which
  cancelled queued/running calls (the documented June kill-cascade gotcha).
  No completed run was ever discarded or rerun; the 25 and then 16
  cancelled configs were relaunched with identical seeds/config per the
  prespec failed-run rule, finally via `.spawn()` against the deployed app
  so no local process could cancel them (`scripts/p9_spawn_missing.py`).
  All 57 cells completed exactly once.
- **Spend:** Phase 8 reanalysis was free (CPU only, ~$0.50 for the dataset
  scan). Phase 9's 57 L4 runs ≈ 96,000 train-example-epochs ≈ **$30–40**.
  Cumulative project spend ≈ **$90 of $200**. (Estimated from run configs
  and the documented L4 rate; the live dashboard could not be read in
  session — the browser extension was disconnected. Human: confirm at
  modal.com/settings → usage.)

### Paper (gate: no placeholders, claims match evidence, Mind2Web null visible)
- Reframed as an evaluation-integrity study; the three pitfalls (M-RoPE
  `inputs_embeds` bug, goal leakage, pseudo-replication) are quantified
  contributions, and the five reversals are the paper's central result.
- Exploratory and confirmatory sections explicitly separated; gold/wrong/zero
  framed as sensitivity/oracle; hit@radius never called task success; the
  low-data "confirmed" claim retracted; the Mind2Web null retained; no SOTA
  claims; limitations cover oracle-gating, the failed control, post-hoc
  per-class reading, degenerate `type` coordinates, near-dup goals, single
  2B backbone, and mirror provenance.
- **Every results number is generated from committed JSONs**
  (`scripts/p10_render_paper_tables.py` → `tables/`); no hand-typed results.
  Both PDFs verified free of the `PENDING` sentinel.
- Contributions and Generative-AI statements are real text (no templates).
- Author list: **Aadi Chauhan + Arthur Ilyasov**. The course version had
  equal-contribution asterisks; the rewrite has none — Arthur must approve
  list, order, markers, and contribution wording (`COAUTHOR_NOTE.md`).
- ⚠️ **Poster (`poster.tex`) still carries the old, now-refuted numbers.**
  Do not distribute it until it is regenerated.

### Builds (gate: arXiv + workshop)
- `scripts/build_paper.sh` produces all three artifacts from one shared
  `body.tex`:
  - `overleaf_submission/paper.pdf` — **arXiv build** (CVPR two-column
    style), **11 pp**, everything inline including the Contributions and
    Generative-AI statements.
  - `overleaf_submission/paper_neurips.pdf` — **workshop build** (official
    NeurIPS 2026 style, submission mode), **16 pp total** with **main text
    ending on p. 9** (references p. 10, appendices after), satisfying
    IAEval's "9 pages excluding references and appendices." Verified
    **fully anonymized**: the style prints "Anonymous Author(s)", and the
    author-identifying Contributions / Generative-AI back matter is gated
    out of this build (a `pypdf` scan finds no author name on any page).
    A `\ifworkshopbuild` toggle relocates dataset-preprocessing detail,
    Stage-1 method, the training objective, the exploratory
    Stage-1/per-class/low-data/e2e/causal subsections, three exploratory
    table floats and three figures into appendices, and compacts Related
    Work. **No result is dropped** — each relocated block keeps a
    macro-backed summary plus a pointer in the main text.
  - `dist/arxiv_src.tar.gz` — sources + `.bbl` + only the referenced
    figures, ready to upload.
  - Verified with `pypdf` text extraction: both PDFs contain **zero**
    `PENDING` sentinels and **zero** unresolved `??` references.
  - The workshop build **is anonymized**: without `[final]`,
    `neurips_2026.sty` both adds line numbers and prints "Anonymous
    Author(s)" in place of the author block (verified in the built PDF), so
    it is double-blind-safe as it stands. If IAEval's finalized CFP turns
    out to be single-blind, add `[final]` to the package options to reveal
    authors and drop line numbers.
- **Venue verification (primary sources, 2026-07-26):**
  - **Evaluation of Interactive Agents @ NeurIPS 2026** (target): Aug 29,
    2026 AoE; official NeurIPS 2026 style; ≤9 pages excl. refs/appendix;
    non-archival. OpenReview venue `NeurIPS.cc/2026/Workshop/IAEval` exists
    but **the submission portal is not live yet and the CFP page says "under
    construction" — the anonymization policy is UNCONFIRMED.** Re-check
    https://eval-interactive-agents-workshop.github.io/ before submitting.
  - **VLM4RWD @ NeurIPS 2026** (fallback): portal live, double-blind, 8 pp.
    ⚠️ Its configured hard close is **2026-08-31 04:00 UTC ≈ Aug 30 end-of-day
    US**, not Aug 31 AoE as the site implies.
  - **Who Verifies the Agents?** (2nd fallback): Aug 29 AoE, 4–9 pp, portal
    live; weaker topical fit.
  - The official NeurIPS 2026 workshop list is not yet published; all three
    have provisioned OpenReview venues (de facto accepted).
  - `neurips_2026.sty` is vendored from a mirror — re-download from the
    finalized CFP before submitting.

### Adversarial audit (two multi-agent passes)

Two adversarial audit passes were run over the paper against the committed
artifacts and the gate rules. The first (23 agents) confirmed 17 findings,
all fixed — including a defect in this repo's own table renderer, which was
computing variant C's descriptive statistics on a parse-success-only
denominator while the delta column used the miss-scored convention.

The second pass (48 agents; some verifications were cut short by usage
limits, so its findings were triaged manually against the artifacts) caught
the more serious set, all since fixed:
- **Arithmetic blocker.** The paper claimed the predicted-vs-oracle gap
  (−0.048) was "larger than the entire oracle advantage over A." It is not:
  the advantage is +0.063, so the gap absorbs 76%. Replaced with the
  traceable arithmetic.
- **Wrong causal attribution.** The Stage-1 accuracy drop was attributed to
  "unseen goals," but the frozen validation split is equally goal-disjoint
  from training. Reworded, with an explicit note that the cause was not
  isolated.
- **Unsupported superlative.** The conclusion called D-token "the better of
  the two mechanisms"; D-hook is numerically higher and no D-hook vs
  D-token contrast was prespecified or run. Now claims neither.
- **Missing oracle marking.** "D-token beats B" appeared without noting
  that D-token reads the gold action type and B reads nothing. Marked at
  every occurrence.
- **Overstated mechanism shift.** "The gain concentrates in scroll" ignored
  that click still supplies 61% of the absolute gain. Now stated as a
  relative-versus-absolute distinction.
- **Same denominator bug in the per-class table** (C's scroll rate was
  inflated 0.113 → corrected 0.074), plus parse rates never being reported
  despite a Methods promise and a PRESPEC commitment — both fixed.
- **Double-blind leak.** The anonymized workshop build named both authors in
  its Contributions section; that back matter is now gated out of the
  submission build.
- Stale exploratory claims still asserted as current (the D-token
  refutation, the control-scope conclusion), an inconsistent 5-seed-vs-3-seed
  baseline behind the exploratory per-class deltas, a placeholder citation,
  a `\pm` convention mismatch between tables, an uncensored Holm p-value,
  and a self-contradiction in this file — all corrected.

### Reproducibility (gate)
- **43/43 tests pass** (29 original + clustered-statistics + manifest
  disjointness tests), in a clean `git clone` + `uv sync` environment.
- **Independent-reader reproduction verified, twice**:
  (1) a fresh agent, given only the README, moved the committed outputs
  aside, reran `p8_dependence_reanalysis.py` + `p10_render_paper_tables.py`
  and reproduced `reanalysis.json`, `REANALYSIS.md`, and all `tables/*.tex`
  **byte-identically** (12.1 s + 0.1 s);
  (2) after the rerun landed and all audit corrections were applied, a fresh
  `git clone` + `uv sync` reproduced the **entire** chain byte-identically:
  `p8_dependence_reanalysis.py` + `p9_analyze_rerun.py` +
  `p10_render_paper_tables.py` regenerate `reanalysis.json`,
  `REANALYSIS.md`, `rerun_analysis.json`, `RERUN_RESULTS.md`, and every
  `tables/*.tex` with zero byte differences, and the 43 tests pass in that
  clone. (Analysis JSON is written with `sort_keys=True` so the artifacts
  are order-stable and this check is exact rather than semantic.)
- **Prespecification audit trail** (git log order): PRESPEC.md and
  `p9_analyze_rerun.py` were both committed **before** the first frozen-split
  result artifact entered the repo. One post-results edit to the analysis
  script is disclosed in its docstring: the Markdown p-value formatter now
  prints `<1e-4` instead of `<=9.999e-05`. No estimator, contrast, metric,
  cluster definition, resampling budget, or correction changed; no computed
  value changed.
- Reproduction path documented in the README:
  ```
  uv run python scripts/p8_dependence_reanalysis.py   # historical, clustered
  uv run python scripts/p9_analyze_rerun.py           # untouched test
  uv run python scripts/p10_render_paper_tables.py    # paper tables/macros
  ```

## Open items / flags

- **Third author**: the original proposal listed a third collaborator; the
  manuscript is two-author. Deliberately unresolved — `COAUTHOR_NOTE.md`
  asks Arthur; if any doubt remains, contact that person before posting.
  **Do not post to arXiv until authorship is closed.**
- **Arthur's approvals**: author list/order/markers, contribution statement,
  AI-disclosure wording, and consent to arXiv + the workshop.
- **IAEval CFP finalization**: anonymization policy and portal.
- **Poster is stale** (old numbers) — regenerate or withhold.

## Human-only submission checklist (in order)

1. Send `COAUTHOR_NOTE.md` (edited to taste) to Arthur; get explicit approval
   on authorship, contributions, disclosure, and venues; resolve the
   third-collaborator question.
2. Confirm Modal spend on the usage dashboard matches the ≈$90 figure above.
3. **arXiv**: log in (cs.CV submission rights; first-time submitters may need
   endorsement for cs.CV) → new submission → upload `dist/arxiv_src.tar.gz`
   → primary **cs.CV**, cross-list **cs.AI** → arXiv non-exclusive license →
   submit (announcement takes ~1–2 business days).
4. **Workshop** (after the CFP finalizes): create the OpenReview submission
   at the IAEval venue, upload `overleaf_submission/paper_neurips.pdf`
   (switch the style option to `[final]` only if the finalized CFP says
   single-blind), fill authors/conflicts, submit before **Aug 29, 2026 AoE**.
   If IAEval slips, VLM4RWD closes ~**Aug 30 end-of-day US**.
5. After arXiv announcement: update the README link and tell Arthur.
