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
| P2 · D-hook vs A | +0.057 [+0.035, +0.079] | 0.0004 | **significant, but oracle** (gold action type at test) |
| P3 · D-token vs B | +0.035 [+0.017, +0.054] | 0.0027 | **reverses in our favour** — the proposal's embedding beats the aux loss |
| P4 · e2e predicted-type vs A | +0.015 [−0.004, +0.034] | 0.309 | **null** — the deployable claim fails |

Supporting facts: Stage-1 action-type accuracy falls 0.795 (val) → 0.708
(unseen goals); the predicted-vs-oracle gap (−0.048) exceeds the entire
oracle gain, so classifier error consumes it. Per-class, the gain is
proportionally largest on **scroll** (0.064 → 0.162, 2.5×), not click
(0.312 → 0.366) — overturning the exploratory "click disambiguation" story.
The **control condition failed**: D-hook gains +0.047 (p=0.033) on the mix
where action type was assumed uninformative, so we no longer have a valid
matched control and say so.

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
  evaluation; `scripts/p9_analyze_rerun.py` written before test data
  existed and run once, unchanged (git history shows both).
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
- `scripts/build_paper.sh` → `overleaf_submission/paper.pdf` (CVPR style,
  arXiv, 8 pp), `overleaf_submission/paper_neurips.pdf` (NeurIPS 2026 style,
  submission mode: anonymized + line numbers, 12 pp with 3 exploratory
  figures relocated to a post-references appendix, so the main body is
  within the 9-page limit), and `dist/arxiv_src.tar.gz` (sources + .bbl +
  used figures, ready to upload).
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

### Reproducibility (gate)
- **43/43 tests pass** (29 original + clustered-statistics + manifest
  disjointness tests), in a clean `git clone` + `uv sync` environment.
- **Independent-reader reproduction verified, twice**:
  (1) a fresh agent, given only the README, moved the committed outputs
  aside, reran `p8_dependence_reanalysis.py` + `p10_render_paper_tables.py`
  and reproduced `reanalysis.json`, `REANALYSIS.md`, and all `tables/*.tex`
  **byte-identically** (12.1 s + 0.1 s);
  (2) after the rerun landed, a second clean `git clone` + `uv sync`
  reproduced the **confirmatory** chain — `p9_analyze_rerun.py` +
  `p10_render_paper_tables.py` regenerate `rerun_analysis.json`,
  `RERUN_RESULTS.md`, and every `tables/*.tex` **byte-identically** from
  the committed run JSONs.
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
