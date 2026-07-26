# REVIEW — repaired CS231N paper: what changed, what the numbers are now, what remains for a human

_Prepared 2026-07-26. Status: FILL-PENDING sections are completed after the
confirmatory rerun lands; nothing here is final until they are._

## TL;DR

The paper was repaired end-to-end per `publication-gate.md` and the
integrity memo: (1) the pooled p-values are retracted and replaced with a
validated episode-clustered analysis; (2) the leaky stream-slice split was
replaced by frozen goal-disjoint train/val/untouched-test manifests and
every variant was retrained and evaluated once on the untouched test under
a prespecified analysis; (3) the paper was reframed as an
evaluation-integrity study ("when does action-type supervision help GUI
grounding, and which implementation/evaluation choices create a false
appearance of benefit"), placeholders removed, authors = Aadi Chauhan +
Arthur Ilyasov; (4) two builds compile: arXiv (CVPR style) and NeurIPS 2026
workshop style. Deliverable ends at the submit button; human-only steps
below.

## What changed (map to the release gate)

### Statistics (gate: "dependence-aware analysis, never quote p=0.0002")
- New module `src/eval/clustered.py`: seed-averaged per-example deltas,
  episode-cluster bootstrap CIs (10k), episode-level sign-flip permutation
  p (10k, never below Monte Carlo resolution, never p=0), per-seed means,
  Welch seed-level sensitivity.
- Validated on synthetic clustered nulls
  (`results/phase8_reanalysis/reanalysis.json → validation`): the legacy
  pooled test rejects a true null 21.2% of the time at nominal 5% (1000
  sims); the clustered test 5.1%. A procedure-level null with run offsets
  drives ANY fixed-checkpoint test to ~55% rejection → estimand fixed as
  "these checkpoints", per-seed effects always reported, no
  procedure-level significance claims from 3 seeds.
- `scripts/p5_paired_bootstrap.py` hard-deprecated (exits with a warning).
- Reanalysis of ALL historical results:
  `results/phase8_reanalysis/REANALYSIS.md`. Headline: B−A survives
  (+0.051 hit@0.10, CI [+0.020,+0.081], p=0.0043, Holm 0.017 across the
  vs-A family) vs the retracted pooled p=0.0002. D-hook−A weakens to
  p=0.023 (Holm 0.068). "D-token significantly worse than B" no longer
  holds (p=0.114). The historical e2e "predicted beats A" is NOT
  significant clustered (+0.016, p=0.42).

### Data split (gate: "group-disjoint splits, untouched test, duplicate checks")
- Leakage quantified from a pinned-revision stream reconstruction
  (alignment PROVEN: re-derived val slices match every run's recorded
  class distribution exactly): episode overlap ~2% of val steps, but
  shared task goals 24% (headline cell) → 85% (n=5000). Dominant channel
  = repeated goal strings, not episode straddling.
- `data/manifests/aitw_frozen_v1.json` (committed, sha-checked): first
  12,000 all_with_coords steps of mirror revision 5c0dc713, grouped by
  normalized goal, split goal-disjointly: train 9,511 / val 1,660 / test
  829 steps. Test groups live entirely beyond stream index 5,500 — past
  the furthest index ANY historical run loaded (5,250) — so test steps
  were never touched by any phase of the project. 17 near-duplicate
  train↔test goal pairs (Jaccard ≥ 0.8) disclosed in the manifest and the
  paper's limitations. Disjointness enforced by tests
  (`tests/test_manifest.py`).
- Frozen image shards + checksums on the Modal volume
  (`frozen/aitw_frozen_v1/BUILD.json`); frozen eval sets identical across
  all variants and training sizes.

### Rerun (decision + protocol)
- Decision: FULL rerun (main 5 variants + e2e × 3 seeds; control mix
  A/B/C/D-hook × 3 seeds; low-data grid A/B/D-hook × {300,500,800} × 3
  seeds = 57 runs), since (a) the effect survived reanalysis, (b) budget
  allowed (~$148 remaining vs ~$40–60 est. for the batch), (c) the paper's
  confirmatory section needs every table's clean-test version.
- `results/phase9_rerun/PRESPEC.md` committed BEFORE any test evaluation:
  estimand, frozen eval sets (awc val 250 / test 600; control 200/400),
  primary contrasts P1 B−A, P2 D-hook−A, P3 B vs D-token, P4 e2e
  predicted−A on hit@0.10 with Holm, failed-run rules, reporting
  commitments. Analysis code (`scripts/p9_analyze_rerun.py`) also written
  before test data existed.
- Frozen data path validated end-to-end by 6 tiny smoke runs before the
  batch (eval-set identity across variants confirmed).

### Confirmatory results (FILL-PENDING)
- P1 (B vs A, hit@0.10): __
- P2 (D-hook vs A): __
- P3 (D-token vs B): __
- P4 (e2e predicted vs A): __
- Control mix: __
- Low-data grid: __
- Failed-run log: __ (any infra relaunches recorded here)
- Actual Modal spend for the rerun: __ (budget note: live dashboard could
  not be checked in-session — browser extension disconnected; figures
  above from documented ballparks. Human: glance at
  modal.com/settings → usage for the true number.)

### Paper (gate: "no placeholders, claims match evidence, Mind2Web null visible")
- Reframed; three pitfalls (M-RoPE inputs_embeds bug, goal leakage,
  pseudo-replication) are quantified contributions. Exploratory vs
  confirmatory sections explicitly separated; gold/wrong/zero framed as
  sensitivity/oracle; hit@radius never called task success; low-data
  "confirmed" retracted and reported as noisy/directional; Mind2Web null
  kept; SOTA claims absent; limitations section covers scope, estimand,
  degenerate `type` coords, near-dup goals, mirror provenance.
- EVERY results table/number generated from committed JSONs
  (`scripts/p10_render_paper_tables.py` → `tables/`); no hand-typed
  results. Contributions + Generative-AI statements are real text.
- Author list: Aadi Chauhan + Arthur Ilyasov. NOTE: the course version had
  equal-contribution asterisks; the rewrite currently has none — Arthur
  must approve list, order, markers, and the contribution wording
  (see `COAUTHOR_NOTE.md`).
- Poster (`poster.tex`) NOT updated with corrected numbers — do not
  distribute it until it is.

### Builds (gate: arXiv + workshop)
- `scripts/build_paper.sh` → `overleaf_submission/paper.pdf` (CVPR-style,
  arXiv) and `overleaf_submission/paper_neurips.pdf` (NeurIPS 2026 style,
  submission mode: anonymized + line numbers). Both compile with tectonic.
- Venue verification (primary sources, 2026-07-26):
  - **Evaluation of Interactive Agents @ NeurIPS 2026** (target): deadline
    Aug 29, 2026 AoE; official NeurIPS 2026 style; up to 9 pages full
    papers excl. refs/appendix; non-archival; OpenReview venue
    `NeurIPS.cc/2026/Workshop/IAEval` exists but the submission portal is
    NOT yet live and the CFP page says "under construction" —
    anonymization policy UNCONFIRMED; re-check
    https://eval-interactive-agents-workshop.github.io/ before submitting.
  - **VLM4RWD @ NeurIPS 2026** (fallback): OpenReview portal live, BUT its
    configured hard close is 2026-08-31 04:00 UTC ≈ Aug 30 EOD US — NOT
    Aug 31 AoE as the site implies. Double-blind, 8 pages, NeurIPS format.
  - **Who Verifies the Agents?** (2nd fallback): Aug 29 AoE, 4–9 pages,
    portal live; weaker topical fit.
  - NeurIPS 2026 official workshop list not yet published; all three have
    provisioned OpenReview venues (de facto accepted).
  - neurips_2026.sty vendored from a mirror of the official file —
    re-download from the finalized CFP before submission.

### Reproducibility (gate)
- Tests: __ passing (was 29; now includes clustered-stats + manifest
  disjointness tests).
- Clean-env smoke: __ (FILL after run)
- Independent-reader reproduction of the principal table from saved
  predictions: __ (FILL: fresh-agent run + diff)

## Open items / flags
- **Third author**: the original proposal lists a third collaborator; the
  manuscript is two-author. Unresolved by design — `COAUTHOR_NOTE.md`
  asks Arthur; if any doubt remains, contact the third person before
  posting. DO NOT post to arXiv until authorship is closed.
- **Arthur's approvals**: author list/order/markers, contribution
  statement, AI-disclosure wording, consent to arXiv + workshop.
- IAEval CFP finalization (anonymization + portal).

## Human-only submission checklist (in order)

1. Send `COAUTHOR_NOTE.md` (edited to taste) to Arthur; obtain explicit
   approval on authorship, contributions, disclosure, and venues; resolve
   the third-collaborator question.
2. Glance at Modal usage dashboard; confirm spend matches this file.
3. arXiv: log in (account with cs.CV submission rights; first-time
   submitters may need endorsement for cs.CV), `Developer → new
   submission`: upload `overleaf_submission/` contents (paper.tex,
   body.tex, tables/, figures/, cvpr.sty deps, paper.bib) or the arXiv
   tarball produced by `scripts/build_paper.sh`; primary cs.CV, cross-list
   cs.AI; license: arXiv non-exclusive. Click submit ~2 business days
   before you want it announced.
4. Workshop (after CFP finalizes): create OpenReview submission at the
   IAEval venue; upload `paper_neurips.pdf` source (flip to [final]
   option only if the CFP says single-blind); fill authors/conflicts;
   click submit before Aug 29, 2026 AoE. If IAEval slips, VLM4RWD closes
   ~Aug 30 EOD US (not Aug 31 AoE) — plan accordingly.
5. After arXiv announcement: update README badge/link, tell Arthur.
