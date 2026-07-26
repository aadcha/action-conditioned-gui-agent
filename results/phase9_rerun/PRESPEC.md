# Phase 9 — prespecified confirmatory analysis (committed BEFORE any test-set evaluation)

This file freezes every data, model, and analysis decision for the
leakage-free rerun. It is committed to git before the first training job that
touches the untouched test split is launched. Nothing below may change after
test results exist; deviations, if any are ever needed, must be reported as
deviations in the paper.

A git commit is not independent preregistration; we claim only "decisions
frozen before test access," which the commit timestamp and run logs document.

## Estimand

Primary inference concerns the **fixed checkpoints** produced by the runs
below, compared pairwise on a population of held-out AITW steps drawn from
episodes AND task descriptions (goals) never seen in training. Statements
about the training *procedure* are limited to descriptive per-seed effects;
3 seeds are too few for procedure-level significance claims, and none will be
made (the synthetic procedure-null in `results/phase8_reanalysis/` quantifies
why).

## Data (frozen)

- Manifest: `data/manifests/aitw_frozen_v1.json`,
  entries_sha256 `7c918faa874043f97bd6e0ee763985b45c97192be4e7a7a23ff4655305139f13`, built from the
  first 12,000 all_with_coords steps of `cjfcsjt/AITW_General` standard/train
  at pinned revision `5c0dc7139aaf714e69f8d8e4bd2ea2bc5a41700e`.
- Groups = normalized goal strings; episodes never span groups. Splits are
  goal-disjoint (hence episode-disjoint). Test groups lie entirely beyond
  filtered-stream index 5,500 — past the furthest index any historical run
  loaded (5,250) — so no test step was ever trained on, evaluated on, or
  inspected in Phases 1–8.
- Counts: train 9,511 / val 1,660 / test 829 steps. 17 near-duplicate
  (token-Jaccard ≥ 0.8) train↔test goal pairs exist and are disclosed in the
  manifest; they are reported in the paper's limitations.
- Training subsets: first n_train steps of the train split in stream order
  under the mix filter (nested across sizes).
- Eval sets (identical for every variant and every training size):
  - `awc-val`: first 250 val-split steps (all_with_coords filter)
  - `awc-test`: first 600 test-split steps (all_with_coords filter)
  - `ts-val`: first 200 val-split steps (taps_and_swipes filter)
  - `ts-test`: first 400 test-split steps (taps_and_swipes filter)
- Exclusions: none. Every step in the frozen eval sets is scored; parse
  failures score the sqrt(2) sentinel distance (counted as misses) and parse
  rate is reported separately. AITW `type` steps are INCLUDED and their
  degenerate-coordinate pathology is reported per-class, as in Phase 7.

## Runs (matched-compute; everything identical to Phase 6 except the split)

Qwen2-VL-2B-Instruct + LoRA (r16, α32, qkvo), lr 2e-5, batch 1, 2 epochs,
coord_scale 1000, seeds {42, 43, 44}, last-epoch reporting (no checkpoint
selection). `frozen_split=aitw_frozen_v1`, `eval_test=true`.

1. Main ablation (all_with_coords, n_train=1200): variants A, B(λ=1),
   C, D-hook(init 0.0), D-token(init 0.02, causal_eval on val), e2e.
   3 seeds each = 18 runs. n_test=600.
2. Control (taps_and_swipes, n_train=1000, n_val=200): A, B, C, D-hook.
   3 seeds each = 12 runs. n_test=400.
3. Low-data (all_with_coords, n_train ∈ {300, 500, 800}): A, B, D-hook.
   3 seeds each = 27 runs. Same frozen eval sets. n_test=600.

Failed-run rule: a run that crashes (infra error) is relaunched with the SAME
seed and config; a run that completes is never rerun or excluded. All runs
are reported.

## Primary confirmatory contrasts (untouched test, all_with_coords n=1200)

Tested with the episode-clustered paired test of `src/eval/clustered.py`
(seed-averaged per-example deltas, cluster bootstrap CI with 10,000 resamples,
episode-level sign-flip permutation p with 10,000 permutations; the same code
validated on synthetic clustered nulls in `results/phase8_reanalysis/`).

- P1: B vs A, hit@0.10 — auxiliary action supervision
- P2: D-hook vs A, hit@0.10 — inference-time action conditioning
- P3: B vs D-token, hit@0.10 — does the proposal's prepended-embedding
  mechanism match the simple auxiliary loss?
- P4: e2e predicted-type pipeline vs A, hit@0.10 — deployable claim

Holm correction across {P1..P4}. Secondary metrics (hit@0.05, hit@0.25, mean
normalized L2), secondary contrasts (C vs A, D-token vs A, D-hook vs B,
predicted vs oracle), per-class decompositions, the control-mix results, the
low-data grid, and the D-token gold/wrong/zero sensitivity are all reported
with CIs but labeled secondary/descriptive — no headline claims hang on them.

Direction: P1/P2/P4 predict positive deltas; P3 is a two-sided comparison with
no predicted winner (Phase 6's exploratory result suggested B ≥ D-token).
Gold-type conditioning results are oracle ceilings, not deployable claims;
wrong/zero-embedding results are sensitivity analyses, not mechanism proofs.

## Reporting commitments

- Effect sizes with cluster-aware 95% CIs are primary; permutation p-values
  are secondary and never reported below Monte Carlo resolution (no p=0).
- Per-seed means are reported for every contrast.
- Parse rate, per-class metrics, and coverage are reported for every run.
- The historical (leaky-split) results remain in the paper as the exploratory
  phase, clearly labeled, alongside the Phase 8 reanalysis.
- If any confirmatory contrast is null or negative on the untouched test, it
  is reported as the headline result and the paper's framing already
  accommodates it (failure-analysis / evaluation-pitfalls contribution).
- The Mind2Web null result remains visible in the paper.
