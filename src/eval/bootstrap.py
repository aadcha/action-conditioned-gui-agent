"""Paired bootstrap + permutation tests for grounding metrics.

The project plan specifies "paired bootstrap or permutation tests at task/episode
level, 95% CIs". Seed-level mean±SE (n=3–5) is low-powered; the right test pools
per-example results from the SAME val set under two variants and resamples over
examples. This removes between-seed variance and is the standard way to compare
two models on a shared eval set.

Inputs are per-example arrays aligned by example index (variant A vs variant D
evaluated on the identical val examples, in the same order). For grounding we use
the normalized-L2 distance per example and derive hit@r as (dist <= r).

Functions here are pure NumPy — no torch, no network — so they run instantly in
tests and CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedResult:
    metric: str
    mean_a: float
    mean_b: float
    delta: float            # b - a  (B is "ours"/D, A is baseline)
    ci_low: float           # 95% CI on the delta
    ci_high: float
    p_value: float          # two-sided, from the bootstrap sign of delta
    n_examples: int
    n_boot: int
    higher_is_better: bool


# Canonical hit@r metric keys used throughout the codebase map to these radii.
# NOTE: "hit_at_010" means radius 0.10, NOT 10 — the keys are zero-padded
# hundredths (005 -> 0.05, 010 -> 0.10, 025 -> 0.25).
_HIT_RADII = {"hit_at_005": 0.05, "hit_at_010": 0.10, "hit_at_025": 0.25}


def _hit_radius(metric: str) -> float:
    """Resolve a hit@r metric key to its radius. Supports the canonical
    zero-padded keys (hit_at_010 -> 0.10) and an explicit-float form
    (hit_at_0.10 -> 0.10)."""
    if metric in _HIT_RADII:
        return _HIT_RADII[metric]
    suffix = metric.split("hit_at_")[1]
    if "." in suffix:
        return float(suffix)              # e.g. "0.10"
    return int(suffix) / 100.0            # e.g. "010" -> 0.10


def _metric_from_dist(dist: np.ndarray, metric: str) -> np.ndarray:
    """Per-example metric value array from per-example normalized distances."""
    if metric == "mean_normalized_l2":
        return dist
    if metric.startswith("hit_at_"):
        r = _hit_radius(metric)
        return (dist <= r).astype(np.float64)
    raise ValueError(f"unknown metric {metric!r}")


def paired_bootstrap(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    metric: str = "hit_at_010",
    n_boot: int = 10000,
    seed: int = 0,
    higher_is_better: bool | None = None,
) -> PairedResult:
    """Paired bootstrap over examples.

    dist_a, dist_b: per-example normalized-L2 distances for variant A and B,
    aligned by example index (same val example at the same position).
    Resamples example indices with replacement, recomputes (mean_B - mean_A)
    of the chosen metric, and reports the 95% percentile CI + a two-sided
    bootstrap p-value (fraction of resamples whose delta crosses 0).
    """
    dist_a = np.asarray(dist_a, dtype=np.float64)
    dist_b = np.asarray(dist_b, dtype=np.float64)
    if dist_a.shape != dist_b.shape:
        raise ValueError(f"shape mismatch: A {dist_a.shape} vs B {dist_b.shape}")
    n = dist_a.shape[0]
    if higher_is_better is None:
        higher_is_better = metric.startswith("hit_at_")

    va = _metric_from_dist(dist_a, metric)
    vb = _metric_from_dist(dist_b, metric)
    obs_delta = float(vb.mean() - va.mean())

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # paired resample: same idx for A and B
        deltas[i] = vb[idx].mean() - va[idx].mean()

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # two-sided p: how often the bootstrap delta is on the opposite side of 0
    # from the observed delta (i.e. evidence the true delta could be 0).
    if obs_delta >= 0:
        p = 2.0 * float((deltas <= 0).mean())
    else:
        p = 2.0 * float((deltas >= 0).mean())
    p = min(1.0, p)

    return PairedResult(
        metric=metric,
        mean_a=float(va.mean()),
        mean_b=float(vb.mean()),
        delta=obs_delta,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p,
        n_examples=n,
        n_boot=n_boot,
        higher_is_better=higher_is_better,
    )


def permutation_test(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    metric: str = "hit_at_010",
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Paired permutation test: randomly swap A/B per example, two-sided p-value
    for the mean metric difference. Complements the bootstrap CI."""
    dist_a = np.asarray(dist_a, dtype=np.float64)
    dist_b = np.asarray(dist_b, dtype=np.float64)
    va = _metric_from_dist(dist_a, metric)
    vb = _metric_from_dist(dist_b, metric)
    obs = abs(vb.mean() - va.mean())
    n = va.shape[0]
    rng = np.random.default_rng(seed)
    diffs = vb - va
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs((diffs * signs).mean()) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def pool_distances_across_seeds(per_seed_dists: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-seed per-example distance arrays into one pooled array.

    Use when each seed evaluated the same val set: pooling increases the
    effective n for the bootstrap (treats (seed, example) pairs as units).
    All arrays must be the same length (same val set) — but pooling works even
    if not, since we just concatenate.
    """
    return np.concatenate([np.asarray(d, dtype=np.float64) for d in per_seed_dists])
