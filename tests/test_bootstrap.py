"""Tests for the paired bootstrap / permutation eval (src.eval.bootstrap)."""

import numpy as np

from src.eval.bootstrap import (
    paired_bootstrap,
    permutation_test,
    pool_distances_across_seeds,
)


def test_identical_distributions_give_null():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 0.5, size=400)
    # B is a copy of A -> delta exactly 0, CI brackets 0, p ~ 1
    res = paired_bootstrap(d, d.copy(), metric="hit_at_010", n_boot=2000, seed=1)
    assert abs(res.delta) < 1e-9
    assert res.ci_low <= 0 <= res.ci_high
    assert res.p_value > 0.5


def test_b_clearly_better_on_hit_is_significant():
    # A misses everything (dist 0.5), B hits everything (dist 0.0) at r=0.10
    n = 300
    a = np.full(n, 0.5)
    b = np.full(n, 0.0)
    res = paired_bootstrap(a, b, metric="hit_at_010", n_boot=2000, seed=2)
    assert res.mean_a == 0.0          # A never within 0.10
    assert res.mean_b == 1.0          # B always within 0.10
    assert res.delta == 1.0
    assert res.ci_low > 0.5           # CI well above 0
    assert res.p_value < 0.05


def test_mean_l2_lower_is_better_flag():
    a = np.full(100, 0.30)
    b = np.full(100, 0.20)
    res = paired_bootstrap(a, b, metric="mean_normalized_l2", n_boot=1000, seed=3)
    assert res.higher_is_better is False
    # delta = mean_b - mean_a = -0.10 (B closer -> better)
    assert abs(res.delta - (-0.10)) < 1e-9


def test_permutation_test_detects_difference():
    n = 200
    a = np.full(n, 0.5)
    b = np.full(n, 0.0)
    p = permutation_test(a, b, metric="hit_at_010", n_perm=1000, seed=4)
    assert p < 0.05


def test_permutation_test_null_is_large_p():
    rng = np.random.default_rng(5)
    d = rng.uniform(0, 0.5, size=200)
    p = permutation_test(d, d.copy(), metric="hit_at_010", n_perm=1000, seed=6)
    assert p > 0.5


def test_pool_distances_concatenates():
    a = [np.array([0.1, 0.2]), np.array([0.3])]
    pooled = pool_distances_across_seeds(a)
    assert pooled.shape == (3,)
    assert np.allclose(pooled, [0.1, 0.2, 0.3])


def test_hit_threshold_parsing():
    a = np.array([0.04, 0.06, 0.20])
    b = np.array([0.04, 0.06, 0.20])
    # hit@0.05: only the 0.04 example counts -> mean 1/3
    res = paired_bootstrap(a, b, metric="hit_at_0.05", n_boot=200, seed=7)
    assert abs(res.mean_a - (1.0 / 3.0)) < 1e-9
