"""Tests for the dependence-aware clustered paired analysis."""

import numpy as np
import pytest

from src.eval.clustered import (
    ClusteredPairedResult,
    clustered_paired_test,
    simulate_clustered_pair,
    validate_type1_error,
)


def test_p_value_never_zero():
    rng = np.random.default_rng(0)
    # Enormous synthetic effect: p must bottom out at 1/(n_perm+1), never 0.
    a = [rng.uniform(0.5, 1.0, size=100)]
    b = [rng.uniform(0.0, 0.05, size=100)]
    r = clustered_paired_test(a, b, list(range(100)), metric="mean_normalized_l2",
                              n_boot=200, n_perm=999, seed=0)
    assert r.p_value == pytest.approx(1.0 / 1000.0)
    assert r.p_value > 0.0
    assert "p<=" in r.p_str()


def test_detects_large_effect_and_ci_excludes_zero():
    rng = np.random.default_rng(1)
    da, db, cids = simulate_clustered_pair(rng, true_delta=0.3, n_episodes=80)
    r = clustered_paired_test(da, db, cids, metric="mean_normalized_l2",
                              n_boot=1000, n_perm=999, seed=1)
    assert r.delta < 0  # B distances lower
    assert r.ci_high < 0
    assert r.p_value < 0.05


def test_null_is_not_rejected_too_often():
    # Small smoke version of the full type-I validation under the
    # fixed-checkpoint population null (checkpoint_sd=0): clustered test near
    # nominal; the pooled test inflated by the shared-across-seeds effect
    # heterogeneity it wrongly treats as independent replication.
    out = validate_type1_error(n_sims=60, n_perm=200, seed=7,
                               n_episodes=40, steps_per_episode=4, n_seeds=3)
    assert out["clustered_reject_rate"] <= 0.15
    assert out["pooled_reject_rate"] > out["clustered_reject_rate"]


def test_seed_means_and_shapes():
    rng = np.random.default_rng(2)
    da = [rng.uniform(0, 1, 50) for _ in range(3)]
    db = [rng.uniform(0, 1, 50) for _ in range(2)]  # unequal seed counts OK
    cids = [i // 5 for i in range(50)]
    r = clustered_paired_test(da, db, cids, metric="hit_at_010",
                              n_boot=100, n_perm=99, seed=0)
    assert r.n_seeds_a == 3 and r.n_seeds_b == 2
    assert len(r.seed_means_a) == 3 and len(r.seed_means_b) == 2
    assert r.n_examples == 50 and r.n_clusters == 10


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        clustered_paired_test([np.zeros(10)], [np.zeros(9)], list(range(10)))
    with pytest.raises(ValueError):
        clustered_paired_test([np.zeros(10)], [np.zeros(10)], list(range(9)))


def test_cluster_structure_matters():
    # With strong episode-level effect heterogeneity, clustering by episode
    # must widen the CI relative to pretending examples are independent.
    rng = np.random.default_rng(3)
    da, db, cids = simulate_clustered_pair(
        rng, true_delta=0.0, n_episodes=30, steps_per_episode=8,
        episode_sd=0.3, example_sd=0.01, noise_sd=0.01,
        effect_episode_sd=0.2, effect_example_sd=0.0)
    r_ep = clustered_paired_test(da, db, cids, metric="mean_normalized_l2",
                                 n_boot=2000, n_perm=99, seed=0)
    r_ex = clustered_paired_test(da, db, list(range(len(cids))),
                                 metric="mean_normalized_l2",
                                 n_boot=2000, n_perm=99, seed=0)
    width_ep = r_ep.ci_high - r_ep.ci_low
    width_ex = r_ex.ci_high - r_ex.ci_low
    assert width_ep > width_ex * 1.5


def test_procedure_null_shows_fixed_checkpoint_limitation():
    # Under a PROCEDURE-level null with run-to-run checkpoint offsets, the
    # fixed-checkpoint clustered test rejects far above nominal — the
    # quantified reason per-seed effects must accompany any clustered result.
    out = validate_type1_error(n_sims=40, n_perm=200, seed=11,
                               n_episodes=40, steps_per_episode=4, n_seeds=3,
                               checkpoint_sd=0.08)
    assert out["clustered_reject_rate"] > 0.3
