"""Dependence-aware (clustered) paired analysis for grounding metrics.

Replaces the pooled ``(seed, example)`` analysis in ``src.eval.bootstrap`` for
every inferential claim. The pooled routine concatenates per-seed per-example
distances and treats each ``(seed, example)`` row as an independent
observation. Repeated predictions of the SAME example by different seeds are
strongly correlated, and examples from the same AITW episode share a task,
app, device and UI state, so they are correlated too. Pooling therefore
overstates the effective sample size and the resulting CI/p-value is
anti-conservative (``validate_type1_error`` below measures how badly).

What this module does instead:

1. collapses the seed dimension: for each example ``i`` the paired difference
   is averaged over seeds, ``delta_i = mean_s m(b_si) - mean_s m(a_si)``,
   which is the paired effect of the FIXED trained checkpoints on example i;
2. clusters examples by an explicit cluster id (AITW episode id; pass example
   indices to get example-level clustering when episode ids are unknown);
3. bootstraps over clusters (episodes resampled with replacement) for the CI;
4. computes an episode-level sign-flip permutation p-value, reported with its
   Monte Carlo resolution ``1/(n_perm+1)`` — it can never be 0.

Inference target: the fixed checkpoints produced by the training runs,
evaluated over the population the eval examples are drawn from. Seed-level
means are returned alongside so seed-to-seed variability stays visible; three
seeds are too few for a training-procedure random-effects claim, and nothing
here pretends otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.eval.bootstrap import _metric_from_dist


@dataclass
class ClusteredPairedResult:
    metric: str
    mean_a: float
    mean_b: float
    delta: float                 # b - a on the seed-averaged per-example metric
    ci_low: float                # 95% cluster-bootstrap percentile CI on delta
    ci_high: float
    p_value: float               # episode-level sign-flip permutation, two-sided
    p_resolution: float          # 1/(n_perm+1): the smallest reportable p
    n_examples: int
    n_clusters: int
    n_seeds_a: int
    n_seeds_b: int
    n_boot: int
    n_perm: int
    seed_means_a: list[float] = field(default_factory=list)
    seed_means_b: list[float] = field(default_factory=list)

    def p_str(self) -> str:
        """Render the p-value honestly: never below Monte Carlo resolution."""
        if self.p_value <= self.p_resolution:
            return f"p<={self.p_resolution:.4g}"
        return f"p={self.p_value:.4g}"


def _stack_metric(dists: Sequence[np.ndarray], metric: str) -> np.ndarray:
    """(n_seeds, n_examples) matrix of metric values from per-seed distances."""
    rows = [np.asarray(_metric_from_dist(np.asarray(d, dtype=np.float64), metric))
            for d in dists]
    n = rows[0].shape[0]
    for r in rows:
        if r.shape[0] != n:
            raise ValueError(f"per-seed arrays have mismatched lengths: {[x.shape[0] for x in rows]}")
    return np.stack(rows, axis=0)


def _cluster_index(cluster_ids: Sequence) -> tuple[np.ndarray, int]:
    """Map arbitrary cluster labels to 0..C-1 (first-appearance order)."""
    seen: dict = {}
    out = np.empty(len(cluster_ids), dtype=np.int64)
    for i, c in enumerate(cluster_ids):
        if c not in seen:
            seen[c] = len(seen)
        out[i] = seen[c]
    return out, len(seen)


def clustered_paired_test(
    dists_a: Sequence[np.ndarray],
    dists_b: Sequence[np.ndarray],
    cluster_ids: Sequence,
    metric: str = "hit_at_010",
    n_boot: int = 10000,
    n_perm: int = 10000,
    seed: int = 0,
) -> ClusteredPairedResult:
    """Cluster-bootstrap CI + sign-flip permutation test on the paired delta.

    dists_a / dists_b: one per-seed array of per-example normalized distances
    per trained checkpoint (list lengths may differ between variants; each
    array is aligned to the same eval examples in the same order).
    cluster_ids: one hashable id per example (episode id). Pass
    ``range(n_examples)`` for example-level clustering.
    """
    A = _stack_metric(dists_a, metric)   # (Sa, N)
    B = _stack_metric(dists_b, metric)   # (Sb, N)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B eval sets differ: {A.shape[1]} vs {B.shape[1]}")
    n = A.shape[1]
    if len(cluster_ids) != n:
        raise ValueError(f"cluster_ids len {len(cluster_ids)} != n_examples {n}")

    a_i = A.mean(axis=0)                 # seed-averaged per-example metric
    b_i = B.mean(axis=0)
    delta_i = b_i - a_i
    obs = float(delta_i.mean())

    cidx, n_clusters = _cluster_index(cluster_ids)
    # Per-cluster sums and sizes for vectorized resampling.
    csum = np.bincount(cidx, weights=delta_i, minlength=n_clusters)
    csize = np.bincount(cidx, minlength=n_clusters).astype(np.float64)

    rng = np.random.default_rng(seed)

    # --- cluster bootstrap CI (resample episodes with replacement) ---
    pick = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot_sum = csum[pick].sum(axis=1)
    boot_n = csize[pick].sum(axis=1)
    boot_delta = boot_sum / boot_n
    ci_low, ci_high = np.percentile(boot_delta, [2.5, 97.5])

    # --- episode-level sign-flip permutation test (two-sided) ---
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n_clusters))
    perm_stat = np.abs(signs @ csum) / n
    count = int((perm_stat >= abs(obs) - 1e-12).sum())
    p = (count + 1) / (n_perm + 1)

    return ClusteredPairedResult(
        metric=metric,
        mean_a=float(a_i.mean()),
        mean_b=float(b_i.mean()),
        delta=obs,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p),
        p_resolution=1.0 / (n_perm + 1),
        n_examples=n,
        n_clusters=n_clusters,
        n_seeds_a=A.shape[0],
        n_seeds_b=B.shape[0],
        n_boot=n_boot,
        n_perm=n_perm,
        seed_means_a=[float(x) for x in A.mean(axis=1)],
        seed_means_b=[float(x) for x in B.mean(axis=1)],
    )


def seed_level_welch(
    dists_a: Sequence[np.ndarray],
    dists_b: Sequence[np.ndarray],
    metric: str = "hit_at_010",
) -> dict:
    """Welch t-test on the per-seed metric means — the (low-powered, honest)
    procedure-level sensitivity check. With 3 seeds per side this has ~2-4
    degrees of freedom; treat it as a robustness report, not the primary test.
    """
    from scipy import stats as _stats

    A = _stack_metric(dists_a, metric).mean(axis=1)
    B = _stack_metric(dists_b, metric).mean(axis=1)
    if len(A) < 2 or len(B) < 2:
        return {"metric": metric, "seed_means_a": A.tolist(), "seed_means_b": B.tolist(),
                "delta": float(B.mean() - A.mean()), "t": None, "df": None, "p_value": None,
                "note": "fewer than 2 seeds on one side; no test possible"}
    t, p = _stats.ttest_ind(B, A, equal_var=False)
    # Welch-Satterthwaite df
    va, vb = A.var(ddof=1) / len(A), B.var(ddof=1) / len(B)
    df = (va + vb) ** 2 / (va ** 2 / (len(A) - 1) + vb ** 2 / (len(B) - 1)) if (va + vb) > 0 else float(len(A) + len(B) - 2)
    return {"metric": metric, "seed_means_a": A.tolist(), "seed_means_b": B.tolist(),
            "delta": float(B.mean() - A.mean()), "t": float(t), "df": float(df),
            "p_value": float(p)}


# ---------------------------------------------------------------------------
# Validation on synthetic clustered data (type-I error / power).
# ---------------------------------------------------------------------------


def simulate_clustered_pair(
    rng: np.random.Generator,
    n_episodes: int = 60,
    steps_per_episode: int = 4,
    n_seeds: int = 3,
    true_delta: float = 0.0,
    episode_sd: float = 0.25,
    example_sd: float = 0.15,
    effect_episode_sd: float = 0.08,
    effect_example_sd: float = 0.08,
    checkpoint_sd: float = 0.0,
    noise_sd: float = 0.15,
    base: float = 0.4,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Simulate per-seed per-example 'distances' with the dependence structure
    of the real data.

    Two distinct nulls, matching the two possible estimands:

    - FIXED-CHECKPOINT / population null (``true_delta=0, checkpoint_sd=0``):
      the B-A effect has zero population mean but varies by episode/example
      (``effect_*_sd``), and that heterogeneity is shared across seeds — the
      same replication structure that makes pooling (seed, example) rows
      anti-conservative. An episode-clustered test should be ~nominal here.
    - PROCEDURE null (``true_delta=0, checkpoint_sd>0``): additionally, each
      trained checkpoint carries a run-level offset. A test that conditions on
      the fixed checkpoints (including the episode-clustered one) is NOT
      calibrated for this null — only seed-level inference is. Simulating this
      quantifies why per-seed effects must be reported alongside.
    """
    n = n_episodes * steps_per_episode
    cluster_ids = [e for e in range(n_episodes) for _ in range(steps_per_episode)]
    ep_eff = rng.normal(0.0, episode_sd, size=n_episodes)
    ex_base = base + rng.normal(0.0, example_sd, size=n) + np.repeat(ep_eff, steps_per_episode)

    # Per-example B-A effect, shared across seeds (heterogeneous, mean true_delta).
    t_i = (
        true_delta
        + np.repeat(rng.normal(0.0, effect_episode_sd, size=n_episodes), steps_per_episode)
        + rng.normal(0.0, effect_example_sd, size=n)
    )

    def draw(effect: np.ndarray | float) -> list[np.ndarray]:
        out = []
        for _ in range(n_seeds):
            ckpt_eff = rng.normal(0.0, checkpoint_sd) if checkpoint_sd > 0 else 0.0
            d = ex_base - effect + ckpt_eff + rng.normal(0.0, noise_sd, size=n)
            out.append(np.clip(d, 0.0, np.sqrt(2.0)))
        return out

    return draw(0.0), draw(t_i), cluster_ids  # lower distance = better B


def validate_type1_error(
    n_sims: int = 300,
    alpha: float = 0.05,
    metric: str = "mean_normalized_l2",
    n_perm: int = 400,
    seed: int = 0,
    **sim_kwargs,
) -> dict:
    """Reject-rate of the clustered test vs the legacy pooled bootstrap on a
    TRUE NULL with clustered structure. The clustered rate should be ~alpha;
    the pooled rate is expected to be inflated. Returns both rates plus their
    Monte Carlo standard errors.
    """
    from src.eval.bootstrap import paired_bootstrap, pool_distances_across_seeds

    rng = np.random.default_rng(seed)
    rej_clustered = 0
    rej_pooled = 0
    for k in range(n_sims):
        da, db, cids = simulate_clustered_pair(rng, true_delta=0.0, **sim_kwargs)
        r = clustered_paired_test(da, db, cids, metric=metric,
                                  n_boot=200, n_perm=n_perm, seed=int(rng.integers(1 << 31)))
        if r.p_value <= alpha:
            rej_clustered += 1
        pooled = paired_bootstrap(
            pool_distances_across_seeds(da),
            pool_distances_across_seeds(db),
            metric=metric, n_boot=400, seed=int(rng.integers(1 << 31)),
        )
        if pooled.p_value <= alpha:
            rej_pooled += 1
    out = {
        "n_sims": n_sims,
        "alpha": alpha,
        "clustered_reject_rate": rej_clustered / n_sims,
        "pooled_reject_rate": rej_pooled / n_sims,
        "mc_se": float(np.sqrt(alpha * (1 - alpha) / n_sims)),
    }
    return out


def validate_power(
    true_delta: float = 0.05,
    n_sims: int = 200,
    alpha: float = 0.05,
    metric: str = "mean_normalized_l2",
    seed: int = 1,
    **sim_kwargs,
) -> dict:
    """Reject-rate of the clustered test under a known real effect, so we can
    say the repaired test still detects effects of the observed magnitude."""
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(n_sims):
        da, db, cids = simulate_clustered_pair(rng, true_delta=true_delta, **sim_kwargs)
        r = clustered_paired_test(da, db, cids, metric=metric,
                                  n_boot=200, n_perm=400, seed=int(rng.integers(1 << 31)))
        if r.p_value <= alpha:
            rej += 1
    return {"n_sims": n_sims, "alpha": alpha, "true_delta": true_delta,
            "clustered_power": rej / n_sims}
