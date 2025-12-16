"""
Bootstrap and Permutation Statistics
=====================================

Bootstrap confidence intervals and permutation tests.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy import stats

from .base import get_n_bootstrap, get_n_permutations, get_ci_level
from .correlation import compute_correlation


def permute_within_groups(
    n: int,
    rng: np.random.Generator,
    groups: Optional[np.ndarray] = None,
    min_group_size: int = 2,
) -> np.ndarray:
    """Generate permutation indices, optionally within groups.
    
    Raises ValueError if any group has fewer than min_group_size samples.
    """
    if groups is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        return idx

    unique, counts = np.unique(groups, return_counts=True)
    small_groups = unique[counts < min_group_size]
    if len(small_groups) > 0:
        raise ValueError(
            f"Groups {small_groups.tolist()} have fewer than {min_group_size} samples. "
            f"Permutation within groups requires at least {min_group_size} per group."
        )

    idx = np.arange(n)
    for g in unique:
        mask = groups == g
        sub = idx[mask]
        rng.shuffle(sub)
        idx[mask] = sub
    return idx


def perm_pval_simple(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_perm: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> float:
    """
    Simple permutation p-value for correlation.
    
    Returns two-sided p-value.
    """
    if n_perm is None:
        n_perm = get_n_permutations(config)
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan

    x_v = x[valid]
    y_v = y[valid]
    groups_v = groups[valid] if groups is not None else None

    r_obs, _ = compute_correlation(x_v, y_v, method)
    if not np.isfinite(r_obs):
        return np.nan

    n_extreme = 0
    for _ in range(n_perm):
        perm_idx = permute_within_groups(len(x_v), rng, groups_v)
        r_perm, _ = compute_correlation(x_v[perm_idx], y_v, method)
        if np.isfinite(r_perm) and np.abs(r_perm) >= np.abs(r_obs):
            n_extreme += 1

    return (n_extreme + 1) / (n_perm + 1)


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_boot: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Bootstrap CI for correlation using percentile method.
    
    Returns (ci_low, ci_high).
    """
    if n_boot is None:
        n_boot = get_n_bootstrap(config)
    if rng is None:
        rng = np.random.default_rng()

    ci_level = get_ci_level(config)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = np.sum(valid)
    if n_valid < 4:
        return np.nan, np.nan

    x_v = x[valid]
    y_v = y[valid]

    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_valid, size=n_valid)
        r, _ = compute_correlation(x_v[idx], y_v[idx], method)
        if np.isfinite(r):
            boot_rs.append(r)

    if len(boot_rs) < 10:
        return np.nan, np.nan

    boot_rs = np.array(boot_rs)
    alpha = 1 - ci_level
    lo = np.percentile(boot_rs, 100 * alpha / 2)
    hi = np.percentile(boot_rs, 100 * (1 - alpha / 2))

    return float(lo), float(hi)


def bootstrap_mean_ci(
    data: np.ndarray,
    n_boot: Optional[int] = None,
    ci_level: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for the mean.

    Returns (mean, ci_low, ci_high).
    """
    if n_boot is None:
        n_boot = get_n_bootstrap(config)
    if ci_level is None:
        ci_level = get_ci_level(config)
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(data).ravel()
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan

    n = x.size
    boot_means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(x[idx]))

    alpha = 1 - float(ci_level)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(x)), lo, hi


def bootstrap_mean_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_boot: Optional[int] = None,
    ci_level: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for mean difference (group2 - group1)."""
    if n_boot is None:
        n_boot = get_n_bootstrap(config)
    if ci_level is None:
        ci_level = get_ci_level(config)
    if rng is None:
        rng = np.random.default_rng()

    g1 = np.asarray(group1).ravel()
    g2 = np.asarray(group2).ravel()
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    if g1.size < 3 or g2.size < 3:
        return np.nan, np.nan, np.nan

    n1 = g1.size
    n2 = g2.size
    boot_diffs = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)
        boot_diffs[i] = float(np.mean(g2[idx2]) - np.mean(g1[idx1]))

    alpha = 1 - float(ci_level)
    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    return float(np.mean(g2) - np.mean(g1)), lo, hi


def bootstrap_ci_bca(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_boot: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    BCa bootstrap CI for correlation (bias-corrected and accelerated).
    
    Returns (ci_low, ci_high).
    """
    if n_boot is None:
        n_boot = get_n_bootstrap(config)
    if rng is None:
        rng = np.random.default_rng()

    ci_level = get_ci_level(config)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = np.sum(valid)
    if n_valid < 5:
        return np.nan, np.nan

    x_v = x[valid]
    y_v = y[valid]

    r_obs, _ = compute_correlation(x_v, y_v, method)
    if not np.isfinite(r_obs):
        return np.nan, np.nan

    # Bootstrap replicates
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_valid, size=n_valid)
        r, _ = compute_correlation(x_v[idx], y_v[idx], method)
        if np.isfinite(r):
            boot_rs.append(r)

    if len(boot_rs) < 50:
        return bootstrap_corr_ci(x, y, method, n_boot, rng, config)

    boot_rs = np.array(boot_rs)

    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(boot_rs < r_obs))
    if not np.isfinite(z0):
        z0 = 0.0

    # Acceleration factor (jackknife)
    jack_rs = []
    for i in range(n_valid):
        mask = np.ones(n_valid, dtype=bool)
        mask[i] = False
        r, _ = compute_correlation(x_v[mask], y_v[mask], method)
        if np.isfinite(r):
            jack_rs.append(r)

    if len(jack_rs) < 3:
        return bootstrap_corr_ci(x, y, method, n_boot, rng, config)

    jack_rs = np.array(jack_rs)
    jack_mean = np.mean(jack_rs)
    num = np.sum((jack_mean - jack_rs) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_rs) ** 2) ** 1.5)

    a = num / denom if np.abs(denom) > 1e-12 else 0.0

    alpha = 1 - ci_level

    def adjusted_percentile(alpha_q):
        z_alpha = stats.norm.ppf(alpha_q)
        adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return stats.norm.cdf(adj)

    lo_pct = adjusted_percentile(alpha / 2)
    hi_pct = adjusted_percentile(1 - alpha / 2)

    lo_pct = np.clip(lo_pct, 0.01, 0.99)
    hi_pct = np.clip(hi_pct, 0.01, 0.99)

    lo = np.percentile(boot_rs, 100 * lo_pct)
    hi = np.percentile(boot_rs, 100 * hi_pct)

    return float(lo), float(hi)

