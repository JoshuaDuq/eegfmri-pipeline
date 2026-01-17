"""
Bootstrap Statistics
===================

Bootstrap confidence intervals for correlations and means.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from scipy import stats

from .base import get_n_bootstrap, get_ci_level

# Minimum sample sizes for valid statistics
MIN_SAMPLES_BOOTSTRAP_CORR = 4
MIN_SAMPLES_BOOTSTRAP_MEAN = 3
MIN_SAMPLES_BCA = 5

# Minimum bootstrap replicates for valid confidence intervals
MIN_BOOTSTRAP_REPLICATES = 10
MIN_BOOTSTRAP_REPLICATES_BCA = 50
MIN_JACKKNIFE_REPLICATES = 3

# Numerical thresholds
DENOMINATOR_THRESHOLD = 1e-12
PERCENTILE_CLIP_MIN = 0.01
PERCENTILE_CLIP_MAX = 0.99


def _get_bootstrap_config(
    n_boot: Optional[int],
    rng: Optional[np.random.Generator],
    config: Optional[Any],
) -> Tuple[int, np.random.Generator]:
    """Get bootstrap configuration values."""
    if n_boot is None:
        n_boot = get_n_bootstrap(config)
    if rng is None:
        rng = np.random.default_rng()
    return n_boot, rng


def _filter_finite_pairs(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter out non-finite values from paired arrays."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _compute_percentile_bounds(
    ci_level: float,
) -> Tuple[float, float]:
    """Compute lower and upper percentile bounds for confidence interval."""
    alpha = 1 - ci_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    return lower_percentile, upper_percentile


def _bootstrap_corr_ci_impl(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    method: str,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Internal implementation of bootstrap CI for correlation.
    
    Returns (ci_low, ci_high).
    """
    # Local import to avoid circular dependency
    from .correlation import compute_correlation
    
    n_valid = len(x_valid)
    if n_valid < MIN_SAMPLES_BOOTSTRAP_CORR:
        return np.nan, np.nan

    bootstrap_correlations = []
    for _ in range(n_boot):
        indices = rng.integers(0, n_valid, size=n_valid)
        correlation, _ = compute_correlation(
            x_valid[indices], y_valid[indices], method
        )
        if np.isfinite(correlation):
            bootstrap_correlations.append(correlation)

    if len(bootstrap_correlations) < MIN_BOOTSTRAP_REPLICATES:
        return np.nan, np.nan

    bootstrap_correlations = np.array(bootstrap_correlations)
    lower_percentile, upper_percentile = _compute_percentile_bounds(ci_level)
    ci_low = np.percentile(bootstrap_correlations, lower_percentile)
    ci_high = np.percentile(bootstrap_correlations, upper_percentile)

    return float(ci_low), float(ci_high)


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
    n_boot, rng = _get_bootstrap_config(n_boot, rng, config)
    ci_level = get_ci_level(config)
    x_valid, y_valid = _filter_finite_pairs(x, y)
    return _bootstrap_corr_ci_impl(x_valid, y_valid, method, n_boot, ci_level, rng)


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
    n_boot, rng = _get_bootstrap_config(n_boot, rng, config)
    if ci_level is None:
        ci_level = get_ci_level(config)

    data_valid = np.asarray(data).ravel()
    data_valid = data_valid[np.isfinite(data_valid)]
    n_valid = data_valid.size
    if n_valid < MIN_SAMPLES_BOOTSTRAP_MEAN:
        return np.nan, np.nan, np.nan

    bootstrap_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        indices = rng.integers(0, n_valid, size=n_valid)
        bootstrap_means[i] = np.mean(data_valid[indices])

    mean_observed = np.mean(data_valid)
    lower_percentile, upper_percentile = _compute_percentile_bounds(ci_level)
    ci_low = np.percentile(bootstrap_means, lower_percentile)
    ci_high = np.percentile(bootstrap_means, upper_percentile)

    return float(mean_observed), float(ci_low), float(ci_high)


def bootstrap_mean_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_boot: Optional[int] = None,
    ci_level: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for mean difference (group2 - group1)."""
    n_boot, rng = _get_bootstrap_config(n_boot, rng, config)
    if ci_level is None:
        ci_level = get_ci_level(config)

    group1_valid = np.asarray(group1).ravel()
    group2_valid = np.asarray(group2).ravel()
    group1_valid = group1_valid[np.isfinite(group1_valid)]
    group2_valid = group2_valid[np.isfinite(group2_valid)]

    n1 = group1_valid.size
    n2 = group2_valid.size
    if n1 < MIN_SAMPLES_BOOTSTRAP_MEAN or n2 < MIN_SAMPLES_BOOTSTRAP_MEAN:
        return np.nan, np.nan, np.nan

    bootstrap_differences = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        indices1 = rng.integers(0, n1, size=n1)
        indices2 = rng.integers(0, n2, size=n2)
        mean1 = np.mean(group1_valid[indices1])
        mean2 = np.mean(group2_valid[indices2])
        bootstrap_differences[i] = mean2 - mean1

    observed_diff = np.mean(group2_valid) - np.mean(group1_valid)
    lower_percentile, upper_percentile = _compute_percentile_bounds(ci_level)
    ci_low = np.percentile(bootstrap_differences, lower_percentile)
    ci_high = np.percentile(bootstrap_differences, upper_percentile)

    return float(observed_diff), float(ci_low), float(ci_high)


def _compute_bias_correction(
    bootstrap_correlations: np.ndarray,
    observed_correlation: float,
) -> float:
    """Compute bias correction factor z0 for BCa method."""
    proportion_below_observed = np.mean(bootstrap_correlations < observed_correlation)
    z0 = stats.norm.ppf(proportion_below_observed)
    return z0 if np.isfinite(z0) else 0.0


def _compute_acceleration_factor(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    method: str,
) -> Optional[float]:
    """Compute acceleration factor a for BCa method using jackknife.
    
    Returns None if insufficient jackknife replicates, otherwise returns
    acceleration factor (may be 0.0 if denominator is too small).
    """
    # Local import to avoid circular dependency
    from .correlation import compute_correlation
    
    n_valid = len(x_valid)
    jackknife_correlations = []
    for i in range(n_valid):
        mask = np.ones(n_valid, dtype=bool)
        mask[i] = False
        correlation, _ = compute_correlation(x_valid[mask], y_valid[mask], method)
        if np.isfinite(correlation):
            jackknife_correlations.append(correlation)

    if len(jackknife_correlations) < MIN_JACKKNIFE_REPLICATES:
        return None

    jackknife_correlations = np.array(jackknife_correlations)
    jackknife_mean = np.mean(jackknife_correlations)
    deviations = jackknife_mean - jackknife_correlations
    numerator = np.sum(deviations ** 3)
    denominator = 6 * (np.sum(deviations ** 2) ** 1.5)

    if np.abs(denominator) > DENOMINATOR_THRESHOLD:
        return numerator / denominator
    return 0.0


def _compute_bca_adjusted_percentile(
    alpha_quantile: float,
    bias_correction: float,
    acceleration: float,
) -> float:
    """Compute BCa-adjusted percentile from quantile."""
    z_alpha = stats.norm.ppf(alpha_quantile)
    denominator = 1 - acceleration * (bias_correction + z_alpha)
    adjusted_z = bias_correction + (bias_correction + z_alpha) / denominator
    return stats.norm.cdf(adjusted_z)


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
    n_boot, rng = _get_bootstrap_config(n_boot, rng, config)
    ci_level = get_ci_level(config)

    # Local import to avoid circular dependency
    from .correlation import compute_correlation
    
    x_valid, y_valid = _filter_finite_pairs(x, y)
    n_valid = len(x_valid)
    if n_valid < MIN_SAMPLES_BCA:
        return np.nan, np.nan

    observed_correlation, _ = compute_correlation(x_valid, y_valid, method)
    if not np.isfinite(observed_correlation):
        return np.nan, np.nan

    bootstrap_correlations = []
    for _ in range(n_boot):
        indices = rng.integers(0, n_valid, size=n_valid)
        correlation, _ = compute_correlation(
            x_valid[indices], y_valid[indices], method
        )
        if np.isfinite(correlation):
            bootstrap_correlations.append(correlation)

    if len(bootstrap_correlations) < MIN_BOOTSTRAP_REPLICATES_BCA:
        return bootstrap_corr_ci(x, y, method, n_boot, rng, config)

    bootstrap_correlations = np.array(bootstrap_correlations)

    bias_correction = _compute_bias_correction(bootstrap_correlations, observed_correlation)
    acceleration = _compute_acceleration_factor(x_valid, y_valid, method)

    if acceleration is None:
        return bootstrap_corr_ci(x, y, method, n_boot, rng, config)

    alpha = 1 - ci_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    lower_percentile = _compute_bca_adjusted_percentile(
        lower_quantile, bias_correction, acceleration
    )
    upper_percentile = _compute_bca_adjusted_percentile(
        upper_quantile, bias_correction, acceleration
    )

    lower_percentile = np.clip(lower_percentile, PERCENTILE_CLIP_MIN, PERCENTILE_CLIP_MAX)
    upper_percentile = np.clip(upper_percentile, PERCENTILE_CLIP_MIN, PERCENTILE_CLIP_MAX)

    ci_low = np.percentile(bootstrap_correlations, 100 * lower_percentile)
    ci_high = np.percentile(bootstrap_correlations, 100 * upper_percentile)

    return float(ci_low), float(ci_high)


def compute_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for correlation.
    
    Convenience wrapper that uses explicit parameters instead of config object.
    
    Parameters
    ----------
    x, y : np.ndarray
        Input arrays
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        Correlation method ('pearson' or 'spearman')
    rng : np.random.Generator, optional
        Random number generator
    config : Optional[Any]
        Configuration object (overridden by explicit parameters)
        
    Returns
    -------
    Tuple[float, float]
        Lower and upper CI bounds
    """
    if rng is None:
        rng = np.random.default_rng()
    x_valid, y_valid = _filter_finite_pairs(x, y)
    return _bootstrap_corr_ci_impl(x_valid, y_valid, method, n_bootstrap, ci_level, rng)

