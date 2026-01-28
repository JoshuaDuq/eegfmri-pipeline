"""
Bootstrap Statistics
===================

Bootstrap confidence intervals for correlations.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .base import get_n_bootstrap, get_ci_level

# Minimum sample sizes for valid statistics
MIN_SAMPLES_BOOTSTRAP_CORR = 4
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

