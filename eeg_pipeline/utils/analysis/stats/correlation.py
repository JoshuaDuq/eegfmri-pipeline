"""
Correlation Statistics
======================

Correlation computation, partial correlations, and Fisher aggregation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_ci_level, get_config_value


def get_correlation_method(use_spearman: bool) -> str:
    """Return correlation method name."""
    return "spearman" if use_spearman else "pearson"


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """
    Compute correlation coefficient and p-value.
    
    Returns (r, p).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan, np.nan

    x_v, y_v = x[valid], y[valid]

    if np.std(x_v) < 1e-12 or np.std(y_v) < 1e-12:
        return np.nan, np.nan

    if method == "spearman":
        r, p = stats.spearmanr(x_v, y_v)
    else:
        r, p = stats.pearsonr(x_v, y_v)

    return float(r), float(p)


def fisher_z(r: float) -> float:
    """Fisher z-transform of correlation coefficient."""
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    """Inverse Fisher z-transform."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def fisher_ci(
    r: float,
    n: int,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute Fisher-based CI for correlation."""
    ci_level = get_ci_level(config)

    if n < 4 or not np.isfinite(r):
        return np.nan, np.nan

    z = fisher_z(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + ci_level) / 2)

    z_lo = z - z_crit * se
    z_hi = z + z_crit * se

    return float(inverse_fisher_z(z_lo)), float(inverse_fisher_z(z_hi))


def fisher_aggregate(
    rs: List[float],
    config: Optional[Any] = None,
) -> Tuple[float, float, float, int]:
    """
    Aggregate correlations via Fisher z-transform.
    
    Returns (r_mean, ci_low, ci_high, n_valid).
    """
    ci_level = get_ci_level(config)
    rs_arr = np.asarray(rs, dtype=float)
    valid = np.isfinite(rs_arr)

    if np.sum(valid) == 0:
        return np.nan, np.nan, np.nan, 0

    rs_v = rs_arr[valid]
    zs = np.array([fisher_z(r) for r in rs_v])

    z_mean = np.mean(zs)
    r_mean = inverse_fisher_z(z_mean)

    if len(zs) > 1:
        se = np.std(zs, ddof=1) / np.sqrt(len(zs))
        z_crit = stats.norm.ppf((1 + ci_level) / 2)
        z_lo = z_mean - z_crit * se
        z_hi = z_mean + z_crit * se
        ci_lo = inverse_fisher_z(z_lo)
        ci_hi = inverse_fisher_z(z_hi)
    else:
        ci_lo = ci_hi = r_mean

    return float(r_mean), float(ci_lo), float(ci_hi), int(np.sum(valid))


def weighted_fisher_aggregate(
    rs: List[float],
    weights: List[float],
    config: Optional[Any] = None,
) -> Tuple[float, float, float, int]:
    """Weighted aggregation of correlations."""
    ci_level = get_ci_level(config)

    rs_arr = np.asarray(rs, dtype=float)
    ws_arr = np.asarray(weights, dtype=float)

    valid = np.isfinite(rs_arr) & np.isfinite(ws_arr) & (ws_arr > 0)
    if np.sum(valid) == 0:
        return np.nan, np.nan, np.nan, 0

    rs_v = rs_arr[valid]
    ws_v = ws_arr[valid]
    ws_v = ws_v / ws_v.sum()

    zs = np.array([fisher_z(r) for r in rs_v])
    z_mean = np.sum(zs * ws_v)
    r_mean = inverse_fisher_z(z_mean)

    if len(zs) > 1:
        var_z = np.sum(ws_v * (zs - z_mean) ** 2)
        se = np.sqrt(var_z / len(zs))
        z_crit = stats.norm.ppf((1 + ci_level) / 2)
        ci_lo = inverse_fisher_z(z_mean - z_crit * se)
        ci_hi = inverse_fisher_z(z_mean + z_crit * se)
    else:
        ci_lo = ci_hi = r_mean

    return float(r_mean), float(ci_lo), float(ci_hi), int(np.sum(valid))


def joint_valid_mask(*arrays: Sequence, require_all: bool = True) -> np.ndarray:
    """Create joint validity mask across multiple arrays."""
    if not arrays:
        return np.array([], dtype=bool)

    masks = [np.isfinite(np.asarray(a)) for a in arrays]
    if require_all:
        return np.all(np.stack(masks), axis=0)
    return np.any(np.stack(masks), axis=0)


def partial_corr_xy_given_Z(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    method: str,
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    """
    Partial correlation of x,y controlling for Z.
    
    Returns (r_partial, p_value, n).
    """
    from scipy.linalg import lstsq

    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    if len(df) < Z.shape[1] + 3:
        return np.nan, np.nan, 0

    X_mat = df[Z.columns].values
    X_mat = np.column_stack([np.ones(len(X_mat)), X_mat])

    x_vals = df["x"].values
    y_vals = df["y"].values

    # Residualize
    try:
        beta_x, *_ = lstsq(X_mat, x_vals)
        beta_y, *_ = lstsq(X_mat, y_vals)
    except Exception:
        return np.nan, np.nan, 0

    res_x = x_vals - X_mat @ beta_x
    res_y = y_vals - X_mat @ beta_y

    r, p = compute_correlation(res_x, res_y, method)
    return float(r), float(p), len(df)


def compute_partial_corr(
    x: pd.Series,
    y: pd.Series,
    Z: Optional[pd.DataFrame],
    method: str,
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "",
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    """
    Compute partial correlation, handling edge cases.
    
    If Z is None or empty, returns simple correlation.
    """
    if Z is None or Z.empty:
        valid = np.isfinite(x.values) & np.isfinite(y.values)
        if np.sum(valid) < 3:
            return np.nan, np.nan, 0
        r, p = compute_correlation(x.values[valid], y.values[valid], method)
        return r, p, int(np.sum(valid))

    return partial_corr_xy_given_Z(x, y, Z, method, config)


def normalize_series(s: pd.Series, epsilon: float = 1e-12) -> pd.Series:
    """Z-score normalize a series."""
    std = s.std()
    if std < epsilon:
        return pd.Series(np.zeros_like(s.values), index=s.index)
    return (s - s.mean()) / std


def compute_correlation_pvalue(r_values: List[float], config: Optional[Any] = None) -> float:
    """Compute p-value for aggregated correlation (one-sample t-test on z)."""
    rs = np.asarray(r_values, dtype=float)
    valid = np.isfinite(rs) & (np.abs(rs) < 1)

    if np.sum(valid) < 2:
        return np.nan

    zs = np.array([fisher_z(r) for r in rs[valid]])
    t_stat = np.mean(zs) / (np.std(zs, ddof=1) / np.sqrt(len(zs)))
    p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(zs) - 1))
    return float(p)


###################################################################
# Bayes Factor for Correlations
###################################################################


def compute_bayes_factor_correlation(
    x: np.ndarray,
    y: np.ndarray,
    prior_width: float = 0.707,
    method: str = "spearman",
) -> Tuple[float, str]:
    """
    Compute Bayes Factor for H1: r≠0 vs H0: r=0.
    
    Uses the Jeffreys-Zellner-Siow (JZS) prior approximation.
    
    Parameters
    ----------
    x, y : array-like
        Data arrays
    prior_width : float
        Width of the Cauchy prior on r (default: sqrt(2)/2 ≈ 0.707)
    method : str
        Correlation method ("spearman" or "pearson")
    
    Returns
    -------
    Tuple[float, str]
        (BF10, interpretation)
        BF10 > 1: evidence for H1 (correlation exists)
        BF10 < 1: evidence for H0 (no correlation)
        
    Interpretation thresholds (Jeffreys):
        BF < 1: Evidence for H0
        1-3: Anecdotal
        3-10: Moderate
        10-30: Strong
        30-100: Very strong
        >100: Extreme
    """
    from scipy.special import hyp2f1
    
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(valid))
    
    if n < 4:
        return np.nan, "insufficient_data"
    
    x_v, y_v = x[valid], y[valid]
    
    # Compute correlation
    if method == "spearman":
        r, _ = stats.spearmanr(x_v, y_v)
    else:
        r, _ = stats.pearsonr(x_v, y_v)
    
    if not np.isfinite(r) or np.abs(r) >= 1:
        return np.nan, "invalid_r"
    
    # JZS Bayes Factor approximation (Wetzels & Wagenmakers, 2012)
    # BF10 ≈ ((1 + t²/ν)^(-(ν+1)/2)) / Beta((1/2), (ν/2)) * integral term
    # Simplified approximation using hypergeometric function
    r2 = r ** 2
    
    # Compute BF10 using the exact formula for correlation
    # Based on Ly et al. (2016) "Harold Jeffreys's default Bayes factor hypothesis tests explained"
    try:
        # Log BF10 for numerical stability
        log_bf = (
            np.log(np.sqrt(2) / prior_width)
            + 0.5 * np.log(n - 1)
            + ((n - 1) / 2) * np.log(1 - r2)
            + np.log(hyp2f1(0.5, 0.5, (n + 1) / 2, r2))
        )
        bf10 = np.exp(log_bf)
    except (ValueError, OverflowError, RuntimeWarning):
        # Fallback: simpler approximation
        t = r * np.sqrt((n - 2) / (1 - r2))
        bf10 = np.sqrt((n + 1) / (2 * np.pi)) * (1 + t**2 / n) ** (-(n + 1) / 2)
    
    # Interpretation
    if bf10 < 1/100:
        interp = "extreme_H0"
    elif bf10 < 1/30:
        interp = "very_strong_H0"
    elif bf10 < 1/10:
        interp = "strong_H0"
    elif bf10 < 1/3:
        interp = "moderate_H0"
    elif bf10 < 1:
        interp = "anecdotal_H0"
    elif bf10 < 3:
        interp = "anecdotal_H1"
    elif bf10 < 10:
        interp = "moderate_H1"
    elif bf10 < 30:
        interp = "strong_H1"
    elif bf10 < 100:
        interp = "very_strong_H1"
    else:
        interp = "extreme_H1"
    
    return float(bf10), interp


###################################################################
# Robust Correlations (Outlier-Resistant)
###################################################################


def compute_robust_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "percentage_bend",
) -> Tuple[float, float]:
    """
    Compute robust correlation resistant to outliers.
    
    Parameters
    ----------
    x, y : array-like
        Data arrays
    method : str
        Robust method:
        - "percentage_bend": Percentage bend correlation (default)
        - "winsorized": Winsorized correlation (20% trimming)
        - "shepherd": Shepherd's pi correlation (removes bivariate outliers)
    
    Returns
    -------
    Tuple[float, float]
        (r, p_value)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(valid))
    
    if n < 4:
        return np.nan, np.nan
    
    x_v, y_v = x[valid], y[valid]
    
    if method == "percentage_bend":
        return _percentage_bend_correlation(x_v, y_v)
    elif method == "winsorized":
        return _winsorized_correlation(x_v, y_v)
    elif method == "shepherd":
        return _shepherd_correlation(x_v, y_v)
    else:
        # Fallback to Spearman (already robust to monotonic outliers)
        return stats.spearmanr(x_v, y_v)


def _percentage_bend_correlation(x: np.ndarray, y: np.ndarray, beta: float = 0.2) -> Tuple[float, float]:
    """
    Percentage bend correlation (Wilcox, 1994).
    
    Downweights observations far from the median.
    """
    n = len(x)
    
    # Compute median and MAD
    mx, my = np.median(x), np.median(y)
    mad_x = np.median(np.abs(x - mx))
    mad_y = np.median(np.abs(y - my))
    
    if mad_x < 1e-12 or mad_y < 1e-12:
        return stats.spearmanr(x, y)
    
    # Bend parameter
    omega_x = (beta * (n - 1) + 0.5) / n
    omega_y = (beta * (n - 1) + 0.5) / n
    
    # Critical values
    crit_x = np.percentile(np.abs(x - mx) / mad_x, 100 * (1 - beta))
    crit_y = np.percentile(np.abs(y - my) / mad_y, 100 * (1 - beta))
    
    # Winsorize
    x_pb = np.clip((x - mx) / mad_x, -crit_x, crit_x)
    y_pb = np.clip((y - my) / mad_y, -crit_y, crit_y)
    
    # Compute correlation on bent data
    r, _ = stats.pearsonr(x_pb, y_pb)
    
    # Approximate p-value using t-distribution
    t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-12))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - 2))
    
    return float(r), float(p)


def _winsorized_correlation(x: np.ndarray, y: np.ndarray, trim: float = 0.2) -> Tuple[float, float]:
    """
    Winsorized correlation (replaces extreme values with percentiles).
    """
    n = len(x)
    k = int(trim * n)
    
    if k < 1:
        return stats.pearsonr(x, y)
    
    # Winsorize both arrays
    def winsorize(arr):
        sorted_arr = np.sort(arr)
        lower, upper = sorted_arr[k], sorted_arr[-(k+1)]
        return np.clip(arr, lower, upper)
    
    x_w = winsorize(x)
    y_w = winsorize(y)
    
    r, _ = stats.pearsonr(x_w, y_w)
    
    # Approximate p-value
    n_eff = n - 2 * k
    if n_eff < 3:
        return float(r), np.nan
    
    t = r * np.sqrt((n_eff - 2) / (1 - r**2 + 1e-12))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n_eff - 2))
    
    return float(r), float(p)


def _shepherd_correlation(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Shepherd's pi correlation (removes bivariate outliers via bootstrap MAD).
    """
    n = len(x)
    
    # Compute Mahalanobis-like distance using MAD
    mx, my = np.median(x), np.median(y)
    mad_x = np.median(np.abs(x - mx)) * 1.4826  # Scale to match std
    mad_y = np.median(np.abs(y - my)) * 1.4826
    
    if mad_x < 1e-12 or mad_y < 1e-12:
        return stats.spearmanr(x, y)
    
    # Standardize
    x_std = (x - mx) / mad_x
    y_std = (y - my) / mad_y
    
    # Distance from center
    dist = np.sqrt(x_std**2 + y_std**2)
    
    # Remove outliers (top alpha fraction by distance)
    threshold = np.percentile(dist, 100 * (1 - alpha))
    inliers = dist <= threshold
    
    if np.sum(inliers) < 4:
        return stats.spearmanr(x, y)
    
    r, p = stats.spearmanr(x[inliers], y[inliers])
    return float(r), float(p)


###################################################################
# LOSO Correlation Stability
###################################################################


def compute_loso_correlation_stability(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    subject_ids: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float, float, List[float]]:
    """
    Compute leave-one-subject-out correlation stability.
    
    Checks if the correlation holds when each subject is left out.
    Low std = stable finding across subjects.
    
    Parameters
    ----------
    feature_values : array-like
        Feature values (one per trial)
    target_values : array-like
        Target values (one per trial)
    subject_ids : array-like
        Subject ID for each trial
    method : str
        Correlation method
    
    Returns
    -------
    Tuple[float, float, float, List[float]]
        (r_mean, r_std, stability_index, per_subject_r)
        stability_index = 1 - (r_std / abs(r_mean)) bounded [0, 1]
    """
    feature_values = np.asarray(feature_values)
    target_values = np.asarray(target_values)
    subject_ids = np.asarray(subject_ids)
    
    unique_subjects = np.unique(subject_ids)
    
    if len(unique_subjects) < 3:
        return np.nan, np.nan, np.nan, []
    
    r_values = []
    
    for subj in unique_subjects:
        mask = subject_ids != subj
        x_loo = feature_values[mask]
        y_loo = target_values[mask]
        
        r, _ = compute_correlation(x_loo, y_loo, method)
        if np.isfinite(r):
            r_values.append(r)
    
    if len(r_values) < 2:
        return np.nan, np.nan, np.nan, r_values
    
    r_mean = float(np.mean(r_values))
    r_std = float(np.std(r_values, ddof=1))
    
    # Stability index: high = stable, low = unstable
    if abs(r_mean) > 1e-6:
        stability = max(0.0, 1.0 - (r_std / abs(r_mean)))
    else:
        stability = 0.0 if r_std > 0.1 else 1.0
    
    return r_mean, r_std, float(stability), r_values


###################################################################
# Split-Half Reliability
###################################################################


def compute_correlation_reliability(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    method: str = "split_half",
    n_iterations: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute reliability of feature-target correlation.
    
    Parameters
    ----------
    feature_values, target_values : array-like
        Data arrays
    method : str
        "split_half": Random split-half with Spearman-Brown correction
        "odd_even": Odd/even split
    n_iterations : int
        Number of random splits (for split_half method)
    seed : int
        Random seed
    
    Returns
    -------
    Tuple[float, float, float]
        (reliability, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    feature_values = np.asarray(feature_values)
    target_values = np.asarray(target_values)
    
    valid = np.isfinite(feature_values) & np.isfinite(target_values)
    n = int(np.sum(valid))
    
    if n < 10:
        return np.nan, np.nan, np.nan
    
    x = feature_values[valid]
    y = target_values[valid]
    
    if method == "odd_even":
        r1, _ = stats.spearmanr(x[::2], y[::2])
        r2, _ = stats.spearmanr(x[1::2], y[1::2])
        r_half = np.corrcoef([r1, r2])[0, 1] if np.isfinite(r1) and np.isfinite(r2) else np.nan
        reliability = _spearman_brown(r_half)
        return reliability, np.nan, np.nan
    
    # Random split-half
    reliabilities = []
    indices = np.arange(n)
    
    for _ in range(n_iterations):
        rng.shuffle(indices)
        half = n // 2
        idx1, idx2 = indices[:half], indices[half:2*half]
        
        r1, _ = stats.spearmanr(x[idx1], y[idx1])
        r2, _ = stats.spearmanr(x[idx2], y[idx2])
        
        if np.isfinite(r1) and np.isfinite(r2):
            r_half = np.corrcoef([r1, r2])[0, 1]
            if np.isfinite(r_half):
                reliabilities.append(_spearman_brown(r_half))
    
    if not reliabilities:
        return np.nan, np.nan, np.nan
    
    reliability = float(np.mean(reliabilities))
    ci_lo = float(np.percentile(reliabilities, 2.5))
    ci_hi = float(np.percentile(reliabilities, 97.5))
    
    return reliability, ci_lo, ci_hi


def _spearman_brown(r: float) -> float:
    """Spearman-Brown prophecy formula for split-half reliability."""
    if not np.isfinite(r) or abs(r) >= 1:
        return np.nan
    return (2 * r) / (1 + abs(r))

