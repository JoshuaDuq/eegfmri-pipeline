"""
Reliability and Validity Statistics
====================================

Functions for assessing measurement reliability and predictive validity:
- ICC (Intraclass Correlation Coefficient)
- Split-half reliability
- Cross-validated predictive modeling
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_subject_seed


###################################################################
# Constants
###################################################################

DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_FOR_CORRELATION = 3
DEFAULT_N_SPLITS = 100
MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF = 20


###################################################################
# Intraclass Correlation Coefficient (ICC)
###################################################################


def _compute_icc_one_way_single(ms_rows: float, ms_within: float, k: int) -> float:
    """Compute ICC(1,1): One-way random, single rater."""
    return (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)


def _compute_icc_two_way_random_single(ms_rows: float, ms_error: float, ms_cols: float, n: int, k: int) -> float:
    """Compute ICC(2,1): Two-way random, single rater."""
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return (ms_rows - ms_error) / denominator


def _compute_icc_two_way_mixed_single(ms_rows: float, ms_error: float, k: int) -> float:
    """Compute ICC(3,1): Two-way mixed, single rater."""
    return (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)


def _compute_icc_one_way_average(ms_rows: float, ms_within: float) -> float:
    """Compute ICC(1,k): One-way random, average of k raters."""
    return (ms_rows - ms_within) / ms_rows


def _compute_icc_two_way_random_average(ms_rows: float, ms_error: float, ms_cols: float, n: int) -> float:
    """Compute ICC(2,k): Two-way random, average of k raters."""
    return (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)


def _compute_icc_two_way_mixed_average(ms_rows: float, ms_error: float) -> float:
    """Compute ICC(3,k): Two-way mixed, average of k raters."""
    return (ms_rows - ms_error) / ms_rows


def _compute_icc_confidence_intervals(
    ms_rows: float,
    ms_error: float,
    n: int,
    k: int,
) -> Tuple[float, float]:
    """Compute 95% confidence intervals for ICC using F-distribution."""
    if ms_error <= 0:
        return np.nan, np.nan
    
    f_value = ms_rows / ms_error
    if not (np.isfinite(f_value) and f_value > 0):
        return np.nan, np.nan
    
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    f_critical_upper = stats.f.ppf(0.975, df1, df2)
    f_critical_lower = stats.f.ppf(0.975, df2, df1)
    
    f_low = f_value / f_critical_upper
    f_high = f_value * f_critical_lower
    
    ci_low = (f_low - 1) / (f_low + k - 1)
    ci_high = (f_high - 1) / (f_high + k - 1)
    
    return float(ci_low), float(ci_high)


def compute_icc(
    data: np.ndarray,
    icc_type: str = "ICC(2,1)",
) -> Tuple[float, float, float]:
    """Compute Intraclass Correlation Coefficient.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_subjects, n_raters/n_sessions).
        Each row is a subject, each column is a rater/session.
    icc_type : str
        Type of ICC to compute:
        - "ICC(1,1)": One-way random, single rater
        - "ICC(2,1)": Two-way random, single rater (default)
        - "ICC(3,1)": Two-way mixed, single rater
        - "ICC(1,k)": One-way random, average of k raters
        - "ICC(2,k)": Two-way random, average of k raters
        - "ICC(3,k)": Two-way mixed, average of k raters
    
    Returns
    -------
    icc : float
        ICC value
    ci_low : float
        Lower 95% CI bound
    ci_high : float
        Upper 95% CI bound
    """
    data = np.asarray(data)
    if data.ndim != 2:
        return np.nan, np.nan, np.nan
    
    n_subjects, n_raters = data.shape
    if n_subjects < 2 or n_raters < 2:
        return np.nan, np.nan, np.nan
    
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = n_raters * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n_subjects * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    
    ms_rows = ss_rows / (n_subjects - 1)
    ms_cols = ss_cols / (n_raters - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
    
    icc_type_upper = icc_type.upper()
    if icc_type_upper in ["ICC(1,1)", "ICC1"]:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_single(ms_rows, ms_within, n_raters)
    elif icc_type_upper in ["ICC(2,1)", "ICC2"]:
        icc = _compute_icc_two_way_random_single(ms_rows, ms_error, ms_cols, n_subjects, n_raters)
    elif icc_type_upper in ["ICC(3,1)", "ICC3"]:
        icc = _compute_icc_two_way_mixed_single(ms_rows, ms_error, n_raters)
    elif icc_type_upper in ["ICC(1,K)", "ICC1K"]:
        ms_within = (ss_cols + ss_error) / (n_subjects * (n_raters - 1))
        icc = _compute_icc_one_way_average(ms_rows, ms_within)
    elif icc_type_upper in ["ICC(2,K)", "ICC2K"]:
        icc = _compute_icc_two_way_random_average(ms_rows, ms_error, ms_cols, n_subjects)
    elif icc_type_upper in ["ICC(3,K)", "ICC3K"]:
        icc = _compute_icc_two_way_mixed_average(ms_rows, ms_error)
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")
    
    ci_low, ci_high = _compute_icc_confidence_intervals(ms_rows, ms_error, n_subjects, n_raters)
    icc_clipped = float(np.clip(icc, -1, 1))
    
    return icc_clipped, ci_low, ci_high


def _apply_spearman_brown(r: float) -> float:
    """Apply Spearman-Brown prophecy formula."""
    if r <= -1 or not np.isfinite(r):
        return np.nan
    return (2 * r) / (1 + r)


###################################################################
# Feature-Extractor-Based Split-Half Reliability
###################################################################


def compute_correlation_split_half_reliability(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_splits: int = DEFAULT_N_SPLITS,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute split-half reliability for a single correlation with Spearman-Brown correction.
    
    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable (e.g., ratings)
    method : str
        Correlation method ('spearman' or 'pearson')
    n_splits : int
        Number of random splits
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    float
        Spearman-Brown corrected reliability
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid_mask.sum())
    
    if n_valid < MIN_SAMPLES_FOR_CORRELATION_SPLIT_HALF:
        return np.nan
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    indices = np.arange(n_valid)
    
    from .correlation import compute_correlation
    
    correlations = []
    half_size = n_valid // 2
    
    for _ in range(n_splits):
        rng.shuffle(indices)
        idx1 = indices[:half_size]
        idx2 = indices[half_size:2 * half_size]
        
        r1, _ = compute_correlation(x_valid[idx1], y_valid[idx1], method)
        r2, _ = compute_correlation(x_valid[idx2], y_valid[idx2], method)
        r1 = r1 if np.isfinite(r1) else np.nan
        r2 = r2 if np.isfinite(r2) else np.nan
        
        if np.isfinite(r1) and np.isfinite(r2):
            mean_correlation = (r1 + r2) / 2
            correlations.append(mean_correlation)
    
    if not correlations:
        return np.nan
    
    mean_half_correlation = np.mean(correlations)
    return float(_apply_spearman_brown(mean_half_correlation))


###################################################################
# Exports
###################################################################

__all__ = [
    "compute_icc",
    "compute_correlation_split_half_reliability",
    "get_subject_seed",
]
