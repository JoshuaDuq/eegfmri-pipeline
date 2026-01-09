"""
Array Utilities
===============

Centralized utilities for safe array operations, NaN handling,
and common numerical patterns used throughout the pipeline.

Usage:
    from eeg_pipeline.utils.analysis.arrays import (
        safe_nanmean, safe_nanstd, mask_valid, robust_zscore,
        safe_divide, clip_outliers, validate_array,
    )
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

# MAD to standard deviation conversion factor (for normal distribution)
MAD_TO_STD_FACTOR = 1.4826

# Numerical stability thresholds
EPSILON_SMALL = 1e-12
EPSILON_STD = 1e-10


# ──────────────────────────────────────────────────────────────────
# Safe Aggregation Functions
# ──────────────────────────────────────────────────────────────────


def safe_nanmean(
    arr: np.ndarray,
    axis: Optional[int] = None,
    min_valid: int = 1,
    default: float = np.nan,
) -> Union[float, np.ndarray]:
    """
    Compute nanmean with minimum valid sample requirement.
    
    Returns `default` if fewer than `min_valid` non-NaN values exist.
    """
    if arr.size == 0:
        return default
    
    valid_count = np.sum(np.isfinite(arr), axis=axis)
    result = np.nanmean(arr, axis=axis)
    
    if axis is None:
        return result if valid_count >= min_valid else default
    
    return np.where(valid_count >= min_valid, result, default)


def safe_nanstd(
    arr: np.ndarray,
    axis: Optional[int] = None,
    min_valid: int = 2,
    default: float = np.nan,
    ddof: int = 1,
) -> Union[float, np.ndarray]:
    """
    Compute nanstd with minimum valid sample requirement.
    
    Returns `default` if fewer than `min_valid` non-NaN values exist.
    """
    if arr.size == 0:
        return default
    
    valid_count = np.sum(np.isfinite(arr), axis=axis)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.nanstd(arr, axis=axis, ddof=ddof)
    
    if axis is None:
        return result if valid_count >= min_valid else default
    
    return np.where(valid_count >= min_valid, result, default)


def safe_nanmedian(
    arr: np.ndarray,
    axis: Optional[int] = None,
    min_valid: int = 1,
    default: float = np.nan,
) -> Union[float, np.ndarray]:
    """Compute nanmedian with minimum valid sample requirement."""
    if arr.size == 0:
        return default
    
    valid_count = np.sum(np.isfinite(arr), axis=axis)
    result = np.nanmedian(arr, axis=axis)
    
    if axis is None:
        return result if valid_count >= min_valid else default
    
    return np.where(valid_count >= min_valid, result, default)


# ──────────────────────────────────────────────────────────────────
# Masking and Filtering
# ──────────────────────────────────────────────────────────────────


def mask_valid(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask where values are finite (not NaN or Inf)."""
    return np.isfinite(arr)


def get_valid(arr: np.ndarray) -> np.ndarray:
    """Return only finite values from array."""
    return arr[np.isfinite(arr)]


def count_valid(arr: np.ndarray, axis: Optional[int] = None) -> Union[int, np.ndarray]:
    """Count finite values along axis."""
    return np.sum(np.isfinite(arr), axis=axis)


def valid_fraction(arr: np.ndarray) -> float:
    """Return fraction of finite values in array."""
    if arr.size == 0:
        return 0.0
    return float(np.sum(np.isfinite(arr)) / arr.size)


# ──────────────────────────────────────────────────────────────────
# Safe Arithmetic
# ──────────────────────────────────────────────────────────────────


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    default: float = 0.0,
    min_denom: float = EPSILON_SMALL,
) -> np.ndarray:
    """
    Safe division avoiding divide-by-zero.
    
    Returns `default` where denominator < min_denom.
    """
    denom = np.asarray(denominator)
    numer = np.asarray(numerator)
    
    is_denom_too_small = np.abs(denom) < min_denom
    safe_denom = np.where(is_denom_too_small, 1.0, denom)
    result = numer / safe_denom
    
    return np.where(is_denom_too_small, default, result)


def safe_log(
    arr: np.ndarray,
    base: float = 10,
    min_val: float = EPSILON_SMALL,
) -> np.ndarray:
    """Safe logarithm clamping minimum value."""
    arr_safe = np.maximum(np.abs(arr), min_val)
    if base == 10:
        return np.log10(arr_safe)
    elif base == np.e:
        return np.log(arr_safe)
    else:
        return np.log(arr_safe) / np.log(base)


def safe_sqrt(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Safe sqrt returning default for negative values."""
    return np.where(arr >= 0, np.sqrt(np.maximum(arr, 0)), default)


# ──────────────────────────────────────────────────────────────────
# Normalization and Scaling
# ──────────────────────────────────────────────────────────────────


def robust_zscore(
    arr: np.ndarray,
    axis: Optional[int] = None,
    min_std: float = EPSILON_STD,
) -> np.ndarray:
    """
    Z-score normalization with minimum std protection.
    
    Uses median and MAD for robustness against outliers.
    """
    median = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr - median), axis=axis, keepdims=True)
    
    std_approx = mad * MAD_TO_STD_FACTOR
    std_approx = np.maximum(std_approx, min_std)
    
    return (arr - median) / std_approx


def zscore(
    arr: np.ndarray,
    axis: Optional[int] = None,
    min_std: float = EPSILON_STD,
) -> np.ndarray:
    """Standard z-score with min_std protection."""
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    std = np.maximum(std, min_std)
    return (arr - mean) / std


def minmax_scale(
    arr: np.ndarray,
    axis: Optional[int] = None,
    feature_range: Tuple[float, float] = (0, 1),
) -> np.ndarray:
    """Min-max scaling to [0, 1] or custom range."""
    arr_min = np.nanmin(arr, axis=axis, keepdims=True)
    arr_max = np.nanmax(arr, axis=axis, keepdims=True)
    
    range_val = arr_max - arr_min
    is_constant = range_val < EPSILON_SMALL
    range_val = np.where(is_constant, 1.0, range_val)
    
    scaled = (arr - arr_min) / range_val
    
    range_low, range_high = feature_range
    return scaled * (range_high - range_low) + range_low


# ──────────────────────────────────────────────────────────────────
# Outlier Handling
# ──────────────────────────────────────────────────────────────────


def clip_outliers(
    arr: np.ndarray,
    n_std: float = 3.0,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Clip values beyond n_std standard deviations."""
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    
    return np.clip(arr, lower_bound, upper_bound)


def winsorize(
    arr: np.ndarray,
    limits: Tuple[float, float] = (0.05, 0.05),
    axis: Optional[int] = None,
) -> np.ndarray:
    """Winsorize array by percentile limits."""
    lower_limit, upper_limit = limits
    lower_percentile = lower_limit * 100
    upper_percentile = (1 - upper_limit) * 100
    
    lower_bound = np.nanpercentile(arr, lower_percentile, axis=axis, keepdims=True)
    upper_bound = np.nanpercentile(arr, upper_percentile, axis=axis, keepdims=True)
    
    return np.clip(arr, lower_bound, upper_bound)


def detect_outliers_zscore(
    arr: np.ndarray,
    threshold: float = 3.0,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Return boolean mask of outliers (|z| > threshold)."""
    z = zscore(arr, axis=axis)
    return np.abs(z) > threshold


def detect_outliers_iqr(
    arr: np.ndarray,
    k: float = 1.5,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Return boolean mask of outliers using IQR method."""
    q1 = np.nanpercentile(arr, 25, axis=axis, keepdims=True)
    q3 = np.nanpercentile(arr, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    is_below_lower = arr < lower_bound
    is_above_upper = arr > upper_bound
    return is_below_lower | is_above_upper


# ──────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────


def validate_array(
    arr: np.ndarray,
    name: str = "array",
    min_size: int = 1,
    allow_nan: bool = True,
    max_nan_fraction: float = 1.0,
) -> Tuple[bool, str]:
    """
    Validate array meets requirements.
    
    Returns (valid, message) tuple.
    """
    if arr is None:
        return False, f"{name} is None"
    
    if not isinstance(arr, np.ndarray):
        return False, f"{name} is not a numpy array"
    
    if arr.size < min_size:
        return False, f"{name} has size {arr.size}, need >= {min_size}"
    
    valid_frac = valid_fraction(arr)
    nan_fraction = 1 - valid_frac
    
    if not allow_nan and nan_fraction > 0:
        return False, f"{name} contains NaN values"
    
    if nan_fraction > max_nan_fraction:
        return False, f"{name} has {nan_fraction:.1%} NaN (max {max_nan_fraction:.1%})"
    
    return True, "OK"


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (add axis if 1D)."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def ensure_finite(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace NaN and Inf with fill_value."""
    return np.where(np.isfinite(arr), arr, fill_value)


def nanmean_with_fraction(data: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Compute NaN-safe mean inside a mask and report finite fractions.
    
    Returns
    -------
    mean_val : float
        Mean value (NaN if no valid samples)
    valid_fraction : float
        Fraction of valid samples (valid / total)
    valid_count : int
        Number of valid samples
    total_count : int
        Total number of samples in mask
    """
    masked_data = data[mask]
    total_count = int(masked_data.size)
    is_finite = np.isfinite(masked_data)
    valid_count = int(np.sum(is_finite))
    
    has_valid_samples = valid_count > 0
    mean_val = float(np.nanmean(masked_data)) if has_valid_samples else np.nan
    
    has_samples = total_count > 0
    valid_fraction = float(valid_count / total_count) if has_samples else 0.0
    
    return mean_val, valid_fraction, valid_count, total_count

