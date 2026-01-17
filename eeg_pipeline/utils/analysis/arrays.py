"""
Array Utilities
===============

Centralized utilities for safe array operations, NaN handling,
and common numerical patterns used throughout the pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


MAD_TO_STD_FACTOR = 1.4826
EPSILON_SMALL = 1e-12
EPSILON_STD = 1e-10


def safe_nanmean(
    arr: np.ndarray,
    axis: Optional[int] = None,
    min_valid: int = 1,
    default: float = np.nan,
) -> Union[float, np.ndarray]:
    """Compute nanmean with minimum valid sample requirement."""
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
    """Compute nanstd with minimum valid sample requirement."""
    if arr.size == 0:
        return default
    
    valid_count = np.sum(np.isfinite(arr), axis=axis)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.nanstd(arr, axis=axis, ddof=ddof)
    
    if axis is None:
        return result if valid_count >= min_valid else default
    
    return np.where(valid_count >= min_valid, result, default)


def mask_valid(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask where values are finite."""
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


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    default: float = 0.0,
    min_denom: float = EPSILON_SMALL,
) -> np.ndarray:
    """Safe division avoiding divide-by-zero."""
    denom = np.asarray(denominator)
    numer = np.asarray(numerator)
    
    is_denom_too_small = np.abs(denom) < min_denom
    safe_denom = np.where(is_denom_too_small, 1.0, denom)
    result = numer / safe_denom
    
    return np.where(is_denom_too_small, default, result)


def robust_zscore(
    arr: np.ndarray,
    axis: Optional[int] = None,
    min_std: float = EPSILON_STD,
) -> np.ndarray:
    """Z-score normalization using median and MAD for robustness."""
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


def validate_array(
    arr: np.ndarray,
    name: str = "array",
    min_size: int = 1,
    allow_nan: bool = True,
    max_nan_fraction: float = 1.0,
) -> Tuple[bool, str]:
    """Validate array meets requirements. Returns (valid, message) tuple."""
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
    
    mean_val = float(np.nanmean(masked_data)) if valid_count > 0 else np.nan
    valid_fraction = float(valid_count / total_count) if total_count > 0 else 0.0
    
    return mean_val, valid_fraction, valid_count, total_count


def extract_finite_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mask where both arrays are finite.
    
    Returns (y_true_finite, y_pred_finite, mask) where mask indicates
    positions where both arrays have finite values.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask], mask

