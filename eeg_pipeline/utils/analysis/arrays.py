"""
Array Utilities
===============

Centralized utilities for safe array operations, NaN handling,
and common numerical patterns used throughout the pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


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

