"""
Regression Utilities
====================

Linear regression and binned statistics.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_linear_residuals(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linear regression residuals."""
    x_series = pd.to_numeric(x_data, errors="coerce")
    y_series = pd.to_numeric(y_data, errors="coerce")
    mask = x_series.notna() & y_series.notna()
    x_clean = x_series[mask].to_numpy(dtype=float)
    y_clean = y_series[mask].to_numpy(dtype=float)
    slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
    fitted = intercept + slope * x_clean
    residuals = y_clean - fitted
    return fitted, residuals, x_clean


def fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    x_range: np.ndarray,
    min_samples: int = 3,
) -> np.ndarray:
    """Fit linear regression and return predictions."""
    if len(x) < min_samples:
        return np.full_like(x_range, np.nan)
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    return polynomial(x_range)


def compute_binned_statistics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute binned means and standard errors."""
    bins = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_centers, bin_means, bin_stds = [], [], []
    
    for i in range(n_bins):
        is_last = i == n_bins - 1
        mask = (y_pred >= bins[i]) & (y_pred <= bins[i+1] if is_last else y_pred < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(y_true[mask]))
            bin_stds.append(np.std(y_true[mask]) / np.sqrt(mask.sum()))
    
    return bin_centers, bin_means, bin_stds





