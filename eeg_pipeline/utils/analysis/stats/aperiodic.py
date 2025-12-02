"""
Aperiodic Fitting
=================

Functions for fitting 1/f aperiodic components.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def fit_aperiodic(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    peak_rejection_z: float = 3.5,
    min_points: int = 5,
) -> Tuple[float, float]:
    """Fit aperiodic (1/f) component to log-log PSD."""
    finite_mask = np.isfinite(log_freqs) & np.isfinite(log_psd)
    freq = log_freqs[finite_mask]
    psd_vals = log_psd[finite_mask]
    
    if freq.size < min_points:
        return np.nan, np.nan
    
    mad = stats.median_abs_deviation(psd_vals, scale="normal", nan_policy="omit")
    median = np.median(psd_vals) if np.isfinite(psd_vals).any() else np.nan
    
    if np.isfinite(mad) and mad > 1e-12 and np.isfinite(median):
        keep_mask = psd_vals <= median + peak_rejection_z * mad
        if keep_mask.sum() >= min_points:
            freq = freq[keep_mask]
            psd_vals = psd_vals[keep_mask]
    
    try:
        slope, intercept = np.polyfit(freq, psd_vals, 1)
        return float(intercept), float(slope)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan


def fit_aperiodic_to_all_epochs(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    peak_rejection_z: float = 3.5,
    min_points: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit aperiodic to all epochs and channels."""
    n_epochs, n_channels, _ = log_psd.shape
    offsets = np.full((n_epochs, n_channels), np.nan)
    slopes = np.full((n_epochs, n_channels), np.nan)
    
    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            intercept, slope = fit_aperiodic(
                log_freqs, log_psd[epoch_idx, channel_idx, :],
                peak_rejection_z=peak_rejection_z, min_points=min_points,
            )
            offsets[epoch_idx, channel_idx] = intercept
            slopes[epoch_idx, channel_idx] = slope
    
    return offsets, slopes





