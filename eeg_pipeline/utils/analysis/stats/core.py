"""
Core Statistics Utilities
=========================

Consolidated module containing:
- Data transformation (centering, z-scoring, pooling)
- Regression utilities (linear residuals, binned statistics)
- Aperiodic fitting (1/f component extraction)
- Coupling statistics (inter-band, group power)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


###################################################################
# Data Transformation
###################################################################


def center_series(series: pd.Series) -> pd.Series:
    """Center series by subtracting mean."""
    return series - series.mean()


def zscore_series(series: pd.Series) -> pd.Series:
    """Z-score normalize series."""
    std_val = series.std(ddof=1)
    if std_val <= 0:
        return pd.Series(dtype=float)
    return (series - series.mean()) / std_val


def apply_pooling_strategy(
    x: pd.Series,
    y: pd.Series,
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series]:
    """Apply pooling strategy for correlation."""
    if pooling_strategy == "within_subject_centered":
        return center_series(x), center_series(y)
    if pooling_strategy == "within_subject_zscored":
        return zscore_series(x), zscore_series(y)
    return x, y


def prepare_data_for_plotting(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[pd.Series, pd.Series, int]:
    """Prepare data for plotting by removing NaNs."""
    mask = x_data.notna() & y_data.notna()
    return x_data[mask], y_data[mask], int(mask.sum())


def prepare_data_without_validation(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[pd.Series, pd.Series, int]:
    """Return data without validation."""
    return x_data, y_data, len(x_data)


def prepare_group_data(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    subj_order: List[str],
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    """Prepare group data for correlation analysis."""
    x_series_list, y_series_list, subject_ids = [], [], []

    for idx, (x_array, y_array) in enumerate(zip(x_lists, y_lists)):
        x_s = pd.Series(np.asarray(x_array))
        y_s = pd.Series(np.asarray(y_array))
        
        min_len = min(len(x_s), len(y_s))
        x_s, y_s = x_s.iloc[:min_len], y_s.iloc[:min_len]
        
        valid = x_s.notna() & y_s.notna()
        x_s, y_s = x_s[valid], y_s[valid]
        
        if x_s.empty:
            continue
        
        x_norm, y_norm = apply_pooling_strategy(x_s, y_s, pooling_strategy)
        if x_norm.empty:
            continue
        
        subject_id = subj_order[idx] if idx < len(subj_order) else str(idx)
        subject_ids.extend([subject_id] * len(x_norm))
        x_series_list.append(x_norm.reset_index(drop=True))
        y_series_list.append(y_norm.reset_index(drop=True))

    if not x_series_list:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    return pd.concat(x_series_list, ignore_index=True), pd.concat(y_series_list, ignore_index=True), subject_ids


###################################################################
# Regression Utilities
###################################################################


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


###################################################################
# Aperiodic Fitting
###################################################################


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


###################################################################
# Coupling Statistics
###################################################################


def compute_consensus_labels(
    labels_all_trials: List[np.ndarray],
    n_timepoints: int,
) -> np.ndarray:
    """Compute consensus microstate labels across trials."""
    labels = np.zeros(n_timepoints, dtype=int)
    for t in range(n_timepoints):
        counts = np.bincount([labels_all_trials[trial][t] for trial in range(len(labels_all_trials))])
        labels[t] = np.argmax(counts)
    return labels


def compute_inter_band_coupling_matrix(
    tfr_avg,
    band_names: List[str],
    features_freq_bands: Dict[str, Tuple[float, float]],
    extract_band_channel_means_func,
) -> np.ndarray:
    """Compute inter-band coupling matrix."""
    from .band_stats import compute_band_spatial_correlation
    
    n = len(band_names)
    mat = np.zeros((n, n))
    
    for i, b1 in enumerate(band_names):
        fmin1, fmax1 = features_freq_bands[b1]
        fm1 = (tfr_avg.freqs >= fmin1) & (tfr_avg.freqs <= fmax1)
        if not fm1.any():
            continue
        mat[i, i] = 1.0
        b1_ch = extract_band_channel_means_func(tfr_avg, fm1)
        
        for j in range(i + 1, n):
            b2 = band_names[j]
            fmin2, fmax2 = features_freq_bands[b2]
            fm2 = (tfr_avg.freqs >= fmin2) & (tfr_avg.freqs <= fmax2)
            if not fm2.any():
                continue
            b2_ch = extract_band_channel_means_func(tfr_avg, fm2)
            r = compute_band_spatial_correlation(b1_ch, b2_ch)
            mat[i, j] = mat[j, i] = r
    
    return mat


def compute_group_channel_power_statistics(
    subj_pow: Dict[str, pd.DataFrame],
    bands: List[str],
    all_channels: List[str],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Compute group channel power statistics."""
    heatmap_rows, stats_rows = [], []
    
    for band in bands:
        band_str = str(band)
        subj_means = []
        for _, df in subj_pow.items():
            vals = []
            for ch in all_channels:
                col = f"pow_{band_str}_{ch}"
                if col in df.columns:
                    vals.append(float(pd.to_numeric(df[col], errors="coerce").mean()))
                else:
                    vals.append(np.nan)
            subj_means.append(vals)
        
        arr = np.asarray(subj_means, dtype=float)
        mean_across = np.nanmean(arr, axis=0)
        heatmap_rows.append(mean_across)
        
        n_eff = np.sum(np.isfinite(arr), axis=0)
        std_across = np.nanstd(arr, axis=0, ddof=1)
        
        for j, ch in enumerate(all_channels):
            stats_rows.append({
                "band": band_str, "channel": ch,
                "mean": float(mean_across[j]) if np.isfinite(mean_across[j]) else np.nan,
                "std": float(std_across[j]) if np.isfinite(std_across[j]) else np.nan,
                "n_subjects": int(n_eff[j]),
            })
    
    return heatmap_rows, stats_rows


__all__ = [
    # Transform
    "center_series",
    "zscore_series",
    "apply_pooling_strategy",
    "prepare_data_for_plotting",
    "prepare_data_without_validation",
    "prepare_group_data",
    # Regression
    "compute_linear_residuals",
    "fit_linear_regression",
    "compute_binned_statistics",
    # Aperiodic
    "fit_aperiodic",
    "fit_aperiodic_to_all_epochs",
    # Coupling
    "compute_consensus_labels",
    "compute_inter_band_coupling_matrix",
    "compute_group_channel_power_statistics",
]
