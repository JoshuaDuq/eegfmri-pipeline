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

from typing import Any, Dict, List, Tuple

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
    mean = series.mean()
    std_val = series.std(ddof=1)
    if std_val <= 0:
        return pd.Series(dtype=float)
    return (series - mean) / std_val


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


def _align_and_validate_series(
    x_array: np.ndarray,
    y_array: np.ndarray,
) -> Tuple[pd.Series, pd.Series]:
    """Align series to same length and remove NaNs."""
    x_series = pd.Series(np.asarray(x_array))
    y_series = pd.Series(np.asarray(y_array))
    
    min_length = min(len(x_series), len(y_series))
    x_series = x_series.iloc[:min_length]
    y_series = y_series.iloc[:min_length]
    
    valid_mask = x_series.notna() & y_series.notna()
    return x_series[valid_mask], y_series[valid_mask]


def _process_subject_data(
    x_array: np.ndarray,
    y_array: np.ndarray,
    subject_id: str,
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    """Process single subject's data and return normalized series with IDs."""
    x_series, y_series = _align_and_validate_series(x_array, y_array)
    
    if x_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    x_normalized, y_normalized = apply_pooling_strategy(
        x_series, y_series, pooling_strategy
    )
    
    if x_normalized.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    n_samples = len(x_normalized)
    subject_ids = [subject_id] * n_samples
    x_reset = x_normalized.reset_index(drop=True)
    y_reset = y_normalized.reset_index(drop=True)
    
    return x_reset, y_reset, subject_ids


def prepare_group_data(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    subj_order: List[str],
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    """Prepare group data for correlation analysis."""
    x_series_list, y_series_list, subject_ids = [], [], []

    for idx, (x_array, y_array) in enumerate(zip(x_lists, y_lists)):
        subject_id = (
            subj_order[idx] if idx < len(subj_order) else str(idx)
        )
        x_normalized, y_normalized, ids = _process_subject_data(
            x_array, y_array, subject_id, pooling_strategy
        )
        
        if x_normalized.empty:
            continue
        
        x_series_list.append(x_normalized)
        y_series_list.append(y_normalized)
        subject_ids.extend(ids)

    if not x_series_list:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    x_concatenated = pd.concat(x_series_list, ignore_index=True)
    y_concatenated = pd.concat(y_series_list, ignore_index=True)
    
    return x_concatenated, y_concatenated, subject_ids


###################################################################
# Regression Utilities
###################################################################


def compute_linear_residuals(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linear regression residuals."""
    x_numeric = pd.to_numeric(x_data, errors="coerce")
    y_numeric = pd.to_numeric(y_data, errors="coerce")
    valid_mask = x_numeric.notna() & y_numeric.notna()
    
    x_valid = x_numeric[valid_mask].to_numpy(dtype=float)
    y_valid = y_numeric[valid_mask].to_numpy(dtype=float)
    
    regression_result = stats.linregress(x_valid, y_valid)
    slope = regression_result.slope
    intercept = regression_result.intercept
    
    fitted_values = intercept + slope * x_valid
    residuals = y_valid - fitted_values
    
    return fitted_values, residuals, x_valid


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


def _create_bin_mask(
    y_pred: np.ndarray,
    bin_edges: np.ndarray,
    bin_index: int,
    is_last_bin: bool,
) -> np.ndarray:
    """Create boolean mask for values in specified bin."""
    lower_bound = bin_edges[bin_index]
    upper_bound = bin_edges[bin_index + 1]
    
    if is_last_bin:
        return (y_pred >= lower_bound) & (y_pred <= upper_bound)
    return (y_pred >= lower_bound) & (y_pred < upper_bound)


def compute_binned_statistics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute binned means and standard errors."""
    y_min = y_pred.min()
    y_max = y_pred.max()
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    
    bin_centers, bin_means, bin_stds = [], [], []
    
    for bin_idx in range(n_bins):
        is_last_bin = bin_idx == n_bins - 1
        bin_mask = _create_bin_mask(y_pred, bin_edges, bin_idx, is_last_bin)
        n_samples_in_bin = bin_mask.sum()
        
        if n_samples_in_bin > 0:
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            y_true_in_bin = y_true[bin_mask]
            bin_mean = np.mean(y_true_in_bin)
            bin_std = np.std(y_true_in_bin) / np.sqrt(n_samples_in_bin)
            
            bin_centers.append(bin_center)
            bin_means.append(bin_mean)
            bin_stds.append(bin_std)
    
    return bin_centers, bin_means, bin_stds


###################################################################
# Aperiodic Fitting
###################################################################


def _reject_peaks(
    frequencies: np.ndarray,
    psd_values: np.ndarray,
    peak_rejection_z: float,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reject spectral peaks using robust outlier detection."""
    mad = stats.median_abs_deviation(
        psd_values, scale="normal", nan_policy="omit"
    )
    median_psd = np.median(psd_values) if np.isfinite(psd_values).any() else np.nan
    
    mad_threshold = 1e-12
    is_mad_valid = np.isfinite(mad) and mad > mad_threshold
    is_median_valid = np.isfinite(median_psd)
    
    if not (is_mad_valid and is_median_valid):
        return frequencies, psd_values
    
    rejection_threshold = median_psd + peak_rejection_z * mad
    keep_mask = psd_values <= rejection_threshold
    
    if keep_mask.sum() >= min_points:
        return frequencies[keep_mask], psd_values[keep_mask]
    
    return frequencies, psd_values


def fit_aperiodic(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    peak_rejection_z: float = 3.5,
    min_points: int = 5,
) -> Tuple[float, float]:
    """Fit aperiodic (1/f) component to log-log PSD."""
    finite_mask = np.isfinite(log_freqs) & np.isfinite(log_psd)
    frequencies = log_freqs[finite_mask]
    psd_values = log_psd[finite_mask]
    
    if frequencies.size < min_points:
        return np.nan, np.nan
    
    frequencies, psd_values = _reject_peaks(
        frequencies, psd_values, peak_rejection_z, min_points
    )
    
    if frequencies.size < min_points:
        return np.nan, np.nan
    
    try:
        slope, intercept = np.polyfit(frequencies, psd_values, 1)
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
    consensus_labels = np.zeros(n_timepoints, dtype=int)
    n_trials = len(labels_all_trials)
    
    for timepoint_idx in range(n_timepoints):
        trial_labels_at_timepoint = [
            labels_all_trials[trial_idx][timepoint_idx]
            for trial_idx in range(n_trials)
        ]
        label_counts = np.bincount(trial_labels_at_timepoint)
        consensus_labels[timepoint_idx] = np.argmax(label_counts)
    
    return consensus_labels


def compute_inter_band_coupling_matrix(
    tfr_avg,
    band_names: List[str],
    features_freq_bands: Dict[str, Tuple[float, float]],
    extract_band_channel_means_func,
) -> np.ndarray:
    """Compute inter-band coupling matrix."""
    from .band_stats import compute_band_spatial_correlation
    
    n_bands = len(band_names)
    coupling_matrix = np.zeros((n_bands, n_bands))
    
    for band_1_idx, band_1_name in enumerate(band_names):
        fmin_1, fmax_1 = features_freq_bands[band_1_name]
        freq_mask_1 = (tfr_avg.freqs >= fmin_1) & (tfr_avg.freqs <= fmax_1)
        
        if not freq_mask_1.any():
            continue
        
        coupling_matrix[band_1_idx, band_1_idx] = 1.0
        band_1_channels = extract_band_channel_means_func(tfr_avg, freq_mask_1)
        
        for band_2_idx in range(band_1_idx + 1, n_bands):
            band_2_name = band_names[band_2_idx]
            fmin_2, fmax_2 = features_freq_bands[band_2_name]
            freq_mask_2 = (tfr_avg.freqs >= fmin_2) & (tfr_avg.freqs <= fmax_2)
            
            if not freq_mask_2.any():
                continue
            
            band_2_channels = extract_band_channel_means_func(
                tfr_avg, freq_mask_2
            )
            correlation = compute_band_spatial_correlation(
                band_1_channels, band_2_channels
            )
            coupling_matrix[band_1_idx, band_2_idx] = correlation
            coupling_matrix[band_2_idx, band_1_idx] = correlation
    
    return coupling_matrix


def _extract_channel_power_values(
    dataframe: pd.DataFrame,
    band_name: str,
    channels: List[str],
) -> List[float]:
    """Extract power values for all channels from dataframe."""
    power_values = []
    for channel in channels:
        column_name = f"pow_{band_name}_{channel}"
        if column_name in dataframe.columns:
            numeric_values = pd.to_numeric(
                dataframe[column_name], errors="coerce"
            )
            mean_value = float(numeric_values.mean())
            power_values.append(mean_value)
        else:
            power_values.append(np.nan)
    return power_values


def _compute_band_statistics(
    subject_power_arrays: np.ndarray,
    band_name: str,
    channels: List[str],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Compute mean, std, and n_subjects for each channel in a band."""
    mean_per_channel = np.nanmean(subject_power_arrays, axis=0)
    n_subjects_per_channel = np.sum(
        np.isfinite(subject_power_arrays), axis=0
    )
    std_per_channel = np.nanstd(subject_power_arrays, axis=0, ddof=1)
    
    statistics = []
    for channel_idx, channel_name in enumerate(channels):
        mean_value = mean_per_channel[channel_idx]
        std_value = std_per_channel[channel_idx]
        n_subjects = n_subjects_per_channel[channel_idx]
        
        statistics.append({
            "band": band_name,
            "channel": channel_name,
            "mean": float(mean_value) if np.isfinite(mean_value) else np.nan,
            "std": float(std_value) if np.isfinite(std_value) else np.nan,
            "n_subjects": int(n_subjects),
        })
    
    return mean_per_channel, statistics


def compute_group_channel_power_statistics(
    subj_pow: Dict[str, pd.DataFrame],
    bands: List[str],
    all_channels: List[str],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Compute group channel power statistics."""
    heatmap_rows, stats_rows = [], []
    
    for band in bands:
        band_name = str(band)
        subject_means = []
        
        for dataframe in subj_pow.values():
            channel_values = _extract_channel_power_values(
                dataframe, band_name, all_channels
            )
            subject_means.append(channel_values)
        
        subject_arrays = np.asarray(subject_means, dtype=float)
        mean_per_channel, band_statistics = _compute_band_statistics(
            subject_arrays, band_name, all_channels
        )
        
        heatmap_rows.append(mean_per_channel)
        stats_rows.extend(band_statistics)
    
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
