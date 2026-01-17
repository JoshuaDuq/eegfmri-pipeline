"""
Band Statistics
===============

Inter-band correlations and power statistics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values

from .correlation import compute_correlation, fisher_z_transform_mean


_EPSILON_STD = 1e-12


def compute_band_spatial_correlation(
    band1_channels: np.ndarray,
    band2_channels: np.ndarray,
) -> float:
    """Compute spatial correlation between bands."""
    if len(band1_channels) <= 1 or len(band2_channels) <= 1:
        return np.nan
    
    correlation_matrix = np.corrcoef(band1_channels, band2_channels)
    return float(correlation_matrix[0, 1])


def compute_band_pair_correlation(
    band_vector_i: Optional[Dict[str, float]],
    band_vector_j: Optional[Dict[str, float]],
) -> float:
    """Compute correlation between band topographies."""
    if band_vector_i is None or band_vector_j is None:
        return np.nan
    
    common_channels = sorted(set(band_vector_i.keys()) & set(band_vector_j.keys()))
    if len(common_channels) < 2:
        return np.nan
    
    values_i = np.array([band_vector_i[ch] for ch in common_channels])
    values_j = np.array([band_vector_j[ch] for ch in common_channels])
    
    if np.std(values_i) < _EPSILON_STD or np.std(values_j) < _EPSILON_STD:
        return np.nan
    
    correlation_matrix = np.corrcoef(values_i, values_j)
    return float(correlation_matrix[0, 1])


def compute_subject_band_correlation_matrix(
    band_vectors: Dict[str, Dict[str, float]],
    band_names: List[str],
) -> np.ndarray:
    """Compute inter-band correlation matrix for one subject."""
    n_bands = len(band_names)
    correlation_matrix = np.eye(n_bands)
    
    for i, band_i in enumerate(band_names):
        for j in range(i + 1, n_bands):
            band_j = band_names[j]
            vector_i = band_vectors.get(band_i)
            vector_j = band_vectors.get(band_j)
            correlation = compute_band_pair_correlation(vector_i, vector_j)
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
    
    return correlation_matrix


def compute_group_band_correlation_matrix(
    per_subject_correlations: List[np.ndarray],
    n_bands: int,
) -> np.ndarray:
    """Aggregate inter-band correlations across subjects."""
    group_matrix = np.eye(n_bands)
    stacked_correlations = np.stack(per_subject_correlations, axis=0)
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            correlation_values = stacked_correlations[:, i, j]
            correlation_values = correlation_values[np.isfinite(correlation_values)]
            
            if correlation_values.size == 0:
                group_matrix[i, j] = np.nan
                group_matrix[j, i] = np.nan
            else:
                mean_correlation = fisher_z_transform_mean(correlation_values)
                group_matrix[i, j] = mean_correlation
                group_matrix[j, i] = mean_correlation
    
    return group_matrix


def compute_band_statistics_array(
    values: np.ndarray,
    ci_multiplier: float = 1.96,
) -> Tuple[float, float, float, int]:
    """Compute band statistics from array."""
    clean_values = values[np.isfinite(values)]
    if clean_values.size == 0:
        return np.nan, np.nan, np.nan, 0
    
    mean_value = float(np.mean(clean_values))
    n_samples = len(clean_values)
    
    if n_samples > 1:
        standard_error = float(np.std(clean_values, ddof=1) / np.sqrt(n_samples))
    else:
        standard_error = np.nan
    
    if np.isfinite(standard_error):
        margin = ci_multiplier * standard_error
        ci_low = mean_value - margin
        ci_high = mean_value + margin
    else:
        ci_low = np.nan
        ci_high = np.nan
    
    return mean_value, ci_low, ci_high, n_samples


def compute_inter_band_correlation_statistics(
    per_subject_correlations: List[np.ndarray],
    band_names: List[str],
    ci_multiplier: float = 1.96,
    config: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Compute inter-band correlation statistics across subjects."""
    clip_min, clip_max = get_fisher_z_clip_values(config)
    results = []
    n_bands = len(band_names)
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            correlation_values = np.array([matrix[i, j] for matrix in per_subject_correlations])
            correlation_values = correlation_values[np.isfinite(correlation_values)]
            
            if correlation_values.size == 0:
                continue
            
            r_clipped = np.clip(correlation_values, clip_min, clip_max)
            z_scores = np.arctanh(r_clipped)
            z_mean = float(np.mean(z_scores))
            n_subjects = len(z_scores)
            
            if n_subjects > 1:
                standard_error = float(np.std(z_scores, ddof=1) / np.sqrt(n_subjects))
            else:
                standard_error = np.nan
            
            r_group = float(np.tanh(z_mean))
            
            if np.isfinite(standard_error):
                z_low = z_mean - ci_multiplier * standard_error
                z_high = z_mean + ci_multiplier * standard_error
                r_ci_low = float(np.tanh(z_low))
                r_ci_high = float(np.tanh(z_high))
            else:
                r_ci_low = np.nan
                r_ci_high = np.nan
            
            results.append({
                "band_i": band_names[i],
                "band_j": band_names[j],
                "r_group": r_group,
                "r_ci_low": r_ci_low,
                "r_ci_high": r_ci_high,
                "n_subjects": n_subjects,
            })
    
    return results


def compute_band_correlations(
    pow_df: pd.DataFrame,
    y: pd.Series,
    band: str,
    min_samples: int = 3,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute channel-wise correlations for a band."""
    band_lower = str(band).lower()
    prefix = "power_"
    token = f"_{band_lower}_ch_"
    
    candidate_columns = [
        col
        for col in pow_df.columns
        if str(col).lower().startswith(prefix) and token in str(col).lower()
    ]
    
    if not candidate_columns:
        return [], np.array([]), np.array([])
    
    pattern = re.compile(
        rf"^power_[^_]+_{re.escape(band_lower)}_ch_(.+?)_",
        re.IGNORECASE
    )
    band_columns: List[str] = []
    channel_names: List[str] = []
    
    for col in candidate_columns:
        match = pattern.match(str(col))
        if match is None:
            continue
        band_columns.append(col)
        channel_names.append(match.group(1))
    
    if not band_columns:
        return [], np.array([]), np.array([])
    
    correlations = []
    p_values = []
    
    for col in band_columns:
        x_values = pow_df[col].to_numpy()
        y_values = y.to_numpy()
        
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_valid = x_values[valid_mask]
        y_valid = y_values[valid_mask]
        
        if len(x_valid) < min_samples:
            correlation = np.nan
            p_value = 1.0
        else:
            correlation, _ = compute_correlation(x_valid, y_valid, method="spearman")
            if np.isfinite(correlation):
                _, p_value = stats.spearmanr(x_valid, y_valid)
            else:
                p_value = 1.0
        
        correlations.append(correlation)
        p_values.append(p_value)
    
    return channel_names, np.array(correlations), np.array(p_values)


def compute_connectivity_correlations(
    conn_df: pd.DataFrame,
    y: pd.Series,
    measure_cols: List[str],
    measure: str,
    band: str,
    min_samples: int = 3,
    min_correlation: float = 0.3,
    max_pvalue: float = 0.05,
) -> Tuple[List[float], List[str]]:
    """Compute significant connectivity correlations."""
    correlations = []
    connections = []
    prefix = f"{measure}_{band}_"
    
    for col in measure_cols:
        x_values = conn_df[col].to_numpy()
        y_values = y.to_numpy()
        
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_valid = x_values[valid_mask]
        y_valid = y_values[valid_mask]
        
        if len(x_valid) < min_samples:
            continue
        
        if np.std(x_valid) < _EPSILON_STD or np.std(y_valid) < _EPSILON_STD:
            continue
        
        correlation, _ = compute_correlation(x_valid, y_valid, method="spearman")
        
        if np.isfinite(correlation):
            _, p_value = stats.spearmanr(x_valid, y_valid)
        else:
            p_value = 1.0
        
        is_significant = abs(correlation) > min_correlation and p_value < max_pvalue
        if is_significant:
            correlations.append(correlation)
            connection_name = col.replace(prefix, "").replace("conn_", "")
            connections.append(connection_name)
    
    return correlations, connections


def compute_inter_band_coupling_matrix(
    tfr_avg,
    band_names: List[str],
    features_freq_bands: Dict[str, Tuple[float, float]],
    extract_band_channel_means_func,
) -> np.ndarray:
    """Compute inter-band coupling matrix."""
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
        subject_means = []
        
        for dataframe in subj_pow.values():
            channel_values = _extract_channel_power_values(
                dataframe, band, all_channels
            )
            subject_means.append(channel_values)
        
        subject_arrays = np.asarray(subject_means, dtype=float)
        mean_per_channel, band_statistics = _compute_band_statistics(
            subject_arrays, band, all_channels
        )
        
        heatmap_rows.append(mean_per_channel)
        stats_rows.extend(band_statistics)
    
    return heatmap_rows, stats_rows

