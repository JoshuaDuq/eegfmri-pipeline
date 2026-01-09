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

from .base import get_statistics_constants
from .bootstrap import bootstrap_corr_ci
from .correlation import compute_correlation


def compute_band_spatial_correlation(
    band1_channels: np.ndarray,
    band2_channels: np.ndarray,
) -> float:
    """Compute spatial correlation between bands."""
    if len(band1_channels) <= 1 or len(band2_channels) <= 1:
        return np.nan
    
    correlation_matrix = np.corrcoef(band1_channels, band2_channels)
    correlation_coefficient = correlation_matrix[0, 1]
    return float(correlation_coefficient)


def compute_band_pair_correlation(
    vec_i: Optional[Dict[str, float]],
    vec_j: Optional[Dict[str, float]],
) -> float:
    """Compute correlation between band topographies."""
    if vec_i is None or vec_j is None:
        return np.nan
    
    common_channels = sorted(set(vec_i.keys()) & set(vec_j.keys()))
    if len(common_channels) < 2:
        return np.nan
    
    values_i = np.array([vec_i[ch] for ch in common_channels])
    values_j = np.array([vec_j[ch] for ch in common_channels])
    
    epsilon_std = 1e-12
    if np.std(values_i) < epsilon_std or np.std(values_j) < epsilon_std:
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


def fisher_z_transform_mean(r_values: np.ndarray, config: Optional[Any] = None) -> float:
    """Compute Fisher z-transformed mean of correlations."""
    clip_min, clip_max = get_fisher_z_clip_values(config)
    r_clipped = np.clip(r_values, clip_min, clip_max)
    z_scores = np.arctanh(r_clipped)
    z_mean = np.mean(z_scores)
    return float(np.tanh(z_mean))


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
    config: Optional[Any] = None,
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


def compute_correlation_ci_fisher(
    z_mean: float,
    se: float,
    ci_multiplier: float = 1.96,
) -> Tuple[float, float]:
    """Compute CI from Fisher z mean and SE."""
    if not np.isfinite(se):
        return np.nan, np.nan
    
    z_low = z_mean - ci_multiplier * se
    z_high = z_mean + ci_multiplier * se
    ci_low = float(np.tanh(z_low))
    ci_high = float(np.tanh(z_high))
    
    return ci_low, ci_high


def compute_band_correlations(
    pow_df: pd.DataFrame,
    y: pd.Series,
    band: str,
    power_prefix: str = "pow_",
    min_samples: int = 3,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute channel-wise correlations for a band."""
    band_lower = str(band).lower()
    
    # Canonical NamingSchema columns:
    # power_{segment}_{band}_ch_{channel}_{stat}
    prefix = "power_"
    token = f"_{band_lower}_ch_"
    candidate_columns = [
        col
        for col in pow_df.columns
        if str(col).lower().startswith(prefix) and token in str(col).lower()
    ]
    
    if not candidate_columns:
        return [], np.array([]), np.array([])
    
    # Example: power_active_alpha_ch_Fz_logratio
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
        valid_data = pd.concat([pow_df[col], y], axis=1).dropna()
        
        if len(valid_data) >= min_samples:
            correlation, p_value = stats.spearmanr(
                valid_data.iloc[:, 0],
                valid_data.iloc[:, 1]
            )
        else:
            correlation = np.nan
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
    epsilon_std = 1e-12
    
    for col in measure_cols:
        valid_mask = ~(conn_df[col].isna() | y.isna())
        if valid_mask.sum() < min_samples:
            continue
        
        x_values = conn_df[col][valid_mask].to_numpy()
        y_values = y[valid_mask].to_numpy()
        
        if np.std(x_values) < epsilon_std or np.std(y_values) < epsilon_std:
            continue
        
        correlation, p_value = stats.spearmanr(x_values, y_values)
        
        is_significant = abs(correlation) > min_correlation and p_value < max_pvalue
        if is_significant:
            correlations.append(correlation)
            connection_name = col.replace(prefix, "").replace("conn_", "")
            connections.append(connection_name)
    
    return correlations, connections


def compute_correlation_stats(
    x: pd.Series,
    y: pd.Series,
    method_code: str,
    bootstrap_ci: int,
    rng: Optional[np.random.Generator],
    min_samples: int = 3,
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute correlation with optional bootstrap CI."""
    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_effective = int(valid_mask.sum())
    
    if n_effective < min_samples:
        return np.nan, np.nan, n_effective, (np.nan, np.nan)
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    correlation, p_value = compute_correlation(x_valid, y_valid, method_code)
    
    if bootstrap_ci > 0:
        confidence_interval = bootstrap_corr_ci(
            x_valid,
            y_valid,
            method_code,
            n_boot=bootstrap_ci,
            rng=rng
        )
    else:
        confidence_interval = (np.nan, np.nan)
    
    return float(correlation), float(p_value), n_effective, confidence_interval


def _safe_float(value: Any) -> float:
    """Convert value to float, handling None and NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return float(value)


def compute_partial_residuals_stats(
    x_res: pd.Series,
    y_res: pd.Series,
    stats_df: Optional[pd.Series],
    n_res: int,
    method_code: str,
    bootstrap_ci: int,
    rng: np.random.Generator,
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Compute partial correlation statistics from residuals."""
    r_residual = np.nan
    p_residual = np.nan
    n_partial = n_res
    
    if stats_df is not None:
        r_residual = _safe_float(stats_df.get("r_partial", np.nan))
        p_residual = _safe_float(stats_df.get("p_partial", np.nan))
        n_partial = int(stats_df.get("n_partial", n_partial))
    
    if not np.isfinite(r_residual):
        # Always use pearson for residuals (even if method is spearman,
        # ranked residuals are Pearson-correlated to get Spearman partial)
        r_residual, p_residual = compute_correlation(
            x_res,
            y_res,
            method="pearson"
        )
    
    if bootstrap_ci > 0:
        confidence_interval = bootstrap_corr_ci(
            x_res,
            y_res,
            method_code,
            n_boot=bootstrap_ci,
            rng=rng
        )
    else:
        confidence_interval = (np.nan, np.nan)
    
    return (
        float(r_residual),
        float(p_residual),
        n_partial,
        confidence_interval
    )

