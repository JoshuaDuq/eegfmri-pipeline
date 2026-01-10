"""
Statistical Aggregation
=======================

Functions for aggregating statistics across subjects/groups.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_ci_level
from .correlation import fisher_z, inverse_fisher_z


_MIN_SAMPLES_FOR_STATISTICS = 2


###################################################################
# Internal Helpers
###################################################################


# Import consolidated z-critical value function from base
from .base import get_z_critical_value as _get_z_critical_value


# Import consolidated finite value filtering from base
from .base import filter_finite_values as _filter_finite_values


def _compute_mean_confidence_interval(
    values: np.ndarray,
    z_critical: float,
) -> Tuple[float, float, float]:
    """Compute mean and confidence interval bounds from values.
    
    Parameters
    ----------
    values : np.ndarray
        Array of numeric values
    z_critical : float
        Z-critical value for confidence interval
        
    Returns
    -------
    Tuple[float, float, float]
        (mean, ci_lower, ci_upper)
    """
    n = len(values)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(n))
    margin_of_error = z_critical * standard_error
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, ci_lower, ci_upper


###################################################################
# Channel Statistics
###################################################################


def compute_group_channel_statistics(
    channel_data: Dict[str, np.ndarray],
    config: Optional[Any] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute mean and confidence intervals for each channel across subjects.
    
    Parameters
    ----------
    channel_data : Dict[str, np.ndarray]
        Dictionary mapping channel names to arrays of values across subjects
    config : Optional[Any]
        Configuration object for CI level
        
    Returns
    -------
    Dict[str, Tuple[float, float, float]]
        Dictionary mapping channel names to (mean, ci_lower, ci_upper)
    """
    ci_level = get_ci_level(config)
    z_critical = _get_z_critical_value(ci_level)
    
    results = {}
    for channel_name, values in channel_data.items():
        valid_values = _filter_finite_values(values)
        
        if len(valid_values) < _MIN_SAMPLES_FOR_STATISTICS:
            results[channel_name] = (np.nan, np.nan, np.nan)
            continue
        
        mean, ci_lower, ci_upper = _compute_mean_confidence_interval(
            valid_values, z_critical
        )
        results[channel_name] = (mean, ci_lower, ci_upper)
    
    return results


def compute_channel_confidence_interval(
    z_scores: np.ndarray,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute confidence interval from z-transformed values.
    
    Parameters
    ----------
    z_scores : np.ndarray
        Array of z-transformed values
    config : Optional[Any]
        Configuration object for CI level
        
    Returns
    -------
    Tuple[float, float]
        (ci_lower, ci_upper) in z-space
    """
    ci_level = get_ci_level(config)
    valid_z_scores = _filter_finite_values(z_scores)
    
    if len(valid_z_scores) < _MIN_SAMPLES_FOR_STATISTICS:
        return np.nan, np.nan
    
    z_critical = _get_z_critical_value(ci_level)
    _, ci_lower, ci_upper = _compute_mean_confidence_interval(
        valid_z_scores, z_critical
    )
    
    return ci_lower, ci_upper


###################################################################
# Data Pooling
###################################################################


def pool_data_by_strategy(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    strategy: str = "concatenate",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool data across subjects using specified strategy.
    
    Parameters
    ----------
    x_lists : List[np.ndarray]
        List of x arrays, one per subject
    y_lists : List[np.ndarray]
        List of y arrays, one per subject
    strategy : str
        Pooling strategy: "concatenate" or "mean"
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pooled (x, y) arrays
        
    Raises
    ------
    ValueError
        If strategy is not recognized
    """
    if strategy == "concatenate":
        return _pool_by_concatenation(x_lists, y_lists)
    if strategy == "mean":
        return _pool_by_mean(x_lists, y_lists)
    
    raise ValueError(f"Unknown pooling strategy: {strategy}")


def _pool_by_concatenation(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate all non-empty arrays."""
    x_pooled = np.concatenate([x for x in x_lists if len(x) > 0])
    y_pooled = np.concatenate([y for y in y_lists if len(y) > 0])
    return x_pooled, y_pooled


def _pool_by_mean(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean of each non-empty array."""
    x_means = [np.nanmean(x) for x in x_lists if len(x) > 0]
    y_means = [np.nanmean(y) for y in y_lists if len(y) > 0]
    return np.array(x_means), np.array(y_means)


###################################################################
# Band Statistics
###################################################################


def compute_band_summary_statistics(
    band_data: pd.Series,
    config: Optional[Any] = None,
) -> Tuple[float, float, float, int]:
    """Compute summary statistics for band data.
    
    Parameters
    ----------
    band_data : pd.Series
        Series of band values across subjects
    config : Optional[Any]
        Configuration object for CI level
        
    Returns
    -------
    Tuple[float, float, float, int]
        (mean, ci_lower, ci_upper, sample_size)
    """
    valid_values = band_data.dropna().values
    n_samples = len(valid_values)
    
    if n_samples < _MIN_SAMPLES_FOR_STATISTICS:
        return np.nan, np.nan, np.nan, n_samples
    
    ci_level = get_ci_level(config)
    z_critical = _get_z_critical_value(ci_level)
    mean, ci_lower, ci_upper = _compute_mean_confidence_interval(
        valid_values, z_critical
    )
    
    return mean, ci_lower, ci_upper, n_samples


def compute_band_summaries(
    means_df: pd.DataFrame,
    bands_present: List[str],
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Compute summary statistics for each band.
    
    Parameters
    ----------
    means_df : pd.DataFrame
        DataFrame with band columns containing mean values per subject
    bands_present : List[str]
        List of band names to process
    config : Optional[Any]
        Configuration object for CI level
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: band, mean, ci_low, ci_high, n
    """
    summaries = []
    for band_name in bands_present:
        if band_name not in means_df.columns:
            continue
        
        mean, ci_lower, ci_upper, sample_size = compute_band_summary_statistics(
            means_df[band_name], config
        )
        summaries.append({
            "band": band_name,
            "mean": mean,
            "ci_low": ci_lower,
            "ci_high": ci_upper,
            "n": sample_size,
        })
    
    return pd.DataFrame(summaries)


def compute_fisher_transformed_mean(
    edge_df: pd.DataFrame,
    config: Optional[Any] = None,
) -> pd.Series:
    """Compute Fisher-transformed mean correlation per edge.
    
    Parameters
    ----------
    edge_df : pd.DataFrame
        DataFrame where each column is an edge and rows are subjects
    config : Optional[Any]
        Configuration object (passed to fisher_z_transform_mean)
        
    Returns
    -------
    pd.Series
        Series of Fisher-aggregated mean correlations per edge
    """
    from .correlation import fisher_z_transform_mean
    
    def _fisher_mean_per_edge(values: pd.Series) -> float:
        """Compute Fisher-transformed mean for a single edge."""
        valid_values = _filter_finite_values(values.values)
        if len(valid_values) == 0:
            return np.nan
        return fisher_z_transform_mean(valid_values, config)
    
    return edge_df.apply(_fisher_mean_per_edge)


def compute_group_band_statistics(
    df: pd.DataFrame,
    bands: List[str],
    ci_multiplier: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[List[str], List[float], List[float], List[float], List[int]]:
    """Compute group statistics for bands.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with band columns and subject rows
    bands : List[str]
        List of band names to process
    ci_multiplier : Optional[float]
        Pre-computed z-critical value. If None, computed from config.
    config : Optional[Any]
        Configuration object for CI level
        
    Returns
    -------
    Tuple[List[str], List[float], List[float], List[float], List[int]]
        (band_names, means, ci_lowers, ci_uppers, sample_sizes)
    """
    if ci_multiplier is None:
        ci_level = get_ci_level(config)
        ci_multiplier = _get_z_critical_value(ci_level)
    
    band_names = []
    means = []
    ci_lowers = []
    ci_uppers = []
    sample_sizes = []
    
    for band_name in bands:
        if band_name not in df.columns:
            continue
        
        valid_values = df[band_name].dropna().values
        sample_size = len(valid_values)
        
        if sample_size < _MIN_SAMPLES_FOR_STATISTICS:
            continue
        
        mean, ci_lower, ci_upper = _compute_mean_confidence_interval(
            valid_values, ci_multiplier
        )
        
        band_names.append(band_name)
        means.append(mean)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        sample_sizes.append(sample_size)
    
    return band_names, means, ci_lowers, ci_uppers, sample_sizes


###################################################################
# Error Bar Conversion
###################################################################


def compute_error_bars_from_ci_dicts(
    values: List[float],
    ci_dicts: List[Optional[Dict[str, List[float]]]],
) -> Tuple[List[float], List[float]]:
    """Convert confidence interval dictionaries to error bar arrays.
    
    Parameters
    ----------
    values : List[float]
        List of mean values
    ci_dicts : List[Optional[Dict[str, List[float]]]]
        List of CI dictionaries with "low" and "high" keys
        
    Returns
    -------
    Tuple[List[float], List[float]]
        (lower_errors, upper_errors) where errors are distances from mean
    """
    lower_errors = []
    upper_errors = []
    
    for mean_value, ci_dict in zip(values, ci_dicts):
        if not _has_valid_ci_dict(ci_dict):
            lower_errors.append(0.0)
            upper_errors.append(0.0)
            continue
        
        ci_lower_value = ci_dict["low"][0] if ci_dict["low"] else mean_value
        ci_upper_value = ci_dict["high"][0] if ci_dict["high"] else mean_value
        
        lower_errors.append(mean_value - ci_lower_value)
        upper_errors.append(ci_upper_value - mean_value)
    
    return lower_errors, upper_errors


def _has_valid_ci_dict(ci_dict: Optional[Dict[str, List[float]]]) -> bool:
    """Check if CI dictionary has required keys."""
    return ci_dict is not None and "low" in ci_dict and "high" in ci_dict


def compute_error_bars_from_arrays(
    means: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
) -> np.ndarray:
    """Convert confidence interval arrays to error bar format.
    
    Parameters
    ----------
    means : List[float]
        List of mean values
    ci_lower : List[float]
        List of lower CI bounds
    ci_upper : List[float]
        List of upper CI bounds
        
    Returns
    -------
    np.ndarray
        Array of shape (2, n) where first row is lower errors,
        second row is upper errors
    """
    lower_errors = [mean - lower for mean, lower in zip(means, ci_lower)]
    upper_errors = [upper - mean for mean, upper in zip(means, ci_upper)]
    return np.array([lower_errors, upper_errors])


###################################################################
# Domain-Specific Helpers
###################################################################


def count_trials_by_condition(
    events_df: pd.DataFrame,
    pain_col: str = "pain_binary",
) -> Dict[str, int]:
    """Count trials by pain/nonpain condition.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame with trial events
    pain_col : str
        Column name containing pain condition (1=pain, 0=nonpain)
        
    Returns
    -------
    Dict[str, int]
        Dictionary with keys: "pain", "nonpain", "total"
    """
    if pain_col not in events_df.columns:
        return {"total": len(events_df)}
    
    pain_values = events_df[pain_col].values
    pain_count = int(np.sum(pain_values == 1))
    nonpain_count = int(np.sum(pain_values == 0))
    total_count = len(events_df)
    
    return {
        "pain": pain_count,
        "nonpain": nonpain_count,
        "total": total_count,
    }


def compute_duration_p_value(
    nonpain_data: np.ndarray,
    pain_data: np.ndarray,
) -> float:
    """Compute p-value for duration difference using Mann-Whitney U test.
    
    Parameters
    ----------
    nonpain_data : np.ndarray
        Array of nonpain condition durations
    pain_data : np.ndarray
        Array of pain condition durations
        
    Returns
    -------
    float
        p-value from Mann-Whitney U test, or np.nan if insufficient data
    """
    valid_nonpain = _filter_finite_values(nonpain_data)
    valid_pain = _filter_finite_values(pain_data)
    
    if len(valid_nonpain) < _MIN_SAMPLES_FOR_STATISTICS:
        return np.nan
    if len(valid_pain) < _MIN_SAMPLES_FOR_STATISTICS:
        return np.nan
    
    _, p_value = stats.mannwhitneyu(
        valid_nonpain, valid_pain, alternative="two-sided"
    )
    return float(p_value)

