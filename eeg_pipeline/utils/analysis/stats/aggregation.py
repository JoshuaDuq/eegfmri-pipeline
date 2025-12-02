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

from .base import get_ci_level, get_config_value
from .correlation import fisher_z, inverse_fisher_z


def compute_group_channel_statistics(
    channel_data: Dict[str, np.ndarray],
    config: Optional[Any] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute mean, CI for each channel across subjects."""
    ci_level = get_ci_level(config)
    z_crit = stats.norm.ppf((1 + ci_level) / 2)
    
    results = {}
    for ch, values in channel_data.items():
        valid = values[np.isfinite(values)]
        if len(valid) < 2:
            results[ch] = (np.nan, np.nan, np.nan)
            continue
        
        mean = np.mean(valid)
        se = np.std(valid, ddof=1) / np.sqrt(len(valid))
        ci_lo = mean - z_crit * se
        ci_hi = mean + z_crit * se
        results[ch] = (mean, ci_lo, ci_hi)
    
    return results


def compute_channel_confidence_interval(
    z_scores: np.ndarray,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute CI from z-transformed values."""
    ci_level = get_ci_level(config)
    valid = z_scores[np.isfinite(z_scores)]
    
    if len(valid) < 2:
        return np.nan, np.nan
    
    z_mean = np.mean(valid)
    se = np.std(valid, ddof=1) / np.sqrt(len(valid))
    z_crit = stats.norm.ppf((1 + ci_level) / 2)
    
    return float(z_mean - z_crit * se), float(z_mean + z_crit * se)


def pool_data_by_strategy(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    strategy: str = "concatenate",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool data across subjects using specified strategy."""
    if strategy == "concatenate":
        x_all = np.concatenate([x for x in x_lists if len(x) > 0])
        y_all = np.concatenate([y for y in y_lists if len(y) > 0])
        return x_all, y_all
    elif strategy == "mean":
        x_means = [np.nanmean(x) for x in x_lists if len(x) > 0]
        y_means = [np.nanmean(y) for y in y_lists if len(y) > 0]
        return np.array(x_means), np.array(y_means)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


def compute_band_summary_statistics(
    band_data: pd.Series,
    config: Optional[Any] = None,
) -> Tuple[float, float, float, int]:
    """Compute mean, CI low, CI high, n for band data."""
    ci_level = get_ci_level(config)
    valid = band_data.dropna().values
    n = len(valid)
    
    if n < 2:
        return np.nan, np.nan, np.nan, n
    
    mean = np.mean(valid)
    se = np.std(valid, ddof=1) / np.sqrt(n)
    z_crit = stats.norm.ppf((1 + ci_level) / 2)
    
    return float(mean), float(mean - z_crit * se), float(mean + z_crit * se), n


def compute_band_summaries(
    means_df: pd.DataFrame,
    bands_present: List[str],
) -> pd.DataFrame:
    """Compute summary statistics for each band."""
    summaries = []
    for band in bands_present:
        if band not in means_df.columns:
            continue
        mean, ci_lo, ci_hi, n = compute_band_summary_statistics(means_df[band])
        summaries.append({
            "band": band,
            "mean": mean,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n": n,
        })
    return pd.DataFrame(summaries)


def compute_fisher_transformed_mean(
    edge_df: pd.DataFrame,
    config: Optional[Any] = None,
) -> pd.Series:
    """Compute Fisher-transformed mean correlation per edge."""
    def fisher_mean(vals):
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0:
            return np.nan
        zs = [fisher_z(v) for v in valid]
        return inverse_fisher_z(np.mean(zs))
    
    return edge_df.apply(fisher_mean)


def compute_group_band_statistics(
    df: pd.DataFrame,
    bands: List[str],
    ci_multiplier: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[List[str], List[float], List[float], List[float], List[int]]:
    """Compute group statistics for bands."""
    if ci_multiplier is None:
        ci_level = get_ci_level(config)
        ci_multiplier = stats.norm.ppf((1 + ci_level) / 2)
    
    band_names = []
    means = []
    ci_lows = []
    ci_highs = []
    counts = []
    
    for band in bands:
        if band not in df.columns:
            continue
        
        values = df[band].dropna().values
        n = len(values)
        if n < 2:
            continue
        
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(n)
        
        band_names.append(band)
        means.append(mean)
        ci_lows.append(mean - ci_multiplier * se)
        ci_highs.append(mean + ci_multiplier * se)
        counts.append(n)
    
    return band_names, means, ci_lows, ci_highs, counts


def compute_error_bars_from_ci_dicts(
    values: List[float],
    ci_dicts: List[Optional[Dict[str, List[float]]]],
) -> Tuple[List[float], List[float]]:
    """Convert CI dicts to error bar arrays."""
    lower_errs = []
    upper_errs = []
    
    for val, ci in zip(values, ci_dicts):
        if ci is None or "low" not in ci or "high" not in ci:
            lower_errs.append(0)
            upper_errs.append(0)
        else:
            lower_errs.append(val - ci["low"][0] if ci["low"] else 0)
            upper_errs.append(ci["high"][0] - val if ci["high"] else 0)
    
    return lower_errs, upper_errs


def compute_error_bars_from_arrays(
    means: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
) -> np.ndarray:
    """Convert CI arrays to error bar format."""
    lower_errs = [m - l for m, l in zip(means, ci_lower)]
    upper_errs = [u - m for m, u in zip(means, ci_upper)]
    return np.array([lower_errs, upper_errs])


def count_trials_by_condition(
    events_df: pd.DataFrame,
    pain_col: str = "pain_binary",
) -> Dict[str, int]:
    """Count trials by pain/nonpain condition."""
    if pain_col not in events_df.columns:
        return {"total": len(events_df)}
    
    pain_vals = events_df[pain_col].values
    return {
        "pain": int(np.sum(pain_vals == 1)),
        "nonpain": int(np.sum(pain_vals == 0)),
        "total": len(events_df),
    }


def compute_duration_p_value(
    nonpain_data: np.ndarray,
    pain_data: np.ndarray,
) -> float:
    """Compute p-value for duration difference (Mann-Whitney)."""
    valid_np = nonpain_data[np.isfinite(nonpain_data)]
    valid_p = pain_data[np.isfinite(pain_data)]
    
    if len(valid_np) < 2 or len(valid_p) < 2:
        return np.nan
    
    _, p = stats.mannwhitneyu(valid_np, valid_p, alternative="two-sided")
    return float(p)

