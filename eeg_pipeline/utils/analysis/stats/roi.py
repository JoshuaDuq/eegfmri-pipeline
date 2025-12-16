"""
ROI Statistics
==============

Region of interest and masked statistics.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_statistics_constants


def extract_roi_statistics(
    df: Optional[pd.DataFrame],
    roi_name: str,
    band_name: str,
) -> Optional[pd.Series]:
    """Extract ROI statistics from dataframe."""
    if df is None or df.empty or "roi" not in df.columns or "band" not in df.columns:
        return None
    
    band_match = df["band"].astype(str).str.lower() == band_name.lower()
    roi_match = df["roi"].astype(str).str.lower() == roi_name.lower()
    mask = band_match & roi_match
    
    return df.loc[mask].iloc[0] if mask.any() else None


def extract_overall_statistics(
    df: Optional[pd.DataFrame],
    band_name: str,
    overall_keys: Optional[List[str]] = None,
) -> Optional[pd.Series]:
    """Extract overall statistics for a band."""
    if overall_keys is None:
        overall_keys = ["overall", "all", "global"]
    
    for key in overall_keys:
        row = extract_roi_statistics(df, key, band_name)
        if row is not None:
            return row
    return None


def update_stats_from_dataframe(
    stats_df: Optional[pd.Series],
    r_val: float,
    p_val: float,
    n_eff: int,
    ci_val: Tuple[float, float],
) -> Tuple[float, float, int, Tuple[float, float]]:
    """Update statistics from dataframe row."""
    if stats_df is None:
        return r_val, p_val, n_eff, ci_val
    
    def _sf(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(v)
    
    return (
        _sf(stats_df.get("r", r_val)),
        _sf(stats_df.get("p", p_val)),
        int(stats_df.get("n", n_eff)),
        (_sf(stats_df.get("r_ci_low", ci_val[0])), _sf(stats_df.get("r_ci_high", ci_val[1]))),
    )


def compute_roi_percentage_change(
    roi_data: np.ndarray,
    is_percent_format: bool,
    config: Optional[Any] = None,
) -> float:
    """Compute ROI percentage change."""
    roi_mean = float(np.nanmean(roi_data))
    if is_percent_format:
        return roi_mean
    constants = get_statistics_constants(config)
    log_base = constants.get("log_base", 10)
    pct_mult = constants.get("percentage_multiplier", 100)
    return (log_base ** roi_mean - 1.0) * pct_mult


def compute_roi_pvalue(
    mask_vec: np.ndarray,
    ch_names: List[str],
    p_ch: Optional[np.ndarray],
    sig_mask: Optional[np.ndarray],
    is_cluster: bool,
    cluster_p_min: Optional[float],
    data_group_a: Optional[np.ndarray] = None,
    data_group_b: Optional[np.ndarray] = None,
    paired: bool = False,
    min_samples: int = 3,
) -> Optional[float]:
    """Compute p-value for ROI comparison."""
    if data_group_a is None or data_group_b is None:
        return None
    
    try:
        roi_a = np.nanmean(data_group_a[:, mask_vec], axis=1)
        roi_b = np.nanmean(data_group_b[:, mask_vec], axis=1)
        
        valid_a, valid_b = np.isfinite(roi_a), np.isfinite(roi_b)
        if np.sum(valid_a) < min_samples or np.sum(valid_b) < min_samples:
            return None
        
        if paired and len(roi_a) == len(roi_b):
            valid_both = valid_a & valid_b
            res = stats.ttest_rel(roi_a[valid_both], roi_b[valid_both])
        else:
            res = stats.ttest_ind(roi_a[valid_a], roi_b[valid_b])
        
        return float(res.pvalue) if np.isfinite(res.pvalue) else None
    except Exception:
        return None


def compute_statistics_for_mask(
    data_values: pd.Series,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """Compute mean and SEM for masked data."""
    masked = data_values[mask].to_numpy()
    masked = masked[np.isfinite(masked)]
    if len(masked) == 0:
        return 0.0, 0.0
    mean_val = np.mean(masked)
    sem_val = np.std(masked) / np.sqrt(len(masked)) if len(masked) > 1 else 0.0
    return float(mean_val), float(sem_val)


def compute_coverage_statistics(
    coverage_values: pd.Series,
    nonpain_mask: np.ndarray,
    pain_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute coverage statistics by condition."""
    mean_np, sem_np = compute_statistics_for_mask(coverage_values, nonpain_mask)
    mean_p, sem_p = compute_statistics_for_mask(coverage_values, pain_mask)
    return mean_np, mean_p, sem_np, sem_p






