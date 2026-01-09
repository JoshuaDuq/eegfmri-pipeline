"""
ROI Statistics
==============

Region of interest and masked statistics.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_statistics_constants


def extract_roi_statistics(
    df: Optional[pd.DataFrame],
    roi_name: str,
    band_name: str,
) -> Optional[pd.Series]:
    """Extract ROI statistics from dataframe matching ROI and band names.

    Args:
        df: Dataframe containing ROI statistics with 'roi' and 'band' columns.
        roi_name: Name of the ROI to extract (case-insensitive).
        band_name: Name of the frequency band to extract (case-insensitive).

    Returns:
        Series containing the matching row, or None if no match found or
        dataframe is invalid.
    """
    if df is None:
        return None
    if df.empty:
        return None
    if "roi" not in df.columns or "band" not in df.columns:
        return None

    band_match = df["band"].astype(str).str.lower() == band_name.lower()
    roi_match = df["roi"].astype(str).str.lower() == roi_name.lower()
    mask = band_match & roi_match

    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def extract_overall_statistics(
    df: Optional[pd.DataFrame],
    band_name: str,
    overall_keys: Optional[List[str]] = None,
) -> Optional[pd.Series]:
    """Extract overall statistics for a frequency band.

    Attempts to find statistics using common overall ROI names.

    Args:
        df: Dataframe containing ROI statistics.
        band_name: Name of the frequency band to extract.
        overall_keys: List of ROI names to try for overall statistics.
            Defaults to ["overall", "all", "global"].

    Returns:
        Series containing the matching overall statistics, or None if not found.
    """
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
    """Update statistics from dataframe row, using defaults if missing.

    Args:
        stats_df: Series containing statistics (may have 'r', 'p', 'n',
            'r_ci_low', 'r_ci_high' keys).
        r_val: Default correlation value.
        p_val: Default p-value.
        n_eff: Default effective sample size.
        ci_val: Default confidence interval (low, high).

    Returns:
        Tuple of (r, p, n, (ci_low, ci_high)) with values from dataframe
        or defaults.
    """
    if stats_df is None:
        return r_val, p_val, n_eff, ci_val

    def safe_float(value: Optional[float]) -> float:
        """Convert value to float, handling None and NaN."""
        if value is None:
            return np.nan
        if isinstance(value, float) and np.isnan(value):
            return np.nan
        return float(value)

    r = safe_float(stats_df.get("r", r_val))
    p = safe_float(stats_df.get("p", p_val))
    n = int(stats_df.get("n", n_eff))
    ci_low = safe_float(stats_df.get("r_ci_low", ci_val[0]))
    ci_high = safe_float(stats_df.get("r_ci_high", ci_val[1]))

    return r, p, n, (ci_low, ci_high)


def compute_roi_percentage_change(
    roi_data: np.ndarray,
    is_percent_format: bool,
    config: Optional[dict] = None,
) -> float:
    """Compute ROI percentage change from mean data.

    If data is already in percent format, returns the mean directly.
    Otherwise, converts from log space to percentage change.

    Args:
        roi_data: Array of ROI values.
        is_percent_format: Whether data is already in percentage format.
        config: Optional configuration dict for statistics constants.

    Returns:
        Percentage change value.
    """
    roi_mean = float(np.nanmean(roi_data))
    if is_percent_format:
        return roi_mean

    constants = get_statistics_constants(config)
    log_base = constants.get("log_base", 10)
    percentage_multiplier = constants.get("percentage_multiplier", 100)

    return (log_base ** roi_mean - 1.0) * percentage_multiplier


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
    """Compute p-value for ROI comparison between two groups.

    Args:
        mask_vec: Boolean mask for channels in ROI.
        ch_names: Channel names (for reference, currently unused).
        p_ch: Per-channel p-values (for reference, currently unused).
        sig_mask: Significance mask (for reference, currently unused).
        is_cluster: Whether using cluster correction (for reference, currently unused).
        cluster_p_min: Minimum cluster p-value (for reference, currently unused).
        data_group_a: Data array for group A (samples x channels).
        data_group_b: Data array for group B (samples x channels).
        paired: Whether to use paired t-test.
        min_samples: Minimum number of valid samples required per group.

    Returns:
        P-value from t-test, or None if insufficient data or computation fails.
    """
    if data_group_a is None or data_group_b is None:
        return None

    try:
        roi_group_a = np.nanmean(data_group_a[:, mask_vec], axis=1)
        roi_group_b = np.nanmean(data_group_b[:, mask_vec], axis=1)

        valid_group_a = np.isfinite(roi_group_a)
        valid_group_b = np.isfinite(roi_group_b)
        n_valid_a = np.sum(valid_group_a)
        n_valid_b = np.sum(valid_group_b)

        if n_valid_a < min_samples or n_valid_b < min_samples:
            return None

        is_paired_test = paired and len(roi_group_a) == len(roi_group_b)
        if is_paired_test:
            valid_both = valid_group_a & valid_group_b
            test_result = stats.ttest_rel(
                roi_group_a[valid_both], roi_group_b[valid_both]
            )
        else:
            test_result = stats.ttest_ind(
                roi_group_a[valid_group_a], roi_group_b[valid_group_b]
            )

        pvalue = test_result.pvalue
        if np.isfinite(pvalue):
            return float(pvalue)
        return None
    except (ValueError, TypeError, IndexError):
        return None


def compute_statistics_for_mask(
    data_values: pd.Series,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """Compute mean and standard error of the mean (SEM) for masked data.

    Args:
        data_values: Series of data values.
        mask: Boolean mask to select values.

    Returns:
        Tuple of (mean, sem). Returns (0.0, 0.0) if no valid data.
    """
    masked_values = data_values[mask].to_numpy()
    valid_values = masked_values[np.isfinite(masked_values)]

    if len(valid_values) == 0:
        return 0.0, 0.0

    mean = float(np.mean(valid_values))
    n_samples = len(valid_values)
    sem = float(np.std(valid_values) / np.sqrt(n_samples)) if n_samples > 1 else 0.0

    return mean, sem


def compute_coverage_statistics(
    coverage_values: pd.Series,
    nonpain_mask: np.ndarray,
    pain_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute coverage statistics for non-pain and pain conditions.

    Args:
        coverage_values: Series of coverage values.
        nonpain_mask: Boolean mask for non-pain condition.
        pain_mask: Boolean mask for pain condition.

    Returns:
        Tuple of (mean_nonpain, mean_pain, sem_nonpain, sem_pain).
    """
    mean_nonpain, sem_nonpain = compute_statistics_for_mask(
        coverage_values, nonpain_mask
    )
    mean_pain, sem_pain = compute_statistics_for_mask(coverage_values, pain_mask)

    return mean_nonpain, mean_pain, sem_nonpain, sem_pain
