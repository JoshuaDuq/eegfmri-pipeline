"""
ROI Statistics
==============

Region of interest and masked statistics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from .base import get_statistics_constants


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
    data_group_a: Optional[np.ndarray] = None,
    data_group_b: Optional[np.ndarray] = None,
    paired: bool = False,
    min_samples: int = 3,
) -> Optional[float]:
    """Compute p-value for ROI comparison between two groups.

    Args:
        mask_vec: Boolean mask for channels in ROI.
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
