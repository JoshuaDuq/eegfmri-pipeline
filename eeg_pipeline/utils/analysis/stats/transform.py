"""
Data Transformation
===================

Data normalization and preparation functions.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


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






