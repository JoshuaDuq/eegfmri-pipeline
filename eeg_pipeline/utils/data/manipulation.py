"""
Data Manipulation Utilities.

Common functions for manipulating DataFrames and other data structures,
including connectivity matrix operations and topomap data preparation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd


def reorder_pivot(
    pivot: pd.DataFrame,
    row_order: List[str],
    col_order: List[str],
) -> pd.DataFrame:
    """Reorder pivot table rows and columns to match specified order.
    
    Args:
        pivot: Pivot table to reorder
        row_order: Desired row order (items not in pivot are skipped)
        col_order: Desired column order (items not in pivot are skipped)
    
    Returns:
        Reordered pivot table
    """
    if pivot.empty:
        return pivot
    
    # Filter to existing rows/columns
    rows = [r for r in row_order if r in pivot.index]
    cols = [c for c in col_order if c in pivot.columns]
    
    # Add any remaining rows/columns not in order
    rows += [r for r in pivot.index if r not in rows]
    cols += [c for c in pivot.columns if c not in cols]
    
    return pivot.reindex(index=rows, columns=cols)


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column from a list of candidates.
    
    Args:
        df: DataFrame to search
        candidates: List of column names to try in order
    
    Returns:
        First matching column name, or None if no match
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _find_power_column(
    pow_cols: List[str],
    band: str,
    channel: str,
    period: str,
) -> Optional[str]:
    """Find power column for a specific band, channel, and period.
    
    Args:
        pow_cols: Available power column names
        band: Power band name
        channel: Channel name
        period: Period name ('active', 'early', 'mid', 'late')
    
    Returns:
        First matching column name, or None if not found
    """
    candidates = [
        f"power_{period}_{band}_ch_{channel}_logratio",
        f"power_{period}_{band}_ch_{channel}_log10raw",
    ]
    pow_cols_set = set(pow_cols)
    for candidate in candidates:
        if candidate in pow_cols_set:
            return candidate
    return None


def _determine_statistic_type(column_names: List[str]) -> str:
    """Determine statistic type from column names.
    
    Args:
        column_names: List of column names to check
    
    Returns:
        Statistic type: 'logratio' or 'log10raw'
    """
    has_logratio = any(name.endswith("_logratio") for name in column_names)
    return "logratio" if has_logratio else "log10raw"


def _add_active_power_column(
    pow_df: pd.DataFrame,
    pow_cols: List[str],
    band: str,
    channel: str,
    col_name_to_series: Dict[str, pd.Series],
    active_cols: List[str],
) -> None:
    """Add active power column for a band-channel combination.
    
    Tries to find a direct active column first, otherwise averages
    early/mid/late periods if all are available.
    
    Args:
        pow_df: DataFrame containing power features
        pow_cols: List of power column names
        band: Power band name
        channel: Channel name
        col_name_to_series: Dictionary to populate with series
        active_cols: List to populate with column names
    """
    preferred = _find_power_column(pow_cols, band, channel, "active")
    if preferred is not None:
        col_name_to_series[preferred] = pow_df[preferred]
        active_cols.append(preferred)
        return

    early = _find_power_column(pow_cols, band, channel, "early")
    mid = _find_power_column(pow_cols, band, channel, "mid")
    late = _find_power_column(pow_cols, band, channel, "late")

    if early is not None and mid is not None and late is not None:
        active_value = pow_df[[early, mid, late]].mean(axis=1)
        statistic_type = _determine_statistic_type([early, mid, late])
        output_name = f"power_active_{band}_ch_{channel}_{statistic_type}"
        col_name_to_series[output_name] = active_value
        active_cols.append(output_name)


def _add_baseline_columns(
    baseline_df: pd.DataFrame,
    band: str,
    channel: str,
    col_name_to_series: Dict[str, pd.Series],
    active_cols: List[str],
) -> None:
    """Add baseline columns for a band-channel combination.
    
    Args:
        baseline_df: DataFrame containing baseline features
        band: Power band name
        channel: Channel name
        col_name_to_series: Dictionary to populate with series
        active_cols: List to populate with column names
    """
    if baseline_df.empty:
        return

    candidates = [
        f"power_baseline_{band}_ch_{channel}_mean",
    ]
    baseline_col = find_column(baseline_df, candidates)
    
    if baseline_col is not None:
        col_name_to_series[baseline_col] = baseline_df[baseline_col]
        active_cols.append(baseline_col)


def build_active_features(
    pow_df: pd.DataFrame,
    pow_cols: List[str],
    baseline_df: pd.DataFrame,
    ch_names: List[str],
    power_bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct active-averaged band power DataFrame.
    
    Args:
        pow_df: DataFrame containing power features
        pow_cols: List of power column names
        baseline_df: DataFrame containing baseline features
        ch_names: List of channel names
        power_bands: List of power band names
    
    Returns:
        Tuple of (active DataFrame, list of active column names)
    """
    col_name_to_series: Dict[str, pd.Series] = {}
    active_cols: List[str] = []

    for band in power_bands:
        for channel in ch_names:
            _add_active_power_column(
                pow_df, pow_cols, band, channel, col_name_to_series, active_cols
            )
            _add_baseline_columns(
                baseline_df, band, channel, col_name_to_series, active_cols
            )

    active_df = pd.DataFrame(col_name_to_series)
    active_df = active_df.reindex(columns=active_cols)
    return active_df, active_cols


###################################################################
# Connectivity Matrix Operations
###################################################################


def flatten_lower_triangles(
    connectivity_trials: np.ndarray,
    labels: Optional[np.ndarray],
    prefix: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Flatten lower triangular connectivity matrices to feature vectors.
    
    Args:
        connectivity_trials: 3D array (trials, nodes, nodes)
        labels: Node labels for naming columns
        prefix: Prefix for column names
    
    Returns:
        Tuple of (DataFrame with flattened values, list of column names)
    
    Raises:
        ValueError: If connectivity_trials is not 3D or has invalid shape
    """
    if connectivity_trials.ndim != 3:
        raise ValueError("Connectivity array must be 3D (trials, nodes, nodes)")
    
    n_trials, n_nodes, n_nodes_check = connectivity_trials.shape
    if n_nodes != n_nodes_check:
        raise ValueError("Connectivity array must be square in last two dimensions")

    lower_triangle_row_indices, lower_triangle_col_indices = np.tril_indices(
        n_nodes, k=-1
    )
    flattened_data = connectivity_trials[
        :, lower_triangle_row_indices, lower_triangle_col_indices
    ]

    has_valid_labels = labels is not None and len(labels) == n_nodes
    if has_valid_labels:
        pair_names = [
            f"{labels[i]}__{labels[j]}"
            for i, j in zip(lower_triangle_row_indices, lower_triangle_col_indices)
        ]
    else:
        pair_names = [
            f"n{i}_n{j}"
            for i, j in zip(lower_triangle_row_indices, lower_triangle_col_indices)
        ]

    column_names = [f"{prefix}_{pair}" for pair in pair_names]
    return pd.DataFrame(flattened_data), column_names


###################################################################
# Topomap Data Preparation
###################################################################


def prepare_topomap_correlation_data(
    band_data: Dict[str, Any],
    info: mne.Info,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare correlation data for topomap visualization.
    
    Args:
        band_data: Dictionary with 'channels', 'correlations', 'significant_mask' keys
        info: MNE Info object with channel information
    
    Returns:
        Tuple of (topomap data array, significance mask array)
    
    Raises:
        KeyError: If band_data is missing required keys
        ValueError: If array lengths are inconsistent
    """
    required_keys = ["channels", "correlations", "significant_mask"]
    missing_keys = [key for key in required_keys if key not in band_data]
    if missing_keys:
        raise KeyError(f"band_data missing required keys: {missing_keys}")

    channels = band_data["channels"]
    correlations = band_data["correlations"]
    significant_mask = band_data["significant_mask"]

    if len(correlations) != len(channels) or len(significant_mask) != len(channels):
        raise ValueError(
            "channels, correlations, and significant_mask must have same length"
        )

    n_info_channels = len(info["ch_names"])
    topomap_data = np.zeros(n_info_channels)
    topomap_mask = np.zeros(n_info_channels, dtype=bool)

    for info_channel_idx, info_channel_name in enumerate(info["ch_names"]):
        if info_channel_name in channels:
            band_channel_idx = channels.index(info_channel_name)
            correlation_value = correlations[band_channel_idx]
            
            if np.isfinite(correlation_value):
                topomap_data[info_channel_idx] = correlation_value
            
            topomap_mask[info_channel_idx] = bool(significant_mask[band_channel_idx])

    return topomap_data, topomap_mask


def extract_pain_masks(pain_vals: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract boolean masks for pain and non-pain trials.
    
    Args:
        pain_vals: Series containing pain binary values (0 or 1)
    
    Returns:
        Tuple of (pain_mask, nonpain_mask) where:
        - pain_mask: Boolean mask where pain_vals == 1
        - nonpain_mask: Boolean mask where pain_vals == 0
    """
    pain_arr = np.asarray(pain_vals)
    pain_mask = pain_arr == 1
    nonpain_mask = pain_arr == 0
    return pain_mask, nonpain_mask


def extract_duration_data(durations: pd.Series, mask: np.ndarray) -> np.ndarray:
    """Extract duration data for masked trials.
    
    Args:
        durations: Series containing duration values
        mask: Boolean mask to apply
    
    Returns:
        Duration values for masked trials
    """
    return durations.values[mask]
