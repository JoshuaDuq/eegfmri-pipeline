"""
Data Manipulation Utilities.

Common functions for manipulating DataFrames and other data structures,
including connectivity matrix operations and topomap data preparation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Any, Tuple

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


def build_plateau_features(
    pow_df: pd.DataFrame,
    pow_cols: List[str],
    baseline_df: pd.DataFrame,
    baseline_cols: List[str],
    ch_names: List[str],
    power_bands: List[str],
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct plateau-averaged band power DataFrame.
    
    Args:
        pow_df: DataFrame containing power features
        pow_cols: List of power column names
        baseline_df: DataFrame containing baseline features
        baseline_cols: List of baseline column names
        ch_names: List of channel names
        power_bands: List of power band names
        logger: Logger instance
    
    Returns:
        Tuple of (plateau DataFrame, list of plateau column names)
    """
    col_name_to_series = {}
    plateau_cols = []

    def _first_present(candidates: List[str]) -> Optional[str]:
        for name in candidates:
            if name in pow_cols:
                return name
        return None

    for band in power_bands:
        for ch in ch_names:
            preferred = _first_present(
                [
                    f"power_plateau_{band}_ch_{ch}_logratio",
                    f"power_plateau_{band}_ch_{ch}_log10raw",
                    f"pow_{band}_{ch}_plateau",
                ]
            )
            if preferred is not None:
                out_name = preferred
                col_name_to_series[out_name] = pow_df[preferred]
                plateau_cols.append(out_name)
                continue

            early = _first_present(
                [
                    f"power_early_{band}_ch_{ch}_logratio",
                    f"power_early_{band}_ch_{ch}_log10raw",
                    f"pow_{band}_{ch}_early",
                ]
            )
            mid = _first_present(
                [
                    f"power_mid_{band}_ch_{ch}_logratio",
                    f"power_mid_{band}_ch_{ch}_log10raw",
                    f"pow_{band}_{ch}_mid",
                ]
            )
            late = _first_present(
                [
                    f"power_late_{band}_ch_{ch}_logratio",
                    f"power_late_{band}_ch_{ch}_log10raw",
                    f"pow_{band}_{ch}_late",
                ]
            )

            if early is not None and mid is not None and late is not None:
                plateau_val = pow_df[[early, mid, late]].mean(axis=1)

                stat = "logratio" if any(s.endswith("_logratio") for s in [early, mid, late]) else "log10raw"
                out_name = f"power_plateau_{band}_ch_{ch}_{stat}"
                col_name_to_series[out_name] = plateau_val
                plateau_cols.append(out_name)

        if not baseline_df.empty:
            for ch in ch_names:
                baseline_col = None
                for candidate in [
                    f"power_baseline_{band}_ch_{ch}_mean",
                    f"baseline_{band}_{ch}",
                ]:
                    if candidate in baseline_cols:
                        baseline_col = candidate
                        break
                if baseline_col is not None:
                    col_name_to_series[baseline_col] = baseline_df[baseline_col]
                    plateau_cols.append(baseline_col)

    plateau_df = pd.DataFrame(col_name_to_series)
    plateau_df = plateau_df.reindex(columns=plateau_cols)
    return plateau_df, plateau_cols


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
    """
    if connectivity_trials.ndim != 3:
        raise ValueError("Connectivity array must be 3D (trials, nodes, nodes)")

    n_trials, n_nodes, _ = connectivity_trials.shape
    lower_tri_i, lower_tri_j = np.tril_indices(n_nodes, k=-1)
    flattened_data = connectivity_trials[:, lower_tri_i, lower_tri_j]

    if labels is not None and len(labels) == n_nodes:
        pair_names = [f"{labels[i]}__{labels[j]}" for i, j in zip(lower_tri_i, lower_tri_j)]
    else:
        pair_names = [f"n{i}_n{j}" for i, j in zip(lower_tri_i, lower_tri_j)]

    column_names = [f"{prefix}_{pair}" for pair in pair_names]
    return pd.DataFrame(flattened_data), column_names


###################################################################
# Topomap Data Preparation
###################################################################


def prepare_topomap_correlation_data(band_data: Dict, info: mne.Info) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare correlation data for topomap visualization.
    
    Args:
        band_data: Dictionary with 'channels', 'correlations', 'significant_mask' keys
        info: MNE Info object with channel information
    
    Returns:
        Tuple of (topomap data array, significance mask array)
    """
    n_info_chs = len(info["ch_names"])
    topo_data = np.zeros(n_info_chs)
    topo_mask = np.zeros(n_info_chs, dtype=bool)

    for j, info_ch in enumerate(info["ch_names"]):
        if info_ch in band_data["channels"]:
            ch_idx = band_data["channels"].index(info_ch)
            if np.isfinite(band_data["correlations"][ch_idx]):
                topo_data[j] = band_data["correlations"][ch_idx]
            topo_mask[j] = bool(band_data["significant_mask"][ch_idx])

    return topo_data, topo_mask
