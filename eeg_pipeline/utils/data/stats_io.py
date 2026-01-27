"""
Statistics I/O Utilities
========================

Functions for loading precomputed statistics and scatter data from files.
This module handles I/O operations for statistics, not statistical computations.

Note: Statistical computation functions are in `eeg_pipeline.utils.analysis.stats`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import read_table
from eeg_pipeline.utils.data.manipulation import find_column
from .covariates import _build_covariate_matrices


def _build_correlation_stats_candidates(
    feature_type: str,
    target_suffix: str,
    target_suffix_alt: Optional[str],
    method_label: Optional[str] = None,
) -> List[str]:
    """Build list of candidate filenames for precomputed correlation stats."""
    method_suffix = f"_{method_label}" if method_label else ""

    def _both_ext(base: str) -> List[str]:
        return [f"{base}.parquet", f"{base}.tsv"]

    base = f"corr_stats_{feature_type}_vs_{target_suffix}{method_suffix}"
    candidates = _both_ext(base)
    
    if target_suffix_alt:
        alt_base = f"corr_stats_{feature_type}_vs_{target_suffix_alt}{method_suffix}"
        candidates.extend(_both_ext(alt_base))
    
    if method_label:
        candidates.extend(_both_ext(f"corr_stats_{feature_type}_vs_{target_suffix}"))
        if target_suffix_alt:
            candidates.extend(_both_ext(f"corr_stats_{feature_type}_vs_{target_suffix_alt}"))
    
    return candidates


def _determine_target_suffixes(target: str) -> Tuple[str, Optional[str]]:
    """Determine target suffix and alternative suffix from target string."""
    is_rating = "rating" in target.lower()
    target_suffix = "rating" if is_rating else "temperature"
    target_suffix_alt = "temp" if target_suffix == "temperature" else None
    return target_suffix, target_suffix_alt


def load_precomputed_correlations(
    stats_dir: Path,
    feature_type: str,
    target: str,
    logger: Optional[logging.Logger] = None,
    method_label: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load precomputed correlation statistics from parquet or TSV files.
    
    Parameters
    ----------
    stats_dir : Path
        Directory containing correlation stats files
    feature_type : str
        Type of feature: "power", "aperiodic", "connectivity", "itpc", "complexity", "power_roi"
    target : str
        Correlation target: "rating" or "temperature"
    logger : Logger, optional
        Logger instance
    method_label : str, optional
        Method label to append to filename
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: feature/channel/roi, band, r, p, n, q_fdr_global, fdr_reject_global
        Returns None if file not found or empty
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    target_suffix, target_suffix_alt = _determine_target_suffixes(target)
    
    candidates = _build_correlation_stats_candidates(
        feature_type,
        target_suffix,
        target_suffix_alt,
        method_label=method_label,
    )

    for filename in candidates:
        filepath = stats_dir / filename
        if not filepath.exists():
            continue
        
        dataframe = read_table(filepath)
        if dataframe is not None and not dataframe.empty:
            return dataframe

    return None


def _find_roi_column(dataframe: pd.DataFrame) -> Optional[str]:
    """Find the ROI/channel/feature column name in the dataframe."""
    return find_column(dataframe, ["roi", "channel", "feature"])


def _find_band_column(dataframe: pd.DataFrame) -> Optional[str]:
    """Find the band column name in the dataframe."""
    return find_column(dataframe, ["band"])


def get_precomputed_stats_for_roi_band(
    stats_df: Optional[pd.DataFrame],
    roi: str,
    band: str,
) -> Dict[str, Any]:
    """
    Extract precomputed stats for a specific ROI and band from a stats DataFrame.
    
    Parameters
    ----------
    stats_df : pd.DataFrame, optional
        DataFrame containing precomputed statistics
    roi : str
        Region of interest identifier
    band : str
        Frequency band identifier
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: r, p, n, ci_low, ci_high, q, fdr_reject
        Returns empty dict if stats_df is None, empty, or matching row not found
    """
    if stats_df is None or stats_df.empty:
        return {}

    roi_column = _find_roi_column(stats_df)
    if roi_column is None:
        return {}

    band_column = _find_band_column(stats_df)
    if band_column is None:
        return {}
        
    matching_rows = stats_df[
        (stats_df[roi_column].astype(str) == roi)
        & (stats_df[band_column].astype(str) == band)
    ]
    
    if matching_rows.empty:
        return {}
        
    row = matching_rows.iloc[0]

    q_global = row.get("q_global", np.nan)
    q_value = q_global if not np.isnan(q_global) else row.get("q", np.nan)
    fdr_reject = bool(row.get("fdr_reject", False))
    
    return {
        "r": row.get("r", np.nan),
        "p": row.get("p", np.nan),
        "n": row.get("n", 0),
        "ci_low": row.get("ci_low", np.nan),
        "ci_high": row.get("ci_high", np.nan),
        "q": q_value,
        "fdr_reject": fdr_reject,
    }


def _load_epochs_for_subject(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
):
    """Load epochs for a subject and task."""
    from .epochs import load_epochs_for_analysis
    
    epochs, _ = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=deriv_root,
        logger=logger,
        config=config,
    )
    return epochs


def _load_features_for_subject(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    epochs,
):
    """Load features and targets for a subject."""
    from .feature_io import _load_features_and_targets
    
    temporal_df, active_df, conn_df, y, info = _load_features_and_targets(
        subject, task, deriv_root, config, epochs=epochs
    )
    return temporal_df, active_df, conn_df, y, info


def _load_aligned_events_and_covariates(
    epochs,
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    partial_covars: Optional[List[str]],
):
    """Load aligned events and build covariate matrices."""
    from .alignment import get_aligned_events
    from .covariates import extract_temperature_data
    
    aligned_events = get_aligned_events(
        epochs,
        subject,
        task,
        strict=True,
        logger=logger,
        config=config,
    )
    
    temp_series, temp_col = extract_temperature_data(aligned_events, config)
    Z_df_full, Z_df_temp = _build_covariate_matrices(
        aligned_events, partial_covars, temp_col, config
    )
    
    return aligned_events, temp_series, Z_df_full, Z_df_temp


def _build_roi_map(info, config):
    """Build ROI map from info object."""
    from ..analysis.tfr import build_rois_from_info
    
    return build_rois_from_info(info, config)


def load_subject_scatter_data(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
    partial_covars: Optional[List[str]] = None,
) -> Tuple[
    Optional[pd.DataFrame],  # temporal_df
    Optional[pd.DataFrame],  # active_df (power)
    Optional[pd.DataFrame],  # y (target)
    Optional[Any],           # info
    Optional[pd.Series],     # temp_series
    Optional[pd.DataFrame],  # Z_df_full
    Optional[pd.DataFrame],  # Z_df_temp
    Optional[Dict],          # roi_map
    Optional[pd.DataFrame],  # conn_df (connectivity)
]:
    """
    Load all data required for subject behavioral scatter plots.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    task : str
        Task identifier
    deriv_root : Path
        Root directory for derivatives
    config
        Configuration object
    logger : logging.Logger
        Logger instance
    partial_covars : List[str], optional
        List of partial covariate names
        
    Returns
    -------
    Tuple of 9 optional values:
        temporal_df, active_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map, conn_df
        Returns tuple of None values if loading fails
    """
    try:
        epochs = _load_epochs_for_subject(subject, task, deriv_root, config, logger)
        
        temporal_df, active_df, conn_df, y, info = _load_features_for_subject(
            subject, task, deriv_root, config, epochs
        )
        
        _, temp_series, Z_df_full, Z_df_temp = _load_aligned_events_and_covariates(
            epochs, subject, task, config, logger, partial_covars
        )
        
        roi_map = _build_roi_map(info, config)
        
        return (
            temporal_df,
            active_df,
            y,
            info,
            temp_series,
            Z_df_full,
            Z_df_temp,
            roi_map,
            conn_df,
        )
        
    except Exception as e:
        logger.error(f"Failed to load scatter data for sub-{subject}: {e}")
        return None, None, None, None, None, None, None, None, None
