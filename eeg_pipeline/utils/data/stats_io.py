"""
Statistics I/O Utilities
========================

Functions for loading scatter data (epochs, features, covariates, ROI map).
I/O for statistical results is in `eeg_pipeline.utils.analysis.stats` and infra.tsv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple

import pandas as pd

from .covariates import _build_covariate_matrices


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
