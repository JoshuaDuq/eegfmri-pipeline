"""
Statistics Loading Utilities
============================

Functions for loading precomputed statistics and scatter data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple
import pandas as pd
import numpy as np

from ..config.loader import load_settings, ConfigDict
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.infra.paths import deriv_stats_path
from .covariates import _build_covariate_matrices


def _build_correlation_stats_candidates(
    feature_type: str,
    target_suffix: str,
    target_suffix_alt: Optional[str],
) -> List[str]:
    file_patterns = {
        "power": [
            f"corr_stats_pow_combined_vs_{target_suffix}.tsv",
            f"corr_stats_power_combined_vs_{target_suffix}.tsv",
            f"corr_stats_power_vs_{target_suffix}.tsv",
            f"corr_stats_power_vs_{target_suffix_alt}.tsv" if target_suffix_alt else None,
        ],
        "power_roi": [
            f"corr_stats_pow_roi_vs_{target_suffix.replace('temperature', 'temp')}.tsv",
            f"corr_stats_power_roi_vs_{target_suffix.replace('temperature', 'temp')}.tsv",
        ],
        "aperiodic": [
            f"corr_stats_aperiodic_vs_{target_suffix}.tsv",
            f"corr_stats_aperiodic_vs_{target_suffix_alt}.tsv" if target_suffix_alt else None,
        ],
        "connectivity": [
            f"corr_stats_connectivity_vs_{target_suffix}.tsv",
            f"corr_stats_connectivity_vs_{target_suffix_alt}.tsv" if target_suffix_alt else None,
        ],
        "itpc": [
            f"corr_stats_itpc_vs_{target_suffix}.tsv",
            f"corr_stats_itpc_vs_{target_suffix_alt}.tsv" if target_suffix_alt else None,
        ],
        "dynamics": [
            f"corr_stats_dynamics_vs_{target_suffix}.tsv",
            f"corr_stats_dynamics_vs_{target_suffix_alt}.tsv" if target_suffix_alt else None,
        ],
    }

    candidates = file_patterns.get(feature_type) or []
    candidates = [c for c in candidates if c]

    # Fallback: unified correlator naming convention corr_stats_<feature_type>_vs_<target>
    # This keeps plotting working as analysis outputs evolve.
    candidates.extend(
        [
            f"corr_stats_{feature_type}_vs_{target_suffix}.tsv",
            f"corr_stats_{feature_type}_vs_{target_suffix_alt}.tsv" if target_suffix_alt else None,
        ]
    )
    return [c for c in candidates if c]


def load_precomputed_correlations(
    stats_dir: Path,
    feature_type: str,
    target: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """
    Load precomputed correlation statistics from TSV files.
    
    Parameters
    ----------
    stats_dir : Path
        Directory containing correlation stats TSV files
    feature_type : str
        Type of feature: "power", "aperiodic", "connectivity", "itpc", "dynamics", "power_roi"
    target : str
        Correlation target: "rating" or "temperature"
    logger : Logger, optional
        Logger instance
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: feature/channel/roi, band, r, p, n, q_fdr_global, fdr_reject_global
        Returns None if file not found or empty
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    target_suffix = "rating" if "rating" in target.lower() else "temperature"
    target_suffix_alt = "temp" if target_suffix == "temperature" else None

    candidates = _build_correlation_stats_candidates(feature_type, target_suffix, target_suffix_alt)
    if not candidates:
        logger.warning(f"Unknown feature type for stats loading: {feature_type}")
        return None

    for fname in candidates:
        fpath = stats_dir / fname
        if not fpath.exists():
            continue
        df = read_tsv(fpath)
        if df is not None and not df.empty:
            return df

    return None


def get_precomputed_stats_for_roi_band(
    stats_df: Optional[pd.DataFrame],
    roi: str,
    band: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Extract precomputed stats for a specific ROI and band from a stats DataFrame.
    
    Returns dict with keys: r, p, n, ci_low, ci_high, fdr_reject
    """
    if stats_df is None or stats_df.empty:
        return {}

    # Filter by ROI/channel and band
    # Column names vary across analysis outputs.
    roi_col = next((c for c in ["roi", "channel", "feature"] if c in stats_df.columns), None)
    if not roi_col:
        return {}

    band_col = "band" if "band" in stats_df.columns else None
    if band_col is None:
        # Some stats tables may encode band in other columns or not at all.
        # For ROI scatter plots, band is required.
        return {}
        
    subset = stats_df[
        (stats_df[roi_col].astype(str) == roi)
        & (stats_df[band_col].astype(str) == band)
    ]
    
    if len(subset) == 0:
        return {}
        
    row = subset.iloc[0]

    # Accept both legacy and newer global FDR column names.
    fdr_reject = bool(
        row.get("fdr_reject_global", False)
        or row.get("fdr_reject", False)
    )
    q_val = row.get("q_fdr_global", row.get("q_global", row.get("q", np.nan)))
    return {
        "r": row.get("r", np.nan),
        "p": row.get("p", np.nan),
        "n": row.get("n", 0),
        "ci_low": row.get("ci_low", np.nan),
        "ci_high": row.get("ci_high", np.nan),
        "q": q_val,
        "fdr_reject": fdr_reject,
    }


def load_subject_scatter_data(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
    partial_covars: Optional[List[str]] = None,
) -> Tuple[
    Optional[pd.DataFrame],  # temporal_df
    Optional[pd.DataFrame],  # plateau_df (power)
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
    
    Returns 9-tuple:
        temporal_df, plateau_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map, conn_df
    """
    from .features_io import _load_features_and_targets
    from .epochs_loading import load_epochs_for_analysis
    from .alignment import get_aligned_events
    from .covariates import extract_temperature_data
    from ..analysis.tfr import build_rois_from_info
    
    try:
        epochs, _ = load_epochs_for_analysis(
            subject, task, 
            align="strict", 
            preload=False, 
            deriv_root=deriv_root,
            bids_root=config.bids_root,
            logger=logger, 
            config=config
        )
        
        temporal_df, plateau_df, conn_df, y, info = _load_features_and_targets(
            subject, task, deriv_root, config, epochs=epochs
        )
        
        aligned_events = get_aligned_events(
            epochs, subject, task, strict=True, 
            logger=logger, bids_root=config.bids_root, config=config
        )
        
        temp_series, temp_col = extract_temperature_data(aligned_events, config)
        Z_df_full, Z_df_temp = _build_covariate_matrices(aligned_events, partial_covars, temp_col, config)
        roi_map = build_rois_from_info(info, config)
        
        return temporal_df, plateau_df, y, info, temp_series, Z_df_full, Z_df_temp, roi_map, conn_df
        
    except Exception as e:
        logger.error(f"Failed to load scatter data for sub-{subject}: {e}")
        return None, None, None, None, None, None, None, None, None
