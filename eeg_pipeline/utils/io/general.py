"""
Backward compatibility module for utils.io.general.

This module re-exports functions from the newly organized submodules
to maintain backward compatibility with existing code.

The code has been split into logical modules:
- paths: Path utilities for BIDS and derivatives
- logging: Logging configuration and utilities
- tsv: TSV file I/O operations
- columns: Column finding utilities
- formatting: String formatting utilities
- plotting: Plotting and visualization utilities
- validation: Data validation utilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import logging
import json
import numpy as np
import pandas as pd
import mne

try:
    from ..config.loader import ConfigDict
except ImportError:
    ConfigDict = dict

EEGConfig = ConfigDict

# Import from new modules
from .paths import (
    ensure_dir,
    find_first,
    bids_sub_eeg_path,
    bids_events_path,
    deriv_sub_eeg_path,
    deriv_features_path,
    deriv_stats_path,
    deriv_plots_path,
    deriv_group_eeg_path,
    deriv_group_stats_path,
    deriv_group_plots_path,
    find_connectivity_features_path,
    _find_clean_epochs_path,
    _load_events_df,
    extract_subject_id_from_path,
    _resolve_deriv_root,
    _resolve_bids_root,
)

from .logging import (
    get_logger,
    get_module_logger,
    log_and_raise_error,
    reset_logging,
    get_subject_logger,
    get_group_logger,
    get_pipeline_logger,
    setup_logger,
    get_default_logger,
)

from .tsv import (
    read_tsv,
    write_tsv,
)

from .columns import (
    find_column_in_events,
    find_pain_column_in_events,
    find_temperature_column_in_events,
    find_column_in_metadata,
    find_pain_column_in_metadata,
    find_temperature_column_in_metadata,
    get_column_from_config,
    get_pain_column_from_config,
    get_temperature_column_from_config,
    _pick_target_column,
)

from .formatting import (
    sanitize_label,
    format_baseline_window_string,
    format_baseline_string,
    parse_analysis_type_from_filename,
    parse_target_from_filename,
    parse_measure_band_from_filename,
    format_band_range,
    build_partial_covars_string,
    format_band_label,
    get_correlation_type_labels,
    format_temperature_label,
    write_group_trial_counts,
    format_channel_list_for_display,
    format_roi_description,
    get_residual_labels,
    get_target_labels,
    get_temporal_xlabel,
    format_time_suffix,
)

from .plotting import (
    SaveFigConfig,
    PlotConfig,
    _get_plot_constants,
    build_footer,
    unwrap_figure,
    get_behavior_footer,
    get_band_color,
    logratio_to_pct,
    pct_to_logratio,
    get_viz_params,
    plot_topomap_on_ax,
    robust_sym_vlim,
    setup_matplotlib,
    extract_plotting_constants,
    extract_eeg_picks,
    log_if_present,
    validate_picks,
    get_default_config,
    save_fig,
)

from ..validation import (
    validate_epochs_for_plotting,
    require_epochs_tfr,
    detect_data_format,
    validate_predictor_file,
    ensure_aligned_lengths,
    _handle_alignment_error,
)


###################################################################
# Remaining utilities that don't fit cleanly into other modules
###################################################################

@dataclass
class EEGDatasetResult:
    epochs: Optional[mne.Epochs]
    events: Optional[pd.DataFrame]
    
    @property
    def empty(self) -> bool:
        return self.epochs is None or self.events is None or len(self.events) == 0


def reconstruct_kept_indices(dropped_trials_path: Path, n_events: int) -> np.ndarray:
    if not dropped_trials_path.exists():
        return np.arange(n_events)
    
    dropped_df = pd.read_csv(dropped_trials_path, sep="\t")
    if "original_index" not in dropped_df.columns or len(dropped_df) == 0:
        return np.arange(n_events)
    
    dropped_indices_raw = pd.to_numeric(dropped_df["original_index"], errors="coerce").dropna()
    if len(dropped_indices_raw) == 0:
        return np.arange(n_events)
    
    dropped_indices = set(dropped_indices_raw.astype(int).tolist())
    kept_indices = np.array([i for i in range(n_events) if i not in dropped_indices])
    return kept_indices


def get_pain_window(constants=None, config: Optional[EEGConfig] = None) -> Tuple[float, float]:
    if config is not None:
        plateau_window = config.get("time_frequency_analysis.plateau_window")
        return tuple(plateau_window)
    
    if constants is None:
        raise ValueError("Either constants or config must be provided to get_pain_window")
    
    if "PLATEAU_WINDOW" not in constants:
        raise KeyError(
            "PLATEAU_WINDOW not found in constants. "
            "Use PLATEAU_WINDOW (tuple) not PLATEAU_END (float)"
        )
    
    return constants["PLATEAU_WINDOW"]


WINDOW_PAIN = get_pain_window


def ensure_derivatives_dataset_description(deriv_root: Optional[Path] = None, constants=None, config=None) -> None:
    root = _resolve_deriv_root(deriv_root, config, constants)
    
    desc_path = root / "dataset_description.json"
    if desc_path.exists():
        return
    
    meta = {
        "Name": "EEG Pipeline Derivatives",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "EEG_fMRI_Analysis Pipeline",
                "Version": "unknown",
                "Description": "Custom EEG analysis (ERP, TFR, features, decoding)",
            }
        ],
    }
    ensure_dir(root)
    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


###################################################################
# File Building Utilities
###################################################################

def build_file_updates_dict(
    file_references: List[Tuple[Path, int]],
    q_array: np.ndarray,
    rejections_array: np.ndarray,
    p_array: np.ndarray,
) -> Dict[Path, List[Tuple[int, float, bool, float]]]:
    from ..analysis.stats import _safe_float
    
    file_updates: Dict[Path, List[Tuple[int, float, bool, float]]] = {}
    
    for index, (file_path, row_index) in enumerate(file_references):
        update_item = (
            row_index,
            _safe_float(q_array[index]),
            bool(rejections_array[index]),
            _safe_float(p_array[index]),
        )
        file_updates.setdefault(file_path, []).append(update_item)
    
    return file_updates


def build_predictor_column_mapping(predictor_type: str) -> Dict[str, str]:
    base_cols = {
        "predictor": "predictor",
        "band": "band",
        "r": "r",
        "p": "p",
        "n": "n",
        "predictor_type": "type",
        "target": "target",
    }
    region_col = "roi" if "roi" in str(predictor_type).lower() else "channel"
    base_cols[region_col] = "region"
    return base_cols


def build_predictor_name(df: pd.DataFrame, predictor_type: str) -> pd.Series:
    region_col = "roi" if "roi" in str(predictor_type).lower() else "channel"
    if region_col not in df.columns:
        # Fallback: use whichever region column exists
        region_col = "roi" if "roi" in df.columns else "channel" if "channel" in df.columns else df.columns[0]
    return df[region_col].astype(str) + " (" + df["band"].astype(str) + ")"


def build_connectivity_heatmap_records(
    n_nodes: int,
    node_names: List[str],
    correlation_matrix: np.ndarray,
    p_value_matrix: np.ndarray,
    rejection_map: Dict[Tuple[int, int], bool],
    critical_value: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            pair_key = (i, j)
            records.append({
                "node_i": node_names[i],
                "node_j": node_names[j],
                "r": correlation_matrix[i, j],
                "p": p_value_matrix[i, j],
                "fdr_reject": rejection_map.get(pair_key, False),
                "fdr_crit_p": critical_value,
            })
    return records


def _get_df_value(df: pd.DataFrame, col: str, row_idx: int, default: str = "") -> str:
    return df.get(col, pd.Series([default] * len(df))).iloc[row_idx]


def build_meta_for_row(
    df: pd.DataFrame,
    row_idx: int,
    filename: str,
    analysis_type: str,
    target: str,
    measure_band: str,
    p_source: str,
) -> Dict[str, Any]:
    meta = {
        "source_file": filename,
        "analysis_type": analysis_type,
        "target": target,
        "measure_band": measure_band,
        "row_index": int(row_idx),
    }
    if p_source:
        meta["p_used_source"] = p_source

    try:
        if analysis_type == "pow_roi":
            roi = _get_df_value(df, "roi", row_idx)
            band = _get_df_value(df, "band", row_idx)
            meta.update({"roi": roi, "band": band})
            meta["test_label"] = f"pow_{band}_ROI {roi} vs {target}"
        elif analysis_type == "conn_roi_summary":
            roi_i = _get_df_value(df, "roi_i", row_idx)
            roi_j = _get_df_value(df, "roi_j", row_idx)
            meta.update({"roi_i": roi_i, "roi_j": roi_j})
            meta["test_label"] = f"conn_{measure_band}_ROI {roi_i}-{roi_j} vs {target}"
        elif analysis_type == "pow_channel":
            channel = _get_df_value(df, "channel", row_idx)
            band = _get_df_value(df, "band", row_idx)
            meta.update({"channel": channel, "band": band})
            meta["test_label"] = f"pow_{band}_Channel {channel} vs {target}"
        elif analysis_type == "conn_edges":
            node_i = _get_df_value(df, "node_i", row_idx)
            node_j = _get_df_value(df, "node_j", row_idx)
            meta.update({"node_i": node_i, "node_j": node_j})
            meta["test_label"] = f"conn_{measure_band}_Edge {node_i}-{node_j} vs {target}"
    except (KeyError, IndexError):
        pass
    
    return meta


__all__ = [
    # Path utilities
    "_find_clean_epochs_path",
    "_load_events_df",
    "_pick_target_column",
    "EEGDatasetResult",
    "PlotConfig",
    "ensure_derivatives_dataset_description",
    "ensure_dir",
    "ensure_aligned_lengths",
    "bids_sub_eeg_path",
    "bids_events_path",
    "deriv_sub_eeg_path",
    "deriv_features_path",
    "deriv_stats_path",
    "deriv_plots_path",
    "deriv_group_eeg_path",
    "deriv_group_stats_path",
    "deriv_group_plots_path",
    "find_connectivity_features_path",
    # Plotting utilities
    "save_fig",
    # Logging utilities
    "get_pipeline_logger",
    "get_subject_logger",
    "get_group_logger",
    "setup_logger",
    # Remaining utilities
    "reconstruct_kept_indices",
    "WINDOW_PAIN",
    "_get_plot_constants",
    "build_footer",
    "unwrap_figure",
    "sanitize_label",
    "get_viz_params",
    "plot_topomap_on_ax",
    "robust_sym_vlim",
    "get_behavior_footer",
    "get_band_color",
    "logratio_to_pct",
    "pct_to_logratio",
    "setup_matplotlib",
    # Column finding
    "find_column_in_events",
    "find_pain_column_in_events",
    "find_temperature_column_in_events",
    "find_column_in_metadata",
    "find_pain_column_in_metadata",
    "find_temperature_column_in_metadata",
    "get_column_from_config",
    "get_pain_column_from_config",
    "get_temperature_column_from_config",
    # Plotting constants
    "extract_plotting_constants",
    "extract_eeg_picks",
    "format_baseline_string",
    "log_if_present",
    "validate_picks",
    "get_default_logger",
    "get_default_config",
    # Formatting utilities
    "parse_analysis_type_from_filename",
    "parse_target_from_filename",
    "parse_measure_band_from_filename",
    "build_partial_covars_string",
    "format_band_label",
    "format_band_range",
    "get_correlation_type_labels",
    "format_channel_list_for_display",
    "extract_subject_id_from_path",
    "write_group_trial_counts",
    "format_temperature_label",
    "format_roi_description",
    "get_residual_labels",
    "get_target_labels",
    "get_temporal_xlabel",
    "format_time_suffix",
    # Validation utilities
    "validate_epochs_for_plotting",
    "require_epochs_tfr",
    "detect_data_format",
    "format_baseline_window_string",
    # File building utilities
    "validate_predictor_file",
    "build_file_updates_dict",
    "build_predictor_column_mapping",
    "build_predictor_name",
    "build_connectivity_heatmap_records",
    "build_meta_for_row",
]
