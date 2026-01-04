"""
Spatial Aggregation Utilities
=============================

Functions for aggregating features across spatial modes:
- ROI: Aggregate by regions of interest
- Channels: Per-channel features (no aggregation)
- Global: Mean across all channels

Aggregation methods:
- mean: Arithmetic mean (default)
- median: Median (robust to outliers)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.channels import build_roi_map


def get_roi_definitions(config: Any) -> Dict[str, List[str]]:
    """Get ROI definitions from config."""
    from eeg_pipeline.utils.config.loader import get_config_value
    
    # Try to get from project-specific section first
    rois = get_config_value(config, "project.roi_definitions", None)
    if rois:
        return rois
        
    # Fallback to general spatial section
    rois = get_config_value(config, "spatial.roi_definitions", {})
    if rois:
        return rois

    # Legacy fallbacks
    rois = get_config_value(config, "rois", {})
    if not rois:
        rois = get_config_value(config, "time_frequency_analysis.rois", {})
    return rois


def _aggregate_func(method: str = "mean"):
    """Return the appropriate aggregation function."""
    if method == "median":
        return np.nanmedian
    return np.nanmean


def aggregate_by_spatial_modes(
    data: np.ndarray,
    ch_names: List[str],
    spatial_modes: List[str],
    config: Any,
    feature_prefix: str = "",
    aggregation_method: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Aggregate data according to spatial modes.
    
    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_epochs, n_channels) or (n_epochs, n_channels, n_times)
    ch_names : list of str
        Channel names
    spatial_modes : list of str
        Spatial modes to compute: 'roi', 'channels', 'global'
    config : Any
        Configuration object
    feature_prefix : str
        Prefix for feature names
    aggregation_method : str
        'mean' or 'median'
        
    Returns
    -------
    dict mapping feature names to arrays of shape (n_epochs,) or (n_epochs, n_times)
    """
    results: Dict[str, np.ndarray] = {}
    agg_func = _aggregate_func(aggregation_method)
    
    # Handle different data shapes
    has_time = data.ndim == 3
    
    if "global" in spatial_modes:
        # Aggregate across all channels
        if has_time:
            global_agg = agg_func(data, axis=1)  # (n_epochs, n_times)
        else:
            global_agg = agg_func(data, axis=1)  # (n_epochs,)
        name = f"{feature_prefix}_global" if feature_prefix else "global"
        results[name] = global_agg
    
    if "roi" in spatial_modes:
        roi_defs = get_roi_definitions(config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)
            for roi_name, ch_indices in roi_map.items():
                if len(ch_indices) > 0:
                    if has_time:
                        roi_agg = agg_func(data[:, ch_indices, :], axis=1)
                    else:
                        roi_agg = agg_func(data[:, ch_indices], axis=1)
                    name = f"{feature_prefix}_{roi_name}" if feature_prefix else roi_name
                    results[name] = roi_agg
    
    if "channels" in spatial_modes:
        for ch_idx, ch_name in enumerate(ch_names):
            if has_time:
                ch_data = data[:, ch_idx, :]
            else:
                ch_data = data[:, ch_idx]
            name = f"{feature_prefix}_{ch_name}" if feature_prefix else ch_name
            results[name] = ch_data
    
    return results


def apply_spatial_aggregation(
    data: np.ndarray,
    ch_names: List[str],
    ctx: Any,
    feature_prefix: str,
    roi_map: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Centralized helper to apply spatial aggregation from context.
    
    Parameters
    ----------
    data : np.ndarray
        Data of shape (n_epochs, n_channels) - already processed (e.g., log-ratio normalized)
    ch_names : list of str
        Channel names
    ctx : FeatureContext
        Context with spatial_modes and config
    feature_prefix : str
        Prefix for feature naming
    roi_map : dict, optional
        Pre-built ROI map (for efficiency when calling multiple times)
    
    Returns
    -------
    dict of feature_name -> values
    """
    spatial_modes = getattr(ctx, 'spatial_modes', ['roi', 'global'])
    aggregation_method = getattr(ctx, 'aggregation_method', 'mean')
    agg_func = _aggregate_func(aggregation_method)
    
    # Build ROI map if not provided and needed
    if roi_map is None and 'roi' in spatial_modes:
        roi_defs = get_roi_definitions(ctx.config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)
        else:
            roi_map = {}
    
    results: Dict[str, np.ndarray] = {}
    
    # Per-channel (only if 'channels' in spatial_modes)
    if 'channels' in spatial_modes:
        for ch_idx, ch_name in enumerate(ch_names):
            results[f"{feature_prefix}_{ch_name}"] = data[:, ch_idx]
    
    # Global aggregation (only if 'global' in spatial_modes)
    if 'global' in spatial_modes:
        results[f"{feature_prefix}_global"] = agg_func(data, axis=1)
    
    # ROI aggregation (only if 'roi' in spatial_modes and roi_map exists)
    if 'roi' in spatial_modes and roi_map:
        for roi_name, ch_indices in roi_map.items():
            if len(ch_indices) > 0:
                results[f"{feature_prefix}_{roi_name}"] = agg_func(data[:, ch_indices], axis=1)
    
    return results


def aggregate_features_df(
    feature_df: pd.DataFrame,
    ch_names: List[str],
    spatial_modes: List[str],
    config: Any,
    base_feature_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate a feature DataFrame that has per-channel columns.
    
    Expects columns named like: {base_feature_name}_{channel}
    
    Returns aggregated DataFrame and list of new column names.
    """
    results_dict: Dict[str, np.ndarray] = {}
    n_rows = len(feature_df)
    
    # Extract per-channel data
    ch_data = {}
    for ch_name in ch_names:
        col_pattern = f"{base_feature_name}_{ch_name}"
        matching_cols = [c for c in feature_df.columns if c == col_pattern or c.startswith(f"{col_pattern}_")]
        if matching_cols:
            ch_data[ch_name] = feature_df[matching_cols[0]].values
    
    if not ch_data:
        return pd.DataFrame(), []
    
    # Stack into array
    ch_list = list(ch_data.keys())
    data_array = np.column_stack([ch_data[ch] for ch in ch_list])  # (n_epochs, n_channels)
    
    aggregated = aggregate_by_spatial_modes(
        data_array,
        ch_list,
        spatial_modes,
        config,
        feature_prefix=base_feature_name,
    )
    
    for name, values in aggregated.items():
        results_dict[name] = values
    
    result_df = pd.DataFrame(results_dict)
    return result_df, list(result_df.columns)


def generate_output_filename(
    base_name: str,
    spatial_modes: List[str],
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> str:
    """
    Generate config-specific output filename.
    
    Examples:
        features_power_roi_global.tsv
        features_power_roi_t0.0-10.0.tsv
    """
    parts = [base_name]
    
    if spatial_modes:
        spatial_str = "_".join(sorted(spatial_modes))
        parts.append(spatial_str)
    
    if tmin is not None or tmax is not None:
        tmin_str = f"{tmin:.1f}" if tmin is not None else "None"
        tmax_str = f"{tmax:.1f}" if tmax is not None else "None"
        parts.append(f"t{tmin_str}-{tmax_str}")
    
    return "_".join(parts)


def crop_epochs_to_time_range(
    epochs: Any,
    tmin: Optional[float],
    tmax: Optional[float],
    logger: Any = None,
) -> Any:
    """
    Crop epochs to specified time range.
    
    Returns cropped epochs (copy) or original if no cropping needed.
    """
    if tmin is None and tmax is None:
        return epochs
    
    actual_tmin = epochs.tmin
    actual_tmax = epochs.tmax
    
    crop_tmin = tmin if tmin is not None else actual_tmin
    crop_tmax = tmax if tmax is not None else actual_tmax
    
    # Swap if needed
    if crop_tmin > crop_tmax:
        if logger:
            logger.warning(f"Time range [{crop_tmin}, {crop_tmax}] is reversed; swapping values.")
        crop_tmin, crop_tmax = crop_tmax, crop_tmin

    # Clamp to available range
    crop_tmin = max(crop_tmin, actual_tmin)
    crop_tmax = min(crop_tmax, actual_tmax)
    
    if crop_tmin >= crop_tmax:
        if logger:
            logger.warning(f"Invalid time range [{crop_tmin}, {crop_tmax}], using full range")
        return epochs
    
    if logger:
        logger.info(f"Cropping epochs to time range [{crop_tmin:.2f}, {crop_tmax:.2f}] s")
    
    # MNE requires data to be loaded for cropping if we want to avoid errors
    if not epochs.preload:
        epochs.load_data()
        
    return epochs.copy().crop(tmin=crop_tmin, tmax=crop_tmax)
