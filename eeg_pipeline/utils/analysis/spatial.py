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

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.channels import build_roi_map


_SPATIAL_MODE_ROI = "roi"
_SPATIAL_MODE_CHANNELS = "channels"
_SPATIAL_MODE_GLOBAL = "global"
_AGGREGATION_MEAN = "mean"
_AGGREGATION_MEDIAN = "median"


def get_roi_definitions(config: Any) -> Dict[str, List[str]]:
    """Get ROI definitions from config with fallback hierarchy."""
    from eeg_pipeline.utils.config.loader import get_config_value
    
    config_paths = [
        "project.roi_definitions",
        "spatial.roi_definitions",
        "rois",
        "time_frequency_analysis.rois",
    ]
    
    for path in config_paths:
        rois = get_config_value(config, path, None)
        if rois:
            return rois
    
    return {}


def _get_aggregation_function(method: str) -> Callable:
    """Return aggregation function for given method."""
    if method == _AGGREGATION_MEDIAN:
        return np.nanmedian
    return np.nanmean


def _aggregate_channels(
    data: np.ndarray,
    channel_indices: np.ndarray,
    aggregation_func: Callable,
    has_time_dimension: bool,
) -> np.ndarray:
    """Aggregate data across specified channel indices."""
    if has_time_dimension:
        return aggregation_func(data[:, channel_indices, :], axis=1)
    return aggregation_func(data[:, channel_indices], axis=1)


def _extract_channel_data(
    data: np.ndarray,
    channel_index: int,
    has_time_dimension: bool,
) -> np.ndarray:
    """Extract data for a single channel."""
    if has_time_dimension:
        return data[:, channel_index, :]
    return data[:, channel_index]


def _build_feature_name(prefix: str, suffix: str) -> str:
    """Build feature name from prefix and suffix."""
    if prefix:
        return f"{prefix}_{suffix}"
    return suffix


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
    if data.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D data, got {data.ndim}D array with shape {data.shape}"
        )
    if len(ch_names) != data.shape[1]:
        raise ValueError(
            f"Channel count mismatch: {len(ch_names)} names for {data.shape[1]} channels"
        )
    
    results: Dict[str, np.ndarray] = {}
    aggregation_func = _get_aggregation_function(aggregation_method)
    has_time_dimension = data.ndim == 3
    
    if _SPATIAL_MODE_GLOBAL in spatial_modes:
        global_data = _aggregate_channels(
            data, np.arange(data.shape[1]), aggregation_func, has_time_dimension
        )
        feature_name = _build_feature_name(feature_prefix, _SPATIAL_MODE_GLOBAL)
        results[feature_name] = global_data
    
    if _SPATIAL_MODE_ROI in spatial_modes:
        roi_definitions = get_roi_definitions(config)
        if roi_definitions:
            roi_map = build_roi_map(ch_names, roi_definitions)
            for roi_name, channel_indices in roi_map.items():
                if len(channel_indices) > 0:
                    roi_data = _aggregate_channels(
                        data, np.array(channel_indices), aggregation_func, has_time_dimension
                    )
                    feature_name = _build_feature_name(feature_prefix, roi_name)
                    results[feature_name] = roi_data
    
    if _SPATIAL_MODE_CHANNELS in spatial_modes:
        for channel_idx, channel_name in enumerate(ch_names):
            channel_data = _extract_channel_data(data, channel_idx, has_time_dimension)
            feature_name = _build_feature_name(feature_prefix, channel_name)
            results[feature_name] = channel_data
    
    return results


def apply_spatial_aggregation(
    data: np.ndarray,
    ch_names: List[str],
    ctx: Any,
    feature_prefix: str,
    roi_map: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply spatial aggregation from FeatureContext.
    
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
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D data (n_epochs, n_channels), got {data.ndim}D array with shape {data.shape}"
        )
    if len(ch_names) != data.shape[1]:
        raise ValueError(
            f"Channel count mismatch: {len(ch_names)} names for {data.shape[1]} channels"
        )
    
    spatial_modes = getattr(ctx, "spatial_modes", [_SPATIAL_MODE_ROI, _SPATIAL_MODE_GLOBAL])
    aggregation_method = getattr(ctx, "aggregation_method", _AGGREGATION_MEAN)
    aggregation_func = _get_aggregation_function(aggregation_method)
    
    if roi_map is None and _SPATIAL_MODE_ROI in spatial_modes:
        roi_definitions = get_roi_definitions(ctx.config)
        roi_map = build_roi_map(ch_names, roi_definitions) if roi_definitions else {}
    
    results: Dict[str, np.ndarray] = {}
    
    if _SPATIAL_MODE_CHANNELS in spatial_modes:
        for channel_idx, channel_name in enumerate(ch_names):
            feature_name = _build_feature_name(feature_prefix, channel_name)
            results[feature_name] = data[:, channel_idx]
    
    if _SPATIAL_MODE_GLOBAL in spatial_modes:
        feature_name = _build_feature_name(feature_prefix, _SPATIAL_MODE_GLOBAL)
        results[feature_name] = aggregation_func(data, axis=1)
    
    if _SPATIAL_MODE_ROI in spatial_modes and roi_map:
        for roi_name, channel_indices in roi_map.items():
            if len(channel_indices) > 0:
                feature_name = _build_feature_name(feature_prefix, roi_name)
                results[feature_name] = aggregation_func(data[:, channel_indices], axis=1)
    
    return results


def _find_channel_column(
    feature_df: pd.DataFrame,
    base_feature_name: str,
    channel_name: str,
) -> Optional[str]:
    """Find column name matching pattern {base_feature_name}_{channel_name} or variants."""
    exact_pattern = f"{base_feature_name}_{channel_name}"
    prefix_pattern = f"{exact_pattern}_"
    
    for column in feature_df.columns:
        column_str = str(column)
        if column_str == exact_pattern or column_str.startswith(prefix_pattern):
            return column_str
    return None


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
    if feature_df.empty:
        return pd.DataFrame(), []
    
    channel_data: Dict[str, np.ndarray] = {}
    found_channel_names: List[str] = []
    
    for channel_name in ch_names:
        column_name = _find_channel_column(feature_df, base_feature_name, channel_name)
        if column_name is not None:
            channel_data[channel_name] = feature_df[column_name].values
            found_channel_names.append(channel_name)
    
    if not channel_data:
        return pd.DataFrame(), []
    
    data_array = np.column_stack([channel_data[ch] for ch in found_channel_names])
    
    aggregated = aggregate_by_spatial_modes(
        data_array,
        found_channel_names,
        spatial_modes,
        config,
        feature_prefix=base_feature_name,
    )
    
    result_df = pd.DataFrame(aggregated)
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
        tmin_formatted = f"{tmin:.1f}" if tmin is not None else "None"
        tmax_formatted = f"{tmax:.1f}" if tmax is not None else "None"
        time_range_str = f"t{tmin_formatted}-{tmax_formatted}"
        parts.append(time_range_str)
    
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
    
    available_tmin = epochs.tmin
    available_tmax = epochs.tmax
    
    requested_tmin = tmin if tmin is not None else available_tmin
    requested_tmax = tmax if tmax is not None else available_tmax
    
    if requested_tmin > requested_tmax:
        if logger:
            logger.warning(
                f"Time range [{requested_tmin}, {requested_tmax}] is reversed; swapping values."
            )
        requested_tmin, requested_tmax = requested_tmax, requested_tmin

    clamped_tmin = max(requested_tmin, available_tmin)
    clamped_tmax = min(requested_tmax, available_tmax)
    
    if clamped_tmin >= clamped_tmax:
        if logger:
            logger.warning(
                f"Invalid time range [{clamped_tmin}, {clamped_tmax}], using full range"
            )
        return epochs
    
    if logger:
        logger.info(f"Cropping epochs to time range [{clamped_tmin:.2f}, {clamped_tmax:.2f}] s")
    
    if not epochs.preload:
        epochs.load_data()
        
    return epochs.copy().crop(tmin=clamped_tmin, tmax=clamped_tmax)
