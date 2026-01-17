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

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from eeg_pipeline.utils.analysis.channels import build_roi_map


_SPATIAL_MODE_ROI = "roi"
_SPATIAL_MODE_CHANNELS = "channels"
_SPATIAL_MODE_GLOBAL = "global"
_AGGREGATION_MEAN = "mean"
_AGGREGATION_MEDIAN = "median"


def get_roi_definitions(config: Any) -> Dict[str, List[str]]:
    """Get ROI definitions from config."""
    from eeg_pipeline.utils.config.loader import get_config_value
    
    return get_config_value(config, "rois", {})


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
