from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.plotting.features.utils import (
    get_band_colors,
    get_band_names,
    get_condition_colors,
)

# Regex patterns for extracting channel information from column names
_CHANNEL_PATTERN_WITH_SEPARATOR = re.compile(r"_ch_([A-Za-z0-9]+)_")
_CHANNEL_PATTERN_END_OF_COLUMN = re.compile(r"_ch_([A-Za-z0-9]+)")
_CHANNEL_PAIR_PATTERN = re.compile(r"_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_")

# Column name suffixes that indicate channel-specific features
_CHANNEL_FEATURE_SUFFIXES = ("_mean", "_logratio", "_lzc", "_pe")


def get_roi_definitions(config: Any) -> Dict[str, List[str]]:
    """Get ROI definitions from config.

    Returns:
        Dict mapping ROI name to list of regex patterns for matching channels.
        Returns empty dict if no ROI definitions found.
    """
    from eeg_pipeline.utils.config.loader import get_config_value

    return get_config_value(config, "rois", {})


def get_roi_channels(roi_patterns: List[str], all_channels: List[str]) -> List[str]:
    """Match channels to ROI regex patterns.

    Args:
        roi_patterns: List of regex patterns for matching channel names.
        all_channels: List of all available channel names.

    Returns:
        List of channel names that match any of the ROI patterns.
    """
    if not roi_patterns or not all_channels:
        return []

    matched_channels = []
    for channel in all_channels:
        for pattern in roi_patterns:
            if re.match(pattern, channel):
                matched_channels.append(channel)
                break
    return matched_channels


def extract_channels_from_columns(columns: List[str]) -> List[str]:
    """Extract unique channel names from feature column names.

    Handles two column naming conventions:
    - Columns with channel info in middle: `feature_ch_CHANNEL_...`
    - Columns with channel info at end: `feature_ch_CHANNEL` (with specific suffixes)

    Args:
        columns: List of feature column names.

    Returns:
        Sorted list of unique channel names found in columns.
    """
    if not columns:
        return []

    channels = set()
    for column in columns:
        match = _CHANNEL_PATTERN_WITH_SEPARATOR.search(column)
        if match:
            channels.add(match.group(1))
        elif column.endswith(_CHANNEL_FEATURE_SUFFIXES):
            match = _CHANNEL_PATTERN_END_OF_COLUMN.search(column)
            if match:
                channels.add(match.group(1))
    return sorted(list(channels))


def aggregate_by_roi(
    features_df: pd.DataFrame,
    col_pattern: str,
    roi_channels: List[str],
) -> pd.Series:
    """Aggregate feature columns by ROI (mean across channels in ROI).

    Args:
        features_df: DataFrame with feature columns.
        col_pattern: Pattern string that must appear in column names to be included.
        roi_channels: List of channel names belonging to the ROI.

    Returns:
        Series with mean values across ROI channels for each row.
        Returns Series of NaNs if no matching columns found.
    """
    if not roi_channels:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)

    matching_columns = []
    for column in features_df.columns:
        if col_pattern not in column:
            continue

        for channel in roi_channels:
            channel_marker = f"_ch_{channel}_"
            channel_marker_end = f"_ch_{channel}"
            if channel_marker in column or column.endswith(channel_marker_end):
                matching_columns.append(column)
                break

    if not matching_columns:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)

    return features_df[matching_columns].mean(axis=1)


def _get_bands_and_palettes(config: Any) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Get bands and color palettes from config.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (band_names, band_colors, condition_colors).
    """
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    condition_colors = get_condition_colors(config)
    return bands, band_colors, condition_colors


def extract_channel_pairs_from_columns(columns: List[str]) -> List[Tuple[str, str]]:
    """Extract unique channel pairs from connectivity column names.

    Args:
        columns: List of connectivity feature column names.

    Returns:
        List of (channel1, channel2) tuples found in columns.
    """
    if not columns:
        return []

    pairs = set()
    for column in columns:
        match = _CHANNEL_PAIR_PATTERN.search(column)
        if match:
            channel1, channel2 = match.group(1), match.group(2)
            pairs.add((channel1, channel2))
    return list(pairs)


def aggregate_connectivity_by_roi(
    features_df: pd.DataFrame,
    col_pattern: str,
    roi_channels: List[str],
) -> pd.Series:
    """Aggregate connectivity within ROI (mean of edges where both channels in ROI).

    Args:
        features_df: DataFrame with connectivity feature columns.
        col_pattern: Pattern string that must appear in column names to be included.
        roi_channels: List of channel names belonging to the ROI.

    Returns:
        Series with mean connectivity values for edges within ROI.
        Returns Series of NaNs if no matching columns found.
    """
    if not roi_channels:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)

    roi_channels_set = set(roi_channels)
    matching_columns = []
    for column in features_df.columns:
        if col_pattern not in column:
            continue

        match = _CHANNEL_PAIR_PATTERN.search(column)
        if match:
            channel1, channel2 = match.group(1), match.group(2)
            if channel1 in roi_channels_set and channel2 in roi_channels_set:
                matching_columns.append(column)

    if not matching_columns:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)

    return features_df[matching_columns].mean(axis=1)


__all__ = [
    "aggregate_by_roi",
    "aggregate_connectivity_by_roi",
    "extract_channel_pairs_from_columns",
    "extract_channels_from_columns",
    "get_roi_channels",
    "get_roi_definitions",
]