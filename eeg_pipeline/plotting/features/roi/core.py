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


def get_roi_definitions(config: Any) -> Dict[str, List[str]]:
    """Get ROI definitions from config.

    Returns dict mapping ROI name to list of regex patterns.
    """

    from eeg_pipeline.utils.config.loader import get_config_value

    rois = get_config_value(config, "rois", {})
    if not rois:
        rois = get_config_value(config, "time_frequency_analysis.rois", {})
    return rois


def get_roi_channels(roi_patterns: List[str], all_channels: List[str]) -> List[str]:
    """Match channels to ROI regex patterns."""

    matched = []
    for ch in all_channels:
        for pattern in roi_patterns:
            if re.match(pattern, ch):
                matched.append(ch)
                break
    return matched


def extract_channels_from_columns(columns: List[str]) -> List[str]:
    """Extract unique channel names from feature column names."""

    channels = set()
    for col in columns:
        match = re.search(r"_ch_([A-Za-z0-9]+)_", col)
        if match:
            channels.add(match.group(1))
        elif col.endswith(("_mean", "_logratio", "_lzc", "_pe")):
            match = re.search(r"_ch_([A-Za-z0-9]+)", col)
            if match:
                channels.add(match.group(1))
    return sorted(list(channels))


def aggregate_by_roi(
    features_df: pd.DataFrame,
    col_pattern: str,
    roi_channels: List[str],
) -> pd.Series:
    """Aggregate feature columns by ROI (mean across channels in ROI)."""

    cols = []
    for col in features_df.columns:
        if col_pattern in col:
            for ch in roi_channels:
                if f"_ch_{ch}_" in col or col.endswith(f"_ch_{ch}"):
                    cols.append(col)
                    break

    if not cols:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)

    return features_df[cols].mean(axis=1)


def _get_bands_and_palettes(config: Any) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Convenience accessor for bands and color palettes."""

    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    condition_colors = get_condition_colors(config)
    return bands, band_colors, condition_colors


def extract_channel_pairs_from_columns(columns: List[str]) -> List[Tuple[str, str]]:
    """Extract unique channel pairs from connectivity column names."""

    pairs = set()
    for col in columns:
        match = re.search(r"_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_", col)
        if match:
            pairs.add((match.group(1), match.group(2)))
    return list(pairs)


def aggregate_connectivity_by_roi(
    features_df: pd.DataFrame,
    col_pattern: str,
    roi_channels: List[str],
) -> pd.Series:
    """Aggregate connectivity within ROI (mean of edges where both channels in ROI)."""

    cols = []
    for col in features_df.columns:
        if col_pattern in col:
            match = re.search(r"_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_", col)
            if match:
                ch1, ch2 = match.group(1), match.group(2)
                if ch1 in roi_channels and ch2 in roi_channels:
                    cols.append(col)

    if not cols:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)

    return features_df[cols].mean(axis=1)


__all__ = [
    "aggregate_by_roi",
    "aggregate_connectivity_by_roi",
    "extract_channel_pairs_from_columns",
    "extract_channels_from_columns",
    "get_roi_channels",
    "get_roi_definitions",
]
