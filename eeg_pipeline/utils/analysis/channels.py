"""
Channel Utilities
=================

Helpers for channel selection and manipulation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np


def pick_eeg_channels(epochs: mne.Epochs) -> Tuple[np.ndarray, List[str]]:
    """Pick EEG channels from epochs.

    Returns:
        Tuple of (channel indices array, channel names list).
    """
    picks = mne.pick_types(
        epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
    )
    channel_names = [epochs.info["ch_names"][idx] for idx in picks]
    return picks, channel_names


def get_eeg_data(
    epochs: mne.Epochs, logger: Any = None, context: str = ""
) -> Optional[Tuple[np.ndarray, List[str], np.ndarray]]:
    """Get EEG data with channel picking and validation.

    Args:
        epochs: MNE Epochs object containing EEG data.
        logger: Optional logger for warning messages.
        context: Optional context string for warning messages.

    Returns:
        Tuple of (data array, channel names, channel indices) if channels found,
        None otherwise.
    """
    picks, channel_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        if logger:
            logger.warning(f"{context}: No EEG channels available")
        return None

    data = epochs.get_data(picks=picks)
    return data, channel_names, picks


def match_channels_to_pattern(
    channel_names: List[str], patterns: List[str]
) -> List[int]:
    """Match channel names against a list of glob-like patterns.

    Args:
        channel_names: List of channel names to match.
        patterns: List of glob-like patterns (supports * and ? wildcards).

    Returns:
        List of indices for channels matching any pattern.
    """
    if not patterns:
        return []

    matched_indices = []
    for channel_idx, channel_name in enumerate(channel_names):
        for pattern in patterns:
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            if re.fullmatch(regex_pattern, channel_name, re.IGNORECASE):
                matched_indices.append(channel_idx)
                break
    return matched_indices


def build_roi_map(
    channel_names: List[str],
    roi_definitions: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    """Build mapping from ROI names to channel indices.

    Args:
        channel_names: List of channel names.
        roi_definitions: Dictionary mapping ROI names to pattern lists.

    Returns:
        Dictionary mapping ROI names to lists of channel indices.
    """
    roi_map: Dict[str, List[int]] = {}
    for roi_name, patterns in roi_definitions.items():
        indices = match_channels_to_pattern(channel_names, patterns)
        if indices:
            roi_map[roi_name] = indices
    return roi_map
