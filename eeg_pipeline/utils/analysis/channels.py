"""
Channel Utilities
=================

Helpers for channel selection and manipulation.
"""

from __future__ import annotations

from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import mne


def pick_eeg_channels(epochs: mne.Epochs) -> Tuple[np.ndarray, List[str]]:
    """Pick EEG channels from epochs. Returns (picks array, channel names list)."""
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    return picks, ch_names


def get_eeg_data(epochs: mne.Epochs, logger: Any = None, context: str = "") -> Optional[Tuple[np.ndarray, List[str], np.ndarray]]:
    """Get EEG data with channel picking and validation. Returns None if no channels."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        if logger:
            logger.warning(f"{context}: No EEG channels available")
        return None
    
    data = epochs.get_data(picks=picks)
    return data, ch_names, picks


def match_channels_to_pattern(ch_names: List[str], patterns: List[str]) -> List[int]:
    """Match channel names against a list of glob-like patterns."""
    import re
    matched = []
    for idx, ch in enumerate(ch_names):
        for pat in patterns:
            # Convert glob to regex
            regex = pat.replace("*", ".*").replace("?", ".")
            if re.fullmatch(regex, ch, re.IGNORECASE):
                matched.append(idx)
                break
    return matched


def build_roi_map(
    ch_names: List[str],
    roi_definitions: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    """Build mapping from ROI names to channel indices."""
    roi_map: Dict[str, List[int]] = {}
    for roi_name, patterns in roi_definitions.items():
        indices = match_channels_to_pattern(ch_names, patterns)
        if indices:
            roi_map[roi_name] = indices
    return roi_map
