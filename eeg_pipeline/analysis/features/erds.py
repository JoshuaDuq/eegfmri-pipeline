from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.tfr import time_mask, freq_mask
from eeg_pipeline.analysis.features.core import (
    compute_band_envelope_fast,
    EPSILON_STD,
)


###################################################################
# Event-Related Desynchronization/Synchronization (ERD/ERS)
###################################################################
# ERD/ERS strongly correlates with BOLD signal changes:
# - Alpha/beta ERD → increased BOLD in sensorimotor cortex
# - Gamma ERS → increased local BOLD
# These are key features for EEG-fMRI prediction


def extract_erds_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    baseline_window: Optional[Tuple[float, float]] = None,
    active_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Event-Related Desynchronization/Synchronization (ERD/ERS) features.

    ERD/ERS = (Active - Baseline) / Baseline * 100
    - Negative values = ERD (desynchronization, power decrease)
    - Positive values = ERS (synchronization, power increase)

    ERD/ERS strongly correlates with BOLD:
    - Alpha/beta ERD in sensorimotor areas → BOLD increase
    - Gamma ERS → local BOLD increase

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands to compute ERD/ERS for
    config : Any
        Configuration object
    logger : Any
        Logger instance
    baseline_window : Optional[Tuple[float, float]]
        Baseline period [tmin, tmax] in seconds. Default from config.
    active_window : Optional[Tuple[float, float]]
        Active/stimulus period [tmin, tmax] in seconds. Default from config (plateau).

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with ERD/ERS features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for ERD/ERS extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    ch_names = [epochs.info["ch_names"][p] for p in picks]

    if baseline_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        baseline_window = tuple(tf_cfg.get("baseline_window", [-5.0, -0.01]))

    if active_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        active_window = tuple(tf_cfg.get("plateau_window", [3.0, 10.5]))

    baseline_mask = time_mask(times, baseline_window[0], baseline_window[1])
    active_mask = time_mask(times, active_window[0], active_window[1])

    if not np.any(baseline_mask) or not np.any(active_mask):
        logger.warning("ERD/ERS: insufficient samples in baseline or active window")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for band in bands:
            if band not in freq_bands:
                continue

            fmin, fmax = freq_bands[band]
            envelope = compute_band_envelope_fast(epoch, sfreq, fmin, fmax)
            if envelope is None:
                for ch_name in ch_names:
                    record[f"erds_{band}_{ch_name}"] = np.nan
                continue

            power = envelope ** 2

            for ch_idx, ch_name in enumerate(ch_names):
                baseline_power = np.mean(power[ch_idx, baseline_mask])
                active_power = np.mean(power[ch_idx, active_mask])

                if baseline_power > epsilon:
                    erds = ((active_power - baseline_power) / baseline_power) * 100
                else:
                    erds = np.nan

                record[f"erds_{band}_{ch_name}"] = float(erds)

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_erds_temporal_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    baseline_window: Optional[Tuple[float, float]] = None,
    n_windows: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract temporally-resolved ERD/ERS features across the plateau period.

    This captures the temporal evolution of ERD/ERS, which may better predict
    the slow hemodynamic response dynamics in fMRI.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands to compute ERD/ERS for
    config : Any
        Configuration object
    logger : Any
        Logger instance
    baseline_window : Optional[Tuple[float, float]]
        Baseline period [tmin, tmax] in seconds
    n_windows : int
        Number of temporal windows in the active period

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with temporal ERD/ERS features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for temporal ERD/ERS extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    ch_names = [epochs.info["ch_names"][p] for p in picks]

    if baseline_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        baseline_window = tuple(tf_cfg.get("baseline_window", [-5.0, -0.01]))

    tf_cfg = config.get("time_frequency_analysis", {})
    plateau_window = tf_cfg.get("plateau_window", [3.0, 10.5])
    plateau_start, plateau_end = plateau_window

    window_duration = (plateau_end - plateau_start) / n_windows
    window_starts = [plateau_start + i * window_duration for i in range(n_windows)]

    baseline_mask = time_mask(times, baseline_window[0], baseline_window[1])
    if not np.any(baseline_mask):
        logger.warning("ERD/ERS temporal: insufficient samples in baseline window")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for band in bands:
            if band not in freq_bands:
                continue

            fmin, fmax = freq_bands[band]
            envelope = compute_band_envelope_fast(epoch, sfreq, fmin, fmax)
            if envelope is None:
                for ch_name in ch_names:
                    for win_idx in range(n_windows):
                        record[f"erds_{band}_{ch_name}_w{win_idx}"] = np.nan
                continue

            power = envelope ** 2

            for ch_idx, ch_name in enumerate(ch_names):
                baseline_power = np.mean(power[ch_idx, baseline_mask])

                for win_idx, win_start in enumerate(window_starts):
                    win_end = win_start + window_duration
                    win_mask = time_mask(times, win_start, win_end)

                    if not np.any(win_mask):
                        record[f"erds_{band}_{ch_name}_w{win_idx}"] = np.nan
                        continue

                    active_power = np.mean(power[ch_idx, win_mask])

                    if baseline_power > epsilon:
                        erds = ((active_power - baseline_power) / baseline_power) * 100
                    else:
                        erds = np.nan

                    record[f"erds_{band}_{ch_name}_w{win_idx}"] = float(erds)

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_erds_slope_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    baseline_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract ERD/ERS slope features (rate of change during stimulus).

    The slope of ERD/ERS may predict the rate of BOLD change,
    useful for modeling hemodynamic response dynamics.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands
    config : Any
        Configuration object
    logger : Any
        Logger instance
    baseline_window : Optional[Tuple[float, float]]
        Baseline period [tmin, tmax] in seconds

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with ERD/ERS slope features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for ERD/ERS slope extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    ch_names = [epochs.info["ch_names"][p] for p in picks]

    if baseline_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        baseline_window = tuple(tf_cfg.get("baseline_window", [-5.0, -0.01]))

    tf_cfg = config.get("time_frequency_analysis", {})
    plateau_window = tf_cfg.get("plateau_window", [3.0, 10.5])

    baseline_mask = time_mask(times, baseline_window[0], baseline_window[1])
    plateau_mask = time_mask(times, plateau_window[0], plateau_window[1])

    if not np.any(baseline_mask) or not np.any(plateau_mask):
        logger.warning("ERD/ERS slope: insufficient samples")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))
    plateau_times = times[plateau_mask]

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for band in bands:
            if band not in freq_bands:
                continue

            fmin, fmax = freq_bands[band]
            envelope = compute_band_envelope_fast(epoch, sfreq, fmin, fmax)
            if envelope is None:
                for ch_name in ch_names:
                    record[f"erds_slope_{band}_{ch_name}"] = np.nan
                    record[f"erds_onset_{band}_{ch_name}"] = np.nan
                continue

            power = envelope ** 2

            for ch_idx, ch_name in enumerate(ch_names):
                baseline_power = np.mean(power[ch_idx, baseline_mask])

                if baseline_power > epsilon:
                    erds_trace = ((power[ch_idx, plateau_mask] - baseline_power) / baseline_power) * 100

                    if len(plateau_times) > 1 and len(erds_trace) > 1:
                        slope, intercept = np.polyfit(plateau_times, erds_trace, 1)
                        record[f"erds_slope_{band}_{ch_name}"] = float(slope)
                        record[f"erds_onset_{band}_{ch_name}"] = float(erds_trace[0])
                    else:
                        record[f"erds_slope_{band}_{ch_name}"] = np.nan
                        record[f"erds_onset_{band}_{ch_name}"] = np.nan
                else:
                    record[f"erds_slope_{band}_{ch_name}"] = np.nan
                    record[f"erds_onset_{band}_{ch_name}"] = np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names

