"""
Time-Domain Feature Extraction
===============================

Statistical and waveform features computed directly in the time domain:
- Statistical: mean, variance, skewness, kurtosis
- Amplitude: RMS, peak-to-peak, MAD
- Waveform: zero crossings, line length, nonlinear energy

All features can be computed on raw or band-filtered signals.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne
from scipy import stats as scipy_stats

from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_zero_crossings as _compute_zero_crossings,
    compute_rms as _compute_rms,
    compute_peak_to_peak as _compute_peak_to_peak,
    compute_line_length as _compute_line_length,
    compute_mean_absolute_deviation as _compute_mean_absolute_deviation,
    compute_nonlinear_energy as _compute_nonlinear_energy,
)


def extract_statistical_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract basic statistical features (mean, variance, skewness, kurtosis) per epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with statistical features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for statistical feature extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]

            record[f"mean_{ch_name}"] = float(np.mean(ch_data))
            record[f"var_{ch_name}"] = float(np.var(ch_data, ddof=1)) if len(ch_data) > 1 else np.nan
            record[f"std_{ch_name}"] = float(np.std(ch_data, ddof=1)) if len(ch_data) > 1 else np.nan
            record[f"skew_{ch_name}"] = float(scipy_stats.skew(ch_data, nan_policy='omit', bias=False))
            record[f"kurt_{ch_name}"] = float(scipy_stats.kurtosis(ch_data, nan_policy='omit', bias=False))
            record[f"median_{ch_name}"] = float(np.median(ch_data))
            record[f"iqr_{ch_name}"] = float(scipy_stats.iqr(ch_data))

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_amplitude_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract amplitude-based features (RMS, peak-to-peak, MAD) per epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with amplitude features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for amplitude feature extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]

            record[f"rms_{ch_name}"] = _compute_rms(ch_data)
            record[f"ptp_{ch_name}"] = _compute_peak_to_peak(ch_data)
            record[f"mad_{ch_name}"] = _compute_mean_absolute_deviation(ch_data)
            record[f"max_{ch_name}"] = float(np.max(ch_data)) if len(ch_data) > 0 else np.nan
            record[f"min_{ch_name}"] = float(np.min(ch_data)) if len(ch_data) > 0 else np.nan
            record[f"absmax_{ch_name}"] = float(np.max(np.abs(ch_data))) if len(ch_data) > 0 else np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_waveform_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract waveform-based features (zero-crossings, line length, nonlinear energy).

    These features capture signal dynamics and are commonly used in seizure detection
    and other EEG applications.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with waveform features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for waveform feature extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]

            ch_data_centered = ch_data - np.mean(ch_data) if len(ch_data) > 0 else ch_data

            zc = _compute_zero_crossings(ch_data_centered)
            record[f"zerocross_{ch_name}"] = float(zc)
            
            duration = len(ch_data) / sfreq if sfreq > 0 else 1.0
            record[f"zerocross_rate_{ch_name}"] = float(zc / duration) if duration > 0 else np.nan

            linelen_val = _compute_line_length(ch_data_centered)
            record[f"linelen_{ch_name}"] = linelen_val
            
            record[f"linelen_norm_{ch_name}"] = float(linelen_val / len(ch_data)) if len(ch_data) > 1 else np.nan

            record[f"nle_{ch_name}"] = _compute_nonlinear_energy(ch_data_centered)

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_percentile_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    percentiles: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract percentile-based features per epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    percentiles : Optional[List[float]]
        Percentiles to compute. Default: [5, 25, 75, 95]

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with percentile features and column names
    """
    if percentiles is None:
        percentiles = [5.0, 25.0, 75.0, 95.0]

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for percentile feature extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]

            if len(ch_data) == 0:
                for p in percentiles:
                    record[f"pct{int(p)}_{ch_name}"] = np.nan
                continue

            pct_values = np.percentile(ch_data, percentiles)
            for p, val in zip(percentiles, pct_values):
                record[f"pct{int(p)}_{ch_name}"] = float(val)

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_derivative_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features based on signal derivatives (first and second).

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with derivative features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for derivative feature extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]

            if len(ch_data) < 2:
                record[f"d1_var_{ch_name}"] = np.nan
                record[f"d1_max_{ch_name}"] = np.nan
                record[f"d2_var_{ch_name}"] = np.nan
                record[f"d2_max_{ch_name}"] = np.nan
                continue

            d1 = np.diff(ch_data) * sfreq
            record[f"d1_var_{ch_name}"] = float(np.var(d1, ddof=1)) if len(d1) > 1 else np.nan
            record[f"d1_max_{ch_name}"] = float(np.max(np.abs(d1)))
            record[f"d1_mean_{ch_name}"] = float(np.mean(np.abs(d1)))

            if len(d1) < 2:
                record[f"d2_var_{ch_name}"] = np.nan
                record[f"d2_max_{ch_name}"] = np.nan
                record[f"d2_mean_{ch_name}"] = np.nan
                continue

            d2 = np.diff(d1) * sfreq
            record[f"d2_var_{ch_name}"] = float(np.var(d2, ddof=1)) if len(d2) > 1 else np.nan
            record[f"d2_max_{ch_name}"] = float(np.max(np.abs(d2)))
            record[f"d2_mean_{ch_name}"] = float(np.mean(np.abs(d2)))

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_all_temporal_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract all time-domain features in one call.

    Includes: statistical, amplitude, waveform, and derivative features.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with all temporal features and column names
    """
    all_dfs: List[pd.DataFrame] = []
    all_cols: List[str] = []

    include_percentiles = bool(config.get("feature_engineering.temporal.include_percentiles", False))
    include_derivatives = bool(config.get("feature_engineering.temporal.include_derivatives", True))

    stat_df, stat_cols = extract_statistical_features(epochs, config, logger)
    if not stat_df.empty:
        all_dfs.append(stat_df)
        all_cols.extend(stat_cols)

    amp_df, amp_cols = extract_amplitude_features(epochs, config, logger)
    if not amp_df.empty:
        all_dfs.append(amp_df)
        all_cols.extend(amp_cols)

    wave_df, wave_cols = extract_waveform_features(epochs, config, logger)
    if not wave_df.empty:
        all_dfs.append(wave_df)
        all_cols.extend(wave_cols)

    if include_percentiles:
        pct_df, pct_cols = extract_percentile_features(epochs, config, logger)
        if not pct_df.empty:
            all_dfs.append(pct_df)
            all_cols.extend(pct_cols)

    if include_derivatives:
        deriv_df, deriv_cols = extract_derivative_features(epochs, config, logger)
        if not deriv_df.empty:
            all_dfs.append(deriv_df)
            all_cols.extend(deriv_cols)

    if not all_dfs:
        return pd.DataFrame(), []

    combined = pd.concat(all_dfs, axis=1)
    return combined, all_cols


###################################################################
# Band-Filtered Temporal Features
###################################################################


# Use shared bandpass filter from core
from eeg_pipeline.analysis.features.core import bandpass_filter_epochs


def extract_band_statistical_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract statistical features per frequency band.

    Features are computed on band-filtered signals, giving features like:
    - var_alpha_Cz, var_beta_Cz, etc.
    - skew_alpha_Cz, kurt_alpha_Cz, etc.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands (e.g., ["alpha", "beta", "gamma"])
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with band-specific statistical features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for band statistical feature extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []

    for band in bands:
        if band not in freq_bands:
            logger.warning(f"Band '{band}' not defined in config; skipping")
            continue

        fmin, fmax = freq_bands[band]
        filtered = bandpass_filter_epochs(data, sfreq, fmin, fmax)
        if filtered is None:
            logger.warning(f"Failed to filter data for band '{band}'; skipping")
            continue

        for epoch_idx, epoch in enumerate(filtered):
            if epoch_idx >= len(feature_records):
                feature_records.append({})
            record = feature_records[epoch_idx]

            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]

                record[f"var_{band}_{ch_name}"] = float(np.var(ch_data, ddof=1)) if len(ch_data) > 1 else np.nan
                record[f"std_{band}_{ch_name}"] = float(np.std(ch_data, ddof=1)) if len(ch_data) > 1 else np.nan
                record[f"skew_{band}_{ch_name}"] = float(scipy_stats.skew(ch_data, nan_policy='omit', bias=False))
                record[f"kurt_{band}_{ch_name}"] = float(scipy_stats.kurtosis(ch_data, nan_policy='omit', bias=False))

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_band_amplitude_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract amplitude features per frequency band.

    Features are computed on band-filtered signals, giving features like:
    - rms_alpha_Cz, rms_beta_Cz, etc.
    - ptp_alpha_Cz, ptp_beta_Cz, etc.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands (e.g., ["alpha", "beta", "gamma"])
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with band-specific amplitude features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for band amplitude feature extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []

    for band in bands:
        if band not in freq_bands:
            continue

        fmin, fmax = freq_bands[band]
        filtered = bandpass_filter_epochs(data, sfreq, fmin, fmax)
        if filtered is None:
            continue

        for epoch_idx, epoch in enumerate(filtered):
            if epoch_idx >= len(feature_records):
                feature_records.append({})
            record = feature_records[epoch_idx]

            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]

                record[f"rms_{band}_{ch_name}"] = _compute_rms(ch_data)
                record[f"ptp_{band}_{ch_name}"] = _compute_peak_to_peak(ch_data)
                record[f"mad_{band}_{ch_name}"] = _compute_mean_absolute_deviation(ch_data)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_band_waveform_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract waveform features per frequency band.

    Features are computed on band-filtered signals, giving features like:
    - zerocross_alpha_Cz, linelen_alpha_Cz, nle_alpha_Cz, etc.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands (e.g., ["alpha", "beta", "gamma"])
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with band-specific waveform features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for band waveform feature extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []

    for band in bands:
        if band not in freq_bands:
            continue

        fmin, fmax = freq_bands[band]
        filtered = bandpass_filter_epochs(data, sfreq, fmin, fmax)
        if filtered is None:
            continue

        for epoch_idx, epoch in enumerate(filtered):
            if epoch_idx >= len(feature_records):
                feature_records.append({})
            record = feature_records[epoch_idx]

            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]

                ch_data_centered = ch_data - np.mean(ch_data) if len(ch_data) > 0 else ch_data

                zc = _compute_zero_crossings(ch_data_centered)
                record[f"zerocross_{band}_{ch_name}"] = float(zc)

                duration = len(ch_data) / sfreq if sfreq > 0 else 1.0
                record[f"zerocross_rate_{band}_{ch_name}"] = float(zc / duration) if duration > 0 else np.nan

                linelen_val = _compute_line_length(ch_data_centered)
                record[f"linelen_{band}_{ch_name}"] = linelen_val
                record[f"nle_{band}_{ch_name}"] = _compute_nonlinear_energy(ch_data_centered)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_all_band_temporal_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract all time-domain features per frequency band.

    Includes: statistical, amplitude, and waveform features for each band.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands (e.g., ["alpha", "beta", "gamma"])
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with all band-specific temporal features and column names
    """
    all_dfs: List[pd.DataFrame] = []
    all_cols: List[str] = []

    stat_df, stat_cols = extract_band_statistical_features(epochs, bands, config, logger)
    if not stat_df.empty:
        all_dfs.append(stat_df)
        all_cols.extend(stat_cols)

    amp_df, amp_cols = extract_band_amplitude_features(epochs, bands, config, logger)
    if not amp_df.empty:
        all_dfs.append(amp_df)
        all_cols.extend(amp_cols)

    wave_df, wave_cols = extract_band_waveform_features(epochs, bands, config, logger)
    if not wave_df.empty:
        all_dfs.append(wave_df)
        all_cols.extend(wave_cols)

    if not all_dfs:
        return pd.DataFrame(), []

    combined = pd.concat(all_dfs, axis=1)
    return combined, all_cols
