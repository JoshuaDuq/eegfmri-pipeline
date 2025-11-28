from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from scipy.stats import entropy

from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.analysis.features.core import pick_eeg_channels


###################################################################
# Spectral Feature Extraction
###################################################################


def _find_peak_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    min_prominence: float = 0.1,
) -> Tuple[float, float]:
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return np.nan, np.nan

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    if band_psd.size == 0 or not np.any(np.isfinite(band_psd)):
        return np.nan, np.nan

    peak_idx = np.nanargmax(band_psd)
    peak_freq = float(band_freqs[peak_idx])
    peak_power = float(band_psd[peak_idx])

    psd_range = np.nanmax(band_psd) - np.nanmin(band_psd)
    if psd_range > 0:
        prominence = (peak_power - np.nanmin(band_psd)) / psd_range
        if prominence < min_prominence:
            return np.nan, np.nan

    return peak_freq, peak_power


def _compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return np.nan
    return float(np.nanmean(psd[band_mask]))


def _compute_spectral_entropy(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    normalize: bool = True,
) -> float:
    if fmin is not None and fmax is not None:
        mask = (freqs >= fmin) & (freqs <= fmax)
        psd = psd[mask]

    if psd.size == 0 or not np.any(np.isfinite(psd)):
        return np.nan

    psd = np.maximum(psd, 0)
    psd_sum = np.nansum(psd)
    if psd_sum <= 0:
        return np.nan

    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]

    if psd_norm.size == 0:
        return np.nan

    se = float(entropy(psd_norm, base=2))

    if normalize and psd_norm.size > 1:
        se = se / np.log2(psd_norm.size)

    return se


def extract_individual_alpha_frequency(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    alpha_range: Optional[Tuple[float, float]] = None,
    method: str = "cog",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Individual Alpha Frequency (IAF) for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract IAF from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    alpha_range : Optional[Tuple[float, float]]
        Alpha frequency range [fmin, fmax]. If None, uses config.
    method : str
        "peak" for peak frequency, "cog" for center of gravity (more robust)

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with IAF features and column names
    """
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for IAF extraction")
        return pd.DataFrame(), []

    if alpha_range is None:
        freq_bands = get_frequency_bands(config)
        alpha_range = freq_bands.get("alpha", (8.0, 13.0))

    fmin, fmax = alpha_range
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []
    column_names: List[str] = []

    data = epochs.get_data(picks=picks)

    for epoch_idx, epoch in enumerate(data):
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            nperseg = min(len(ch_data), int(2 * sfreq))

            try:
                freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
            except (ValueError, RuntimeError):
                record[f"iaf_{ch_name}"] = np.nan
                record[f"iaf_power_{ch_name}"] = np.nan
                continue

            alpha_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(alpha_mask):
                record[f"iaf_{ch_name}"] = np.nan
                record[f"iaf_power_{ch_name}"] = np.nan
                continue

            alpha_freqs = freqs[alpha_mask]
            alpha_psd = psd[alpha_mask]

            if method == "cog":
                psd_sum = np.nansum(alpha_psd)
                if psd_sum > 0:
                    iaf = float(np.nansum(alpha_freqs * alpha_psd) / psd_sum)
                    iaf_power = float(np.nanmax(alpha_psd))
                else:
                    iaf, iaf_power = np.nan, np.nan
            else:
                iaf, iaf_power = _find_peak_frequency(freqs, psd, fmin, fmax)

            record[f"iaf_{ch_name}"] = iaf
            record[f"iaf_power_{ch_name}"] = iaf_power

        feature_records.append(record)

    if feature_records:
        column_names = list(feature_records[0].keys())

    global_iaf = []
    for record in feature_records:
        iaf_vals = [v for k, v in record.items() if k.startswith("iaf_") and not k.startswith("iaf_power_")]
        iaf_vals = [v for v in iaf_vals if np.isfinite(v)]
        record["iaf_global"] = float(np.median(iaf_vals)) if iaf_vals else np.nan

    if feature_records:
        column_names = list(feature_records[0].keys())

    return pd.DataFrame(feature_records), column_names


def extract_relative_band_power(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    total_range: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract relative band power (band power / total power) for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands to extract (e.g., ["delta", "theta", "alpha", "beta", "gamma"])
    config : Any
        Configuration object
    logger : Any
        Logger instance
    total_range : Optional[Tuple[float, float]]
        Frequency range for total power calculation. If None, uses 1-80 Hz.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with relative power features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for relative power extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    if total_range is None:
        total_range = (1.0, 80.0)

    sfreq = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            nperseg = min(len(ch_data), int(2 * sfreq))

            try:
                freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
            except (ValueError, RuntimeError):
                for band in bands:
                    record[f"relpow_{band}_{ch_name}"] = np.nan
                continue

            total_power = _compute_band_power(freqs, psd, total_range[0], total_range[1])

            for band in bands:
                if band not in freq_bands:
                    logger.warning(f"Band '{band}' not defined in config; skipping")
                    continue

                fmin, fmax = freq_bands[band]
                band_power = _compute_band_power(freqs, psd, fmin, fmax)

                if total_power > 0 and np.isfinite(band_power):
                    record[f"relpow_{band}_{ch_name}"] = float(band_power / total_power)
                else:
                    record[f"relpow_{band}_{ch_name}"] = np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_band_power_ratios(
    epochs: mne.Epochs,
    ratios: List[Tuple[str, str]],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract band power ratios (e.g., theta/beta for attention, alpha/theta for alertness).

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    ratios : List[Tuple[str, str]]
        List of (numerator_band, denominator_band) tuples
        e.g., [("theta", "beta"), ("alpha", "theta")]
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with ratio features and column names
    """
    if not ratios:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for band ratio extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    sfreq = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks)
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            nperseg = min(len(ch_data), int(2 * sfreq))

            try:
                freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
            except (ValueError, RuntimeError):
                for num_band, denom_band in ratios:
                    record[f"ratio_{num_band}_{denom_band}_{ch_name}"] = np.nan
                continue

            for num_band, denom_band in ratios:
                if num_band not in freq_bands or denom_band not in freq_bands:
                    logger.warning(f"Band ratio {num_band}/{denom_band} skipped: band not in config")
                    record[f"ratio_{num_band}_{denom_band}_{ch_name}"] = np.nan
                    continue

                num_fmin, num_fmax = freq_bands[num_band]
                denom_fmin, denom_fmax = freq_bands[denom_band]

                num_power = _compute_band_power(freqs, psd, num_fmin, num_fmax)
                denom_power = _compute_band_power(freqs, psd, denom_fmin, denom_fmax)

                if np.isfinite(num_power) and np.isfinite(denom_power) and denom_power > epsilon:
                    record[f"ratio_{num_band}_{denom_band}_{ch_name}"] = float(num_power / denom_power)
                else:
                    record[f"ratio_{num_band}_{denom_band}_{ch_name}"] = np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_spectral_entropy_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract spectral entropy features for each frequency band.

    Spectral entropy measures the "uniformity" of the power spectrum.
    Lower entropy = more peaked spectrum (more rhythmic activity)
    Higher entropy = flatter spectrum (more noise-like)

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands to compute entropy for
    config : Any
        Configuration object
    logger : Any
        Logger instance
    normalize : bool
        If True, normalize entropy to [0, 1] range

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with spectral entropy features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for spectral entropy extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    sfreq = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            nperseg = min(len(ch_data), int(2 * sfreq))

            try:
                freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
            except (ValueError, RuntimeError):
                for band in bands:
                    record[f"se_{band}_{ch_name}"] = np.nan
                record[f"se_broadband_{ch_name}"] = np.nan
                continue

            for band in bands:
                if band not in freq_bands:
                    continue
                fmin, fmax = freq_bands[band]
                se = _compute_spectral_entropy(freqs, psd, fmin, fmax, normalize=normalize)
                record[f"se_{band}_{ch_name}"] = se

            se_broad = _compute_spectral_entropy(freqs, psd, 1.0, 80.0, normalize=normalize)
            record[f"se_broadband_{ch_name}"] = se_broad

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_peak_frequencies(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract peak frequency within each band for each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    bands : List[str]
        Frequency bands to find peaks in
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with peak frequency features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for peak frequency extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    sfreq = float(epochs.info["sfreq"])
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            nperseg = min(len(ch_data), int(2 * sfreq))

            try:
                freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
            except (ValueError, RuntimeError):
                for band in bands:
                    record[f"peakfreq_{band}_{ch_name}"] = np.nan
                continue

            for band in bands:
                if band not in freq_bands:
                    continue
                fmin, fmax = freq_bands[band]
                peak_freq, _ = _find_peak_frequency(freqs, psd, fmin, fmax)
                record[f"peakfreq_{band}_{ch_name}"] = peak_freq

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names

