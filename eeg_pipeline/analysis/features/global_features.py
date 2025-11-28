from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne
from scipy import stats as scipy_stats

from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.tfr import time_mask
from eeg_pipeline.analysis.features.core import (
    compute_gfp,
    compute_gfp_with_peaks,
    compute_band_envelope_fast,
    compute_band_phase_fast,
)


###################################################################
# Global EEG Features for fMRI Prediction
###################################################################
# Global features capture brain-wide activity patterns that correlate
# with global BOLD fluctuations and vigilance states.
#
# Uses shared utilities from core.py to avoid code duplication.


def extract_gfp_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    active_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Global Field Power (GFP) features.

    GFP = spatial standard deviation across channels at each time point.
    GFP dynamics correlate with global BOLD fluctuations.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    active_window : Optional[Tuple[float, float]]
        Time window for feature extraction. Default: plateau window.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with GFP features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for GFP extraction")
        return pd.DataFrame(), []

    times = epochs.times
    sfreq = float(epochs.info["sfreq"])

    if active_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        active_window = tuple(tf_cfg.get("plateau_window", [3.0, 10.5]))

    active_mask = time_mask(times, active_window[0], active_window[1])
    if not np.any(active_mask):
        logger.warning("GFP: no samples in active window")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    min_peak_distance = int(config.get("feature_engineering.gfp.min_peak_distance_samples", 10))

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        # GFP = spatial std across channels at each time point
        gfp_full = np.std(epoch, axis=0)  # (times,)
        gfp = gfp_full[active_mask]

        record["gfp_mean"] = float(np.mean(gfp))
        record["gfp_std"] = float(np.std(gfp))
        record["gfp_max"] = float(np.max(gfp))
        record["gfp_min"] = float(np.min(gfp))
        record["gfp_range"] = float(np.max(gfp) - np.min(gfp))
        record["gfp_cv"] = float(np.std(gfp) / np.mean(gfp)) if np.mean(gfp) > 0 else np.nan

        # Compute peaks within active window only for consistent peak rate calculation
        _, peaks = compute_gfp_with_peaks(epoch[:, active_mask], min_peak_distance)
        duration = len(gfp) / sfreq if sfreq > 0 else 1.0
        record["gfp_peak_rate"] = float(len(peaks) / duration) if duration > 0 else np.nan

        if len(peaks) > 0:
            record["gfp_peak_mean"] = float(np.mean(gfp[peaks]))
        else:
            record["gfp_peak_mean"] = np.nan

        record["gfp_skew"] = float(scipy_stats.skew(gfp))
        record["gfp_kurt"] = float(scipy_stats.kurtosis(gfp))

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_gfp_band_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    active_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract band-specific GFP features.

    Computes GFP after bandpass filtering in each frequency band.
    Band-specific GFP may differentially predict BOLD in different regions.

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
    active_window : Optional[Tuple[float, float]]
        Time window for feature extraction

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with band-GFP features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for band-GFP extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])

    if active_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        active_window = tuple(tf_cfg.get("plateau_window", [3.0, 10.5]))

    active_mask = time_mask(times, active_window[0], active_window[1])
    if not np.any(active_mask):
        logger.warning("Band-GFP: no samples in active window")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for band in bands:
            if band not in freq_bands:
                continue

            fmin, fmax = freq_bands[band]

            try:
                filtered = mne.filter.filter_data(
                    epoch,
                    sfreq,
                    l_freq=fmin,
                    h_freq=fmax,
                    n_jobs=1,
                    verbose=False,
                )
                # GFP = spatial std across channels at each time point
                gfp = np.std(filtered, axis=0)
                gfp_active = gfp[active_mask]

                record[f"gfp_{band}_mean"] = float(np.mean(gfp_active))
                record[f"gfp_{band}_std"] = float(np.std(gfp_active))
                record[f"gfp_{band}_max"] = float(np.max(gfp_active))

            except (ValueError, RuntimeError, IndexError):
                record[f"gfp_{band}_mean"] = np.nan
                record[f"gfp_{band}_std"] = np.nan
                record[f"gfp_{band}_max"] = np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_global_synchrony_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    active_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract global synchrony features (mean pairwise phase coherence).

    Global synchrony captures brain-wide coordination and may predict
    global BOLD signal changes and vigilance fluctuations.

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
    active_window : Optional[Tuple[float, float]]
        Time window for feature extraction

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with global synchrony features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for global synchrony extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])

    if active_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        active_window = tuple(tf_cfg.get("plateau_window", [3.0, 10.5]))

    active_mask = time_mask(times, active_window[0], active_window[1])
    if not np.any(active_mask):
        logger.warning("Global synchrony: no samples in active window")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for band in bands:
            if band not in freq_bands:
                continue

            fmin, fmax = freq_bands[band]

            try:
                # Use shared function for band-limited phase
                phases_full = compute_band_phase_fast(epoch, sfreq, fmin, fmax)
                if phases_full is None:
                    record[f"global_plv_{band}"] = np.nan
                    continue
                
                phases = phases_full[:, active_mask]

                n_channels = phases.shape[0]
                plv_sum = 0.0
                n_pairs = 0

                for i in range(n_channels):
                    for j in range(i + 1, n_channels):
                        phase_diff = phases[i] - phases[j]
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                        plv_sum += plv
                        n_pairs += 1

                if n_pairs > 0:
                    record[f"global_plv_{band}"] = float(plv_sum / n_pairs)
                else:
                    record[f"global_plv_{band}"] = np.nan

            except (ValueError, RuntimeError, IndexError):
                record[f"global_plv_{band}"] = np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_variance_explained_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    n_components: int = 5,
    active_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract PCA variance explained features.

    The distribution of variance across principal components indicates
    whether activity is focal (few components) or distributed (many).
    This may relate to the spatial extent of BOLD activation.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    n_components : int
        Number of PCA components to consider
    active_window : Optional[Tuple[float, float]]
        Time window for feature extraction

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with variance features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for variance features")
        return pd.DataFrame(), []

    times = epochs.times

    if active_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        active_window = tuple(tf_cfg.get("plateau_window", [3.0, 10.5]))

    active_mask = time_mask(times, active_window[0], active_window[1])
    if not np.any(active_mask):
        logger.warning("Variance features: no samples in active window")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    n_components = min(n_components, len(picks))

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        epoch_active = epoch[:, active_mask]

        if epoch_active.shape[1] < 2:
            for i in range(n_components):
                record[f"pca_var_pc{i+1}"] = np.nan
            record["pca_var_cumsum_3"] = np.nan
            record["pca_var_entropy"] = np.nan
            feature_records.append(record)
            continue

        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(epoch_active.T)

            var_explained = pca.explained_variance_ratio_

            for i, var in enumerate(var_explained):
                record[f"pca_var_pc{i+1}"] = float(var)

            record["pca_var_cumsum_3"] = float(np.sum(var_explained[:3])) if len(var_explained) >= 3 else np.nan

            var_norm = var_explained / np.sum(var_explained)
            var_entropy = -np.sum(var_norm * np.log2(var_norm + 1e-12))
            record["pca_var_entropy"] = float(var_entropy)

        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            for i in range(n_components):
                record[f"pca_var_pc{i+1}"] = np.nan
            record["pca_var_cumsum_3"] = np.nan
            record["pca_var_entropy"] = np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_all_global_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract all global EEG features in one call.

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

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with all global features and column names
    """
    all_dfs: List[pd.DataFrame] = []
    all_cols: List[str] = []

    gfp_df, gfp_cols = extract_gfp_features(epochs, config, logger)
    if not gfp_df.empty:
        all_dfs.append(gfp_df)
        all_cols.extend(gfp_cols)

    gfp_band_df, gfp_band_cols = extract_gfp_band_features(epochs, bands, config, logger)
    if not gfp_band_df.empty:
        all_dfs.append(gfp_band_df)
        all_cols.extend(gfp_band_cols)

    sync_df, sync_cols = extract_global_synchrony_features(epochs, bands, config, logger)
    if not sync_df.empty:
        all_dfs.append(sync_df)
        all_cols.extend(sync_cols)

    var_df, var_cols = extract_variance_explained_features(epochs, config, logger)
    if not var_df.empty:
        all_dfs.append(var_df)
        all_cols.extend(var_cols)

    if not all_dfs:
        return pd.DataFrame(), []

    combined = pd.concat(all_dfs, axis=1)
    return combined, all_cols

