from __future__ import annotations

import re
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.analysis.features.core import (
    match_channels_to_pattern,
    build_roi_map,
)


###################################################################
# ROI-Averaged Features for fMRI Prediction
###################################################################
# ROI averaging:
# 1. Reduces dimensionality (64 channels → ~10 ROIs)
# 2. Maps to fMRI-relevant brain regions
# 3. Reduces noise through spatial averaging
# 4. Provides anatomically interpretable features
#
# Uses shared utilities from core.py.


def _get_roi_definitions(config: Any) -> Dict[str, List[str]]:
    """Get ROI definitions from config."""
    rois = config.get("rois", {})
    if not rois:
        rois = config.get("time_frequency_analysis", {}).get("rois", {})
    return rois


def _build_roi_channel_map(
    ch_names: List[str],
    roi_definitions: Dict[str, List[str]],
    logger: Any,
) -> Dict[str, List[int]]:
    """Build ROI channel map with logging for missing ROIs."""
    roi_map = build_roi_map(ch_names, roi_definitions)
    
    # Log ROIs with no matching channels
    for roi_name in roi_definitions:
        if roi_name not in roi_map:
            logger.debug(f"ROI '{roi_name}' has no matching channels")
    
    return roi_map


def extract_roi_power_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    channel_features_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract ROI-averaged power features from channel-level features or raw epochs.

    ROI averaging maps EEG channels to fMRI-relevant brain regions:
    - Frontal → prefrontal/ACC (pain processing)
    - Sensorimotor → S1/M1 (somatosensory)
    - Parietal → posterior parietal (attention/salience)

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
    channel_features_df : Optional[pd.DataFrame]
        Pre-computed channel-level features. If provided, averages these.
        If None, computes power from epochs directly.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with ROI-averaged features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for ROI feature extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    roi_definitions = _get_roi_definitions(config)
    
    if not roi_definitions:
        logger.warning("No ROI definitions found in config; cannot compute ROI features")
        return pd.DataFrame(), []

    roi_map = _build_roi_channel_map(ch_names, roi_definitions, logger)
    
    if not roi_map:
        logger.warning("No ROIs matched any channels; skipping ROI features")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    feature_records: List[Dict[str, float]] = []

    if channel_features_df is not None and not channel_features_df.empty:
        for row_idx in range(len(channel_features_df)):
            record: Dict[str, float] = {}
            
            for band in bands:
                if band not in freq_bands:
                    continue
                
                for roi_name, ch_indices in roi_map.items():
                    roi_ch_names = [ch_names[i] for i in ch_indices]
                    
                    col_candidates = [
                        f"pow_{band}_{ch}" for ch in roi_ch_names
                    ] + [
                        f"pow_{band}_{ch}_plateau" for ch in roi_ch_names
                    ]
                    
                    values = []
                    for col in col_candidates:
                        if col in channel_features_df.columns:
                            val = channel_features_df.iloc[row_idx][col]
                            if np.isfinite(val):
                                values.append(val)
                    
                    if values:
                        record[f"roi_pow_{band}_{roi_name}_mean"] = float(np.mean(values))
                        record[f"roi_pow_{band}_{roi_name}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
                    else:
                        record[f"roi_pow_{band}_{roi_name}_mean"] = np.nan
                        record[f"roi_pow_{band}_{roi_name}_std"] = np.nan
            
            feature_records.append(record)
    else:
        from scipy.signal import welch
        
        data = epochs.get_data(picks=picks)
        sfreq = float(epochs.info["sfreq"])
        
        for epoch in data:
            record: Dict[str, float] = {}
            
            for band in bands:
                if band not in freq_bands:
                    continue
                
                fmin, fmax = freq_bands[band]
                
                for roi_name, ch_indices in roi_map.items():
                    roi_powers = []
                    
                    for ch_idx in ch_indices:
                        ch_data = epoch[ch_idx]
                        nperseg = min(len(ch_data), int(2 * sfreq))
                        
                        try:
                            freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
                            band_mask = (freqs >= fmin) & (freqs <= fmax)
                            if np.any(band_mask):
                                band_power = np.mean(psd[band_mask])
                                roi_powers.append(band_power)
                        except (ValueError, RuntimeError):
                            continue
                    
                    if roi_powers:
                        record[f"roi_pow_{band}_{roi_name}_mean"] = float(np.mean(roi_powers))
                        record[f"roi_pow_{band}_{roi_name}_std"] = float(np.std(roi_powers)) if len(roi_powers) > 1 else 0.0
                    else:
                        record[f"roi_pow_{band}_{roi_name}_mean"] = np.nan
                        record[f"roi_pow_{band}_{roi_name}_std"] = np.nan
            
            feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_roi_asymmetry_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract hemispheric asymmetry features within ROIs.

    Asymmetry = (Right - Left) / (Right + Left)
    
    Particularly relevant for:
    - Contralateral vs ipsilateral sensorimotor activation during unilateral pain
    - Frontal asymmetry (related to affect/approach-withdrawal)

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
        DataFrame with asymmetry features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for asymmetry extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    freq_bands = get_frequency_bands(config)

    asymmetry_pairs = config.get("feature_engineering.roi_features.asymmetry_pairs", [
        {"left": "Sensorimotor_Ipsi_L", "right": "Sensorimotor_Contra_R", "name": "sensorimotor"},
        {"left": "Temporal_Ipsi_L", "right": "Temporal_Contra_R", "name": "temporal"},
        {"left": "ParOccipital_Ipsi_L", "right": "ParOccipital_Contra_R", "name": "paroccipital"},
    ])

    roi_definitions = _get_roi_definitions(config)
    if not roi_definitions:
        logger.warning("No ROI definitions found; cannot compute asymmetry")
        return pd.DataFrame(), []

    roi_map = _build_roi_channel_map(ch_names, roi_definitions, logger)

    from scipy.signal import welch
    
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    epsilon = float(config.get("feature_engineering.constants.epsilon_std", 1e-12))

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for band in bands:
            if band not in freq_bands:
                continue

            fmin, fmax = freq_bands[band]

            roi_powers: Dict[str, float] = {}
            for roi_name, ch_indices in roi_map.items():
                powers = []
                for ch_idx in ch_indices:
                    ch_data = epoch[ch_idx]
                    nperseg = min(len(ch_data), int(2 * sfreq))
                    try:
                        freqs, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
                        band_mask = (freqs >= fmin) & (freqs <= fmax)
                        if np.any(band_mask):
                            powers.append(np.mean(psd[band_mask]))
                    except (ValueError, RuntimeError):
                        continue
                if powers:
                    roi_powers[roi_name] = float(np.mean(powers))

            for pair in asymmetry_pairs:
                left_roi = pair.get("left", "")
                right_roi = pair.get("right", "")
                pair_name = pair.get("name", f"{left_roi}_{right_roi}")

                left_power = roi_powers.get(left_roi, np.nan)
                right_power = roi_powers.get(right_roi, np.nan)

                if np.isfinite(left_power) and np.isfinite(right_power):
                    denom = left_power + right_power
                    if denom > epsilon:
                        asymmetry = (right_power - left_power) / denom
                    else:
                        asymmetry = np.nan
                else:
                    asymmetry = np.nan

                record[f"asym_{band}_{pair_name}"] = float(asymmetry) if np.isfinite(asymmetry) else np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_roi_connectivity_features(
    connectivity_df: pd.DataFrame,
    config: Any,
    logger: Any,
    *,
    bands: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate connectivity features to ROI-level (inter-ROI connectivity).

    Reduces pairwise connectivity (n_channels² features) to ROI pairs
    (~10 ROIs → ~45 pairs), making it tractable for ML.

    Parameters
    ----------
    connectivity_df : pd.DataFrame
        Pre-computed connectivity features (pairwise)
    config : Any
        Configuration object
    logger : Any
        Logger instance

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with ROI-level connectivity and column names
    """
    if connectivity_df is None or connectivity_df.empty:
        return pd.DataFrame(), []

    roi_definitions = _get_roi_definitions(config)
    if not roi_definitions:
        logger.warning("No ROI definitions found; cannot aggregate connectivity")
        return pd.DataFrame(), []

    roi_names = list(roi_definitions.keys())
    
    # Use configured bands instead of hardcoded list
    if bands is None:
        freq_bands = get_frequency_bands(config)
        bands = list(freq_bands.keys())

    feature_records: List[Dict[str, float]] = []

    for row_idx in range(len(connectivity_df)):
        record: Dict[str, float] = {}
        row = connectivity_df.iloc[row_idx]

        # Detect connectivity measures from DataFrame columns dynamically
        available_measures = set()
        for col in connectivity_df.columns:
            for measure_candidate in ["wpli", "aec", "imcoh", "pli", "plv"]:
                if col.startswith(f"{measure_candidate}_"):
                    available_measures.add(measure_candidate)
                    break
        
        for measure in sorted(available_measures):
            for band in bands:
                for i, roi_i in enumerate(roi_names):
                    patterns_i = roi_definitions[roi_i]

                    within_values = []
                    for col in connectivity_df.columns:
                        if not col.startswith(f"{measure}_{band}_"):
                            continue
                        parts = col.replace(f"{measure}_{band}_", "").split("_")
                        if len(parts) < 2:
                            continue
                        ch1, ch2 = parts[0], parts[-1]

                        ch1_in_roi = any(re.match(p, ch1) for p in patterns_i)
                        ch2_in_roi = any(re.match(p, ch2) for p in patterns_i)

                        if ch1_in_roi and ch2_in_roi:
                            val = row[col]
                            if np.isfinite(val):
                                within_values.append(val)

                    if within_values:
                        record[f"roi_{measure}_{band}_{roi_i}_within"] = float(np.mean(within_values))

                    for j, roi_j in enumerate(roi_names):
                        if j <= i:
                            continue

                        patterns_j = roi_definitions[roi_j]
                        between_values = []

                        for col in connectivity_df.columns:
                            if not col.startswith(f"{measure}_{band}_"):
                                continue
                            parts = col.replace(f"{measure}_{band}_", "").split("_")
                            if len(parts) < 2:
                                continue
                            ch1, ch2 = parts[0], parts[-1]

                            ch1_in_i = any(re.match(p, ch1) for p in patterns_i)
                            ch2_in_j = any(re.match(p, ch2) for p in patterns_j)
                            ch1_in_j = any(re.match(p, ch1) for p in patterns_j)
                            ch2_in_i = any(re.match(p, ch2) for p in patterns_i)

                            if (ch1_in_i and ch2_in_j) or (ch1_in_j and ch2_in_i):
                                val = row[col]
                                if np.isfinite(val):
                                    between_values.append(val)

                        if between_values:
                            record[f"roi_{measure}_{band}_{roi_i}_{roi_j}"] = float(np.mean(between_values))

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_pain_roi_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract features from pain-specific ROIs (pain matrix regions).

    Pain matrix regions mapped from fMRI:
    - Sensorimotor (contralateral) → S1/S2
    - Midline (Fz, Cz, CPz) → ACC/MCC
    - Frontal → Prefrontal/anterior insula proxy
    - Temporal → Posterior insula proxy

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
        DataFrame with pain-ROI features and column names
    """
    pain_rois = config.get("feature_engineering.roi_features.pain_rois", [
        "Sensorimotor_Contra_R",
        "Midline_ACC_MCC",
        "Frontal",
        "Temporal_Contra_R",
    ])

    roi_df, roi_cols = extract_roi_power_features(epochs, bands, config, logger)
    
    if roi_df.empty:
        return pd.DataFrame(), []

    pain_cols = [col for col in roi_cols if any(roi in col for roi in pain_rois)]
    
    if not pain_cols:
        logger.warning("No pain ROI columns found in features")
        return pd.DataFrame(), []

    return roi_df[pain_cols], pain_cols

