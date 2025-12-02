"""
Complexity Feature Extraction
==============================

Nonlinear dynamics features for EEG analysis:
- Permutation Entropy (PE): Ordinal pattern complexity
- Hjorth Parameters: Activity, mobility, complexity
- Lempel-Ziv Complexity (LZC): Algorithmic complexity
- Sample Entropy: Regularity measure

All features can be computed on raw or band-filtered signals.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.analysis.features.core import MIN_SAMPLES_FOR_ENTROPY, pick_eeg_channels
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_permutation_entropy as _permutation_entropy,
    compute_sample_entropy as _sample_entropy,
    compute_hjorth_parameters as _hjorth_parameters,
    compute_lempel_ziv_complexity as _lempel_ziv_complexity,
)


def extract_permutation_entropy_features(
    epochs: mne.Epochs, config: Any, logger: Any, *, order: int = 3, delay: int = 1, normalize: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract permutation entropy per epoch/channel. Lower PE = more regular."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for permutation entropy extraction")
        return pd.DataFrame(), []

    order = int(config.get("feature_engineering.complexity.pe_order", order))
    delay = int(config.get("feature_engineering.complexity.pe_delay", delay))
    min_samples = int(config.get("feature_engineering.complexity.min_samples_for_entropy", MIN_SAMPLES_FOR_ENTROPY))
    if (order - 1) * delay + 1 > min_samples:
        min_samples = (order - 1) * delay + 1
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []
    n_too_short = 0

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            if len(ch_data) < min_samples:
                record[f"pe_{ch_name}"] = np.nan
                n_too_short += 1
                continue
            pe = _permutation_entropy(ch_data, order=order, delay=delay, normalize=normalize)
            record[f"pe_{ch_name}"] = pe

        pe_vals = [v for v in record.values() if np.isfinite(v)]
        record["pe_global_mean"] = float(np.mean(pe_vals)) if pe_vals else np.nan
        record["pe_global_std"] = float(np.std(pe_vals)) if len(pe_vals) > 1 else np.nan

        feature_records.append(record)

    if n_too_short > 0:
        logger.warning(
            "Permutation entropy: %d channel-epochs shorter than min_samples=%d; returning NaN for those entries.",
            n_too_short,
            min_samples,
        )

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_sample_entropy_features(
    epochs: mne.Epochs, config: Any, logger: Any, *, m: int = 2, r_multiplier: float = 0.2, max_samples: int = 500
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract sample entropy per epoch/channel. Lower = more regular."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for sample entropy extraction")
        return pd.DataFrame(), []

    m = int(config.get("feature_engineering.complexity.sampen_m", m))
    r_multiplier = float(config.get("feature_engineering.complexity.sampen_r_mult", r_multiplier))
    max_samples = int(config.get("feature_engineering.complexity.sampen_max_samples", max_samples))
    min_samples = int(config.get("feature_engineering.complexity.min_samples_for_entropy", MIN_SAMPLES_FOR_ENTROPY))
    if m + 2 > min_samples:
        min_samples = m + 2

    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []
    n_too_short = 0

    for epoch_idx, epoch in enumerate(data):
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            if len(ch_data) < min_samples:
                record[f"sampen_{ch_name}"] = np.nan
                n_too_short += 1
                continue
            if len(ch_data) > max_samples:
                sample_idx = np.linspace(0, len(ch_data) - 1, num=max_samples, dtype=int)
                sample_idx = np.unique(sample_idx)  # avoid duplicate indices when len ~ max_samples
                ch_data = ch_data[sample_idx]
                if epoch_idx == 0 and ch_idx == 0:
                    logger.info(
                        "Downsampled channel data to %d samples for sample entropy (uniform coverage, computational efficiency)",
                        len(ch_data),
                    )
            
            sampen = _sample_entropy(ch_data, m=m, r_multiplier=r_multiplier)
            record[f"sampen_{ch_name}"] = sampen

        sampen_vals = [v for v in record.values() if np.isfinite(v)]
        record["sampen_global_mean"] = float(np.mean(sampen_vals)) if sampen_vals else np.nan

        feature_records.append(record)

    if n_too_short > 0:
        logger.warning(
            "Sample entropy: %d channel-epochs shorter than min_samples=%d; returning NaN for those entries.",
            n_too_short,
            min_samples,
        )

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_hjorth_parameters(epochs: mne.Epochs, config: Any, logger: Any) -> Tuple[pd.DataFrame, List[str]]:
    """Extract Hjorth parameters (Activity, Mobility, Complexity) per epoch."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for Hjorth parameter extraction")
        return pd.DataFrame(), []
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            activity, mobility, complexity = _hjorth_parameters(ch_data)

            record[f"hjorth_activity_{ch_name}"] = activity
            record[f"hjorth_mobility_{ch_name}"] = mobility
            record[f"hjorth_complexity_{ch_name}"] = complexity

        activity_vals = [v for k, v in record.items() if "activity" in k and np.isfinite(v)]
        mobility_vals = [v for k, v in record.items() if "mobility" in k and np.isfinite(v)]
        complexity_vals = [v for k, v in record.items() if "complexity" in k and np.isfinite(v)]

        record["hjorth_activity_global"] = float(np.mean(activity_vals)) if activity_vals else np.nan
        record["hjorth_mobility_global"] = float(np.mean(mobility_vals)) if mobility_vals else np.nan
        record["hjorth_complexity_global"] = float(np.mean(complexity_vals)) if complexity_vals else np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_lempel_ziv_complexity(epochs: mne.Epochs, config: Any, logger: Any, *, normalize: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Extract Lempel-Ziv complexity per epoch/channel. Higher = more random."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for Lempel-Ziv complexity extraction")
        return pd.DataFrame(), []

    min_samples = int(config.get("feature_engineering.complexity.min_samples_for_entropy", MIN_SAMPLES_FOR_ENTROPY))
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []
    n_too_short = 0

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            if len(ch_data) < min_samples:
                record[f"lzc_{ch_name}"] = np.nan
                n_too_short += 1
                continue
            lzc = _lempel_ziv_complexity(ch_data, normalize=normalize)
            record[f"lzc_{ch_name}"] = lzc

        lzc_vals = [v for v in record.values() if np.isfinite(v)]
        record["lzc_global_mean"] = float(np.mean(lzc_vals)) if lzc_vals else np.nan

        feature_records.append(record)

    if n_too_short > 0:
        logger.warning(
            "Lempel-Ziv complexity: %d channel-epochs shorter than min_samples=%d; returning NaN for those entries.",
            n_too_short,
            min_samples,
        )

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_all_complexity_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract all complexity features in one call.

    Includes: Permutation entropy, Hjorth parameters, Lempel-Ziv complexity.
    Note: Sample entropy is excluded by default due to computational cost.

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
        DataFrame with all complexity features and column names
    """
    all_dfs: List[pd.DataFrame] = []
    all_cols: List[str] = []

    include_sampen = bool(config.get("feature_engineering.complexity.include_sample_entropy", False))

    pe_df, pe_cols = extract_permutation_entropy_features(epochs, config, logger)
    if not pe_df.empty:
        all_dfs.append(pe_df)
        all_cols.extend(pe_cols)

    if include_sampen:
        sampen_df, sampen_cols = extract_sample_entropy_features(epochs, config, logger)
        if not sampen_df.empty:
            all_dfs.append(sampen_df)
            all_cols.extend(sampen_cols)

    hjorth_df, hjorth_cols = extract_hjorth_parameters(epochs, config, logger)
    if not hjorth_df.empty:
        all_dfs.append(hjorth_df)
        all_cols.extend(hjorth_cols)

    lzc_df, lzc_cols = extract_lempel_ziv_complexity(epochs, config, logger)
    if not lzc_df.empty:
        all_dfs.append(lzc_df)
        all_cols.extend(lzc_cols)

    if not all_dfs:
        return pd.DataFrame(), []

    combined = pd.concat(all_dfs, axis=1)
    return combined, all_cols


###################################################################
# Band-Filtered Complexity Features
###################################################################


# Use shared bandpass filter from core
from eeg_pipeline.analysis.features.core import bandpass_filter_epochs


def extract_band_permutation_entropy_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract permutation entropy per frequency band.

    Features are computed on band-filtered signals, giving features like:
    - pe_alpha_Cz, pe_beta_Cz, etc.

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
    order : int
        Embedding dimension (pattern length). Default 3.
    delay : int
        Time delay between samples in pattern. Default 1.
    normalize : bool
        If True, normalize to [0, 1]. Default True.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with band-specific PE features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for band PE extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    order = int(config.get("feature_engineering.complexity.pe_order", order))
    delay = int(config.get("feature_engineering.complexity.pe_delay", delay))
    min_samples = int(config.get("feature_engineering.complexity.min_samples_for_entropy", MIN_SAMPLES_FOR_ENTROPY))
    if (order - 1) * delay + 1 > min_samples:
        min_samples = (order - 1) * delay + 1

    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []
    n_too_short = 0

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

            pe_vals = []
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]
                if len(ch_data) < min_samples:
                    record[f"pe_{band}_{ch_name}"] = np.nan
                    n_too_short += 1
                    continue
                pe = _permutation_entropy(ch_data, order=order, delay=delay, normalize=normalize)
                record[f"pe_{band}_{ch_name}"] = pe
                if np.isfinite(pe):
                    pe_vals.append(pe)

            record[f"pe_{band}_global_mean"] = float(np.mean(pe_vals)) if pe_vals else np.nan
            record[f"pe_{band}_global_std"] = float(np.std(pe_vals)) if len(pe_vals) > 1 else np.nan

    if n_too_short > 0:
        logger.warning(
            "Band permutation entropy: %d channel-epochs shorter than min_samples=%d; returning NaN for those entries.",
            n_too_short,
            min_samples,
        )

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_band_hjorth_parameters(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Hjorth parameters per frequency band.

    Features are computed on band-filtered signals, giving features like:
    - hjorth_activity_alpha_Cz, hjorth_mobility_alpha_Cz, hjorth_complexity_alpha_Cz, etc.

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
        DataFrame with band-specific Hjorth features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for band Hjorth extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []
    min_samples = int(config.get("feature_engineering.complexity.min_samples_for_entropy", MIN_SAMPLES_FOR_ENTROPY))
    n_too_short = 0

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

            act_vals, mob_vals, comp_vals = [], [], []
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]
                if len(ch_data) < min_samples:
                    record[f"hjorth_activity_{band}_{ch_name}"] = np.nan
                    record[f"hjorth_mobility_{band}_{ch_name}"] = np.nan
                    record[f"hjorth_complexity_{band}_{ch_name}"] = np.nan
                    n_too_short += 1
                    continue
                activity, mobility, complexity = _hjorth_parameters(ch_data)

                record[f"hjorth_activity_{band}_{ch_name}"] = activity
                record[f"hjorth_mobility_{band}_{ch_name}"] = mobility
                record[f"hjorth_complexity_{band}_{ch_name}"] = complexity

                if np.isfinite(activity):
                    act_vals.append(activity)
                if np.isfinite(mobility):
                    mob_vals.append(mobility)
                if np.isfinite(complexity):
                    comp_vals.append(complexity)

            record[f"hjorth_activity_{band}_global"] = float(np.mean(act_vals)) if act_vals else np.nan
            record[f"hjorth_mobility_{band}_global"] = float(np.mean(mob_vals)) if mob_vals else np.nan
            record[f"hjorth_complexity_{band}_global"] = float(np.mean(comp_vals)) if comp_vals else np.nan

    if n_too_short > 0:
        logger.warning(
            "Band Hjorth parameters: %d channel-epochs shorter than min_samples=%d; returning NaN for those entries.",
            n_too_short,
            min_samples,
        )

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_band_lempel_ziv_complexity(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Lempel-Ziv complexity per frequency band.

    Features are computed on band-filtered signals, giving features like:
    - lzc_alpha_Cz, lzc_beta_Cz, etc.

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
        DataFrame with band-specific LZC features and column names
    """
    if not bands:
        return pd.DataFrame(), []

    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for band LZC extraction")
        return pd.DataFrame(), []

    min_samples = int(config.get("feature_engineering.complexity.min_samples_for_entropy", MIN_SAMPLES_FOR_ENTROPY))
    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    feature_records: List[Dict[str, float]] = []
    n_too_short = 0

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

            lzc_vals = []
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]
                if len(ch_data) < min_samples:
                    record[f"lzc_{band}_{ch_name}"] = np.nan
                    n_too_short += 1
                    continue
                lzc = _lempel_ziv_complexity(ch_data)
                record[f"lzc_{band}_{ch_name}"] = lzc
                if np.isfinite(lzc):
                    lzc_vals.append(lzc)

            record[f"lzc_{band}_global_mean"] = float(np.mean(lzc_vals)) if lzc_vals else np.nan
            record[f"lzc_{band}_global_std"] = float(np.std(lzc_vals)) if len(lzc_vals) > 1 else np.nan

    if n_too_short > 0:
        logger.warning(
            "Band Lempel-Ziv complexity: %d channel-epochs shorter than min_samples=%d; returning NaN for those entries.",
            n_too_short,
            min_samples,
        )

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_all_band_complexity_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract all complexity features per frequency band.

    Includes: permutation entropy, Hjorth parameters, and Lempel-Ziv complexity
    for each band.

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
        DataFrame with all band-specific complexity features and column names
    """
    all_dfs: List[pd.DataFrame] = []
    all_cols: List[str] = []

    pe_df, pe_cols = extract_band_permutation_entropy_features(epochs, bands, config, logger)
    if not pe_df.empty:
        all_dfs.append(pe_df)
        all_cols.extend(pe_cols)

    hjorth_df, hjorth_cols = extract_band_hjorth_parameters(epochs, bands, config, logger)
    if not hjorth_df.empty:
        all_dfs.append(hjorth_df)
        all_cols.extend(hjorth_cols)

    lzc_df, lzc_cols = extract_band_lempel_ziv_complexity(epochs, bands, config, logger)
    if not lzc_df.empty:
        all_dfs.append(lzc_df)
        all_cols.extend(lzc_cols)

    if not all_dfs:
        return pd.DataFrame(), []

    combined = pd.concat(all_dfs, axis=1)
    return combined, all_cols
