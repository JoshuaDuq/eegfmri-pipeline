from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
from math import factorial

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.config.loader import get_frequency_bands


###################################################################
# Complexity Feature Extraction
###################################################################


def _embed_time_series(x: np.ndarray, order: int, delay: int) -> np.ndarray:
    n = len(x)
    if n < (order - 1) * delay + 1:
        return np.array([])
    
    n_vectors = n - (order - 1) * delay
    embedded = np.zeros((n_vectors, order))
    
    for i in range(n_vectors):
        for j in range(order):
            embedded[i, j] = x[i + j * delay]
    
    return embedded


def _permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    if len(x) < (order - 1) * delay + 1:
        return np.nan
    
    embedded = _embed_time_series(x, order, delay)
    if embedded.size == 0:
        return np.nan
    
    n_vectors = embedded.shape[0]
    perm_counts: Dict[tuple, int] = {}
    
    for i in range(n_vectors):
        pattern = tuple(np.argsort(embedded[i]))
        perm_counts[pattern] = perm_counts.get(pattern, 0) + 1
    
    probs = np.array(list(perm_counts.values())) / n_vectors
    probs = probs[probs > 0]
    pe = -np.sum(probs * np.log2(probs))
    
    if normalize:
        max_entropy = np.log2(factorial(order))
        if max_entropy > 0:
            pe = pe / max_entropy
    
    return float(pe)


def _sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    r_multiplier: float = 0.2,
) -> float:
    """
    Compute sample entropy using vectorized Chebyshev distance.
    
    Optimized to avoid O(n²) nested Python loops by using NumPy broadcasting.
    """
    n = len(x)
    if n < m + 2:
        return np.nan
    
    if r is None:
        std = np.std(x, ddof=1)
        if std == 0:
            return np.nan
        r = r_multiplier * std
    
    def count_matches_vectorized(template_length: int) -> int:
        """Count template matches using vectorized Chebyshev distance."""
        n_templates = n - template_length
        if n_templates < 2:
            return 0
        
        # Build template matrix: (n_templates, template_length)
        templates = np.array([x[i:i + template_length] for i in range(n_templates)])
        
        # Compute pairwise Chebyshev distances using broadcasting
        # diff[i, j, k] = templates[i, k] - templates[j, k]
        diff = templates[:, np.newaxis, :] - templates[np.newaxis, :, :]  # (n, n, m)
        chebyshev = np.max(np.abs(diff), axis=2)  # (n, n)
        
        # Count pairs where distance < r (upper triangle only, excluding diagonal)
        triu_idx = np.triu_indices(n_templates, k=1)
        matches = np.sum(chebyshev[triu_idx] < r)
        return int(matches)
    
    a = count_matches_vectorized(m + 1)
    b = count_matches_vectorized(m)
    
    if b == 0:
        return np.nan
    
    return float(-np.log(a / b)) if a > 0 else np.nan


def _hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    
    diff1 = np.diff(x)
    diff2 = np.diff(diff1)
    
    var_x = np.var(x)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    
    if var_x == 0:
        return np.nan, np.nan, np.nan
    
    activity = float(var_x)
    
    if var_x > 0:
        mobility = float(np.sqrt(var_d1 / var_x))
    else:
        mobility = np.nan
    
    if var_d1 > 0 and mobility > 0:
        complexity = float(np.sqrt(var_d2 / var_d1) / mobility)
    else:
        complexity = np.nan
    
    return activity, mobility, complexity


def _lempel_ziv_complexity(
    x: np.ndarray,
    threshold: Optional[float] = None,
    normalize: bool = True,
) -> float:
    if len(x) < 2:
        return np.nan
    
    if threshold is None:
        threshold = np.median(x)
    
    binary = (x > threshold).astype(int)
    s = "".join(map(str, binary))
    n = len(s)
    
    if n == 0:
        return np.nan
    
    i = 0
    c = 1
    l = 1
    k = 1
    k_max = 1
    
    while i + k <= n:
        if s[i:i + k] in s[:i + l - 1]:
            k += 1
            k_max = max(k, k_max)
        else:
            c += 1
            i += k_max
            l = 1
            k = 1
            k_max = 1
        
        if i + k > n:
            break
    
    lzc = float(c)
    
    if normalize and n > 1:
        b = n / np.log2(n)
        lzc = lzc / b
    
    return lzc


def extract_permutation_entropy_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract permutation entropy for each epoch and channel.

    Permutation entropy measures the complexity/irregularity of time series
    based on the ordinal patterns of consecutive values. It is robust to noise
    and computational efficient.

    Lower PE = more regular/predictable signal
    Higher PE = more complex/irregular signal

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    order : int
        Embedding dimension (pattern length). Default 3.
        Higher values capture longer-range patterns but need more data.
    delay : int
        Time delay between samples in pattern. Default 1.
    normalize : bool
        If True, normalize to [0, 1]. Default True.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with PE features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for permutation entropy extraction")
        return pd.DataFrame(), []

    order = int(config.get("feature_engineering.complexity.pe_order", order))
    delay = int(config.get("feature_engineering.complexity.pe_delay", delay))

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            pe = _permutation_entropy(ch_data, order=order, delay=delay, normalize=normalize)
            record[f"pe_{ch_name}"] = pe

        pe_vals = [v for v in record.values() if np.isfinite(v)]
        record["pe_global_mean"] = float(np.mean(pe_vals)) if pe_vals else np.nan
        record["pe_global_std"] = float(np.std(pe_vals)) if len(pe_vals) > 1 else np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_sample_entropy_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    m: int = 2,
    r_multiplier: float = 0.2,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract sample entropy for each epoch and channel.

    Sample entropy measures time series regularity/predictability.
    More robust than approximate entropy for short time series.

    Lower SampEn = more regular (e.g., sine wave)
    Higher SampEn = more irregular/complex

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    m : int
        Embedding dimension. Default 2.
    r_multiplier : float
        Tolerance as fraction of std. Default 0.2 (i.e., r = 0.2 * std)

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with SampEn features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for sample entropy extraction")
        return pd.DataFrame(), []

    m = int(config.get("feature_engineering.complexity.sampen_m", m))
    r_multiplier = float(config.get("feature_engineering.complexity.sampen_r_mult", r_multiplier))

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch_idx, epoch in enumerate(data):
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            
            if len(ch_data) > 500:
                ch_data = ch_data[:500]
                if epoch_idx == 0 and ch_idx == 0:
                    logger.info("Truncating data to 500 samples for sample entropy (computational efficiency)")
            
            sampen = _sample_entropy(ch_data, m=m, r_multiplier=r_multiplier)
            record[f"sampen_{ch_name}"] = sampen

        sampen_vals = [v for v in record.values() if np.isfinite(v)]
        record["sampen_global_mean"] = float(np.mean(sampen_vals)) if sampen_vals else np.nan

        feature_records.append(record)

    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


def extract_hjorth_parameters(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Hjorth parameters (Activity, Mobility, Complexity) for each epoch.

    Activity: Variance of the signal (related to power)
    Mobility: Mean frequency (std of derivative / std of signal)
    Complexity: Bandwidth (change in frequency)

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
        DataFrame with Hjorth parameters and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for Hjorth parameter extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
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


def extract_lempel_ziv_complexity(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Lempel-Ziv complexity for each epoch and channel.

    LZC measures the number of distinct patterns in a binarized signal.
    Higher complexity = more unique patterns = more random/complex.
    Lower complexity = more repetitive patterns.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    normalize : bool
        If True, normalize by theoretical maximum. Default True.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with LZC features and column names
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for Lempel-Ziv complexity extraction")
        return pd.DataFrame(), []

    ch_names = [epochs.info["ch_names"][p] for p in picks]
    data = epochs.get_data(picks=picks)

    feature_records: List[Dict[str, float]] = []

    for epoch in data:
        record: Dict[str, float] = {}

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = epoch[ch_idx]
            lzc = _lempel_ziv_complexity(ch_data, normalize=normalize)
            record[f"lzc_{ch_name}"] = lzc

        lzc_vals = [v for v in record.values() if np.isfinite(v)]
        record["lzc_global_mean"] = float(np.mean(lzc_vals)) if lzc_vals else np.nan

        feature_records.append(record)

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

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for band PE extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    order = int(config.get("feature_engineering.complexity.pe_order", order))
    delay = int(config.get("feature_engineering.complexity.pe_delay", delay))

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

            pe_vals = []
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]
                pe = _permutation_entropy(ch_data, order=order, delay=delay, normalize=normalize)
                record[f"pe_{band}_{ch_name}"] = pe
                if np.isfinite(pe):
                    pe_vals.append(pe)

            record[f"pe_{band}_global_mean"] = float(np.mean(pe_vals)) if pe_vals else np.nan
            record[f"pe_{band}_global_std"] = float(np.std(pe_vals)) if len(pe_vals) > 1 else np.nan

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

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for band Hjorth extraction")
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

            act_vals, mob_vals, comp_vals = [], [], []
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]
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

    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for band LZC extraction")
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

            lzc_vals = []
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = epoch[ch_idx]
                lzc = _lempel_ziv_complexity(ch_data)
                record[f"lzc_{band}_{ch_name}"] = lzc
                if np.isfinite(lzc):
                    lzc_vals.append(lzc)

            record[f"lzc_{band}_global_mean"] = float(np.mean(lzc_vals)) if lzc_vals else np.nan
            record[f"lzc_{band}_global_std"] = float(np.std(lzc_vals)) if len(lzc_vals) > 1 else np.nan

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

