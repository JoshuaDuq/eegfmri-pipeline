from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import mne
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

from eeg_pipeline.utils.tfr_utils import (
    time_mask,
    freq_mask,
    extract_tfr_object,
    extract_band_power,
    process_temporal_bin,
)
from eeg_pipeline.utils.io_utils import (
    get_column_from_config,
    get_pain_column_from_config,
    find_connectivity_arrays,
    load_connectivity_labels,
)
from eeg_pipeline.utils.stats_utils import (
    compute_correlation_for_metric_state,
    compute_duration_p_value,
    fit_aperiodic_to_all_epochs,
    compute_residuals,
    extract_pain_masks,
    extract_duration_data,
)
from eeg_pipeline.utils.data_loading import (
    flatten_lower_triangles,
    align_feature_blocks,
    extract_epoch_data,
)
from eeg_pipeline.utils.config_loader import (
    get_config_value,
    get_config_int,
    get_config_float,
    get_frequency_bands,
    parse_temporal_bin_config,
    get_default_frequency_bands,
    get_frequency_bands_for_aperiodic,
    load_settings,
)


###################################################################
# Microstate Core Functions
###################################################################


def zscore_maps(maps: np.ndarray, axis: int = 1, eps: Optional[float] = None) -> np.ndarray:
    if maps.size == 0:
        return maps
    
    if eps is None:
        config_local = load_settings()
        eps = config_local.get("feature_engineering.constants.epsilon_std", 1e-12)
        eps = float(eps)
    
    if axis < 0 or axis >= maps.ndim:
        raise ValueError(f"Invalid axis {axis} for array with {maps.ndim} dimensions")
    
    mu = np.mean(maps, axis=axis, keepdims=True)
    sd = np.std(maps, axis=axis, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (maps - mu) / sd


def compute_gfp(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.array([])
    if data.ndim < 2:
        raise ValueError(f"compute_gfp requires at least 2D array, got shape {data.shape}")
    return np.std(data, axis=0)


def corr_maps(maps_a: np.ndarray, maps_b: np.ndarray) -> np.ndarray:
    if maps_a.size == 0 or maps_b.size == 0:
        raise ValueError("corr_maps requires non-empty arrays")
    
    if maps_a.ndim != 2 or maps_b.ndim != 2:
        raise ValueError(f"corr_maps requires 2D arrays, got shapes {maps_a.shape} and {maps_b.shape}")
    
    if maps_a.shape[1] != maps_b.shape[1]:
        raise ValueError(
            f"corr_maps requires matching second dimension, got {maps_a.shape[1]} and {maps_b.shape[1]}"
        )
    
    n_samples = maps_a.shape[1]
    if n_samples <= 1:
        raise ValueError(f"corr_maps requires at least 2 samples, got {n_samples}")
    
    return (maps_a @ maps_b.T) / (n_samples - 1)


def label_timecourse(channel_data: np.ndarray, templates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if channel_data.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    if templates.size == 0 or templates.shape[0] == 0:
        raise ValueError("label_timecourse requires non-empty templates")
    
    if channel_data.ndim != 2:
        raise ValueError(f"channel_data must be 2D, got shape {channel_data.shape}")
    
    if templates.ndim != 2:
        raise ValueError(f"templates must be 2D, got shape {templates.shape}")
    
    if channel_data.shape[0] != templates.shape[1]:
        raise ValueError(
            f"Dimension mismatch: channel_data has {channel_data.shape[0]} channels, "
            f"but templates expect {templates.shape[1]} channels"
        )
    
    zscored = zscore_maps(channel_data.T, axis=1)
    correlations = corr_maps(zscored, templates)
    state_indices = np.argmax(np.abs(correlations), axis=1)
    assigned_correlations = correlations[np.arange(correlations.shape[0]), state_indices]
    return state_indices.astype(int), assigned_correlations.astype(float)


def _extract_peak_maps_from_epoch(
    epoch: np.ndarray,
    min_dist_samples: int,
    peaks_per_epoch: int,
) -> Optional[np.ndarray]:
    if epoch.size == 0:
        return None
    
    if epoch.ndim != 2:
        return None
    
    if min_dist_samples < 1:
        min_dist_samples = 1
    
    if peaks_per_epoch < 1:
        return None
    
    gfp = compute_gfp(epoch)
    if gfp.size == 0 or np.allclose(gfp, 0):
        return None
    
    peaks, _ = find_peaks(gfp, distance=min_dist_samples)
    if peaks.size == 0:
        return None
    
    sorted_indices = np.argsort(gfp[peaks])[::-1]
    n_select = min(peaks_per_epoch, peaks.size)
    selected_peaks = peaks[sorted_indices[:n_select]]
    epoch_maps = epoch[:, selected_peaks].T
    return zscore_maps(epoch_maps, axis=1)


def extract_templates_from_trials(
    trial_data: np.ndarray,
    sfreq: float,
    n_states: int,
    config: Any,
) -> Optional[np.ndarray]:
    if n_states <= 0:
        raise ValueError(f"Number of states must be positive, got {n_states}")
    
    min_peak_distance_ms = get_config_float(
        config, "feature_engineering.microstates.min_peak_distance_ms", 20.0
    )
    peaks_per_epoch = get_config_int(
        config, "feature_engineering.microstates.peaks_per_epoch", 5
    )
    random_state = get_config_int(config, "random.seed", 42)

    min_dist_samples = max(1, int((min_peak_distance_ms / 1000.0) * sfreq))
    peak_maps: List[np.ndarray] = []
    
    for epoch in trial_data:
        epoch_maps = _extract_peak_maps_from_epoch(epoch, min_dist_samples, peaks_per_epoch)
        if epoch_maps is not None:
            peak_maps.append(epoch_maps)
    
    if not peak_maps:
        return None
    
    all_peak_maps = np.vstack(peak_maps)
    if all_peak_maps.shape[0] < n_states:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Insufficient peak maps ({all_peak_maps.shape[0]}) for requested number of states ({n_states}); "
            "cannot fit KMeans. Returning None."
        )
        return None
    
    kmeans_n_init = get_config_int(config, "feature_engineering.constants.kmeans_n_init", 20)
    kmeans = KMeans(n_clusters=int(n_states), n_init=kmeans_n_init, random_state=random_state)
    kmeans.fit(all_peak_maps)
    templates = kmeans.cluster_centers_
    return zscore_maps(templates, axis=1)


def _state_labels(n_states: int, config: Any = None) -> List[str]:
    if config is None:
        config = load_settings()
    ascii_uppercase_a = config.get("feature_engineering.constants.ascii_uppercase_a", 65)
    max_ascii = 127
    if ascii_uppercase_a + n_states > max_ascii:
        return [f"State_{i}" for i in range(n_states)]
    return [chr(ascii_uppercase_a + i) for i in range(n_states)]


@dataclass
class MicrostateTransitionStats:
    nonpain: np.ndarray
    pain: np.ndarray
    state_labels: List[str]


@dataclass
class MicrostateDurationStat:
    state: str
    nonpain: np.ndarray
    pain: np.ndarray
    p_value: float


def _detect_number_of_states(ms_df: pd.DataFrame) -> int:
    state_indices = (
        int(col.split("_")[-1])
        for col in ms_df.columns
        if col.startswith("ms_") and col.rsplit("_", 1)[-1].isdigit()
    )
    max_idx = max(state_indices, default=-1)
    return max_idx + 1 if max_idx >= 0 else 0


def compute_microstate_metric_correlations(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    config: Any,
    metrics: Optional[List[str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    method: str = "spearman",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rating_col = get_column_from_config(config, "event_columns.rating", events_df)
    if rating_col is None:
        return pd.DataFrame(), pd.DataFrame()

    ratings = pd.to_numeric(events_df[rating_col], errors="coerce")
    min_correlation_samples = config.get("feature_engineering.constants.min_correlation_samples", 5)
    if ratings.notna().sum() < min_correlation_samples:
        return pd.DataFrame(), pd.DataFrame()

    metrics = metrics or ["coverage", "duration", "occurrence", "gev"]
    metric_labels = metric_labels or {
        "coverage": "Coverage",
        "duration": "Duration",
        "occurrence": "Occurrence",
        "gev": "GEV",
    }

    n_states_detected = _detect_number_of_states(ms_df)
    state_labels = _state_labels(n_states_detected, config) if n_states_detected > 0 else []

    corr = pd.DataFrame(
        np.nan,
        index=[metric_labels.get(m, m.capitalize()) for m in metrics],
        columns=state_labels,
    )
    pvals = corr.copy()

    for metric in metrics:
        metric_label = metric_labels.get(metric, metric.capitalize())
        for state_idx, state_label in enumerate(state_labels):
            col = f"ms_{metric}_{state_idx}"
            if col not in ms_df.columns:
                continue
            
            metric_vals = pd.to_numeric(ms_df[col], errors="coerce")
            result = compute_correlation_for_metric_state(metric_vals, ratings, method)
            if result is not None:
                r, p = result
                corr.at[metric_label, state_label] = r
                pvals.at[metric_label, state_label] = p

    return corr, pvals


def _compute_transition_mean(trans_vals: pd.Series, mask: np.ndarray) -> float:
    data = trans_vals[mask].to_numpy(dtype=float)
    finite_data = data[np.isfinite(data)]
    return np.mean(finite_data) if finite_data.size > 0 else 0.0


def compute_microstate_transition_stats(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    n_states: int,
    config: Any,
) -> MicrostateTransitionStats:
    trans_nonpain = np.zeros((n_states, n_states), dtype=float)
    trans_pain = np.zeros((n_states, n_states), dtype=float)
    state_labels = _state_labels(n_states, config)

    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        return MicrostateTransitionStats(trans_nonpain, trans_pain, state_labels)

    pain_col = get_pain_column_from_config(config, events_df)
    if pain_col is None:
        return MicrostateTransitionStats(trans_nonpain, trans_pain, state_labels)

    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    nonpain_mask, pain_mask = extract_pain_masks(pain_vals)

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            col = f"ms_trans_{i}_to_{j}"
            if col not in ms_df.columns:
                continue
            
            trans_vals = pd.to_numeric(ms_df[col], errors="coerce")
            trans_nonpain[i, j] = _compute_transition_mean(trans_vals, nonpain_mask)
            trans_pain[i, j] = _compute_transition_mean(trans_vals, pain_mask)

    return MicrostateTransitionStats(trans_nonpain, trans_pain, state_labels)


def compute_microstate_duration_stats(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    n_states: int,
    config: Any,
) -> List[MicrostateDurationStat]:
    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        return []

    pain_col = get_pain_column_from_config(config, events_df)
    if pain_col is None:
        return []

    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    nonpain_mask, pain_mask = extract_pain_masks(pain_vals)
    stats_list: List[MicrostateDurationStat] = []

    for idx, state_label in enumerate(_state_labels(n_states, config)):
        col = f"ms_duration_{idx}"
        if col not in ms_df.columns:
            continue
        
        durations = pd.to_numeric(ms_df[col], errors="coerce")
        nonpain_data = extract_duration_data(durations, nonpain_mask)
        pain_data = extract_duration_data(durations, pain_mask)
        p_value = compute_duration_p_value(nonpain_data, pain_data)

        stats_list.append(
            MicrostateDurationStat(
                state=state_label,
                nonpain=nonpain_data,
                pain=pain_data,
                p_value=p_value,
            )
        )

    return stats_list


###################################################################
# Feature Extraction Functions
###################################################################


def _prepare_tfr_power_inputs(tfr: Any, config: Any, logger: Any) -> Optional[Tuple[Any, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    tfr_obj = extract_tfr_object(tfr)
    if tfr_obj is None:
        return None

    tfr_data = tfr_obj.data
    if tfr_data.ndim != 4:
        raise RuntimeError("TFR data must have 4D shape (epochs, channels, freqs, times)")

    freqs = tfr_obj.freqs
    times = tfr_obj.times
    channel_names = tfr_obj.info["ch_names"]
    
    return tfr_obj, tfr_data, freqs, times, channel_names


def extract_baseline_power_features(
    tfr: Any, bands: List[str], baseline_window: Tuple[float, float], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    if not bands:
        return pd.DataFrame(), []
    
    result = _prepare_tfr_power_inputs(tfr, config, logger)
    if result is None:
        return pd.DataFrame(), []
    
    tfr_obj, tfr_data, freqs, times, channel_names = result
    
    baseline_start, baseline_end = baseline_window
    baseline_mask = time_mask(times, baseline_start, baseline_end)
    if not np.any(baseline_mask):
        logger.warning(f"No time points in baseline window ({baseline_start}, {baseline_end})")
        return pd.DataFrame(), []
    
    frequency_bands = get_frequency_bands(config)
    feature_arrays = []
    column_names = []
    
    for band in bands:
        if band not in frequency_bands:
            continue
        
        fmin, fmax = frequency_bands[band]
        band_power = extract_band_power(tfr_data, freqs, fmin, fmax, baseline_mask)
        if band_power is None:
            continue
        
        feature_arrays.append(band_power)
        column_names.extend([f"baseline_{band}_{ch}" for ch in channel_names])

    if not feature_arrays:
        return pd.DataFrame(), []

    return pd.DataFrame(np.concatenate(feature_arrays, axis=1)), column_names


def extract_band_power_features(
    tfr: Any, bands: List[str], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    if not bands:
        return pd.DataFrame(), []
    
    result = _prepare_tfr_power_inputs(tfr, config, logger)
    if result is None:
        return pd.DataFrame(), []
    
    tfr_obj, tfr_data, freqs, times, channel_names = result
    
    frequency_bands = get_frequency_bands(config)
    temporal_bins = get_config_value(config, "feature_engineering.features.temporal_bins", [])
    feature_arrays = []
    column_names = []
    
    for band in bands:
        if band not in frequency_bands:
            logger.warning(f"Band '{band}' not defined in config; skipping")
            continue
        
        fmin, fmax = frequency_bands[band]
        freq_mask_indices = freq_mask(freqs, fmin, fmax)
        if not np.any(freq_mask_indices):
            logger.warning(f"TFR freqs contain no points in band '{band}' ({fmin}-{fmax} Hz); skipping")
            continue
        
        for bin_config in temporal_bins:
            bin_params = parse_temporal_bin_config(bin_config)
            if bin_params is None:
                logger.warning(f"Invalid temporal bin configuration: {bin_config}; skipping")
                continue
            
            time_start, time_end, time_label = bin_params
            result = process_temporal_bin(
                tfr_data, freqs, times, channel_names,
                band, fmin, fmax, time_start, time_end, time_label, logger
            )
            if result is not None:
                band_power, cols = result
                feature_arrays.append(band_power)
                column_names.extend(cols)

    if not feature_arrays:
        return pd.DataFrame(), []

    return pd.DataFrame(np.concatenate(feature_arrays, axis=1)), column_names


def _load_connectivity_block(
    path: Optional[Path],
    labels: Optional[np.ndarray],
    prefix: str,
    logger: Any,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    if path is None or not path.exists():
        logger.debug(f"Connectivity file missing for {prefix}: {path}")
        return None
    
    try:
        arr = np.load(path)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load connectivity file {path}: {e}")
        return None
    
    if arr.ndim != 3:
        logger.warning(f"Unexpected connectivity shape at {path}: {arr.shape}")
        return None
    
    df_flat, cols = flatten_lower_triangles(arr, labels, prefix=prefix)
    return df_flat, cols


def extract_connectivity_features(
    subject: str, task: str, bands: List[str], deriv_root: Path, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    subj_dir = deriv_root / f"sub-{subject}" / "eeg"
    if not subj_dir.exists():
        return pd.DataFrame(), []

    labels = load_connectivity_labels(subject, task, deriv_root)
    all_blocks: List[pd.DataFrame] = []
    all_cols: List[str] = []
    missing_files: List[str] = []

    for band in bands:
        aec_path, wpli_path = find_connectivity_arrays(subject, task, band, deriv_root)
        
        for measure, path in (("aec", aec_path), ("wpli", wpli_path)):
            result = _load_connectivity_block(path, labels, f"{measure}_{band}", logger)
            if result is not None:
                df_flat, cols = result
                all_blocks.append(df_flat)
                all_cols.extend(cols)
            else:
                missing_files.append(f"{measure}_{band}")

    if not all_blocks:
        if missing_files:
            logger.debug(f"No connectivity files found for subject {subject}, task {task}. Missing: {', '.join(missing_files)}")
        return pd.DataFrame(), []

    aligned_blocks = align_feature_blocks(all_blocks)
    if not aligned_blocks:
        return pd.DataFrame(), []

    combined_df = pd.concat(aligned_blocks, axis=1)
    if len(combined_df.columns) != len(all_cols):
        logger.warning(
            f"Column count mismatch: DataFrame has {len(combined_df.columns)} columns "
            f"but {len(all_cols)} column names provided. Using DataFrame column names."
        )
        return combined_df, list(combined_df.columns)
    
    combined_df.columns = all_cols
    return combined_df, all_cols


def _build_microstate_column_names(n_states: int) -> List[str]:
    column_names = []
    metrics = ("coverage", "duration", "occurrence", "gev")
    for metric in metrics:
        for state_idx in range(n_states):
            column_names.append(f"ms_{metric}_{state_idx}")
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                column_names.append(f"ms_trans_{i}_to_{j}")
    return column_names


def _compute_coverage_metrics(state_labels: np.ndarray, n_states: int, record: Dict[str, float]) -> None:
    state_counts = np.bincount(state_labels, minlength=n_states).astype(float)
    coverage = state_counts / float(state_labels.size) if state_labels.size > 0 else np.zeros(n_states, dtype=float)
    for state_idx in range(n_states):
        record[f"ms_coverage_{state_idx}"] = float(coverage[state_idx])


def _compute_transition_metrics(
    state_labels: np.ndarray,
    sfreq: float,
    n_states: int,
    record: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    if state_labels.size == 0:
        return np.zeros(n_states, dtype=float), np.zeros(n_states, dtype=float)
    
    if sfreq <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {sfreq}")
    
    valid_mask = (state_labels >= 0) & (state_labels < n_states)
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        logging.getLogger(__name__).warning(
            f"Found {invalid_count} invalid state labels (out of range [0, {n_states})). "
            "Clipping to valid range."
        )
        state_labels = np.clip(state_labels, 0, n_states - 1)
    
    occurrences = np.zeros(n_states, dtype=float)
    dwell_times = np.zeros(n_states, dtype=float)
    prev_state = int(state_labels[0])
    run_length = 1

    for sample_idx in range(1, state_labels.size):
        current_state = int(state_labels[sample_idx])
        if current_state == prev_state:
            run_length += 1
            continue
        
        occurrences[prev_state] += 1.0
        dwell_times[prev_state] += run_length / sfreq
        trans_key = f"ms_trans_{prev_state}_to_{current_state}"
        if trans_key in record:
            record[trans_key] += 1.0
        prev_state = current_state
        run_length = 1
    
    occurrences[prev_state] += 1.0
    dwell_times[prev_state] += run_length / sfreq
    return occurrences, dwell_times


def _compute_duration_occurrence_metrics(
    occurrences: np.ndarray,
    dwell_times: np.ndarray,
    n_states: int,
    record: Dict[str, float],
) -> None:
    for state_idx in range(n_states):
        occurrence_count = occurrences[state_idx]
        record[f"ms_occurrence_{state_idx}"] = float(occurrence_count)
        record[f"ms_duration_{state_idx}"] = float(dwell_times[state_idx] / occurrence_count) if occurrence_count > 0 else 0.0


def _compute_gev_metrics(
    state_labels: np.ndarray,
    correlation_values: np.ndarray,
    gfp: np.ndarray,
    gfp_energy: float,
    n_states: int,
    record: Dict[str, float],
) -> None:
    for state_idx in range(n_states):
        state_mask = state_labels == state_idx
        if not np.any(state_mask):
            record[f"ms_gev_{state_idx}"] = 0.0
            continue
        gev_numerator = np.sum((correlation_values[state_mask] ** 2) * (gfp[state_mask] ** 2))
        record[f"ms_gev_{state_idx}"] = float(gev_numerator / gfp_energy)


def _compute_microstate_metrics_for_trial(
    trial_data: np.ndarray,
    templates: np.ndarray,
    sfreq: float,
    n_states: int,
    column_names: List[str],
    logger: Any = None,
    trial_id: Optional[str] = None,
) -> Dict[str, float]:
    record = {col: 0.0 for col in column_names}
    state_labels, correlation_values = label_timecourse(trial_data, templates)
    
    if state_labels.size == 0:
        return record

    gfp = compute_gfp(trial_data)
    gfp_energy = float(np.sum(gfp ** 2))
    
    gfp_energy_valid = np.isfinite(gfp_energy) and gfp_energy > 0
    if not gfp_energy_valid:
        trial_info = f" (trial: {trial_id})" if trial_id else ""
        log_msg = (
            f"Invalid GFP energy ({gfp_energy}) for microstate metrics{trial_info}; "
            "skipping GEV computation (coverage/duration still computed)."
        )
        if logger:
            logger.warning(log_msg)
        else:
            logging.getLogger(__name__).warning(log_msg)
        gfp_energy = 0.0

    _compute_coverage_metrics(state_labels, n_states, record)
    occurrences, dwell_times = _compute_transition_metrics(state_labels, sfreq, n_states, record)
    _compute_duration_occurrence_metrics(occurrences, dwell_times, n_states, record)
    
    if gfp_energy_valid:
        _compute_gev_metrics(state_labels, correlation_values, gfp, gfp_energy, n_states, record)
    else:
        for state_idx in range(n_states):
            record[f"ms_gev_{state_idx}"] = 0.0

    return record


def extract_microstate_features(
    epochs: Any, n_states: int, config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str], Optional[np.ndarray]]:
    if n_states <= 0:
        logger.warning(f"Invalid number of states ({n_states}); must be positive. Returning empty features.")
        return pd.DataFrame(), [], None
    
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for microstate feature extraction")
        return pd.DataFrame(), [], None

    epoch_data = extract_epoch_data(epochs, picks)
    if epoch_data.size == 0:
        logger.warning("Epoch data empty after picking EEG channels; skipping microstate features")
        return pd.DataFrame(), [], None

    sfreq = float(epochs.info.get("sfreq", 1.0))
    templates = extract_templates_from_trials(epoch_data, sfreq, n_states, config)
    if templates is None or templates.shape[0] == 0:
        logger.warning("Failed to derive microstate templates; skipping microstate features")
        return pd.DataFrame(), [], None

    column_names = _build_microstate_column_names(n_states)
    feature_rows = []
    
    for trial_idx, trial in enumerate(epoch_data):
        record = _compute_microstate_metrics_for_trial(
            trial, templates, sfreq, n_states, column_names, logger=logger, trial_id=str(trial_idx)
        )
        feature_rows.append(record)

    ms_df = pd.DataFrame.from_records(feature_rows, columns=column_names)
    return ms_df, column_names, templates


def _build_aperiodic_feature_records(
    offsets: np.ndarray,
    slopes: np.ndarray,
    residuals: np.ndarray,
    freqs: np.ndarray,
    channel_names: List[str],
    bands: List[str],
    freq_bands: Dict[str, List[float]],
) -> List[Dict[str, float]]:
    if offsets.shape[0] == 0:
        return []
    
    if offsets.shape[0] != slopes.shape[0] or offsets.shape[0] != residuals.shape[0]:
        raise ValueError(
            f"Shape mismatch: offsets={offsets.shape[0]}, slopes={slopes.shape[0]}, "
            f"residuals={residuals.shape[0]} epochs"
        )
    
    if offsets.shape[1] != len(channel_names) or slopes.shape[1] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: offsets/slopes have {offsets.shape[1]} channels, "
            f"but {len(channel_names)} channel names provided"
        )
    
    if residuals.ndim != 3 or residuals.shape[1] != len(channel_names):
        raise ValueError(
            f"Residuals shape mismatch: expected (n_epochs, n_channels, n_freqs), "
            f"got {residuals.shape}"
        )
    
    band_masks = {}
    for band in bands:
        if band in freq_bands:
            fmin, fmax = freq_bands[band]
            band_masks[band] = (freqs >= fmin) & (freqs <= fmax)
    
    n_epochs = offsets.shape[0]
    feature_records = []
    
    for epoch_idx in range(n_epochs):
        record = {}
        for channel_idx, channel_name in enumerate(channel_names):
            record[f"aper_slope_{channel_name}"] = float(slopes[epoch_idx, channel_idx])
            record[f"aper_offset_{channel_name}"] = float(offsets[epoch_idx, channel_idx])
        
        for band, mask in band_masks.items():
            if not np.any(mask):
                for channel_name in channel_names:
                    record[f"powcorr_{band}_{channel_name}"] = np.nan
            else:
                for channel_idx, channel_name in enumerate(channel_names):
                    band_residual_mean = np.mean(residuals[epoch_idx, channel_idx, mask])
                    record[f"powcorr_{band}_{channel_name}"] = float(band_residual_mean)
        
        feature_records.append(record)
    
    return feature_records


def _adjust_baseline_end(baseline_end: float) -> float:
    return 0.0 if baseline_end > 0 else baseline_end


def extract_aperiodic_features(
    epochs: Any,
    baseline_window: Tuple[float, float],
    bands: List[str],
    config: Any,
    logger: Any,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    n_fft: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for aperiodic feature extraction")
        return pd.DataFrame(), []
    
    baseline_start, baseline_end = baseline_window
    baseline_end = _adjust_baseline_end(baseline_end)
    freq_bands = get_frequency_bands_for_aperiodic(config)
    
    if fmin is None:
        fmin = config.get("feature_engineering.constants.aperiodic_fmin", 2.0)
    if fmax is None:
        fmax = config.get("feature_engineering.constants.aperiodic_fmax", 40.0)
    
    psds, freqs = mne.time_frequency.psd_welch(
        epochs,
        picks=picks,
        fmin=float(fmin),
        fmax=float(fmax),
        tmin=baseline_start,
        tmax=baseline_end,
        n_fft=n_fft,
        n_overlap=None,
        average='mean',
        verbose=False,
    )
    
    if psds.ndim != 3:
        logger.error(f"Unexpected PSD shape: {psds.shape}")
        return pd.DataFrame(), []
    
    log_freqs = np.log10(freqs)
    epsilon_psd = config.get("feature_engineering.constants.epsilon_psd", 1e-20)
    epsilon_psd = float(epsilon_psd)
    log_psd = np.log10(np.maximum(psds, epsilon_psd))
    
    offsets, slopes = fit_aperiodic_to_all_epochs(log_freqs, log_psd)
    residuals = compute_residuals(log_freqs, log_psd, offsets, slopes)
    
    channel_names = [epochs.info["ch_names"][p] for p in picks]
    feature_records = _build_aperiodic_feature_records(
        offsets, slopes, residuals, freqs, channel_names, bands, freq_bands
    )
    
    feature_df = pd.DataFrame(feature_records)
    column_names = list(feature_df.columns)
    return feature_df, column_names

