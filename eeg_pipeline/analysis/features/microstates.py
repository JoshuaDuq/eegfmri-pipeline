from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import mne
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from scipy import stats

from eeg_pipeline.utils.io.general import (
    get_column_from_config,
    get_pain_column_from_config,
)
from eeg_pipeline.utils.analysis.stats import (
    compute_correlation_for_metric_state,
    compute_duration_p_value,
    extract_pain_masks,
    extract_duration_data,
    fdr_bh,
)
from eeg_pipeline.utils.data.loading import extract_epoch_data
from eeg_pipeline.utils.config.loader import (
    get_config_int,
    get_config_float,
    load_settings,
)


###################################################################
# Microstate Feature Extraction
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


def compute_gfp_with_floor(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Global Field Power with epsilon floor for microstate analysis.
    
    This version adds an epsilon floor to prevent division by zero
    in microstate template matching. For general GFP computation, use
    `eeg_pipeline.analysis.features.core.compute_gfp`.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (channels, times) - single epoch
    eps : float
        Epsilon floor value
        
    Returns
    -------
    np.ndarray
        GFP values with shape (times,)
    """
    if data.size == 0:
        return np.array([])
    if data.ndim < 2:
        raise ValueError(f"compute_gfp_with_floor requires at least 2D array, got shape {data.shape}")
    gfp = np.std(data, axis=0)
    # Prevent division by zero in downstream template matching
    gfp = np.where(gfp < eps, eps, gfp)
    return gfp


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
    
    # For z-scored maps (mean=0, std=1), correlation = dot product / (n_samples - 1)
    # Using n_samples - 1 (Bessel's correction) provides unbiased correlation estimate
    # since zscore_maps uses ddof=0 (population std) for standardization
    denominator = max(n_samples - 1, 1)
    return (maps_a @ maps_b.T) / denominator


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
    
    gfp = compute_gfp_with_floor(epoch)
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
    
    # --- Fix for Polarity Invariance ---
    # Standard KMeans distinguishes between Map and -Map, but EEG microstates are polarity invariant.
    # We align all maps so that the channel with the largest absolute amplitude is positive.
    # This acts as a heuristic to map all dipoles to the same hemisphere of the hypersphere.
    n_maps = all_peak_maps.shape[0]
    logger = logging.getLogger(__name__)
    
    # Check if user wants polarity-invariant clustering (experimental)
    use_polarity_invariant = bool(config.get(
        "feature_engineering.microstates.polarity_invariant_kmeans", False
    ))
    
    if use_polarity_invariant:
        logger.info(
            f"Using polarity-invariant K-means for {n_maps} peak maps. "
            "Note: This is experimental and may produce different results than standard alignment."
        )
        # For polarity-invariant clustering, we augment the data with both polarities
        # and then deduplicate cluster centers
        all_peak_maps_augmented = np.vstack([all_peak_maps, -all_peak_maps])
        kmeans_n_init = get_config_int(config, "feature_engineering.constants.kmeans_n_init", 20)
        kmeans = KMeans(n_clusters=int(n_states * 2), n_init=kmeans_n_init, random_state=random_state)
        kmeans.fit(all_peak_maps_augmented)
        # Deduplicate by keeping only positive-polarity representatives
        centers = kmeans.cluster_centers_
        unique_centers = []
        for center in centers:
            max_idx = np.argmax(np.abs(center))
            if center[max_idx] < 0:
                center = -center
            # Check if this center is already in unique_centers (within tolerance)
            is_duplicate = False
            for existing in unique_centers:
                if np.corrcoef(center, existing)[0, 1] > 0.95:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_centers.append(center)
        # Take top n_states by variance explained
        unique_centers = np.array(unique_centers[:n_states])
        if len(unique_centers) < n_states:
            logger.warning(
                f"Polarity-invariant K-means produced only {len(unique_centers)} unique states "
                f"(requested {n_states}). Falling back to standard alignment."
            )
            use_polarity_invariant = False
        else:
            return zscore_maps(unique_centers, axis=1)
    
    logger.info(f"Aligning polarity of {n_maps} peak maps for standard KMeans...")
    
    # Find index of max absolute value for each map
    max_indices = np.argmax(np.abs(all_peak_maps), axis=1)
    # Get the sign of the value at that index
    # efficient indexing: all_peak_maps[row, col]
    max_values = all_peak_maps[np.arange(n_maps), max_indices]
    signs = np.sign(max_values)
    # Handle zero sign if any (unlikely for z-scored data) by treating as 1
    signs[signs == 0] = 1
    
    # Multiply each map by its sign to flip it if necessary
    all_peak_maps = all_peak_maps * signs[:, np.newaxis]
    # -----------------------------------

    if all_peak_maps.shape[0] < n_states:
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
    p_values: Optional[np.ndarray] = None
    q_values: Optional[np.ndarray] = None


@dataclass
class MicrostateDurationStat:
    state: str
    nonpain: np.ndarray
    pain: np.ndarray
    p_value: float
    q_value: float


def _reorder_templates_to_picks(
    templates: np.ndarray,
    template_ch_names: Optional[List[str]],
    picks: np.ndarray,
    info: Any,
    logger: Any,
) -> Optional[np.ndarray]:
    if template_ch_names is None:
        logger.error(
            "Fixed microstate templates provided without channel names; cannot align to current data."
        )
        return None
    
    if templates.ndim != 2:
        logger.error(f"Expected templates as 2D array (states x channels); got shape {templates.shape}")
        return None
    
    if templates.shape[1] != len(template_ch_names):
        logger.error(
            "Template channel count (%d) does not match provided channel names (%d).",
            templates.shape[1], len(template_ch_names),
        )
        return None
    
    pick_names = [info["ch_names"][i] for i in picks]
    missing = [ch for ch in pick_names if ch not in template_ch_names]
    if missing:
        logger.error(
            "Fixed templates missing %d channels required by current data: %s",
            len(missing), ", ".join(missing),
        )
        return None
    
    reorder_indices = [template_ch_names.index(ch) for ch in pick_names]
    aligned = templates[:, reorder_indices]
    return aligned


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
    min_correlation_samples = config.get("feature_engineering.constants.min_correlation_samples", 15)
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
    qvals = corr.copy()

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
                qvals.at[metric_label, state_label] = p

    if not qvals.empty:
        flat_p = qvals.to_numpy(dtype=float).ravel()
        q_vals_flat = fdr_bh(flat_p, config=config)
        q_vals = q_vals_flat.reshape(qvals.shape)
        qvals = pd.DataFrame(q_vals, index=qvals.index, columns=qvals.columns)

    return corr, qvals


def _compute_transition_mean(trans_vals: pd.Series, mask: np.ndarray) -> float:
    data = trans_vals[mask].to_numpy(dtype=float)
    finite_data = data[np.isfinite(data)]
    return np.mean(finite_data) if finite_data.size > 0 else 0.0


def _compute_transition_p_value(trans_vals: pd.Series, nonpain_mask: np.ndarray, pain_mask: np.ndarray, min_trials: int) -> float:
    nonpain = trans_vals[nonpain_mask].to_numpy(dtype=float)
    pain = trans_vals[pain_mask].to_numpy(dtype=float)
    nonpain = nonpain[np.isfinite(nonpain)]
    pain = pain[np.isfinite(pain)]
    if nonpain.size < min_trials or pain.size < min_trials:
        return np.nan
    try:
        _, p_val = stats.mannwhitneyu(nonpain, pain, alternative="two-sided")
        return float(p_val) if np.isfinite(p_val) else np.nan
    except ValueError:
        return np.nan


def compute_microstate_transition_stats(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    n_states: int,
    config: Any,
) -> MicrostateTransitionStats:
    trans_nonpain = np.zeros((n_states, n_states), dtype=float)
    trans_pain = np.zeros((n_states, n_states), dtype=float)
    p_mat = np.full((n_states, n_states), np.nan, dtype=float)
    state_labels = _state_labels(n_states, config)
    min_trials = int(config.get("behavior_analysis.statistics.min_transition_samples", 5)) if hasattr(config, "get") else 5

    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        return MicrostateTransitionStats(trans_nonpain, trans_pain, state_labels, p_mat, p_mat)

    pain_col = get_pain_column_from_config(config, events_df)
    if pain_col is None:
        return MicrostateTransitionStats(trans_nonpain, trans_pain, state_labels, p_mat, p_mat)

    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    nonpain_mask, pain_mask = extract_pain_masks(pain_vals)

    p_values_flat: List[float] = []
    coords: List[Tuple[int, int]] = []

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
            p_val = _compute_transition_p_value(trans_vals, nonpain_mask, pain_mask, min_trials)
            p_mat[i, j] = p_val
            if np.isfinite(p_val):
                p_values_flat.append(p_val)
                coords.append((i, j))

    if p_values_flat:
        q_flat = fdr_bh(p_values_flat, config=config)
        q_mat = np.full_like(p_mat, np.nan, dtype=float)
        for (i, j), q in zip(coords, q_flat):
            q_mat[i, j] = float(q)
    else:
        q_mat = p_mat

    return MicrostateTransitionStats(trans_nonpain, trans_pain, state_labels, p_mat, q_mat)


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
    p_values: List[float] = []

    # First pass: collect per-state p-values
    state_records: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
    for idx, state_label in enumerate(_state_labels(n_states, config)):
        col = f"ms_duration_{idx}"
        if col not in ms_df.columns:
            continue
        
        durations = pd.to_numeric(ms_df[col], errors="coerce")
        nonpain_data = extract_duration_data(durations, nonpain_mask)
        pain_data = extract_duration_data(durations, pain_mask)
        p_value = compute_duration_p_value(nonpain_data, pain_data)

        state_records.append((state_label, nonpain_data, pain_data, p_value))
        p_values.append(p_value)

    if not state_records:
        return []

    # Apply FDR across states
    q_values = fdr_bh(p_values, config=config)

    for (state_label, nonpain_data, pain_data, p_value), q_value in zip(state_records, q_values):
        stats_list.append(
            MicrostateDurationStat(
                state=state_label,
                nonpain=nonpain_data,
                pain=pain_data,
                p_value=p_value,
                q_value=float(q_value),
            )
        )

    return stats_list


def _build_microstate_column_names(n_states: int) -> List[str]:
    column_names = ["ms_valid_fraction"]
    metrics = ("coverage", "duration", "occurrence", "gev")
    for metric in metrics:
        for state_idx in range(n_states):
            column_names.append(f"ms_{metric}_{state_idx}")
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                column_names.append(f"ms_trans_{i}_to_{j}")
    return column_names


def _compute_coverage_metrics(
    state_labels: np.ndarray,
    n_states: int,
    record: Dict[str, float],
    total_samples: Optional[int] = None,
) -> None:
    state_counts = np.bincount(state_labels, minlength=n_states).astype(float)
    denom = float(total_samples) if total_samples is not None and total_samples > 0 else float(state_labels.size)
    if denom <= 0:
        coverage = np.zeros(n_states, dtype=float)
    else:
        coverage = state_counts / denom
    for state_idx in range(n_states):
        record[f"ms_coverage_{state_idx}"] = float(coverage[state_idx])


def _compute_transition_metrics(
    state_labels: np.ndarray,
    sfreq: float,
    n_states: int,
    record: Dict[str, float],
    epoch_duration_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if state_labels.size == 0:
        return np.zeros(n_states, dtype=float), np.zeros(n_states, dtype=float), 0.0
    
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
    
    epoch_duration = epoch_duration_override if epoch_duration_override is not None else float(state_labels.size) / sfreq
    if epoch_duration <= 0:
        return np.zeros(n_states, dtype=float), np.zeros(n_states, dtype=float), 0.0
    
    occurrences = np.zeros(n_states, dtype=float)
    dwell_times = np.zeros(n_states, dtype=float)
    trans_counts = np.zeros((n_states, n_states), dtype=float)
    prev_state = int(state_labels[0])
    run_length = 1

    for sample_idx in range(1, state_labels.size):
        current_state = int(state_labels[sample_idx])
        if current_state == prev_state:
            run_length += 1
            continue
        
        occurrences[prev_state] += 1.0
        dwell_times[prev_state] += run_length / sfreq
        trans_counts[prev_state, current_state] += 1.0
        prev_state = current_state
        run_length = 1
    
    occurrences[prev_state] += 1.0
    dwell_times[prev_state] += run_length / sfreq
    
    # Store transition rates (per second) rather than raw counts
    trans_rates = trans_counts / epoch_duration
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            trans_key = f"ms_trans_{i}_to_{j}"
            if trans_key in record:
                record[trans_key] = float(trans_rates[i, j])
    
    return occurrences, dwell_times, epoch_duration


def _merge_short_runs(state_labels: np.ndarray, min_run_samples: int) -> np.ndarray:
    """
    Collapse runs shorter than min_run_samples into neighboring states to avoid
    spurious transitions driven by isolated samples.
    """
    if state_labels.size == 0 or min_run_samples <= 1:
        return state_labels

    merged: List[int] = []
    run_start = 0
    while run_start < state_labels.size:
        current = int(state_labels[run_start])
        run_end = run_start
        while run_end + 1 < state_labels.size and state_labels[run_end + 1] == current:
            run_end += 1

        run_len = run_end - run_start + 1
        if run_len >= min_run_samples:
            merged.extend([current] * run_len)
        else:
            prev_state = merged[-1] if merged else None
            next_state = int(state_labels[run_end + 1]) if run_end + 1 < state_labels.size else None
            if prev_state is not None:
                merged.extend([prev_state] * run_len)
            elif next_state is not None:
                merged.extend([next_state] * run_len)

        run_start = run_end + 1

    return np.asarray(merged, dtype=int)


def _compute_duration_occurrence_metrics(
    occurrences: np.ndarray,
    dwell_times: np.ndarray,
    n_states: int,
    record: Dict[str, float],
    epoch_duration: float,
) -> None:
    for state_idx in range(n_states):
        occurrence_count = occurrences[state_idx]
        occurrence_rate = float(occurrence_count / epoch_duration) if epoch_duration > 0 else 0.0
        record[f"ms_occurrence_{state_idx}"] = occurrence_rate
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
    config: Any,
    logger: Any = None,
    trial_id: Optional[str] = None,
) -> Dict[str, float]:
    record = {col: 0.0 for col in column_names}
    state_labels_full, correlation_values_full = label_timecourse(trial_data, templates)
    
    if state_labels_full.size == 0:
        return record

    # Filter noisy samples: require sufficient GFP and map correlation
    config_local = config or load_settings()
    corr_min = float(config_local.get("feature_engineering.microstates.min_correlation", 0.25))
    gfp_percentile = float(config_local.get("feature_engineering.microstates.gfp_min_percentile", 25.0))
    gfp_full = compute_gfp_with_floor(trial_data)
    gfp_threshold = float(np.nanpercentile(gfp_full, gfp_percentile)) if gfp_full.size else 0.0
    valid_mask = (np.abs(correlation_values_full) >= corr_min) & (gfp_full >= gfp_threshold)
    if not np.any(valid_mask):
        return record

    total_samples = trial_data.shape[1]
    valid_samples = int(valid_mask.sum())
    valid_fraction = float(valid_samples) / float(total_samples) if total_samples > 0 else 0.0
    record["ms_valid_fraction"] = valid_fraction
    min_valid_fraction = float(config_local.get("feature_engineering.microstates.min_valid_fraction", 0.5))
    if valid_fraction < min_valid_fraction:
        if logger:
            logger.warning(
                "Microstate QC: valid fraction %.3f below minimum %.3f; marking trial as unusable%s",
                valid_fraction,
                min_valid_fraction,
                f" (trial {trial_id})" if trial_id else "",
            )
        for key in record:
            if key != "ms_valid_fraction":
                record[key] = np.nan
        return record

    # Restrict to valid samples and recompute correlations so labels and correlations stay aligned
    trial_data_valid = trial_data[:, valid_mask]
    state_labels = state_labels_full[valid_mask]
    gfp = gfp_full[valid_mask]

    if trial_data_valid.shape[1] <= 1:
        if logger:
            logger.warning("Insufficient valid samples after microstate QC; skipping metrics for this trial.")
        return record

    zscored_valid = zscore_maps(trial_data_valid.T, axis=1)
    correlations_matrix = corr_maps(zscored_valid, templates)

    # Enforce minimum run length (merge very short segments into neighbors)
    min_run_ms = float(config_local.get("feature_engineering.microstates.min_run_ms", 10.0))
    min_run_samples = max(1, int((min_run_ms / 1000.0) * sfreq))
    state_labels = _merge_short_runs(state_labels, min_run_samples)
    if state_labels.size == 0:
        return record

    # Keep correlations/GFP aligned to merged labels (truncate to common length if needed)
    max_len = min(state_labels.size, correlations_matrix.shape[0], gfp.shape[0])
    state_labels = state_labels[:max_len]
    correlations_matrix = correlations_matrix[:max_len]
    gfp = gfp[:max_len]

    correlation_values = correlations_matrix[np.arange(state_labels.size), state_labels]
    
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

    valid_samples_post = state_labels.size
    epoch_duration_valid = float(valid_samples_post) / sfreq if (sfreq > 0 and valid_samples_post > 0) else 0.0
    epoch_duration_full = float(total_samples) / sfreq if (sfreq > 0 and total_samples > 0) else 0.0

    _compute_coverage_metrics(
        state_labels, n_states, record,
        total_samples=total_samples if total_samples > 0 else None,
    )
    occurrences, dwell_times, epoch_duration = _compute_transition_metrics(
        state_labels, sfreq, n_states, record,
        epoch_duration_override=epoch_duration_full if epoch_duration_full > 0 else None,
    )
    _compute_duration_occurrence_metrics(
        occurrences,
        dwell_times,
        n_states,
        record,
        epoch_duration_full if epoch_duration_full > 0 else epoch_duration_valid,
    )
    
    if gfp_energy_valid:
        _compute_gev_metrics(state_labels, correlation_values, gfp, gfp_energy, n_states, record)
    else:
        for state_idx in range(n_states):
            record[f"ms_gev_{state_idx}"] = 0.0

    return record


def extract_microstate_features(
    epochs: Any, n_states: int, config: Any, logger: Any,
    fixed_templates: Optional[np.ndarray] = None,
    fixed_template_ch_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[np.ndarray]]:
    """
    Extract microstate features (coverage, duration, etc.) for each epoch.
    
    Args:
        epochs: MNE Epochs object
        n_states: Number of microstate classes (k)
        config: Configuration dict/object
        logger: Logger instance
        fixed_templates: Optional (n_channels, n_states) array of fixed templates to use.
                         If provided, clustering is skipped and these templates are used directly.
                         This is CRITICAL for group-level consistency if individual subject templates are not aligned.
    
    Returns:
        Tuple (DataFrame of features, list of column names, templates used)
    """
    if n_states <= 0:
        logger.warning(f"Invalid number of states ({n_states}); must be positive. Returning empty features.")
        return pd.DataFrame(), [], None
    
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for microstate feature extraction")
        return pd.DataFrame(), [], None

    # Enforce average reference for microstate analysis (standard practice)
    # Use a copy to avoid side-effects on the main processing pipeline
    logger.info("Applying average reference for microstate analysis...")
    epochs_micro = epochs.copy()
    epochs_micro.load_data()
    epochs_micro.set_eeg_reference('average', projection=False, verbose=False)

    epoch_data = extract_epoch_data(epochs_micro, picks)
    if epoch_data.size == 0:
        logger.warning("Epoch data empty after picking EEG channels; skipping microstate features")
        return pd.DataFrame(), [], None

    sfreq = float(epochs.info.get("sfreq", 1.0))
    
    if fixed_templates is not None:
        logger.info("Using provided fixed microstate templates (skipping individual clustering).")
        if fixed_template_ch_names is None:
            logger.error("Fixed templates provided without channel names; cannot ensure alignment. Skipping microstate features.")
            return pd.DataFrame(), [], None
        aligned_templates = _reorder_templates_to_picks(
            fixed_templates, list(fixed_template_ch_names), picks, epochs.info, logger
        )
        if aligned_templates is None:
            return pd.DataFrame(), [], None
        templates = aligned_templates
    else:
        logger.info(f"Computing individual microstate templates (k={n_states}) via K-Means...")
        templates = extract_templates_from_trials(epoch_data, sfreq, n_states, config)
        
        # WARNING for Scientific Validity
        logger.warning(
            "CAUTION: Microstate templates computed individually for this subject. "
            "State labels (0, 1, 2...) may NOT correspond across subjects. "
            "Group-level aggregation of these features without template alignment is scientifically invalid."
        )

    if templates is None or templates.shape[0] == 0:
        logger.warning("Failed to derive microstate templates; skipping microstate features")
        return pd.DataFrame(), [], None

    column_names = _build_microstate_column_names(n_states)
    feature_rows = []
    
    for trial_idx, trial in enumerate(epoch_data):
        record = _compute_microstate_metrics_for_trial(
            trial, templates, sfreq, n_states, column_names, config,
            logger=logger, trial_id=str(trial_idx)
        )
        feature_rows.append(record)

    ms_df = pd.DataFrame.from_records(feature_rows, columns=column_names)
    return ms_df, column_names, templates

