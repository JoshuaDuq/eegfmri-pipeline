"""
Phase Feature Extraction
=========================

Phase-based features for EEG analysis:
- ITPC: Inter-Trial Phase Coherence (global across trials; leakage-safe)
- PAC: Phase-Amplitude Coupling (cross-frequency coupling)
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Any, Dict
import logging
import numpy as np
import pandas as pd
import mne
from scipy.signal import find_peaks

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.utils.analysis.windowing import make_mask_for_times, get_segment_masks
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.parallel import get_n_jobs

# Constants
_EPSILON_COMPLEX = 1e-12
_MIN_PEAKS_FOR_SHARPNESS = 2
_MIN_EPOCHS_FOR_ITPC = 2
_MIN_TIMES_FOR_SURROGATES = 3
_MIN_CHANNELS_FOR_PARALLEL = 4  # Lower threshold for better parallelization

# --- Helpers ---

def _safe_config_get(config: Any, key: str, default: Any = None) -> Any:
    """Safely get value from config, handling dict-like and object-like configs.
    
    If default is a dict and the result is not a dict, returns default.
    This ensures type consistency when expecting dict values.
    """
    if config is None:
        return default
    if hasattr(config, "get"):
        result = config.get(key, default)
    elif isinstance(config, dict):
        result = config.get(key, default)
    else:
        result = default
    
    if isinstance(default, dict) and not isinstance(result, dict):
        return default
    return result

def _get_itpc_method(config: Any) -> str:
    """Get ITPC computation method from config.
    
    Supported methods:
    - 'global': Compute ITPC across all trials, broadcast to each trial.
                WARNING: Creates cross-trial dependence; can leak in CV/machine learning.
    - 'fold_global': Compute ITPC from training trials only (requires ctx.train_mask),
                     broadcast to all trials. This is leakage-safe for CV/machine learning.
    - 'condition': Compute ITPC per condition group (avoids pseudo-replication for
                   condition-level analyses). Requires condition_column in config.
    - 'loo': Leave-one-out ITPC (per-trial). Requires train_mask and explicit opt-in.
    """
    method = str(_safe_config_get(config, "feature_engineering.itpc.method", "fold_global")).strip().lower()
    if method not in {"loo", "global", "fold_global", "condition"}:
        raise ValueError(
            "Invalid ITPC method. Supported values are 'loo', 'global', 'fold_global', and 'condition'. "
            f"Got: {method!r}"
        )
    if method == "loo" and not bool(_safe_config_get(config, "feature_engineering.itpc.allow_unsafe_loo", False)):
        raise ValueError(
            "ITPC method 'loo' is disabled by default because it creates cross-trial dependence "
            "and can cause leakage in trial-level analyses. Set "
            "feature_engineering.itpc.allow_unsafe_loo=true only if you compute features "
            "within CV folds and pass an explicit training mask."
        )
    return method


def _compute_condition_itpc_map(
    data: np.ndarray,
    condition_labels: np.ndarray,
    *,
    train_mask: Optional[np.ndarray],
    min_trials_per_condition: int,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute per-trial ITPC maps by condition.

    Returns an array with shape (n_epochs, n_ch, n_freqs, n_times) where all
    trials in the same condition receive the same ITPC map (computed within the
    condition). If train_mask is provided, the ITPC map for each condition is
    computed using only training trials in that condition.
    """
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError(f"Expected complex TFR data with 4 dims, got shape={data.shape}.")
    n_epochs = int(data.shape[0])
    if condition_labels is None:
        raise ValueError("ITPC(method='condition') requires condition labels.")
    labels = np.asarray(condition_labels)
    if labels.shape[0] != n_epochs:
        raise ValueError(
            f"Condition label length ({labels.shape[0]}) does not match n_epochs ({n_epochs})."
        )

    valid_labels = np.ones(n_epochs, dtype=bool)
    if labels.dtype.kind in {"f"}:
        valid_labels = np.isfinite(labels.astype(float))

    itpc_map = np.full((n_epochs,) + tuple(data.shape[1:]), np.nan, dtype=np.float32)
    for cond in np.unique(labels[valid_labels]):
        cond_mask = (labels == cond) & valid_labels
        if train_mask is None:
            cond_train = cond_mask
        else:
            tm = np.asarray(train_mask, dtype=bool)
            if tm.shape[0] != n_epochs:
                raise ValueError(
                    f"train_mask length ({tm.shape[0]}) does not match n_epochs ({n_epochs})."
                )
            cond_train = cond_mask & tm

        n_train = int(np.sum(cond_train))
        if n_train < int(min_trials_per_condition):
            if logger is not None:
                logger.warning(
                    "ITPC(condition): condition '%s' has only %d training trials (<%d); output set to NaN.",
                    cond,
                    n_train,
                    int(min_trials_per_condition),
                )
            continue

        cond_map = _compute_fold_global_itpc_map(data, cond_train, n_jobs=n_jobs, logger=logger)
        itpc_map[cond_mask] = cond_map[None, ...]

    return itpc_map


def _compute_peak_distance(
    sfreq_hz: float,
    fmax_hz: Optional[float],
) -> Optional[int]:
    """Compute minimum distance between peaks based on frequency constraints."""
    if fmax_hz is None:
        return None
    try:
        fmax_hz = float(fmax_hz)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid fmax_hz for peak distance computation: {fmax_hz!r}") from exc
    if not (np.isfinite(fmax_hz) and fmax_hz > 0):
        return None
    half_period_samples = float(sfreq_hz) / (2.0 * float(fmax_hz))
    return max(1, int(round(half_period_samples)))


def _compute_mean_sharpness(
    signal: np.ndarray,
    peak_indices: np.ndarray,
    offset_samples: int,
) -> float:
    """Compute mean sharpness for given peak indices."""
    sharpness_values: List[float] = []
    for idx in peak_indices:
        if idx - offset_samples < 0 or idx + offset_samples >= signal.size:
            continue
        left_diff = abs(signal[idx] - signal[idx - offset_samples])
        right_diff = abs(signal[idx] - signal[idx + offset_samples])
        sharpness_values.append(left_diff + right_diff)
    return float(np.mean(sharpness_values)) if sharpness_values else np.nan


def _sharpness_log_ratio(
    x: np.ndarray,
    sfreq_hz: float,
    offset_ms: float,
    *,
    fmax_hz: Optional[float] = None,
) -> float:
    """Cole/Voytek-style sharpness ratio proxy (log peak sharpness / trough sharpness)."""
    x = np.asarray(x, dtype=float)
    if x.size < 10 or not np.isfinite(x).any() or not np.isfinite(sfreq_hz) or sfreq_hz <= 0:
        return np.nan

    offset_samples = float(offset_ms) * float(sfreq_hz) / 1000.0
    offset_samples = int(round(offset_samples))
    max_offset = (x.size - 1) // 4
    offset_samples = max(1, min(offset_samples, max_offset))

    peak_distance = _compute_peak_distance(sfreq_hz, fmax_hz)

    cleaned_signal = np.nan_to_num(x, nan=np.nanmedian(x))
    peaks, _ = find_peaks(cleaned_signal, distance=peak_distance)
    troughs, _ = find_peaks(-cleaned_signal, distance=peak_distance)
    
    if peaks.size < _MIN_PEAKS_FOR_SHARPNESS or troughs.size < _MIN_PEAKS_FOR_SHARPNESS:
        return np.nan

    peak_sharpness = _compute_mean_sharpness(cleaned_signal, peaks, offset_samples)
    trough_sharpness = _compute_mean_sharpness(cleaned_signal, troughs, offset_samples)
    
    if (np.isfinite(peak_sharpness) and np.isfinite(trough_sharpness) and 
        peak_sharpness > 0 and trough_sharpness > 0):
        return float(np.log(peak_sharpness / trough_sharpness))
    return np.nan

def _normalize_complex_to_unit_vectors(data: np.ndarray) -> np.ndarray:
    """Normalize complex data to unit vectors."""
    return data / (np.abs(data) + _EPSILON_COMPLEX)


###################################################################
# Single-Channel ITPC Workers (for parallel execution)
###################################################################


def _compute_loo_itpc_single_channel(
    ch_data: np.ndarray,
    train_mask: np.ndarray,
    train_indices: np.ndarray,
    n_train: int,
) -> np.ndarray:
    """Compute LOO-ITPC for a single channel."""
    unit_vectors = _normalize_complex_to_unit_vectors(ch_data)
    sum_train = np.sum(unit_vectors[train_mask], axis=0)
    
    n_epochs = ch_data.shape[0]
    result = np.zeros((n_epochs,) + ch_data.shape[1:], dtype=np.float32)
    
    mean_test = sum_train / max(1, n_train)
    result[:] = np.abs(mean_test)
    
    if n_train > 1 and train_indices.size:
        loo_train = (sum_train[None, ...] - unit_vectors[train_indices]) / (n_train - 1)
        result[train_indices] = np.abs(loo_train)
    
    return result


def _compute_global_itpc_single_channel(ch_data: np.ndarray) -> np.ndarray:
    """Compute global ITPC for a single channel."""
    unit_vectors = _normalize_complex_to_unit_vectors(ch_data)
    return np.abs(np.mean(unit_vectors, axis=0)).astype(np.float32)


def _compute_fold_global_itpc_single_channel(
    ch_data: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Compute fold-global ITPC for a single channel."""
    unit_vectors = _normalize_complex_to_unit_vectors(ch_data)
    train_unit_vectors = unit_vectors[train_mask]
    return np.abs(np.mean(train_unit_vectors, axis=0)).astype(np.float32)


###################################################################
# Parallel ITPC Computation
###################################################################


def _compute_loo_itpc(
    data: np.ndarray,
    train_mask: Optional[np.ndarray] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute Leave-One-Out ITPC with optional parallelization.
    
    Args:
        data: Complex TFR of shape (n_epochs, n_ch, n_freqs, n_times)
        train_mask: Boolean mask indicating training trials
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        
    Returns:
        ITPC values of shape (n_epochs, n_ch, n_freqs, n_times)
    """
    n_epochs, n_ch = data.shape[0], data.shape[1]
    
    if n_epochs < _MIN_EPOCHS_FOR_ITPC:
        return np.zeros_like(np.abs(data), dtype=np.float32)
    
    if train_mask is None:
        train_mask = np.ones(n_epochs, dtype=bool)
    n_train = int(np.sum(train_mask))
    
    if n_train < 1:
        return np.zeros_like(np.abs(data), dtype=np.float32)
    
    train_indices = np.flatnonzero(train_mask)
    loo_itpc = np.zeros(data.shape, dtype=np.float32)
    
    use_parallel = n_jobs != 1 and n_ch >= _MIN_CHANNELS_FOR_PARALLEL
    
    if use_parallel:
        from joblib import Parallel, delayed
        from eeg_pipeline.utils.parallel import _normalize_n_jobs
        
        n_jobs_actual = _normalize_n_jobs(n_jobs)
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_loo_itpc_single_channel)(
                data[:, ch], train_mask, train_indices, n_train
            )
            for ch in range(n_ch)
        )
        for ch, result in enumerate(results):
            loo_itpc[:, ch] = result
    else:
        for ch in range(n_ch):
            loo_itpc[:, ch] = _compute_loo_itpc_single_channel(
                data[:, ch], train_mask, train_indices, n_train
            )
    
    return loo_itpc


def _compute_global_itpc_map(data: np.ndarray, n_jobs: int = 1, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Compute ITPC map across epochs with optional parallelization."""
    n_ch = data.shape[1]
    itpc_map = np.zeros(data.shape[1:], dtype=np.float32)
    
    use_parallel = n_jobs != 1 and n_ch >= _MIN_CHANNELS_FOR_PARALLEL
    
    if use_parallel:
        from joblib import Parallel, delayed
        from eeg_pipeline.utils.parallel import _normalize_n_jobs
        
        n_jobs_actual = _normalize_n_jobs(n_jobs)
        if logger is not None:
            logger.debug(f"ITPC: Using parallel computation ({n_jobs_actual} jobs, {n_ch} channels)")
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_global_itpc_single_channel)(data[:, ch])
            for ch in range(n_ch)
        )
        for ch, result in enumerate(results):
            itpc_map[ch] = result
    else:
        if logger is not None and n_jobs != 1:
            logger.debug(f"ITPC: Sequential computation (n_channels={n_ch} < {_MIN_CHANNELS_FOR_PARALLEL})")
        for ch in range(n_ch):
            itpc_map[ch] = _compute_global_itpc_single_channel(data[:, ch])
    
    return itpc_map


def _compute_fold_global_itpc_map(
    data: np.ndarray,
    train_mask: np.ndarray,
    n_jobs: int = 1,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Compute ITPC map from TRAINING trials only with optional parallelization."""
    if train_mask is None or not np.any(train_mask):
        return _compute_global_itpc_map(data, n_jobs=n_jobs, logger=logger)
    
    n_ch = data.shape[1]
    itpc_map = np.zeros(data.shape[1:], dtype=np.float32)
    
    use_parallel = n_jobs != 1 and n_ch >= _MIN_CHANNELS_FOR_PARALLEL
    
    if use_parallel:
        from joblib import Parallel, delayed
        from eeg_pipeline.utils.parallel import _normalize_n_jobs
        
        n_jobs_actual = _normalize_n_jobs(n_jobs)
        if logger is not None:
            logger.debug(f"ITPC: Using parallel computation ({n_jobs_actual} jobs, {n_ch} channels)")
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_fold_global_itpc_single_channel)(data[:, ch], train_mask)
            for ch in range(n_ch)
        )
        for ch, result in enumerate(results):
            itpc_map[ch] = result
    else:
        if logger is not None and n_jobs != 1:
            logger.debug(f"ITPC: Sequential computation (n_channels={n_ch} < {_MIN_CHANNELS_FOR_PARALLEL})")
        for ch in range(n_ch):
            itpc_map[ch] = _compute_fold_global_itpc_single_channel(data[:, ch], train_mask)
    
    return itpc_map


###################################################################
# Precomputed ITPC (3D data: epochs, channels, time)
###################################################################


def _compute_itpc_single_channel_precomputed(
    ch_data: np.ndarray,
    train_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Compute ITPC for a single channel from precomputed complex vectors (3D)."""
    if train_mask is not None and np.any(train_mask):
        return np.abs(np.mean(ch_data[train_mask], axis=0)).astype(np.float32)
    return np.abs(np.mean(ch_data, axis=0)).astype(np.float32)


def _compute_itpc_map_precomputed(
    complex_vectors: np.ndarray,
    train_mask: Optional[np.ndarray],
    n_jobs: int = 1,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Compute ITPC map from precomputed complex vectors with optional parallelization.
    
    Args:
        complex_vectors: Complex unit vectors of shape (n_epochs, n_ch, n_times)
        train_mask: Optional boolean mask for training trials
        n_jobs: Number of parallel jobs
        logger: Optional logger for debug messages
        
    Returns:
        ITPC map of shape (n_ch, n_times)
    """
    n_ch = complex_vectors.shape[1]
    itpc_map = np.zeros((n_ch, complex_vectors.shape[2]), dtype=np.float32)
    
    use_parallel = n_jobs != 1 and n_ch >= _MIN_CHANNELS_FOR_PARALLEL
    
    if use_parallel:
        from joblib import Parallel, delayed
        from eeg_pipeline.utils.parallel import _normalize_n_jobs
        
        n_jobs_actual = _normalize_n_jobs(n_jobs)
        if logger is not None:
            logger.debug(f"ITPC (precomputed): Using parallel computation ({n_jobs_actual} jobs, {n_ch} channels)")
        results = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_itpc_single_channel_precomputed)(
                complex_vectors[:, ch], train_mask
            )
            for ch in range(n_ch)
        )
        for ch, result in enumerate(results):
            itpc_map[ch] = result
    else:
        if logger is not None and n_jobs != 1:
            logger.debug(f"ITPC (precomputed): Sequential computation (n_channels={n_ch} < {_MIN_CHANNELS_FOR_PARALLEL})")
        for ch in range(n_ch):
            itpc_map[ch] = _compute_itpc_single_channel_precomputed(
                complex_vectors[:, ch], train_mask
            )
    
    return itpc_map


def _compute_condition_itpc_precomputed(
    segment_complex: np.ndarray,
    condition_labels: np.ndarray,
    train_mask: Optional[np.ndarray],
    min_trials: int,
    logger: Optional[logging.Logger],
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute per-trial ITPC by condition from precomputed complex vectors.
    
    Args:
        segment_complex: Complex vectors of shape (n_epochs, n_ch, n_times)
        condition_labels: Condition labels for each epoch
        train_mask: Optional training mask
        min_trials: Minimum trials per condition
        logger: Logger instance
        n_jobs: Number of parallel jobs
        
    Returns:
        ITPC values of shape (n_epochs, n_ch)
    """
    n_ep, n_ch, _ = segment_complex.shape
    out = np.full((n_ep, n_ch), np.nan, dtype=np.float32)
    
    labels_arr = np.asarray(condition_labels)
    if labels_arr.shape[0] != n_ep:
        raise ValueError(
            f"Condition label length ({labels_arr.shape[0]}) != n_epochs ({n_ep})."
        )
    
    valid_labels = np.ones(n_ep, dtype=bool)
    if labels_arr.dtype.kind in {"f"}:
        valid_labels = np.isfinite(labels_arr.astype(float))

    tm = None
    if train_mask is not None:
        tm = np.asarray(train_mask, dtype=bool)
        if tm.shape[0] != n_ep:
            raise ValueError(
                f"train_mask length ({tm.shape[0]}) != n_epochs ({n_ep})."
            )

    for cond in np.unique(labels_arr[valid_labels]):
        cond_mask = (labels_arr == cond) & valid_labels
        cond_train = cond_mask if tm is None else (cond_mask & tm)
        n_train = int(np.sum(cond_train))
        
        if n_train < int(min_trials):
            if logger is not None:
                logger.warning(
                    "ITPC(condition): condition '%s' has only %d training trials (<%d); output set to NaN.",
                    cond,
                    n_train,
                    int(min_trials),
                )
            continue
        
        itpc_map = _compute_itpc_map_precomputed(segment_complex, cond_train, n_jobs=n_jobs, logger=logger)
        itpc_ch = np.nanmean(itpc_map, axis=1)
        out[cond_mask] = itpc_ch[None, :]

    return out


def _broadcast_per_trial(values_ch: np.ndarray, n_epochs: int) -> np.ndarray:
    """Broadcast per-channel values to all trials.
    
    Args:
        values_ch: 1D array of values per channel
        n_epochs: Number of epochs to broadcast to
        
    Returns:
        2D array of shape (n_epochs, n_channels) with values repeated
    """
    values_ch = np.asarray(values_ch, dtype=float)
    if values_ch.ndim != 1:
        raise ValueError("Expected 1D channel vector for broadcasting.")
    return np.broadcast_to(values_ch[None, :], (int(n_epochs), int(values_ch.shape[0]))).copy()


def _aggregate_spatial_features(
    feature_matrix: np.ndarray,
    feature_type: str,
    segment_name: str,
    band_or_pair: str,
    stat_type: str,
    ch_names: List[str],
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """Aggregate features across spatial dimensions (channels, ROI, global).
    
    Args:
        feature_matrix: Array of shape (n_epochs, n_channels) with feature values
        feature_type: Feature type name (e.g., "itpc", "pac")
        segment_name: Name of the time segment
        band_or_pair: Frequency band name or band pair name
        stat_type: Statistic type (e.g., "val", "z", "lf_sharpness_ratio")
        ch_names: List of channel names
        spatial_modes: List of spatial aggregation modes to include
        roi_map: Dictionary mapping ROI names to channel indices
        
    Returns:
        Dictionary mapping column names to feature arrays
    """
    results = {}
    
    if 'channels' in spatial_modes:
        for channel_idx, channel_name in enumerate(ch_names):
            column_name = NamingSchema.build(
                feature_type, segment_name, band_or_pair, "ch", stat_type,
                channel=channel_name
            )
            results[column_name] = feature_matrix[:, channel_idx]
    
    if 'roi' in spatial_modes and roi_map:
        for roi_name, channel_indices in roi_map.items():
            if channel_indices:
                roi_values = np.nanmean(feature_matrix[:, channel_indices], axis=1)
                column_name = NamingSchema.build(
                    feature_type, segment_name, band_or_pair, "roi", stat_type,
                    channel=roi_name
                )
                results[column_name] = roi_values
    
    if 'global' in spatial_modes:
        global_values = np.nanmean(feature_matrix, axis=1)
        column_name = NamingSchema.build(
            feature_type, segment_name, band_or_pair, "global", stat_type
        )
        results[column_name] = global_values
    
    return results


def _compute_itpc_map_by_method(
    data: np.ndarray,
    method: str,
    train_mask: Optional[np.ndarray],
    analysis_mode: Optional[str],
    logger: logging.Logger,
    *,
    condition_labels: Optional[np.ndarray] = None,
    min_trials_per_condition: int = 10,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute ITPC map using the specified method.
    
    Args:
        data: Complex TFR data of shape (n_epochs, n_ch, n_freqs, n_times)
        method: ITPC computation method ('loo', 'fold_global', 'global', or 'condition')
        train_mask: Boolean mask indicating training trials (required in trial_ml_safe mode)
        analysis_mode: Feature analysis mode (e.g., 'trial_ml_safe', 'group_stats')
        logger: Logger instance for warnings
        condition_labels: Condition labels when method='condition'
        min_trials_per_condition: Minimum training trials per condition
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        
    Returns:
        ITPC map with shape (n_epochs, n_ch, n_freqs, n_times) for 'loo'/'condition',
        or (n_ch, n_freqs, n_times) for 'global'/'fold_global'
    """
    mode = str(analysis_mode or "").strip().lower()
    if mode == "trial_ml_safe" and method == "global":
        raise ValueError(
            "ITPC(method='global') is not allowed in analysis_mode='trial_ml_safe' because it "
            "uses all trials (including test trials) and leaks information. Use method='fold_global' "
            "or method='loo' computed within CV folds."
        )

    if method == "loo":
        if train_mask is None:
            raise ValueError(
                "ITPC(method='loo') requires ctx.train_mask to be provided (training set trials only). "
                "Compute ITPC within each CV fold to avoid leakage."
            )
        return _compute_loo_itpc(data, train_mask=train_mask, n_jobs=n_jobs)

    if method == "fold_global":
        if train_mask is None:
            if mode == "trial_ml_safe":
                raise ValueError(
                    "ITPC(method='fold_global') requires ctx.train_mask in analysis_mode='trial_ml_safe'. "
                    "Pass a fold-specific train_mask or switch to analysis_mode='group_stats'."
                )
            if logger is not None:
                logger.info(
                    "ITPC(method='fold_global') requested but ctx.train_mask is None; computing ITPC across all trials "
                    "(equivalent to method='global'; not CV-safe)."
                )
            return _compute_global_itpc_map(data, n_jobs=n_jobs, logger=logger)
        n_training_trials = int(np.sum(train_mask))
        logger.info(
            "ITPC: Using fold_global mode - computing from %d training trials only",
            n_training_trials
        )
        return _compute_fold_global_itpc_map(data, train_mask, n_jobs=n_jobs, logger=logger)

    if method == "condition":
        if condition_labels is None:
            raise ValueError(
                "ITPC(method='condition') requires condition labels. Set "
                "feature_engineering.itpc.condition_column and ensure aligned_events contains it."
            )
        if mode == "trial_ml_safe" and train_mask is None:
            raise ValueError(
                "ITPC(method='condition') in analysis_mode='trial_ml_safe' requires ctx.train_mask so the "
                "condition-level maps can be computed from training trials only."
            )
        itpc_map = _compute_condition_itpc_map(
            data,
            condition_labels,
            train_mask=train_mask,
            min_trials_per_condition=int(min_trials_per_condition),
            logger=logger,
            n_jobs=n_jobs,
        )
        if logger is not None:
            n_ch = data.shape[1]
            from eeg_pipeline.utils.parallel import _normalize_n_jobs
            n_jobs_actual = _normalize_n_jobs(n_jobs)
            use_parallel = n_jobs != 1 and n_ch >= _MIN_CHANNELS_FOR_PARALLEL
            if use_parallel:
                logger.debug(f"ITPC(condition): Using parallel computation ({n_jobs_actual} jobs, {n_ch} channels)")
        return itpc_map

    if method == "global":
        if train_mask is not None:
            logger.info(
                "ITPC: train_mask detected with method='global'. Consider using method='fold_global' "
                "for leakage-safe CV/machine learning. Using global ITPC (all trials) as requested."
            )
        return _compute_global_itpc_map(data, n_jobs=n_jobs, logger=logger)

    raise ValueError(f"Unsupported ITPC method: {method!r}")


def _extract_band_frequencies(
    band_name: str,
    tf_bands: Dict[str, List[float]],
) -> Optional[Tuple[float, float]]:
    """Extract frequency range for a band name."""
    if band_name not in tf_bands:
        return None
    try:
        fmin = float(tf_bands[band_name][0])
        fmax = float(tf_bands[band_name][1])
        return (fmin, fmax)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid frequency band definition for '{band_name}': {tf_bands.get(band_name)}") from exc


def _check_harmonic_overlap(
    phase_min: float,
    phase_max: float,
    amp_min: float,
    amp_max: float,
    max_harmonic: int,
    tolerance_hz: float,
) -> bool:
    """Check if amplitude band overlaps with harmonics of phase band."""
    for harmonic in range(2, max_harmonic + 1):
        harmonic_min = (phase_min * harmonic) - tolerance_hz
        harmonic_max = (phase_max * harmonic) + tolerance_hz
        if (amp_max >= harmonic_min) and (amp_min <= harmonic_max):
            return True
    return False


def _validate_pac_band_pair(
    phase_band: str,
    amp_band: str,
    tf_bands: Dict[str, List[float]],
    allow_harmonic_overlap: bool,
    max_harm: int,
    tol_hz: float,
    logger: logging.Logger,
) -> Optional[Tuple[float, float, float, float]]:
    """Validate a PAC band pair and check for harmonic overlap.
    
    Args:
        phase_band: Name of phase frequency band
        amp_band: Name of amplitude frequency band
        tf_bands: Dictionary mapping band names to [fmin, fmax] tuples
        allow_harmonic_overlap: Whether to allow harmonic overlap
        max_harm: Maximum harmonic to check
        tol_hz: Tolerance in Hz for harmonic overlap detection
        logger: Logger instance for warnings
        
    Returns:
        Tuple of (pmin, pmax, amin, amax) if valid, None otherwise
    """
    phase_range = _extract_band_frequencies(phase_band, tf_bands)
    amp_range = _extract_band_frequencies(amp_band, tf_bands)
    
    if phase_range is None or amp_range is None:
        return None
    
    pmin, pmax = phase_range
    amin, amax = amp_range
    
    if amin <= pmax:
        return None
    
    should_check_harmonics = (not allow_harmonic_overlap and 
                              max_harm >= 2 and 
                              np.isfinite(tol_hz) and 
                              tol_hz >= 0)
    
    if should_check_harmonics:
        has_overlap = _check_harmonic_overlap(pmin, pmax, amin, amax, max_harm, tol_hz)
        if has_overlap:
            logger.warning(
                "PAC: skipping pair %s→%s due to harmonic overlap (phase=%0.1f-%0.1fHz, amp=%0.1f-%0.1fHz). "
                "Set feature_engineering.pac.allow_harmonic_overlap=true to override.",
                phase_band,
                amp_band,
                pmin,
                pmax,
                amin,
                amax,
            )
            return None
    
    return (pmin, pmax, amin, amax)


def _extract_pac_config(config: Any) -> Dict[str, Any]:
    """Extract and validate PAC configuration."""
    pac_cfg = _safe_config_get(config, "feature_engineering.pac", {})
    method = str(pac_cfg.get("method", "mvl")).strip().lower()
    if method != "mvl":
        raise ValueError(f"PAC: unsupported method '{method}'. Only 'mvl' is implemented.")
    return pac_cfg


def _extract_frequency_ranges(
    pac_cfg: Dict[str, Any],
    default_phase: Tuple[float, float] = (4.0, 8.0),
    default_amp: Tuple[float, float] = (30.0, 80.0),
) -> Tuple[float, float, float, float]:
    """Extract phase and amplitude frequency ranges from config."""
    phase_range = pac_cfg.get("phase_range", list(default_phase))
    amp_range = pac_cfg.get("amp_range", list(default_amp))
    try:
        phase_min = float(phase_range[0])
        phase_max = float(phase_range[1])
        amp_min = float(amp_range[0])
        amp_max = float(amp_range[1])
    except (IndexError, ValueError, TypeError):
        phase_min, phase_max = default_phase
        amp_min, amp_max = default_amp
    return phase_min, phase_max, amp_min, amp_max


def _prepare_pac_data_and_times(
    tfr_complex: mne.time_frequency.EpochsTFR,
    freqs: np.ndarray,
    times: np.ndarray,
    segment_window: Optional[Tuple[float, float]],
    segment_name: str,
    logger: logging.Logger,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Prepare PAC data and time arrays, applying segment window if provided."""
    data = tfr_complex.data  # (n_epochs, n_ch, n_freqs, n_times)
    tfr_freqs = np.asarray(getattr(tfr_complex, "freqs", freqs), dtype=float)
    tfr_times = np.asarray(getattr(tfr_complex, "times", times), dtype=float)

    if segment_window is not None:
        start, end = float(segment_window[0]), float(segment_window[1])
        time_mask = (tfr_times >= start) & (tfr_times < end)
        if not np.any(time_mask):
            logger.warning("PAC: empty time window for %s [%0.3f, %0.3f)", segment_name, start, end)
            return None, None, None
        data = data[..., time_mask]
        tfr_times = tfr_times[time_mask]

    n_times = data.shape[-1]
    if n_times == 0:
        logger.warning("PAC: no samples available for %s; skipping", segment_name)
        return None, None, None

    return data, tfr_freqs, tfr_times


def _compute_pac_surrogates(
    phase_unit_vectors: np.ndarray,
    amplitudes: np.ndarray,
    n_surrogates: int,
    normalize: bool,
    epsilon: float,
    n_times: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute PAC surrogate distribution for z-scoring.
    
    Args:
        phase_unit_vectors: Phase unit vectors of shape (n_epochs, n_times) or (n_epochs, n_ch, n_times)
        amplitudes: Amplitudes of shape (n_epochs, n_times) or (n_epochs, n_ch, n_times)
        n_surrogates: Number of surrogate samples
        normalize: Whether to normalize by sum of amplitudes
        epsilon: Small value to prevent division by zero
        n_times: Number of time points
        rng: Random number generator
        
    Returns:
        Surrogate PAC values of shape (n_epochs, n_surrogates) or (n_epochs, n_ch, n_surrogates)
    """
    n_epochs = phase_unit_vectors.shape[0]
    is_per_channel = phase_unit_vectors.ndim == 3
    
    if is_per_channel:
        n_ch = phase_unit_vectors.shape[1]
        surrogates = np.full((n_epochs, n_ch, n_surrogates), np.nan, dtype=float)
    else:
        surrogates = np.full((n_epochs, n_surrogates), np.nan, dtype=float)
    
    for epoch_idx in range(n_epochs):
        if is_per_channel:
            for ch_idx in range(n_ch):
                epoch_amplitude = amplitudes[epoch_idx, ch_idx]
                epoch_phase = phase_unit_vectors[epoch_idx, ch_idx]
                
                if not (np.isfinite(epoch_amplitude).any() and np.isfinite(epoch_phase).any()):
                    continue
                
                if normalize:
                    denominator = np.nansum(epoch_amplitude) + epsilon
                else:
                    denominator = float(n_times)
                
                for surrogate_idx in range(n_surrogates):
                    shift = int(rng.integers(1, n_times - 1))
                    shifted_amplitude = np.roll(epoch_amplitude, shift)
                    
                    if normalize:
                        numerator = np.nansum(shifted_amplitude * epoch_phase)
                    else:
                        numerator = np.nanmean(shifted_amplitude * epoch_phase)
                    
                    surrogates[epoch_idx, ch_idx, surrogate_idx] = float(np.abs(numerator / denominator))
        else:
            epoch_amplitude = amplitudes[epoch_idx]
            epoch_phase = phase_unit_vectors[epoch_idx]
            
            if not (np.isfinite(epoch_amplitude).any() and np.isfinite(epoch_phase).any()):
                continue
            
            if normalize:
                denominator = np.nansum(epoch_amplitude) + epsilon
            else:
                denominator = float(n_times)
            
            for surrogate_idx in range(n_surrogates):
                shift = int(rng.integers(1, n_times - 1))
                shifted_amplitude = np.roll(epoch_amplitude, shift)
                
                if normalize:
                    numerator = np.nansum(shifted_amplitude * epoch_phase)
                else:
                    numerator = np.nanmean(shifted_amplitude * epoch_phase)
                
                surrogates[epoch_idx, surrogate_idx] = float(np.abs(numerator / denominator))
    
    return surrogates


def _compute_pac_z_scores(
    pac_values: np.ndarray,
    surrogates: np.ndarray,
) -> np.ndarray:
    """Compute z-scores from PAC values and surrogate distribution."""
    surrogate_mean = np.nanmean(surrogates, axis=1)
    surrogate_std = np.nanstd(surrogates, axis=1, ddof=1)
    surrogate_std = np.where(surrogate_std > 0, surrogate_std, np.nan)
    z_scores = (pac_values - surrogate_mean) / surrogate_std
    return z_scores


def _get_channel_names_from_tfr(
    tfr_complex: mne.time_frequency.EpochsTFR,
    n_channels: int,
    logger: logging.Logger,
) -> List[str]:
    """Extract channel names from TFR, handling dimension mismatches."""
    tfr_ch_names = tfr_complex.info['ch_names']
    if len(tfr_ch_names) != n_channels:
        logger.warning(
            "PAC channel count mismatch: TFR has %d channels but info has %d; using TFR channel list",
            n_channels, len(tfr_ch_names)
        )
    if len(tfr_ch_names) >= n_channels:
        return tfr_ch_names[:n_channels]
    return tfr_ch_names


def _build_roi_map_if_needed(
    spatial_modes: List[str],
    ch_names: List[str],
    config: Any,
) -> Dict[str, List[int]]:
    """Build ROI map if ROI spatial mode is requested."""
    if 'roi' not in spatial_modes:
        return {}
    
    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
    from eeg_pipeline.utils.analysis.channels import build_roi_map
    
    roi_defs = get_roi_definitions(config)
    if not roi_defs:
        return {}
    
    return build_roi_map(ch_names, roi_defs)


# --- Main API ---

def _get_baseline_correction_mode(config: Any) -> str:
    """Extract baseline correction mode from config."""
    itpc_cfg = _safe_config_get(config, "feature_engineering.itpc", {})
    baseline_correction = str(itpc_cfg.get("baseline_correction", "none")).strip().lower()
    if baseline_correction not in {"none", "subtract"}:
        return "none"
    return baseline_correction


def _compute_itpc_for_segment_and_band(
    itpc_map: np.ndarray,
    segment_mask: np.ndarray,
    baseline_mask: Optional[np.ndarray],
    frequency_mask: np.ndarray,
    itpc_method: str,
    n_epochs: int,
) -> np.ndarray:
    """Compute ITPC for a segment and frequency band, with optional baseline correction."""
    if itpc_method in {"loo", "condition"}:
        itpc_segment = itpc_map[..., segment_mask]  # (epochs, ch, freq, time_seg)
        itpc_segment_mean_time = np.nanmean(itpc_segment, axis=-1)  # (epochs, ch, freq)
        itpc_band = np.nanmean(itpc_segment_mean_time[..., frequency_mask], axis=-1)  # (epochs, ch)
        
        if baseline_mask is not None:
            itpc_baseline = itpc_map[..., baseline_mask]
            baseline_mean_time = np.nanmean(itpc_baseline, axis=-1)  # (epochs, ch, freq)
            baseline_band = np.nanmean(baseline_mean_time[..., frequency_mask], axis=-1)
            itpc_band = itpc_band - baseline_band
    else:
        itpc_segment = itpc_map[..., segment_mask]  # (ch, freq, time_seg)
        itpc_segment_mean_time = np.nanmean(itpc_segment, axis=-1)  # (ch, freq)
        itpc_band_ch = np.nanmean(itpc_segment_mean_time[:, frequency_mask], axis=-1)  # (ch,)
        
        if baseline_mask is not None:
            itpc_baseline = itpc_map[..., baseline_mask]
            baseline_mean_time = np.nanmean(itpc_baseline, axis=-1)  # (ch, freq)
            baseline_band_ch = np.nanmean(baseline_mean_time[:, frequency_mask], axis=-1)
            itpc_band_ch = itpc_band_ch - baseline_band_ch
        
        itpc_band = _broadcast_per_trial(itpc_band_ch, n_epochs)  # (epochs, ch)
    
    return itpc_band


def extract_phase_features(
    ctx: Any,
    bands: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract ITPC phase features from complex TFR data."""
    config = ctx.config
    epochs = ctx.epochs
    logger = getattr(ctx, "logger", None)

    itpc_method = _get_itpc_method(config)
    
    tfr = ctx.tfr_complex
    if tfr is None:
        ctx.logger.error("Phase: complex TFR missing; skipping extraction.")
        return pd.DataFrame(), []
    
    data = tfr.data  # (epochs, ch, freq, time)
    times = tfr.times
    freqs = tfr.freqs
    ch_names = tfr.info['ch_names']
    n_epochs = int(data.shape[0])
    
    train_mask = getattr(ctx, "train_mask", None)
    analysis_mode = getattr(ctx, "analysis_mode", None)
    
    n_jobs = get_n_jobs(config, default=-1, config_path="feature_engineering.parallel.n_jobs_itpc")
    if logger is not None:
        from eeg_pipeline.utils.parallel import _normalize_n_jobs
        n_jobs_actual = _normalize_n_jobs(n_jobs)
        logger.info(f"ITPC: n_jobs={n_jobs_actual} (from config: {n_jobs})")

    condition_labels = None
    min_trials_per_condition = 10
    if itpc_method == "condition":
        itpc_cfg = _safe_config_get(config, "feature_engineering.itpc", {})
        condition_column = itpc_cfg.get("condition_column")
        min_trials_per_condition = int(itpc_cfg.get("min_trials_per_condition", 10))
        if not condition_column:
            raise ValueError(
                "ITPC(method='condition') requires feature_engineering.itpc.condition_column to be set."
            )
        events = getattr(ctx, "aligned_events", None)
        if events is None or getattr(events, "empty", True) or len(events) != n_epochs:
            raise ValueError(
                "ITPC(method='condition') requires ctx.aligned_events aligned to epochs."
            )
        if condition_column not in events.columns:
            raise ValueError(
                f"ITPC(method='condition') condition_column '{condition_column}' not found in aligned_events."
            )
        condition_labels = events[condition_column].to_numpy()

    itpc_map = _compute_itpc_map_by_method(
        data,
        itpc_method,
        train_mask,
        analysis_mode,
        logger,
        condition_labels=condition_labels,
        min_trials_per_condition=min_trials_per_condition,
        n_jobs=n_jobs,
    )
    
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)
    
    results = {}
    spatial_modes = list(getattr(ctx, "spatial_modes", ["roi", "global"]))
    roi_map = _build_roi_map_if_needed(spatial_modes, ch_names, config)
    
    windows = ctx.windows
    target_name = getattr(ctx, "name", None)
    allow_full_epoch_fallback = bool(_safe_config_get(config, "feature_engineering.windows.allow_full_epoch_fallback", False))
    
    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        # windows.get_mask() may be built on a different time axis than tfr.times.
        # If lengths mismatch, rebuild using explicit window ranges on the TFR time axis.
        if mask is not None and getattr(mask, "shape", (0,))[0] != times.shape[0]:
            mask = None
        if mask is None or not np.any(mask):
            mask = make_mask_for_times(windows, target_name, times)

        if mask is not None and np.any(mask):
            segment_masks = {target_name: mask}
        else:
            if logger:
                if allow_full_epoch_fallback:
                    logger.warning(
                        "ITPC: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                        target_name,
                    )
                else:
                    logger.error(
                        "ITPC: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                segment_masks = {target_name: np.ones_like(times, dtype=bool)}
            else:
                return pd.DataFrame(), []
        segments = [target_name]
    else:
        segment_masks = get_segment_masks(epochs.times, windows, config)
        segments = list(segment_masks.keys())
    
    if not segments:
        logger.warning("ITPC: No valid segments found; returning empty results.")
        return pd.DataFrame(), []
    
    baseline_correction = _get_baseline_correction_mode(config)

    for segment_name in segments:
        segment_mask = segment_masks.get(segment_name)
        if segment_mask is None or getattr(segment_mask, "shape", (0,))[0] != times.shape[0]:
            segment_mask = make_mask_for_times(windows, segment_name, times)
            
        if not np.any(segment_mask):
            continue

        baseline_mask = None
        if baseline_correction == "subtract":
            try:
                baseline_mask = make_mask_for_times(windows, "baseline", times)
            except (KeyError, ValueError, AttributeError):
                baseline_mask = None
            if baseline_mask is not None and not np.any(baseline_mask):
                baseline_mask = None
            if segment_name == "baseline":
                baseline_mask = None
        
        for band in bands:
            if band not in freq_bands:
                continue
            
            fmin, fmax = freq_bands[band]
            frequency_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(frequency_mask):
                continue
            
            itpc_band = _compute_itpc_for_segment_and_band(
                itpc_map, segment_mask, baseline_mask, frequency_mask,
                itpc_method, n_epochs
            )
            
            spatial_results = _aggregate_spatial_features(
                itpc_band, "itpc", segment_name, band, "val",
                ch_names, spatial_modes, roi_map
            )
            results.update(spatial_results)
                
    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)


def _parse_requested_pac_pairs(requested_pairs: Any) -> List[Tuple[str, str]]:
    """Parse and validate requested PAC band pairs."""
    if requested_pairs is None:
        return [("theta", "gamma"), ("alpha", "gamma")]
    
    pairs: List[Tuple[str, str]] = []
    for pair in requested_pairs:
        if pair and len(pair) >= 2:
            pairs.append((str(pair[0]), str(pair[1])))
    return pairs


def _get_valid_pac_pairs(
    pairs: List[Tuple[str, str]],
    tf_bands: Dict[str, List[float]],
    allow_harmonic_overlap: bool,
    max_harm: int,
    tol_hz: float,
    logger: logging.Logger,
) -> List[Tuple[str, str, Tuple[float, float], Tuple[float, float]]]:
    """Validate and filter PAC band pairs."""
    valid_pairs: List[Tuple[str, str, Tuple[float, float], Tuple[float, float]]] = []
    for phase_band, amp_band in pairs:
        band_range = _validate_pac_band_pair(
            phase_band, amp_band, tf_bands,
            allow_harmonic_overlap, max_harm, tol_hz, logger
        )
        if band_range is not None:
            pmin, pmax, amin, amax = band_range
            valid_pairs.append((phase_band, amp_band, (pmin, pmax), (amin, amax)))
    return valid_pairs


def _compute_pac_for_channel_band_pair(
    data: np.ndarray,
    channel_idx: int,
    phase_freqs: np.ndarray,
    amp_freqs: np.ndarray,
    phase_indices: np.ndarray,
    amp_indices: np.ndarray,
    phase_band_range: Tuple[float, float],
    amp_band_range: Tuple[float, float],
    normalize: bool,
    epsilon: float,
    n_times: int,
) -> Optional[np.ndarray]:
    """Compute PAC for a single channel and band pair.
    
    Returns:
        PAC values of shape (n_epochs,) or None if band ranges are invalid
    """
    pmin, pmax = phase_band_range
    amin, amax = amp_band_range
    
    phase_mask = (phase_freqs >= pmin) & (phase_freqs <= pmax)
    amp_mask = (amp_freqs >= amin) & (amp_freqs <= amax)
    
    if not (np.any(phase_mask) and np.any(amp_mask)):
        return None
    
    phase_idx = phase_indices[phase_mask]
    amp_idx = amp_indices[amp_mask]
    
    # Phase: average unit vectors across phase freqs -> complex (epochs, times)
    phase_angles = np.angle(data[:, channel_idx, phase_idx, :])
    phase_unit_vectors = np.exp(1j * phase_angles)
    mean_phase_vector = np.nanmean(phase_unit_vectors, axis=1)  # (epochs, times)
    
    # Amplitude: average across amp freqs -> (epochs, times)
    amplitudes = np.abs(data[:, channel_idx, amp_idx, :])
    mean_amplitude = np.nanmean(amplitudes, axis=1)  # (epochs, times)
    
    if normalize:
        denominator = np.nansum(mean_amplitude, axis=1) + epsilon
        numerator = np.nansum(mean_amplitude * mean_phase_vector, axis=1)
    else:
        denominator = float(n_times)
        numerator = np.nanmean(mean_amplitude * mean_phase_vector, axis=1)
    
    pac_values = np.abs(numerator / denominator)
    return pac_values


def _aggregate_pac_results_to_dataframe(
    pair_channel_data: Dict[str, np.ndarray],
    pair_channel_data_z: Dict[str, np.ndarray],
    segment_name: str,
    ch_names: List[str],
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
) -> Optional[pd.DataFrame]:
    """Aggregate PAC results into a DataFrame."""
    trials_pac_list = []
    
    for band_pair, matrix in pair_channel_data.items():
        spatial_results = _aggregate_spatial_features(
            matrix, "pac", segment_name, band_pair, "val",
            ch_names, spatial_modes, roi_map
        )
        for col_name, values in spatial_results.items():
            trials_pac_list.append(pd.Series(values, name=col_name))
    
    for band_pair, matrix in pair_channel_data_z.items():
        spatial_results = _aggregate_spatial_features(
            matrix, "pac", segment_name, band_pair, "z",
            ch_names, spatial_modes, roi_map
        )
        for col_name, values in spatial_results.items():
            trials_pac_list.append(pd.Series(values, name=col_name))
    
    if not trials_pac_list:
        return None
    
    return pd.concat(trials_pac_list, axis=1)


def compute_pac_comodulograms(
    tfr_complex: mne.time_frequency.EpochsTFR,
    freqs: np.ndarray,
    times: np.ndarray,
    info: mne.Info,
    config: Any,
    logger: logging.Logger,
    *,
    segment_name: str = "full",
    segment_window: Optional[Tuple[float, float]] = None,
    spatial_modes: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compute Phase-Amplitude Coupling (PAC) using Mean Vector Length (MVL).
    
    Returns:
        pac_df: Aggregated PAC (mean over trials) per channel & freq-pair.
        pac_phase_freqs: Phase frequencies used.
        pac_amp_freqs: Amplitude frequencies used.
        pac_trials_df: Trial-wise PAC (averaged over freq pairs per band or selected pairs).
        pac_time_df: Time-resolved PAC (optional, currently None).
    """
    if tfr_complex is None:
        return None, None, None, None, None
    
    result = _prepare_pac_data_and_times(
        tfr_complex, freqs, times, segment_window, segment_name, logger
    )
    if result[0] is None:
        return None, None, None, None, None
    
    data, tfr_freqs, tfr_times = result
    n_epochs, n_ch, _, _ = data.shape
    n_times = data.shape[-1]
    
    pac_cfg = _extract_pac_config(config)
    n_surrogates = int(pac_cfg.get("n_surrogates", 0))
    min_epochs = int(pac_cfg.get("min_epochs", 2))
    if n_epochs < min_epochs:
        logger.warning("PAC: insufficient epochs (%d < %d); skipping", n_epochs, min_epochs)
        return None, None, None, None, None
    
    phase_min, phase_max, amp_min, amp_max = _extract_frequency_ranges(pac_cfg)
    
    phase_mask = (tfr_freqs >= phase_min) & (tfr_freqs <= phase_max)
    amp_mask = (tfr_freqs >= amp_min) & (tfr_freqs <= amp_max)
    
    if not (np.any(phase_mask) and np.any(amp_mask)):
        logger.warning("No valid phase/amplitude frequencies for PAC")
        return None, None, None, None, None
    
    phase_freqs = tfr_freqs[phase_mask]
    amp_freqs = tfr_freqs[amp_mask]
    phase_indices = np.where(phase_mask)[0]
    amp_indices = np.where(amp_mask)[0]
    
    normalize = bool(pac_cfg.get("normalize", True))
    epsilon = float(_safe_config_get(config, "feature_engineering.constants.epsilon_amp", _EPSILON_COMPLEX))
    
    seed = pac_cfg.get("random_seed", None)
    rng = np.random.default_rng(None if seed in (None, "", 0) else int(seed))
    
    ch_names = _get_channel_names_from_tfr(tfr_complex, n_ch, logger)
    
    tf_bands = _safe_config_get(config, "time_frequency_analysis.bands", {})
    requested_pairs = pac_cfg.get("pairs")
    pairs = _parse_requested_pac_pairs(requested_pairs)
    
    allow_harmonic_overlap = bool(pac_cfg.get("allow_harmonic_overlap", False))
    max_harm = int(pac_cfg.get("max_harmonic", 6))
    tol_hz = float(pac_cfg.get("harmonic_tolerance_hz", 1.0))
    
    valid_pairs = _get_valid_pac_pairs(
        pairs, tf_bands, allow_harmonic_overlap, max_harm, tol_hz, logger
    )
    
    if not valid_pairs:
        logger.warning("PAC: no valid band pairs found in config; skipping")
        return None, phase_freqs, amp_freqs, None, None
    
    pair_channel_data = {}
    pair_channel_data_z = {}
    
    for channel_idx in range(n_ch):
        for phase_band, amp_band, phase_range, amp_range in valid_pairs:
            pac_values = _compute_pac_for_channel_band_pair(
                data, channel_idx, phase_freqs, amp_freqs,
                phase_indices, amp_indices, phase_range, amp_range,
                normalize, epsilon, n_times
            )
            
            if pac_values is None:
                continue
            
            band_pair_name = f"{phase_band}_{amp_band}"
            if band_pair_name not in pair_channel_data:
                pair_channel_data[band_pair_name] = np.full((n_epochs, n_ch), np.nan)
            pair_channel_data[band_pair_name][:, channel_idx] = pac_values
            
            if n_surrogates > 0 and n_times > _MIN_TIMES_FOR_SURROGATES:
                pmin, pmax = phase_range
                amin, amax = amp_range
                
                phase_idx = phase_indices[(phase_freqs >= pmin) & (phase_freqs <= pmax)]
                amp_idx = amp_indices[(amp_freqs >= amin) & (amp_freqs <= amax)]
                
                phase_angles = np.angle(data[:, channel_idx, phase_idx, :])
                amplitudes = np.abs(data[:, channel_idx, amp_idx, :])
                phase_unit_vectors = np.exp(1j * phase_angles)
                mean_phase = np.nanmean(phase_unit_vectors, axis=1)  # (epochs, times)
                mean_amplitude = np.nanmean(amplitudes, axis=1)  # (epochs, times)
                
                surrogates = _compute_pac_surrogates(
                    mean_phase, mean_amplitude, n_surrogates,
                    normalize, epsilon, n_times, rng
                )
                pac_z = _compute_pac_z_scores(pac_values, surrogates)
                
                if band_pair_name not in pair_channel_data_z:
                    pair_channel_data_z[band_pair_name] = np.full((n_epochs, n_ch), np.nan)
                pair_channel_data_z[band_pair_name][:, channel_idx] = pac_z
    
    if spatial_modes is None:
        spatial_modes = _safe_config_get(config, "feature_engineering.spatial_modes", ["roi", "global"])
    
    roi_map = _build_roi_map_if_needed(spatial_modes, ch_names, config)
    
    pac_trials_df = _aggregate_pac_results_to_dataframe(
        pair_channel_data, pair_channel_data_z, segment_name,
        ch_names, spatial_modes, roi_map
    )
    
    return None, phase_freqs, amp_freqs, pac_trials_df, None


def extract_itpc_from_precomputed(
    precomputed: Any,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute ITPC-style metrics directly from precomputed band phases.
    
    ITPC = | (1/N) * sum( exp(i*phase) ) |
    
    Supports multiple computation modes:
    - 'global': Compute across all trials, broadcast to each (pseudo-replication warning)
    - 'fold_global': Compute from training trials only (CV-safe)
    - 'condition': Compute per condition group (avoids pseudo-replication)
    
    Args:
        precomputed: PrecomputedData with band phases
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    """
    logger = getattr(precomputed, "logger", None)
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        if logger is not None:
            logger.warning("ITPC: %s; skipping extraction.", err_msg)
        return pd.DataFrame(), []

    cfg = precomputed.config or {}
    itpc_method = _get_itpc_method(cfg)
    if itpc_method == "loo":
        raise ValueError(
            "ITPC(method='loo') is not supported in precomputed mode because it requires "
            "fold-specific training masks to avoid leakage. Use ITPC(method='global') "
            "or compute LOO-ITPC within your CV loop."
        )
    
    n_jobs = get_n_jobs(cfg, default=n_jobs, config_path="feature_engineering.parallel.n_jobs_itpc")
    if logger is not None:
        from eeg_pipeline.utils.parallel import _normalize_n_jobs
        n_jobs_actual = _normalize_n_jobs(n_jobs)
        logger.info(f"ITPC (precomputed): n_jobs={n_jobs_actual} (from config: {n_jobs})")
    
    analysis_mode = str(cfg.get("feature_engineering.analysis_mode", "group_stats") or "group_stats").strip().lower()
    train_mask = getattr(precomputed, "train_mask", None)

    if analysis_mode == "trial_ml_safe" and itpc_method == "global":
        raise ValueError(
            "ITPC(method='global') is not allowed in analysis_mode='trial_ml_safe' because it uses all trials "
            "(including held-out trials) and leaks information. Use method='fold_global' with a fold-specific train_mask."
        )
    if analysis_mode == "trial_ml_safe" and itpc_method in {"fold_global", "condition"} and train_mask is None:
        raise ValueError(
            f"ITPC(method='{itpc_method}') in analysis_mode='trial_ml_safe' requires precomputed.train_mask."
        )

    itpc_cfg = _safe_config_get(cfg, "feature_engineering.itpc", {})
    condition_column = itpc_cfg.get("condition_column", None)
    min_trials_per_condition = int(itpc_cfg.get("min_trials_per_condition", 10))

    condition_labels = None
    if itpc_method == "condition":
        if not condition_column:
            raise ValueError(
                "ITPC(method='condition') requires feature_engineering.itpc.condition_column to be set."
            )
        metadata = getattr(precomputed, "metadata", None)
        if metadata is None or getattr(metadata, "empty", True):
            raise ValueError(
                "ITPC(method='condition') requires precomputed.metadata to be populated with aligned events."
            )
        if condition_column not in metadata.columns:
            raise ValueError(
                f"ITPC(method='condition') condition_column '{condition_column}' not found in precomputed.metadata."
            )
        condition_labels = metadata[condition_column].to_numpy()
        
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    
    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    allow_full_epoch_fallback = bool(_safe_config_get(cfg, "feature_engineering.windows.allow_full_epoch_fallback", False))
    
    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            masks = {target_name: mask}
        else:
            if logger:
                if allow_full_epoch_fallback:
                    logger.warning(
                        "ITPC: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                        target_name,
                    )
                else:
                    logger.error(
                        "ITPC: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                masks = {target_name: np.ones(len(precomputed.times), dtype=bool)}
            else:
                return pd.DataFrame(), []
    else:
        masks = get_segment_masks(precomputed.times, windows, precomputed.config)
    
    if not masks:
        return pd.DataFrame(), []
    
    ch_names = precomputed.ch_names
    n_epochs = precomputed.data.shape[0]
    if n_epochs < _MIN_EPOCHS_FOR_ITPC:
        if logger is not None:
            logger.warning("ITPC: Fewer than %d epochs available; skipping extraction.", _MIN_EPOCHS_FOR_ITPC)
        return pd.DataFrame(), []
    
    baseline_correction = _get_baseline_correction_mode(cfg)
    spatial_modes = getattr(precomputed, "spatial_modes", None) or ["roi", "global"]
    roi_map = _build_roi_map_if_needed(spatial_modes, ch_names, cfg)

    baseline_mask = masks.get("baseline") if baseline_correction == "subtract" else None
    results = {}

    for band, band_data in precomputed.band_data.items():
        phases = band_data.phase
        if phases is None or phases.size == 0:
            continue

        complex_vectors = np.exp(1j * phases)  # (epochs, ch, time)

        baseline_itpc = None
        if baseline_mask is not None and np.any(baseline_mask):
            baseline_complex = complex_vectors[:, :, baseline_mask]
            if itpc_method == "condition":
                baseline_itpc = _compute_condition_itpc_precomputed(
                    baseline_complex,
                    condition_labels,
                    train_mask,
                    min_trials_per_condition,
                    logger,
                    n_jobs=n_jobs,
                )
            else:
                if itpc_method == "fold_global" and train_mask is None:
                    if logger is not None:
                        logger.info(
                            "ITPC(method='fold_global') requested but precomputed.train_mask is None; computing baseline ITPC across all trials "
                            "(equivalent to method='global'; not CV-safe)."
                        )
                    use_mask = None
                else:
                    use_mask = train_mask if itpc_method == "fold_global" else None
                base_map = _compute_itpc_map_precomputed(
                    baseline_complex, use_mask, n_jobs=n_jobs, logger=logger
                )
                base_ch = np.nanmean(base_map, axis=1)
                baseline_itpc = _broadcast_per_trial(base_ch, n_epochs)

        for seg_name, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            
            segment_complex = complex_vectors[:, :, mask]
            
            if itpc_method == "condition":
                itpc_seg = _compute_condition_itpc_precomputed(
                    segment_complex,
                    condition_labels,
                    train_mask,
                    min_trials_per_condition,
                    logger,
                    n_jobs=n_jobs,
                )
            else:
                if itpc_method == "fold_global":
                    if train_mask is None:
                        if logger is not None:
                            logger.info(
                                "ITPC(method='fold_global') requested but precomputed.train_mask is None; computing ITPC across all trials "
                                "(equivalent to method='global'; not CV-safe)."
                            )
                        use_mask = None
                    else:
                        use_mask = np.asarray(train_mask, dtype=bool)
                else:
                    use_mask = None
                itpc_map = _compute_itpc_map_precomputed(
                    segment_complex, use_mask, n_jobs=n_jobs, logger=logger
                )
                itpc_ch = np.nanmean(itpc_map, axis=1)
                itpc_seg = _broadcast_per_trial(itpc_ch, n_epochs)

            if baseline_itpc is not None and seg_name != "baseline":
                itpc_seg = itpc_seg - baseline_itpc

            spatial_results = _aggregate_spatial_features(
                itpc_seg, "itpc", seg_name, band, "val",
                ch_names, spatial_modes, roi_map
            )
            results.update(spatial_results)
                
    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)


def extract_pac_from_precomputed(
    precomputed: Any,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute PAC using precomputed analytic signals.
    
    PAC(fp, fa) = | mean( A_fa * exp(i * phi_fp) ) | over time.
    """
    logger = getattr(precomputed, "logger", None)
    config = config or getattr(precomputed, "config", None) or {}
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        if logger is not None:
            logger.warning("PAC (precomputed): %s; skipping extraction.", err_msg)
        return pd.DataFrame(), []

    # Get PAC config
    pac_cfg = _safe_config_get(config, "feature_engineering.pac", {})
    method = str(pac_cfg.get("method", "mvl")).strip().lower()
    if method != "mvl":
        raise ValueError(f"PAC (precomputed): unsupported method '{method}'. Only 'mvl' is implemented.")
    n_surrogates = int(pac_cfg.get("n_surrogates", 0))
    requested_pairs = pac_cfg.get("pairs", [("theta", "gamma"), ("alpha", "gamma")])
    normalize = bool(pac_cfg.get("normalize", True))
    eps_amp = float(_safe_config_get(config, "feature_engineering.constants.epsilon_amp", _EPSILON_COMPLEX))
    seed = pac_cfg.get("random_seed", None)
    rng = np.random.default_rng(None if seed in (None, "", 0) else int(seed))
    allow_harmonic_overlap = bool(pac_cfg.get("allow_harmonic_overlap", False))
    max_harm = int(pac_cfg.get("max_harmonic", 6))
    tol_hz = float(pac_cfg.get("harmonic_tolerance_hz", 1.0))
    
    # Get spatial modes and ROI map
    spatial_modes = getattr(precomputed, "spatial_modes", None)
    if spatial_modes is None:
        spatial_modes = _safe_config_get(config, "feature_engineering.spatial_modes", ["roi", "global"])
    
    # Standard bands for pair lookup
    tf_bands = _safe_config_get(config, "time_frequency_analysis.bands", {})
    
    ch_names = precomputed.ch_names
    n_ch = len(ch_names)  # Required for surrogate and waveform QC loops
    n_epochs = precomputed.data.shape[0]
    windows = precomputed.windows

    allow_full_epoch_fallback = bool(_safe_config_get(config, "feature_engineering.windows.allow_full_epoch_fallback", False))
    target_name = getattr(windows, "name", None) if windows else None

    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            masks = {target_name: mask}
        else:
            if logger is not None:
                if allow_full_epoch_fallback:
                    logger.warning(
                        "PAC: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                        target_name,
                    )
                else:
                    logger.error(
                        "PAC: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                masks = {target_name: np.ones_like(precomputed.times, dtype=bool)}
            else:
                return pd.DataFrame(), []
    else:
        masks = get_segment_masks(precomputed.times, windows, config)

    if not masks:
        return pd.DataFrame(), []

    # Pre-calculate sqrt(power) for all bands to get amplitude
    amplitudes = {}
    phases = {}
    for band, bd in precomputed.band_data.items():
        if bd.power is not None:
            # Power is (nep, nch, ntimes)
            amplitudes[band] = np.sqrt(np.maximum(bd.power, 0))
        if bd.phase is not None:
            phases[band] = bd.phase

    results = {}
    
    roi_map = _build_roi_map_if_needed(spatial_modes, ch_names, config)

    pairs = _parse_requested_pac_pairs(requested_pairs)
    valid_pairs = _get_valid_pac_pairs(
        pairs, tf_bands, allow_harmonic_overlap, max_harm, tol_hz, logger
    )

    for segment_name, mask in masks.items():
        if mask is None or not np.any(mask):
            continue
        n_times = int(np.sum(mask))

        for phase_band, amp_band, _, _ in valid_pairs:
            if phase_band not in phases or amp_band not in amplitudes:
                continue

            phase_data = phases[phase_band][..., mask]
            amplitude_data = amplitudes[amp_band][..., mask]

            phase_unit_vectors = np.exp(1j * phase_data)

            if normalize:
                denominator = np.nansum(amplitude_data, axis=-1) + eps_amp
                numerator = np.nansum(amplitude_data * phase_unit_vectors, axis=-1)
            else:
                denominator = float(n_times)
                numerator = np.nanmean(amplitude_data * phase_unit_vectors, axis=-1)

            pac_val = np.abs(numerator / denominator)

            pac_z = None
            if n_surrogates > 0 and n_times > _MIN_TIMES_FOR_SURROGATES:
                surrogates = _compute_pac_surrogates(
                    phase_unit_vectors, amplitude_data, n_surrogates,
                    normalize, eps_amp, n_times, rng
                )
                pac_z = np.full_like(pac_val, np.nan, dtype=float)
                for epoch_idx in range(n_epochs):
                    for channel_idx in range(n_ch):
                        epoch_surrogates = surrogates[epoch_idx, channel_idx]
                        if np.isfinite(epoch_surrogates).any():
                            surrogate_mean = float(np.nanmean(epoch_surrogates))
                            surrogate_std = float(np.nanstd(epoch_surrogates, ddof=1))
                            if np.isfinite(surrogate_std) and surrogate_std > 0:
                                pac_z[epoch_idx, channel_idx] = (
                                    pac_val[epoch_idx, channel_idx] - surrogate_mean
                                ) / surrogate_std

            pair_label = f"{phase_band}_{amp_band}"

            spatial_results = _aggregate_spatial_features(
                pac_val,
                "pac",
                segment_name,
                pair_label,
                "val",
                ch_names,
                spatial_modes,
                roi_map,
            )
            results.update(spatial_results)

            if pac_z is not None:
                spatial_results_z = _aggregate_spatial_features(
                    pac_z,
                    "pac",
                    segment_name,
                    pair_label,
                    "z",
                    ch_names,
                    spatial_modes,
                    roi_map,
                )
                results.update(spatial_results_z)

            if bool(pac_cfg.get("compute_waveform_qc", False)) and phase_band in precomputed.band_data:
                try:
                    filt = precomputed.band_data[phase_band].filtered[..., mask]  # (epochs, ch, time)
                    offset_ms = float(pac_cfg.get("waveform_offset_ms", 5.0))
                    sf = float(getattr(precomputed, "sfreq", np.nan))
                    fmax_hz = float(getattr(precomputed.band_data[phase_band], "fmax", np.nan))
                    if not np.isfinite(sf) or sf <= 0:
                        raise ValueError("missing/invalid sampling rate for waveform QC")

                    ratios = np.full((n_epochs, n_ch), np.nan)
                    for ep in range(n_epochs):
                        for ch in range(n_ch):
                            ratios[ep, ch] = _sharpness_log_ratio(
                                filt[ep, ch], sf, offset_ms, fmax_hz=fmax_hz
                            )

                    sharpness_results = _aggregate_spatial_features(
                        ratios,
                        "pac",
                        segment_name,
                        pair_label,
                        "lf_sharpness_ratio",
                        ch_names,
                        spatial_modes,
                        roi_map,
                    )
                    results.update(sharpness_results)
                except (ValueError, AttributeError, KeyError, IndexError) as exc:
                    if logger is not None:
                        logger.warning("PAC waveform QC failed for pair %s: %s", pair_label, exc)
            
    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)
