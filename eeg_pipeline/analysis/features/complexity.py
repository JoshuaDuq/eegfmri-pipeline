"""
Complexity Feature Extraction
=============================

Nonlinear dynamics features for EEG analysis:
- Lempel-Ziv Complexity (LZC): Algorithmic complexity
- Permutation Entropy (PE): Ordinal pattern complexity

Features are computed per user-defined time windows and frequency bands
using band-limited amplitude envelopes from precomputed data.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_permutation_entropy as _permutation_entropy,
    compute_lempel_ziv_complexity as _lempel_ziv_complexity,
)

# Constants
MIN_SAMPLES_FOR_COMPLEXITY = 10
MIN_TARGET_SAMPLES = 10
DEFAULT_PE_ORDER = 3
DEFAULT_PE_DELAY = 1
DEFAULT_TARGET_HZ = 100.0
DEFAULT_TARGET_N_SAMPLES = 500
MIN_FINITE_SAMPLES_FOR_ZSCORE = 2


def _extract_params(config: Any) -> Dict[str, Any]:
    """Extract complexity parameters from config."""
    return {
        "pe_order": int(config.get("feature_engineering.complexity.pe_order", DEFAULT_PE_ORDER)),
        "pe_delay": int(config.get("feature_engineering.complexity.pe_delay", DEFAULT_PE_DELAY)),
        "target_hz": float(config.get("feature_engineering.complexity.target_hz", DEFAULT_TARGET_HZ)),
        "target_n_samples": int(config.get("feature_engineering.complexity.target_n_samples", DEFAULT_TARGET_N_SAMPLES)),
        "zscore": bool(config.get("feature_engineering.complexity.zscore", True)),
    }


def _resample_1d(trace: np.ndarray, *, target_n: int) -> np.ndarray:
    """Resample 1D trace to target length using polyphase or interpolation."""
    trace = np.asarray(trace, dtype=float)
    
    if target_n <= 1 or trace.size <= 1:
        return trace
    if trace.size == target_n:
        return trace
    
    try:
        from math import gcd
        from scipy.signal import resample_poly

        gcd_value = gcd(int(trace.size), int(target_n))
        up_factor = int(target_n // gcd_value)
        down_factor = int(trace.size // gcd_value)
        resampled = resample_poly(
            trace, up=up_factor, down=down_factor, window=("kaiser", 5.0), padtype="line"
        )
        if resampled.size == target_n:
            return resampled.astype(float, copy=False)
    except Exception:
        pass

    original_indices = np.linspace(0.0, 1.0, trace.size)
    target_indices = np.linspace(0.0, 1.0, target_n)
    return np.interp(target_indices, original_indices, trace)


def _standardize_trace(trace: np.ndarray, *, zscore: bool) -> np.ndarray:
    """Standardize trace by z-scoring if requested."""
    trace = np.asarray(trace, dtype=float)
    if not zscore:
        return trace
    
    finite_values = trace[np.isfinite(trace)]
    if finite_values.size < MIN_FINITE_SAMPLES_FOR_ZSCORE:
        return trace
    
    mean_value = float(np.mean(finite_values))
    std_value = float(np.std(finite_values, ddof=1))
    
    if not np.isfinite(std_value) or std_value <= 0:
        return trace - mean_value
    
    return (trace - mean_value) / std_value


def _compute_target_sample_count(
    trace_length: int, sfreq: float, target_hz: float, default_target_n: int
) -> int:
    """Compute target sample count for resampling based on duration and target Hz."""
    target_n = default_target_n
    
    if np.isfinite(sfreq) and sfreq > 0 and np.isfinite(target_hz) and target_hz > 0:
        duration_seconds = float(trace_length) / float(sfreq)
        target_n = max(MIN_TARGET_SAMPLES, int(round(duration_seconds * target_hz)))
    
    return max(MIN_TARGET_SAMPLES, int(target_n))


def _compute_channel_complexity(
    trace: np.ndarray, sfreq: float, params: Dict[str, Any]
) -> Tuple[float, float]:
    """Compute LZC and permutation entropy for a single channel trace."""
    target_hz = float(params.get("target_hz", DEFAULT_TARGET_HZ))
    default_target_n = int(params.get("target_n_samples", DEFAULT_TARGET_N_SAMPLES))
    target_n = _compute_target_sample_count(trace.size, sfreq, target_hz, default_target_n)
    
    resampled_trace = _resample_1d(trace, target_n=target_n)
    standardized_trace = _standardize_trace(
        resampled_trace, zscore=bool(params.get("zscore", True))
    )
    
    try:
        lzc = _lempel_ziv_complexity(standardized_trace)
    except (ValueError, RuntimeError):
        lzc = np.nan
    
    try:
        pe = _permutation_entropy(
            standardized_trace,
            order=params.get("pe_order", DEFAULT_PE_ORDER),
            delay=params.get("pe_delay", DEFAULT_PE_DELAY),
        )
    except (ValueError, RuntimeError):
        pe = np.nan
    
    return float(lzc), float(pe)


def _record_channel_features(
    record: Dict[str, float],
    segment_name: str,
    band: str,
    channel_name: str,
    lzc: float,
    pe: float,
) -> None:
    """Record channel-level complexity features."""
    lzc_column = NamingSchema.build("comp", segment_name, band, "ch", "lzc", channel=channel_name)
    pe_column = NamingSchema.build("comp", segment_name, band, "ch", "pe", channel=channel_name)
    record[lzc_column] = lzc
    record[pe_column] = pe


def _record_roi_features(
    record: Dict[str, float],
    segment_name: str,
    band: str,
    roi_map: Dict[str, List[int]],
    lzc_per_channel: np.ndarray,
    pe_per_channel: np.ndarray,
) -> None:
    """Record ROI-level complexity features."""
    for roi_name, channel_indices in roi_map.items():
        if not channel_indices:
            continue
        
        lzc_value = float(np.nanmean(lzc_per_channel[channel_indices]))
        pe_value = float(np.nanmean(pe_per_channel[channel_indices]))
        
        lzc_column = NamingSchema.build("comp", segment_name, band, "roi", "lzc", channel=roi_name)
        pe_column = NamingSchema.build("comp", segment_name, band, "roi", "pe", channel=roi_name)
        record[lzc_column] = lzc_value
        record[pe_column] = pe_value


def _record_global_features(
    record: Dict[str, float],
    segment_name: str,
    band: str,
    lzc_per_channel: np.ndarray,
    pe_per_channel: np.ndarray,
) -> None:
    """Record global complexity features."""
    lzc_global = float(np.nanmean(lzc_per_channel))
    pe_global = float(np.nanmean(pe_per_channel))
    
    lzc_column = NamingSchema.build("comp", segment_name, band, "global", "lzc")
    pe_column = NamingSchema.build("comp", segment_name, band, "global", "pe")
    record[lzc_column] = lzc_global
    record[pe_column] = pe_global


def _process_complexity_epoch(
    ep_idx: int,
    band_data: Dict[str, Any],
    ch_names: List[str],
    segment_mask: np.ndarray,
    segment_name: str,
    params: Dict[str, Any],
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
    sfreq: float,
) -> Dict[str, float]:
    """Process complexity for a single epoch (parallel worker)."""
    record: Dict[str, float] = {}
    
    for band, band_data_obj in band_data.items():
        envelope = band_data_obj.envelope[ep_idx]
        envelope_segment = (
            envelope[:, segment_mask] if not isinstance(segment_mask, slice) else envelope
        )
        
        if envelope_segment.shape[1] < MIN_SAMPLES_FOR_COMPLEXITY:
            continue

        n_channels = len(ch_names)
        lzc_per_channel = np.full(n_channels, np.nan)
        pe_per_channel = np.full(n_channels, np.nan)
        
        for ch_idx, channel_name in enumerate(ch_names):
            trace = envelope_segment[ch_idx]
            lzc, pe = _compute_channel_complexity(trace, sfreq, params)
            
            lzc_per_channel[ch_idx] = lzc
            pe_per_channel[ch_idx] = pe
            
            if "channels" in spatial_modes:
                _record_channel_features(record, segment_name, band, channel_name, lzc, pe)
        
        if "roi" in spatial_modes and roi_map:
            _record_roi_features(record, segment_name, band, roi_map, lzc_per_channel, pe_per_channel)

        if "global" in spatial_modes:
            _record_global_features(record, segment_name, band, lzc_per_channel, pe_per_channel)
    
    return record


def _compute_complexity_for_segment(
    precomputed: Any,
    segment_mask: np.ndarray,
    segment_name: str,
    params: Dict[str, Any],
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
    n_jobs: int = 1,
) -> List[Dict[str, float]]:
    """Compute complexity metrics for a single segment using precomputed envelopes."""
    n_epochs = precomputed.data.shape[0]
    sampling_freq = float(getattr(precomputed, "sfreq", np.nan))
    
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_complexity_epoch)(
                ep_idx,
                precomputed.band_data,
                precomputed.ch_names,
                segment_mask,
                segment_name,
                params,
                spatial_modes,
                roi_map,
                sampling_freq,
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_complexity_epoch(
                ep_idx,
                precomputed.band_data,
                precomputed.ch_names,
                segment_mask,
                segment_name,
                params,
                spatial_modes,
                roi_map,
                sampling_freq,
            )
            for ep_idx in range(n_epochs)
        ]
    
    return records


def _prepare_spatial_configuration(
    precomputed: Any, spatial_modes: List[str]
) -> Dict[str, List[int]]:
    """Prepare ROI mapping if ROI mode is enabled."""
    roi_map: Dict[str, List[int]] = {}
    
    if "roi" in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        
        roi_definitions = get_roi_definitions(precomputed.config)
        if roi_definitions:
            roi_map = build_roi_map(precomputed.ch_names, roi_definitions)
    
    return roi_map


def _merge_segment_records(
    all_records: List[Dict[str, float]], segment_records: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    """Merge segment records into existing records."""
    if not all_records:
        return segment_records
    
    for epoch_idx, segment_record in enumerate(segment_records):
        all_records[epoch_idx].update(segment_record)
    
    return all_records


def extract_complexity_from_precomputed(
    precomputed: Any,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute complexity metrics (LZC, permutation entropy) using precomputed data."""
    if precomputed.windows is None:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    
    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    logger = getattr(precomputed, "logger", None)
    
    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segments = {target_name: mask}
        else:
            if logger:
                logger.warning(
                    "Complexity: targeted window '%s' has no valid mask; using full epoch.",
                    target_name,
                )
            segments = {target_name: np.ones(len(precomputed.times), dtype=bool)}
    else:
        segments = get_segment_masks(precomputed.times, windows, precomputed.config)
    
    params = _extract_params(precomputed.config)
    spatial_modes = getattr(precomputed, "spatial_modes", None) or ["roi", "global"]
    roi_map = _prepare_spatial_configuration(precomputed, spatial_modes)
    
    all_records: List[Dict[str, float]] = []
    
    for segment_label, segment_mask in segments.items():
        if segment_mask is None or not np.any(segment_mask):
            continue
        
        segment_records = _compute_complexity_for_segment(
            precomputed,
            segment_mask,
            segment_label,
            params,
            spatial_modes,
            roi_map,
            n_jobs=n_jobs,
        )
        all_records = _merge_segment_records(all_records, segment_records)
    
    if not all_records or all(len(record) == 0 for record in all_records):
        return pd.DataFrame(), []

    features_df = pd.DataFrame(all_records)
    feature_columns = list(features_df.columns)
    return features_df, feature_columns
