"""
Burst Feature Extraction
========================

Detects oscillatory bursts using band-limited amplitude envelopes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.windowing import get_segment_masks


_MAD_TO_STD_SCALE = 1.4826
_MIN_PERCENTILE = 50.0
_MAX_PERCENTILE = 99.9
_MS_TO_SECONDS = 1000.0


def _find_burst_intervals(
    above_threshold: np.ndarray,
    n_samples: int,
) -> Tuple[List[int], List[int]]:
    """Find start and end indices of intervals where trace exceeds threshold."""
    diff = np.diff(above_threshold.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)

    if above_threshold[0]:
        starts = [0] + starts
    if above_threshold[-1]:
        ends = ends + [n_samples]

    return starts, ends


def _extract_burst_metrics(
    trace: np.ndarray,
    sfreq: float,
    threshold: float,
    min_samples: int,
) -> Dict[str, float]:
    """Extract burst detection metrics from a single trace."""
    above_threshold = np.asarray(trace) > threshold
    n_samples = int(trace.size)
    duration_sec = float(n_samples / sfreq) if sfreq > 0 else np.nan

    has_valid_duration = np.isfinite(duration_sec) and duration_sec > 0
    has_samples = n_samples > 0
    has_above_threshold = np.any(above_threshold)

    if not has_samples or not has_above_threshold:
        return {
            "count": 0.0,
            "rate": 0.0 if has_valid_duration else np.nan,
            "duration_mean": np.nan,
            "amp_mean": np.nan,
            "fraction": 0.0,
        }

    starts, ends = _find_burst_intervals(above_threshold, n_samples)
    durations = np.array([end - start for start, end in zip(starts, ends)], dtype=int)
    meets_min_duration = durations >= min_samples

    if not np.any(meets_min_duration):
        fraction_above = float(np.mean(above_threshold))
        return {
            "count": 0.0,
            "rate": 0.0 if has_valid_duration else np.nan,
            "duration_mean": np.nan,
            "amp_mean": np.nan,
            "fraction": fraction_above,
        }

    valid_starts = [start for start, keep in zip(starts, meets_min_duration) if keep]
    valid_ends = [end for end, keep in zip(ends, meets_min_duration) if keep]
    valid_durations = durations[meets_min_duration]

    peak_amplitudes = [
        float(np.nanmax(trace[start:end]))
        for start, end in zip(valid_starts, valid_ends)
    ]

    burst_count = float(len(valid_durations))
    mean_duration_sec = (
        float(np.mean(valid_durations) / sfreq) if sfreq > 0 else np.nan
    )
    mean_amplitude = (
        float(np.mean(peak_amplitudes)) if peak_amplitudes else np.nan
    )
    burst_rate = (
        float(burst_count / duration_sec) if has_valid_duration else np.nan
    )
    fraction_above = float(np.mean(above_threshold))

    return {
        "count": burst_count,
        "rate": burst_rate,
        "duration_mean": mean_duration_sec,
        "amp_mean": mean_amplitude,
        "fraction": fraction_above,
    }


def _parse_burst_config(
    config: Any,
    default_bands: List[str],
) -> Dict[str, Any]:
    """Parse burst detection configuration from context."""
    burst_config = config.get("feature_engineering.bursts", {}) if hasattr(config, "get") else {}
    
    threshold_method = str(burst_config.get("threshold_method", "percentile")).strip().lower()
    valid_methods = {"percentile", "zscore", "mad"}
    if threshold_method not in valid_methods:
        threshold_method = "percentile"

    threshold_reference = str(burst_config.get("threshold_reference", "trial")).strip().lower()
    if threshold_reference not in {"trial", "subject", "condition"}:
        threshold_reference = "subject"

    return {
        "bands": burst_config.get("bands") or default_bands,
        "threshold_method": threshold_method,
        "threshold_reference": threshold_reference,
        "threshold_z": float(burst_config.get("threshold_z", 2.0)),
        "threshold_percentile": float(burst_config.get("threshold_percentile", 95.0)),
        "min_duration_ms": float(burst_config.get("min_duration_ms", 50.0)),
        "min_cycles": float(burst_config.get("min_cycles", 3.0)),
        "min_trials_per_condition": int(burst_config.get("min_trials_per_condition", 10)),
    }


def _compute_thresholds_zscore(
    baseline_envelope: np.ndarray,
    threshold_z: float,
) -> np.ndarray:
    """Compute thresholds using z-score method."""
    baseline_mean = np.nanmean(baseline_envelope, axis=2)
    baseline_std = np.nanstd(baseline_envelope, axis=2)
    return baseline_mean + (threshold_z * baseline_std)


def _compute_thresholds_mad(
    baseline_envelope: np.ndarray,
    threshold_z: float,
) -> np.ndarray:
    """Compute thresholds using median absolute deviation method."""
    baseline_median = np.nanmedian(baseline_envelope, axis=2)
    median_abs_deviation = np.nanmedian(
        np.abs(baseline_envelope - baseline_median[:, :, None]),
        axis=2,
    )
    return baseline_median + (threshold_z * _MAD_TO_STD_SCALE * median_abs_deviation)


def _compute_thresholds_percentile(
    baseline_envelope: np.ndarray,
    threshold_percentile: float,
) -> np.ndarray:
    """Compute thresholds using percentile method."""
    clipped_percentile = float(np.clip(threshold_percentile, _MIN_PERCENTILE, _MAX_PERCENTILE))
    return np.nanpercentile(baseline_envelope, q=clipped_percentile, axis=2)

def _compute_thresholds_subject(
    baseline_envelope: np.ndarray,
    method: str,
    threshold_z: float,
    threshold_percentile: float,
) -> np.ndarray:
    """Compute burst thresholds per channel using all baseline samples across epochs."""
    if method == "zscore":
        mean = np.nanmean(baseline_envelope, axis=(0, 2))
        std = np.nanstd(baseline_envelope, axis=(0, 2))
        thr = mean + (threshold_z * std)
        return np.broadcast_to(thr[None, :], (baseline_envelope.shape[0], baseline_envelope.shape[1]))

    if method == "mad":
        median = np.nanmedian(baseline_envelope, axis=(0, 2))
        mad = np.nanmedian(np.abs(baseline_envelope - median[None, :, None]), axis=(0, 2))
        thr = median + (threshold_z * _MAD_TO_STD_SCALE * mad)
        return np.broadcast_to(thr[None, :], (baseline_envelope.shape[0], baseline_envelope.shape[1]))

    clipped_percentile = float(np.clip(threshold_percentile, _MIN_PERCENTILE, _MAX_PERCENTILE))
    thr = np.nanpercentile(baseline_envelope, q=clipped_percentile, axis=(0, 2))
    return np.broadcast_to(thr[None, :], (baseline_envelope.shape[0], baseline_envelope.shape[1]))


def _extract_condition_labels(ctx: Any, n_epochs: int) -> Optional[np.ndarray]:
    labels = getattr(ctx, "condition", None)
    if labels is not None:
        labels = np.asarray(labels)
        return labels if labels.shape[0] == n_epochs else None

    aligned = getattr(ctx, "aligned_events", None)
    if isinstance(aligned, pd.DataFrame) and "condition" in aligned.columns:
        labels = aligned["condition"].to_numpy()
        return labels if labels.shape[0] == n_epochs else None

    return None


def _compute_thresholds_condition(
    baseline_envelope: np.ndarray,
    condition_labels: np.ndarray,
    *,
    method: str,
    threshold_z: float,
    threshold_percentile: float,
    min_trials_per_condition: int,
) -> np.ndarray:
    """Compute burst thresholds per channel within each condition group."""
    n_epochs, n_channels, _ = baseline_envelope.shape
    out = np.full((n_epochs, n_channels), np.nan)

    for cond in np.unique(condition_labels):
        mask = condition_labels == cond
        n = int(np.sum(mask))
        if n < int(min_trials_per_condition):
            continue
        out[mask] = _compute_thresholds_subject(
            baseline_envelope[mask],
            method=method,
            threshold_z=threshold_z,
            threshold_percentile=threshold_percentile,
        )

    # Fill any missing conditions with subject-level thresholds.
    if not np.isfinite(out).any():
        return _compute_thresholds_subject(
            baseline_envelope,
            method=method,
            threshold_z=threshold_z,
            threshold_percentile=threshold_percentile,
        )

    missing = ~np.isfinite(out).any(axis=1)
    if np.any(missing):
        subj = _compute_thresholds_subject(
            baseline_envelope,
            method=method,
            threshold_z=threshold_z,
            threshold_percentile=threshold_percentile,
        )
        out[missing] = subj[missing]

    return out


def _compute_thresholds_condition_trainmask(
    baseline_envelope: np.ndarray,
    condition_labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    method: str,
    threshold_z: float,
    threshold_percentile: float,
    min_trials_per_condition: int,
) -> np.ndarray:
    """Compute condition thresholds from training trials only; apply to all epochs."""
    n_epochs, n_channels, _ = baseline_envelope.shape
    train_mask = np.asarray(train_mask, dtype=bool).ravel()
    if train_mask.size != n_epochs or not np.any(train_mask):
        return _compute_thresholds_subject(
            baseline_envelope,
            method=method,
            threshold_z=threshold_z,
            threshold_percentile=threshold_percentile,
        )

    subj_train = _compute_thresholds_subject(
        baseline_envelope[train_mask],
        method=method,
        threshold_z=threshold_z,
        threshold_percentile=threshold_percentile,
    )
    subj_thr = subj_train[0] if subj_train.shape[0] else np.full((n_channels,), np.nan)

    out = np.broadcast_to(subj_thr[None, :], (n_epochs, n_channels)).copy()
    for cond in np.unique(condition_labels):
        cond_all = condition_labels == cond
        cond_train = cond_all & train_mask
        n_train = int(np.sum(cond_train))
        if n_train < int(min_trials_per_condition):
            continue
        thr_train = _compute_thresholds_subject(
            baseline_envelope[cond_train],
            method=method,
            threshold_z=threshold_z,
            threshold_percentile=threshold_percentile,
        )
        thr_vec = thr_train[0] if thr_train.shape[0] else subj_thr
        out[cond_all] = np.broadcast_to(thr_vec[None, :], (int(np.sum(cond_all)), n_channels))

    return out


def _compute_burst_thresholds(
    baseline_envelope: np.ndarray,
    method: str,
    threshold_z: float,
    threshold_percentile: float,
) -> np.ndarray:
    """Compute burst detection thresholds from baseline envelope."""
    if method == "zscore":
        return _compute_thresholds_zscore(baseline_envelope, threshold_z)
    elif method == "mad":
        return _compute_thresholds_mad(baseline_envelope, threshold_z)
    else:
        return _compute_thresholds_percentile(baseline_envelope, threshold_percentile)


def _compute_min_samples(
    min_duration_ms: float,
    sfreq: float,
    band_data: Any,
    min_cycles: float,
) -> int:
    """Compute minimum samples for burst detection from duration and cycle constraints."""
    min_samples_from_duration = (
        max(1, int(round(min_duration_ms * sfreq / _MS_TO_SECONDS)))
        if sfreq > 0
        else 1
    )

    try:
        fmin = float(band_data.fmin)
        fmax = float(band_data.fmax)
        center_frequency = float(np.sqrt(fmin * fmax)) if fmin > 0 else np.nan

        has_valid_frequency = (
            np.isfinite(center_frequency)
            and center_frequency > 0
            and np.isfinite(min_cycles)
            and min_cycles > 0
        )

        if has_valid_frequency:
            min_samples_from_cycles = int(round((min_cycles / center_frequency) * sfreq))
            return max(min_samples_from_duration, max(1, min_samples_from_cycles))
    except (AttributeError, ValueError, TypeError):
        pass

    return min_samples_from_duration


def _build_roi_map(
    config: Any,
    channel_names: List[str],
    spatial_modes: List[str],
) -> Dict[str, List[int]]:
    """Build ROI mapping from configuration if ROI mode is enabled."""
    if "roi" not in spatial_modes:
        return {}

    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
    from eeg_pipeline.utils.analysis.channels import build_roi_map

    roi_definitions = get_roi_definitions(config)
    if not roi_definitions:
        return {}

    return build_roi_map(channel_names, roi_definitions)


def _process_channel_features(
    segment_envelope: np.ndarray,
    thresholds: np.ndarray,
    segment_name: str,
    band: str,
    channel_names: List[str],
    epoch_index: int,
    sfreq: float,
    min_samples: int,
    record: Dict[str, float],
) -> None:
    """Extract burst features for individual channels."""
    for channel_index, channel_name in enumerate(channel_names):
        channel_trace = segment_envelope[epoch_index, channel_index]
        channel_threshold = thresholds[epoch_index, channel_index]
        
        metrics = _extract_burst_metrics(
            channel_trace,
            sfreq,
            channel_threshold,
            min_samples,
        )
        
        for statistic, value in metrics.items():
            column_name = NamingSchema.build(
                "bursts", segment_name, band, "ch", statistic, channel=channel_name
            )
            record[column_name] = float(value)


def _process_roi_features(
    segment_envelope: np.ndarray,
    thresholds: np.ndarray,
    segment_name: str,
    band: str,
    roi_map: Dict[str, List[int]],
    epoch_index: int,
    sfreq: float,
    min_samples: int,
    record: Dict[str, float],
) -> None:
    """Extract burst features for ROI aggregations."""
    for roi_name, channel_indices in roi_map.items():
        if not channel_indices:
            continue

        roi_trace = np.nanmean(segment_envelope[epoch_index, channel_indices], axis=0)
        roi_threshold = float(np.nanmean(thresholds[epoch_index, channel_indices]))
        
        metrics = _extract_burst_metrics(roi_trace, sfreq, roi_threshold, min_samples)
        
        for statistic, value in metrics.items():
            column_name = NamingSchema.build(
                "bursts", segment_name, band, "roi", statistic, channel=roi_name
            )
            record[column_name] = float(value)


def _process_global_features(
    segment_envelope: np.ndarray,
    thresholds: np.ndarray,
    segment_name: str,
    band: str,
    epoch_index: int,
    sfreq: float,
    min_samples: int,
    record: Dict[str, float],
) -> None:
    """Extract burst features for global (all-channel) aggregation."""
    global_trace = np.nanmean(segment_envelope[epoch_index], axis=0)
    global_threshold = float(np.nanmean(thresholds[epoch_index]))
    
    metrics = _extract_burst_metrics(global_trace, sfreq, global_threshold, min_samples)
    
    for statistic, value in metrics.items():
        column_name = NamingSchema.build(
            "bursts", segment_name, band, "global", statistic
        )
        record[column_name] = float(value)


def extract_burst_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract burst detection features from precomputed band envelopes."""
    precomputed = getattr(ctx, "precomputed", None)
    if precomputed is None:
        ctx.logger.warning("Bursts: missing precomputed intermediates; skipping extraction.")
        return pd.DataFrame(), []

    is_valid, error_message = validate_precomputed(
        precomputed, require_windows=True, require_bands=True
    )
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("Bursts: %s; skipping extraction.", error_message)
        return pd.DataFrame(), []

    config = ctx.config if hasattr(ctx, "config") else precomputed.config
    burst_config = _parse_burst_config(config, bands)

    sampling_frequency = float(getattr(precomputed, "sfreq", np.nan))
    baseline_mask = precomputed.windows.get_mask("baseline")
    
    if baseline_mask is None or not np.any(baseline_mask):
        ctx.logger.warning("Bursts: baseline window missing; skipping extraction.")
        return pd.DataFrame(), []

    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    allow_full_epoch_fallback = bool(
        config.get("feature_engineering.windows.allow_full_epoch_fallback", False)
        if hasattr(config, "get")
        else False
    )
    
    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segment_masks = {target_name: mask}
        else:
            if ctx.logger:
                if allow_full_epoch_fallback:
                    ctx.logger.warning(
                        "Bursts: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                        target_name,
                    )
                else:
                    ctx.logger.error(
                        "Bursts: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                segment_masks = {target_name: np.ones(len(precomputed.times), dtype=bool)}
            else:
                return pd.DataFrame(), []
    else:
        segment_masks = get_segment_masks(precomputed.times, windows, precomputed.config)
    segment_names = [name for name in segment_masks.keys() if name != "baseline"]
    
    if not segment_names:
        return pd.DataFrame(), []

    spatial_modes = getattr(ctx, "spatial_modes", ["roi", "global"])
    roi_map = _build_roi_map(config, precomputed.ch_names, spatial_modes)

    n_epochs = precomputed.data.shape[0]
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    analysis_mode = str(getattr(ctx, "analysis_mode", "") or "").strip().lower()
    train_mask = getattr(precomputed, "train_mask", None)
    if train_mask is None:
        train_mask = getattr(ctx, "train_mask", None)

    threshold_method = str(burst_config["threshold_method"])
    threshold_reference = str(burst_config.get("threshold_reference", "subject")).strip().lower()
    if analysis_mode == "trial_ml_safe" and threshold_reference in {"subject", "condition"} and train_mask is None:
        ctx.logger.warning(
            "Bursts: threshold_reference=%s requires cross-trial baselines; forcing threshold_reference='trial' "
            "in trial_ml_safe mode without train_mask.",
            threshold_reference,
        )
        threshold_reference = "trial"

    for band in burst_config["bands"]:
        if band not in precomputed.band_data:
            continue

        envelope = precomputed.band_data[band].envelope
        baseline_envelope = envelope[:, :, baseline_mask]

        if threshold_reference == "trial":
            thresholds = _compute_burst_thresholds(
                baseline_envelope,
                threshold_method,
                burst_config["threshold_z"],
                burst_config["threshold_percentile"],
            )
        elif threshold_reference == "condition":
            condition_labels = _extract_condition_labels(ctx, n_epochs)
            if condition_labels is None:
                thresholds_src = baseline_envelope[train_mask] if train_mask is not None else baseline_envelope
                thresholds_train = _compute_thresholds_subject(
                    thresholds_src,
                    method=threshold_method,
                    threshold_z=burst_config["threshold_z"],
                    threshold_percentile=burst_config["threshold_percentile"],
                )
                thresholds = (
                    np.broadcast_to(thresholds_train[0][None, :], (n_epochs, thresholds_train.shape[1]))
                    if thresholds_train.size
                    else np.full((n_epochs, baseline_envelope.shape[1]), np.nan)
                )
            elif train_mask is not None:
                thresholds = _compute_thresholds_condition_trainmask(
                    baseline_envelope,
                    condition_labels,
                    train_mask,
                    method=threshold_method,
                    threshold_z=burst_config["threshold_z"],
                    threshold_percentile=burst_config["threshold_percentile"],
                    min_trials_per_condition=burst_config.get("min_trials_per_condition", 10),
                )
            else:
                thresholds = _compute_thresholds_condition(
                    baseline_envelope,
                    condition_labels,
                    method=threshold_method,
                    threshold_z=burst_config["threshold_z"],
                    threshold_percentile=burst_config["threshold_percentile"],
                    min_trials_per_condition=burst_config.get("min_trials_per_condition", 10),
                )
        else:
            thresholds_src = baseline_envelope[train_mask] if train_mask is not None else baseline_envelope
            thresholds_train = _compute_thresholds_subject(
                thresholds_src,
                method=threshold_method,
                threshold_z=burst_config["threshold_z"],
                threshold_percentile=burst_config["threshold_percentile"],
            )
            thresholds = (
                np.broadcast_to(thresholds_train[0][None, :], (n_epochs, thresholds_train.shape[1]))
                if thresholds_train.size
                else np.full((n_epochs, baseline_envelope.shape[1]), np.nan)
            )

        min_samples = _compute_min_samples(
            burst_config["min_duration_ms"],
            sampling_frequency,
            precomputed.band_data[band],
            burst_config["min_cycles"],
        )

        for segment_name in segment_names:
            segment_mask = segment_masks.get(segment_name)
            if segment_mask is None or not np.any(segment_mask):
                continue

            segment_envelope = envelope[:, :, segment_mask]
            if segment_envelope.shape[-1] < min_samples:
                continue

            for epoch_index in range(n_epochs):
                record = records[epoch_index]

                if "channels" in spatial_modes:
                    _process_channel_features(
                        segment_envelope,
                        thresholds,
                        segment_name,
                        band,
                        precomputed.ch_names,
                        epoch_index,
                        sampling_frequency,
                        min_samples,
                        record,
                    )

                if "roi" in spatial_modes and roi_map:
                    _process_roi_features(
                        segment_envelope,
                        thresholds,
                        segment_name,
                        band,
                        roi_map,
                        epoch_index,
                        sampling_frequency,
                        min_samples,
                        record,
                    )

                if "global" in spatial_modes:
                    _process_global_features(
                        segment_envelope,
                        thresholds,
                        segment_name,
                        band,
                        epoch_index,
                        sampling_frequency,
                        min_samples,
                        record,
                    )

    if not records or all(len(record) == 0 for record in records):
        return pd.DataFrame(), []

    dataframe = pd.DataFrame(records)
    dataframe.attrs["threshold_reference"] = str(threshold_reference)
    dataframe.attrs["threshold_method"] = str(threshold_method)
    dataframe.attrs["threshold_train_mask_used"] = bool(
        train_mask is not None and threshold_reference in {"subject", "condition"}
    )
    return dataframe, list(dataframe.columns)


__all__ = ["extract_burst_features"]
