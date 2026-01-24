"""
ERP/LEP Feature Extraction
==========================

Time-domain features for pain-related evoked potentials.
Computed per-trial within user-defined time windows, including
peak-to-peak features for matching N/P components (e.g., N2-P2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import MIN_SAMPLES_DEFAULT, validate_extractor_inputs
from eeg_pipeline.utils.analysis.channels import pick_eeg_channels, build_roi_map
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
from eeg_pipeline.utils.analysis.windowing import get_segment_masks


_MICROVOLTS_TO_VOLTS = 1e-6
_SAVGOL_POLYORDER = 2
_MIN_SMOOTH_WINDOW_LENGTH = 5
_DEFAULT_LOWPASS_HZ = 30.0
_MILLISECONDS_PER_SECOND = 1000.0
_MIN_EPOCHS_FOR_ERP = 2


def _infer_peak_mode(
    window_name: str,
    *,
    peak_mode_by_segment: Optional[Dict[str, str]] = None,
) -> str:
    name_raw = str(window_name).strip()
    name = name_raw.lower()

    if peak_mode_by_segment:
        direct = peak_mode_by_segment.get(name_raw) or peak_mode_by_segment.get(name)
        if direct is not None:
            mode = str(direct).strip().lower()
            if mode in {"neg", "pos", "abs"}:
                return mode
            if mode in {"n", "negative"}:
                return "neg"
            if mode in {"p", "positive"}:
                return "pos"

    if name.startswith("n"):
        return "neg"
    if name.startswith("p"):
        return "pos"
    return "abs"


def _parse_peak_label(window_name: str) -> Optional[Tuple[str, str]]:
    name = str(window_name).strip().lower()
    if not name:
        return None
    polarity = name[0]
    if polarity not in {"n", "p"}:
        return None
    suffix = name[1:]
    if not suffix:
        return None
    return polarity, suffix


def _apply_smoothing(
    data: np.ndarray,
    smooth_samples: int,
) -> np.ndarray:
    n_times = data.shape[2]
    
    if smooth_samples < _MIN_SMOOTH_WINDOW_LENGTH or smooth_samples >= n_times:
        return data
    
    window_length = smooth_samples if smooth_samples % 2 == 1 else smooth_samples + 1
    
    return savgol_filter(
        data,
        window_length=window_length,
        polyorder=_SAVGOL_POLYORDER,
        axis=2,
        mode="interp",
    )


def _find_peak_in_signal(
    signal: np.ndarray,
    times: np.ndarray,
    mode: str,
    prominence: Optional[float],
) -> Tuple[float, float]:
    cleaned_signal = np.nan_to_num(signal, nan=np.nanmedian(signal))
    
    if mode == "neg":
        search_signal = -cleaned_signal
    elif mode == "pos":
        search_signal = cleaned_signal
    else:
        search_signal = np.abs(cleaned_signal)
    
    has_valid_prominence = (
        prominence is not None
        and np.isfinite(prominence)
        and prominence > 0
    )
    
    if has_valid_prominence:
        peaks, properties = find_peaks(search_signal, prominence=prominence)
        if peaks.size > 0:
            prominences = properties.get("prominences", np.ones_like(peaks))
            best_peak_idx = peaks[np.argmax(prominences)]
            return float(cleaned_signal[best_peak_idx]), float(times[best_peak_idx])
    
    best_peak_idx = np.nanargmax(search_signal)
    return float(cleaned_signal[best_peak_idx]), float(times[best_peak_idx])


def _compute_peaks(
    data: np.ndarray,
    times: np.ndarray,
    mode: str,
    *,
    smooth_samples: int = 0,
    prominence: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_epochs, n_series, _ = data.shape
    peak_vals = np.full((n_epochs, n_series), np.nan)
    peak_times = np.full((n_epochs, n_series), np.nan)
    
    smoothed_data = _apply_smoothing(data, smooth_samples)
    has_finite = np.isfinite(smoothed_data).any(axis=2)
    
    for epoch_idx in range(n_epochs):
        for series_idx in range(n_series):
            if not has_finite[epoch_idx, series_idx]:
                continue
            
            signal = smoothed_data[epoch_idx, series_idx]
            peak_value, peak_time = _find_peak_in_signal(
                signal,
                times,
                mode,
                prominence,
            )
            peak_vals[epoch_idx, series_idx] = peak_value
            peak_times[epoch_idx, series_idx] = peak_time
    
    return peak_vals, peak_times


def _build_feature_names(
    segment: str,
    scope: str,
    peak_mode: str,
    channel_name: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    peak_stat = f"peak_{peak_mode}"
    latency_stat = f"latency_{peak_mode}"
    
    if scope == "global":
        mean_name = NamingSchema.build("erp", segment, "broadband", "global", "mean")
        peak_name = NamingSchema.build("erp", segment, "broadband", "global", peak_stat)
        latency_name = NamingSchema.build("erp", segment, "broadband", "global", latency_stat)
        auc_name = NamingSchema.build("erp", segment, "broadband", "global", "auc")
    else:
        mean_name = NamingSchema.build("erp", segment, "broadband", scope, "mean", channel=channel_name)
        peak_name = NamingSchema.build("erp", segment, "broadband", scope, peak_stat, channel=channel_name)
        latency_name = NamingSchema.build("erp", segment, "broadband", scope, latency_stat, channel=channel_name)
        auc_name = NamingSchema.build("erp", segment, "broadband", scope, "auc", channel=channel_name)
    
    return mean_name, peak_name, latency_name, auc_name


def _compute_auc(
    data: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    has_finite = np.isfinite(data).any(axis=2)
    auc_vals = np.trapz(np.nan_to_num(data, nan=0.0), times, axis=2)
    auc_vals[~has_finite] = np.nan
    return auc_vals


def _append_series_features(
    output: Dict[str, np.ndarray],
    data: np.ndarray,
    names: List[str],
    times: np.ndarray,
    segment: str,
    scope: str,
    peak_mode: str,
    *,
    smooth_samples: int = 0,
    prominence: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mean_vals = np.nanmean(data, axis=2)
    peak_vals, peak_times = _compute_peaks(
        data,
        times,
        peak_mode,
        smooth_samples=smooth_samples,
        prominence=prominence,
    )
    auc_vals = _compute_auc(data, times)
    
    for idx, channel_name in enumerate(names):
        mean_name, peak_name, latency_name, auc_name = _build_feature_names(
            segment,
            scope,
            peak_mode,
            channel_name if scope != "global" else None,
        )
        
        output[mean_name] = mean_vals[:, idx]
        output[peak_name] = peak_vals[:, idx]
        output[latency_name] = peak_times[:, idx]
        output[auc_name] = auc_vals[:, idx]
    
    return peak_vals, peak_times


def _build_peak_pair_names(
    pair_label: str,
    scope: str,
    channel_name: Optional[str] = None,
) -> Tuple[str, str]:
    if scope == "global":
        ptp_name = NamingSchema.build("erp", pair_label, "broadband", "global", "ptp")
        latency_diff_name = NamingSchema.build("erp", pair_label, "broadband", "global", "latency_diff")
    else:
        ptp_name = NamingSchema.build("erp", pair_label, "broadband", scope, "ptp", channel=channel_name)
        latency_diff_name = NamingSchema.build("erp", pair_label, "broadband", scope, "latency_diff", channel=channel_name)
    return ptp_name, latency_diff_name


def _append_peak_pair_features(
    output: Dict[str, np.ndarray],
    *,
    scope: str,
    pair_label: str,
    names: List[str],
    ptp_vals: np.ndarray,
    lat_diff: np.ndarray,
) -> None:
    for idx, channel_name in enumerate(names):
        ptp_name, latency_diff_name = _build_peak_pair_names(
            pair_label,
            scope,
            channel_name if scope != "global" else None,
        )
        output[ptp_name] = ptp_vals[:, idx]
        output[latency_diff_name] = lat_diff[:, idx]


def _build_component_masks(
    times: np.ndarray,
    erp_cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    components = erp_cfg.get("components", [])
    if not isinstance(components, list):
        return {}
    
    masks: Dict[str, np.ndarray] = {}
    for comp in components:
        if not isinstance(comp, dict):
            continue
        
        name = str(comp.get("name", "")).strip().lower()
        start = comp.get("start")
        end = comp.get("end")
        
        if not name or start is None or end is None:
            continue
        
        try:
            start_time = float(start)
            end_time = float(end)
        except (TypeError, ValueError):
            continue
        
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            continue
        if end_time <= start_time:
            continue
        
        mask = (times >= start_time) & (times < end_time)
        if np.any(mask):
            masks[name] = mask
    
    return masks


def _parse_erp_config(
    config: Any,
) -> Dict[str, Any]:
    return config.get("feature_engineering.erp", {})


def _parse_lowpass_filter(
    erp_cfg: Dict[str, Any],
) -> Optional[float]:
    lowpass_hz = erp_cfg.get("lowpass_hz", _DEFAULT_LOWPASS_HZ)
    if lowpass_hz is None:
        return None
    
    try:
        lowpass_hz = float(lowpass_hz)
    except (TypeError, ValueError):
        return _DEFAULT_LOWPASS_HZ
    
    if not np.isfinite(lowpass_hz) or lowpass_hz <= 0:
        return None
    
    return lowpass_hz


def _apply_lowpass_filter(
    epochs: Any,
    picks: np.ndarray,
    lowpass_hz: float,
    logger: Any,
) -> np.ndarray:
    try:
        filtered_epochs = epochs.copy().filter(
            l_freq=None,
            h_freq=lowpass_hz,
            picks=picks,
            verbose=False,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"ERP low-pass filtering failed (h_freq={lowpass_hz}).") from exc

    data = filtered_epochs.get_data(picks=picks)
    logger.info("ERP: Applied %.1f Hz low-pass filter for peak detection", lowpass_hz)
    return data


def _parse_smoothing_config(
    erp_cfg: Dict[str, Any],
    sampling_rate: float,
) -> int:
    smooth_ms = erp_cfg.get("smooth_ms", 0.0)
    try:
        smooth_ms = float(smooth_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ERP smooth_ms must be a float (got {smooth_ms!r}).") from exc
    
    if smooth_ms <= 0:
        return 0
    
    return int(round(sampling_rate * smooth_ms / _MILLISECONDS_PER_SECOND))


def _parse_peak_prominence(
    erp_cfg: Dict[str, Any],
) -> Optional[float]:
    peak_prom_uv = erp_cfg.get("peak_prominence_uv", None)
    if peak_prom_uv is None:
        return None
    
    try:
        peak_prominence = float(peak_prom_uv) * _MICROVOLTS_TO_VOLTS
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ERP peak_prominence_uv must be a float (got {peak_prom_uv!r}).") from exc
    
    return peak_prominence


def _compute_baseline_mask_for_times(
    times: np.ndarray,
    windows: Any,
) -> Optional[np.ndarray]:
    """Compute baseline mask relative to the given times array.
    
    This handles the case where epochs have been cropped to a specific time range,
    so we need to recompute the baseline mask against the current times rather than
    using a pre-computed mask from the original full epoch.
    """
    if windows is None:
        return None
    
    baseline_range = getattr(windows, "baseline_range", None)
    if baseline_range is None:
        return None
    
    try:
        tmin, tmax = baseline_range
        if not (np.isfinite(tmin) and np.isfinite(tmax)):
            return None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid baseline_range in TimeWindows: {baseline_range!r}") from exc
    
    # Use a half-open interval [tmin, tmax) to avoid including t=0 exactly when
    # users specify baseline_end=0.0 (common in ERP practice).
    mask = (times >= tmin) & (times < tmax)
    return mask if np.any(mask) else None


def _apply_baseline_correction(
    data: np.ndarray,
    times: np.ndarray,
    windows: Any,
    allow_no_baseline: bool,
    logger: Any,
    original_epochs: Any = None,
    picks: Any = None,
) -> np.ndarray:
    """Apply baseline correction using the baseline range from windows.
    
    If the baseline period is not in the current (cropped) data, uses the
    original uncropped epochs to compute the baseline mean.
    """
    baseline_mask = _compute_baseline_mask_for_times(times, windows)
    
    if baseline_mask is not None and np.any(baseline_mask):
        baseline = np.nanmean(data[:, :, baseline_mask], axis=2, keepdims=True)
        return data - baseline
    
    if original_epochs is not None and windows is not None:
        original_times = original_epochs.times
        original_baseline_mask = _compute_baseline_mask_for_times(original_times, windows)
        
        if original_baseline_mask is not None and np.any(original_baseline_mask):
            original_data = original_epochs.get_data(picks=picks)
            baseline = np.nanmean(
                original_data[:, :, original_baseline_mask], axis=2, keepdims=True
            )
            logger.info(
                "ERP: Using baseline from original epochs (baseline period not in current window)"
            )
            return data - baseline
    
    if allow_no_baseline:
        logger.info("ERP: baseline window missing; proceeding without baseline correction.")
        return data
    
    raise ValueError(
        "ERP baseline correction requested but baseline window is missing or empty."
    )


def _process_channel_features(
    output: Dict[str, np.ndarray],
    seg_data: np.ndarray,
    seg_times: np.ndarray,
    ch_names: List[str],
    seg_name: str,
    peak_mode: str,
    smooth_samples: int,
    peak_prominence: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    peak_vals, peak_times = _append_series_features(
        output,
        seg_data,
        ch_names,
        seg_times,
        seg_name,
        "ch",
        peak_mode,
        smooth_samples=smooth_samples,
        prominence=peak_prominence,
    )
    return peak_vals, peak_times


def _process_roi_features(
    output: Dict[str, np.ndarray],
    seg_data: np.ndarray,
    seg_times: np.ndarray,
    roi_map: Dict[str, List[int]],
    seg_name: str,
    peak_mode: str,
    smooth_samples: int,
    peak_prominence: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    roi_names = []
    roi_series = []
    
    for roi_name, channel_indices in roi_map.items():
        if not channel_indices:
            continue
        roi_names.append(roi_name)
        roi_series.append(np.nanmean(seg_data[:, channel_indices, :], axis=1))
    
    if not roi_series:
        return np.array([]), np.array([]), []
    
    roi_stack = np.stack(roi_series, axis=1)
    peak_vals, peak_times = _append_series_features(
        output,
        roi_stack,
        roi_names,
        seg_times,
        seg_name,
        "roi",
        peak_mode,
        smooth_samples=smooth_samples,
        prominence=peak_prominence,
    )
    return peak_vals, peak_times, roi_names


def _process_global_features(
    output: Dict[str, np.ndarray],
    seg_data: np.ndarray,
    seg_times: np.ndarray,
    seg_name: str,
    peak_mode: str,
    smooth_samples: int,
    peak_prominence: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    global_series = np.nanmean(seg_data, axis=1, keepdims=True)
    peak_vals, peak_times = _append_series_features(
        output,
        global_series,
        ["global"],
        seg_times,
        seg_name,
        "global",
        peak_mode,
        smooth_samples=smooth_samples,
        prominence=peak_prominence,
    )
    return peak_vals, peak_times


def _find_matching_peak_pairs(
    segments: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
) -> List[Tuple[str, str, str]]:
    negative_by_suffix: Dict[str, str] = {}
    positive_by_suffix: Dict[str, str] = {}
    
    for segment_name in segments.keys():
        parsed = _parse_peak_label(segment_name)
        if not parsed:
            continue
        
        polarity, suffix = parsed
        if polarity == "n" and suffix not in negative_by_suffix:
            negative_by_suffix[suffix] = segment_name
        elif polarity == "p" and suffix not in positive_by_suffix:
            positive_by_suffix[suffix] = segment_name
    
    matching_pairs = []
    common_suffixes = sorted(set(negative_by_suffix) & set(positive_by_suffix))
    
    for suffix in common_suffixes:
        negative_segment = negative_by_suffix[suffix]
        positive_segment = positive_by_suffix[suffix]
        matching_pairs.append((negative_segment, positive_segment, suffix))
    
    return matching_pairs


def extract_erp_features(
    ctx: Any,  # FeatureContext
) -> Tuple[pd.DataFrame, List[str]]:
    valid, err = validate_extractor_inputs(ctx, "ERP", min_epochs=_MIN_EPOCHS_FOR_ERP)
    if not valid:
        ctx.logger.warning(err)
        return pd.DataFrame(), []

    epochs = ctx.epochs
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        ctx.logger.warning("ERP: No EEG channels available; skipping extraction.")
        return pd.DataFrame(), []

    erp_cfg = _parse_erp_config(ctx.config)
    times = epochs.times
    spatial_modes = getattr(ctx, "spatial_modes", ["roi", "global"])
    
    lowpass_hz = _parse_lowpass_filter(erp_cfg)
    if lowpass_hz is not None:
        data = _apply_lowpass_filter(epochs, picks, lowpass_hz, ctx.logger)
    else:
        data = epochs.get_data(picks=picks)
    
    sampling_rate = float(epochs.info["sfreq"])
    smooth_samples = _parse_smoothing_config(erp_cfg, sampling_rate)
    peak_prominence = _parse_peak_prominence(erp_cfg)
    
    baseline_correction = bool(erp_cfg.get("baseline_correction", True))
    allow_no_baseline = bool(erp_cfg.get("allow_no_baseline", False))

    # Avoid double baseline correction: epochs created by the preprocessing pipeline may
    # already have baseline correction applied (mne.Epochs.baseline != None).
    already_baselined = getattr(epochs, "baseline", None)
    if baseline_correction and already_baselined not in (None, (None, None)):
        ctx.logger.info(
            "ERP: skipping baseline correction because epochs are already baseline-corrected (epochs.baseline=%s).",
            already_baselined,
        )
        baseline_correction = False

    if baseline_correction:
        original_epochs = getattr(ctx, "_original_epochs", None)
        data = _apply_baseline_correction(
            data, times, ctx.windows, allow_no_baseline, ctx.logger,
            original_epochs=original_epochs, picks=picks
        )

    windows = ctx.windows
    target_name = getattr(ctx, "name", None)
    allow_full_epoch_fallback = bool(
        ctx.config.get("feature_engineering.windows.allow_full_epoch_fallback", False)
    )
    
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segment_masks = {target_name: mask}
        else:
            if allow_full_epoch_fallback:
                ctx.logger.warning(
                    "ERP: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                    target_name,
                )
                segment_masks = {target_name: np.ones_like(times, dtype=bool)}
            else:
                ctx.logger.error(
                    "ERP: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                    target_name,
                )
                return pd.DataFrame(), []
    else:
        segment_masks = get_segment_masks(times, windows, ctx.config)
        component_masks = _build_component_masks(times, erp_cfg)
        
        for name, mask in component_masks.items():
            if name not in segment_masks:
                segment_masks[name] = mask
    
    if not segment_masks:
        return pd.DataFrame(), []

    min_samples = int(erp_cfg.get("min_samples", MIN_SAMPLES_DEFAULT))
    output: Dict[str, np.ndarray] = {}
    peak_cache: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]] = {
        "ch": {},
        "roi": {},
        "global": {},
    }

    roi_map = {}
    if "roi" in spatial_modes:
        roi_defs = get_roi_definitions(ctx.config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)

    for seg_name, mask in segment_masks.items():
        if seg_name == "baseline":
            continue
        if mask is None or np.count_nonzero(mask) < min_samples:
            continue
        
        seg_times = times[mask]
        seg_data = data[:, :, mask]
        peak_mode = _infer_peak_mode(seg_name, peak_mode_by_segment=erp_cfg.get("peak_mode_by_segment"))

        if "channels" in spatial_modes:
            peak_vals, peak_times = _process_channel_features(
                output,
                seg_data,
                seg_times,
                ch_names,
                seg_name,
                peak_mode,
                smooth_samples,
                peak_prominence,
            )
            peak_cache["ch"][seg_name] = (peak_vals, peak_times, ch_names)

        if "roi" in spatial_modes and roi_map:
            peak_vals, peak_times, roi_names = _process_roi_features(
                output,
                seg_data,
                seg_times,
                roi_map,
                seg_name,
                peak_mode,
                smooth_samples,
                peak_prominence,
            )
            if roi_names:
                peak_cache["roi"][seg_name] = (peak_vals, peak_times, roi_names)

        if "global" in spatial_modes:
            peak_vals, peak_times = _process_global_features(
                output,
                seg_data,
                seg_times,
                seg_name,
                peak_mode,
                smooth_samples,
                peak_prominence,
            )
            peak_cache["global"][seg_name] = (peak_vals, peak_times, ["global"])

    for scope, segments in peak_cache.items():
        if not segments:
            continue
        
        matching_pairs = _find_matching_peak_pairs(segments)
        
        for neg_segment, pos_segment, suffix in matching_pairs:
            neg_vals, neg_times, neg_names = segments[neg_segment]
            pos_vals, pos_times, pos_names = segments[pos_segment]
            
            if scope != "global" and neg_names != pos_names:
                continue

            pair_label = f"{neg_segment}{pos_segment}".replace("_", "")
            ptp_vals = pos_vals - neg_vals
            latency_diff = pos_times - neg_times
            channel_names = pos_names if scope != "global" else ["global"]
            
            _append_peak_pair_features(
                output,
                scope=scope,
                pair_label=pair_label,
                names=channel_names,
                ptp_vals=ptp_vals,
                lat_diff=latency_diff,
            )

    if not output:
        return pd.DataFrame(), []

    return pd.DataFrame(output), list(output.keys())


__all__ = ["extract_erp_features"]
