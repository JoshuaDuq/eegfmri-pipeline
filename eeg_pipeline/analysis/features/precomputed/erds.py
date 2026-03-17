from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.arrays import nanmean_with_fraction
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
from eeg_pipeline.utils.analysis.channels import build_roi_map
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.analysis.features.rest import raise_if_rest_incompatible
from eeg_pipeline.utils.config.loader import get_feature_constant

from .extras import validate_window_masks


_DEFAULT_LATERALITY_MARKER = "Somatosensory_Contralateral"
_DEFAULT_LEFT_SOMATOSENSORY_CHANNELS = (
    "C1",
    "C3",
    "C5",
    "CP1",
    "CP3",
    "CP5",
    "FC1",
    "FC3",
    "FC5",
)
_DEFAULT_RIGHT_SOMATOSENSORY_CHANNELS = (
    "C2",
    "C4",
    "C6",
    "CP2",
    "CP4",
    "CP6",
    "FC2",
    "FC4",
    "FC6",
)
_DEFAULT_LATERALITY_COLUMNS = (
    "stim_side",
    "stimulated_side",
    "side",
    "stimulus_side",
    "hand",
)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _coerce_str_list(value: Any, fallback: Sequence[str]) -> List[str]:
    if value is None:
        return [str(v).strip() for v in fallback if str(v).strip()]
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for token in value:
            text = str(token).strip()
            if text:
                out.append(text)
        return out
    return [str(v).strip() for v in fallback if str(v).strip()]


def _extract_laterality(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None

    token = str(value).strip().lower()
    if not token:
        return None

    if token in {"l", "left", "lh", "left_hand", "left-arm", "left_arm"}:
        return "left"
    if token in {"r", "right", "rh", "right_hand", "right-arm", "right_arm"}:
        return "right"

    has_left = "left" in token or "_l" in token or token.endswith("-l")
    has_right = "right" in token or "_r" in token or token.endswith("-r")
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    return None


def _trial_stim_side(
    metadata: Optional[pd.DataFrame],
    epoch_idx: int,
    laterality_columns: Sequence[str],
) -> Optional[str]:
    if metadata is None or metadata.empty:
        return None
    if epoch_idx < 0 or epoch_idx >= len(metadata):
        return None

    for column in laterality_columns:
        if column not in metadata.columns:
            continue
        side = _extract_laterality(metadata.iloc[epoch_idx].get(column))
        if side is not None:
            return side
    return None


def _first_sustained_crossing(mask: np.ndarray, min_samples: int) -> Optional[int]:
    run = 0
    required = max(1, int(min_samples))
    for idx, flag in enumerate(mask):
        if bool(flag):
            run += 1
            if run >= required:
                return int(idx - required + 1)
        else:
            run = 0
    return None


def _aggregate_trace(
    traces_by_channel: Mapping[int, np.ndarray],
    indices: np.ndarray,
    n_times: int,
) -> Tuple[np.ndarray, int]:
    if indices.size == 0:
        return np.full((n_times,), np.nan, dtype=float), 0

    traces: List[np.ndarray] = []
    for idx in indices:
        trace = traces_by_channel.get(int(idx))
        if trace is None:
            continue
        if not np.any(np.isfinite(trace)):
            continue
        traces.append(trace)

    if not traces:
        return np.full((n_times,), np.nan, dtype=float), 0

    stacked = np.vstack(traces)
    with np.errstate(all="ignore"):
        return np.nanmean(stacked, axis=0), len(traces)


def _resolve_laterality_roi_indices(
    ch_names: Sequence[str],
    erds_cfg: Mapping[str, Any],
    roi_map: Mapping[str, Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    lookup = {str(ch).strip().lower(): idx for idx, ch in enumerate(ch_names)}

    def _indices_from_names(names: Sequence[str]) -> np.ndarray:
        indices: List[int] = []
        for raw_name in names:
            name = str(raw_name).strip().lower()
            if not name:
                continue
            idx = lookup.get(name)
            if idx is not None:
                indices.append(int(idx))
        if not indices:
            return np.array([], dtype=int)
        return np.asarray(sorted(set(indices)), dtype=int)

    left_names = _coerce_str_list(
        erds_cfg.get("somatosensory_left_channels"),
        _DEFAULT_LEFT_SOMATOSENSORY_CHANNELS,
    )
    right_names = _coerce_str_list(
        erds_cfg.get("somatosensory_right_channels"),
        _DEFAULT_RIGHT_SOMATOSENSORY_CHANNELS,
    )

    left_idx = _indices_from_names(left_names)
    right_idx = _indices_from_names(right_names)

    if left_idx.size == 0 or right_idx.size == 0:
        for roi_name, indices in roi_map.items():
            roi_key = str(roi_name).strip().lower()
            roi_indices = np.asarray(indices, dtype=int)
            if roi_indices.size == 0:
                continue

            if left_idx.size == 0 and (
                "sensorimotor_left" in roi_key
                or "somatosensory_left" in roi_key
                or roi_key.endswith("_left")
            ):
                left_idx = roi_indices
            if right_idx.size == 0 and (
                "sensorimotor_right" in roi_key
                or "somatosensory_right" in roi_key
                or roi_key.endswith("_right")
            ):
                right_idx = roi_indices

    return left_idx, right_idx


def _compute_laterality_metrics(
    trace: np.ndarray,
    active_times: np.ndarray,
    sfreq: float,
    *,
    baseline_noise_pct: float,
    onset_threshold_sigma: float,
    onset_min_threshold_percent: float,
    onset_min_samples: int,
    rebound_threshold_sigma: float,
    rebound_min_threshold_percent: float,
    rebound_min_latency_ms: float,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if trace.size == 0 or not np.any(np.isfinite(trace)):
        return metrics

    with np.errstate(all="ignore"):
        peak_idx = int(np.nanargmin(trace))
    peak_value = float(trace[peak_idx])
    if np.isfinite(peak_value):
        metrics["peak_latency"] = float(active_times[peak_idx])
        metrics["erd_magnitude"] = float(abs(min(peak_value, 0.0)))

    noise_pct = float(baseline_noise_pct) if np.isfinite(baseline_noise_pct) else 0.0
    onset_threshold = max(float(onset_min_threshold_percent), float(onset_threshold_sigma) * noise_pct)

    onset_mask = np.isfinite(trace) & (trace <= -onset_threshold)
    onset_idx = _first_sustained_crossing(onset_mask, onset_min_samples)
    if onset_idx is not None:
        metrics["onset_latency"] = float(active_times[onset_idx])

    latency_samples = int(round(max(0.0, rebound_min_latency_ms) * sfreq / 1000.0))
    rebound_start = min(len(trace), peak_idx + latency_samples)
    if rebound_start >= len(trace):
        return metrics

    rebound_trace = trace[rebound_start:]
    if rebound_trace.size == 0 or not np.any(np.isfinite(rebound_trace)):
        return metrics

    rebound_threshold = max(
        float(rebound_min_threshold_percent),
        float(rebound_threshold_sigma) * noise_pct,
    )
    rebound_mask = np.isfinite(rebound_trace) & (rebound_trace >= rebound_threshold)
    rebound_onset_idx = _first_sustained_crossing(rebound_mask, onset_min_samples)
    if rebound_onset_idx is None:
        return metrics

    finite_rebound = np.where(np.isfinite(rebound_trace), rebound_trace, -np.inf)
    rebound_peak_rel = int(np.argmax(finite_rebound))
    rebound_peak_val = float(finite_rebound[rebound_peak_rel])
    if not np.isfinite(rebound_peak_val) or rebound_peak_val < rebound_threshold:
        return metrics

    rebound_peak_idx = rebound_start + rebound_peak_rel
    metrics["rebound_magnitude"] = rebound_peak_val
    metrics["rebound_latency"] = float(active_times[rebound_peak_idx])
    # Keep compatibility with existing ERDS semantics while adding explicit rebound fields.
    metrics["ers_magnitude"] = rebound_peak_val
    return metrics


def extract_erds_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    raise_if_rest_incompatible(precomputed.config or {}, feature_name="ERDS")
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("ERDS: %s; skipping extraction.", err_msg)
        return pd.DataFrame(), [], {}

    n_epochs = precomputed.data.shape[0]
    try:
        min_epochs = int(get_feature_constant(precomputed.config, "MIN_EPOCHS_FOR_FEATURES", 10))
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Invalid config constant MIN_EPOCHS_FOR_FEATURES; expected an int-like value."
        ) from e
    if n_epochs < min_epochs:
        if precomputed.logger:
            precomputed.logger.warning(
                "ERDS extraction skipped: only %d epochs available (min=%d). "
                "Insufficient trials for stable ERD/ERS estimation.",
                n_epochs,
                min_epochs,
            )
        return pd.DataFrame(), [], {"skipped_reason": "insufficient_epochs", "n_epochs": n_epochs}

    if not validate_window_masks(precomputed, precomputed.logger):
        return pd.DataFrame(), [], {}

    try:
        epsilon = float(get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12))
    except (TypeError, ValueError) as e:
        raise ValueError("Invalid config constant EPSILON_STD; expected a float-like value.") from e

    try:
        min_valid_fraction = float(
            get_feature_constant(precomputed.config, "MIN_VALID_FRACTION", 0.5)
        )
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Invalid config constant MIN_VALID_FRACTION; expected a float-like value."
        ) from e

    config = precomputed.config or {}
    erds_cfg = config.get("feature_engineering.erds", {})
    min_baseline_power = float(
        erds_cfg.get(
            "min_baseline_power",
            config.get("feature_engineering.features.min_baseline_power", epsilon),
        )
    )
    min_active_power = float(erds_cfg.get("min_active_power", epsilon))
    use_log_ratio = bool(erds_cfg.get("use_log_ratio", False))

    laterality_metrics_enabled = _bool_or_default(
        erds_cfg.get("enable_laterality_markers"),
        False,
    )
    laterality_marker_identifier = str(erds_cfg.get("laterality_marker_identifier", _DEFAULT_LATERALITY_MARKER))
    laterality_marker_bands = {
        str(token).strip().lower()
        for token in _coerce_str_list(erds_cfg.get("laterality_marker_bands"), ["alpha"])
    }
    laterality_columns = _coerce_str_list(
        erds_cfg.get("laterality_columns"),
        _DEFAULT_LATERALITY_COLUMNS,
    )
    infer_contralateral = _bool_or_default(
        erds_cfg.get("infer_contralateral_when_missing"),
        True,
    )

    onset_threshold_sigma = _float_or_default(erds_cfg.get("onset_threshold_sigma"), 1.0)
    onset_min_threshold_percent = _float_or_default(erds_cfg.get("onset_min_threshold_percent"), 5.0)
    onset_min_duration_ms = _float_or_default(erds_cfg.get("onset_min_duration_ms"), 30.0)
    rebound_threshold_sigma = _float_or_default(erds_cfg.get("rebound_threshold_sigma"), 1.0)
    rebound_min_threshold_percent = _float_or_default(erds_cfg.get("rebound_min_threshold_percent"), 5.0)
    rebound_min_latency_ms = _float_or_default(erds_cfg.get("rebound_min_latency_ms"), 100.0)

    onset_min_samples = max(1, _int_or_default(round(onset_min_duration_ms * precomputed.sfreq / 1000.0), 1))

    clamped_baselines = 0
    windows = precomputed.windows

    target_name = getattr(windows, "name", None) if windows else None

    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segment_masks = {target_name: mask}
        else:
            if precomputed.logger:
                precomputed.logger.error(
                    "ERDS: targeted window '%s' has no valid mask; skipping.",
                    target_name,
                )
            return pd.DataFrame(), [], {"error": f"invalid_target_window_mask:{target_name}"}
    else:
        segment_masks = get_segment_masks(precomputed.times, windows, precomputed.config)

    # Filter out baseline - ERDS uses baseline only as reference
    active_segments = {k: v for k, v in segment_masks.items() if k != "baseline" and v is not None and np.any(v)}

    spatial_modes = getattr(precomputed, "spatial_modes", ["roi", "global"])
    roi_map = {}
    if "roi" in spatial_modes:
        roi_defs = get_roi_definitions(precomputed.config)
        if roi_defs:
            roi_map = build_roi_map(precomputed.ch_names, roi_defs)

    left_somato_idx, right_somato_idx = _resolve_laterality_roi_indices(
        precomputed.ch_names,
        erds_cfg,
        roi_map,
    )

    if not active_segments:
        if precomputed.logger:
            precomputed.logger.warning("ERDS: No non-baseline segments defined; skipping extraction.")
        return pd.DataFrame(), [], {}

    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
    qc_payload: Dict[str, Dict[str, Any]] = {}
    times = precomputed.times

    # Iterate over ALL active segments
    for segment_label, active_mask in active_segments.items():
        active_times = times[active_mask]

        for ep_idx in range(n_epochs):
            record = records[ep_idx]

            for band in bands:
                if band not in precomputed.band_data:
                    continue

                power = precomputed.band_data[band].power[ep_idx]
                all_erds_full: List[float] = []
                all_log_full: List[float] = [] if use_log_ratio else []
                clamped_channels_for_band = 0
                baseline_valid_count = 0
                n_channels = len(precomputed.ch_names)
                baseline_ref_by_channel = np.full((n_channels,), np.nan, dtype=float)
                active_mean_by_channel = np.full((n_channels,), np.nan, dtype=float)
                baseline_noise_pct_by_channel = np.full((n_channels,), np.nan, dtype=float)
                channel_erds_trace: Dict[int, np.ndarray] = {}

                for ch_idx, ch_name in enumerate(precomputed.ch_names):
                    baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                        power[ch_idx], windows.baseline_mask
                    )
                    baseline_std = float(np.nanstd(power[ch_idx, windows.baseline_mask]))

                    # Flag low-SNR channels as invalid instead of clamping
                    # Clamping would produce artificially huge ERD/ERS ratios
                    baseline_too_low = baseline_power < min_baseline_power
                    if baseline_too_low:
                        clamped_baselines += 1
                        clamped_channels_for_band += 1

                    baseline_ref = baseline_power

                    baseline_valid = (
                        not baseline_too_low
                        and baseline_ref > epsilon
                        and baseline_frac >= min_valid_fraction
                        and baseline_total > 0
                    )
                    if baseline_valid:
                        baseline_valid_count += 1
                        baseline_noise_pct_by_channel[ch_idx] = (
                            baseline_std / baseline_ref * 100.0
                            if baseline_ref > epsilon
                            else np.nan
                        )

                    # Use segment-specific active_mask, not windows.active_mask
                    active_power_trace = power[ch_idx, active_mask]
                    active_power_mean, _, _, _ = nanmean_with_fraction(power[ch_idx], active_mask)
                    np.maximum(active_power_trace, min_active_power)
                    safe_active_mean = (
                        float(active_power_mean) if np.isfinite(active_power_mean) else min_active_power
                    )

                    if baseline_valid:
                        erds_full = ((active_power_mean - baseline_ref) / baseline_ref) * 100
                        erds_trace = ((active_power_trace - baseline_ref) / baseline_ref) * 100
                        if use_log_ratio:
                            erds_full_db = 10 * np.log10(safe_active_mean / baseline_ref)
                        else:
                            erds_full_db = np.nan
                        if np.isfinite(active_power_mean):
                            baseline_ref_by_channel[ch_idx] = float(baseline_ref)
                            active_mean_by_channel[ch_idx] = float(active_power_mean)
                    else:
                        erds_full = np.nan
                        erds_trace = np.full_like(active_power_trace, np.nan)
                        erds_full_db = np.nan

                    channel_erds_trace[ch_idx] = erds_trace

                    if "channels" in spatial_modes:
                        record[
                            NamingSchema.build("erds", segment_label, band, "ch", "percent", channel=ch_name)
                        ] = float(erds_full)
                        if use_log_ratio:
                            record[
                                NamingSchema.build("erds", segment_label, band, "ch", "db", channel=ch_name)
                            ] = float(erds_full_db)

                    all_erds_full.append(float(erds_full) if np.isfinite(erds_full) else np.nan)
                    if use_log_ratio:
                        all_log_full.append(float(erds_full_db) if np.isfinite(erds_full_db) else np.nan)

                    if "channels" in spatial_modes and np.any(np.isfinite(erds_trace)) and len(active_times) > 1:
                        valid_mask_trace = np.isfinite(erds_trace)
                        if np.sum(valid_mask_trace) > 2:
                            slope, _ = np.polyfit(active_times[valid_mask_trace], erds_trace[valid_mask_trace], 1)
                            record[
                                NamingSchema.build("erds", segment_label, band, "ch", "slope", channel=ch_name)
                            ] = float(slope)

                        peak_idx = int(np.nanargmax(np.abs(erds_trace)))
                        record[
                            NamingSchema.build("erds", segment_label, band, "ch", "peak_latency", channel=ch_name)
                        ] = float(active_times[peak_idx])

                        threshold = (
                            baseline_std / baseline_ref * 100 if baseline_ref > epsilon else np.inf
                        )
                        onset_mask = np.abs(erds_trace) > threshold
                        if np.any(onset_mask):
                            onset_idx = int(np.argmax(onset_mask))
                            record[
                                NamingSchema.build(
                                    "erds",
                                    segment_label,
                                    band,
                                    "ch",
                                    "onset_latency",
                                    channel=ch_name,
                                )
                            ] = float(active_times[onset_idx])

                        erd_vals = erds_trace[erds_trace < 0]
                        ers_vals = erds_trace[erds_trace > 0]
                        if baseline_valid:
                            if len(erd_vals) > 0:
                                erd_magnitude = float(np.mean(np.abs(erd_vals)))
                                erd_duration = float(len(erd_vals) / precomputed.sfreq)
                            else:
                                erd_magnitude = 0.0
                                erd_duration = 0.0
                            record[
                                NamingSchema.build("erds", segment_label, band, "ch", "erd_magnitude", channel=ch_name)
                            ] = erd_magnitude
                            record[
                                NamingSchema.build("erds", segment_label, band, "ch", "erd_duration", channel=ch_name)
                            ] = erd_duration

                            if len(ers_vals) > 0:
                                ers_magnitude = float(np.mean(ers_vals))
                                ers_duration = float(len(ers_vals) / precomputed.sfreq)
                            else:
                                ers_magnitude = 0.0
                                ers_duration = 0.0
                            record[
                                NamingSchema.build("erds", segment_label, band, "ch", "ers_magnitude", channel=ch_name)
                            ] = ers_magnitude
                            record[
                                NamingSchema.build("erds", segment_label, band, "ch", "ers_duration", channel=ch_name)
                            ] = ers_duration

                if (
                    laterality_metrics_enabled
                    and band.lower() in laterality_marker_bands
                    and len(active_times) > 1
                    and laterality_marker_identifier
                ):
                    stim_side = _trial_stim_side(
                        getattr(precomputed, "metadata", None),
                        ep_idx,
                        laterality_columns,
                    )

                    left_trace, left_n = _aggregate_trace(channel_erds_trace, left_somato_idx, len(active_times))
                    right_trace, right_n = _aggregate_trace(channel_erds_trace, right_somato_idx, len(active_times))

                    selected_indices: np.ndarray
                    selected_trace: np.ndarray
                    if stim_side == "left" and right_n > 0:
                        selected_indices = right_somato_idx
                        selected_trace = right_trace
                    elif stim_side == "right" and left_n > 0:
                        selected_indices = left_somato_idx
                        selected_trace = left_trace
                    elif infer_contralateral and left_n > 0 and right_n > 0:
                        left_peak = float(np.nanmin(left_trace)) if np.any(np.isfinite(left_trace)) else np.nan
                        right_peak = float(np.nanmin(right_trace)) if np.any(np.isfinite(right_trace)) else np.nan
                        if np.isfinite(left_peak) and np.isfinite(right_peak):
                            if left_peak <= right_peak:
                                selected_indices = left_somato_idx
                                selected_trace = left_trace
                            else:
                                selected_indices = right_somato_idx
                                selected_trace = right_trace
                        elif np.isfinite(left_peak):
                            selected_indices = left_somato_idx
                            selected_trace = left_trace
                        elif np.isfinite(right_peak):
                            selected_indices = right_somato_idx
                            selected_trace = right_trace
                        else:
                            selected_indices = np.array([], dtype=int)
                            selected_trace = np.full((len(active_times),), np.nan, dtype=float)
                    elif left_n > 0:
                        selected_indices = left_somato_idx
                        selected_trace = left_trace
                    elif right_n > 0:
                        selected_indices = right_somato_idx
                        selected_trace = right_trace
                    else:
                        selected_indices = np.array([], dtype=int)
                        selected_trace = np.full((len(active_times),), np.nan, dtype=float)

                    if selected_indices.size > 0 and np.any(np.isfinite(selected_trace)):
                        noise_vals = baseline_noise_pct_by_channel[selected_indices]
                        baseline_noise_pct = float(np.nanmedian(noise_vals)) if np.any(np.isfinite(noise_vals)) else np.nan
                        laterality_metrics = _compute_laterality_metrics(
                            selected_trace,
                            active_times,
                            precomputed.sfreq,
                            baseline_noise_pct=baseline_noise_pct,
                            onset_threshold_sigma=onset_threshold_sigma,
                            onset_min_threshold_percent=onset_min_threshold_percent,
                            onset_min_samples=onset_min_samples,
                            rebound_threshold_sigma=rebound_threshold_sigma,
                            rebound_min_threshold_percent=rebound_min_threshold_percent,
                            rebound_min_latency_ms=rebound_min_latency_ms,
                        )
                        for stat, value in laterality_metrics.items():
                            record[
                                NamingSchema.build(
                                    "erds",
                                    segment_label,
                                    band,
                                    "roi",
                                    stat,
                                    channel=laterality_marker_identifier,
                                )
                            ] = float(value)

                valid_erds = [e for e in all_erds_full if np.isfinite(e)]
                valid_log = [e for e in all_log_full if np.isfinite(e)] if use_log_ratio else []
                baseline_valid_fraction = (
                    baseline_valid_count / n_channels if n_channels > 0 else 0.0
                )

                if band not in qc_payload:
                    qc_payload[band] = {
                        "clamped_channels": [],
                        "baseline_min_power": min_baseline_power,
                        "valid_fractions": [],
                    }
                qc_payload[band]["clamped_channels"].append(int(clamped_channels_for_band))
                qc_payload[band]["valid_fractions"].append(float(baseline_valid_fraction))

                if "global" in spatial_modes:
                    if baseline_valid_fraction < min_valid_fraction:
                        record[
                            NamingSchema.build("erds", segment_label, band, "global", "percent_mean")
                        ] = np.nan
                        record[
                            NamingSchema.build("erds", segment_label, band, "global", "percent_std")
                        ] = np.nan
                    else:
                        # Scientific validity: compute ERDS on spatially-aggregated power
                        # (mean across channels) rather than averaging per-channel ERDS (ratio).
                        valid_mask_ch = np.isfinite(baseline_ref_by_channel) & np.isfinite(active_mean_by_channel)
                        if not np.any(valid_mask_ch):
                            record[
                                NamingSchema.build("erds", segment_label, band, "global", "percent_mean")
                            ] = np.nan
                            record[
                                NamingSchema.build("erds", segment_label, band, "global", "percent_std")
                            ] = np.nan
                        else:
                            baseline_mean = float(np.nanmean(baseline_ref_by_channel[valid_mask_ch]))
                            active_mean = float(np.nanmean(active_mean_by_channel[valid_mask_ch]))
                            if baseline_mean > epsilon and np.isfinite(baseline_mean) and np.isfinite(active_mean):
                                global_percent_mean = float(((active_mean - baseline_mean) / baseline_mean) * 100)
                            else:
                                global_percent_mean = np.nan
                            record[
                                NamingSchema.build("erds", segment_label, band, "global", "percent_mean")
                            ] = global_percent_mean
                            record[
                                NamingSchema.build("erds", segment_label, band, "global", "percent_std")
                            ] = float(np.std(valid_erds)) if valid_erds else np.nan

                    if use_log_ratio:
                        if baseline_valid_fraction < min_valid_fraction:
                            record[NamingSchema.build("erds", segment_label, band, "global", "db_mean")] = np.nan
                            record[NamingSchema.build("erds", segment_label, band, "global", "db_std")] = np.nan
                        else:
                            valid_mask_ch = np.isfinite(baseline_ref_by_channel) & np.isfinite(active_mean_by_channel)
                            if not np.any(valid_mask_ch):
                                record[NamingSchema.build("erds", segment_label, band, "global", "db_mean")] = np.nan
                                record[NamingSchema.build("erds", segment_label, band, "global", "db_std")] = np.nan
                            else:
                                baseline_mean = float(np.nanmean(baseline_ref_by_channel[valid_mask_ch]))
                                active_mean = float(np.nanmean(active_mean_by_channel[valid_mask_ch]))
                                if (
                                    baseline_mean > epsilon
                                    and active_mean > 0
                                    and np.isfinite(baseline_mean)
                                    and np.isfinite(active_mean)
                                ):
                                    db_mean = float(10 * np.log10(max(active_mean, min_active_power) / baseline_mean))
                                else:
                                    db_mean = np.nan
                                record[NamingSchema.build("erds", segment_label, band, "global", "db_mean")] = db_mean
                                record[NamingSchema.build("erds", segment_label, band, "global", "db_std")] = (
                                    float(np.std(valid_log)) if valid_log else np.nan
                                )

                if "roi" in spatial_modes and roi_map:
                    for roi_name, roi_indices in roi_map.items():
                        roi_idx = np.asarray(roi_indices, dtype=int)
                        if roi_idx.size == 0:
                            record[
                                NamingSchema.build(
                                    "erds", segment_label, band, "roi", "percent_mean", channel=roi_name
                                )
                            ] = np.nan
                            if use_log_ratio:
                                record[
                                    NamingSchema.build(
                                        "erds", segment_label, band, "roi", "db_mean", channel=roi_name
                                    )
                                ] = np.nan
                            continue

                        valid_roi = np.isfinite(baseline_ref_by_channel[roi_idx]) & np.isfinite(
                            active_mean_by_channel[roi_idx]
                        )
                        if not np.any(valid_roi):
                            record[
                                NamingSchema.build(
                                    "erds", segment_label, band, "roi", "percent_mean", channel=roi_name
                                )
                            ] = np.nan
                        else:
                            b_roi = float(np.nanmean(baseline_ref_by_channel[roi_idx][valid_roi]))
                            a_roi = float(np.nanmean(active_mean_by_channel[roi_idx][valid_roi]))
                            if b_roi > epsilon and np.isfinite(b_roi) and np.isfinite(a_roi):
                                roi_percent_mean = float(((a_roi - b_roi) / b_roi) * 100)
                            else:
                                roi_percent_mean = np.nan
                            record[
                                NamingSchema.build(
                                    "erds", segment_label, band, "roi", "percent_mean", channel=roi_name
                                )
                            ] = roi_percent_mean

                        if use_log_ratio:
                            if not np.any(valid_roi):
                                record[
                                    NamingSchema.build(
                                        "erds", segment_label, band, "roi", "db_mean", channel=roi_name
                                    )
                                ] = np.nan
                            else:
                                if b_roi > epsilon and a_roi > 0 and np.isfinite(b_roi) and np.isfinite(a_roi):
                                    roi_db_mean = float(10 * np.log10(max(a_roi, min_active_power) / b_roi))
                                else:
                                    roi_db_mean = np.nan
                                record[
                                    NamingSchema.build(
                                        "erds", segment_label, band, "roi", "db_mean", channel=roi_name
                                    )
                                ] = roi_db_mean

    if clamped_baselines > 0 and precomputed.logger:
        precomputed.logger.info(
            "Clamped %d baseline power values below min_baseline_power=%.3e to stabilize ERD/ERS ratios (use_log_ratio=%s).",
            clamped_baselines,
            min_baseline_power,
            use_log_ratio,
        )

    qc_summary: Dict[str, Any] = {}
    for band, stats in qc_payload.items():
        clamp_list = stats.get("clamped_channels", [])
        valid_frac_list = stats.get("valid_fractions", [])
        qc_summary[band] = {
            "median_clamped_channels": float(np.median(clamp_list)) if clamp_list else 0.0,
            "max_clamped_channels": int(np.max(clamp_list)) if clamp_list else 0,
            "min_baseline_power": float(stats.get("baseline_min_power", min_baseline_power)),
            "median_baseline_valid_fraction": float(np.median(valid_frac_list)) if valid_frac_list else 0.0,
            "min_baseline_valid_fraction": float(np.min(valid_frac_list)) if valid_frac_list else 0.0,
            "n_epochs_low_validity": int(sum(1 for f in valid_frac_list if f < min_valid_fraction)),
        }

    for band, stats in qc_summary.items():
        if stats["n_epochs_low_validity"] > 0 and precomputed.logger:
            precomputed.logger.warning(
                "ERDS band '%s': %d/%d epochs had baseline_valid_fraction < %.1f%%; global summaries set to NaN for these epochs.",
                band,
                stats["n_epochs_low_validity"],
                n_epochs,
                min_valid_fraction * 100,
            )

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, qc_summary


__all__ = ["extract_erds_from_precomputed"]
