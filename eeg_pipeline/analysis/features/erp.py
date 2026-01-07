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


def _infer_peak_mode(window_name: str) -> str:
    name = str(window_name).strip().lower()
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


def _compute_peaks(
    data: np.ndarray,
    times: np.ndarray,
    mode: str,
    *,
    smooth_samples: int = 0,
    prominence: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    has_finite = np.isfinite(data).any(axis=2)
    n_epochs, n_series, n_times = data.shape
    peak_vals = np.full((n_epochs, n_series), np.nan)
    peak_times = np.full((n_epochs, n_series), np.nan)

    use_smooth = int(smooth_samples) if smooth_samples is not None else 0
    if use_smooth and use_smooth >= 5 and use_smooth < n_times:
        if use_smooth % 2 == 0:
            use_smooth += 1
        try:
            data_s = savgol_filter(data, window_length=use_smooth, polyorder=2, axis=2, mode="interp")
        except Exception:
            data_s = data
    else:
        data_s = data

    for e in range(n_epochs):
        for s in range(n_series):
            y = data_s[e, s]
            if not np.isfinite(y).any():
                continue
            y0 = np.nan_to_num(y, nan=np.nanmedian(y))
            if mode == "neg":
                y_search = -y0
            elif mode == "pos":
                y_search = y0
            else:
                y_search = np.abs(y0)

            if prominence is not None and np.isfinite(prominence) and prominence > 0:
                peaks, props = find_peaks(y_search, prominence=float(prominence))
                if peaks.size:
                    best = int(peaks[np.argmax(props.get("prominences", np.ones_like(peaks)))])
                    peak_vals[e, s] = float(y0[best])
                    peak_times[e, s] = float(times[best])
                    continue

            # Fallback: argmax on requested mode
            best = int(np.nanargmax(y_search))
            peak_vals[e, s] = float(y0[best])
            peak_times[e, s] = float(times[best])

    peak_vals[~has_finite] = np.nan
    peak_times[~has_finite] = np.nan
    return peak_vals, peak_times


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
        smooth_samples=int(smooth_samples),
        prominence=prominence,
    )
    has_finite = np.isfinite(data).any(axis=2)
    auc_vals = np.trapz(np.nan_to_num(data, nan=0.0), times, axis=2)
    auc_vals[~has_finite] = np.nan

    peak_stat = f"peak_{peak_mode}"
    lat_stat = f"latency_{peak_mode}"

    for idx, name in enumerate(names):
        if scope == "global":
            mean_col = NamingSchema.build("erp", segment, "broadband", "global", "mean")
            peak_col = NamingSchema.build("erp", segment, "broadband", "global", peak_stat)
            lat_col = NamingSchema.build("erp", segment, "broadband", "global", lat_stat)
            auc_col = NamingSchema.build("erp", segment, "broadband", "global", "auc")
        else:
            mean_col = NamingSchema.build("erp", segment, "broadband", scope, "mean", channel=name)
            peak_col = NamingSchema.build("erp", segment, "broadband", scope, peak_stat, channel=name)
            lat_col = NamingSchema.build("erp", segment, "broadband", scope, lat_stat, channel=name)
            auc_col = NamingSchema.build("erp", segment, "broadband", scope, "auc", channel=name)

        output[mean_col] = mean_vals[:, idx]
        output[peak_col] = peak_vals[:, idx]
        output[lat_col] = peak_times[:, idx]
        output[auc_col] = auc_vals[:, idx]
    return peak_vals, peak_times


def _append_peak_pair_features(
    output: Dict[str, np.ndarray],
    *,
    scope: str,
    pair_label: str,
    names: List[str],
    ptp_vals: np.ndarray,
    lat_diff: np.ndarray,
) -> None:
    for idx, name in enumerate(names):
        if scope == "global":
            ptp_col = NamingSchema.build("erp", pair_label, "broadband", "global", "ptp")
            lat_col = NamingSchema.build("erp", pair_label, "broadband", "global", "latency_diff")
        else:
            ptp_col = NamingSchema.build("erp", pair_label, "broadband", scope, "ptp", channel=name)
            lat_col = NamingSchema.build("erp", pair_label, "broadband", scope, "latency_diff", channel=name)
        output[ptp_col] = ptp_vals[:, idx]
        output[lat_col] = lat_diff[:, idx]


def _build_component_masks(
    times: np.ndarray,
    erp_cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    components = erp_cfg.get("components", [])
    masks: Dict[str, np.ndarray] = {}
    if not isinstance(components, list):
        return masks
    for comp in components:
        if not isinstance(comp, dict):
            continue
        name = str(comp.get("name", "")).strip().lower()
        start = comp.get("start")
        end = comp.get("end")
        if not name or start is None or end is None:
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(start_f) or not np.isfinite(end_f) or end_f <= start_f:
            continue
        mask = (times >= start_f) & (times < end_f)
        if np.any(mask):
            masks[name] = mask
    return masks


def extract_erp_features(
    ctx: Any,  # FeatureContext
) -> Tuple[pd.DataFrame, List[str]]:
    valid, err = validate_extractor_inputs(ctx, "ERP", min_epochs=2)
    if not valid:
        ctx.logger.warning(err)
        return pd.DataFrame(), []

    epochs = ctx.epochs
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        ctx.logger.warning("ERP: No EEG channels available; skipping extraction.")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    times = epochs.times
    spatial_modes = getattr(ctx, "spatial_modes", ["roi", "global"])

    erp_cfg = ctx.config.get("feature_engineering.erp", {}) if hasattr(ctx.config, "get") else {}
    baseline_correction = bool(erp_cfg.get("baseline_correction", True))
    allow_no_baseline = bool(erp_cfg.get("allow_no_baseline", False))
    smooth_ms = erp_cfg.get("smooth_ms", 0.0)
    try:
        smooth_ms = float(smooth_ms)
    except Exception:
        smooth_ms = 0.0
    smooth_samples = int(round(float(epochs.info["sfreq"]) * smooth_ms / 1000.0)) if smooth_ms > 0 else 0
    peak_prom_uv = erp_cfg.get("peak_prominence_uv", None)
    peak_prominence = None
    if peak_prom_uv is not None:
        try:
            peak_prominence = float(peak_prom_uv) * 1e-6
        except Exception:
            peak_prominence = None

    if baseline_correction:
        baseline_mask = ctx.windows.get_mask("baseline") if ctx.windows is not None else None
        if baseline_mask is None or not np.any(baseline_mask):
            if allow_no_baseline:
                ctx.logger.info("ERP: baseline window missing; proceeding without baseline correction.")
            else:
                raise ValueError(
                    "ERP baseline correction requested but baseline window is missing or empty."
                )
        else:
            baseline = np.nanmean(data[:, :, baseline_mask], axis=2, keepdims=True)
            data = data - baseline

    segment_masks = get_segment_masks(times, ctx.windows, ctx.config)
    component_masks = _build_component_masks(times, erp_cfg)
    if component_masks:
        for name, mask in component_masks.items():
            if name not in segment_masks:
                segment_masks[name] = mask
    if not segment_masks:
        return pd.DataFrame(), []

    output: Dict[str, np.ndarray] = {}
    min_samples = int(erp_cfg.get("min_samples", MIN_SAMPLES_DEFAULT))
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
        if mask is None or np.sum(mask) < min_samples:
            continue
        seg_times = times[mask]
        seg_data = data[:, :, mask]
        peak_mode = _infer_peak_mode(seg_name)

        if "channels" in spatial_modes:
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
            peak_cache["ch"][seg_name] = (peak_vals, peak_times, ch_names)

        if "roi" in spatial_modes and roi_map:
            roi_names = []
            roi_series = []
            for roi_name, ch_indices in roi_map.items():
                if ch_indices:
                    roi_names.append(roi_name)
                    roi_series.append(np.nanmean(seg_data[:, ch_indices, :], axis=1))
            if roi_series:
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
                peak_cache["roi"][seg_name] = (peak_vals, peak_times, roi_names)

        if "global" in spatial_modes:
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
            peak_cache["global"][seg_name] = (peak_vals, peak_times, ["global"])

    for scope, segments in peak_cache.items():
        if not segments:
            continue
        neg_by_suffix: Dict[str, str] = {}
        pos_by_suffix: Dict[str, str] = {}
        for seg_name in segments.keys():
            parsed = _parse_peak_label(seg_name)
            if not parsed:
                continue
            polarity, suffix = parsed
            if polarity == "n" and suffix not in neg_by_suffix:
                neg_by_suffix[suffix] = seg_name
            elif polarity == "p" and suffix not in pos_by_suffix:
                pos_by_suffix[suffix] = seg_name

        for suffix in sorted(set(neg_by_suffix) & set(pos_by_suffix)):
            neg_seg = neg_by_suffix[suffix]
            pos_seg = pos_by_suffix[suffix]
            neg_vals, neg_times, neg_names = segments[neg_seg]
            pos_vals, pos_times, pos_names = segments[pos_seg]
            if scope != "global" and neg_names != pos_names:
                continue

            pair_label = f"{neg_seg}{pos_seg}".replace("_", "")
            ptp_vals = pos_vals - neg_vals
            lat_diff = pos_times - neg_times
            names = ["global"] if scope == "global" else pos_names
            _append_peak_pair_features(
                output,
                scope=scope,
                pair_label=pair_label,
                names=names,
                ptp_vals=ptp_vals,
                lat_diff=lat_diff,
            )

    if not output:
        return pd.DataFrame(), []

    df = pd.DataFrame(output)
    return df, list(df.columns)


__all__ = ["extract_erp_features"]
