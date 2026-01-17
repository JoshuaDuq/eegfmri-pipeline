from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.arrays import nanmean_with_fraction
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
from eeg_pipeline.utils.analysis.channels import build_roi_map
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.utils.config.loader import get_feature_constant

from .extras import validate_window_masks


def extract_erds_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
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
    clamped_baselines = 0
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
            if precomputed.logger:
                if allow_full_epoch_fallback:
                    precomputed.logger.warning(
                        "ERDS: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                        target_name,
                    )
                else:
                    precomputed.logger.error(
                        "ERDS: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                segment_masks = {target_name: np.ones(len(precomputed.times), dtype=bool)}
            else:
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

                    # Use segment-specific active_mask, not windows.active_mask
                    active_power_trace = power[ch_idx, active_mask]
                    active_power_mean, _, _, _ = nanmean_with_fraction(power[ch_idx], active_mask)
                    safe_active_trace = np.maximum(active_power_trace, min_active_power)
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
                    else:
                        erds_full = np.nan
                        erds_trace = np.full_like(active_power_trace, np.nan)
                        erds_full_db = np.nan

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
                    elif valid_erds:
                        record[
                            NamingSchema.build("erds", segment_label, band, "global", "percent_mean")
                        ] = float(np.mean(valid_erds))
                        record[
                            NamingSchema.build("erds", segment_label, band, "global", "percent_std")
                        ] = float(np.std(valid_erds))

                    if use_log_ratio:
                        if baseline_valid_fraction < min_valid_fraction or not valid_log:
                            record[NamingSchema.build("erds", segment_label, band, "global", "db_mean")] = np.nan
                            record[NamingSchema.build("erds", segment_label, band, "global", "db_std")] = np.nan
                        else:
                            record[NamingSchema.build("erds", segment_label, band, "global", "db_mean")] = float(np.mean(valid_log))
                            record[NamingSchema.build("erds", segment_label, band, "global", "db_std")] = float(np.std(valid_log))

                if "roi" in spatial_modes and roi_map:
                    for roi_name, roi_indices in roi_map.items():
                        roi_erds = [all_erds_full[idx] for idx in roi_indices if np.isfinite(all_erds_full[idx])]
                        if roi_erds:
                            record[NamingSchema.build("erds", segment_label, band, "roi", "percent_mean", channel=roi_name)] = float(np.mean(roi_erds))
                        else:
                            record[NamingSchema.build("erds", segment_label, band, "roi", "percent_mean", channel=roi_name)] = np.nan
                        
                        if use_log_ratio:
                            roi_log = [all_log_full[idx] for idx in roi_indices if np.isfinite(all_log_full[idx])]
                            if roi_log:
                                record[NamingSchema.build("erds", segment_label, band, "roi", "db_mean", channel=roi_name)] = float(np.mean(roi_log))
                            else:
                                record[NamingSchema.build("erds", segment_label, band, "roi", "db_mean", channel=roi_name)] = np.nan

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
