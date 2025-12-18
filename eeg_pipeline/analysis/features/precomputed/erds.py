from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.arrays import nanmean_with_fraction
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

    records: List[Dict[str, float]] = []
    qc_payload: Dict[str, Dict[str, Any]] = {}
    times = precomputed.times
    active_times = times[windows.active_mask]

    segment_label = "plateau"
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}

        for band in bands:
            if band not in precomputed.band_data:
                continue

            power = precomputed.band_data[band].power[ep_idx]
            all_erds_full: List[float] = []
            all_log_full: List[float] = []
            clamped_channels_for_band = 0
            baseline_valid_count = 0
            n_channels = len(precomputed.ch_names)

            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                    power[ch_idx], windows.baseline_mask
                )
                baseline_std = float(np.nanstd(power[ch_idx, windows.baseline_mask]))
                baseline_ref = (
                    baseline_power if baseline_power >= min_baseline_power else min_baseline_power
                )
                if baseline_power < min_baseline_power:
                    clamped_baselines += 1
                    clamped_channels_for_band += 1

                baseline_valid = (
                    baseline_ref > epsilon
                    and baseline_frac >= min_valid_fraction
                    and baseline_total > 0
                )
                if baseline_valid:
                    baseline_valid_count += 1

                active_power_trace = power[ch_idx, windows.active_mask]
                active_power_mean, _, _, _ = nanmean_with_fraction(power[ch_idx], windows.active_mask)
                safe_active_trace = np.maximum(active_power_trace, min_active_power)
                safe_active_mean = (
                    float(active_power_mean) if np.isfinite(active_power_mean) else min_active_power
                )

                if baseline_valid:
                    erds_full = ((active_power_mean - baseline_ref) / baseline_ref) * 100
                    erds_trace = ((active_power_trace - baseline_ref) / baseline_ref) * 100
                    erds_full_db = 10 * np.log10(safe_active_mean / baseline_ref)
                    erds_trace_db = 10 * np.log10(safe_active_trace / baseline_ref)
                else:
                    erds_full = np.nan
                    erds_trace = np.full_like(active_power_trace, np.nan)
                    erds_full_db = np.nan
                    erds_trace_db = np.full_like(active_power_trace, np.nan)

                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "percent", channel=ch_name)
                ] = float(erds_full)
                if use_log_ratio:
                    record[
                        NamingSchema.build("erds", segment_label, band, "ch", "db", channel=ch_name)
                    ] = float(erds_full_db)
                all_erds_full.append(float(erds_full) if np.isfinite(erds_full) else np.nan)
                all_log_full.append(float(erds_full_db) if np.isfinite(erds_full_db) else np.nan)

                coarse_values: Dict[str, float] = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if not np.any(win_mask):
                        continue
                    win_power = np.mean(power[ch_idx, win_mask])
                    if baseline_valid:
                        erds_win = ((win_power - baseline_ref) / baseline_ref) * 100
                        erds_win_db = 10 * np.log10(max(win_power, min_active_power) / baseline_ref)
                    else:
                        erds_win = np.nan
                        erds_win_db = np.nan
                    record[
                        NamingSchema.build("erds", win_label, band, "ch", "percent", channel=ch_name)
                    ] = float(erds_win)
                    if use_log_ratio:
                        record[
                            NamingSchema.build("erds", win_label, band, "ch", "db", channel=ch_name)
                        ] = float(erds_win_db)
                    coarse_values[win_label] = float(erds_win)

                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if not np.any(win_mask):
                        continue
                    win_power = np.mean(power[ch_idx, win_mask])
                    if baseline_valid:
                        erds_win = ((win_power - baseline_ref) / baseline_ref) * 100
                        erds_win_db = 10 * np.log10(max(win_power, min_active_power) / baseline_ref)
                    else:
                        erds_win = np.nan
                        erds_win_db = np.nan
                    record[
                        NamingSchema.build("erds", win_label, band, "ch", "percent", channel=ch_name)
                    ] = float(erds_win)
                    if use_log_ratio:
                        record[
                            NamingSchema.build("erds", win_label, band, "ch", "db", channel=ch_name)
                        ] = float(erds_win_db)

                record[NamingSchema.build("erds", segment_label, band, "ch", "slope", channel=ch_name)] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "peak_latency", channel=ch_name)
                ] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "onset_latency", channel=ch_name)
                ] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "early_late_diff", channel=ch_name)
                ] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "erd_magnitude", channel=ch_name)
                ] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "erd_duration", channel=ch_name)
                ] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "ers_magnitude", channel=ch_name)
                ] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "ch", "ers_duration", channel=ch_name)
                ] = np.nan

                if np.any(np.isfinite(erds_trace)) and len(active_times) > 1:
                    valid_mask = np.isfinite(erds_trace)
                    if np.sum(valid_mask) > 2:
                        slope, _ = np.polyfit(active_times[valid_mask], erds_trace[valid_mask], 1)
                        record[
                            NamingSchema.build("erds", segment_label, band, "ch", "slope", channel=ch_name)
                        ] = float(slope)

                    if "early" in coarse_values and "late" in coarse_values:
                        diff = coarse_values["late"] - coarse_values["early"]
                        record[
                            NamingSchema.build(
                                "erds",
                                segment_label,
                                band,
                                "ch",
                                "early_late_diff",
                                channel=ch_name,
                            )
                        ] = float(diff)

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
            valid_log = [e for e in all_log_full if np.isfinite(e)]
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

            if baseline_valid_fraction < min_valid_fraction:
                record[
                    NamingSchema.build("erds", segment_label, band, "global", "percent_mean")
                ] = np.nan
                record[
                    NamingSchema.build("erds", segment_label, band, "global", "percent_std")
                ] = np.nan
                for win_label in windows.coarse_labels:
                    record[
                        NamingSchema.build("erds", win_label, band, "global", "percent_mean")
                    ] = np.nan
                    if use_log_ratio:
                        record[
                            NamingSchema.build("erds", win_label, band, "global", "db_mean")
                        ] = np.nan
                if use_log_ratio:
                    record[
                        NamingSchema.build("erds", segment_label, band, "global", "db_mean")
                    ] = np.nan
                    record[
                        NamingSchema.build("erds", segment_label, band, "global", "db_std")
                    ] = np.nan
            elif valid_erds:
                record[
                    NamingSchema.build("erds", segment_label, band, "global", "percent_mean")
                ] = float(np.mean(valid_erds))
                record[
                    NamingSchema.build("erds", segment_label, band, "global", "percent_std")
                ] = float(np.std(valid_erds))

                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if not np.any(win_mask):
                        continue
                    win_erds: List[float] = []
                    win_log: List[float] = []
                    for ch_idx in range(len(precomputed.ch_names)):
                        bp = np.mean(power[ch_idx, windows.baseline_mask])
                        bp_ref = bp if bp >= min_baseline_power else min_baseline_power
                        if bp_ref > epsilon:
                            wp = np.mean(power[ch_idx, win_mask])
                            win_erds.append(((wp - bp_ref) / bp_ref) * 100)
                            if use_log_ratio:
                                win_log.append(10 * np.log10(max(wp, min_active_power) / bp_ref))
                    if win_erds:
                        record[
                            NamingSchema.build("erds", win_label, band, "global", "percent_mean")
                        ] = float(np.mean(win_erds))
                    if use_log_ratio and win_log:
                        record[
                            NamingSchema.build("erds", win_label, band, "global", "db_mean")
                        ] = float(np.mean(win_log))

                if use_log_ratio and valid_log:
                    record[
                        NamingSchema.build("erds", segment_label, band, "global", "db_mean")
                    ] = float(np.mean(valid_log))
                    record[
                        NamingSchema.build("erds", segment_label, band, "global", "db_std")
                    ] = float(np.std(valid_log))

        records.append(record)

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
