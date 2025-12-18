from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.analysis.arrays import nanmean_with_fraction
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.utils.config.loader import get_feature_constant, get_frequency_bands

from .extras import validate_window_masks


def extract_power_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("Power (precomputed): %s; skipping extraction.", err_msg)
        return pd.DataFrame(), [], {}

    n_epochs = precomputed.data.shape[0]
    min_epochs = get_feature_constant(precomputed.config, "MIN_EPOCHS_FOR_FEATURES", 10)
    if n_epochs < min_epochs:
        if precomputed.logger:
            precomputed.logger.warning(
                "Power extraction skipped: only %d epochs available (min=%d). "
                "Insufficient trials for stable power estimation.",
                n_epochs,
                min_epochs,
            )
        return pd.DataFrame(), [], {"skipped_reason": "insufficient_epochs", "n_epochs": n_epochs}

    if not validate_window_masks(precomputed, precomputed.logger):
        return pd.DataFrame(), [], {}

    epsilon = float(get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12))
    min_valid_fraction = float(get_feature_constant(precomputed.config, "MIN_VALID_FRACTION", 0.5))
    windows = precomputed.windows

    records: List[Dict[str, float]] = []
    times = precomputed.times
    active_times = times[windows.active_mask]

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}

        for band in bands:
            if band not in precomputed.band_data:
                continue

            power = precomputed.band_data[band].power[ep_idx]
            baseline_valid_count = 0
            total_channels = len(precomputed.ch_names)
            all_power_full: List[float] = []

            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                    power[ch_idx], windows.baseline_mask
                )
                active_power, _, _, _ = nanmean_with_fraction(power[ch_idx], windows.active_mask)
                baseline_valid = (
                    baseline_power > epsilon
                    and baseline_frac >= min_valid_fraction
                    and baseline_total > 0
                )
                if baseline_valid:
                    baseline_valid_count += 1

                if baseline_valid:
                    logratio = np.log10(active_power / baseline_power)
                else:
                    logratio = np.nan
                record[
                    NamingSchema.build("spectral", "plateau", band, "ch", "logratio", channel=ch_name)
                ] = float(logratio)
                all_power_full.append(float(logratio) if np.isfinite(logratio) else np.nan)

                coarse_values: Dict[str, float] = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if not np.any(win_mask):
                        continue
                    win_power, win_frac, _, _ = nanmean_with_fraction(power[ch_idx], win_mask)
                    if baseline_valid and win_frac >= min_valid_fraction:
                        win_logratio = np.log10(win_power / baseline_power)
                    else:
                        win_logratio = np.nan
                    record[
                        NamingSchema.build("spectral", win_label, band, "ch", "logratio", channel=ch_name)
                    ] = float(win_logratio)
                    coarse_values[win_label] = float(win_logratio)

                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if not np.any(win_mask):
                        continue
                    win_power, win_frac, _, _ = nanmean_with_fraction(power[ch_idx], win_mask)
                    if baseline_valid and win_frac >= min_valid_fraction:
                        win_logratio = np.log10(win_power / baseline_power)
                    else:
                        win_logratio = np.nan
                    record[
                        NamingSchema.build("spectral", win_label, band, "ch", "logratio", channel=ch_name)
                    ] = float(win_logratio)

                if len(active_times) > 2:
                    active_power_trace = power[ch_idx, windows.active_mask]
                    if baseline_valid:
                        logratio_trace = np.log10(active_power_trace / baseline_power)
                        valid_mask = np.isfinite(logratio_trace)
                        if np.sum(valid_mask) > 2:
                            slope, _ = np.polyfit(
                                active_times[valid_mask], logratio_trace[valid_mask], 1
                            )
                            record[
                                NamingSchema.build("spectral", "plateau", band, "ch", "slope", channel=ch_name)
                            ] = float(slope)
                        else:
                            record[
                                NamingSchema.build("spectral", "plateau", band, "ch", "slope", channel=ch_name)
                            ] = np.nan
                    else:
                        record[
                            NamingSchema.build("spectral", "plateau", band, "ch", "slope", channel=ch_name)
                        ] = np.nan

                    if "early" in coarse_values and "late" in coarse_values:
                        diff = coarse_values["late"] - coarse_values["early"]
                        record[
                            NamingSchema.build(
                                "power",
                                "plateau",
                                band,
                                "ch",
                                "early_late_diff",
                                channel=ch_name,
                            )
                        ] = float(diff)

            valid_power = [p for p in all_power_full if np.isfinite(p)]
            baseline_valid_fraction = (
                baseline_valid_count / total_channels if total_channels > 0 else 0.0
            )
            record[
                NamingSchema.build("spectral", "baseline", band, "global", "valid_fraction")
            ] = float(baseline_valid_fraction)

            if baseline_valid_fraction < min_valid_fraction:
                record[
                    NamingSchema.build("spectral", "plateau", band, "global", "logratio_mean")
                ] = np.nan
                record[
                    NamingSchema.build("spectral", "plateau", band, "global", "logratio_std")
                ] = np.nan
                for win_label in windows.coarse_labels:
                    record[
                        NamingSchema.build("spectral", win_label, band, "global", "logratio_mean")
                    ] = np.nan
            elif valid_power:
                record[
                    NamingSchema.build("spectral", "plateau", band, "global", "logratio_mean")
                ] = float(np.mean(valid_power))
                record[
                    NamingSchema.build("spectral", "plateau", band, "global", "logratio_std")
                ] = float(np.std(valid_power))
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if not np.any(win_mask):
                        continue
                    win_powers: List[float] = []
                    for ch_idx in range(len(precomputed.ch_names)):
                        bp, bp_frac, _, bp_total = nanmean_with_fraction(
                            power[ch_idx], windows.baseline_mask
                        )
                        if bp > epsilon and bp_frac >= min_valid_fraction and bp_total > 0:
                            wp, wp_frac, _, _ = nanmean_with_fraction(power[ch_idx], win_mask)
                            if wp_frac >= min_valid_fraction:
                                win_powers.append(np.log10(wp / bp))
                    if win_powers:
                        record[
                            NamingSchema.build("spectral", win_label, band, "global", "logratio_mean")
                        ] = float(np.mean(win_powers))

        records.append(record)

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, {}


def _compute_spectral_peak(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> Tuple[float, float]:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, np.nan
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    idx = np.argmax(psd_band)
    return float(freqs_band[idx]), float(psd_band[idx])


def _process_spectral_extras_epoch(
    ep_idx: int,
    psd: np.ndarray,
    freqs: np.ndarray,
    ch_names: List[str],
    bands: List[str],
    freq_bands: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    record: Dict[str, float] = {}
    for ch_idx, ch_name in enumerate(ch_names):
        psd_ep = psd[ep_idx, ch_idx, :]

        for band in bands:
            if band not in freq_bands:
                continue
            fmin, fmax = freq_bands[band]
            pf, pp = _compute_spectral_peak(psd_ep, freqs, fmin, fmax)
            record[
                NamingSchema.build("spectral", "baseline", band, "ch", "peakfreq", channel=ch_name)
            ] = pf
            record[
                NamingSchema.build("spectral", "baseline", band, "ch", "peakpow", channel=ch_name)
            ] = pp

        mask_fit = (freqs >= 1) & (freqs <= 40)
        if np.sum(mask_fit) > 2:
            log_f = np.log10(freqs[mask_fit])
            log_p = np.log10(psd_ep[mask_fit])
            slope, intercept = np.polyfit(log_f, log_p, 1)
            record[
                NamingSchema.build(
                    "spectral",
                    "baseline",
                    "broadband",
                    "ch",
                    "aperiodic_slope",
                    channel=ch_name,
                )
            ] = float(slope)
            record[
                NamingSchema.build(
                    "spectral",
                    "baseline",
                    "broadband",
                    "ch",
                    "aperiodic_offset",
                    channel=ch_name,
                )
            ] = float(intercept)
    return record


def extract_spectral_extras_from_precomputed(
    precomputed: Any,
    bands: List[str],
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    logger = getattr(precomputed, "logger", None)
    if precomputed.psd_data is None:
        if logger is not None:
            logger.warning("Spectral extras: No PSD data available; skipping extraction.")
        return pd.DataFrame(), []

    psd = precomputed.psd_data.psd
    freqs = precomputed.psd_data.freqs
    n_epochs = psd.shape[0]
    freq_bands = get_frequency_bands(precomputed.config)

    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_spectral_extras_epoch)(
                ep_idx, psd, freqs, precomputed.ch_names, bands, freq_bands
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_spectral_extras_epoch(
                ep_idx, psd, freqs, precomputed.ch_names, bands, freq_bands
            )
            for ep_idx in range(n_epochs)
        ]

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


def _process_segment_power_epoch(
    ep_idx: int,
    band_data: Dict[str, Any],
    bands: List[str],
    ch_names: List[str],
    baseline_mask: np.ndarray,
    segments_active: Dict[str, np.ndarray],
    epsilon: float,
) -> Dict[str, float]:
    record: Dict[str, float] = {}
    for band in bands:
        if band not in band_data:
            continue
        power = band_data[band].power[ep_idx]
        baseline_power_per_ch = np.nanmean(power[:, baseline_mask], axis=1)

        for ch_idx, ch_name in enumerate(ch_names):
            if baseline_power_per_ch[ch_idx] > epsilon:
                baseline_log = np.log10(baseline_power_per_ch[ch_idx])
            else:
                baseline_log = np.nan
            col = NamingSchema.build("spectral", "baseline", band, "ch", "logpower", channel=ch_name)
            record[col] = float(baseline_log)

        valid_baseline = baseline_power_per_ch[baseline_power_per_ch > epsilon]
        if len(valid_baseline) > 0:
            col_glob = NamingSchema.build("spectral", "baseline", band, "global", "logpower")
            record[col_glob] = float(np.nanmean(np.log10(valid_baseline)))

        for seg_name, seg_mask in segments_active.items():
            seg_power = power[:, seg_mask] if not isinstance(seg_mask, slice) else power
            seg_mean = np.nanmean(seg_power, axis=1)

            for ch_idx, ch_name in enumerate(ch_names):
                if baseline_power_per_ch[ch_idx] > epsilon:
                    logratio = np.log10(seg_mean[ch_idx] / baseline_power_per_ch[ch_idx])
                else:
                    logratio = np.nan
                col = NamingSchema.build("spectral", seg_name, band, "ch", "logratio", channel=ch_name)
                record[col] = float(logratio)

            valid = baseline_power_per_ch > epsilon
            if np.any(valid):
                col_glob = NamingSchema.build("spectral", seg_name, band, "global", "logratio")
                record[col_glob] = float(
                    np.nanmean(np.log10(seg_mean[valid] / baseline_power_per_ch[valid]))
                )
    return record


def extract_segment_power_from_precomputed(
    precomputed: Any,
    bands: List[str],
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    baseline_mask = masks.get("baseline")
    if baseline_mask is None or not np.any(baseline_mask):
        return pd.DataFrame(), []

    epsilon = float(get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12))
    segments_active = {
        k: v
        for k, v in masks.items()
        if k in ["ramp", "offset"] and v is not None and np.any(v)
    }
    n_epochs = precomputed.data.shape[0]

    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_segment_power_epoch)(
                ep_idx,
                precomputed.band_data,
                bands,
                precomputed.ch_names,
                baseline_mask,
                segments_active,
                epsilon,
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_segment_power_epoch(
                ep_idx,
                precomputed.band_data,
                bands,
                precomputed.ch_names,
                baseline_mask,
                segments_active,
                epsilon,
            )
            for ep_idx in range(n_epochs)
        ]

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = [
    "extract_power_from_precomputed",
    "extract_spectral_extras_from_precomputed",
    "extract_segment_power_from_precomputed",
]
