"""
Feature Extractors
==================

Functions to extract specific feature sets from precomputed data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.config.loader import get_feature_constant, get_frequency_bands, get_config_value
from eeg_pipeline.utils.analysis.arrays import nanmean_with_fraction
from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from joblib import Parallel, delayed


def validate_window_masks(
    precomputed: PrecomputedData,
    logger: Optional[logging.Logger] = None,
    *,
    require_baseline: bool = True,
    require_active: bool = True,
) -> bool:
    """
    Ensure baseline/active masks exist and contain samples.

    Returns False and logs a warning when masks are missing or empty so that
    feature extractors can fail fast instead of producing all-NaN outputs.
    """
    windows = precomputed.windows
    if windows is None:
        if logger:
            logger.warning("Time windows are missing; skipping feature extraction.")
        return False

    if require_baseline:
        baseline_mask = getattr(windows, "baseline_mask", None)
        if baseline_mask is None or not np.any(baseline_mask):
            if logger:
                logger.warning(
                    "Baseline window is empty; configured/used range: %s. Skipping feature extraction.",
                    getattr(windows, "baseline_range", None),
                )
            return False

    if require_active:
        active_mask = getattr(windows, "active_mask", None)
        if active_mask is None or not np.any(active_mask):
            if logger:
                logger.warning(
                    "Active window is empty; configured/used range: %s. Skipping feature extraction.",
                    getattr(windows, "active_range", None),
                )
            return False

    return True


def extract_erds_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Extract comprehensive ERD/ERS features using precomputed band power.
    
    ERD/ERS = (Active - Baseline) / Baseline * 100
    Negative = ERD (desynchronization), Positive = ERS (synchronization)
    
    Features extracted per channel/band:
    - Mean ERD/ERS over full active period
    - ERD/ERS per coarse temporal window (early, mid, late)
    - ERD/ERS per fine temporal window (t1-t7) for HRF modeling
    - Temporal dynamics: slope, onset latency, peak latency, early-late diff
    - ERD vs ERS separation (magnitude of negative vs positive values)
    - Global (cross-channel) statistics per band
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), [], {}

    # Early bail: enforce minimum epoch count for stable ERDS estimation
    n_epochs = precomputed.data.shape[0]
    min_epochs = get_feature_constant(precomputed.config, "MIN_EPOCHS_FOR_FEATURES", 10)
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

    epsilon = get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12)
    min_valid_fraction = get_feature_constant(precomputed.config, "MIN_VALID_FRACTION", 0.5)
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
    n_epochs = precomputed.data.shape[0]
    times = precomputed.times
    active_times = times[windows.active_mask]
    
    segment_label = "plateau"
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            all_erds_full = []
            all_log_full = []
            clamped_channels_for_band = 0
            baseline_valid_count = 0  # Track how many channels have valid baselines
            n_channels = len(precomputed.ch_names)
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                    power[ch_idx], windows.baseline_mask
                )
                baseline_std = float(np.nanstd(power[ch_idx, windows.baseline_mask]))
                baseline_ref = baseline_power if baseline_power >= min_baseline_power else min_baseline_power
                if baseline_power < min_baseline_power:
                    clamped_baselines += 1
                    clamped_channels_for_band += 1
                baseline_valid = baseline_ref > epsilon and baseline_frac >= min_valid_fraction and baseline_total > 0
                if baseline_valid:
                    baseline_valid_count += 1
                active_power_trace = power[ch_idx, windows.active_mask]
                active_power_mean, active_frac, _, _ = nanmean_with_fraction(power[ch_idx], windows.active_mask)
                safe_active_trace = np.maximum(active_power_trace, min_active_power)
                safe_active_mean = float(active_power_mean) if np.isfinite(active_power_mean) else min_active_power
                
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
                
                # Full period ERD/ERS
                record[NamingSchema.build("erds", segment_label, band, "ch", "percent", channel=ch_name)] = float(erds_full)
                if use_log_ratio:
                    record[NamingSchema.build("erds", segment_label, band, "ch", "db", channel=ch_name)] = float(erds_full_db)
                all_erds_full.append(erds_full)
                all_log_full.append(erds_full_db)
                
                # === Coarse temporal bins (early, mid, late) ===
                coarse_values = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_valid:
                            erds_win = ((win_power - baseline_ref) / baseline_ref) * 100
                            erds_win_db = 10 * np.log10(max(win_power, min_active_power) / baseline_ref)
                        else:
                            erds_win = np.nan
                            erds_win_db = np.nan
                        record[NamingSchema.build("erds", win_label, band, "ch", "percent", channel=ch_name)] = float(erds_win)
                        if use_log_ratio:
                            record[NamingSchema.build("erds", win_label, band, "ch", "db", channel=ch_name)] = float(erds_win_db)
                        coarse_values[win_label] = erds_win
                
                # === Fine temporal bins (t1-t7) for HRF modeling ===
                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if np.any(win_mask):
                        win_power = np.mean(power[ch_idx, win_mask])
                        if baseline_valid:
                            erds_win = ((win_power - baseline_ref) / baseline_ref) * 100
                            erds_win_db = 10 * np.log10(max(win_power, min_active_power) / baseline_ref)
                        else:
                            erds_win = np.nan
                            erds_win_db = np.nan
                        record[NamingSchema.build("erds", win_label, band, "ch", "percent", channel=ch_name)] = float(erds_win)
                        if use_log_ratio:
                            record[NamingSchema.build("erds", win_label, band, "ch", "db", channel=ch_name)] = float(erds_win_db)
                
                # === Temporal dynamics ===
                # Default temporal metrics to NaN
                record[NamingSchema.build("erds", segment_label, band, "ch", "slope", channel=ch_name)] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "ch", "peak_latency", channel=ch_name)] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "ch", "onset_latency", channel=ch_name)] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "ch", "early_late_diff", channel=ch_name)] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "ch", "erd_magnitude", channel=ch_name)] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "ch", "erd_duration", channel=ch_name)] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "ch", "ers_magnitude", channel=ch_name)] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "ch", "ers_duration", channel=ch_name)] = np.nan
                
                if np.any(np.isfinite(erds_trace)) and len(active_times) > 1:
                    valid_mask = np.isfinite(erds_trace)
                    
                    # Slope (linear trend over plateau)
                    if np.sum(valid_mask) > 2:
                        slope, _ = np.polyfit(active_times[valid_mask], erds_trace[valid_mask], 1)
                        record[NamingSchema.build("erds", segment_label, band, "ch", "slope", channel=ch_name)] = float(slope)
                    else:
                        record[NamingSchema.build("erds", segment_label, band, "ch", "slope", channel=ch_name)] = np.nan
                    
                    # Early-late difference
                    if "early" in coarse_values and "late" in coarse_values:
                        diff = coarse_values["late"] - coarse_values["early"]
                        record[NamingSchema.build("erds", segment_label, band, "ch", "early_late_diff", channel=ch_name)] = float(diff)
                    
                    # Peak latency
                    peak_idx = np.nanargmax(np.abs(erds_trace))
                    record[NamingSchema.build("erds", segment_label, band, "ch", "peak_latency", channel=ch_name)] = float(active_times[peak_idx])

                    # Onset latency
                    threshold = baseline_std / baseline_ref * 100 if baseline_ref > epsilon else np.inf
                    onset_mask = np.abs(erds_trace) > threshold
                    if np.any(onset_mask):
                        onset_idx = np.argmax(onset_mask)
                        record[NamingSchema.build("erds", segment_label, band, "ch", "onset_latency", channel=ch_name)] = float(active_times[onset_idx])
                    else:
                        record[NamingSchema.build("erds", segment_label, band, "ch", "onset_latency", channel=ch_name)] = np.nan
                    
                    # === ERD vs ERS separation ===
                    erd_vals = erds_trace[erds_trace < 0]
                    ers_vals = erds_trace[erds_trace > 0]
                    
                    if baseline_valid:
                        if len(erd_vals) > 0:
                            erd_magnitude = float(np.mean(np.abs(erd_vals)))
                            erd_duration = float(len(erd_vals) / precomputed.sfreq)
                        else:
                            erd_magnitude = 0.0
                            erd_duration = 0.0

                        record[NamingSchema.build("erds", segment_label, band, "ch", "erd_magnitude", channel=ch_name)] = erd_magnitude
                        record[NamingSchema.build("erds", segment_label, band, "ch", "erd_duration", channel=ch_name)] = erd_duration
                        
                        if len(ers_vals) > 0:
                            ers_magnitude = float(np.mean(ers_vals))
                            ers_duration = float(len(ers_vals) / precomputed.sfreq)
                        else:
                            ers_magnitude = 0.0
                            ers_duration = 0.0

                        record[NamingSchema.build("erds", segment_label, band, "ch", "ers_magnitude", channel=ch_name)] = ers_magnitude
                        record[NamingSchema.build("erds", segment_label, band, "ch", "ers_duration", channel=ch_name)] = ers_duration
            
            # === Global statistics per band ===
            valid_erds = [e for e in all_erds_full if np.isfinite(e)]
            valid_log = [e for e in all_log_full if np.isfinite(e)]
            baseline_valid_fraction = baseline_valid_count / n_channels if n_channels > 0 else 0.0
            
            if band not in qc_payload:
                qc_payload[band] = {"clamped_channels": [], "baseline_min_power": min_baseline_power, "valid_fractions": []}
            qc_payload[band]["clamped_channels"].append(int(clamped_channels_for_band))
            qc_payload[band]["valid_fractions"].append(float(baseline_valid_fraction))
            
            # Skip global summaries when baseline-valid fraction is too low to avoid mixing valid/invalid channels
            if baseline_valid_fraction < min_valid_fraction:
                record[NamingSchema.build("erds", segment_label, band, "global", "percent_mean")] = np.nan
                record[NamingSchema.build("erds", segment_label, band, "global", "percent_std")] = np.nan
                for win_label in windows.coarse_labels:
                    record[NamingSchema.build("erds", win_label, band, "global", "percent_mean")] = np.nan
                    if use_log_ratio:
                        record[NamingSchema.build("erds", win_label, band, "global", "db_mean")] = np.nan
                if use_log_ratio:
                    record[NamingSchema.build("erds", segment_label, band, "global", "db_mean")] = np.nan
                    record[NamingSchema.build("erds", segment_label, band, "global", "db_std")] = np.nan
            elif valid_erds:
                record[NamingSchema.build("erds", segment_label, band, "global", "percent_mean")] = float(np.mean(valid_erds))
                record[NamingSchema.build("erds", segment_label, band, "global", "percent_std")] = float(np.std(valid_erds))
                # Global per coarse bin
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_erds = []
                        win_log = []
                        for ch_idx in range(len(precomputed.ch_names)):
                            bp = np.mean(power[ch_idx, windows.baseline_mask])
                            bp_ref = bp if bp >= min_baseline_power else min_baseline_power
                            if bp_ref > epsilon:
                                wp = np.mean(power[ch_idx, win_mask])
                                win_erds.append(((wp - bp_ref) / bp_ref) * 100)
                                if use_log_ratio:
                                    win_log.append(10 * np.log10(max(wp, min_active_power) / bp_ref))
                        if win_erds:
                            record[NamingSchema.build("erds", win_label, band, "global", "percent_mean")] = float(np.mean(win_erds))
                        if use_log_ratio and win_log:
                            record[NamingSchema.build("erds", win_label, band, "global", "db_mean")] = float(np.mean(win_log))
                if use_log_ratio and valid_log:
                    record[NamingSchema.build("erds", segment_label, band, "global", "db_mean")] = float(np.mean(valid_log))
                    record[NamingSchema.build("erds", segment_label, band, "global", "db_std")] = float(np.std(valid_log))

        records.append(record)

    if clamped_baselines > 0 and precomputed.logger:
        precomputed.logger.info(
            "Clamped %d baseline power values below min_baseline_power=%.3e to stabilize ERD/ERS ratios (use_log_ratio=%s).",
            clamped_baselines,
            min_baseline_power,
            use_log_ratio,
        )
    
    # QC summary per band
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
    
    # Log warning if many epochs have low baseline validity
    for band, stats in qc_summary.items():
        if stats["n_epochs_low_validity"] > 0 and precomputed.logger:
            precomputed.logger.warning(
                "ERDS band '%s': %d/%d epochs had baseline_valid_fraction < %.1f%%; "
                "global summaries set to NaN for these epochs.",
                band,
                stats["n_epochs_low_validity"],
                n_epochs,
                min_valid_fraction * 100,
            )

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, qc_summary


def extract_power_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Extract time-resolved band power features using precomputed band data.
    
    Features extracted per channel/band:
    - Power per coarse temporal window (early, mid, late)
    - Power per fine temporal window (t1-t7) for HRF modeling
    - Temporal dynamics: slope, early-late diff
    - Baseline-normalized power (log-ratio)
    - Global (cross-channel) statistics per band
    """
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), [], {}

    # Early bail: enforce minimum epoch count for stable power estimation
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
    
    epsilon = get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12)
    min_valid_fraction = get_feature_constant(precomputed.config, "MIN_VALID_FRACTION", 0.5)
    windows = precomputed.windows
    
    records: List[Dict[str, float]] = []
    qc_payload: Dict[str, Dict[str, Any]] = {}
    n_epochs = precomputed.data.shape[0]
    times = precomputed.times
    active_times = times[windows.active_mask]
    
    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        
        for band in bands:
            if band not in precomputed.band_data:
                continue
            
            power = precomputed.band_data[band].power[ep_idx]  # (channels, times)
            baseline_valid_count = 0
            total_channels = len(precomputed.ch_names)
            all_power_full = []
            
            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                baseline_power, baseline_frac, _, baseline_total = nanmean_with_fraction(
                    power[ch_idx], windows.baseline_mask
                )
                active_power, active_frac, _, _ = nanmean_with_fraction(
                    power[ch_idx], windows.active_mask
                )
                baseline_valid = baseline_power > epsilon and baseline_frac >= min_valid_fraction and baseline_total > 0
                if baseline_valid:
                    baseline_valid_count += 1
                
                # Full period power (log-ratio normalized)
                if baseline_valid:
                    logratio = np.log10(active_power / baseline_power)
                else:
                    logratio = np.nan
                record[NamingSchema.build("power", "plateau", band, "ch", "logratio", channel=ch_name)] = float(logratio)
                all_power_full.append(logratio)
                
                # === Coarse temporal bins (early, mid, late) ===
                coarse_values = {}
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_power, win_frac, _, _ = nanmean_with_fraction(power[ch_idx], win_mask)
                        if baseline_valid and win_frac >= min_valid_fraction:
                            win_logratio = np.log10(win_power / baseline_power)
                        else:
                            win_logratio = np.nan
                        record[NamingSchema.build("power", win_label, band, "ch", "logratio", channel=ch_name)] = float(win_logratio)
                        coarse_values[win_label] = win_logratio
                
                # === Fine temporal bins (t1-t7) for HRF modeling ===
                for win_mask, win_label in zip(windows.fine_masks, windows.fine_labels):
                    if np.any(win_mask):
                        win_power, win_frac, _, _ = nanmean_with_fraction(power[ch_idx], win_mask)
                        if baseline_valid and win_frac >= min_valid_fraction:
                            win_logratio = np.log10(win_power / baseline_power)
                        else:
                            win_logratio = np.nan
                        record[NamingSchema.build("power", win_label, band, "ch", "logratio", channel=ch_name)] = float(win_logratio)
                
                # === Temporal dynamics ===
                if len(active_times) > 2:
                    active_power_trace = power[ch_idx, windows.active_mask]
                    if baseline_valid:
                        logratio_trace = np.log10(active_power_trace / baseline_power)
                        valid_mask = np.isfinite(logratio_trace)
                        if np.sum(valid_mask) > 2:
                            slope, _ = np.polyfit(active_times[valid_mask], logratio_trace[valid_mask], 1)
                            record[NamingSchema.build("power", "plateau", band, "ch", "slope", channel=ch_name)] = float(slope)
                        else:
                            record[NamingSchema.build("power", "plateau", band, "ch", "slope", channel=ch_name)] = np.nan
                    else:
                        record[NamingSchema.build("power", "plateau", band, "ch", "slope", channel=ch_name)] = np.nan
                    
                    # Early-late difference
                    if "early" in coarse_values and "late" in coarse_values:
                        diff = coarse_values["late"] - coarse_values["early"]
                        record[NamingSchema.build("power", "plateau", band, "ch", "early_late_diff", channel=ch_name)] = float(diff)
            
            # === Global statistics per band ===
            valid_power = [p for p in all_power_full if np.isfinite(p)]
            baseline_valid_fraction = baseline_valid_count / total_channels if total_channels > 0 else 0.0
            record[NamingSchema.build("power", "baseline", band, "global", "valid_fraction")] = float(baseline_valid_fraction)
            
            # Skip global summaries when baseline-valid fraction is too low
            if baseline_valid_fraction < min_valid_fraction:
                record[NamingSchema.build("power", "plateau", band, "global", "logratio_mean")] = np.nan
                record[NamingSchema.build("power", "plateau", band, "global", "logratio_std")] = np.nan
                for win_label in windows.coarse_labels:
                    record[NamingSchema.build("power", win_label, band, "global", "logratio_mean")] = np.nan
            elif valid_power:
                record[NamingSchema.build("power", "plateau", band, "global", "logratio_mean")] = float(np.mean(valid_power))
                record[NamingSchema.build("power", "plateau", band, "global", "logratio_std")] = float(np.std(valid_power))
                
                # Global per coarse bin
                for win_mask, win_label in zip(windows.coarse_masks, windows.coarse_labels):
                    if np.any(win_mask):
                        win_powers = []
                        for ch_idx in range(len(precomputed.ch_names)):
                            bp, bp_frac, _, bp_total = nanmean_with_fraction(power[ch_idx], windows.baseline_mask)
                            if bp > epsilon and bp_frac >= min_valid_fraction and bp_total > 0:
                                wp, wp_frac, _, _ = nanmean_with_fraction(power[ch_idx], win_mask)
                                if wp_frac >= min_valid_fraction:
                                    win_powers.append(np.log10(wp / bp))
                        if win_powers:
                            record[NamingSchema.build("power", win_label, band, "global", "logratio_mean")] = float(np.mean(win_powers))

        records.append(record)

    columns = list(records[0].keys()) if records else []
    return pd.DataFrame(records), columns, {}


###################################################################
# Precomputed Power Helpers (Spectral, Asymmetry, Segment Power)
###################################################################


def _compute_spectral_peak(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> Tuple[float, float]:
    """Find peak frequency and power within a band."""
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
    """Process spectral extras for a single epoch (parallel worker)."""
    record: Dict[str, float] = {}
    for ch_idx, ch_name in enumerate(ch_names):
        psd_ep = psd[ep_idx, ch_idx, :]
        
        for band in bands:
            if band not in freq_bands:
                continue
            fmin, fmax = freq_bands[band]
            pf, pp = _compute_spectral_peak(psd_ep, freqs, fmin, fmax)
            record[NamingSchema.build("spectral", "baseline", band, "ch", "peakfreq", channel=ch_name)] = pf
            record[NamingSchema.build("spectral", "baseline", band, "ch", "peakpow", channel=ch_name)] = pp

        mask_fit = (freqs >= 1) & (freqs <= 40)
        if np.sum(mask_fit) > 2:
            log_f = np.log10(freqs[mask_fit])
            log_p = np.log10(psd_ep[mask_fit])
            slope, intercept = np.polyfit(log_f, log_p, 1)
            record[NamingSchema.build("spectral", "baseline", "broadband", "ch", "aperiodic_slope", channel=ch_name)] = float(slope)
            record[NamingSchema.build("spectral", "baseline", "broadband", "ch", "aperiodic_offset", channel=ch_name)] = float(intercept)
    return record


def extract_spectral_extras_from_precomputed(
    precomputed: Any,
    bands: List[str],
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute spectral features that require PSD (Peak Power, Peak Freq, 1/f slope)."""
    if precomputed.psd_data is None:
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
    return pd.DataFrame(records), list(pd.DataFrame(records).columns)


###################################################################
# Additional precomputed groups advertised in config
###################################################################


def extract_gfp_from_precomputed(precomputed: PrecomputedData, config: Any) -> Tuple[pd.DataFrame, List[str]]:
    if precomputed.data is None or precomputed.windows is None:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    windows = precomputed.windows
    masks = get_segment_masks(precomputed.times, windows, config)

    records: List[Dict[str, float]] = []
    for ep_idx in range(precomputed.data.shape[0]):
        x = precomputed.data[ep_idx]
        gfp_t = np.nanstd(x, axis=0)
        rec: Dict[str, float] = {}
        for seg, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            vals = gfp_t[mask]
            rec[NamingSchema.build("gfp", seg, "broadband", "global", "mean")] = float(np.nanmean(vals))
            rec[NamingSchema.build("gfp", seg, "broadband", "global", "max")] = float(np.nanmax(vals))
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


def _compile_roi_indices(ch_names: List[str], roi_defs: Dict[str, List[str]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for roi_name, patterns in (roi_defs or {}).items():
        indices: List[int] = []
        compiled = [re.compile(p) for p in patterns] if isinstance(patterns, list) else []
        for idx, ch in enumerate(ch_names):
            if any(rgx.match(ch) for rgx in compiled):
                indices.append(idx)
        if indices:
            out[str(roi_name)] = indices
    return out


def extract_roi_features_from_precomputed(precomputed: PrecomputedData, bands: List[str], config: Any) -> Tuple[pd.DataFrame, List[str]]:
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []

    roi_defs = get_config_value(config, "time_frequency_analysis.rois", {})
    roi_to_idx = _compile_roi_indices(precomputed.ch_names, roi_defs)
    if not roi_to_idx:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    windows = precomputed.windows
    masks = get_segment_masks(precomputed.times, windows, config)
    baseline_mask = masks.get("baseline")
    segments: Dict[str, Optional[np.ndarray]] = {
        "ramp": masks.get("ramp"),
        "plateau": masks.get("plateau"),
    }
    epsilon = float(get_feature_constant(config, "EPSILON_STD", 1e-12))

    if baseline_mask is None or not np.any(baseline_mask):
        return pd.DataFrame(), []

    records: List[Dict[str, float]] = []
    for ep_idx in range(precomputed.data.shape[0]):
        rec: Dict[str, float] = {}
        for band in bands:
            if band not in precomputed.band_data:
                continue
            power = precomputed.band_data[band].power[ep_idx]  # (ch, t)
            base_ch = np.nanmean(power[:, baseline_mask], axis=1)
            for roi_name, idxs in roi_to_idx.items():
                base = float(np.nanmean(base_ch[idxs]))
                base = base if np.isfinite(base) and base > epsilon else np.nan
                for seg, mask in segments.items():
                    if mask is None or not np.any(mask):
                        continue
                    seg_ch = np.nanmean(power[:, mask], axis=1)
                    seg_val = float(np.nanmean(seg_ch[idxs]))
                    if base is not None and np.isfinite(base) and base > epsilon and np.isfinite(seg_val):
                        logratio = float(np.log10(seg_val / base))
                    else:
                        logratio = np.nan
                    rec[NamingSchema.build("roi", seg, band, "global", f"{roi_name}_logratio_mean")] = logratio
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


def extract_temporal_features_from_precomputed(precomputed: PrecomputedData, config: Any) -> Tuple[pd.DataFrame, List[str]]:
    if precomputed.data is None or precomputed.windows is None:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    windows = precomputed.windows
    masks = get_segment_masks(precomputed.times, windows, config)

    records: List[Dict[str, float]] = []
    for ep_idx in range(precomputed.data.shape[0]):
        x = precomputed.data[ep_idx]
        rec: Dict[str, float] = {}
        for seg, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            seg_x = x[:, mask]
            # Simple, robust summaries averaged over channels
            var_mean = float(np.nanmean(np.nanvar(seg_x, axis=1)))
            rms_mean = float(np.nanmean(np.sqrt(np.nanmean(seg_x ** 2, axis=1))))
            ll_mean = float(np.nanmean(np.nanmean(np.abs(np.diff(seg_x, axis=1)), axis=1))) if seg_x.shape[1] > 1 else np.nan
            rec[NamingSchema.build("temporal", seg, "broadband", "global", "var_mean")] = var_mean
            rec[NamingSchema.build("temporal", seg, "broadband", "global", "rms_mean")] = rms_mean
            rec[NamingSchema.build("temporal", seg, "broadband", "global", "line_length_mean")] = ll_mean
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


def extract_band_ratios_from_precomputed(precomputed: PrecomputedData, config: Any) -> Tuple[pd.DataFrame, List[str]]:
    if not precomputed.band_data or precomputed.windows is None:
        return pd.DataFrame(), []

    ratio_pairs = get_config_value(config, "feature_engineering.spectral.ratio_pairs", [])
    pairs: List[Tuple[str, str]] = []
    for entry in ratio_pairs:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            pairs.append((str(entry[0]), str(entry[1])))
    if not pairs:
        return pd.DataFrame(), []

    plateau_mask = getattr(precomputed.windows, "active_mask", None)
    if plateau_mask is None or not np.any(plateau_mask):
        return pd.DataFrame(), []

    eps = float(get_feature_constant(config, "EPSILON_STD", 1e-12))
    records: List[Dict[str, float]] = []
    for ep_idx in range(precomputed.data.shape[0]):
        rec: Dict[str, float] = {}
        band_means: Dict[str, float] = {}
        for band, bd in precomputed.band_data.items():
            p = bd.power[ep_idx]
            band_means[band] = float(np.nanmean(p[:, plateau_mask]))
        for num, den in pairs:
            if num not in band_means or den not in band_means:
                continue
            denom = band_means[den]
            if not np.isfinite(denom) or denom <= eps:
                ratio = np.nan
            else:
                ratio = float(band_means[num] / denom)
            rec[NamingSchema.build("ratios", "plateau", f"{num}_{den}", "global", "power_ratio")] = ratio
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


def _process_asymmetry_epoch(
    ep_idx: int,
    band_data: Dict[str, Any],
    valid_pairs: List[Tuple[str, str, int, int]],
    active_mask: np.ndarray,
) -> Dict[str, float]:
    """Process asymmetry for a single epoch (parallel worker)."""
    record: Dict[str, float] = {}
    for band, bd in band_data.items():
        power = bd.power[ep_idx]
        p_active = power[:, active_mask] if not isinstance(active_mask, slice) else power
        p_mean = np.nanmean(p_active, axis=1)
        
        for l_name, r_name, l_idx, r_idx in valid_pairs:
            pl, pr = p_mean[l_idx], p_mean[r_idx]
            denom = pr + pl
            asym = (pr - pl) / denom if denom > 1e-12 else 0.0
            pair = f"{l_name}-{r_name}"
            record[NamingSchema.build("asymmetry", "plateau", band, "chpair", "index", channel_pair=pair)] = float(asym)
            if pr > 0 and pl > 0:
                record[NamingSchema.build("asymmetry", "plateau", band, "chpair", "logdiff", channel_pair=pair)] = float(np.log(pr) - np.log(pl))
    return record


def extract_asymmetry_from_precomputed(
    precomputed: Any,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute frontal alpha asymmetry and other hemispheric asymmetries."""
    if not precomputed.band_data:
        return pd.DataFrame(), []

    pairs = [("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]
    valid_pairs = []
    ch_map = {name: i for i, name in enumerate(precomputed.ch_names)}
    for l, r in pairs:
        if l in ch_map and r in ch_map:
            valid_pairs.append((l, r, ch_map[l], ch_map[r]))
    if not valid_pairs:
        return pd.DataFrame(), []
    
    active_mask = getattr(precomputed.windows, "active_mask", None)
    if active_mask is None or not np.any(active_mask):
        active_mask = slice(None)

    n_epochs = precomputed.data.shape[0]
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_asymmetry_epoch)(ep_idx, precomputed.band_data, valid_pairs, active_mask)
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_asymmetry_epoch(ep_idx, precomputed.band_data, valid_pairs, active_mask)
            for ep_idx in range(n_epochs)
        ]
    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    return pd.DataFrame(records), list(pd.DataFrame(records).columns)


def _process_segment_power_epoch(
    ep_idx: int,
    band_data: Dict[str, Any],
    bands: List[str],
    ch_names: List[str],
    baseline_mask: np.ndarray,
    segments_active: Dict[str, np.ndarray],
    epsilon: float,
) -> Dict[str, float]:
    """Process segment power for a single epoch (parallel worker)."""
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
            col = NamingSchema.build("power", "baseline", band, "ch", "logpower", channel=ch_name)
            record[col] = float(baseline_log)
        
        valid_baseline = baseline_power_per_ch[baseline_power_per_ch > epsilon]
        if len(valid_baseline) > 0:
            col_glob = NamingSchema.build("power", "baseline", band, "global", "logpower")
            record[col_glob] = float(np.nanmean(np.log10(valid_baseline)))
        
        for seg_name, seg_mask in segments_active.items():
            seg_power = power[:, seg_mask] if not isinstance(seg_mask, slice) else power
            seg_mean = np.nanmean(seg_power, axis=1)
            
            for ch_idx, ch_name in enumerate(ch_names):
                if baseline_power_per_ch[ch_idx] > epsilon:
                    logratio = np.log10(seg_mean[ch_idx] / baseline_power_per_ch[ch_idx])
                else:
                    logratio = np.nan
                col = NamingSchema.build("power", seg_name, band, "ch", "logratio", channel=ch_name)
                record[col] = float(logratio)
            
            valid = baseline_power_per_ch > epsilon
            if np.any(valid):
                col_glob = NamingSchema.build("power", seg_name, band, "global", "logratio")
                record[col_glob] = float(np.nanmean(np.log10(seg_mean[valid] / baseline_power_per_ch[valid])))
    return record


def extract_segment_power_from_precomputed(
    precomputed: Any,
    bands: List[str],
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute baseline/ramp/plateau/offset band power with proper naming schema."""
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    baseline_mask = masks.get("baseline")
    if baseline_mask is None or not np.any(baseline_mask):
        return pd.DataFrame(), []

    epsilon = float(get_feature_constant(precomputed.config, "EPSILON_STD", 1e-12))
    
    # NOTE: plateau logratio features are produced by extract_power_from_precomputed().
    # To avoid duplicate column names, this extractor contributes only non-plateau segments.
    segments_active = {k: v for k, v in masks.items() if k in ["ramp", "offset"] and v is not None and np.any(v)}
    n_epochs = precomputed.data.shape[0]

    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_segment_power_epoch)(
                ep_idx, precomputed.band_data, bands, precomputed.ch_names,
                baseline_mask, segments_active, epsilon
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_segment_power_epoch(
                ep_idx, precomputed.band_data, bands, precomputed.ch_names,
                baseline_mask, segments_active, epsilon
            )
            for ep_idx in range(n_epochs)
        ]

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    return pd.DataFrame(records), list(pd.DataFrame(records).columns)
