"""
Extra Precomputed Feature Extractors.

Consolidated module for smaller/specialized precomputed feature extractors:
- Band ratios
- Hemispheric asymmetry
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.analysis.features.rest import (
    is_resting_state_feature_mode,
    select_single_rest_analysis_segment,
    valid_rest_analysis_segment_masks,
)
from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.utils.config.loader import get_config_value, get_feature_constant, get_frequency_bands


###################################################################
# VALIDATION
###################################################################

def validate_window_masks(
    precomputed: PrecomputedData,
    logger: Optional[logging.Logger] = None,
    *,
    require_baseline: bool = True,
    require_active: bool = True,
) -> bool:
    windows = precomputed.windows
    if windows is None:
        if logger:
            logger.warning("Time windows are missing; skipping feature extraction.")
        return False

    if require_baseline:
        baseline_mask = windows.get_mask("baseline")
        if baseline_mask is None or not np.any(baseline_mask):
            if logger:
                logger.warning(
                    "Baseline window is missing or empty. This may affect features "
                    "requiring baseline normalization."
                )
            return False

    if require_active:
        target_name = getattr(windows, "name", None)
        if target_name and target_name in windows.masks:
            mask = windows.get_mask(target_name)
            if mask is not None and np.any(mask):
                return True
        
        has_active = any(
            name.lower() != "baseline" and np.any(mask) 
            for name, mask in windows.masks.items()
        )
        if not has_active:
            if logger:
                logger.warning("No active/task windows found; skipping feature extraction.")
            return False

    return True


###################################################################
# BAND RATIO FEATURES
###################################################################

def _get_psd_config(config: Any, sfreq: float) -> Dict[str, Any]:
    """Extract PSD configuration from config."""
    psd_method = get_config_value(config, "feature_engineering.spectral.psd_method", "multitaper")
    psd_method = str(psd_method).strip().lower()
    if psd_method not in {"welch", "multitaper"}:
        psd_method = "multitaper"
    
    fmin_psd = float(get_config_value(config, "feature_engineering.spectral.fmin", 1.0))
    fmax_psd = float(get_config_value(config, "feature_engineering.spectral.fmax", min(80.0, sfreq / 2.0 - 0.5)))
    
    multitaper_adaptive = bool(
        get_config_value(config, "feature_engineering.spectral.multitaper_adaptive", False) or
        get_config_value(config, "feature_engineering.spectral.psd_adaptive", False)
    )
    
    exclude_line = bool(get_config_value(config, "feature_engineering.spectral.exclude_line_noise", True))

    default_line_freq = get_config_value(config, "preprocessing.line_freq", 50.0)
    try:
        default_line_freq = float(default_line_freq)
        if not (np.isfinite(default_line_freq) and default_line_freq > 0):
            default_line_freq = 50.0
    except (TypeError, ValueError):
        default_line_freq = 50.0

    line_freqs_raw = get_config_value(
        config,
        "feature_engineering.spectral.line_noise_freqs",
        [default_line_freq],
    )
    try:
        if isinstance(line_freqs_raw, (list, tuple)):
            line_freqs = [float(f) for f in line_freqs_raw]
        else:
            line_freqs = [default_line_freq]
    except (ValueError, TypeError):
        line_freqs = [default_line_freq]
    
    line_width = float(get_config_value(config, "feature_engineering.spectral.line_noise_width_hz", 1.0))
    n_harm = int(get_config_value(config, "feature_engineering.spectral.line_noise_harmonics", 3))
    
    return {
        "psd_method": psd_method,
        "fmin": fmin_psd,
        "fmax": fmax_psd,
        "adaptive": multitaper_adaptive,
        "exclude_line": exclude_line,
        "line_freqs": line_freqs,
        "line_width": line_width,
        "n_harmonics": n_harm,
    }


def _compute_psd_band_power_for_segment(
    data: np.ndarray,
    sfreq: float,
    band_ranges: Dict[str, Tuple[float, float]],
    config: Any,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Compute PSD-integrated band power for a data segment.
    
    Uses centralized PSD settings from config (method, fmin/fmax, line-noise exclusion).
    Returns dict mapping band names to power arrays:
    - (n_channels,) if data is 2D (channels, times)
    - (n_epochs, n_channels) if data is 3D (epochs, channels, times)
    """
    from eeg_pipeline.utils.analysis.spectral import compute_psd_bandpower
    
    psd_cfg = _get_psd_config(config, sfreq)
    
    if data.ndim == 2:
        data_3d = data[np.newaxis, :, :]
    else:
        data_3d = data
    
    result = compute_psd_bandpower(
        data_3d,
        sfreq,
        band_ranges,
        psd_method=psd_cfg["psd_method"],
        fmin=psd_cfg["fmin"],
        fmax=psd_cfg["fmax"],
        normalize_by_bandwidth=True,
        adaptive=psd_cfg["adaptive"],
        exclude_line_noise=psd_cfg["exclude_line"],
        line_freqs=psd_cfg["line_freqs"] if psd_cfg["exclude_line"] else None,
        line_width=psd_cfg["line_width"] if psd_cfg["exclude_line"] else None,
        n_harmonics=psd_cfg["n_harmonics"] if psd_cfg["exclude_line"] else None,
        logger=logger,
    )
    
    if result is None:
        return None
    
    if data.ndim == 2:
        return {band: arr[0] for band, arr in result.items()}
    return result


def _get_segment_masks_with_fallback(
    precomputed: PrecomputedData,
    config: Any,
    logger: Optional[logging.Logger],
    feature_name: str,
) -> Dict[str, np.ndarray]:
    """Get valid segment masks for precomputed extras."""
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks

    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    task_is_rest = is_resting_state_feature_mode(config)

    def _valid_analysis_masks(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return valid_rest_analysis_segment_masks(masks)

    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            return {target_name: mask}

        if task_is_rest:
            segment_name, segment_mask = select_single_rest_analysis_segment(
                get_segment_masks(precomputed.times, windows, config),
                feature_name=feature_name,
                target_name=str(target_name),
            )
            if logger:
                logger.info(
                    "%s: resting-state mode found no valid target window '%s'; "
                    "using available analysis segment '%s' instead.",
                    feature_name,
                    target_name,
                    segment_name,
                )
            return {segment_name: segment_mask}

        if logger:
            logger.error(
                "%s: targeted window '%s' has no valid mask; skipping.",
                feature_name,
                target_name,
            )
        return {}

    segment_masks = get_segment_masks(precomputed.times, windows, config)
    if task_is_rest:
        return _valid_analysis_masks(segment_masks)
    return segment_masks


def extract_band_ratios_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract band power ratios for ALL user-defined segments.
    
    Uses PSD-integrated band power (scientifically valid for ratios).
    Power is bandwidth-normalized (power per Hz) for comparability across bands.
    """
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("Band ratios: %s; skipping extraction.", err_msg)
        return pd.DataFrame(), []

    ratio_pairs = get_config_value(config, "feature_engineering.spectral.ratio_pairs", [])
    pairs: List[Tuple[str, str]] = []
    for entry in ratio_pairs:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            pairs.append((str(entry[0]), str(entry[1])))
    if not pairs:
        return pd.DataFrame(), []

    logger = getattr(precomputed, "logger", None)
    segment_masks = _get_segment_masks_with_fallback(precomputed, config, logger, "Band ratios")
    
    if not segment_masks:
        return pd.DataFrame(), []

    spatial_modes = getattr(precomputed, 'spatial_modes', ['roi', 'global'])
    roi_map = {}
    if 'roi' in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(config)
        if roi_defs:
            roi_map = build_roi_map(precomputed.ch_names, roi_defs)

    eps = float(get_feature_constant(config, "EPSILON_STD", 1e-12))
    include_log = bool(get_config_value(config, "feature_engineering.spectral.include_log_ratios", True))
    min_segment_sec = float(get_config_value(config, "feature_engineering.ratios.min_segment_sec", 1.0))
    min_cycles = float(get_config_value(config, "feature_engineering.ratios.min_cycles_at_fmin", 3.0))
    skip_invalid = bool(get_config_value(config, "feature_engineering.ratios.skip_invalid_segments", True))
    
    freq_bands = get_frequency_bands(config)
    needed_bands = {band for pair in pairs for band in pair}
    band_ranges = {b: tuple(freq_bands[b]) for b in needed_bands if b in freq_bands}
    
    n_epochs = precomputed.data.shape[0]
    sfreq = precomputed.sfreq
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    min_segment_samples = max(0, int(round(min_segment_sec * sfreq)))
    warned_short: set[str] = set()
    warned_cycles: set[tuple[str, str]] = set()
    
    for seg_label, seg_mask in segment_masks.items():
        if seg_mask is None or not np.any(seg_mask):
            continue

        seg_n = int(np.sum(seg_mask))
        if min_segment_samples > 0 and seg_n < min_segment_samples:
            if logger and seg_label not in warned_short:
                logger.warning(
                    "Band ratios: segment '%s' is too short for PSD ratios (%.3fs, %d samples < %d); skipping.",
                    seg_label,
                    seg_n / sfreq,
                    seg_n,
                    min_segment_samples,
                )
                warned_short.add(seg_label)
            if skip_invalid:
                continue
        
        seg_data_all = precomputed.data[:, :, seg_mask]  # (epochs, ch, time)
        band_power = _compute_psd_band_power_for_segment(
            seg_data_all, sfreq, band_ranges, config, logger
        )
        if band_power is None:
            continue

        seg_sec = float(seg_n) / float(sfreq) if sfreq > 0 else 0.0

        for num, den in pairs:
            if num not in band_power or den not in band_power:
                continue

            try:
                fmin_num = float(band_ranges.get(num, (np.nan, np.nan))[0])
                fmin_den = float(band_ranges.get(den, (np.nan, np.nan))[0])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid frequency band definitions for ratio pair '{num}/{den}': "
                    f"{band_ranges.get(num)} / {band_ranges.get(den)}"
                ) from exc
            fmin_pair = np.nanmin([fmin_num, fmin_den])
            req_sec = max(float(min_segment_sec), (float(min_cycles) / float(fmin_pair)) if np.isfinite(fmin_pair) and fmin_pair > 0 else np.inf)
            if seg_sec < req_sec:
                if logger and (seg_label, f"{num}/{den}") not in warned_cycles:
                    logger.warning(
                        "Band ratios: segment '%s' too short for '%s/%s' (%.3fs < %.3fs; min_cycles=%s at fmin=%.2f).",
                        seg_label,
                        num,
                        den,
                        seg_sec,
                        req_sec,
                        min_cycles,
                        float(fmin_pair) if np.isfinite(fmin_pair) else np.nan,
                    )
                    warned_cycles.add((seg_label, f"{num}/{den}"))
                if skip_invalid:
                    continue

            p_num = band_power[num]  # (epochs, ch)
            p_den = band_power[den]  # (epochs, ch)

            with np.errstate(divide="ignore", invalid="ignore"):
                r_all = p_num / p_den
                r_all[p_den <= eps] = np.nan

            log_r_all = None
            if include_log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_r_all = np.log(p_num + eps) - np.log(p_den + eps)
                    log_r_all[(p_num <= eps) | (p_den <= eps)] = np.nan

            pair_label = f"{num}_{den}"

            for ep_idx in range(n_epochs):
                rec = records[ep_idx]
                r_ch = r_all[ep_idx]
                log_r_ch = log_r_all[ep_idx] if log_r_all is not None else None
                p_num_ch = p_num[ep_idx]
                p_den_ch = p_den[ep_idx]

                valid_ch = np.isfinite(p_num_ch) & np.isfinite(p_den_ch) & (p_den_ch > eps)

                if "channels" in spatial_modes:
                    for c, ch in enumerate(precomputed.ch_names):
                        col = NamingSchema.build(
                            "ratios", seg_label, pair_label, "ch", "power_ratio", channel=ch
                        )
                        rec[col] = float(r_ch[c])
                        if include_log and log_r_ch is not None:
                            col_log = NamingSchema.build(
                                "ratios", seg_label, pair_label, "ch", "log_ratio", channel=ch
                            )
                            rec[col_log] = float(log_r_ch[c])

                if "roi" in spatial_modes and roi_map:
                    for roi_name, idxs in roi_map.items():
                        idx = np.asarray(idxs, dtype=int)
                        if idx.size == 0:
                            continue

                        roi_valid = valid_ch[idx]
                        if np.any(roi_valid):
                            mean_num_roi = float(np.nanmean(p_num_ch[idx][roi_valid]))
                            mean_den_roi = float(np.nanmean(p_den_ch[idx][roi_valid]))
                            val = float(mean_num_roi / mean_den_roi) if mean_den_roi > eps else np.nan
                            val_log = (
                                float(np.log(mean_num_roi) - np.log(mean_den_roi))
                                if mean_num_roi > eps and mean_den_roi > eps
                                else np.nan
                            )
                        else:
                            val = np.nan
                            val_log = np.nan

                        col = NamingSchema.build(
                            "ratios", seg_label, pair_label, "roi", "power_ratio", channel=roi_name
                        )
                        rec[col] = float(val)
                        if include_log:
                            col_log = NamingSchema.build(
                                "ratios", seg_label, pair_label, "roi", "log_ratio", channel=roi_name
                            )
                            rec[col_log] = float(val_log)

                if "global" in spatial_modes:
                    if np.any(valid_ch):
                        mean_num_global = float(np.nanmean(p_num_ch[valid_ch]))
                        mean_den_global = float(np.nanmean(p_den_ch[valid_ch]))
                        val = float(mean_num_global / mean_den_global) if mean_den_global > eps else np.nan
                        val_log = (
                            float(np.log(mean_num_global) - np.log(mean_den_global))
                            if mean_num_global > eps and mean_den_global > eps
                            else np.nan
                        )
                    else:
                        val = np.nan
                        val_log = np.nan

                    col = NamingSchema.build("ratios", seg_label, pair_label, "global", "power_ratio")
                    rec[col] = float(val)
                    if include_log:
                        col_log = NamingSchema.build("ratios", seg_label, pair_label, "global", "log_ratio")
                        rec[col_log] = float(val_log)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    df.attrs["precomputed_feature_family"] = str(getattr(precomputed, "feature_family", "") or "")
    df.attrs["precomputed_spatial_transform"] = str(getattr(precomputed, "spatial_transform", "") or "")
    df.attrs["precomputed_evoked_subtracted"] = bool(getattr(precomputed, "evoked_subtracted", False))
    df.attrs["precomputed_evoked_subtracted_conditionwise"] = bool(
        getattr(precomputed, "evoked_subtracted_conditionwise", False)
    )
    return df, list(df.columns)


###################################################################
# ASYMMETRY FEATURES
###################################################################

def _process_asymmetry_epoch(
    ep_idx: int,
    band_power: Dict[str, np.ndarray],
    valid_pairs: List[Tuple[str, str, int, int]],
    segment_label: str,
    *,
    eps: float,
    emit_activation_convention: bool,
    activation_bands: set[str],
) -> Dict[str, float]:
    """Process asymmetry for a single epoch using pre-computed band power.
    
    Args:
        ep_idx: Epoch index (for logging only)
        band_power: Dict mapping band names to (n_channels,) power arrays
        valid_pairs: List of (left_name, right_name, left_idx, right_idx) tuples
        segment_label: Name of the time segment
    """
    record: Dict[str, float] = {}
    
    for band, p_mean in band_power.items():
        band_lc = str(band).strip().lower()
        for l_name, r_name, l_idx, r_idx in valid_pairs:
            pl, pr = p_mean[l_idx], p_mean[r_idx]
            pair = f"{l_name}-{r_name}"
            
            # Use NaN when asymmetry is undefined (tiny denominator)
            denom = pr + pl
            if denom > eps and np.isfinite(pl) and np.isfinite(pr):
                asym = (pr - pl) / denom
            else:
                asym = np.nan
            
            record[
                NamingSchema.build(
                    "asymmetry",
                    segment_label,
                    band,
                    "chpair",
                    "index",
                    channel_pair=pair,
                )
            ] = float(asym)
            
            # Log-difference (ln(R) - ln(L)) - primary metric for frontal alpha asymmetry
            if pr > eps and pl > eps and np.isfinite(pl) and np.isfinite(pr):
                logdiff = float(np.log(pr) - np.log(pl))
            else:
                logdiff = np.nan
            
            record[
                NamingSchema.build(
                    "asymmetry",
                    segment_label,
                    band,
                    "chpair",
                    "logdiff",
                    channel_pair=pair,
                )
            ] = logdiff

            if emit_activation_convention and band_lc in activation_bands:
                record[
                    NamingSchema.build(
                        "asymmetry",
                        segment_label,
                        band,
                        "chpair",
                        "logdiff_activation",
                        channel_pair=pair,
                    )
                ] = float(-logdiff) if np.isfinite(logdiff) else np.nan
    return record


def extract_asymmetry_from_precomputed(
    precomputed: Any,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract hemispheric asymmetry features for ALL user-defined segments.
    
    Uses PSD-integrated band power (scientifically valid for asymmetry indices).
    Power is bandwidth-normalized (power per Hz) for comparability across bands.
    """
    logger = getattr(precomputed, "logger", None)
    config = precomputed.config
    
    default_pairs = [("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]
    pairs_cfg = get_config_value(config, "feature_engineering.asymmetry.channel_pairs", None)
    pairs: List[Tuple[str, str]] = []
    if isinstance(pairs_cfg, list):
        for entry in pairs_cfg:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                left, right = entry[0], entry[1]
                if isinstance(left, str) and isinstance(right, str):
                    pairs.append((left, right))
    if not pairs:
        pairs = default_pairs

    ch_map = {name: i for i, name in enumerate(precomputed.ch_names)}
    valid_pairs = [(l, r, ch_map[l], ch_map[r]) for l, r in pairs if l in ch_map and r in ch_map]
    if not valid_pairs:
        return pd.DataFrame(), []

    segment_masks = _get_segment_masks_with_fallback(precomputed, config, logger, "Asymmetry")
    
    if not segment_masks:
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    band_ranges = {b: tuple(freq_bands[b]) for b in freq_bands}

    n_epochs = precomputed.data.shape[0]
    sfreq = precomputed.sfreq
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    min_segment_sec = float(get_config_value(config, "feature_engineering.asymmetry.min_segment_sec", 1.0))
    min_cycles = float(get_config_value(config, "feature_engineering.asymmetry.min_cycles_at_fmin", 3.0))
    skip_invalid = bool(get_config_value(config, "feature_engineering.asymmetry.skip_invalid_segments", True))
    eps = float(get_feature_constant(config, "EPSILON_STD", 1e-12))
    min_segment_samples = max(0, int(round(min_segment_sec * sfreq)))
    warned_short: set[str] = set()
    warned_cycles: set[tuple[str, str]] = set()

    emit_activation_convention = bool(
        get_config_value(config, "feature_engineering.asymmetry.emit_activation_convention", False)
    )
    activation_bands_cfg = get_config_value(
        config, "feature_engineering.asymmetry.activation_bands", ["alpha"]
    )
    activation_bands = (
        {str(b).strip().lower() for b in activation_bands_cfg}
        if isinstance(activation_bands_cfg, (list, tuple))
        else {"alpha"}
    )
    
    for seg_label, seg_mask in segment_masks.items():
        if seg_mask is None or not np.any(seg_mask):
            continue

        seg_n = int(np.sum(seg_mask))
        if min_segment_samples > 0 and seg_n < min_segment_samples:
            if logger and seg_label not in warned_short:
                logger.warning(
                    "Asymmetry: segment '%s' is too short for PSD asymmetry (%.3fs, %d samples < %d); skipping.",
                    seg_label,
                    seg_n / sfreq,
                    seg_n,
                    min_segment_samples,
                )
                warned_short.add(seg_label)
            if skip_invalid:
                continue
        
        seg_data_all = precomputed.data[:, :, seg_mask]

        seg_sec = float(seg_n) / float(sfreq) if sfreq > 0 else 0.0
        eligible_bands: Dict[str, Tuple[float, float]] = {}
        for band, (fmin, fmax) in band_ranges.items():
            try:
                fmin_f = float(fmin)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid fmin for band '{band}': {fmin}") from exc
            req_sec = max(float(min_segment_sec), (float(min_cycles) / fmin_f) if fmin_f > 0 else np.inf)
            if seg_sec >= req_sec:
                eligible_bands[band] = (fmin, fmax)
            else:
                if logger and (seg_label, str(band)) not in warned_cycles:
                    logger.warning(
                        "Asymmetry: segment '%s' too short for '%s' (%.3fs < %.3fs; min_cycles=%s at fmin=%.2f).",
                        seg_label,
                        band,
                        seg_sec,
                        req_sec,
                        min_cycles,
                        fmin_f,
                    )
                    warned_cycles.add((seg_label, str(band)))

        if not eligible_bands and skip_invalid:
            continue

        band_power_all = _compute_psd_band_power_for_segment(
            seg_data_all, sfreq, eligible_bands if eligible_bands else band_ranges, config, logger
        )
        if band_power_all is None:
            band_power_all = {
                b: np.full((n_epochs, len(precomputed.ch_names)), np.nan) for b in (eligible_bands if eligible_bands else band_ranges)
            }
        
        epoch_band_power = [
            {band: arr[ep_idx] for band, arr in band_power_all.items()}
            for ep_idx in range(n_epochs)
        ]
        
        if n_jobs != 1:
            seg_records = Parallel(n_jobs=n_jobs)(
                delayed(_process_asymmetry_epoch)(
                    ep_idx,
                    power,
                    valid_pairs,
                    seg_label,
                    eps=eps,
                    emit_activation_convention=emit_activation_convention,
                    activation_bands=activation_bands,
                )
                for ep_idx, power in enumerate(epoch_band_power)
            )
        else:
            seg_records = [
                _process_asymmetry_epoch(
                    ep_idx,
                    power,
                    valid_pairs,
                    seg_label,
                    eps=eps,
                    emit_activation_convention=emit_activation_convention,
                    activation_bands=activation_bands,
                )
                for ep_idx, power in enumerate(epoch_band_power)
            ]
        
        for ep_idx, seg_rec in enumerate(seg_records):
            records[ep_idx].update(seg_rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    df.attrs["precomputed_feature_family"] = str(getattr(precomputed, "feature_family", "") or "")
    df.attrs["precomputed_spatial_transform"] = str(getattr(precomputed, "spatial_transform", "") or "")
    df.attrs["precomputed_evoked_subtracted"] = bool(getattr(precomputed, "evoked_subtracted", False))
    df.attrs["precomputed_evoked_subtracted_conditionwise"] = bool(
        getattr(precomputed, "evoked_subtracted_conditionwise", False)
    )
    return df, list(df.columns)


__all__ = [
    "validate_window_masks",
    "extract_band_ratios_from_precomputed",
    "extract_asymmetry_from_precomputed",
]
