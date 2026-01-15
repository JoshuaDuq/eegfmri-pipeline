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

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import get_segment_mask, validate_precomputed
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
        # Check if a targeted window is set and exists
        target_name = getattr(windows, "name", None)
        if target_name and target_name in windows.masks:
            mask = windows.get_mask(target_name)
            if mask is not None and np.any(mask):
                return True
        
        # Otherwise check if there's at least one non-baseline mask
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
    
    # Get PSD settings from config
    spec_cfg = config.get("feature_engineering.spectral", {}) if hasattr(config, "get") else {}
    psd_method = str(spec_cfg.get("psd_method", "multitaper")).strip().lower()
    if psd_method not in {"welch", "multitaper"}:
        psd_method = "multitaper"
    
    fmin_psd = float(spec_cfg.get("fmin", 1.0))
    fmax_psd = float(spec_cfg.get("fmax", min(80.0, sfreq / 2.0 - 0.5)))

    multitaper_adaptive = bool(spec_cfg.get("multitaper_adaptive", spec_cfg.get("psd_adaptive", False)))
    
    # Line noise exclusion
    exclude_line = bool(spec_cfg.get("exclude_line_noise", True))
    line_freqs = spec_cfg.get("line_noise_freqs", [50.0])
    try:
        line_freqs = [float(f) for f in line_freqs]
    except Exception:
        line_freqs = [50.0]
    line_width = float(spec_cfg.get("line_noise_width_hz", 1.0))
    n_harm = int(spec_cfg.get("line_noise_harmonics", 3))
    
    # Reshape to (1, n_channels, n_times) for compute_psd_bandpower
    if data.ndim == 2:
        data_3d = data[np.newaxis, :, :]
    else:
        data_3d = data
    
    result = compute_psd_bandpower(
        data_3d,
        sfreq,
        band_ranges,
        psd_method=psd_method,
        fmin=fmin_psd,
        fmax=fmax_psd,
        normalize_by_bandwidth=True,
        adaptive=multitaper_adaptive,
        exclude_line_noise=exclude_line,
        line_freqs=line_freqs if exclude_line else None,
        line_width=line_width if exclude_line else None,
        n_harmonics=n_harm if exclude_line else None,
        logger=logger,
    )
    
    if result is None:
        return None
    
    # Squeeze out epoch dimension for single-epoch 2D input
    if data.ndim == 2:
        return {band: arr[0] for band, arr in result.items()}
    return result


def extract_band_ratios_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract band power ratios for ALL user-defined segments.
    
    Uses PSD-integrated band power (scientifically valid for ratios).
    Power is bandwidth-normalized (power per Hz) for comparability across bands.
    """
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
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

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    
    logger = getattr(precomputed, "logger", None)
    windows = precomputed.windows
    target_name = getattr(windows, "name", None) if windows else None
    allow_full_epoch_fallback = bool(
        config.get("feature_engineering.windows.allow_full_epoch_fallback", False)
        if hasattr(config, "get")
        else False
    )
    
    # Always derive mask from windows - never use np.ones() blindly
    # If data is pre-cropped, the mask will naturally be all-True
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segment_masks = {target_name: mask}
        else:
            if logger:
                if allow_full_epoch_fallback:
                    logger.warning(
                        "Band ratios: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                        target_name,
                    )
                else:
                    logger.error(
                        "Band ratios: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                segment_masks = {target_name: np.ones(len(precomputed.times), dtype=bool)}
            else:
                return pd.DataFrame(), []
    else:
        segment_masks = get_segment_masks(precomputed.times, windows, config)
    
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
    
    # Get frequency bands for PSD computation
    freq_bands = get_frequency_bands(config)
    needed_bands = set()
    for num, den in pairs:
        needed_bands.add(num)
        needed_bands.add(den)
    band_ranges = {b: tuple(freq_bands[b]) for b in needed_bands if b in freq_bands}
    
    n_epochs = precomputed.data.shape[0]
    sfreq = precomputed.sfreq
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    spec_cfg = config.get("feature_engineering.spectral", {}) if hasattr(config, "get") else {}
    min_segment_sec = float(spec_cfg.get("min_segment_sec", 2.0))
    min_segment_samples = max(0, int(round(min_segment_sec * float(sfreq))))
    warned_short: set[str] = set()
    
    for seg_label, seg_mask in segment_masks.items():
        if seg_mask is None or not np.any(seg_mask):
            continue

        seg_n = int(np.sum(seg_mask))
        if min_segment_samples > 0 and seg_n < min_segment_samples:
            if logger and seg_label not in warned_short:
                logger.warning(
                    "Band ratios: segment '%s' is too short for PSD ratios (%.3fs, %d samples < %d); skipping.",
                    seg_label,
                    float(seg_n) / float(sfreq),
                    seg_n,
                    min_segment_samples,
                )
                warned_short.add(seg_label)
            continue
        
        seg_data_all = precomputed.data[:, :, seg_mask]  # (epochs, ch, time)
        band_power = _compute_psd_band_power_for_segment(
            seg_data_all, sfreq, band_ranges, config, logger
        )
        if band_power is None:
            continue

        for num, den in pairs:
            if num not in band_power or den not in band_power:
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
                        if idxs:
                            val = np.nanmean(r_ch[idxs])
                            col = NamingSchema.build(
                                "ratios", seg_label, pair_label, "roi", "power_ratio", channel=roi_name
                            )
                            rec[col] = float(val)
                            if include_log and log_r_ch is not None:
                                val_log = np.nanmean(log_r_ch[idxs])
                                col_log = NamingSchema.build(
                                    "ratios", seg_label, pair_label, "roi", "log_ratio", channel=roi_name
                                )
                                rec[col_log] = float(val_log)

                if "global" in spatial_modes:
                    val = np.nanmean(r_ch)
                    col = NamingSchema.build("ratios", seg_label, pair_label, "global", "power_ratio")
                    rec[col] = float(val)
                    if include_log and log_r_ch is not None:
                        val_log = np.nanmean(log_r_ch)
                        col_log = NamingSchema.build("ratios", seg_label, pair_label, "global", "log_ratio")
                        rec[col_log] = float(val_log)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


###################################################################
# ASYMMETRY FEATURES
###################################################################

def _process_asymmetry_epoch(
    ep_idx: int,
    band_power: Dict[str, np.ndarray],
    valid_pairs: List[Tuple[str, str, int, int]],
    segment_label: str,
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
        for l_name, r_name, l_idx, r_idx in valid_pairs:
            pl, pr = p_mean[l_idx], p_mean[r_idx]
            pair = f"{l_name}-{r_name}"
            
            # Use NaN when asymmetry is undefined (tiny denominator)
            denom = pr + pl
            if denom > 1e-12 and np.isfinite(pl) and np.isfinite(pr):
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
            if pr > 1e-12 and pl > 1e-12 and np.isfinite(pl) and np.isfinite(pr):
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

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    
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
            if logger:
                if allow_full_epoch_fallback:
                    logger.warning(
                        "Asymmetry: targeted window '%s' has no valid mask; using full epoch (allow_full_epoch_fallback=True).",
                        target_name,
                    )
                else:
                    logger.error(
                        "Asymmetry: targeted window '%s' has no valid mask; skipping (allow_full_epoch_fallback=False).",
                        target_name,
                    )
            if allow_full_epoch_fallback:
                segment_masks = {target_name: np.ones(len(precomputed.times), dtype=bool)}
            else:
                return pd.DataFrame(), []
    else:
        segment_masks = get_segment_masks(precomputed.times, windows, config)
    
    if not segment_masks:
        return pd.DataFrame(), []

    # Get frequency bands for PSD computation
    freq_bands = get_frequency_bands(config)
    band_ranges = {b: tuple(freq_bands[b]) for b in freq_bands}

    n_epochs = precomputed.data.shape[0]
    sfreq = precomputed.sfreq
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    spec_cfg = config.get("feature_engineering.spectral", {}) if hasattr(config, "get") else {}
    min_segment_sec = float(spec_cfg.get("min_segment_sec", 2.0))
    min_segment_samples = max(0, int(round(min_segment_sec * float(sfreq))))
    warned_short: set[str] = set()
    
    for seg_label, seg_mask in segment_masks.items():
        if seg_mask is None or not np.any(seg_mask):
            continue

        seg_n = int(np.sum(seg_mask))
        if min_segment_samples > 0 and seg_n < min_segment_samples:
            if logger and seg_label not in warned_short:
                logger.warning(
                    "Asymmetry: segment '%s' is too short for PSD asymmetry (%.3fs, %d samples < %d); skipping.",
                    seg_label,
                    float(seg_n) / float(sfreq),
                    seg_n,
                    min_segment_samples,
                )
                warned_short.add(seg_label)
            continue
        
        seg_data_all = precomputed.data[:, :, seg_mask]
        band_power_all = _compute_psd_band_power_for_segment(
            seg_data_all, sfreq, band_ranges, config, logger
        )
        if band_power_all is None:
            band_power_all = {
                b: np.full((n_epochs, len(precomputed.ch_names)), np.nan) for b in band_ranges
            }
        
        # Process asymmetry (parallelizable)
        if n_jobs != 1:
            seg_records = Parallel(n_jobs=n_jobs)(
                delayed(_process_asymmetry_epoch)(
                    ep_idx,
                    {band: arr[ep_idx] for band, arr in band_power_all.items()},
                    valid_pairs,
                    seg_label,
                )
                for ep_idx in range(n_epochs)
            )
        else:
            seg_records = [
                _process_asymmetry_epoch(
                    ep_idx,
                    {band: arr[ep_idx] for band, arr in band_power_all.items()},
                    valid_pairs,
                    seg_label,
                )
                for ep_idx in range(n_epochs)
            ]
        
        for ep_idx, seg_rec in enumerate(seg_records):
            records[ep_idx].update(seg_rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = [
    "validate_window_masks",
    "extract_band_ratios_from_precomputed",
    "extract_asymmetry_from_precomputed",
]
