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
from eeg_pipeline.utils.config.loader import get_config_value, get_feature_constant


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


###################################################################
# BAND RATIO FEATURES
###################################################################

def extract_band_ratios_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract band power ratios for ALL user-defined segments."""
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

    # Get ALL segment masks from windows
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    segment_masks = get_segment_masks(precomputed.times, precomputed.windows, config)
    
    if not segment_masks:
        return pd.DataFrame(), []

    # Spatial info
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
    
    # ratio_source: 'raw' uses absolute power, 'powcorr' uses aperiodic-adjusted power
    # Using powcorr reduces confounding from broadband artifacts and aperiodic slope
    ratio_source = str(get_config_value(config, "feature_engineering.spectral.ratio_source", "raw")).strip().lower()
    if ratio_source not in {"raw", "powcorr"}:
        ratio_source = "raw"
    
    n_epochs = precomputed.data.shape[0]
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
    
    # Iterate over ALL segments
    for seg_label, seg_mask in segment_masks.items():
        if seg_mask is None or not np.any(seg_mask):
            continue
            
        for ep_idx in range(n_epochs):
            rec = records[ep_idx]
            
            # Compute band power for this segment
            band_power_ch = {}
            for band, bd in precomputed.band_data.items():
                if ratio_source == "powcorr" and hasattr(bd, 'powcorr') and bd.powcorr is not None:
                    # Use aperiodic-corrected power (ratio relative to 1/f fit)
                    p = bd.powcorr[ep_idx]
                else:
                    # Use raw absolute power
                    p = bd.power[ep_idx]
                band_power_ch[band] = np.nanmean(p[:, seg_mask], axis=1)


            for num, den in pairs:
                if num not in band_power_ch or den not in band_power_ch:
                    continue
                    
                p_num = band_power_ch[num]
                p_den = band_power_ch[den]
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    r_ch = p_num / p_den
                    r_ch[p_den <= eps] = np.nan

                if include_log:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        log_r_ch = np.log(p_num + eps) - np.log(p_den + eps)
                        log_r_ch[(p_num <= eps) | (p_den <= eps)] = np.nan
                
                pair_label = f"{num}_{den}"
                
                # Channels
                if 'channels' in spatial_modes:
                    for c, ch in enumerate(precomputed.ch_names):
                        col = NamingSchema.build("ratios", seg_label, pair_label, "ch", "power_ratio", channel=ch)
                        rec[col] = float(r_ch[c])
                        if include_log:
                            col_log = NamingSchema.build("ratios", seg_label, pair_label, "ch", "log_ratio", channel=ch)
                            rec[col_log] = float(log_r_ch[c])
                
                # ROI
                if 'roi' in spatial_modes and roi_map:
                    for roi_name, idxs in roi_map.items():
                        if idxs:
                            val = np.nanmean(r_ch[idxs])
                            col = NamingSchema.build("ratios", seg_label, pair_label, "roi", "power_ratio", channel=roi_name)
                            rec[col] = float(val)
                            if include_log:
                                val_log = np.nanmean(log_r_ch[idxs])
                                col_log = NamingSchema.build("ratios", seg_label, pair_label, "roi", "log_ratio", channel=roi_name)
                                rec[col_log] = float(val_log)
                
                # Global
                if 'global' in spatial_modes:
                    val = np.nanmean(r_ch)
                    col = NamingSchema.build("ratios", seg_label, pair_label, "global", "power_ratio")
                    rec[col] = float(val)
                    if include_log:
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
    band_data: Dict[str, Any],
    valid_pairs: List[Tuple[str, str, int, int]],
    active_mask: np.ndarray,
    segment_label: str,
) -> Dict[str, float]:
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
            if pr > 0 and pl > 0:
                record[
                    NamingSchema.build(
                        "asymmetry",
                        segment_label,
                        band,
                        "chpair",
                        "logdiff",
                        channel_pair=pair,
                    )
                ] = float(np.log(pr) - np.log(pl))
    return record


def extract_asymmetry_from_precomputed(
    precomputed: Any,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract hemispheric asymmetry features for ALL user-defined segments."""
    logger = getattr(precomputed, "logger", None)
    if not precomputed.band_data:
        if logger is not None:
            logger.warning("Asymmetry: No band data available; skipping extraction.")
        return pd.DataFrame(), []

    default_pairs = [("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]
    pairs_cfg = get_config_value(precomputed.config, "feature_engineering.asymmetry.channel_pairs", None)
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

    # Get ALL segment masks from windows
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    segment_masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    
    if not segment_masks:
        return pd.DataFrame(), []

    n_epochs = precomputed.data.shape[0]
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
    
    # Iterate over ALL segments
    for seg_label, seg_mask in segment_masks.items():
        if seg_mask is None or not np.any(seg_mask):
            continue
        
        if n_jobs != 1:
            seg_records = Parallel(n_jobs=n_jobs)(
                delayed(_process_asymmetry_epoch)(
                    ep_idx,
                    precomputed.band_data,
                    valid_pairs,
                    seg_mask,
                    seg_label,
                )
                for ep_idx in range(n_epochs)
            )
        else:
            seg_records = [
                _process_asymmetry_epoch(ep_idx, precomputed.band_data, valid_pairs, seg_mask, seg_label)
                for ep_idx in range(n_epochs)
            ]
        
        # Merge segment records into main records
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
