"""
Extra Precomputed Feature Extractors.

Consolidated module for smaller/specialized precomputed feature extractors:
- GFP (Global Field Power)
- Temporal (variance, RMS, line length)
- Band ratios
- Hemispheric asymmetry
- ROI aggregations
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import get_segment_mask, SEGMENT_PLATEAU, validate_precomputed
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
# GFP FEATURES
###################################################################

def extract_gfp_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    logger = getattr(precomputed, "logger", None)
    if precomputed.data is None or precomputed.windows is None:
        if logger is not None:
            logger.warning("GFP: No data or windows available; skipping extraction.")
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
            rec[NamingSchema.build("gfp", seg, "broadband", "global", "mean")] = float(
                np.nanmean(vals)
            )
            rec[NamingSchema.build("gfp", seg, "broadband", "global", "max")] = float(
                np.nanmax(vals)
            )
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


###################################################################
# TEMPORAL FEATURES
###################################################################

def extract_temporal_features_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    logger = getattr(precomputed, "logger", None)
    if precomputed.data is None or precomputed.windows is None:
        if logger is not None:
            logger.warning("Temporal: No data or windows available; skipping extraction.")
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
            var_mean = float(np.nanmean(np.nanvar(seg_x, axis=1)))
            rms_mean = float(np.nanmean(np.sqrt(np.nanmean(seg_x**2, axis=1))))
            ll_mean = (
                float(np.nanmean(np.nanmean(np.abs(np.diff(seg_x, axis=1)), axis=1)))
                if seg_x.shape[1] > 1
                else np.nan
            )
            rec[NamingSchema.build("temporal", seg, "broadband", "global", "var_mean")] = var_mean
            rec[NamingSchema.build("temporal", seg, "broadband", "global", "rms_mean")] = rms_mean
            rec[
                NamingSchema.build("temporal", seg, "broadband", "global", "line_length_mean")
            ] = ll_mean
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


###################################################################
# BAND RATIO FEATURES
###################################################################

def extract_band_ratios_from_precomputed(
    precomputed: PrecomputedData,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
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
            rec[
                NamingSchema.build(
                    "ratios",
                    "plateau",
                    f"{num}_{den}",
                    "global",
                    "power_ratio",
                )
            ] = ratio
        records.append(rec)

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

    segment_label = str(
        get_config_value(precomputed.config, "feature_engineering.asymmetry.segment_label", "plateau")
    )

    ch_map = {name: i for i, name in enumerate(precomputed.ch_names)}
    valid_pairs = [(l, r, ch_map[l], ch_map[r]) for l, r in pairs if l in ch_map and r in ch_map]
    if not valid_pairs:
        return pd.DataFrame(), []

    mask = get_segment_mask(precomputed.windows, SEGMENT_PLATEAU)
    if mask is None or not np.any(mask):
        mask = get_segment_mask(precomputed.windows, "active")
    if mask is None or not np.any(mask):
        mask = slice(None)

    n_epochs = precomputed.data.shape[0]
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_asymmetry_epoch)(
                ep_idx,
                precomputed.band_data,
                valid_pairs,
                mask,
                segment_label,
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_asymmetry_epoch(ep_idx, precomputed.band_data, valid_pairs, mask, segment_label)
            for ep_idx in range(n_epochs)
        ]

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    return df, list(df.columns)


###################################################################
# ROI FEATURES
###################################################################

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


def extract_roi_features_from_precomputed(
    precomputed: PrecomputedData,
    bands: List[str],
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("ROI: %s; skipping extraction.", err_msg)
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
            power = precomputed.band_data[band].power[ep_idx]
            base_ch = np.nanmean(power[:, baseline_mask], axis=1)
            for roi_name, idxs in roi_to_idx.items():
                base = float(np.nanmean(base_ch[idxs]))
                base = base if np.isfinite(base) and base > epsilon else np.nan
                for seg, mask in segments.items():
                    if mask is None or not np.any(mask):
                        continue
                    seg_ch = np.nanmean(power[:, mask], axis=1)
                    seg_val = float(np.nanmean(seg_ch[idxs]))
                    if np.isfinite(base) and base > epsilon and np.isfinite(seg_val):
                        logratio = float(np.log10(seg_val / base))
                    else:
                        logratio = np.nan
                    rec[
                        NamingSchema.build(
                            "roi",
                            seg,
                            band,
                            "global",
                            f"{roi_name}_logratio_mean",
                        )
                    ] = logratio
        records.append(rec)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = [
    "validate_window_masks",
    "extract_gfp_from_precomputed",
    "extract_temporal_features_from_precomputed",
    "extract_band_ratios_from_precomputed",
    "extract_asymmetry_from_precomputed",
    "extract_roi_features_from_precomputed",
]
