"""
Dynamics Feature Extraction
===========================

Extracts dynamic features including:
- Global Field Power (GFP) metrics
- Temporal dynamics of band power
- Burst detection and properties
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from joblib import Parallel, delayed

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import (
    EPSILON_STD,
    validate_precomputed,
)
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.config.loader import get_config_value

DEFAULT_BURST_PERCENTILE = 75

def _process_single_epoch_dynamics(
    ep_idx: int,
    gfp: np.ndarray,
    gfp_band: Dict[str, np.ndarray],
    band_data: Dict[str, Any],
    times: np.ndarray,
    sfreq: float,
    active_mask: np.ndarray,
    baseline_mask: np.ndarray,
    seg_masks: Dict[str, np.ndarray],
    ch_names: List[str],
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
    segment_label: str = "active",
    burst_percentile: float = DEFAULT_BURST_PERCENTILE,
) -> Dict[str, float]:
    """Process dynamics for a single epoch (parallel worker)."""
    record: Dict[str, float] = {}
    
    # 1. GFP Dynamics (Always Global)
    if gfp is not None and gfp.size > 0:
        gfp_trace = gfp[ep_idx]
        gfp_active = gfp_trace[active_mask] if not isinstance(active_mask, slice) else gfp_trace
        
        if gfp_active.size:
            if 'global' in spatial_modes:
                record[NamingSchema.build("dynamics", segment_label, "broadband", "global", "gfp_mean")] = float(np.nanmean(gfp_active))
                record[NamingSchema.build("dynamics", segment_label, "broadband", "global", "gfp_std")] = float(np.nanstd(gfp_active))
                record[NamingSchema.build("dynamics", segment_label, "broadband", "global", "gfp_fano")] = float(
                    np.nanvar(gfp_active) / (np.nanmean(gfp_active) + EPSILON_STD)
                )

                if gfp_active.size > 1:
                    times_active = times[active_mask] if not isinstance(active_mask, slice) else times
                    valid = np.isfinite(gfp_active)
                    if np.sum(valid) > 2:
                        slope, _ = np.polyfit(times_active[valid], gfp_active[valid], 1)
                        record[NamingSchema.build("dynamics", segment_label, "broadband", "global", "gfp_slope")] = float(slope)
    
    if gfp_band and 'global' in spatial_modes:
        for band_name, gfp_b in gfp_band.items():
            gfp_band_active = gfp_b[ep_idx][active_mask] if not isinstance(active_mask, slice) else gfp_b[ep_idx]
            if gfp_band_active.size:
                record[NamingSchema.build("dynamics", segment_label, band_name, "global", "gfp_mean")] = float(np.nanmean(gfp_band_active))
                record[NamingSchema.build("dynamics", segment_label, band_name, "global", "gfp_std")] = float(np.nanstd(gfp_band_active))

    # 2. Band Power Dynamics & Bursts
    for band, bd in band_data.items():
        power = bd.power[ep_idx]
        
        # Baseline power for logratio
        if baseline_mask is not None and np.any(baseline_mask):
            baseline_power = np.nanmean(power[:, baseline_mask], axis=1)
        else:
            baseline_power = np.full(power.shape[0], np.nan)
        
        active_power = power[:, active_mask] if not isinstance(active_mask, slice) else power
        # Mean across time per channel -> (n_ch,)
        ch_active_mean = np.nanmean(active_power, axis=1)
        
        # Logratio per channel
        with np.errstate(divide='ignore', invalid='ignore'):
            ch_logratio = np.log10(ch_active_mean / (baseline_power + EPSILON_STD))
            
        # Variability per channel
        ch_fano = np.nanvar(active_power, axis=1) / (np.nanmean(active_power, axis=1) + EPSILON_STD)

        # Burst detection on envelopes
        env = bd.envelope[ep_idx]
        env_active = env[:, active_mask] if not isinstance(active_mask, slice) else env
        
        n_ch = env_active.shape[0]
        ch_burst_rate = np.zeros(n_ch)
        ch_burst_dur = np.full(n_ch, np.nan)
        ch_burst_amp = np.full(n_ch, np.nan)
        
        if env_active.size > 0:
            thresh = np.nanpercentile(env_active, burst_percentile)
            burst_mask = env_active > thresh
            for c in range(n_ch):
                c_mask = burst_mask[c]
                if c_mask.size == 0 or not np.any(c_mask): continue
                diff = np.diff(np.concatenate([[0], c_mask.astype(int), [0]]))
                starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
                durs = (ends - starts) / sfreq
                ch_burst_rate[c] = len(durs)
                if durs.size > 0:
                    ch_burst_dur[c] = np.mean(durs)
                    ch_burst_amp[c] = np.mean([np.nanmax(env_active[c, s:e]) for s, e in zip(starts, ends)])

        metrics = {
            "mean_active": ch_active_mean,
            "logratio": ch_logratio,
            "power_fano": ch_fano,
            "burst_rate": ch_burst_rate,
            "burst_mean_duration": ch_burst_dur,
            "burst_mean_amplitude": ch_burst_amp,
        }
        
        for k, vals in metrics.items():
            # Channels
            if 'channels' in spatial_modes:
                for c, ch in enumerate(ch_names):
                    col = NamingSchema.build("dynamics", segment_label, band, "ch", k, channel=ch)
                    record[col] = float(vals[c])

            # ROI
            if 'roi' in spatial_modes and roi_map:
                for roi_name, idxs in roi_map.items():
                    if idxs:
                        val = np.nanmean(vals[idxs])
                        col = NamingSchema.build("dynamics", segment_label, band, "roi", f"{k}_mean", channel=roi_name)
                        record[col] = float(val)
            
            # Global
            if 'global' in spatial_modes:
                val = np.nanmean(vals)
                col = NamingSchema.build("dynamics", segment_label, band, "global", f"{k}_mean")
                record[col] = float(val)

    return record


def extract_dynamics_from_precomputed(precomputed: "PrecomputedData", n_jobs: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    """Compute dynamics metrics from GFP, band power/envelopes, and burst structure."""
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        return pd.DataFrame(), []

    ctx_name = getattr(precomputed.windows, "name", "active") or "active"
    active_mask = precomputed.windows.get_mask(ctx_name)
    baseline_mask = precomputed.windows.get_mask("baseline")
    if active_mask is None or (isinstance(active_mask, np.ndarray) and not np.any(active_mask)):
        active_mask = precomputed.windows.active_mask
    if active_mask is None or (isinstance(active_mask, np.ndarray) and not np.any(active_mask)):
        active_mask = slice(None)

    # Pre-calculate segment masks for burst analysis (e.g. gamma ramp)
    seg_masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    n_epochs = precomputed.data.shape[0]
    ch_names = precomputed.ch_names
    
    # Spatial info
    spatial_modes = getattr(precomputed, 'spatial_modes', ['roi', 'global'])
    roi_map = {}
    if 'roi' in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(precomputed.config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)
            
    burst_percentile = float(get_config_value(
        precomputed.config, "feature_engineering.dynamics.burst_percentile", DEFAULT_BURST_PERCENTILE
    ))
    
    seg_label = getattr(precomputed.windows, "name", "active") or "active"

    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_epoch_dynamics)(
                ep_idx,
                precomputed.gfp,
                precomputed.gfp_band,
                precomputed.band_data,
                precomputed.times,
                precomputed.sfreq,
                active_mask,
                baseline_mask,
                seg_masks,
                ch_names,
                spatial_modes,
                roi_map,
                seg_label,
                burst_percentile,
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_single_epoch_dynamics(
                ep_idx,
                precomputed.gfp,
                precomputed.gfp_band,
                precomputed.band_data,
                precomputed.times,
                precomputed.sfreq,
                active_mask,
                baseline_mask,
                seg_masks,
                ch_names,
                spatial_modes,
                roi_map,
                seg_label,
                burst_percentile,
            )
            for ep_idx in range(n_epochs)
        ]

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    return df, list(df.columns)
