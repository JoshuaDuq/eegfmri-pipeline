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
from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.utils.analysis.windowing import get_segment_masks


def _get_segment_masks(precomputed: "PrecomputedData") -> Dict[str, np.ndarray]:
    """Derive ramp/plateau/offset masks based on times and config.
    
    Wrapper around the canonical get_segment_masks in utils/analysis/windowing.py.
    """
    return get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)

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
) -> Dict[str, float]:
    """Process dynamics for a single epoch (parallel worker)."""
    record: Dict[str, float] = {}

    segment_label = "plateau"
    
    # 1. GFP Dynamics
    if gfp is not None and gfp.size > 0:
        gfp_trace = gfp[ep_idx]
        gfp_active = gfp_trace[active_mask] if not isinstance(active_mask, slice) else gfp_trace
        
        if gfp_active.size:
            record["gfp_mean_active"] = float(np.nanmean(gfp_active))
            record["gfp_std_active"] = float(np.nanstd(gfp_active))
            record["gfp_fano_active"] = float(
                np.nanvar(gfp_active) / (np.nanmean(gfp_active) + 1e-12)
            )

            record[NamingSchema.build("gfp", segment_label, "broadband", "global", "mean")] = record["gfp_mean_active"]
            record[NamingSchema.build("gfp", segment_label, "broadband", "global", "std")] = record["gfp_std_active"]
            record[NamingSchema.build("gfp", segment_label, "broadband", "global", "fano")] = record["gfp_fano_active"]

            if gfp_active.size > 1:
                times_active = times[active_mask] if not isinstance(active_mask, slice) else times
                valid = np.isfinite(gfp_active)
                if np.sum(valid) > 2:
                    slope, _ = np.polyfit(times_active[valid], gfp_active[valid], 1)
                    record["gfp_slope_active"] = float(slope)
                    record[NamingSchema.build("gfp", segment_label, "broadband", "global", "slope")] = record["gfp_slope_active"]
    
    if gfp_band:
        for band_name, gfp_b in gfp_band.items():
            gfp_band_active = gfp_b[ep_idx][active_mask] if not isinstance(active_mask, slice) else gfp_b[ep_idx]
            if gfp_band_active.size:
                record[f"gfp_{band_name}_mean_active"] = float(np.nanmean(gfp_band_active))
                record[f"gfp_{band_name}_std_active"] = float(np.nanstd(gfp_band_active))

                record[NamingSchema.build("gfp", segment_label, band_name, "global", "mean")] = record[f"gfp_{band_name}_mean_active"]
                record[NamingSchema.build("gfp", segment_label, band_name, "global", "std")] = record[f"gfp_{band_name}_std_active"]

    # 2. Band Power Dynamics & Bursts
    for band, bd in band_data.items():
        power = bd.power[ep_idx]
        
        # Baseline power for logratio
        if baseline_mask is not None and np.any(baseline_mask):
            baseline_power = np.nanmean(power[:, baseline_mask], axis=1)
        else:
            baseline_power = np.full(power.shape[0], np.nan)
        
        active_power = power[:, active_mask] if not isinstance(active_mask, slice) else power
        active_mean = np.nanmean(active_power, axis=1)
        
        record[f"dynamics_{band}_mean_active"] = float(np.nanmean(active_mean))
        record[NamingSchema.build("dynamics", segment_label, band, "global", "mean_active")] = record[f"dynamics_{band}_mean_active"]
        if np.any(np.isfinite(baseline_power)):
            record[f"dynamics_{band}_logratio"] = float(
                np.nanmean(np.log10(active_mean / (baseline_power + 1e-12)))
            )
            record[NamingSchema.build("dynamics", segment_label, band, "global", "logratio")] = record[f"dynamics_{band}_logratio"]

        # Trial-level variability (Fano) across time
        record[f"dynamics_{band}_power_fano"] = float(
            np.nanvar(active_power) / (np.nanmean(active_power) + 1e-12)
        )
        record[NamingSchema.build("dynamics", segment_label, band, "global", "power_fano")] = record[f"dynamics_{band}_power_fano"]

        # Burst detection on envelopes (simple percentile threshold)
        env = bd.envelope[ep_idx]
        env_active = env[:, active_mask] if not isinstance(active_mask, slice) else env
        
        if env_active.size > 0:
            thresh = np.nanpercentile(env_active, 75)
            burst_mask = env_active > thresh
            durations: List[float] = []
            amps: List[float] = []
            
            for ch_idx in range(env_active.shape[0]):
                ch_mask = burst_mask[ch_idx]
                if ch_mask.size == 0:
                    continue
                    
                # Find contiguous regions
                diff = np.diff(np.concatenate([[0], ch_mask.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                for s, e in zip(starts, ends):
                    dur = (e - s) / sfreq
                    durations.append(dur)
                    amps.append(float(np.nanmax(env_active[ch_idx, s:e]) if e > s else np.nan))
                    
            if durations:
                record[f"dynamics_{band}_burst_rate"] = float(len(durations))
                record[f"dynamics_{band}_burst_mean_duration"] = float(np.nanmean(durations))
                record[f"dynamics_{band}_burst_mean_amplitude"] = float(np.nanmean(amps))
            else:
                record[f"dynamics_{band}_burst_rate"] = 0.0
                record[f"dynamics_{band}_burst_mean_duration"] = np.nan
                record[f"dynamics_{band}_burst_mean_amplitude"] = np.nan

            record[NamingSchema.build("dynamics", segment_label, band, "global", "burst_rate")] = record[f"dynamics_{band}_burst_rate"]
            record[NamingSchema.build("dynamics", segment_label, band, "global", "burst_mean_duration")] = record[f"dynamics_{band}_burst_mean_duration"]
            record[NamingSchema.build("dynamics", segment_label, band, "global", "burst_mean_amplitude")] = record[f"dynamics_{band}_burst_mean_amplitude"]
        
        # Ramp-specific burst metrics for gamma
        if band.lower() == "gamma":
            ramp_mask = seg_masks.get("ramp")
            if ramp_mask is not None and np.any(ramp_mask):
                env_ramp = env[:, ramp_mask]
                thresh_ramp = np.nanpercentile(env_ramp, 75) if env_ramp.size > 0 else np.nan
                
                if env_ramp.size > 0 and np.isfinite(thresh_ramp):
                    burst_mask_ramp = env_ramp > thresh_ramp
                    durations_r: List[float] = []
                    amps_r: List[float] = []
                    
                    for ch_idx in range(env_ramp.shape[0]):
                        ch_mask = burst_mask_ramp[ch_idx]
                        if ch_mask.size == 0:
                            continue
                        diff = np.diff(np.concatenate([[0], ch_mask.astype(int), [0]]))
                        starts = np.where(diff == 1)[0]
                        ends = np.where(diff == -1)[0]
                        for s, e in zip(starts, ends):
                            dur = (e - s) / sfreq
                            durations_r.append(dur)
                            amps_r.append(float(np.nanmax(env_ramp[ch_idx, s:e]) if e > s else np.nan))
                            
                    record["dynamics_gamma_ramp_burst_rate"] = float(len(durations_r)) if durations_r else 0.0
                    record["dynamics_gamma_ramp_burst_mean_duration"] = float(np.nanmean(durations_r)) if durations_r else np.nan
                    record["dynamics_gamma_ramp_burst_mean_amplitude"] = float(np.nanmean(amps_r)) if amps_r else np.nan

                    record[NamingSchema.build("dynamics", "ramp", "gamma", "global", "burst_rate")] = record["dynamics_gamma_ramp_burst_rate"]
                    record[NamingSchema.build("dynamics", "ramp", "gamma", "global", "burst_mean_duration")] = record["dynamics_gamma_ramp_burst_mean_duration"]
                    record[NamingSchema.build("dynamics", "ramp", "gamma", "global", "burst_mean_amplitude")] = record["dynamics_gamma_ramp_burst_mean_amplitude"]

    return record


def extract_dynamics_from_precomputed(precomputed: "PrecomputedData", n_jobs: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    """Compute dynamics metrics from GFP, band power/envelopes, and burst structure."""
    if precomputed.windows is None:
        return pd.DataFrame(), []

    active_mask = getattr(precomputed.windows, "active_mask", None)
    baseline_mask = getattr(precomputed.windows, "baseline_mask", None)
    if active_mask is None or not np.any(active_mask):
        active_mask = slice(None)

    # Pre-calculate segment masks for burst analysis (e.g. gamma ramp)
    seg_masks = _get_segment_masks(precomputed)
    n_epochs = precomputed.data.shape[0]
    
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
            )
            for ep_idx in range(n_epochs)
        ]

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    return df, list(df.columns)
