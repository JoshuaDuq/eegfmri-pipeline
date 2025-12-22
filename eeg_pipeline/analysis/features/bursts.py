"""
Burst Feature Extraction
========================

Detects oscillatory bursts using band-limited amplitude envelopes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.windowing import get_segment_masks


def _extract_burst_metrics(
    trace: np.ndarray,
    sfreq: float,
    threshold: float,
    min_samples: int,
) -> Dict[str, float]:
    above = np.asarray(trace) > threshold
    n_samples = int(trace.size)
    duration_sec = float(n_samples / sfreq) if sfreq > 0 else np.nan

    if n_samples == 0 or not np.any(above):
        return {
            "count": 0.0,
            "rate": 0.0 if np.isfinite(duration_sec) and duration_sec > 0 else np.nan,
            "duration_mean": np.nan,
            "amp_mean": np.nan,
            "fraction": 0.0,
        }

    diff = np.diff(above.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)

    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [n_samples]

    durations = np.array([e - s for s, e in zip(starts, ends)], dtype=int)
    keep = durations >= min_samples
    if not np.any(keep):
        return {
            "count": 0.0,
            "rate": 0.0 if np.isfinite(duration_sec) and duration_sec > 0 else np.nan,
            "duration_mean": np.nan,
            "amp_mean": np.nan,
            "fraction": float(np.mean(above)),
        }

    kept_starts = [s for s, k in zip(starts, keep) if k]
    kept_ends = [e for e, k in zip(ends, keep) if k]
    kept_durations = durations[keep]

    peak_amps = []
    for s, e in zip(kept_starts, kept_ends):
        peak_amps.append(float(np.nanmax(trace[s:e])))

    count = float(len(kept_durations))
    dur_mean = float(np.mean(kept_durations) / sfreq) if sfreq > 0 else np.nan
    amp_mean = float(np.mean(peak_amps)) if peak_amps else np.nan
    rate = float(count / duration_sec) if np.isfinite(duration_sec) and duration_sec > 0 else np.nan
    fraction = float(np.mean(above))

    return {
        "count": count,
        "rate": rate,
        "duration_mean": dur_mean,
        "amp_mean": amp_mean,
        "fraction": fraction,
    }


def extract_burst_features(
    ctx: Any,  # FeatureContext
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    precomputed = getattr(ctx, "precomputed", None)
    if precomputed is None:
        ctx.logger.warning("Bursts: missing precomputed intermediates; skipping extraction.")
        return pd.DataFrame(), []

    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        logger = getattr(precomputed, "logger", None)
        if logger is not None:
            logger.warning("Bursts: %s; skipping extraction.", err_msg)
        return pd.DataFrame(), []

    cfg = ctx.config if hasattr(ctx, "config") else precomputed.config
    burst_cfg = cfg.get("feature_engineering.bursts", {}) if hasattr(cfg, "get") else {}
    burst_bands = burst_cfg.get("bands") or bands
    threshold_z = float(burst_cfg.get("threshold_z", 2.0))
    min_duration_ms = float(burst_cfg.get("min_duration_ms", 50.0))

    sfreq = float(getattr(precomputed, "sfreq", np.nan))
    min_samples = max(1, int(round(min_duration_ms * sfreq / 1000.0))) if sfreq > 0 else 1

    baseline_mask = precomputed.windows.get_mask("baseline")
    if baseline_mask is None or not np.any(baseline_mask):
        ctx.logger.warning("Bursts: baseline window missing; skipping extraction.")
        return pd.DataFrame(), []

    masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    segments = [k for k in masks.keys() if k != "baseline"]
    if not segments:
        return pd.DataFrame(), []

    spatial_modes = getattr(ctx, "spatial_modes", ["roi", "global"])
    roi_map = {}
    if "roi" in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(cfg)
        if roi_defs:
            roi_map = build_roi_map(precomputed.ch_names, roi_defs)

    n_epochs = precomputed.data.shape[0]
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]

    for band in burst_bands:
        if band not in precomputed.band_data:
            continue
        env = precomputed.band_data[band].envelope

        baseline_env = env[:, :, baseline_mask]
        base_mean = np.nanmean(baseline_env, axis=2)
        base_std = np.nanstd(baseline_env, axis=2)
        thresholds = base_mean + (threshold_z * base_std)

        for seg_name in segments:
            seg_mask = masks.get(seg_name)
            if seg_mask is None or not np.any(seg_mask):
                continue
            seg_env = env[:, :, seg_mask]
            if seg_env.shape[-1] < min_samples:
                continue

            for ep_idx in range(n_epochs):
                rec = records[ep_idx]

                if "channels" in spatial_modes:
                    for ch_idx, ch_name in enumerate(precomputed.ch_names):
                        metrics = _extract_burst_metrics(
                            seg_env[ep_idx, ch_idx],
                            sfreq,
                            thresholds[ep_idx, ch_idx],
                            min_samples,
                        )
                        for stat, val in metrics.items():
                            col = NamingSchema.build("bursts", seg_name, band, "ch", stat, channel=ch_name)
                            rec[col] = float(val)

                if "roi" in spatial_modes and roi_map:
                    for roi_name, ch_indices in roi_map.items():
                        if not ch_indices:
                            continue
                        roi_trace = np.nanmean(seg_env[ep_idx, ch_indices], axis=0)
                        roi_thr = float(np.nanmean(thresholds[ep_idx, ch_indices]))
                        metrics = _extract_burst_metrics(roi_trace, sfreq, roi_thr, min_samples)
                        for stat, val in metrics.items():
                            col = NamingSchema.build("bursts", seg_name, band, "roi", stat, channel=roi_name)
                            rec[col] = float(val)

                if "global" in spatial_modes:
                    glob_trace = np.nanmean(seg_env[ep_idx], axis=0)
                    glob_thr = float(np.nanmean(thresholds[ep_idx]))
                    metrics = _extract_burst_metrics(glob_trace, sfreq, glob_thr, min_samples)
                    for stat, val in metrics.items():
                        col = NamingSchema.build("bursts", seg_name, band, "global", stat)
                        rec[col] = float(val)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    return df, list(df.columns)


__all__ = ["extract_burst_features"]
