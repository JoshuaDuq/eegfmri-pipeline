"""
Complexity Feature Extraction
=============================

Nonlinear dynamics features for EEG analysis:
- Lempel-Ziv Complexity (LZC): Algorithmic complexity
- Permutation Entropy (PE): Ordinal pattern complexity

Features are computed per user-defined time windows and frequency bands
using band-limited amplitude envelopes from precomputed data.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_permutation_entropy as _permutation_entropy,
    compute_lempel_ziv_complexity as _lempel_ziv_complexity,
)

# --- Helpers ---

def _extract_params(config: Any) -> Dict[str, Any]:
    """Extract complexity parameters from config."""
    return {
        "pe_order": int(config.get("feature_engineering.complexity.pe_order", 3)),
        "pe_delay": int(config.get("feature_engineering.complexity.pe_delay", 1)),
        "target_hz": float(config.get("feature_engineering.complexity.target_hz", 100.0)),
        "target_n_samples": int(config.get("feature_engineering.complexity.target_n_samples", 500)),
        "zscore": bool(config.get("feature_engineering.complexity.zscore", True)),
    }


def _resample_1d(x: np.ndarray, *, target_n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if target_n <= 1 or x.size <= 1:
        return x
    if x.size == target_n:
        return x
    try:
        # Prefer polyphase resampling (anti-aliasing) when available.
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(int(x.size), int(target_n))
        up = int(target_n // g)
        down = int(x.size // g)
        y = resample_poly(x, up=up, down=down, window=("kaiser", 5.0), padtype="line")
        if y.size == target_n:
            return y.astype(float, copy=False)
    except Exception:
        pass

    xp = np.linspace(0.0, 1.0, x.size)
    xq = np.linspace(0.0, 1.0, target_n)
    return np.interp(xq, xp, x)


def _standardize_trace(x: np.ndarray, *, zscore: bool) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not zscore:
        return x
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return x
    mu = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return x - mu
    return (x - mu) / sd

# =============================================================================
# Precomputed Data Extractors (Moved from pipeline.py)
# =============================================================================

def _process_complexity_epoch(
    ep_idx: int,
    band_data: Dict[str, Any],
    ch_names: List[str],
    segment_mask: np.ndarray,
    segment_name: str,
    params: Dict[str, Any],
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
    sfreq: float,
) -> Dict[str, float]:
    """Process complexity for a single epoch (parallel worker)."""
    record: Dict[str, float] = {}
    for band, bd in band_data.items():
        env = bd.envelope[ep_idx]
        env_seg = env[:, segment_mask] if not isinstance(segment_mask, slice) else env
        
        if env_seg.shape[1] < 10:
            continue

        n_ch = len(ch_names)
        lzc_ch = np.full(n_ch, np.nan)
        pe_ch = np.full(n_ch, np.nan)
        
        for ch_idx, ch_name in enumerate(ch_names):
            trace = env_seg[ch_idx]

            # Complexity metrics are sensitive to sampling rate and window length.
            # Standardize by resampling to a fixed target length and (optionally) z-scoring.
            target_hz = float(params.get("target_hz", 100.0))
            target_n = int(params.get("target_n_samples", 500))
            if np.isfinite(sfreq) and sfreq > 0 and np.isfinite(target_hz) and target_hz > 0:
                duration_sec = float(trace.size) / float(sfreq)
                target_n = max(10, int(round(duration_sec * target_hz)))
            target_n = max(10, int(target_n))

            trace_rs = _resample_1d(trace, target_n=target_n)
            trace_rs = _standardize_trace(trace_rs, zscore=bool(params.get("zscore", True)))
            
            try:
                lzc = _lempel_ziv_complexity(trace_rs)
            except (ValueError, RuntimeError):
                lzc = np.nan
            
            try:
                pe = _permutation_entropy(
                    trace_rs,
                    order=params.get("pe_order", 3),
                    delay=params.get("pe_delay", 1),
                )
            except (ValueError, RuntimeError):
                pe = np.nan

            lzc_ch[ch_idx] = lzc
            pe_ch[ch_idx] = pe
            
            if "channels" in spatial_modes:
                col_lzc = NamingSchema.build("comp", segment_name, band, "ch", "lzc", channel=ch_name)
                col_pe = NamingSchema.build("comp", segment_name, band, "ch", "pe", channel=ch_name)
                record[col_lzc] = float(lzc)
                record[col_pe] = float(pe)
            
        if "roi" in spatial_modes and roi_map:
            for roi_name, ch_indices in roi_map.items():
                if not ch_indices:
                    continue
                lzc_val = float(np.nanmean(lzc_ch[ch_indices]))
                pe_val = float(np.nanmean(pe_ch[ch_indices]))
                record[NamingSchema.build("comp", segment_name, band, "roi", "lzc", channel=roi_name)] = lzc_val
                record[NamingSchema.build("comp", segment_name, band, "roi", "pe", channel=roi_name)] = pe_val

        if "global" in spatial_modes:
            record[NamingSchema.build("comp", segment_name, band, "global", "lzc")] = float(np.nanmean(lzc_ch))
            record[NamingSchema.build("comp", segment_name, band, "global", "pe")] = float(np.nanmean(pe_ch))
            
    return record


def _compute_complexity_for_segment(
    precomputed: Any,
    segment_mask: np.ndarray,
    segment_name: str,
    params: Dict[str, Any],
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
    n_jobs: int = 1,
) -> List[Dict[str, float]]:
    """Compute complexity metrics for a single segment using precomputed envelopes."""
    n_epochs = precomputed.data.shape[0]
    
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_complexity_epoch)(
                ep_idx,
                precomputed.band_data,
                precomputed.ch_names,
                segment_mask,
                segment_name,
                params,
                spatial_modes,
                roi_map,
                float(getattr(precomputed, "sfreq", np.nan)),
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_complexity_epoch(
                ep_idx,
                precomputed.band_data,
                precomputed.ch_names,
                segment_mask,
                segment_name,
                params,
                spatial_modes,
                roi_map,
                float(getattr(precomputed, "sfreq", np.nan)),
            )
            for ep_idx in range(n_epochs)
        ]
    
    return records


def extract_complexity_from_precomputed(
    precomputed: Any, # PrecomputedData
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute complexity metrics (LZC, permutation entropy) using precomputed data."""
    if precomputed.windows is None:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    segments = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    
    all_records = []
    params = _extract_params(precomputed.config)
    spatial_modes = getattr(precomputed, "spatial_modes", None) or ["roi", "global"]
    roi_map: Dict[str, List[int]] = {}
    if "roi" in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(precomputed.config)
        if roi_defs:
            roi_map = build_roi_map(precomputed.ch_names, roi_defs)
    
    for seg_label, mask in segments.items():
        if mask is not None and np.any(mask):
            seg_recs = _compute_complexity_for_segment(
                precomputed,
                mask,
                seg_label,
                params,
                spatial_modes,
                roi_map,
                n_jobs=n_jobs,
            )
            if not all_records:
                all_records = seg_recs
            else:
                for i, rec in enumerate(seg_recs):
                    all_records[i].update(rec)
    
    if not all_records or all(len(r) == 0 for r in all_records):
        return pd.DataFrame(), []

    df = pd.DataFrame(all_records)
    return df, list(df.columns)
