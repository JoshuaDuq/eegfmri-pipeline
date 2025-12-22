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
    }

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
            
            try:
                lzc = _lempel_ziv_complexity(trace)
            except (ValueError, RuntimeError):
                lzc = np.nan
            
            try:
                pe = _permutation_entropy(trace, order=params.get("pe_order", 3), delay=params.get("pe_delay", 1))
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
