"""
Complexity Feature Extraction (Dynamics)
=========================================

Nonlinear dynamics features for EEG analysis:
- Permutation Entropy (PE): Ordinal pattern complexity
- Hjorth Parameters: Activity, mobility, complexity
- Lempel-Ziv Complexity (LZC): Algorithmic complexity
- Sample Entropy: Regularity measure
- Fractal Dimension: (via Higuchi or Katz, if implemented in signal_metrics)

Features are computed on the 'plateau' window (pain) by default.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import (
    MIN_SAMPLES_COMPLEXITY,
    get_segment_mask,
    validate_extractor_inputs,
)
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_permutation_entropy as _permutation_entropy,
    compute_sample_entropy as _sample_entropy,
    compute_hjorth_parameters as _hjorth_parameters,
    compute_lempel_ziv_complexity as _lempel_ziv_complexity,
)
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.spectral import bandpass_filter_epochs

# --- Helpers ---

def _extract_params(config: Any) -> Dict[str, Any]:
    """Extract complexity parameters from config."""
    return {
        "pe_order": int(config.get("feature_engineering.complexity.pe_order", 3)),
        "pe_delay": int(config.get("feature_engineering.complexity.pe_delay", 1)),
        "sampen_m": int(config.get("feature_engineering.complexity.sampen_m", 2)),
        "sampen_r": float(config.get("feature_engineering.complexity.sampen_r_mult", 0.2)),
        "include_sampen": config.get("feature_engineering.complexity.include_sample_entropy", False),
    }

def _compute_metrics_for_trace(data: np.ndarray, params: Dict[str, Any]) -> Dict[str, float]:
    """Compute metrics for a single 1D trace."""
    res = {}
    
    # PE
    res["pe"] = _permutation_entropy(data, order=params["pe_order"], delay=params["pe_delay"])
    
    # Hjorth parameters - can fail on constant or degenerate signals
    try:
        act, mob, comp = _hjorth_parameters(data)
        res["hjorth_activity"] = act
        res["hjorth_mobility"] = mob
        res["hjorth_complexity"] = comp
    except (ValueError, ZeroDivisionError, RuntimeError):
        res["hjorth_activity"] = np.nan
        res["hjorth_mobility"] = np.nan
        res["hjorth_complexity"] = np.nan
        
    # LZC
    res["lzc"] = _lempel_ziv_complexity(data)
    
    # Sample Entropy (expensive, optional)
    if params["include_sampen"]:
        res["sampen"] = _sample_entropy(data, m=params["sampen_m"], r_multiplier=params["sampen_r"])
        
    return res

def _process_epoch_metrics(
    epoch_data: np.ndarray, 
    ch_names: List[str], 
    params: Dict[str, Any],
    seg: str,
    band: str,
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
) -> Dict[str, float]:
    """Process all channels for a single epoch and aggregate spatially."""
    n_ch = len(ch_names)
    metric_keys = ["pe", "hjorth_mobility", "hjorth_complexity", "lzc"]
    if params.get("include_sampen"): 
        metric_keys.append("sampen")
    
    results_mat = np.full((len(metric_keys), n_ch), np.nan)
    for c in range(n_ch):
        mets = _compute_metrics_for_trace(epoch_data[c], params)
        for m_idx, k in enumerate(metric_keys):
            results_mat[m_idx, c] = mets.get(k, np.nan)
            
    epoch_res = {}
    for m_idx, k in enumerate(metric_keys):
        # Channels
        if 'channels' in spatial_modes:
            for c, ch in enumerate(ch_names):
                col = NamingSchema.build("dynamics", seg, band, "ch", k, channel=ch)
                epoch_res[col] = results_mat[m_idx, c]
        
        # ROI
        if 'roi' in spatial_modes and roi_map:
            for roi_name, ch_indices in roi_map.items():
                if ch_indices:
                    roi_val = np.nanmean(results_mat[m_idx, ch_indices])
                    col = NamingSchema.build("dynamics", seg, band, "roi", k, channel=roi_name)
                    epoch_res[col] = roi_val
                    
        # Global
        if 'global' in spatial_modes:
            global_val = np.nanmean(results_mat[m_idx, :])
            col = NamingSchema.build("dynamics", seg, band, "global", k)
            epoch_res[col] = global_val
            
    return epoch_res

# --- Main API ---

def _extract_dynamics_for_segment(
    data: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    bands: List[str],
    freq_bands: Dict[str, Tuple[float, float]],
    params: Dict[str, Any],
    segment_name: str,
    n_jobs: int,
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
) -> pd.DataFrame:
    """Extract dynamics features for a single segment."""
    n_epochs = len(data)
    dfs = []
    
    for band in bands:
        if band not in freq_bands: 
            continue
        fmin, fmax = freq_bands[band]
        
        filtered = bandpass_filter_epochs(data, sfreq, fmin, fmax, n_jobs=n_jobs)
        if filtered is None: 
            continue

        # Use parallelism for epoch processing - bandpass filter is already complete at this point
        # so there's no nested parallelism concern
        effective_jobs = n_jobs if n_epochs > 4 else 1  # Only parallelize if worthwhile
        epoch_dicts = Parallel(n_jobs=effective_jobs, prefer="threads")(
            delayed(_process_epoch_metrics)(filtered[e], ch_names, params, segment_name, band, spatial_modes, roi_map)
            for e in range(n_epochs)
        )
        
        band_df = pd.DataFrame(epoch_dicts)
        dfs.append(band_df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, axis=1)


def extract_dynamics_features(
    ctx: Any, # FeatureContext
    bands: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    
    valid, err = validate_extractor_inputs(ctx, "Complexity", min_epochs=2)
    if not valid:
        ctx.logger.warning(err)
        return pd.DataFrame(), []
    
    epochs = ctx.epochs
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        ctx.logger.warning("Complexity: No EEG channels available; skipping extraction.")
        return pd.DataFrame(), []
    
    full_data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]
    from eeg_pipeline.utils.parallel import get_n_jobs
    freq_bands = get_frequency_bands(ctx.config)
    n_jobs = get_n_jobs(ctx.config, default=-1, config_path="feature_engineering.parallel.n_jobs_complexity")
    params = _extract_params(ctx.config)
    n_epochs = len(full_data) # Added to define n_epochs for effective_jobs
    effective_jobs = n_jobs if n_epochs > 4 else 1
    
    # Get spatial modes and ROI map from context
    spatial_modes = getattr(ctx, 'spatial_modes', ['roi', 'global'])
    roi_map = {}
    if 'roi' in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(ctx.config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)

    all_dfs = []
    segments = get_segment_masks(epochs.times, ctx.windows, ctx.config)
    for seg_name, mask in segments.items():
        if mask is None or np.sum(mask) < int(sfreq):
            continue
        
        seg_data = full_data[:, :, mask]
        seg_df = _extract_dynamics_for_segment(
            seg_data, ch_names, sfreq, bands, freq_bands, params, seg_name, n_jobs,
            spatial_modes=spatial_modes, roi_map=roi_map
        )
        if not seg_df.empty:
            all_dfs.append(seg_df)
            ctx.logger.info(f"Computed Dynamics for {seg_name}: {np.sum(mask)} samples")
    
    if not all_dfs:
        ctx.logger.warning("No valid segments for dynamics")
        return pd.DataFrame(), []
    
    final_df = pd.concat(all_dfs, axis=1)
    return final_df, list(final_df.columns)


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
) -> Dict[str, float]:
    """Process complexity for a single epoch (parallel worker)."""
    record: Dict[str, float] = {}
    for band, bd in band_data.items():
        env = bd.envelope[ep_idx]
        env_seg = env[:, segment_mask] if not isinstance(segment_mask, slice) else env
        
        if env_seg.shape[1] < 10: 
            continue

        lzc_vals = []
        pe_vals = []
        
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
            
            col_lzc = NamingSchema.build("complexity", segment_name, band, "ch", "lzc", channel=ch_name)
            col_pe = NamingSchema.build("complexity", segment_name, band, "ch", "pe", channel=ch_name)
            record[col_lzc] = float(lzc)
            record[col_pe] = float(pe)
            
            if np.isfinite(lzc): lzc_vals.append(lzc)
            if np.isfinite(pe): pe_vals.append(pe)
            
        if lzc_vals:
            col_glob_lzc = NamingSchema.build("complexity", segment_name, band, "global", "lzc")
            record[col_glob_lzc] = float(np.mean(lzc_vals))
        if pe_vals:
            col_glob_pe = NamingSchema.build("complexity", segment_name, band, "global", "pe")
            record[col_glob_pe] = float(np.mean(pe_vals))
            
    return record


def _compute_complexity_for_segment(
    precomputed: Any,
    segment_mask: np.ndarray,
    segment_name: str,
    params: Dict[str, Any],
    n_jobs: int = 1,
) -> List[Dict[str, float]]:
    """Compute complexity metrics for a single segment using precomputed envelopes."""
    n_epochs = precomputed.data.shape[0]
    
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_complexity_epoch)(
                ep_idx, precomputed.band_data, precomputed.ch_names, segment_mask, segment_name, params
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_complexity_epoch(
                ep_idx, precomputed.band_data, precomputed.ch_names, segment_mask, segment_name, params
            )
            for ep_idx in range(n_epochs)
        ]
    
    return records


def extract_complexity_from_precomputed(
    precomputed: Any, # PrecomputedData
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """Computing complexity metrics (LZC, entropy, DFA-lite) using precomputed data."""
    if precomputed.windows is None:
        return pd.DataFrame(), []

    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    segments = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
    
    all_records = []
    params = _extract_params(precomputed.config)
    
    for seg_label, mask in segments.items():
        if mask is not None and np.any(mask):
            seg_recs = _compute_complexity_for_segment(precomputed, mask, seg_label, params, n_jobs=n_jobs)
            if not all_records:
                all_records = seg_recs
            else:
                for i, rec in enumerate(seg_recs):
                    all_records[i].update(rec)
    
    if not all_records or all(len(r) == 0 for r in all_records):
        return pd.DataFrame(), []

    df = pd.DataFrame(all_records)
    return df, list(df.columns)
