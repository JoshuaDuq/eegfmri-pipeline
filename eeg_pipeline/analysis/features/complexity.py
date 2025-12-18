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
    STANDARD_SEGMENTS,
    get_segment_mask,
    validate_extractor_inputs,
)
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
    band: str
) -> Dict[str, float]:
    """Process all channels for a single epoch and return named features."""
    # epoch_data shape: (n_ch, n_times)
    epoch_res = {}
    
    # Global accumulators for this epoch (mean across channels)
    acc: Dict[str, List[float]] = {
        "pe": [], 
        "hjorth_mobility": [], 
        "hjorth_complexity": [], 
        "lzc": []
    }
    if params["include_sampen"]:
        acc["sampen"] = []
    
    # Per-channel metrics
    for c, ch in enumerate(ch_names):
        trace = epoch_data[c]
        mets = _compute_metrics_for_trace(trace, params)
        
        for k, v in mets.items():
            # Build column name: dynamics_plateau_theta_ch_pe_Fp1
            col_name = NamingSchema.build("dynamics", seg, band, "ch", k, channel=ch)
            epoch_res[col_name] = v
            
            if k in acc and np.isfinite(v):
                acc[k].append(v)
                
    # Global mean metrics for this epoch
    for k, vals in acc.items():
        # Build column name: dynamics_plateau_theta_global_pe
        col_global = NamingSchema.build("dynamics", seg, band, "global", k)
        if vals:
            epoch_res[col_global] = np.mean(vals)
        else:
            epoch_res[col_global] = np.nan
            
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
            delayed(_process_epoch_metrics)(filtered[e], ch_names, params, segment_name, band)
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
    min_samps = MIN_SAMPLES_COMPLEXITY
    
    all_dfs = []
    
    for seg_name in STANDARD_SEGMENTS:
        mask = get_segment_mask(ctx.windows, seg_name)
        if mask is None or np.sum(mask) < min_samps:
            continue
        data_seg = full_data[..., mask]
        seg_df = _extract_dynamics_for_segment(
            data_seg, ch_names, sfreq, bands, freq_bands, params, seg_name, n_jobs
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
                pe = _permutation_entropy(trace, order=3, delay=1)
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
    n_jobs: int = 1,
) -> List[Dict[str, float]]:
    """Compute complexity metrics for a single segment using precomputed envelopes."""
    n_epochs = precomputed.data.shape[0]
    
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_complexity_epoch)(
                ep_idx, precomputed.band_data, precomputed.ch_names, segment_mask, segment_name
            )
            for ep_idx in range(n_epochs)
        )
    else:
        records = [
            _process_complexity_epoch(
                ep_idx, precomputed.band_data, precomputed.ch_names, segment_mask, segment_name
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

    all_records: List[Dict[str, float]] = []
    
    baseline_mask = precomputed.windows.get_mask("baseline")
    active_mask = precomputed.windows.get_mask("plateau")
    
    if baseline_mask is not None and np.any(baseline_mask):
        baseline_records = _compute_complexity_for_segment(precomputed, baseline_mask, "baseline", n_jobs=n_jobs)
        if baseline_records:
            if not all_records:
                all_records = baseline_records
            else:
                for i, rec in enumerate(baseline_records):
                    all_records[i].update(rec)
    
    # Process plateau segment
    if active_mask is not None and np.any(active_mask):
        plateau_records = _compute_complexity_for_segment(precomputed, active_mask, "plateau", n_jobs=n_jobs)
        if plateau_records:
            if not all_records:
                all_records = plateau_records
            else:
                for i, rec in enumerate(plateau_records):
                    all_records[i].update(rec)
    
    if not all_records or all(len(r) == 0 for r in all_records):
        return pd.DataFrame(), []

    df = pd.DataFrame(all_records)
    return df, list(df.columns)
