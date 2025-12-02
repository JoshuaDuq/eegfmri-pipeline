from __future__ import annotations

from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from joblib import Parallel, delayed

from eeg_pipeline.utils.analysis.features.metadata import NamingSchema
from eeg_pipeline.utils.analysis.channels import pick_eeg_channels

# --- Helpers (Preserved logic, condensed where possible) ---

def zscore_maps(maps: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    if maps.size == 0: return maps
    mu = np.mean(maps, axis=axis, keepdims=True)
    sd = np.std(maps, axis=axis, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (maps - mu) / sd

def compute_gfp_with_floor(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if data.size == 0: return np.array([])
    gfp = np.std(data, axis=0)
    return np.where(gfp < eps, eps, gfp)

def corr_maps(maps_a: np.ndarray, maps_b: np.ndarray) -> np.ndarray:
    # Pearson correlation between sets of maps
    # Assumes z-scored input for efficiency
    if maps_a.shape[1] != maps_b.shape[1]: raise ValueError("Shape mismatch")
    n = maps_a.shape[1]
    denom = max(n - 1, 1)
    return (maps_a @ maps_b.T) / denom

def label_timecourse(channel_data: np.ndarray, templates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Label each timepoint with the best-matching microstate template.
    
    Returns:
        state_indices: (n_times,) array of state labels
        assigned_correlations: (n_times,) array of correlation with assigned state
        gev: Global Explained Variance - how well templates explain the data
    """
    if channel_data.size == 0: 
        return np.array([], dtype=int), np.array([], dtype=float), np.nan
    
    zscored = zscore_maps(channel_data.T, axis=1)  # (n_times, n_ch)
    correlations = corr_maps(zscored, templates)    # (n_times, n_states)
    state_indices = np.argmax(np.abs(correlations), axis=1)
    assigned_correlations = correlations[np.arange(correlations.shape[0]), state_indices]
    
    # Compute GEV: sum of squared correlations weighted by GFP^2
    gfp = compute_gfp_with_floor(channel_data)  # (n_times,)
    gfp_squared = gfp ** 2
    
    # GEV = sum(r^2 * GFP^2) / sum(GFP^2)
    r_squared = assigned_correlations ** 2
    total_gfp_sq = np.sum(gfp_squared)
    if total_gfp_sq > 0:
        gev = float(np.sum(r_squared * gfp_squared) / total_gfp_sq)
    else:
        gev = np.nan
    
    return state_indices.astype(int), assigned_correlations.astype(float), gev

def _extract_peak_maps_from_epoch(epoch, min_dist_samples, peaks_per_epoch):
    if epoch.size == 0: return None
    gfp = compute_gfp_with_floor(epoch)
    peaks, _ = find_peaks(gfp, distance=min_dist_samples)
    if peaks.size == 0: return None
    sorted_indices = np.argsort(gfp[peaks])[::-1]
    n_select = min(peaks_per_epoch, peaks.size)
    selected_peaks = peaks[sorted_indices[:n_select]]
    return zscore_maps(epoch[:, selected_peaks].T, axis=1)

def extract_templates_from_trials(trial_data, sfreq, n_states):
    # (Simplified for brevity but retaining core logic)
    peak_maps = []
    min_dist = max(1, int((20.0/1000)*sfreq))
    for ep in trial_data:
        m = _extract_peak_maps_from_epoch(ep, min_dist, 5)
        if m is not None: peak_maps.append(m)
    
    if not peak_maps: return None
    all_maps = np.vstack(peak_maps)
    
    # Sign flip for polarity invariance
    max_indices = np.argmax(np.abs(all_maps), axis=1)
    signs = np.sign(all_maps[np.arange(all_maps.shape[0]), max_indices])
    signs[signs==0] = 1
    all_maps = all_maps * signs[:, np.newaxis]
    
    kmeans = KMeans(n_clusters=n_states, n_init=10, random_state=42)
    kmeans.fit(all_maps)
    return zscore_maps(kmeans.cluster_centers_, axis=1)

def _reorder_templates_to_picks(templates, template_ch_names, picks, info, logger):
    # Align templates to current data channels
    pick_names = [info["ch_names"][i] for i in picks]
    indices = [template_ch_names.index(ch) for ch in pick_names if ch in template_ch_names]
    if len(indices) != len(pick_names):
        logger.warning("Template alignment mismatch")
        return None
    return templates[:, indices]

def _compute_metrics(state_labels, sfreq, n_states, record):
    # Compute Coverage, Duration, Occurrence
    total_samples = float(state_labels.size)
    duration_sec = total_samples / sfreq if sfreq > 0 else 0
    counts = np.bincount(state_labels, minlength=n_states)
    
    # Coverage
    coverage = counts / total_samples if total_samples > 0 else np.zeros(n_states)
    for i in range(n_states):
        record[f"coverage_state{i}"] = coverage[i]
        
    # Duration / Occurrence
    # Run Logic
    if total_samples == 0: return
    
    runs = []
    curr = state_labels[0]
    length = 1
    for x in state_labels[1:]:
        if x == curr: length += 1
        else:
            runs.append((curr, length))
            curr = x
            length = 1
    runs.append((curr, length))
    
    durations = {i: [] for i in range(n_states)}
    for s, l in runs:
        durations[s].append(l / sfreq)
        
    for i in range(n_states):
        durs = durations[i]
        mean_dur = np.mean(durs) if durs else 0.0
        occ = len(durs) / duration_sec if duration_sec > 0 else 0.0
        record[f"duration_state{i}"] = mean_dur
        record[f"occurrence_state{i}"] = occ
        
    # Transition (Markov)
    # Simple transition matrix
    trans = np.zeros((n_states, n_states))
    for (s1, l1), (s2, l2) in zip(runs[:-1], runs[1:]):
        trans[s1, s2] += 1
    # Rate? Or Probability? Plan doesn't specify deeply, usually probability or rate.
    # Legacy was rate.
    trans_rate = trans / duration_sec if duration_sec > 0 else trans
    for i in range(n_states):
        for j in range(n_states):
            if i==j: continue
            record[f"trans_{i}_to_{j}"] = trans_rate[i,j]

# --- Main API ---

def _process_single_trial_microstates(
    trial_data: np.ndarray,
    templates: np.ndarray,
    sfreq: float,
    n_states: int,
) -> Dict[str, float]:
    """Process a single trial for microstate features (parallel worker)."""
    lbls, _, gev = label_timecourse(trial_data, templates)
    rec = {}
    _compute_metrics(lbls, sfreq, n_states, rec)
    rec["gev"] = gev
    return rec


def _extract_microstates_for_segment(
    data: np.ndarray,
    templates: np.ndarray,
    sfreq: float,
    n_states: int,
    segment_name: str,
    n_jobs: int = 1,
) -> Dict[str, List[float]]:
    """Extract microstate features for a single segment.
    
    Features include:
    - Coverage, Duration, Occurrence per state
    - Transition rates between states
    - GEV (Global Explained Variance) - template fit quality
    """
    formatted_data = {}
    
    if n_jobs != 1:
        records = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_trial_microstates)(data[i], templates, sfreq, n_states)
            for i in range(len(data))
        )
    else:
        records = [
            _process_single_trial_microstates(data[i], templates, sfreq, n_states)
            for i in range(len(data))
        ]
    
    if not records:
        return {}
    
    keys = records[0].keys()
    for k in keys:
        vals = [r[k] for r in records]
        col = NamingSchema.build("microstates", segment_name, "broadband", "global", k)
        formatted_data[col] = vals
    
    return formatted_data


def extract_microstate_features(
    ctx: Any, # FeatureContext
) -> Tuple[pd.DataFrame, List[str], Optional[np.ndarray]]:
    
    n_states = int(ctx.config.get("feature_engineering.microstates.n_states", 4))
    epochs = ctx.epochs
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0: 
        return pd.DataFrame(), [], None
    
    full_data = epochs.get_data(picks=picks)
    sfreq = epochs.info["sfreq"]
    n_jobs = int(ctx.config.get("system.n_jobs", -1))
    min_samples = 10
    
    # Get templates from plateau data (primary segment)
    mask_plateau = ctx.windows.get_mask("plateau")
    if mask_plateau is None or not np.any(mask_plateau):
        ctx.logger.warning("No plateau window for microstates")
        return pd.DataFrame(), [], None
    
    data_plateau = full_data[..., mask_plateau]
    if data_plateau.shape[-1] < min_samples:
        return pd.DataFrame(), [], None
    
    # Templates
    fixed = ctx.fixed_templates
    fixed_names = ctx.fixed_template_ch_names
    
    if fixed is not None and fixed_names is not None:
        templates = _reorder_templates_to_picks(fixed, fixed_names, picks, epochs.info, ctx.logger)
    else:
        templates = extract_templates_from_trials(data_plateau, sfreq, n_states)
        
    if templates is None:
        return pd.DataFrame(), [], None
    
    all_data = {}
    
    # Process baseline segment
    mask_baseline = ctx.windows.get_mask("baseline")
    if mask_baseline is not None and np.sum(mask_baseline) >= min_samples:
        data_baseline = full_data[..., mask_baseline]
        baseline_features = _extract_microstates_for_segment(
            data_baseline, templates, sfreq, n_states, "baseline", n_jobs=n_jobs
        )
        all_data.update(baseline_features)
        ctx.logger.info(f"Computed Microstates for baseline: {np.sum(mask_baseline)} samples")
    
    # Process ramp segment
    from eeg_pipeline.utils.config.loader import get_config_value
    times = epochs.times
    ramp_end = float(get_config_value(ctx.config, "feature_engineering.features.ramp_end", 3.0))
    ramp_mask = (times >= 0) & (times <= ramp_end)
    if np.sum(ramp_mask) >= min_samples:
        data_ramp = full_data[..., ramp_mask]
        ramp_features = _extract_microstates_for_segment(
            data_ramp, templates, sfreq, n_states, "ramp", n_jobs=n_jobs
        )
        all_data.update(ramp_features)
        ctx.logger.info(f"Computed Microstates for ramp: {np.sum(ramp_mask)} samples")
    
    # Process plateau segment
    plateau_features = _extract_microstates_for_segment(
        data_plateau, templates, sfreq, n_states, "plateau", n_jobs=n_jobs
    )
    all_data.update(plateau_features)
    ctx.logger.info(f"Computed Microstates for plateau: {np.sum(mask_plateau)} samples")
    
    if not all_data:
        return pd.DataFrame(), [], templates
        
    df = pd.DataFrame(all_data)
    return df, list(df.columns), templates
