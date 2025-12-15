from __future__ import annotations

from typing import List, Dict, Tuple, Any, Optional

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
    if maps.size == 0:
        return maps
    mu = np.mean(maps, axis=axis, keepdims=True)
    sd = np.std(maps, axis=axis, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (maps - mu) / sd

def compute_gfp_with_floor(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if data.size == 0:
        return np.array([])
    gfp = np.std(data, axis=0)
    return np.where(gfp < eps, eps, gfp)

def corr_maps(maps_a: np.ndarray, maps_b: np.ndarray) -> np.ndarray:
    # Pearson correlation between sets of maps
    # Assumes z-scored input for efficiency
    if maps_a.shape[1] != maps_b.shape[1]:
        raise ValueError("Shape mismatch")
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
    if epoch.size == 0:
        return None
    gfp = compute_gfp_with_floor(epoch)
    peaks, _ = find_peaks(gfp, distance=min_dist_samples)
    if peaks.size == 0:
        return None
    sorted_indices = np.argsort(gfp[peaks])[::-1]
    n_select = min(peaks_per_epoch, peaks.size)
    selected_peaks = peaks[sorted_indices[:n_select]]
    return zscore_maps(epoch[:, selected_peaks].T, axis=1)

def extract_templates_from_trials(trial_data, sfreq, n_states, config):
    peak_maps = []
    min_peak_distance_ms = float(config.get("feature_engineering.microstates.min_peak_distance_ms", 20.0))
    peaks_per_epoch = int(config.get("feature_engineering.microstates.peaks_per_epoch", 5))
    max_gfp_peaks_per_epoch = int(config.get("feature_engineering.microstates.max_gfp_peaks_per_epoch", 100))
    min_dist = max(1, int((min_peak_distance_ms / 1000.0) * sfreq))

    for ep in trial_data:
        if ep.size == 0:
            continue
        gfp = compute_gfp_with_floor(ep)
        peaks, _ = find_peaks(gfp, distance=min_dist)
        if peaks.size == 0:
            continue
        if peaks.size > max_gfp_peaks_per_epoch:
            sorted_peaks = peaks[np.argsort(gfp[peaks])[::-1][:max_gfp_peaks_per_epoch]]
            peaks = np.sort(sorted_peaks)
        sorted_indices = np.argsort(gfp[peaks])[::-1]
        n_select = min(peaks_per_epoch, peaks.size)
        selected_peaks = peaks[sorted_indices[:n_select]]
        m = zscore_maps(ep[:, selected_peaks].T, axis=1)
        if m is not None and m.size > 0:
            peak_maps.append(m)

    if not peak_maps:
        return None
    all_maps = np.vstack(peak_maps)

    polarity_invariant = bool(config.get("feature_engineering.microstates.polarity_invariant_kmeans", False))
    if polarity_invariant:
        max_indices = np.argmax(np.abs(all_maps), axis=1)
        signs = np.sign(all_maps[np.arange(all_maps.shape[0]), max_indices])
        signs[signs == 0] = 1
        all_maps = all_maps * signs[:, np.newaxis]

    random_state = int(config.get("project.random_state", 42))
    n_init = int(config.get("feature_engineering.microstates.kmeans_n_init", 10))
    kmeans = KMeans(n_clusters=n_states, n_init=n_init, random_state=random_state)
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

def _compute_metrics(
    state_labels: np.ndarray,
    sfreq: float,
    n_states: int,
    record: Dict[str, float],
    *,
    min_run_ms: float,
    valid_mask: Optional[np.ndarray] = None,
):
    # Compute Coverage, Duration, Occurrence
    if valid_mask is None:
        valid_mask = np.ones(state_labels.size, dtype=bool)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape[0] != state_labels.shape[0]:
        raise ValueError("valid_mask must have same length as state_labels")

    valid_labels = state_labels[valid_mask]
    valid_samples = int(valid_labels.size)
    total_samples = int(state_labels.size)
    duration_sec_total = total_samples / sfreq if sfreq > 0 else 0.0
    duration_sec_valid = valid_samples / sfreq if sfreq > 0 else 0.0

    counts = np.bincount(valid_labels, minlength=n_states) if valid_samples > 0 else np.zeros(n_states, dtype=int)
    
    # Coverage
    coverage = counts / float(valid_samples) if valid_samples > 0 else np.zeros(n_states)
    for i in range(n_states):
        record[f"coverage_state{i}"] = coverage[i]
        
    if valid_samples == 0:
        return

    # Build runs within each continuous valid segment.
    # When valid_mask has gaps, we avoid counting transitions that involve the
    # first run after a segment boundary (boundary artifacts).
    runs_all: List[Tuple[int, int]] = []
    transition_segments: List[List[Tuple[int, int]]] = []
    segment_runs: List[Tuple[int, int]] = []
    segment_index = 0
    curr: Optional[int] = None
    length = 0
    in_segment = False

    def _flush_segment() -> None:
        nonlocal segment_runs, segment_index
        if not segment_runs:
            return
        runs_all.extend(segment_runs)
        if segment_index == 0:
            transition_segments.append(list(segment_runs))
        else:
            transition_segments.append(list(segment_runs[1:]))
        segment_runs = []
        segment_index += 1

    for is_valid, x in zip(valid_mask, state_labels):
        if not bool(is_valid):
            if curr is not None and length > 0:
                segment_runs.append((int(curr), int(length)))
            curr = None
            length = 0
            if in_segment:
                _flush_segment()
            in_segment = False
            continue

        in_segment = True
        xi = int(x)
        if curr is None:
            curr = xi
            length = 1
            continue

        if xi == int(curr):
            length += 1
        else:
            segment_runs.append((int(curr), int(length)))
            curr = xi
            length = 1

    if curr is not None and length > 0:
        segment_runs.append((int(curr), int(length)))
    if in_segment:
        _flush_segment()

    min_run_samples = max(1, int((min_run_ms / 1000.0) * sfreq)) if sfreq > 0 else 1
    valid_runs = [(s, l) for (s, l) in runs_all if int(l) >= min_run_samples]
    transition_segments = [
        [(s, l) for (s, l) in seg if int(l) >= min_run_samples]
        for seg in transition_segments
    ]
    
    durations = {i: [] for i in range(n_states)}
    for s, l in valid_runs:
        durations[s].append(l / sfreq)
        
    for i in range(n_states):
        durs = durations[i]
        mean_dur = np.mean(durs) if durs else 0.0
        occ = len(durs) / duration_sec_valid if duration_sec_valid > 0 else 0.0
        record[f"duration_state{i}"] = mean_dur
        record[f"occurrence_state{i}"] = occ
        
    # Transition (Markov)
    trans = np.zeros((n_states, n_states))
    for seg_runs in transition_segments:
        for (s1, _), (s2, _) in zip(seg_runs[:-1], seg_runs[1:]):
            trans[int(s1), int(s2)] += 1
    # Rate? Or Probability? Plan doesn't specify deeply, usually probability or rate.
    # Legacy was rate.
    trans_rate = trans / duration_sec_valid if duration_sec_valid > 0 else trans
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
    *,
    min_run_ms: float,
    gfp_min_percentile: float,
    min_valid_fraction: float,
    min_correlation: float,
) -> Dict[str, float]:
    """Process a single trial for microstate features (parallel worker)."""
    def _empty_record() -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i in range(n_states):
            out[f"coverage_state{i}"] = np.nan
            out[f"duration_state{i}"] = np.nan
            out[f"occurrence_state{i}"] = np.nan
        for i in range(n_states):
            for j in range(n_states):
                if i == j:
                    continue
                out[f"trans_{i}_to_{j}"] = np.nan
        out["gev"] = np.nan
        out["valid_fraction"] = np.nan
        out["mean_abs_corr"] = np.nan
        return out

    lbls, assigned_corr, gev = label_timecourse(trial_data, templates)
    if lbls.size == 0:
        return _empty_record()

    gfp = compute_gfp_with_floor(trial_data)
    if gfp.size != lbls.size:
        return _empty_record()

    gfp_thresh = float(np.nanpercentile(gfp, gfp_min_percentile)) if np.isfinite(gfp_min_percentile) else np.nan
    valid_mask = np.isfinite(gfp)
    if np.isfinite(gfp_thresh):
        valid_mask &= gfp >= gfp_thresh

    if np.isfinite(min_correlation):
        valid_mask &= np.abs(assigned_corr) >= float(min_correlation)

    valid_fraction = float(np.mean(valid_mask)) if valid_mask.size else 0.0
    rec = _empty_record()
    rec["valid_fraction"] = valid_fraction
    if np.any(valid_mask):
        rec["mean_abs_corr"] = float(np.nanmean(np.abs(assigned_corr[valid_mask])))

    if valid_fraction < float(min_valid_fraction) or int(np.sum(valid_mask)) < 2:
        return rec

    _compute_metrics(lbls, sfreq, n_states, rec, min_run_ms=min_run_ms, valid_mask=valid_mask)
    rec["gev"] = float(gev) if np.isfinite(gev) else np.nan
    return rec


def _extract_microstates_for_segment(
    data: np.ndarray,
    templates: np.ndarray,
    sfreq: float,
    n_states: int,
    segment_name: str,
    n_jobs: int = 1,
    *,
    min_run_ms: float,
    gfp_min_percentile: float,
    min_valid_fraction: float,
    min_correlation: float,
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
            delayed(_process_single_trial_microstates)(
                data[i],
                templates,
                sfreq,
                n_states,
                min_run_ms=min_run_ms,
                gfp_min_percentile=gfp_min_percentile,
                min_valid_fraction=min_valid_fraction,
                min_correlation=min_correlation,
            )
            for i in range(len(data))
        )
    else:
        records = [
            _process_single_trial_microstates(
                data[i],
                templates,
                sfreq,
                n_states,
                min_run_ms=min_run_ms,
                gfp_min_percentile=gfp_min_percentile,
                min_valid_fraction=min_valid_fraction,
                min_correlation=min_correlation,
            )
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

    # Microstate computation parallelizes across epochs/trials.
    # Prefer a dedicated knob if available; otherwise fall back to the legacy key.
    n_jobs = int(
        ctx.config.get(
            "feature_engineering.parallel.n_jobs_microstates",
            ctx.config.get("feature_engineering.parallel.n_jobs_feature_groups", -1),
        )
    )
    min_samples = 10

    min_run_ms = float(ctx.config.get("feature_engineering.microstates.min_run_ms", 10.0))
    gfp_min_percentile = float(ctx.config.get("feature_engineering.microstates.gfp_min_percentile", 25.0))
    min_valid_fraction = float(ctx.config.get("feature_engineering.microstates.min_valid_fraction", 0.5))
    min_correlation = float(ctx.config.get("feature_engineering.microstates.min_correlation", 0.25))
    
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
        templates = extract_templates_from_trials(data_plateau, sfreq, n_states, ctx.config)
        
    if templates is None:
        return pd.DataFrame(), [], None
    
    all_data = {}
    
    # Process baseline segment
    mask_baseline = ctx.windows.get_mask("baseline")
    if mask_baseline is not None and np.sum(mask_baseline) >= min_samples:
        data_baseline = full_data[..., mask_baseline]
        baseline_features = _extract_microstates_for_segment(
            data_baseline,
            templates,
            sfreq,
            n_states,
            "baseline",
            n_jobs=n_jobs,
            min_run_ms=min_run_ms,
            gfp_min_percentile=gfp_min_percentile,
            min_valid_fraction=min_valid_fraction,
            min_correlation=min_correlation,
        )
        all_data.update(baseline_features)
        ctx.logger.info(f"Computed Microstates for baseline: {np.sum(mask_baseline)} samples")
    
    # Process ramp segment
    mask_ramp = ctx.windows.get_mask("ramp")
    if mask_ramp is not None and np.sum(mask_ramp) >= min_samples:
        data_ramp = full_data[..., mask_ramp]
        ramp_features = _extract_microstates_for_segment(
            data_ramp,
            templates,
            sfreq,
            n_states,
            "ramp",
            n_jobs=n_jobs,
            min_run_ms=min_run_ms,
            gfp_min_percentile=gfp_min_percentile,
            min_valid_fraction=min_valid_fraction,
            min_correlation=min_correlation,
        )
        all_data.update(ramp_features)
        ctx.logger.info(f"Computed Microstates for ramp: {np.sum(mask_ramp)} samples")
    
    # Process plateau segment
    plateau_features = _extract_microstates_for_segment(
        data_plateau,
        templates,
        sfreq,
        n_states,
        "plateau",
        n_jobs=n_jobs,
        min_run_ms=min_run_ms,
        gfp_min_percentile=gfp_min_percentile,
        min_valid_fraction=min_valid_fraction,
        min_correlation=min_correlation,
    )
    all_data.update(plateau_features)
    ctx.logger.info(f"Computed Microstates for plateau: {np.sum(mask_plateau)} samples")
    
    if not all_data:
        return pd.DataFrame(), [], templates
        
    df = pd.DataFrame(all_data)
    return df, list(df.columns), templates


def extract_microstate_features_from_epochs(
    epochs: mne.Epochs,
    n_states: int,
    config: Any,
    logger: Any,
    *,
    fixed_templates: Optional[np.ndarray] = None,
    fixed_template_ch_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[np.ndarray]]:
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        return pd.DataFrame(), [], None

    full_data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])

    # Microstate computation parallelizes across epochs/trials.
    # Prefer a dedicated knob if available; otherwise fall back to the legacy key.
    n_jobs = int(
        config.get(
            "feature_engineering.parallel.n_jobs_microstates",
            config.get("feature_engineering.parallel.n_jobs_feature_groups", -1),
        )
    )
    min_samples = 10

    min_run_ms = float(config.get("feature_engineering.microstates.min_run_ms", 10.0))
    gfp_min_percentile = float(config.get("feature_engineering.microstates.gfp_min_percentile", 25.0))
    min_valid_fraction = float(config.get("feature_engineering.microstates.min_valid_fraction", 0.5))
    min_correlation = float(config.get("feature_engineering.microstates.min_correlation", 0.25))

    from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec

    spec = TimeWindowSpec(
        times=epochs.times,
        config=config,
        sampling_rate=float(sfreq),
        logger=logger,
    )
    baseline_mask = spec.get_mask("baseline")
    ramp_mask = spec.get_mask("ramp")
    plateau_mask = spec.get_mask("plateau")

    if plateau_mask is None or not np.any(plateau_mask):
        return pd.DataFrame(), [], None

    data_plateau = full_data[..., plateau_mask]
    if data_plateau.shape[-1] < min_samples:
        return pd.DataFrame(), [], None

    if fixed_templates is not None and fixed_template_ch_names is not None:
        templates = _reorder_templates_to_picks(
            fixed_templates,
            list(fixed_template_ch_names),
            picks,
            epochs.info,
            logger,
        )
    else:
        templates = extract_templates_from_trials(data_plateau, sfreq, n_states, config)

    if templates is None:
        return pd.DataFrame(), [], None

    all_data: Dict[str, List[float]] = {}

    if baseline_mask is not None and int(np.sum(baseline_mask)) >= min_samples:
        data_baseline = full_data[..., baseline_mask]
        all_data.update(
            _extract_microstates_for_segment(
                data_baseline,
                templates,
                sfreq,
                n_states,
                "baseline",
                n_jobs=n_jobs,
                min_run_ms=min_run_ms,
                gfp_min_percentile=gfp_min_percentile,
                min_valid_fraction=min_valid_fraction,
                min_correlation=min_correlation,
            )
        )

    if ramp_mask is not None and int(np.sum(ramp_mask)) >= min_samples:
        data_ramp = full_data[..., ramp_mask]
        all_data.update(
            _extract_microstates_for_segment(
                data_ramp,
                templates,
                sfreq,
                n_states,
                "ramp",
                n_jobs=n_jobs,
                min_run_ms=min_run_ms,
                gfp_min_percentile=gfp_min_percentile,
                min_valid_fraction=min_valid_fraction,
                min_correlation=min_correlation,
            )
        )

    all_data.update(
        _extract_microstates_for_segment(
            data_plateau,
            templates,
            sfreq,
            n_states,
            "plateau",
            n_jobs=n_jobs,
            min_run_ms=min_run_ms,
            gfp_min_percentile=gfp_min_percentile,
            min_valid_fraction=min_valid_fraction,
            min_correlation=min_correlation,
        )
    )

    if not all_data:
        return pd.DataFrame(), [], templates

    df = pd.DataFrame(all_data)
    return df, list(df.columns), templates
