from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import mne
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks

from eeg_pipeline.utils.tfr_utils import time_mask, freq_mask
from eeg_pipeline.utils.io_utils import find_first


###################################################################
# Microstate Core Functions
###################################################################


def zscore_maps(X: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(X, axis=axis, keepdims=True)
    sd = np.std(X, axis=axis, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (X - mu) / sd


def compute_gfp(data: np.ndarray) -> np.ndarray:
    return np.std(data, axis=0)


def corr_maps(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A @ B.T) / (A.shape[1] - 1)


def label_timecourse(Xch: np.ndarray, templates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Zt = zscore_maps(Xch.T, axis=1)
    C = corr_maps(Zt, templates)
    idx = np.argmax(np.abs(C), axis=1)
    corr_assigned = C[np.arange(C.shape[0]), idx]
    return idx.astype(int), corr_assigned.astype(float)


def extract_templates_from_trials(
    X: np.ndarray,
    sfreq: float,
    n_states: int,
    config,
) -> Optional[np.ndarray]:
    min_peak_distance_ms = float(
        config.get("feature_engineering.microstates.min_peak_distance_ms", 20.0)
    ) if config else 20.0
    peaks_per_epoch = int(
        config.get("feature_engineering.microstates.peaks_per_epoch", 5)
    ) if config else 5
    microstate_random_state = int(config.get("random.seed", 42) if config else 42)

    peak_maps: List[np.ndarray] = []
    min_dist = max(1, int((min_peak_distance_ms / 1000.0) * sfreq))
    for ep in X:
        gfp = compute_gfp(ep)
        if np.allclose(gfp, 0):
            continue
        peaks, _ = find_peaks(gfp, distance=min_dist)
        if peaks.size == 0:
            continue
        order = np.argsort(gfp[peaks])[::-1]
        sel = peaks[order[:peaks_per_epoch]]
        maps = ep[:, sel].T
        maps = zscore_maps(maps, axis=1)
        peak_maps.append(maps)
    if not peak_maps:
        return None
    M = np.vstack(peak_maps)

    kmeans = KMeans(n_clusters=int(n_states), n_init=20, random_state=microstate_random_state)
    kmeans.fit(M)
    templates = kmeans.cluster_centers_
    templates = zscore_maps(templates, axis=1)
    return templates


def _state_labels(n_states: int) -> List[str]:
    return [chr(65 + i) for i in range(n_states)]


@dataclass
class MicrostateTransitionStats:
    nonpain: np.ndarray
    pain: np.ndarray
    state_labels: List[str]


@dataclass
class MicrostateDurationStat:
    state: str
    nonpain: np.ndarray
    pain: np.ndarray
    p_value: float


def compute_microstate_metric_correlations(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    config,
    metrics: Optional[List[str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    method: str = "spearman",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rating_col = None
    if config is not None:
        rating_col = next(
            (col for col in config.get("event_columns.rating", []) if col in events_df.columns),
            None,
        )
    if rating_col is None:
        numeric_cols = events_df.select_dtypes(include=[np.number]).columns
        rating_col = numeric_cols[0] if len(numeric_cols) else None
    if rating_col is None:
        return pd.DataFrame(), pd.DataFrame()

    ratings = pd.to_numeric(events_df[rating_col], errors="coerce")
    if ratings.notna().sum() < 5:
        return pd.DataFrame(), pd.DataFrame()

    metrics = metrics or ["coverage", "duration", "occurrence", "gev"]
    metric_labels = metric_labels or {
        "coverage": "Coverage",
        "duration": "Duration",
        "occurrence": "Occurrence",
        "gev": "GEV",
    }

    n_states_detected = max(
        (
            int(col.split("_")[-1])
            for col in ms_df.columns
            if col.startswith("ms_") and col.rsplit("_", 1)[-1].isdigit()
        ),
        default=-1,
    ) + 1
    state_labels = _state_labels(n_states_detected) if n_states_detected > 0 else []

    corr = pd.DataFrame(
        np.nan,
        index=[metric_labels.get(m, m.capitalize()) for m in metrics],
        columns=state_labels,
    )
    pvals = corr.copy()

    for metric in metrics:
        metric_label = metric_labels.get(metric, metric.capitalize())
        for state_idx, state_label in enumerate(state_labels):
            col = f"ms_{metric}_{state_idx}"
            if col not in ms_df.columns:
                continue
            metric_vals = pd.to_numeric(ms_df[col], errors="coerce")
            valid_mask = metric_vals.notna() & ratings.notna()
            if valid_mask.sum() < 5:
                continue
            x = metric_vals[valid_mask].to_numpy()
            y = ratings[valid_mask].to_numpy()
            if np.std(x) <= 0 or np.std(y) <= 0:
                continue
            if method == "pearson":
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)
            corr.at[metric_label, state_label] = float(r)
            pvals.at[metric_label, state_label] = float(p)

    return corr, pvals


def compute_microstate_transition_stats(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    n_states: int,
    config,
) -> MicrostateTransitionStats:
    trans_nonpain = np.zeros((n_states, n_states), dtype=float)
    trans_pain = np.zeros((n_states, n_states), dtype=float)

    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        return MicrostateTransitionStats(trans_nonpain, trans_pain, _state_labels(n_states))

    pain_col = None
    if config is not None:
        pain_col = next(
            (col for col in config.get("event_columns.pain_binary", []) if col in events_df.columns),
            None,
        )
    if pain_col is None:
        return MicrostateTransitionStats(trans_nonpain, trans_pain, _state_labels(n_states))

    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    valid_mask = pain_vals.notna()

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            col = f"ms_trans_{i}_to_{j}"
            if col not in ms_df.columns:
                continue
            trans_vals = pd.to_numeric(ms_df[col], errors="coerce")
            nonpain_mask = valid_mask & (pain_vals == 0)
            pain_mask = valid_mask & (pain_vals == 1)

            nonpain_data = trans_vals[nonpain_mask].to_numpy(dtype=float)
            pain_data = trans_vals[pain_mask].to_numpy(dtype=float)

            nonpain_data = nonpain_data[np.isfinite(nonpain_data)]
            pain_data = pain_data[np.isfinite(pain_data)]

            trans_nonpain[i, j] = np.mean(nonpain_data) if nonpain_data.size else 0.0
            trans_pain[i, j] = np.mean(pain_data) if pain_data.size else 0.0

    return MicrostateTransitionStats(trans_nonpain, trans_pain, _state_labels(n_states))


def compute_microstate_duration_stats(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    n_states: int,
    config,
) -> List[MicrostateDurationStat]:
    stats_list: List[MicrostateDurationStat] = []

    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        return stats_list

    pain_col = None
    if config is not None:
        pain_col = next(
            (col for col in config.get("event_columns.pain_binary", []) if col in events_df.columns),
            None,
        )
    if pain_col is None:
        return stats_list

    pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
    valid_mask = pain_vals.notna()

    for idx, state_label in enumerate(_state_labels(n_states)):
        col = f"ms_duration_{idx}"
        if col not in ms_df.columns:
            continue
        durations = pd.to_numeric(ms_df[col], errors="coerce")
        nonpain_mask = valid_mask & (pain_vals == 0)
        pain_mask = valid_mask & (pain_vals == 1)

        nonpain_data = durations[nonpain_mask].to_numpy(dtype=float)
        pain_data = durations[pain_mask].to_numpy(dtype=float)

        nonpain_data = nonpain_data[np.isfinite(nonpain_data)]
        pain_data = pain_data[np.isfinite(pain_data)]

        p_val = np.nan
        if nonpain_data.size and pain_data.size:
            try:
                _, p_val = stats.mannwhitneyu(nonpain_data, pain_data, alternative="two-sided")
            except ValueError:
                p_val = np.nan

        stats_list.append(
            MicrostateDurationStat(
                state=state_label,
                nonpain=nonpain_data,
                pain=pain_data,
                p_value=float(p_val) if np.isfinite(p_val) else np.nan,
            )
        )

    return stats_list


###################################################################
# Helper Functions
###################################################################

def _fit_aperiodic(logf: np.ndarray, logpsd: np.ndarray) -> Tuple[float, float]:
    try:
        b, a = np.polyfit(logf, logpsd, 1)
        return float(a), float(b)
    except Exception:
        return float("nan"), float("nan")


def _find_connectivity_arrays(subject: str, task: str, band: str, deriv_root: Path):
    def _find_measure(measure: str) -> Optional[Path]:
        patterns = [
            f"sub-{subject}/eeg/sub-{subject}_task-{task}_*connectivity_{measure}_{band}*_all_trials.npy",
            f"sub-{subject}/eeg/sub-{subject}_task-{task}_connectivity_{measure}_{band}*_all_trials.npy",
        ]
        for pattern in patterns:
            path = find_first(str((deriv_root / pattern).as_posix()))
            if path:
                return path
        return None
    
    return _find_measure("aec"), _find_measure("wpli")


def _load_labels(subject: str, task: str, deriv_root: Path) -> Optional[np.ndarray]:
    patterns = [
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_*connectivity_labels*.npy",
        f"sub-{subject}/eeg/sub-{subject}_task-{task}_connectivity_labels*.npy",
    ]
    for pattern in patterns:
        path = find_first(str((deriv_root / pattern).as_posix()))
        if path:
            return np.load(path, allow_pickle=True)
    return None


def _flatten_lower_triangles(conn_trials, labels, prefix):
    if conn_trials.ndim != 3:
        raise ValueError("Connectivity array must be 3D (trials, nodes, nodes)")
    n_trials, n_nodes, _ = conn_trials.shape
    idx_i, idx_j = np.tril_indices(n_nodes, k=-1)
    out = conn_trials[:, idx_i, idx_j]

    if labels is not None and len(labels) == n_nodes:
        pair_names = [f"{labels[i]}__{labels[j]}" for i, j in zip(idx_i, idx_j)]
    else:
        pair_names = [f"n{i}_n{j}" for i, j in zip(idx_i, idx_j)]
    cols = [f"{prefix}_{p}" for p in pair_names]
    return pd.DataFrame(out), cols


def align_feature_blocks(blocks: List[pd.DataFrame]) -> List[pd.DataFrame]:
    if not blocks:
        return []
    
    n_trials_ref = None
    aligned_blocks = []
    
    for block in blocks:
        if block is None or block.empty:
            continue
        
        if n_trials_ref is None:
            n_trials_ref = len(block)
            aligned_blocks.append(block)
        else:
            min_n = min(n_trials_ref, len(block))
            aligned_blocks.append(block.iloc[:min_n, :])
            aligned_blocks = [b.iloc[:min_n, :] for b in aligned_blocks]
            n_trials_ref = min_n
    
    return aligned_blocks


###################################################################
# Feature Extraction Functions
###################################################################

def extract_baseline_power_features(tfr, bands, baseline_window, config, logger):
    if tfr is None or (isinstance(tfr, list) and len(tfr) == 0):
        return pd.DataFrame(), []

    tfr_obj = tfr[0] if isinstance(tfr, list) else tfr
    data = tfr_obj.data
    if data.ndim != 4:
        raise RuntimeError("TFR data must have 4D shape (epochs, channels, freqs, times)")

    b_start, b_end = baseline_window
    times = tfr_obj.times
    ch_names = tfr_obj.info["ch_names"]
    
    baseline_mask = time_mask(times, b_start, b_end)
    if not np.any(baseline_mask):
        logger.warning(f"No time points in baseline window ({b_start}, {b_end})")
        return pd.DataFrame(), []
    
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    features = []
    colnames = []
    
    for band in bands:
        if band not in features_freq_bands:
            continue
        
        fmin, fmax = features_freq_bands[band]
        freq_mask_idx = freq_mask(tfr_obj.freqs, fmin, fmax)
        if not np.any(freq_mask_idx):
            continue
        
        band_power = data[:, :, freq_mask_idx, :][:, :, :, baseline_mask].mean(axis=(2, 3))
        features.append(band_power)
        colnames.extend([f"baseline_{band}_{ch}" for ch in ch_names])

    if not features:
        return pd.DataFrame(), []

    return pd.DataFrame(np.concatenate(features, axis=1)), colnames


def extract_band_power_features(tfr, bands, config, logger):
    if tfr is None or (isinstance(tfr, list) and len(tfr) == 0):
        return pd.DataFrame(), []

    tfr_obj = tfr[0] if isinstance(tfr, list) else tfr
    data = tfr_obj.data
    if data.ndim != 4:
        raise RuntimeError("TFR data must have 4D shape (epochs, channels, freqs, times)")

    times = tfr_obj.times
    ch_names = tfr_obj.info["ch_names"]
    
    features_freq_bands = config.get("time_frequency_analysis.bands") or config.frequency_bands
    features = []
    colnames = []
    
    for band in bands:
        if band not in features_freq_bands:
            logger.warning(f"Band '{band}' not defined in config; skipping")
            continue
        
        fmin, fmax = features_freq_bands[band]
        freq_mask_idx = freq_mask(tfr_obj.freqs, fmin, fmax)
        if not np.any(freq_mask_idx):
            logger.warning(f"TFR freqs contain no points in band '{band}' ({fmin}-{fmax} Hz); skipping")
            continue
        
        bins = config.get("feature_engineering.features.temporal_bins", [])
        for bin_config in bins:
            if isinstance(bin_config, dict):
                t_start = float(bin_config.get("start", 0.0))
                t_end = float(bin_config.get("end", 0.0))
                t_label = str(bin_config.get("label", "unknown"))
            elif isinstance(bin_config, (list, tuple)) and len(bin_config) >= 3:
                t_start = float(bin_config[0])
                t_end = float(bin_config[1])
                t_label = str(bin_config[2])
            else:
                logger.warning(f"Invalid temporal bin configuration: {bin_config}; skipping")
                continue
            
            time_mask_arr = time_mask(times, t_start, t_end)
            if not np.any(time_mask_arr):
                logger.warning(f"No time points in bin {t_label} ({t_start}-{t_end}s) for band '{band}'")
                continue
            
            band_power = data[:, :, freq_mask_idx, :][:, :, :, time_mask_arr].mean(axis=(2, 3))
            features.append(band_power)
            colnames.extend([f"pow_{band}_{ch}_{t_label}" for ch in ch_names])

    if not features:
        return pd.DataFrame(), []

    return pd.DataFrame(np.concatenate(features, axis=1)), colnames


def extract_connectivity_features(subject: str, task: str, bands, deriv_root, logger):
    subj_dir = deriv_root / f"sub-{subject}" / "eeg"
    if not subj_dir.exists():
        return pd.DataFrame(), []

    labels = _load_labels(subject, task, deriv_root)
    all_blocks: List[pd.DataFrame] = []
    all_cols: List[str] = []

    for band in bands:
        aec_path, wpli_path = _find_connectivity_arrays(subject, task, band, deriv_root)
        
        for measure, path in (("aec", aec_path), ("wpli", wpli_path)):
            if path is None or not path.exists():
                logger.warning(f"Connectivity file missing for {measure} {band}: {path}")
                continue
            
            try:
                arr = np.load(path)
            except (OSError, IOError, ValueError) as e:
                logger.warning(f"Failed to load connectivity file {path}: {e}")
                continue
            
            if arr.ndim != 3:
                logger.warning(f"Unexpected connectivity shape at {path}: {arr.shape}")
                continue
            
            df_flat, cols = _flatten_lower_triangles(arr, labels, prefix=f"{measure}_{band}")
            
            all_blocks.append(df_flat)
            all_cols.extend(cols)

    if not all_blocks:
        return pd.DataFrame(), []

    aligned_blocks = align_feature_blocks(all_blocks)
    if not aligned_blocks:
        return pd.DataFrame(), []

    combined_df = pd.concat(aligned_blocks, axis=1)
    combined_df.columns = all_cols
    return combined_df, all_cols


def extract_microstate_features(epochs, n_states, config, logger):
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for microstate feature extraction")
        return pd.DataFrame(), [], None

    try:
        data = epochs.get_data(picks=picks)
    except TypeError:
        data = epochs.get_data()[:, picks, :]

    if data.size == 0:
        logger.warning("Epoch data empty after picking EEG channels; skipping microstate features")
        return pd.DataFrame(), [], None

    sfreq = float(epochs.info.get("sfreq", 1.0))
    templates = extract_templates_from_trials(data, sfreq, n_states, config)
    if templates is None or len(templates) == 0:
        logger.warning("Failed to derive microstate templates; skipping microstate features")
        return pd.DataFrame(), [], None

    column_order = []
    metrics = ("coverage", "duration", "occurrence", "gev")
    for metric in metrics:
        for state_idx in range(n_states):
            column_order.append(f"ms_{metric}_{state_idx}")
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            column_order.append(f"ms_trans_{i}_to_{j}")

    feature_rows = []
    for trial_idx, trial in enumerate(data):
        record = {col: 0.0 for col in column_order}
        labels, corr_vals = label_timecourse(trial, templates)
        if labels.size == 0:
            feature_rows.append(record)
            continue

        gfp = compute_gfp(trial)
        gfp_energy = float(np.sum(gfp ** 2))
        if not np.isfinite(gfp_energy) or gfp_energy <= 0:
            gfp_energy = 1e-12

        counts = np.bincount(labels, minlength=n_states).astype(float)
        coverage = counts / float(labels.size) if labels.size > 0 else np.zeros(n_states, dtype=float)
        for state_idx in range(n_states):
            record[f"ms_coverage_{state_idx}"] = float(coverage[state_idx])

        occurrences = np.zeros(n_states, dtype=float)
        dwell_time = np.zeros(n_states, dtype=float)

        prev_state = int(labels[0])
        run_len = 1
        for sample_idx in range(1, labels.size):
            state = int(labels[sample_idx])
            if state == prev_state:
                run_len += 1
                continue
            occurrences[prev_state] += 1.0
            dwell_time[prev_state] += run_len / sfreq
            record[f"ms_trans_{prev_state}_to_{state}"] += 1.0
            prev_state = state
            run_len = 1
        occurrences[prev_state] += 1.0
        dwell_time[prev_state] += run_len / sfreq

        for state_idx in range(n_states):
            occ = occurrences[state_idx]
            record[f"ms_occurrence_{state_idx}"] = float(occ)
            record[f"ms_duration_{state_idx}"] = float(dwell_time[state_idx] / occ) if occ > 0 else 0.0

        for state_idx in range(n_states):
            mask = labels == state_idx
            if not np.any(mask):
                record[f"ms_gev_{state_idx}"] = 0.0
                continue
            gev_num = np.sum((corr_vals[mask] ** 2) * (gfp[mask] ** 2))
            record[f"ms_gev_{state_idx}"] = float(gev_num / gfp_energy)

        feature_rows.append(record)

    ms_df = pd.DataFrame.from_records(feature_rows, columns=column_order)
    return ms_df, column_order, templates


def extract_aperiodic_features(epochs, baseline_window, bands, config, logger, fmin: float = 2.0, fmax: float = 40.0, n_fft: Optional[int] = None):
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for aperiodic feature extraction")
        return pd.DataFrame(), []
    
    b_start, b_end = baseline_window
    if b_end > 0:
        b_end = 0.0
    
    freq_bands = config.get("time_frequency_analysis.bands") or {}
    if not freq_bands:
        freq_bands = {
            "delta": [1.0, 3.9],
            "theta": [4.0, 7.9],
            "alpha": [8.0, 12.9],
            "beta": [13.0, 30.0],
            "gamma": [30.1, 80.0],
        }
    
    psds, freqs = mne.time_frequency.psd_welch(
        epochs,
        picks=picks,
        fmin=float(fmin),
        fmax=float(fmax),
        tmin=b_start,
        tmax=b_end,
        n_fft=n_fft,
        n_overlap=None,
        average='mean',
        verbose=False,
    )
    
    if psds.ndim != 3:
        logger.error(f"Unexpected PSD shape: {psds.shape}")
        return pd.DataFrame(), []
    
    eps = 1e-20
    logf = np.log10(freqs)
    logpsd = np.log10(np.maximum(psds, eps))
    
    n_ep, n_ch, n_fr = logpsd.shape
    
    offset = np.full((n_ep, n_ch), np.nan)
    slope = np.full((n_ep, n_ch), np.nan)
    for i in range(n_ep):
        for j in range(n_ch):
            a, b = _fit_aperiodic(logf, logpsd[i, j, :])
            offset[i, j] = a
            slope[i, j] = b
    
    resid = np.empty_like(logpsd)
    for i in range(n_ep):
        for j in range(n_ch):
            resid[i, j, :] = logpsd[i, j, :] - (offset[i, j] + slope[i, j] * logf)
    
    band_masks = {}
    for band in bands:
        if band in freq_bands:
            lo, hi = freq_bands[band]
            band_masks[band] = (freqs >= lo) & (freqs <= hi)
    
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    rows = []
    for i in range(n_ep):
        rec = {}
        for j, ch in enumerate(ch_names):
            rec[f"aper_slope_{ch}"] = float(slope[i, j])
            rec[f"aper_offset_{ch}"] = float(offset[i, j])
        
        for b, mask in band_masks.items():
            if not np.any(mask):
                for j, ch in enumerate(ch_names):
                    rec[f"powcorr_{b}_{ch}"] = np.nan
            else:
                for j, ch in enumerate(ch_names):
                    rec[f"powcorr_{b}_{ch}"] = float(np.mean(resid[i, j, mask]))
        rows.append(rec)
    
    feat_df = pd.DataFrame(rows)
    colnames = list(feat_df.columns)
    return feat_df, colnames

