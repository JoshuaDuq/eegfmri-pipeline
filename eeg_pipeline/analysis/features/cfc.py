"""
Cross-Frequency Coupling Features
==================================

Advanced CFC measures for EEG-fMRI analysis:
- Modulation Index PAC (Tort et al. 2010)
- Phase-Phase Coupling (n:m locking)
- Bicoherence (bispectral analysis)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert
from scipy.stats import entropy as scipy_entropy
from joblib import Parallel, delayed

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.utils.analysis.spectral import bandpass_filter_epochs
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.features.metadata import NamingSchema


def _compute_modulation_index(phase: np.ndarray, amp: np.ndarray, n_bins: int = 18) -> float:
    """Compute Modulation Index (Tort et al. 2010)."""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    # Vectorized binning
    # Digitizing is faster than loop
    indices = np.digitize(phase, bin_edges) - 1
    # Handle edge case pi
    indices[indices == n_bins] = n_bins - 1
    
    # Aggregate amp in bins using bincount?
    # Amp is weighted by itself.
    # We need mean amplitude in each bin.
    # sum(amp) in bin / count in bin.
    
    # Logic:
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Fast aggregation
    # Unweighted bincount of indices gives counts
    # Weighted bincount gives sums
    # Filter out invalid indices if any (shouldn't be)
    mask = (indices >= 0) & (indices < n_bins)
    if np.sum(mask) == 0: return np.nan
    
    valid_ind = indices[mask]
    valid_amp = amp[mask]
    
    bin_counts = np.bincount(valid_ind, minlength=n_bins)
    bin_sums = np.bincount(valid_ind, weights=valid_amp, minlength=n_bins)
    
    # Mean
    with np.errstate(divide='ignore', invalid='ignore'):
        amp_in_bins = bin_sums / bin_counts
        
    # If any bin is empty, it's NaN. Replace with 0 or handle?
    # Tort MI requires distribution. Empty bin means 0 prob?
    # Usually we ignore or fill with mean?
    # Original code skipped empty bins.
    # If sum(amp_in_bins) == 0 -> nan.
    amp_in_bins = np.nan_to_num(amp_in_bins)
    
    if np.sum(amp_in_bins) == 0:
        return np.nan
    
    amp_dist = amp_in_bins / np.sum(amp_in_bins)
    amp_dist = np.clip(amp_dist, 1e-12, None)
    
    h_max = np.log(n_bins)
    h_obs = scipy_entropy(amp_dist)
    mi = (h_max - h_obs) / h_max
    
    return float(mi)


def _compute_glm_pac(phase: np.ndarray, amp: np.ndarray) -> float:
    """Compute GLM-based PAC (Penny et al. 2008)."""
    X = np.column_stack([np.cos(phase), np.sin(phase), np.ones_like(phase)])
    
    try:
        # lstsq is somewhat slow for many calls. 
        # But for 1D array it's okay.
        beta, residuals, rank, s = np.linalg.lstsq(X, amp, rcond=None)
        y_pred = X @ beta
        ss_reg = np.sum((y_pred - np.mean(amp)) ** 2)
        ss_tot = np.sum((amp - np.mean(amp)) ** 2)
        r_squared = ss_reg / (ss_tot + 1e-12)
        return float(r_squared)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def _compute_pac_epoch(
    ep_idx: int, 
    filtered_signals: Dict[str, np.ndarray], 
    phase_bands: List[str], 
    amp_bands: List[str], 
    freq_bands: Dict[str, Tuple[float, float]], 
    ch_names: List[str], 
    n_bins: int
) -> Dict[str, float]:
    """Compute PAC for a single epoch (parallel worker)."""
    record = {}
    
    for phase_band in phase_bands:
        if phase_band not in filtered_signals: continue
        
        p_signal_all = filtered_signals[phase_band][ep_idx] # (n_ch, n_times)
        p_phase_all = np.angle(p_signal_all)
        
        for amp_band in amp_bands:
            if amp_band not in filtered_signals: continue
            
            # Freq check (amp > phase)
            p_fmin, p_fmax = freq_bands[phase_band]
            a_fmin, a_fmax = freq_bands[amp_band]
            if a_fmin <= p_fmax: continue
            
            a_signal_all = filtered_signals[amp_band][ep_idx]
            a_amp_all = np.abs(a_signal_all)
            
            mi_values = []
            for ch_idx in range(len(ch_names)):
                mi = _compute_modulation_index(
                    p_phase_all[ch_idx], 
                    a_amp_all[ch_idx], 
                    n_bins
                )
                if np.isfinite(mi):
                    mi_values.append(mi)
                    
            pair_label = f"{phase_band}_{amp_band}"
            col_mean = NamingSchema.build("cfc", "full", pair_label, "global", "mi_mean")
            col_max = NamingSchema.build("cfc", "full", pair_label, "global", "mi_max")
            record[col_mean] = float(np.mean(mi_values)) if mi_values else np.nan
            record[col_max] = float(np.max(mi_values)) if mi_values else np.nan
            
    return record


def _compute_glm_pac_epoch(
    ep_idx: int,
    filtered_signals: Dict[str, np.ndarray],
    phase_bands: List[str],
    amp_bands: List[str],
    freq_bands: Dict[str, Tuple[float, float]],
    ch_names: List[str]
) -> Dict[str, float]:
    """Compute GLM-PAC for a single epoch (parallel worker)."""
    record = {}
    for phase_band in phase_bands:
        if phase_band not in filtered_signals: continue
        p_phase_all = np.angle(filtered_signals[phase_band][ep_idx])
        
        for amp_band in amp_bands:
            if amp_band not in filtered_signals: continue
            p_fmin, p_fmax = freq_bands[phase_band]
            a_fmin, a_fmax = freq_bands[amp_band]
            if a_fmin <= p_fmax: continue
            
            a_amp_all = np.abs(filtered_signals[amp_band][ep_idx])
            
            glm_values = []
            for ch_idx in range(len(ch_names)):
                r2 = _compute_glm_pac(p_phase_all[ch_idx], a_amp_all[ch_idx])
                if np.isfinite(r2):
                    glm_values.append(r2)
                    
            pair_label = f"{phase_band}_{amp_band}"
            col_mean = NamingSchema.build("cfc", "full", pair_label, "global", "glm_r2_mean")
            record[col_mean] = float(np.mean(glm_values)) if glm_values else np.nan
            
    return record


def extract_modulation_index_pac(
    epochs: mne.Epochs, phase_bands: List[str], amp_bands: List[str], config: Any, logger: Any, *, n_bins: int = 18
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract Modulation Index PAC (Tort et al. 2010)."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]
    n_jobs = int(config.get("feature_engineering.parallel.n_jobs_pac", -1))
    
    # 1. Pre-filter all relevant bands
    # Collect unique bands needed
    all_bands = set(phase_bands) | set(amp_bands)
    filtered_signals = {} # band -> analytic signal (n_epochs, n_ch, n_times)
    
    for band in all_bands:
        if band in freq_bands:
            fmin, fmax = freq_bands[band]
            # Use parallel filtering
            filt = bandpass_filter_epochs(data, sfreq, fmin, fmax, n_jobs=n_jobs)
            if filt is not None:
                # Pre-compute analytic signal to save time?
                # Using hilbert inside loop is okay if loop is parallelized, 
                # but better to do it once if memory allows.
                # Hilbert on large array is fast if parallelized?
                # scipy.signal.hilbert supports axis but not n_jobs.
                # bandpass_filter_epochs returns filtered real data.
                # Let's compute Hilbert here.
                # 1.2 GB per band complex? Too big maybe (data is float64).
                # 64*61*7700*16 bytes ~ 480MB. 
                # 5 bands ~ 2.5 GB. 
                # Acceptable for modern machines.
                filtered_signals[band] = hilbert(filt, axis=-1)
    
    if not filtered_signals:
         return pd.DataFrame(), []

    # Prepare data for parallel execution
    # Convert filtered_signals dict to a more pickle-friendly format if needed,
    # or just pass the dict if it contains numpy arrays (efficiently shared in 'processes' mode on Linux/Mac copy-on-write,
    # but 'loky' pickles. Actually joblib uses memmapping for large arrays automatically).
    
    # We need to construct arguments for each epoch
    
    records = Parallel(n_jobs=n_jobs)(
        delayed(_compute_pac_epoch)(
            i, 
            filtered_signals, 
            phase_bands, 
            amp_bands, 
            freq_bands, 
            ch_names, 
            n_bins
        ) for i in range(n_epochs)
    )
    
    if not records:
        return pd.DataFrame(), []
        
    return pd.DataFrame(records), list(records[0].keys())


def extract_glm_pac(
    epochs: mne.Epochs, phase_bands: List[str], amp_bands: List[str], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract GLM-based PAC."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0: return pd.DataFrame(), []
    
    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]
    n_jobs = int(config.get("feature_engineering.parallel.n_jobs_pac", -1))

    # Pre-filter
    all_bands = set(phase_bands) | set(amp_bands)
    filtered_signals = {}
    
    for band in all_bands:
        if band in freq_bands:
            fmin, fmax = freq_bands[band]
            filt = bandpass_filter_epochs(data, sfreq, fmin, fmax, n_jobs=n_jobs)
            if filt is not None:
                filtered_signals[band] = hilbert(filt, axis=-1)
                
    records = Parallel(n_jobs=n_jobs)(
        delayed(_compute_glm_pac_epoch)(
            i, filtered_signals, phase_bands, amp_bands, freq_bands, ch_names
        ) for i in range(n_epochs)
    )

    
    if not records: return pd.DataFrame(), []
    return pd.DataFrame(records), list(records[0].keys())


def _compute_phase_locking_nm(phase1: np.ndarray, phase2: np.ndarray, n: int, m: int) -> float:
    phase_diff = n * phase1 - m * phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return float(plv)


def _compute_ppc_epoch(
    ep_idx: int,
    filtered_signals: Dict[str, np.ndarray],
    band_pairs: List[Tuple[str, str, int, int]],
    ch_names: List[str]
) -> Dict[str, float]:
    """Compute Phase-Phase Coupling for a single epoch (parallel worker)."""
    record = {}
    for b1, b2, n, m in band_pairs:
        if b1 not in filtered_signals or b2 not in filtered_signals: continue
        
        p1_all = np.angle(filtered_signals[b1][ep_idx])
        p2_all = np.angle(filtered_signals[b2][ep_idx])
        
        ppc_values = []
        for ch_idx in range(len(ch_names)):
            plv = _compute_phase_locking_nm(p1_all[ch_idx], p2_all[ch_idx], n, m)
            if np.isfinite(plv):
                ppc_values.append(plv)
                
        pair_label = f"{b1}_{b2}_{n}to{m}"
        col_mean = NamingSchema.build("cfc", "full", pair_label, "global", "ppc_mean")
        col_max = NamingSchema.build("cfc", "full", pair_label, "global", "ppc_max")
        record[col_mean] = float(np.mean(ppc_values)) if ppc_values else np.nan
        record[col_max] = float(np.max(ppc_values)) if ppc_values else np.nan
        
    return record


def extract_phase_phase_coupling(
    epochs: mne.Epochs, band_pairs: List[Tuple[str, str, int, int]], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract n:m phase-phase coupling."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0: return pd.DataFrame(), []
    
    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]
    n_jobs = int(config.get("feature_engineering.parallel.n_jobs_pac", -1))
    
    # Pre-filter
    needed_bands = set()
    for b1, b2, _, _ in band_pairs:
        needed_bands.add(b1); needed_bands.add(b2)
        
    filtered_signals = {}
    for band in needed_bands:
        if band in freq_bands:
            fmin, fmax = freq_bands[band]
            filt = bandpass_filter_epochs(data, sfreq, fmin, fmax, n_jobs=n_jobs)
            if filt is not None:
                filtered_signals[band] = hilbert(filt, axis=-1)
                
    records = Parallel(n_jobs=n_jobs)(
        delayed(_compute_ppc_epoch)(i, filtered_signals, band_pairs, ch_names) for i in range(n_epochs)
    )
    
    if not records: return pd.DataFrame(), []
    return pd.DataFrame(records), list(records[0].keys())

# Bicoherence: Remains as is or optimized if needed.
# Given length, I will omit re-implementation for brevity unless strict requirement.
# Wait, I must provide full file content.
# I'll implement bicoherence with parallel epoch processing.

def _compute_bicoherence(x: np.ndarray, sfreq: float, fmax: float = 50.0, nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.fft import fft
    if nperseg is None:
        nperseg = min(len(x), int(2 * sfreq))
    n_segments = max(1, len(x) // nperseg)
    nfft = nperseg
    freqs = np.fft.fftfreq(nfft, 1 / sfreq)
    freq_mask = (freqs >= 0) & (freqs <= fmax)
    freqs_pos = freqs[freq_mask]
    n_freqs = len(freqs_pos)
    
    bispectrum = np.zeros((n_freqs, n_freqs), dtype=complex)
    power_f1 = np.zeros(n_freqs)
    power_f2 = np.zeros(n_freqs)
    power_f12 = np.zeros((n_freqs, n_freqs))
    
    for seg in range(n_segments):
        start = seg * nperseg; end = start + nperseg
        if end > len(x): break
        segment = x[start:end]
        X = fft(segment)
        X_pos = X[freq_mask]
        
        # Optimize loop?
        # Outer product to get all i,j pairs?
        # X[i] * X[j] * conj(X[i+j])
        # Indices i, j such that i+j < n_freqs
        
        # Vectorized implementation for speed:
        # Construct (i,j) meshgrid or masked
        # This is hard to fully vectorize without huge memory.
        # Stick to loops for stability or Numba.
        # Since we parallelize epochs, loops are acceptable here.
        for i in range(n_freqs):
           for j in range(i, n_freqs):
               if i + j < n_freqs:
                   val = X_pos[i] * X_pos[j] * np.conj(X_pos[i + j])
                   bispectrum[i, j] += val
                   power_f12[i, j] += np.abs(X_pos[i + j]) ** 2
           power_f1[i] += np.abs(X_pos[i]) ** 2
        power_f2 += np.abs(X_pos) ** 2
        
    denom = np.sqrt(np.outer(power_f1, power_f2) * power_f12 + 1e-12)
    bicoherence = np.abs(bispectrum) / denom
    return bicoherence, freqs_pos


def _compute_bicoherence_epoch(
    ep_idx: int,
    data: np.ndarray,
    sfreq: float,
    fmax: float,
    ch_names: List[str]
) -> Dict[str, float]:
    """Compute bicoherence for a single epoch (parallel worker)."""
    record = {}
    bic_values = []
    # epoch data: (n_ch, n_times)
    epoch_data = data[ep_idx]
    
    for ch_idx in range(len(ch_names)):
        try:
            bic, freqs = _compute_bicoherence(epoch_data[ch_idx], sfreq, fmax)
            bic_values.append(bic)
        except (ValueError, RuntimeError):
            continue
            
    if bic_values:
        mean_bic = np.nanmean(bic_values, axis=0)
        record["bicoherence_mean"] = float(np.nanmean(mean_bic))
        record["bicoherence_max"] = float(np.nanmax(mean_bic))
        
        # Peak location
        idx = np.nanargmax(mean_bic)
        i_idx, j_idx = np.unravel_index(idx, mean_bic.shape)
        record["bicoherence_peak_f1"] = float(freqs[i_idx]) if i_idx < len(freqs) else np.nan
        record["bicoherence_peak_f2"] = float(freqs[j_idx]) if j_idx < len(freqs) else np.nan

    else:
        record["bicoherence_mean"] = np.nan
        record["bicoherence_max"] = np.nan
        
    return record


def extract_bicoherence_features(
    epochs: mne.Epochs, config: Any, logger: Any, *, freq_range: Tuple[float, float] = (1, 50)
) -> Tuple[pd.DataFrame, List[str]]:
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0: return pd.DataFrame(), []
    
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]
    n_jobs = int(config.get("feature_engineering.parallel.n_jobs_pac", -1))
    fmax = min(freq_range[1], sfreq / 2 - 1)
    
    records = Parallel(n_jobs=n_jobs)(
        delayed(_compute_bicoherence_epoch)(i, data, sfreq, fmax, ch_names) for i in range(n_epochs)
    )
    
    return pd.DataFrame(records), list(records[0].keys()) if records else []


# =============================================================================
# Precomputed Data Extractors (Moved from pipeline.py)
# =============================================================================

def extract_pac_from_precomputed(
    precomputed: Any, # PrecomputedData
    config: Any
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute phase-amplitude coupling using multiple estimators on precomputed analytic signals."""
    if not precomputed.band_data:
        return pd.DataFrame(), []

    logger = getattr(precomputed, "logger", None)

    def _apply_time_mask(arr_2d: np.ndarray, time_mask: Any) -> np.ndarray:
        if arr_2d.ndim != 2:
            raise ValueError(f"Expected 2D array (ch, time); got shape {arr_2d.shape}")
        if time_mask is None:
            return arr_2d
        if isinstance(time_mask, slice):
            return arr_2d[:, time_mask]
        if isinstance(time_mask, np.ndarray):
            if time_mask.dtype != bool:
                raise ValueError("Time mask must be boolean")
            return arr_2d[:, time_mask]
        raise TypeError(f"Unsupported mask type: {type(time_mask)}")

    pac_cfg = config.get("feature_engineering.pac", {})
    cfc_cfg = config.get("feature_engineering.cross_frequency", {})
    n_bins = int(pac_cfg.get("n_bins", cfc_cfg.get("n_bins", 18)))
    default_pairs = pac_cfg.get("pairs")
    if default_pairs is None:
        default_pairs = [("theta", "gamma"), ("alpha", "gamma")]
    
    # Filter valid pairs
    pairs = []
    for p in default_pairs:
        if p and len(p) >= 2:
            pairs.append((p[0], p[1]))

    if not pairs:
        return pd.DataFrame(), []

    active_mask = getattr(precomputed.windows, "active_mask", None)
    if active_mask is None or not np.any(active_mask):
        active_mask = slice(None)

    raw_estimators = pac_cfg.get("estimators", ["tort", "cl_corr"])
    if raw_estimators is None:
        raw_estimators = []

    est_set = set()
    for est in raw_estimators:
        if est is None:
            continue
        est_norm = str(est).strip().lower()
        if est_norm in {"clcorr", "cl_corr", "cl-corr"}:
            est_norm = "cl_corr"
        est_set.add(est_norm)

    supported = {"tort", "cl_corr"}
    unsupported = sorted([e for e in est_set if e not in supported])
    if unsupported and logger is not None:
        logger.warning(
            "PAC (precomputed): ignoring unsupported estimators=%s. Supported=%s",
            ", ".join(unsupported),
            ", ".join(sorted(supported)),
        )
    estimators = {e for e in est_set if e in supported}
    records: List[Dict[str, float]] = []
    
    for ep_idx in range(precomputed.data.shape[0]):
        record: Dict[str, float] = {}
        for phase_band, amp_band in pairs:
            if phase_band not in precomputed.band_data or amp_band not in precomputed.band_data:
                continue

            phase = precomputed.band_data[phase_band].phase[ep_idx]
            amp_env = precomputed.band_data[amp_band].envelope[ep_idx]
            
            # Use mask
            phase_seg = _apply_time_mask(phase, active_mask)
            amp_seg = _apply_time_mask(amp_env, active_mask)

            if phase_seg.shape[1] == 0 or amp_seg.shape[1] == 0:
                continue

            for ch_idx, ch_name in enumerate(precomputed.ch_names):
                ph = phase_seg[ch_idx]
                amp = amp_seg[ch_idx]
                
                # Tort/MI (KL divergence)
                if "tort" in estimators:
                    # Reuse _compute_modulation_index helper if available, or inline
                    # Since _compute_modulation_index is defined in this file (cfc.py), we can use it!
                    mi = _compute_modulation_index(ph, amp, n_bins)
                    col = NamingSchema.build(
                        "pac",
                        "plateau",
                        f"{phase_band}_{amp_band}",
                        "ch",
                        "hilbert_mi",
                        channel=str(ch_name),
                    )
                    record[col] = mi

                # Circular-linear correlation (phase vs amplitude)
                if "cl_corr" in estimators:
                    sin_ph = np.sin(ph)
                    cos_ph = np.cos(ph)
                    r_num = (np.corrcoef(sin_ph, amp)[0, 1] ** 2 + np.corrcoef(cos_ph, amp)[0, 1] ** 2)
                    clc = float(np.sqrt(max(r_num, 0.0)))
                    col = NamingSchema.build(
                        "pac",
                        "plateau",
                        f"{phase_band}_{amp_band}",
                        "ch",
                        "hilbert_clcorr",
                        channel=str(ch_name),
                    )
                    record[col] = clc
        
        # Global summaries per epoch across channels
        if record:
            for phase_band, amp_band in pairs:
                pair_label = f"{phase_band}_{amp_band}"
                for stat in ("hilbert_mi", "hilbert_clcorr"):
                    prefix = f"pac_plateau_{pair_label}_ch_"
                    vals = [
                        v
                        for k, v in record.items()
                        if str(k).startswith(prefix) and str(k).endswith(f"_{stat}")
                    ]
                    if vals:
                        col_g = NamingSchema.build(
                            "pac",
                            "plateau",
                            pair_label,
                            "global",
                            f"{stat}_mean",
                        )
                        record[col_g] = float(np.nanmean(vals))
        records.append(record)

    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []

    df = pd.DataFrame(records)
    return df, list(df.columns)


def extract_all_cfc_features(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
    *,
    include_pac: bool = True,
    include_aac: bool = True,
    include_ppc: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract all cross-frequency coupling features.
    
    Combines multiple CFC measures:
    - PAC: Phase-Amplitude Coupling (Modulation Index)
    - AAC: Amplitude-Amplitude Coupling (via GLM-PAC proxy)
    - PPC: Phase-Phase Coupling (n:m locking)
    """
    dfs = []
    all_cols = []
    
    freq_bands = get_frequency_bands(config)
    cfc_cfg = config.get("feature_engineering.cross_frequency", {})
    phase_bands = [b for b in cfc_cfg.get("phase_bands", ["theta", "alpha"]) if b in freq_bands]
    amp_bands = [b for b in cfc_cfg.get("amp_bands", ["gamma"]) if b in freq_bands]
    n_bins = int(cfc_cfg.get("n_bins", 18))
    
    if include_pac and phase_bands and amp_bands:
        logger.info("Extracting Modulation Index PAC...")
        pac_df, pac_cols = extract_modulation_index_pac(
            epochs, phase_bands, amp_bands, config, logger, n_bins=n_bins
        )
        if not pac_df.empty:
            dfs.append(pac_df)
            all_cols.extend(pac_cols)
    
    if include_aac and phase_bands and amp_bands:
        logger.info("Extracting GLM-PAC...")
        glm_df, glm_cols = extract_glm_pac(epochs, phase_bands, amp_bands, config, logger)
        if not glm_df.empty:
            dfs.append(glm_df)
            all_cols.extend(glm_cols)
    
    if include_ppc:
        logger.info("Extracting Phase-Phase Coupling...")
        band_pairs = cfc_cfg.get(
            "phase_phase_pairs",
            [
                ["theta", "alpha", 1, 2],
                ["theta", "beta", 1, 3],
                ["alpha", "beta", 1, 2],
            ],
        )
        valid_pairs = []
        for entry in band_pairs:
            if not entry or len(entry) < 4:
                continue
            b1, b2, n, m = entry[0], entry[1], entry[2], entry[3]
            if b1 in freq_bands and b2 in freq_bands:
                try:
                    valid_pairs.append((str(b1), str(b2), int(n), int(m)))
                except Exception:
                    continue
        if valid_pairs:
            ppc_df, ppc_cols = extract_phase_phase_coupling(epochs, valid_pairs, config, logger)
            if not ppc_df.empty:
                dfs.append(ppc_df)
                all_cols.extend(ppc_cols)
    
    if not dfs:
        return pd.DataFrame(), []
    
    combined = pd.concat(dfs, axis=1)
    return combined, all_cols
