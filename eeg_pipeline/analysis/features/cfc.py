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

from eeg_pipeline.analysis.features.core import pick_eeg_channels, bandpass_filter_epochs
from eeg_pipeline.utils.config.loader import get_frequency_bands


def _compute_modulation_index(phase: np.ndarray, amp: np.ndarray, n_bins: int = 18) -> float:
    """Compute Modulation Index (Tort et al. 2010)."""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    amp_in_bins = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) > 0:
            amp_in_bins[i] = np.mean(amp[mask])
    
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
        beta, residuals, rank, s = np.linalg.lstsq(X, amp, rcond=None)
        y_pred = X @ beta
        ss_reg = np.sum((y_pred - np.mean(amp)) ** 2)
        ss_tot = np.sum((amp - np.mean(amp)) ** 2)
        r_squared = ss_reg / (ss_tot + 1e-12)
        return float(r_squared)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def extract_modulation_index_pac(
    epochs: mne.Epochs, phase_bands: List[str], amp_bands: List[str], config: Any, logger: Any, *, n_bins: int = 18
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract Modulation Index PAC (Tort et al. 2010) for phase-amplitude coupling."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels for MI-PAC extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]

    feature_records: List[Dict[str, float]] = []

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        ep_data = data[ep_idx]

        for phase_band in phase_bands:
            if phase_band not in freq_bands:
                continue
            p_fmin, p_fmax = freq_bands[phase_band]

            for amp_band in amp_bands:
                if amp_band not in freq_bands:
                    continue
                a_fmin, a_fmax = freq_bands[amp_band]

                if a_fmin <= p_fmax:
                    continue

                mi_values = []
                for ch_idx in range(len(ch_names)):
                    ch_data = ep_data[ch_idx]

                    phase_filt = bandpass_filter_epochs(ch_data.reshape(1, -1), sfreq, p_fmin, p_fmax)
                    amp_filt = bandpass_filter_epochs(ch_data.reshape(1, -1), sfreq, a_fmin, a_fmax)

                    if phase_filt is None or amp_filt is None:
                        continue

                    phase_signal = np.angle(hilbert(phase_filt.flatten()))
                    amp_signal = np.abs(hilbert(amp_filt.flatten()))

                    mi = _compute_modulation_index(phase_signal, amp_signal, n_bins)
                    if np.isfinite(mi):
                        mi_values.append(mi)

                key = f"mi_pac_{phase_band}_{amp_band}"
                record[f"{key}_mean"] = float(np.mean(mi_values)) if mi_values else np.nan
                record[f"{key}_max"] = float(np.max(mi_values)) if mi_values else np.nan

        feature_records.append(record)

    columns = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), columns


def extract_glm_pac(
    epochs: mne.Epochs, phase_bands: List[str], amp_bands: List[str], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract GLM-based PAC (R² of amplitude ~ cos(phase) + sin(phase))."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels for GLM-PAC extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]

    feature_records: List[Dict[str, float]] = []

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        ep_data = data[ep_idx]

        for phase_band in phase_bands:
            if phase_band not in freq_bands:
                continue
            p_fmin, p_fmax = freq_bands[phase_band]

            for amp_band in amp_bands:
                if amp_band not in freq_bands:
                    continue
                a_fmin, a_fmax = freq_bands[amp_band]

                if a_fmin <= p_fmax:
                    continue

                glm_values = []
                for ch_idx in range(len(ch_names)):
                    ch_data = ep_data[ch_idx]

                    phase_filt = bandpass_filter_epochs(ch_data.reshape(1, -1), sfreq, p_fmin, p_fmax)
                    amp_filt = bandpass_filter_epochs(ch_data.reshape(1, -1), sfreq, a_fmin, a_fmax)

                    if phase_filt is None or amp_filt is None:
                        continue

                    phase_signal = np.angle(hilbert(phase_filt.flatten()))
                    amp_signal = np.abs(hilbert(amp_filt.flatten()))

                    r2 = _compute_glm_pac(phase_signal, amp_signal)
                    if np.isfinite(r2):
                        glm_values.append(r2)

                key = f"glm_pac_{phase_band}_{amp_band}"
                record[f"{key}_mean"] = float(np.mean(glm_values)) if glm_values else np.nan

        feature_records.append(record)

    columns = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), columns


def _compute_phase_locking_nm(phase1: np.ndarray, phase2: np.ndarray, n: int, m: int) -> float:
    """Compute n:m phase locking value."""
    phase_diff = n * phase1 - m * phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return float(plv)


def extract_phase_phase_coupling(
    epochs: mne.Epochs, band_pairs: List[Tuple[str, str, int, int]], config: Any, logger: Any
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract n:m phase-phase coupling between frequency bands.
    
    band_pairs: List of (band1, band2, n, m) for n:m phase locking ratio
    """
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels for PPC extraction")
        return pd.DataFrame(), []

    freq_bands = get_frequency_bands(config)
    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]

    feature_records: List[Dict[str, float]] = []

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        ep_data = data[ep_idx]

        for band1, band2, n_ratio, m_ratio in band_pairs:
            if band1 not in freq_bands or band2 not in freq_bands:
                continue

            f1_min, f1_max = freq_bands[band1]
            f2_min, f2_max = freq_bands[band2]

            ppc_values = []
            for ch_idx in range(len(ch_names)):
                ch_data = ep_data[ch_idx]

                filt1 = bandpass_filter_epochs(ch_data.reshape(1, -1), sfreq, f1_min, f1_max)
                filt2 = bandpass_filter_epochs(ch_data.reshape(1, -1), sfreq, f2_min, f2_max)

                if filt1 is None or filt2 is None:
                    continue

                phase1 = np.angle(hilbert(filt1.flatten()))
                phase2 = np.angle(hilbert(filt2.flatten()))

                plv = _compute_phase_locking_nm(phase1, phase2, n_ratio, m_ratio)
                if np.isfinite(plv):
                    ppc_values.append(plv)

            key = f"ppc_{band1}_{band2}_{n_ratio}to{m_ratio}"
            record[f"{key}_mean"] = float(np.mean(ppc_values)) if ppc_values else np.nan
            record[f"{key}_max"] = float(np.max(ppc_values)) if ppc_values else np.nan

        feature_records.append(record)

    columns = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), columns


def _compute_bicoherence(x: np.ndarray, sfreq: float, fmax: float = 50.0, nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bicoherence matrix."""
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
        start = seg * nperseg
        end = start + nperseg
        if end > len(x):
            break
            
        segment = x[start:end]
        X = fft(segment)
        X_pos = X[freq_mask]
        
        for i in range(n_freqs):
            for j in range(i, n_freqs):
                if i + j < n_freqs:
                    bispectrum[i, j] += X_pos[i] * X_pos[j] * np.conj(X_pos[i + j])
                    power_f12[i, j] += np.abs(X_pos[i + j]) ** 2
            power_f1[i] += np.abs(X_pos[i]) ** 2
        power_f2 += np.abs(X_pos) ** 2
    
    denom = np.sqrt(np.outer(power_f1, power_f2) * power_f12 + 1e-12)
    bicoherence = np.abs(bispectrum) / denom
    
    return bicoherence, freqs_pos


def extract_bicoherence_features(
    epochs: mne.Epochs, config: Any, logger: Any, *, freq_range: Tuple[float, float] = (1, 50)
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract bicoherence features for nonlinear cross-frequency interactions."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels for bicoherence extraction")
        return pd.DataFrame(), []

    data = epochs.get_data(picks=picks)
    sfreq = float(epochs.info["sfreq"])
    n_epochs = data.shape[0]
    fmax = min(freq_range[1], sfreq / 2 - 1)

    feature_records: List[Dict[str, float]] = []

    for ep_idx in range(n_epochs):
        record: Dict[str, float] = {}
        ep_data = data[ep_idx]

        bic_values = []
        for ch_idx in range(len(ch_names)):
            try:
                bic, freqs = _compute_bicoherence(ep_data[ch_idx], sfreq, fmax)
                bic_values.append(bic)
            except (ValueError, RuntimeError):
                continue

        if bic_values:
            mean_bic = np.nanmean(bic_values, axis=0)
            record["bicoherence_mean"] = float(np.nanmean(mean_bic))
            record["bicoherence_max"] = float(np.nanmax(mean_bic))
            record["bicoherence_peak_f1"] = float(np.unravel_index(np.nanargmax(mean_bic), mean_bic.shape)[0])
            record["bicoherence_peak_f2"] = float(np.unravel_index(np.nanargmax(mean_bic), mean_bic.shape)[1])
        else:
            record["bicoherence_mean"] = np.nan
            record["bicoherence_max"] = np.nan
            record["bicoherence_peak_f1"] = np.nan
            record["bicoherence_peak_f2"] = np.nan

        feature_records.append(record)

    columns = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), columns

