"""
Directed Connectivity Feature Extraction
=========================================

Computes directed (causal) connectivity features from EEG data:
- Phase Slope Index (PSI): Direction of information flow based on phase slope
- Directed Transfer Function (DTF): Multivariate autoregressive model-based directionality

These measures capture the direction of information flow between brain regions,
which is critical for understanding pain processing pathways (e.g., S1 → insula → PFC).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.windowing import get_segment_masks

try:
    from mne_connectivity import spectral_connectivity_epochs
except ImportError:
    spectral_connectivity_epochs = None


###################################################################
# Phase Slope Index (PSI) Computation
###################################################################

def _compute_cross_spectrum(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_fft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-spectral density matrix using Welch's method.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    fmin : float
        Minimum frequency of interest
    fmax : float
        Maximum frequency of interest
    n_fft : int, optional
        FFT length. If None, uses n_times.
        
    Returns
    -------
    csd : np.ndarray
        Cross-spectral density of shape (n_epochs, n_channels, n_channels, n_freqs)
    freqs : np.ndarray
        Frequency vector
    """
    from scipy.signal import csd as scipy_csd
    
    n_epochs, n_channels, n_times = data.shape
    
    if n_fft is None:
        n_fft = min(n_times, int(sfreq * 2))
    n_fft = max(n_fft, 64)
    
    n_overlap = n_fft // 2
    
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_band = freqs[freq_mask]
    n_freqs = len(freqs_band)
    
    if n_freqs < 2:
        return np.array([]), freqs_band
    
    csd_matrix = np.zeros((n_epochs, n_channels, n_channels, n_freqs), dtype=complex)
    
    for ep_idx in range(n_epochs):
        for i in range(n_channels):
            for j in range(n_channels):
                _, csd_ij = scipy_csd(
                    data[ep_idx, i],
                    data[ep_idx, j],
                    fs=sfreq,
                    nperseg=n_fft,
                    noverlap=n_overlap,
                    return_onesided=True,
                )
                csd_matrix[ep_idx, i, j, :] = csd_ij[freq_mask]
    
    return csd_matrix, freqs_band


def _compute_psi_from_csd(
    csd: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Compute Phase Slope Index from cross-spectral density.
    
    PSI measures the direction of information flow based on the slope of the
    phase spectrum. Positive PSI(i→j) indicates information flows from i to j.
    
    Reference: Nolte et al. (2008) "Robustly estimating the flow direction of 
    information in complex physical systems"
    
    Parameters
    ----------
    csd : np.ndarray
        Cross-spectral density of shape (n_epochs, n_channels, n_channels, n_freqs)
    freqs : np.ndarray
        Frequency vector
        
    Returns
    -------
    psi : np.ndarray
        Phase slope index of shape (n_epochs, n_channels, n_channels)
        Positive values indicate flow from row to column index.
    """
    n_epochs, n_channels, _, n_freqs = csd.shape
    
    if n_freqs < 2:
        return np.full((n_epochs, n_channels, n_channels), np.nan)
    
    psi = np.zeros((n_epochs, n_channels, n_channels))
    
    for ep_idx in range(n_epochs):
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    continue
                    
                csd_ij = csd[ep_idx, i, j, :]
                
                coherency = csd_ij / np.sqrt(
                    np.abs(csd[ep_idx, i, i, :]) * np.abs(csd[ep_idx, j, j, :]) + 1e-12
                )
                
                phase = np.angle(coherency)
                
                phase_unwrapped = np.unwrap(phase)
                
                if len(freqs) > 1:
                    slope, _ = np.polyfit(freqs, phase_unwrapped, 1)
                else:
                    slope = 0.0
                
                psi[ep_idx, i, j] = slope
    
    return psi


def _compute_psi_imaginary(
    csd: np.ndarray,
) -> np.ndarray:
    """
    Compute Phase Slope Index using imaginary part of coherency.
    
    This is a more robust version that uses the imaginary part of coherency
    to reduce sensitivity to volume conduction (zero-lag effects).
    
    Parameters
    ----------
    csd : np.ndarray
        Cross-spectral density of shape (n_epochs, n_channels, n_channels, n_freqs)
        
    Returns
    -------
    psi : np.ndarray
        Phase slope index of shape (n_epochs, n_channels, n_channels)
    """
    n_epochs, n_channels, _, n_freqs = csd.shape
    
    if n_freqs < 2:
        return np.full((n_epochs, n_channels, n_channels), np.nan)
    
    psi = np.zeros((n_epochs, n_channels, n_channels))
    
    for ep_idx in range(n_epochs):
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    continue
                
                csd_ij = csd[ep_idx, i, j, :]
                
                norm = np.sqrt(
                    np.abs(csd[ep_idx, i, i, :]) * np.abs(csd[ep_idx, j, j, :]) + 1e-12
                )
                coherency = csd_ij / norm
                
                imag_coh = np.imag(coherency)
                
                psi_sum = 0.0
                for f_idx in range(n_freqs - 1):
                    psi_sum += imag_coh[f_idx] * np.conj(coherency[f_idx + 1])
                
                psi[ep_idx, i, j] = np.imag(psi_sum)
    
    return psi


###################################################################
# Directed Transfer Function (DTF) Computation
###################################################################

def _fit_mvar_model(
    data: np.ndarray,
    order: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Multivariate Autoregressive (MVAR) model using Yule-Walker equations.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_channels, n_times)
    order : int
        Model order (number of lags)
        
    Returns
    -------
    A : np.ndarray
        AR coefficients of shape (order, n_channels, n_channels)
    sigma : np.ndarray
        Residual covariance of shape (n_channels, n_channels)
    """
    n_channels, n_times = data.shape
    
    if n_times <= order * n_channels:
        return np.array([]), np.array([])
    
    data_centered = data - data.mean(axis=1, keepdims=True)
    
    R = np.zeros((order + 1, n_channels, n_channels))
    for lag in range(order + 1):
        if lag == 0:
            R[0] = np.dot(data_centered, data_centered.T) / n_times
        else:
            R[lag] = np.dot(data_centered[:, lag:], data_centered[:, :-lag].T) / (n_times - lag)
    
    block_size = n_channels
    R_matrix = np.zeros((order * block_size, order * block_size))
    r_vector = np.zeros((order * block_size, block_size))
    
    for i in range(order):
        for j in range(order):
            lag = abs(i - j)
            if i >= j:
                R_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = R[lag]
            else:
                R_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = R[lag].T
        r_vector[i*block_size:(i+1)*block_size, :] = R[i + 1]
    
    try:
        A_flat = np.linalg.solve(R_matrix, r_vector)
    except np.linalg.LinAlgError:
        return np.array([]), np.array([])
    
    A = np.zeros((order, n_channels, n_channels))
    for i in range(order):
        A[i] = A_flat[i*block_size:(i+1)*block_size, :].T
    
    sigma = R[0].copy()
    for i in range(order):
        sigma -= np.dot(A[i], R[i + 1].T)
    
    return A, sigma


def _compute_dtf_from_mvar(
    A: np.ndarray,
    sigma: np.ndarray,
    freqs: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Compute Directed Transfer Function from MVAR coefficients.
    
    DTF measures the causal influence from channel j to channel i at each frequency.
    
    Reference: Kaminski & Blinowska (1991) "A new method of the description of the 
    information flow in the brain structures"
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients of shape (order, n_channels, n_channels)
    sigma : np.ndarray
        Residual covariance of shape (n_channels, n_channels)
    freqs : np.ndarray
        Frequency vector
    sfreq : float
        Sampling frequency
        
    Returns
    -------
    dtf : np.ndarray
        Directed transfer function of shape (n_channels, n_channels, n_freqs)
        DTF[i, j, f] = causal influence from j to i at frequency f
    """
    if A.size == 0:
        return np.array([])
    
    order, n_channels, _ = A.shape
    n_freqs = len(freqs)
    
    H = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
    
    for f_idx, freq in enumerate(freqs):
        A_f = np.eye(n_channels, dtype=complex)
        for k in range(order):
            A_f -= A[k] * np.exp(-2j * np.pi * freq * (k + 1) / sfreq)
        
        try:
            H[:, :, f_idx] = np.linalg.inv(A_f)
        except np.linalg.LinAlgError:
            H[:, :, f_idx] = np.nan
    
    dtf = np.zeros((n_channels, n_channels, n_freqs))
    
    for f_idx in range(n_freqs):
        H_f = H[:, :, f_idx]
        
        for i in range(n_channels):
            norm = np.sqrt(np.sum(np.abs(H_f[i, :]) ** 2) + 1e-12)
            for j in range(n_channels):
                dtf[i, j, f_idx] = np.abs(H_f[i, j]) / norm
    
    return dtf


def _compute_pdc_from_mvar(
    A: np.ndarray,
    sigma: np.ndarray,
    freqs: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Compute Partial Directed Coherence from MVAR coefficients.
    
    PDC measures the direct causal influence from channel j to channel i,
    partialling out indirect effects through other channels.
    
    Reference: Baccala & Sameshima (2001) "Partial directed coherence: a new 
    concept in neural structure determination"
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients of shape (order, n_channels, n_channels)
    sigma : np.ndarray
        Residual covariance of shape (n_channels, n_channels)
    freqs : np.ndarray
        Frequency vector
    sfreq : float
        Sampling frequency
        
    Returns
    -------
    pdc : np.ndarray
        Partial directed coherence of shape (n_channels, n_channels, n_freqs)
        PDC[i, j, f] = direct causal influence from j to i at frequency f
    """
    if A.size == 0:
        return np.array([])
    
    order, n_channels, _ = A.shape
    n_freqs = len(freqs)
    
    pdc = np.zeros((n_channels, n_channels, n_freqs))
    
    for f_idx, freq in enumerate(freqs):
        A_f = np.eye(n_channels, dtype=complex)
        for k in range(order):
            A_f -= A[k] * np.exp(-2j * np.pi * freq * (k + 1) / sfreq)
        
        for j in range(n_channels):
            norm = np.sqrt(np.sum(np.abs(A_f[:, j]) ** 2) + 1e-12)
            for i in range(n_channels):
                pdc[i, j, f_idx] = np.abs(A_f[i, j]) / norm
    
    return pdc


###################################################################
# Main Extraction Functions
###################################################################

def _compute_directed_connectivity_epoch(
    ep_idx: int,
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_freqs: int,
    mvar_order: int,
    methods: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute directed connectivity for a single epoch.
    
    Parameters
    ----------
    ep_idx : int
        Epoch index (for logging)
    data : np.ndarray
        EEG data of shape (n_channels, n_times)
    sfreq : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    n_freqs : int
        Number of frequency bins
    mvar_order : int
        MVAR model order
    methods : List[str]
        List of methods to compute ('psi', 'dtf', 'pdc')
        
    Returns
    -------
    results : Dict[str, np.ndarray]
        Dictionary mapping method names to connectivity matrices
    """
    results = {}
    n_channels = data.shape[0]
    
    freqs = np.linspace(fmin, fmax, n_freqs)
    
    if "psi" in methods:
        csd, freqs_csd = _compute_cross_spectrum(
            data[np.newaxis, :, :], sfreq, fmin, fmax
        )
        if csd.size > 0:
            psi = _compute_psi_imaginary(csd)
            results["psi"] = psi[0]
        else:
            results["psi"] = np.full((n_channels, n_channels), np.nan)
    
    if "dtf" in methods or "pdc" in methods:
        A, sigma = _fit_mvar_model(data, mvar_order)
        
        if A.size > 0:
            if "dtf" in methods:
                dtf = _compute_dtf_from_mvar(A, sigma, freqs, sfreq)
                if dtf.size > 0:
                    results["dtf"] = np.nanmean(dtf, axis=2)
                else:
                    results["dtf"] = np.full((n_channels, n_channels), np.nan)
            
            if "pdc" in methods:
                pdc = _compute_pdc_from_mvar(A, sigma, freqs, sfreq)
                if pdc.size > 0:
                    results["pdc"] = np.nanmean(pdc, axis=2)
                else:
                    results["pdc"] = np.full((n_channels, n_channels), np.nan)
        else:
            if "dtf" in methods:
                results["dtf"] = np.full((n_channels, n_channels), np.nan)
            if "pdc" in methods:
                results["pdc"] = np.full((n_channels, n_channels), np.nan)
    
    return results


def extract_directed_connectivity_from_precomputed(
    precomputed: Any,
    *,
    bands: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    config: Any = None,
    logger: Any = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract directed connectivity features from precomputed data.
    
    Parameters
    ----------
    precomputed : PrecomputedData
        Precomputed intermediate data with band-filtered signals
    bands : List[str], optional
        Frequency bands to process
    segments : List[str], optional
        Time segments to process
    config : Any
        Configuration object
    logger : Any
        Logger instance
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with directed connectivity features
    columns : List[str]
        List of feature column names
    """
    if not precomputed.band_data:
        return pd.DataFrame(), []
    
    config = config or getattr(precomputed, "config", None) or {}
    logger = logger or getattr(precomputed, "logger", None)
    
    bands_use = (
        list(precomputed.band_data.keys()) 
        if bands is None 
        else [b for b in bands if b in precomputed.band_data]
    )
    if not bands_use:
        return pd.DataFrame(), []
    
    conn_cfg = config.get("feature_engineering.connectivity", {}) if hasattr(config, "get") else {}
    directed_cfg = config.get("feature_engineering.directed_connectivity", {}) if hasattr(config, "get") else {}
    
    enable_psi = bool(directed_cfg.get("enable_psi", True))
    enable_dtf = bool(directed_cfg.get("enable_dtf", False))
    enable_pdc = bool(directed_cfg.get("enable_pdc", False))
    
    methods = []
    if enable_psi:
        methods.append("psi")
    if enable_dtf:
        methods.append("dtf")
    if enable_pdc:
        methods.append("pdc")
    
    if not methods:
        if logger is not None:
            logger.info("Directed connectivity: no methods enabled; skipping extraction.")
        return pd.DataFrame(), []
    
    output_level = str(directed_cfg.get("output_level", "full")).strip().lower()
    if output_level not in {"full", "global_only"}:
        output_level = "full"
    
    mvar_order = int(directed_cfg.get("mvar_order", 10))
    n_freqs = int(directed_cfg.get("n_freqs", 16))
    min_segment_samples = int(directed_cfg.get("min_segment_samples", 100))
    
    try:
        sfreq = float(getattr(precomputed, "sfreq", None))
    except Exception:
        if logger is not None:
            logger.error("Directed connectivity: invalid sampling frequency.")
        return pd.DataFrame(), []
    
    if not np.isfinite(sfreq) or sfreq <= 0:
        return pd.DataFrame(), []
    
    ch_names = list(getattr(precomputed, "ch_names", []))
    n_channels = len(ch_names)
    if n_channels < 2:
        if logger is not None:
            logger.warning("Directed connectivity: fewer than 2 channels; skipping.")
        return pd.DataFrame(), []
    
    pair_i, pair_j = np.triu_indices(n_channels, k=1)
    pair_names = [f"{ch_names[i]}-{ch_names[j]}" for i, j in zip(pair_i, pair_j)]
    
    freq_bands = getattr(precomputed, "frequency_bands", None) or get_frequency_bands(config)
    
    target_name = getattr(precomputed.windows, "name", None) if precomputed.windows else None
    
    if target_name:
        seg_mask_map = {target_name: np.ones(len(precomputed.times), dtype=bool)}
    else:
        masks = get_segment_masks(precomputed.times, precomputed.windows, precomputed.config)
        seg_mask_map = {k: v for k, v in masks.items() if v is not None}
    
    segments_use = segments if segments is not None else sorted(seg_mask_map.keys()) or ["full"]
    
    n_epochs = int(precomputed.data.shape[0])
    records: List[Dict[str, float]] = [dict() for _ in range(n_epochs)]
    
    if logger is not None:
        logger.info(
            "Directed connectivity extraction: epochs=%d, channels=%d, bands=%d, "
            "segments=%d, methods=%s",
            n_epochs, n_channels, len(bands_use), len(segments_use), methods
        )
    
    t0 = time.perf_counter()
    
    for seg_name in segments_use:
        seg_mask = seg_mask_map.get(seg_name)
        if seg_mask is None and seg_name == "full":
            seg_data = precomputed.data
        elif seg_mask is not None and np.any(seg_mask):
            seg_data = precomputed.data[:, :, seg_mask]
        else:
            continue
        
        if seg_data.shape[-1] < min_segment_samples:
            continue
        
        for band in bands_use:
            if band not in freq_bands:
                continue
            
            fmin, fmax = freq_bands[band]
            try:
                fmin = float(fmin)
                fmax = float(fmax)
            except Exception:
                continue
            
            if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin:
                continue
            
            for ep_idx in range(n_epochs):
                epoch_data = seg_data[ep_idx]
                
                results = _compute_directed_connectivity_epoch(
                    ep_idx,
                    epoch_data,
                    sfreq,
                    fmin,
                    fmax,
                    n_freqs,
                    mvar_order,
                    methods,
                )
                
                for method, conn_matrix in results.items():
                    if conn_matrix is None or not np.isfinite(conn_matrix).any():
                        continue
                    
                    if output_level == "full":
                        for idx, (i, j) in enumerate(zip(pair_i, pair_j)):
                            col_fwd = NamingSchema.build(
                                "dconn", seg_name, band, "chpair",
                                f"{method}_fwd", channel_pair=pair_names[idx]
                            )
                            col_bwd = NamingSchema.build(
                                "dconn", seg_name, band, "chpair",
                                f"{method}_bwd", channel_pair=pair_names[idx]
                            )
                            records[ep_idx][col_fwd] = float(conn_matrix[i, j])
                            records[ep_idx][col_bwd] = float(conn_matrix[j, i])
                    
                    upper_vals = conn_matrix[pair_i, pair_j]
                    lower_vals = conn_matrix[pair_j, pair_i]
                    
                    col_mean_fwd = NamingSchema.build(
                        "dconn", seg_name, band, "global", f"{method}_fwd_mean"
                    )
                    col_mean_bwd = NamingSchema.build(
                        "dconn", seg_name, band, "global", f"{method}_bwd_mean"
                    )
                    col_asymmetry = NamingSchema.build(
                        "dconn", seg_name, band, "global", f"{method}_asymmetry"
                    )
                    
                    records[ep_idx][col_mean_fwd] = float(np.nanmean(upper_vals))
                    records[ep_idx][col_mean_bwd] = float(np.nanmean(lower_vals))
                    
                    asymmetry = np.nanmean(upper_vals) - np.nanmean(lower_vals)
                    records[ep_idx][col_asymmetry] = float(asymmetry)
    
    if logger is not None:
        logger.info(
            "Directed connectivity extraction completed in %.2fs",
            time.perf_counter() - t0
        )
    
    if not records or all(len(r) == 0 for r in records):
        return pd.DataFrame(), []
    
    df = pd.DataFrame(records)
    return df, list(df.columns)


def extract_directed_connectivity_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract directed connectivity features from FeatureContext.
    
    This is the main entry point for the features pipeline.
    
    Parameters
    ----------
    ctx : FeatureContext
        Feature extraction context
    bands : List[str]
        Frequency bands to process
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with directed connectivity features
    columns : List[str]
        List of feature column names
    """
    if not bands:
        return pd.DataFrame(), []
    
    precomputed = getattr(ctx, "precomputed", None)
    if precomputed is None:
        if getattr(ctx, "epochs", None) is None:
            return pd.DataFrame(), []
        
        from eeg_pipeline.analysis.features.preparation import precompute_data
        
        if not getattr(ctx.epochs, "preload", False):
            ctx.logger.info("Preloading epochs data...")
            ctx.epochs.load_data()
        
        precomputed = precompute_data(
            ctx.epochs,
            bands,
            ctx.config,
            ctx.logger,
            windows_spec=ctx.windows,
        )
        try:
            ctx.set_precomputed(precomputed)
        except Exception:
            pass
    
    ctx_name = getattr(ctx, "name", None)
    segments: List[str] = []
    if ctx_name:
        segments = [ctx_name]
    elif getattr(ctx, "windows", None) is not None:
        for key in ("active", "plateau"):
            mask = ctx.windows.get_mask(key)
            if mask is not None and np.any(mask):
                segments = [key]
                break
        if not segments:
            mask_names = [k for k in ctx.windows.masks.keys() if k != "baseline"]
            if mask_names:
                segments = [mask_names[0]]
    if not segments:
        segments = ["full"]
    
    df, cols = extract_directed_connectivity_from_precomputed(
        precomputed,
        bands=bands,
        segments=segments,
        config=ctx.config,
        logger=ctx.logger,
    )
    
    return df, cols


__all__ = [
    "extract_directed_connectivity_features",
    "extract_directed_connectivity_from_precomputed",
]
