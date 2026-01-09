"""
Spectral Feature Extraction
============================

Extracts spectral peak features for pain research:
- Individual Alpha Frequency (IAF): Key biomarker for pain sensitivity
- Peak power per band
- Spectral center of gravity
- Spectral bandwidth
- Spectral edge frequency
- Spectral entropy (normalized)
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels, build_roi_map
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.windowing import get_segment_masks


###################################################################
# Core Spectral Computations
###################################################################


def remove_aperiodic_component(
    psd: np.ndarray,
    freqs: np.ndarray,
    fit_range: Tuple[float, float] = (2.0, 40.0),
) -> np.ndarray:
    """
    Remove 1/f aperiodic component from PSD using robust linear fit in log-log space.
    
    This addresses the scientific validity concern that raw PSD peak detection
    is biased by 1/f changes (common in pain/arousal/alertness states).
    
    Parameters
    ----------
    psd : np.ndarray
        Power spectral density (1D array)
    freqs : np.ndarray
        Frequency values
    fit_range : tuple
        Frequency range for fitting 1/f model (Hz)
        
    Returns
    -------
    residual : np.ndarray
        Aperiodic-adjusted PSD (residual in log space)
    """
    if psd.size == 0 or freqs.size == 0:
        return psd.copy()
    
    log_f = np.log10(np.maximum(freqs, 1e-6))
    log_p = np.log10(np.maximum(psd, 1e-20))
    
    fit_mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & np.isfinite(log_p)
    if np.sum(fit_mask) < 5:
        return psd.copy()
    
    try:
        slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
        aperiodic_fit = intercept + slope * log_f
        residual = log_p - aperiodic_fit
        return 10 ** residual
    except (np.linalg.LinAlgError, ValueError):
        return psd.copy()


def compute_peak_frequency(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    aperiodic_adjusted: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Compute peak frequency and peak power within a frequency range.
    
    Parameters
    ----------
    psd : np.ndarray
        Power spectral density (1D array)
    freqs : np.ndarray
        Frequency values
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    aperiodic_adjusted : bool
        If True, remove 1/f component before peak detection to avoid
        bias from aperiodic power changes (default: True)
    
    Returns
    -------
    peak_freq : float
        Frequency of maximum power (on aperiodic-adjusted spectrum if enabled)
    peak_power : float
        Power at peak frequency (from original PSD)
    peak_ratio : float
        Ratio of raw power to aperiodic fit at peak frequency (oscillatory 
        prominence relative to 1/f background). Values > 1 indicate a peak
        above the aperiodic floor.
    peak_residual : float
        Log10(power) - log10(aperiodic_fit) at peak frequency. Positive values
        indicate oscillatory power above the aperiodic background.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    if len(psd_band) == 0 or np.all(np.isnan(psd_band)):
        return np.nan, np.nan, np.nan, np.nan
    
    # Always compute aperiodic fit for prominence metrics
    log_f = np.log10(np.maximum(freqs, 1e-6))
    log_p = np.log10(np.maximum(psd, 1e-20))
    
    fit_range = (2.0, 40.0)  # Standard fit range
    fit_mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & np.isfinite(log_p)
    
    aperiodic_fit = None
    if np.sum(fit_mask) >= 5:
        try:
            slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
            aperiodic_fit = 10 ** (intercept + slope * log_f)  # Linear in log-log space
        except (np.linalg.LinAlgError, ValueError):
            pass
    
    if aperiodic_adjusted and aperiodic_fit is not None:
        residual = log_p - (np.log10(aperiodic_fit))
        psd_for_peak = (10 ** residual)[mask]
    else:
        psd_for_peak = psd_band
    
    if np.all(np.isnan(psd_for_peak)):
        psd_for_peak = psd_band
    
    peak_idx = np.nanargmax(psd_for_peak)
    peak_freq = float(freqs_band[peak_idx])
    peak_power = float(psd_band[peak_idx])
    
    # Compute peak prominence metrics
    peak_ratio = np.nan
    peak_residual = np.nan
    
    if aperiodic_fit is not None:
        # Find the global index for the peak frequency
        global_peak_idx = np.where(mask)[0][peak_idx]
        aperiodic_at_peak = aperiodic_fit[global_peak_idx]
        
        if np.isfinite(aperiodic_at_peak) and aperiodic_at_peak > 0:
            peak_ratio = float(peak_power / aperiodic_at_peak)
            peak_residual = float(np.log10(peak_power) - np.log10(aperiodic_at_peak))
    
    return peak_freq, peak_power, peak_ratio, peak_residual


def compute_spectral_center(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Compute spectral center of gravity (centroid) within a frequency range.
    
    Uses Δf weighting for non-uniform frequency grids (e.g., log-spaced).
    Formula: Σ(f * P * Δf) / Σ(P * Δf)
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan
    
    center = float(np.nansum(freqs_band * mass) / total_mass)
    return center


def compute_spectral_bandwidth(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Compute spectral bandwidth (standard deviation of frequency distribution).
    
    Uses Δf weighting for non-uniform frequency grids.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan
    
    center = np.nansum(freqs_band * mass) / total_mass
    variance = np.nansum(mass * (freqs_band - center) ** 2) / total_mass
    bandwidth = float(np.sqrt(variance))
    
    return bandwidth


def compute_spectral_edge(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    percentile: float = 0.95,
) -> float:
    """
    Compute spectral edge frequency (frequency below which X% of power lies).
    
    Uses Δf weighting for non-uniform frequency grids.
    
    Parameters
    ----------
    percentile : float
        Cumulative power threshold (default 0.95 = 95%)
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan
    
    cumsum = np.nancumsum(mass) / total_mass
    edge_idx = np.searchsorted(cumsum, percentile)
    edge_idx = min(edge_idx, len(freqs_band) - 1)
    
    return float(freqs_band[edge_idx])


def compute_spectral_entropy(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Compute normalized spectral entropy within a frequency range.
    
    Uses Δf weighting for non-uniform frequency grids.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan

    psd_band = psd[mask]
    freqs_band = freqs[mask]
    if len(psd_band) == 0 or np.all(np.isnan(psd_band)):
        return np.nan

    psd_band = np.maximum(psd_band, 0)
    
    # Compute frequency bin widths for proper weighting
    df = np.gradient(freqs_band) if len(freqs_band) > 1 else np.ones_like(freqs_band)
    mass = psd_band * df
    
    total_mass = np.nansum(mass)
    if total_mass <= 0 or np.isnan(total_mass):
        return np.nan

    probs = mass / total_mass
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if probs.size == 0:
        return np.nan

    entropy = -np.sum(probs * np.log(probs))
    norm = np.log(float(probs.size))
    if norm > 0:
        entropy /= norm
    return float(entropy)


###################################################################
# Main Extractor
###################################################################


def extract_spectral_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract spectral features including IAF (Individual Alpha Frequency).
    
    Features extracted:
    - Peak frequency per band (IAF for alpha band)
    - Peak power per band
    - Spectral center of gravity
    - Spectral bandwidth
    - Spectral entropy (normalized)
    - Spectral edge frequency (broadband, 95%)
    """
    if not bands:
        return pd.DataFrame(), []
    
    epochs = ctx.epochs
    config = ctx.config
    logger = ctx.logger
    
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("Spectral: No EEG channels available; skipping.")
        return pd.DataFrame(), []
    
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)
    spatial_modes = getattr(ctx, "spatial_modes", ["roi", "global"])
    
    roi_map = {}
    if "roi" in spatial_modes:
        roi_defs = get_roi_definitions(config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)
    
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data(picks=picks)
    n_epochs = data.shape[0]
    n_channels = data.shape[1]

    spec_cfg = config.get("feature_engineering.spectral", {}) if hasattr(config, "get") else {}
    psd_method = str(spec_cfg.get("psd_method", "multitaper")).strip().lower()
    if psd_method not in {"welch", "multitaper"}:
        psd_method = "multitaper"

    fmin_psd = float(spec_cfg.get("fmin", 1.0))
    fmax_psd = float(spec_cfg.get("fmax", min(80.0, float(sfreq) / 2.0 - 0.5)))

    exclude_line = bool(spec_cfg.get("exclude_line_noise", True))
    line_freqs = spec_cfg.get("line_noise_freqs", [50.0])
    try:
        line_freqs = [float(f) for f in line_freqs]
    except Exception:
        line_freqs = [50.0]
    line_width = float(spec_cfg.get("line_noise_width_hz", 1.0))
    n_harm = int(spec_cfg.get("line_noise_harmonics", 3))
    
    segment_masks = get_segment_masks(epochs.times, ctx.windows, config)
    
    # Restrict to configured segments (default: baseline only for spectral/IAF validity)
    allowed_segments = spec_cfg.get("segments", ["baseline"])
    if isinstance(allowed_segments, str):
        allowed_segments = [allowed_segments]
    
    # Filter to allowed segments that exist
    segments: List[str] = [s for s in allowed_segments if s in segment_masks or s == "full"]
    
    # Fallback if no segments defined
    if not segments:
        segments = ["baseline"] if "baseline" in segment_masks else ["full"]
    
    # Segment duration validation parameters
    min_segment_sec = float(spec_cfg.get("min_segment_sec", 2.0))
    min_cycles_at_fmin = float(spec_cfg.get("min_cycles_at_fmin", 3.0))

    records = [dict() for _ in range(n_epochs)]

    for segment_name in segments:
        if segment_name == "full":
            mask = np.ones(data.shape[2], dtype=bool)
        else:
            mask = segment_masks.get(segment_name)
            if mask is None or not np.any(mask):
                continue

        seg_data = data[:, :, mask]
        seg_duration_sec = float(seg_data.shape[2]) / float(sfreq)
        
        # Validate minimum segment duration
        if seg_duration_sec < min_segment_sec:
            logger.warning(
                "Spectral: segment '%s' duration (%.2fs) is shorter than min_segment_sec (%.2fs); skipping.",
                segment_name, seg_duration_sec, min_segment_sec
            )
            continue
        
        if seg_data.shape[2] < 2:
            continue

        try:
            import mne
            if psd_method == "multitaper":
                psds, freqs = mne.time_frequency.psd_array_multitaper(
                    seg_data,
                    sfreq=float(sfreq),
                    fmin=fmin_psd,
                    fmax=fmax_psd,
                    adaptive=True,
                    normalization="full",
                    verbose=False,
                )
            else:
                n_times = int(seg_data.shape[2])
                n_per_seg = min(int(float(sfreq) * 2.0), n_times)
                psds, freqs = mne.time_frequency.psd_array_welch(
                    seg_data,
                    sfreq=float(sfreq),
                    fmin=fmin_psd,
                    fmax=fmax_psd,
                    n_fft=min(n_times, max(64, n_per_seg)),
                    n_overlap=0,
                    verbose=False,
                )
        except Exception as exc:
            logger.warning("Spectral: PSD computation failed for segment '%s' (%s); skipping.", segment_name, exc)
            continue

        freqs = np.asarray(freqs, dtype=float)
        psds = np.asarray(psds, dtype=float)
        if psds.ndim != 3:
            continue

        freq_keep_mask = np.ones_like(freqs, dtype=bool)
        if exclude_line and freqs.size > 0 and line_width > 0 and n_harm > 0:
            for base in line_freqs:
                if not np.isfinite(base) or base <= 0:
                    continue
                for h in range(1, n_harm + 1):
                    f0 = base * h
                    freq_keep_mask &= ~(
                        (freqs >= (f0 - line_width)) & (freqs <= (f0 + line_width))
                    )

        freqs_use = freqs[freq_keep_mask] if np.any(~freq_keep_mask) else freqs
        psds_use = psds[:, :, freq_keep_mask] if np.any(~freq_keep_mask) else psds

        for ep_idx in range(n_epochs):
            record = records[ep_idx]
            channel_psd = psds_use[ep_idx]

            for band in bands:
                if band not in freq_bands:
                    continue
                fmin, fmax = freq_bands[band]

                # Require enough cycles at the lowest frequency in the band.
                if min_cycles_at_fmin > 0:
                    try:
                        fmin_hz = float(fmin)
                    except Exception:
                        fmin_hz = np.nan
                    if np.isfinite(fmin_hz) and fmin_hz > 0:
                        min_req_sec = float(min_cycles_at_fmin) / fmin_hz
                        if seg_duration_sec < min_req_sec:
                            continue

                if "channels" in spatial_modes:
                    for ch_idx, ch_name in enumerate(ch_names):
                        psd = channel_psd[ch_idx]
                        peak_freq, peak_power, peak_ratio, peak_residual = compute_peak_frequency(
                            psd, freqs_use, fmin, fmax, aperiodic_adjusted=True
                        )
                        center_freq = compute_spectral_center(psd, freqs_use, fmin, fmax)
                        bandwidth = compute_spectral_bandwidth(psd, freqs_use, fmin, fmax)
                        entropy = compute_spectral_entropy(psd, freqs_use, fmin, fmax)

                        record[f"spectral_{segment_name}_{band}_ch_{ch_name}_peak_freq"] = peak_freq
                        record[f"spectral_{segment_name}_{band}_ch_{ch_name}_peak_power"] = peak_power
                        record[f"spectral_{segment_name}_{band}_ch_{ch_name}_peak_ratio"] = peak_ratio
                        record[f"spectral_{segment_name}_{band}_ch_{ch_name}_peak_residual"] = peak_residual
                        record[f"spectral_{segment_name}_{band}_ch_{ch_name}_center_freq"] = center_freq
                        record[f"spectral_{segment_name}_{band}_ch_{ch_name}_bandwidth"] = bandwidth
                        record[f"spectral_{segment_name}_{band}_ch_{ch_name}_entropy"] = entropy

                if "global" in spatial_modes:
                    global_psd = np.nanmean(channel_psd, axis=0)
                    g_peak_freq, g_peak_power, g_peak_ratio, g_peak_residual = compute_peak_frequency(
                        global_psd, freqs_use, fmin, fmax, aperiodic_adjusted=True
                    )
                    g_center = compute_spectral_center(global_psd, freqs_use, fmin, fmax)
                    g_bandwidth = compute_spectral_bandwidth(global_psd, freqs_use, fmin, fmax)
                    g_entropy = compute_spectral_entropy(global_psd, freqs_use, fmin, fmax)

                    record[f"spectral_{segment_name}_{band}_global_peak_freq"] = g_peak_freq
                    record[f"spectral_{segment_name}_{band}_global_peak_power"] = g_peak_power
                    record[f"spectral_{segment_name}_{band}_global_peak_ratio"] = g_peak_ratio
                    record[f"spectral_{segment_name}_{band}_global_peak_residual"] = g_peak_residual
                    record[f"spectral_{segment_name}_{band}_global_center_freq"] = g_center
                    record[f"spectral_{segment_name}_{band}_global_bandwidth"] = g_bandwidth
                    record[f"spectral_{segment_name}_{band}_global_entropy"] = g_entropy

                if "roi" in spatial_modes and roi_map:
                    for roi_name, roi_indices in roi_map.items():
                        if not roi_indices:
                            continue
                        roi_psd = np.nanmean(channel_psd[roi_indices], axis=0)
                        r_peak_freq, r_peak_power, r_peak_ratio, r_peak_residual = compute_peak_frequency(
                            roi_psd, freqs_use, fmin, fmax, aperiodic_adjusted=True
                        )
                        r_center = compute_spectral_center(roi_psd, freqs_use, fmin, fmax)
                        r_bandwidth = compute_spectral_bandwidth(roi_psd, freqs_use, fmin, fmax)
                        r_entropy = compute_spectral_entropy(roi_psd, freqs_use, fmin, fmax)

                        record[f"spectral_{segment_name}_{band}_roi_{roi_name}_peak_freq"] = r_peak_freq
                        record[f"spectral_{segment_name}_{band}_roi_{roi_name}_peak_power"] = r_peak_power
                        record[f"spectral_{segment_name}_{band}_roi_{roi_name}_peak_ratio"] = r_peak_ratio
                        record[f"spectral_{segment_name}_{band}_roi_{roi_name}_peak_residual"] = r_peak_residual
                        record[f"spectral_{segment_name}_{band}_roi_{roi_name}_center_freq"] = r_center
                        record[f"spectral_{segment_name}_{band}_roi_{roi_name}_bandwidth"] = r_bandwidth
                        record[f"spectral_{segment_name}_{band}_roi_{roi_name}_entropy"] = r_entropy

            global_psd = np.nanmean(channel_psd, axis=0)
            edge_fmax = float(freqs_use[-1]) if freqs_use.size else (float(sfreq) / 2.0 - 0.5)
            edge_95 = compute_spectral_edge(global_psd, freqs_use, 1.0, edge_fmax, 0.95)
            record[f"spectral_{segment_name}_broadband_global_edge_freq_95"] = edge_95

            if "roi" in spatial_modes and roi_map:
                for roi_name, roi_indices in roi_map.items():
                    if not roi_indices:
                        continue
                    roi_psd = np.nanmean(channel_psd[roi_indices], axis=0)
                    roi_edge = compute_spectral_edge(roi_psd, freqs_use, 1.0, edge_fmax, 0.95)
                    record[f"spectral_{segment_name}_broadband_roi_{roi_name}_edge_freq_95"] = roi_edge
    
    if not records:
        return pd.DataFrame(), []
    
    df = pd.DataFrame(records)
    cols = list(df.columns)
    
    logger.info(f"Extracted {len(cols)} spectral features for {n_epochs} epochs")
    
    return df, cols


__all__ = [
    "extract_spectral_features",
    "compute_peak_frequency",
    "compute_spectral_center",
    "compute_spectral_bandwidth",
    "compute_spectral_edge",
    "compute_spectral_entropy",
    "remove_aperiodic_component",
]
