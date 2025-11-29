"""
Aperiodic (1/f) Feature Extraction
===================================

Extracts aperiodic spectral features using FOOOF-like fitting:
- Slope: 1/f exponent (related to E/I balance)
- Offset: Broadband power level
- Power-corrected band power: Ratio of observed to 1/f background power within band

These features separate oscillatory from aperiodic neural activity.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.analysis.tfr import validate_baseline_indices
from eeg_pipeline.utils.analysis.stats import (
    fit_aperiodic_to_all_epochs,
    compute_residuals,
)
from eeg_pipeline.utils.config.loader import get_frequency_bands_for_aperiodic


# =============================================================================
# Helper Functions
# =============================================================================


def _adjust_baseline_end(baseline_end: float) -> float:
    return 0.0 if baseline_end > 0 else baseline_end


def _build_aperiodic_feature_records(
    offsets: np.ndarray,
    slopes: np.ndarray,
    residuals: np.ndarray,
    freqs: np.ndarray,
    channel_names: List[str],
    bands: List[str],
    freq_bands: Dict[str, List[float]],
) -> List[Dict[str, float]]:
    if offsets.shape[0] == 0:
        return []
    
    if offsets.shape[0] != slopes.shape[0] or offsets.shape[0] != residuals.shape[0]:
        raise ValueError(
            f"Shape mismatch: offsets={offsets.shape[0]}, slopes={slopes.shape[0]}, "
            f"residuals={residuals.shape[0]} epochs"
        )
    
    if offsets.shape[1] != len(channel_names) or slopes.shape[1] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: offsets/slopes have {offsets.shape[1]} channels, "
            f"but {len(channel_names)} channel names provided"
        )
    
    if residuals.ndim != 3 or residuals.shape[1] != len(channel_names):
        raise ValueError(
            f"Residuals shape mismatch: expected (n_epochs, n_channels, n_freqs), "
            f"got {residuals.shape}"
        )
    
    band_masks = {}
    for band in bands:
        if band in freq_bands:
            fmin, fmax = freq_bands[band]
            band_masks[band] = (freqs >= fmin) & (freqs <= fmax)
    
    n_epochs = offsets.shape[0]
    feature_records = []
    
    for epoch_idx in range(n_epochs):
        record = {}
        for channel_idx, channel_name in enumerate(channel_names):
            record[f"aper_slope_{channel_name}"] = float(slopes[epoch_idx, channel_idx])
            record[f"aper_offset_{channel_name}"] = float(offsets[epoch_idx, channel_idx])
        
        for band, mask in band_masks.items():
            if not np.any(mask):
                for channel_name in channel_names:
                    record[f"powcorr_{band}_{channel_name}"] = np.nan
            else:
                freq_subset = freqs[mask]
                if freq_subset.size < 2:
                    for channel_name in channel_names:
                        record[f"powcorr_{band}_{channel_name}"] = np.nan
                    continue
                bandwidth = float(freq_subset[-1] - freq_subset[0])
                if bandwidth <= 0:
                    for channel_name in channel_names:
                        record[f"powcorr_{band}_{channel_name}"] = np.nan
                    continue

                for channel_idx, channel_name in enumerate(channel_names):
                    band_residual = residuals[epoch_idx, channel_idx, mask]
                    if band_residual.size == 0 or not np.any(np.isfinite(band_residual)):
                        record[f"powcorr_{band}_{channel_name}"] = np.nan
                        continue

                    # Convert log10 residuals to linear ratio (PSD / aperiodic fit) and average
                    band_residual_clean = np.power(10.0, band_residual)
                    if not np.any(np.isfinite(band_residual_clean)):
                        record[f"powcorr_{band}_{channel_name}"] = np.nan
                        continue

                    # Frequency-weighted average ratio to avoid bias from non-uniform bins
                    band_ratio_area = np.trapz(band_residual_clean, freq_subset)
                    band_ratio = band_ratio_area / bandwidth if bandwidth > 0 else np.nan
                    record[f"powcorr_{band}_{channel_name}"] = float(band_ratio)
        
        feature_records.append(record)
    
    return feature_records


def extract_aperiodic_features(
    epochs: Any,
    baseline_window: Tuple[float, float],
    bands: List[str],
    config: Any,
    logger: Any,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    n_fft: Optional[int] = None,
    events_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        logger.warning("No EEG channels available for aperiodic feature extraction")
        return pd.DataFrame(), [], {}
    
    baseline_start, baseline_end = baseline_window
    baseline_end = _adjust_baseline_end(baseline_end)
    default_psd_window = (baseline_start, baseline_end)
    aper_cfg = config.get("feature_engineering.aperiodic", {})
    psd_window_cfg = aper_cfg.get("psd_window", None)
    psd_window = psd_window_cfg if psd_window_cfg is not None else default_psd_window

    try:
        psd_start, psd_end = float(psd_window[0]), float(psd_window[1])
    except (TypeError, ValueError, IndexError):
        psd_start, psd_end = default_psd_window
        logger.warning("Invalid aperiodic PSD window in config; defaulting to baseline window.")

    if psd_end > 0:
        logger.warning(
            "Aperiodic PSD window must end at or before 0 s (pre-stimulus); got [%.3f, %.3f]. "
            "Clamping end to 0 and reusing baseline start (%.3f).",
            psd_start, psd_end, baseline_start,
        )
        psd_end = 0.0
        if psd_start >= psd_end:
            psd_start = baseline_start

    if psd_start >= psd_end:
        logger.warning(
            "Aperiodic PSD window start (%.3f) must be < end (%.3f); defaulting to baseline window.",
            psd_start, psd_end,
        )
        psd_start, psd_end = default_psd_window
    # Validate PSD window has enough pre-stimulus samples
    try:
        validate_baseline_indices(
            np.asarray(epochs.times), (psd_start, psd_end),
            min_samples=int(config.get("feature_engineering.aperiodic.min_samples", 5)),
            logger=logger, config=config,
        )
    except ValueError as exc:
        logger.error("Invalid PSD window for aperiodic fit: %s", exc)
        return pd.DataFrame(), [], {}
    freq_bands = get_frequency_bands_for_aperiodic(config)
    
    if fmin is None:
        fmin = config.get("feature_engineering.constants.aperiodic_fmin", 2.0)
    if fmax is None:
        fmax = config.get("feature_engineering.constants.aperiodic_fmax", 40.0)

    peak_rejection_z = float(aper_cfg.get("peak_rejection_z", 3.5))
    min_fit_points = int(aper_cfg.get("min_fit_points", 5))
    min_r2 = float(aper_cfg.get("min_r2", 0.5))

    logger.info(
        "Computing aperiodic PSD in window [%.3f, %.3f] s (fmin=%.1f, fmax=%.1f, robust peak rejection z=%.1f)",
        psd_start, psd_end, fmin, fmax, peak_rejection_z,
    )
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=float(fmin),
        fmax=float(fmax),
        tmin=psd_start,
        tmax=psd_end,
        n_fft=n_fft,
        n_overlap=0,
        picks=picks,
        average=False,  # keep trial-level variance; do not average across epochs
        verbose=False,
    )
    
    # Get data in (epochs, channels, freqs)
    psds = spectrum.get_data(return_freqs=False)
    freqs = spectrum.freqs
    
    if psds.ndim == 4:
        n_epochs, n_channels, n_freqs, n_segments = psds.shape
        logger.info(
            "Averaging PSD across %d window(s) per epoch to obtain shape (epochs, channels, freqs)",
            n_segments,
        )
        psds = np.mean(psds, axis=-1)
    if psds.ndim != 3:
        logger.error(f"Unexpected PSD shape: {psds.shape}")
        return pd.DataFrame(), [], {}
    if psds.shape[0] != len(epochs):
        logger.error(
            "PSD epoch count mismatch: psds has %d epochs but input had %d. "
            "Averaging across epochs would invalidate aperiodic fitting.",
            psds.shape[0],
            len(epochs),
        )
        return pd.DataFrame(), [], {}
    
    log_freqs = np.log10(freqs)
    epsilon_psd = config.get("feature_engineering.constants.epsilon_psd", 1e-20)
    epsilon_psd = float(epsilon_psd)
    log_psd = np.log10(np.maximum(psds, epsilon_psd))
    
    offsets, slopes = fit_aperiodic_to_all_epochs(
        log_freqs, log_psd,
        peak_rejection_z=peak_rejection_z,
        min_points=min_fit_points,
    )
    residuals = compute_residuals(log_freqs, log_psd, offsets, slopes)
    # R^2 per epoch/channel
    ss_res = np.nansum(residuals ** 2, axis=2)
    mu = np.nanmean(log_psd, axis=2)
    ss_tot = np.nansum((log_psd - mu[:, :, np.newaxis]) ** 2, axis=2)
    r2 = np.where(ss_tot > 0, 1.0 - (ss_res / ss_tot), np.nan)
    
    channel_names = [epochs.info["ch_names"][p] for p in picks]
    feature_records = _build_aperiodic_feature_records(
        offsets, slopes, residuals, freqs, channel_names, bands, freq_bands
    )

    # Invalidate poorly fit channels to avoid propagating unreliable slopes/offsets/powcorr
    n_bad = 0
    for epoch_idx in range(len(epochs)):
        for ch_idx, ch in enumerate(channel_names):
            if not np.isfinite(r2[epoch_idx, ch_idx]) or r2[epoch_idx, ch_idx] < min_r2:
                n_bad += 1
                feature_records[epoch_idx][f"aper_slope_{ch}"] = np.nan
                feature_records[epoch_idx][f"aper_offset_{ch}"] = np.nan
                for band in bands:
                    key = f"powcorr_{band}_{ch}"
                    if key in feature_records[epoch_idx]:
                        feature_records[epoch_idx][key] = np.nan
    if n_bad > 0:
        logger.warning(
            "Aperiodic fits: %d epoch-channel combinations fell below min_r2=%.2f and were set to NaN.",
            n_bad,
            min_r2,
        )

    # Add QC metrics (r2 and residual RMS per epoch/channel)
    for epoch_idx in range(len(epochs)):
        for ch_idx, ch in enumerate(channel_names):
            feature_records[epoch_idx][f"aper_r2_{ch}"] = float(r2[epoch_idx, ch_idx])
            feature_records[epoch_idx][f"aper_resid_rms_{ch}"] = float(np.sqrt(np.nanmean(residuals[epoch_idx, ch_idx, :] ** 2)))

    feature_df = pd.DataFrame(feature_records)
    column_names = list(feature_df.columns)

    # Build QC payload for plotting
    residual_mean = np.nanmean(residuals, axis=0)  # (channels, freqs)
    run_labels = None
    if events_df is not None and len(events_df) == len(epochs):
        for cand in ["run_id", "run", "block"]:
            if cand in events_df.columns:
                run_labels = events_df[cand].astype(str).to_numpy()
                break

    qc_payload = {
        "freqs": freqs,
        "residual_mean": residual_mean,
        "r2": r2,
        "slopes": slopes,
        "offsets": offsets,
        "channel_names": np.array(channel_names),
        "run_labels": run_labels,
    }

    return feature_df, column_names, qc_payload

