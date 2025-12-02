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
from scipy import stats

from eeg_pipeline.analysis.features.core import pick_eeg_channels
from eeg_pipeline.utils.analysis.tfr import validate_baseline_indices
from eeg_pipeline.utils.analysis.stats import compute_residuals
from eeg_pipeline.utils.config.loader import get_frequency_bands_for_aperiodic


# =============================================================================
# Helper Functions
# =============================================================================


def _adjust_baseline_end(baseline_end: float) -> float:
    return 0.0 if baseline_end > 0 else baseline_end


def _fit_single_epoch_channel(
    epoch_idx: int,
    channel_idx: int,
    log_freqs: np.ndarray,
    psd_vals: np.ndarray,
    peak_rejection_z: float,
    min_fit_points: int,
) -> Tuple[int, int, float, float, int, int, bool, np.ndarray, int]:
    """Fit aperiodic component for a single epoch×channel combination.
    
    Returns: (epoch_idx, channel_idx, offset, slope, valid_bins, kept_bins, 
              peak_rejected, fit_mask_indices, status)
    Status: 0=success, 1=insufficient_bins, 2=insufficient_after_rejection, 3=fit_failed
    """
    finite_freq_mask = np.isfinite(log_freqs)
    finite_mask = finite_freq_mask & np.isfinite(psd_vals)
    valid_bins = int(np.sum(finite_mask))
    finite_indices = np.where(finite_mask)[0]
    
    if valid_bins < min_fit_points:
        return (epoch_idx, channel_idx, np.nan, np.nan, valid_bins, 0, False, 
                np.array([], dtype=int), 1)
    
    freq = log_freqs[finite_mask]
    psd_clean = psd_vals[finite_mask]
    
    keep_mask = np.ones_like(psd_clean, dtype=bool)
    mad = stats.median_abs_deviation(psd_clean, scale="normal", nan_policy="omit")
    median = np.median(psd_clean) if np.isfinite(psd_clean).any() else np.nan
    peak_rejected = False
    
    if np.isfinite(mad) and mad > 1e-12 and np.isfinite(median):
        candidate_keep = psd_clean <= median + peak_rejection_z * mad
        if np.sum(candidate_keep) >= min_fit_points:
            keep_mask = candidate_keep
            peak_rejected = bool(np.any(~candidate_keep))
            freq = freq[keep_mask]
            psd_clean = psd_clean[keep_mask]
    
    final_indices = finite_indices[keep_mask] if finite_indices.size else np.array([], dtype=int)
    kept_bins = int(psd_clean.size)
    
    if psd_clean.size < min_fit_points:
        return (epoch_idx, channel_idx, np.nan, np.nan, valid_bins, kept_bins, 
                peak_rejected, np.array([], dtype=int), 2)
    
    try:
        slope, intercept = np.polyfit(freq, psd_clean, 1)
        return (epoch_idx, channel_idx, float(intercept), float(slope), valid_bins, 
                kept_bins, peak_rejected, final_indices, 0)
    except (ValueError, np.linalg.LinAlgError):
        return (epoch_idx, channel_idx, np.nan, np.nan, valid_bins, kept_bins, 
                peak_rejected, np.array([], dtype=int), 3)


def _fit_aperiodic_with_qc(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    peak_rejection_z: float,
    min_fit_points: int,
    logger: Any = None,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit aperiodic components while tracking basic QC metrics.
    
    Short-circuits epoch/channel combinations that lack sufficient valid bins
    for stable fitting. Returns comprehensive QC tracking.
    """
    n_epochs, n_channels, _ = log_psd.shape
    offsets = np.full((n_epochs, n_channels), np.nan)
    slopes = np.full((n_epochs, n_channels), np.nan)
    valid_bins = np.zeros((n_epochs, n_channels), dtype=int)
    kept_bins = np.zeros((n_epochs, n_channels), dtype=int)
    peak_rejected = np.zeros((n_epochs, n_channels), dtype=bool)
    fit_masks = np.zeros((n_epochs, n_channels, log_freqs.shape[0]), dtype=bool)
    
    # QC tracking
    n_skipped_insufficient_bins = 0
    n_skipped_after_peak_rejection = 0
    n_fit_failed = 0
    n_successful = 0
    
    # Generate all epoch×channel combinations
    tasks = [
        (ep_idx, ch_idx, log_freqs, log_psd[ep_idx, ch_idx, :], peak_rejection_z, min_fit_points)
        for ep_idx in range(n_epochs)
        for ch_idx in range(n_channels)
    ]
    
    # Process in parallel or sequentially
    if n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            if logger:
                logger.debug(f"Parallel aperiodic fitting: {len(tasks)} tasks with {n_jobs} workers")
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_fit_single_epoch_channel)(*task) for task in tasks
            )
        except Exception as exc:
            if logger:
                logger.warning(f"Parallel aperiodic fitting failed ({exc}); falling back to sequential.")
            results = [_fit_single_epoch_channel(*task) for task in tasks]
    else:
        results = [_fit_single_epoch_channel(*task) for task in tasks]
    
    # Aggregate results
    for ep_idx, ch_idx, offset, slope, v_bins, k_bins, p_rej, fit_idx, status in results:
        offsets[ep_idx, ch_idx] = offset
        slopes[ep_idx, ch_idx] = slope
        valid_bins[ep_idx, ch_idx] = v_bins
        kept_bins[ep_idx, ch_idx] = k_bins
        peak_rejected[ep_idx, ch_idx] = p_rej
        if fit_idx.size > 0:
            fit_masks[ep_idx, ch_idx, fit_idx] = True
        
        if status == 0:
            n_successful += 1
        elif status == 1:
            n_skipped_insufficient_bins += 1
        elif status == 2:
            n_skipped_after_peak_rejection += 1
        elif status == 3:
            n_fit_failed += 1
    
    # Build QC summary
    total = n_epochs * n_channels
    fit_qc = {
        "n_epoch_channel_pairs": total,
        "n_successful_fits": n_successful,
        "n_skipped_insufficient_bins": n_skipped_insufficient_bins,
        "n_skipped_after_peak_rejection": n_skipped_after_peak_rejection,
        "n_fit_failed": n_fit_failed,
        "success_rate": n_successful / total if total > 0 else 0.0,
    }
    
    # Log warning if too few successful fits
    if logger and n_successful < total * 0.5:
        logger.warning(
            "Aperiodic fitting: only %d/%d (%.1f%%) epoch-channel pairs had successful fits. "
            "Skipped: %d insufficient bins, %d after peak rejection, %d fit failures.",
            n_successful, total, 100 * n_successful / total if total > 0 else 0,
            n_skipped_insufficient_bins, n_skipped_after_peak_rejection, n_fit_failed,
        )

    return offsets, slopes, valid_bins, kept_bins, peak_rejected, fit_masks, fit_qc


def _compute_fit_r2_and_rms(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    offsets: np.ndarray,
    slopes: np.ndarray,
    fit_masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute R^2 and residual RMS using only the frequency bins kept for each fit.
    
    This avoids penalizing fits for bins intentionally excluded during peak rejection.
    """
    n_epochs, n_channels, _ = log_psd.shape
    r2 = np.full((n_epochs, n_channels), np.nan, dtype=float)
    rms = np.full((n_epochs, n_channels), np.nan, dtype=float)

    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            mask = fit_masks[epoch_idx, ch_idx, :]
            if not np.any(mask):
                continue
            if not np.isfinite(offsets[epoch_idx, ch_idx]) or not np.isfinite(slopes[epoch_idx, ch_idx]):
                continue

            y_true = log_psd[epoch_idx, ch_idx, mask]
            y_pred = offsets[epoch_idx, ch_idx] + slopes[epoch_idx, ch_idx] * log_freqs[mask]
            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            if np.sum(finite) < 2:
                continue

            y_true = y_true[finite]
            y_pred = y_pred[finite]
            resid = y_true - y_pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            if ss_tot > 0:
                r2[epoch_idx, ch_idx] = 1.0 - (ss_res / ss_tot)
            rms[epoch_idx, ch_idx] = float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan

    return r2, rms


def _build_aperiodic_feature_records(
    offsets: np.ndarray,
    slopes: np.ndarray,
    residuals: np.ndarray,
    freqs: np.ndarray,
    channel_names: List[str],
    bands: List[str],
    freq_bands: Dict[str, List[float]],
    logger: Any = None,
    min_band_bins: int = 2,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, Any]]]:
    if offsets.shape[0] == 0:
        return [], {}
    
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
    band_qc: Dict[str, Dict[str, Any]] = {}
    warned_bands = set()
    for band in bands:
        if band in freq_bands:
            fmin, fmax = freq_bands[band]
            band_masks[band] = (freqs >= fmin) & (freqs <= fmax)
            band_qc[band] = {
                "n_freq_bins": int(np.sum(band_masks[band])),
                "min_required_bins": int(min_band_bins),
                "valid": bool(np.sum(band_masks[band]) >= min_band_bins),
            }
            if not band_qc[band]["valid"] and band not in warned_bands and logger:
                logger.warning(
                    "Aperiodic powcorr band '%s' has %d bins (<%d); powcorr will be NaN.",
                    band,
                    band_qc[band]["n_freq_bins"],
                    min_band_bins,
                )
                warned_bands.add(band)
    
    n_epochs = offsets.shape[0]
    feature_records = []
    
    for epoch_idx in range(n_epochs):
        record = {}
        for channel_idx, channel_name in enumerate(channel_names):
            record[f"aper_slope_{channel_name}"] = float(slopes[epoch_idx, channel_idx])
            record[f"aper_offset_{channel_name}"] = float(offsets[epoch_idx, channel_idx])
        
        for band, mask in band_masks.items():
            if not np.any(mask) or mask.sum() < min_band_bins:
                for channel_name in channel_names:
                    record[f"powcorr_{band}_{channel_name}"] = np.nan
            else:
                freq_subset = freqs[mask]
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

    return feature_records, band_qc


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
    picks, ch_names = pick_eeg_channels(epochs)
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
    
    n_jobs_aperiodic = int(config.get("feature_engineering.parallel.n_jobs_aperiodic", 1))
    offsets, slopes, valid_bins, kept_bins, peak_rejected, fit_masks, fit_qc = _fit_aperiodic_with_qc(
        log_freqs,
        log_psd,
        peak_rejection_z=peak_rejection_z,
        min_fit_points=min_fit_points,
        logger=logger,
        n_jobs=n_jobs_aperiodic,
    )
    
    # Early bail if success rate is critically low
    if fit_qc["success_rate"] < 0.1:
        logger.error(
            "Aperiodic extraction aborted: only %.1f%% of fits succeeded (threshold: 10%%). "
            "Check PSD data quality or frequency range settings.",
            fit_qc["success_rate"] * 100,
        )
        return pd.DataFrame(), [], {"fit_qc": fit_qc, "aborted": True}
    residuals = compute_residuals(log_freqs, log_psd, offsets, slopes)
    # R^2 per epoch/channel using only bins retained for fitting
    r2, resid_rms = _compute_fit_r2_and_rms(log_freqs, log_psd, offsets, slopes, fit_masks)
    
    channel_names = [epochs.info["ch_names"][p] for p in picks]
    min_band_bins = int(aper_cfg.get("min_band_bins", 2))
    feature_records, band_qc = _build_aperiodic_feature_records(
        offsets, slopes, residuals, freqs, channel_names, bands, freq_bands, logger=logger, min_band_bins=min_band_bins
    )

    # Invalidate poorly fit channels to avoid propagating unreliable slopes/offsets/powcorr
    n_low_r2 = 0
    n_missing_fit = 0
    for epoch_idx in range(len(epochs)):
        for ch_idx, ch in enumerate(channel_names):
            fit_attempted = bool(np.any(fit_masks[epoch_idx, ch_idx, :]))
            has_params = np.isfinite(offsets[epoch_idx, ch_idx]) and np.isfinite(slopes[epoch_idx, ch_idx])
            if not (fit_attempted and has_params):
                n_missing_fit += 1
                feature_records[epoch_idx][f"aper_slope_{ch}"] = np.nan
                feature_records[epoch_idx][f"aper_offset_{ch}"] = np.nan
                for band in bands:
                    key = f"powcorr_{band}_{ch}"
                    if key in feature_records[epoch_idx]:
                        feature_records[epoch_idx][key] = np.nan
                continue

            if not np.isfinite(r2[epoch_idx, ch_idx]) or r2[epoch_idx, ch_idx] < min_r2:
                n_low_r2 += 1
                feature_records[epoch_idx][f"aper_slope_{ch}"] = np.nan
                feature_records[epoch_idx][f"aper_offset_{ch}"] = np.nan
                for band in bands:
                    key = f"powcorr_{band}_{ch}"
                    if key in feature_records[epoch_idx]:
                        feature_records[epoch_idx][key] = np.nan
    if n_low_r2 > 0:
        logger.warning(
            "Aperiodic fits: %d epoch-channel combinations fell below min_r2=%.2f and were set to NaN "
            "(%d additional combinations lacked a valid fit and remain NaN).",
            n_low_r2,
            min_r2,
            n_missing_fit,
        )
    elif n_missing_fit > 0:
        logger.info(
            "Aperiodic fits: %d epoch-channel combinations lacked a valid fit (insufficient bins or fit failure); values remain NaN.",
            n_missing_fit,
        )

    # Add QC metrics (r2 and residual RMS per epoch/channel)
    for epoch_idx in range(len(epochs)):
        for ch_idx, ch in enumerate(channel_names):
            feature_records[epoch_idx][f"aper_r2_{ch}"] = float(r2[epoch_idx, ch_idx])
            feature_records[epoch_idx][f"aper_resid_rms_{ch}"] = float(resid_rms[epoch_idx, ch_idx])

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
        "fit_stats": {
            "valid_bins": valid_bins,
            "kept_bins_after_peak_rejection": kept_bins,
            "peak_rejected": peak_rejected,
            "min_fit_points": min_fit_points,
            "peak_rejected_fraction": float(np.mean(peak_rejected)) if peak_rejected.size else 0.0,
            **fit_qc,  # Include detailed fit QC from _fit_aperiodic_with_qc
        },
        "band_qc": band_qc,
        "min_band_bins": min_band_bins,
        "n_bad_r2": n_low_r2,
        "n_missing_fit": n_missing_fit,
        "min_r2": min_r2,
    }

    return feature_df, column_names, qc_payload
