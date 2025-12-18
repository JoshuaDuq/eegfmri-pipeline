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
from joblib import Parallel, delayed

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import STANDARD_SEGMENTS, get_segment_mask, validate_extractor_inputs
from eeg_pipeline.utils.config.loader import get_frequency_bands_for_aperiodic
from eeg_pipeline.utils.analysis.stats import compute_residuals
from eeg_pipeline.utils.parallel import get_n_jobs

# --- Helper Functions (Preserved) ---

def _fit_single_epoch_channel(
    epoch_idx: int,
    channel_idx: int,
    log_freqs: np.ndarray,
    psd_vals: np.ndarray,
    peak_rejection_z: float,
    min_fit_points: int,
) -> Tuple[int, int, float, float, int, int, bool, np.ndarray, int]:
    finite_mask = np.isfinite(log_freqs) & np.isfinite(psd_vals)
    finite_indices = np.flatnonzero(finite_mask)
    valid_bins = int(finite_indices.size)
    if valid_bins < min_fit_points:
        return (epoch_idx, channel_idx, np.nan, np.nan, valid_bins, 0, False, np.array([], dtype=int), 1)

    psd_finite = psd_vals[finite_indices]
    keep_mask = np.ones(valid_bins, dtype=bool)

    mad = stats.median_abs_deviation(psd_finite, scale="normal", nan_policy="omit")
    median = np.median(psd_finite) if np.isfinite(psd_finite).any() else np.nan
    peak_rejected = False

    if np.isfinite(mad) and mad > 1e-12 and np.isfinite(median):
        candidate_keep = psd_finite <= median + peak_rejection_z * mad
        if int(np.sum(candidate_keep)) >= min_fit_points:
            keep_mask = candidate_keep
            peak_rejected = bool(np.any(~candidate_keep))

    kept_indices = finite_indices[keep_mask]
    kept_bins = int(kept_indices.size)
    if kept_bins < min_fit_points:
        return (epoch_idx, channel_idx, np.nan, np.nan, valid_bins, kept_bins, peak_rejected, np.array([], dtype=int), 2)

    try:
        freq_fit = log_freqs[kept_indices]
        psd_fit = psd_vals[kept_indices]
        slope, intercept = np.polyfit(freq_fit, psd_fit, 1)
        return (
            epoch_idx,
            channel_idx,
            float(intercept),
            float(slope),
            valid_bins,
            kept_bins,
            peak_rejected,
            kept_indices.astype(int),
            0,
        )
    except Exception:
        return (epoch_idx, channel_idx, np.nan, np.nan, valid_bins, kept_bins, peak_rejected, np.array([], dtype=int), 3)

def _fit_aperiodic_with_qc(log_freqs, log_psd, peak_rejection_z, min_fit_points, logger, n_jobs=1):
    # Driver for parallel fitting
    n_epochs, n_channels, _ = log_psd.shape
    offsets = np.full((n_epochs, n_channels), np.nan)
    slopes = np.full((n_epochs, n_channels), np.nan)
    valid_bins = np.zeros((n_epochs, n_channels), dtype=int)
    kept_bins = np.zeros((n_epochs, n_channels), dtype=int)
    peak_rejected = np.zeros((n_epochs, n_channels), dtype=bool)
    fit_masks = np.zeros((n_epochs, n_channels, log_freqs.shape[0]), dtype=bool)
    
    tasks = [
        (ep_idx, ch_idx, log_freqs, log_psd[ep_idx, ch_idx, :], peak_rejection_z, min_fit_points)
        for ep_idx in range(n_epochs) for ch_idx in range(n_channels)
    ]
    
    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_epoch_channel)(*task) for task in tasks
        )
    else:
        results = [_fit_single_epoch_channel(*task) for task in tasks]
    
    for ep_idx, ch_idx, offset, slope, v_bins, k_bins, p_rej, fit_idx, status in results:
        offsets[ep_idx, ch_idx] = offset
        slopes[ep_idx, ch_idx] = slope
        valid_bins[ep_idx, ch_idx] = v_bins
        kept_bins[ep_idx, ch_idx] = k_bins
        peak_rejected[ep_idx, ch_idx] = p_rej
        if fit_idx.size > 0: fit_masks[ep_idx, ch_idx, fit_idx] = True
        
    return offsets, slopes, valid_bins, kept_bins, peak_rejected, fit_masks, {}

def _compute_fit_r2_and_rms(log_freqs, log_psd, offsets, slopes, fit_masks):
    n_epochs, n_channels, _ = log_psd.shape
    r2 = np.full((n_epochs, n_channels), np.nan)
    rms = np.full((n_epochs, n_channels), np.nan)
    
    for e in range(n_epochs):
        for c in range(n_channels):
            mask = fit_masks[e, c, :]
            if not np.any(mask) or np.isnan(offsets[e,c]): continue
            y_true = log_psd[e, c, mask]
            y_pred = offsets[e,c] + slopes[e,c] * log_freqs[mask]
            resid = y_true - y_pred
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            if ss_tot > 0: r2[e,c] = 1.0 - (ss_res/ss_tot)
            rms[e,c] = np.sqrt(np.mean(resid**2))
    return r2, rms

# --- Main API ---

def _find_alpha_peak_frequency(freqs: np.ndarray, psd: np.ndarray, alpha_range: Tuple[float, float] = (7.0, 13.0)) -> float:
    """Find individual alpha peak frequency within alpha range.
    
    Uses center of gravity method for robustness.
    """
    alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    if not np.any(alpha_mask):
        return np.nan
    
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]
    
    if np.all(np.isnan(alpha_psd)) or np.sum(alpha_psd) <= 0:
        return np.nan
    
    # Center of gravity (more robust than simple argmax)
    alpha_psd_pos = np.maximum(alpha_psd, 0)
    total_power = np.sum(alpha_psd_pos)
    if total_power > 0:
        cog = np.sum(alpha_freqs * alpha_psd_pos) / total_power
        return float(cog)
    return np.nan


def _extract_aperiodic_for_segment(
    epochs: mne.Epochs,
    picks: np.ndarray,
    ch_names: List[str],
    segment_name: str,
    start_t: float,
    end_t: float,
    bands: List[str],
    config: Any,
    logger: Any,
) -> Dict[str, np.ndarray]:
    """Extract aperiodic and spectral features for a single segment.
    
    Features:
    - Aperiodic slope/offset (1/f)
    - Power-corrected band power
    - Alpha Peak Frequency (APF)
    - Theta/Beta Ratio (TBR)
    """
    fmin = float(config.get("feature_engineering.constants.aperiodic_fmin", 2.0))
    fmax = float(config.get("feature_engineering.constants.aperiodic_fmax", 40.0))
    
    try:
        spectrum = epochs.compute_psd(
            method="welch", fmin=fmin, fmax=fmax, tmin=start_t, tmax=end_t,
            picks=picks, average=False, verbose=False
        )
    except Exception as e:
        logger.warning(f"PSD computation failed for {segment_name}: {e}")
        return {}
        
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    if psds.ndim != 3:
        if psds.ndim == 4: 
            psds = np.mean(psds, axis=-1)
    
    log_freqs = np.log10(freqs)
    log_psd = np.log10(np.maximum(psds, 1e-20))
    
    peak_z = float(config.get("feature_engineering.aperiodic.peak_rejection_z", 3.5))
    min_pts = int(config.get("feature_engineering.aperiodic.min_fit_points", 5))
    n_jobs = get_n_jobs(config, default=-1, config_path="feature_engineering.parallel.n_jobs_aperiodic")
    
    offsets, slopes, valid_bins, kept_bins, peak_rej, fit_masks, fit_qc = _fit_aperiodic_with_qc(
        log_freqs, log_psd, peak_z, min_pts, logger, n_jobs=n_jobs
    )
    
    residuals = compute_residuals(log_freqs, log_psd, offsets, slopes)
    r2, rms = _compute_fit_r2_and_rms(log_freqs, log_psd, offsets, slopes, fit_masks)

    min_r2 = float(config.get("feature_engineering.aperiodic.min_r2", 0.0))
    if not np.isfinite(min_r2):
        min_r2 = 0.0

    max_rms = config.get("feature_engineering.aperiodic.max_rms", None)
    if max_rms is not None:
        try:
            max_rms = float(max_rms)
        except Exception:
            max_rms = None
    if max_rms is not None and not np.isfinite(max_rms):
        max_rms = None

    fit_ok = np.isfinite(r2)
    if min_r2 > 0:
        fit_ok &= (r2 >= min_r2)

    if max_rms is not None:
        fit_ok &= (np.isfinite(rms) & (rms <= max_rms))

    data_dict: Dict[str, np.ndarray] = {}
    freq_bands = get_frequency_bands_for_aperiodic(config)
    n_epochs = psds.shape[0]
    
    # Get theta and beta ranges for TBR
    theta_range = freq_bands.get("theta", (4.0, 8.0))
    beta_range = freq_bands.get("beta", (13.0, 30.0))
    alpha_range = freq_bands.get("alpha", (8.0, 13.0))
    
    theta_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1])
    beta_mask = (freqs >= beta_range[0]) & (freqs <= beta_range[1])
    
    for i, ch in enumerate(ch_names):
        # Aperiodic features
        col_slope = NamingSchema.build("aperiodic", segment_name, "broadband", "ch", "slope", channel=ch)
        col_offset = NamingSchema.build("aperiodic", segment_name, "broadband", "ch", "offset", channel=ch)
        col_r2 = NamingSchema.build("aperiodic", segment_name, "broadband", "ch", "r2", channel=ch)
        col_rms = NamingSchema.build("aperiodic", segment_name, "broadband", "ch", "rms", channel=ch)
        
        slope_vals = slopes[:, i].copy()
        offset_vals = offsets[:, i].copy()
        bad = ~fit_ok[:, i]
        if np.any(bad):
            slope_vals[bad] = np.nan
            offset_vals[bad] = np.nan

        data_dict[col_slope] = slope_vals
        data_dict[col_offset] = offset_vals
        data_dict[col_r2] = r2[:, i]
        data_dict[col_rms] = rms[:, i]
        
        # Alpha Peak Frequency (vectorized across epochs)
        # Use center of gravity for each epoch
        alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
        if np.any(alpha_mask):
            alpha_freqs = freqs[alpha_mask]
            alpha_psd = psds[:, i, alpha_mask]  # (n_epochs, n_alpha_freqs)
            alpha_psd_pos = np.maximum(alpha_psd, 0)
            total_power = np.sum(alpha_psd_pos, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                apf_vals = np.where(
                    total_power > 0,
                    np.sum(alpha_freqs * alpha_psd_pos, axis=1) / total_power,
                    np.nan
                )
        else:
            apf_vals = np.full(n_epochs, np.nan)
        col_apf = NamingSchema.build("aperiodic", segment_name, "alpha", "ch", "peakfreq", channel=ch)
        data_dict[col_apf] = apf_vals
        
        # Theta/Beta Ratio (vectorized across epochs)
        if np.any(theta_mask) and np.any(beta_mask):
            theta_power = np.mean(psds[:, i, theta_mask], axis=1)  # (n_epochs,)
            beta_power = np.mean(psds[:, i, beta_mask], axis=1)    # (n_epochs,)
            with np.errstate(invalid='ignore', divide='ignore'):
                tbr_vals = np.where(
                    (beta_power > 0) & np.isfinite(theta_power) & np.isfinite(beta_power),
                    theta_power / beta_power,
                    np.nan
                )
        else:
            tbr_vals = np.full(n_epochs, np.nan)
        col_tbr = NamingSchema.build("aperiodic", segment_name, "broadband", "ch", "tbr", channel=ch)
        data_dict[col_tbr] = tbr_vals
        
        # Power-corrected band power
        for b in bands:
            if b not in freq_bands: 
                continue
            bfmin, bfmax = freq_bands[b]
            fmask = (freqs >= bfmin) & (freqs <= bfmax)
            if not np.any(fmask): 
                continue
            
            res_band = residuals[:, i, fmask]
            ratio = np.power(10.0, res_band)
            mean_ratio = np.mean(ratio, axis=1)
            if np.any(bad):
                mean_ratio = mean_ratio.copy()
                mean_ratio[bad] = np.nan
            
            col_powcorr = NamingSchema.build("aperiodic", segment_name, b, "ch", "powcorr", channel=ch)
            data_dict[col_powcorr] = mean_ratio
    
    # Global APF and TBR (vectorized mean across channels)
    apf_ch_keys = [NamingSchema.build("aperiodic", segment_name, "alpha", "ch", "peakfreq", channel=ch) for ch in ch_names]
    tbr_ch_keys = [NamingSchema.build("aperiodic", segment_name, "broadband", "ch", "tbr", channel=ch) for ch in ch_names]
    
    apf_stack = np.column_stack([data_dict[k] for k in apf_ch_keys])  # (n_epochs, n_channels)
    tbr_stack = np.column_stack([data_dict[k] for k in tbr_ch_keys])  # (n_epochs, n_channels)
    
    apf_global = np.nanmean(apf_stack, axis=1)
    tbr_global = np.nanmean(tbr_stack, axis=1)
    
    col_apf_glob = NamingSchema.build("aperiodic", segment_name, "alpha", "global", "peakfreq")
    col_tbr_glob = NamingSchema.build("aperiodic", segment_name, "broadband", "global", "tbr")
    data_dict[col_apf_glob] = apf_global
    data_dict[col_tbr_glob] = tbr_global

    data_dict["__qc__"] = {
        "segment": segment_name,
        "freqs": freqs,
        "log_freqs": log_freqs,
        "slopes": slopes,
        "offsets": offsets,
        "r2": r2,
        "rms": rms,
        "min_r2": float(min_r2),
        "max_rms": float(max_rms) if max_rms is not None else None,
        "fit_ok_fraction": float(np.nanmean(fit_ok)) if fit_ok.size else np.nan,
        "residual_mean": np.nanmean(residuals, axis=2) if residuals.ndim == 3 else None,
        "valid_bins": valid_bins,
        "kept_bins": kept_bins,
        "peak_rejected": peak_rej,
        "channel_names": ch_names,
    }
    
    return data_dict


def extract_aperiodic_features(
    ctx: Any, # FeatureContext
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    
    valid, err = validate_extractor_inputs(ctx, "Aperiodic", min_epochs=2)
    if not valid:
        ctx.logger.warning(err)
        return pd.DataFrame(), [], {}
    
    epochs = ctx.epochs
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        ctx.logger.warning("Aperiodic: No EEG channels available; skipping extraction.")
        return pd.DataFrame(), [], {}
    
    config = ctx.config
    logger = ctx.logger
    sfreq = epochs.info["sfreq"]
    times = epochs.times
    min_samples = int(sfreq)
    
    all_data: Dict[str, Any] = {}
    qc_payload: Dict[str, Any] = {
        "segments": {},
        "channel_names": ch_names,
    }
    
    for seg_name in STANDARD_SEGMENTS:
        mask = get_segment_mask(ctx.windows, seg_name)
        if mask is None or np.sum(mask) < min_samples:
            continue
        t_seg = times[mask]
        seg_data = _extract_aperiodic_for_segment(
            epochs, picks, ch_names, seg_name,
            t_seg[0], t_seg[-1], bands, config, logger
        )
        qc_payload["segments"][seg_name] = seg_data.get("__qc__")
        seg_data.pop("__qc__", None)
        all_data.update(seg_data)
        logger.info(f"Computed Aperiodic for {seg_name}: [{t_seg[0]:.2f}, {t_seg[-1]:.2f}]")
    
    if not all_data:
        logger.warning("No valid segments for Aperiodic; returning empty result.")
        return pd.DataFrame(), [], {}
    
    df = pd.DataFrame(all_data)
    
    segments_done = sorted([k for k, v in qc_payload.get("segments", {}).items() if v])
    qc_payload["segments_computed"] = segments_done
    plateau_qc = qc_payload.get("segments", {}).get("plateau")
    baseline_qc = qc_payload.get("segments", {}).get("baseline")
    chosen = plateau_qc or baseline_qc
    if chosen:
        qc_payload["freqs"] = chosen.get("freqs")
        qc_payload["residual_mean"] = chosen.get("residual_mean")
        qc_payload["r2"] = chosen.get("r2")
        qc_payload["slopes"] = chosen.get("slopes")
        qc_payload["offsets"] = chosen.get("offsets")
        qc_payload["channel_names"] = chosen.get("channel_names")
    
    return df, list(df.columns), qc_payload


def extract_aperiodic_features_from_epochs(
    epochs: mne.Epochs,
    baseline_window: Tuple[float, float],
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    events_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("Aperiodic: No EEG channels available; skipping extraction.")
        return pd.DataFrame(), [], {}

    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    min_samples = int(sfreq)

    def _clamp_window(window: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if times.size == 0:
            return None
        start, end = float(window[0]), float(window[1])
        start = max(start, float(times[0]))
        end = min(end, float(times[-1]))
        if end <= start:
            return None
        return (start, end)

    baseline = _clamp_window(baseline_window)
    if baseline is None:
        return pd.DataFrame(), [], {"skipped_reason": "invalid_baseline_window"}

    from eeg_pipeline.utils.config.loader import get_config_value

    ramp_end = float(get_config_value(config, "feature_engineering.features.ramp_end", 3.0))
    plateau_window = get_config_value(config, "time_frequency_analysis.plateau_window", [3.0, 10.5])
    plateau = _clamp_window((float(plateau_window[0]), float(plateau_window[1])))
    ramp = _clamp_window((0.0, ramp_end))

    all_data: Dict[str, np.ndarray] = {}
    segments_done: List[str] = []

    baseline_data = _extract_aperiodic_for_segment(
        epochs,
        picks,
        ch_names,
        "baseline",
        baseline[0],
        baseline[1],
        bands,
        config,
        logger,
    )
    if baseline_data:
        all_data.update(baseline_data)
        segments_done.append("baseline")

    if ramp is not None:
        ramp_mask = (times >= ramp[0]) & (times <= ramp[1])
        if int(np.sum(ramp_mask)) >= min_samples:
            ramp_data = _extract_aperiodic_for_segment(
                epochs,
                picks,
                ch_names,
                "ramp",
                ramp[0],
                ramp[1],
                bands,
                config,
                logger,
            )
            if ramp_data:
                all_data.update(ramp_data)
                segments_done.append("ramp")

    if plateau is not None:
        plateau_mask = (times >= plateau[0]) & (times <= plateau[1])
        if int(np.sum(plateau_mask)) >= min_samples:
            plateau_data = _extract_aperiodic_for_segment(
                epochs,
                picks,
                ch_names,
                "plateau",
                plateau[0],
                plateau[1],
                bands,
                config,
                logger,
            )
            if plateau_data:
                all_data.update(plateau_data)
                segments_done.append("plateau")

    if not all_data:
        return pd.DataFrame(), [], {"skipped_reason": "empty_result"}

    df = pd.DataFrame(all_data)
    qc_payload = {
        "segments_computed": sorted(set(segments_done)),
        "baseline_window": (float(baseline[0]), float(baseline[1])),
        "plateau_window": (float(plateau[0]), float(plateau[1])) if plateau is not None else None,
    }
    return df, list(df.columns), qc_payload
