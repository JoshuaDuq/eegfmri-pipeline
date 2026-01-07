"""
Phase Feature Extraction
=========================

Phase-based features for EEG analysis:
- ITPC: Inter-Trial Phase Coherence (global across trials; leakage-safe)
- PAC: Phase-Amplitude Coupling (cross-frequency coupling)
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Any
import logging
import numpy as np
import pandas as pd
import mne
from scipy.signal import find_peaks

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.utils.analysis.windowing import make_mask_for_times, get_segment_masks
from eeg_pipeline.utils.config.loader import get_frequency_bands

# --- Helpers ---

def _get_itpc_method(config: Any) -> str:
    method = str(config.get("feature_engineering.itpc.method", "global")).strip().lower()
    if method not in {"loo", "global"}:
        raise ValueError(
            "Invalid ITPC method. Supported values are 'loo' and 'global'. "
            f"Got: {method!r}"
        )
    if method == "loo" and not bool(config.get("feature_engineering.itpc.allow_unsafe_loo", False)):
        raise ValueError(
            "ITPC method 'loo' is disabled by default because it creates cross-trial dependence "
            "and can cause leakage in trial-level analyses. Set "
            "feature_engineering.itpc.allow_unsafe_loo=true only if you compute features "
            "within CV folds and pass an explicit training mask."
        )
    return method

def _sharpness_log_ratio(
    x: np.ndarray,
    sfreq_hz: float,
    offset_ms: float,
    *,
    fmax_hz: Optional[float] = None,
) -> float:
    """Cole/Voytek-style sharpness ratio proxy (log peak sharpness / trough sharpness)."""
    x = np.asarray(x, dtype=float)
    if x.size < 10 or not np.isfinite(x).any() or not np.isfinite(sfreq_hz) or sfreq_hz <= 0:
        return np.nan

    dt = int(round(float(offset_ms) * float(sfreq_hz) / 1000.0))
    dt = max(1, min(dt, (x.size - 1) // 4))

    distance = None
    if fmax_hz is not None:
        try:
            fmax_hz = float(fmax_hz)
        except Exception:
            fmax_hz = None
    if fmax_hz is not None and np.isfinite(fmax_hz) and fmax_hz > 0:
        # Avoid multiple peaks per cycle in noisy signals (half-period at fmax).
        distance = max(1, int(round(float(sfreq_hz) / (2.0 * float(fmax_hz)))))

    y = np.nan_to_num(x, nan=np.nanmedian(x))
    peaks, _ = find_peaks(y, distance=distance)
    troughs, _ = find_peaks(-y, distance=distance)
    if peaks.size < 2 or troughs.size < 2:
        return np.nan

    def _mean_sharp(idxs: np.ndarray) -> float:
        vals: List[float] = []
        for i in idxs:
            if i - dt < 0 or i + dt >= y.size:
                continue
            vals.append(abs(y[i] - y[i - dt]) + abs(y[i] - y[i + dt]))
        return float(np.mean(vals)) if vals else np.nan

    sp = _mean_sharp(peaks)
    st = _mean_sharp(troughs)
    if np.isfinite(sp) and np.isfinite(st) and sp > 0 and st > 0:
        return float(np.log(sp / st))
    return np.nan

def _compute_loo_itpc(data, train_mask=None):
    """
    Compute Leave-One-Out ITPC.
    data: (n_epochs, n_ch, n_freqs, n_times) complex TFR
    Returns: (n_epochs, n_ch, n_freqs, n_times) ITPC values
    """
    n_epochs = data.shape[0]
    if n_epochs < 2: return np.zeros_like(np.abs(data))
    
    # Normalize to unit circle
    eps = 1e-12
    unit = data / (np.abs(data) + eps)
    
    if train_mask is None:
        train_mask = np.ones(n_epochs, dtype=bool)
    n_train = np.sum(train_mask)
    
    if n_train < 1: return np.zeros_like(np.abs(data))
    
    sum_train = np.sum(unit[train_mask], axis=0)  # (ch, freq, time)

    mean_test = sum_train / max(1, n_train)
    loo_itpc = np.broadcast_to(np.abs(mean_test), (n_epochs,) + mean_test.shape).copy()

    if n_train > 1:
        train_indices = np.flatnonzero(train_mask)
        if train_indices.size:
            loo_train = (sum_train[None, ...] - unit[train_indices]) / (n_train - 1)
            loo_itpc[train_indices] = np.abs(loo_train)

    return loo_itpc


def _compute_global_itpc_map(data: np.ndarray) -> np.ndarray:
    """Compute ITPC map across epochs: |mean_e exp(i*phi_e)|."""
    eps = 1e-12
    unit = data / (np.abs(data) + eps)
    return np.abs(np.mean(unit, axis=0))  # (ch, freq, time)


def _broadcast_per_trial(values_ch: np.ndarray, n_epochs: int) -> np.ndarray:
    values_ch = np.asarray(values_ch, dtype=float)
    if values_ch.ndim != 1:
        raise ValueError("Expected 1D channel vector for broadcasting.")
    return np.broadcast_to(values_ch[None, :], (int(n_epochs), int(values_ch.shape[0]))).copy()

# --- Main API ---

def extract_phase_features(
    ctx: Any, # FeatureContext
    bands: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    
    config = ctx.config
    epochs = ctx.epochs

    itpc_method = _get_itpc_method(config)
    
    # Ensure we have complex TFR (computed upstream when ITPC/PAC requested)
    tfr = ctx.tfr_complex
    if tfr is None:
        ctx.logger.error("Phase: complex TFR missing; skipping extraction.")
        return pd.DataFrame(), []
    
    data = tfr.data # (epochs, ch, freq, time)
    times = tfr.times
    freqs = tfr.freqs
    ch_names = tfr.info['ch_names']
    n_epochs = int(data.shape[0])
    
    # ITPC is defined across trials. A per-trial LOO variant can be used in limited
    # contexts but must be computed within CV folds to avoid leakage.
    if itpc_method == "loo":
        train_mask = getattr(ctx, "train_mask", None)
        if train_mask is None:
            raise ValueError(
                "ITPC(method='loo') requires ctx.train_mask to be provided (training set trials only). "
                "Compute ITPC within each CV fold to avoid leakage."
            )
        itpc_map = _compute_loo_itpc(data, train_mask=train_mask)  # (epochs, ch, freq, time)
    else:
        itpc_map = _compute_global_itpc_map(data)  # (ch, freq, time)
    
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)
    
    results = {}
    spatial_modes = list(getattr(ctx, "spatial_modes", ["roi", "global"]))
    roi_map = {}
    if "roi" in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)
    
    segments = list(get_segment_masks(epochs.times, ctx.windows, config).keys())
    if not segments:
        segments = ["full"]
    
    itpc_cfg = config.get("feature_engineering.itpc", {}) if hasattr(config, "get") else {}
    baseline_correction = str(itpc_cfg.get("baseline_correction", "none")).strip().lower()
    if baseline_correction not in {"none", "subtract"}:
        baseline_correction = "none"

    for seg in segments:
        if seg == "full":
            seg_mask_tfr = np.ones_like(times, dtype=bool)
        else:
            seg_mask_tfr = make_mask_for_times(ctx.windows, seg, times)
            
        if not np.any(seg_mask_tfr): continue

        baseline_mask_tfr = None
        if baseline_correction == "subtract":
            try:
                baseline_mask_tfr = make_mask_for_times(ctx.windows, "baseline", times)
            except Exception:
                baseline_mask_tfr = None
            if baseline_mask_tfr is not None and not np.any(baseline_mask_tfr):
                baseline_mask_tfr = None
        
        if itpc_method == "loo":
            itpc_seg = itpc_map[..., seg_mask_tfr]  # (epochs, ch, freq, time_seg)
            itpc_seg_mean_t = np.nanmean(itpc_seg, axis=-1)  # (epochs, ch, freq)
            base_seg_mean_t = None
            if baseline_mask_tfr is not None and seg != "baseline":
                itpc_base = itpc_map[..., baseline_mask_tfr]
                base_seg_mean_t = np.nanmean(itpc_base, axis=-1)  # (epochs, ch, freq)
        else:
            itpc_seg = itpc_map[..., seg_mask_tfr]  # (ch, freq, time_seg)
            itpc_seg_mean_t = np.nanmean(itpc_seg, axis=-1)  # (ch, freq)
            base_seg_mean_t = None
            if baseline_mask_tfr is not None and seg != "baseline":
                itpc_base = itpc_map[..., baseline_mask_tfr]
                base_seg_mean_t = np.nanmean(itpc_base, axis=-1)  # (ch, freq)
        
        for band in bands:
            if band not in freq_bands: continue
            fmin, fmax = freq_bands[band]
            f_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(f_mask): continue
            
            if itpc_method == "loo":
                itpc_band = np.nanmean(itpc_seg_mean_t[..., f_mask], axis=-1)  # (epochs, ch)
                if base_seg_mean_t is not None:
                    itpc_band = itpc_band - np.nanmean(base_seg_mean_t[..., f_mask], axis=-1)
            else:
                itpc_band_ch = np.nanmean(itpc_seg_mean_t[:, f_mask], axis=-1)  # (ch,)
                if base_seg_mean_t is not None:
                    itpc_band_ch = itpc_band_ch - np.nanmean(base_seg_mean_t[:, f_mask], axis=-1)
                itpc_band = _broadcast_per_trial(itpc_band_ch, n_epochs)  # (epochs, ch)
            
            # Per-channel
            if 'channels' in spatial_modes:
                for i, ch in enumerate(ch_names):
                    col = NamingSchema.build("itpc", seg, band, "ch", "val", channel=ch)
                    results[col] = itpc_band[:, i]
            
            # ROI Mean
            if 'roi' in spatial_modes and roi_map:
                for roi_name, ch_indices in roi_map.items():
                    if ch_indices:
                        roi_val = np.nanmean(itpc_band[:, ch_indices], axis=1)
                        col = NamingSchema.build("itpc", seg, band, "roi", "val", channel=roi_name)
                        results[col] = roi_val
                        
            # Global Mean
            if 'global' in spatial_modes:
                global_val = np.nanmean(itpc_band, axis=1)
                col = NamingSchema.build("itpc", seg, band, "global", "val")
                results[col] = global_val
                
    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)


def compute_pac_comodulograms(
    tfr_complex: mne.time_frequency.EpochsTFR,
    freqs: np.ndarray,
    times: np.ndarray,
    info: mne.Info,
    config: Any,
    logger: logging.Logger,
    *,
    segment_name: str = "active",
    segment_window: Optional[Tuple[float, float]] = None,
    spatial_modes: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compute Phase-Amplitude Coupling (PAC) using Mean Vector Length (MVL).
    
    Returns:
        pac_df: Aggregated PAC (mean over trials) per channel & freq-pair.
        pac_phase_freqs: Phase frequencies used.
        pac_amp_freqs: Amplitude frequencies used.
        pac_trials_df: Trial-wise PAC (averaged over freq pairs per band or selected pairs).
        pac_time_df: Time-resolved PAC (optional, currently None).
    """
    if tfr_complex is None:
        return None, None, None, None, None
        
    data = tfr_complex.data  # (n_epochs, n_ch, n_freqs, n_times)
    n_epochs, n_ch, _, _ = data.shape

    tfr_freqs = np.asarray(getattr(tfr_complex, "freqs", freqs), dtype=float)
    tfr_times = np.asarray(getattr(tfr_complex, "times", times), dtype=float)

    if segment_window is not None:
        start, end = float(segment_window[0]), float(segment_window[1])
        t_mask = (tfr_times >= start) & (tfr_times < end)
        if not np.any(t_mask):
            logger.warning("PAC: empty time window for %s [%0.3f, %0.3f)", segment_name, start, end)
            return None, None, None, None, None
        data = data[..., t_mask]
        tfr_times = tfr_times[t_mask]

    n_times = data.shape[-1]
    if n_times == 0:
        logger.warning("PAC: no samples available for %s; skipping", segment_name)
        return None, None, None, None, None

    pac_cfg = config.get("feature_engineering.pac", {}) if config is not None else {}
    method = str(pac_cfg.get("method", "mvl")).strip().lower()
    if method != "mvl":
        raise ValueError(f"PAC: unsupported method '{method}'. Only 'mvl' is implemented.")
    n_surrogates = int(pac_cfg.get("n_surrogates", 0))
    min_epochs = int(pac_cfg.get("min_epochs", 2))
    if n_epochs < min_epochs:
        logger.warning("PAC: insufficient epochs (%d < %d); skipping", n_epochs, min_epochs)
        return None, None, None, None, None

    phase_range = pac_cfg.get("phase_range", [4.0, 8.0])
    amp_range = pac_cfg.get("amp_range", [30.0, 80.0])
    try:
        phase_min, phase_max = float(phase_range[0]), float(phase_range[1])
        amp_min, amp_max = float(amp_range[0]), float(amp_range[1])
    except Exception:
        phase_min, phase_max = 4.0, 8.0
        amp_min, amp_max = 30.0, 80.0

    phase_mask = (tfr_freqs >= phase_min) & (tfr_freqs <= phase_max)
    amp_mask = (tfr_freqs >= amp_min) & (tfr_freqs <= amp_max)
    
    if not np.any(phase_mask) or not np.any(amp_mask):
        logger.warning("No valid phase/amplitude frequencies for PAC")
        return None, None, None, None, None
        
    phase_freqs = tfr_freqs[phase_mask]
    amp_freqs = tfr_freqs[amp_mask]
    
    phase_indices = np.where(phase_mask)[0]
    amp_indices = np.where(amp_mask)[0]
    
    # Pre-compute phase and amplitude
    # We need (epochs, ch, freq, time)
    # Be careful with memory.
    
    # Logic: For each channel, for each trial, compute PAC Matrix (n_phase, n_amp)
    # MVL = | mean( A_amp * exp(i * phi_phase) ) | over time
    
    # We want to perform this efficienty.
    # 1. Extract Phase angles for phase freqs
    # 2. Extract Amplitudes for amp freqs
    
    normalize = bool(pac_cfg.get("normalize", True))
    eps_amp = float(config.get("feature_engineering.constants.epsilon_amp", 1e-12)) if hasattr(config, "get") else 1e-12
    seed = pac_cfg.get("random_seed", None)
    rng = np.random.default_rng(None if seed in (None, "", 0) else int(seed))
    
    # PAC: For each pair (fp, fa), compute |mean(A * expphi)|
    # We can't form the full (nep, nch, nph, namp, ntimes) tensor -> too big.
    
    # For trial-wise: we probably want specific band pairs?
    # e.g. Theta-Gamma, Alpha-Gamma matches.
    # Let's aggregate by standard bands for trial-wise output
    
    # Output arrays
    # Aggregated: (n_ch, n_phase, n_amp) - Mean over epochs and time
    # Trial-wise: (n_epochs, n_ch, n_phase, n_amp) - Mean over time
    
    # Use TFR's own channel names - they match the data dimensions
    # (info passed in may have more channels than the TFR data)
    tfr_ch_names = tfr_complex.info['ch_names']
    if len(tfr_ch_names) != n_ch:
        logger.warning(
            "PAC channel count mismatch: TFR has %d channels but info has %d; using TFR channel list",
            n_ch, len(tfr_ch_names)
        )
    ch_names = tfr_ch_names[:n_ch] if len(tfr_ch_names) >= n_ch else tfr_ch_names
    
    trials_pac_list = []
    pair_channel_data = {}
    pair_channel_data_z = {}

    tf_bands = config.get("time_frequency_analysis.bands", {}) if config is not None else {}
    requested_pairs = pac_cfg.get("pairs")
    if requested_pairs is None:
        requested_pairs = [("theta", "gamma"), ("alpha", "gamma")]

    allow_harmonic_overlap = bool(pac_cfg.get("allow_harmonic_overlap", False))
    max_harm = int(pac_cfg.get("max_harmonic", 6))
    tol_hz = float(pac_cfg.get("harmonic_tolerance_hz", 1.0))

    pairs: List[Tuple[str, str]] = []
    for p in requested_pairs:
        if not p or len(p) < 2:
            continue
        pairs.append((str(p[0]), str(p[1])))

    valid_pairs: List[Tuple[str, str, Tuple[float, float], Tuple[float, float]]] = []
    for phase_band, amp_band in pairs:
        if phase_band not in tf_bands or amp_band not in tf_bands:
            continue
        try:
            pmin, pmax = float(tf_bands[phase_band][0]), float(tf_bands[phase_band][1])
            amin, amax = float(tf_bands[amp_band][0]), float(tf_bands[amp_band][1])
        except Exception:
            continue
        if amin <= pmax:
            continue

        if not allow_harmonic_overlap and max_harm >= 2 and np.isfinite(tol_hz) and tol_hz >= 0:
            overlaps = False
            for h in range(2, max_harm + 1):
                hmin = (pmin * h) - tol_hz
                hmax = (pmax * h) + tol_hz
                if (amax >= hmin) and (amin <= hmax):
                    overlaps = True
                    break
            if overlaps:
                logger.warning(
                    "PAC: skipping pair %s→%s due to harmonic overlap (phase=%0.1f-%0.1fHz, amp=%0.1f-%0.1fHz). "
                    "Set feature_engineering.pac.allow_harmonic_overlap=true to override.",
                    str(phase_band),
                    str(amp_band),
                    float(pmin),
                    float(pmax),
                    float(amin),
                    float(amax),
                )
                continue

        valid_pairs.append((phase_band, amp_band, (pmin, pmax), (amin, amax)))

    if not valid_pairs:
        logger.warning("PAC: no valid band pairs found in config; skipping")
        return None, phase_freqs, amp_freqs, None, None

    for ch_idx in range(n_ch):
        # Compute per requested band-pair PAC for this channel.
        for phase_band, amp_band, (pmin, pmax), (amin, amax) in valid_pairs:
            p_mask_band = (phase_freqs >= pmin) & (phase_freqs <= pmax)
            a_mask_band = (amp_freqs >= amin) & (amp_freqs <= amax)
            if not np.any(p_mask_band) or not np.any(a_mask_band):
                continue

            p_idx = phase_indices[p_mask_band]
            a_idx = amp_indices[a_mask_band]

            # Phase: average unit vectors across phase freqs -> complex (epochs, times)
            phi = np.angle(data[:, ch_idx, p_idx, :])
            z_p = np.nanmean(np.exp(1j * phi), axis=1)

            # Amplitude: average across amp freqs -> (epochs, times)
            A = np.nanmean(np.abs(data[:, ch_idx, a_idx, :]), axis=1)
            denom = np.nansum(A, axis=1) + eps_amp if normalize else float(n_times)
            numer = np.nansum(A * z_p, axis=1) if normalize else np.nanmean(A * z_p, axis=1)
            pac_raw = np.abs(numer / denom)

            band_pair_name = f"{phase_band}_{amp_band}"
            if band_pair_name not in pair_channel_data:
                pair_channel_data[band_pair_name] = np.full((n_epochs, n_ch), np.nan)
            pair_channel_data[band_pair_name][:, ch_idx] = pac_raw

            if n_surrogates > 0 and n_times > 3:
                sur = np.full((n_epochs, n_surrogates), np.nan, dtype=float)
                for ep in range(n_epochs):
                    a_ep = A[ep]
                    z_ep = z_p[ep]
                    if not np.isfinite(a_ep).any() or not np.isfinite(z_ep).any():
                        continue
                    denom_ep = (np.nansum(a_ep) + eps_amp) if normalize else float(n_times)
                    for s in range(n_surrogates):
                        shift = int(rng.integers(1, n_times - 1))
                        a_shift = np.roll(a_ep, shift)
                        numer_s = np.nansum(a_shift * z_ep) if normalize else np.nanmean(a_shift * z_ep)
                        sur[ep, s] = float(np.abs(numer_s / denom_ep))
                mu = np.nanmean(sur, axis=1)
                sd = np.nanstd(sur, axis=1, ddof=1)
                sd = np.where(sd > 0, sd, np.nan)
                pac_z = (pac_raw - mu) / sd

                if band_pair_name not in pair_channel_data_z:
                    pair_channel_data_z[band_pair_name] = np.full((n_epochs, n_ch), np.nan)
                pair_channel_data_z[band_pair_name][:, ch_idx] = pac_z

    # Spatial aggregation for PAC
    if spatial_modes is None:
        if hasattr(config, "get"):
            spatial_modes = config.get("feature_engineering.spatial_modes", ["roi", "global"])
        else:
            spatial_modes = ["roi", "global"]
    
    # Build ROI map
    roi_map = {}
    if 'roi' in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)

    for band_pair, matrix in pair_channel_data.items():
        # matrix is (n_epochs, n_ch)
        
        # Channels
        if 'channels' in spatial_modes:
            for ch_idx, ch in enumerate(ch_names):
                col = NamingSchema.build("pac", segment_name, band_pair, "ch", "val", channel=ch)
                trials_pac_list.append(pd.Series(matrix[:, ch_idx], name=col))
        
        # ROI
        if 'roi' in spatial_modes and roi_map:
            for roi_name, ch_indices in roi_map.items():
                if ch_indices:
                    roi_val = np.nanmean(matrix[:, ch_indices], axis=1)
                    col = NamingSchema.build("pac", segment_name, band_pair, "roi", "val", channel=roi_name)
                    trials_pac_list.append(pd.Series(roi_val, name=col))
                    
        # Global
        if 'global' in spatial_modes:
            global_val = np.nanmean(matrix, axis=1)
            col = NamingSchema.build("pac", segment_name, band_pair, "global", "val")
            trials_pac_list.append(pd.Series(global_val, name=col))

    if pair_channel_data_z:
        for band_pair, matrix in pair_channel_data_z.items():
            if 'channels' in spatial_modes:
                for ch_idx, ch in enumerate(ch_names):
                    col = NamingSchema.build("pac", segment_name, band_pair, "ch", "z", channel=ch)
                    trials_pac_list.append(pd.Series(matrix[:, ch_idx], name=col))
            if 'roi' in spatial_modes and roi_map:
                for roi_name, ch_indices in roi_map.items():
                    if ch_indices:
                        roi_val = np.nanmean(matrix[:, ch_indices], axis=1)
                        col = NamingSchema.build("pac", segment_name, band_pair, "roi", "z", channel=roi_name)
                        trials_pac_list.append(pd.Series(roi_val, name=col))
            if 'global' in spatial_modes:
                global_val = np.nanmean(matrix, axis=1)
                col = NamingSchema.build("pac", segment_name, band_pair, "global", "z")
                trials_pac_list.append(pd.Series(global_val, name=col))

    # Combine all trial series
    if not trials_pac_list:
        return None, phase_freqs, amp_freqs, None, None
        
    pac_trials_df = pd.concat(trials_pac_list, axis=1)
    return None, phase_freqs, amp_freqs, pac_trials_df, None


def extract_itpc_from_precomputed(
    precomputed: Any
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute ITPC-style metrics directly from precomputed band phases.
    
    ITPC = | (1/N) * sum( exp(i*phase) ) |
    """
    logger = getattr(precomputed, "logger", None)
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        if logger is not None:
            logger.warning("ITPC: %s; skipping extraction.", err_msg)
        return pd.DataFrame(), []

    cfg = precomputed.config or {}
    itpc_method = _get_itpc_method(cfg)
    if itpc_method == "loo":
        raise ValueError(
            "ITPC(method='loo') is not supported in precomputed mode because it requires "
            "fold-specific training masks to avoid leakage. Use ITPC(method='global') "
            "or compute LOO-ITPC within your CV loop."
        )
        
    windows = precomputed.windows
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    masks = get_segment_masks(precomputed.times, windows, precomputed.config)
    if not masks:
        return pd.DataFrame(), []
    
    ch_names = precomputed.ch_names
    n_epochs = precomputed.data.shape[0]
    if n_epochs < 2:
        if logger is not None:
            logger.warning("ITPC: Fewer than 2 epochs available; skipping extraction.")
        return pd.DataFrame(), []
    
    itpc_cfg = cfg.get("feature_engineering.itpc", {}) if hasattr(cfg, "get") else {}
    baseline_correction = str(itpc_cfg.get("baseline_correction", "none")).strip().lower()
    if baseline_correction not in {"none", "subtract"}:
        baseline_correction = "none"

    results = {}
    
    for band, bd in precomputed.band_data.items():
        # bd.phase: (n_epochs, n_ch, n_times)
        phases = bd.phase
        if phases is None or phases.size == 0:
            continue

        complex_vectors = np.exp(1j * phases)  # (epochs, ch, time)

        baseline_mask = masks.get("baseline") if baseline_correction == "subtract" else None
        baseline_itpc_ch = None
        if baseline_mask is not None and np.any(baseline_mask):
            base_map = np.abs(np.mean(complex_vectors[:, :, baseline_mask], axis=0))  # (ch, time)
            baseline_itpc_ch = np.nanmean(base_map, axis=1)  # (ch,)

        # Get spatial modes and ROI map from precomputed or context
        spatial_modes = getattr(precomputed, "spatial_modes", None) or ["roi", "global"]
        roi_map = {}
        if "roi" in spatial_modes:
            from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
            from eeg_pipeline.utils.analysis.channels import build_roi_map
            roi_defs = get_roi_definitions(cfg)
            if roi_defs:
                roi_map = build_roi_map(ch_names, roi_defs)

        for seg_name, mask in masks.items():
            if mask is None or not np.any(mask):
                continue
            itpc_map = np.abs(np.mean(complex_vectors[:, :, mask], axis=0))  # (ch, time)
            itpc_ch = np.nanmean(itpc_map, axis=1)  # (ch,)
            if baseline_itpc_ch is not None and seg_name != "baseline":
                itpc_ch = itpc_ch - baseline_itpc_ch

            itpc_seg = _broadcast_per_trial(itpc_ch, n_epochs)  # (epochs, ch)
            
            # Channels
            if 'channels' in spatial_modes:
                for ch_idx, ch in enumerate(ch_names):
                    col = NamingSchema.build("itpc", seg_name, band, "ch", "val", channel=ch)
                    results[col] = itpc_seg[:, ch_idx]

            # ROI
            if 'roi' in spatial_modes and roi_map:
                for roi_name, ch_indices in roi_map.items():
                    if ch_indices:
                        roi_val = np.nanmean(itpc_seg[:, ch_indices], axis=1)
                        col = NamingSchema.build("itpc", seg_name, band, "roi", "val", channel=roi_name)
                        results[col] = roi_val

            # Global
            if 'global' in spatial_modes:
                global_val = np.nanmean(itpc_seg, axis=1)
                col = NamingSchema.build("itpc", seg_name, band, "global", "val")
                results[col] = global_val
                
    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)


def extract_pac_from_precomputed(
    precomputed: Any,
    config: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute PAC using precomputed analytic signals.
    
    PAC(fp, fa) = | mean( A_fa * exp(i * phi_fp) ) | over time.
    """
    logger = getattr(precomputed, "logger", None)
    config = config or getattr(precomputed, "config", None) or {}
    is_valid, err_msg = validate_precomputed(precomputed, require_windows=True, require_bands=True)
    if not is_valid:
        if logger is not None:
            logger.warning("PAC (precomputed): %s; skipping extraction.", err_msg)
        return pd.DataFrame(), []

    # Get PAC config
    pac_cfg = config.get("feature_engineering.pac", {}) if hasattr(config, "get") else {}
    method = str(pac_cfg.get("method", "mvl")).strip().lower()
    if method != "mvl":
        raise ValueError(f"PAC (precomputed): unsupported method '{method}'. Only 'mvl' is implemented.")
    n_surrogates = int(pac_cfg.get("n_surrogates", 0))
    requested_pairs = pac_cfg.get("pairs", [("theta", "gamma"), ("alpha", "gamma")])
    normalize = bool(pac_cfg.get("normalize", True))
    eps_amp = float(config.get("feature_engineering.constants.epsilon_amp", 1e-12)) if hasattr(config, "get") else 1e-12
    seed = pac_cfg.get("random_seed", None)
    rng = np.random.default_rng(None if seed in (None, "", 0) else int(seed))
    allow_harmonic_overlap = bool(pac_cfg.get("allow_harmonic_overlap", False))
    max_harm = int(pac_cfg.get("max_harmonic", 6))
    tol_hz = float(pac_cfg.get("harmonic_tolerance_hz", 1.0))
    
    # Get spatial modes and ROI map
    spatial_modes = getattr(precomputed, "spatial_modes", None)
    if spatial_modes is None:
        spatial_modes = config.get("feature_engineering.spatial_modes", ["roi", "global"]) if hasattr(config, "get") else ["roi", "global"]
    
    # Standard bands for pair lookup
    tf_bands = config.get("time_frequency_analysis.bands", {}) if hasattr(config, "get") else {}
    
    ch_names = precomputed.ch_names
    n_epochs = precomputed.data.shape[0]
    windows = precomputed.windows
    segment_name = getattr(windows, "name", "active") or "active"
    mask = getattr(windows, "active_mask", None)
    
    if mask is None or not np.any(mask):
        return pd.DataFrame(), []
    n_times = int(np.sum(mask))

    # Pre-calculate sqrt(power) for all bands to get amplitude
    amplitudes = {}
    phases = {}
    for band, bd in precomputed.band_data.items():
        if bd.power is not None:
            # Power is (nep, nch, ntimes)
            amplitudes[band] = np.sqrt(np.maximum(bd.power, 0))
        if bd.phase is not None:
            phases[band] = bd.phase

    results = {}
    
    roi_map = {}
    if 'roi' in spatial_modes:
        from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
        from eeg_pipeline.utils.analysis.channels import build_roi_map
        roi_defs = get_roi_definitions(config)
        if roi_defs:
            roi_map = build_roi_map(ch_names, roi_defs)

    for p_band, a_band in requested_pairs:
        if not allow_harmonic_overlap and max_harm >= 2 and p_band in tf_bands and a_band in tf_bands:
            try:
                pmin, pmax = float(tf_bands[p_band][0]), float(tf_bands[p_band][1])
                amin, amax = float(tf_bands[a_band][0]), float(tf_bands[a_band][1])
            except Exception:
                pmin = pmax = amin = amax = np.nan
            if np.isfinite(pmin) and np.isfinite(pmax) and np.isfinite(amin) and np.isfinite(amax):
                overlaps = False
                for h in range(2, max_harm + 1):
                    hmin = (pmin * h) - tol_hz
                    hmax = (pmax * h) + tol_hz
                    if (amax >= hmin) and (amin <= hmax):
                        overlaps = True
                        break
                if overlaps:
                    if logger is not None:
                        logger.warning(
                            "PAC (precomputed): skipping pair %s→%s due to harmonic overlap (phase=%0.1f-%0.1fHz, amp=%0.1f-%0.1fHz).",
                            str(p_band),
                            str(a_band),
                            float(pmin),
                            float(pmax),
                            float(amin),
                            float(amax),
                        )
                    continue
        if p_band not in phases or a_band not in amplitudes:
            continue
            
        phi = phases[p_band][..., mask]
        amp = amplitudes[a_band][..., mask]
        
        # PAC(trial, ch) = | sum_t( amp(t) * exp(i*phi(t)) ) / sum_t amp(t) |
        expiphi = np.exp(1j * phi)
        if normalize:
            denom = np.nansum(amp, axis=-1) + eps_amp
            numer = np.nansum(amp * expiphi, axis=-1)
            pac_val = np.abs(numer / denom)
        else:
            pac_val = np.abs(np.nanmean(amp * expiphi, axis=-1))

        pac_z = None
        if n_surrogates > 0 and n_times > 3:
            pac_z = np.full_like(pac_val, np.nan, dtype=float)
            for ep in range(n_epochs):
                for ch in range(n_ch):
                    a = amp[ep, ch]
                    z = expiphi[ep, ch]
                    if not np.isfinite(a).any() or not np.isfinite(z).any():
                        continue
                    denom_ep = (np.nansum(a) + eps_amp) if normalize else float(n_times)
                    sur = np.full((n_surrogates,), np.nan, dtype=float)
                    for s in range(n_surrogates):
                        shift = int(rng.integers(1, n_times - 1))
                        a_shift = np.roll(a, shift)
                        numer_s = np.nansum(a_shift * z) if normalize else np.nanmean(a_shift * z)
                        sur[s] = float(np.abs(numer_s / denom_ep))
                    mu = float(np.nanmean(sur))
                    sd = float(np.nanstd(sur, ddof=1))
                    if np.isfinite(sd) and sd > 0:
                        pac_z[ep, ch] = (pac_val[ep, ch] - mu) / sd
        
        pair_label = f"{p_band}_{a_band}"
        
        # Channels
        if 'channels' in spatial_modes:
            for i, ch in enumerate(ch_names):
                col = NamingSchema.build("pac", segment_name, pair_label, "ch", "val", channel=ch)
                results[col] = pac_val[:, i]
                if pac_z is not None:
                    colz = NamingSchema.build("pac", segment_name, pair_label, "ch", "z", channel=ch)
                    results[colz] = pac_z[:, i]
                
        # ROI
        if 'roi' in spatial_modes and roi_map:
            for roi_name, idxs in roi_map.items():
                if idxs:
                    roi_val = np.nanmean(pac_val[:, idxs], axis=1)
                    col = NamingSchema.build("pac", segment_name, pair_label, "roi", "val", channel=roi_name)
                    results[col] = roi_val
                    if pac_z is not None:
                        roi_z = np.nanmean(pac_z[:, idxs], axis=1)
                        colz = NamingSchema.build("pac", segment_name, pair_label, "roi", "z", channel=roi_name)
                        results[colz] = roi_z
                    
        # Global
        if 'global' in spatial_modes:
            global_val = np.nanmean(pac_val, axis=1)
            col = NamingSchema.build("pac", segment_name, pair_label, "global", "val")
            results[col] = global_val
            if pac_z is not None:
                glob_z = np.nanmean(pac_z, axis=1)
                colz = NamingSchema.build("pac", segment_name, pair_label, "global", "z")
                results[colz] = glob_z

        # Waveform confound metrics for phase band (non-sinusoidality proxy).
        if bool(pac_cfg.get("compute_waveform_qc", False)) and p_band in precomputed.band_data:
            try:
                filt = precomputed.band_data[p_band].filtered[..., mask]  # (epochs, ch, time)
                offset_ms = float(pac_cfg.get("waveform_offset_ms", 5.0))
                sf = float(getattr(precomputed, "sfreq", np.nan))
                fmax_hz = float(getattr(precomputed.band_data[p_band], "fmax", np.nan))
                if not np.isfinite(sf) or sf <= 0:
                    raise ValueError("missing/invalid sampling rate for waveform QC")

                ratios = np.full((n_epochs, n_ch), np.nan)
                for ep in range(n_epochs):
                    for ch in range(n_ch):
                        ratios[ep, ch] = _sharpness_log_ratio(filt[ep, ch], sf, offset_ms, fmax_hz=fmax_hz)

                # Channels
                if 'channels' in spatial_modes:
                    for i, ch_name in enumerate(ch_names):
                        col = NamingSchema.build("pac", segment_name, pair_label, "ch", "lf_sharpness_ratio", channel=ch_name)
                        results[col] = ratios[:, i]
                # ROI
                if 'roi' in spatial_modes and roi_map:
                    for roi_name, idxs in roi_map.items():
                        if idxs:
                            roi_val = np.nanmean(ratios[:, idxs], axis=1)
                            col = NamingSchema.build("pac", segment_name, pair_label, "roi", "lf_sharpness_ratio", channel=roi_name)
                            results[col] = roi_val
                # Global
                if 'global' in spatial_modes:
                    glob = np.nanmean(ratios, axis=1)
                    col = NamingSchema.build("pac", segment_name, pair_label, "global", "lf_sharpness_ratio")
                    results[col] = glob
            except Exception as exc:
                if logger is not None:
                    logger.warning("PAC waveform QC failed for pair %s: %s", pair_label, exc)
            
    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)
