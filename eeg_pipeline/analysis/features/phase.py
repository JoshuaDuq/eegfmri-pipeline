"""
Phase Feature Extraction
=========================

Phase-based features for EEG analysis:
- ITPC: Inter-Trial Phase Coherence (trial-wise via Leave-One-Out)
- PAC: Phase-Amplitude Coupling (cross-frequency coupling)
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Any
import logging
import numpy as np
import pandas as pd
import mne

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import validate_precomputed
from eeg_pipeline.utils.analysis.windowing import make_mask_for_times, get_segment_masks
from eeg_pipeline.utils.config.loader import get_frequency_bands

# --- Helpers ---

def _get_itpc_method(config: Any) -> str:
    method = str(config.get("feature_engineering.itpc.method", "loo")).strip().lower()
    if method not in {"loo", "global"}:
        raise ValueError(
            "Invalid ITPC method. Supported values are 'loo' and 'global'. "
            f"Got: {method!r}"
        )
    return method

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

# --- Main API ---

def extract_phase_features(
    ctx: Any, # FeatureContext
    bands: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    
    config = ctx.config
    epochs = ctx.epochs

    itpc_method = _get_itpc_method(config)
    if itpc_method != "loo":
        raise ValueError(
            "ITPC method 'global' is not supported for trial-wise feature columns. "
            "Set feature_engineering.itpc.method='loo' for per-trial ITPC."
        )
    
    # Ensure we have complex TFR (computed upstream when ITPC/PAC requested)
    tfr = ctx.tfr_complex
    if tfr is None:
        ctx.logger.error("Phase: complex TFR missing; skipping extraction.")
        return pd.DataFrame(), []
    
    data = tfr.data # (epochs, ch, freq, time)
    times = tfr.times
    freqs = tfr.freqs
    ch_names = tfr.info['ch_names']
    
    # 1. ITPC (Trial-wise LOO)
    # ------------------------
    # Compute full LOO map once
    itpc_map = _compute_loo_itpc(data) #(epochs, ch, freq, time) (Real)
    
    freq_bands = get_frequency_bands(config)
    
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
    
    for seg in segments:
        if seg == "full":
            seg_mask_tfr = np.ones_like(times, dtype=bool)
        else:
            seg_mask_tfr = make_mask_for_times(ctx.windows, seg, times)
            
        if not np.any(seg_mask_tfr): continue
        
        # Average over time window
        # slice: (epochs, ch, freq, time_seg)
        itpc_seg = itpc_map[..., seg_mask_tfr]
        # Mean over time -> (epochs, ch, freq)
        itpc_seg_mean_t = np.mean(itpc_seg, axis=-1)
        
        for band in bands:
            if band not in freq_bands: continue
            fmin, fmax = freq_bands[band]
            f_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(f_mask): continue
            
            # Mean over freq -> (epochs, ch)
            itpc_band = np.mean(itpc_seg_mean_t[..., f_mask], axis=-1)
            
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
    if n_surrogates > 0:
        raise ValueError(
            "PAC: surrogate correction is not implemented. "
            "Set feature_engineering.pac.n_surrogates=0."
        )
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
    
    # Let's vectorize over trials and time?
    # (n_epochs, n_ch, n_phase, n_times)
    phases = np.angle(data[:, :, phase_indices, :])
    expphi = np.exp(1j * phases)
    
    # (n_epochs, n_ch, n_amp, n_times)
    amps = np.abs(data[:, :, amp_indices, :])
    
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
    pair_channel_data: Dict[str, np.ndarray] = {}

    tf_bands = config.get("time_frequency_analysis.bands", {}) if config is not None else {}
    requested_pairs = pac_cfg.get("pairs")
    if requested_pairs is None:
        requested_pairs = [("theta", "gamma"), ("alpha", "gamma")]

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
        valid_pairs.append((phase_band, amp_band, (pmin, pmax), (amin, amax)))

    if not valid_pairs:
        logger.warning("PAC: no valid band pairs found in config; skipping")
        return None, phase_freqs, amp_freqs, None, None
    
    for ch_idx in range(n_ch):
        # (n_epochs, n_phase, n_times)
        ch_expphi = expphi[:, ch_idx, :, :]
        # (n_epochs, n_amp, n_times)
        ch_amps = amps[:, ch_idx, :, :]
        
        # MVL per trial:
        # Cross product: (n_epochs, n_phase, 1, n_time) * (n_epochs, 1, n_amp, n_time)
        # = (n_epochs, n_phase, n_amp, n_time)
        # Sum over time
        
        # Optimize: tensordot over time?
        # A * exp(iphi) summed over t
        # shape: (n_epochs, n_phase, n_amp)
        # einsum: 'ept, eat -> epa'
        
        mvl_complex = np.einsum('ipt,iat->ipa', ch_expphi, ch_amps)
        mvl_complex /= n_times # Mean over time
        
        pac_val = np.abs(mvl_complex) # (n_epochs, n_phase, n_amp)

        # Prepare per-pair PAC values
        for phase_band, amp_band, (pmin, pmax), (amin, amax) in valid_pairs:
            p_mask_band = (phase_freqs >= pmin) & (phase_freqs <= pmax)
            a_mask_band = (amp_freqs >= amin) & (amp_freqs <= amax)
            if not np.any(p_mask_band) or not np.any(a_mask_band):
                continue
            
            sub_pac = pac_val[:, p_mask_band, :][:, :, a_mask_band]
            # Mean across phase/amp freqs -> (n_epochs,) for this channel/pair
            pair_vals = np.mean(sub_pac, axis=(1, 2))
            
            band_pair_name = f"{phase_band}_{amp_band}"
            
            # Store in a temporary structure for later spatial aggregation if needed
            # For simplicity, we'll just handle it per channel here and add Roi/Global later
            # Wait, better to store all channel values for this pair first.
            if band_pair_name not in pair_channel_data:
                pair_channel_data[band_pair_name] = np.full((n_epochs, n_ch), np.nan)
            pair_channel_data[band_pair_name][:, ch_idx] = pair_vals

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
    if itpc_method != "loo":
        raise ValueError(
            "ITPC method 'global' is not supported for trial-wise feature columns from precomputed phases. "
            "Set feature_engineering.itpc.method='loo' for per-trial ITPC."
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
    
    results = {}
    
    for band, bd in precomputed.band_data.items():
        # bd.phase: (n_epochs, n_ch, n_times)
        phases = bd.phase
        if phases is None or phases.size == 0:
            continue

        # Trial-wise ITPC via leave-one-out (LOO):
        #   itpc_i(t) = | (sum_{k!=i} exp(1j*phi_k(t))) / (N-1) |
        # This produces one value per trial after time-averaging within each segment.
        complex_vectors = np.exp(1j * phases)
        sum_vec = np.sum(complex_vectors, axis=0)  # (n_ch, n_times)

        loo_means = (sum_vec[None, :, :] - complex_vectors) / (n_epochs - 1)
        loo_itpc_t = np.abs(loo_means)  # (n_epochs, n_ch, n_times)

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
            loo_itpc_seg = np.mean(loo_itpc_t[:, :, mask], axis=2)  # (n_epochs, n_ch)
            
            # Channels
            if 'channels' in spatial_modes:
                for ch_idx, ch in enumerate(ch_names):
                    col = NamingSchema.build("itpc", seg_name, band, "ch", "val", channel=ch)
                    results[col] = loo_itpc_seg[:, ch_idx]

            # ROI
            if 'roi' in spatial_modes and roi_map:
                for roi_name, ch_indices in roi_map.items():
                    if ch_indices:
                        roi_val = np.nanmean(loo_itpc_seg[:, ch_indices], axis=1)
                        col = NamingSchema.build("itpc", seg_name, band, "roi", "val", channel=roi_name)
                        results[col] = roi_val

            # Global
            if 'global' in spatial_modes:
                global_val = np.nanmean(loo_itpc_seg, axis=1)
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
    if n_surrogates > 0:
        raise ValueError(
            "PAC (precomputed): surrogate correction is not implemented. "
            "Set feature_engineering.pac.n_surrogates=0."
        )
    requested_pairs = pac_cfg.get("pairs", [("theta", "gamma"), ("alpha", "gamma")])
    
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
        if p_band not in phases or a_band not in amplitudes:
            continue
            
        phi = phases[p_band][..., mask]
        amp = amplitudes[a_band][..., mask]
        
        # PAC(trial, ch) = | mean( amp * exp(i*phi) ) | over time
        mvl_complex = np.mean(amp * np.exp(1j * phi), axis=-1)
        pac_val = np.abs(mvl_complex) # (n_epochs, n_ch)
        
        pair_label = f"{p_band}_{a_band}"
        
        # Channels
        if 'channels' in spatial_modes:
            for i, ch in enumerate(ch_names):
                col = NamingSchema.build("pac", segment_name, pair_label, "ch", "val", channel=ch)
                results[col] = pac_val[:, i]
                
        # ROI
        if 'roi' in spatial_modes and roi_map:
            for roi_name, idxs in roi_map.items():
                if idxs:
                    roi_val = np.nanmean(pac_val[:, idxs], axis=1)
                    col = NamingSchema.build("pac", segment_name, pair_label, "roi", "val", channel=roi_name)
                    results[col] = roi_val
                    
        # Global
        if 'global' in spatial_modes:
            global_val = np.nanmean(pac_val, axis=1)
            col = NamingSchema.build("pac", segment_name, pair_label, "global", "val")
            results[col] = global_val
            
    if not results:
        return pd.DataFrame(), []
        
    df = pd.DataFrame(results)
    return df, list(df.columns)
