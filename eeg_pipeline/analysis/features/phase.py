"""
Phase Feature Extraction
=========================

Phase-based features for EEG analysis:
- ITPC: Inter-Trial Phase Coherence (phase locking across trials)
- PAC: Phase-Amplitude Coupling (cross-frequency coupling)

ITPC measures stimulus-locked phase consistency, relevant for
evoked responses and attention. PAC measures coupling between
low-frequency phase and high-frequency amplitude.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.analysis.features.core import pick_eeg_channels
from eeg_pipeline.utils.analysis.tfr import (
    time_mask,
    freq_mask,
    compute_itpc_map,
    compute_itpc_band_time,
    compute_adaptive_n_cycles,
    get_tfr_config,
    build_rois_from_info,
    resolve_tfr_workers,
)
from eeg_pipeline.utils.analysis.stats import fdr_bh
from eeg_pipeline.utils.config.loader import (
    get_config_value,
    get_frequency_bands,
    parse_temporal_bin_config,
)


# =============================================================================
# ITPC Features
# =============================================================================


def extract_itpc_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    tfr_complex: Optional[Any] = None,
    freqs_override: Optional[np.ndarray] = None,
    times_override: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Compute inter-trial phase coherence (ITPC) features per band and temporal bin.
    Returns a long-form DataFrame (channel, band, time_bin, itpc) plus the full ITPC map.
    """
    if epochs is None or not bands:
        return pd.DataFrame(), [], None, None, None, None

    if tfr_complex is None:
        freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
        workers_default = int(config.get("time_frequency_analysis.tfr.workers", -1))
        workers = resolve_tfr_workers(workers_default=workers_default)

        logger.info("Computing complex TFR for ITPC...")
        try:
            tfr_complex = epochs.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                decim=tfr_decim,
                picks=tfr_picks,
                use_fft=True,
                return_itc=False,
                average=False,
                output="complex",
                n_jobs=workers,
            )
        except Exception as exc:
            logger.error("Failed to compute complex TFR for ITPC: %s", exc)
            tfr_complex = None

        if tfr_complex is None:
            logger.error("Failed to compute complex TFR for ITPC; skipping ITPC features")
            return pd.DataFrame(columns=["channel", "band", "time_bin", "itpc"]), [], None, None, None, None
        freqs_use = freqs
    else:
        freqs = np.asarray(tfr_complex.freqs) if hasattr(tfr_complex, "freqs") else np.asarray(freqs_override)
        freqs_use = freqs

    itpc_map = compute_itpc_map(tfr_complex.data, logger=logger)
    times = np.asarray(times_override if times_override is not None else tfr_complex.times)
    ch_names = list(tfr_complex.info["ch_names"])

    frequency_bands = get_frequency_bands(config)
    temporal_bins = get_config_value(config, "feature_engineering.features.temporal_bins", [])

    rows: List[Dict[str, Any]] = []
    column_names = ["channel", "band", "time_bin", "itpc"]

    for band in bands:
        if band not in frequency_bands:
            logger.warning(f"Band '{band}' not defined in config; skipping ITPC")
            continue

        fmin, fmax = frequency_bands[band]
        for bin_config in temporal_bins:
            bin_params = parse_temporal_bin_config(bin_config)
            if bin_params is None:
                logger.warning(f"Invalid temporal bin configuration: {bin_config}; skipping ITPC bin")
                continue

            time_start, time_end, time_label = bin_params
            band_vals = compute_itpc_band_time(itpc_map, freqs_use, times, fmin, fmax, time_start, time_end)
            if band_vals is None:
                logger.debug(
                    "No data in ITPC bin '%s' (%s-%ss) for band %s; skipping",
                    time_label, time_start, time_end, band
                )
                continue

            for ch, val in zip(ch_names, band_vals):
                rows.append(
                    {
                        "channel": ch,
                        "band": band,
                        "time_bin": time_label,
                        "itpc": float(val),
                    }
                )

    itpc_df = pd.DataFrame(rows, columns=column_names) if rows else pd.DataFrame(columns=column_names)
    return itpc_df, column_names, itpc_map, freqs, times, ch_names


def extract_trialwise_itpc_features(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    tfr_complex: Optional[Any] = None,
    freqs_override: Optional[np.ndarray] = None,
    times_override: Optional[np.ndarray] = None,
    train_indices: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute trial-level ITPC features by leave-one-out mean vector length.
    Returns wide DataFrame (trials x channels) with columns per band/time bin.
    If train_indices is provided, leave-one-out is computed within the training
    set only, and test trials use the training mean (no leakage from held-out).
    """
    if epochs is None or len(epochs) < 2 or not bands:
        return pd.DataFrame(), []

    # Require minimum 15 training epochs for reliable leave-one-out ITPC.
    # With fewer epochs, leaving one out results in too few remaining trials
    # for stable phase coherence estimation (e.g., with 3 epochs, LOO uses only 2).
    min_epochs_for_loo = 15
    if len(epochs) < min_epochs_for_loo:
        logger.warning(
            f"Need at least {min_epochs_for_loo} epochs for reliable leave-one-out ITPC "
            f"(got {len(epochs)}); skipping trialwise ITPC features"
        )
        return pd.DataFrame(), []

    if tfr_complex is None:
        freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
        n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
        workers_default = int(config.get("time_frequency_analysis.tfr.workers", -1))
        workers = resolve_tfr_workers(workers_default=workers_default)
        try:
            tfr_complex = epochs.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                decim=tfr_decim,
                picks=tfr_picks,
                use_fft=True,
                return_itc=False,
                average=False,
                output="complex",
                n_jobs=workers,
            )
        except Exception as exc:
            logger.error("Failed to compute complex TFR for trialwise ITPC: %s", exc)
            tfr_complex = None
        if tfr_complex is None:
            logger.error("Failed to compute complex TFR for trialwise ITPC; skipping ITPC features")
            return pd.DataFrame(), []
        freqs_use = freqs
        times = np.asarray(tfr_complex.times)
    else:
        freqs_use = np.asarray(tfr_complex.freqs) if hasattr(tfr_complex, "freqs") else np.asarray(freqs_override)
        times = np.asarray(times_override if times_override is not None else tfr_complex.times)

    data = np.asarray(tfr_complex.data)
    if data.ndim != 4:
        logger.error(f"Unexpected TFR shape for ITPC: {data.shape}")
        return pd.DataFrame(), []

    n_epochs = data.shape[0]
    if n_epochs < 2:
        logger.warning("Need at least 2 epochs to compute trialwise ITPC; skipping")
        return pd.DataFrame(), []

    # Unit phasors for each trial
    eps = float(config.get("feature_engineering.constants.itpc_epsilon_amp", 1e-12))
    unit = data / (np.abs(data) + eps)

    if train_indices is not None:
        train_mask = np.zeros(n_epochs, dtype=bool)
        try:
            train_mask[np.asarray(train_indices, dtype=int)] = True
        except (IndexError, ValueError, TypeError):
            logger.warning("Invalid train_indices for ITPC trialwise features; falling back to all epochs.")
            train_mask[:] = True
    else:
        train_mask = np.ones(n_epochs, dtype=bool)

    n_train = int(train_mask.sum())
    if n_train < 1:
        logger.warning("No training epochs available for ITPC trialwise features; skipping")
        return pd.DataFrame(), []

    sum_train = np.sum(unit[train_mask], axis=0)  # (ch, freq, time)

    frequency_bands = get_frequency_bands(config)
    temporal_bins = get_config_value(config, "feature_engineering.features.temporal_bins", [])
    ch_names = list(tfr_complex.info["ch_names"])

    feature_blocks: List[pd.DataFrame] = []
    column_names: List[str] = []

    for band in bands:
        if band not in frequency_bands:
            logger.warning("Band '%s' not defined in config; skipping trialwise ITPC", band)
            continue
        fmin, fmax = frequency_bands[band]
        f_mask = freq_mask(freqs_use, fmin, fmax)
        if not np.any(f_mask):
            logger.warning("No frequencies in band %s for ITPC trialwise extraction", band)
            continue

        for bin_config in temporal_bins:
            bin_params = parse_temporal_bin_config(bin_config)
            if bin_params is None:
                logger.warning("Invalid temporal bin configuration: %s; skipping ITPC trial bin", bin_config)
                continue
            tmin, tmax, tlabel = bin_params
            t_mask = time_mask(times, tmin, tmax)
            if not np.any(t_mask):
                logger.debug("No samples in ITPC bin %s (%s-%s); skipping", tlabel, tmin, tmax)
                continue

            # Memory-friendly computation: avoid full mean_without_trial array
            mean_train = sum_train / float(max(1, n_train))
            band_bin_vals = np.zeros((n_epochs, len(ch_names)), dtype=float)

            for ep_idx in range(n_epochs):
                if train_mask[ep_idx] and n_train > 1:
                    mean_ep = (sum_train - unit[ep_idx]) / float(n_train - 1)
                elif train_mask[ep_idx] and n_train == 1:
                    mean_ep = mean_train
                else:
                    # test epochs: use training mean to avoid leakage
                    mean_ep = mean_train

                mean_ep_band = mean_ep[:, f_mask][:, :, t_mask]
                band_bin_vals[ep_idx] = np.abs(mean_ep_band).mean(axis=(1, 2))

            cols = [f"itpc_{band}_{ch}_{tlabel}" for ch in ch_names]
            feature_blocks.append(pd.DataFrame(band_bin_vals, columns=cols))
            column_names.extend(cols)

    if not feature_blocks:
        return pd.DataFrame(), []

    df_itpc = pd.concat(feature_blocks, axis=1)
    return df_itpc, column_names


def compute_pac_comodulograms(
    tfr_complex,
    freqs: np.ndarray,
    times: np.ndarray,
    info: mne.Info,
    config: Any,
    logger: Any,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Compute ROI-level PAC comodulograms using amplitude-weighted phase locking.
    Returns (pac_df_long, phase_freqs, amp_freqs, pac_trial_df, pac_time_df).
    
    Requires minimum 20 epochs for reliable PAC estimation with surrogate testing.
    """
    if tfr_complex is None:
        return pd.DataFrame(), np.array([]), np.array([]), pd.DataFrame(), pd.DataFrame()
    
    # Validate minimum epochs for reliable PAC estimation
    # PAC with surrogate testing requires sufficient trials to avoid spurious coupling
    min_epochs_pac = int(config.get("feature_engineering.pac.min_epochs", 20))
    n_epochs = tfr_complex.data.shape[0] if hasattr(tfr_complex, 'data') else 0
    if n_epochs < min_epochs_pac:
        logger.warning(
            f"Insufficient epochs for reliable PAC estimation: {n_epochs} < {min_epochs_pac} minimum. "
            f"PAC results with fewer epochs are unreliable due to insufficient surrogate distribution sampling. "
            f"Skipping PAC computation."
        )
        return pd.DataFrame(), np.array([]), np.array([]), pd.DataFrame(), pd.DataFrame()

    pac_cfg = config.get("feature_engineering.pac", {})
    phase_min, phase_max = pac_cfg.get("phase_range", [4.0, 8.0])
    amp_min, amp_max = pac_cfg.get("amp_range", [30.0, 80.0])
    phase_step = float(pac_cfg.get("phase_step_hz", 1.0))
    amp_step = float(pac_cfg.get("amp_step_hz", 5.0))
    plateau_window = pac_cfg.get("plateau_window") or config.get("time_frequency_analysis.plateau_window", [3.0, 10.5])
    n_surrogates = int(pac_cfg.get("n_surrogates", 200))
    rng = np.random.default_rng(int(config.get("project.random_state", 42)))

    phase_freqs = np.arange(phase_min, phase_max + 1e-6, phase_step)
    amp_freqs = np.arange(amp_min, amp_max + 1e-6, amp_step)
    freq_arr = np.asarray(freqs)

    plateau_mask = time_mask(times, plateau_window[0], plateau_window[1])
    if not np.any(plateau_mask):
        logger.warning("No samples in plateau window for PAC; skipping")
        return pd.DataFrame(), phase_freqs, amp_freqs, pd.DataFrame(), pd.DataFrame()

    if info is None:
        logger.warning("No info available for PAC computation; skipping")
        return pd.DataFrame(), phase_freqs, amp_freqs, pd.DataFrame(), pd.DataFrame()

    roi_map = build_rois_from_info(info, config=config)
    if not roi_map:
        logger.warning("No ROIs found for PAC computation; skipping")
        return pd.DataFrame(), phase_freqs, amp_freqs, pd.DataFrame(), pd.DataFrame()

    data = tfr_complex.data  # (epochs, channels, freqs, times)
    amp = np.abs(data)
    phase = np.angle(data)
    eps_amp = float(config.get("feature_engineering.constants.pac_epsilon_amp", 1e-9))
    n_jobs_pac = int(config.get("feature_engineering.parallel.n_jobs_pac", 1))

    tfr_ch_names = tfr_complex.info.ch_names
    rows = []
    trial_rows = []
    time_rows = []
    freq_arr = np.asarray(freqs)

    n_rois = len(roi_map)
    n_phase_freqs = len(phase_freqs)
    n_amp_freqs = len(amp_freqs)
    total_combinations = n_rois * n_phase_freqs * n_amp_freqs

    max_surrogate_iters = int(pac_cfg.get("max_surrogate_iterations", 200000))
    n_surrogates = max(0, n_surrogates)
    effective_surrogates = n_surrogates
    if max_surrogate_iters > 0 and total_combinations > 0:
        est_iters = total_combinations * max(1, n_surrogates)
        if est_iters > max_surrogate_iters:
            effective_surrogates = max(1, max_surrogate_iters // total_combinations)
            logger.warning(
                "Reducing PAC surrogates from %d to %d to cap total iterations (%d) "
                "across %d ROI/phase/amp combinations.",
                n_surrogates,
                effective_surrogates,
                max_surrogate_iters,
                total_combinations,
            )

    surrogate_perms = [rng.permutation(n_epochs) for _ in range(effective_surrogates)] if effective_surrogates > 0 else []

    for roi_idx, (roi, chs) in enumerate(roi_map.items()):
        chs_available = [ch for ch in chs if ch in tfr_ch_names]
        if len(chs_available) == 0:
            continue
        picks = mne.pick_channels(tfr_ch_names, include=chs_available, ordered=True)
        if len(picks) == 0:
            continue

        amp_norm_by_ch_freq = {}
        for f_amp in amp_freqs:
            amp_mask = freq_mask(freq_arr, f_amp * 0.9, f_amp * 1.1)
            if not np.any(amp_mask):
                continue
            for ch in picks:
                amp_vals = amp[:, ch, amp_mask, :]
                amp_plateau = amp_vals[:, :, plateau_mask]
                amp_mean = np.nanmean(amp_plateau, axis=(1, 2))[:, np.newaxis, np.newaxis]
                amp_norm = amp_plateau / (amp_mean + eps_amp)
                if ch not in amp_norm_by_ch_freq:
                    amp_norm_by_ch_freq[ch] = {}
                amp_norm_by_ch_freq[ch][f_amp] = amp_norm

        # Cache phase data per frequency to avoid recomputation inside workers
        phase_cache: Dict[float, Dict[int, np.ndarray]] = {}
        for f_phase in phase_freqs:
            phase_mask = freq_mask(freq_arr, f_phase * 0.9, f_phase * 1.1)
            if not np.any(phase_mask):
                continue
            phase_plateau_by_ch = {}
            for ch in picks:
                phase_vals = phase[:, ch, phase_mask, :]
                phase_mean = np.exp(1j * phase_vals).mean(axis=1)
                phase_plateau_by_ch[ch] = phase_mean[:, plateau_mask]
            phase_cache[f_phase] = phase_plateau_by_ch

        combos: List[Tuple[float, float]] = []
        for f_phase, phase_plateau_by_ch in phase_cache.items():
            for f_amp in amp_freqs:
                if f_amp <= f_phase * 1.5:
                    continue
                if not all(f_amp in amp_norm_by_ch_freq.get(ch, {}) for ch in picks):
                    continue
                combos.append((f_phase, f_amp))

        def _compute_pac_combo(f_phase: float, f_amp: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
            phase_plateau_by_ch = phase_cache[f_phase]

            pac_complex_by_ch: List[complex] = []
            pac_trial_by_ch: List[np.ndarray] = []

            for ch in picks:
                phase_plateau = phase_plateau_by_ch[ch]
                amp_norm = amp_norm_by_ch_freq[ch][f_amp]
                z = amp_norm * np.exp(1j * phase_plateau[:, np.newaxis, :])

                pac_complex = np.nanmean(z)
                pac_complex_by_ch.append(pac_complex)
                pac_trial_by_ch.append(np.nanmean(z, axis=(1, 2)))

            if not pac_complex_by_ch:
                return [], [], []

            pac_roi_complex = np.nanmean(np.array(pac_complex_by_ch))
            pac_val = np.abs(pac_roi_complex)

            p_perm = np.nan
            if effective_surrogates > 0:
                phase_plateau_all = np.stack([phase_plateau_by_ch[ch] for ch in picks], axis=0)
                amp_norm_all = np.stack([amp_norm_by_ch_freq[ch][f_amp] for ch in picks], axis=0)

                null = np.zeros(effective_surrogates, dtype=float)

                for s_idx, trial_perm in enumerate(surrogate_perms):
                    amp_shuffled = amp_norm_all[:, trial_perm, :, :]
                    z_null = amp_shuffled * np.exp(1j * phase_plateau_all[:, :, np.newaxis, :])
                    pac_null_ch = np.nanmean(z_null, axis=(1, 2, 3))
                    pac_null = np.abs(np.nanmean(pac_null_ch))
                    null[s_idx] = pac_null

                p_perm = (1 + np.sum(null >= pac_val)) / (effective_surrogates + 1)

            rows_local = [
                {
                    "roi": roi,
                    "phase_freq": float(f_phase),
                    "amp_freq": float(f_amp),
                    "pac": float(pac_val),
                    "p_perm": float(p_perm) if np.isfinite(p_perm) else np.nan,
                    "n_surrogates": effective_surrogates,
                }
            ]

            trial_rows_local: List[Dict[str, Any]] = []
            pac_trials_complex = np.nanmedian(np.stack(pac_trial_by_ch, axis=0), axis=0)
            for trial_idx, val in enumerate(pac_trials_complex):
                trial_rows_local.append(
                    {
                        "trial": trial_idx,
                        "roi": roi,
                        "phase_freq": float(f_phase),
                        "amp_freq": float(f_amp),
                        "pac": float(np.abs(val)),
                    }
                )

            time_rows_local: List[Dict[str, Any]] = []
            z_time = []
            for ch in picks:
                phase_plateau = phase_plateau_by_ch[ch]
                amp_norm = amp_norm_by_ch_freq[ch][f_amp]
                z_time.append(amp_norm * np.exp(1j * phase_plateau[:, np.newaxis, :]))
            z_time = np.nanmedian(np.stack(z_time, axis=0), axis=0)
            pac_time = np.abs(np.nanmean(z_time, axis=0))
            for t_idx, t_val in enumerate(times[plateau_mask]):
                time_rows_local.append(
                    {
                        "roi": roi,
                        "phase_freq": float(f_phase),
                        "amp_freq": float(f_amp),
                        "time": float(t_val),
                        "pac": float(np.nanmean(pac_time[:, t_idx])),
                    }
                )

            return rows_local, trial_rows_local, time_rows_local

        if combos:
            if n_jobs_pac > 1:
                try:
                    from joblib import Parallel, delayed
                    combo_results = Parallel(n_jobs=n_jobs_pac, prefer="processes")(
                        delayed(_compute_pac_combo)(f_phase, f_amp) for f_phase, f_amp in combos
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("PAC parallel computation failed (%s); falling back to sequential.", exc)
                    combo_results = [_compute_pac_combo(f_phase, f_amp) for f_phase, f_amp in combos]
            else:
                combo_results = [_compute_pac_combo(f_phase, f_amp) for f_phase, f_amp in combos]

            for r_list, tr_list, ti_list in combo_results:
                rows.extend(r_list)
                trial_rows.extend(tr_list)
                time_rows.extend(ti_list)

    pac_df = pd.DataFrame(rows, columns=["roi", "phase_freq", "amp_freq", "pac", "p_perm", "n_surrogates"]) if rows else pd.DataFrame(columns=["roi", "phase_freq", "amp_freq", "pac", "p_perm", "n_surrogates"])
    pac_trials_df = pd.DataFrame(trial_rows, columns=["trial", "roi", "phase_freq", "amp_freq", "pac"]) if trial_rows else pd.DataFrame(columns=["trial", "roi", "phase_freq", "amp_freq", "pac"])
    pac_time_df = pd.DataFrame(time_rows, columns=["roi", "phase_freq", "amp_freq", "time", "pac"]) if time_rows else pd.DataFrame(columns=["roi", "phase_freq", "amp_freq", "time", "pac"])

    if n_surrogates > 0 and not pac_df.empty and "p_perm" in pac_df.columns:
        pac_df["q_perm"] = fdr_bh(pac_df["p_perm"].to_numpy(dtype=float), config=config)

    return pac_df, phase_freqs, amp_freqs, pac_trials_df, pac_time_df


# =============================================================================
# Amplitude-Amplitude Coupling (AAC)
# =============================================================================


def extract_aac_features(
    epochs: mne.Epochs,
    band_pairs: List[Tuple[str, str]],
    config: Any,
    logger: Any,
    *,
    active_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Amplitude-Amplitude Coupling (AAC) features.
    
    AAC measures the correlation between amplitude envelopes of two 
    frequency bands. High AAC indicates co-modulation of power across bands.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    band_pairs : List[Tuple[str, str]]
        List of (low_band, high_band) pairs to compute AAC for.
        E.g., [("theta", "gamma"), ("alpha", "gamma")]
    config : Any
        Configuration object
    logger : Any
        Logger instance
    active_window : Optional[Tuple[float, float]]
        Time window for AAC computation. Default: plateau window.
    
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with AAC features and column names
    """
    if not band_pairs:
        return pd.DataFrame(), []
    
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for AAC extraction")
        return pd.DataFrame(), []
    
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    from scipy.signal import hilbert
    
    freq_bands = get_frequency_bands(config)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    
    if active_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        active_window = tuple(tf_cfg.get("plateau_window", [3.0, 10.5]))
    
    active_mask = time_mask(times, active_window[0], active_window[1])
    if not np.any(active_mask):
        logger.warning("AAC: no samples in active window")
        return pd.DataFrame(), []
    
    data = epochs.get_data(picks=picks)
    n_epochs = data.shape[0]
    
    feature_records: List[Dict[str, float]] = []
    
    for ep_idx in range(n_epochs):
        epoch = data[ep_idx]
        record: Dict[str, float] = {}
        
        for low_band, high_band in band_pairs:
            if low_band not in freq_bands or high_band not in freq_bands:
                logger.debug(f"AAC band pair ({low_band}, {high_band}) not in config")
                continue
            
            fmin_low, fmax_low = freq_bands[low_band]
            fmin_high, fmax_high = freq_bands[high_band]
            
            try:
                # Filter and compute envelopes
                filtered_low = mne.filter.filter_data(
                    epoch, sfreq, l_freq=fmin_low, h_freq=fmax_low,
                    n_jobs=1, verbose=False
                )
                filtered_high = mne.filter.filter_data(
                    epoch, sfreq, l_freq=fmin_high, h_freq=fmax_high,
                    n_jobs=1, verbose=False
                )
                
                env_low = np.abs(hilbert(filtered_low, axis=-1))
                env_high = np.abs(hilbert(filtered_high, axis=-1))
                
            except (ValueError, RuntimeError) as e:
                logger.debug(f"AAC filtering failed for {low_band}-{high_band}: {e}")
                for ch_name in ch_names:
                    record[f"aac_{low_band}_{high_band}_{ch_name}"] = np.nan
                continue
            
            aac_values = []
            for ch_idx, ch_name in enumerate(ch_names):
                env_low_active = env_low[ch_idx, active_mask]
                env_high_active = env_high[ch_idx, active_mask]
                
                if len(env_low_active) < 2:
                    record[f"aac_{low_band}_{high_band}_{ch_name}"] = np.nan
                    continue
                
                # Pearson correlation of envelopes
                corr = np.corrcoef(env_low_active, env_high_active)[0, 1]
                record[f"aac_{low_band}_{high_band}_{ch_name}"] = float(corr) if np.isfinite(corr) else np.nan
                
                if np.isfinite(corr):
                    aac_values.append(corr)
            
            # Global AAC (mean across channels)
            record[f"aac_{low_band}_{high_band}_global"] = float(np.mean(aac_values)) if aac_values else np.nan
        
        feature_records.append(record)
    
    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


# =============================================================================
# Phase-Phase Coupling (PPC) / n:m Phase Locking
# =============================================================================


def extract_ppc_features(
    epochs: mne.Epochs,
    band_pairs: List[Tuple[str, str, int, int]],
    config: Any,
    logger: Any,
    *,
    active_window: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract Phase-Phase Coupling (n:m phase locking) features.
    
    PPC measures the consistency of phase relationships between two bands
    at different frequency ratios (e.g., 1:2 theta-alpha locking).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    band_pairs : List[Tuple[str, str, int, int]]
        List of (low_band, high_band, n, m) tuples.
        n:m indicates the phase ratio (e.g., 1:2 means high_phase = 2*low_phase).
        E.g., [("theta", "alpha", 1, 2), ("delta", "theta", 1, 2)]
    config : Any
        Configuration object
    logger : Any
        Logger instance
    active_window : Optional[Tuple[float, float]]
        Time window for PPC computation. Default: plateau window.
    
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with PPC features and column names
    """
    if not band_pairs:
        return pd.DataFrame(), []
    
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("No EEG channels available for PPC extraction")
        return pd.DataFrame(), []
    
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    from scipy.signal import hilbert
    
    freq_bands = get_frequency_bands(config)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    
    if active_window is None:
        tf_cfg = config.get("time_frequency_analysis", {})
        active_window = tuple(tf_cfg.get("plateau_window", [3.0, 10.5]))
    
    active_mask = time_mask(times, active_window[0], active_window[1])
    if not np.any(active_mask):
        logger.warning("PPC: no samples in active window")
        return pd.DataFrame(), []
    
    data = epochs.get_data(picks=picks)
    n_epochs = data.shape[0]
    
    feature_records: List[Dict[str, float]] = []
    
    for ep_idx in range(n_epochs):
        epoch = data[ep_idx]
        record: Dict[str, float] = {}
        
        for low_band, high_band, n, m in band_pairs:
            if low_band not in freq_bands or high_band not in freq_bands:
                logger.debug(f"PPC band pair ({low_band}, {high_band}) not in config")
                continue
            
            fmin_low, fmax_low = freq_bands[low_band]
            fmin_high, fmax_high = freq_bands[high_band]
            
            try:
                # Filter and compute phases
                filtered_low = mne.filter.filter_data(
                    epoch, sfreq, l_freq=fmin_low, h_freq=fmax_low,
                    n_jobs=1, verbose=False
                )
                filtered_high = mne.filter.filter_data(
                    epoch, sfreq, l_freq=fmin_high, h_freq=fmax_high,
                    n_jobs=1, verbose=False
                )
                
                phase_low = np.angle(hilbert(filtered_low, axis=-1))
                phase_high = np.angle(hilbert(filtered_high, axis=-1))
                
            except (ValueError, RuntimeError) as e:
                logger.debug(f"PPC filtering failed for {low_band}-{high_band}: {e}")
                for ch_name in ch_names:
                    record[f"ppc_{n}_{m}_{low_band}_{high_band}_{ch_name}"] = np.nan
                continue
            
            ppc_values = []
            for ch_idx, ch_name in enumerate(ch_names):
                phase_low_active = phase_low[ch_idx, active_mask]
                phase_high_active = phase_high[ch_idx, active_mask]
                
                if len(phase_low_active) < 2:
                    record[f"ppc_{n}_{m}_{low_band}_{high_band}_{ch_name}"] = np.nan
                    continue
                
                # n:m phase locking value
                # Compute: |mean(exp(i * (n*phase_low - m*phase_high)))|
                phase_diff = n * phase_low_active - m * phase_high_active
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                
                record[f"ppc_{n}_{m}_{low_band}_{high_band}_{ch_name}"] = float(plv)
                ppc_values.append(plv)
            
            # Global PPC (mean across channels)
            record[f"ppc_{n}_{m}_{low_band}_{high_band}_global"] = float(np.mean(ppc_values)) if ppc_values else np.nan
        
        feature_records.append(record)
    
    column_names = list(feature_records[0].keys()) if feature_records else []
    return pd.DataFrame(feature_records), column_names


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
    Extract all cross-frequency coupling features (PAC, AAC, PPC).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to extract features from
    config : Any
        Configuration object
    logger : Any
        Logger instance
    include_pac : bool
        Include phase-amplitude coupling (requires TFR computation)
    include_aac : bool
        Include amplitude-amplitude coupling
    include_ppc : bool
        Include phase-phase coupling
    
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with CFC features and column names
    """
    all_dfs: List[pd.DataFrame] = []
    all_cols: List[str] = []
    
    cfc_cfg = config.get("feature_engineering.cfc", {})
    
    # AAC
    if include_aac:
        aac_pairs = cfc_cfg.get("aac_pairs", [
            ("theta", "gamma"),
            ("alpha", "gamma"),
            ("alpha", "beta"),
        ])
        # Convert to list of tuples if needed
        aac_pairs = [(p[0], p[1]) for p in aac_pairs]
        
        aac_df, aac_cols = extract_aac_features(epochs, aac_pairs, config, logger)
        if not aac_df.empty:
            all_dfs.append(aac_df)
            all_cols.extend(aac_cols)
    
    # PPC
    if include_ppc:
        ppc_pairs = cfc_cfg.get("ppc_pairs", [
            ("delta", "theta", 1, 2),
            ("theta", "alpha", 1, 2),
            ("alpha", "beta", 1, 2),
        ])
        # Convert to list of tuples if needed
        ppc_pairs = [(p[0], p[1], p[2], p[3]) for p in ppc_pairs]
        
        ppc_df, ppc_cols = extract_ppc_features(epochs, ppc_pairs, config, logger)
        if not ppc_df.empty:
            all_dfs.append(ppc_df)
            all_cols.extend(ppc_cols)
    
    if not all_dfs:
        return pd.DataFrame(), []
    
    combined = pd.concat(all_dfs, axis=1)
    return combined, all_cols
