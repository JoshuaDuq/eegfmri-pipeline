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
        workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
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
        workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1))
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
    min_epochs_pac = int(config.get("pac_analysis.min_epochs", 20))
    n_epochs = tfr_complex.data.shape[0] if hasattr(tfr_complex, 'data') else 0
    if n_epochs < min_epochs_pac:
        logger.warning(
            f"Insufficient epochs for reliable PAC estimation: {n_epochs} < {min_epochs_pac} minimum. "
            f"PAC results with fewer epochs are unreliable due to insufficient surrogate distribution sampling. "
            f"Skipping PAC computation."
        )
        return pd.DataFrame(), np.array([]), np.array([]), pd.DataFrame(), pd.DataFrame()

    pac_cfg = config.get("pac_analysis", {})
    phase_min, phase_max = pac_cfg.get("phase_range", [4.0, 8.0])
    amp_min, amp_max = pac_cfg.get("amp_range", [30.0, 80.0])
    phase_step = float(pac_cfg.get("phase_step_hz", 1.0))
    amp_step = float(pac_cfg.get("amp_step_hz", 5.0))
    plateau_window = pac_cfg.get("plateau_window") or config.get("time_frequency_analysis", {}).get("plateau_window", [3.0, 10.5])
    n_surrogates = int(pac_cfg.get("n_surrogates", 200))
    rng = np.random.default_rng(int(config.get("random.seed", 42)))

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

    tfr_ch_names = tfr_complex.info.ch_names
    rows = []
    trial_rows = []
    time_rows = []
    freq_arr = np.asarray(freqs)

    n_rois = len(roi_map)
    n_phase_freqs = len(phase_freqs)
    n_amp_freqs = len(amp_freqs)
    total_combinations = n_rois * n_phase_freqs * n_amp_freqs
    processed = 0

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

        for f_phase_idx, f_phase in enumerate(phase_freqs):
            phase_mask = freq_mask(freq_arr, f_phase * 0.9, f_phase * 1.1)
            if not np.any(phase_mask):
                continue

            phase_plateau_by_ch = {}
            for ch in picks:
                phase_vals = phase[:, ch, phase_mask, :]
                phase_mean = np.exp(1j * phase_vals).mean(axis=1)
                phase_plateau_by_ch[ch] = phase_mean[:, plateau_mask]

            for f_amp_idx, f_amp in enumerate(amp_freqs):
                if f_amp <= f_phase * 1.5:
                    continue
                # Check if f_amp exists for all channels in picks
                if not all(f_amp in amp_norm_by_ch_freq.get(ch, {}) for ch in picks):
                    continue

                processed += 1
                progress_pct = (processed / total_combinations) * 100
                if processed == 1 or progress_pct >= 25 and (processed - 1) / total_combinations * 100 < 25:
                    logger.info(f"PAC progress: {progress_pct:.0f}% ({processed}/{total_combinations})")
                elif progress_pct >= 50 and (processed - 1) / total_combinations * 100 < 50:
                    logger.info(f"PAC progress: {progress_pct:.0f}% ({processed}/{total_combinations})")
                elif progress_pct >= 75 and (processed - 1) / total_combinations * 100 < 75:
                    logger.info(f"PAC progress: {progress_pct:.0f}% ({processed}/{total_combinations})")
                elif processed == total_combinations:
                    logger.info(f"PAC progress: 100% ({processed}/{total_combinations})")

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
                    continue

                pac_roi_complex = np.nanmean(np.array(pac_complex_by_ch))
                pac_val = np.abs(pac_roi_complex)

                p_perm = np.nan
                if effective_surrogates > 0:
                    phase_plateau_all = np.stack([phase_plateau_by_ch[ch] for ch in picks], axis=0)
                    amp_norm_all = np.stack([amp_norm_by_ch_freq[ch][f_amp] for ch in picks], axis=0)
                    n_channels, n_epochs, n_time = phase_plateau_all.shape

                    null = np.zeros(effective_surrogates, dtype=float)

                    for s_idx, trial_perm in enumerate(surrogate_perms):
                        # Trial-shuffle surrogate: shuffle trial order for amplitude
                        # while keeping phase fixed. This breaks PAC while preserving
                        # within-trial temporal structure (more valid than time-shift).
                        amp_shuffled = amp_norm_all[:, trial_perm, :, :]
                        z_null = amp_shuffled * np.exp(1j * phase_plateau_all[:, :, np.newaxis, :])
                        pac_null_ch = np.nanmean(z_null, axis=(1, 2, 3))
                        pac_null = np.abs(np.nanmean(pac_null_ch))
                        null[s_idx] = pac_null

                    # Use >= for consistent permutation p-value calculation
                    # (includes observed value in null distribution per Phipson & Smyth 2010)
                    p_perm = (1 + np.sum(null >= pac_val)) / (effective_surrogates + 1)

                rows.append(
                    {
                        "roi": roi,
                        "phase_freq": float(f_phase),
                        "amp_freq": float(f_amp),
                        "pac": float(pac_val),
                        "p_perm": float(p_perm) if np.isfinite(p_perm) else np.nan,
                        "n_surrogates": effective_surrogates,
                    }
                )

                pac_trials_complex = np.nanmedian(np.stack(pac_trial_by_ch, axis=0), axis=0)
                for trial_idx, val in enumerate(pac_trials_complex):
                    trial_rows.append(
                        {
                            "trial": trial_idx,
                            "roi": roi,
                            "phase_freq": float(f_phase),
                            "amp_freq": float(f_amp),
                            "pac": float(np.abs(val)),
                        }
                    )

                z_time = []
                for ch in picks:
                    phase_plateau = phase_plateau_by_ch[ch]
                    amp_norm = amp_norm_by_ch_freq[ch][f_amp]
                    z_time.append(amp_norm * np.exp(1j * phase_plateau[:, np.newaxis, :]))
                z_time = np.nanmedian(np.stack(z_time, axis=0), axis=0)
                pac_time = np.abs(np.nanmean(z_time, axis=0))
                for t_idx, t_val in enumerate(times[plateau_mask]):
                    time_rows.append(
                        {
                            "roi": roi,
                            "phase_freq": float(f_phase),
                            "amp_freq": float(f_amp),
                            "time": float(t_val),
                            "pac": float(np.nanmean(pac_time[:, t_idx])),
                        }
                    )

    pac_df = pd.DataFrame(rows, columns=["roi", "phase_freq", "amp_freq", "pac", "p_perm", "n_surrogates"]) if rows else pd.DataFrame(columns=["roi", "phase_freq", "amp_freq", "pac", "p_perm", "n_surrogates"])
    pac_trials_df = pd.DataFrame(trial_rows, columns=["trial", "roi", "phase_freq", "amp_freq", "pac"]) if trial_rows else pd.DataFrame(columns=["trial", "roi", "phase_freq", "amp_freq", "pac"])
    pac_time_df = pd.DataFrame(time_rows, columns=["roi", "phase_freq", "amp_freq", "time", "pac"]) if time_rows else pd.DataFrame(columns=["roi", "phase_freq", "amp_freq", "time", "pac"])

    if n_surrogates > 0 and not pac_df.empty and "p_perm" in pac_df.columns:
        pac_df["q_perm"] = fdr_bh(pac_df["p_perm"].to_numpy(dtype=float), config=config)

    return pac_df, phase_freqs, amp_freqs, pac_trials_df, pac_time_df

