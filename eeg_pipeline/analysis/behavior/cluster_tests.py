import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import mne
from scipy import sparse
from mne.stats import permutation_cluster_test, combine_adjacency

from eeg_pipeline.utils.data.loading import (
    load_epochs_for_analysis,
)
from eeg_pipeline.utils.io.general import (
    deriv_stats_path,
    ensure_dir,
    get_pain_column_from_config,
    write_tsv,
)
from eeg_pipeline.utils.analysis.stats import (
    get_eeg_adjacency,
    validate_pain_binary_values,
    fdr_bh_values,
    _resolve_cluster_n_jobs,
)
from eeg_pipeline.utils.analysis.tfr import (
    compute_tfr_morlet,
    get_bands_for_tfr,
    get_tfr_config,
)


def compute_pain_nonpain_time_cluster_test(
    subject: str,
    pain_epochs: mne.Epochs,
    nonpain_epochs: mne.Epochs,
    output_dir: Path,
    config,
    bands: Optional[dict] = None,
    n_permutations: int = 1024,
    alpha: float = 0.05,
) -> dict:
    """
    Time-domain cluster permutation test contrasting pain vs. non-pain trials.

    Uses the pipeline's TFR parameters (Morlet; average=False) and applies the
    standard baseline before testing. Tests from time 0 onwards. Results are saved per-band as TSVs.
    """
    from eeg_pipeline.utils.analysis.tfr import apply_baseline_to_tfr
    
    logger = logging.getLogger(__name__)
    ensure_dir(output_dir)

    freq_min, freq_max, _n_freqs, n_cycles_factor, decim, picks = get_tfr_config(config)
    if bands is None:
        bands = get_bands_for_tfr(max_freq_available=freq_max, config=config)

    n_jobs = _resolve_cluster_n_jobs(config=config)
    results: dict = {}

    for band_name, (fmin, fmax) in bands.items():
        if fmin >= fmax:
            logger.warning("Band %s has invalid range [%s, %s]; skipping.", band_name, fmin, fmax)
            continue

        freqs = np.arange(fmin, fmax, 2.0)
        if freqs.size == 0:
            logger.warning("Band %s produced no frequency bins; skipping.", band_name)
            continue

        logger.info(
            "Computing pain vs. non-pain time-cluster test for %s band [%s, %s] (sub-%s)",
            band_name, fmin, fmax, subject,
        )

        tfr_pain = compute_tfr_morlet(
            pain_epochs,
            config,
            logger=logger,
            freqs=freqs,
            picks=picks,
            decim=decim,
        )
        tfr_nonpain = compute_tfr_morlet(
            nonpain_epochs,
            config,
            logger=logger,
            freqs=freqs,
            picks=picks,
            decim=decim,
        )
        apply_baseline_to_tfr(tfr_pain, config, logger)
        apply_baseline_to_tfr(tfr_nonpain, config, logger)

        time_mask = tfr_pain.times >= 0.0
        band_power_pain = tfr_pain.data[:, :, :, time_mask].mean(axis=2)  # (n_epochs, n_channels, n_times)
        band_power_nonpain = tfr_nonpain.data[:, :, :, time_mask].mean(axis=2)
        time_vec = tfr_pain.times[time_mask]

        if band_power_pain.shape[0] < 2 or band_power_nonpain.shape[0] < 2:
            logger.warning(
                "Insufficient epochs for band %s (pain=%d, nonpain=%d); skipping cluster test.",
                band_name, band_power_pain.shape[0], band_power_nonpain.shape[0],
            )
            results[band_name] = {"error": "insufficient_epochs"}
            continue

        n_channels = band_power_pain.shape[1]
        n_times = band_power_pain.shape[2]

        adjacency_eeg, _, _ = get_eeg_adjacency(tfr_pain.info, logger=logger)
        cluster_records = []

        if adjacency_eeg is not None:
            adjacency = combine_adjacency(n_times, adjacency_eeg)
            X_pain = band_power_pain.reshape(band_power_pain.shape[0], -1)
            X_nonpain = band_power_nonpain.reshape(band_power_nonpain.shape[0], -1)

            T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
                [X_pain, X_nonpain],
                n_permutations=int(n_permutations),
                tail=0,
                n_jobs=n_jobs,
                adjacency=adjacency,
                out_type="mask",
            )

            sig_inds = np.where(cluster_p_values < alpha)[0]
            sig_mask = np.zeros((n_channels, n_times), dtype=bool)
            T_obs_grid = T_obs.reshape(n_channels, n_times)

            for idx in sig_inds:
                c_mask_flat = clusters[idx]
                c_mask = c_mask_flat.reshape(n_channels, n_times)
                sig_mask |= c_mask
                t_inds = np.where(c_mask.any(axis=0))[0]
                ch_inds = np.where(c_mask.any(axis=1))[0]
                if t_inds.size == 0 or ch_inds.size == 0:
                    continue
                t_start = float(time_vec[t_inds[0]])
                t_end = float(time_vec[t_inds[-1]])
                cluster_records.append({
                    "subject": f"sub-{subject}",
                    "band": band_name,
                    "cluster_index": int(idx),
                    "p_value": float(cluster_p_values[idx]),
                    "t_start": t_start,
                    "t_end": t_end,
                    "n_timepoints": int(len(t_inds)),
                    "n_channels": int(len(ch_inds)),
                    "channels": ",".join(np.array(tfr_pain.ch_names)[ch_inds]),
                    "t_stat_min": float(np.min(T_obs_grid[c_mask])),
                    "t_stat_max": float(np.max(T_obs_grid[c_mask])),
                })

            results[band_name] = {
                "significant": len(cluster_records) > 0,
                "cluster_records": cluster_records,
                "times": time_vec,
                "time_mask": sig_mask.any(axis=0),
                "time_mask_channels": sig_mask,
            }
        else:
            logger.warning(
                "EEG adjacency unavailable; running time-only cluster tests per-channel with BH across channels."
            )
            time_adjacency = sparse.diags([1, 1, 1], [-1, 0, 1], shape=(n_times, n_times), format="csr")
            channel_cluster_records = []
            channel_pvals = []

            for ch_idx, ch_name in enumerate(tfr_pain.ch_names):
                X_pain_ch = band_power_pain[:, ch_idx, :]
                X_nonpain_ch = band_power_nonpain[:, ch_idx, :]
                if X_pain_ch.shape[0] < 2 or X_nonpain_ch.shape[0] < 2:
                    continue
                T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
                    [X_pain_ch, X_nonpain_ch],
                    n_permutations=int(n_permutations),
                    tail=0,
                    n_jobs=n_jobs,
                    adjacency=time_adjacency,
                    out_type="mask",
                )
                channel_pvals.extend(cluster_p_values.tolist())
                for idx, p_val in enumerate(cluster_p_values):
                    t_inds = np.where(clusters[idx])[0]
                    channel_cluster_records.append({
                        "subject": f"sub-{subject}",
                        "band": band_name,
                        "channel": ch_name,
                        "cluster_index": int(idx),
                        "p_raw": float(p_val),
                        "t_start": float(time_vec[t_inds[0]]),
                        "t_end": float(time_vec[t_inds[-1]]),
                        "n_timepoints": int(len(t_inds)),
                        "t_stat_min": float(np.min(T_obs[t_inds])),
                        "t_stat_max": float(np.max(T_obs[t_inds])),
                    })

            if channel_pvals:
                _, q_vals = fdr_bh_values(np.asarray(channel_pvals), alpha=alpha)
                q_iter = iter(q_vals if q_vals is not None else [])
                for rec in channel_cluster_records:
                    try:
                        q_val = next(q_iter)
                    except StopIteration:
                        q_val = np.nan
                    rec["p_raw"] = float(rec.get("p_raw", np.nan))
                    rec["p_fdr_local"] = float(q_val) if np.isfinite(q_val) else np.nan
                    rec["p_value"] = rec["p_fdr_local"] if np.isfinite(rec["p_fdr_local"]) else rec["p_raw"]
                    rec["fdr_reject_local"] = bool(np.isfinite(q_val) and q_val < alpha)
                cluster_records = channel_cluster_records

            results[band_name] = {
                "significant": any(rec.get("fdr_reject_local", False) for rec in cluster_records),
                "cluster_records": cluster_records,
                "times": time_vec,
                "time_mask": np.zeros_like(time_vec, dtype=bool),
                "time_mask_channels": np.zeros((n_channels, n_times), dtype=bool),
            }

        out_path = Path(output_dir) / f"pain_nonpain_time_clusters_{band_name}.tsv"
        if cluster_records:
            write_tsv(pd.DataFrame(cluster_records), out_path)
        else:
            write_tsv(pd.DataFrame([{
                "subject": f"sub-{subject}",
                "band": band_name,
                "cluster_index": -1,
                "p_value": np.nan,
                "t_start": np.nan,
                "t_end": np.nan,
                "n_timepoints": 0,
                "t_stat_min": np.nan,
                "t_stat_max": np.nan,
            }]), out_path)
        logger.info("Saved pain vs. non-pain cluster results for %s to %s", band_name, out_path)

    return results


def _run_pain_nonpain_cluster_test(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
) -> None:
    """
    Wrapper to compute pain vs. non-pain time clusters using configured ROI and bands.
    Executes only when a pain column is available and both conditions have >=2 trials.
    """
    from eeg_pipeline.utils.analysis.tfr import restrict_epochs_to_roi
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )
    if epochs is None or aligned_events is None:
        logger.warning("Cannot run pain vs. non-pain cluster test: epochs or events unavailable.")
        return

    heatmap_config = config.get("behavior_analysis", {}).get("time_frequency_heatmap", {})
    roi_selection = heatmap_config.get("roi_selection")
    epochs_roi = restrict_epochs_to_roi(epochs, roi_selection, config, logger)

    pain_col = get_pain_column_from_config(config, aligned_events)
    if pain_col is None or pain_col not in aligned_events.columns:
        logger.warning("Pain column not found; skipping pain vs. non-pain cluster test.")
        return

    pain_series = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    valid_pain_mask = pain_series.isin([0, 1])
    invalid_trials = int((~valid_pain_mask).sum())
    if invalid_trials > 0:
        logger.error(
            "Pain column %s contains %d invalid/NaN entries; dropping those trials for pain vs. non-pain cluster test.",
            pain_col, invalid_trials,
        )
        pain_series = pain_series[valid_pain_mask]
        epochs_roi = epochs_roi[valid_pain_mask.to_numpy()]
        aligned_events = aligned_events.loc[valid_pain_mask].reset_index(drop=True)
    if len(pain_series) == 0:
        logger.warning("No valid pain trials remain after filtering; skipping pain vs. non-pain cluster test.")
        return

    try:
        pain_values, _ = validate_pain_binary_values(pain_series, pain_col, logger=logger)
    except ValueError as exc:
        logger.error("Pain column validation failed: %s", exc)
        return

    pain_mask = np.asarray(pain_values == 1, dtype=bool)
    nonpain_mask = np.asarray(pain_values == 0, dtype=bool)

    if pain_mask.sum() < 2 or nonpain_mask.sum() < 2:
        logger.warning(
            "Insufficient pain/non-pain trials (pain=%d, non-pain=%d); skipping pain cluster test.",
            int(pain_mask.sum()), int(nonpain_mask.sum())
        )
        return

    stats_cfg = config.get("behavior_analysis", {}).get("statistics", {})
    # Minimum 5000 permutations for reliable p-value estimation at alpha=0.05
    n_perm = int(stats_cfg.get("n_permutations", 5000))
    alpha = float(stats_cfg.get("sig_alpha", config.get("statistics.sig_alpha", 0.05)))

    bands = get_bands_for_tfr(
        max_freq_available=get_tfr_config(config)[1],
        config=config
    )

    compute_pain_nonpain_time_cluster_test(
        subject=subject,
        pain_epochs=epochs_roi[pain_mask],
        nonpain_epochs=epochs_roi[nonpain_mask],
        output_dir=deriv_stats_path(deriv_root, subject),
        config=config,
        bands=bands,
        n_permutations=n_perm,
        alpha=alpha,
    )

