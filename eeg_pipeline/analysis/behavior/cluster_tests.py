"""Cluster permutation tests for pain vs. non-pain conditions.

Enhanced with:
- Exposed cluster-forming thresholds
- Effect-size maps (Cohen's d) alongside p-maps
- Min-cluster-size filters
- Null distribution data for stability visualization
- FWER control options
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

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
    fdr_bh,
    resolve_cluster_n_jobs,
)
from eeg_pipeline.utils.analysis.tfr import (
    compute_tfr_morlet,
    get_bands_for_tfr,
    get_tfr_config,
)

if TYPE_CHECKING:
    from eeg_pipeline.context.behavior import BehaviorContext


###################################################################
# Cluster Test Configuration
###################################################################


def get_cluster_test_config(config: Any) -> Dict[str, Any]:
    """Extract and validate cluster test configuration parameters.
    
    Returns dict with all cluster test settings for transparency/reproducibility.
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    
    return {
        "n_permutations": int(get_config_value(
            config, "behavior_analysis.statistics.default_n_permutations",
            get_config_value(config, "behavior_analysis.cluster_correction.default_n_permutations", 1000)
        )),
        "alpha": float(get_config_value(
            config, "statistics.sig_alpha",
            get_config_value(config, "behavior_analysis.statistics.default_alpha", 0.05)
        )),
        "fdr_alpha": float(get_config_value(
            config, "behavior_analysis.statistics.fdr_alpha",
            get_config_value(config, "statistics.fdr_alpha", 0.05)
        )),
        "cluster_forming_threshold": float(get_config_value(
            config, "behavior_analysis.cluster_correction.cluster_forming_threshold", 0.05
        )),
        "min_timepoints": int(get_config_value(
            config, "behavior_analysis.cluster_correction.min_timepoints", 2
        )),
        "min_channels": int(get_config_value(
            config, "behavior_analysis.cluster_correction.min_channels", 1
        )),
        "min_cluster_size": int(get_config_value(
            config, "behavior_analysis.cluster_correction.min_cluster_size", 5
        )),
        "tail": int(get_config_value(
            config, "behavior_analysis.cluster_correction.tail", 0
        )),
        "random_seed": int(get_config_value(config, "project.random_state", 42)),
        "fwer_method": get_config_value(
            config, "behavior_analysis.cluster_correction.fwer_method", "cluster"
        ),
    }


def compute_effect_size_map(
    data_pain: np.ndarray,
    data_nonpain: np.ndarray,
) -> np.ndarray:
    """Compute Cohen's d effect size map (channels x times or flattened).
    
    Parameters
    ----------
    data_pain : array (n_trials_pain, n_features)
    data_nonpain : array (n_trials_nonpain, n_features)
    
    Returns
    -------
    d_map : array (n_features,)
        Cohen's d for each feature
    """
    n1 = data_pain.shape[0]
    n2 = data_nonpain.shape[0]
    
    mean1 = np.mean(data_pain, axis=0)
    mean2 = np.mean(data_nonpain, axis=0)
    
    var1 = np.var(data_pain, axis=0, ddof=1)
    var2 = np.var(data_nonpain, axis=0, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Avoid division by zero
    pooled_std = np.where(pooled_std > 0, pooled_std, 1e-10)
    
    d_map = (mean1 - mean2) / pooled_std
    
    return d_map


def save_null_distribution_data(
    null_cluster_masses: np.ndarray,
    observed_masses: np.ndarray,
    output_path: Path,
    band_name: str,
) -> None:
    """Save null distribution data for diagnostic plotting."""
    data = {
        "band": band_name,
        "null_masses": null_cluster_masses.tolist() if len(null_cluster_masses) < 10000 else null_cluster_masses[:10000].tolist(),
        "observed_masses": observed_masses.tolist(),
        "null_mean": float(np.mean(null_cluster_masses)),
        "null_std": float(np.std(null_cluster_masses)),
        "null_95pct": float(np.percentile(null_cluster_masses, 95)),
        "null_99pct": float(np.percentile(null_cluster_masses, 99)),
        "n_permutations": len(null_cluster_masses),
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def compute_pain_nonpain_time_cluster_test(
    subject: str,
    pain_epochs: mne.Epochs,
    nonpain_epochs: mne.Epochs,
    output_dir: Path,
    config,
    bands: Optional[dict] = None,
    n_permutations: Optional[int] = None,
    alpha: Optional[float] = None,
    save_null_distributions: bool = True,
) -> dict:
    """Time-domain cluster permutation test for pain vs. non-pain trials.
    
    Enhanced with:
    - Exposed cluster-forming thresholds in output
    - Effect-size maps (Cohen's d) alongside p-maps
    - Min-cluster-size filters
    - Null distribution data for stability visualization
    """
    from eeg_pipeline.utils.analysis.tfr import apply_baseline_to_tfr
    
    logger = logging.getLogger(__name__)
    ensure_dir(output_dir)
    
    from eeg_pipeline.utils.config.loader import get_config_value
    
    # Get comprehensive cluster test configuration
    cluster_cfg = get_cluster_test_config(config)
    
    if n_permutations is None:
        n_permutations = cluster_cfg["n_permutations"]
    if alpha is None:
        alpha = cluster_cfg["alpha"]
    
    fdr_alpha = cluster_cfg["fdr_alpha"]
    min_timepoints = cluster_cfg["min_timepoints"]
    min_channels = cluster_cfg["min_channels"]
    min_cluster_size = cluster_cfg["min_cluster_size"]
    rng_seed = cluster_cfg["random_seed"]
    
    # Save configuration for reproducibility
    cluster_cfg["n_permutations_used"] = n_permutations
    cluster_cfg["alpha_used"] = alpha

    freq_min, freq_max, _n_freqs, n_cycles_factor, decim, picks = get_tfr_config(config)
    if bands is None:
        bands = get_bands_for_tfr(max_freq_available=freq_max, config=config)

    n_jobs = resolve_cluster_n_jobs(config=config)
    results: dict = {}
    band_cluster_refs = []

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
            
            # Compute effect size map (Cohen's d)
            d_map_flat = compute_effect_size_map(X_pain, X_nonpain)
            d_map_grid = d_map_flat.reshape(n_channels, n_times)

            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                [X_pain, X_nonpain],
                n_permutations=int(n_permutations),
                tail=cluster_cfg["tail"],
                n_jobs=n_jobs,
                adjacency=adjacency,
                out_type="mask",
                seed=rng_seed,
            )
            
            # Save null distribution for diagnostic plots
            if save_null_distributions and H0 is not None:
                null_dist_path = output_dir / f"null_distribution_{band_name}.json"
                observed_masses = []
                for c_idx, c_mask in enumerate(clusters):
                    if c_mask.sum() > 0:
                        observed_masses.append(float(np.abs(T_obs[c_mask]).sum()))
                save_null_distribution_data(
                    np.abs(H0), np.array(observed_masses), null_dist_path, band_name
                )

            sig_inds = np.where(cluster_p_values < alpha)[0]
            sig_mask = np.zeros((n_channels, n_times), dtype=bool)
            T_obs_grid = T_obs.reshape(n_channels, n_times)

            for idx in sig_inds:
                c_mask_flat = clusters[idx]
                c_mask = c_mask_flat.reshape(n_channels, n_times)
                
                # Compute cluster mass for stability metrics
                cluster_mass = float(np.abs(T_obs[c_mask_flat]).sum())
                cluster_size = int(c_mask.sum())
                
                # Enforce min_cluster_size filter
                if cluster_size < min_cluster_size:
                    logger.debug(
                        f"Cluster {idx} rejected: size {cluster_size} < min_cluster_size {min_cluster_size}"
                    )
                    continue
                
                t_inds = np.where(c_mask.any(axis=0))[0]
                ch_inds = np.where(c_mask.any(axis=1))[0]
                if t_inds.size == 0 or ch_inds.size == 0:
                    continue
                if t_inds.size < min_timepoints or ch_inds.size < min_channels:
                    continue
                
                # Compute cluster stability metrics from null distribution
                null_percentile = float(np.mean(np.abs(H0) <= cluster_mass) * 100) if H0 is not None else np.nan
                mc_se = np.sqrt(cluster_p_values[idx] * (1 - cluster_p_values[idx]) / n_permutations)
                
                # Update sig_mask only for clusters that pass all filters
                sig_mask |= c_mask
                t_start = float(time_vec[t_inds[0]])
                t_end = float(time_vec[t_inds[-1]])
                
                # Compute cluster-level effect size (Cohen's d)
                cluster_pain = band_power_pain[:, ch_inds, :][:, :, t_inds].mean(axis=(1, 2))
                cluster_nonpain = band_power_nonpain[:, ch_inds, :][:, :, t_inds].mean(axis=(1, 2))
                
                n1, n2 = len(cluster_pain), len(cluster_nonpain)
                mean_diff = cluster_pain.mean() - cluster_nonpain.mean()
                pooled_std = np.sqrt(((n1 - 1) * cluster_pain.std()**2 + (n2 - 1) * cluster_nonpain.std()**2) / (n1 + n2 - 2))
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
                
                # Bootstrap CI for effect size (vectorized)
                rng_boot = np.random.default_rng(rng_seed)
                n_boot_es = 1000
                
                # Generate all bootstrap indices at once
                boot_idx_pain = rng_boot.integers(0, n1, size=(n_boot_es, n1))
                boot_idx_nonpain = rng_boot.integers(0, n2, size=(n_boot_es, n2))
                
                # Vectorized bootstrap
                boot_pain_samples = cluster_pain[boot_idx_pain]  # (n_boot, n1)
                boot_nonpain_samples = cluster_nonpain[boot_idx_nonpain]  # (n_boot, n2)
                
                boot_means_pain = boot_pain_samples.mean(axis=1)
                boot_means_nonpain = boot_nonpain_samples.mean(axis=1)
                boot_vars_pain = boot_pain_samples.var(axis=1, ddof=1)
                boot_vars_nonpain = boot_nonpain_samples.var(axis=1, ddof=1)
                
                boot_pooled = np.sqrt(((n1 - 1) * boot_vars_pain + (n2 - 1) * boot_vars_nonpain) / (n1 + n2 - 2))
                boot_ds = np.where(boot_pooled > 0, (boot_means_pain - boot_means_nonpain) / boot_pooled, np.nan)
                boot_ds_valid = boot_ds[np.isfinite(boot_ds)]
                
                d_ci_low = np.percentile(boot_ds_valid, 2.5) if len(boot_ds_valid) > 0 else np.nan
                d_ci_high = np.percentile(boot_ds_valid, 97.5) if len(boot_ds_valid) > 0 else np.nan
                
                # Coverage metrics
                total_ch_time = n_channels * n_times
                cluster_coverage = c_mask.sum() / total_ch_time * 100
                
                rec = {
                    "subject": f"sub-{subject}",
                    "band": band_name,
                    "cluster_index": int(idx),
                    "p_value": float(cluster_p_values[idx]),
                    "t_start": t_start,
                    "t_end": t_end,
                    "duration_ms": float((t_end - t_start) * 1000),
                    "n_timepoints": int(len(t_inds)),
                    "n_channels": int(len(ch_inds)),
                    "channels": ",".join(np.array(tfr_pain.ch_names)[ch_inds]),
                    "t_stat_min": float(np.min(T_obs_grid[c_mask])),
                    "t_stat_max": float(np.max(T_obs_grid[c_mask])),
                    "t_stat_mean": float(np.mean(T_obs_grid[c_mask])),
                    # Effect size metrics
                    "cohens_d": float(cohens_d),
                    "d_ci_low": float(d_ci_low),
                    "d_ci_high": float(d_ci_high),
                    # Coverage metrics
                    "coverage_pct": float(cluster_coverage),
                    "n_ch_time_points": int(c_mask.sum()),
                    "total_ch_time_points": int(total_ch_time),
                    # Stability metrics
                    "cluster_mass": cluster_mass,
                    "cluster_size": cluster_size,
                    "null_percentile": null_percentile,
                    "mc_standard_error": float(mc_se),
                    "min_cluster_size_used": min_cluster_size,
                }
                cluster_records.append(rec)
                band_cluster_refs.append(rec)

            results[band_name] = {
                "significant": len(cluster_records) > 0,
                "cluster_records": cluster_records,
                "times": time_vec,
                "time_mask": sig_mask.any(axis=0),
                "time_mask_channels": sig_mask,
                # Effect size map for visualization
                "effect_size_map": d_map_grid,
                "t_stat_map": T_obs_grid,
                # Configuration for reproducibility
                "config": {
                    "n_permutations": n_permutations,
                    "alpha": alpha,
                    "min_timepoints": min_timepoints,
                    "min_channels": min_channels,
                    "min_cluster_size": min_cluster_size,
                    "random_seed": rng_seed,
                    "n_pain_trials": band_power_pain.shape[0],
                    "n_nonpain_trials": band_power_nonpain.shape[0],
                },
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
                    seed=rng_seed,
                )
                channel_pvals.extend(cluster_p_values.tolist())
                for idx, p_val in enumerate(cluster_p_values):
                    t_inds = np.where(clusters[idx])[0]
                    if t_inds.size < min_timepoints:
                        continue
                    
                    # Compute effect size for this channel cluster
                    cluster_pain_ch = X_pain_ch[:, t_inds].mean(axis=1)
                    cluster_nonpain_ch = X_nonpain_ch[:, t_inds].mean(axis=1)
                    
                    n1_ch, n2_ch = len(cluster_pain_ch), len(cluster_nonpain_ch)
                    mean_diff_ch = cluster_pain_ch.mean() - cluster_nonpain_ch.mean()
                    pooled_std_ch = np.sqrt(((n1_ch - 1) * cluster_pain_ch.std()**2 + 
                                             (n2_ch - 1) * cluster_nonpain_ch.std()**2) / (n1_ch + n2_ch - 2))
                    cohens_d_ch = mean_diff_ch / pooled_std_ch if pooled_std_ch > 0 else 0.0
                    
                    # Bootstrap CI
                    rng_boot_ch = np.random.default_rng(rng_seed + ch_idx)
                    boot_ds_ch = []
                    for _ in range(500):  # Fewer iterations for per-channel
                        boot_p = cluster_pain_ch[rng_boot_ch.integers(0, n1_ch, size=n1_ch)]
                        boot_np = cluster_nonpain_ch[rng_boot_ch.integers(0, n2_ch, size=n2_ch)]
                        boot_diff = boot_p.mean() - boot_np.mean()
                        boot_pool = np.sqrt(((n1_ch - 1) * boot_p.std()**2 + 
                                            (n2_ch - 1) * boot_np.std()**2) / (n1_ch + n2_ch - 2))
                        if boot_pool > 0:
                            boot_ds_ch.append(boot_diff / boot_pool)
                    
                    d_ci_low_ch = np.percentile(boot_ds_ch, 2.5) if boot_ds_ch else np.nan
                    d_ci_high_ch = np.percentile(boot_ds_ch, 97.5) if boot_ds_ch else np.nan
                    
                    rec = {
                        "subject": f"sub-{subject}",
                        "band": band_name,
                        "channel": ch_name,
                        "cluster_index": int(idx),
                        "p_raw": float(p_val),
                        "t_start": float(time_vec[t_inds[0]]),
                        "t_end": float(time_vec[t_inds[-1]]),
                        "duration_ms": float((time_vec[t_inds[-1]] - time_vec[t_inds[0]]) * 1000),
                        "n_timepoints": int(len(t_inds)),
                        "t_stat_min": float(np.min(T_obs[t_inds])),
                        "t_stat_max": float(np.max(T_obs[t_inds])),
                        "t_stat_mean": float(np.mean(T_obs[t_inds])),
                        # Effect size
                        "cohens_d": float(cohens_d_ch),
                        "d_ci_low": float(d_ci_low_ch),
                        "d_ci_high": float(d_ci_high_ch),
                        # Coverage
                        "coverage_pct": float(len(t_inds) / n_times * 100),
                    }
                    channel_cluster_records.append(rec)
                    band_cluster_refs.append(rec)

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

    # Apply band-wise FDR across all collected clusters
    if band_cluster_refs:
        p_vals = np.array([rec.get("p_value", np.nan) for rec in band_cluster_refs], dtype=float)
        q_vals = fdr_bh(p_vals, alpha=fdr_alpha, config=config)
        for rec, q_val in zip(band_cluster_refs, q_vals):
            rec["p_fdr_global"] = float(q_val) if np.isfinite(q_val) else np.nan
            rec["fdr_reject_global"] = bool(np.isfinite(q_val) and q_val < fdr_alpha)

    # Update significance flags after global FDR and persist to disk
    for band_name, res in results.items():
        records = res.get("cluster_records", [])
        if records:
            res["significant"] = res.get("significant", False) or any(
                rec.get("fdr_reject_global", False) for rec in records
            )
        out_path = Path(output_dir) / f"pain_nonpain_time_clusters_{band_name}.tsv"
        if records:
            write_tsv(pd.DataFrame(records), out_path)
        else:
            write_tsv(pd.DataFrame([{
                "subject": f"sub-{subject}",
                "band": band_name,
                "cluster_index": -1,
                "p_value": np.nan,
                "p_fdr_global": np.nan,
                "t_start": np.nan,
                "t_end": np.nan,
                "n_timepoints": 0,
                "t_stat_min": np.nan,
                "t_stat_max": np.nan,
            }]), out_path)
        logger.info("Saved pain vs. non-pain cluster results for %s to %s", band_name, out_path)

    return results


def _run_cluster_test_core(
    subject: str,
    epochs: mne.Epochs,
    aligned_events: pd.DataFrame,
    output_dir: Path,
    config,
    logger: logging.Logger,
    n_perm: int,
) -> None:
    """Core implementation for pain vs. non-pain cluster test."""
    from eeg_pipeline.utils.analysis.tfr import restrict_epochs_to_roi
    
    if epochs is None or aligned_events is None:
        logger.warning("Cannot run pain vs. non-pain cluster test: epochs or events unavailable.")
        return
    
    if not epochs.preload:
        epochs.load_data()
    
    heatmap_config = config.get("behavior_analysis.time_frequency_heatmap", {})
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
        logger.warning(f"Pain column {pain_col} contains {invalid_trials} invalid/NaN entries; dropping.")
        pain_series = pain_series[valid_pain_mask]
        epochs_roi = epochs_roi[valid_pain_mask.to_numpy()]
        aligned_events = aligned_events.loc[valid_pain_mask].reset_index(drop=True)
    
    if len(pain_series) == 0:
        logger.warning("No valid pain trials remain; skipping cluster test.")
        return
    
    try:
        pain_values, _ = validate_pain_binary_values(pain_series, pain_col, logger=logger)
    except ValueError as exc:
        logger.error(f"Pain column validation failed: {exc}")
        return
    
    pain_mask = np.asarray(pain_values == 1, dtype=bool)
    nonpain_mask = np.asarray(pain_values == 0, dtype=bool)
    
    if pain_mask.sum() < 2 or nonpain_mask.sum() < 2:
        logger.warning(f"Insufficient trials (pain={int(pain_mask.sum())}, non-pain={int(nonpain_mask.sum())}); skipping.")
        return
    
    stats_cfg = config.get("behavior_analysis.statistics", {})
    n_perm = n_perm if n_perm > 0 else int(stats_cfg.get("n_permutations", 100))
    alpha = float(stats_cfg.get("sig_alpha", config.get("statistics.sig_alpha", 0.05)))
    bands = get_bands_for_tfr(max_freq_available=get_tfr_config(config)[1], config=config)
    
    compute_pain_nonpain_time_cluster_test(
        subject=subject,
        pain_epochs=epochs_roi[pain_mask],
        nonpain_epochs=epochs_roi[nonpain_mask],
        output_dir=output_dir,
        config=config,
        bands=bands,
        n_permutations=n_perm,
        alpha=alpha,
    )


def _run_pain_nonpain_cluster_test(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
) -> None:
    """Compute pain vs. non-pain time clusters by loading data from disk."""
    epochs, aligned_events = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )
    from eeg_pipeline.utils.config.loader import get_config_value
    default_n_perm = int(get_config_value(config, "behavior_analysis.cluster_correction.default_n_permutations", get_config_value(config, "behavior_analysis.statistics.default_n_permutations", 100)))
    _run_cluster_test_core(
        subject, epochs, aligned_events,
        deriv_stats_path(deriv_root, subject), config, logger, n_perm=default_n_perm
    )


def run_cluster_test_from_context(ctx: "BehaviorContext") -> None:
    """Run pain vs. non-pain cluster test using pre-loaded data from context."""
    _run_cluster_test_core(
        ctx.subject, ctx.epochs, ctx.aligned_events,
        ctx.stats_dir, ctx.config, ctx.logger, ctx.n_perm
    )
