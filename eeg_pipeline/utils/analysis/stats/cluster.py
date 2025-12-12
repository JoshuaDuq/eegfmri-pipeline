"""
Cluster Statistics
==================

EEG cluster-based permutation tests and adjacency utilities.
"""

from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse
from mne.stats import permutation_cluster_test, combine_adjacency

if TYPE_CHECKING:
    import mne

from .base import get_statistics_constants, get_fdr_alpha, get_config_value, ensure_config
from eeg_pipeline.utils.io.paths import ensure_dir
from eeg_pipeline.utils.io.tsv import write_tsv
from eeg_pipeline.utils.io.columns import get_pain_column_from_config
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh_values, fdr_bh
from eeg_pipeline.utils.analysis.stats.validation import validate_pain_binary_values
from eeg_pipeline.utils.analysis.tfr import (
    apply_baseline_to_tfr,
    compute_tfr_morlet,
    get_bands_for_tfr,
    get_tfr_config,
)


###################################################################
# Cluster Test Configuration and Utilities
###################################################################


def get_cluster_test_config(config: Any) -> Dict[str, Any]:
    """Extract and validate cluster test configuration parameters."""
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
    """Compute Cohen's d effect size map (channels x times or flattened)."""
    n1 = data_pain.shape[0]
    n2 = data_nonpain.shape[0]

    mean1 = np.mean(data_pain, axis=0)
    mean2 = np.mean(data_nonpain, axis=0)

    var1 = np.var(data_pain, axis=0, ddof=1)
    var2 = np.var(data_nonpain, axis=0, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    pooled_std = np.where(pooled_std > 0, pooled_std, 1e-10)

    return (mean1 - mean2) / pooled_std


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

try:
    from ...config.loader import load_settings
except ImportError:
    load_settings = None


def build_distance_adjacency(
    info_eeg: "mne.Info",
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> Tuple[Any, List[str]]:
    """Build distance-based adjacency matrix as fallback."""
    from scipy.spatial import distance_matrix
    from scipy import sparse

    positions = np.array([ch["loc"][:3] for ch in info_eeg["chs"]])
    if np.all(np.isnan(positions)) or np.allclose(positions, 0):
        logger.warning("Invalid channel positions, returning None adjacency")
        return None, []

    dist = distance_matrix(positions, positions)
    constants = get_statistics_constants(config)
    n_neighbors = min(constants["k_neighbors_adjacency"], len(positions) - 1)

    adj = np.zeros((len(positions), len(positions)), dtype=bool)
    for i in range(len(positions)):
        nearest = np.argsort(dist[i])[1 : n_neighbors + 1]
        adj[i, nearest] = True
        adj[nearest, i] = True

    return sparse.csr_matrix(adj), [ch["ch_name"] for ch in info_eeg["chs"]]


def get_eeg_adjacency(
    info: "mne.Info",
    restrict_picks: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[Any], Optional[np.ndarray], Optional["mne.Info"]]:
    """Get EEG channel adjacency for cluster tests."""
    import mne

    if logger is None:
        logger = logging.getLogger(__name__)

    eeg_picks_all = mne.pick_types(info, eeg=True, exclude=[])
    if len(eeg_picks_all) == 0:
        return None, None, None

    if restrict_picks is not None:
        restrict_set = set(np.asarray(restrict_picks, dtype=int))
        eeg_picks = np.array([p for p in eeg_picks_all if p in restrict_set], dtype=int)
    else:
        eeg_picks = np.asarray(eeg_picks_all, dtype=int)

    if eeg_picks.size == 0:
        return None, None, None

    info_eeg = mne.pick_info(info, sel=eeg_picks.tolist())

    try:
        adjacency, _ = mne.channels.find_ch_adjacency(info_eeg, ch_type="eeg")
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Delaunay adjacency failed ({type(e).__name__}), using distance fallback")
        adjacency, _ = build_distance_adjacency(info_eeg, logger)
        if adjacency is None:
            return None, eeg_picks, info_eeg

    return adjacency, eeg_picks, info_eeg


def build_full_mask_from_eeg(
    sig_mask_eeg: np.ndarray,
    n_ch_total: int,
    eeg_picks: np.ndarray,
) -> np.ndarray:
    """Expand EEG-only mask to full channel mask."""
    full_mask = np.zeros(n_ch_total, dtype=bool)
    full_mask[eeg_picks] = sig_mask_eeg.astype(bool)
    return full_mask


def extract_cluster_indices(cluster: Any, expected_length: int) -> np.ndarray:
    """Extract cluster indices from cluster object."""
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool and cluster.shape[0] == expected_length:
        return np.where(cluster)[0]
    return np.asarray(cluster)


def compute_cluster_masses(clusters: List[Any], t_stat: np.ndarray) -> List[float]:
    """Compute mass (sum of |t|) for each cluster."""
    masses = []
    for cluster in clusters:
        idx = extract_cluster_indices(cluster, t_stat.shape[0])
        mass = float(np.nansum(np.abs(t_stat[idx]))) if idx.size > 0 else 0.0
        masses.append(mass)
    return masses


def cluster_mask_from_clusters(
    clusters: List[Any],
    p_values: np.ndarray,
    n_features: int,
    alpha: float,
) -> np.ndarray:
    """Create significant mask from cluster results."""
    mask = np.zeros(n_features, dtype=bool)
    for cluster, pval in zip(clusters, p_values):
        if float(pval) > float(alpha):
            continue
        idx = extract_cluster_indices(cluster, n_features)
        if isinstance(cluster, np.ndarray) and cluster.dtype == bool and cluster.shape[0] == n_features:
            mask |= cluster
        else:
            mask[idx] = True
    return mask


def resolve_cluster_n_jobs(config=None) -> int:
    """Resolve number of parallel jobs for cluster tests."""
    raw = os.getenv("EEG_CLUSTER_N_JOBS")
    if raw and raw.strip().lower() not in {"auto", ""}:
        try:
            return max(1, int(raw))
        except ValueError:
            pass

    if config is None and load_settings is not None:
        try:
            config = load_settings()
        except Exception:
            pass

    default = -1
    if config is not None:
        default = int(get_config_value(config, "statistics.cluster_n_jobs", -1))
    return default


def cluster_test_two_sample(
    group_a: np.ndarray,
    group_b: np.ndarray,
    info: "mne.Info",
    alpha: Optional[float] = None,
    paired: bool = False,
    n_permutations: Optional[int] = None,
    restrict_picks: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """
    Two-sample cluster permutation test on channel data.
    
    Returns (sig_mask, p_value, cluster_size, cluster_mass).
    """
    from mne.stats import permutation_cluster_test, permutation_cluster_1samp_test

    config = ensure_config(config)

    if alpha is None:
        alpha = get_fdr_alpha(config)
    if n_permutations is None:
        n_permutations = int(get_config_value(config, "statistics.cluster_n_perm", get_config_value(config, "statistics.cluster_n_perm", 100)))

    adjacency, eeg_picks, info_eeg = get_eeg_adjacency(info, restrict_picks=restrict_picks)
    if eeg_picks is None:
        return None, None, None, None

    if n_jobs is None:
        n_jobs = resolve_cluster_n_jobs(config)

    a_eeg = np.asarray(group_a)[:, eeg_picks]
    b_eeg = np.asarray(group_b)[:, eeg_picks]

    if paired and a_eeg.shape[0] == b_eeg.shape[0]:
        diff = a_eeg - b_eeg
        t_stat, clusters, pvals, _ = permutation_cluster_1samp_test(
            diff, n_permutations=n_permutations, adjacency=adjacency, tail=0, out_type="mask", n_jobs=n_jobs
        )
    else:
        t_stat, clusters, pvals, _ = permutation_cluster_test(
            [a_eeg, b_eeg], n_permutations=n_permutations, adjacency=adjacency, tail=0, out_type="mask", n_jobs=n_jobs
        )

    sig_eeg = cluster_mask_from_clusters(clusters, pvals, n_features=a_eeg.shape[1], alpha=alpha)
    sig_full = build_full_mask_from_eeg(sig_eeg, len(info["ch_names"]), eeg_picks)

    if len(clusters) == 0:
        return sig_full, None, None, None

    masses = compute_cluster_masses(clusters, t_stat)
    if not masses:
        return sig_full, None, None, None

    best = int(np.nanargmax(masses))
    best_idx = extract_cluster_indices(clusters[best], t_stat.shape[0])

    p = float(pvals[best]) if np.isfinite(pvals[best]) else None
    size = int(best_idx.size)
    mass = float(np.nansum(np.abs(t_stat[best_idx]))) if best_idx.size > 0 else 0.0

    return sig_full, p, size, mass


def cluster_test_epochs(
    tfr_epochs: "mne.time_frequency.EpochsTFR",
    group_a_mask: np.ndarray,
    group_b_mask: np.ndarray,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
    paired: bool = False,
    alpha: Optional[float] = None,
    n_permutations: Optional[int] = None,
    restrict_picks: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """Cluster test on EpochsTFR data.
    
    Args:
        tfr_epochs: MNE EpochsTFR object
        group_a_mask: Boolean mask for first group
        group_b_mask: Boolean mask for second group
        fmin: Minimum frequency
        fmax: Maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        paired: Whether to use paired test
        alpha: Significance threshold
        n_permutations: Number of permutations
        restrict_picks: Optional channel restriction
        n_jobs: Number of parallel jobs
        config: Configuration object
        logger: Optional logger
    
    Returns:
        Tuple of (significance_mask, cluster_p_min, cluster_k, cluster_mass)
    """
    import mne
    
    info = tfr_epochs.info
    eeg_picks = mne.pick_types(info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        if logger:
            logger.warning("No EEG channels found")
        return None, None, None, None
    
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    t_mask = (times >= tmin) & (times < tmax)
    
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None, None, None, None
    
    data = np.asarray(tfr_epochs.data)[:, :, f_mask, :][:, :, :, t_mask]
    ch_power = data.mean(axis=(2, 3))
    
    if ch_power.shape[1] != len(info["ch_names"]):
        if logger:
            logger.error("Channel dimension mismatch")
        return None, None, None, None
    
    group_a = ch_power[np.asarray(group_a_mask, dtype=bool), :]
    group_b = ch_power[np.asarray(group_b_mask, dtype=bool), :]
    
    if group_a.shape[0] < 2 or group_b.shape[0] < 2:
        return None, None, None, None
    
    return cluster_test_two_sample(
        group_a, group_b, info, alpha=alpha, paired=paired,
        n_permutations=n_permutations, restrict_picks=restrict_picks,
        n_jobs=n_jobs, config=config
    )


###################################################################
# Pain vs Non-Pain Cluster Tests
###################################################################


def compute_pain_nonpain_time_cluster_test(
    subject: str,
    pain_epochs: "mne.Epochs",
    nonpain_epochs: "mne.Epochs",
    output_dir: Path,
    config,
    bands: Optional[dict] = None,
    n_permutations: Optional[int] = None,
    alpha: Optional[float] = None,
    save_null_distributions: bool = True,
) -> dict:
    """Time-domain cluster permutation test for pain vs. non-pain trials."""
    logger = logging.getLogger(__name__)
    ensure_dir(output_dir)

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
        band_power_pain = tfr_pain.data[:, :, :, time_mask].mean(axis=2)
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

                boot_idx_pain = rng_boot.integers(0, n1, size=(n_boot_es, n1))
                boot_idx_nonpain = rng_boot.integers(0, n2, size=(n_boot_es, n2))

                boot_pain_samples = cluster_pain[boot_idx_pain]
                boot_nonpain_samples = cluster_nonpain[boot_idx_nonpain]

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
                    "cohens_d": float(cohens_d),
                    "d_ci_low": float(d_ci_low),
                    "d_ci_high": float(d_ci_high),
                    "coverage_pct": float(cluster_coverage),
                    "n_ch_time_points": int(c_mask.sum()),
                    "total_ch_time_points": int(total_ch_time),
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
                "effect_size_map": d_map_grid,
                "t_stat_map": T_obs_grid,
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

                    cluster_pain_ch = X_pain_ch[:, t_inds].mean(axis=1)
                    cluster_nonpain_ch = X_nonpain_ch[:, t_inds].mean(axis=1)

                    n1_ch, n2_ch = len(cluster_pain_ch), len(cluster_nonpain_ch)
                    mean_diff_ch = cluster_pain_ch.mean() - cluster_nonpain_ch.mean()
                    pooled_std_ch = np.sqrt(((n1_ch - 1) * cluster_pain_ch.std()**2 +
                                             (n2_ch - 1) * cluster_nonpain_ch.std()**2) / (n1_ch + n2_ch - 2))
                    cohens_d_ch = mean_diff_ch / pooled_std_ch if pooled_std_ch > 0 else 0.0

                    rng_boot_ch = np.random.default_rng(rng_seed + ch_idx)
                    boot_ds_ch = []
                    for _ in range(500):
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
                        "cohens_d": float(cohens_d_ch),
                        "d_ci_low": float(d_ci_low_ch),
                        "d_ci_high": float(d_ci_high_ch),
                    }
                    channel_cluster_records.append(rec)

            if channel_cluster_records:
                p_vals = np.array([rec["p_raw"] for rec in channel_cluster_records])
                _, q_vals = fdr_bh_values(np.asarray(channel_pvals), alpha=alpha)
                for rec, q_val in zip(channel_cluster_records, q_vals):
                    rec["p_fdr_local"] = float(q_val) if np.isfinite(q_val) else np.nan
                    rec["p_value"] = rec["p_fdr_local"] if np.isfinite(rec["p_fdr_local"]) else rec["p_raw"]
                    rec["fdr_reject_local"] = bool(np.isfinite(q_val) and q_val < alpha)

                records = pd.DataFrame(channel_cluster_records)
                results[band_name] = {
                    "significant": any(rec.get("fdr_reject_local", False) for rec in channel_cluster_records),
                    "cluster_records": channel_cluster_records,
                    "times": time_vec,
                }
            else:
                results[band_name] = {"significant": False, "cluster_records": []}

    # Global FDR across bands (when adjacency is available)
    if band_cluster_refs:
        p_vals = [rec["p_value"] for rec in band_cluster_refs if np.isfinite(rec["p_value"])]
        q_vals = fdr_bh(p_vals, alpha=fdr_alpha, config=config) if p_vals else []
        for rec, q_val in zip(band_cluster_refs, q_vals):
            rec["p_fdr_global"] = float(q_val) if np.isfinite(q_val) else np.nan
            rec["fdr_reject_global"] = bool(np.isfinite(q_val) and q_val < fdr_alpha)

    # Save TSV outputs
    for band_name, result in results.items():
        records = result.get("cluster_records", [])
        if not records:
            continue
        out_path = output_dir / f"cluster_results_{band_name}.tsv"
        write_tsv(pd.DataFrame(records), out_path)
        logger.info("Saved pain vs. non-pain cluster results for %s to %s", band_name, out_path)

    return results


def _run_cluster_test_core(
    subject: str,
    epochs: "mne.Epochs",
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

