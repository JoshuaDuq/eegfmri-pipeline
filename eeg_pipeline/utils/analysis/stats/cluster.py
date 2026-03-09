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

from .base import (
    get_statistics_constants,
    get_fdr_alpha,
    get_config_value,
    ensure_config,
)
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.tsv import write_tsv
from eeg_pipeline.utils.data.columns import (
    get_binary_outcome_column_from_config,
    get_condition_column_from_config,
)
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh_values, fdr_bh
from eeg_pipeline.utils.analysis.stats.effect_size import compute_cohens_d_with_bootstrap_ci


###################################################################
# Cluster Test Configuration and Utilities
###################################################################


def _get_config_int(config: Any, *paths: str, default: int) -> int:
    """Extract integer config value from multiple possible paths."""
    for path in paths:
        val = get_config_value(config, path, None)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return int(default)


def _get_config_float(config: Any, *paths: str, default: float) -> float:
    """Extract float config value from multiple possible paths."""
    for path in paths:
        val = get_config_value(config, path, None)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return float(default)


def get_cluster_test_config(config: Any) -> Dict[str, Any]:
    """Extract and validate cluster test configuration parameters."""

    return {
        "n_permutations": _get_config_int(
            config,
            "behavior_analysis.cluster.n_permutations",
            "behavior_analysis.cluster_correction.n_permutations",
            "behavior_analysis.statistics.n_permutations",
            "statistics.n_permutations",
            default=1000,
        ),
        "alpha": _get_config_float(
            config,
            "statistics.alpha",
            "statistics.sig_alpha",
            "behavior_analysis.statistics.alpha",
            default=0.05,
        ),
        "fdr_alpha": _get_config_float(
            config,
            "behavior_analysis.statistics.fdr_alpha",
            "statistics.fdr_alpha",
            default=0.05,
        ),
        "cluster_forming_threshold": _get_config_float(
            config,
            "behavior_analysis.cluster.forming_threshold",
            "behavior_analysis.cluster_correction.cluster_forming_threshold",
            default=0.05,
        ),
        "min_timepoints": _get_config_int(
            config,
            "behavior_analysis.cluster.min_timepoints",
            "behavior_analysis.cluster_correction.min_timepoints",
            default=2,
        ),
        "min_channels": _get_config_int(
            config,
            "behavior_analysis.cluster.min_channels",
            "behavior_analysis.cluster_correction.min_channels",
            default=1,
        ),
        "min_cluster_size": _get_config_int(
            config,
            "behavior_analysis.cluster.min_cluster_size",
            "behavior_analysis.cluster_correction.min_cluster_size",
            default=5,
        ),
        "tail": _get_config_int(
            config,
            "behavior_analysis.cluster.tail",
            "behavior_analysis.cluster_correction.tail",
            default=0,
        ),
        "random_seed": _get_config_int(config, "project.random_state", default=42),
        "fwer_method": get_config_value(
            config,
            "behavior_analysis.cluster.fwer_method",
            get_config_value(
                config,
                "behavior_analysis.cluster_correction.fwer_method",
                "cluster",
            ),
        ),
    }


def compute_effect_size_map(
    data_group_a: np.ndarray,
    data_group_b: np.ndarray,
) -> np.ndarray:
    """Compute Cohen's d effect size map (channels x times or flattened)."""
    n_group_a = data_group_a.shape[0]
    n_group_b = data_group_b.shape[0]

    mean_group_a = np.mean(data_group_a, axis=0)
    mean_group_b = np.mean(data_group_b, axis=0)

    var_group_a = np.var(data_group_a, axis=0, ddof=1)
    var_group_b = np.var(data_group_b, axis=0, ddof=1)

    pooled_std = np.sqrt(
        ((n_group_a - 1) * var_group_a + (n_group_b - 1) * var_group_b)
        / (n_group_a + n_group_b - 2)
    )
    pooled_std = np.where(pooled_std > 0, pooled_std, 1e-10)

    return (mean_group_a - mean_group_b) / pooled_std


def save_null_distribution_data(
    null_cluster_masses: np.ndarray,
    observed_masses: np.ndarray,
    output_path: Path,
    band_name: str,
) -> None:
    """Save null distribution data for diagnostic plotting."""
    max_masses_to_save = 10000
    null_masses_list = (
        null_cluster_masses.tolist()
        if len(null_cluster_masses) < max_masses_to_save
        else null_cluster_masses[:max_masses_to_save].tolist()
    )

    data = {
        "band": band_name,
        "null_masses": null_masses_list,
        "observed_masses": observed_masses.tolist(),
        "null_mean": float(np.mean(null_cluster_masses)),
        "null_std": float(np.std(null_cluster_masses)),
        "null_95pct": float(np.percentile(null_cluster_masses, 95)),
        "null_99pct": float(np.percentile(null_cluster_masses, 99)),
        "n_permutations": len(null_cluster_masses),
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def build_distance_adjacency(
    info_eeg: "mne.Info",
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> Tuple[Any, List[str]]:
    """Build distance-based adjacency matrix as fallback."""
    from scipy.spatial import distance_matrix

    positions = np.array([ch["loc"][:3] for ch in info_eeg["chs"]])
    if np.all(np.isnan(positions)) or np.allclose(positions, 0):
        logger.warning("Invalid channel positions, returning None adjacency")
        return None, []

    distance_matrix_2d = distance_matrix(positions, positions)
    constants = get_statistics_constants(config)
    n_neighbors = min(constants["k_neighbors_adjacency"], len(positions) - 1)

    n_positions = len(positions)
    adjacency = np.zeros((n_positions, n_positions), dtype=bool)
    for i in range(n_positions):
        nearest_indices = np.argsort(distance_matrix_2d[i])[1 : n_neighbors + 1]
        adjacency[i, nearest_indices] = True
        adjacency[nearest_indices, i] = True

    return sparse.csr_matrix(adjacency), [ch["ch_name"] for ch in info_eeg["chs"]]


_EEG_ADJ_CACHE: dict[tuple[str, ...], Any] = {}


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
    cache_key = tuple(info_eeg.get("ch_names", []))
    if cache_key and cache_key in _EEG_ADJ_CACHE:
        return _EEG_ADJ_CACHE[cache_key], eeg_picks, info_eeg

    try:
        adjacency, _ = mne.channels.find_ch_adjacency(info_eeg, ch_type="eeg")
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Delaunay adjacency failed ({type(e).__name__}), using distance fallback")
        adjacency, _ = build_distance_adjacency(info_eeg, logger)
        if adjacency is None:
            return None, eeg_picks, info_eeg

    if cache_key:
        _EEG_ADJ_CACHE[cache_key] = adjacency
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


def resolve_cluster_n_jobs(config: Optional[Any] = None) -> int:
    """Resolve number of parallel jobs for cluster tests."""
    env_value = os.getenv("EEG_CLUSTER_N_JOBS")
    if env_value and env_value.strip().lower() not in {"auto", ""}:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass

    if config is None:
        try:
            from eeg_pipeline.utils.config.loader import load_config

            config = load_config()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to load config for cluster n_jobs resolution; using defaults: %s",
                exc,
            )

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
        n_permutations = int(get_config_value(config, "statistics.cluster_n_perm", 1000))

    adjacency, eeg_picks, info_eeg = get_eeg_adjacency(info, restrict_picks=restrict_picks)
    if eeg_picks is None:
        return None, None, None, None

    if n_jobs is None:
        n_jobs = resolve_cluster_n_jobs(config)

    group_a_eeg = np.asarray(group_a)[:, eeg_picks]
    group_b_eeg = np.asarray(group_b)[:, eeg_picks]

    if paired and group_a_eeg.shape[0] == group_b_eeg.shape[0]:
        differences = group_a_eeg - group_b_eeg
        t_stat, clusters, pvals, _ = permutation_cluster_1samp_test(
            differences,
            n_permutations=n_permutations,
            adjacency=adjacency,
            tail=0,
            out_type="mask",
            n_jobs=n_jobs,
        )
    else:
        t_stat, clusters, pvals, _ = permutation_cluster_test(
            [group_a_eeg, group_b_eeg],
            n_permutations=n_permutations,
            adjacency=adjacency,
            tail=0,
            out_type="mask",
            n_jobs=n_jobs,
        )

    sig_eeg = cluster_mask_from_clusters(clusters, pvals, n_features=group_a_eeg.shape[1], alpha=alpha)

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
# Two-Condition Cluster Tests
###################################################################


def _create_band_summary_record(
    subject: str,
    band_name: str,
    fmin: float,
    fmax: float,
    condition_column: Optional[str],
    condition_labels: Tuple[str, str],
    condition_values: Optional[Tuple[Any, Any]],
    n_clusters_found: int,
    n_significant: int,
    cluster_p_values: np.ndarray,
    n_condition_a_trials: int,
    n_condition_b_trials: int,
    n_permutations: int,
    alpha: float,
    n_channels: int,
    n_timepoints: int,
    mean_effect_size: float = np.nan,
    max_effect_size: float = np.nan,
) -> Dict[str, Any]:
    """Create diagnostic summary record for a band."""
    return {
        "subject": f"sub-{subject}",
        "band": band_name,
        "condition_column": str(condition_column) if condition_column else "",
        "condition_a_label": str(condition_labels[0]),
        "condition_b_label": str(condition_labels[1]),
        "condition_a_value": (
            str(condition_values[0]) if condition_values is not None else ""
        ),
        "condition_b_value": (
            str(condition_values[1]) if condition_values is not None else ""
        ),
        "fmin": float(fmin),
        "fmax": float(fmax),
        "n_clusters_found": n_clusters_found,
        "n_significant": n_significant,
        "min_p_value": (
            float(np.min(cluster_p_values)) if len(cluster_p_values) > 0 else np.nan
        ),
        "n_condition_a_trials": n_condition_a_trials,
        "n_condition_b_trials": n_condition_b_trials,
        "n_permutations": n_permutations,
        "alpha": alpha,
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "mean_effect_size": mean_effect_size,
        "max_effect_size": max_effect_size,
        "status": (
            "significant"
            if n_significant > 0
            else ("clusters_found" if n_clusters_found > 0 else "no_clusters")
        ),
    }


def compute_two_condition_time_cluster_test(
    subject: str,
    group_a_epochs: "mne.Epochs",
    group_b_epochs: "mne.Epochs",
    output_dir: Path,
    config,
    bands: Optional[dict] = None,
    n_permutations: Optional[int] = None,
    alpha: Optional[float] = None,
    save_null_distributions: bool = True,
    *,
    condition_column: Optional[str] = None,
    condition_values: Optional[Tuple[Any, Any]] = None,
    condition_labels: Optional[Tuple[str, str]] = None,
) -> dict:
    """Time-domain cluster permutation test for two trial groups.
    
    Generic two-condition comparison that can work with any condition column.
    Condition is specified via condition_column
    and condition_values parameters.
    """
    logger = logging.getLogger(__name__)
    ensure_dir(output_dir)

    cluster_cfg = get_cluster_test_config(config)

    if n_permutations is None:
        n_permutations = cluster_cfg["n_permutations"]
    if alpha is None:
        alpha = cluster_cfg["alpha"]

    # Warn if n_permutations is too low for meaningful results
    if n_permutations < 100:
        logger.warning(
            "Cluster test using only %d permutations - results may not be statistically meaningful. "
            "For publication-quality results, use at least 1000 permutations.",
            n_permutations,
        )

    fdr_alpha = cluster_cfg["fdr_alpha"]
    min_timepoints = cluster_cfg["min_timepoints"]
    min_channels = cluster_cfg["min_channels"]
    min_cluster_size = cluster_cfg["min_cluster_size"]
    rng_seed = cluster_cfg["random_seed"]

    cluster_cfg["n_permutations_used"] = n_permutations
    cluster_cfg["alpha_used"] = alpha

    # Local import to avoid circular dependency
    from eeg_pipeline.utils.analysis.tfr import get_tfr_config, get_bands_for_tfr
    freq_min, freq_max, _n_freqs, n_cycles_factor, decim, picks = get_tfr_config(config)
    if bands is None:
        bands = get_bands_for_tfr(max_freq_available=freq_max, config=config)

    n_jobs = resolve_cluster_n_jobs(config=config)
    results: dict = {}
    band_cluster_refs = []
    band_summaries: List[Dict[str, Any]] = []

    if condition_labels is None:
        if condition_values is not None and len(condition_values) == 2:
            condition_labels = (str(condition_values[0]), str(condition_values[1]))
        else:
            condition_labels = ("group_a", "group_b")

    for band_name, (fmin, fmax) in bands.items():
        if fmin >= fmax:
            logger.warning("Band %s has invalid range [%s, %s]; skipping.", band_name, fmin, fmax)
            continue

        freqs = np.arange(fmin, fmax, 2.0)
        if freqs.size == 0:
            logger.warning("Band %s produced no frequency bins; skipping.", band_name)
            continue

        logger.info(
            "Computing %s vs %s time-cluster test for %s band [%s, %s] (sub-%s)",
            condition_labels[0],
            condition_labels[1],
            band_name,
            fmin,
            fmax,
            subject,
        )

        # Local import to avoid circular dependency
        from eeg_pipeline.utils.analysis.tfr import compute_tfr_morlet, apply_baseline_to_tfr
        tfr_group_a = compute_tfr_morlet(
            group_a_epochs,
            config,
            logger=logger,
            freqs=freqs,
            picks=picks,
            decim=decim,
        )
        tfr_group_b = compute_tfr_morlet(
            group_b_epochs,
            config,
            logger=logger,
            freqs=freqs,
            picks=picks,
            decim=decim,
        )
        apply_baseline_to_tfr(tfr_group_a, config, logger)
        apply_baseline_to_tfr(tfr_group_b, config, logger)

        time_mask = tfr_group_a.times >= 0.0
        band_power_group_a = tfr_group_a.data[:, :, :, time_mask].mean(axis=2)
        band_power_group_b = tfr_group_b.data[:, :, :, time_mask].mean(axis=2)
        time_vec = tfr_group_a.times[time_mask]

        if band_power_group_a.shape[0] < 2 or band_power_group_b.shape[0] < 2:
            logger.warning(
                "Insufficient epochs for band %s (%s=%d, %s=%d); skipping cluster test.",
                band_name,
                condition_labels[0],
                band_power_group_a.shape[0],
                condition_labels[1],
                band_power_group_b.shape[0],
            )
            results[band_name] = {"error": "insufficient_epochs"}
            continue

        n_channels = band_power_group_a.shape[1]
        n_times = band_power_group_a.shape[2]

        adjacency_eeg, _, _ = get_eeg_adjacency(tfr_group_a.info, logger=logger)
        cluster_records = []

        if adjacency_eeg is not None:
            adjacency = combine_adjacency(n_times, adjacency_eeg)
            band_power_group_a_flat = band_power_group_a.reshape(band_power_group_a.shape[0], -1)
            band_power_group_b_flat = band_power_group_b.reshape(band_power_group_b.shape[0], -1)

            effect_size_map_flat = compute_effect_size_map(
                band_power_group_a_flat, band_power_group_b_flat
            )
            effect_size_map = effect_size_map_flat.reshape(n_channels, n_times)

            cluster_test_result = permutation_cluster_test(
                [band_power_group_a_flat, band_power_group_b_flat],
                n_permutations=int(n_permutations),
                tail=0,
                n_jobs=n_jobs,
                adjacency=adjacency,
                out_type="mask",
                seed=rng_seed,
            )
            t_stat_observed, clusters, cluster_p_values, null_distribution = (
                cluster_test_result
            )

            if save_null_distributions and null_distribution is not None:
                null_dist_path = output_dir / f"null_distribution_{band_name}.json"
                observed_masses = []
                for cluster_idx, cluster_mask in enumerate(clusters):
                    if cluster_mask.sum() > 0:
                        observed_masses.append(
                            float(np.abs(t_stat_observed[cluster_mask]).sum())
                        )
                save_null_distribution_data(
                    np.abs(null_distribution),
                    np.array(observed_masses),
                    null_dist_path,
                    band_name,
                )

            n_clusters_found = len(clusters)
            significant_indices = np.where(cluster_p_values < alpha)[0]
            n_significant = len(significant_indices)

            if n_clusters_found > 0:
                min_p = float(np.min(cluster_p_values)) if len(cluster_p_values) > 0 else np.nan
                logger.info(
                    "%s band: found %d clusters, %d significant (p < %.3f), min p-value=%.4f",
                    band_name, n_clusters_found, n_significant, alpha, min_p,
                )
            else:
                logger.info("%s band: no clusters found", band_name)
            
            significant_mask = np.zeros((n_channels, n_times), dtype=bool)
            t_stat_grid = t_stat_observed.reshape(n_channels, n_times)

            for cluster_idx in significant_indices:
                cluster_mask_flat = clusters[cluster_idx]
                cluster_mask = cluster_mask_flat.reshape(n_channels, n_times)

                cluster_mass = float(np.abs(t_stat_observed[cluster_mask_flat]).sum())
                cluster_size = int(cluster_mask.sum())

                if cluster_size < min_cluster_size:
                    logger.debug(
                        "Cluster %d rejected: size %d < min_cluster_size %d",
                        cluster_idx,
                        cluster_size,
                        min_cluster_size,
                    )
                    continue

                time_indices = np.where(cluster_mask.any(axis=0))[0]
                channel_indices = np.where(cluster_mask.any(axis=1))[0]
                if time_indices.size == 0 or channel_indices.size == 0:
                    continue
                if time_indices.size < min_timepoints or channel_indices.size < min_channels:
                    continue

                null_percentile = (
                    float(np.mean(np.abs(null_distribution) <= cluster_mass) * 100)
                    if null_distribution is not None
                    else np.nan
                )
                monte_carlo_se = np.sqrt(
                    cluster_p_values[cluster_idx]
                    * (1 - cluster_p_values[cluster_idx])
                    / n_permutations
                )

                significant_mask |= cluster_mask
                time_start = float(time_vec[time_indices[0]])
                time_end = float(time_vec[time_indices[-1]])

                cluster_group_a = (
                    band_power_group_a[:, channel_indices, :][:, :, time_indices].mean(axis=(1, 2))
                )
                cluster_group_b = (
                    band_power_group_b[:, channel_indices, :][:, :, time_indices].mean(axis=(1, 2))
                )

                cohens_d, d_ci_low, d_ci_high = compute_cohens_d_with_bootstrap_ci(
                    cluster_group_a,
                    cluster_group_b,
                    random_seed=rng_seed,
                )

                total_channel_time_points = n_channels * n_times
                cluster_coverage = cluster_mask.sum() / total_channel_time_points * 100

                cluster_record = {
                    "subject": f"sub-{subject}",
                    "band": band_name,
                    "condition_column": str(condition_column) if condition_column else "",
                    "condition_a_label": str(condition_labels[0]),
                    "condition_b_label": str(condition_labels[1]),
                    "condition_a_value": (
                        str(condition_values[0]) if condition_values is not None else ""
                    ),
                    "condition_b_value": (
                        str(condition_values[1]) if condition_values is not None else ""
                    ),
                    "cluster_index": int(cluster_idx),
                    "p_value": float(cluster_p_values[cluster_idx]),
                    "t_start": time_start,
                    "t_end": time_end,
                    "duration_ms": float((time_end - time_start) * 1000),
                    "n_timepoints": int(len(time_indices)),
                    "n_channels": int(len(channel_indices)),
                    "channels": ",".join(np.array(tfr_group_a.ch_names)[channel_indices]),
                    "t_stat_min": float(np.min(t_stat_grid[cluster_mask])),
                    "t_stat_max": float(np.max(t_stat_grid[cluster_mask])),
                    "t_stat_mean": float(np.mean(t_stat_grid[cluster_mask])),
                    "cohens_d": cohens_d,
                    "d_ci_low": d_ci_low,
                    "d_ci_high": d_ci_high,
                    "coverage_pct": float(cluster_coverage),
                    "n_ch_time_points": int(cluster_mask.sum()),
                    "total_ch_time_points": int(total_channel_time_points),
                    "cluster_mass": cluster_mass,
                    "cluster_size": cluster_size,
                    "null_percentile": null_percentile,
                    "mc_standard_error": monte_carlo_se,
                    "min_cluster_size_used": min_cluster_size,
                }
                cluster_records.append(cluster_record)
                band_cluster_refs.append(cluster_record)

            band_summaries.append(
                _create_band_summary_record(
                    subject=subject,
                    band_name=band_name,
                    fmin=fmin,
                    fmax=fmax,
                    condition_column=condition_column,
                    condition_labels=condition_labels,
                    condition_values=condition_values,
                    n_clusters_found=n_clusters_found,
                    n_significant=n_significant,
                    cluster_p_values=cluster_p_values,
                    n_condition_a_trials=int(band_power_group_a.shape[0]),
                    n_condition_b_trials=int(band_power_group_b.shape[0]),
                    n_permutations=n_permutations,
                    alpha=alpha,
                    n_channels=n_channels,
                    n_timepoints=n_times,
                    mean_effect_size=float(np.nanmean(np.abs(effect_size_map))),
                    max_effect_size=float(np.nanmax(np.abs(effect_size_map))),
                )
            )

            results[band_name] = {
                "significant": len(cluster_records) > 0,
                "cluster_records": cluster_records,
                "times": time_vec,
                "time_mask": significant_mask.any(axis=0),
                "time_mask_channels": significant_mask,
                "effect_size_map": effect_size_map,
                "t_stat_map": t_stat_grid,
                "n_clusters_found": n_clusters_found,
                "n_significant": n_significant,
                "config": {
                    "n_permutations": n_permutations,
                    "alpha": alpha,
                    "min_timepoints": min_timepoints,
                    "min_channels": min_channels,
                    "min_cluster_size": min_cluster_size,
                    "random_seed": rng_seed,
                    "condition_column": str(condition_column) if condition_column else "",
                    "condition_a_label": str(condition_labels[0]),
                    "condition_b_label": str(condition_labels[1]),
                    "condition_a_value": str(condition_values[0]) if condition_values is not None else "",
                    "condition_b_value": str(condition_values[1]) if condition_values is not None else "",
                    "n_condition_a_trials": int(band_power_group_a.shape[0]),
                    "n_condition_b_trials": int(band_power_group_b.shape[0]),
                },
            }
        else:
            logger.warning(
                "EEG adjacency unavailable; running time-only cluster tests per-channel with BH across channels."
            )
            time_adjacency = sparse.diags(
                [1, 1, 1], [-1, 0, 1], shape=(n_times, n_times), format="csr"
            )
            channel_cluster_records = []
            channel_pvals = []

            for channel_idx, channel_name in enumerate(tfr_group_a.ch_names):
                band_power_group_a_ch = band_power_group_a[:, channel_idx, :]
                band_power_group_b_ch = band_power_group_b[:, channel_idx, :]
                if band_power_group_a_ch.shape[0] < 2 or band_power_group_b_ch.shape[0] < 2:
                    continue
                t_stat_ch, clusters_ch, cluster_p_values_ch, _ = permutation_cluster_test(
                    [band_power_group_a_ch, band_power_group_b_ch],
                    n_permutations=int(n_permutations),
                    tail=0,
                    n_jobs=n_jobs,
                    adjacency=time_adjacency,
                    out_type="mask",
                    seed=rng_seed,
                )
                channel_pvals.extend(cluster_p_values_ch.tolist())
                for cluster_idx_ch, p_value_ch in enumerate(cluster_p_values_ch):
                    time_indices_ch = np.where(clusters_ch[cluster_idx_ch])[0]
                    if time_indices_ch.size < min_timepoints:
                        continue

                    cluster_group_a_ch = band_power_group_a_ch[:, time_indices_ch].mean(axis=1)
                    cluster_group_b_ch = band_power_group_b_ch[:, time_indices_ch].mean(axis=1)

                    cohens_d_ch, d_ci_low_ch, d_ci_high_ch = (
                        compute_cohens_d_with_bootstrap_ci(
                            cluster_group_a_ch,
                            cluster_group_b_ch,
                            random_seed=rng_seed + channel_idx,
                            n_bootstrap=500,
                        )
                    )

                    channel_record = {
                        "subject": f"sub-{subject}",
                        "band": band_name,
                        "channel": channel_name,
                        "cluster_index": int(cluster_idx_ch),
                        "p_raw": float(p_value_ch),
                        "t_start": float(time_vec[time_indices_ch[0]]),
                        "t_end": float(time_vec[time_indices_ch[-1]]),
                        "duration_ms": float(
                            (time_vec[time_indices_ch[-1]] - time_vec[time_indices_ch[0]]) * 1000
                        ),
                        "n_timepoints": int(len(time_indices_ch)),
                        "t_stat_min": float(np.min(t_stat_ch[time_indices_ch])),
                        "t_stat_max": float(np.max(t_stat_ch[time_indices_ch])),
                        "t_stat_mean": float(np.mean(t_stat_ch[time_indices_ch])),
                        "cohens_d": cohens_d_ch,
                        "d_ci_low": d_ci_low_ch,
                        "d_ci_high": d_ci_high_ch,
                    }
                    channel_cluster_records.append(channel_record)

            if channel_cluster_records:
                _, q_values = fdr_bh_values(np.asarray(channel_pvals), alpha=alpha)
                for channel_record, q_value in zip(channel_cluster_records, q_values):
                    p_fdr_local = float(q_value) if np.isfinite(q_value) else np.nan
                    channel_record["p_fdr_local"] = p_fdr_local
                    channel_record["p_value"] = (
                        p_fdr_local if np.isfinite(p_fdr_local) else channel_record["p_raw"]
                    )
                    channel_record["fdr_reject_local"] = (
                        bool(np.isfinite(q_value) and q_value < alpha)
                    )

                n_significant_channels = sum(
                    1
                    for record in channel_cluster_records
                    if record.get("fdr_reject_local", False)
                )
                results[band_name] = {
                    "significant": n_significant_channels > 0,
                    "cluster_records": channel_cluster_records,
                    "times": time_vec,
                    "n_clusters_found": len(channel_cluster_records),
                    "n_significant": n_significant_channels,
                }
                cluster_p_values_channel = np.array(
                    [record.get("p_raw", 1.0) for record in channel_cluster_records]
                )
                band_summaries.append(
                    _create_band_summary_record(
                        subject=subject,
                        band_name=band_name,
                        fmin=fmin,
                        fmax=fmax,
                        condition_column=condition_column,
                        condition_labels=condition_labels,
                        condition_values=condition_values,
                        n_clusters_found=len(channel_cluster_records),
                        n_significant=n_significant_channels,
                        cluster_p_values=cluster_p_values_channel,
                        n_condition_a_trials=int(band_power_group_a.shape[0]),
                        n_condition_b_trials=int(band_power_group_b.shape[0]),
                        n_permutations=n_permutations,
                        alpha=alpha,
                        n_channels=n_channels,
                        n_timepoints=n_times,
                        mean_effect_size=np.nan,
                        max_effect_size=np.nan,
                    )
                )
            else:
                results[band_name] = {
                    "significant": False,
                    "cluster_records": [],
                    "n_clusters_found": 0,
                    "n_significant": 0,
                }
                band_summaries.append(
                    _create_band_summary_record(
                        subject=subject,
                        band_name=band_name,
                        fmin=fmin,
                        fmax=fmax,
                        condition_column=condition_column,
                        condition_labels=condition_labels,
                        condition_values=condition_values,
                        n_clusters_found=0,
                        n_significant=0,
                        cluster_p_values=np.array([]),
                        n_condition_a_trials=int(band_power_group_a.shape[0]),
                        n_condition_b_trials=int(band_power_group_b.shape[0]),
                        n_permutations=n_permutations,
                        alpha=alpha,
                        n_channels=n_channels,
                        n_timepoints=n_times,
                        mean_effect_size=np.nan,
                        max_effect_size=np.nan,
                    )
                )

    if band_cluster_refs:
        p_values_global = [
            record["p_value"]
            for record in band_cluster_refs
            if np.isfinite(record["p_value"])
        ]
        q_values_global = (
            fdr_bh(p_values_global, alpha=fdr_alpha, config=config)
            if p_values_global
            else []
        )
        for record, q_value_global in zip(band_cluster_refs, q_values_global):
            record["p_fdr_global"] = (
                float(q_value_global) if np.isfinite(q_value_global) else np.nan
            )
            record["fdr_reject_global"] = (
                bool(np.isfinite(q_value_global) and q_value_global < fdr_alpha)
            )

    for band_name, result in results.items():
        cluster_records = result.get("cluster_records", [])
        if not cluster_records:
            continue
        output_path = output_dir / f"cluster_results_{band_name}.tsv"
        write_tsv(pd.DataFrame(cluster_records), output_path)
        logger.info("Saved cluster results for %s to %s", band_name, output_path)

    if band_summaries:
        summary_df = pd.DataFrame(band_summaries)
        summary_path = output_dir / "cluster_summary.tsv"
        write_tsv(summary_df, summary_path)

        total_clusters_found = int(summary_df["n_clusters_found"].sum())
        total_significant = int(summary_df["n_significant"].sum())
        bands_with_significant = int((summary_df["n_significant"] > 0).sum())

        logger.info(
            "Cluster test summary: %d bands tested, %d total clusters found, "
            "%d significant across %d bands",
            len(band_summaries),
            total_clusters_found,
            total_significant,
            bands_with_significant,
        )

        if total_clusters_found > 0 and total_significant == 0:
            min_p_overall = float(summary_df["min_p_value"].min())
            logger.info(
                "No clusters reached significance (alpha=%.3f). "
                "Minimum p-value observed: %.4f. "
                "Consider increasing n_permutations for more stable p-value estimates.",
                alpha,
                min_p_overall,
            )

    return results


def _run_cluster_test_core(
    subject: str,
    epochs: "mne.Epochs",
    aligned_events: pd.DataFrame,
    output_dir: Path,
    config,
    logger: logging.Logger,
    n_perm: int,
) -> Optional[Dict[str, Any]]:
    """Core implementation for two-condition cluster test."""
    from eeg_pipeline.utils.analysis.tfr import restrict_epochs_to_roi

    if epochs is None or aligned_events is None:
        logger.warning("Cannot run cluster test: epochs or events unavailable.")
        return None

    if not epochs.preload:
        epochs.load_data()

    temporal_cfg = config.get("behavior_analysis.temporal", {}) or {}
    roi_selection = temporal_cfg.get("roi_selection")
    epochs_roi = restrict_epochs_to_roi(epochs, roi_selection, config, logger)

    condition_column_config = str(
        get_config_value(config, "behavior_analysis.cluster.condition_column", "") or ""
    ).strip()
    condition_column = (
        condition_column_config
        if condition_column_config and condition_column_config in aligned_events.columns
        else (
            get_condition_column_from_config(config, aligned_events)
            or get_binary_outcome_column_from_config(config, aligned_events)
        )
    )
    if condition_column is None or condition_column not in aligned_events.columns:
        logger.warning("Cluster condition column not found; skipping cluster test.")
        return None

    condition_values_config = (
        get_config_value(config, "behavior_analysis.cluster.condition_values", []) or []
    )
    condition_values_list: List[Any] = (
        list(condition_values_config)
        if isinstance(condition_values_config, (list, tuple))
        else [condition_values_config]
    )
    if len(condition_values_list) >= 2:
        value_a, value_b = condition_values_list[0], condition_values_list[1]
    else:
        value_a, value_b = 0, 1

    condition_series = aligned_events[condition_column]

    def _match_condition_values(
        series: pd.Series, val_a: Any, val_b: Any
    ) -> Tuple[pd.Series, pd.Series]:
        if series.dtype == object:
            series_normalized = series.astype(str).str.strip().str.lower()
            val_a_normalized = str(val_a).strip().lower()
            val_b_normalized = str(val_b).strip().lower()
            return (series_normalized == val_a_normalized), (
                series_normalized == val_b_normalized
            )

        series_numeric = pd.to_numeric(series, errors="coerce")
        try:
            val_a_numeric = float(val_a)
            val_b_numeric = float(val_b)
            return (series_numeric == val_a_numeric), (series_numeric == val_b_numeric)
        except (TypeError, ValueError):
            series_str = series.astype(str).str.strip()
            return (series_str == str(val_a).strip()), (series_str == str(val_b).strip())

    mask_group_a, mask_group_b = _match_condition_values(
        condition_series, value_a, value_b
    )
    keep_mask = (mask_group_a | mask_group_b) & condition_series.notna()
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        logger.warning(
            "Cluster condition split produced zero trials; column=%s values=%s/%s",
            condition_column,
            value_a,
            value_b,
        )
        return None

    epochs_roi = epochs_roi[keep_mask.to_numpy()]
    aligned_events = aligned_events.loc[keep_mask].reset_index(drop=True)
    mask_group_a = mask_group_a.loc[keep_mask].reset_index(drop=True)
    mask_group_b = mask_group_b.loc[keep_mask].reset_index(drop=True)

    if int(mask_group_a.sum()) < 2 or int(mask_group_b.sum()) < 2:
        logger.warning(
            "Insufficient trials (%s=%d, %s=%d); skipping cluster test.",
            str(value_a),
            int(mask_group_a.sum()),
            str(value_b),
            int(mask_group_b.sum()),
        )
        return None

    statistics_config = config.get("behavior_analysis.statistics", {})
    n_permutations_used = (
        n_perm if n_perm > 0 else int(statistics_config.get("n_permutations", 100))
    )
    alpha_used = float(
        statistics_config.get("sig_alpha", config.get("statistics.sig_alpha", 0.05))
    )
    # Local import to avoid circular dependency
    from eeg_pipeline.utils.analysis.tfr import get_tfr_config, get_bands_for_tfr
    _, max_freq_available, _, _, _, _ = get_tfr_config(config)
    bands = get_bands_for_tfr(max_freq_available=max_freq_available, config=config)

    return compute_two_condition_time_cluster_test(
        subject=subject,
        group_a_epochs=epochs_roi[mask_group_a.to_numpy()],
        group_b_epochs=epochs_roi[mask_group_b.to_numpy()],
        output_dir=output_dir,
        config=config,
        bands=bands,
        n_permutations=n_permutations_used,
        alpha=alpha_used,
        condition_column=str(condition_column),
        condition_values=(value_a, value_b),
        condition_labels=(str(value_a), str(value_b)),
    )


###################################################################
# 2D Cluster Correction (Time-Frequency)
###################################################################

from joblib import Parallel, delayed, cpu_count

from .correlation import compute_correlation

# Constants for 2D cluster correction
_MAX_RNG_SEED = 2**31
_MIN_PARALLEL_JOBS = 1
_MIN_PERMUTATIONS_FOR_PARALLEL = 10
_NUMERICAL_STABILITY_EPSILON = 1e-15
_DEFAULT_CLUSTER_STRUCTURE = np.ones((3, 3), dtype=int)


def _get_default_cluster_structure(config: Optional[Any] = None) -> np.ndarray:
    """Get default cluster structure from config or return default."""
    constants = get_statistics_constants(config)
    raw_structure = constants.get("cluster_structure_2d", None)
    if raw_structure is not None:
        return np.array(raw_structure, dtype=int)
    return _DEFAULT_CLUSTER_STRUCTURE.copy()


def compute_cluster_masses_2d(
    correlation_matrix: np.ndarray,
    pvalue_matrix: np.ndarray,
    cluster_alpha: float,
    cluster_forming_threshold: Optional[float] = None,
    cluster_structure: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute cluster masses from 2D correlation matrix."""
    from scipy.ndimage import label
    
    if cluster_structure is None:
        cluster_structure = _get_default_cluster_structure(config)
    
    finite_mask = np.isfinite(correlation_matrix)
    if cluster_forming_threshold is not None:
        significant_mask = finite_mask & (np.abs(correlation_matrix) >= cluster_forming_threshold)
    else:
        significant_mask = (
            finite_mask
            & np.isfinite(pvalue_matrix)
            & (pvalue_matrix < cluster_alpha)
        )
    
    labels, n_clusters = label(significant_mask, structure=cluster_structure)
    
    masses = {}
    for cluster_id in range(1, n_clusters + 1):
        cluster_region = (labels == cluster_id)
        if cluster_region.any():
            cluster_mass = np.nansum(np.abs(correlation_matrix[cluster_region]))
            masses[cluster_id] = float(cluster_mass)
    
    return labels, masses


def _permute_indices_2d(
    n_samples: int,
    rng_seed: int,
    groups: Optional[np.ndarray],
    *,
    scheme: str = "shuffle",
) -> np.ndarray:
    """Generate permuted indices (stateless for parallel use)."""
    rng = np.random.default_rng(rng_seed)
    groups_array = np.asarray(groups) if groups is not None else None

    # IMPORTANT scientific validity:
    # In repeated-trial paradigms, trial order often carries autocorrelation (habituation/drift).
    # Support circular-shift permutations (optionally within groups) to preserve
    # serial dependence when requested.
    from .permutation import permute_within_groups

    return permute_within_groups(
        n_samples,
        rng,
        groups=groups_array,
        min_group_size=1,
        scheme=scheme,
    )


def _single_permutation_threshold(
    perm_seed: int,
    n_samples: int,
    groups: Optional[np.ndarray],
    scheme: str,
    informative_bins: List[Tuple[int, int]],
    bin_data: np.ndarray,
    y_array: np.ndarray,
    residual_cache: Dict,
    min_valid_points: int,
    use_spearman: bool,
    covariate_count: int,
) -> float:
    """Single permutation for threshold derivation.
    
    Returns maximum absolute t-statistic across all bins for this permutation.
    """
    permuted_indices = _permute_indices_2d(
        n_samples,
        perm_seed,
        groups,
        scheme=scheme,
    )
    max_absolute_value = 0.0
    
    for frequency_idx, time_idx in informative_bins:
        t_statistic, _ = _compute_single_bin_corr(
            frequency_idx,
            time_idx,
            permuted_indices,
            residual_cache,
            bin_data,
            y_array,
            min_valid_points,
            use_spearman,
            covariate_count,
        )
        if np.isfinite(t_statistic):
            max_absolute_value = max(
                max_absolute_value,
                abs(float(t_statistic)),
            )
    
    return max_absolute_value


def _single_permutation_mass(
    perm_seed: int,
    n_samples: int,
    groups: Optional[np.ndarray],
    scheme: str,
    informative_bins: List[Tuple[int, int]],
    bin_data: np.ndarray,
    y_array: np.ndarray,
    residual_cache: Dict,
    correlations_shape: Tuple[int, ...],
    min_valid_points: int,
    use_spearman: bool,
    covariate_count: int,
    cluster_alpha: float,
    cluster_forming_threshold: float,
    cluster_structure: Optional[np.ndarray],
) -> float:
    """Single permutation for max mass computation."""
    permuted_indices = _permute_indices_2d(
        n_samples,
        perm_seed,
        groups,
        scheme=scheme,
    )
    permuted_correlations = np.full(correlations_shape, np.nan)
    permuted_pvalues = np.full(correlations_shape, np.nan)
    
    for frequency_idx, time_idx in informative_bins:
        t_statistic, p_value = _compute_single_bin_corr(
            frequency_idx,
            time_idx,
            permuted_indices,
            residual_cache,
            bin_data,
            y_array,
            min_valid_points,
            use_spearman,
            covariate_count,
        )
        permuted_correlations[frequency_idx, time_idx] = t_statistic
        permuted_pvalues[frequency_idx, time_idx] = p_value
    
    _, cluster_masses = compute_cluster_masses_2d(
        permuted_correlations,
        permuted_pvalues,
        cluster_alpha,
        cluster_forming_threshold,
        cluster_structure,
    )
    
    return max(cluster_masses.values()) if cluster_masses else 0.0


def _correlation_to_t_statistic(
    correlation: float,
    degrees_of_freedom: int,
) -> Tuple[float, float]:
    """Convert correlation coefficient to t-statistic and p-value."""
    from scipy import stats as scipy_stats
    
    if degrees_of_freedom <= 0 or not np.isfinite(correlation) or abs(correlation) >= 1:
        return np.nan, np.nan
    
    denominator = max(_NUMERICAL_STABILITY_EPSILON, 1 - correlation**2)
    t_statistic = correlation * np.sqrt(degrees_of_freedom / denominator)
    p_value = float(2 * scipy_stats.t.sf(np.abs(t_statistic), degrees_of_freedom))
    
    return t_statistic, p_value


def _compute_bin_correlation_with_residuals(
    frequency_idx: int,
    time_idx: int,
    permuted_indices: np.ndarray,
    residual_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, Dict[int, int]]],
    min_valid_points: int,
    use_spearman: bool,
    covariate_count: int,
) -> Tuple[float, float]:
    """Compute correlation for a bin using pre-computed residuals."""
    bin_key = (frequency_idx, time_idx)
    if bin_key not in residual_cache:
        return np.nan, np.nan
    
    x_residuals, y_residuals, index_map = residual_cache[bin_key]
    if x_residuals.size < min_valid_points:
        return np.nan, np.nan
    
    permuted_order = [index_map[i] for i in permuted_indices if i in index_map]
    if len(permuted_order) != y_residuals.size:
        return np.nan, np.nan
    
    y_permuted = y_residuals[permuted_order]
    correlation_method = "spearman" if use_spearman else "pearson"
    correlation, _ = compute_correlation(x_residuals, y_permuted, correlation_method)
    
    degrees_of_freedom = x_residuals.size - covariate_count - 2
    return _correlation_to_t_statistic(correlation, degrees_of_freedom)


def _compute_bin_correlation_direct(
    frequency_idx: int,
    time_idx: int,
    permuted_indices: np.ndarray,
    bin_data: np.ndarray,
    y_array: np.ndarray,
    min_valid_points: int,
    use_spearman: bool,
) -> Tuple[float, float]:
    """Compute correlation for a bin directly from raw data."""
    bin_values = bin_data[frequency_idx, time_idx, :]
    y_permuted = y_array[permuted_indices]
    
    valid_mask = np.isfinite(bin_values) & np.isfinite(y_permuted)
    n_valid = int(valid_mask.sum())
    
    if n_valid < min_valid_points:
        return np.nan, np.nan
    
    correlation_method = "spearman" if use_spearman else "pearson"
    correlation, _ = compute_correlation(
        bin_values[valid_mask],
        y_permuted[valid_mask],
        correlation_method,
    )
    
    degrees_of_freedom = n_valid - 2
    return _correlation_to_t_statistic(correlation, degrees_of_freedom)


def _compute_single_bin_corr(
    frequency_idx: int,
    time_idx: int,
    permuted_indices: np.ndarray,
    residual_cache: Dict,
    bin_data: np.ndarray,
    y_array: np.ndarray,
    min_valid_points: int,
    use_spearman: bool,
    covariate_count: int,
) -> Tuple[float, float]:
    """Compute correlation for a single bin with permuted y.
    
    Returns t-statistic and p-value (not raw correlation).
    """
    if residual_cache:
        return _compute_bin_correlation_with_residuals(
            frequency_idx,
            time_idx,
            permuted_indices,
            residual_cache,
            min_valid_points,
            use_spearman,
            covariate_count,
        )
    
    return _compute_bin_correlation_direct(
        frequency_idx,
        time_idx,
        permuted_indices,
        bin_data,
        y_array,
        min_valid_points,
        use_spearman,
    )


def _determine_parallel_jobs(n_jobs: int) -> int:
    """Determine number of parallel jobs to use."""
    if n_jobs == -1:
        return max(_MIN_PARALLEL_JOBS, cpu_count() - 1)
    return max(_MIN_PARALLEL_JOBS, n_jobs)


def _should_use_parallel(n_jobs: int, n_iterations: int) -> bool:
    """Determine if parallel execution should be used."""
    return n_jobs > 1 and n_iterations > _MIN_PERMUTATIONS_FOR_PARALLEL


def _run_parallel_or_sequential(
    func,
    n_iterations: int,
    base_seed: int,
    n_jobs: int,
    **func_kwargs,
) -> List[Any]:
    """Run function in parallel or sequentially based on configuration."""
    if _should_use_parallel(n_jobs, n_iterations):
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(func)(base_seed + i, **func_kwargs)
            for i in range(n_iterations)
        )
    return [func(base_seed + i, **func_kwargs) for i in range(n_iterations)]


def _build_residual_cache(
    bin_data: np.ndarray,
    informative_bins: List[Tuple[int, int]],
    y_array: np.ndarray,
    covariates_matrix: np.ndarray,
    min_valid_points: int,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, Dict[int, int]]]:
    """Build cache of residuals for partial correlation computation."""
    residual_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, Dict[int, int]]] = {}
    covariate_count = covariates_matrix.shape[1]
    covariates = np.asarray(covariates_matrix, dtype=float)
    
    for frequency_idx, time_idx in informative_bins:
        bin_values = bin_data[frequency_idx, time_idx, :]
        valid_mask = (
            np.isfinite(bin_values)
            & np.isfinite(y_array)
            & np.all(np.isfinite(covariates), axis=1)
        )
        
        min_required = max(min_valid_points, covariate_count + 1)
        if valid_mask.sum() < min_required:
            continue
        
        design_matrix = np.column_stack([
            np.ones(valid_mask.sum()),
            covariates[valid_mask],
        ])
        
        try:
            beta_x = np.linalg.lstsq(design_matrix, bin_values[valid_mask], rcond=None)[0]
            beta_y = np.linalg.lstsq(design_matrix, y_array[valid_mask], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        
        x_residuals = bin_values[valid_mask] - design_matrix @ beta_x
        y_residuals = y_array[valid_mask] - design_matrix @ beta_y
        
        valid_indices = np.where(valid_mask)[0]
        index_map = {int(idx): pos for pos, idx in enumerate(valid_indices)}
        residual_cache[(frequency_idx, time_idx)] = (x_residuals, y_residuals, index_map)
    
    return residual_cache


def _derive_cluster_forming_threshold(
    n_samples: int,
    groups: Optional[np.ndarray],
    scheme: str,
    informative_bins: List[Tuple[int, int]],
    bin_data: np.ndarray,
    y_array: np.ndarray,
    residual_cache: Dict,
    min_valid_points: int,
    use_spearman: bool,
    covariate_count: int,
    n_cluster_perm: int,
    cluster_alpha: float,
    base_seed: int,
    n_jobs: int,
) -> float:
    """Derive cluster forming threshold from permutation distribution."""
    max_absolute_values = _run_parallel_or_sequential(
        _single_permutation_threshold,
        n_cluster_perm,
        base_seed,
        n_jobs,
        n_samples=n_samples,
        groups=groups,
        scheme=scheme,
        informative_bins=informative_bins,
        bin_data=bin_data,
        y_array=y_array,
        residual_cache=residual_cache,
        min_valid_points=min_valid_points,
        use_spearman=use_spearman,
        covariate_count=covariate_count,
    )
    
    if not max_absolute_values:
        return 0.0
    
    percentile = 100 * (1 - cluster_alpha)
    return float(np.nanpercentile(max_absolute_values, percentile))


def compute_permutation_max_masses(
    bin_data: np.ndarray,
    informative_bins: List[Tuple[int, int]],
    y_array: np.ndarray,
    correlations_shape: Tuple[int, ...],
    cluster_alpha: float,
    min_valid_points: int,
    use_spearman: bool,
    n_cluster_perm: int,
    cluster_rng: np.random.Generator,
    cluster_structure: Optional[np.ndarray] = None,
    covariates_matrix: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    cluster_forming_threshold: Optional[float] = None,
    n_jobs: int = -1,
    *,
    scheme: str = "shuffle",
) -> Tuple[List[float], float]:
    """Compute permutation distribution of max cluster masses.
    
    Uses parallel processing with loky backend for speed.
    """
    if n_cluster_perm <= 0:
        return [], cluster_forming_threshold or 0.0
    
    n_jobs_actual = _determine_parallel_jobs(n_jobs)
    groups_array = np.asarray(groups) if groups is not None else None
    n_samples = len(y_array)

    scheme = str(scheme or "shuffle").strip().lower()
    if scheme not in {"shuffle", "circular_shift"}:
        scheme = "shuffle"
    
    residual_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, Dict[int, int]]] = {}
    covariate_count = 0
    if covariates_matrix is not None:
        covariate_count = covariates_matrix.shape[1]
        residual_cache = _build_residual_cache(
            bin_data,
            informative_bins,
            y_array,
            covariates_matrix,
            min_valid_points,
        )
    
    base_seed = int(cluster_rng.integers(0, _MAX_RNG_SEED))
    
    if cluster_forming_threshold is None:
        cluster_forming_threshold = _derive_cluster_forming_threshold(
            n_samples,
            groups_array,
            scheme,
            informative_bins,
            bin_data,
            y_array,
            residual_cache,
            min_valid_points,
            use_spearman,
            covariate_count,
            n_cluster_perm,
            cluster_alpha,
            base_seed,
            n_jobs_actual,
        )
    
    offset_seed = base_seed + n_cluster_perm
    permutation_max_masses = _run_parallel_or_sequential(
        _single_permutation_mass,
        n_cluster_perm,
        offset_seed,
        n_jobs_actual,
        n_samples=n_samples,
        groups=groups_array,
        scheme=scheme,
        informative_bins=informative_bins,
        bin_data=bin_data,
        y_array=y_array,
        residual_cache=residual_cache,
        correlations_shape=correlations_shape,
        min_valid_points=min_valid_points,
        use_spearman=use_spearman,
        covariate_count=covariate_count,
        cluster_alpha=cluster_alpha,
        cluster_forming_threshold=cluster_forming_threshold,
        cluster_structure=cluster_structure,
    )
    
    return permutation_max_masses, cluster_forming_threshold


def compute_cluster_pvalues_2d(
    cluster_labels: np.ndarray,
    cluster_masses: Dict[int, float],
    perm_max_masses: List[float],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Compute cluster p-values from permutation distribution."""
    pvalues = np.full_like(cluster_labels, np.nan, dtype=float)
    significant_mask = np.zeros_like(cluster_labels, dtype=bool)
    records = []
    
    if not perm_max_masses:
        for cluster_id, mass in cluster_masses.items():
            cluster_region = (cluster_labels == cluster_id)
            records.append({
                "cluster_id": int(cluster_id),
                "mass": mass,
                "size": int(cluster_region.sum()),
                "p_value": np.nan,
            })
        return pvalues, significant_mask, records
    
    denominator = len(perm_max_masses) + 1
    permutation_array = np.asarray(perm_max_masses)
    
    for cluster_id, cluster_mass in cluster_masses.items():
        cluster_region = (cluster_labels == cluster_id)
        cluster_size = int(cluster_region.sum())
        
        n_exceeding = np.sum(permutation_array >= cluster_mass)
        p_value = (n_exceeding + 1) / denominator
        
        pvalues[cluster_region] = p_value
        if p_value <= alpha:
            significant_mask[cluster_region] = True
        
        records.append({
            "cluster_id": int(cluster_id),
            "mass": float(cluster_mass),
            "size": cluster_size,
            "p_value": float(p_value),
        })
    
    return pvalues, significant_mask, records


def compute_cluster_correction_2d(
    correlations: np.ndarray,
    p_values: np.ndarray,
    bin_data: np.ndarray,
    informative_bins: List[Tuple[int, int]],
    y_array: np.ndarray,
    cluster_alpha: float,
    n_cluster_perm: int,
    alpha: float,
    min_valid_points: int,
    use_spearman: bool,
    cluster_rng: np.random.Generator,
    cluster_structure: Optional[np.ndarray] = None,
    covariates_matrix: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    cluster_forming_threshold: Optional[float] = None,
    n_jobs: int = -1,
    *,
    config: Optional[Any] = None,
    scheme: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[float], float]:
    """Full 2D cluster correction pipeline."""
    if not informative_bins:
        empty_labels = np.zeros_like(correlations, dtype=int)
        empty_pvalues = np.full_like(correlations, np.nan)
        empty_mask = np.zeros_like(correlations, dtype=bool)
        default_threshold = float(cluster_forming_threshold or 0.0)
        return empty_labels, empty_pvalues, empty_mask, [], [], default_threshold
    
    if scheme is None:
        scheme = str(get_config_value(config, "behavior_analysis.permutation.scheme", "shuffle")).strip().lower()

    permutation_max_masses, derived_threshold = compute_permutation_max_masses(
        bin_data,
        informative_bins,
        y_array,
        correlations.shape,
        cluster_alpha,
        min_valid_points,
        use_spearman,
        n_cluster_perm,
        cluster_rng,
        cluster_structure,
        covariates_matrix,
        groups,
        cluster_forming_threshold,
        n_jobs,
        scheme=scheme,
    )
    
    final_threshold = (
        cluster_forming_threshold
        if cluster_forming_threshold is not None
        else derived_threshold
    )
    
    observed_labels, cluster_masses = compute_cluster_masses_2d(
        correlations,
        p_values,
        cluster_alpha,
        final_threshold,
        cluster_structure,
    )
    
    if not cluster_masses:
        empty_pvalues = np.full_like(correlations, np.nan)
        for row_idx, col_idx in informative_bins:
            empty_pvalues[row_idx, col_idx] = 1.0
        empty_mask = np.zeros_like(correlations, dtype=bool)
        return (
            observed_labels,
            empty_pvalues,
            empty_mask,
            [],
            permutation_max_masses,
            final_threshold,
        )
    
    pvalues, significant_mask, records = compute_cluster_pvalues_2d(
        observed_labels,
        cluster_masses,
        permutation_max_masses,
        alpha,
    )
    for row_idx, col_idx in informative_bins:
        if not np.isfinite(pvalues[row_idx, col_idx]):
            pvalues[row_idx, col_idx] = 1.0
    
    return (
        observed_labels,
        pvalues,
        significant_mask,
        records,
        permutation_max_masses,
        final_threshold,
    )
