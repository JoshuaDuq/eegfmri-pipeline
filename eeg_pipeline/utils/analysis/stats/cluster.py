"""
Cluster Statistics
==================

EEG cluster-based permutation tests and adjacency utilities.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mne

from .base import get_statistics_constants, get_fdr_alpha, get_config_value, ensure_config

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

