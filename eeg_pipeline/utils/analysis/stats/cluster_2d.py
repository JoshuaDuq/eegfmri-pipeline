"""
2D Cluster Correction
=====================

Cluster-based permutation tests for 2D (time-frequency) data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label

from .base import get_statistics_constants
from .correlation import compute_correlation


def compute_cluster_masses_2d(
    correlation_matrix: np.ndarray,
    pvalue_matrix: np.ndarray,
    cluster_alpha: float,
    cluster_forming_threshold: Optional[float] = None,
    cluster_structure: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute cluster masses from 2D correlation matrix."""
    if cluster_structure is None:
        try:
            constants = get_statistics_constants(config)
            cluster_structure = np.array(constants.get("cluster_structure_2d", [[1,1,1],[1,1,1],[1,1,1]]))
        except:
            cluster_structure = np.ones((3, 3))
    
    finite = np.isfinite(correlation_matrix)
    if cluster_forming_threshold is not None:
        sig = finite & (np.abs(correlation_matrix) >= cluster_forming_threshold)
    else:
        sig = finite & np.isfinite(pvalue_matrix) & (pvalue_matrix < cluster_alpha)
    
    labels, n_clusters = label(sig, structure=cluster_structure)
    
    masses = {}
    for cid in range(1, n_clusters + 1):
        region = (labels == cid)
        if region.any():
            masses[cid] = float(np.nansum(np.abs(correlation_matrix[region])))
    
    return labels, masses


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
) -> Tuple[List[float], float]:
    """Compute permutation distribution of max cluster masses."""
    perm_max = []
    if n_cluster_perm <= 0:
        return perm_max, cluster_forming_threshold or 0.0
    
    def _permute(n, rng, grps):
        if grps is None:
            return rng.permutation(n)
        perm = []
        for g in np.unique(grps):
            idx = np.where(grps == g)[0]
            perm.extend(idx[rng.permutation(len(idx))])
        return np.asarray(perm)
    
    # Build residual cache for partial correlation
    resid_cache = {}
    cov_count = covariates_matrix.shape[1] if covariates_matrix is not None else 0
    if covariates_matrix is not None:
        cov = np.asarray(covariates_matrix, dtype=float)
        for fi, ti in informative_bins:
            bv = bin_data[fi, ti, :]
            mask = np.isfinite(bv) & np.isfinite(y_array) & np.all(np.isfinite(cov), axis=1)
            req = max(min_valid_points, cov_count + 1)
            if mask.sum() < req:
                continue
            X = np.column_stack([np.ones(mask.sum()), cov[mask]])
            try:
                beta_x = np.linalg.lstsq(X, bv[mask], rcond=None)[0]
                beta_y = np.linalg.lstsq(X, y_array[mask], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            x_res = bv[mask] - X @ beta_x
            y_res = y_array[mask] - X @ beta_y
            idx_map = {int(i): pos for pos, i in enumerate(np.where(mask)[0])}
            resid_cache[(fi, ti)] = (x_res, y_res, idx_map)
    
    def _corr(fi, ti, perm_idx):
        if resid_cache:
            if (fi, ti) not in resid_cache:
                return np.nan, np.nan
            x_res, y_res, idx_map = resid_cache[(fi, ti)]
            if x_res.size < min_valid_points:
                return np.nan, np.nan
            order = [idx_map[i] for i in perm_idx if i in idx_map]
            if len(order) != y_res.size:
                return np.nan, np.nan
            y_perm = y_res[order]
            r, _ = compute_correlation(x_res, y_perm, "spearman" if use_spearman else "pearson")
            dof = x_res.size - cov_count - 2
            if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
                return np.nan, np.nan
            t = r * np.sqrt(dof / max(1e-15, 1 - r**2))
            p = float(2 * stats.t.sf(np.abs(t), dof))
            return t, p
        
        bv = bin_data[fi, ti, :]
        yv = y_array[perm_idx]
        mask = np.isfinite(bv) & np.isfinite(yv)
        n = int(mask.sum())
        if n < min_valid_points:
            return np.nan, np.nan
        r, _ = compute_correlation(bv[mask], yv[mask], "spearman" if use_spearman else "pearson")
        dof = n - 2
        if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
            return np.nan, np.nan
        t = r * np.sqrt(dof / max(1e-15, 1 - r**2))
        p = float(2 * stats.t.sf(np.abs(t), dof))
        return t, p
    
    template = np.arange(len(y_array))
    grps = np.asarray(groups) if groups is not None else None
    
    # Derive threshold if needed
    if cluster_forming_threshold is None:
        max_abs = []
        for _ in range(n_cluster_perm):
            perm = _permute(len(template), cluster_rng, grps)
            ma = 0.0
            for fi, ti in informative_bins:
                r, _ = _corr(fi, ti, perm)
                if np.isfinite(r):
                    ma = max(ma, abs(float(r)))
            max_abs.append(ma)
        cluster_forming_threshold = float(np.nanpercentile(max_abs, 100 * (1 - cluster_alpha))) if max_abs else 0.0
    
    # Main permutation loop
    for _ in range(n_cluster_perm):
        perm_corr = np.full(correlations_shape, np.nan)
        perm_p = np.full(correlations_shape, np.nan)
        perm_idx = _permute(len(template), cluster_rng, grps)
        
        for fi, ti in informative_bins:
            r, p = _corr(fi, ti, perm_idx)
            perm_corr[fi, ti] = r
            perm_p[fi, ti] = p
        
        _, masses = compute_cluster_masses_2d(perm_corr, perm_p, cluster_alpha, cluster_forming_threshold, cluster_structure)
        perm_max.append(max(masses.values()) if masses else 0.0)
    
    return perm_max, cluster_forming_threshold


def compute_cluster_pvalues(
    cluster_labels: np.ndarray,
    cluster_masses: Dict[int, float],
    perm_max_masses: List[float],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Compute cluster p-values from permutation distribution."""
    pvals = np.full_like(cluster_labels, np.nan, dtype=float)
    sig_mask = np.zeros_like(cluster_labels, dtype=bool)
    records = []
    
    if not perm_max_masses:
        for cid, mass in cluster_masses.items():
            region = (cluster_labels == cid)
            records.append({"cluster_id": int(cid), "mass": mass, "size": int(region.sum()), "p_value": np.nan})
        return pvals, sig_mask, records
    
    denom = len(perm_max_masses) + 1
    perm_arr = np.asarray(perm_max_masses)
    
    for cid, mass in cluster_masses.items():
        region = (cluster_labels == cid)
        size = int(region.sum())
        p = (np.sum(perm_arr >= mass) + 1) / denom
        pvals[region] = p
        if p <= alpha:
            sig_mask[region] = True
        records.append({"cluster_id": int(cid), "mass": float(mass), "size": size, "p_value": float(p)})
    
    return pvals, sig_mask, records


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[float], float]:
    """Full 2D cluster correction pipeline."""
    labels = np.zeros_like(correlations, dtype=int)
    pvals = np.full_like(correlations, np.nan)
    sig_mask = np.zeros_like(correlations, dtype=bool)
    records = []
    
    if not informative_bins:
        return labels, pvals, sig_mask, records, [], float(cluster_forming_threshold or 0.0)
    
    perm_max, thresh = compute_permutation_max_masses(
        bin_data, informative_bins, y_array, correlations.shape, cluster_alpha, min_valid_points,
        use_spearman, n_cluster_perm, cluster_rng, cluster_structure,
        covariates_matrix, groups, cluster_forming_threshold,
    )
    cluster_forming_threshold = cluster_forming_threshold if cluster_forming_threshold is not None else thresh
    
    labels_obs, masses = compute_cluster_masses_2d(correlations, p_values, cluster_alpha, cluster_forming_threshold, cluster_structure)
    if not masses:
        return labels_obs, pvals, sig_mask, records, perm_max, cluster_forming_threshold
    
    labels = labels_obs
    pvals, sig_mask, records = compute_cluster_pvalues(labels_obs, masses, perm_max, alpha)
    
    return labels, pvals, sig_mask, records, perm_max, cluster_forming_threshold


# ============================================================================
# 1D Cluster Correction (for topomaps)
# ============================================================================

def compute_cluster_masses_1d(
    correlation_vector: np.ndarray,
    pvalue_vector: np.ndarray,
    cluster_alpha: float,
    ch_to_eeg_idx: Dict[int, int],
    eeg_picks: np.ndarray,
    adjacency: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute 1D cluster masses using EEG adjacency."""
    from scipy.sparse.csgraph import connected_components
    
    sig = np.isfinite(correlation_vector) & np.isfinite(pvalue_vector) & (pvalue_vector < cluster_alpha)
    if not np.any(sig):
        return np.zeros(len(correlation_vector), dtype=int), {}
    
    eeg_mask = np.zeros(len(eeg_picks), dtype=bool)
    for ch_idx, eeg_idx in ch_to_eeg_idx.items():
        if sig[ch_idx]:
            eeg_mask[eeg_idx] = True
    
    if not np.any(eeg_mask):
        return np.zeros(len(correlation_vector), dtype=int), {}
    
    eeg_indices = np.where(eeg_mask)[0]
    if len(eeg_indices) < 2:
        return np.zeros(len(correlation_vector), dtype=int), {}
    
    adj_sub = adjacency[eeg_indices, :][:, eeg_indices]
    n_comp, eeg_labels = connected_components(csgraph=adj_sub, directed=False, return_labels=True)
    
    full_labels = np.zeros(len(correlation_vector), dtype=int)
    eeg_to_ch = {eeg_idx: ch_idx for ch_idx, eeg_idx in ch_to_eeg_idx.items()}
    for local, global_eeg in enumerate(eeg_indices):
        if global_eeg in eeg_to_ch:
            full_labels[eeg_to_ch[global_eeg]] = eeg_labels[local] + 1
    
    masses = {}
    for cid in range(1, n_comp + 1):
        region = (full_labels == cid)
        if region.any():
            masses[cid] = float(np.nansum(np.abs(correlation_vector[region])))
    
    return full_labels, masses


def compute_topomap_permutation_masses(
    channel_data: np.ndarray,
    temp_series: pd.Series,
    n_channels: int,
    n_cluster_perm: int,
    cluster_alpha: float,
    min_valid_points: int,
    use_spearman: bool,
    ch_to_eeg_idx: Dict[int, int],
    eeg_picks: np.ndarray,
    adjacency: np.ndarray,
    cluster_rng: np.random.Generator,
) -> List[float]:
    """Compute permutation max masses for topomap cluster correction."""
    perm_max = []
    if n_cluster_perm <= 0:
        return perm_max
    
    temp_arr = temp_series.to_numpy(dtype=float)
    
    for _ in range(n_cluster_perm):
        temp_perm = cluster_rng.permutation(temp_arr)
        corrs = np.full(n_channels, np.nan)
        pvals = np.full(n_channels, np.nan)
        
        for ch in range(n_channels):
            ch_vec = channel_data[ch, :]
            mask = np.isfinite(ch_vec) & np.isfinite(temp_perm)
            if mask.sum() >= min_valid_points:
                r, p = compute_correlation(ch_vec[mask], temp_perm[mask], "spearman" if use_spearman else "pearson")
                corrs[ch], pvals[ch] = r, p
        
        _, masses = compute_cluster_masses_1d(corrs, pvals, cluster_alpha, ch_to_eeg_idx, eeg_picks, adjacency)
        perm_max.append(max(masses.values()) if masses else 0.0)
    
    return perm_max


def compute_cluster_pvalues_1d(
    cluster_labels: np.ndarray,
    cluster_masses: Dict[int, float],
    perm_max_masses: List[float],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Compute 1D cluster p-values (alias for compute_cluster_pvalues)."""
    return compute_cluster_pvalues(cluster_labels, cluster_masses, perm_max_masses, alpha)


