"""
2D Cluster Correction
=====================

Cluster-based permutation tests for 2D (time-frequency) data.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label
from joblib import Parallel, delayed, cpu_count

from .base import get_statistics_constants
from .correlation import compute_correlation


# Constants
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


def _permute_indices(
    n_samples: int,
    rng_seed: int,
    groups: Optional[np.ndarray],
) -> np.ndarray:
    """Generate permuted indices (stateless for parallel use)."""
    rng = np.random.default_rng(rng_seed)
    if groups is None:
        return rng.permutation(n_samples)
    
    permuted_indices = []
    unique_groups = np.unique(groups)
    for group in unique_groups:
        group_indices = np.where(groups == group)[0]
        permuted_group = rng.permutation(len(group_indices))
        permuted_indices.extend(group_indices[permuted_group])
    
    return np.asarray(permuted_indices)


def _single_permutation_threshold(
    perm_seed: int,
    n_samples: int,
    groups: Optional[np.ndarray],
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
    permuted_indices = _permute_indices(n_samples, perm_seed, groups)
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
    permuted_indices = _permute_indices(n_samples, perm_seed, groups)
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
    if degrees_of_freedom <= 0 or not np.isfinite(correlation) or abs(correlation) >= 1:
        return np.nan, np.nan
    
    denominator = max(_NUMERICAL_STABILITY_EPSILON, 1 - correlation**2)
    t_statistic = correlation * np.sqrt(degrees_of_freedom / denominator)
    p_value = float(2 * stats.t.sf(np.abs(t_statistic), degrees_of_freedom))
    
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
    func: Callable,
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
) -> Tuple[List[float], float]:
    """Compute permutation distribution of max cluster masses.
    
    Uses parallel processing with loky backend for speed.
    """
    if n_cluster_perm <= 0:
        return [], cluster_forming_threshold or 0.0
    
    n_jobs_actual = _determine_parallel_jobs(n_jobs)
    groups_array = np.asarray(groups) if groups is not None else None
    n_samples = len(y_array)
    
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



def compute_cluster_pvalues(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[float], float]:
    """Full 2D cluster correction pipeline."""
    if not informative_bins:
        empty_labels = np.zeros_like(correlations, dtype=int)
        empty_pvalues = np.full_like(correlations, np.nan)
        empty_mask = np.zeros_like(correlations, dtype=bool)
        default_threshold = float(cluster_forming_threshold or 0.0)
        return empty_labels, empty_pvalues, empty_mask, [], [], default_threshold
    
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
        empty_mask = np.zeros_like(correlations, dtype=bool)
        return (
            observed_labels,
            empty_pvalues,
            empty_mask,
            [],
            permutation_max_masses,
            final_threshold,
        )
    
    pvalues, significant_mask, records = compute_cluster_pvalues(
        observed_labels,
        cluster_masses,
        permutation_max_masses,
        alpha,
    )
    
    return (
        observed_labels,
        pvalues,
        significant_mask,
        records,
        permutation_max_masses,
        final_threshold,
    )


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
    
    significant_mask = (
        np.isfinite(correlation_vector)
        & np.isfinite(pvalue_vector)
        & (pvalue_vector < cluster_alpha)
    )
    
    if not np.any(significant_mask):
        empty_labels = np.zeros(len(correlation_vector), dtype=int)
        return empty_labels, {}
    
    eeg_significant_mask = np.zeros(len(eeg_picks), dtype=bool)
    for channel_idx, eeg_idx in ch_to_eeg_idx.items():
        if significant_mask[channel_idx]:
            eeg_significant_mask[eeg_idx] = True
    
    if not np.any(eeg_significant_mask):
        empty_labels = np.zeros(len(correlation_vector), dtype=int)
        return empty_labels, {}
    
    eeg_significant_indices = np.where(eeg_significant_mask)[0]
    if len(eeg_significant_indices) < 2:
        empty_labels = np.zeros(len(correlation_vector), dtype=int)
        return empty_labels, {}
    
    adjacency_subset = adjacency[eeg_significant_indices, :][:, eeg_significant_indices]
    n_components, eeg_labels = connected_components(
        csgraph=adjacency_subset,
        directed=False,
        return_labels=True,
    )
    
    full_labels = np.zeros(len(correlation_vector), dtype=int)
    eeg_to_channel = {eeg_idx: ch_idx for ch_idx, eeg_idx in ch_to_eeg_idx.items()}
    
    for local_idx, global_eeg_idx in enumerate(eeg_significant_indices):
        if global_eeg_idx in eeg_to_channel:
            channel_idx = eeg_to_channel[global_eeg_idx]
            full_labels[channel_idx] = eeg_labels[local_idx] + 1
    
    cluster_masses = {}
    for cluster_id in range(1, n_components + 1):
        cluster_region = (full_labels == cluster_id)
        if cluster_region.any():
            cluster_mass = np.nansum(np.abs(correlation_vector[cluster_region]))
            cluster_masses[cluster_id] = float(cluster_mass)
    
    return full_labels, cluster_masses


def _single_topomap_permutation(
    perm_seed: int,
    channel_data: np.ndarray,
    temp_arr: np.ndarray,
    n_channels: int,
    cluster_alpha: float,
    min_valid_points: int,
    use_spearman: bool,
    ch_to_eeg_idx: Dict[int, int],
    eeg_picks: np.ndarray,
    adjacency: np.ndarray,
) -> float:
    """Single permutation for topomap cluster correction."""
    rng = np.random.default_rng(perm_seed)
    temp_permuted = rng.permutation(temp_arr)
    correlations = np.full(n_channels, np.nan)
    pvalues = np.full(n_channels, np.nan)
    
    correlation_method = "spearman" if use_spearman else "pearson"
    
    for channel_idx in range(n_channels):
        channel_values = channel_data[channel_idx, :]
        valid_mask = np.isfinite(channel_values) & np.isfinite(temp_permuted)
        
        if valid_mask.sum() >= min_valid_points:
            correlation, p_value = compute_correlation(
                channel_values[valid_mask],
                temp_permuted[valid_mask],
                correlation_method,
            )
            correlations[channel_idx] = correlation
            pvalues[channel_idx] = p_value
    
    _, cluster_masses = compute_cluster_masses_1d(
        correlations,
        pvalues,
        cluster_alpha,
        ch_to_eeg_idx,
        eeg_picks,
        adjacency,
    )
    
    return max(cluster_masses.values()) if cluster_masses else 0.0


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
    n_jobs: int = -1,
) -> List[float]:
    """Compute permutation max masses for topomap cluster correction.
    
    Uses parallel processing with loky backend for speed.
    """
    if n_cluster_perm <= 0:
        return []
    
    n_jobs_actual = _determine_parallel_jobs(n_jobs)
    temp_array = temp_series.to_numpy(dtype=float)
    base_seed = int(cluster_rng.integers(0, _MAX_RNG_SEED))
    
    permutation_max_masses = _run_parallel_or_sequential(
        _single_topomap_permutation,
        n_cluster_perm,
        base_seed,
        n_jobs_actual,
        channel_data=channel_data,
        temp_arr=temp_array,
        n_channels=n_channels,
        cluster_alpha=cluster_alpha,
        min_valid_points=min_valid_points,
        use_spearman=use_spearman,
        ch_to_eeg_idx=ch_to_eeg_idx,
        eeg_picks=eeg_picks,
        adjacency=adjacency,
    )
    
    return permutation_max_masses


def compute_cluster_pvalues_1d(
    cluster_labels: np.ndarray,
    cluster_masses: Dict[int, float],
    perm_max_masses: List[float],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Compute 1D cluster p-values (alias for compute_cluster_pvalues)."""
    return compute_cluster_pvalues(cluster_labels, cluster_masses, perm_max_masses, alpha)


