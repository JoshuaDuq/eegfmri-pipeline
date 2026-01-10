"""Parallel utilities.

Centralized parallelization helpers (joblib-based) used across analysis modules.

This module replaces the previous location `eeg_pipeline.analysis.behavior.parallel`.
A compatibility wrapper remains at that import path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

T = TypeVar("T")

# Constants
_MIN_PARALLEL_JOBS = 1
_DEFAULT_CPU_RESERVE = 1
_MIN_FEATURES_FOR_PARALLEL_CORRELATION = 10
_MIN_FEATURES_FOR_PARALLEL_CONDITION = 10
_MIN_FEATURES_FOR_PARALLEL_REGRESSION = 10
_MIN_FEATURES_FOR_PARALLEL_STABILITY = 10
_MIN_FEATURES_FOR_PARALLEL_INFLUENCE = 5
_MIN_FEATURE_TYPES_FOR_PARALLEL = 2
_NUMERIC_TOLERANCE = 1e-12
_SPLIT_HALF_RELIABILITY_SPLITS = 50
_MIN_FEATURES_FOR_BATCH_CONDITION = 100  # Use vectorized batch for 100+ features


def _normalize_n_jobs(n_jobs: int) -> int:
    """Normalize n_jobs value to valid positive integer."""
    if n_jobs == -1:
        return max(_MIN_PARALLEL_JOBS, cpu_count() - _DEFAULT_CPU_RESERVE)
    return max(_MIN_PARALLEL_JOBS, n_jobs)


def get_n_jobs(
    config: Optional[Any] = None,
    default: int = -1,
    config_path: str = "behavior_analysis.n_jobs",
) -> int:
    """Get number of parallel jobs from config or environment."""
    n_jobs = default

    if config is not None:
        from eeg_pipeline.utils.config.loader import get_config_value

        n_jobs = int(get_config_value(config, config_path, default))

    env_jobs = os.environ.get("EEG_PIPELINE_N_JOBS")
    if env_jobs is not None:
        try:
            n_jobs = int(env_jobs)
        except ValueError:
            pass

    return _normalize_n_jobs(n_jobs)


def _should_use_parallel(n_jobs: int, num_items: int, min_items: int = 1) -> bool:
    """Determine if parallel execution should be used."""
    normalized_jobs = _normalize_n_jobs(n_jobs)
    return normalized_jobs > 1 and num_items >= min_items


def _execute_parallel(
    func: Callable[..., T],
    items: List[Any],
    n_jobs: int,
    backend: str = "loky",
    verbose: int = 0,
    **kwargs: Any,
) -> List[T]:
    """Execute function on items in parallel using joblib."""
    normalized_jobs = _normalize_n_jobs(n_jobs)
    return Parallel(n_jobs=normalized_jobs, backend=backend, verbose=verbose)(
        delayed(func)(item, **kwargs) for item in items
    )


def parallel_map(
    func: Callable[..., T],
    items: List[Any],
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
    desc: str = "",
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> List[T]:
    """Apply function to items in parallel."""
    if not items:
        return []

    if not _should_use_parallel(n_jobs, len(items)):
        return [func(item, **kwargs) for item in items]

    if logger and desc:
        normalized_jobs = _normalize_n_jobs(n_jobs)
        logger.debug(f"Parallel {desc}: {len(items)} items, {normalized_jobs} jobs")

    return _execute_parallel(func, items, n_jobs, backend, verbose, **kwargs)


def parallel_correlate_features(
    feature_columns: List[str],
    feature_df: pd.DataFrame,
    target_arr: np.ndarray,
    temp_arr: Optional[np.ndarray],
    order_arr: Optional[np.ndarray],
    method: str,
    min_samples: int,
    compute_reliability: bool,
    rng_seed: int,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Correlate multiple features with target.
    
    Uses vectorized batch computation when partial correlations and reliability
    are not needed. Falls back to per-feature computation otherwise.
    """
    if not feature_columns:
        return []
    
    n_features = len(feature_columns)
    
    # Use vectorized batch for simple correlations (no partials, no reliability)
    needs_partials = temp_arr is not None or order_arr is not None
    needs_reliability = compute_reliability
    
    if n_features >= _MIN_FEATURES_FOR_BATCH_CONDITION and not needs_partials and not needs_reliability:
        from eeg_pipeline.utils.analysis.stats.correlation import compute_batch_correlations
        return compute_batch_correlations(
            feature_columns=feature_columns,
            feature_df=feature_df,
            target_arr=target_arr,
            method=method,
            min_samples=min_samples,
            logger=logger,
        )
    
    # Fall back to per-feature computation for partial correlations or reliability
    if logger:
        logger.info(f"Correlations: {n_features} features (per-feature mode, partials={needs_partials}, reliability={needs_reliability})")
    
    if not _should_use_parallel(n_jobs, n_features, _MIN_FEATURES_FOR_PARALLEL_CORRELATION):
        results = []
        for col in feature_columns:
            result = _correlate_single_column(
                col,
                feature_df,
                target_arr,
                temp_arr,
                order_arr,
                method,
                min_samples,
                compute_reliability,
                rng_seed,
            )
            if result is not None:
                results.append(result)
        return results

    normalized_jobs = _normalize_n_jobs(n_jobs)
    results = Parallel(n_jobs=normalized_jobs, backend="loky")(
        delayed(_correlate_single_column)(
            col,
            feature_df,
            target_arr,
            temp_arr,
            order_arr,
            method,
            min_samples,
            compute_reliability,
            rng_seed + i,
        )
        for i, col in enumerate(feature_columns)
    )

    return [result for result in results if result is not None]


def _extract_band_from_column_name(column_name: str) -> str:
    """Extract frequency band name from column name."""
    frequency_bands = ["delta", "theta", "alpha", "beta", "gamma"]
    column_lower = column_name.lower()
    for band in frequency_bands:
        if band in column_lower:
            return band
    return "broadband"


def _correlate_single_column(
    col: str,
    feature_df: pd.DataFrame,
    target_arr: np.ndarray,
    temp_arr: Optional[np.ndarray],
    order_arr: Optional[np.ndarray],
    method: str,
    min_samples: int,
    compute_reliability: bool,
    rng_seed: int,
) -> Optional[Dict[str, Any]]:
    """Correlate a single feature column with target."""
    from eeg_pipeline.utils.analysis.stats.correlation import (
        correlate_single_feature,
        interpret_correlation,
    )
    from eeg_pipeline.utils.analysis.stats.reliability import (
        compute_correlation_split_half_reliability as compute_split_half_reliability,
    )

    feature_arr = pd.to_numeric(feature_df[col], errors="coerce").values

    is_change_score = "_change_" in col
    band = _extract_band_from_column_name(col)

    correlation_results = correlate_single_feature(
        feature_arr, target_arr, temp_arr, order_arr, method, min_samples
    )
    r_raw, p_raw, r_pt, p_pt, r_po, p_po, r_pf, p_pf, n = correlation_results

    if not np.isfinite(r_raw):
        return None

    r_primary = r_pt if np.isfinite(r_pt) else r_raw
    p_primary = p_pt if np.isfinite(p_pt) else p_raw
    effect_interpretation = interpret_correlation(r_primary)

    reliability = np.nan
    if compute_reliability:
        rng = np.random.default_rng(rng_seed)
        reliability = compute_split_half_reliability(
            feature_arr, target_arr, method, n_splits=_SPLIT_HALF_RELIABILITY_SPLITS, rng=rng
        )

    return {
        "feature": col,
        "band": band,
        "r_raw": float(r_raw),
        "p_raw": float(p_raw),
        "n": n,
        "r_partial_temp": float(r_pt) if np.isfinite(r_pt) else np.nan,
        "p_partial_temp": float(p_pt) if np.isfinite(p_pt) else np.nan,
        "r_partial_order": float(r_po) if np.isfinite(r_po) else np.nan,
        "p_partial_order": float(p_po) if np.isfinite(p_po) else np.nan,
        "r_partial_full": float(r_pf) if np.isfinite(r_pf) else np.nan,
        "p_partial_full": float(p_pf) if np.isfinite(p_pf) else np.nan,
        "effect_interpretation": effect_interpretation,
        "reliability": reliability,
        "is_change_score": is_change_score,
        "method": method,
        "r_primary": r_primary,
        "p_primary": p_primary,
    }


def parallel_condition_effects(
    feature_columns: List[str],
    features_df: pd.DataFrame,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    min_samples: int,
    n_jobs: int = -1,
    *,
    groups: Optional[np.ndarray] = None,
    n_perm: int = 0,
    base_seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Compute condition effects for multiple features.
    
    Uses vectorized batch computation for speed when possible.
    Falls back to per-feature parallel computation for small feature sets.
    """
    from eeg_pipeline.utils.analysis.stats.effect_size import compute_batch_condition_effects
    
    if not feature_columns:
        return []
    
    n_features = len(feature_columns)
    
    # Use vectorized batch computation for large feature sets (much faster)
    if n_features >= _MIN_FEATURES_FOR_BATCH_CONDITION:
        return compute_batch_condition_effects(
            feature_columns=feature_columns,
            features_df=features_df,
            pain_mask=pain_mask,
            nonpain_mask=nonpain_mask,
            n_perm=n_perm,
            base_seed=base_seed,
            groups=groups,
            logger=logger,
        )
    
    # Fall back to per-feature computation for small feature sets
    if logger:
        logger.info(f"Computing condition effects for {n_features} features (per-feature mode)")
    
    if not _should_use_parallel(n_jobs, n_features, _MIN_FEATURES_FOR_PARALLEL_CONDITION):
        results = []
        log_interval = max(1, n_features // 20)
        for i, col in enumerate(feature_columns):
            if logger and i > 0 and i % log_interval == 0:
                logger.info(f"  Condition effects progress: {i}/{n_features} ({100*i//n_features}%)")
            result = _compute_single_condition_effect(
                col,
                features_df,
                pain_mask,
                nonpain_mask,
                min_samples,
                groups=groups,
                n_perm=n_perm,
                base_seed=base_seed,
            )
            if result is not None:
                results.append(result)
        return results

    normalized_jobs = _normalize_n_jobs(n_jobs)
    results = Parallel(n_jobs=normalized_jobs, backend="loky")(
        delayed(_compute_single_condition_effect)(
            col,
            features_df,
            pain_mask,
            nonpain_mask,
            min_samples,
            groups=groups,
            n_perm=n_perm,
            base_seed=base_seed,
        )
        for col in feature_columns
    )

    return [result for result in results if result is not None]


def _compute_ttest_statistics(
    pain_values: np.ndarray,
    nonpain_values: np.ndarray,
    paired: bool = False,
) -> Tuple[float, float]:
    """Compute t-test statistics for two groups.
    
    Args:
        pain_values: Values for pain condition
        nonpain_values: Values for non-pain condition
        paired: If True, use paired t-test (for within-subject designs).
                If False, use independent samples Welch t-test.
    
    Note:
        For within-subject thermal pain paradigms, paired=True is scientifically
        correct because pain/non-pain conditions are within the same subject.
        Using unpaired tests inflates Type I error by ignoring subject-level
        correlation structure.
    """
    from scipy import stats

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            if paired:
                # Paired t-test for within-subject designs
                # Requires equal-length arrays (matched pairs)
                if len(pain_values) != len(nonpain_values):
                    # Fall back to unpaired if lengths don't match
                    t_stat, p_val = stats.ttest_ind(pain_values, nonpain_values, equal_var=False)
                else:
                    t_stat, p_val = stats.ttest_rel(pain_values, nonpain_values)
            else:
                t_stat, p_val = stats.ttest_ind(pain_values, nonpain_values, equal_var=False)
            return float(t_stat), float(p_val)
    except RuntimeWarning:
        return np.nan, np.nan


def _generate_column_seed(column_name: str, base_seed: int) -> int:
    """Generate deterministic seed for column using hash."""
    hash_bytes = hashlib.sha256(str(column_name).encode("utf-8")).digest()[:8]
    hash_value = int.from_bytes(hash_bytes, "little") % 2_000_000_000
    return int(base_seed + hash_value)


def _extract_valid_groups(
    groups: Optional[np.ndarray], values: np.ndarray, finite_mask: np.ndarray
) -> Optional[np.ndarray]:
    """Extract valid group labels for permutation testing."""
    if groups is None:
        return None

    try:
        groups_array = np.asarray(groups)
        if groups_array.shape[0] != values.shape[0]:
            return None

        groups_subset = groups_array[finite_mask]
        if pd.isna(groups_subset).any():
            return None

        return groups_subset
    except Exception:
        return None


def _compute_permutation_p_value(
    values: np.ndarray,
    labels: np.ndarray,
    observed_statistic: float,
    num_permutations: int,
    groups: Optional[np.ndarray],
    rng: np.random.Generator,
) -> float:
    """Compute permutation-based p-value with optional block-aware shuffling."""
    num_exceeded = 1
    denominator = num_permutations + 1

    for _ in range(num_permutations):
        if groups is None:
            permuted_labels = labels[rng.permutation(len(labels))]
        else:
            permuted_labels = labels.copy()
            for group_id in np.unique(groups):
                group_indices = np.where(groups == group_id)[0]
                if group_indices.size <= 1:
                    continue
                permuted_labels[group_indices] = labels[group_indices][rng.permutation(group_indices.size)]

        permuted_statistic = float(np.abs(np.nanmean(values[permuted_labels]) - np.nanmean(values[~permuted_labels])))
        if permuted_statistic >= observed_statistic - _NUMERIC_TOLERANCE:
            num_exceeded += 1

    return float(num_exceeded / denominator)


def _compute_single_condition_effect(
    col: str,
    features_df: pd.DataFrame,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    min_samples: int,
    *,
    groups: Optional[np.ndarray] = None,
    n_perm: int = 0,
    base_seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Compute condition effect for a single feature."""
    from eeg_pipeline.utils.analysis.stats import hedges_g
    from eeg_pipeline.utils.analysis.stats.correlation import interpret_effect_size

    # Suppress numpy RuntimeWarnings (empty slices, invalid divides in variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        values = pd.to_numeric(features_df[col], errors="coerce").values

        pain_values = values[pain_mask]
        nonpain_values = values[nonpain_mask]

        pain_valid = pain_values[np.isfinite(pain_values)]
        nonpain_valid = nonpain_values[np.isfinite(nonpain_values)]

        mean_pain = float(np.mean(pain_valid))
        mean_nonpain = float(np.mean(nonpain_valid))
        std_pain = float(np.std(pain_valid, ddof=1))
        std_nonpain = float(np.std(nonpain_valid, ddof=1))

        hedges_g_value = hedges_g(pain_valid, nonpain_valid)

        has_zero_variance = std_pain < _NUMERIC_TOLERANCE and std_nonpain < _NUMERIC_TOLERANCE
        has_zero_mean_difference = abs(mean_pain - mean_nonpain) < _NUMERIC_TOLERANCE

        if has_zero_variance and has_zero_mean_difference:
            t_statistic = 0.0
            p_value = 1.0
            hedges_g_value = 0.0
        else:
            t_statistic, p_value = _compute_ttest_statistics(pain_valid, nonpain_valid)

        p_permutation = np.nan
        if n_perm > 0:
            finite_mask = np.isfinite(values) & (pain_mask | nonpain_mask)
            finite_values = values[finite_mask].astype(float, copy=False)
            finite_labels = pain_mask[finite_mask].astype(bool, copy=False)

            has_sufficient_data = finite_values.size >= 4
            has_both_conditions = finite_labels.any() and (~finite_labels).any()

            if has_sufficient_data and has_both_conditions:
                observed_statistic = float(
                    np.abs(np.nanmean(finite_values[finite_labels]) - np.nanmean(finite_values[~finite_labels]))
                )

                rng_seed = _generate_column_seed(col, base_seed)
                rng = np.random.default_rng(rng_seed)

                valid_groups = _extract_valid_groups(groups, values, finite_mask)

                p_permutation = _compute_permutation_p_value(
                    finite_values, finite_labels, observed_statistic, n_perm, valid_groups, rng
                )

        effect_interpretation = interpret_effect_size(hedges_g_value) if np.isfinite(hedges_g_value) else "unknown"

    return {
        "feature": col,
        "mean_pain": mean_pain,
        "mean_nonpain": mean_nonpain,
        "std_pain": std_pain,
        "std_nonpain": std_nonpain,
        "hedges_g": float(hedges_g_value),
        "effect_interpretation": effect_interpretation,
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "p_raw": float(p_value),
        "p_perm": float(p_permutation) if np.isfinite(p_permutation) else np.nan,
        "n_permutations": n_perm if n_perm > 0 else 0,
        "n_pain": len(pain_valid),
        "n_nonpain": len(nonpain_valid),
    }


def parallel_feature_types(
    feature_dfs: Dict[str, pd.DataFrame],
    targets: pd.Series,
    correlate_func: Callable,
    corr_config: Any,
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Correlate multiple feature types in parallel."""
    items = list(feature_dfs.items())

    if not _should_use_parallel(n_jobs, len(items), _MIN_FEATURE_TYPES_FOR_PARALLEL):
        results: Dict[str, Any] = {}
        for name, df in items:
            results[name] = correlate_func(df, targets, corr_config, name)
        return results

    if logger:
        normalized_jobs = _normalize_n_jobs(n_jobs)
        logger.debug(f"Parallel feature types: {len(items)} types, {normalized_jobs} jobs")

    normalized_jobs = _normalize_n_jobs(n_jobs)
    parallel_results = Parallel(n_jobs=normalized_jobs, backend="loky")(
        delayed(correlate_func)(df, targets, corr_config, name) for name, df in items
    )

    return {name: result for (name, _), result in zip(items, parallel_results)}


def _safe_process_subject(
    subject: str, process_func: Callable[[str], T], logger: Optional[logging.Logger]
) -> Tuple[str, Optional[T]]:
    """Safely process a single subject, returning subject ID and result."""
    try:
        return subject, process_func(subject)
    except Exception as e:
        if logger:
            logger.error(f"Failed sub-{subject}: {e}")
        return subject, None


def parallel_subjects(
    subjects: List[str],
    process_func: Callable[[str], T],
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[T]]:
    """Process multiple subjects in parallel."""
    if not _should_use_parallel(n_jobs, len(subjects)):
        results: Dict[str, Optional[T]] = {}
        for subject in subjects:
            _, result = _safe_process_subject(subject, process_func, logger)
            results[subject] = result
        return results

    if logger:
        normalized_jobs = _normalize_n_jobs(n_jobs)
        logger.info(f"Parallel processing: {len(subjects)} subjects, {normalized_jobs} jobs")

    normalized_jobs = _normalize_n_jobs(n_jobs)
    parallel_results = Parallel(n_jobs=normalized_jobs, backend="loky")(
        delayed(_safe_process_subject)(subject, process_func, logger) for subject in subjects
    )

    return {subject: result for subject, result in parallel_results}


def parallel_regression_features(
    feature_args: List[Tuple[Any, ...]],
    process_func: Callable[..., Optional[Dict[str, Any]]],
    n_jobs: int = -1,
    min_features_for_parallel: int = _MIN_FEATURES_FOR_PARALLEL_REGRESSION,
) -> List[Dict[str, Any]]:
    """Run regression for multiple features in parallel.

    Parameters
    ----------
    feature_args : List[Tuple]
        List of argument tuples, one per feature
    process_func : Callable
        Function to process a single feature, returns dict or None
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs - 1)
    min_features_for_parallel : int
        Minimum features to use parallelization

    Returns
    -------
    List[Dict]
        Results for each feature (excludes None results)
    """
    if not _should_use_parallel(n_jobs, len(feature_args), min_features_for_parallel):
        results = [process_func(*args) for args in feature_args]
        return [result for result in results if result is not None]

    normalized_jobs = _normalize_n_jobs(n_jobs)
    results = Parallel(n_jobs=normalized_jobs, backend="loky")(
        delayed(process_func)(*args) for args in feature_args
    )
    return [result for result in results if result is not None]


def parallel_stability_features(
    feature_args: List[Tuple[Any, ...]],
    process_func: Callable[..., Optional[Dict[str, Any]]],
    n_jobs: int = -1,
    min_features_for_parallel: int = _MIN_FEATURES_FOR_PARALLEL_STABILITY,
) -> List[Dict[str, Any]]:
    """Compute stability for multiple features in parallel."""
    if not _should_use_parallel(n_jobs, len(feature_args), min_features_for_parallel):
        results = [process_func(*args) for args in feature_args]
        return [result for result in results if result is not None]

    normalized_jobs = _normalize_n_jobs(n_jobs)
    results = Parallel(n_jobs=normalized_jobs, backend="loky")(
        delayed(process_func)(*args) for args in feature_args
    )
    return [result for result in results if result is not None]


def parallel_influence_features(
    feature_args: List[Tuple[Any, ...]],
    process_func: Callable[..., Optional[Dict[str, Any]]],
    n_jobs: int = -1,
    min_features_for_parallel: int = _MIN_FEATURES_FOR_PARALLEL_INFLUENCE,
) -> List[Dict[str, Any]]:
    """Compute influence diagnostics for multiple features in parallel."""
    if not _should_use_parallel(n_jobs, len(feature_args), min_features_for_parallel):
        results = [process_func(*args) for args in feature_args]
        return [result for result in results if result is not None]

    normalized_jobs = _normalize_n_jobs(n_jobs)
    results = Parallel(n_jobs=normalized_jobs, backend="loky")(
        delayed(process_func)(*args) for args in feature_args
    )
    return [result for result in results if result is not None]


__all__ = [
    "get_n_jobs",
    "parallel_map",
    "parallel_correlate_features",
    "parallel_condition_effects",
    "parallel_feature_types",
    "parallel_subjects",
    "parallel_regression_features",
    "parallel_stability_features",
    "parallel_influence_features",
]










