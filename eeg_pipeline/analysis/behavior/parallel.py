"""
Parallel Utilities for Behavior Analysis
=========================================

Provides parallelization helpers for compute-intensive behavior analysis operations.
Uses joblib for cross-platform multiprocessing with proper memory management.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

T = TypeVar("T")


def get_n_jobs(config: Optional[Any] = None, default: int = -1) -> int:
    """Get number of parallel jobs from config or environment.
    
    Args:
        config: Configuration object with behavior_analysis.n_jobs setting
        default: Default value if not specified (-1 = all cores)
    
    Returns:
        Number of jobs to use (positive integer)
    """
    n_jobs = default
    
    if config is not None:
        from eeg_pipeline.utils.config.loader import get_config_value
        n_jobs = int(get_config_value(config, "behavior_analysis.n_jobs", default))
    
    env_jobs = os.environ.get("EEG_PIPELINE_N_JOBS")
    if env_jobs is not None:
        try:
            n_jobs = int(env_jobs)
        except ValueError:
            pass
    
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    elif n_jobs <= 0:
        n_jobs = 1
    
    return n_jobs


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
    """Apply function to items in parallel.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        n_jobs: Number of parallel jobs (-1 = all cores minus 1)
        backend: Joblib backend ("loky", "threading", "multiprocessing")
        verbose: Verbosity level for joblib
        desc: Description for logging
        logger: Logger instance
        **kwargs: Additional arguments passed to func
    
    Returns:
        List of results in same order as items
    """
    if not items:
        return []
    
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    
    if n_jobs == 1 or len(items) == 1:
        return [func(item, **kwargs) for item in items]
    
    if logger and desc:
        logger.debug(f"Parallel {desc}: {len(items)} items, {n_jobs} jobs")
    
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(item, **kwargs) for item in items
    )
    
    return results


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
) -> List[Dict[str, Any]]:
    """Correlate multiple features with target in parallel.
    
    Args:
        feature_columns: List of column names to correlate
        feature_df: DataFrame containing feature data
        target_arr: Target values array
        temp_arr: Temperature array for partial correlations (optional)
        order_arr: Trial order array for partial correlations (optional)
        method: Correlation method ("spearman" or "pearson")
        min_samples: Minimum valid samples required
        compute_reliability: Whether to compute split-half reliability
        rng_seed: Random seed for reliability computation
        n_jobs: Number of parallel jobs
    
    Returns:
        List of correlation result dictionaries
    """
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    
    if n_jobs == 1 or len(feature_columns) < 10:
        return [
            _correlate_single_column(
                col, feature_df, target_arr, temp_arr, order_arr,
                method, min_samples, compute_reliability, rng_seed
            )
            for col in feature_columns
        ]
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_correlate_single_column)(
            col, feature_df, target_arr, temp_arr, order_arr,
            method, min_samples, compute_reliability, rng_seed + i
        )
        for i, col in enumerate(feature_columns)
    )
    
    return [r for r in results if r is not None]


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
    from eeg_pipeline.analysis.behavior.correlations import (
        _correlate_single_feature,
        compute_split_half_reliability,
        interpret_correlation,
    )
    
    feature_arr = pd.to_numeric(feature_df[col], errors="coerce").values
    
    is_change = "_change_" in col
    band = "broadband"
    for b in ["delta", "theta", "alpha", "beta", "gamma"]:
        if b in col.lower():
            band = b
            break
    
    r_raw, p_raw, r_pt, p_pt, r_po, p_po, r_pf, p_pf, n = _correlate_single_feature(
        feature_arr, target_arr, temp_arr, order_arr, method, min_samples
    )
    
    if not np.isfinite(r_raw):
        return None
    
    r_primary = r_pt if np.isfinite(r_pt) else r_raw
    effect = interpret_correlation(r_primary)
    
    reliability = np.nan
    if compute_reliability:
        rng = np.random.default_rng(rng_seed)
        reliability = compute_split_half_reliability(
            feature_arr, target_arr, method, n_splits=50, rng=rng
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
        "effect_interpretation": effect,
        "reliability": reliability,
        "is_change_score": is_change,
        "method": method,
        "r_primary": r_pt if np.isfinite(r_pt) else r_raw,
        "p_primary": p_pt if np.isfinite(p_pt) else p_raw,
    }


def parallel_condition_effects(
    feature_columns: List[str],
    features_df: pd.DataFrame,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    min_samples: int,
    n_jobs: int = -1,
) -> List[Dict[str, Any]]:
    """Compute condition effects for multiple features in parallel.
    
    Args:
        feature_columns: List of column names to process
        features_df: DataFrame containing feature data
        pain_mask: Boolean mask for pain trials
        nonpain_mask: Boolean mask for non-pain trials
        min_samples: Minimum samples per condition
        n_jobs: Number of parallel jobs
    
    Returns:
        List of effect size result dictionaries
    """
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    
    if n_jobs == 1 or len(feature_columns) < 10:
        return [
            _compute_single_condition_effect(
                col, features_df, pain_mask, nonpain_mask, min_samples
            )
            for col in feature_columns
        ]
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_single_condition_effect)(
            col, features_df, pain_mask, nonpain_mask, min_samples
        )
        for col in feature_columns
    )
    
    return [r for r in results if r is not None]


def _compute_single_condition_effect(
    col: str,
    features_df: pd.DataFrame,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    min_samples: int,
) -> Optional[Dict[str, Any]]:
    """Compute condition effect for a single feature."""
    from scipy import stats
    from eeg_pipeline.utils.analysis.stats import hedges_g
    from eeg_pipeline.analysis.behavior.correlations import interpret_effect_size
    
    vals = pd.to_numeric(features_df[col], errors="coerce").values
    
    pain_vals = vals[pain_mask]
    nonpain_vals = vals[nonpain_mask]
    
    pain_valid = pain_vals[np.isfinite(pain_vals)]
    nonpain_valid = nonpain_vals[np.isfinite(nonpain_vals)]
    
    if len(pain_valid) < min_samples or len(nonpain_valid) < min_samples:
        return None
    
    g = hedges_g(pain_valid, nonpain_valid)
    t_stat, p_val = stats.ttest_ind(pain_valid, nonpain_valid, equal_var=False)
    interp = interpret_effect_size(g)
    
    return {
        "feature": col,
        "mean_pain": float(np.mean(pain_valid)),
        "mean_nonpain": float(np.mean(nonpain_valid)),
        "std_pain": float(np.std(pain_valid, ddof=1)),
        "std_nonpain": float(np.std(nonpain_valid, ddof=1)),
        "hedges_g": float(g),
        "effect_interpretation": interp,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
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
    """Correlate multiple feature types in parallel.
    
    Args:
        feature_dfs: Dictionary of feature type name -> DataFrame
        targets: Target series for correlation
        correlate_func: Function to correlate a single DataFrame
        corr_config: Correlation configuration
        n_jobs: Number of parallel jobs
        logger: Logger instance
    
    Returns:
        Dictionary of feature type name -> correlation results
    """
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    
    items = list(feature_dfs.items())
    
    if n_jobs == 1 or len(items) <= 2:
        results = {}
        for name, df in items:
            results[name] = correlate_func(df, targets, corr_config, name)
        return results
    
    if logger:
        logger.debug(f"Parallel feature types: {len(items)} types, {n_jobs} jobs")
    
    parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(correlate_func)(df, targets, corr_config, name)
        for name, df in items
    )
    
    return {name: result for (name, _), result in zip(items, parallel_results)}


def parallel_subjects(
    subjects: List[str],
    process_func: Callable[[str], T],
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, T]:
    """Process multiple subjects in parallel.
    
    Args:
        subjects: List of subject IDs
        process_func: Function to process a single subject
        n_jobs: Number of parallel jobs
        logger: Logger instance
    
    Returns:
        Dictionary of subject ID -> result
    """
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    
    if n_jobs == 1 or len(subjects) == 1:
        results = {}
        for subject in subjects:
            try:
                results[subject] = process_func(subject)
            except Exception as e:
                if logger:
                    logger.error(f"Failed sub-{subject}: {e}")
                results[subject] = None
        return results
    
    if logger:
        logger.info(f"Parallel processing: {len(subjects)} subjects, {n_jobs} jobs")
    
    def safe_process(subject: str) -> Tuple[str, Optional[T]]:
        try:
            return subject, process_func(subject)
        except Exception as e:
            if logger:
                logger.error(f"Failed sub-{subject}: {e}")
            return subject, None
    
    parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(safe_process)(subject) for subject in subjects
    )
    
    return {subject: result for subject, result in parallel_results}
