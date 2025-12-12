"""Parallel utilities.

Centralized parallelization helpers (joblib-based) used across analysis modules.

This module replaces the previous location `eeg_pipeline.analysis.behavior.parallel`.
A compatibility wrapper remains at that import path.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count

T = TypeVar("T")


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
    """Apply function to items in parallel."""
    if not items:
        return []

    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    if n_jobs == 1 or len(items) == 1:
        return [func(item, **kwargs) for item in items]

    if logger and desc:
        logger.debug(f"Parallel {desc}: {len(items)} items, {n_jobs} jobs")

    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(item, **kwargs) for item in items
    )


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
    """Correlate multiple features with target in parallel."""
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    if n_jobs == 1 or len(feature_columns) < 10:
        return [
            _correlate_single_column(
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
            for col in feature_columns
        ]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
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
    from eeg_pipeline.utils.analysis.stats.correlation import (
        correlate_single_feature,
        interpret_correlation,
    )
    from eeg_pipeline.utils.analysis.stats.reliability import (
        compute_correlation_split_half_reliability as compute_split_half_reliability,
    )

    feature_arr = pd.to_numeric(feature_df[col], errors="coerce").values

    is_change = "_change_" in col
    band = "broadband"
    for b in ["delta", "theta", "alpha", "beta", "gamma"]:
        if b in col.lower():
            band = b
            break

    r_raw, p_raw, r_pt, p_pt, r_po, p_po, r_pf, p_pf, n = correlate_single_feature(
        feature_arr, target_arr, temp_arr, order_arr, method, min_samples
    )

    if not np.isfinite(r_raw):
        return None

    r_primary = r_pt if np.isfinite(r_pt) else r_raw
    effect = interpret_correlation(r_primary)

    reliability = np.nan
    if compute_reliability:
        rng = np.random.default_rng(rng_seed)
        reliability = compute_split_half_reliability(feature_arr, target_arr, method, n_splits=50, rng=rng)

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
    """Compute condition effects for multiple features in parallel."""
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    if n_jobs == 1 or len(feature_columns) < 10:
        return [
            _compute_single_condition_effect(col, features_df, pain_mask, nonpain_mask, min_samples)
            for col in feature_columns
        ]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_compute_single_condition_effect)(col, features_df, pain_mask, nonpain_mask, min_samples)
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
    from eeg_pipeline.utils.analysis.stats.correlation import interpret_effect_size

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
    """Correlate multiple feature types in parallel."""
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    items = list(feature_dfs.items())

    if n_jobs == 1 or len(items) <= 2:
        results: Dict[str, Any] = {}
        for name, df in items:
            results[name] = correlate_func(df, targets, corr_config, name)
        return results

    if logger:
        logger.debug(f"Parallel feature types: {len(items)} types, {n_jobs} jobs")

    parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(correlate_func)(df, targets, corr_config, name) for name, df in items
    )

    return {name: result for (name, _), result in zip(items, parallel_results)}


def parallel_subjects(
    subjects: List[str],
    process_func: Callable[[str], T],
    n_jobs: int = -1,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[T]]:
    """Process multiple subjects in parallel."""
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)

    if n_jobs == 1 or len(subjects) == 1:
        results: Dict[str, Optional[T]] = {}
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

    parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(safe_process)(subject) for subject in subjects)

    return {subject: result for subject, result in parallel_results}


__all__ = [
    "get_n_jobs",
    "parallel_map",
    "parallel_correlate_features",
    "parallel_condition_effects",
    "parallel_feature_types",
    "parallel_subjects",
]


