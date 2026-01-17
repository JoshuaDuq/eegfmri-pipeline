"""
Block/Run Stability (Subject-Level)
===================================

Non-gating stability diagnostics for feature→outcome associations across repeated
blocks/runs within a subject (common in pain paradigms).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.analysis.stats.partial import compute_partial_corr
from eeg_pipeline.utils.parallel import get_n_jobs, parallel_stability_features
from eeg_pipeline.utils.analysis.stats.base import safe_get_config_value as _get_config_value


# Constants
_MIN_TRIALS_FOR_ANALYSIS = 10
_MIN_VARIANCE_THRESHOLD = 1e-12


def _compute_group_correlation(
    feature_values: np.ndarray,
    outcome_values: np.ndarray,
    method: str,
) -> Tuple[float, float]:
    """Compute correlation for a single group."""
    r, p = compute_correlation(feature_values, outcome_values, method=method)
    r_float = float(r) if np.isfinite(r) else np.nan
    p_float = float(p) if np.isfinite(p) else np.nan
    return r_float, p_float


def _compute_partial_correlation_for_group(
    feature_values: pd.Series,
    outcome_values: pd.Series,
    temperature_values: pd.Series,
    method: str,
) -> Tuple[float, float]:
    """Compute partial correlation controlling for temperature."""
    try:
        r_partial, p_partial, _ = compute_partial_corr(
            feature_values,
            outcome_values,
            pd.DataFrame({"temperature": temperature_values}),
            method=method,
        )
        r_float = float(r_partial) if np.isfinite(r_partial) else np.nan
        p_float = float(p_partial) if np.isfinite(p_partial) else np.nan
        return r_float, p_float
    except (ValueError, np.linalg.LinAlgError, AttributeError):
        return np.nan, np.nan


def _compute_groupwise_statistics(
    correlations: np.ndarray,
    p_values: np.ndarray,
    overall_correlation: float,
    alpha: float,
) -> Dict[str, float]:
    """Compute summary statistics across groups."""
    valid_mask = np.isfinite(correlations)
    n_groups_valid = int(valid_mask.sum())
    
    if not valid_mask.any():
        return {
            "n_groups_valid": n_groups_valid,
            "r_group_mean": np.nan,
            "r_group_std": np.nan,
            "r_group_min": np.nan,
            "r_group_max": np.nan,
            "sign_consistency": np.nan,
            "frac_groups_p_lt_alpha": np.nan,
        }
    
    valid_correlations = correlations[valid_mask]
    valid_p_values = p_values[valid_mask]
    
    has_overall_correlation = np.isfinite(overall_correlation)
    if has_overall_correlation:
        sign_consistency = float(
            (np.sign(valid_correlations) == np.sign(overall_correlation)).mean()
        )
    else:
        sign_consistency = np.nan
    
    return {
        "n_groups_valid": n_groups_valid,
        "r_group_mean": float(np.nanmean(valid_correlations)),
        "r_group_std": float(np.nanstd(valid_correlations, ddof=1)) if len(valid_correlations) > 1 else np.nan,
        "r_group_min": float(np.nanmin(valid_correlations)),
        "r_group_max": float(np.nanmax(valid_correlations)),
        "sign_consistency": sign_consistency,
        "frac_groups_p_lt_alpha": float((valid_p_values < alpha).mean()),
    }


def _process_single_stability_feature(
    feature_name: str,
    trial_df: pd.DataFrame,
    outcome_series: pd.Series,
    group_series: pd.Series,
    groups: List[Any],
    outcome: str,
    group_col: str,
    method: str,
    alpha: float,
    use_partial_temp: bool,
    has_temp: bool,
) -> Optional[Dict[str, Any]]:
    """Process stability for a single feature across groups."""
    feature_values = pd.to_numeric(trial_df[feature_name], errors="coerce")
    valid_mask = feature_values.notna() & outcome_series.notna()
    
    valid_feature = feature_values[valid_mask].to_numpy(dtype=float)
    valid_outcome = outcome_series[valid_mask].to_numpy(dtype=float)
    overall_r, overall_p = compute_correlation(
        valid_feature,
        valid_outcome,
        method=method,
    )
    overall_r = float(overall_r) if np.isfinite(overall_r) else np.nan
    overall_p = float(overall_p) if np.isfinite(overall_p) else np.nan

    use_partial = use_partial_temp and has_temp

    group_correlations = []
    group_p_values = []
    group_sample_sizes = []
    partial_correlations = []
    partial_p_values = []

    for group in groups:
        group_mask = (group_series == group) & feature_values.notna() & outcome_series.notna()
        sample_size = int(group_mask.sum())
        
        group_feature = feature_values[group_mask].to_numpy(dtype=float)
        group_outcome = outcome_series[group_mask].to_numpy(dtype=float)
        r, p = _compute_group_correlation(group_feature, group_outcome, method)
        group_correlations.append(r)
        group_p_values.append(p)
        group_sample_sizes.append(sample_size)

        if use_partial:
            temperature_values = pd.to_numeric(
                trial_df.loc[group_mask, "temperature"],
                errors="coerce"
            )
            valid_temp_mask = temperature_values.notna()
            
            if int(valid_temp_mask.sum()) >= _MIN_TRIALS_FOR_ANALYSIS:
                valid_temp_array = valid_temp_mask.values
                partial_feature = pd.Series(group_feature[valid_temp_array])
                partial_outcome = pd.Series(group_outcome[valid_temp_array])
                partial_temp = temperature_values[valid_temp_mask]
                
                r_partial, p_partial = _compute_partial_correlation_for_group(
                    partial_feature,
                    partial_outcome,
                    partial_temp,
                    method,
                )
                partial_correlations.append(r_partial)
                partial_p_values.append(p_partial)

    correlations_array = np.asarray(group_correlations, dtype=float)
    p_values_array = np.asarray(group_p_values, dtype=float)
    
    group_stats = _compute_groupwise_statistics(
        correlations_array,
        p_values_array,
        overall_r,
        alpha,
    )

    record: Dict[str, Any] = {
        "feature": feature_name,
        "target": outcome,
        "group_column": group_col,
        "method": method,
        "n_groups_total": int(len(groups)),
        "n_trials_total": int(valid_mask.sum()),
        "r_overall": overall_r,
        "p_overall": overall_p,
        "mean_group_n": float(np.mean(group_sample_sizes)) if group_sample_sizes else np.nan,
        **group_stats,
    }

    if partial_correlations:
        partial_corr_array = np.asarray(partial_correlations, dtype=float)
        partial_p_array = np.asarray(partial_p_values, dtype=float)
        valid_partial_mask = np.isfinite(partial_corr_array)
        
        if valid_partial_mask.any():
            valid_partial_corr = partial_corr_array[valid_partial_mask]
            valid_partial_p = partial_p_array[valid_partial_mask]
            
            record["r_partial_temp_group_mean"] = float(np.nanmean(valid_partial_corr))
            record["r_partial_temp_group_std"] = (
                float(np.nanstd(valid_partial_corr, ddof=1))
                if len(valid_partial_corr) > 1
                else np.nan
            )
            record["frac_groups_partial_p_lt_alpha"] = float(
                (valid_partial_p < alpha).mean()
            )

    return record


def _extract_configuration(
    config: Optional[Any],
) -> Dict[str, Any]:
    """Extract and validate configuration parameters."""
    method = str(
        _get_config_value(config, "behavior_analysis.stability.method", "spearman")
    ).strip().lower()
    max_features = int(
        _get_config_value(config, "behavior_analysis.stability.max_features", 50)
    )
    alpha = float(
        _get_config_value(config, "behavior_analysis.stability.alpha", 0.05)
    )
    use_partial_temp = bool(
        _get_config_value(
            config, "behavior_analysis.stability.partial_temperature", True
        )
    )
    n_jobs = int(_get_config_value(config, "behavior_analysis.n_jobs", -1))
    
    return {
        "method": method,
        "max_features": max_features,
        "alpha": alpha,
        "use_partial_temp": use_partial_temp,
        "n_jobs": n_jobs,
    }


def _validate_inputs(
    trial_df: pd.DataFrame,
    outcome: str,
    group_col: str,
) -> bool:
    """Validate input DataFrame and column names."""
    return outcome in trial_df.columns and group_col in trial_df.columns


def _select_candidate_features(
    trial_df: pd.DataFrame,
    feature_cols: List[str],
    outcome_series: pd.Series,
    method: str,
    max_features: int,
) -> Tuple[List[str], int]:
    """Select top features based on absolute correlation strength.
    
    Returns:
        Tuple of (selected_feature_names, total_candidates_considered)
    """
    candidates = []
    
    for col in feature_cols:
        if col not in trial_df.columns:
            continue
        
        feature_values = pd.to_numeric(trial_df[col], errors="coerce")
        valid_mask = feature_values.notna() & outcome_series.notna()
        n_valid = int(valid_mask.sum())
        
        if n_valid < _MIN_TRIALS_FOR_ANALYSIS:
            continue
        
        feature_array = feature_values.to_numpy(dtype=float)
        feature_std = float(np.nanstd(feature_array, ddof=1))
        if feature_std <= _MIN_VARIANCE_THRESHOLD:
            continue
        
        outcome_array = outcome_series.to_numpy(dtype=float)
        r, _ = compute_correlation(
            feature_array,
            outcome_array,
            method=method,
        )
        
        abs_r = abs(float(r)) if np.isfinite(r) else 0.0
        candidates.append((abs_r, col))

    if not candidates:
        return [], 0
    
    candidates.sort(reverse=True)
    selected = [col for _abs_r, col in candidates[:max_features]]
    return selected, len(candidates)


def compute_groupwise_stability(
    trial_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    outcome: str,
    group_col: str,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute per-feature association stability across groups (e.g., runs/blocks).

    This is intentionally descriptive and non-gating: it reports variability and
    sign consistency but does not exclude features.
    """
    cfg = _extract_configuration(config)
    method = cfg["method"]
    max_features = cfg["max_features"]
    alpha = cfg["alpha"]
    use_partial_temp = cfg["use_partial_temp"]
    n_jobs_actual = get_n_jobs(config, cfg["n_jobs"])

    meta: Dict[str, Any] = {
        "method": method,
        "max_features": max_features,
        "alpha": alpha,
        "partial_temperature": use_partial_temp,
        "outcome": outcome,
        "group_col": group_col,
    }

    if not _validate_inputs(trial_df, outcome, group_col):
        return pd.DataFrame(), {**meta, "status": "missing_columns"}

    outcome_series = pd.to_numeric(trial_df[outcome], errors="coerce")
    group_series = trial_df[group_col]

    selected_features, n_candidates = _select_candidate_features(
        trial_df,
        feature_cols,
        outcome_series,
        method,
        max_features,
    )

    if not selected_features:
        return pd.DataFrame(), {**meta, "status": "empty"}

    meta["n_features_considered"] = n_candidates
    meta["n_features_selected"] = len(selected_features)
    meta["has_partial_corr"] = True
    has_temp = "temperature" in trial_df.columns

    groups = group_series.dropna().unique().tolist()
    meta["n_groups_total"] = len(groups)

    feature_args = [
        (
            feat,
            trial_df,
            outcome_series,
            group_series,
            groups,
            outcome,
            group_col,
            method,
            alpha,
            use_partial_temp,
            has_temp,
        )
        for feat in selected_features
    ]

    records = parallel_stability_features(
        feature_args,
        _process_single_stability_feature,
        n_jobs=n_jobs_actual,
        min_features_for_parallel=10,
    )

    result_df = pd.DataFrame(records) if records else pd.DataFrame()
    return result_df, {**meta, "status": "ok"}


__all__ = ["compute_groupwise_stability"]

