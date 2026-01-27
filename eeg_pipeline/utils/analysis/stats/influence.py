"""
Influence Diagnostics (Subject-Level, Non-Gating)
=================================================

Computes leverage and Cook's distance for per-feature subject-level models:

  outcome ~ (temperature control) + trial order + run/block dummies + feature (+ optional interaction)

Used to detect when single trials dominate an effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.parallel import get_n_jobs, parallel_influence_features
from eeg_pipeline.utils.analysis.stats._regression_utils import (
    _ols_fit as _fit_ols,
    _build_covariate_design,
    _build_temperature_covariates as _build_temp_cov_shared,
)
from eeg_pipeline.utils.analysis.stats.transforms import zscore_array as _standardize


# Constants
_MIN_SAMPLES_FOR_FIT = 12
_MIN_SAMPLES_ABOVE_PARAMS = 5
_COOKS_THRESHOLD_NUMERATOR = 4.0
_LEVERAGE_THRESHOLD_MULTIPLIER = 2.0
_NUMERICAL_STABILITY_EPSILON = 1e-12
_MIN_DOF = 1
_MIN_FEATURES_FOR_PARALLEL = 5
_DEFAULT_COOKS_THRESHOLD_DIVISOR = 1


from .base import get_config_value as _get_config_value


def _compute_hat_diagonal(X: np.ndarray) -> Optional[np.ndarray]:
    """Compute diagonal of hat matrix (leverage values)."""
    try:
        XtX_inverse = np.linalg.inv(X.T @ X)
        hat_diagonal = np.einsum("ij,jk,ik->i", X, XtX_inverse, X)
        return hat_diagonal
    except np.linalg.LinAlgError:
        return None


def _compute_cooks_distance(
    residuals: np.ndarray,
    leverage: np.ndarray,
    *,
    n_parameters: int,
) -> np.ndarray:
    """Compute Cook's distance for each observation."""
    residuals = np.asarray(residuals, dtype=float)
    leverage = np.asarray(leverage, dtype=float)
    n_samples = int(len(residuals))
    degrees_of_freedom = max(n_samples - n_parameters, _MIN_DOF)
    
    if degrees_of_freedom > 0:
        mean_squared_error = float(np.nansum(residuals**2) / degrees_of_freedom)
    else:
        mean_squared_error = np.nan
    
    denominator = (1.0 - leverage) ** 2
    denominator = np.where(
        np.isfinite(denominator) & (denominator > _NUMERICAL_STABILITY_EPSILON),
        denominator,
        np.nan,
    )
    
    cooks_distance = (
        (residuals**2 / (n_parameters * mean_squared_error + _NUMERICAL_STABILITY_EPSILON))
        * (leverage / denominator)
    )
    return cooks_distance


@dataclass
class InfluenceConfig:
    enabled: bool = True
    outcomes: Optional[List[str]] = None
    max_features: int = 20
    include_trial_order: bool = True
    include_run_block: bool = True
    include_temperature: bool = True
    temperature_control: str = "linear"  # "linear" | "rating_hat" | "spline"
    include_interaction: bool = False
    standardize: bool = True
    n_jobs: int = -1

    @classmethod
    def from_config(cls, config: Any) -> "InfluenceConfig":
        """Create config from configuration object."""
        default_outcomes = ["rating", "pain_residual"]
        outcomes = _get_config_value(config, "behavior_analysis.influence.outcomes", default_outcomes)
        if not isinstance(outcomes, (list, tuple)) or not outcomes:
            outcomes = default_outcomes
        
        temperature_control_raw = _get_config_value(
            config, "behavior_analysis.influence.temperature_control", "linear"
        )
        temperature_control = str(temperature_control_raw).strip().lower()
        
        return cls(
            enabled=bool(_get_config_value(config, "behavior_analysis.influence.enabled", True)),
            outcomes=[str(x) for x in outcomes],
            max_features=int(_get_config_value(config, "behavior_analysis.influence.max_features", 20)),
            include_trial_order=bool(
                _get_config_value(config, "behavior_analysis.influence.include_trial_order", True)
            ),
            include_run_block=bool(
                _get_config_value(config, "behavior_analysis.influence.include_run_block", True)
            ),
            include_temperature=bool(
                _get_config_value(config, "behavior_analysis.influence.include_temperature", True)
            ),
            temperature_control=temperature_control,
            include_interaction=bool(
                _get_config_value(config, "behavior_analysis.influence.include_interaction", False)
            ),
            standardize=bool(_get_config_value(config, "behavior_analysis.influence.standardize", True)),
            n_jobs=int(_get_config_value(config, "behavior_analysis.n_jobs", -1)),
        )


def _extract_p_values_from_dataframe(df: pd.DataFrame, p_column: str) -> List[Tuple[float, str]]:
    """Extract feature-score pairs from dataframe using p-value column."""
    candidates = []
    if p_column not in df.columns:
        return candidates
    
    df["score"] = pd.to_numeric(df[p_column], errors="coerce")
    for _, row in df.dropna(subset=["feature"]).iterrows():
        score = float(row.get("score", np.nan))
        feature = str(row["feature"])
        candidates.append((score, feature))
    return candidates


def _filter_rating_target(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to rating target rows."""
    if "target" in df.columns:
        return df[df["target"].astype(str) == "rating"]
    return df


def _find_p_value_column(df: pd.DataFrame) -> Optional[str]:
    """Find the first available p-value column."""
    p_columns = ["p_primary", "p_raw", "p_value"]
    for col in p_columns:
        if col in df.columns:
            return col
    return None


def _extract_candidates_from_regression_df(regression_df: pd.DataFrame) -> List[Tuple[float, str]]:
    """Extract feature candidates from regression dataframe."""
    df = regression_df.copy()
    df = _filter_rating_target(df)
    return _extract_p_values_from_dataframe(df, "p_primary")


def _extract_candidates_from_models_df(models_df: pd.DataFrame) -> List[Tuple[float, str]]:
    """Extract feature candidates from models dataframe."""
    df = models_df.copy()
    if "model_family" in df.columns:
        df = df[df["model_family"].astype(str) == "ols_hc3"]
    df = _filter_rating_target(df)
    return _extract_p_values_from_dataframe(df, "p_primary")


def _extract_candidates_from_corr_df(corr_df: pd.DataFrame) -> List[Tuple[float, str]]:
    """Extract feature candidates from correlation dataframe."""
    df = corr_df.copy()
    df = _filter_rating_target(df)
    p_column = _find_p_value_column(df)
    if p_column is None:
        return []
    return _extract_p_values_from_dataframe(df, p_column)


def _extract_fallback_candidates_from_regression_df(regression_df: pd.DataFrame) -> List[Tuple[float, str]]:
    """Extract candidates using absolute beta values as fallback."""
    df = regression_df.copy()
    if "beta_feature" not in df.columns:
        return []
    
    df["score"] = -pd.to_numeric(df["beta_feature"], errors="coerce").abs()
    candidates = []
    for _, row in df.dropna(subset=["feature"]).iterrows():
        score = float(row.get("score", np.nan))
        feature = str(row["feature"])
        candidates.append((score, feature))
    return candidates


def _deduplicate_and_rank_candidates(
    candidates: List[Tuple[float, str]],
    max_features: int,
) -> List[str]:
    """Deduplicate candidates keeping best score and return top features."""
    best_scores: Dict[str, float] = {}
    for score, feature in candidates:
        if not np.isfinite(score):
            continue
        if feature not in best_scores or score < best_scores[feature]:
            best_scores[feature] = score
    
    ranked_features = sorted(best_scores.items(), key=lambda item: item[1])
    n_features = max(1, int(max_features))
    return [feature for feature, _ in ranked_features[:n_features]]


def _select_top_features(
    *,
    corr_df: Optional[pd.DataFrame],
    regression_df: Optional[pd.DataFrame],
    models_df: Optional[pd.DataFrame],
    max_features: int,
) -> List[str]:
    """Select top features based on p-values from available dataframes."""
    candidates: List[Tuple[float, str]] = []

    if regression_df is not None and not regression_df.empty:
        candidates.extend(_extract_candidates_from_regression_df(regression_df))

    if models_df is not None and not models_df.empty:
        candidates.extend(_extract_candidates_from_models_df(models_df))

    if corr_df is not None and not corr_df.empty:
        candidates.extend(_extract_candidates_from_corr_df(corr_df))

    if not candidates and regression_df is not None and not regression_df.empty:
        candidates.extend(_extract_fallback_candidates_from_regression_df(regression_df))

    return _deduplicate_and_rank_candidates(candidates, max_features)


def _build_feature_design_matrix(
    feature_values: np.ndarray,
    base_design_matrix: np.ndarray,
    base_design_names: List[str],
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray,
    config: "InfluenceConfig",
) -> Tuple[np.ndarray, List[str]]:
    """Build design matrix including feature and optional interaction."""
    design_parts = [base_design_matrix[valid_mask], feature_values[:, None]]
    design_names = [*base_design_names, "feature"]

    if config.include_interaction and "temperature" in trial_df.columns:
        temperature_raw = pd.to_numeric(trial_df["temperature"], errors="coerce").to_numpy(dtype=float)
        temperature_valid = temperature_raw[valid_mask]
        temperature_standardized = (
            _standardize(temperature_valid) if config.standardize else temperature_valid
        )
        if np.isfinite(temperature_standardized).any():
            interaction = (feature_values * temperature_standardized)[:, None]
            design_parts.append(interaction)
            design_names.append("feature_x_temperature")

    design_matrix = np.column_stack(design_parts)
    return design_matrix, design_names


def _compute_thresholds(
    n_samples: int,
    n_parameters: int,
    config: Optional[Any],
) -> Tuple[float, float]:
    """Compute Cook's distance and leverage thresholds."""
    cooks_threshold_config = _get_config_value(
        config, "behavior_analysis.influence.cooks_threshold", None
    )
    leverage_threshold_config = _get_config_value(
        config, "behavior_analysis.influence.leverage_threshold", None
    )
    
    n_samples_safe = max(int(n_samples), _DEFAULT_COOKS_THRESHOLD_DIVISOR)
    
    if cooks_threshold_config is not None:
        cooks_threshold = float(cooks_threshold_config)
    else:
        cooks_threshold = float(_COOKS_THRESHOLD_NUMERATOR / n_samples_safe)
    
    if leverage_threshold_config is not None:
        leverage_threshold = float(leverage_threshold_config)
    else:
        leverage_threshold = float(_LEVERAGE_THRESHOLD_MULTIPLIER * n_parameters / n_samples_safe)
    
    return cooks_threshold, leverage_threshold


def _find_worst_trial_indices(
    cooks_distance: np.ndarray,
    leverage: np.ndarray,
    valid_mask: np.ndarray,
    trial_df: pd.DataFrame,
) -> Tuple[int, int, float]:
    """Find trial indices with worst Cook's distance and leverage."""
    worst_cooks_index = int(np.nanargmax(cooks_distance)) if np.isfinite(cooks_distance).any() else -1
    worst_leverage_index = int(np.nanargmax(leverage)) if np.isfinite(leverage).any() else -1

    valid_indices = np.where(valid_mask)[0]
    worst_cooks_trial = (
        int(valid_indices[worst_cooks_index]) if worst_cooks_index >= 0 else -1
    )
    worst_leverage_trial = (
        int(valid_indices[worst_leverage_index]) if worst_leverage_index >= 0 else -1
    )

    epoch_column = "epoch" if "epoch" in trial_df.columns else None
    worst_cooks_epoch = (
        int(trial_df.iloc[worst_cooks_trial][epoch_column])
        if epoch_column and worst_cooks_trial >= 0
        else np.nan
    )

    return worst_cooks_trial, worst_leverage_trial, worst_cooks_epoch


def _process_single_influence_feature(
    feat: str,
    trial_df: pd.DataFrame,
    outcome_values: np.ndarray,
    base_design_matrix: np.ndarray,
    base_design_names: List[str],
    cfg: "InfluenceConfig",
    outcome: str,
    config: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Process influence diagnostics for a single feature."""
    if feat not in trial_df.columns:
        return None
    
    feature_raw = pd.to_numeric(trial_df[feat], errors="coerce").to_numpy(dtype=float)
    feature_standardized = _standardize(feature_raw) if cfg.standardize else feature_raw

    valid_mask = (
        np.isfinite(outcome_values)
        & np.isfinite(feature_standardized)
        & np.all(np.isfinite(base_design_matrix), axis=1)
    )
    
    min_samples_required = max(_MIN_SAMPLES_FOR_FIT, base_design_matrix.shape[1] + _MIN_SAMPLES_ABOVE_PARAMS)
    if int(valid_mask.sum()) < min_samples_required:
        return None

    outcome_valid = outcome_values[valid_mask]
    feature_valid = feature_standardized[valid_mask]

    design_matrix, design_names = _build_feature_design_matrix(
        feature_valid,
        base_design_matrix,
        base_design_names,
        trial_df,
        valid_mask,
        cfg,
    )

    coefficients = _fit_ols(design_matrix, outcome_valid)
    if coefficients is None:
        return None

    predicted = design_matrix @ coefficients
    residuals = outcome_valid - predicted
    leverage = _compute_hat_diagonal(design_matrix)
    if leverage is None:
        return None

    n_parameters = int(design_matrix.shape[1])
    cooks_distance = _compute_cooks_distance(residuals, leverage, n_parameters=n_parameters)

    cooks_threshold, leverage_threshold = _compute_thresholds(
        len(outcome_valid), n_parameters, config
    )

    worst_cooks_trial, worst_leverage_trial, worst_cooks_epoch = _find_worst_trial_indices(
        cooks_distance, leverage, valid_mask, trial_df
    )

    return {
        "feature": feat,
        "target": str(outcome),
        "n": int(len(outcome_valid)),
        "p": n_parameters,
        "temperature_control": cfg.temperature_control,
        "cooks_threshold": cooks_threshold,
        "leverage_threshold": leverage_threshold,
        "max_cooks": float(np.nanmax(cooks_distance)) if np.isfinite(cooks_distance).any() else np.nan,
        "n_cooks_gt_threshold": (
            int((cooks_distance > cooks_threshold).sum())
            if np.isfinite(cooks_distance).any()
            else 0
        ),
        "max_leverage": float(np.nanmax(leverage)) if np.isfinite(leverage).any() else np.nan,
        "n_leverage_gt_threshold": (
            int((leverage > leverage_threshold).sum()) if np.isfinite(leverage).any() else 0
        ),
        "worst_cooks_trial_index": worst_cooks_trial,
        "worst_cooks_epoch": worst_cooks_epoch,
        "worst_leverage_trial_index": worst_leverage_trial,
    }


def _add_trial_order_covariate(trial_df: pd.DataFrame) -> Optional[str]:
    """Add trial order covariate if available."""
    trial_order_columns = ["trial_index_within_group", "trial_index"]
    for col in trial_order_columns:
        if col in trial_df.columns:
            return col
    return None


def _add_run_block_covariate(trial_df: pd.DataFrame, config: Optional[Any]) -> Optional[str]:
    """Add run/block covariate if available."""
    run_column_config = _get_config_value(config, "behavior_analysis.run_adjustment.column", "run_id")
    run_column = str(run_column_config or "run_id").strip()
    
    candidate_columns = [run_column, "run_id", "run", "block"]
    seen = set()
    for col in candidate_columns:
        if col and col not in seen and col in trial_df.columns:
            seen.add(col)
            return col
    return None


def _build_base_covariate_dataframe(
    trial_df: pd.DataFrame,
    covariates: List[str],
    temperature_design_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build base dataframe with covariate columns."""
    present_covariates = [col for col in covariates if col in trial_df.columns]
    base_df = trial_df[present_covariates].copy() if present_covariates else pd.DataFrame(index=trial_df.index)
    
    if temperature_design_df is not None:
        for col in covariates:
            if col not in base_df.columns and col in temperature_design_df.columns:
                base_df[col] = temperature_design_df[col]
    
    return base_df


def _process_outcome_influence(
    outcome: str,
    trial_df: pd.DataFrame,
    selected_features: List[str],
    config: "InfluenceConfig",
    global_config: Optional[Any],
    n_jobs: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process influence diagnostics for a single outcome."""
    if outcome not in trial_df.columns:
        return [], {}
    
    outcome_values = pd.to_numeric(trial_df[outcome], errors="coerce").to_numpy(dtype=float)
    
    covariates: List[str] = []
    temperature_control_type = str(config.temperature_control or "linear").strip().lower()
    
    if config.include_temperature:
        temp_covariates, temp_design_df, temp_metadata = _build_temp_cov_shared(
            trial_df=trial_df,
            outcome=outcome,
            temperature_control=temperature_control_type,
            include_temperature=True,
            config=global_config,
            key_prefix="behavior_analysis.influence.temperature_spline",
        )
        covariates.extend(temp_covariates)
    else:
        temp_design_df = None
        temp_metadata = {"temperature_control_requested": temperature_control_type}
    
    if config.include_trial_order:
        trial_order_col = _add_trial_order_covariate(trial_df)
        if trial_order_col is not None:
            covariates.append(trial_order_col)
    
    if config.include_run_block:
        run_block_col = _add_run_block_covariate(trial_df, global_config)
        if run_block_col is not None:
            covariates.append(run_block_col)
    
    base_covariate_df = _build_base_covariate_dataframe(trial_df, covariates, temp_design_df)
    base_design_matrix, base_design_names = _build_covariate_design(
        base_covariate_df, covariates, add_intercept=True
    )
    
    feature_args = [
        (feat, trial_df, outcome_values, base_design_matrix, base_design_names, config, outcome, global_config)
        for feat in selected_features
    ]
    
    outcome_records = parallel_influence_features(
        feature_args,
        _process_single_influence_feature,
        n_jobs=n_jobs,
        min_features_for_parallel=_MIN_FEATURES_FOR_PARALLEL,
    )
    
    return outcome_records, temp_metadata


def compute_influence_diagnostics(
    trial_df: pd.DataFrame,
    *,
    corr_df: Optional[pd.DataFrame] = None,
    regression_df: Optional[pd.DataFrame] = None,
    models_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute influence diagnostics (leverage and Cook's distance) for features."""
    cfg = InfluenceConfig.from_config(config)
    metadata: Dict[str, Any] = {"enabled": cfg.enabled}
    if not cfg.enabled:
        return pd.DataFrame(), {**metadata, "status": "disabled"}

    n_jobs_actual = get_n_jobs(config, cfg.n_jobs)

    selected_features = _select_top_features(
        corr_df=corr_df,
        regression_df=regression_df,
        models_df=models_df,
        max_features=cfg.max_features,
    )
    if not selected_features:
        return pd.DataFrame(), {**metadata, "status": "empty"}
    metadata["selected_features"] = selected_features

    all_records: List[Dict[str, Any]] = []
    temperature_control_by_outcome: Dict[str, Dict[str, Any]] = {}

    for outcome in cfg.outcomes:
        outcome_records, temp_metadata = _process_outcome_influence(
            outcome, trial_df, selected_features, cfg, config, n_jobs_actual
        )
        all_records.extend(outcome_records)
        temperature_control_by_outcome[str(outcome)] = temp_metadata

    if not all_records:
        return pd.DataFrame(), {**metadata, "status": "empty_after_fit"}

    metadata["temperature_control_by_outcome"] = temperature_control_by_outcome
    result_df = pd.DataFrame(all_records)
    return result_df, {**metadata, "status": "ok", "n_rows": int(len(result_df))}


__all__ = ["InfluenceConfig", "compute_influence_diagnostics"]
