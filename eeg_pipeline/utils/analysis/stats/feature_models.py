"""
Feature Models (Subject-Level)
==============================

Per-feature model families that complement correlations/regressions for event-related studies.

This module is intended for single-subject analysis tables (one row per trial),
and supports multiple outcomes (e.g., `rating`, `predictor_residual`, and `binary_outcome`).
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.analysis.stats.base import get_config_value as _get
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.transforms import zscore_array as _zscore
from eeg_pipeline.utils.parallel import get_n_jobs, parallel_regression_features
from eeg_pipeline.utils.analysis.stats._regression_utils import (
    _ols_fit,
    _hc3_se,
    _r2,
    _build_covariate_design,
    _build_predictor_covariates as _build_temp_cov_shared,
)

# Constants
_NUMERICAL_TOLERANCE = 1e-12
_LOGIT_MAX_ITERATIONS = 200
_QUANTILE_MEDIAN = 0.5
_DEFAULT_FDR_ALPHA = 0.05
_MIN_FEATURES_FOR_PARALLEL = 10
_MIN_BINARY_LEVELS = 2
logger = logging.getLogger(__name__)


@dataclass
class FeatureModelsConfig:
    enabled: bool = False
    outcomes: Optional[List[str]] = None
    families: Optional[List[str]] = None
    include_predictor: bool = True
    predictor_control: str = "linear"  # "linear" | "outcome_hat" | "spline"
    include_trial_order: bool = True
    include_prev_terms: bool = False
    include_run_block: bool = True
    include_interaction: bool = True
    standardize: bool = True
    min_samples: int = 20
    max_features: Optional[int] = 100
    binary_outcome: str = "binary_outcome"  # or "outcome_median"
    n_jobs: int = 1

    @classmethod
    def from_config(cls, config: Any) -> "FeatureModelsConfig":
        outcomes = _get(config, "behavior_analysis.models.outcomes", ["outcome", "predictor_residual"])
        if not isinstance(outcomes, (list, tuple)) or not outcomes:
            outcomes = ["outcome", "predictor_residual"]
        families = _get(config, "behavior_analysis.models.families", ["ols_hc3", "robust_rlm", "quantile_50", "logit"])
        if not isinstance(families, (list, tuple)) or not families:
            families = ["ols_hc3", "robust_rlm", "quantile_50", "logit"]
        return cls(
            enabled=bool(_get(config, "behavior_analysis.models.enabled", False)),
            outcomes=[str(x) for x in outcomes],
            families=[str(x).strip().lower() for x in families],
            include_predictor=bool(_get(config, "behavior_analysis.models.include_predictor", True)),
            predictor_control=str(_get(config, "behavior_analysis.models.predictor_control", "linear")).strip().lower(),
            include_trial_order=bool(_get(config, "behavior_analysis.models.include_trial_order", True)),
            include_prev_terms=bool(_get(config, "behavior_analysis.models.include_prev_terms", False)),
            include_run_block=bool(_get(config, "behavior_analysis.models.include_run_block", True)),
            include_interaction=bool(_get(config, "behavior_analysis.models.include_interaction", True)),
            standardize=bool(_get(config, "behavior_analysis.models.standardize", True)),
            min_samples=int(_get(config, "behavior_analysis.models.min_samples", 20)),
            max_features=_get(config, "behavior_analysis.models.max_features", 100),
            binary_outcome=str(_get(config, "behavior_analysis.models.binary_outcome", "binary_outcome")).strip().lower(),
            n_jobs=int(_get(config, "behavior_analysis.n_jobs", 1)),
        )


def _extract_interaction_terms(
    feature_idx: int,
    names: List[str],
    model_params: Any,
    model_pvalues: Optional[Any] = None,
) -> Tuple[float, float]:
    """Extract interaction term beta and p-value if present."""
    if "feature_x_predictor" not in names:
        return np.nan, np.nan

    interaction_idx = names.index("feature_x_predictor")
    beta_interaction = float(model_params[interaction_idx])
    
    if model_pvalues is not None and hasattr(model_pvalues, "__getitem__"):
        p_interaction = float(model_pvalues[interaction_idx])
    else:
        p_interaction = np.nan
    
    return beta_interaction, p_interaction


def _build_base_record(
    feature: str,
    target: str,
    model_family: str,
    n_samples: int,
    beta_feature: float,
    se_feature: float,
    stat_feature: float,
    p_feature: float,
    p_source: str,
    beta_interaction: float = np.nan,
    p_interaction: float = np.nan,
) -> Dict[str, Any]:
    """Build base record dictionary for model results."""
    return {
        "feature": feature,
        "target": target,
        "model_family": model_family,
        "n": n_samples,
        "beta_feature": beta_feature,
        "se_feature": se_feature,
        "stat_feature": stat_feature,
        "p_feature": p_feature,
        "odds_ratio": np.nan,
        "auc": np.nan,
        "delta_auc": np.nan,
        "pseudo_r2_mcfadden": np.nan,
        "beta_interaction": beta_interaction,
        "p_interaction": p_interaction,
        "p_primary": p_feature,
        "p_raw": p_feature,
        "p_kind_primary": "p_feature",
        "p_primary_source": p_source,
    }


def _fit_logit_model(
    feature: str,
    target: str,
    y_binary: np.ndarray,
    X_base: np.ndarray,
    X_full: np.ndarray,
    names: List[str],
    n_samples: int,
    sm: Any,
) -> Optional[Dict[str, Any]]:
    """Fit logistic regression model and return results record."""
    unique_values = np.unique(y_binary[np.isfinite(y_binary)])
    if len(unique_values) < _MIN_BINARY_LEVELS:
        return None
    
    try:
        model_reduced = sm.Logit(y_binary, X_base).fit(disp=False, maxiter=_LOGIT_MAX_ITERATIONS)
        model_full = sm.Logit(y_binary, X_full).fit(disp=False, maxiter=_LOGIT_MAX_ITERATIONS)
    except (ValueError, np.linalg.LinAlgError, Exception):
        return None
    
    feature_idx = names.index("feature")
    beta = float(model_full.params[feature_idx])
    se = float(model_full.bse[feature_idx]) if np.isfinite(model_full.bse[feature_idx]) else np.nan
    
    if hasattr(model_full, "tvalues"):
        z = float(model_full.tvalues[feature_idx])
    else:
        z = beta / (se + _NUMERICAL_TOLERANCE)
    
    if hasattr(model_full, "pvalues"):
        p = float(model_full.pvalues[feature_idx])
    else:
        p = float(2 * stats.norm.sf(abs(z)))
    
    odds_ratio = float(np.exp(beta)) if np.isfinite(beta) else np.nan
    
    log_likelihood_full = float(model_full.llf) if hasattr(model_full, "llf") else np.nan
    log_likelihood_reduced = float(model_reduced.llf) if hasattr(model_reduced, "llf") else np.nan
    
    if np.isfinite(log_likelihood_full) and np.isfinite(log_likelihood_reduced) and log_likelihood_reduced != 0:
        mcfadden_r2 = 1.0 - (log_likelihood_full / log_likelihood_reduced)
    else:
        mcfadden_r2 = np.nan
    
    auc = np.nan
    delta_auc = np.nan
    try:
        from sklearn.metrics import roc_auc_score
        yhat_full = np.asarray(model_full.predict(X_full), dtype=float)
        yhat_reduced = np.asarray(model_reduced.predict(X_base), dtype=float)
        
        if np.isfinite(yhat_full).all():
            auc = float(roc_auc_score(y_binary, yhat_full))
        if np.isfinite(yhat_reduced).all():
            auc_reduced = float(roc_auc_score(y_binary, yhat_reduced))
            if np.isfinite(auc) and np.isfinite(auc_reduced):
                delta_auc = auc - auc_reduced
    except Exception as exc:
        logger.debug("Unable to compute AUC diagnostics for feature=%s: %s", feature, exc)
    
    beta_interaction, p_interaction = _extract_interaction_terms(
        feature_idx, names, model_full.params, model_full.pvalues if hasattr(model_full, "pvalues") else None
    )
    
    record = _build_base_record(
        feature, target, "logit", n_samples, beta, se, z, p, "mle",
        beta_interaction, p_interaction
    )
    record.update({
        "odds_ratio": odds_ratio,
        "auc": auc,
        "delta_auc": delta_auc,
        "pseudo_r2_mcfadden": mcfadden_r2,
    })
    return record


def _fit_quantile_model(
    feature: str,
    target: str,
    y_continuous: np.ndarray,
    X_full: np.ndarray,
    names: List[str],
    n_samples: int,
    sm: Any,
) -> Optional[Dict[str, Any]]:
    """Fit quantile regression model and return results record."""
    try:
        model = sm.QuantReg(y_continuous, X_full)
        result = model.fit(q=_QUANTILE_MEDIAN)
    except (ValueError, np.linalg.LinAlgError, Exception):
        return None
    
    feature_idx = names.index("feature")
    beta = float(result.params[feature_idx])
    
    if hasattr(result, "bse"):
        se = float(result.bse[feature_idx])
    else:
        se = np.nan
    
    if hasattr(result, "tvalues"):
        t_stat = float(result.tvalues[feature_idx])
    else:
        t_stat = beta / (se + _NUMERICAL_TOLERANCE)
    
    if hasattr(result, "pvalues"):
        p = float(result.pvalues[feature_idx])
    else:
        p = float(2 * stats.norm.sf(abs(t_stat)))
    
    beta_interaction, p_interaction = _extract_interaction_terms(
        feature_idx, names, result.params,
        result.pvalues if hasattr(result, "pvalues") else None
    )
    
    record = _build_base_record(
        feature, target, "quantile_50", n_samples, beta, se, t_stat, p, "quantreg",
        beta_interaction, p_interaction
    )
    return record


def _fit_robust_rlm_model(
    feature: str,
    target: str,
    y_continuous: np.ndarray,
    X_full: np.ndarray,
    names: List[str],
    n_samples: int,
    sm: Any,
) -> Optional[Dict[str, Any]]:
    """Fit robust regression model and return results record."""
    try:
        result = sm.RLM(y_continuous, X_full, M=sm.robust.norms.HuberT()).fit()
    except (ValueError, np.linalg.LinAlgError, Exception):
        return None
    
    feature_idx = names.index("feature")
    beta = float(result.params[feature_idx])
    
    if hasattr(result, "bse"):
        se = float(result.bse[feature_idx])
    else:
        se = np.nan
    
    if np.isfinite(se):
        z = beta / (se + _NUMERICAL_TOLERANCE)
    else:
        z = np.nan
    
    if np.isfinite(z):
        p = float(2 * stats.norm.sf(abs(z)))
    else:
        p = np.nan
    
    beta_interaction, _ = _extract_interaction_terms(feature_idx, names, result.params)
    
    record = _build_base_record(
        feature, target, "robust_rlm", n_samples, beta, se, z, p, "rlm",
        beta_interaction, np.nan
    )
    return record


def _fit_ols_hc3_model(
    feature: str,
    target: str,
    y_continuous: np.ndarray,
    X_full: np.ndarray,
    names: List[str],
    n_samples: int,
) -> Optional[Dict[str, Any]]:
    """Fit OLS with HC3 standard errors and return results record."""
    beta = _ols_fit(X_full, y_continuous)
    if beta is None:
        return None
    
    y_predicted = X_full @ beta
    r2_full = _r2(y_continuous, y_predicted)
    se = _hc3_se(X_full, y_continuous, beta)
    
    degrees_of_freedom = max(int(len(y_continuous) - X_full.shape[1]), 1)
    t_stats = beta / (se + _NUMERICAL_TOLERANCE)
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=degrees_of_freedom)
    
    feature_idx = names.index("feature")
    beta_feature = float(beta[feature_idx])
    se_feature = float(se[feature_idx]) if np.isfinite(se[feature_idx]) else np.nan
    t_stat_feature = float(t_stats[feature_idx]) if np.isfinite(t_stats[feature_idx]) else np.nan
    p_feature = float(p_values[feature_idx]) if np.isfinite(p_values[feature_idx]) else np.nan
    
    beta_interaction, p_interaction = _extract_interaction_terms(
        feature_idx, names, beta, p_values
    )
    
    record = _build_base_record(
        feature, target, "ols_hc3", n_samples, beta_feature, se_feature,
        t_stat_feature, p_feature, "hc3", beta_interaction, p_interaction
    )
    record["r2"] = float(r2_full) if np.isfinite(r2_full) else np.nan
    return record


def _derive_binary_outcome(df: pd.DataFrame, kind: str) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    meta: Dict[str, Any] = {"binary_outcome_kind": kind}
    if kind == "binary_outcome" and "binary_outcome" in df.columns:
        return pd.to_numeric(df["binary_outcome"], errors="coerce"), meta
    if kind in ("outcome_median", "outcome_median_split") and "outcome" in df.columns:
        r = pd.to_numeric(df["outcome"], errors="coerce")
        med = float(r.median(skipna=True)) if r.notna().any() else np.nan
        meta["outcome_median"] = med
        return (r > med).astype(float), meta
    return None, {"binary_outcome_kind": kind, "status": "missing"}


def _prepare_feature_data(
    feature: str,
    trial_df: pd.DataFrame,
    y_all: pd.Series,
    X_base: np.ndarray,
    X_base_names: List[str],
    cfg: "FeatureModelsConfig",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str], int]:
    """Prepare feature and outcome data, returning validated arrays and design matrix."""
    feature_raw = pd.to_numeric(trial_df[feature], errors="coerce")
    feature_values = feature_raw.to_numpy(dtype=float)
    if cfg.standardize:
        feature_values = _zscore(feature_values)

    predictor_values = None
    if cfg.include_interaction and "predictor" in trial_df.columns:
        predictor_raw = pd.to_numeric(trial_df["predictor"], errors="coerce").to_numpy(dtype=float)
        predictor_values = _zscore(predictor_raw) if cfg.standardize else predictor_raw

    outcome_values = pd.to_numeric(y_all, errors="coerce").to_numpy(dtype=float)
    is_valid = (
        np.isfinite(outcome_values) &
        np.isfinite(feature_values) &
        np.all(np.isfinite(X_base), axis=1)
    )
    if predictor_values is not None:
        is_valid = is_valid & np.isfinite(predictor_values)
    
    n_valid = int(is_valid.sum())
    if n_valid < cfg.min_samples:
        return None, None, None, None, [], 0

    outcome_valid = outcome_values[is_valid]
    feature_valid = feature_values[is_valid]
    X_base_valid = X_base[is_valid]

    design_parts = [X_base_valid, feature_valid[:, None]]
    design_names = [*X_base_names, "feature"]
    
    if predictor_values is not None:
        predictor_valid = predictor_values[is_valid]
        interaction_term = (feature_valid * predictor_valid)[:, None]
        design_parts.append(interaction_term)
        design_names.append("feature_x_predictor")
    
    X_full = np.column_stack(design_parts)
    return outcome_valid, feature_valid, X_base_valid, X_full, design_names, n_valid


def _process_single_feature_models(
    feat: str,
    trial_df: pd.DataFrame,
    y_all: pd.Series,
    X_base: np.ndarray,
    X_base_names: List[str],
    cfg: "FeatureModelsConfig",
    families: List[str],
    is_binary: bool,
    out_name: str,
    has_statsmodels: bool,
) -> List[Dict[str, Any]]:
    """Process all model families for a single feature."""
    records: List[Dict[str, Any]] = []
    
    prepared = _prepare_feature_data(feat, trial_df, y_all, X_base, X_base_names, cfg)
    y_valid, _, X_base_valid, X_full, names, n_samples = prepared
    
    if y_valid is None:
        return records

    sm = None
    if has_statsmodels:
        try:
            import statsmodels.api as sm
        except ImportError as exc:
            logger.debug("statsmodels import failed in run_feature_model_for_target: %s", exc)

    for family in families:
        family_lower = str(family).strip().lower()
        
        if family_lower in ("logit", "logistic", "logistic_regression"):
            if not is_binary or sm is None:
                continue
            record = _fit_logit_model(feat, out_name, y_valid, X_base_valid, X_full, names, n_samples, sm)
            if record is not None:
                records.append(record)
        
        elif family_lower in ("quantile_50", "quantile", "median"):
            if is_binary or sm is None:
                continue
            record = _fit_quantile_model(feat, out_name, y_valid, X_full, names, n_samples, sm)
            if record is not None:
                records.append(record)
        
        elif family_lower in ("robust_rlm", "rlm", "huber"):
            if is_binary or sm is None:
                continue
            record = _fit_robust_rlm_model(feat, out_name, y_valid, X_full, names, n_samples, sm)
            if record is not None:
                records.append(record)
        
        elif family_lower in ("ols_hc3", "ols"):
            if is_binary:
                continue
            record = _fit_ols_hc3_model(feat, out_name, y_valid, X_full, names, n_samples)
            if record is not None:
                records.append(record)

    return records


def _select_valid_features(
    trial_df: pd.DataFrame,
    feature_cols: List[str],
    min_samples: int,
    max_features: Optional[int],
) -> Tuple[List[str], Dict[str, Any]]:
    """Select valid features based on data quality and variance."""
    meta: Dict[str, Any] = {}
    candidates: List[str] = []
    
    for col in feature_cols:
        if col not in trial_df.columns:
            continue
        feature_values = pd.to_numeric(trial_df[col], errors="coerce")
        n_valid = int(feature_values.notna().sum())
        if n_valid < min_samples:
            continue
        
        feature_array = feature_values.to_numpy(dtype=float)
        std_dev = float(np.nanstd(feature_array, ddof=1))
        if std_dev <= _NUMERICAL_TOLERANCE:
            continue
        candidates.append(col)

    if max_features is not None and max_features > 0 and len(candidates) > max_features:
        variance_feature_pairs = []
        for col in candidates:
            feature_values = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)
            variance = float(np.nanvar(feature_values, ddof=1))
            variance_feature_pairs.append((variance, col))
        
        variance_feature_pairs.sort(reverse=True)
        candidates = [col for _, col in variance_feature_pairs[:max_features]]
        meta["max_features_applied"] = max_features
    
    return candidates, meta


def _build_additional_covariates(
    trial_df: pd.DataFrame,
    cfg: "FeatureModelsConfig",
    config: Any,
) -> List[str]:
    """Build additional covariates (trial order, previous terms, run/block)."""
    covariates: List[str] = []
    
    if cfg.include_trial_order:
        if "trial_index_within_group" in trial_df.columns:
            covariates.append("trial_index_within_group")
        elif "trial_index" in trial_df.columns:
            covariates.append("trial_index")
    
    if cfg.include_prev_terms:
        prev_term_candidates = ["prev_predictor", "prev_outcome", "delta_predictor", "delta_outcome"]
        for col in prev_term_candidates:
            if col in trial_df.columns:
                covariates.append(col)
    
    if cfg.include_run_block:
        run_col_config = str(_get(config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
        run_col_candidates = [run_col_config, "run_id", "run", "block"]
        seen = set()
        for col in run_col_candidates:
            if col and col not in seen and col in trial_df.columns:
                seen.add(col)
                covariates.append(col)
    
    return covariates


def _prepare_outcome_data(
    trial_df: pd.DataFrame,
    outcome_name: str,
) -> Tuple[Optional[pd.Series], bool, str, Dict[str, Any]]:
    """Prepare outcome data, handling binary and continuous outcomes."""
    outcome_name_str = str(outcome_name)
    meta: Dict[str, Any] = {}
    
    if outcome_name_str == "binary_outcome":
        outcome_series, binary_meta = _derive_binary_outcome(trial_df, "binary_outcome")
        meta.update(binary_meta)
        if outcome_series is None:
            return None, False, outcome_name_str, meta
        return outcome_series, True, outcome_name_str, meta
    
    if outcome_name_str in ("outcome_median", "outcome_median_split"):
        outcome_series, binary_meta = _derive_binary_outcome(trial_df, outcome_name_str)
        meta.update(binary_meta)
        if outcome_series is None:
            return None, False, outcome_name_str, meta
        return outcome_series, True, "binary_outcome_derived", meta
    
    if outcome_name_str not in trial_df.columns:
        return None, False, outcome_name_str, meta
    
    outcome_series = pd.to_numeric(trial_df[outcome_name_str], errors="coerce")
    return outcome_series, False, outcome_name_str, meta


def run_feature_model_families(
    trial_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    config: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit multiple model families per feature for subject-level trialwise data.
    """
    cfg = FeatureModelsConfig.from_config(config)
    meta: Dict[str, Any] = {
        "enabled": cfg.enabled,
        "families": cfg.families,
        "outcomes": cfg.outcomes,
        "predictor_control": cfg.predictor_control,
    }
    if not cfg.enabled:
        return pd.DataFrame(), {**meta, "status": "disabled"}

    meta["random_state"] = int(_get(config, "project.random_state", 42))
    n_jobs_actual = get_n_jobs(config, cfg.n_jobs)

    candidates, feature_selection_meta = _select_valid_features(
        trial_df, feature_cols, cfg.min_samples, cfg.max_features
    )
    meta.update(feature_selection_meta)

    has_statsmodels = importlib.util.find_spec("statsmodels") is not None
    meta["has_statsmodels"] = has_statsmodels

    records: List[Dict[str, Any]] = []
    for outcome in cfg.outcomes:
        outcome_series, is_binary, outcome_name, outcome_meta = _prepare_outcome_data(trial_df, outcome)
        if outcome_series is None:
            continue
        
        meta.setdefault("binary_outcome", {}).update(outcome_meta)

        from eeg_pipeline.utils.data.columns import resolve_predictor_column

        predictor_col = resolve_predictor_column(trial_df, config) or "predictor"
        temp_covariates, temp_design_df, temp_meta = _build_temp_cov_shared(
            trial_df=trial_df,
            outcome=outcome_name,
            predictor_control=cfg.predictor_control,
            include_predictor=cfg.include_predictor,
            config=config,
            predictor_col=predictor_col,
            key_prefix="behavior_analysis.models.predictor_spline",
        )
        additional_covariates = _build_additional_covariates(trial_df, cfg, config)
        
        all_covariates = temp_covariates + additional_covariates
        covariates_present = [c for c in all_covariates if c in trial_df.columns]
        
        base_df = trial_df[covariates_present].copy() if covariates_present else pd.DataFrame(index=trial_df.index)
        if temp_design_df is not None:
            for col in all_covariates:
                if col not in base_df.columns and col in temp_design_df.columns:
                    base_df[col] = temp_design_df[col]

        X_base, X_base_names = _build_covariate_design(base_df, all_covariates, add_intercept=True)
        meta.setdefault("predictor_control_by_outcome", {})[outcome_name] = temp_meta
        meta.setdefault("covariates_by_outcome", {})[outcome_name] = list(all_covariates)

        feature_args = [
            (feat, trial_df, outcome_series, X_base, X_base_names, cfg, cfg.families,
             is_binary, outcome_name, has_statsmodels)
            for feat in candidates
        ]
        
        outcome_records = parallel_regression_features(
            feature_args,
            _process_single_feature_models,
            n_jobs=n_jobs_actual,
            min_features_for_parallel=_MIN_FEATURES_FOR_PARALLEL,
        )
        for record_list in outcome_records:
            records.extend(record_list)

    if not records:
        return pd.DataFrame(), {**meta, "status": "empty"}

    results_df = pd.DataFrame(records)
    p_values_for_fdr = pd.to_numeric(results_df["p_primary"], errors="coerce").to_numpy()
    fdr_alpha = float(_get(config, "behavior_analysis.statistics.fdr_alpha", _DEFAULT_FDR_ALPHA))
    results_df["p_fdr"] = fdr_bh(p_values_for_fdr, alpha=fdr_alpha, config=config)
    
    meta["status"] = "ok"
    meta["n_rows"] = int(len(results_df))
    meta["n_features"] = int(results_df["feature"].nunique()) if "feature" in results_df.columns else 0
    return results_df, meta


__all__ = ["FeatureModelsConfig", "run_feature_model_families"]
