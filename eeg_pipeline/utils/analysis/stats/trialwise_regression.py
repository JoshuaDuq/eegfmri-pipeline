"""
Trialwise Regression (Subject-Level)
====================================

Per-subject regression analyses that complement correlations:

Primary model (default):
    rating ~ temperature + trial_order (+ run/block dummies) + feature

Optional moderation:
    + feature*temperature

Outputs robust (HC3) standard errors and optional Freedman–Lane permutation p-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.analysis.stats.base import safe_get_config_value as _get
from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.transforms import zscore_array as _zscore
from eeg_pipeline.utils.parallel import get_n_jobs, parallel_regression_features
from eeg_pipeline.utils.analysis.stats._regression_utils import (
    _ols_fit,
    _hc3_se,
    _r2,
    _build_covariate_design,
    _build_temperature_covariates as _build_temp_cov_shared,
)


# Constants
_MIN_VARIANCE_THRESHOLD = 1e-12
_MIN_DENOMINATOR_THRESHOLD = 1e-12
_MIN_SE_THRESHOLD = 1e-12
_MAX_DUMMY_VARIABLES = 20
_MIN_FEATURES_FOR_PARALLEL = 10


# OLS fit, HC3 SE, and R² functions are now imported from _regression_utils
# Alias for backward compatibility with existing code
_hc3_se_for_beta = _hc3_se


# _build_covariate_design is now imported from _regression_utils
# Wrapper to maintain backward compatibility with return_design_df=True
def _build_covariate_design_with_df(
    df: pd.DataFrame,
    covariate_cols: List[str],
    *,
    add_intercept: bool = True,
    max_dummies: int = _MAX_DUMMY_VARIABLES,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Build covariate design matrix and return design DataFrame.
    
    Wrapper around consolidated _build_covariate_design that always returns
    the design DataFrame for backward compatibility.
    """
    X, names, design_df = _build_covariate_design(
        df, covariate_cols, add_intercept=add_intercept, max_dummies=max_dummies, return_design_df=True
    )
    return X, names, design_df


# _build_temperature_covariates is now imported from _regression_utils
# Wrapper to maintain backward compatibility with config object interface
def _build_temperature_covariates(
    trial_df: pd.DataFrame,
    outcome: str,
    cfg: "TrialwiseRegressionConfig",
    config: Any,
) -> Tuple[List[str], Optional[pd.DataFrame], Dict[str, Any]]:
    """Build temperature-related covariates based on configuration.
    
    Wrapper around consolidated _build_temperature_covariates that uses config object.
    
    Returns
    -------
    Tuple[List[str], Optional[pd.DataFrame], Dict[str, Any]]
        Covariate column names, optional spline design DataFrame, metadata
    """
    return _build_temp_cov_shared(
        trial_df=trial_df,
        outcome=outcome,
        temperature_control=cfg.temperature_control or "linear",
        include_temperature=cfg.include_temperature,
        config=config,
        key_prefix="behavior_analysis.regression.temperature_spline",
    )


def _build_trial_order_covariates(
    trial_df: pd.DataFrame,
    cfg: "TrialwiseRegressionConfig",
) -> List[str]:
    """Build trial order covariates."""
    covariates: List[str] = []
    if cfg.include_trial_order:
        if "trial_index_within_group" in trial_df.columns:
            covariates.append("trial_index_within_group")
        elif "trial_index" in trial_df.columns:
            covariates.append("trial_index")
    return covariates


def _build_previous_term_covariates(
    trial_df: pd.DataFrame,
    cfg: "TrialwiseRegressionConfig",
) -> List[str]:
    """Build previous trial term covariates."""
    covariates: List[str] = []
    if cfg.include_prev_terms:
        prev_columns = ["prev_temperature", "prev_rating", "delta_temperature", "delta_rating"]
        for col in prev_columns:
            if col in trial_df.columns:
                covariates.append(col)
    return covariates


def _build_run_block_covariates(
    trial_df: pd.DataFrame,
    cfg: "TrialwiseRegressionConfig",
    config: Any,
) -> List[str]:
    """Build run/block grouping covariates."""
    covariates: List[str] = []
    if cfg.include_run_block:
        run_col = str(_get(config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
        candidates = [run_col, "run_id", "run", "block"]
        seen = set()
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            if c in trial_df.columns:
                covariates.append(c)
    return covariates


def _screen_feature_candidates(
    trial_df: pd.DataFrame,
    feature_cols: List[str],
    valid_mask: np.ndarray,
    cfg: "TrialwiseRegressionConfig",
) -> List[str]:
    """Screen features for validity and variance."""
    candidates: List[str] = []
    for col in feature_cols:
        if col not in trial_df.columns:
            continue
        x = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)[valid_mask]
        n_valid = int(np.isfinite(x).sum())
        if n_valid < cfg.min_samples:
            continue
        if np.nanstd(x, ddof=1) <= _MIN_VARIANCE_THRESHOLD:
            continue
        candidates.append(col)
    return candidates


def _select_top_variance_features(
    trial_df: pd.DataFrame,
    candidates: List[str],
    valid_mask: np.ndarray,
    max_features: int,
) -> List[str]:
    """Select top features by variance."""
    variances = []
    for col in candidates:
        x = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)[valid_mask]
        variances.append((float(np.nanvar(x, ddof=1)), col))
    variances.sort(reverse=True)
    return [col for _var, col in variances[:max_features]]


def _prepare_base_dataframe(
    trial_df: pd.DataFrame,
    outcome: str,
    covariates: List[str],
    temp_design_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Prepare base dataframe with outcome and covariates."""
    base_cols = list(dict.fromkeys([outcome, *covariates]))
    base_present = [c for c in base_cols if c in trial_df.columns]
    if outcome not in base_present:
        base_present = [outcome, *[c for c in base_present if c != outcome]]
    base = trial_df[base_present].copy()
    
    if temp_design_df is not None:
        for c in covariates:
            if c not in base.columns and c in temp_design_df.columns:
                base[c] = temp_design_df[c]
    
    return base


def _fit_reduced_model(
    base: pd.DataFrame,
    outcome: str,
    covariates: List[str],
    min_samples: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[np.ndarray], Optional[List[str]]]:
    """Fit reduced model (covariates only) and return fitted values.
    
    Returns
    -------
    Tuple containing:
        - y_valid: outcome values for valid rows
        - Xz_valid: design matrix for valid rows
        - y_hat: fitted values
        - resid: residuals
        - r2_reduced: R-squared of reduced model
        - valid_mask: boolean mask for valid rows
        - Xz_names: column names for design matrix
    """
    Xz, Xz_names, _ = _build_covariate_design_with_df(base, covariates, add_intercept=True)
    y = pd.to_numeric(base[outcome], errors="coerce").to_numpy(dtype=float)
    
    valid_mask = np.isfinite(y) & np.all(np.isfinite(Xz), axis=1)
    if valid_mask.sum() < min_samples:
        return None, None, None, None, None, None, None
    
    y_valid = y[valid_mask]
    Xz_valid = Xz[valid_mask]
    beta_z = _ols_fit(Xz_valid, y_valid)
    if beta_z is None:
        return None, None, None, None, None, None, None
    
    y_hat = Xz_valid @ beta_z
    resid = y_valid - y_hat
    r2_reduced = _r2(y_valid, y_hat)
    
    return y_valid, Xz_valid, y_hat, resid, r2_reduced, valid_mask, Xz_names


@dataclass
class TrialwiseRegressionConfig:
    outcome: str = "rating"
    include_temperature: bool = True
    temperature_control: str = "linear"  # "linear" | "rating_hat" | "spline"
    include_trial_order: bool = True
    include_prev_terms: bool = False
    include_run_block: bool = True
    include_interaction: bool = True
    standardize: bool = True
    min_samples: int = 15
    n_permutations: int = 0
    max_features: Optional[int] = None
    n_jobs: int = 1

    @classmethod
    def from_config(cls, config: Any) -> "TrialwiseRegressionConfig":
        """Create config from configuration object."""
        base_path = "behavior_analysis.regression"
        outcome = str(_get(config, f"{base_path}.outcome", "rating"))
        include_temperature = bool(_get(config, f"{base_path}.include_temperature", True))
        temperature_control_raw = str(_get(config, f"{base_path}.temperature_control", "linear"))
        temperature_control = temperature_control_raw.strip().lower()
        include_trial_order = bool(_get(config, f"{base_path}.include_trial_order", True))
        include_prev_terms = bool(_get(config, f"{base_path}.include_prev_terms", False))
        include_run_block = bool(_get(config, f"{base_path}.include_run_block", True))
        include_interaction = bool(_get(config, f"{base_path}.include_interaction", True))
        standardize = bool(_get(config, f"{base_path}.standardize", True))
        min_samples = int(_get(config, f"{base_path}.min_samples", 15))
        n_permutations = int(_get(config, f"{base_path}.n_permutations", 0))
        max_features = _get(config, f"{base_path}.max_features", None)
        n_jobs = int(_get(config, "behavior_analysis.n_jobs", 1))
        
        return cls(
            outcome=outcome,
            include_temperature=include_temperature,
            temperature_control=temperature_control,
            include_trial_order=include_trial_order,
            include_prev_terms=include_prev_terms,
            include_run_block=include_run_block,
            include_interaction=include_interaction,
            standardize=standardize,
            min_samples=min_samples,
            n_permutations=n_permutations,
            max_features=max_features,
            n_jobs=n_jobs,
        )


def _compute_permutation_pvalues(
    X: np.ndarray,
    y_f: np.ndarray,
    beta: np.ndarray,
    y_hat_z: np.ndarray,
    resid_z: np.ndarray,
    valid_feat: np.ndarray,
    groups_v: Optional[np.ndarray],
    names: List[str],
    idx_feature: int,
    beta_feature: float,
    beta_int: float,
    n_permutations: int,
    rng_seed: int,
) -> Tuple[float, float]:
    """Compute permutation p-values for feature and interaction terms.
    
    Returns
    -------
    Tuple[float, float]
        Permutation p-values for feature and interaction (NaN if not computed)
    """
    if n_permutations <= 0:
        return np.nan, np.nan
    
    rng = np.random.default_rng(rng_seed)
    exceed_feature = 1
    exceed_int = 1
    denom = n_permutations + 1
    has_interaction = "feature_x_temperature" in names
    
    for _ in range(n_permutations):
        perm_idx = permute_within_groups(len(resid_z), rng, groups_v)
        y_perm = y_hat_z + resid_z[perm_idx]
        y_perm_f = y_perm[valid_feat]
        beta_p = _ols_fit(X, y_perm_f)
        if beta_p is None:
            continue
        
        beta_perm_feature = float(beta_p[idx_feature])
        if np.abs(beta_perm_feature) >= np.abs(beta_feature):
            exceed_feature += 1
        
        if has_interaction:
            idx_int = names.index("feature_x_temperature")
            beta_perm_int = float(beta_p[idx_int])
            if np.abs(beta_perm_int) >= np.abs(beta_int):
                exceed_int += 1
    
    p_perm_feature = exceed_feature / denom
    p_perm_int = exceed_int / denom if has_interaction else np.nan
    return p_perm_feature, p_perm_int


def _prepare_feature_and_interaction(
    col: str,
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray,
    cfg: "TrialwiseRegressionConfig",
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Extract and prepare feature values and optional interaction term.
    
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]
        Feature values, interaction term (if applicable), validity mask
    """
    x_raw = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)[valid_mask]
    x = _zscore(x_raw) if cfg.standardize else x_raw
    
    x_int = None
    if cfg.include_interaction and "temperature" in trial_df.columns and np.isfinite(x).any():
        temperature = pd.to_numeric(trial_df["temperature"], errors="coerce").to_numpy(dtype=float)[valid_mask]
        temperature_standardized = _zscore(temperature) if cfg.standardize else temperature
        x_int = x * temperature_standardized
    
    valid_feat = np.isfinite(x)
    if x_int is not None:
        valid_feat = valid_feat & np.isfinite(x_int)
    
    return x, x_int, valid_feat


def _build_full_design_matrix(
    Xz_f: np.ndarray,
    x_f: np.ndarray,
    x_int: Optional[np.ndarray],
    Xz_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build full design matrix including feature and optional interaction."""
    X_parts = [Xz_f, x_f[:, None]]
    names = [*Xz_names, "feature"]
    
    if x_int is not None:
        X_parts.append(x_int[:, None])
        names.append("feature_x_temperature")
    
    X = np.column_stack(X_parts)
    return X, names


def _extract_coefficient_results(
    beta: np.ndarray,
    se: np.ndarray,
    t_stats: np.ndarray,
    p_vals: np.ndarray,
    names: List[str],
) -> Tuple[float, float, float, float, float]:
    """Extract feature and interaction coefficient statistics.
    
    Returns
    -------
    Tuple[float, float, float, float, float]
        beta_feature, p_feature, beta_int, p_int, se_feature
    """
    idx_feature = names.index("feature")
    beta_feature = float(beta[idx_feature])
    se_feature = float(se[idx_feature]) if np.isfinite(se[idx_feature]) else np.nan
    p_feature = float(p_vals[idx_feature]) if np.isfinite(p_vals[idx_feature]) else np.nan
    
    beta_int = np.nan
    p_int = np.nan
    if "feature_x_temperature" in names:
        idx_int = names.index("feature_x_temperature")
        beta_int = float(beta[idx_int])
        p_int = float(p_vals[idx_int]) if np.isfinite(p_vals[idx_int]) else np.nan
    
    return beta_feature, p_feature, beta_int, p_int, se_feature


def _process_single_regression_feature(
    col: str,
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray,
    y_v: np.ndarray,
    Xz_v: np.ndarray,
    Xz_names: List[str],
    y_hat_z: np.ndarray,
    resid_z: np.ndarray,
    r2_reduced: float,
    groups_v: Optional[np.ndarray],
    cfg: "TrialwiseRegressionConfig",
    rng_seed: int,
    out_col: str,
) -> Optional[Dict[str, Any]]:
    """Process a single feature for regression analysis."""
    x, x_int, valid_feat = _prepare_feature_and_interaction(col, trial_df, valid_mask, cfg)
    
    if int(valid_feat.sum()) < cfg.min_samples:
        return None

    y_f = y_v[valid_feat]
    Xz_f = Xz_v[valid_feat]
    x_f = x[valid_feat]
    x_int_f = x_int[valid_feat] if x_int is not None else None
    
    X, names = _build_full_design_matrix(Xz_f, x_f, x_int_f, Xz_names)

    beta = _ols_fit(X, y_f)
    if beta is None:
        return None

    y_hat = X @ beta
    r2_full = _r2(y_f, y_hat)
    delta_r2 = r2_full - r2_reduced if np.isfinite(r2_full) and np.isfinite(r2_reduced) else np.nan

    se = _hc3_se(X, y_f, beta, min_denominator=_MIN_DENOMINATOR_THRESHOLD)
    n_params = X.shape[1]
    dof = max(int(len(y_f) - n_params), 1)
    t_stats = beta / (se + _MIN_SE_THRESHOLD)
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=dof)

    beta_feature, p_feature, beta_int, p_int, se_feature = _extract_coefficient_results(
        beta, se, t_stats, p_vals, names
    )
    
    idx_feature = names.index("feature")
    p_perm_feature, p_perm_int = _compute_permutation_pvalues(
        X,
        y_f,
        beta,
        y_hat_z,
        resid_z,
        valid_feat,
        groups_v,
        names,
        idx_feature,
        beta_feature,
        beta_int,
        cfg.n_permutations,
        rng_seed,
    )

    return {
        "feature": col,
        "target": out_col,
        "n": int(len(y_f)),
        "beta_feature": beta_feature,
        "se_feature_hc3": se_feature,
        "t_feature_hc3": float(t_stats[idx_feature]) if np.isfinite(t_stats[idx_feature]) else np.nan,
        "p_feature": p_feature,
        "beta_interaction": beta_int,
        "p_interaction": p_int,
        "r2_reduced": float(r2_reduced) if np.isfinite(r2_reduced) else np.nan,
        "r2_full": float(r2_full) if np.isfinite(r2_full) else np.nan,
        "delta_r2": float(delta_r2) if np.isfinite(delta_r2) else np.nan,
        "n_covariates": int(len(Xz_names) - 1),
        "n_permutations": int(cfg.n_permutations),
        "p_perm_feature": p_perm_feature,
        "p_perm_interaction": p_perm_int,
        "p_primary": p_perm_feature if np.isfinite(p_perm_feature) else p_feature,
        "p_raw": p_feature,
        "p_kind_primary": "p_perm_feature" if np.isfinite(p_perm_feature) else "p_feature",
        "p_primary_source": "permutation" if np.isfinite(p_perm_feature) else "hc3",
    }


def run_trialwise_feature_regressions(
    trial_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    config: Any,
    groups_for_permutation: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run per-feature regression on a subject's trial table."""
    cfg = TrialwiseRegressionConfig.from_config(config)
    out_col = cfg.outcome
    meta: Dict[str, Any] = {"outcome": out_col, "temperature_control": cfg.temperature_control}

    if out_col not in trial_df.columns:
        return pd.DataFrame(), {"status": "missing_outcome", **meta}

    y_all = pd.to_numeric(trial_df[out_col], errors="coerce")
    if y_all.notna().sum() < cfg.min_samples:
        return pd.DataFrame(), {"status": "insufficient_samples", "n_valid": int(y_all.notna().sum()), **meta}

    temp_covariates, temp_design_df, temp_meta = _build_temperature_covariates(trial_df, out_col, cfg, config)
    meta.update(temp_meta)
    
    covariates = (
        temp_covariates
        + _build_trial_order_covariates(trial_df, cfg)
        + _build_previous_term_covariates(trial_df, cfg)
        + _build_run_block_covariates(trial_df, cfg, config)
    )

    base = _prepare_base_dataframe(trial_df, out_col, covariates, temp_design_df)
    
    y_v, Xz_v, y_hat_z, resid_z, r2_reduced, valid_mask, Xz_names = _fit_reduced_model(
        base, out_col, covariates, cfg.min_samples
    )
    
    if y_v is None:
        status = "insufficient_samples_after_covariates" if valid_mask is None else "failed_reduced_fit"
        n_valid = int(valid_mask.sum()) if valid_mask is not None else 0
        return pd.DataFrame(), {"status": status, "n_valid": n_valid, **meta}

    # Prepare permutation groups (aligned to valid rows)
    groups_v = None
    if cfg.n_permutations > 0 and groups_for_permutation is not None:
        groups_arr = np.asarray(groups_for_permutation)
        if groups_arr.shape[0] == len(trial_df):
            groups_v = groups_arr[valid_mask]

    rng_seed = int(_get(config, "project.random_state", 42))

    candidates = _screen_feature_candidates(trial_df, feature_cols, valid_mask, cfg)
    
    if cfg.max_features is not None:
        try:
            max_features = int(cfg.max_features)
        except Exception:
            max_features = None
        if max_features is not None and max_features > 0 and len(candidates) > max_features:
            candidates = _select_top_variance_features(trial_df, candidates, valid_mask, max_features)
            meta["max_features_applied"] = max_features

    n_jobs_actual = get_n_jobs(config, cfg.n_jobs)
    
    feature_args = [
        (
            col,
            trial_df,
            valid_mask,
            y_v,
            Xz_v,
            Xz_names,
            y_hat_z,
            resid_z,
            r2_reduced,
            groups_v,
            cfg,
            rng_seed + i,
            out_col,
        )
        for i, col in enumerate(candidates)
    ]
    
    records = parallel_regression_features(
        feature_args,
        _process_single_regression_feature,
        n_jobs=n_jobs_actual,
        min_features_for_parallel=_MIN_FEATURES_FOR_PARALLEL,
    )

    if not records:
        return pd.DataFrame(), {"status": "empty", **meta}

    out = pd.DataFrame(records)
    p_for_fdr = pd.to_numeric(out["p_primary"], errors="coerce").to_numpy()
    fdr_alpha = float(_get(config, "behavior_analysis.statistics.fdr_alpha", 0.05))
    out["p_fdr"] = fdr_bh(p_for_fdr, alpha=fdr_alpha, config=config)
    meta["status"] = "ok"
    meta["n_features_tested"] = int(len(out))
    meta["covariates"] = covariates
    return out, meta


__all__ = ["TrialwiseRegressionConfig", "run_trialwise_feature_regressions"]
