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

from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.parallel import get_n_jobs, parallel_regression_features


def _get(config: Any, key: str, default: Any) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:
        pass
    return default


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(x, np.nan)
    return (x - mu) / sd


def _permute_within_groups(
    n: int,
    rng: np.random.Generator,
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    if groups is None:
        return rng.permutation(n)
    groups = np.asarray(groups)
    perm: List[int] = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        perm.extend(idx[rng.permutation(len(idx))])
    return np.asarray(perm, dtype=int)


def _hc3_se_for_beta(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return HC3 SEs for OLS coefficients."""
    n, p = X.shape
    resid = y - X @ beta
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)

    # hat diag: h_i = x_i^T (X'X)^-1 x_i
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    denom = (1.0 - h)
    denom = np.where(np.isfinite(denom) & (np.abs(denom) > 1e-12), denom, np.nan)
    w = (resid**2) / (denom**2)
    w = np.where(np.isfinite(w), w, 0.0)

    middle = X.T @ (X * w[:, None])
    cov = XtX_inv @ middle @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return se


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return beta


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _build_covariate_design(
    df: pd.DataFrame,
    covariate_cols: List[str],
    *,
    add_intercept: bool = True,
    max_dummies: int = 20,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Build numeric covariate design matrix with categorical dummies when appropriate."""
    use = df[covariate_cols].copy() if covariate_cols else pd.DataFrame(index=df.index)
    expanded_cols: List[str] = []

    parts = []
    if add_intercept:
        parts.append(pd.Series(1.0, index=df.index, name="intercept"))
        expanded_cols.append("intercept")

    for col in covariate_cols:
        s = use[col]
        if s.dtype == object or str(s.dtype).startswith("category"):
            # Bound dummy explosion.
            n_levels = int(pd.Series(s).nunique(dropna=True))
            if n_levels <= 1 or n_levels > max_dummies:
                continue
            dummies = pd.get_dummies(s.astype("category"), prefix=str(col), drop_first=True)
            for c in dummies.columns:
                parts.append(pd.to_numeric(dummies[c], errors="coerce").fillna(0.0))
                expanded_cols.append(str(c))
        else:
            parts.append(pd.to_numeric(s, errors="coerce"))
            expanded_cols.append(str(col))

    design_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    X = design_df.to_numpy(dtype=float)
    return X, expanded_cols, design_df


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
        return cls(
            outcome=str(_get(config, "behavior_analysis.regression.outcome", "rating")),
            include_temperature=bool(_get(config, "behavior_analysis.regression.include_temperature", True)),
            temperature_control=str(_get(config, "behavior_analysis.regression.temperature_control", "linear")).strip().lower(),
            include_trial_order=bool(_get(config, "behavior_analysis.regression.include_trial_order", True)),
            include_prev_terms=bool(_get(config, "behavior_analysis.regression.include_prev_terms", False)),
            include_run_block=bool(_get(config, "behavior_analysis.regression.include_run_block", True)),
            include_interaction=bool(_get(config, "behavior_analysis.regression.include_interaction", True)),
            standardize=bool(_get(config, "behavior_analysis.regression.standardize", True)),
            min_samples=int(_get(config, "behavior_analysis.regression.min_samples", 15)),
            n_permutations=int(_get(config, "behavior_analysis.regression.n_permutations", 0)),
            max_features=_get(config, "behavior_analysis.regression.max_features", None),
            n_jobs=int(_get(config, "behavior_analysis.n_jobs", 1)),
        )


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
    x_raw = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)[valid_mask]
    if cfg.standardize:
        x = _zscore(x_raw)
    else:
        x = x_raw
    x_int = None
    if cfg.include_interaction and "temperature" in trial_df.columns and np.isfinite(x).any():
        t = pd.to_numeric(trial_df["temperature"], errors="coerce").to_numpy(dtype=float)[valid_mask]
        t_use = _zscore(t) if cfg.standardize else t
        x_int = x * t_use

    valid_feat = np.isfinite(x)
    if x_int is not None:
        valid_feat = valid_feat & np.isfinite(x_int)
    if int(valid_feat.sum()) < cfg.min_samples:
        return None

    y_f = y_v[valid_feat]
    Xz_f = Xz_v[valid_feat]
    x_f = x[valid_feat]
    X_parts = [Xz_f, x_f[:, None]]
    names = [*Xz_names, "feature"]
    if x_int is not None:
        X_parts.append(x_int[valid_feat][:, None])
        names.append("feature_x_temperature")
    X = np.column_stack(X_parts)

    beta = _ols_fit(X, y_f)
    if beta is None:
        return None

    y_hat = X @ beta
    r2_full = _r2(y_f, y_hat)
    delta_r2 = r2_full - r2_reduced if np.isfinite(r2_full) and np.isfinite(r2_reduced) else np.nan

    se = _hc3_se_for_beta(X, y_f, beta)
    p = X.shape[1]
    dof = max(int(len(y_f) - p), 1)
    t_stats = beta / (se + 1e-12)
    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=dof)

    idx_feature = names.index("feature")
    beta_feature = float(beta[idx_feature])
    p_feature = float(p_vals[idx_feature]) if np.isfinite(p_vals[idx_feature]) else np.nan

    beta_int = np.nan
    p_int = np.nan
    if "feature_x_temperature" in names:
        idx_int = names.index("feature_x_temperature")
        beta_int = float(beta[idx_int])
        p_int = float(p_vals[idx_int]) if np.isfinite(p_vals[idx_int]) else np.nan

    p_perm_feature = np.nan
    p_perm_int = np.nan
    if cfg.n_permutations > 0:
        rng = np.random.default_rng(rng_seed)
        exceed_feature = 1
        exceed_int = 1
        denom = cfg.n_permutations + 1
        for _ in range(cfg.n_permutations):
            perm_idx = _permute_within_groups(len(resid_z), rng, groups_v)
            y_perm = y_hat_z + resid_z[perm_idx]
            y_perm_f = y_perm[valid_feat]
            beta_p = _ols_fit(X, y_perm_f)
            if beta_p is None:
                continue
            bf = float(beta_p[idx_feature])
            if np.abs(bf) >= np.abs(beta_feature):
                exceed_feature += 1
            if "feature_x_temperature" in names:
                bi = float(beta_p[names.index("feature_x_temperature")])
                if np.abs(bi) >= np.abs(beta_int):
                    exceed_int += 1
        p_perm_feature = exceed_feature / denom
        if "feature_x_temperature" in names:
            p_perm_int = exceed_int / denom

    return {
        "feature": col,
        "target": out_col,
        "n": int(len(y_f)),
        "beta_feature": beta_feature,
        "se_feature_hc3": float(se[idx_feature]) if np.isfinite(se[idx_feature]) else np.nan,
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

    covariates: List[str] = []
    temp_ctrl = str(cfg.temperature_control or "linear").strip().lower()
    # Do not include temperature control when temperature is the outcome.
    include_temp_ctrl = bool(cfg.include_temperature and out_col != "temperature")
    temp_design_df = None
    if include_temp_ctrl:
        # Nonlinear control option 1: use predicted rating from temperature (computed in trial table)
        # instead of a linear temperature term. Avoid using this for pain_residual since
        # it is derived from the same curve.
        if temp_ctrl in ("rating_hat", "rating_hat_from_temp", "nonlinear") and out_col not in ("pain_residual",):
            if "rating_hat_from_temp" in trial_df.columns:
                covariates.append("rating_hat_from_temp")
                meta["temperature_control_used"] = "rating_hat"
                meta["temperature_control_column"] = "rating_hat_from_temp"
            elif "temperature" in trial_df.columns:
                covariates.append("temperature")
                meta["temperature_control_used"] = "linear"
                meta["temperature_control_fallback"] = "temperature"
        # Nonlinear control option 2: spline terms for physical temperature.
        elif temp_ctrl in ("spline", "rcs", "restricted_cubic"):
            if "temperature" in trial_df.columns:
                from eeg_pipeline.utils.analysis.stats.splines import build_temperature_rcs_design

                temp_design_df, spline_cols, spline_meta = build_temperature_rcs_design(
                    trial_df["temperature"],
                    config=config,
                    key_prefix="behavior_analysis.regression.temperature_spline",
                    name_prefix="temperature_rcs",
                )
                for c in spline_cols:
                    if c not in covariates:
                        covariates.append(c)
                meta["temperature_control_used"] = "spline" if spline_meta.get("status") in ("ok", "ok_linear_only") else "linear"
                meta["temperature_spline"] = spline_meta
                meta["temperature_control_column"] = "temperature"
            elif "rating_hat_from_temp" in trial_df.columns and out_col not in ("pain_residual",):
                covariates.append("rating_hat_from_temp")
                meta["temperature_control_used"] = "rating_hat_fallback"
                meta["temperature_control_column"] = "rating_hat_from_temp"
        elif "temperature" in trial_df.columns:
            covariates.append("temperature")
            meta["temperature_control_used"] = "linear"
            meta["temperature_control_column"] = "temperature"
    if cfg.include_trial_order:
        if "trial_index_within_group" in trial_df.columns:
            covariates.append("trial_index_within_group")
        elif "trial_index" in trial_df.columns:
            covariates.append("trial_index")
    if cfg.include_prev_terms:
        for c in ["prev_temperature", "prev_rating", "delta_temperature", "delta_rating"]:
            if c in trial_df.columns:
                covariates.append(c)
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

    # Prepare base table with covariates and outcome.
    base_cols = list(dict.fromkeys([out_col, *covariates]))
    base_present = [c for c in base_cols if c in trial_df.columns]
    if out_col not in base_present:
        base_present = [out_col, *[c for c in base_present if c != out_col]]
    base = trial_df[base_present].copy()

    # Add any derived spline columns that are not part of the trial table.
    if temp_design_df is not None:
        for c in covariates:
            if c in base.columns:
                continue
            if c in temp_design_df.columns:
                base[c] = temp_design_df[c]

    # Reduced model matrix (covariates only).
    Xz, Xz_names, _ = _build_covariate_design(base, covariates, add_intercept=True)
    y = pd.to_numeric(base[out_col], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(y) & np.all(np.isfinite(Xz), axis=1)
    if valid_mask.sum() < cfg.min_samples:
        return pd.DataFrame(), {"status": "insufficient_samples_after_covariates", "n_valid": int(valid_mask.sum()), **meta}

    y_v = y[valid_mask]
    Xz_v = Xz[valid_mask]
    beta_z = _ols_fit(Xz_v, y_v)
    if beta_z is None:
        return pd.DataFrame(), {"status": "failed_reduced_fit", **meta}
    y_hat_z = Xz_v @ beta_z
    resid_z = y_v - y_hat_z
    r2_reduced = _r2(y_v, y_hat_z)

    # Prepare permutation groups (aligned to valid rows)
    groups_v = None
    if cfg.n_permutations > 0 and groups_for_permutation is not None:
        groups_arr = np.asarray(groups_for_permutation)
        if groups_arr.shape[0] == len(trial_df):
            groups_v = groups_arr[valid_mask]

    rng_seed = int(_get(config, "project.random_state", 42))
    rng = np.random.default_rng(rng_seed)

    # Feature screening and optional max_features.
    candidates = []
    for col in feature_cols:
        if col not in trial_df.columns:
            continue
        x = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)[valid_mask]
        n_ok = int(np.isfinite(x).sum())
        if n_ok < cfg.min_samples:
            continue
        if np.nanstd(x, ddof=1) <= 1e-12:
            continue
        candidates.append(col)

    if cfg.max_features is not None:
        try:
            max_f = int(cfg.max_features)
        except Exception:
            max_f = None
        if max_f is not None and max_f > 0 and len(candidates) > max_f:
            # Keep highest-variance features among candidates.
            vars_ = []
            for col in candidates:
                x = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)[valid_mask]
                vars_.append((float(np.nanvar(x, ddof=1)), col))
            vars_.sort(reverse=True)
            candidates = [c for _v, c in vars_[:max_f]]
            meta["max_features_applied"] = max_f

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
        min_features_for_parallel=10,
    )

    if not records:
        return pd.DataFrame(), {"status": "empty", **meta}

    out = pd.DataFrame(records)
    p_for_fdr = pd.to_numeric(out["p_primary"], errors="coerce").to_numpy()
    out["p_fdr"] = fdr_bh(p_for_fdr, alpha=float(_get(config, "behavior_analysis.statistics.fdr_alpha", 0.05)), config=config)
    meta["status"] = "ok"
    meta["n_features_tested"] = int(len(out))
    meta["covariates"] = covariates
    return out, meta


__all__ = ["TrialwiseRegressionConfig", "run_trialwise_feature_regressions"]
