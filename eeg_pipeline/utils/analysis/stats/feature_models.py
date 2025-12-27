"""
Feature Models (Subject-Level)
==============================

Per-feature model families that complement correlations/regressions for pain studies.

This module is intended for single-subject analysis tables (one row per trial),
and supports multiple outcomes (e.g., `rating`, `pain_residual`, and `pain_binary`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh


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


def _build_covariate_design(
    df: pd.DataFrame,
    covariate_cols: List[str],
    *,
    add_intercept: bool = True,
    max_dummies: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    use = df[covariate_cols].copy() if covariate_cols else pd.DataFrame(index=df.index)
    parts = []
    names: List[str] = []
    if add_intercept:
        parts.append(pd.Series(1.0, index=df.index, name="intercept"))
        names.append("intercept")

    for col in covariate_cols:
        s = use[col]
        if s.dtype == object or str(s.dtype).startswith("category"):
            n_levels = int(pd.Series(s).nunique(dropna=True))
            if n_levels <= 1 or n_levels > max_dummies:
                continue
            dummies = pd.get_dummies(s.astype("category"), prefix=str(col), drop_first=True)
            for c in dummies.columns:
                parts.append(pd.to_numeric(dummies[c], errors="coerce").fillna(0.0))
                names.append(str(c))
        else:
            parts.append(pd.to_numeric(s, errors="coerce"))
            names.append(str(col))

    design_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    X = design_df.to_numpy(dtype=float)
    return X, names


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return beta


def _hc3_se(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n, p = X.shape
    resid = y - X @ beta
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    denom = (1.0 - h)
    denom = np.where(np.isfinite(denom) & (np.abs(denom) > 1e-12), denom, np.nan)
    w = (resid**2) / (denom**2)
    w = np.where(np.isfinite(w), w, 0.0)
    middle = X.T @ (X * w[:, None])
    cov = XtX_inv @ middle @ XtX_inv
    return np.sqrt(np.diag(cov))


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


@dataclass
class FeatureModelsConfig:
    enabled: bool = False
    feature_set: str = "pain_summaries"  # or "all"
    outcomes: List[str] = None  # type: ignore[assignment]
    families: List[str] = None  # type: ignore[assignment]
    include_temperature: bool = True
    temperature_control: str = "linear"  # "linear" or "rating_hat"
    include_trial_order: bool = True
    include_prev_terms: bool = False
    include_run_block: bool = True
    include_interaction: bool = True
    standardize: bool = True
    min_samples: int = 20
    max_features: Optional[int] = 100
    binary_outcome: str = "pain_binary"  # or "rating_median"
    n_jobs: int = 1

    @classmethod
    def from_config(cls, config: Any) -> "FeatureModelsConfig":
        outcomes = _get(config, "behavior_analysis.models.outcomes", ["rating", "pain_residual"])
        if not isinstance(outcomes, (list, tuple)) or not outcomes:
            outcomes = ["rating", "pain_residual"]
        families = _get(config, "behavior_analysis.models.families", ["ols_hc3", "robust_rlm", "quantile_50", "logit"])
        if not isinstance(families, (list, tuple)) or not families:
            families = ["ols_hc3", "robust_rlm", "quantile_50", "logit"]
        return cls(
            enabled=bool(_get(config, "behavior_analysis.models.enabled", False)),
            feature_set=str(_get(config, "behavior_analysis.models.feature_set", "pain_summaries")).strip().lower(),
            outcomes=[str(x) for x in outcomes],
            families=[str(x).strip().lower() for x in families],
            include_temperature=bool(_get(config, "behavior_analysis.models.include_temperature", True)),
            temperature_control=str(_get(config, "behavior_analysis.models.temperature_control", "linear")).strip().lower(),
            include_trial_order=bool(_get(config, "behavior_analysis.models.include_trial_order", True)),
            include_prev_terms=bool(_get(config, "behavior_analysis.models.include_prev_terms", False)),
            include_run_block=bool(_get(config, "behavior_analysis.models.include_run_block", True)),
            include_interaction=bool(_get(config, "behavior_analysis.models.include_interaction", True)),
            standardize=bool(_get(config, "behavior_analysis.models.standardize", True)),
            min_samples=int(_get(config, "behavior_analysis.models.min_samples", 20)),
            max_features=_get(config, "behavior_analysis.models.max_features", 100),
            binary_outcome=str(_get(config, "behavior_analysis.models.binary_outcome", "pain_binary")).strip().lower(),
            n_jobs=int(_get(config, "behavior_analysis.n_jobs", 1)),
        )


def _derive_binary_outcome(df: pd.DataFrame, kind: str) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    meta: Dict[str, Any] = {"binary_outcome_kind": kind}
    if kind == "pain_binary" and "pain_binary" in df.columns:
        s = pd.to_numeric(df["pain_binary"], errors="coerce")
        ok = s.dropna().isin([0, 1]).all() if s.dropna().size else False
        meta["pain_binary_is_0_1"] = bool(ok)
        return s, meta
    if kind in ("rating_median", "rating_median_split") and "rating" in df.columns:
        r = pd.to_numeric(df["rating"], errors="coerce")
        med = float(r.median(skipna=True)) if r.notna().any() else np.nan
        meta["rating_median"] = med
        return (r > med).astype(float), meta
    return None, {"binary_outcome_kind": kind, "status": "missing"}


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
        "feature_set": cfg.feature_set,
        "temperature_control": cfg.temperature_control,
    }
    if not cfg.enabled:
        return pd.DataFrame(), {**meta, "status": "disabled"}

    # Covariates shared across models.
    covariates: List[str] = []
    if cfg.include_temperature:
        temp_ctrl = cfg.temperature_control
        if temp_ctrl in ("rating_hat", "rating_hat_from_temp", "nonlinear"):
            if "rating_hat_from_temp" in trial_df.columns:
                covariates.append("rating_hat_from_temp")
                meta["temperature_control_column"] = "rating_hat_from_temp"
            elif "temperature" in trial_df.columns:
                covariates.append("temperature")
                meta["temperature_control_fallback"] = "temperature"
        elif "temperature" in trial_df.columns:
            covariates.append("temperature")
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
        for c in ["run", "block"]:
            if c in trial_df.columns:
                covariates.append(c)

    base_cols = list(dict.fromkeys([*covariates]))
    base = trial_df[base_cols].copy() if base_cols else pd.DataFrame(index=trial_df.index)
    X_base, X_base_names = _build_covariate_design(base, covariates, add_intercept=True)

    rng_seed = int(_get(config, "project.random_state", 42))
    meta["random_state"] = rng_seed

    # Feature filtering
    candidates: List[str] = []
    for col in feature_cols:
        if col not in trial_df.columns:
            continue
        x = pd.to_numeric(trial_df[col], errors="coerce")
        if int(x.notna().sum()) < cfg.min_samples:
            continue
        if float(np.nanstd(x.to_numpy(dtype=float), ddof=1)) <= 1e-12:
            continue
        candidates.append(col)

    if cfg.max_features is not None:
        try:
            max_f = int(cfg.max_features)
        except Exception:
            max_f = None
        if max_f is not None and max_f > 0 and len(candidates) > max_f:
            vars_ = []
            for col in candidates:
                x = pd.to_numeric(trial_df[col], errors="coerce").to_numpy(dtype=float)
                vars_.append((float(np.nanvar(x, ddof=1)), col))
            vars_.sort(reverse=True)
            candidates = [c for _v, c in vars_[:max_f]]
            meta["max_features_applied"] = max_f

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf  # noqa: F401
        _has_statsmodels = True
    except Exception:
        sm = None  # type: ignore[assignment]
        _has_statsmodels = False
    meta["has_statsmodels"] = _has_statsmodels

    records: List[Dict[str, Any]] = []
    for outcome in cfg.outcomes:
        out_name = str(outcome)

        if out_name == "pain_binary":
            y_s, bin_meta = _derive_binary_outcome(trial_df, "pain_binary")
            meta.setdefault("binary_outcome", {}).update(bin_meta)
            if y_s is None:
                continue
            y_all = y_s
            is_binary = True
        elif out_name in ("rating_median", "rating_median_split"):
            y_s, bin_meta = _derive_binary_outcome(trial_df, out_name)
            meta.setdefault("binary_outcome", {}).update(bin_meta)
            if y_s is None:
                continue
            y_all = y_s
            is_binary = True
            out_name = "pain_binary_derived"
        else:
            if out_name not in trial_df.columns:
                continue
            y_all = pd.to_numeric(trial_df[out_name], errors="coerce")
            is_binary = False

        for feat in candidates:
            x_raw = pd.to_numeric(trial_df[feat], errors="coerce")
            x = x_raw.to_numpy(dtype=float)
            if cfg.standardize:
                x = _zscore(x)

            t_use = None
            # Moderation uses physical temperature even if temperature is controlled nonlinearly.
            if cfg.include_interaction and "temperature" in trial_df.columns:
                t = pd.to_numeric(trial_df["temperature"], errors="coerce").to_numpy(dtype=float)
                t_use = _zscore(t) if cfg.standardize else t

            y = pd.to_numeric(y_all, errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(X_base), axis=1)
            if t_use is not None:
                valid = valid & np.isfinite(t_use)
            if int(valid.sum()) < cfg.min_samples:
                continue

            y_v = y[valid]
            x_v = x[valid]
            Xb_v = X_base[valid]

            X_parts = [Xb_v, x_v[:, None]]
            names = [*X_base_names, "feature"]
            if t_use is not None:
                X_parts.append((x_v * t_use[valid])[:, None])
                names.append("feature_x_temperature")
            X = np.column_stack(X_parts)

            for fam in cfg.families:
                fam = str(fam).lower()
                if fam in ("logit", "logistic", "logistic_regression"):
                    if not is_binary:
                        continue
                    if not _has_statsmodels:
                        continue
                    # Require both classes.
                    y_bin = y_v
                    if int(np.unique(y_bin[np.isfinite(y_bin)]).size) < 2:
                        continue
                    try:
                        model_red = sm.Logit(y_bin, Xb_v).fit(disp=False, maxiter=200)
                        model = sm.Logit(y_bin, X).fit(disp=False, maxiter=200)
                    except Exception:
                        continue
                    idx = names.index("feature")
                    beta = float(model.params[idx])
                    se = float(model.bse[idx]) if np.isfinite(model.bse[idx]) else np.nan
                    z = float(model.tvalues[idx]) if hasattr(model, "tvalues") else (beta / (se + 1e-12))
                    p = float(model.pvalues[idx]) if hasattr(model, "pvalues") else float(2 * stats.norm.sf(abs(z)))
                    or_ = float(np.exp(beta)) if np.isfinite(beta) else np.nan
                    llf = float(model.llf) if hasattr(model, "llf") else np.nan
                    llf0 = float(model_red.llf) if hasattr(model_red, "llf") else np.nan
                    mcfadden = 1.0 - (llf / llf0) if np.isfinite(llf) and np.isfinite(llf0) and llf0 != 0 else np.nan

                    auc = np.nan
                    delta_auc = np.nan
                    try:
                        from sklearn.metrics import roc_auc_score

                        yhat = np.asarray(model.predict(X), dtype=float)
                        yhat0 = np.asarray(model_red.predict(Xb_v), dtype=float)
                        auc = float(roc_auc_score(y_bin, yhat)) if np.isfinite(yhat).all() else np.nan
                        auc0 = float(roc_auc_score(y_bin, yhat0)) if np.isfinite(yhat0).all() else np.nan
                        delta_auc = auc - auc0 if np.isfinite(auc) and np.isfinite(auc0) else np.nan
                    except Exception:
                        pass

                    records.append(
                        {
                            "feature": feat,
                            "target": out_name,
                            "model_family": "logit",
                            "n": int(valid.sum()),
                            "beta_feature": beta,
                            "se_feature": se,
                            "stat_feature": z,
                            "p_feature": p,
                            "odds_ratio": or_,
                            "auc": auc,
                            "delta_auc": delta_auc,
                            "pseudo_r2_mcfadden": mcfadden,
                            "beta_interaction": float(model.params[names.index("feature_x_temperature")]) if "feature_x_temperature" in names else np.nan,
                            "p_interaction": float(model.pvalues[names.index("feature_x_temperature")]) if "feature_x_temperature" in names and hasattr(model, "pvalues") else np.nan,
                            "p_primary": p,
                            "p_raw": p,
                            "p_kind_primary": "p_feature",
                            "p_primary_source": "mle",
                        }
                    )
                    continue

                if fam in ("quantile_50", "quantile", "median"):
                    if is_binary:
                        continue
                    if not _has_statsmodels:
                        continue
                    try:
                        mod = sm.QuantReg(y_v, X)
                        res = mod.fit(q=0.5)
                    except Exception:
                        continue
                    idx = names.index("feature")
                    beta = float(res.params[idx])
                    se = float(res.bse[idx]) if hasattr(res, "bse") else np.nan
                    t = float(res.tvalues[idx]) if hasattr(res, "tvalues") else (beta / (se + 1e-12))
                    p = float(res.pvalues[idx]) if hasattr(res, "pvalues") else float(2 * stats.norm.sf(abs(t)))
                    records.append(
                        {
                            "feature": feat,
                            "target": out_name,
                            "model_family": "quantile_50",
                            "n": int(valid.sum()),
                            "beta_feature": beta,
                            "se_feature": se,
                            "stat_feature": t,
                            "p_feature": p,
                            "odds_ratio": np.nan,
                            "auc": np.nan,
                            "delta_auc": np.nan,
                            "pseudo_r2_mcfadden": np.nan,
                            "beta_interaction": float(res.params[names.index("feature_x_temperature")]) if "feature_x_temperature" in names else np.nan,
                            "p_interaction": float(res.pvalues[names.index("feature_x_temperature")]) if "feature_x_temperature" in names and hasattr(res, "pvalues") else np.nan,
                            "p_primary": p,
                            "p_raw": p,
                            "p_kind_primary": "p_feature",
                            "p_primary_source": "quantreg",
                        }
                    )
                    continue

                if fam in ("robust_rlm", "rlm", "huber"):
                    if is_binary:
                        continue
                    if not _has_statsmodels:
                        continue
                    try:
                        res = sm.RLM(y_v, X, M=sm.robust.norms.HuberT()).fit()
                    except Exception:
                        continue
                    idx = names.index("feature")
                    beta = float(res.params[idx])
                    se = float(res.bse[idx]) if hasattr(res, "bse") else np.nan
                    z = beta / (se + 1e-12) if np.isfinite(se) else np.nan
                    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
                    records.append(
                        {
                            "feature": feat,
                            "target": out_name,
                            "model_family": "robust_rlm",
                            "n": int(valid.sum()),
                            "beta_feature": beta,
                            "se_feature": se,
                            "stat_feature": float(z) if np.isfinite(z) else np.nan,
                            "p_feature": p,
                            "odds_ratio": np.nan,
                            "auc": np.nan,
                            "delta_auc": np.nan,
                            "pseudo_r2_mcfadden": np.nan,
                            "beta_interaction": float(res.params[names.index("feature_x_temperature")]) if "feature_x_temperature" in names else np.nan,
                            "p_interaction": np.nan,
                            "p_primary": p,
                            "p_raw": p,
                            "p_kind_primary": "p_feature",
                            "p_primary_source": "rlm",
                        }
                    )
                    continue

                if fam in ("ols_hc3", "ols"):
                    if is_binary:
                        continue
                    beta = _ols_fit(X, y_v)
                    if beta is None:
                        continue
                    y_hat = X @ beta
                    r2_full = _r2(y_v, y_hat)
                    se = _hc3_se(X, y_v, beta)
                    dof = max(int(len(y_v) - X.shape[1]), 1)
                    t_stats = beta / (se + 1e-12)
                    p_vals = 2 * stats.t.sf(np.abs(t_stats), df=dof)
                    idx = names.index("feature")
                    beta_f = float(beta[idx])
                    p_f = float(p_vals[idx]) if np.isfinite(p_vals[idx]) else np.nan
                    records.append(
                        {
                            "feature": feat,
                            "target": out_name,
                            "model_family": "ols_hc3",
                            "n": int(valid.sum()),
                            "beta_feature": beta_f,
                            "se_feature": float(se[idx]) if np.isfinite(se[idx]) else np.nan,
                            "stat_feature": float(t_stats[idx]) if np.isfinite(t_stats[idx]) else np.nan,
                            "p_feature": p_f,
                            "odds_ratio": np.nan,
                            "auc": np.nan,
                            "delta_auc": np.nan,
                            "pseudo_r2_mcfadden": np.nan,
                            "r2": float(r2_full) if np.isfinite(r2_full) else np.nan,
                            "beta_interaction": float(beta[names.index("feature_x_temperature")]) if "feature_x_temperature" in names else np.nan,
                            "p_interaction": float(p_vals[names.index("feature_x_temperature")]) if "feature_x_temperature" in names else np.nan,
                            "p_primary": p_f,
                            "p_raw": p_f,
                            "p_kind_primary": "p_feature",
                            "p_primary_source": "hc3",
                        }
                    )

    if not records:
        return pd.DataFrame(), {**meta, "status": "empty"}

    out = pd.DataFrame(records)
    # Within-file FDR (non-global)
    p_for_fdr = pd.to_numeric(out["p_primary"], errors="coerce").to_numpy()
    out["p_fdr"] = fdr_bh(p_for_fdr, alpha=float(_get(config, "behavior_analysis.statistics.fdr_alpha", 0.05)), config=config)
    meta["status"] = "ok"
    meta["n_rows"] = int(len(out))
    meta["n_features"] = int(out["feature"].nunique()) if "feature" in out.columns else 0
    meta["covariates"] = covariates
    return out, meta


__all__ = ["FeatureModelsConfig", "run_feature_model_families"]
