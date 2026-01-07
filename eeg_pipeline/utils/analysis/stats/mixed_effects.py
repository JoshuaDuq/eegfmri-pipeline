"""Mixed-effects and mediation utilities shared across analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.analysis.stats.reliability import compute_icc as _compute_icc_array
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.base import get_fdr_alpha
from eeg_pipeline.utils.analysis.stats.mediation import analyze_mediation_for_features


@dataclass
class MixedEffectsResult:
    """Result container for mixed-effects model."""
    fixed_effect: float
    fixed_se: float
    fixed_p: float
    random_variance: float
    residual_variance: float
    icc: float
    n_subjects: int
    n_observations: int
    converged: bool
    aic: float = np.nan
    bic: float = np.nan


def _fit_lmer_manual(
    y: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
    random_slope: bool = False,
) -> MixedEffectsResult:
    """Fit linear mixed-effects model using a simple iterative estimator."""
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_obs = len(y)

    if n_groups < 3 or n_obs < 10:
        return MixedEffectsResult(
            fixed_effect=np.nan, fixed_se=np.nan, fixed_p=np.nan,
            random_variance=np.nan, residual_variance=np.nan,
            icc=np.nan, n_subjects=n_groups, n_observations=n_obs,
            converged=False
        )

    X = np.column_stack([np.ones(n_obs), x])
    try:
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return MixedEffectsResult(
            fixed_effect=np.nan, fixed_se=np.nan, fixed_p=np.nan,
            random_variance=np.nan, residual_variance=np.nan,
            icc=np.nan, n_subjects=n_groups, n_observations=n_obs,
            converged=False
        )

    residuals = y - X @ beta_ols
    ss_total = np.sum(residuals ** 2)

    group_means = np.array([np.mean(residuals[groups == g]) for g in unique_groups])
    group_sizes = np.array([np.sum(groups == g) for g in unique_groups])

    ss_between = np.sum(group_sizes * (group_means - np.mean(residuals)) ** 2)
    ss_within = ss_total - ss_between

    ms_between = ss_between / max(n_groups - 1, 1)
    ms_within = ss_within / max(n_obs - n_groups, 1)

    n0 = (n_obs - np.sum(group_sizes ** 2) / n_obs) / max(n_groups - 1, 1)

    sigma2_u = max(0, (ms_between - ms_within) / n0)
    sigma2_e = ms_within

    icc = sigma2_u / (sigma2_u + sigma2_e + 1e-12)

    try:
        V_inv = np.zeros((n_obs, n_obs))
        for g_id in unique_groups:
            mask = (groups == g_id)
            n_g = mask.sum()
            idx = np.where(mask)[0]

            a = 1.0 / sigma2_e
            b = sigma2_u / (sigma2_e * (sigma2_e + n_g * sigma2_u) + 1e-12)

            V_inv[np.ix_(idx, idx)] = -b
            V_inv[idx, idx] = a - b

        XtVX = X.T @ V_inv @ X
        XtVy = X.T @ V_inv @ y
        beta_gls = np.linalg.solve(XtVX, XtVy)

        cov_beta = np.linalg.inv(XtVX)
        se = np.sqrt(np.diag(cov_beta))

        fixed_effect = beta_gls[1]
        fixed_se = se[1]

        df = max(n_groups - 2, 1)
        t_stat = fixed_effect / (fixed_se + 1e-12)
        fixed_p = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        ll = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(sigma2_e + sigma2_u) + 1)
        k = 4
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(n_obs)

        return MixedEffectsResult(
            fixed_effect=float(fixed_effect),
            fixed_se=float(fixed_se),
            fixed_p=float(fixed_p),
            random_variance=float(sigma2_u),
            residual_variance=float(sigma2_e),
            icc=float(icc),
            n_subjects=n_groups,
            n_observations=n_obs,
            converged=True,
            aic=float(aic),
            bic=float(bic),
        )
    except (np.linalg.LinAlgError, ValueError):
        return MixedEffectsResult(
            fixed_effect=np.nan, fixed_se=np.nan, fixed_p=np.nan,
            random_variance=float(sigma2_u), residual_variance=float(sigma2_e),
            icc=float(icc), n_subjects=n_groups, n_observations=n_obs,
            converged=False
        )


def fit_mixed_effects_model(
    df: pd.DataFrame,
    feature_col: str,
    behavior_col: str,
    subject_col: str = "subject",
    random_effects: str = "intercept",
    covariates: List[str] = None,
) -> Dict[str, Any]:
    """Fit linear mixed-effects model for repeated measures."""
    required_cols = [feature_col, behavior_col, subject_col]
    if covariates:
        required_cols.extend(covariates)

    df_clean = df[required_cols].dropna()

    if len(df_clean) < 10:
        return {
            "converged": False,
            "error": "Insufficient data",
            "n_observations": len(df_clean),
        }

    y = df_clean[behavior_col].values
    x = df_clean[feature_col].values
    groups = df_clean[subject_col].values

    x_std = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-12)

    result = _fit_lmer_manual(y_std, x_std, groups, random_slope=(random_effects == "slope"))

    scale_factor = np.std(y) / (np.std(x) + 1e-12)

    return {
        "converged": result.converged,
        "fixed_effect": result.fixed_effect * scale_factor,
        "fixed_effect_std": result.fixed_effect,
        "fixed_se": result.fixed_se * scale_factor,
        "fixed_p": result.fixed_p,
        "random_variance": result.random_variance,
        "residual_variance": result.residual_variance,
        "icc": result.icc,
        "n_subjects": result.n_subjects,
        "n_observations": result.n_observations,
        "aic": result.aic,
        "bic": result.bic,
    }


def compute_icc(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    icc_type: str = "ICC(1,1)",
) -> Tuple[float, Tuple[float, float]]:
    """Compute Intraclass Correlation Coefficient from DataFrame."""
    df_clean = df[[value_col, group_col]].dropna()

    groups = df_clean[group_col].unique()
    n_groups = len(groups)

    if n_groups < 2:
        return np.nan, (np.nan, np.nan)

    try:
        pivot = df_clean.pivot_table(
            index=group_col,
            columns=df_clean.groupby(group_col).cumcount(),
            values=value_col,
            aggfunc="first"
        ).dropna()

        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return np.nan, (np.nan, np.nan)

        icc, ci_low, ci_high = _compute_icc_array(pivot.values, icc_type=icc_type)
        return float(icc), (float(ci_low), float(ci_high))
    except Exception:
        return np.nan, (np.nan, np.nan)


def run_multilevel_correlation_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    behavior_col: str,
    subject_col: str = "subject",
    covariates: List[str] = None,
    config: Any = None,
) -> pd.DataFrame:
    """Run multilevel models for multiple features."""
    results = []

    for feature in feature_cols:
        if feature not in df.columns:
            continue

        result = fit_mixed_effects_model(
            df, feature, behavior_col, subject_col,
            random_effects="intercept", covariates=covariates
        )

        result["feature"] = feature
        result["behavior"] = behavior_col
        results.append(result)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if "fixed_p" in results_df.columns:
        valid_p = results_df["fixed_p"].dropna()
        if len(valid_p) > 0:
            q_vals = fdr_bh(valid_p.values)
            results_df.loc[valid_p.index, "fixed_p_fdr"] = q_vals
            alpha_thresh = get_fdr_alpha(config) if config is not None else 0.05
            results_df.loc[valid_p.index, "significant_fdr"] = q_vals < alpha_thresh

    return results_df


def run_mediation_analysis(
    df: pd.DataFrame,
    x_col: str,
    mediator_cols: List[str],
    y_col: str,
    n_bootstrap: int = 1000,
    min_effect_size: float = 0.05,
) -> pd.DataFrame:
    """Run mediation analysis for multiple potential mediators."""
    X = df[x_col].values
    Y = df[y_col].values

    valid_cols = [c for c in mediator_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame()

    feature_df = df[valid_cols]

    results = analyze_mediation_for_features(
        X=X,
        feature_df=feature_df,
        Y=Y,
        n_boot=n_bootstrap,
        min_effect_size=min_effect_size,
        x_label=x_col,
        y_label=y_col,
    )

    if not results:
        return pd.DataFrame()

    records = []
    for r in results:
        records.append({
            "mediator": r.m_label,
            "a_path": r.a,
            "b_path": r.b,
            "c_path": r.c,
            "c_prime": r.c_prime,
            "indirect_effect": r.ab,
            "indirect_ci_low": r.ci_ab_low,
            "indirect_ci_high": r.ci_ab_high,
            "proportion_mediated": r.proportion_mediated,
            "sobel_p": r.p_ab,
            "significant": r.is_significant_mediation(),
        })

    return pd.DataFrame(records)


__all__ = [
    "MixedEffectsResult",
    "_fit_lmer_manual",
    "fit_mixed_effects_model",
    "compute_icc",
    "run_multilevel_correlation_analysis",
    "run_mediation_analysis",
]














