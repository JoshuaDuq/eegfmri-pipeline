"""Mixed-effects and mediation utilities shared across analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.analysis.stats.reliability import compute_icc as _compute_icc_array
from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
from eeg_pipeline.utils.analysis.stats.base import get_fdr_alpha
from eeg_pipeline.utils.analysis.stats.mediation import analyze_mediation_for_features


# Numerical stability constants
EPSILON = 1e-12
MIN_GROUPS_FOR_MIXED_EFFECTS = 3
MIN_OBSERVATIONS_FOR_MIXED_EFFECTS = 10
MIN_GROUPS_FOR_ICC = 2
MIN_OBSERVATIONS_FOR_ANALYSIS = 10
MIN_PIVOT_ROWS_FOR_ICC = 2
MIN_PIVOT_COLS_FOR_ICC = 2


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


def _create_failed_result(
    n_subjects: int,
    n_observations: int,
    random_variance: float = np.nan,
    residual_variance: float = np.nan,
    icc: float = np.nan,
) -> MixedEffectsResult:
    """Create a failed MixedEffectsResult with NaN values."""
    return MixedEffectsResult(
        fixed_effect=np.nan,
        fixed_se=np.nan,
        fixed_p=np.nan,
        random_variance=random_variance,
        residual_variance=residual_variance,
        icc=icc,
        n_subjects=n_subjects,
        n_observations=n_observations,
        converged=False,
    )


def _fit_lmer_manual(
    y: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
) -> MixedEffectsResult:
    """Fit linear mixed-effects model using a simple iterative estimator."""
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_obs = len(y)

    if n_groups < MIN_GROUPS_FOR_MIXED_EFFECTS or n_obs < MIN_OBSERVATIONS_FOR_MIXED_EFFECTS:
        return _create_failed_result(n_groups, n_obs)

    design_matrix = np.column_stack([np.ones(n_obs), x])
    try:
        beta_ols = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _create_failed_result(n_groups, n_obs)

    residuals = y - design_matrix @ beta_ols
    ss_total = np.sum(residuals ** 2)

    group_means = np.array([np.mean(residuals[groups == g]) for g in unique_groups])
    group_sizes = np.array([np.sum(groups == g) for g in unique_groups])

    grand_mean_residual = np.mean(residuals)
    ss_between = np.sum(group_sizes * (group_means - grand_mean_residual) ** 2)
    ss_within = ss_total - ss_between

    df_between = max(n_groups - 1, 1)
    df_within = max(n_obs - n_groups, 1)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    sum_squared_group_sizes = np.sum(group_sizes ** 2)
    n0 = (n_obs - sum_squared_group_sizes / n_obs) / df_between

    variance_ratio = (ms_between - ms_within) / n0
    sigma2_u = max(0.0, variance_ratio)
    sigma2_e = ms_within

    total_variance = sigma2_u + sigma2_e
    icc = sigma2_u / (total_variance + EPSILON)

    try:
        precision_matrix = _build_precision_matrix(
            groups, unique_groups, n_obs, sigma2_u, sigma2_e
        )

        design_t_precision = design_matrix.T @ precision_matrix
        xtvx = design_t_precision @ design_matrix
        xtvy = design_t_precision @ y
        beta_gls = np.linalg.solve(xtvx, xtvy)

        cov_beta = np.linalg.inv(xtvx)
        se = np.sqrt(np.diag(cov_beta))

        fixed_effect = beta_gls[1]
        fixed_se = se[1]

        df = max(n_groups - 2, 1)
        t_stat = fixed_effect / (fixed_se + EPSILON)
        fixed_p = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        log_likelihood = _compute_log_likelihood(n_obs, sigma2_u, sigma2_e)
        n_params = 4
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_obs)

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
        return _create_failed_result(n_groups, n_obs, sigma2_u, sigma2_e, icc)


def _build_precision_matrix(
    groups: np.ndarray,
    unique_groups: np.ndarray,
    n_obs: int,
    sigma2_u: float,
    sigma2_e: float,
) -> np.ndarray:
    """Build the precision matrix (inverse covariance) for GLS estimation."""
    precision_matrix = np.zeros((n_obs, n_obs))
    residual_precision = 1.0 / sigma2_e

    for group_id in unique_groups:
        group_mask = groups == group_id
        n_group = group_mask.sum()
        group_indices = np.where(group_mask)[0]

        denominator = sigma2_e * (sigma2_e + n_group * sigma2_u) + EPSILON
        off_diagonal = -sigma2_u / denominator
        diagonal = residual_precision - off_diagonal

        precision_matrix[np.ix_(group_indices, group_indices)] = off_diagonal
        precision_matrix[group_indices, group_indices] = diagonal

    return precision_matrix


def _compute_log_likelihood(
    n_obs: int,
    sigma2_u: float,
    sigma2_e: float,
) -> float:
    """Compute log-likelihood for the mixed-effects model."""
    total_variance = sigma2_e + sigma2_u
    log_variance = np.log(total_variance)
    constant_term = np.log(2 * np.pi) + log_variance + 1
    return -0.5 * n_obs * constant_term


def fit_mixed_effects_model(
    df: pd.DataFrame,
    feature_col: str,
    behavior_col: str,
    subject_col: str = "subject",
    random_effects: str = "intercept",
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Fit linear mixed-effects model for repeated measures."""
    required_cols = [feature_col, behavior_col, subject_col]
    if covariates:
        required_cols.extend(covariates)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_clean = df[required_cols].dropna()

    if len(df_clean) < MIN_OBSERVATIONS_FOR_ANALYSIS:
        return {
            "converged": False,
            "error": "Insufficient data",
            "n_observations": len(df_clean),
        }

    y = df_clean[behavior_col].values
    x = df_clean[feature_col].values
    groups = df_clean[subject_col].values

    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_std = np.std(y)

    x_standardized = (x - x_mean) / (x_std + EPSILON)
    y_standardized = (y - y_mean) / (y_std + EPSILON)

    result = _fit_lmer_manual(y_standardized, x_standardized, groups)

    scale_factor = y_std / (x_std + EPSILON)

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
) -> tuple[float, tuple[float, float]]:
    """Compute Intraclass Correlation Coefficient from DataFrame."""
    if value_col not in df.columns or group_col not in df.columns:
        return np.nan, (np.nan, np.nan)

    df_clean = df[[value_col, group_col]].dropna()

    groups = df_clean[group_col].unique()
    n_groups = len(groups)

    if n_groups < MIN_GROUPS_FOR_ICC:
        return np.nan, (np.nan, np.nan)

    try:
        pivot = df_clean.pivot_table(
            index=group_col,
            columns=df_clean.groupby(group_col).cumcount(),
            values=value_col,
            aggfunc="first"
        ).dropna()

        n_rows, n_cols = pivot.shape
        if n_rows < MIN_PIVOT_ROWS_FOR_ICC or n_cols < MIN_PIVOT_COLS_FOR_ICC:
            return np.nan, (np.nan, np.nan)

        icc, ci_low, ci_high = _compute_icc_array(pivot.values, icc_type=icc_type)
        return float(icc), (float(ci_low), float(ci_high))
    except (ValueError, KeyError, IndexError):
        return np.nan, (np.nan, np.nan)


def run_multilevel_correlation_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    behavior_col: str,
    subject_col: str = "subject",
    covariates: list[str] | None = None,
    config: Any = None,
) -> pd.DataFrame:
    """Run multilevel models for multiple features."""
    if behavior_col not in df.columns:
        return pd.DataFrame()

    valid_features = [f for f in feature_cols if f in df.columns]
    if not valid_features:
        return pd.DataFrame()

    results = []
    for feature in valid_features:
        try:
            result = fit_mixed_effects_model(
                df,
                feature,
                behavior_col,
                subject_col,
                random_effects="intercept",
                covariates=covariates,
            )
            result["feature"] = feature
            result["behavior"] = behavior_col
            results.append(result)
        except (ValueError, KeyError):
            continue

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if "fixed_p" in results_df.columns:
        valid_p = results_df["fixed_p"].dropna()
        if len(valid_p) > 0:
            q_vals = fdr_bh(valid_p.values)
            results_df.loc[valid_p.index, "fixed_p_fdr"] = q_vals
            alpha_threshold = get_fdr_alpha(config) if config is not None else 0.05
            results_df.loc[valid_p.index, "significant_fdr"] = q_vals < alpha_threshold

    return results_df


def run_mediation_analysis(
    df: pd.DataFrame,
    x_col: str,
    mediator_cols: list[str],
    y_col: str,
    n_bootstrap: int = 1000,
    min_effect_size: float = 0.05,
) -> pd.DataFrame:
    """Run mediation analysis for multiple potential mediators."""
    required_cols = [x_col, y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return pd.DataFrame()

    x_values = df[x_col].values
    y_values = df[y_col].values

    valid_mediators = [col for col in mediator_cols if col in df.columns]
    if not valid_mediators:
        return pd.DataFrame()

    mediator_df = df[valid_mediators]

    results = analyze_mediation_for_features(
        X=x_values,
        feature_df=mediator_df,
        Y=y_values,
        n_boot=n_bootstrap,
        min_effect_size=min_effect_size,
        x_label=x_col,
        y_label=y_col,
    )

    if not results:
        return pd.DataFrame()

    records = [
        {
            "mediator": result.m_label,
            "a_path": result.a,
            "b_path": result.b,
            "c_path": result.c,
            "c_prime": result.c_prime,
            "indirect_effect": result.ab,
            "indirect_ci_low": result.ci_ab_low,
            "indirect_ci_high": result.ci_ab_high,
            "proportion_mediated": result.proportion_mediated,
            "sobel_p": result.p_ab,
            "significant": result.is_significant_mediation(),
        }
        for result in results
    ]

    return pd.DataFrame(records)


__all__ = [
    "MixedEffectsResult",
    "_fit_lmer_manual",
    "fit_mixed_effects_model",
    "compute_icc",
    "run_multilevel_correlation_analysis",
    "run_mediation_analysis",
]





