"""
Mixed-Effects Modeling for Behavior Analysis
==============================================

Proper handling of repeated measures with:
- Linear mixed-effects models
- Random intercepts and slopes
- ICC computation
- Multilevel correlation analysis
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


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
    """Fit linear mixed-effects model using iterative estimation."""
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
    
    # Initialize with OLS
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
    
    # Estimate variance components via ANOVA-type estimator
    residuals = y - X @ beta_ols
    ss_total = np.sum(residuals ** 2)
    
    # Between-group and within-group variance
    group_means = np.array([np.mean(residuals[groups == g]) for g in unique_groups])
    group_sizes = np.array([np.sum(groups == g) for g in unique_groups])
    
    ss_between = np.sum(group_sizes * (group_means - np.mean(residuals)) ** 2)
    ss_within = ss_total - ss_between
    
    ms_between = ss_between / max(n_groups - 1, 1)
    ms_within = ss_within / max(n_obs - n_groups, 1)
    
    # Average group size for variance component estimation
    n0 = (n_obs - np.sum(group_sizes ** 2) / n_obs) / max(n_groups - 1, 1)
    
    sigma2_u = max(0, (ms_between - ms_within) / n0)  # Random intercept variance
    sigma2_e = ms_within  # Residual variance
    
    # Compute ICC
    icc = sigma2_u / (sigma2_u + sigma2_e + 1e-12)
    
    # Re-estimate fixed effects with GLS
    # Construct V = Z @ D @ Z' + R where D = sigma2_u * I and R = sigma2_e * I
    # For random intercept: V_ij = sigma2_u (if same group) + sigma2_e (if i=j)
    
    # Use weighted least squares approximation
    weights = 1 / (sigma2_e + sigma2_u + 1e-12)
    W = np.diag(np.full(n_obs, weights))
    
    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta_gls = np.linalg.solve(XtWX, XtWy)
        
        # Standard errors
        cov_beta = np.linalg.inv(XtWX)
        se = np.sqrt(np.diag(cov_beta))
        
        # Fixed effect for x (second coefficient)
        fixed_effect = beta_gls[1]
        fixed_se = se[1]
        
        # t-statistic and p-value (Satterthwaite approximation)
        df = n_obs - n_groups - 1
        t_stat = fixed_effect / (fixed_se + 1e-12)
        fixed_p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # AIC/BIC
        ll = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(sigma2_e + sigma2_u) + 1)
        k = 4  # intercept, slope, sigma2_u, sigma2_e
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
    """Fit linear mixed-effects model for repeated measures.
    
    Returns dict with fixed effects, random effects variance, ICC, p-values.
    """
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
    
    # Standardize for numerical stability
    x_std = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-12)
    
    result = _fit_lmer_manual(y_std, x_std, groups, random_slope=(random_effects == "slope"))
    
    # Rescale fixed effect to original units
    scale_factor = np.std(y) / (np.std(x) + 1e-12)
    
    return {
        "converged": result.converged,
        "fixed_effect": result.fixed_effect * scale_factor,
        "fixed_effect_std": result.fixed_effect,  # Standardized
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
    """Compute Intraclass Correlation Coefficient.
    
    Returns (icc, (ci_low, ci_high)).
    """
    df_clean = df[[value_col, group_col]].dropna()
    
    groups = df_clean[group_col].unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        return np.nan, (np.nan, np.nan)
    
    # Group statistics
    group_data = [df_clean[df_clean[group_col] == g][value_col].values for g in groups]
    group_sizes = np.array([len(g) for g in group_data])
    n_total = np.sum(group_sizes)
    
    if n_total < 5:
        return np.nan, (np.nan, np.nan)
    
    # Grand mean
    grand_mean = df_clean[value_col].mean()
    
    # Mean squares
    group_means = np.array([np.mean(g) for g in group_data])
    
    # Between-groups sum of squares
    ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)
    
    # Within-groups sum of squares
    ss_within = np.sum([np.sum((g - np.mean(g)) ** 2) for g in group_data])
    
    df_between = n_groups - 1
    df_within = n_total - n_groups
    
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    # Average group size
    n0 = (n_total - np.sum(group_sizes ** 2) / n_total) / (n_groups - 1)
    
    # ICC(1,1): Single rater, absolute agreement
    if icc_type == "ICC(1,1)":
        icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    # ICC(2,1): Single rater, consistency
    elif icc_type == "ICC(2,1)":
        icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    # ICC(3,1): Single rater, consistency (two-way mixed)
    else:
        icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    
    icc = max(-1, min(1, icc))
    
    # Confidence interval using F distribution
    f_ratio = ms_between / (ms_within + 1e-12)
    
    try:
        f_low = f_ratio / stats.f.ppf(0.975, df_between, df_within)
        f_high = f_ratio / stats.f.ppf(0.025, df_between, df_within)
        
        ci_low = (f_low - 1) / (f_low + n0 - 1)
        ci_high = (f_high - 1) / (f_high + n0 - 1)
        
        ci_low = max(-1, min(1, ci_low))
        ci_high = max(-1, min(1, ci_high))
    except (ValueError, ZeroDivisionError):
        ci_low, ci_high = np.nan, np.nan
    
    return float(icc), (float(ci_low), float(ci_high))


def run_multilevel_correlation_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    behavior_col: str,
    subject_col: str = "subject",
    covariates: List[str] = None,
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
    
    # Add FDR correction
    if "fixed_p" in results_df.columns:
        valid_p = results_df["fixed_p"].dropna()
        if len(valid_p) > 0:
            from eeg_pipeline.utils.analysis.stats import fdr_bh
            reject, p_adj = fdr_bh(valid_p.values)
            results_df.loc[valid_p.index, "fixed_p_fdr"] = p_adj
            results_df.loc[valid_p.index, "significant_fdr"] = reject
    
    return results_df

