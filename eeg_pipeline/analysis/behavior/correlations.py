from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sm = None
    smf = None

from eeg_pipeline.utils.analysis.stats import (
    _safe_float,
    prepare_aligned_data,
    compute_correlation,
    compute_bootstrap_ci,
    get_fdr_alpha_from_config,
    compute_partial_correlations,
    compute_permutation_pvalues,
    compute_temp_permutation_pvalues,
    fdr_bh,
    CorrelationStats,
)


@dataclass
class AnalysisConfig:
    """Configuration object for behavior analysis to reduce argument passing."""
    subject: str
    config: Any
    logger: Any
    rng: np.random.Generator
    stats_dir: Optional[Path] = None
    bootstrap: int = 0
    n_perm: int = 0
    use_spearman: bool = True
    method: str = "spearman"
    min_samples_channel: int = 10
    min_samples_roi: int = 20
    groups: Optional[np.ndarray] = None


def _align_groups_to_series(
    series: pd.Series,
    groups: Optional[Union[pd.Series, np.ndarray]],
) -> Optional[np.ndarray]:
    """
    Align group labels to a cleaned/filtered series using its index.
    Returns None if alignment fails.
    """
    if groups is None:
        return None
    try:
        if isinstance(groups, pd.Series):
            aligned = groups.loc[series.index]
        else:
            arr = np.asarray(groups)
            if arr.size != len(series):
                return None
            aligned = pd.Series(arr, index=series.index)
        return aligned.to_numpy()
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _safe_import_statsmodels(logger=None):
    if sm is None or smf is None:
        if logger:
            logger.warning("statsmodels not installed; mixed-effects models unavailable.")
        return None, None
    return sm, smf


def _fit_mixedlm(
    data: pd.DataFrame,
    formula: str,
    groups: str,
    re_formula: Optional[str] = None,
    method: str = "lbfgs",
    logger=None,
):
    sm_local, smf_local = _safe_import_statsmodels(logger)
    if sm_local is None:
        return None
    try:
        model = smf_local.mixedlm(formula, data=data, groups=data[groups], re_formula=re_formula)
        result = model.fit(method=method, disp=False)
        return result
    except Exception as exc:
        if logger:
            logger.warning(f"MixedLM fit failed for {formula}: {exc}")
        return None


def _extract_fixed_effects(result) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    fe = result.params
    se = result.bse
    tvals = result.tvalues
    pvals = result.pvalues
    df = pd.DataFrame({
        "term": fe.index,
        "estimate": fe.values,
        "se": se.values,
        "t": tvals.values,
        "p": pvals.values,
    })
    return df


def _add_fdr(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if df.empty or "p" not in df.columns:
        return df
    q = fdr_bh(df["p"].to_numpy(dtype=float), alpha=alpha)
    df = df.copy()
    df["q"] = q
    return df


def _compute_partial_residuals(result, data: pd.DataFrame, predictor: str) -> Optional[pd.Series]:
    if result is None or predictor not in result.params.index:
        return None
    fitted = result.fittedvalues
    fe = result.params
    y = data[result.model.endog_names]
    term = fe[predictor] * data[predictor]
    partial_resid = y - (fitted - term)
    return partial_resid


def _prepare_mixedlm_data(x: pd.Series, y: pd.Series, groups: pd.Series, covars: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = pd.DataFrame({"x": x, "y": y, "group": groups})
    if covars is not None and not covars.empty:
        for col in covars.columns:
            df[col] = covars[col]
    return df.dropna()




def _build_temp_record_unified(
    x_values: pd.Series,
    temp_series: Optional[pd.Series],
    covariates_without_temp_df: Optional[pd.DataFrame],
    identifier: str,
    identifier_key: str,
    band: str,
    analysis_cfg: AnalysisConfig,
    groups: Optional[np.ndarray] = None,
    **extra_fields,
) -> Optional[Dict[str, Any]]:
    if temp_series is None or temp_series.empty:
        return None
    
    context_temp = f"temperature {identifier_key} {identifier} ({band})"
    x_aligned, temp_aligned, covariates_aligned, _, _ = prepare_aligned_data(
        x_values,
        temp_series,
        covariates_without_temp_df,
        min_samples=analysis_cfg.min_samples_channel if identifier_key == "channel" else analysis_cfg.min_samples_roi,
        logger=analysis_cfg.logger,
        context=context_temp,
    )
    
    if x_aligned is None or temp_aligned is None:
        return None
    
    group_source = groups if groups is not None else analysis_cfg.groups
    groups_aligned = _align_groups_to_series(x_aligned, group_source)
    
    # Use standardized safe_correlation instead of compute_correlation
    from eeg_pipeline.analysis.behavior.core import safe_correlation
    correlation_temp, p_value_temp, _ = safe_correlation(
        x_aligned, temp_aligned, 
        method=analysis_cfg.method,
        min_samples=analysis_cfg.min_samples_channel if identifier_key == "channel" else analysis_cfg.min_samples_roi
    )
    
    ci_low = ci_high = np.nan
    p_perm_temp = np.nan
    
    if analysis_cfg.bootstrap is not None and analysis_cfg.bootstrap > 0:
        ci_low, ci_high = compute_bootstrap_ci(
            x_aligned, temp_aligned, analysis_cfg.bootstrap, analysis_cfg.use_spearman, analysis_cfg.rng,
            analysis_cfg.min_samples_channel if identifier_key == "channel" else analysis_cfg.min_samples_roi,
            logger=analysis_cfg.logger, config=analysis_cfg.config, groups=groups_aligned
        )
    
    if analysis_cfg.n_perm is not None and analysis_cfg.n_perm > 0:
        p_perm_temp, _ = compute_temp_permutation_pvalues(
            x_aligned, temp_aligned, covariates_aligned, analysis_cfg.method,
            analysis_cfg.n_perm, analysis_cfg.rng, band, identifier, analysis_cfg.logger, groups=groups_aligned
        )
    
    from eeg_pipeline.analysis.behavior.core import build_correlation_record
    
    record = build_correlation_record(
        identifier=identifier,
        band=band,
        r=correlation_temp,
        p=p_value_temp,
        n=int(len(x_aligned)),
        method=analysis_cfg.method,
        ci_low=ci_low,
        ci_high=ci_high,
        p_perm=p_perm_temp,
        identifier_type=identifier_key,
        **extra_fields,
    )
    
    return record.to_dict()


def _compute_correlation_statistics(
    x_values: pd.Series,
    y_values: pd.Series,
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    covariates_df: Optional[pd.DataFrame],
    bootstrap: int,
    n_perm: int,
    use_spearman: bool,
    method: str,
    rng: np.random.Generator,
    n_eff: int,
    min_samples: int,
    logger,
    config=None,
    temp_series: Optional[pd.Series] = None,
    context: str = "",
    band: str = "",
    identifier: str = "",
    groups: Optional[np.ndarray] = None,
    mixed_effects: bool = False,
    covariates_df_full: Optional[pd.DataFrame] = None,
    mixed_effects_records: Optional[List[Dict[str, Any]]] = None,
) -> CorrelationStats:
    correlation, p_value = compute_correlation(x_aligned, y_aligned, use_spearman)
    
    (
        r_partial,
        p_partial,
        n_partial,
        r_partial_temp,
        p_partial_temp,
        n_partial_temp,
    ) = compute_partial_correlations(
        x_values, y_values, covariates_df, temp_series, method, context, logger, min_samples
    )
    
    if mixed_effects and covariates_df_full is not None:
        df_mixed = _prepare_mixedlm_data(
            x_values, y_values,
            pd.Series(groups) if groups is not None else pd.Series(np.arange(len(x_values))),
            covariates_df_full,
        )
        if not df_mixed.empty:
            fixed_terms = ["x"] + [c for c in covariates_df_full.columns if c in df_mixed.columns]
            formula = "y ~ " + " + ".join(fixed_terms)
            result = _fit_mixedlm(df_mixed, formula=formula, groups="group", re_formula=None, logger=logger)
            fe_df = _add_fdr(_extract_fixed_effects(result), alpha=get_fdr_alpha_from_config(config))
            if not fe_df.empty and "x" in fe_df["term"].values:
                row = fe_df[fe_df["term"] == "x"].iloc[0]
                r_partial = row.get("estimate", r_partial)
                p_partial = row.get("p", p_partial)
            if mixed_effects_records is not None and not fe_df.empty:
                mixed_effects_records.append(
                    {
                        "band": band,
                        "identifier": identifier,
                        "context": context,
                        "formula": formula,
                        "n": len(df_mixed),
                        "terms": fe_df.to_dict(orient="records"),
                    }
                )

    ci_low, ci_high = compute_bootstrap_ci(
        x_aligned, y_aligned, bootstrap, use_spearman, rng,
        min_samples, logger=logger, config=config, groups=groups
    )
    
    p_perm, p_partial_perm, p_partial_temp_perm = compute_permutation_pvalues(
        x_aligned, y_aligned, covariates_df, temp_series, method,
        n_perm, n_eff, rng, band, identifier, groups=groups,
    )
    
    return CorrelationStats(
        correlation=correlation,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        r_partial=r_partial,
        p_partial=p_partial,
        n_partial=n_partial,
        r_partial_temp=r_partial_temp,
        p_partial_temp=p_partial_temp,
        n_partial_temp=n_partial_temp,
        p_perm=p_perm,
        p_partial_perm=p_partial_perm,
        p_partial_temp_perm=p_partial_temp_perm,
    )


def _compute_roi_correlation_stats(
    x_values: pd.Series,
    y_values: pd.Series,
    x_aligned: np.ndarray,
    y_aligned: np.ndarray,
    covariates_df: Optional[pd.DataFrame],
    temp_series: Optional[pd.Series],
    n_eff: int,
    band: str,
    roi: str,
    context: str,
    analysis_cfg: AnalysisConfig,
    groups: Optional[np.ndarray] = None,
    mixed_effects_records: Optional[List[Dict[str, Any]]] = None,
) -> CorrelationStats:
    return _compute_correlation_statistics(
        x_values, y_values, x_aligned, y_aligned,
        covariates_df, analysis_cfg.bootstrap, analysis_cfg.n_perm, analysis_cfg.use_spearman, analysis_cfg.method,
        analysis_cfg.rng, n_eff, analysis_cfg.min_samples_roi, analysis_cfg.logger, analysis_cfg.config,
        temp_series, context, band, roi, groups=groups,
        mixed_effects=True,
        covariates_df_full=covariates_df,
        mixed_effects_records=mixed_effects_records,
    )

