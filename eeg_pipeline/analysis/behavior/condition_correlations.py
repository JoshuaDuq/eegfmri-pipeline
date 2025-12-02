"""Condition-specific correlations (pain vs non-pain trials)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from eeg_pipeline.utils.io.general import get_pain_column_from_config, write_tsv
from eeg_pipeline.utils.analysis.stats import (
    fdr_bh,
    correlation_difference_effect,
    hedges_g,
    cohens_d,
)
from eeg_pipeline.analysis.behavior.core import (
    ComputationResult,
    ComputationStatus,
    CorrelationRecord,
    build_correlation_record,
    correlate_features_loop,
    save_correlation_results,
    MIN_SAMPLES_DEFAULT,
    MIN_TRIALS_PER_CONDITION,
)

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.core import BehaviorContext


###################################################################
# Feature Configuration
###################################################################

# Centralized feature configs: (name, attr_name, col_filter)
# Used consistently across all condition correlation functions
FEATURE_CONFIGS = [
    ("power", "power_df", "pow_"),
    ("connectivity", "connectivity_df", None),
    ("precomputed", "precomputed_df", None),
    ("microstates", "microstates_df", None),
    ("aperiodic", "aperiodic_df", "aper_"),
]


def _get_feature_dfs(ctx: "BehaviorContext") -> List[Tuple[str, Optional[pd.DataFrame], Optional[str]]]:
    """Get all feature DataFrames from context with their configs."""
    return [
        (name, getattr(ctx, attr, None), col_filter)
        for name, attr, col_filter in FEATURE_CONFIGS
    ]


###################################################################
# Trial Splitting
###################################################################


def split_data_by_condition(
    aligned_events: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split trials into pain and non-pain conditions."""
    pain_col = get_pain_column_from_config(config, aligned_events)
    
    if pain_col is None:
        # Try common column names
        for col in ["pain_binary", "is_pain", "pain", "condition"]:
            if col in aligned_events.columns:
                pain_col = col
                break
    
    if pain_col is None or pain_col not in aligned_events.columns:
        logger.warning("No pain column found; cannot split by condition")
        return np.array([]), np.array([]), 0, 0
    
    pain_series = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    
    pain_mask = (pain_series == 1).values
    nonpain_mask = (pain_series == 0).values
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int(nonpain_mask.sum())
    
    logger.info(f"Condition split: {n_pain} pain trials, {n_nonpain} non-pain trials")
    
    return pain_mask, nonpain_mask, n_pain, n_nonpain


###################################################################
# Condition-Specific Correlation Functions
###################################################################


def _correlate_features_for_condition(
    feature_df: pd.DataFrame,
    target_values: pd.Series,
    condition_mask: np.ndarray,
    condition_name: str,
    min_samples: int,
    method: str,
    logger: logging.Logger,
    feature_prefix: str = "",
) -> Tuple[pd.DataFrame, List[CorrelationRecord]]:
    """Correlate features with target within a specific condition."""
    if feature_df is None or feature_df.empty:
        return pd.DataFrame(), []
    
    analysis_type = f"{feature_prefix}_{condition_name}" if feature_prefix else condition_name
    identifier_type = feature_prefix or "feature"
    
    records, results_df = correlate_features_loop(
        feature_df=feature_df,
        target_values=target_values,
        method=method,
        min_samples=min_samples,
        logger=logger,
        condition_mask=condition_mask,
        identifier_type=identifier_type,
        analysis_type=analysis_type,
    )
    
    if not results_df.empty:
        results_df["condition"] = condition_name
    
    return results_df, records


def _correlate_df_by_condition(
    feature_df: Optional[pd.DataFrame],
    targets: pd.Series,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    feature_name: str,
    min_samples: int,
    method: str,
    logger: logging.Logger,
    col_filter: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correlate any feature DataFrame by condition."""
    if feature_df is None or feature_df.empty or targets is None:
        return pd.DataFrame(), pd.DataFrame()
    
    if col_filter:
        cols = [c for c in feature_df.columns if str(c).startswith(col_filter)]
        if not cols:
            return pd.DataFrame(), pd.DataFrame()
        feature_df = feature_df[cols]
    
    pain_df, _ = _correlate_features_for_condition(
        feature_df, targets, pain_mask, "pain", min_samples, method, logger, feature_name
    )
    nonpain_df, _ = _correlate_features_for_condition(
        feature_df, targets, nonpain_mask, "nonpain", min_samples, method, logger, feature_name
    )
    # Ensure identifier column exists for downstream comparison
    for df in (pain_df, nonpain_df):
        if not df.empty and "identifier" not in df.columns:
            # Try common identifier columns; fallback to row index
            for cand in ("channel", "roi", "feature", "name", feature_name):
                if cand in df.columns:
                    df["identifier"] = df[cand]
                    break
            else:
                # Last resort: use index
                df["identifier"] = df.index.astype(str)
    return pain_df, nonpain_df


###################################################################
# Partial Correlations (controlling for temperature)
###################################################################


def compute_partial_correlations_controlling_temperature(
    feature_df: pd.DataFrame,
    targets: pd.Series,
    temperature: pd.Series,
    condition_mask: np.ndarray,
    condition_name: str,
    min_samples: int,
    method: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute partial correlations between features and rating, controlling for temperature.
    
    This answers: "Is the feature-rating relationship independent of temperature?"
    """
    from scipy import stats as scipy_stats
    from eeg_pipeline.utils.analysis.stats import partial_corr_xy_given_Z
    
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()
    
    # Apply condition mask (convert boolean mask to integer indices for iloc)
    if condition_mask.dtype == bool:
        idx = np.where(condition_mask)[0]
    else:
        idx = condition_mask
    
    feat_cond = feature_df.iloc[idx].reset_index(drop=True)
    target_series = pd.Series(targets.iloc[idx].values, name="rating")
    temp_series = pd.Series(temperature.iloc[idx].values, name="temperature")
    temp_df = temp_series.to_frame()
    
    records = []
    for col in feat_cond.columns:
        feat_series = pd.to_numeric(feat_cond[col], errors="coerce")
        feat_series.name = "feature"
        
        # Simple correlation
        valid = np.isfinite(feat_series.values) & np.isfinite(target_series.values)
        n_valid = int(valid.sum())
        if n_valid < min_samples:
            continue
        
        if method == "spearman":
            r, p = scipy_stats.spearmanr(feat_series.values[valid], target_series.values[valid])
        else:
            r, p = scipy_stats.pearsonr(feat_series.values[valid], target_series.values[valid])
        
        # Partial correlation controlling for temperature
        r_partial, p_partial, n_partial = partial_corr_xy_given_Z(
            feat_series, target_series, temp_df, method
        )
        
        # Change in correlation when controlling for temperature
        r_change = r - r_partial if np.isfinite(r_partial) else np.nan
        
        records.append({
            "feature": col,
            "condition": condition_name,
            "r": float(r),
            "p": float(p),
            "r_partial_temp": float(r_partial),
            "p_partial_temp": float(p_partial),
            "r_change_temp": float(r_change),
            "n": n_valid,
            "n_partial": n_partial,
            "method": method,
        })
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # FDR correction for partial p-values
    if "p_partial_temp" in df.columns:
        valid_p = df["p_partial_temp"].notna()
        if valid_p.any():
            df.loc[valid_p, "q_partial_temp"] = fdr_bh(df.loc[valid_p, "p_partial_temp"].values)
    
    # Flag features where temperature explains the relationship
    # (significant simple correlation but non-significant partial)
    df["temp_mediated"] = (df["p"] < 0.05) & (df["p_partial_temp"] >= 0.05)
    
    n_mediated = df["temp_mediated"].sum()
    if n_mediated > 0:
        logger.info(f"  {condition_name}: {n_mediated} features potentially mediated by temperature")
    
    return df


###################################################################
# Condition Comparison
###################################################################


def compare_condition_correlations(
    pain_df: pd.DataFrame,
    nonpain_df: pd.DataFrame,
    logger: logging.Logger,
    compute_effect_sizes: bool = True,
) -> pd.DataFrame:
    """Compare correlations between pain and non-pain conditions.
    
    Args:
        pain_df: DataFrame with pain condition correlations
        nonpain_df: DataFrame with non-pain condition correlations
        logger: Logger instance
        compute_effect_sizes: If True, compute Cohen's q for correlation differences
        
    Returns:
        DataFrame with comparison statistics including effect sizes
    """
    if pain_df.empty or nonpain_df.empty:
        return pd.DataFrame()
    
    # Resolve identifier column (supports legacy naming per identifier_type)
    # Include all possible feature type names that might be used as identifier columns
    id_candidates = [
        "identifier", "feature", "channel", "roi", "name",
        "power", "connectivity", "precomputed", "microstates", "aperiodic",
        "temporal", "complexity", "erds", "itpc", "pac", "gfp",
    ]
    id_col = next((c for c in id_candidates if c in pain_df.columns and c in nonpain_df.columns), None)
    if id_col is None:
        logger.warning("No identifier column; cannot compare conditions")
        return pd.DataFrame()
    pain_features = set(pain_df[id_col])
    nonpain_features = set(nonpain_df[id_col])
    
    common_features = pain_features & nonpain_features
    
    if not common_features:
        logger.warning("No common features between conditions")
        return pd.DataFrame()
    
    comparison_records = []
    
    for feature in common_features:
        pain_row = pain_df[pain_df[id_col] == feature].iloc[0]
        nonpain_row = nonpain_df[nonpain_df[id_col] == feature].iloc[0]
        
        r_pain = float(pain_row.get("r", pain_row.get("correlation", np.nan)))
        r_nonpain = float(nonpain_row.get("r", nonpain_row.get("correlation", np.nan)))
        
        # Get p-value with fallback to alternative column names
        p_pain = float(pain_row.get("p", pain_row.get("p_value", pain_row.get("p_perm", np.nan))))
        p_nonpain = float(nonpain_row.get("p", nonpain_row.get("p_value", nonpain_row.get("p_perm", np.nan))))
        
        n_pain = int(pain_row.get("n", pain_row.get("n_valid", 0)))
        n_nonpain = int(nonpain_row.get("n", nonpain_row.get("n_valid", 0)))
        
        # Compute effect sizes using centralized function
        if compute_effect_sizes:
            effect_stats = correlation_difference_effect(
                r_pain, r_nonpain, n_pain, n_nonpain
            )
        else:
            effect_stats = {
                "z_stat": np.nan,
                "p_value": np.nan,
                "cohens_q": np.nan,
                "r_diff": np.nan,
            }
        
        # Determine specificity
        sig_pain = p_pain < 0.05
        sig_nonpain = p_nonpain < 0.05
        
        if sig_pain and not sig_nonpain:
            specificity = "pain_specific"
        elif sig_nonpain and not sig_pain:
            specificity = "nonpain_specific"
        elif sig_pain and sig_nonpain:
            specificity = "both"
        else:
            specificity = "neither"
        
        record = {
            "identifier": feature,
            "r_pain": r_pain,
            "r_nonpain": r_nonpain,
            "r_diff": effect_stats.get("r_diff", r_pain - r_nonpain),
            "p_pain": p_pain,
            "p_nonpain": p_nonpain,
            "z_diff": effect_stats["z_stat"],
            "z_test": effect_stats["z_stat"],
            "p_diff": effect_stats["p_value"],
            "n_pain": n_pain,
            "n_nonpain": n_nonpain,
            "specificity": specificity,
        }
        
        # Add effect size if computed
        if compute_effect_sizes:
            record["cohens_q"] = effect_stats["cohens_q"]
            # Interpret Cohen's q: 0.1=small, 0.3=medium, 0.5=large
            q_val = effect_stats["cohens_q"]
            try:
                q_val_float = float(q_val) if q_val is not None else np.nan
            except (ValueError, TypeError):
                q_val_float = np.nan
            if np.isfinite(q_val_float):
                if q_val_float >= 0.5:
                    record["effect_magnitude"] = "large"
                elif q_val_float >= 0.3:
                    record["effect_magnitude"] = "medium"
                elif q_val_float >= 0.1:
                    record["effect_magnitude"] = "small"
                else:
                    record["effect_magnitude"] = "negligible"
            else:
                record["effect_magnitude"] = "NA"
        
        comparison_records.append(record)
    
    comparison_df = pd.DataFrame(comparison_records)
    
    # Add FDR for difference p-values
    if len(comparison_df) > 0 and "p_diff" in comparison_df.columns:
        valid_p = comparison_df["p_diff"].dropna()
        if len(valid_p) > 0:
            comparison_df["q_diff"] = np.nan
            comparison_df.loc[comparison_df["p_diff"].notna(), "q_diff"] = fdr_bh(
                comparison_df.loc[comparison_df["p_diff"].notna(), "p_diff"].values
            )
    
    # Log effect size summary
    if compute_effect_sizes and "effect_magnitude" in comparison_df.columns:
        effect_counts = comparison_df["effect_magnitude"].value_counts()
        logger.info(f"  Effect size distribution: {effect_counts.to_dict()}")
    
    return comparison_df


###################################################################
# Feature Distribution Effect Sizes
###################################################################


def compute_condition_effect_sizes(
    feature_df: pd.DataFrame,
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute effect sizes (Hedges' g) for features between pain vs non-pain conditions.
    
    This measures how much the feature values differ between conditions,
    independent of correlation with rating.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature DataFrame with one row per trial
    pain_mask, nonpain_mask : np.ndarray
        Boolean masks for condition trials
    logger : logging.Logger
        Logger instance
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, hedges_g, cohens_d, effect_magnitude, p_ttest
    """
    from scipy import stats
    
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int(nonpain_mask.sum())
    
    if n_pain < 5 or n_nonpain < 5:
        logger.warning(f"Too few trials: {n_pain} pain, {n_nonpain} non-pain")
        return pd.DataFrame()
    
    records = []
    
    for col in feature_df.columns:
        try:
            vals = pd.to_numeric(feature_df[col], errors="coerce").values
            pain_vals = vals[pain_mask]
            nonpain_vals = vals[nonpain_mask]
            
            # Filter valid values
            pain_valid = pain_vals[np.isfinite(pain_vals)]
            nonpain_valid = nonpain_vals[np.isfinite(nonpain_vals)]
            
            if len(pain_valid) < 5 or len(nonpain_valid) < 5:
                continue
            
            # Effect sizes
            g = hedges_g(pain_valid, nonpain_valid)
            d = cohens_d(pain_valid, nonpain_valid)
            
            # T-test
            _, p_ttest = stats.ttest_ind(pain_valid, nonpain_valid, equal_var=False)
            
            # Mean difference
            mean_diff = float(np.mean(pain_valid) - np.mean(nonpain_valid))
            
            # Effect magnitude interpretation
            abs_g = abs(g) if np.isfinite(g) else 0
            if abs_g >= 0.8:
                magnitude = "large"
            elif abs_g >= 0.5:
                magnitude = "medium"
            elif abs_g >= 0.2:
                magnitude = "small"
            else:
                magnitude = "negligible"
            
            records.append({
                "feature": col,
                "mean_pain": float(np.mean(pain_valid)),
                "mean_nonpain": float(np.mean(nonpain_valid)),
                "mean_diff": mean_diff,
                "std_pain": float(np.std(pain_valid, ddof=1)),
                "std_nonpain": float(np.std(nonpain_valid, ddof=1)),
                "hedges_g": g,
                "cohens_d": d,
                "effect_magnitude": magnitude,
                "p_ttest": float(p_ttest),
                "n_pain": len(pain_valid),
                "n_nonpain": len(nonpain_valid),
            })
        except Exception:
            continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # FDR correction
    if "p_ttest" in df.columns:
        valid_p = df["p_ttest"].notna()
        if valid_p.any():
            df.loc[valid_p, "q_ttest"] = fdr_bh(df.loc[valid_p, "p_ttest"].values)
    
    # Sort by effect size
    df = df.assign(abs_g=df["hedges_g"].abs()).sort_values("abs_g", ascending=False).drop(columns=["abs_g"])
    
    # Log summary
    n_large = (df["effect_magnitude"] == "large").sum()
    n_medium = (df["effect_magnitude"] == "medium").sum()
    logger.info(f"  Effect sizes: {n_large} large, {n_medium} medium out of {len(df)} features")
    
    return df


###################################################################
# Main Entry Point
###################################################################


def _correlate_features_with_temperature_by_condition(
    ctx: "BehaviorContext",
    pain_mask: np.ndarray,
    nonpain_mask: np.ndarray,
    min_samples: int,
    method: str,
    method_suffix: str,
) -> None:
    """Correlate features with temperature separately by condition."""
    if ctx.temperature is None or len(ctx.temperature.dropna()) < MIN_SAMPLES_DEFAULT:
        ctx.logger.info("No temperature data for condition-specific correlations")
        return
    
    ctx.logger.info("Computing temperature correlations by condition...")
    
    # Use centralized feature configs
    all_pain_temp, all_nonpain_temp = [], []
    for feat_name, feat_df, col_filter in _get_feature_dfs(ctx):
        pain_df, nonpain_df = _correlate_df_by_condition(
            feat_df, ctx.temperature, pain_mask, nonpain_mask,
            feat_name, min_samples, method, ctx.logger, col_filter
        )
        if not pain_df.empty:
            save_correlation_results(
                pain_df, 
                ctx.stats_dir / f"corr_stats_{feat_name}_vs_temp_pain{method_suffix}.tsv",
                apply_fdr=True, config=ctx.config, logger=ctx.logger
            )
            all_pain_temp.append(pain_df)
        if not nonpain_df.empty:
            save_correlation_results(
                nonpain_df,
                ctx.stats_dir / f"corr_stats_{feat_name}_vs_temp_nonpain{method_suffix}.tsv",
                apply_fdr=True, config=ctx.config, logger=ctx.logger
            )
            all_nonpain_temp.append(nonpain_df)
    
    if all_pain_temp:
        write_tsv(pd.concat(all_pain_temp, ignore_index=True), ctx.stats_dir / f"corr_stats_all_vs_temp_pain{method_suffix}.tsv")
    if all_nonpain_temp:
        write_tsv(pd.concat(all_nonpain_temp, ignore_index=True), ctx.stats_dir / f"corr_stats_all_vs_temp_nonpain{method_suffix}.tsv")


def compute_condition_correlations(ctx: "BehaviorContext") -> ComputationResult:
    """Compute correlations separately for pain and non-pain trials."""
    logger = ctx.logger
    method = "spearman" if ctx.use_spearman else "pearson"
    method_suffix = f"_{method}"
    
    if ctx.targets is None:
        return ComputationResult(
            name="condition_correlations",
            status=ComputationStatus.SKIPPED,
            metadata={"reason": "No target variable"},
        )
    
    if ctx.aligned_events is None:
        return ComputationResult(
            name="condition_correlations",
            status=ComputationStatus.SKIPPED,
            metadata={"reason": "No aligned events"},
        )
    
    # Split once, use everywhere
    pain_mask, nonpain_mask, n_pain, n_nonpain = split_data_by_condition(
        ctx.aligned_events, ctx.config, logger
    )
    
    if n_pain < MIN_TRIALS_PER_CONDITION and n_nonpain < MIN_TRIALS_PER_CONDITION:
        return ComputationResult(
            name="condition_correlations",
            status=ComputationStatus.SKIPPED,
            metadata={"reason": f"Insufficient trials (pain={n_pain}, nonpain={n_nonpain})"},
        )
    
    logger.info("Computing condition-specific correlations (pain vs non-pain)...")
    min_samples = int(ctx.config.get("behavior_analysis.statistics.min_samples_channel", MIN_SAMPLES_DEFAULT))
    
    # Use centralized feature configs
    feature_configs = _get_feature_dfs(ctx)
    
    all_pain_dfs, all_nonpain_dfs, all_comparison_dfs, all_effect_dfs = [], [], [], []
    
    try:
        for feat_name, feat_df, col_filter in feature_configs:
            if feat_df is None or feat_df.empty:
                continue
            
            logger.info(f"  Correlating {feat_name} features by condition...")
            pain_df, nonpain_df = _correlate_df_by_condition(
                feat_df, ctx.targets, pain_mask, nonpain_mask,
                feat_name, min_samples, method, logger, col_filter
            )
            
            if not pain_df.empty:
                save_correlation_results(
                    pain_df,
                    ctx.stats_dir / f"corr_stats_{feat_name}_pain{method_suffix}.tsv",
                    apply_fdr=True, config=ctx.config, logger=ctx.logger
                )
                all_pain_dfs.append(pain_df)
            
            if not nonpain_df.empty:
                save_correlation_results(
                    nonpain_df,
                    ctx.stats_dir / f"corr_stats_{feat_name}_nonpain{method_suffix}.tsv",
                    apply_fdr=True, config=ctx.config, logger=ctx.logger
                )
                all_nonpain_dfs.append(nonpain_df)
            
            comp_df = compare_condition_correlations(pain_df, nonpain_df, logger)
            if not comp_df.empty:
                write_tsv(comp_df, ctx.stats_dir / f"corr_stats_{feat_name}_condition_comparison{method_suffix}.tsv")
                all_comparison_dfs.append(comp_df)
            
            # Compute effect sizes for feature distributions (pain vs nonpain)
            filtered_df = feat_df
            if col_filter:
                cols = [c for c in feat_df.columns if str(c).startswith(col_filter)]
                if cols:
                    filtered_df = feat_df[cols]
            
            effect_df = compute_condition_effect_sizes(filtered_df, pain_mask, nonpain_mask, logger)
            if not effect_df.empty:
                effect_df["feature_type"] = feat_name
                write_tsv(effect_df, ctx.stats_dir / f"effect_sizes_{feat_name}_pain_vs_nonpain.tsv")
                all_effect_dfs.append(effect_df)
        
        # Combined summaries
        if all_pain_dfs:
            combined_pain = pd.concat(all_pain_dfs, ignore_index=True)
            write_tsv(combined_pain, ctx.stats_dir / f"corr_stats_all_pain{method_suffix}.tsv")
            logger.info(f"  Pain: {len(combined_pain)} correlations, {int((combined_pain['q'] < 0.05).sum())} significant")
        
        if all_nonpain_dfs:
            combined_nonpain = pd.concat(all_nonpain_dfs, ignore_index=True)
            write_tsv(combined_nonpain, ctx.stats_dir / f"corr_stats_all_nonpain{method_suffix}.tsv")
            logger.info(f"  Non-pain: {len(combined_nonpain)} correlations, {int((combined_nonpain['q'] < 0.05).sum())} significant")
        
        if all_comparison_dfs:
            combined_comparison = pd.concat(all_comparison_dfs, ignore_index=True)
            write_tsv(combined_comparison, ctx.stats_dir / f"corr_stats_condition_comparison_all{method_suffix}.tsv")
            logger.info(f"  Condition specificity: {combined_comparison['specificity'].value_counts().to_dict()}")
        
        # Combined effect sizes
        if all_effect_dfs:
            combined_effects = pd.concat(all_effect_dfs, ignore_index=True)
            write_tsv(combined_effects, ctx.stats_dir / "effect_sizes_all_pain_vs_nonpain.tsv")
            n_large = (combined_effects["effect_magnitude"] == "large").sum()
            n_medium = (combined_effects["effect_magnitude"] == "medium").sum()
            logger.info(f"  Effect sizes (pain vs nonpain): {n_large} large, {n_medium} medium out of {len(combined_effects)}")
        
        _correlate_features_with_temperature_by_condition(
            ctx, pain_mask, nonpain_mask, min_samples, method, method_suffix
        )
        
        # Compute partial correlations controlling for temperature
        n_partial = 0
        if ctx.temperature is not None and len(ctx.temperature.dropna()) >= MIN_SAMPLES_DEFAULT:
            logger.info("Computing partial correlations controlling for temperature...")
            all_partial_dfs = []
            for feat_name, feat_df, col_filter in feature_configs:
                if feat_df is None or feat_df.empty:
                    continue
                
                filtered_df = feat_df
                if col_filter:
                    cols = [c for c in feat_df.columns if str(c).startswith(col_filter)]
                    if cols:
                        filtered_df = feat_df[cols]
                
                # Pain condition
                pain_partial = compute_partial_correlations_controlling_temperature(
                    filtered_df, ctx.targets, ctx.temperature, pain_mask,
                    "pain", min_samples, method, logger
                )
                if not pain_partial.empty:
                    pain_partial["feature_type"] = feat_name
                    all_partial_dfs.append(pain_partial)
                
                # Non-pain condition
                nonpain_partial = compute_partial_correlations_controlling_temperature(
                    filtered_df, ctx.targets, ctx.temperature, nonpain_mask,
                    "nonpain", min_samples, method, logger
                )
                if not nonpain_partial.empty:
                    nonpain_partial["feature_type"] = feat_name
                    all_partial_dfs.append(nonpain_partial)
            
            if all_partial_dfs:
                combined_partial = pd.concat(all_partial_dfs, ignore_index=True)
                write_tsv(combined_partial, ctx.stats_dir / f"partial_corr_controlling_temp{method_suffix}.tsv")
                n_partial = len(combined_partial)
                n_mediated = combined_partial["temp_mediated"].sum()
                logger.info(f"  Partial correlations: {n_partial} computed, {n_mediated} potentially temperature-mediated")
        
        return ComputationResult(
            name="condition_correlations",
            status=ComputationStatus.SUCCESS,
            metadata={
                "n_pain_trials": n_pain,
                "n_nonpain_trials": n_nonpain,
                "n_pain_correlations": sum(len(df) for df in all_pain_dfs),
                "n_nonpain_correlations": sum(len(df) for df in all_nonpain_dfs),
                "n_effect_sizes": sum(len(df) for df in all_effect_dfs),
                "n_partial_correlations": n_partial,
                "has_temp_correlations": ctx.temperature is not None,
            },
        )
        
    except Exception as exc:
        logger.error(f"condition_correlations failed: {exc}")
        return ComputationResult(
            name="condition_correlations",
            status=ComputationStatus.FAILED,
            error=str(exc),
        )
