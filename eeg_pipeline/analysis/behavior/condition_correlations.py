"""Condition-specific correlations (pain vs non-pain trials)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from eeg_pipeline.utils.io.general import (
    get_pain_column_from_config,
    write_tsv,
)
from eeg_pipeline.analysis.behavior.core import save_correlation_results
from eeg_pipeline.analysis.behavior.core import (
    ComputationResult,
    ComputationStatus,
    CorrelationRecord,
    build_correlation_record,
    safe_correlation,
    MIN_SAMPLES_DEFAULT,
    MIN_TRIALS_PER_CONDITION,
)

if TYPE_CHECKING:
    from eeg_pipeline.analysis.behavior.core import BehaviorContext


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
    from eeg_pipeline.analysis.behavior.core import correlate_features_loop
    
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
    return pain_df, nonpain_df


###################################################################
# Condition Comparison
###################################################################


def compare_condition_correlations(
    pain_df: pd.DataFrame,
    nonpain_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compare correlations between pain and non-pain conditions."""
    if pain_df.empty or nonpain_df.empty:
        return pd.DataFrame()
    
    # Get common features
    if "identifier" in pain_df.columns:
        pain_features = set(pain_df["identifier"])
        nonpain_features = set(nonpain_df["identifier"])
    else:
        logger.warning("No identifier column; cannot compare conditions")
        return pd.DataFrame()
    
    common_features = pain_features & nonpain_features
    
    if not common_features:
        logger.warning("No common features between conditions")
        return pd.DataFrame()
    
    comparison_records = []
    
    for feature in common_features:
        pain_row = pain_df[pain_df["identifier"] == feature].iloc[0]
        nonpain_row = nonpain_df[nonpain_df["identifier"] == feature].iloc[0]
        
        r_pain = pain_row["r"]
        r_nonpain = nonpain_row["r"]
        p_pain = pain_row["p"]
        p_nonpain = nonpain_row["p"]
        n_pain = pain_row["n"]
        n_nonpain = nonpain_row["n"]
        
        # Difference
        r_diff = r_pain - r_nonpain
        
        # Fisher z-transform for comparison
        z_pain = np.arctanh(np.clip(r_pain, -0.999, 0.999))
        z_nonpain = np.arctanh(np.clip(r_nonpain, -0.999, 0.999))
        z_diff = z_pain - z_nonpain
        
        # Standard error of difference
        se_diff = np.sqrt(1/(n_pain - 3) + 1/(n_nonpain - 3)) if n_pain > 3 and n_nonpain > 3 else np.nan
        
        # Z-test for difference
        z_test = z_diff / se_diff if np.isfinite(se_diff) and se_diff > 0 else np.nan
        from scipy import stats
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_test))) if np.isfinite(z_test) else np.nan
        
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
        
        comparison_records.append({
            "identifier": feature,
            "r_pain": r_pain,
            "r_nonpain": r_nonpain,
            "r_diff": r_diff,
            "p_pain": p_pain,
            "p_nonpain": p_nonpain,
            "z_diff": z_diff,
            "z_test": z_test,
            "p_diff": p_diff,
            "n_pain": n_pain,
            "n_nonpain": n_nonpain,
            "specificity": specificity,
        })
    
    comparison_df = pd.DataFrame(comparison_records)
    
    # Add FDR for difference p-values
    from eeg_pipeline.utils.analysis.stats import fdr_bh
    if len(comparison_df) > 0 and "p_diff" in comparison_df.columns:
        valid_p = comparison_df["p_diff"].dropna()
        if len(valid_p) > 0:
            comparison_df["q_diff"] = np.nan
            comparison_df.loc[comparison_df["p_diff"].notna(), "q_diff"] = fdr_bh(
                comparison_df.loc[comparison_df["p_diff"].notna(), "p_diff"].values
            )
    
    return comparison_df


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
    temp_configs = [
        ("power", ctx.power_df, "pow_"),
        ("connectivity", ctx.connectivity_df, None),
        ("precomputed", ctx.precomputed_df, None),
        ("microstates", ctx.microstates_df, None),
    ]
    
    all_pain_temp, all_nonpain_temp = [], []
    for feat_name, feat_df, col_filter in temp_configs:
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
    
    # Feature configs: (name, dataframe, col_filter)
    feature_configs = [
        ("power", ctx.power_df, "pow_"),
        ("connectivity", ctx.connectivity_df, None),
        ("precomputed", ctx.precomputed_df, None),
        ("microstates", ctx.microstates_df, None),
    ]
    
    all_pain_dfs, all_nonpain_dfs, all_comparison_dfs = [], [], []
    
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
        
        _correlate_features_with_temperature_by_condition(
            ctx, pain_mask, nonpain_mask, min_samples, method, method_suffix
        )
        
        return ComputationResult(
            name="condition_correlations",
            status=ComputationStatus.SUCCESS,
            metadata={
                "n_pain_trials": n_pain,
                "n_nonpain_trials": n_nonpain,
                "n_pain_correlations": len(all_pain_dfs[0]) if all_pain_dfs else 0,
                "n_nonpain_correlations": len(all_nonpain_dfs[0]) if all_nonpain_dfs else 0,
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

