"""Precomputed feature correlations with behavioral measures."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.utils.io.general import (
    deriv_features_path,
    read_tsv,
)
from eeg_pipeline.utils.analysis.stats import (
    prepare_aligned_data,
)
from eeg_pipeline.analysis.behavior.core import (
    BehaviorContext,
    ComputationResult,
    ComputationStatus,
    CorrelationRecord,
    build_correlation_record,
    safe_correlation,
    MIN_SAMPLES_DEFAULT,
)


FEATURE_PATTERNS = {
    # ERD/ERS features: erds_{band}_{channel} or erds_{band}_{channel}_{window}
    "erds": re.compile(r"^erds_(\w+)_(.+)$"),
    
    # Spectral features
    "iaf": re.compile(r"^iaf_(.+)$"),  # Individual alpha frequency
    "relative_power": re.compile(r"^relative_(\w+)_(.+)$"),
    "band_ratio": re.compile(r"^ratio_(\w+)_(\w+)_(.+)$"),
    "spectral_entropy": re.compile(r"^spectral_entropy_(.+)$"),
    "peak_freq": re.compile(r"^peak_freq_(\w+)_(.+)$"),
    
    # Temporal/time-domain features
    "temporal_stat": re.compile(r"^(mean|var|std|skew|kurt|median|iqr)_(.+)$"),
    "amplitude": re.compile(r"^(rms|p2p|line_length|nle)_(.+)$"),
    "zero_cross": re.compile(r"^zero_cross_(.+)$"),
    
    # Complexity features
    "permutation_entropy": re.compile(r"^pe_(.+)$"),
    "sample_entropy": re.compile(r"^sampen_(.+)$"),
    "hjorth": re.compile(r"^hjorth_(activity|mobility|complexity)_(.+)$"),
    "lzc": re.compile(r"^lzc_(.+)$"),
    
    # Global features
    "gfp": re.compile(r"^gfp_(.+)$"),
    "global_plv": re.compile(r"^global_plv_(\w+)$"),
    "variance_explained": re.compile(r"^var_explained_(.+)$"),
    
    # ROI features
    "roi_power": re.compile(r"^roi_pow_(\w+)_(.+)$"),
    "roi_asymmetry": re.compile(r"^asymmetry_(\w+)_(.+)$"),
    "roi_erds": re.compile(r"^roi_erds_(\w+)_(.+)$"),
    
    # Microstate features
    "ms_coverage": re.compile(r"^ms_coverage_(\w+)$"),
    "ms_duration": re.compile(r"^ms_duration_(\w+)$"),
    "ms_occurrence": re.compile(r"^ms_occurrence_(\w+)$"),
    "ms_transition": re.compile(r"^ms_transition_(\w+)_(\w+)$"),
    "ms_gev": re.compile(r"^ms_gev_(\w+)$"),
}


def classify_feature(column_name: str) -> Tuple[str, Dict[str, str]]:
    """Classify a feature column name into its type and extract metadata."""
    for feature_type, pattern in FEATURE_PATTERNS.items():
        match = pattern.match(column_name)
        if match:
            groups = match.groups()
            
            # Build metadata based on feature type
            if feature_type == "erds":
                return feature_type, {"band": groups[0], "identifier": groups[1]}
            elif feature_type == "relative_power":
                return feature_type, {"band": groups[0], "channel": groups[1]}
            elif feature_type == "band_ratio":
                return feature_type, {"band1": groups[0], "band2": groups[1], "channel": groups[2]}
            elif feature_type in ("temporal_stat", "amplitude"):
                return feature_type, {"stat": groups[0], "channel": groups[1]}
            elif feature_type == "hjorth":
                return feature_type, {"param": groups[0], "channel": groups[1]}
            elif feature_type == "roi_power":
                return feature_type, {"band": groups[0], "roi": groups[1]}
            elif feature_type == "roi_asymmetry":
                return feature_type, {"band": groups[0], "pair": groups[1]}
            elif feature_type == "ms_transition":
                return feature_type, {"from_state": groups[0], "to_state": groups[1]}
            else:
                # Simple single-group patterns
                return feature_type, {"identifier": groups[0] if groups else column_name}
    
    # Unknown feature type
    return "unknown", {"identifier": column_name}


def correlate_feature_with_behavior(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    feature_name: str,
    feature_type: str,
    metadata: Dict[str, str],
    method: str = "spearman",
    min_samples: int = MIN_SAMPLES_DEFAULT,
) -> Optional[CorrelationRecord]:
    """Compute correlation between a feature and target variable."""
    r, p, n = safe_correlation(feature_values, target_values, method, min_samples)
    
    if not np.isfinite(r):
        return None
    
    # Determine identifier and band from metadata
    identifier = metadata.get("identifier", metadata.get("channel", feature_name))
    band = metadata.get("band", "N/A")
    
    return build_correlation_record(
        identifier=identifier,
        band=band,
        r=r,
        p=p,
        n=n,
        method=method,
        identifier_type=feature_type,
        analysis_type=feature_type,
    )


def _correlate_dataframe_features(
    feature_df: pd.DataFrame,
    target_values: pd.Series,
    config: Any,
    logger: logging.Logger,
    *,
    use_spearman: bool = True,
    feature_source: str = "unknown",
) -> Tuple[pd.DataFrame, List[CorrelationRecord]]:
    """Correlate all columns in a feature DataFrame with target values."""
    from eeg_pipeline.analysis.behavior.core import correlate_features_loop
    
    method = "spearman" if use_spearman else "pearson"
    min_samples = int(config.get("behavior_analysis.statistics.min_samples_channel", MIN_SAMPLES_DEFAULT))
    
    records, results_df = correlate_features_loop(
        feature_df=feature_df,
        target_values=target_values,
        method=method,
        min_samples=min_samples,
        logger=logger,
        identifier_type="feature",
        analysis_type=feature_source,
        feature_classifier=classify_feature,
    )
    
    return results_df, records


def correlate_precomputed_features(
    subject: str,
    deriv_root: Path,
    target_values: pd.Series,
    config: Any,
    logger: logging.Logger,
    *,
    use_spearman: bool = True,
) -> Tuple[pd.DataFrame, List[CorrelationRecord]]:
    """Correlate all precomputed features with behavior."""
    features_dir = deriv_features_path(deriv_root, subject)
    precomputed_path = features_dir / "features_precomputed.tsv"
    
    if not precomputed_path.exists():
        logger.warning(f"Precomputed features not found: {precomputed_path}")
        return pd.DataFrame(), []
    
    precomputed_df = read_tsv(precomputed_path)
    
    if precomputed_df.empty:
        logger.warning("Precomputed features file is empty")
        return pd.DataFrame(), []
    
    method = "spearman" if use_spearman else "pearson"
    min_samples = int(config.get("behavior_analysis.statistics.min_samples_channel", MIN_SAMPLES_DEFAULT))
    
    # Align target with features
    n_features = len(precomputed_df)
    n_targets = len(target_values)
    
    if n_features != n_targets:
        logger.warning(
            f"Length mismatch: precomputed={n_features}, targets={n_targets}. "
            "Using minimum length."
        )
        n_use = min(n_features, n_targets)
        precomputed_df = precomputed_df.iloc[:n_use]
        target_values = target_values.iloc[:n_use]
    
    target_arr = target_values.to_numpy()
    records: List[CorrelationRecord] = []
    
    logger.info(f"Correlating {len(precomputed_df.columns)} precomputed features...")
    
    # Group features by type for organized output
    feature_groups: Dict[str, List[CorrelationRecord]] = {}
    
    for col in precomputed_df.columns:
        feature_values = precomputed_df[col].to_numpy()
        feature_type, metadata = classify_feature(col)
        
        record = correlate_feature_with_behavior(
            feature_values=feature_values,
            target_values=target_arr,
            feature_name=col,
            feature_type=feature_type,
            metadata=metadata,
            method=method,
            min_samples=min_samples,
        )
        
        if record is not None:
            records.append(record)
            
            if feature_type not in feature_groups:
                feature_groups[feature_type] = []
            feature_groups[feature_type].append(record)
    
    # Log summary by feature type
    for ftype, recs in feature_groups.items():
        n_sig = sum(1 for r in recs if r.is_significant)
        logger.info(f"  {ftype}: {len(recs)} features, {n_sig} significant (p<0.05)")
    
    # Convert to DataFrame
    if not records:
        return pd.DataFrame(), []
    
    results_df = pd.DataFrame([r.to_dict() for r in records])
    
    return results_df, records


def correlate_microstate_features(
    subject: str,
    deriv_root: Path,
    target_values: pd.Series,
    config: Any,
    logger: logging.Logger,
    *,
    use_spearman: bool = True,
) -> Tuple[pd.DataFrame, List[CorrelationRecord]]:
    """Correlate microstate features with behavior."""
    features_dir = deriv_features_path(deriv_root, subject)
    ms_path = features_dir / "features_microstates.tsv"
    
    if not ms_path.exists():
        logger.warning(f"Microstate features not found: {ms_path}")
        return pd.DataFrame(), []
    
    ms_df = read_tsv(ms_path)
    
    if ms_df.empty:
        logger.warning("Microstate features file is empty")
        return pd.DataFrame(), []
    
    method = "spearman" if use_spearman else "pearson"
    min_samples = int(config.get("behavior_analysis.statistics.min_samples_channel", MIN_SAMPLES_DEFAULT))
    
    # Align
    n_ms = len(ms_df)
    n_targets = len(target_values)
    
    if n_ms != n_targets:
        logger.warning(f"Length mismatch: microstates={n_ms}, targets={n_targets}")
        n_use = min(n_ms, n_targets)
        ms_df = ms_df.iloc[:n_use]
        target_values = target_values.iloc[:n_use]
    
    target_arr = target_values.to_numpy()
    records: List[CorrelationRecord] = []
    
    logger.info(f"Correlating {len(ms_df.columns)} microstate features...")
    
    for col in ms_df.columns:
        feature_values = ms_df[col].to_numpy()
        feature_type, metadata = classify_feature(col)
        
        # Override type to microstate
        if not feature_type.startswith("ms_"):
            feature_type = "microstate"
        
        record = correlate_feature_with_behavior(
            feature_values=feature_values,
            target_values=target_arr,
            feature_name=col,
            feature_type=feature_type,
            metadata=metadata,
            method=method,
            min_samples=min_samples,
        )
        
        if record is not None:
            records.append(record)
    
    n_sig = sum(1 for r in records if r.is_significant)
    logger.info(f"  Microstates: {len(records)} features, {n_sig} significant")
    
    if not records:
        return pd.DataFrame(), []
    
    results_df = pd.DataFrame([r.to_dict() for r in records])
    return results_df, records


def compute_precomputed_correlations(
    ctx: BehaviorContext,
) -> ComputationResult:
    """Compute correlations for all precomputed features."""
    if ctx.targets is None:
        return ComputationResult(
            name="precomputed_correlations",
            status=ComputationStatus.SKIPPED,
            metadata={"reason": "No target variable"},
        )
    
    method_suffix = "_spearman" if ctx.use_spearman else "_pearson"
    
    try:
        # Use precomputed_df from context if available (avoids re-reading file)
        precomp_df_result = pd.DataFrame()
        precomp_records: List[CorrelationRecord] = []
        
        if ctx.precomputed_df is not None and not ctx.precomputed_df.empty:
            precomp_df_result, precomp_records = _correlate_dataframe_features(
                feature_df=ctx.precomputed_df,
                target_values=ctx.targets,
                config=ctx.config,
                logger=ctx.logger,
                use_spearman=ctx.use_spearman,
                feature_source="precomputed",
            )
        else:
            # Fallback to file loading if not in context
            precomp_df_result, precomp_records = correlate_precomputed_features(
                subject=ctx.subject,
                deriv_root=ctx.deriv_root,
                target_values=ctx.targets,
                config=ctx.config,
                logger=ctx.logger,
                use_spearman=ctx.use_spearman,
            )
        
        # Use microstates_df from context if available
        ms_df_result = pd.DataFrame()
        ms_records: List[CorrelationRecord] = []
        
        if ctx.microstates_df is not None and not ctx.microstates_df.empty:
            ms_df_result, ms_records = _correlate_dataframe_features(
                feature_df=ctx.microstates_df,
                target_values=ctx.targets,
                config=ctx.config,
                logger=ctx.logger,
                use_spearman=ctx.use_spearman,
                feature_source="microstates",
            )
        else:
            # Fallback to file loading
            ms_df_result, ms_records = correlate_microstate_features(
                subject=ctx.subject,
                deriv_root=ctx.deriv_root,
                target_values=ctx.targets,
                config=ctx.config,
                logger=ctx.logger,
                use_spearman=ctx.use_spearman,
            )
        
        precomp_df = precomp_df_result
        ms_df = ms_df_result
        
        all_records = precomp_records + ms_records
        
        # Save rating correlations using standardized function
        from eeg_pipeline.analysis.behavior.core import save_correlation_results
        
        if not precomp_df.empty:
            output_path = ctx.stats_dir / f"corr_stats_precomputed_vs_rating{method_suffix}.tsv"
            save_correlation_results(precomp_df, output_path, apply_fdr=True, config=ctx.config, logger=ctx.logger)
            
            # Also save by feature type
            for ftype in precomp_df["analysis"].unique():
                ftype_df = precomp_df[precomp_df["analysis"] == ftype].copy()
                ftype_path = ctx.stats_dir / f"corr_stats_{ftype}_vs_rating{method_suffix}.tsv"
                save_correlation_results(ftype_df, ftype_path, apply_fdr=True, config=ctx.config, logger=ctx.logger)
        
        if not ms_df.empty:
            ms_output = ctx.stats_dir / f"corr_stats_microstates_vs_rating{method_suffix}.tsv"
            save_correlation_results(ms_df, ms_output, apply_fdr=True, config=ctx.config, logger=ctx.logger)
        
        # Temperature correlations
        temp_precomp_df = pd.DataFrame()
        temp_ms_df = pd.DataFrame()
        
        if ctx.temperature is not None and len(ctx.temperature.dropna()) >= MIN_SAMPLES_DEFAULT:
            ctx.logger.info("Computing temperature correlations for precomputed features...")
            
            if ctx.precomputed_df is not None and not ctx.precomputed_df.empty:
                temp_precomp_df, _ = _correlate_dataframe_features(
                    feature_df=ctx.precomputed_df,
                    target_values=ctx.temperature,
                    config=ctx.config,
                    logger=ctx.logger,
                    use_spearman=ctx.use_spearman,
                    feature_source="precomputed",
                )
            
            if ctx.microstates_df is not None and not ctx.microstates_df.empty:
                temp_ms_df, _ = _correlate_dataframe_features(
                    feature_df=ctx.microstates_df,
                    target_values=ctx.temperature,
                    config=ctx.config,
                    logger=ctx.logger,
                    use_spearman=ctx.use_spearman,
                    feature_source="microstates",
                )
            
            # Save temperature correlations using standardized function
            if not temp_precomp_df.empty:
                temp_output = ctx.stats_dir / f"corr_stats_precomputed_vs_temp{method_suffix}.tsv"
                save_correlation_results(temp_precomp_df, temp_output, apply_fdr=True, config=ctx.config, logger=ctx.logger)
                
                # By feature type
                for ftype in temp_precomp_df["analysis"].unique():
                    ftype_df = temp_precomp_df[temp_precomp_df["analysis"] == ftype].copy()
                    ftype_path = ctx.stats_dir / f"corr_stats_{ftype}_vs_temp{method_suffix}.tsv"
                    save_correlation_results(ftype_df, ftype_path, apply_fdr=True, config=ctx.config, logger=ctx.logger)
            
            if not temp_ms_df.empty:
                temp_ms_output = ctx.stats_dir / f"corr_stats_microstates_vs_temp{method_suffix}.tsv"
                save_correlation_results(temp_ms_df, temp_ms_output, apply_fdr=True, config=ctx.config, logger=ctx.logger)
        
        return ComputationResult(
            name="precomputed_correlations",
            status=ComputationStatus.SUCCESS,
            records=all_records,
            dataframe=pd.concat([precomp_df, ms_df], ignore_index=True) if not precomp_df.empty or not ms_df.empty else pd.DataFrame(),
            metadata={
                "n_precomputed": len(precomp_records),
                "n_microstates": len(ms_records),
                "has_temp_correlations": ctx.temperature is not None,
            },
        )
        
    except Exception as exc:
        ctx.logger.error(f"precomputed_correlations failed: {exc}")
        return ComputationResult(
            name="precomputed_correlations",
            status=ComputationStatus.FAILED,
            error=str(exc),
        )
