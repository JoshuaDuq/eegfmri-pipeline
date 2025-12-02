"""
Behavior Analysis Pipeline (Canonical)
======================================

Pipeline class for EEG-behavior correlation analysis.
This module provides the PipelineBase subclass for behavior analysis.

The actual statistical routines remain in analysis/behavior/:
- correlations.py: Enhanced correlations with controls
- feature_correlator.py: Unified feature correlator
- cluster_tests.py: Cluster permutation tests
- mixed_effects.py: Mixed-effects models
- temporal.py: Time-frequency correlations
- condition.py: Pain vs non-pain comparison

Usage:
    pipeline = BehaviorPipeline(config=config)
    pipeline.process_subject("0001", "thermalactive")
    
    # Or batch processing
    pipeline.run_batch(["0001", "0002"])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.pipelines.base import PipelineBase


###################################################################
# Configuration
###################################################################


@dataclass
class BehaviorPipelineConfig:
    method: str = "spearman"
    min_samples: int = 10
    control_temperature: bool = True
    control_trial_order: bool = True
    compute_change_scores: bool = True
    compute_pain_sensitivity: bool = True
    compute_reliability: bool = True
    
    run_condition_comparison: bool = True
    run_temporal_correlations: bool = True
    run_cluster_tests: bool = False
    run_mediation: bool = False
    run_mixed_effects: bool = False
    
    fdr_alpha: float = 0.05
    n_permutations: int = 1000
    n_jobs: int = -1
    
    @classmethod
    def from_config(cls, config: Any) -> "BehaviorPipelineConfig":
        from eeg_pipeline.utils.config.loader import get_config_value
        return cls(
            method=get_config_value(config, "behavior_analysis.statistics.correlation_method", "spearman"),
            min_samples=int(get_config_value(config, "behavior_analysis.min_samples.default", 10)),
            fdr_alpha=float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", 0.05)),
            n_permutations=int(get_config_value(config, "behavior_analysis.statistics.n_permutations", 1000)),
            run_temporal_correlations=get_config_value(config, "behavior_analysis.time_frequency_heatmap.enabled", True),
            run_cluster_tests=get_config_value(config, "behavior_analysis.cluster_correction.enabled", False),
            run_mediation=get_config_value(config, "behavior_analysis.mediation.enabled", False),
            run_mixed_effects=get_config_value(config, "behavior_analysis.mixed_effects.enabled", False),
            n_jobs=int(get_config_value(config, "behavior_analysis.n_jobs", -1)),
        )


@dataclass
class BehaviorPipelineResults:
    subject: str
    correlations: Optional[pd.DataFrame] = None
    pain_sensitivity: Optional[pd.DataFrame] = None
    condition_effects: Optional[pd.DataFrame] = None
    mediation: Optional[pd.DataFrame] = None
    mixed_effects: Optional[pd.DataFrame] = None
    cluster: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary(self) -> Dict[str, Any]:
        s = {"subject": self.subject}
        if self.correlations is not None:
            s["n_features"] = len(self.correlations)
            s["n_sig_raw"] = int((self.correlations["p_raw"] < 0.05).sum())
            s["n_sig_controlled"] = int((self.correlations["p_partial_temp"].fillna(1) < 0.05).sum())
        if self.condition_effects is not None:
            s["n_condition_effects"] = len(self.condition_effects)
            s["n_large_effects"] = int((self.condition_effects["hedges_g"].abs() >= 0.8).sum())
        return s


###################################################################
# Helper Functions
###################################################################


def _combine_features(ctx: BehaviorContext) -> pd.DataFrame:
    dfs = []
    for name, df in [
        ("power", ctx.power_df),
        ("connectivity", ctx.connectivity_df),
        ("microstates", ctx.microstates_df),
        ("aperiodic", ctx.aperiodic_df),
        ("itpc", ctx.itpc_df),
        ("pac", ctx.pac_df),
        ("precomputed", ctx.precomputed_df),
    ]:
        if df is not None and not df.empty:
            if not any(str(c).startswith(name) for c in df.columns):
                df = df.add_prefix(f"{name}_")
            dfs.append(df)
    
    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()


###################################################################
# Pipeline Stages
###################################################################


def _stage_load(ctx: BehaviorContext) -> bool:
    if not ctx.load_data():
        ctx.logger.warning("Failed to load data")
        return False
    
    ctx.logger.info(f"Loaded {ctx.n_trials} trials")
    return True


def _stage_correlate(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
) -> tuple:
    from eeg_pipeline.analysis.behavior import (
        run_pain_sensitivity_correlations,
        run_unified_feature_correlations,
    )
    from eeg_pipeline.utils.io.general import read_tsv
    
    ctx.logger.info("Running unified feature correlator...")
    result = run_unified_feature_correlations(ctx)
    
    corr_df = pd.DataFrame()
    if result.status.name == "SUCCESS":
        combined_path = ctx.stats_dir / "corr_stats_all_features_vs_rating.tsv"
        if combined_path.exists():
            corr_df = read_tsv(combined_path)
            ctx.logger.info(f"Loaded {len(corr_df)} correlation results from unified correlator")
    
    psi_df = pd.DataFrame()
    if config.compute_pain_sensitivity and ctx.temperature is not None:
        features = _combine_features(ctx)
        if not features.empty:
            psi_df = run_pain_sensitivity_correlations(
                features_df=features,
                ratings=ctx.targets,
                temperatures=ctx.temperature,
                method=config.method,
                min_samples=config.min_samples,
                logger=ctx.logger,
            )
    
    return corr_df, psi_df


def _stage_condition(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
) -> pd.DataFrame:
    from eeg_pipeline.analysis.behavior import split_by_condition, compute_condition_effects
    
    if ctx.aligned_events is None:
        return pd.DataFrame()
    
    pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(
        ctx.aligned_events, ctx.config, ctx.logger
    )
    
    if n_pain < 5 or n_nonpain < 5:
        ctx.logger.warning(f"Insufficient trials: {n_pain} pain, {n_nonpain} non-pain")
        return pd.DataFrame()
    
    features = _combine_features(ctx)
    if features.empty:
        return pd.DataFrame()
    
    return compute_condition_effects(
        features, pain_mask, nonpain_mask,
        min_samples=5, fdr_alpha=config.fdr_alpha, logger=ctx.logger,
        n_jobs=config.n_jobs, config=ctx.config
    )


def _stage_temporal(ctx: BehaviorContext) -> None:
    from eeg_pipeline.analysis.behavior import (
        compute_time_frequency_from_context,
        compute_temporal_from_context,
    )
    
    ctx.logger.info("Computing time-frequency correlations...")
    try:
        compute_time_frequency_from_context(ctx)
    except Exception as e:
        ctx.logger.warning(f"Time-frequency correlations failed: {e}")
    
    ctx.logger.info("Computing temporal correlations by condition...")
    try:
        compute_temporal_from_context(ctx)
    except Exception as e:
        ctx.logger.warning(f"Temporal correlations failed: {e}")


def _stage_cluster(ctx: BehaviorContext, config: BehaviorPipelineConfig) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior import run_cluster_test_from_context
    
    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = config.n_permutations
    
    try:
        run_cluster_test_from_context(ctx)
        return {"status": "completed"}
    except Exception as e:
        ctx.logger.warning(f"Cluster tests failed: {e}")
        return {"status": "failed", "error": str(e)}


def _stage_advanced(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
    results: BehaviorPipelineResults,
) -> None:
    from eeg_pipeline.analysis.behavior import run_mediation_analysis, run_multilevel_correlation_analysis
    
    if ctx.precomputed_df is None:
        return
    
    if config.run_mediation:
        ctx.logger.info("Running mediation analysis...")
        feature_cols = [c for c in ctx.precomputed_df.columns 
                       if c not in ["subject", "epoch", "trial", "condition", "temperature", "rating"]]
        if feature_cols:
            variances = ctx.precomputed_df[feature_cols].var()
            mediators = variances.nlargest(20).index.tolist()
            results.mediation = run_mediation_analysis(
                ctx.precomputed_df, "temperature", mediators, "rating", n_bootstrap=1000
            )
    
    if config.run_mixed_effects and "subject" in ctx.precomputed_df.columns:
        ctx.logger.info("Running mixed-effects analysis...")
        feature_cols = [c for c in ctx.precomputed_df.columns 
                       if c not in ["subject", "epoch", "trial", "condition", "temperature", "rating"]]
        if feature_cols:
            results.mixed_effects = run_multilevel_correlation_analysis(
                ctx.precomputed_df, feature_cols[:50], "rating", "subject"
            )


def _stage_validate(ctx: BehaviorContext, config: BehaviorPipelineConfig) -> None:
    from eeg_pipeline.utils.analysis.stats.fdr import apply_global_fdr
    
    ctx.logger.info("Applying global FDR correction...")
    apply_global_fdr(ctx.stats_dir, alpha=config.fdr_alpha, logger=ctx.logger)


def _stage_export(
    ctx: BehaviorContext,
    results: BehaviorPipelineResults,
) -> List[Path]:
    from eeg_pipeline.utils.io.general import write_tsv, ensure_dir
    
    ensure_dir(ctx.stats_dir)
    saved = []
    
    if results.correlations is not None and not results.correlations.empty:
        path = ctx.stats_dir / "correlations.tsv"
        write_tsv(results.correlations, path)
        saved.append(path)
    
    if results.pain_sensitivity is not None and not results.pain_sensitivity.empty:
        path = ctx.stats_dir / "pain_sensitivity.tsv"
        write_tsv(results.pain_sensitivity, path)
        saved.append(path)
    
    if results.condition_effects is not None and not results.condition_effects.empty:
        path = ctx.stats_dir / "condition_effects.tsv"
        write_tsv(results.condition_effects, path)
        saved.append(path)
    
    if results.mediation is not None and not results.mediation.empty:
        path = ctx.stats_dir / "mediation.tsv"
        write_tsv(results.mediation, path)
        saved.append(path)
    
    summary = results.to_summary()
    with open(ctx.stats_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    saved.append(ctx.stats_dir / "summary.json")
    
    ctx.logger.info(f"Saved {len(saved)} output files")
    return saved


###################################################################
# Pipeline Class
###################################################################


class BehaviorPipeline(PipelineBase):
    """Pipeline for EEG-behavior correlation analysis."""
    
    def __init__(self, config: Optional[Any] = None, pipeline_config: Optional[BehaviorPipelineConfig] = None):
        super().__init__(name="behavior_analysis", config=config)
        self.pipeline_config = pipeline_config or BehaviorPipelineConfig.from_config(self.config)

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> BehaviorPipelineResults:
        from eeg_pipeline.utils.io.general import deriv_stats_path, ensure_dir, get_subject_logger
        
        task = task or self.config.get("project.task", "thermalactive")
        
        stats_dir = deriv_stats_path(self.deriv_root, subject)
        ensure_dir(stats_dir)
        
        logger = get_subject_logger(
            "behavior_analysis", subject,
            self.config.get("logging.log_file_name", "behavior_analysis.log"),
            config=self.config
        )
        
        logger.info(f"{'='*60}")
        logger.info(f"Behavior Pipeline: sub-{subject}")
        logger.info(f"{'='*60}")
        
        ctx = BehaviorContext(
            subject=subject,
            task=task,
            config=self.config,
            logger=logger,
            deriv_root=self.deriv_root,
            stats_dir=stats_dir,
            use_spearman=(self.pipeline_config.method == "spearman"),
            rng=np.random.default_rng(42),
        )
        
        results = BehaviorPipelineResults(subject=subject)
        
        logger.info("Stage 1: Loading data...")
        if not _stage_load(ctx):
            return results
        
        logger.info("Stage 2: Running unified correlations...")
        results.correlations, results.pain_sensitivity = _stage_correlate(ctx, self.pipeline_config)
        
        if self.pipeline_config.run_condition_comparison:
            logger.info("Stage 3: Condition comparison...")
            results.condition_effects = _stage_condition(ctx, self.pipeline_config)
        
        if self.pipeline_config.run_temporal_correlations:
            logger.info("Stage 4: Temporal correlations...")
            _stage_temporal(ctx)
        
        if self.pipeline_config.run_cluster_tests:
            logger.info("Stage 5: Cluster permutation tests...")
            results.cluster = _stage_cluster(ctx, self.pipeline_config)
        
        if self.pipeline_config.run_mediation or self.pipeline_config.run_mixed_effects:
            logger.info("Stage 6: Advanced analyses...")
            _stage_advanced(ctx, self.pipeline_config, results)
        
        logger.info("Stage 7: Global FDR correction...")
        _stage_validate(ctx, self.pipeline_config)
        
        logger.info("Stage 8: Saving results...")
        _stage_export(ctx, results)
        
        summary = results.to_summary()
        logger.info(f"{'='*60}")
        logger.info(f"Complete: {summary.get('n_features', 0)} features")
        logger.info(f"  Significant (raw): {summary.get('n_sig_raw', 0)}")
        logger.info(f"  Significant (controlled): {summary.get('n_sig_controlled', 0)}")
        logger.info(f"{'='*60}")
        
        return results


###################################################################
# Module-Level Entry Points
###################################################################


def run_pipeline(
    subject: str,
    task: Optional[str] = None,
    config: Optional[Any] = None,
    pipeline_config: Optional[BehaviorPipelineConfig] = None,
) -> BehaviorPipelineResults:
    pipeline = BehaviorPipeline(config=config, pipeline_config=pipeline_config)
    return pipeline.process_subject(subject, task=task)


def run_pipeline_batch(
    subjects: List[str],
    task: Optional[str] = None,
    config: Optional[Any] = None,
    run_group_aggregation: bool = True,
    parallel_subjects: bool = False,
    n_jobs: int = -1,
) -> Dict[str, BehaviorPipelineResults]:
    from eeg_pipeline.utils.config.loader import load_settings
    from eeg_pipeline.utils.progress import BatchProgress
    from eeg_pipeline.utils.io.general import get_logger
    from eeg_pipeline.analysis.behavior.parallel import parallel_subjects as run_parallel_subjects, get_n_jobs
    
    if config is None:
        config = load_settings()
    
    logger = get_logger(__name__)
    pipeline_config = BehaviorPipelineConfig.from_config(config)
    n_jobs_actual = get_n_jobs(config, n_jobs)
    
    if parallel_subjects and len(subjects) > 1:
        logger.info(f"Parallel subject processing: {len(subjects)} subjects, {n_jobs_actual} jobs")
        
        def process_single_subject(subject: str) -> BehaviorPipelineResults:
            return run_pipeline(subject, task, config, pipeline_config)
        
        results = run_parallel_subjects(
            subjects=subjects,
            process_func=process_single_subject,
            n_jobs=n_jobs_actual,
            logger=logger,
        )
        
        for subject in subjects:
            if results.get(subject) is None:
                results[subject] = BehaviorPipelineResults(subject=subject)
        
        return results
    
    results = {}
    
    with BatchProgress(subjects=subjects, logger=logger, desc="Behavior") as batch:
        for subject in subjects:
            start = batch.start_subject(subject)
            try:
                results[subject] = run_pipeline(subject, task, config, pipeline_config)
                batch.finish_subject(subject, start)
            except Exception as e:
                logger.error(f"Failed sub-{subject}: {e}")
                results[subject] = BehaviorPipelineResults(subject=subject)
                batch.finish_subject(subject, start)
    
    return results


def compute_behavior_correlations_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    correlation_method: str = "spearman",
    bootstrap: int = 0,
    n_perm: int = 100,
    rng_seed: Optional[int] = None,
    computations: Optional[List[str]] = None,
    parallel_subjects: bool = False,
    n_jobs: int = -1,
) -> Dict[str, BehaviorPipelineResults]:
    from eeg_pipeline.utils.config.loader import load_settings
    
    config = load_settings()
    
    def _ensure_nested(d: dict, *keys) -> dict:
        for key in keys:
            if key not in d:
                d[key] = {}
            d = d[key]
        return d
    
    if correlation_method:
        _ensure_nested(config, "behavior_analysis", "statistics")["correlation_method"] = correlation_method
    if bootstrap is not None and bootstrap > 0:
        _ensure_nested(config, "behavior_analysis")["bootstrap"] = bootstrap
    if n_perm is not None and n_perm > 0:
        _ensure_nested(config, "behavior_analysis", "statistics")["n_permutations"] = n_perm
    if rng_seed is not None:
        _ensure_nested(config, "project")["random_state"] = rng_seed
    if n_jobs != -1:
        _ensure_nested(config, "behavior_analysis")["n_jobs"] = n_jobs
    
    if computations:
        comp_set = set(computations)
        ba = _ensure_nested(config, "behavior_analysis")
        ba["temporal_enabled"] = "temporal" in comp_set
        ba["cluster_enabled"] = "cluster" in comp_set
        ba["mediation_enabled"] = "mediation" in comp_set
    
    return run_pipeline_batch(subjects, task, config, parallel_subjects=parallel_subjects, n_jobs=n_jobs)


__all__ = [
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
    "run_pipeline",
    "run_pipeline_batch",
    "compute_behavior_correlations_for_subjects",
]
