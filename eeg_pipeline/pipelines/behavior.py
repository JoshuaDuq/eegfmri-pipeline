"""
Behavior Analysis Pipeline (Canonical)
======================================

Pipeline class for EEG-behavior correlation analysis.
This module provides the PipelineBase subclass for behavior analysis,
with statistical routines consolidated in eeg_pipeline.analysis.behavior.api.

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
from eeg_pipeline.utils.analysis.stats.correlation import (
    normalize_correlation_method,
    format_correlation_method_label,
)
from eeg_pipeline.utils.analysis.stats.reliability import get_subject_seed
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.analysis.behavior.orchestration import (
    add_change_scores as _add_change_scores_impl,
    build_behavior_qc as _build_behavior_qc_impl,
    combine_features as _combine_features_impl,
    stage_advanced as _stage_advanced_impl,
    stage_cluster as _stage_cluster_impl,
    stage_condition as _stage_condition_impl,
    stage_correlate as _stage_correlate_impl,
    stage_export as _stage_export_impl,
    stage_load as _stage_load_impl,
    stage_temporal as _stage_temporal_impl,
    stage_validate as _stage_validate_impl,
    write_analysis_metadata as _write_analysis_metadata_impl,
    write_outputs_manifest,
)


###################################################################
# Configuration
###################################################################

BEHAVIOR_COMPUTATION_FLAGS = [
    "correlations",
    "pain_sensitivity",
    "condition",
    "temporal",
    "cluster",
    "mediation",
    "mixed_effects",
]


def _resolve_behavior_computation_flags(
    requested: Optional[List[str]],
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, bool]]:
    """
    Normalize requested behavior computations into stage flags.
    
    If requested is None, returns None to indicate no override (use config).
    """
    if requested is None:
        return None
    
    flags = {k: False for k in BEHAVIOR_COMPUTATION_FLAGS}
    normalized = [str(item).lower() for item in requested]
    
    unknown = [k for k in normalized if k not in BEHAVIOR_COMPUTATION_FLAGS]
    if unknown and logger:
        logger.warning("Ignoring unrecognized behavior computations: %s", ", ".join(sorted(set(unknown))))
    
    for key in normalized:
        if key in flags:
            flags[key] = True
    
    return flags


@dataclass
class BehaviorPipelineConfig:
    method: str = "spearman"
    min_samples: int = 10
    control_temperature: bool = True
    control_trial_order: bool = True
    compute_change_scores: bool = True
    compute_pain_sensitivity: bool = True
    compute_reliability: bool = False
    compute_bayes_factors: bool = False
    compute_loso_stability: bool = True
    bootstrap: int = 0
    robust_method: Optional[str] = None
    method_label: str = "spearman"
    
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
        raw_method = get_config_value(config, "behavior_analysis.statistics.correlation_method", None)
        if raw_method is None:
            raw_method = get_config_value(config, "behavior_analysis.correlation_method", "spearman")
        method = normalize_correlation_method(raw_method, default="spearman")
        robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
        if robust_method is not None:
            robust_method = str(robust_method).strip().lower() or None
        method_label = format_correlation_method_label(method, robust_method)
        return cls(
            method=method,
            min_samples=int(get_config_value(config, "behavior_analysis.min_samples.default", 10)),
            control_temperature=bool(get_config_value(config, "behavior_analysis.control_temperature", True)),
            control_trial_order=bool(get_config_value(config, "behavior_analysis.control_trial_order", True)),
            compute_change_scores=bool(get_config_value(config, "behavior_analysis.compute_change_scores", True)),
            compute_pain_sensitivity=bool(get_config_value(config, "behavior_analysis.compute_pain_sensitivity", True)),
            compute_reliability=bool(get_config_value(config, "behavior_analysis.statistics.compute_reliability", False)),
            compute_bayes_factors=bool(get_config_value(config, "behavior_analysis.compute_bayes_factors", False)),
            compute_loso_stability=bool(get_config_value(config, "behavior_analysis.loso_stability", True)),
            bootstrap=int(get_config_value(config, "behavior_analysis.bootstrap", get_config_value(config, "behavior_analysis.statistics.default_n_bootstrap", 1000))),
            robust_method=robust_method,
            method_label=method_label,
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
            df = self.correlations
            s["n_features"] = len(df)
            p_raw = (
                df["p_raw"]
                if "p_raw" in df.columns
                else df.get("p_value")
                if "p_value" in df.columns
                else df.get("p")
            )
            p_primary = df["p_primary"] if "p_primary" in df.columns else None
            if "p_fdr" in df.columns:
                p_fdr = df["p_fdr"]
            elif "q_value" in df.columns:
                p_fdr = df["q_value"]
            else:
                p_fdr = None
            s["n_sig_raw"] = int((p_raw.fillna(1) < 0.05).sum()) if p_raw is not None else 0
            s["n_sig_controlled"] = int((p_primary.fillna(1) < 0.05).sum()) if p_primary is not None else 0
            if p_fdr is not None:
                s["n_sig_fdr"] = int((p_fdr.fillna(1) < 0.05).sum())
        if self.condition_effects is not None:
            s["n_condition_effects"] = len(self.condition_effects)
            s["n_large_effects"] = int((self.condition_effects["hedges_g"].abs() >= 0.8).sum())
        return s



###################################################################
# Pipeline Class
###################################################################


class BehaviorPipeline(PipelineBase):
    """Pipeline for EEG-behavior correlation analysis."""
    
    def __init__(
        self,
        config: Optional[Any] = None,
        pipeline_config: Optional[BehaviorPipelineConfig] = None,
        computations: Optional[List[str]] = None,
        feature_categories: Optional[List[str]] = None,
        feature_files: Optional[List[str]] = None,
        computation_features: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(name="behavior_analysis", config=config)
        self.pipeline_config = pipeline_config or BehaviorPipelineConfig.from_config(self.config)
        self.feature_categories = feature_categories
        self.feature_files = feature_files
        self.computation_features = computation_features or {}
        
        self._run_correlations = True
        self._run_export = True
        self._run_validation = True
        
        comp_flags = _resolve_behavior_computation_flags(computations, logger=self.logger)
        if comp_flags is not None:
            self._run_correlations = comp_flags["correlations"]
            self.pipeline_config.run_condition_comparison = comp_flags["condition"]
            self.pipeline_config.run_temporal_correlations = comp_flags["temporal"]
            self.pipeline_config.run_cluster_tests = comp_flags["cluster"]
            self.pipeline_config.run_mediation = comp_flags["mediation"]
            self.pipeline_config.run_mixed_effects = comp_flags["mixed_effects"]
            self.pipeline_config.compute_pain_sensitivity = comp_flags["pain_sensitivity"]
            self._run_validation = any(
                comp_flags[key] for key in BEHAVIOR_COMPUTATION_FLAGS
            )
            
            selected = [k for k, v in comp_flags.items() if v]
            self.logger.info("Behavior computations (override): %s", ", ".join(selected) if selected else "none")
        else:
            self._run_validation = any(
                [
                    self._run_correlations,
                    self.pipeline_config.run_condition_comparison,
                    self.pipeline_config.run_temporal_correlations,
                    self.pipeline_config.run_cluster_tests,
                    self.pipeline_config.run_mediation,
                    self.pipeline_config.run_mixed_effects,
                    self.pipeline_config.compute_pain_sensitivity,
                ]
            )
        
        if self.feature_categories:
            self.logger.info("Feature categories filter: %s", ", ".join(self.feature_categories))
        
        if self.computation_features:
            for comp, feats in self.computation_features.items():
                self.logger.info("  %s features: %s", comp, ", ".join(feats))

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> BehaviorPipelineResults:
        from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
        from eeg_pipeline.infra.logging import get_subject_logger
        from eeg_pipeline.cli.common import ProgressReporter
        import sys
        import time
        import resource
        
        task = task or self.config.get("project.task", "thermalactive")
        
        # Initialize progress reporter
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        validate_only = bool(kwargs.get("validate_only", False))
        
        # Calculate total steps dynamically based on enabled stages
        run_advanced = self.pipeline_config.run_mediation or self.pipeline_config.run_mixed_effects
        enabled_stages = [
            True,  # Load (always runs)
            self._run_correlations,
            self.pipeline_config.run_condition_comparison,
            self.pipeline_config.run_temporal_correlations,
            self.pipeline_config.run_cluster_tests,
            run_advanced,
            self._run_validation,
            True,  # Export (always runs)
        ]
        total_steps = sum(enabled_stages) if not validate_only else 2
        current_step = 0
        
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
        
        progress.subject_start(f"sub-{subject}")

        base_seed = int(get_config_value(self.config, "behavior_analysis.statistics.base_seed", 42))
        rng = np.random.default_rng(get_subject_seed(base_seed, subject))

        stats_cfg = self.config.get("behavior_analysis.statistics", {})
        partial_covars = stats_cfg.get("partial_covariates", None)

        ctx = BehaviorContext(
            subject=subject,
            task=task,
            config=self.config,
            logger=logger,
            deriv_root=self.deriv_root,
            stats_dir=stats_dir,
            use_spearman=(self.pipeline_config.method == "spearman"),
            bootstrap=int(self.pipeline_config.bootstrap),
            n_perm=int(self.pipeline_config.n_permutations),
            rng=rng,
            partial_covars=partial_covars,
            control_temperature=self.pipeline_config.control_temperature,
            control_trial_order=self.pipeline_config.control_trial_order,
            compute_change_scores=self.pipeline_config.compute_change_scores,
            compute_reliability=self.pipeline_config.compute_reliability,
            stats_config=self.pipeline_config,
            feature_categories=self.feature_categories,
            selected_feature_files=self.feature_files,
        )
        
        results = BehaviorPipelineResults(subject=subject)
        
        def _rss_mb() -> float:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024 * 1024)
            return rss / 1024

        stage_metrics = {}

        def _record_stage(name: str, start_time: float, start_rss: float) -> None:
            end_time = time.perf_counter()
            end_rss = _rss_mb()
            stage_metrics[name] = {
                "duration_s": round(end_time - start_time, 4),
                "rss_mb_start": round(start_rss, 3),
                "rss_mb_end": round(end_rss, 3),
                "rss_mb_delta": round(end_rss - start_rss, 3),
            }

        stage_start = time.perf_counter()
        stage_rss = _rss_mb()
        current_step += 1
        progress.step("Loading data", current=current_step, total=total_steps)
        logger.info("Loading data...")
        if not _stage_load_impl(ctx):
            progress.error("load_failed", "Failed to load data")
            _record_stage("load", stage_start, stage_rss)
            return results
        _record_stage("load", stage_start, stage_rss)

        if validate_only:
            current_step += 1
            progress.step("Exporting validation", current=current_step, total=total_steps)
            logger.info("Validation-only mode: skipping computations.")
            ctx.data_qc["validate_only"] = True
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            _stage_export_impl(ctx, self.pipeline_config, results)
            _record_stage("export", stage_start, stage_rss)
            outputs_manifest_path = ctx.stats_dir / "outputs_manifest.json"
            try:
                _write_analysis_metadata_impl(
                    ctx, self.pipeline_config, results,
                    stage_metrics=stage_metrics,
                    outputs_manifest=outputs_manifest_path,
                )
            except Exception as e:
                logger.warning(f"Failed to write analysis metadata: {e}")
            write_outputs_manifest(ctx, self.pipeline_config, results, stage_metrics)
            progress.subject_done(f"sub-{subject}", success=True)
            return results
        
        if self._run_correlations:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Running correlations", current=current_step, total=total_steps)
            logger.info("Running correlations...")
            results.correlations, results.pain_sensitivity = _stage_correlate_impl(ctx, self.pipeline_config)
            _record_stage("correlations", stage_start, stage_rss)
        
        if self.pipeline_config.run_condition_comparison:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Condition comparison", current=current_step, total=total_steps)
            logger.info("Running condition comparison...")
            results.condition_effects = _stage_condition_impl(ctx, self.pipeline_config)
            _record_stage("condition", stage_start, stage_rss)
        
        if self.pipeline_config.run_temporal_correlations:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Temporal correlations", current=current_step, total=total_steps)
            logger.info("Running temporal correlations...")
            _stage_temporal_impl(ctx)
            _record_stage("temporal", stage_start, stage_rss)
        
        if self.pipeline_config.run_cluster_tests:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Cluster permutation tests", current=current_step, total=total_steps)
            logger.info("Running cluster permutation tests...")
            results.cluster = _stage_cluster_impl(ctx, self.pipeline_config)
            _record_stage("cluster", stage_start, stage_rss)
        
        if run_advanced:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Advanced analyses", current=current_step, total=total_steps)
            logger.info("Running advanced analyses...")
            _stage_advanced_impl(ctx, self.pipeline_config, results)
            _record_stage("advanced", stage_start, stage_rss)
        
        if self._run_validation:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Global FDR correction", current=current_step, total=total_steps)
            logger.info("Running global FDR correction...")
            _stage_validate_impl(ctx, self.pipeline_config)
            _record_stage("fdr", stage_start, stage_rss)
        
        current_step += 1
        progress.step("Saving results", current=current_step, total=total_steps)
        logger.info("Saving results...")
        stage_start = time.perf_counter()
        stage_rss = _rss_mb()
        _stage_export_impl(ctx, self.pipeline_config, results)
        _record_stage("export", stage_start, stage_rss)

        outputs_manifest_path = ctx.stats_dir / "outputs_manifest.json"
        try:
            _write_analysis_metadata_impl(
                ctx, self.pipeline_config, results,
                stage_metrics=stage_metrics,
                outputs_manifest=outputs_manifest_path,
            )
        except Exception as e:
            logger.warning(f"Failed to write analysis metadata: {e}")
        write_outputs_manifest(ctx, self.pipeline_config, results, stage_metrics)
        
        summary = results.to_summary()
        logger.info(f"{'='*60}")
        logger.info(f"Complete: {summary.get('n_features', 0)} features")
        logger.info(f"  Significant (raw): {summary.get('n_sig_raw', 0)}")
        logger.info(f"  Significant (controlled): {summary.get('n_sig_controlled', 0)}")
        logger.info(f"{'='*60}")
        
        progress.subject_done(f"sub-{subject}", success=True)
        
        return results


__all__ = [
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
]
