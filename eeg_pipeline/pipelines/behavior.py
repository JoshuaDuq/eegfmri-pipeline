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
    stage_trial_table as _stage_trial_table_impl,
    stage_trial_table_validate as _stage_trial_table_validate_impl,
    stage_confounds as _stage_confounds_impl,
    stage_regression as _stage_regression_impl,
    stage_models as _stage_models_impl,
    stage_stability as _stage_stability_impl,
    stage_consistency as _stage_consistency_impl,
    stage_influence as _stage_influence_impl,
    stage_advanced as _stage_advanced_impl,
    stage_cluster as _stage_cluster_impl,
    stage_condition as _stage_condition_impl,
    stage_correlate as _stage_correlate_impl,
    stage_export as _stage_export_impl,
    stage_load as _stage_load_impl,
    stage_temporal as _stage_temporal_impl,
    stage_validate as _stage_validate_impl,
    stage_report as _stage_report_impl,
    write_analysis_metadata as _write_analysis_metadata_impl,
    write_outputs_manifest,
)


###################################################################
# Configuration
###################################################################

BEHAVIOR_COMPUTATION_FLAGS = [
    "trial_table",
    "confounds",
    "regression",
    "models",
    "stability",
    "consistency",
    "influence",
    "report",
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
    
    # Computation flags
    trial_table_only: bool = True
    run_trial_table: bool = True
    run_confounds: bool = True
    run_regression: bool = False
    run_models: bool = False
    run_stability: bool = True
    run_consistency: bool = True
    run_influence: bool = True
    run_report: bool = True
    run_correlations: bool = True
    run_condition_comparison: bool = True
    run_temporal_correlations: bool = True
    run_cluster_tests: bool = False
    run_mediation: bool = False
    run_mixed_effects: bool = False
    
    # General stats
    fdr_alpha: float = 0.05
    n_permutations: int = 1000
    n_jobs: int = -1
    
    # Condition-specific
    condition_effect_threshold: float = 0.5
    condition_min_trials: int = 10
    
    # Temporal-specific
    temporal_resolution_ms: int = 50
    temporal_smooth_ms: int = 100
    
    # Cluster-specific
    cluster_threshold: float = 0.05
    cluster_min_size: int = 2
    cluster_tail: int = 0
    
    # Mediation-specific
    mediation_n_bootstrap: int = 1000
    mediation_min_effect: float = 0.05
    mediation_max_mediators: int = 20
    
    # Mixed effects-specific
    mixed_effects_type: str = "intercept"
    mixed_effects_max_features: int = 50
    
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
            compute_change_scores=bool(get_config_value(config, "behavior_analysis.correlations.compute_change_scores", True)),
            compute_pain_sensitivity=bool(get_config_value(config, "behavior_analysis.pain_sensitivity.enabled", True)),
            compute_reliability=bool(get_config_value(config, "behavior_analysis.statistics.compute_reliability", False)),
            compute_bayes_factors=bool(get_config_value(config, "behavior_analysis.correlations.compute_bayes_factors", False)),
            compute_loso_stability=bool(get_config_value(config, "behavior_analysis.correlations.loso_stability", True)),
            bootstrap=int(get_config_value(config, "behavior_analysis.bootstrap", get_config_value(config, "behavior_analysis.statistics.default_n_bootstrap", 1000))),
            robust_method=robust_method,
            method_label=method_label,
            trial_table_only=bool(get_config_value(config, "behavior_analysis.trial_table_only.enabled", True)),
            run_trial_table=bool(get_config_value(config, "behavior_analysis.trial_table.enabled", True)),
            run_confounds=bool(get_config_value(config, "behavior_analysis.confounds.enabled", True)),
            run_regression=bool(get_config_value(config, "behavior_analysis.regression.enabled", False)),
            run_models=bool(get_config_value(config, "behavior_analysis.models.enabled", False)),
            run_stability=bool(get_config_value(config, "behavior_analysis.stability.enabled", True)),
            run_consistency=bool(get_config_value(config, "behavior_analysis.consistency.enabled", True)),
            run_influence=bool(get_config_value(config, "behavior_analysis.influence.enabled", True)),
            run_report=bool(get_config_value(config, "behavior_analysis.report.enabled", True)),
            run_correlations=get_config_value(config, "behavior_analysis.correlations.enabled", True),
            run_condition_comparison=get_config_value(config, "behavior_analysis.condition.enabled", True),
            run_temporal_correlations=get_config_value(config, "behavior_analysis.temporal.enabled", True),
            run_cluster_tests=get_config_value(config, "behavior_analysis.cluster.enabled", False),
            run_mediation=get_config_value(config, "behavior_analysis.mediation.enabled", False),
            run_mixed_effects=get_config_value(config, "behavior_analysis.mixed_effects.enabled", False),
            fdr_alpha=float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", 0.05)),
            n_permutations=int(
                get_config_value(
                    config,
                    "behavior_analysis.cluster.n_permutations",
                    get_config_value(config, "behavior_analysis.statistics.n_permutations", 1000),
                )
            ),
            n_jobs=int(get_config_value(config, "behavior_analysis.n_jobs", -1)),
            # Condition-specific
            condition_effect_threshold=float(get_config_value(config, "behavior_analysis.condition.effect_size_threshold", 0.5)),
            condition_min_trials=int(get_config_value(config, "behavior_analysis.condition.min_trials_per_condition", 10)),
            # Temporal-specific
            temporal_resolution_ms=int(get_config_value(config, "behavior_analysis.temporal.time_resolution_ms", 50)),
            temporal_smooth_ms=int(get_config_value(config, "behavior_analysis.temporal.smooth_window_ms", 100)),
            # Cluster-specific
            cluster_threshold=float(get_config_value(config, "behavior_analysis.cluster.forming_threshold", 0.05)),
            cluster_min_size=int(get_config_value(config, "behavior_analysis.cluster.min_cluster_size", 2)),
            cluster_tail=int(get_config_value(config, "behavior_analysis.cluster.tail", 0)),
            # Mediation-specific
            mediation_n_bootstrap=int(get_config_value(config, "behavior_analysis.mediation.n_bootstrap", 1000)),
            mediation_min_effect=float(get_config_value(config, "behavior_analysis.mediation.min_effect_size", 0.05)),
            mediation_max_mediators=int(get_config_value(config, "behavior_analysis.mediation.max_mediators", 20)),
            # Mixed effects-specific
            mixed_effects_type=str(get_config_value(config, "behavior_analysis.mixed_effects.random_effects", "intercept")),
            mixed_effects_max_features=int(get_config_value(config, "behavior_analysis.mixed_effects.max_features", 50)),
        )


@dataclass
class BehaviorPipelineResults:
    subject: str
    trial_table_path: Optional[str] = None
    trial_table_validation: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None
    confounds: Optional[pd.DataFrame] = None
    regression: Optional[pd.DataFrame] = None
    models: Optional[pd.DataFrame] = None
    stability: Optional[pd.DataFrame] = None
    consistency: Optional[pd.DataFrame] = None
    influence: Optional[pd.DataFrame] = None
    correlations: Optional[pd.DataFrame] = None
    pain_sensitivity: Optional[pd.DataFrame] = None
    condition_effects: Optional[pd.DataFrame] = None
    mediation: Optional[pd.DataFrame] = None
    mixed_effects: Optional[pd.DataFrame] = None
    cluster: Optional[Dict[str, Any]] = None
    temporal: Optional[Dict[str, Any]] = None
    tf: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary(self) -> Dict[str, Any]:
        s = {"subject": self.subject}
        if self.trial_table_path:
            s["trial_table_path"] = self.trial_table_path
        if self.report_path:
            s["report_path"] = self.report_path
        n_total = 0
        n_sig_raw = 0
        n_sig_controlled = 0
        n_sig_fdr = 0

        if self.correlations is not None and not self.correlations.empty:
            df = self.correlations
            n = len(df)
            n_total += n
            
            p_raw = (
                df["p_raw"]
                if "p_raw" in df.columns
                else df.get("p_value")
                if "p_value" in df.columns
                else df.get("p")
            )
            p_primary = df["p_primary"] if "p_primary" in df.columns else None
            p_fdr = df["q_global"] if "q_global" in df.columns else df.get("q_value")
            
            n_sig_raw += int((p_raw.fillna(1) < 0.05).sum()) if p_raw is not None else 0
            n_sig_controlled += int((p_primary.fillna(1) < 0.05).sum()) if p_primary is not None else 0
            n_sig_fdr += int((p_fdr.fillna(1) < 0.05).sum()) if p_fdr is not None else 0
        
        if self.pain_sensitivity is not None and not self.pain_sensitivity.empty:
            df_psi = self.pain_sensitivity
            n = len(df_psi)
            n_total += n
            s["n_pain_sensitivity_features"] = n
            
            p_psi = df_psi["p_psi"] if "p_psi" in df_psi.columns else df_psi.get("p_value")
            p_fdr = df_psi["q_global"] if "q_global" in df_psi.columns else df_psi.get("q_value")
            
            if p_psi is not None:
                n_sig_psi_raw = int((p_psi.fillna(1) < 0.05).sum())
                n_sig_raw += n_sig_psi_raw
                s["n_sig_psi_raw"] = n_sig_psi_raw
            
            if p_fdr is not None:
                n_sig_fdr += int((p_fdr.fillna(1) < 0.05).sum())

        if self.condition_effects is not None and not self.condition_effects.empty:
            df = self.condition_effects
            n = len(df)
            n_total += n
            s["n_condition_effects"] = n
            
            p_raw = df["p_value"] if "p_value" in df.columns else df.get("p")
            p_fdr = df["q_global"] if "q_global" in df.columns else df.get("q_value")
            
            n_sig_raw += int((p_raw.fillna(1) < 0.05).sum()) if p_raw is not None else 0
            if p_fdr is not None:
                n_sig_fdr += int((p_fdr.fillna(1) < 0.05).sum())

        if self.regression is not None and not self.regression.empty:
            df = self.regression
            n = len(df)
            n_total += n
            s["n_regression_features"] = n
            p_raw = df["p_primary"] if "p_primary" in df.columns else df.get("p_feature")
            p_fdr = df["q_global"] if "q_global" in df.columns else df.get("p_fdr")
            if p_raw is not None:
                n_sig_raw += int((pd.to_numeric(p_raw, errors="coerce").fillna(1) < 0.05).sum())
            if p_fdr is not None:
                n_sig_fdr += int((pd.to_numeric(p_fdr, errors="coerce").fillna(1) < 0.05).sum())
                
            s["n_large_effects"] = int((df["hedges_g"].abs() >= 0.8).sum()) if "hedges_g" in df.columns else 0

        if self.mediation is not None and not self.mediation.empty:
            df = self.mediation
            n = len(df)
            n_total += n
            s["n_mediation_mediators"] = n
            
            p_raw = df["sobel_p"] if "sobel_p" in df.columns else df.get("p_value")
            p_fdr = df["q_global"] if "q_global" in df.columns else None
            
            n_sig_raw += int((p_raw.fillna(1) < 0.05).sum()) if p_raw is not None else 0
            if p_fdr is not None:
                n_sig_fdr += int((p_fdr.fillna(1) < 0.05).sum())

        if self.mixed_effects is not None and not self.mixed_effects.empty:
            df = self.mixed_effects
            n = len(df)
            n_total += n
            s["n_mixed_effects_features"] = n
            
            p_raw = df["fixed_p"] if "fixed_p" in df.columns else df.get("p_value")
            p_fdr = df["q_global"] if "q_global" in df.columns else df.get("fixed_p_fdr")
            
            n_sig_raw += int((p_raw.fillna(1) < 0.05).sum()) if p_raw is not None else 0
            if p_fdr is not None:
                n_sig_fdr += int((p_fdr.fillna(1) < 0.05).sum())

        if self.tf is not None:
             n = self.tf.get("n_tests", 0)
             n_total += n
             n_sig_raw += self.tf.get("n_sig_raw", 0)
             n_sig_fdr += self.tf.get("n_sig_fdr", 0)
             s["n_tf_tests"] = n

        if self.temporal is not None:
             n = self.temporal.get("n_tests", 0)
             n_total += n
             n_sig_raw += self.temporal.get("n_sig_raw", 0)
             n_sig_fdr += self.temporal.get("n_sig_fdr", 0)
             s["n_temporal_tests"] = n

        if self.cluster is not None:
            n_clusters = 0
            n_sig_clusters = 0
            for band, res in self.cluster.items():
                if isinstance(res, dict):
                    recs = res.get("cluster_records", [])
                    n_clusters += len(recs)
                    # Check q_global first for FDR significance, fallback to p_value
                    for r in recs:
                        p_val = r.get("q_global") if "q_global" in r else r.get("p_value", 1.0)
                        if p_val < 0.05:
                            n_sig_clusters += 1
            
            s["n_clusters"] = n_clusters
            s["n_sig_clusters"] = n_sig_clusters
            n_total += n_clusters
            n_sig_raw += n_sig_clusters
            n_sig_fdr += n_sig_clusters # Cluster tests are already FWER corrected

        s["n_features"] = n_total
        s["n_sig_raw"] = n_sig_raw
        s["n_sig_controlled"] = n_sig_controlled
        s["n_sig_fdr"] = n_sig_fdr
        
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
        
        self._run_validation = True
        
        comp_flags = _resolve_behavior_computation_flags(computations, logger=self.logger)
        if comp_flags is not None:
            self.pipeline_config.run_trial_table = comp_flags["trial_table"]
            self.pipeline_config.run_confounds = comp_flags["confounds"]
            self.pipeline_config.run_regression = comp_flags["regression"]
            self.pipeline_config.run_models = comp_flags["models"]
            self.pipeline_config.run_stability = comp_flags["stability"]
            self.pipeline_config.run_consistency = comp_flags["consistency"]
            self.pipeline_config.run_influence = comp_flags["influence"]
            self.pipeline_config.run_report = comp_flags["report"]
            self.pipeline_config.run_correlations = comp_flags["correlations"]
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
                    self.pipeline_config.run_confounds,
                    self.pipeline_config.run_regression,
                    self.pipeline_config.run_models,
                    self.pipeline_config.run_stability,
                    self.pipeline_config.run_consistency,
                    self.pipeline_config.run_influence,
                    self.pipeline_config.run_report,
                    self.pipeline_config.run_correlations,
                    self.pipeline_config.run_condition_comparison,
                    self.pipeline_config.run_temporal_correlations,
                    self.pipeline_config.run_cluster_tests,
                    self.pipeline_config.run_mediation,
                    self.pipeline_config.run_mixed_effects,
                    self.pipeline_config.compute_pain_sensitivity,
                ]
            )

        # Enforce subject-level "trial-table-only" mode by skipping computations that require epochs/time-frequency arrays.
        if bool(getattr(self.pipeline_config, "trial_table_only", True)):
            if self.pipeline_config.run_temporal_correlations:
                self.logger.info("trial_table_only enabled: skipping `temporal` computation.")
                self.pipeline_config.run_temporal_correlations = False
            if self.pipeline_config.run_cluster_tests:
                self.logger.info("trial_table_only enabled: skipping `cluster` computation.")
                self.pipeline_config.run_cluster_tests = False
        
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
        run_correlate = self.pipeline_config.run_correlations or self.pipeline_config.compute_pain_sensitivity
        enabled_stages = [
            True,  # Load (always runs)
            self.pipeline_config.run_trial_table,
            self.pipeline_config.run_confounds,
            self.pipeline_config.run_regression,
            self.pipeline_config.run_models,
            self.pipeline_config.run_stability,
            run_correlate,
            self.pipeline_config.run_consistency,
            self.pipeline_config.run_influence,
            self.pipeline_config.run_condition_comparison,
            self.pipeline_config.run_temporal_correlations,
            self.pipeline_config.run_cluster_tests,
            run_advanced,
            self._run_validation,
            self.pipeline_config.run_report,
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

        if self.pipeline_config.run_trial_table:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Building trial table", current=current_step, total=total_steps)
            logger.info("Building trial table...")
            try:
                # Default: save inside stats_dir
                out_path = _stage_trial_table_impl(ctx, self.pipeline_config)
                if out_path is not None:
                    results.trial_table_path = str(out_path)
            except Exception as exc:
                logger.warning(f"Trial table build failed: {exc}")

            # Always attempt non-gating trial-table validation when enabled.
            try:
                results.trial_table_validation = _stage_trial_table_validate_impl(ctx, self.pipeline_config)
            except Exception as exc:
                logger.debug("Trial table validation failed: %s", exc)
            _record_stage("trial_table", stage_start, stage_rss)

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

        if self.pipeline_config.run_confounds:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Confound audit", current=current_step, total=total_steps)
            logger.info("Running confound audit...")
            try:
                results.confounds = _stage_confounds_impl(ctx, self.pipeline_config)
            except Exception as exc:
                logger.warning(f"Confound audit failed: {exc}")
            _record_stage("confounds", stage_start, stage_rss)

        if self.pipeline_config.run_regression:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Trialwise regression", current=current_step, total=total_steps)
            logger.info("Running trialwise regression...")
            try:
                results.regression = _stage_regression_impl(ctx, self.pipeline_config)
            except Exception as exc:
                logger.warning(f"Regression stage failed: {exc}")
            _record_stage("regression", stage_start, stage_rss)

        if self.pipeline_config.run_models:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Model families", current=current_step, total=total_steps)
            logger.info("Running model families...")
            try:
                results.models = _stage_models_impl(ctx, self.pipeline_config)
            except Exception as exc:
                logger.warning(f"Models stage failed: {exc}")
            _record_stage("models", stage_start, stage_rss)

        if self.pipeline_config.run_stability:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Stability (run/block)", current=current_step, total=total_steps)
            logger.info("Running stability diagnostics...")
            try:
                results.stability = _stage_stability_impl(ctx, self.pipeline_config)
            except Exception as exc:
                logger.warning(f"Stability stage failed: {exc}")
            _record_stage("stability", stage_start, stage_rss)
        
        if self.pipeline_config.run_correlations or self.pipeline_config.compute_pain_sensitivity:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Running correlations", current=current_step, total=total_steps)
            logger.info("Running correlations...")
            results.correlations, results.pain_sensitivity = _stage_correlate_impl(ctx, self.pipeline_config)
            _record_stage("correlations", stage_start, stage_rss)

        if self.pipeline_config.run_consistency:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Consistency summary", current=current_step, total=total_steps)
            logger.info("Building effect-direction consistency summary...")
            try:
                results.consistency = _stage_consistency_impl(ctx, self.pipeline_config, results)
            except Exception as exc:
                logger.warning(f"Consistency stage failed: {exc}")
            _record_stage("consistency", stage_start, stage_rss)

        if self.pipeline_config.run_influence:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Influence diagnostics", current=current_step, total=total_steps)
            logger.info("Computing influence diagnostics...")
            try:
                results.influence = _stage_influence_impl(ctx, self.pipeline_config, results)
            except Exception as exc:
                logger.warning(f"Influence stage failed: {exc}")
            _record_stage("influence", stage_start, stage_rss)
        
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
            results.tf, results.temporal = _stage_temporal_impl(ctx)
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
        
        current_step += 1
        progress.step("Saving results", current=current_step, total=total_steps)
        logger.info("Saving results...")
        stage_start = time.perf_counter()
        stage_rss = _rss_mb()
        _stage_export_impl(ctx, self.pipeline_config, results)
        _record_stage("export", stage_start, stage_rss)

        if self._run_validation:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Global FDR correction", current=current_step, total=total_steps)
            logger.info("Running global FDR correction...")
            _stage_validate_impl(ctx, self.pipeline_config, results=results)
            _record_stage("fdr", stage_start, stage_rss)

        if self.pipeline_config.run_report:
            stage_start = time.perf_counter()
            stage_rss = _rss_mb()
            current_step += 1
            progress.step("Subject report", current=current_step, total=total_steps)
            logger.info("Writing subject report...")
            try:
                rp = _stage_report_impl(ctx, self.pipeline_config)
                if rp is not None:
                    results.report_path = str(rp)
            except Exception as exc:
                logger.warning("Report stage failed: %s", exc)
            _record_stage("report", stage_start, stage_rss)

        # Final metadata and manifest (must be after all stages including validation)
        outputs_manifest_path = ctx.stats_dir / "outputs_manifest.json"
        try:
            _write_analysis_metadata_impl(
                ctx, self.pipeline_config, results,
                stage_metrics=stage_metrics,
                outputs_manifest=outputs_manifest_path,
            )
        except Exception as e:
            logger.warning(f"Failed to write analysis metadata: {e}")
            
        # Write final outputs manifest and summary.json (re-writing it to include global FDR if needed)
        write_outputs_manifest(ctx, self.pipeline_config, results, stage_metrics)
        
        # Save summary.json again to ensure it includes Global FDR results
        summary = results.to_summary()
        from eeg_pipeline.infra.tsv import write_tsv
        (ctx.stats_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        
        # Also re-write normalized results if validation was run to include Global FDR q-values
        if self.pipeline_config.run_correlations or self.pipeline_config.run_condition_comparison:
             from eeg_pipeline.analysis.behavior.orchestration import _write_normalized_results
             _write_normalized_results(ctx, self.pipeline_config, results)

        outputs_log = [
            f"Complete: {summary.get('n_features', 0)} features",
            f"  Significant (raw): {summary.get('n_sig_raw', 0)}",
            f"  Significant (controlled): {summary.get('n_sig_controlled', 0)}",
            f"  Significant (Global FDR): {summary.get('n_sig_fdr', 0)}",
        ]
        if summary.get("n_clusters"):
            outputs_log.append(f"  Clusters identified: {summary.get('n_clusters')}")
            outputs_log.append(f"  Significant clusters: {summary.get('n_sig_clusters')}")
            
        logger.info(f"{'='*60}")
        for line in outputs_log:
            logger.info(line)
        logger.info(f"{'='*60}")
        
        progress.subject_done(f"sub-{subject}", success=True)
        
        return results


__all__ = [
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
]
