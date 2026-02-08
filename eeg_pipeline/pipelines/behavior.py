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
    run_behavior_stages,
    write_analysis_metadata as _write_analysis_metadata_impl,
    write_outputs_manifest,
    get_behavior_output_dir,
)


SIGNIFICANCE_THRESHOLD = 0.05

BEHAVIOR_COMPUTATION_FLAGS = [
    "trial_table",
    "lag_features",
    "pain_residual",
    "temperature_models",
    "regression",
    "models",
    "stability",
    "consistency",
    "influence",
    "report",
    "correlations",
    "multilevel_correlations",
    "pain_sensitivity",
    "condition",
    "temporal",
    "cluster",
    "mediation",
    "moderation",
    "mixed_effects",
]

# Bundled computation aliases for cleaner TUI/CLI
BEHAVIOR_COMPUTATION_BUNDLES = {
    "validation": ["consistency", "influence"],
}


def _resolve_behavior_computation_flags(
    requested: Optional[List[str]],
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, bool]]:
    """
    Normalize requested behavior computations into stage flags.
    
    If requested is None, returns None to indicate no override (use config).
    Handles bundled aliases like 'validation' -> ['consistency', 'influence'].
    """
    if requested is None:
        return None
    
    flags = {k: False for k in BEHAVIOR_COMPUTATION_FLAGS}
    
    # Expand bundled aliases
    expanded = []
    for item in requested:
        key = str(item).lower()
        if key in BEHAVIOR_COMPUTATION_BUNDLES:
            expanded.extend(BEHAVIOR_COMPUTATION_BUNDLES[key])
        else:
            expanded.append(key)
    
    unknown = [k for k in expanded if k not in BEHAVIOR_COMPUTATION_FLAGS]
    if unknown and logger:
        logger.warning("Ignoring unrecognized behavior computations: %s", ", ".join(sorted(set(unknown))))
    
    for key in expanded:
        if key in flags:
            flags[key] = True
    
    return flags


def _get_optional_int(config: Any, key: str, default: Optional[int]) -> Optional[int]:
    """Get optional integer from config, returning None if not set or explicitly None."""
    value = get_config_value(config, key, default)
    if value is None:
        return None
    return int(value)


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
    correlation_types: List[str] = field(default_factory=lambda: ["partial_cov_temp"])
    
    # Computation flags
    run_trial_table: bool = True
    run_lag_features: bool = True
    run_pain_residual: bool = True
    run_temperature_models: bool = True
    run_feature_qc: bool = False
    run_regression: bool = False
    run_models: bool = False
    run_stability: bool = True
    run_consistency: bool = True
    run_influence: bool = True
    run_report: bool = True
    run_correlations: bool = True
    run_multilevel_correlations: bool = False
    run_condition_comparison: bool = True
    run_temporal_correlations: bool = True
    run_cluster_tests: bool = False
    run_mediation: bool = False
    run_moderation: bool = False
    run_mixed_effects: bool = False
    
    # General stats
    fdr_alpha: float = 0.05
    n_permutations: int = 0
    n_jobs: int = -1
    
    # Condition-specific
    condition_effect_threshold: float = 0.5
    
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
    mediation_max_mediators: Optional[int] = 20  # None means unlimited
    
    # Moderation-specific
    moderation_max_features: Optional[int] = 50  # None means unlimited
    moderation_min_samples: int = 15
    
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
            bootstrap=int(
                get_config_value(
                    config,
                    "behavior_analysis.bootstrap",
                    get_config_value(config, "behavior_analysis.statistics.default_n_bootstrap", 1000),
                )
            ),
            robust_method=robust_method,
            method_label=method_label,
            correlation_types=get_config_value(config, "behavior_analysis.correlations.types", ["partial_cov_temp"]),
            run_trial_table=bool(get_config_value(config, "behavior_analysis.trial_table.enabled", True)),
            run_lag_features=bool(get_config_value(config, "behavior_analysis.lag_features.enabled", True)),
            run_pain_residual=bool(get_config_value(config, "behavior_analysis.pain_residual.enabled", True)),
            run_temperature_models=bool(get_config_value(config, "behavior_analysis.temperature_models.enabled", True)),
            run_feature_qc=bool(get_config_value(config, "behavior_analysis.feature_qc.enabled", False)),
            run_regression=bool(get_config_value(config, "behavior_analysis.regression.enabled", False)),
            run_models=bool(get_config_value(config, "behavior_analysis.models.enabled", False)),
            run_stability=bool(get_config_value(config, "behavior_analysis.stability.enabled", True)),
            run_consistency=bool(get_config_value(config, "behavior_analysis.consistency.enabled", True)),
            run_influence=bool(get_config_value(config, "behavior_analysis.influence.enabled", True)),
            run_report=bool(get_config_value(config, "behavior_analysis.report.enabled", True)),
            run_correlations=bool(get_config_value(config, "behavior_analysis.correlations.enabled", True)),
            run_condition_comparison=bool(get_config_value(config, "behavior_analysis.condition.enabled", True)),
            run_temporal_correlations=bool(get_config_value(config, "behavior_analysis.temporal.enabled", True)),
            run_cluster_tests=bool(get_config_value(config, "behavior_analysis.cluster.enabled", False)),
            run_mediation=bool(get_config_value(config, "behavior_analysis.mediation.enabled", False)),
            run_moderation=bool(get_config_value(config, "behavior_analysis.moderation.enabled", False)),
            run_mixed_effects=bool(get_config_value(config, "behavior_analysis.mixed_effects.enabled", False)),
            fdr_alpha=float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", 0.05)),
            n_permutations=int(
                get_config_value(
                    config,
                    "behavior_analysis.cluster.n_permutations",
                    get_config_value(config, "behavior_analysis.statistics.n_permutations", 0),
                )
            ),
            n_jobs=int(get_config_value(config, "behavior_analysis.n_jobs", -1)),
            # Condition-specific
            condition_effect_threshold=float(get_config_value(config, "behavior_analysis.condition.effect_size_threshold", 0.5)),
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
            mediation_max_mediators=_get_optional_int(config, "behavior_analysis.mediation.max_mediators", None),
            # Moderation-specific
            moderation_max_features=_get_optional_int(config, "behavior_analysis.moderation.max_features", None),
            moderation_min_samples=int(get_config_value(config, "behavior_analysis.moderation.min_samples", 15)),
            # Mixed effects-specific
            mixed_effects_type=str(get_config_value(config, "behavior_analysis.mixed_effects.random_effects", "intercept")),
            mixed_effects_max_features=int(get_config_value(config, "behavior_analysis.mixed_effects.max_features", 50)),
        )


def _extract_p_value_column(df: pd.DataFrame, primary_cols: List[str], fallback_cols: List[str]) -> Optional[pd.Series]:
    """Extract p-value column from dataframe using primary and fallback column names."""
    for col in primary_cols:
        if col in df.columns:
            return df[col]
    for col in fallback_cols:
        if col in df.columns:
            return df[col]
    return None


def _count_significant(p_values: Optional[pd.Series], threshold: float = SIGNIFICANCE_THRESHOLD) -> int:
    """Count significant p-values below threshold."""
    if p_values is None:
        return 0
    return int((p_values.fillna(1.0) < threshold).sum())


@dataclass
class BehaviorPipelineResults:
    subject: str
    trial_table_path: Optional[str] = None
    report_path: Optional[str] = None
    regression: Optional[pd.DataFrame] = None
    models: Optional[pd.DataFrame] = None
    stability: Optional[pd.DataFrame] = None
    consistency: Optional[pd.DataFrame] = None
    influence: Optional[pd.DataFrame] = None
    correlations: Optional[pd.DataFrame] = None
    pain_sensitivity: Optional[pd.DataFrame] = None
    condition_effects: Optional[pd.DataFrame] = None
    mediation: Optional[pd.DataFrame] = None
    moderation: Optional[pd.DataFrame] = None
    mixed_effects: Optional[pd.DataFrame] = None
    cluster: Optional[Dict[str, Any]] = None
    temporal: Optional[Dict[str, Any]] = None
    tf: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary(self) -> Dict[str, Any]:
        summary = {"subject": self.subject}
        if self.trial_table_path:
            summary["trial_table_path"] = self.trial_table_path
        if self.report_path:
            summary["report_path"] = self.report_path
        
        n_total = 0
        n_sig_raw = 0
        n_sig_controlled = 0
        n_sig_fdr = 0

        if self.correlations is not None and not self.correlations.empty:
            df = self.correlations
            n_total += len(df)
            
            p_raw = _extract_p_value_column(df, ["p_raw"], ["p_value", "p"])
            p_primary = _extract_p_value_column(df, ["p_primary"], [])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["q_value"])
            
            n_sig_raw += _count_significant(p_raw)
            n_sig_controlled += _count_significant(p_primary)
            n_sig_fdr += _count_significant(p_fdr)
        
        if self.pain_sensitivity is not None and not self.pain_sensitivity.empty:
            df = self.pain_sensitivity
            n_total += len(df)
            summary["n_pain_sensitivity_features"] = len(df)
            
            p_psi = _extract_p_value_column(df, ["p_psi"], ["p_value"])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["q_value"])
            
            n_sig_psi_raw = _count_significant(p_psi)
            n_sig_raw += n_sig_psi_raw
            summary["n_sig_psi_raw"] = n_sig_psi_raw
            n_sig_fdr += _count_significant(p_fdr)

        if self.condition_effects is not None and not self.condition_effects.empty:
            df = self.condition_effects
            n_total += len(df)
            summary["n_condition_effects"] = len(df)
            
            p_raw = _extract_p_value_column(df, ["p_value"], ["p"])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["q_value"])
            
            n_sig_raw += _count_significant(p_raw)
            n_sig_fdr += _count_significant(p_fdr)

        if self.regression is not None and not self.regression.empty:
            df = self.regression
            n_total += len(df)
            summary["n_regression_features"] = len(df)
            
            p_raw = _extract_p_value_column(df, ["p_primary"], ["p_feature"])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["p_fdr"])
            
            if p_raw is not None:
                p_raw_numeric = pd.to_numeric(p_raw, errors="coerce")
                n_sig_raw += _count_significant(p_raw_numeric)
            if p_fdr is not None:
                p_fdr_numeric = pd.to_numeric(p_fdr, errors="coerce")
                n_sig_fdr += _count_significant(p_fdr_numeric)
                
            if "hedges_g" in df.columns:
                summary["n_large_effects"] = int((df["hedges_g"].abs() >= 0.8).sum())

        if self.mediation is not None and not self.mediation.empty:
            df = self.mediation
            n_total += len(df)
            summary["n_mediation_mediators"] = len(df)
            
            p_raw = _extract_p_value_column(df, ["sobel_p"], ["p_value"])
            p_fdr = _extract_p_value_column(df, ["q_global"], [])
            
            n_sig_raw += _count_significant(p_raw)
            n_sig_fdr += _count_significant(p_fdr)

        if self.moderation is not None and not self.moderation.empty:
            df = self.moderation
            n_total += len(df)
            summary["n_moderation_features"] = len(df)
            
            p_raw = _extract_p_value_column(df, ["p_interaction"], ["p_value"])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["p_fdr"])
            
            n_sig_raw += _count_significant(p_raw)
            if p_fdr is not None:
                p_fdr_numeric = pd.to_numeric(p_fdr, errors="coerce")
                n_sig_fdr += _count_significant(p_fdr_numeric)

        if self.mixed_effects is not None and not self.mixed_effects.empty:
            df = self.mixed_effects
            n_total += len(df)
            summary["n_mixed_effects_features"] = len(df)
            
            p_raw = _extract_p_value_column(df, ["fixed_p"], ["p_value"])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["fixed_p_fdr"])
            
            n_sig_raw += _count_significant(p_raw)
            n_sig_fdr += _count_significant(p_fdr)

        if self.tf is not None:
            n_tests = self.tf.get("n_tests", 0)
            n_total += n_tests
            n_sig_raw += self.tf.get("n_sig_raw", 0)
            n_sig_fdr += self.tf.get("n_sig_fdr", 0)
            summary["n_tf_tests"] = n_tests

        if self.temporal is not None:
            n_tests = self.temporal.get("n_tests", 0)
            n_total += n_tests
            n_sig_raw += self.temporal.get("n_sig_raw", 0)
            n_sig_fdr += self.temporal.get("n_sig_fdr", 0)
            summary["n_temporal_tests"] = n_tests

        if self.cluster is not None:
            n_clusters = 0
            n_sig_clusters = 0
            for band, res in self.cluster.items():
                if isinstance(res, dict):
                    cluster_records = res.get("cluster_records", [])
                    n_clusters += len(cluster_records)
                    for record in cluster_records:
                        p_value = record.get("q_global", record.get("p_value", 1.0))
                        if p_value < SIGNIFICANCE_THRESHOLD:
                            n_sig_clusters += 1
            
            summary["n_clusters"] = n_clusters
            summary["n_sig_clusters"] = n_sig_clusters
            n_total += n_clusters
            n_sig_raw += n_sig_clusters
            n_sig_fdr += n_sig_clusters

        summary["n_features"] = n_total
        summary["n_sig_raw"] = n_sig_raw
        summary["n_sig_controlled"] = n_sig_controlled
        summary["n_sig_fdr"] = n_sig_fdr
        
        return summary


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
        
        comp_flags = _resolve_behavior_computation_flags(computations, logger=self.logger)
        if comp_flags is not None:
            any_requested = any(comp_flags.values())
            other_computations_requested = any(
                comp_flags[k] for k in comp_flags.keys() if k != "trial_table"
            )
            needs_trial_table = any_requested and other_computations_requested
            
            if needs_trial_table and not comp_flags.get("trial_table", False):
                self.logger.info("Auto-enabling `trial_table` (required by selected computations).")
                comp_flags["trial_table"] = True

            self.pipeline_config.run_trial_table = comp_flags["trial_table"]
            self.pipeline_config.run_lag_features = comp_flags["lag_features"]
            self.pipeline_config.run_pain_residual = comp_flags["pain_residual"]
            self.pipeline_config.run_temperature_models = comp_flags["temperature_models"]
            self.pipeline_config.run_regression = comp_flags["regression"]
            self.pipeline_config.run_models = comp_flags["models"]
            self.pipeline_config.run_stability = comp_flags["stability"]
            self.pipeline_config.run_consistency = comp_flags["consistency"]
            self.pipeline_config.run_influence = comp_flags["influence"]
            self.pipeline_config.run_report = comp_flags["report"]
            self.pipeline_config.run_correlations = comp_flags["correlations"]
            self.pipeline_config.run_multilevel_correlations = comp_flags["multilevel_correlations"]
            self.pipeline_config.run_condition_comparison = comp_flags["condition"]
            self.pipeline_config.run_temporal_correlations = comp_flags["temporal"]
            self.pipeline_config.run_cluster_tests = comp_flags["cluster"]
            self.pipeline_config.run_mediation = comp_flags["mediation"]
            self.pipeline_config.run_moderation = comp_flags["moderation"]
            self.pipeline_config.run_mixed_effects = comp_flags["mixed_effects"]
            self.pipeline_config.compute_pain_sensitivity = comp_flags["pain_sensitivity"]
            
            selected_computations = [k for k, v in comp_flags.items() if v]
            selected_text = ", ".join(selected_computations) if selected_computations else "none"
            self.logger.info("Behavior computations (override): %s", selected_text)

        if self.feature_categories:
            self.logger.info("Feature categories filter: %s", ", ".join(self.feature_categories))
        
        if self.computation_features:
            for comp, feats in self.computation_features.items():
                self.logger.info("  %s features: %s", comp, ", ".join(feats))

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> BehaviorPipelineResults:
        """Process a single subject using the DAG-based stage executor.
        
        This is the canonical entry point. All stage execution is delegated to
        run_behavior_stages() which resolves dependencies and runs stages in order.
        """
        from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
        from eeg_pipeline.infra.logging import get_subject_logger
        from eeg_pipeline.cli.common import ProgressReporter
        from eeg_pipeline.analysis.behavior.orchestration import _cache
        import time
        
        # Clear cache at pipeline entry point to prevent stale data
        _cache.clear()
        
        task = task or self.config.get("project.task", "thermalactive")
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        validate_only = bool(kwargs.get("validate_only", False))
        
        stats_dir = deriv_stats_path(self.deriv_root, subject)
        ensure_dir(stats_dir)
        
        logger = get_subject_logger("behavior_analysis", subject)
        
        logger.info("=== Behavior analysis: sub-%s, task-%s ===", subject, task)
        method_label = self.pipeline_config.method_label or self.pipeline_config.method
        controls = []
        if self.pipeline_config.control_temperature:
            controls.append("temperature")
        if self.pipeline_config.control_trial_order:
            controls.append("trial_order")
        logger.info(
            "Method: %s, controls: %s, bootstrap: %d, permutations: %d",
            method_label,
            ", ".join(controls) if controls else "none",
            self.pipeline_config.bootstrap,
            self.pipeline_config.n_permutations,
        )
        
        progress.subject_start(f"sub-{subject}")

        base_seed = int(get_config_value(self.config, "behavior_analysis.statistics.base_seed", 42))
        rng = np.random.default_rng(get_subject_seed(base_seed, subject))

        stats_cfg = self.config.get("behavior_analysis.statistics", {})
        partial_covars = stats_cfg.get("partial_covariates", None)
        
        output_cfg = self.config.get("behavior_analysis.output", {})
        also_save_csv = bool(output_cfg.get("also_save_csv", False))
        overwrite = bool(output_cfg.get("overwrite", True))

        # Build context (step 1)
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
            selected_bands=kwargs.get("bands"),
            computation_features=self.computation_features,
            also_save_csv=also_save_csv,
            overwrite=overwrite,
        )
        
        results = BehaviorPipelineResults(subject=subject)
        
        # Handle validate_only mode
        if validate_only:
            logger.info("Validation-only mode: running minimal stages.")
            ctx.data_qc["validate_only"] = True
            # Override config to only run load + trial_table + export
            self.pipeline_config.run_correlations = False
            self.pipeline_config.run_condition_comparison = False
            self.pipeline_config.run_temporal_correlations = False
            self.pipeline_config.run_cluster_tests = False
            self.pipeline_config.run_mediation = False
            self.pipeline_config.run_moderation = False
            self.pipeline_config.run_mixed_effects = False
        
        # Run all stages via DAG executor (step 2)
        start_time = time.perf_counter()
        try:
            run_behavior_stages(
                ctx=ctx,
                pipeline_config=self.pipeline_config,
                results=results,
                progress=progress,
            )
        except Exception as exc:
            logger.error(f"Pipeline failed: {exc}")
            progress.error("pipeline_failed", str(exc))
            progress.subject_done(f"sub-{subject}", success=False)
            return results
        
        elapsed = time.perf_counter() - start_time
        logger.info("Stage execution completed in %.1fs", elapsed)
        
        # Persist metadata (step 3)
        outputs_manifest_path = write_outputs_manifest(ctx, self.pipeline_config, results, {})
        try:
            _write_analysis_metadata_impl(
                ctx, self.pipeline_config, results,
                stage_metrics={},
                outputs_manifest=outputs_manifest_path,
            )
        except Exception as e:
            logger.warning(f"Failed to write analysis metadata: {e}")
        
        summary = results.to_summary()
        summary_dir = get_behavior_output_dir(ctx, "summary", ensure=True)
        summary_path = summary_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

        n_features = summary.get("n_features", 0)
        n_sig_raw = summary.get("n_sig_raw", 0)
        n_sig_controlled = summary.get("n_sig_controlled", 0)
        n_sig_fdr = summary.get("n_sig_fdr", 0)
        
        cluster_info = ""
        n_clusters = summary.get("n_clusters")
        if n_clusters:
            n_sig_clusters = summary.get("n_sig_clusters", 0)
            cluster_info = f", clusters: {n_sig_clusters}/{n_clusters} sig"
        logger.info(
            "Results: %d features tested, sig raw=%d, controlled=%d, FDR=%d%s (%.1fs)",
            n_features, n_sig_raw, n_sig_controlled, n_sig_fdr, cluster_info, elapsed,
        )
        
        progress.subject_done(f"sub-{subject}", success=True)
        
        return results
    
    def run_group_level(self, subjects: List[str], **kwargs) -> Any:
        """Run group-level behavior analysis across multiple subjects.
        
        Implements multi-subject analyses including:
        - Mixed-effects models with subject random effects
        - Multilevel correlations with block-restricted permutations
        - Hierarchical FDR correction by feature family
        
        Parameters
        ----------
        subjects : List[str]
            List of subject IDs to include
        **kwargs : dict
            run_mixed_effects : bool, default False
                Run mixed-effects models (only runs if explicitly requested)
            run_multilevel_correlations : bool, default False
                Run multilevel correlations with block-restricted permutations (opt-in)
            output_dir : Path, optional
                Custom output directory (default: deriv_root/group/stats)
        
        Returns
        -------
        GroupLevelResult
            Aggregated group-level results with mixed-effects and multilevel correlations
        """
        from eeg_pipeline.analysis.behavior.orchestration import (
            run_group_level_analysis,
            GroupLevelResult,
        )
        from eeg_pipeline.infra.paths import ensure_dir
        
        run_mixed_effects = kwargs.get("run_mixed_effects")
        if run_mixed_effects is None:
            run_mixed_effects = getattr(self.pipeline_config, "run_mixed_effects", False)
        
        run_multilevel_correlations = kwargs.get("run_multilevel_correlations")
        if run_multilevel_correlations is None:
            run_multilevel_correlations = getattr(self.pipeline_config, "run_multilevel_correlations", False)
        
        # Only run group-level analysis if at least one computation is enabled
        if not run_mixed_effects and not run_multilevel_correlations:
            return None
        
        output_dir = kwargs.get("output_dir")
        if output_dir is None:
            output_dir = self.deriv_root / "group" / "stats" / "behavior"
        
        ensure_dir(output_dir)
        
        self.logger.info("="*60)
        self.logger.info("Group-Level Behavior Analysis")
        self.logger.info("="*60)
        self.logger.info("Subjects (%d): %s", len(subjects), ", ".join(subjects))
        
        result = run_group_level_analysis(
            subjects=subjects,
            deriv_root=self.deriv_root,
            config=self.config,
            logger=self.logger,
            run_mixed_effects=run_mixed_effects,
            run_multilevel_correlations=run_multilevel_correlations,
            output_dir=output_dir,
        )
        
        if result.mixed_effects and result.mixed_effects.df is not None:
            n_sig = result.mixed_effects.n_significant
            self.logger.info("Mixed-effects: %d significant features", n_sig)
        
        if result.multilevel_correlations is not None and not result.multilevel_correlations.empty:
            q_values = result.multilevel_correlations.get("q_within_family", pd.Series([1.0]))
            n_sig = (q_values < SIGNIFICANCE_THRESHOLD).sum()
            self.logger.info("Multilevel correlations: %d significant", n_sig)
        
        self.logger.info("Group-level results saved to: %s", output_dir)
        
        return result


__all__ = [
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
]
