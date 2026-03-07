"""
Behavior Analysis Pipeline (Canonical)
======================================

Pipeline class for EEG-behavior correlation analysis.
This module provides the PipelineBase subclass for behavior analysis,
with statistical routines consolidated in eeg_pipeline.analysis.behavior.api.

Usage:
    pipeline = BehaviorPipeline(config=config)
    pipeline.process_subject("0001", "task")
    
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
from eeg_pipeline.pipelines.progress import ensure_progress_reporter
from eeg_pipeline.utils.analysis.stats.correlation import (
    format_correlation_method_label,
    normalize_robust_correlation_method,
)
from eeg_pipeline.utils.analysis.stats.base import get_subject_seed
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.analysis.behavior.config_resolver import resolve_correlation_method
from eeg_pipeline.analysis.behavior.stage_catalog import (
    COMPUTATION_TO_PIPELINE_ATTR,
    apply_computation_flags_impl as _apply_computation_flags_impl,
)
from eeg_pipeline.analysis.behavior.orchestration import (
    create_behavior_runtime,
    run_behavior_stages,
    write_analysis_metadata as _write_analysis_metadata_impl,
    write_outputs_manifest,
    get_behavior_output_dir,
)


SIGNIFICANCE_THRESHOLD = 0.05

BEHAVIOR_COMPUTATION_FLAGS = list(COMPUTATION_TO_PIPELINE_ATTR)

# Bundled computation aliases for cleaner TUI/CLI
BEHAVIOR_COMPUTATION_BUNDLES: dict[str, list[str]] = {}


def _resolve_behavior_computation_flags(
    requested: Optional[List[str]],
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, bool]]:
    """
    Normalize requested behavior computations into stage flags.
    
    If requested is None, returns None to indicate no override (use config).
    Bundled aliases are expanded through `BEHAVIOR_COMPUTATION_BUNDLES`.
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
    control_predictor: bool = True
    control_trial_order: bool = True
    compute_change_scores: bool = True
    compute_reliability: bool = False
    compute_bayes_factors: bool = False
    compute_loso_stability: bool = True
    bootstrap: int = 0
    robust_method: Optional[str] = None
    method_label: str = "spearman"
    correlation_types: List[str] = field(default_factory=lambda: ["partial_cov_predictor"])
    
    # Computation flags
    run_trial_table: bool = True
    run_predictor_residual: bool = True
    run_regression: bool = False
    run_icc: bool = True
    run_validation: bool = True
    run_report: bool = True
    run_correlations: bool = True
    run_multilevel_correlations: bool = False
    run_condition_comparison: bool = True
    run_temporal_correlations: bool = True
    run_cluster_tests: bool = False
    
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
    
    @classmethod
    def from_config(cls, config: Any) -> "BehaviorPipelineConfig":
        method = resolve_correlation_method(config, default="spearman")
        robust_method = normalize_robust_correlation_method(
            get_config_value(config, "behavior_analysis.robust_correlation", None),
            default=None,
            strict=True,
        )
        method_label = format_correlation_method_label(method, robust_method)
        return cls(
            method=method,
            min_samples=int(get_config_value(config, "behavior_analysis.min_samples.default", 10)),
            control_predictor=bool(get_config_value(config, "behavior_analysis.predictor_control_enabled", True)),
            control_trial_order=bool(get_config_value(config, "behavior_analysis.control_trial_order", True)),
            compute_change_scores=bool(get_config_value(config, "behavior_analysis.correlations.compute_change_scores", True)),
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
            correlation_types=get_config_value(config, "behavior_analysis.correlations.types", ["partial_cov_predictor"]),
            run_trial_table=bool(get_config_value(config, "behavior_analysis.trial_table.enabled", True)),
            run_predictor_residual=bool(get_config_value(config, "behavior_analysis.predictor_residual.enabled", True)),
            run_regression=bool(get_config_value(config, "behavior_analysis.regression.enabled", False)),
            run_icc=bool(
                get_config_value(
                    config,
                    "behavior_analysis.icc.enabled",
                    True,
                )
            ),
            run_validation=bool(get_config_value(config, "behavior_analysis.validation.enabled", True)),
            run_report=bool(get_config_value(config, "behavior_analysis.report.enabled", True)),
            run_correlations=bool(get_config_value(config, "behavior_analysis.correlations.enabled", True)),
            run_condition_comparison=bool(get_config_value(config, "behavior_analysis.condition.enabled", True)),
            run_temporal_correlations=bool(get_config_value(config, "behavior_analysis.temporal.enabled", True)),
            run_cluster_tests=bool(get_config_value(config, "behavior_analysis.cluster.enabled", False)),
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
    icc: Optional[pd.DataFrame] = None
    correlations: Optional[pd.DataFrame] = None
    condition_effects: Optional[pd.DataFrame] = None
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
        
        if self.condition_effects is not None and not self.condition_effects.empty:
            df = self.condition_effects
            n_total += len(df)
            summary["n_condition_effects"] = len(df)
            
            p_raw = _extract_p_value_column(df, ["p_raw", "p_value"], ["p"])
            p_primary = _extract_p_value_column(df, ["p_primary"], [])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["q_value"])
            
            n_sig_raw += _count_significant(p_raw)
            n_sig_controlled += _count_significant(p_primary)
            n_sig_fdr += _count_significant(p_fdr)

        if self.regression is not None and not self.regression.empty:
            df = self.regression
            n_total += len(df)
            summary["n_regression_features"] = len(df)
            
            p_raw = _extract_p_value_column(df, ["p_raw", "p_feature"], [])
            p_primary = _extract_p_value_column(df, ["p_primary"], ["p_feature"])
            p_fdr = _extract_p_value_column(df, ["q_global"], ["p_fdr"])
            
            if p_raw is not None:
                p_raw_numeric = pd.to_numeric(p_raw, errors="coerce")
                n_sig_raw += _count_significant(p_raw_numeric)
            if p_primary is not None:
                p_primary_numeric = pd.to_numeric(p_primary, errors="coerce")
                n_sig_controlled += _count_significant(p_primary_numeric)
            if p_fdr is not None:
                p_fdr_numeric = pd.to_numeric(p_fdr, errors="coerce")
                n_sig_fdr += _count_significant(p_fdr_numeric)
                
            if "hedges_g" in df.columns:
                summary["n_large_effects"] = int((df["hedges_g"].abs() >= 0.8).sum())

        if self.icc is not None and not self.icc.empty:
            df = self.icc
            n_total += len(df)
            summary["n_icc_features"] = len(df)

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

            _apply_computation_flags_impl(self.pipeline_config, comp_flags)
            
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
        import time
        
        task = task or self.config.get("project.task", "task")
        progress = ensure_progress_reporter(kwargs.get("progress"))
        stats_dir = deriv_stats_path(self.deriv_root, subject)
        ensure_dir(stats_dir)
        
        logger = get_subject_logger("behavior_analysis", subject)
        
        logger.info("=== Behavior analysis: sub-%s, task-%s ===", subject, task)
        method_label = (
            getattr(self.pipeline_config, "method_label", None)
            or getattr(self.pipeline_config, "method", "spearman")
        )
        controls = []
        if self.pipeline_config.control_predictor:
            controls.append("predictor")
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
            control_predictor=self.pipeline_config.control_predictor,
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
        # Isolated runtime per subject prevents cross-subject cache leakage.
        setattr(ctx, "_behavior_runtime", create_behavior_runtime())
        
        results = BehaviorPipelineResults(subject=subject)
        
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
            progress.error("pipeline_failed", str(exc))
            progress.subject_done(f"sub-{subject}", success=False)
            raise
        
        elapsed = time.perf_counter() - start_time
        logger.info("Stage execution completed in %.1fs", elapsed)
        
        # Persist metadata (step 3)
        outputs_manifest_path = write_outputs_manifest(ctx, self.pipeline_config, results, {})
        _write_analysis_metadata_impl(
            ctx, self.pipeline_config, results,
            stage_metrics={},
            outputs_manifest=outputs_manifest_path,
        )
        
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
            logger.info("Clusters identified: %d total, %d significant", n_clusters, n_sig_clusters)
        logger.info(
            "Results: %d features tested, sig raw=%d, controlled=%d, FDR=%d%s (%.1fs)",
            n_features, n_sig_raw, n_sig_controlled, n_sig_fdr, cluster_info, elapsed,
        )
        
        progress.subject_done(f"sub-{subject}", success=True)
        
        return results
    
    def run_group_level(self, subjects: List[str], **kwargs) -> Any:
        """Run group-level behavior analysis across multiple subjects.
        
        Implements multi-subject analyses including:
        - Multilevel correlations with block-restricted permutations
        
        Parameters
        ----------
        subjects : List[str]
            List of subject IDs to include
        **kwargs : dict
            run_multilevel_correlations : bool, default False
                Run multilevel correlations with block-restricted permutations (opt-in)
            output_dir : Path, optional
                Custom output directory (default: deriv_root/group/stats)
        
        Returns
        -------
        GroupLevelResult
            Aggregated group-level results
        """
        from eeg_pipeline.analysis.behavior.orchestration import (
            run_group_level_analysis,
        )
        from eeg_pipeline.infra.paths import ensure_dir
        
        run_multilevel_correlations = kwargs.get("run_multilevel_correlations")
        if run_multilevel_correlations is None:
            pipeline_flag = bool(
                getattr(self.pipeline_config, "run_multilevel_correlations", False)
            )
            run_multilevel_correlations = not pipeline_flag
        
        # Only run group-level analysis if at least one computation is enabled
        if not run_multilevel_correlations:
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
            run_multilevel_correlations=run_multilevel_correlations,
            output_dir=output_dir,
        )
        
        if result.multilevel_correlations is not None and not result.multilevel_correlations.empty:
            reject = result.multilevel_correlations.get("reject_within_family")
            if reject is not None:
                n_sig = int(pd.Series(reject).fillna(False).astype(bool).sum())
            else:
                q_values = result.multilevel_correlations.get("q_within_family", pd.Series([1.0]))
                n_sig = int((pd.to_numeric(q_values, errors="coerce") < SIGNIFICANCE_THRESHOLD).sum())
            self.logger.info("Multilevel correlations: %d significant", n_sig)
        
        self.logger.info("Group-level results saved to: %s", output_dir)
        
        return result


__all__ = [
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
]
