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
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.analysis.stats.reliability import get_subject_seed
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
    "export",
]

# Legacy aliases supported for backward-compatible CLI choices
_BEHAVIOR_COMP_ALIASES = {
    "condition_correlations": "condition",
    "time_frequency": "temporal",
    "temporal_correlations": "temporal",
    "cluster_test": "cluster",
    "exports": "export",
    "precomputed_correlations": "correlations",
    "power_roi": "correlations",
    "connectivity_roi": "correlations",
    "connectivity_heatmaps": "correlations",
    "sliding_connectivity": "temporal",
}


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
    normalized: List[str] = []
    for item in requested:
        key = str(item).lower()
        key = _BEHAVIOR_COMP_ALIASES.get(key, key)
        normalized.append(key)
    
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
    compute_reliability: bool = True
    bootstrap: int = 0
    
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
            control_temperature=bool(get_config_value(config, "behavior_analysis.control_temperature", True)),
            control_trial_order=bool(get_config_value(config, "behavior_analysis.control_trial_order", True)),
            compute_change_scores=bool(get_config_value(config, "behavior_analysis.compute_change_scores", True)),
            compute_pain_sensitivity=bool(get_config_value(config, "behavior_analysis.compute_pain_sensitivity", True)),
            compute_reliability=bool(get_config_value(config, "behavior_analysis.statistics.compute_reliability", True)),
            bootstrap=int(get_config_value(config, "behavior_analysis.bootstrap", get_config_value(config, "behavior_analysis.statistics.default_n_bootstrap", 0))),
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
# Helper Functions
###################################################################


def _combine_features(ctx: BehaviorContext) -> pd.DataFrame:
    return _combine_features_impl(ctx)


def _add_change_scores(ctx: BehaviorContext) -> None:
    """Compute and append change scores (plateau-baseline) once per context."""
    return _add_change_scores_impl(ctx)


###################################################################
# Pipeline Stages
###################################################################


def _stage_load(ctx: BehaviorContext) -> bool:
    return _stage_load_impl(ctx)


def _stage_correlate(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
) -> tuple:
    return _stage_correlate_impl(ctx, config)


def _build_behavior_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    return _build_behavior_qc_impl(ctx)


def _write_analysis_metadata(
    ctx: BehaviorContext,
    pipeline_config: BehaviorPipelineConfig,
    results: BehaviorPipelineResults,
) -> None:
    return _write_analysis_metadata_impl(ctx, pipeline_config, results)


def _stage_condition(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
) -> pd.DataFrame:
    return _stage_condition_impl(ctx, config)


def _stage_temporal(ctx: BehaviorContext) -> None:
    return _stage_temporal_impl(ctx)


def _stage_cluster(ctx: BehaviorContext, config: BehaviorPipelineConfig) -> Dict[str, Any]:
    return _stage_cluster_impl(ctx, config)


def _stage_advanced(
    ctx: BehaviorContext,
    config: BehaviorPipelineConfig,
    results: BehaviorPipelineResults,
) -> None:
    return _stage_advanced_impl(ctx, config, results)


def _stage_validate(ctx: BehaviorContext, config: BehaviorPipelineConfig) -> None:
    return _stage_validate_impl(ctx, config)


def _stage_export(
    ctx: BehaviorContext,
    results: BehaviorPipelineResults,
) -> List[Path]:
    return _stage_export_impl(ctx, results)


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
    ):
        super().__init__(name="behavior_analysis", config=config)
        self.pipeline_config = pipeline_config or BehaviorPipelineConfig.from_config(self.config)
        self.feature_categories = feature_categories
        
        self._run_correlations = True
        self._run_export = True
        self._run_validation = True
        
        comp_flags = _resolve_behavior_computation_flags(computations, logger=self.logger)
        if comp_flags is not None:
            self._run_correlations = comp_flags["correlations"]
            self._run_export = comp_flags["export"]
            self.pipeline_config.run_condition_comparison = comp_flags["condition"]
            self.pipeline_config.run_temporal_correlations = comp_flags["temporal"]
            self.pipeline_config.run_cluster_tests = comp_flags["cluster"]
            self.pipeline_config.run_mediation = comp_flags["mediation"]
            self.pipeline_config.run_mixed_effects = comp_flags["mixed_effects"]
            self.pipeline_config.compute_pain_sensitivity = comp_flags["pain_sensitivity"]
            self._run_validation = any(
                comp_flags[key] for key in BEHAVIOR_COMPUTATION_FLAGS if key != "export"
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

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> BehaviorPipelineResults:
        from eeg_pipeline.infra.paths import deriv_stats_path, ensure_dir
        from eeg_pipeline.infra.logging import get_subject_logger
        
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

        base_seed = 42
        try:
            base_seed = int(self.config.get("behavior_analysis.statistics.base_seed", base_seed))
        except Exception:
            base_seed = 42
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
            feature_categories=self.feature_categories,
        )
        
        results = BehaviorPipelineResults(subject=subject)
        
        logger.info("Stage 1: Loading data...")
        if not _stage_load(ctx):
            return results
        
        if self._run_correlations:
            logger.info("Stage 2: Running unified correlations...")
            results.correlations, results.pain_sensitivity = _stage_correlate(ctx, self.pipeline_config)
        else:
            logger.info("Stage 2: Skipped correlations (CLI selection)")
        
        if self.pipeline_config.run_condition_comparison:
            logger.info("Stage 3: Condition comparison...")
            results.condition_effects = _stage_condition(ctx, self.pipeline_config)
        else:
            logger.info("Stage 3: Skipped condition comparison (CLI selection)")
        
        if self.pipeline_config.run_temporal_correlations:
            logger.info("Stage 4: Temporal correlations...")
            _stage_temporal(ctx)
        else:
            logger.info("Stage 4: Skipped temporal correlations (CLI selection)")
        
        if self.pipeline_config.run_cluster_tests:
            logger.info("Stage 5: Cluster permutation tests...")
            results.cluster = _stage_cluster(ctx, self.pipeline_config)
        else:
            logger.info("Stage 5: Skipped cluster tests (CLI selection)")
        
        if self.pipeline_config.run_mediation or self.pipeline_config.run_mixed_effects:
            logger.info("Stage 6: Advanced analyses...")
            _stage_advanced(ctx, self.pipeline_config, results)
        else:
            logger.info("Stage 6: Skipped advanced analyses (CLI selection)")
        
        if self._run_validation:
            logger.info("Stage 7: Global FDR correction...")
            _stage_validate(ctx, self.pipeline_config)
        else:
            logger.info("Stage 7: Skipped global FDR (no computations selected)")
        
        if self._run_export:
            logger.info("Stage 8: Saving results...")
            _stage_export(ctx, results)
        else:
            logger.info("Stage 8: Skipped export (CLI selection)")

        try:
            _write_analysis_metadata(ctx, self.pipeline_config, results)
        except Exception as e:
            logger.warning(f"Failed to write analysis metadata: {e}")
        
        summary = results.to_summary()
        logger.info(f"{'='*60}")
        logger.info(f"Complete: {summary.get('n_features', 0)} features")
        logger.info(f"  Significant (raw): {summary.get('n_sig_raw', 0)}")
        logger.info(f"  Significant (controlled): {summary.get('n_sig_controlled', 0)}")
        logger.info(f"{'='*60}")
        
        return results


__all__ = [
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
]
