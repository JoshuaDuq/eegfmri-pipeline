"""
Behavior Analysis Pipeline.

Orchestrates brain-behavior correlation analysis for EEG data.

Usage:
    # Single subject
    process_subject("0001", task="thermalactive", deriv_root=deriv_root, config=config)

    # Multiple subjects
    compute_behavior_correlations_for_subjects(["0001", "0002"], run_group_aggregation=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from eeg_pipeline.analysis.behavior.core import (
    BehaviorContext,
    ComputationResult,
    ComputationStatus,
)


###################################################################
# Computation Registry
###################################################################

ALL_COMPUTATIONS = [
    "power_roi",
    "connectivity_roi",
    "connectivity_heatmaps",
    "sliding_connectivity",
    "time_frequency",
    "temporal_correlations",
    "cluster_test",
    "precomputed_correlations",
    "condition_correlations",
    "exports",
]


def _make_computation(
    name: str,
    func: Callable[[BehaviorContext], None],
    critical: bool = True,
) -> Callable[[BehaviorContext], ComputationResult]:
    """Factory for computation wrappers with consistent error handling."""

    def wrapper(ctx: BehaviorContext) -> ComputationResult:
        try:
            func(ctx)
            return ComputationResult(name=name, status=ComputationStatus.SUCCESS)
        except Exception as exc:
            log_method = ctx.logger.error if critical else ctx.logger.warning
            log_method(f"{name} failed: {exc}")
            return ComputationResult(name=name, status=ComputationStatus.FAILED, error=str(exc))

    return wrapper


def _get_computation_registry() -> Dict[str, Callable[[BehaviorContext], ComputationResult]]:
    """Build computation registry with lazy imports."""
    from eeg_pipeline.analysis.behavior.power_roi import compute_power_roi_stats_from_context
    from eeg_pipeline.analysis.behavior.connectivity import (
        correlate_connectivity_roi_from_context,
        correlate_connectivity_heatmaps,
        _correlate_sliding_connectivity,
    )
    from eeg_pipeline.analysis.behavior.temporal import (
        compute_time_frequency_from_context,
        compute_temporal_from_context,
    )
    from eeg_pipeline.analysis.behavior.cluster_tests import run_cluster_test_from_context
    from eeg_pipeline.analysis.behavior.precomputed_correlations import compute_precomputed_correlations
    from eeg_pipeline.analysis.behavior.condition_correlations import compute_condition_correlations
    from eeg_pipeline.analysis.behavior.exports import export_combined_power_corr_stats

    def _sliding_connectivity(ctx: BehaviorContext) -> None:
        if ctx.connectivity_df is None:
            return
        _correlate_sliding_connectivity(
            conn_df=ctx.connectivity_df,
            ratings=ctx.targets,
            config=ctx.config,
            stats_dir=ctx.stats_dir,
            logger=ctx.logger,
            use_spearman=ctx.use_spearman,
        )

    def _connectivity_heatmaps(ctx: BehaviorContext) -> None:
        correlate_connectivity_heatmaps(ctx.subject, ctx.task, use_spearman=ctx.use_spearman)

    def _exports(ctx: BehaviorContext) -> None:
        export_combined_power_corr_stats(ctx.subject)

    return {
        "power_roi": _make_computation("power_roi", compute_power_roi_stats_from_context),
        "connectivity_roi": _make_computation("connectivity_roi", correlate_connectivity_roi_from_context),
        "connectivity_heatmaps": _make_computation("connectivity_heatmaps", _connectivity_heatmaps),
        "sliding_connectivity": _make_computation("sliding_connectivity", _sliding_connectivity),
        "time_frequency": _make_computation("time_frequency", compute_time_frequency_from_context),
        "temporal_correlations": _make_computation("temporal_correlations", compute_temporal_from_context),
        "cluster_test": _make_computation("cluster_test", run_cluster_test_from_context, critical=False),
        "precomputed_correlations": lambda ctx: compute_precomputed_correlations(ctx),
        "condition_correlations": lambda ctx: compute_condition_correlations(ctx),
        "exports": _make_computation("exports", _exports),
    }


###################################################################
# Context Creation
###################################################################


def initialize_analysis_context(
    subject: str,
    task: Optional[str],
    config: Any = None,
):
    """
    Initialize analysis context for behavior analysis.
    
    Returns (config, task, deriv_root, stats_dir, logger).
    """
    from eeg_pipeline.utils.config.loader import load_settings
    from eeg_pipeline.utils.io.general import deriv_stats_path, ensure_dir, get_subject_logger

    if not subject:
        raise ValueError("Subject must be provided")

    if config is None:
        config = load_settings()

    if task is None:
        task = config.get("project.task", "thermalactive")

    logger = get_subject_logger(
        "behavior_analysis",
        subject,
        config.get("output.log_file_name", "behavior_analysis.log"),
        config=config,
    )

    deriv_root = Path(config.deriv_root)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    return config, task, deriv_root, stats_dir, logger


def create_context(
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    logger: logging.Logger,
    *,
    use_spearman: bool = True,
    bootstrap: int = 0,
    n_perm: int = 100,
    rng_seed: int = 42,
) -> BehaviorContext:
    """Create BehaviorContext for a subject."""
    from eeg_pipeline.utils.io.general import deriv_stats_path, ensure_dir
    from eeg_pipeline.utils.analysis.reliability import get_subject_seed

    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(stats_dir)

    return BehaviorContext(
        subject=subject,
        task=task,
        config=config,
        logger=logger,
        deriv_root=deriv_root,
        stats_dir=stats_dir,
        use_spearman=use_spearman,
        bootstrap=bootstrap,
        n_perm=n_perm,
        rng=np.random.default_rng(get_subject_seed(rng_seed, subject)),
    )


###################################################################
# Pipeline Execution
###################################################################


def run_computations(ctx: BehaviorContext, computations: Optional[List[str]] = None) -> None:
    """Run selected computations on a context."""
    from eeg_pipeline.utils.progress import PipelineProgress

    to_run = list(computations) if computations else list(ALL_COMPUTATIONS)

    # Always include condition_correlations
    if "condition_correlations" not in to_run:
        to_run.append("condition_correlations")
        ctx.logger.info("Auto-including 'condition_correlations'")

    registry = _get_computation_registry()
    valid = [c for c in to_run if c in registry]

    if not valid:
        ctx.logger.warning("No valid computations to run")
        return

    progress = PipelineProgress(total=len(valid), logger=ctx.logger, desc="Behavior")
    progress.start()

    for name in valid:
        result = registry[name](ctx)
        ctx.add_result(name, result)
        progress.step(message=f"Completed {name}")

    progress.finish()


def apply_fdr_and_export(ctx: BehaviorContext, alpha: Optional[float] = None) -> None:
    """Apply global FDR correction and export significant predictors."""
    from eeg_pipeline.analysis.behavior.fdr_correction import apply_global_fdr
    from eeg_pipeline.analysis.behavior.exports import export_all_significant_predictors

    ctx.logger.info("Applying global FDR correction...")

    try:
        apply_global_fdr(ctx.subject)
    except Exception as exc:
        ctx.logger.error(f"FDR correction failed: {exc}")
        raise RuntimeError("FDR correction failed - required for valid inference") from exc

    sig_alpha = alpha or float(ctx.config.get("statistics.sig_alpha", 0.05))
    ctx.logger.info(f"Exporting significant predictors (alpha={sig_alpha})...")
    export_all_significant_predictors(ctx.subject, alpha=sig_alpha, use_fdr=True)


###################################################################
# Public API
###################################################################


def process_subject(
    subject: str,
    deriv_root: Path,
    task: str,
    config: Any,
    logger: Optional[logging.Logger] = None,
    correlation_method: str = "spearman",
    bootstrap: int = 0,
    n_perm: int = 0,
    rng_seed: int = 42,
    computations: Optional[List[str]] = None,
) -> None:
    """Process a single subject for behavior analysis."""
    from eeg_pipeline.utils.io.general import get_subject_logger

    if not subject or not task:
        raise ValueError(f"subject and task required, got: {subject=}, {task=}")

    if logger is None:
        logger = get_subject_logger(
            "behavior_analysis",
            subject,
            config.get("output.log_file_name", "behavior_analysis.log"),
            config=config,
        )

    logger.info(f"=== Behavior analysis: sub-{subject}, task-{task} ===")

    # Resolve n_perm from config
    if n_perm == 0:
        n_perm = int(config.get("behavior_analysis.statistics.n_permutations", 100))

    ctx = create_context(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        config=config,
        logger=logger,
        use_spearman=(correlation_method == "spearman"),
        bootstrap=bootstrap,
        n_perm=n_perm,
        rng_seed=rng_seed,
    )

    if not ctx.load_data():
        logger.warning("No target variable found; skipping")
        return

    run_computations(ctx, computations)
    apply_fdr_and_export(ctx)

    n_success = sum(1 for r in ctx.results.values() if r.status == ComputationStatus.SUCCESS)
    n_failed = sum(1 for r in ctx.results.values() if r.status == ComputationStatus.FAILED)
    logger.info(f"Completed: {n_success} succeeded, {n_failed} failed")


def compute_behavior_correlations_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    correlation_method: Optional[str] = None,
    bootstrap: int = 0,
    n_perm: Optional[int] = None,
    rng_seed: int = 42,
    run_group_aggregation: bool = True,
    computations: Optional[List[str]] = None,
) -> None:
    """Run behavior analysis for multiple subjects."""
    from eeg_pipeline.utils.config.loader import load_settings
    from eeg_pipeline.utils.io.general import (
        setup_matplotlib,
        ensure_derivatives_dataset_description,
        get_logger,
    )
    from eeg_pipeline.utils.progress import BatchProgress

    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        config = load_settings()

    if deriv_root is None:
        deriv_root = Path(config.deriv_root)

    setup_matplotlib(config)
    ensure_derivatives_dataset_description(deriv_root=deriv_root)

    task = task or config.get("project.task", "thermalactive")
    correlation_method = correlation_method or config.get(
        "behavior_analysis.statistics.correlation_method", "spearman"
    )
    n_perm = n_perm or int(config.get("behavior_analysis.statistics.n_permutations", 100))

    logger = get_logger(__name__)

    with BatchProgress(subjects=subjects, logger=logger, desc="Behavior Analysis") as batch:
        for subject in subjects:
            start_time = batch.start_subject(subject)
            try:
                process_subject(
                    subject=subject,
                    deriv_root=deriv_root,
                    task=task,
                    config=config,
                    correlation_method=correlation_method,
                    bootstrap=bootstrap,
                    n_perm=n_perm,
                    rng_seed=rng_seed,
                    computations=computations,
                )
                batch.finish_subject(subject, start_time)
            except Exception as exc:
                logger.error(f"Failed sub-{subject}: {exc}")
                batch.finish_subject(subject, start_time)

    if run_group_aggregation and len(subjects) >= 2:
        logger.info("Running group-level aggregation...")
        from eeg_pipeline.analysis.group import aggregate_behavior_correlations

        aggregate_behavior_correlations(
            subjects=subjects,
            task=task,
            deriv_root=deriv_root,
            pooling_strategy="within_subject_centered",
            config=config,
        )
