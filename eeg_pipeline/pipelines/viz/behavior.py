"""Behavior visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for behavioral visualizations.
Plot primitives live in `eeg_pipeline.plotting.behavioral.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from eeg_pipeline.utils.data.loading import load_behavior_stats_files
from eeg_pipeline.io.paths import deriv_plots_path, deriv_stats_path, ensure_dir, resolve_deriv_root
from eeg_pipeline.io.plot_collections import collect_significant_plots
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.io.logging import get_logger
from eeg_pipeline.plotting.behavioral.registry import (
    BehaviorPlotContext,
    BehaviorPlotManager,
)

# Import plotters for side-effects: registers plot functions into BehaviorPlotRegistry.
# This import must not call back into this module at import time.
from eeg_pipeline.plotting.behavioral import registrations as _behavior_plotters  # noqa: F401


###################################################################
# Subject-level Behavioral Visualization
###################################################################


def _build_behavior_plot_context(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
) -> BehaviorPlotContext:
    """Create a context object shared by all behavioral plotters."""
    plots_dir = deriv_plots_path(deriv_root, subject, subdir="behavior")
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    stats_config = config.get("behavior_analysis", {}).get("statistics", {})
    use_spearman = stats_config.get("correlation_method") == "spearman"
    rating_stats, temp_stats = load_behavior_stats_files(stats_dir, logger)

    return BehaviorPlotContext(
        subject=subject,
        task=task,
        config=config,
        logger=logger,
        deriv_root=deriv_root,
        plots_dir=plots_dir,
        stats_dir=stats_dir,
        use_spearman=use_spearman,
        rating_stats=rating_stats,
        temp_stats=temp_stats,
        all_results=[],
    )


def visualize_subject_behavior(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    scatter_only: bool = False,
    temporal_only: bool = False,
    plots: Optional[List[str]] = None,
    deriv_root: Optional[Path] = None,
) -> None:
    """Visualize behavioral correlations for a single subject."""
    logger.info(f"Visualizing behavioral correlations for sub-{subject}...")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    ctx = _build_behavior_plot_context(subject, task, effective_deriv_root, config, logger)
    manager = BehaviorPlotManager(ctx)

    if scatter_only and temporal_only:
        raise ValueError("Cannot specify both scatter_only and temporal_only")

    if scatter_only:
        plot_names = [
            "psychometrics",
            "power_roi_scatter",
            "dynamics_scatter",
            "aperiodic_scatter",
            "connectivity_scatter",
            "itpc_scatter",
        ]
        logger.info("Running scatter-only behavioral plots via registry...")
        manager.run_selected(plot_names)
    elif temporal_only:
        logger.info("Running temporal-only behavioral plots via registry...")
        manager.run_selected(["temporal_topomaps"])
    else:
        plot_names = plots if plots is not None else [
            "psychometrics",
            "power_roi_scatter",
            "dynamics_scatter",
            "aperiodic_scatter",
            "connectivity_scatter",
            "itpc_scatter",
            "temporal_topomaps",
            "pain_clusters",
            "dose_response",
        ]
        logger.info("Running behavioral plots via registry...")
        manager.run_selected(plot_names) if plots else manager.run_all()

    if ctx.all_results:
        logger.info("Collecting significant plots...")
        collect_significant_plots(subject, effective_deriv_root, ctx.all_results, config=config)

    logger.info(f"Behavioral visualizations saved to {ctx.plots_dir}")


###################################################################
# Batch Processing
###################################################################


def visualize_behavior_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    logger: Optional[logging.Logger] = None,
    scatter_only: bool = False,
    temporal_only: bool = False,
) -> None:
    """Batch process behavioral visualizations for multiple subjects."""
    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        from eeg_pipeline.utils.config.loader import load_settings

        config = load_settings()

    setup_matplotlib(config)

    task = task or config.get("project.task", "thermalactive")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    if scatter_only and temporal_only:
        raise ValueError("Cannot specify both scatter_only and temporal_only")

    mode_str = "scatter-only" if scatter_only else "temporal-only" if temporal_only else "full"
    logger.info(
        f"Starting behavioral visualization ({mode_str}): {len(subjects)} subject(s), task={task}"
    )

    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        visualize_subject_behavior(
            subject,
            task,
            config,
            logger,
            scatter_only=scatter_only,
            temporal_only=temporal_only,
            deriv_root=effective_deriv_root,
        )

    logger.info("Behavioral visualization complete")


__all__ = [
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
]

