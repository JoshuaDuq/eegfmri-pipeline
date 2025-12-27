"""Behavior visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for behavioral visualizations.
Plot primitives live in `eeg_pipeline.plotting.behavioral.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from eeg_pipeline.utils.data.behavior import load_behavior_stats_files
from eeg_pipeline.infra.paths import deriv_plots_path, deriv_stats_path, ensure_dir, resolve_deriv_root
from eeg_pipeline.plotting.io.collections import collect_significant_plots
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.plotting.behavioral.registry import (
    BehaviorPlotContext,
    BehaviorPlotManager,
)
from eeg_pipeline.utils.analysis.stats.correlation import (
    format_correlation_method_label,
    normalize_correlation_method,
)
from eeg_pipeline.utils.config.loader import get_config_value

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
    plot_cfg = get_plot_config(config)
    behavioral_config = plot_cfg.get_behavioral_config()
    plot_subdir = behavioral_config.get("plot_subdir", "behavior")

    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)

    raw_method = get_config_value(config, "behavior_analysis.statistics.correlation_method", None)
    if raw_method is None:
        raw_method = get_config_value(config, "behavior_analysis.correlation_method", "spearman")
    method = normalize_correlation_method(raw_method, default="spearman")
    use_spearman = method == "spearman"
    robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
    if robust_method is not None:
        robust_method = str(robust_method).strip().lower() or None
    method_label = format_correlation_method_label(method, robust_method)
    rating_stats, temp_stats = load_behavior_stats_files(stats_dir, logger, method_label=method_label)

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


CATEGORY_TO_PLOTS = {
    "psychometrics": ["psychometrics"],
    "power": ["power_roi_scatter"],
    "complexity": ["complexity_scatter"],
    "aperiodic": ["aperiodic_scatter"],
    "connectivity": ["connectivity_scatter"],
    "itpc": ["itpc_scatter"],
    "temporal": ["temporal_topomaps", "pain_clusters"],
    "dose_response": ["dose_response"],
    "trial_table": ["trial_table_overview"],
    "confounds": ["confounds_audit"],
    "regression": ["regression_summary"],
    "temperature_models": ["temperature_models"],
    "stability": ["stability_groupwise"],
}


def visualize_subject_behavior(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    scatter_only: bool = False,
    temporal_only: bool = False,
    plots: Optional[List[str]] = None,
    deriv_root: Optional[Path] = None,
    visualize_categories: Optional[List[str]] = None,
) -> None:
    """Visualize behavioral correlations for a single subject.
    
    Parameters
    ----------
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        Maps to specific plot names: power->power_roi_scatter, etc.
        If None, all plots are generated.
    """
    logger.info(f"Visualizing behavioral correlations for sub-{subject}...")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    ctx = _build_behavior_plot_context(subject, task, effective_deriv_root, config, logger)
    manager = BehaviorPlotManager(ctx)

    if scatter_only and temporal_only:
        raise ValueError("Cannot specify both scatter_only and temporal_only")

    if visualize_categories:
        plot_names = []
        for cat in visualize_categories:
            plot_names.extend(CATEGORY_TO_PLOTS.get(cat, []))
        logger.info(f"Running category-specific behavioral plots: {', '.join(visualize_categories)}")
        manager.run_selected(plot_names)
    elif scatter_only:
        plot_names = [
            "psychometrics",
            "power_roi_scatter",
            "complexity_scatter",
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
            "complexity_scatter",
            "aperiodic_scatter",
            "connectivity_scatter",
            "itpc_scatter",
            "temporal_topomaps",
            "pain_clusters",
            "dose_response",
            "top_predictors",
            "trial_table_overview",
            "confounds_audit",
            "regression_summary",
            "temperature_models",
            "stability_groupwise",
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
    visualize_categories: Optional[List[str]] = None,
    plots: Optional[List[str]] = None,
) -> None:
    """Batch process behavioral visualizations for multiple subjects.
    
    Parameters
    ----------
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        Maps to specific plot names. If None, all plots are generated.
    """
    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        from eeg_pipeline.utils.config.loader import load_config

        config = load_config()

    setup_matplotlib(config)

    task = task or config.get("project.task", "thermalactive")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    if scatter_only and temporal_only:
        raise ValueError("Cannot specify both scatter_only and temporal_only")

    if visualize_categories:
        mode_str = f"categories: {', '.join(visualize_categories)}"
    elif scatter_only:
        mode_str = "scatter-only"
    elif temporal_only:
        mode_str = "temporal-only"
    else:
        mode_str = "full"
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
            plots=plots,
            deriv_root=effective_deriv_root,
            visualize_categories=visualize_categories,
        )

    logger.info("Behavioral visualization complete")


__all__ = [
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
]
