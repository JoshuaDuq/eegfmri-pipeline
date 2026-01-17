"""Behavior visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for behavioral visualizations.
Plot primitives live in `eeg_pipeline.plotting.behavioral.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.infra.paths import (
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    resolve_deriv_root,
)
from eeg_pipeline.plotting.behavioral.registry import (
    BehaviorPlotContext,
    BehaviorPlotManager,
)
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.collections import collect_significant_plots
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.utils.analysis.stats.correlation import (
    format_correlation_method_label,
    normalize_correlation_method,
)
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.behavior import load_behavior_stats_files

# Import plotters for side-effects: registers plot functions into BehaviorPlotRegistry.
# This import must not call back into this module at import time.
from eeg_pipeline.plotting.behavioral import registrations as _behavior_plotters  # noqa: F401


def _resolve_correlation_method(config) -> str:
    """Resolve correlation method from config.
    
    Returns
    -------
    str
        Normalized method name.
    """
    raw_method = get_config_value(
        config, "behavior_analysis.statistics.correlation_method", "spearman"
    )
    return normalize_correlation_method(raw_method, default="spearman")


def _resolve_robust_method(config) -> str | None:
    """Resolve robust correlation method from config."""
    robust_method = get_config_value(config, "behavior_analysis.robust_correlation", None)
    if robust_method is not None:
        robust_method = str(robust_method).strip().lower() or None
    return robust_method


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

    method = _resolve_correlation_method(config)
    robust_method = _resolve_robust_method(config)
    method_label = format_correlation_method_label(method, robust_method)
    rating_stats, temp_stats = load_behavior_stats_files(
        stats_dir, logger, method_label=method_label
    )

    return BehaviorPlotContext(
        subject=subject,
        task=task,
        config=config,
        logger=logger,
        deriv_root=deriv_root,
        plots_dir=plots_dir,
        stats_dir=stats_dir,
        use_spearman=method == "spearman",
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
    "temperature_models": ["temperature_models"],
    "stability": ["stability_groupwise"],
}


def _select_plot_names(
    visualize_categories: list[str] | None,
    plots: list[str] | None,
) -> list[str] | None:
    """Select plot names based on provided options.
    
    Returns
    -------
    list[str] | None
        List of plot names to run, or None to run all plots.
    """
    if visualize_categories:
        plot_names = []
        for category in visualize_categories:
            plot_names.extend(CATEGORY_TO_PLOTS.get(category, []))
        return plot_names
    
    return plots


def _run_plots(
    manager: BehaviorPlotManager,
    plot_names: list[str] | None,
    logger: logging.Logger,
) -> None:
    """Execute selected plots via the manager."""
    if plot_names is not None:
        logger.info("Running selected behavioral plots via registry...")
        manager.run_selected(plot_names)
    else:
        logger.info("Running all behavioral plots via registry...")
        manager.run_all()


def visualize_subject_behavior(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    plots: list[str] | None = None,
    deriv_root: Path | None = None,
    visualize_categories: list[str] | None = None,
) -> None:
    """Visualize behavioral correlations for a single subject.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    task : str
        Task name.
    config
        Configuration object.
    logger : logging.Logger
        Logger instance.
    plots : list of str, optional
        Specific plot names to generate. If None, uses categories or all plots.
    deriv_root : Path, optional
        Derived data root directory. If None, resolved from config.
    visualize_categories : list of str, optional
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        Maps to specific plot names: power->power_roi_scatter, etc.
        If None, all plots are generated.
    """
    logger.info(f"Visualizing behavioral correlations for sub-{subject}...")

    deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    ctx = _build_behavior_plot_context(subject, task, deriv_root, config, logger)
    manager = BehaviorPlotManager(ctx)

    plot_names = _select_plot_names(visualize_categories, plots)
    _run_plots(manager, plot_names, logger)

    if ctx.all_results:
        logger.info("Collecting significant plots...")
        collect_significant_plots(subject, deriv_root, ctx.all_results, config=config)

    logger.info(f"Behavioral visualizations saved to {ctx.plots_dir}")


def visualize_behavior_for_subjects(
    subjects: list[str],
    task: str | None = None,
    deriv_root: Path | None = None,
    config=None,
    logger: logging.Logger | None = None,
    visualize_categories: list[str] | None = None,
    plots: list[str] | None = None,
) -> None:
    """Batch process behavioral visualizations for multiple subjects.
    
    Parameters
    ----------
    subjects : list of str
        List of subject identifiers to process.
    task : str, optional
        Task name. If None, resolved from config.
    deriv_root : Path, optional
        Derived data root directory. If None, resolved from config.
    config
        Configuration object. If None, loaded from default location.
    logger : logging.Logger, optional
        Logger instance. If None, created for this module.
    visualize_categories : list of str, optional
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        Maps to specific plot names. If None, all plots are generated.
    plots : list of str, optional
        Specific plot names to generate. If None, uses categories or all plots.
    """
    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        from eeg_pipeline.utils.config.loader import load_config

        config = load_config()

    setup_matplotlib(config)

    task = task or config.get("project.task", "thermalactive")
    deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    mode_str = (
        f"categories: {', '.join(visualize_categories)}"
        if visualize_categories
        else "full"
    )
    logger.info(
        f"Starting behavioral visualization ({mode_str}): "
        f"{len(subjects)} subject(s), task={task}"
    )

    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        visualize_subject_behavior(
            subject,
            task,
            config,
            logger,
            plots=plots,
            deriv_root=deriv_root,
            visualize_categories=visualize_categories,
        )

    logger.info("Behavioral visualization complete")


__all__ = [
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
]
