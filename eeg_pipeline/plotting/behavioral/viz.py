"""
Behavioral visualization orchestration functions.

High-level entry points for creating behavioral correlation visualizations
at both subject and group levels. Coordinates calls to specialized plotting modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging

from ...utils.data.loading import load_behavior_stats_files
from ...utils.io.general import deriv_plots_path, deriv_stats_path, ensure_dir, setup_matplotlib, get_logger
from .scatter import (
    plot_psychometrics,
    plot_power_roi_scatter,
)
from .group import (
    plot_group_power_roi_scatter,
)
from .temporal import (
    plot_temporal_correlation_topomaps_by_temperature,
    plot_temporal_correlation_topomaps_by_pain,
    plot_pain_nonpain_clusters,
    plot_pac_behavior_correlations,
    plot_behavior_reliability,
)
from .temporal_group import (
    plot_group_temporal_topomaps,
)


###################################################################
# Subject-level Behavioral Visualization
###################################################################


def visualize_subject_behavior(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    scatter_only: bool = False,
    temporal_only: bool = False,
    plots: Optional[List[str]] = None,
) -> None:
    """Visualize behavioral correlations for a single subject.
    
    Creates comprehensive behavioral correlation visualizations including:
    - Psychometric plots
    - ROI scatter plots
    - Temporal correlation topomaps
    - PAC-behavior correlations
    - ITPC-rating scatter grids
    - Reliability diagnostics
    - Pain/non-pain clusters
    
    Args:
        subject: Subject ID (without 'sub-' prefix)
        task: Task name
        config: Configuration object
        logger: Logger instance
        scatter_only: If True, only create scatter plots
        temporal_only: If True, only create temporal topomap plots
    """
    logger.info(f"Visualizing behavioral correlations for sub-{subject}...")
    
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="behavior")
    ensure_dir(plots_dir)
    stats_dir = deriv_stats_path(config.deriv_root, subject)
    
    stats_config = config.get("behavior_analysis", {}).get("statistics", {})
    use_spearman = (stats_config.get("correlation_method") == "spearman")
    
    rating_stats, temp_stats = load_behavior_stats_files(stats_dir, logger)
    
    all_plots = {
        "psychometrics": lambda: plot_psychometrics(subject, config.deriv_root, task, config),
        "power_roi_scatter": lambda: plot_power_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            partial_covars=None, do_temp=True, bootstrap_ci=0,
            rng=None, rating_stats=rating_stats, temp_stats=temp_stats,
            plots_dir=plots_dir, config=config
        ),
        "temporal_topomaps_temp": lambda: plot_temporal_correlation_topomaps_by_temperature(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        ),
        "temporal_topomaps_pain": lambda: plot_temporal_correlation_topomaps_by_pain(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        ),
        "pac_behavior": lambda: plot_pac_behavior_correlations(
            subject=subject,
            stats_dir=stats_dir,
            plots_dir=plots_dir,
            config=config,
            logger=logger,
        ),
        "pain_clusters": lambda: plot_pain_nonpain_clusters(
            subject=subject,
            stats_dir=stats_dir,
            plots_dir=plots_dir,
            config=config,
            logger=logger,
        ),
        "reliability": lambda: plot_behavior_reliability(
            subject=subject,
            stats_dir=stats_dir,
            plots_dir=plots_dir,
            config=config,
            logger=logger,
        ),
    }
    
    if scatter_only:
        logger.info("Plotting scatter plots only...")
        rating_stats, temp_stats = load_behavior_stats_files(stats_dir, logger)
        plot_power_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            partial_covars=None, do_temp=True, bootstrap_ci=0,
            rng=None, rating_stats=rating_stats, temp_stats=temp_stats,
            plots_dir=plots_dir, config=config
        )
        plot_psychometrics(subject, config.deriv_root, task, config)
        logger.info(f"Scatter visualizations saved to {plots_dir}")
        return
    
    if temporal_only:
        logger.info("Plotting temporal topomaps only...")
        plot_temporal_correlation_topomaps_by_temperature(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        )
        plot_temporal_correlation_topomaps_by_pain(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        )
        logger.info(f"Temporal topomap visualizations saved to {plots_dir}")
        return
    
    plots_to_run = plots if plots is not None else list(all_plots.keys())
    
    if "psychometrics" in plots_to_run:
        logger.info("Plotting psychometrics...")
        all_plots["psychometrics"]()
    
    if "power_roi_scatter" in plots_to_run:
        logger.info("Plotting ROI scatter plots...")
        rating_stats, temp_stats = load_behavior_stats_files(stats_dir, logger)
        plot_power_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            partial_covars=None, do_temp=True, bootstrap_ci=0,
            rng=None, rating_stats=rating_stats, temp_stats=temp_stats,
            plots_dir=plots_dir, config=config
        )
    
    if "temporal_topomaps_temp" in plots_to_run:
        logger.info("Plotting temporal correlation topomaps by temperature...")
        all_plots["temporal_topomaps_temp"]()
    
    if "temporal_topomaps_pain" in plots_to_run:
        logger.info("Plotting temporal correlation topomaps by pain...")
        all_plots["temporal_topomaps_pain"]()
    
    if "pac_behavior" in plots_to_run:
        logger.info("Plotting PAC-behavior correlations...")
        all_plots["pac_behavior"]()
    
    if "pain_clusters" in plots_to_run:
        logger.info("Plotting pain vs. non-pain clusters...")
        all_plots["pain_clusters"]()
    
    if "reliability" in plots_to_run:
        logger.info("Plotting reliability diagnostics...")
        all_plots["reliability"]()
    
    logger.info(f"Behavioral visualizations saved to {plots_dir}")


###################################################################
# Group-level Behavioral Visualization
###################################################################


def visualize_group_behavior(
    subjects: List[str],
    task: str,
    config,
    logger: logging.Logger,
    scatter_only: bool = False,
    temporal_only: bool = False,
) -> None:
    """Visualize behavioral correlations for a group of subjects.
    
    Creates group-level behavioral correlation visualizations including:
    - Group ROI scatter plots
    - Group temporal correlation topomaps
    
    Args:
        subjects: List of subject IDs (without 'sub-' prefix)
        task: Task name
        config: Configuration object
        logger: Logger instance
        scatter_only: If True, only create scatter plots
        temporal_only: If True, only create temporal topomap plots
    """
    if len(subjects) < 2:
        logger.warning(f"Group visualization requires at least 2 subjects, got {len(subjects)}")
        return
    
    logger.info(f"Visualizing group behavioral correlations for {len(subjects)} subjects...")
    
    group_dir = config.deriv_root / "group" / "eeg" / "plots" / "behavior"
    ensure_dir(group_dir)
    
    stats_config = config.get("behavior_analysis", {}).get("statistics", {})
    use_spearman = (stats_config.get("correlation_method") == "spearman")
    
    if scatter_only:
        logger.info("Plotting group scatter plots only...")
        plot_group_power_roi_scatter(
            subjects=subjects,
            deriv_root=config.deriv_root,
            task=task,
            use_spearman=use_spearman,
            partial_covars=None,
            do_temp=True,
            bootstrap_ci=0,
            rng=None,
            plots_dir=group_dir,
            config=config,
            logger=logger,
        )
        logger.info(f"Group scatter visualizations saved to {group_dir}")
        return
    
    if temporal_only:
        logger.info("Plotting group temporal topomaps only...")
        plot_group_temporal_topomaps(
            subjects=subjects,
            deriv_root=config.deriv_root,
            plots_dir=group_dir,
            config=config,
            logger=logger,
            use_spearman=use_spearman,
            condition="pain",
        )
        plot_group_temporal_topomaps(
            subjects=subjects,
            deriv_root=config.deriv_root,
            plots_dir=group_dir,
            config=config,
            logger=logger,
            use_spearman=use_spearman,
            condition="temperature",
        )
        logger.info(f"Group temporal topomap visualizations saved to {group_dir}")
        return
    
    logger.info("Plotting group ROI scatter plots...")
    plot_group_power_roi_scatter(
        subjects=subjects,
        deriv_root=config.deriv_root,
        task=task,
        use_spearman=use_spearman,
        partial_covars=None,
        do_temp=True,
        bootstrap_ci=0,
        rng=None,
        plots_dir=group_dir,
        config=config,
        logger=logger,
    )
    
    logger.info("Plotting group temporal topomaps by pain...")
    plot_group_temporal_topomaps(
        subjects=subjects,
        deriv_root=config.deriv_root,
        plots_dir=group_dir,
        config=config,
        logger=logger,
        use_spearman=use_spearman,
        condition="pain",
    )
    
    logger.info("Plotting group temporal topomaps by temperature...")
    plot_group_temporal_topomaps(
        subjects=subjects,
        deriv_root=config.deriv_root,
        plots_dir=group_dir,
        config=config,
        logger=logger,
        use_spearman=use_spearman,
        condition="temperature",
    )
    
    logger.info(f"Group behavioral visualizations saved to {group_dir}")


###################################################################
# Batch Processing
###################################################################


def visualize_behavior_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    logger: Optional[logging.Logger] = None,
    group: bool = False,
    scatter_only: bool = False,
    temporal_only: bool = False,
) -> None:
    """Batch process behavioral visualizations for multiple subjects.
    
    High-level entry point for creating behavioral visualizations for a list of subjects,
    optionally including group-level analysis.
    
    Args:
        subjects: List of subject IDs (without 'sub-' prefix)
        task: Optional task name (defaults to config)
        deriv_root: Optional derivatives root path
        config: Optional configuration object (loads from settings if None)
        logger: Optional logger instance
        group: If True, create group-level visualizations
        scatter_only: If True, only create scatter plots
        temporal_only: If True, only create temporal topomap plots
    """
    if not subjects:
        raise ValueError("No subjects specified")
    
    if config is None:
        from ...utils.config.loader import load_settings
        config = load_settings()
    
    setup_matplotlib(config)
    
    task = task or config.get("project.task", "thermalactive")
    
    if logger is None:
        logger = get_logger(__name__)
    
    if scatter_only and temporal_only:
        raise ValueError("Cannot specify both scatter_only and temporal_only")
    
    mode_str = "scatter-only" if scatter_only else "temporal-only" if temporal_only else "full"
    logger.info(f"Starting behavioral visualization ({mode_str}): {len(subjects)} subject(s), task={task}")
    
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        visualize_subject_behavior(
            subject, task, config, logger,
            scatter_only=scatter_only,
            temporal_only=temporal_only
        )
    
    if group and len(subjects) >= 2:
        logger.info("Creating group visualizations...")
        visualize_group_behavior(
            subjects, task, config, logger,
            scatter_only=scatter_only,
            temporal_only=temporal_only
        )
    
    logger.info("Behavioral visualization complete")


__all__ = [
    "visualize_subject_behavior",
    "visualize_group_behavior",
    "visualize_behavior_for_subjects",
]

