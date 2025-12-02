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
    plot_dynamics_roi_scatter,
    plot_aperiodic_roi_scatter,
    plot_connectivity_roi_scatter,
    plot_itpc_roi_scatter,
)
from .collect_significant import collect_significant_plots
from .temporal import (
    plot_temporal_correlation_topomaps_by_pain,
    plot_pain_nonpain_clusters,
)
from .dose_response import visualize_dose_response
from .diagnostics import plot_comprehensive_diagnostics
from .mediation_plots import (
    plot_mediation_path_diagram,
    plot_indirect_effect_distribution,
    plot_mediation_summary_table,
)
from .moderation_plots import plot_simple_slopes, plot_johnson_neyman


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
    
    all_results = []
    
    if scatter_only:
        logger.info("Plotting scatter plots only (all feature types)...")
        rating_stats, temp_stats = load_behavior_stats_files(stats_dir, logger)
        
        plot_power_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            partial_covars=None, do_temp=True, bootstrap_ci=0,
            rng=None, rating_stats=rating_stats, temp_stats=temp_stats,
            plots_dir=plots_dir, config=config
        )
        
        logger.info("Plotting dynamics scatter...")
        dynamics_results = plot_dynamics_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(dynamics_results)
        
        logger.info("Plotting aperiodic scatter...")
        aperiodic_results = plot_aperiodic_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(aperiodic_results)
        
        logger.info("Plotting connectivity scatter...")
        conn_results = plot_connectivity_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(conn_results)
        
        logger.info("Plotting ITPC scatter...")
        itpc_results = plot_itpc_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(itpc_results)
        
        plot_psychometrics(subject, config.deriv_root, task, config)
        
        logger.info("Collecting significant plots...")
        collect_significant_plots(subject, config.deriv_root, all_results, config=config)
        
        logger.info(f"Scatter visualizations saved to {plots_dir}")
        return
    
    if temporal_only:
        logger.info("Plotting temporal topomaps only...")
        plot_temporal_correlation_topomaps_by_pain(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        )
        logger.info(f"Temporal topomap visualizations saved to {plots_dir}")
        return
    
    plots_to_run = plots if plots is not None else [
        "psychometrics", "power_roi_scatter", "dynamics_scatter", "aperiodic_scatter",
        "connectivity_scatter", "itpc_scatter", "temporal_topomaps_pain", "pain_clusters",
        "dose_response"
    ]
    
    if "psychometrics" in plots_to_run:
        logger.info("Plotting psychometrics...")
        plot_psychometrics(subject, config.deriv_root, task, config)
    
    if "power_roi_scatter" in plots_to_run:
        logger.info("Plotting power ROI scatter plots...")
        rating_stats, temp_stats = load_behavior_stats_files(stats_dir, logger)
        plot_power_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            partial_covars=None, do_temp=True, bootstrap_ci=0,
            rng=None, rating_stats=rating_stats, temp_stats=temp_stats,
            plots_dir=plots_dir, config=config
        )
    
    if "dynamics_scatter" in plots_to_run:
        logger.info("Plotting dynamics ROI scatter plots...")
        dynamics_results = plot_dynamics_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(dynamics_results)
    
    if "aperiodic_scatter" in plots_to_run:
        logger.info("Plotting aperiodic ROI scatter plots...")
        aperiodic_results = plot_aperiodic_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(aperiodic_results)
    
    if "connectivity_scatter" in plots_to_run:
        logger.info("Plotting connectivity ROI scatter plots...")
        conn_results = plot_connectivity_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(conn_results)
    
    if "itpc_scatter" in plots_to_run:
        logger.info("Plotting ITPC ROI scatter plots...")
        itpc_results = plot_itpc_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            do_temp=True, plots_dir=plots_dir, config=config
        )
        all_results.append(itpc_results)
    
    if "temporal_topomaps_pain" in plots_to_run:
        logger.info("Plotting temporal correlation topomaps by pain...")
        plot_temporal_correlation_topomaps_by_pain(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        )
    
    if "pain_clusters" in plots_to_run:
        logger.info("Plotting pain vs. non-pain clusters...")
        plot_pain_nonpain_clusters(
            subject=subject, stats_dir=stats_dir, plots_dir=plots_dir,
            config=config, logger=logger,
        )
    
    if "dose_response" in plots_to_run:
        logger.info("Plotting dose-response relationships...")
        visualize_dose_response(
            subject, config.deriv_root, task, config, logger
        )
    
    if all_results:
        logger.info("Collecting significant plots...")
        collect_significant_plots(subject, config.deriv_root, all_results, config=config)
    
    logger.info(f"Behavioral visualizations saved to {plots_dir}")


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
    """Batch process behavioral visualizations for multiple subjects.
    
    High-level entry point for creating behavioral visualizations for a list of subjects.
    
    Args:
        subjects: List of subject IDs (without 'sub-' prefix)
        task: Optional task name (defaults to config)
        deriv_root: Optional derivatives root path
        config: Optional configuration object (loads from settings if None)
        logger: Optional logger instance
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
    
    logger.info("Behavioral visualization complete")


__all__ = [
    "visualize_subject_behavior",
    "visualize_behavior_for_subjects",
]
