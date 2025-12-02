"""
Behavioral visualization orchestration functions.

High-level entry points for creating behavioral correlation visualizations
at both subject and group levels. Coordinates calls to specialized plotting modules.

This module provides:
- Subject-level visualization (`visualize_subject_behavior`)
- Group-level visualization (`visualize_group_behavior`)
- Batch processing (`visualize_behavior_for_subjects`)
- Comprehensive dashboard generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import pandas as pd

from ...utils.data.loading import load_behavior_stats_files
from ...utils.io.general import (
    deriv_plots_path, 
    deriv_stats_path, 
    ensure_dir, 
    setup_matplotlib, 
    get_logger,
    read_tsv,
)
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
)
from .temporal_group import (
    plot_group_temporal_topomaps,
)
from .effect_sizes import (
    plot_correlation_forest,
    plot_effect_size_comparison,
    plot_effect_size_heatmap,
    plot_condition_effect_sizes,
    plot_temperature_mediation,
)
from .mediation import (
    plot_mediation_summary,
    plot_mediation_paths_grid,
)
from .mixed_effects import (
    plot_icc_bar_chart,
    plot_variance_decomposition,
    plot_mixed_effects_forest,
)
from .robust import (
    plot_bootstrap_ci_comparison,
)
from .distributions import (
    plot_feature_distributions,
    plot_behavioral_summary,
    plot_feature_by_condition,
    plot_feature_correlation_matrix,
    plot_top_predictors_summary,
)
from .summary import (
    plot_analysis_dashboard,
    plot_group_summary_dashboard,
    plot_quality_overview,
)
from .feature_behavior_plots import visualize_feature_behavior_correlations


###################################################################
# Plot Definition Registry
###################################################################

# Available plot types with their definitions
AVAILABLE_PLOTS = {
    # Comprehensive feature-behavior visualization
    "feature_behavior": {
        "category": "comprehensive",
        "description": "All feature-behavior correlations with organized output",
        "quick": False,
    },
    # Core scatter plots
    "psychometrics": {
        "category": "scatter",
        "description": "Temperature vs rating scatter with psychometric curve",
        "quick": True,
    },
    "power_roi_scatter": {
        "category": "scatter", 
        "description": "Power-behavior scatter plots by ROI and band",
        "quick": True,
    },
    # Temporal/spatial plots
    "temporal_topomaps_temp": {
        "category": "topomap",
        "description": "Temporal correlation topomaps by temperature",
        "quick": False,
    },
    "temporal_topomaps_pain": {
        "category": "topomap",
        "description": "Temporal correlation topomaps by pain condition",
        "quick": False,
    },
    "pac_behavior": {
        "category": "heatmap",
        "description": "Phase-amplitude coupling behavior correlations",
        "quick": False,
    },
    "pain_clusters": {
        "category": "topomap",
        "description": "Pain vs non-pain cluster visualization",
        "quick": False,
    },
    # Effect size plots
    "effect_size_forest": {
        "category": "forest",
        "description": "Forest plot of effect sizes with CIs",
        "quick": True,
    },
    # Advanced analyses
    "mediation": {
        "category": "diagram",
        "description": "Mediation analysis path diagrams",
        "quick": False,
    },
    "mixed_effects": {
        "category": "forest",
        "description": "Mixed-effects model results",
        "quick": False,
    },
    "bootstrap_ci": {
        "category": "forest",
        "description": "Bootstrap confidence interval comparison",
        "quick": False,
    },
    # New distribution plots
    "feature_distributions": {
        "category": "distribution",
        "description": "Feature distribution violin plots",
        "quick": True,
    },
    "behavioral_summary": {
        "category": "summary",
        "description": "Behavioral variables summary",
        "quick": True,
    },
    "feature_correlations": {
        "category": "heatmap",
        "description": "Feature correlation matrix",
        "quick": True,
    },
    "condition_comparison": {
        "category": "comparison",
        "description": "Features by condition comparison",
        "quick": False,
    },
    # Quality
    "quality_overview": {
        "category": "summary",
        "description": "Feature quality overview",
        "quick": True,
    },
    # Condition effect sizes
    "condition_effect_sizes": {
        "category": "forest",
        "description": "Pain vs non-pain effect sizes (Hedges' g)",
        "quick": True,
    },
    # Temperature mediation
    "temperature_mediation": {
        "category": "forest",
        "description": "Temperature mediation of feature-rating correlations",
        "quick": True,
    },
}


###################################################################
# Subject-level Behavioral Visualization
###################################################################


def _load_data_for_visualization(
    subject: str,
    stats_dir: Path,
    config: Any,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Load all data needed for visualization once."""
    data = {
        "rating_stats": None,
        "temp_stats": None,
        "mediation_results": None,
        "mixed_effects_results": None,
        "power_df": None,
        "precomputed_df": None,
        "quality_report": None,
    }
    
    # Load stats files
    data["rating_stats"], data["temp_stats"] = load_behavior_stats_files(stats_dir, logger)
    
    # Load additional stats if available
    mediation_path = stats_dir / "mediation_results.tsv"
    if mediation_path.exists():
        try:
            data["mediation_results"] = pd.read_csv(mediation_path, sep="\t")
        except Exception:
            pass
    
    me_path = stats_dir / "mixed_effects_results.tsv"
    if me_path.exists():
        try:
            data["mixed_effects_results"] = pd.read_csv(me_path, sep="\t")
        except Exception:
            pass
    
    # Load precomputed features if available
    from ...utils.io.general import deriv_features_path
    features_dir = deriv_features_path(config.deriv_root, subject)
    precomputed_path = features_dir / "features_precomputed.tsv"
    if precomputed_path.exists():
        try:
            data["precomputed_df"] = read_tsv(precomputed_path)
        except Exception:
            pass
    
    # Load quality report if available
    quality_path = stats_dir / "feature_quality_report.json"
    if quality_path.exists():
        try:
            import json
            with open(quality_path) as f:
                data["quality_report"] = json.load(f)
        except Exception:
            pass
    
    return data


def visualize_subject_behavior(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    scatter_only: bool = False,
    temporal_only: bool = False,
    plots: Optional[List[str]] = None,
    include_dashboard: bool = True,
) -> None:
    """Visualize behavioral correlations for a single subject.
    
    Creates comprehensive behavioral correlation visualizations including:
    - Psychometric plots
    - ROI scatter plots
    - Temporal correlation topomaps
    - PAC-behavior correlations
    - Effect size visualizations
    - Feature distribution summaries
    - Analysis dashboard
    
    Args:
        subject: Subject ID (without 'sub-' prefix)
        task: Task name
        config: Configuration object
        logger: Logger instance
        scatter_only: If True, only create scatter plots
        temporal_only: If True, only create temporal topomap plots
        plots: Specific plots to generate (None = all)
        include_dashboard: Whether to include summary dashboard
    """
    logger.info(f"Visualizing behavioral correlations for sub-{subject}...")
    
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="behavior")
    ensure_dir(plots_dir)
    stats_dir = deriv_stats_path(config.deriv_root, subject)
    
    stats_config = config.get("behavior_analysis.statistics", {})
    use_spearman = (stats_config.get("correlation_method") == "spearman")
    
    # Load all data once (avoid redundant loading)
    data = _load_data_for_visualization(subject, stats_dir, config, logger)
    rating_stats = data["rating_stats"]
    temp_stats = data["temp_stats"]
    
    # Define plot executors
    def _run_psychometrics():
        plot_psychometrics(subject, config.deriv_root, task, config)
    
    def _run_power_roi_scatter():
        plot_power_roi_scatter(
            subject, config.deriv_root, task=task, use_spearman=use_spearman,
            partial_covars=None, do_temp=True, bootstrap_ci=0,
            rng=None, rating_stats=rating_stats, temp_stats=temp_stats,
            plots_dir=plots_dir, config=config
        )
    
    def _run_temporal_topomaps_temp():
        if temp_stats is None or (hasattr(temp_stats, "empty") and temp_stats.empty):
            logger.info("No temperature stats available; skipping temporal topomaps by temperature.")
            return
        plot_temporal_correlation_topomaps_by_temperature(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        )
    
    def _run_temporal_topomaps_pain():
        plot_temporal_correlation_topomaps_by_pain(
            subject, task, plots_dir, stats_dir, config, logger, use_spearman=use_spearman
        )
    
    def _run_pac_behavior():
        plot_pac_behavior_correlations(
            subject=subject, stats_dir=stats_dir, plots_dir=plots_dir,
            config=config, logger=logger,
        )
    
    def _run_pain_clusters():
        plot_pain_nonpain_clusters(
            subject=subject, stats_dir=stats_dir, plots_dir=plots_dir,
            config=config, logger=logger,
        )
    
    def _run_feature_distributions():
        if data["precomputed_df"] is not None:
            dist_dir = plots_dir / "distributions"
            ensure_dir(dist_dir)
            plot_feature_distributions(
                data["precomputed_df"],
                dist_dir / f"sub-{subject}_feature_distributions",
                max_features=50, config=config,
            )
    
    def _run_feature_correlations():
        if data["precomputed_df"] is not None:
            heatmap_dir = plots_dir / "heatmaps"
            ensure_dir(heatmap_dir)
            plot_feature_correlation_matrix(
                data["precomputed_df"],
                heatmap_dir / f"sub-{subject}_feature_correlation_matrix",
                max_features=50, config=config,
            )
    
    def _run_quality_overview():
        if data["quality_report"]:
            summary_dir = plots_dir / "summary"
            ensure_dir(summary_dir)
            plot_quality_overview(
                data["quality_report"],
                summary_dir / f"sub-{subject}_quality_overview",
                subject=subject, config=config,
            )
    
    def _run_feature_behavior():
        """Run comprehensive feature-behavior visualization with organized output.
        This includes dashboards, forest plots, heatmaps - all in organized subdirs.
        """
        visualize_feature_behavior_correlations(
            subject=subject,
            deriv_root=config.deriv_root,
            config=config,
            logger=logger,
            targets=None,
            temperature=None,
            pain_condition=None,
        )
    
    def _run_condition_effect_sizes():
        effect_file = stats_dir / "effect_sizes_all_pain_vs_nonpain.tsv"
        if effect_file.exists():
            effect_df = read_tsv(effect_file)
            if effect_df is not None and not effect_df.empty:
                forest_dir = plots_dir / "forest"
                ensure_dir(forest_dir)
                plot_condition_effect_sizes(
                    effect_df,
                    forest_dir / f"sub-{subject}_condition_effect_sizes",
                    config=config,
                )
    
    def _run_temperature_mediation():
        for method in ["spearman", "pearson"]:
            partial_file = stats_dir / f"partial_corr_controlling_temp_{method}.tsv"
            if partial_file.exists():
                partial_df = read_tsv(partial_file)
                if partial_df is not None and not partial_df.empty:
                    forest_dir = plots_dir / "forest"
                    ensure_dir(forest_dir)
                    plot_temperature_mediation(
                        partial_df,
                        forest_dir / f"sub-{subject}_temperature_mediation_{method}",
                        config=config,
                    )
                    break  # Only need one method
    
    all_plots = {
        # Comprehensive visualization (includes dashboard, forest, heatmaps)
        "feature_behavior": _run_feature_behavior,
        # Core scatter plots
        "psychometrics": _run_psychometrics,
        "power_roi_scatter": _run_power_roi_scatter,
        # Temporal/spatial plots  
        "temporal_topomaps_temp": _run_temporal_topomaps_temp,
        "temporal_topomaps_pain": _run_temporal_topomaps_pain,
        "pac_behavior": _run_pac_behavior,
        "pain_clusters": _run_pain_clusters,
        # Effect sizes and statistics
        "effect_size_forest": lambda: _plot_effect_sizes(rating_stats, plots_dir, logger),
        "condition_effect_sizes": _run_condition_effect_sizes,
        "temperature_mediation": _run_temperature_mediation,
        "mediation": lambda: _plot_mediation_results(stats_dir, plots_dir, logger),
        "mixed_effects": lambda: _plot_mixed_effects(stats_dir, plots_dir, logger),
        "bootstrap_ci": lambda: _plot_bootstrap_cis(rating_stats, plots_dir, logger),
        # Distributions 
        "feature_distributions": _run_feature_distributions,
        "feature_correlations": _run_feature_correlations,
        "quality_overview": _run_quality_overview,
    }
    
    # Handle quick modes
    if scatter_only:
        logger.info("Plotting scatter plots only...")
        _run_power_roi_scatter()
        _run_psychometrics()
        logger.info(f"Scatter visualizations saved to {plots_dir}")
        return
    
    if temporal_only:
        logger.info("Plotting temporal topomaps only...")
        _run_temporal_topomaps_temp()
        _run_temporal_topomaps_pain()
        logger.info(f"Temporal topomap visualizations saved to {plots_dir}")
        return
    
    # Determine which plots to run
    plots_to_run = plots if plots is not None else list(all_plots.keys())
    
    # Always include feature_behavior (which includes dashboard) if requested
    if include_dashboard and "feature_behavior" not in plots_to_run:
        plots_to_run.append("feature_behavior")
    
    # Execute each plot with error handling
    n_success = 0
    n_failed = 0
    
    for plot_name in plots_to_run:
        if plot_name not in all_plots:
            logger.warning(f"Unknown plot type: {plot_name}")
            continue
        
        logger.info(f"Plotting {plot_name}...")
        try:
            all_plots[plot_name]()
            n_success += 1
        except Exception as exc:
            logger.warning(f"Failed to create {plot_name}: {exc}")
            n_failed += 1
    
    logger.info(f"Behavioral visualizations complete: {n_success} succeeded, {n_failed} failed")
    logger.info(f"Output directory: {plots_dir}")


###################################################################
# Advanced Visualization Helpers
###################################################################


def _plot_effect_sizes(stats_df, plots_dir, logger):
    """Helper to plot effect size forest."""
    import pandas as pd
    if stats_df is None or (isinstance(stats_df, pd.DataFrame) and stats_df.empty):
        logger.warning("No stats data for effect size forest")
        return
    
    try:
        forest_dir = plots_dir / "forest"
        ensure_dir(forest_dir)
        save_path = forest_dir / "effect_size_forest.png"
        plot_correlation_forest(stats_df, save_path, title="Effect Sizes with 95% CI")
        logger.info(f"Saved effect size forest to {save_path}")
    except Exception as e:
        logger.warning(f"Effect size forest failed: {e}")


def _plot_mediation_results(stats_dir, plots_dir, logger):
    """Helper to plot mediation results if available."""
    import pandas as pd
    mediation_path = stats_dir / "mediation_results.tsv"
    if not mediation_path.exists():
        return
    
    try:
        results_df = pd.read_csv(mediation_path, sep="\t")
        if results_df.empty:
            return
        
        summary_dir = plots_dir / "summary"
        ensure_dir(summary_dir)
        save_path = summary_dir / "mediation_summary.png"
        plot_mediation_summary(results_df, save_path)
        logger.info(f"Saved mediation summary to {save_path}")
        
        grid_path = summary_dir / "mediation_paths_grid.png"
        plot_mediation_paths_grid(results_df, grid_path)
        logger.info(f"Saved mediation paths grid to {grid_path}")
    except Exception as e:
        logger.warning(f"Mediation plots failed: {e}")


def _plot_mixed_effects(stats_dir, plots_dir, logger):
    """Helper to plot mixed-effects results if available."""
    import pandas as pd
    me_path = stats_dir / "mixed_effects_results.tsv"
    if not me_path.exists():
        return
    
    try:
        results_df = pd.read_csv(me_path, sep="\t")
        if results_df.empty:
            return
        
        forest_dir = plots_dir / "forest"
        ensure_dir(forest_dir)
        summary_dir = plots_dir / "summary"
        ensure_dir(summary_dir)
        
        # Forest plot
        save_path = forest_dir / "mixed_effects_forest.png"
        plot_mixed_effects_forest(results_df, save_path)
        logger.info(f"Saved mixed-effects forest to {save_path}")
        
        # Variance decomposition
        if "random_variance" in results_df.columns:
            var_path = summary_dir / "variance_decomposition.png"
            plot_variance_decomposition(results_df, var_path)
            logger.info(f"Saved variance decomposition to {var_path}")
        
        # ICC bar chart
        if "icc" in results_df.columns:
            icc_path = summary_dir / "icc_bar_chart.png"
            plot_icc_bar_chart(results_df, icc_path, icc_col="icc", feature_col="feature")
            logger.info(f"Saved ICC bar chart to {icc_path}")
    except Exception as e:
        logger.warning(f"Mixed-effects plots failed: {e}")


def _plot_bootstrap_cis(stats_df, plots_dir, logger):
    """Helper to plot bootstrap CI comparison."""
    import pandas as pd
    if stats_df is None or (isinstance(stats_df, pd.DataFrame) and stats_df.empty):
        return
    
    if "ci_low" not in stats_df.columns or "ci_high" not in stats_df.columns:
        return
    
    try:
        forest_dir = plots_dir / "forest"
        ensure_dir(forest_dir)
        save_path = forest_dir / "bootstrap_ci_comparison.png"
        plot_bootstrap_ci_comparison(stats_df, save_path)
        logger.info(f"Saved bootstrap CI comparison to {save_path}")
    except Exception as e:
        logger.warning(f"Bootstrap CI plot failed: {e}")


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
    
    stats_config = config.get("behavior_analysis.statistics", {})
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
    # Orchestration functions
    "visualize_subject_behavior",
    "visualize_group_behavior",
    "visualize_behavior_for_subjects",
    # Plot registry
    "AVAILABLE_PLOTS",
]
