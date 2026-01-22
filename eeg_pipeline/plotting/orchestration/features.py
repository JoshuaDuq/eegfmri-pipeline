"""Feature visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for descriptive feature visualizations.
Plot primitives and plot registries live in `eeg_pipeline.plotting.features.*`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.infra.paths import (
    deriv_features_path,
    deriv_plots_path,
    deriv_stats_path,
    ensure_dir,
    resolve_deriv_root,
)
from eeg_pipeline.plotting.io.figures import setup_matplotlib

# Import plotters for side-effects: registers plot functions into VisualizationRegistry.
# This import must not call back into this module at import time.
from eeg_pipeline.plotting.features import registrations as _feature_plotters  # noqa: F401
from eeg_pipeline.plotting.features.context import (
    FeaturePlotContext,
    VisualizationManager,
    VisualizationRegistry,
)


# Known feature types for manifest generation
# These must match the categories registered in VisualizationRegistry
_KNOWN_FEATURE_TYPES = [
    "aperiodic",
    "bursts",
    "complexity",
    "connectivity",
    "erds",
    "erp",
    "itpc",
    "pac",
    "power",
    "ratios",
    "spectral",
    "asymmetry",
    "temporal",
]

_PLOT_FILE_EXTENSIONS = ["png", "svg", "pdf"]


def _parse_plotter_tokens(tokens: List[str]) -> Dict[str, Set[str]]:
    """Parse plotter tokens into category-to-names mapping.
    
    Tokens should be in format "category.name" (e.g., "power.spectral").
    Invalid tokens are silently skipped.
    
    Parameters
    ----------
    tokens : list of str
        Plotter tokens in "category.name" format.
        
    Returns
    -------
    dict mapping category names to sets of plotter names.
    """
    selected_by_category: Dict[str, Set[str]] = {}
    
    for token in tokens:
        token = str(token).strip()
        if "." not in token:
            continue
        
        category, _, name = token.partition(".")
        category = category.strip()
        name = name.strip()
        
        if category and name:
            selected_by_category.setdefault(category, set()).add(name)
    
    return selected_by_category


def _get_filtered_plotters(category: str, wanted_names: Set[str]) -> List[Tuple[str, Any]]:
    """Get plotters for a category filtered by wanted names.
    
    Parameters
    ----------
    category : str
        Category name.
    wanted_names : set of str
        Names of plotters to include.
        
    Returns
    -------
    list of (name, function) tuples for matching plotters.
    """
    all_plotters = VisualizationRegistry.get_plotters(category)
    return [(name, func) for name, func in all_plotters if name in wanted_names]


def _run_visualizations(
    manager: VisualizationManager,
    visualize_categories: Optional[List[str]],
    selected_by_category: Dict[str, Set[str]],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Run visualizations based on category and plotter selections.
    
    Parameters
    ----------
    manager : VisualizationManager
        Manager instance to run visualizations.
    visualize_categories : optional list of str
        Specific categories to visualize. If None, all categories are used.
    selected_by_category : dict
        Mapping of category names to sets of wanted plotter names.
    logger : logging.Logger
        Logger instance.
        
    Returns
    -------
    dict mapping plot names to file paths.
    """
    has_plotter_filter = bool(selected_by_category)
    
    if visualize_categories is not None:
        logger.info(f"Visualizing specific categories: {', '.join(visualize_categories)}")
        categories_to_run = visualize_categories
    elif has_plotter_filter:
        categories_to_run = list(selected_by_category.keys())
    else:
        return manager.run_all()
    
    for category in categories_to_run:
        if has_plotter_filter:
            wanted_names = selected_by_category.get(category)
            if not wanted_names:
                continue
            plotters = _get_filtered_plotters(category, wanted_names)
            manager.run_category(category, plotters=plotters)
        else:
            manager.run_category(category)
    
    return manager.saved_plots


def _create_plot_context(
    subject: str,
    deriv_root: Path,
    config: Optional[Any],
    logger: logging.Logger,
    epochs_info: Optional[Any],
    aligned_events: Optional[Any],
    epochs: Optional[Any],
    tfr: Optional[Any],
    plot_name_patterns: Optional[List[str]] = None,
) -> FeaturePlotContext:
    """Create and initialize feature plot context.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    deriv_root : Path
        Root directory for derivatives.
    config : optional
        Configuration object.
    logger : logging.Logger
        Logger instance.
    epochs_info : optional
        Epochs info object.
    aligned_events : optional
        Aligned events DataFrame.
    epochs : optional
        Epochs object.
    tfr : optional
        Time-frequency representation.
        
    Returns
    -------
    Initialized FeaturePlotContext.
    """
    features_dir = deriv_features_path(deriv_root, subject)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir="features")
    stats_dir = deriv_stats_path(deriv_root, subject)
    ensure_dir(plots_dir)
    
    context = FeaturePlotContext(
        subject=subject,
        plots_dir=plots_dir,
        features_dir=features_dir,
        config=config,
        logger=logger,
        plot_name_patterns=plot_name_patterns,
        stats_dir=stats_dir,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        tfr=tfr,
    )
    context.load_data()
    
    return context


def _validate_context_has_data(context: FeaturePlotContext, subject: str, logger: logging.Logger) -> bool:
    """Validate that context has sufficient data for visualization.
    
    Parameters
    ----------
    context : FeaturePlotContext
        Context to validate.
    subject : str
        Subject identifier for logging.
    logger : logging.Logger
        Logger instance.
        
    Returns
    -------
    True if context has sufficient data, False otherwise.
    """
    has_feature_data = context.power_df is not None or context.connectivity_df is not None
    
    if not has_feature_data:
        logger.warning(f"No feature data found for subject {subject}")
        if context.epochs is None:
            logger.warning("No epochs data found either")
            return False
    
    return True


def visualize_features(
    subject: str,
    deriv_root: Path,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    epochs_info: Optional[Any] = None,
    aligned_events: Optional[Any] = None,
    epochs: Optional[Any] = None,
    tfr: Optional[Any] = None,
    visualize_categories: Optional[List[str]] = None,
    feature_plotters: Optional[List[str]] = None,
    plot_name_patterns: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Generate descriptive feature visualizations using registered plotters.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    deriv_root : Path
        Root directory for derivatives.
    config : optional
        Configuration object.
    logger : optional logging.Logger
        Logger instance. If None, creates a new logger.
    epochs_info : optional
        Epochs info object.
    aligned_events : optional
        Aligned events DataFrame.
    epochs : optional
        Epochs object.
    tfr : optional
        Time-frequency representation.
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        If None, all registered categories are visualized.
    feature_plotters : optional list of str
        Specific plotters to use in "category.name" format.
        If None, all plotters for selected categories are used.
        
    Returns
    -------
    dict mapping plot names to file paths.
    """
    if logger is None:
        logger = get_logger(__name__)
    
    context = _create_plot_context(
        subject=subject,
        deriv_root=deriv_root,
        config=config,
        logger=logger,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        tfr=tfr,
        plot_name_patterns=plot_name_patterns,
    )
    
    if not _validate_context_has_data(context, subject, logger):
        return {}
    
    manager = VisualizationManager(context)
    
    selected_by_category = _parse_plotter_tokens(feature_plotters) if feature_plotters else {}
    
    saved_plots = _run_visualizations(
        manager=manager,
        visualize_categories=visualize_categories,
        selected_by_category=selected_by_category,
        logger=logger,
    )
    
    _save_plot_manifest(
        plots_dir=context.plots_dir,
        subject=subject,
        logger=logger,
    )
    
    return saved_plots


def _detect_feature_type_from_path(path: Path) -> str:
    """Detect feature type from plot file path.
    
    Parameters
    ----------
    path : Path
        Path to plot file.
        
    Returns
    -------
    Feature type string, or "unknown" if not detected.
    """
    for feature_type in _KNOWN_FEATURE_TYPES:
        if feature_type in path.parts:
            return feature_type
    return "unknown"


def _collect_plot_files(plots_dir: Path) -> List[Path]:
    """Collect all plot files from directory.
    
    Parameters
    ----------
    plots_dir : Path
        Directory containing plot files.
        
    Returns
    -------
    Sorted list of plot file paths.
    """
    plot_files = [
        path
        for extension in _PLOT_FILE_EXTENSIONS
        for path in plots_dir.rglob(f"*.{extension}")
    ]
    return sorted(plot_files)


def _create_plot_entry(path: Path, plots_dir: Path) -> Dict[str, str]:
    """Create manifest entry for a single plot file.
    
    Parameters
    ----------
    path : Path
        Path to plot file.
    plots_dir : Path
        Base plots directory for relative path calculation.
        
    Returns
    -------
    Dictionary with plot metadata.
    """
    return {
        "name": path.stem,
        "path": str(path.relative_to(plots_dir)),
        "feature_type": _detect_feature_type_from_path(path),
        "format": path.suffix[1:],
    }


def _save_plot_manifest(
    plots_dir: Path,
    subject: str,
    logger: logging.Logger,
) -> None:
    """Save plot manifest JSON file listing all generated plots.
    
    Parameters
    ----------
    plots_dir : Path
        Directory containing plot files.
    subject : str
        Subject identifier.
    logger : logging.Logger
        Logger instance.
    """
    plot_files = _collect_plot_files(plots_dir)
    
    manifest = {
        "subject": subject,
        "generated_at": datetime.now().isoformat(),
        "plots_directory": str(plots_dir),
        "total_plots": len(plot_files),
        "plots": [_create_plot_entry(path, plots_dir) for path in plot_files],
    }
    
    manifest_path = plots_dir / f"sub-{subject}_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    
    logger.info(f"Saved plot manifest ({len(plot_files)} plots)")


def _load_config_if_needed(config: Any) -> Any:
    """Load configuration if not provided.
    
    Parameters
    ----------
    config : optional
        Configuration object. If None, loads default config.
        
    Returns
    -------
    Configuration object.
    """
    if config is not None:
        return config
    from eeg_pipeline.utils.config.loader import load_config
    return load_config()


def _resolve_task(task: Optional[str], config: Any) -> str:
    """Resolve task name from parameter or config.
    
    Parameters
    ----------
    task : optional str
        Task name. If None, uses config default.
    config : Any
        Configuration object.
        
    Returns
    -------
    Task name string.
    """
    if task is not None:
        return task
    return config.get("project.task", "thermalactive")


def _log_visualization_start(
    subjects: List[str],
    task: str,
    visualize_categories: Optional[List[str]],
    logger: logging.Logger,
) -> None:
    """Log start of visualization process.
    
    Parameters
    ----------
    subjects : list of str
        Subject identifiers.
    task : str
        Task name.
    visualize_categories : optional list of str
        Categories being visualized.
    logger : logging.Logger
        Logger instance.
    """
    category_suffix = f" ({', '.join(visualize_categories)})" if visualize_categories else ""
    logger.info(
        f"Starting feature visualization{category_suffix}: "
        f"{len(subjects)} subject(s), task={task}"
    )


def _visualize_single_subject(
    subject: str,
    task: str,
    effective_deriv_root: Path,
    config: Any,
    logger: logging.Logger,
    visualize_categories: Optional[List[str]],
    feature_plotters: Optional[List[str]],
    plot_name_patterns: Optional[List[str]],
    subject_index: int,
    total_subjects: int,
) -> None:
    """Visualize features for a single subject.
    
    Parameters
    ----------
    subject : str
        Subject identifier.
    task : str
        Task name.
    effective_deriv_root : Path
        Resolved derivatives root directory.
    config : Any
        Configuration object.
    logger : logging.Logger
        Logger instance.
    visualize_categories : optional list of str
        Categories to visualize.
    feature_plotters : optional list of str
        Specific plotters to use.
    subject_index : int
        Current subject index (1-based).
    total_subjects : int
        Total number of subjects.
    """
    logger.info(f"[{subject_index}/{total_subjects}] Visualizing sub-{subject}")
    
    epochs, aligned_events = load_epochs_for_analysis(
        subject=subject,
        task=task,
        align="strict",
        preload=False,
        deriv_root=effective_deriv_root,
        config=config,
        logger=logger,
    )
    
    epochs_info = epochs.info if epochs else None
    
    visualize_features(
        subject=subject,
        deriv_root=effective_deriv_root,
        config=config,
        logger=logger,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        visualize_categories=visualize_categories,
        feature_plotters=feature_plotters,
        plot_name_patterns=plot_name_patterns,
    )


def visualize_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
    visualize_categories: Optional[List[str]] = None,
    feature_plotters: Optional[List[str]] = None,
    plot_name_patterns: Optional[List[str]] = None,
) -> None:
    """Visualize features for multiple subjects.
    
    Parameters
    ----------
    subjects : list of str
        Subject identifiers.
    task : optional str
        Task name. If None, uses config default.
    deriv_root : optional Path
        Root directory for derivatives. If None, resolves from config.
    config : optional
        Configuration object. If None, loads default config.
    logger : optional logging.Logger
        Logger instance. If None, creates a new logger.
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        If None, all registered categories are visualized.
    feature_plotters : optional list of str
        Specific plotters to use in "category.name" format.
        If None, all plotters for selected categories are used.
    """
    if not subjects:
        raise ValueError("No subjects specified")
    
    config = _load_config_if_needed(config)
    setup_matplotlib(config)
    
    task = _resolve_task(task, config)
    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    
    if logger is None:
        logger = get_logger(__name__)
    
    _log_visualization_start(subjects, task, visualize_categories, logger)
    
    for index, subject in enumerate(subjects, 1):
        _visualize_single_subject(
            subject=subject,
            task=task,
            effective_deriv_root=effective_deriv_root,
            config=config,
            logger=logger,
            visualize_categories=visualize_categories,
            feature_plotters=feature_plotters,
            plot_name_patterns=plot_name_patterns,
            subject_index=index,
            total_subjects=len(subjects),
        )
    
    logger.info("Feature visualization complete")


__all__ = [
    "visualize_features",
    "visualize_features_for_subjects",
]
