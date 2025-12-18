"""Feature visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for descriptive feature visualizations.
Plot primitives and plot registries live in `eeg_pipeline.plotting.features.*`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.infra.paths import deriv_features_path, deriv_plots_path, ensure_dir, resolve_deriv_root
from eeg_pipeline.plotting.io.figures import setup_matplotlib

# Import plotters for side-effects: registers plot functions into VisualizationRegistry.
# This import must not call back into this module at import time.
from eeg_pipeline.plotting.features import registrations as _feature_plotters  # noqa: F401
from eeg_pipeline.plotting.features.context import FeaturePlotContext, VisualizationManager


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
) -> Dict[str, Path]:
    """Generate descriptive feature visualizations using registered plotters.
    
    Parameters
    ----------
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        If None, all registered categories are visualized.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    features_dir = deriv_features_path(deriv_root, subject)
    plots_dir = deriv_plots_path(deriv_root, subject, subdir="features")
    ensure_dir(plots_dir)

    ctx = FeaturePlotContext(
        subject=subject,
        plots_dir=plots_dir,
        features_dir=features_dir,
        config=config,
        logger=logger,
        epochs_info=epochs_info,
        aligned_events=aligned_events,
        epochs=epochs,
        tfr=tfr,
    )
    ctx.load_data()

    if ctx.power_df is None and ctx.connectivity_df is None:
        logger.warning(f"No feature data found for subject {subject}")
        if ctx.epochs is None:
            logger.warning("No epochs data found either.")
            return {}

    manager = VisualizationManager(ctx)
    
    if visualize_categories:
        logger.info(f"Visualizing specific categories: {', '.join(visualize_categories)}")
        for category in visualize_categories:
            manager.run_category(category)
        saved_plots = manager.saved_plots
    else:
        saved_plots = manager.run_all()

    _save_plot_manifest(plots_dir=plots_dir, subject=subject, logger=logger)

    return saved_plots


def _save_plot_manifest(
    plots_dir: Path,
    subject: str,
    logger: logging.Logger,
) -> None:
    plot_files: List[Path] = []
    for ext in ["png", "svg", "pdf"]:
        plot_files.extend(plots_dir.rglob(f"*.{ext}"))

    manifest = {
        "subject": subject,
        "generated_at": datetime.now().isoformat(),
        "plots_directory": str(plots_dir),
        "total_plots": len(plot_files),
        "plots": [],
    }

    for path in sorted(plot_files):
        path_str = str(path)
        feature_type = "unknown"
        for ftype in [
            "power",
            "connectivity",
            "microstates",
            "pac",
            "itpc",
            "complexity",
            "burst",
            "aperiodic",
            "dynamics",
            "summary",
            "erds",
        ]:
            if f"/{ftype}/" in path_str or f"\\{ftype}\\" in path_str:
                feature_type = ftype
                break

        manifest["plots"].append(
            {
                "name": path.stem,
                "path": str(path.relative_to(plots_dir)),
                "feature_type": feature_type,
                "format": path.suffix[1:],
            }
        )

    manifest_path = plots_dir / f"sub-{subject}_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(f"Saved plot manifest ({len(plot_files)} plots)")


def visualize_features_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
    visualize_categories: Optional[List[str]] = None,
) -> None:
    """Visualize features for multiple subjects.
    
    Parameters
    ----------
    visualize_categories : optional list of str
        Specific categories to visualize (e.g., ["power", "connectivity"]).
        If None, all registered categories are visualized.
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

    cat_str = f" ({', '.join(visualize_categories)})" if visualize_categories else ""
    logger.info(f"Starting feature visualization{cat_str}: {len(subjects)} subject(s), task={task}")

    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")

        epochs, aligned_events = load_epochs_for_analysis(
            subject,
            task,
            align="strict",
            preload=False,
            deriv_root=effective_deriv_root,
            bids_root=config.bids_root,
            config=config,
            logger=logger,
        )

        visualize_features(
            subject=subject,
            deriv_root=effective_deriv_root,
            config=config,
            logger=logger,
            epochs_info=epochs.info if epochs else None,
            aligned_events=aligned_events,
            epochs=epochs,
            visualize_categories=visualize_categories,
        )

    logger.info("Feature visualization complete")


__all__ = [
    "visualize_features",
    "visualize_features_for_subjects",
]
