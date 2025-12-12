"""ERP visualization orchestration (pipeline-level).

Plot primitives live in `eeg_pipeline.plotting.erp.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from eeg_pipeline.utils.io.paths import deriv_plots_path, ensure_dir
from eeg_pipeline.utils.io.logging import get_logger
from eeg_pipeline.utils.io.plotting import setup_matplotlib
from eeg_pipeline.plotting.erp.contrasts import erp_contrast_pain
from eeg_pipeline.plotting.erp.temperature import erp_by_temperature
from eeg_pipeline.plotting.erp.registry import (
    ERPPlotContext,
    ERPPlotManager,
    ERPPlotRegistry,
)


###################################################################
# Subject-level ERP Visualization
###################################################################


def visualize_subject_erp(
    subject: str,
    task: str,
    config,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    plots: Optional[List[str]] = None,
) -> None:
    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Visualizing ERP for sub-{subject}...")

    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="erp")
    ensure_dir(plots_dir)

    from eeg_pipeline.pipelines.erp import get_erp_config, load_and_prepare_epochs

    erp_cfg = get_erp_config(config)

    epochs = load_and_prepare_epochs(
        subject,
        task,
        config,
        crop_tmin,
        crop_tmax,
        erp_cfg["include_tmax_in_crop"],
        logger,
    )

    if epochs is None:
        logger.error(f"Failed to load data for sub-{subject}")
        return

    ctx = ERPPlotContext(
        subject=subject,
        task=task,
        config=config,
        plots_dir=plots_dir,
        epochs=epochs,
        erp_cfg=erp_cfg,
        logger=logger,
    )
    manager = ERPPlotManager(ctx)

    plots_to_run = plots if plots is not None else ["pain_contrast", "temperature"]
    manager.run_selected(plots_to_run) if plots else manager.run_all()

    logger.info(f"ERP visualizations saved to {plots_dir}")


###################################################################
# Batch processing
###################################################################


def visualize_erp_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        from eeg_pipeline.utils.config.loader import load_settings

        config = load_settings()

    setup_matplotlib(config)

    task = task or config.get("project.task", "thermalactive")

    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Starting ERP visualization: {len(subjects)} subject(s), task={task}")

    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        visualize_subject_erp(
            subject,
            task,
            config,
            crop_tmin=crop_tmin,
            crop_tmax=crop_tmax,
            logger=logger,
        )

    logger.info("ERP visualization complete")


###################################################################
# Registry adapters
###################################################################


@ERPPlotRegistry.register("pain_contrast")
def run_pain_contrast(ctx: ERPPlotContext, saved_plots):
    erp_contrast_pain(
        ctx.epochs,
        ctx.plots_dir,
        ctx.config,
        ctx.erp_cfg["baseline_window"],
        ctx.erp_cfg["picks"],
        ctx.erp_cfg["pain_color"],
        ctx.erp_cfg["nonpain_color"],
        ctx.erp_cfg["combine"],
        ctx.erp_cfg["output_files"],
        ctx.logger,
        subject=ctx.subject,
    )
    saved_plots["pain_contrast"] = ctx.plots_dir


@ERPPlotRegistry.register("temperature")
def run_temperature(ctx: ERPPlotContext, saved_plots):
    erp_by_temperature(
        ctx.epochs,
        ctx.plots_dir,
        ctx.config,
        ctx.erp_cfg["baseline_window"],
        ctx.erp_cfg["picks"],
        ctx.erp_cfg["combine"],
        ctx.erp_cfg["output_files"],
        ctx.logger,
        subject=ctx.subject,
    )
    saved_plots["temperature"] = ctx.plots_dir


__all__ = [
    "visualize_subject_erp",
    "visualize_erp_for_subjects",
]

