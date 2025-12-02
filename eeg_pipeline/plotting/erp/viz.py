"""
ERP visualization orchestration functions.

High-level entry points for creating ERP visualizations for individual subjects
and groups, coordinating calls to contrast and temperature analysis modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging

from ...utils.io.general import deriv_plots_path, ensure_dir, get_logger
from .contrasts import erp_contrast_pain
from .temperature import erp_by_temperature


###################################################################
# ERP visualization helpers
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
    """Visualize ERP plots for a single subject.
    
    Args:
        subject: Subject identifier
        task: Task name
        config: Config dictionary
        crop_tmin: Optional minimum time to crop epochs
        crop_tmax: Optional maximum time to crop epochs
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Visualizing ERP for sub-{subject}...")
    
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="erp")
    ensure_dir(plots_dir)
    
    from eeg_pipeline.pipelines.erp import get_erp_config, load_and_prepare_epochs
    
    erp_cfg = get_erp_config(config)
    
    epochs = load_and_prepare_epochs(
        subject, task, config, crop_tmin, crop_tmax,
        erp_cfg["include_tmax_in_crop"], logger
    )
    
    if epochs is None:
        logger.error(f"Failed to load data for sub-{subject}")
        return
    
    plots_to_run = plots if plots is not None else ["pain_contrast", "temperature"]
    
    if "pain_contrast" in plots_to_run:
        logger.info("Plotting pain contrast...")
        erp_contrast_pain(
            epochs, plots_dir, config, erp_cfg["baseline_window"], erp_cfg["picks"],
            erp_cfg["pain_color"], erp_cfg["nonpain_color"], erp_cfg["combine"],
            erp_cfg["output_files"], logger, subject=subject
        )
    
    if "temperature" in plots_to_run:
        logger.info("Plotting temperature analysis...")
        erp_by_temperature(
            epochs, plots_dir, config, erp_cfg["baseline_window"], erp_cfg["picks"],
            erp_cfg["combine"], erp_cfg["output_files"], logger, subject=subject
        )
    
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
    """Visualize ERP plots for multiple subjects.
    
    Args:
        subjects: List of subject identifiers
        task: Optional task name (uses config default if not provided)
        deriv_root: Optional derivatives root path
        config: Optional config dictionary (loads default if not provided)
        crop_tmin: Optional minimum time to crop epochs
        crop_tmax: Optional maximum time to crop epochs
        logger: Optional logger instance
    
    Raises:
        ValueError: If no subjects are specified
    """
    if not subjects:
        raise ValueError("No subjects specified")
    
    if config is None:
        from ...utils.config.loader import load_settings
        config = load_settings()
    
    from ...utils.io.general import setup_matplotlib
    
    setup_matplotlib(config)
    
    task = task or config.get("project.task", "thermalactive")
    
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Starting ERP visualization: {len(subjects)} subject(s), task={task}")
    
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
        visualize_subject_erp(subject, task, config, crop_tmin=crop_tmin, crop_tmax=crop_tmax, logger=logger)
    
    logger.info("ERP visualization complete")

