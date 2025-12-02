"""
ERP Analysis Pipeline.

Extracts event-related potential statistics from preprocessed epochs.

Usage:
    # Single subject
    extract_erp_stats("0001", "thermalactive", config=config)

    # Multiple subjects
    extract_erp_stats_for_subjects(["0001", "0002"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd


###################################################################
# Configuration
###################################################################


def get_erp_config(config=None) -> dict:
    """Return ERP configuration with defaults."""
    from eeg_pipeline.utils.config.loader import load_settings

    if config is None:
        config = load_settings()

    erp = config.get("erp_analysis.erp", {})

    return {
        "baseline_window": tuple(erp.get("baseline_window", [-0.2, 0.0])),
        "picks": erp.get("picks", "eeg"),
        "pain_color": erp.get("pain_color", "crimson"),
        "nonpain_color": erp.get("nonpain_color", "navy"),
        "combine": erp.get("combine", "gfp"),
        "include_tmax_in_crop": bool(erp.get("include_tmax_in_crop", False)),
        "default_crop_tmin": erp.get("default_crop_tmin"),
        "default_crop_tmax": erp.get("default_crop_tmax"),
        "output_files": erp.get(
            "output_files",
            {
                "pain_gfp": "erp_pain_binary_gfp.png",
                "pain_butterfly": "erp_pain_binary_butterfly.png",
                "temp_gfp": "erp_by_temperature_gfp.png",
                "temp_butterfly": "erp_by_temperature_butterfly.png",
            },
        ),
    }


###################################################################
# Data Loading
###################################################################


def load_and_prepare_epochs(
    subject: str,
    task: str,
    config,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    logger: logging.Logger,
):
    """Load epochs, attach metadata, and crop if requested."""
    from eeg_pipeline.utils.data.loading import load_epochs_for_analysis, crop_epochs

    epochs, aligned_events = load_epochs_for_analysis(
        subject, task,
        align="strict",
        preload=False,
        deriv_root=config.deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )

    if epochs is None or aligned_events is None:
        return None

    epochs.metadata = aligned_events

    if crop_tmin is not None or crop_tmax is not None:
        epochs = crop_epochs(epochs, crop_tmin, crop_tmax, include_tmax_in_crop, logger)

    return epochs


###################################################################
# Statistics Extraction
###################################################################


def extract_erp_stats(
    subject: str,
    task: str,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
    include_tmax_in_crop: bool = False,
    config=None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Extract ERP statistics for a single subject."""
    from eeg_pipeline.utils.io.general import (
        get_logger,
        deriv_stats_path,
        ensure_dir,
        write_tsv,
        find_pain_column_in_metadata,
        find_temperature_column_in_metadata,
    )
    from eeg_pipeline.utils.analysis.stats import count_trials_by_condition

    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Loading epochs for sub-{subject}...")
    epochs = load_and_prepare_epochs(
        subject, task, config, crop_tmin, crop_tmax, include_tmax_in_crop, logger
    )

    if epochs is None:
        logger.error(f"Failed to load data for sub-{subject}, task-{task}")
        return

    logger.info(f"Loaded {len(epochs)} epochs")

    # Count trials by condition
    n_pain = 0
    n_nonpain = 0
    temperatures = []

    pain_col = find_pain_column_in_metadata(epochs, config)
    if pain_col:
        n_pain, n_nonpain = count_trials_by_condition(epochs, pain_col, logger=None)

    temp_col = find_temperature_column_in_metadata(epochs, config)
    if temp_col:
        temperatures = sorted(epochs.metadata[temp_col].unique().tolist())

    logger.info(f"ERP stats: pain={n_pain}, non-pain={n_nonpain}, temps={temperatures}")

    # Save stats
    stats_dir = deriv_stats_path(config.deriv_root, subject)
    ensure_dir(stats_dir)
    counts_file_name = config.get("erp_analysis.erp.output_files.counts_file_name", "erp_trial_counts.tsv")
    stats_path = stats_dir / counts_file_name

    try:
        write_tsv(
            pd.DataFrame([{
                "subject": subject,
                "n_trials_pain": n_pain,
                "n_trials_nonpain": n_nonpain,
                "temperatures": ",".join(map(str, temperatures)),
            }]),
            stats_path,
        )
        logger.info(f"Saved ERP stats to {stats_path}")
    except Exception as e:
        logger.error(f"Failed to write ERP stats: {e}")


###################################################################
# Batch Processing
###################################################################


def extract_erp_stats_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
) -> None:
    """Extract ERP statistics for multiple subjects."""
    from eeg_pipeline.utils.config.loader import load_settings
    from eeg_pipeline.utils.io.general import (
        get_logger,
        setup_matplotlib,
        ensure_derivatives_dataset_description,
    )

    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        config = load_settings()

    if deriv_root is None:
        deriv_root = Path(config.deriv_root)

    setup_matplotlib(config)
    ensure_derivatives_dataset_description(deriv_root=deriv_root)

    task = task or config.get("project.task", "thermalactive")
    erp_config = get_erp_config(config)

    if crop_tmin is None:
        crop_tmin = erp_config.get("default_crop_tmin")
    if crop_tmax is None:
        crop_tmax = erp_config.get("default_crop_tmax")

    include_tmax = bool(erp_config.get("include_tmax_in_crop", False))
    logger = get_logger(__name__)

    logger.info(f"Starting ERP extraction: {len(subjects)} subjects, task={task}")

    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Processing sub-{subject}")
        extract_erp_stats(
            subject, task, crop_tmin, crop_tmax, include_tmax, config, logger
        )

    logger.info(f"ERP extraction complete: {len(subjects)} subjects")
