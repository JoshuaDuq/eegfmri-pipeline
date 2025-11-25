from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import logging

import pandas as pd

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import load_epochs_for_analysis, crop_epochs
from eeg_pipeline.utils.io.general import (
    ensure_dir,
    deriv_stats_path,
    get_logger,
    write_tsv,
)
from eeg_pipeline.utils.analysis.stats import count_trials_by_condition
from eeg_pipeline.utils.io.general import find_pain_column_in_metadata, find_temperature_column_in_metadata


def get_erp_config(config=None):
    """Return ERP configuration with defaults applied."""
    if config is None:
        config = load_settings()
    erp_config = config.get("foundational_analysis", {}).get("erp", {})
    return {
        "baseline_window": tuple(erp_config.get("baseline_window", [-0.2, 0.0])),
        "picks": erp_config.get("picks", "eeg"),
        "pain_color": erp_config.get("pain_color", "crimson"),
        "nonpain_color": erp_config.get("nonpain_color", "navy"),
        "combine": erp_config.get("combine", "gfp"),
        "include_tmax_in_crop": bool(erp_config.get("include_tmax_in_crop", False)),
        "default_crop_tmin": erp_config.get("default_crop_tmin"),
        "default_crop_tmax": erp_config.get("default_crop_tmax"),
        "output_files": erp_config.get(
            "output_files",
            {
                "pain_gfp": "erp_pain_binary_gfp.png",
                "pain_butterfly": "erp_pain_binary_butterfly.png",
                "temp_gfp": "erp_by_temperature_gfp.png",
                "temp_butterfly": "erp_by_temperature_butterfly.png",
            },
        ),
    }


def load_and_prepare_epochs(
    subject: str,
    task: str,
    config,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    logger: logging.Logger,
):
    """Load epochs/events, attach metadata, and crop if requested."""
    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
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


def extract_erp_stats(
    subject: str,
    task: str,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax_in_crop: bool,
    config,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"  Loading epochs for sub-{subject}...")
    epochs = load_and_prepare_epochs(
        subject, task, config, crop_tmin, crop_tmax,
        include_tmax_in_crop, logger
    )
    
    if epochs is None:
        logger.error(f"Failed to load data for sub-{subject}, task-{task}")
        return
    
    logger.info(f"  Loaded {len(epochs)} epochs")
    
    n_trials_pain = 0
    n_trials_nonpain = 0
    temperatures_detected = []
    
    pain_column = find_pain_column_in_metadata(epochs, config)
    if pain_column:
        n_trials_pain, n_trials_nonpain = count_trials_by_condition(epochs, pain_column, logger=None)
    
    temperature_column = find_temperature_column_in_metadata(epochs, config)
    if temperature_column:
        temperatures_detected = sorted(epochs.metadata[temperature_column].unique().tolist())
    
    logger.info(
        f"  ERP stats: pain={n_trials_pain}, "
        f"non-pain={n_trials_nonpain}, "
        f"temperatures={temperatures_detected}"
    )
    
    stats_dir = deriv_stats_path(config.deriv_root, subject)
    ensure_dir(stats_dir)
    stats_path = stats_dir / "erp_trial_counts.tsv"
    
    try:
        write_tsv(pd.DataFrame([{
            "subject": subject,
            "n_trials_pain": n_trials_pain,
            "n_trials_nonpain": n_trials_nonpain,
            "temperatures": ",".join(map(str, temperatures_detected)),
        }]), stats_path)
        logger.info(f"  Saved ERP stats to {stats_path}")
    except Exception as e:
        logger.error(f"Failed to write ERP stats to {stats_path}: {e}")


###################################################################
# Batch processing
###################################################################


def extract_erp_stats_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    crop_tmin: Optional[float] = None,
    crop_tmax: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not subjects:
        raise ValueError("No subjects specified. Use --group all|A,B,C, or --subject (can repeat).")
    
    if config is None:
        config = load_settings()
    
    from eeg_pipeline.utils.io.general import ensure_derivatives_dataset_description, setup_matplotlib
    
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
    include_tmax_in_crop = bool(erp_config.get("include_tmax_in_crop", False))
    
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Starting ERP statistics extraction: {len(subjects)} subject(s), task={task}")
    logger.info(f"Subjects: {', '.join(subjects)}")
    
    for idx, subject in enumerate(subjects, 1):
        logger.info(f"[{idx}/{len(subjects)}] Processing sub-{subject}")
        extract_erp_stats(subject, task, crop_tmin, crop_tmax, include_tmax_in_crop, config, logger)
    
    logger.info(f"ERP statistics extraction complete: {len(subjects)} subjects processed")
