"""
TFR visualization orchestration functions.

High-level entry points for creating time-frequency representation (TFR) visualizations
at both subject and group levels. Coordinates calls to specialized plotting modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import logging

from joblib import Parallel, delayed, cpu_count

from ...utils.data.loading import load_epochs_for_analysis
from ...utils.io.general import deriv_plots_path, ensure_dir
from ...utils.analysis.tfr import (
    compute_tfr_for_visualization,
    extract_roi_tfrs,
)
from ...utils.analysis.stats import validate_baseline_window_pre_stimulus
from .scalpmean import (
    plot_scalpmean_all_trials,
    contrast_scalpmean_pain_nonpain,
)
from .channels import (
    plot_channels_all_trials,
)
from .rois import (
    plot_rois_all_trials,
)
from .contrasts import (
    plot_bands_pain_temp_contrasts,
)
from .topomaps import (
    plot_topomap_grid_baseline_temps,
    plot_pain_nonpain_temporal_topomaps_diff_allbands,
    plot_temporal_topomaps_allbands_plateau,
)
from .channels import (
    contrast_channels_pain_nonpain,
)
from .rois import (
    contrast_pain_nonpain_rois,
)
from .band_evolution import (
    visualize_band_evolution,
)


###################################################################
# TFR visualization helpers
###################################################################


def _plot_topomaps(
    power,
    events_df,
    plots_dir,
    config,
    baseline_window,
    plateau_window,
    logger: logging.Logger,
) -> None:
    """Plot all topomap visualizations.
    
    Consolidates topomap plotting logic to avoid duplication.
    """
    plot_bands_pain_temp_contrasts(
        power, events_df, plots_dir, config=config, baseline=baseline_window,
        plateau_window=plateau_window, logger=logger
    )
    
    plot_topomap_grid_baseline_temps(
        power, events_df, plots_dir, config=config, baseline=baseline_window,
        plateau_window=plateau_window, logger=logger
    )
    
    window_size_ms = config.get(
        "erp_analysis.topomap_windows.pain_nonpain_temporal_diff_allbands.window_size_ms", 100.0
    )
    plot_pain_nonpain_temporal_topomaps_diff_allbands(
        power, events_df, plots_dir, config=config, baseline=baseline_window,
        plateau_window=plateau_window, window_size_ms=window_size_ms, logger=logger
    )
    
    window_count = config.get(
        "erp_analysis.topomap_windows.temporal_allbands_plateau.window_count", 5
    )
    plot_temporal_topomaps_allbands_plateau(
        power, events_df, plots_dir, config=config, baseline=baseline_window,
        plateau_window=plateau_window, window_count=window_count, logger=logger
    )


def visualize_subject_tfr(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    tfr_roi_only: bool = False,
    tfr_topomaps_only: bool = False,
    plots: Optional[List[str]] = None,
) -> None:
    """Visualize TFR for a single subject.
    
    Creates comprehensive time-frequency representation visualizations including:
    - Scalp-mean TFR plots
    - Channel-level TFR plots
    - ROI-level TFR plots
    - Topomap visualizations
    - Pain/non-pain contrasts
    
    Args:
        subject: Subject ID (without 'sub-' prefix)
        task: Task name
        config: Configuration object
        logger: Logger instance
        tfr_roi_only: If True, only create ROI-level plots
        tfr_topomaps_only: If True, only create topomap plots
    """
    logger.info(f"Visualizing TFR for sub-{subject}...")
    
    plots_dir = deriv_plots_path(config.deriv_root, subject, subdir="tfr")
    ensure_dir(plots_dir)
    
    epochs, events_df = load_epochs_for_analysis(
        subject, task, align="strict", preload=True,
        deriv_root=config.deriv_root, bids_root=config.bids_root,
        config=config, logger=logger
    )
    
    if epochs is None:
        logger.error(f"Failed to load epochs for sub-{subject}")
        return
    
    if events_df is None:
        logger.warning("No events available; limited visualizations")
        events_df = epochs.metadata if hasattr(epochs, 'metadata') else None
    
    tfr_analysis = config.get("time_frequency_analysis", {})
    baseline_window_raw = tuple(tfr_analysis.get("baseline_window", [-2.0, 0.0]))
    baseline_window = validate_baseline_window_pre_stimulus(baseline_window_raw, logger=logger)
    plateau_window = tuple(tfr_analysis.get("plateau_window", [3.0, 10.5]))
    
    power = compute_tfr_for_visualization(epochs, config, logger)
    
    
    plots_to_run = plots if plots is not None else (
        ["rois", "rois_contrast"] if tfr_roi_only else
        ["topomaps"] if tfr_topomaps_only else
        ["scalpmean", "scalpmean_contrast", "channels", "channels_contrast", "rois", "rois_contrast", "topomaps", "band_evolution"]
    )
    
    if tfr_roi_only:
        logger.info("Computing and plotting ROI-level TFR only...")
        if "rois" in plots_to_run or "rois_contrast" in plots_to_run:
            roi_tfrs = extract_roi_tfrs(power, config, logger)
            if roi_tfrs:
                if "rois" in plots_to_run:
                    logger.info("Plotting ROI-level TFR...")
                    plot_rois_all_trials(
                        roi_tfrs, plots_dir, config=config, baseline=baseline_window, logger=logger
                    )
                
                if "rois_contrast" in plots_to_run and events_df is not None:
                    logger.info("Plotting ROI pain contrast...")
                    contrast_pain_nonpain_rois(
                        roi_tfrs, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger
                    )
        logger.info(f"TFR ROI visualizations saved to {plots_dir}")
        return
    
    if tfr_topomaps_only:
        logger.info("Plotting topomaps only...")
        if events_df is None:
            logger.warning("Topomaps require events_df; skipping.")
            return
        
        if "topomaps" in plots_to_run:
            _plot_topomaps(
                power, events_df, plots_dir, config,
                baseline_window, plateau_window, logger
            )
        
        logger.info(f"TFR topomap visualizations saved to {plots_dir}")
        return
    
    if "scalpmean" in plots_to_run:
        logger.info("Plotting scalp-mean TFR...")
        plot_scalpmean_all_trials(
            power, plots_dir, config=config, baseline=baseline_window,
            plateau_window=plateau_window, subject=subject, task=task, logger=logger
        )
    
    if "scalpmean_contrast" in plots_to_run and events_df is not None:
        logger.info("Plotting pain contrast...")
        contrast_scalpmean_pain_nonpain(
            power, events_df, plots_dir, config=config, baseline=baseline_window,
            plateau_window=plateau_window, logger=logger, subject=subject
        )
    
    if "channels" in plots_to_run:
        logger.info("Plotting channel-level TFR...")
        plot_channels_all_trials(
            power, plots_dir, config=config, baseline=baseline_window,
            logger=logger, subject=subject, task=task
        )
    
    if "channels_contrast" in plots_to_run and events_df is not None:
        logger.info("Plotting channel pain contrast...")
        contrast_channels_pain_nonpain(
            power, events_df, plots_dir, config=config, baseline=baseline_window,
            logger=logger, subject=subject
        )
    
    if "rois" in plots_to_run or "rois_contrast" in plots_to_run:
        logger.info("Extracting and plotting ROI-level TFR...")
        roi_tfrs = extract_roi_tfrs(power, config, logger)
        if roi_tfrs:
            if "rois" in plots_to_run:
                plot_rois_all_trials(
                    roi_tfrs, plots_dir, config=config, baseline=baseline_window, logger=logger
                )
            
            if "rois_contrast" in plots_to_run and events_df is not None:
                contrast_pain_nonpain_rois(
                    roi_tfrs, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger
                )
    
    if "topomaps" in plots_to_run and events_df is not None:
        logger.info("Plotting topomaps...")
        _plot_topomaps(
            power, events_df, plots_dir, config,
            baseline_window, plateau_window, logger
        )
    
    # Band power evolution plots
    if "band_evolution" in plots_to_run and events_df is not None:
        logger.info("Plotting band power evolution...")
        visualize_band_evolution(
            power, events_df, plots_dir, config=config,
            baseline=baseline_window, plateau_window=plateau_window, logger=logger
        )
    
    logger.info(f"TFR visualizations saved to {plots_dir}")


###################################################################
# Batch processing
###################################################################


def _get_n_jobs(config=None) -> int:
    """Get number of parallel jobs from config or environment."""
    n_jobs = -1
    
    if config is not None:
        from ...utils.config.loader import get_config_value
        n_jobs = int(get_config_value(config, "time_frequency_analysis.n_jobs", -1))
    
    env_jobs = os.environ.get("EEG_PIPELINE_N_JOBS")
    if env_jobs is not None:
        try:
            n_jobs = int(env_jobs)
        except ValueError:
            pass
    
    if n_jobs == -1:
        n_jobs = max(1, cpu_count() - 1)
    elif n_jobs <= 0:
        n_jobs = 1
    
    return n_jobs


def _visualize_single_subject(
    subject: str,
    task: str,
    config,
    tfr_roi_only: bool,
    tfr_topomaps_only: bool,
) -> str:
    """Worker function for parallel TFR visualization.
    
    Returns subject ID on success for logging.
    """
    from ...utils.io.general import setup_matplotlib, get_logger
    
    setup_matplotlib(config)
    logger = get_logger(__name__)
    
    try:
        visualize_subject_tfr(
            subject, task, config, logger,
            tfr_roi_only=tfr_roi_only,
            tfr_topomaps_only=tfr_topomaps_only
        )
        return subject
    except Exception as e:
        logger.error(f"Failed to visualize sub-{subject}: {e}")
        return None


def visualize_tfr_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    logger: Optional[logging.Logger] = None,
    tfr_roi_only: bool = False,
    tfr_topomaps_only: bool = False,
    n_jobs: Optional[int] = None,
) -> None:
    """Batch process TFR visualizations for multiple subjects.
    
    High-level entry point for creating TFR visualizations for a list of subjects.
    Supports parallel processing across subjects.
    
    Args:
        subjects: List of subject IDs (without 'sub-' prefix)
        task: Optional task name (defaults to config)
        deriv_root: Optional derivatives root path
        config: Optional configuration object (loads from settings if None)
        logger: Optional logger instance
        tfr_roi_only: If True, only create ROI-level plots
        tfr_topomaps_only: If True, only create topomap plots
        n_jobs: Number of parallel jobs (-1 = all cores minus 1, 1 = sequential)
    """
    if not subjects:
        raise ValueError("No subjects specified")
    
    if config is None:
        from ...utils.config.loader import load_settings
        config = load_settings()
    
    from ...utils.io.general import setup_matplotlib, get_logger
    
    setup_matplotlib(config)
    
    task = task or config.get("project.task", "thermalactive")
    
    if logger is None:
        logger = get_logger(__name__)
    
    if tfr_roi_only and tfr_topomaps_only:
        raise ValueError("Cannot specify both tfr_roi_only and tfr_topomaps_only")
    
    if n_jobs is None:
        n_jobs = _get_n_jobs(config)
    
    mode_str = "ROI-only" if tfr_roi_only else "topomaps-only" if tfr_topomaps_only else "full"
    
    if n_jobs == 1 or len(subjects) == 1:
        logger.info(f"Starting TFR visualization ({mode_str}): {len(subjects)} subject(s), task={task} [sequential]")
        for idx, subject in enumerate(subjects, 1):
            logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
            visualize_subject_tfr(subject, task, config, logger, tfr_roi_only=tfr_roi_only, tfr_topomaps_only=tfr_topomaps_only)
    else:
        logger.info(f"Starting TFR visualization ({mode_str}): {len(subjects)} subject(s), task={task} [parallel, n_jobs={n_jobs}]")
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_visualize_single_subject)(
                subject, task, config, tfr_roi_only, tfr_topomaps_only
            )
            for subject in subjects
        )
        successful = [r for r in results if r is not None]
        logger.info(f"Completed {len(successful)}/{len(subjects)} subjects")
    
    logger.info("TFR visualization complete")


__all__ = [
    "visualize_subject_tfr",
    "visualize_tfr_for_subjects",
]

