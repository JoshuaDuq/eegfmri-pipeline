"""TFR visualization orchestration (pipeline-level).

This module is the canonical orchestration layer for time-frequency (TFR) visualizations.
Plot primitives live in `eeg_pipeline.plotting.tfr.*`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from joblib import Parallel, delayed

from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.infra.paths import deriv_plots_path, ensure_dir, resolve_deriv_root
from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization, extract_roi_tfrs
from eeg_pipeline.utils.analysis.stats import validate_baseline_window_pre_stimulus
from eeg_pipeline.utils.parallel import get_n_jobs

from eeg_pipeline.plotting.tfr.scalpmean import (
    plot_scalpmean_all_trials,
    contrast_scalpmean_pain_nonpain,
)
from eeg_pipeline.plotting.tfr.channels import plot_channels_all_trials, contrast_channels_pain_nonpain
from eeg_pipeline.plotting.tfr.rois import plot_rois_all_trials, contrast_pain_nonpain_rois
from eeg_pipeline.plotting.tfr.contrasts import plot_bands_pain_temp_contrasts
from eeg_pipeline.plotting.tfr.topomaps import (
    plot_topomap_grid_baseline_temps,
    plot_pain_nonpain_temporal_topomaps_diff_allbands,
    plot_temporal_topomaps_allbands_plateau,
)
from eeg_pipeline.plotting.tfr.band_evolution import visualize_band_evolution


###################################################################
# TFR visualization helpers
###################################################################


def _plot_topomaps(
    power,
    events_df,
    plots_dir: Path,
    config,
    baseline_window,
    plateau_window,
    logger: logging.Logger,
) -> None:
    """Plot all topomap visualizations."""
    plot_bands_pain_temp_contrasts(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        plateau_window=plateau_window,
        logger=logger,
    )

    plot_topomap_grid_baseline_temps(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        plateau_window=plateau_window,
        logger=logger,
    )

    window_size_ms = config.get(
        "erp_analysis.topomap_windows.pain_nonpain_temporal_diff_allbands.window_size_ms", 100.0
    )
    plot_pain_nonpain_temporal_topomaps_diff_allbands(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        plateau_window=plateau_window,
        window_size_ms=window_size_ms,
        logger=logger,
    )

    window_count = config.get("erp_analysis.topomap_windows.temporal_allbands_plateau.window_count", 5)
    plot_temporal_topomaps_allbands_plateau(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        plateau_window=plateau_window,
        window_count=window_count,
        logger=logger,
    )


###################################################################
# Subject-level visualization
###################################################################


def visualize_subject_tfr(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
    tfr_roi_only: bool = False,
    tfr_topomaps_only: bool = False,
    plots: Optional[List[str]] = None,
    deriv_root: Optional[Path] = None,
) -> None:
    logger.info(f"Visualizing TFR for sub-{subject}...")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    plots_dir = deriv_plots_path(effective_deriv_root, subject, subdir="tfr")
    ensure_dir(plots_dir)

    epochs, events_df = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=True,
        deriv_root=effective_deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )

    if epochs is None:
        logger.error(f"Failed to load epochs for sub-{subject}")
        return

    if events_df is None:
        logger.warning("No events available; limited visualizations")
        events_df = epochs.metadata if hasattr(epochs, "metadata") else None

    tfr_analysis = config.get("time_frequency_analysis", {})
    baseline_window_raw = tuple(tfr_analysis.get("baseline_window", [-2.0, 0.0]))
    baseline_window = validate_baseline_window_pre_stimulus(baseline_window_raw, logger=logger)
    plateau_window = tuple(tfr_analysis.get("plateau_window", [3.0, 10.5]))

    power = compute_tfr_for_visualization(epochs, config, logger)

    plots_to_run = plots if plots is not None else (
        ["rois", "rois_contrast"]
        if tfr_roi_only
        else ["topomaps"]
        if tfr_topomaps_only
        else [
            "scalpmean",
            "scalpmean_contrast",
            "channels",
            "channels_contrast",
            "rois",
            "rois_contrast",
            "topomaps",
            "band_evolution",
        ]
    )

    if tfr_roi_only:
        logger.info("Computing and plotting ROI-level TFR only...")
        if "rois" in plots_to_run or "rois_contrast" in plots_to_run:
            roi_tfrs = extract_roi_tfrs(power, config, logger)
            if roi_tfrs:
                if "rois" in plots_to_run:
                    logger.info("Plotting ROI-level TFR...")
                    plot_rois_all_trials(roi_tfrs, plots_dir, config=config, baseline=baseline_window, logger=logger)

                if "rois_contrast" in plots_to_run and events_df is not None:
                    logger.info("Plotting ROI pain contrast...")
                    contrast_pain_nonpain_rois(roi_tfrs, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger)
        logger.info(f"TFR ROI visualizations saved to {plots_dir}")
        return

    if tfr_topomaps_only:
        logger.info("Plotting topomaps only...")
        if events_df is None:
            logger.warning("Topomaps require events_df; skipping.")
            return

        if "topomaps" in plots_to_run:
            _plot_topomaps(power, events_df, plots_dir, config, baseline_window, plateau_window, logger)

        logger.info(f"TFR topomap visualizations saved to {plots_dir}")
        return

    if "scalpmean" in plots_to_run:
        logger.info("Plotting scalp-mean TFR...")
        plot_scalpmean_all_trials(
            power,
            plots_dir,
            config=config,
            baseline=baseline_window,
            plateau_window=plateau_window,
            subject=subject,
            task=task,
            logger=logger,
        )

    if "scalpmean_contrast" in plots_to_run and events_df is not None:
        logger.info("Plotting pain contrast...")
        contrast_scalpmean_pain_nonpain(
            power,
            events_df,
            plots_dir,
            config=config,
            baseline=baseline_window,
            plateau_window=plateau_window,
            logger=logger,
            subject=subject,
        )

    if "channels" in plots_to_run:
        logger.info("Plotting channel-level TFR...")
        plot_channels_all_trials(
            power,
            plots_dir,
            config=config,
            baseline=baseline_window,
            logger=logger,
            subject=subject,
            task=task,
        )

    if "channels_contrast" in plots_to_run and events_df is not None:
        logger.info("Plotting channel pain contrast...")
        contrast_channels_pain_nonpain(
            power,
            events_df,
            plots_dir,
            config=config,
            baseline=baseline_window,
            logger=logger,
            subject=subject,
        )

    if "rois" in plots_to_run or "rois_contrast" in plots_to_run:
        logger.info("Extracting and plotting ROI-level TFR...")
        roi_tfrs = extract_roi_tfrs(power, config, logger)
        if roi_tfrs:
            if "rois" in plots_to_run:
                plot_rois_all_trials(roi_tfrs, plots_dir, config=config, baseline=baseline_window, logger=logger)

            if "rois_contrast" in plots_to_run and events_df is not None:
                contrast_pain_nonpain_rois(roi_tfrs, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger)

    if "topomaps" in plots_to_run and events_df is not None:
        logger.info("Plotting topomaps...")
        _plot_topomaps(power, events_df, plots_dir, config, baseline_window, plateau_window, logger)

    if "band_evolution" in plots_to_run and events_df is not None:
        logger.info("Plotting band power evolution...")
        visualize_band_evolution(
            power,
            events_df,
            plots_dir,
            config=config,
            baseline=baseline_window,
            plateau_window=plateau_window,
            logger=logger,
        )

    logger.info(f"TFR visualizations saved to {plots_dir}")


###################################################################
# Batch processing
###################################################################


def _visualize_single_subject(
    subject: str,
    task: str,
    config,
    tfr_roi_only: bool,
    tfr_topomaps_only: bool,
    deriv_root: Path,
) -> Optional[str]:
    """Worker function for parallel TFR visualization."""
    from eeg_pipeline.plotting.io.figures import setup_matplotlib
    from eeg_pipeline.infra.logging import get_logger

    setup_matplotlib(config)
    logger = get_logger(__name__)

    try:
        visualize_subject_tfr(
            subject,
            task,
            config,
            logger,
            tfr_roi_only=tfr_roi_only,
            tfr_topomaps_only=tfr_topomaps_only,
            deriv_root=deriv_root,
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
    """Batch process TFR visualizations for multiple subjects."""
    if not subjects:
        raise ValueError("No subjects specified")

    if config is None:
        from eeg_pipeline.utils.config.loader import load_settings

        config = load_settings()

    from eeg_pipeline.plotting.io.figures import setup_matplotlib
    from eeg_pipeline.infra.logging import get_logger

    setup_matplotlib(config)

    task = task or config.get("project.task", "thermalactive")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    if tfr_roi_only and tfr_topomaps_only:
        raise ValueError("Cannot specify both tfr_roi_only and tfr_topomaps_only")

    if n_jobs is None:
        n_jobs = get_n_jobs(config, config_path="time_frequency_analysis.n_jobs")

    mode_str = "ROI-only" if tfr_roi_only else "topomaps-only" if tfr_topomaps_only else "full"

    if n_jobs == 1 or len(subjects) == 1:
        logger.info(
            f"Starting TFR visualization ({mode_str}): {len(subjects)} subject(s), task={task} [sequential]"
        )
        for idx, subject in enumerate(subjects, 1):
            logger.info(f"[{idx}/{len(subjects)}] Visualizing sub-{subject}")
            visualize_subject_tfr(
                subject,
                task,
                config,
                logger,
                tfr_roi_only=tfr_roi_only,
                tfr_topomaps_only=tfr_topomaps_only,
                deriv_root=effective_deriv_root,
            )
    else:
        logger.info(
            f"Starting TFR visualization ({mode_str}): {len(subjects)} subject(s), task={task} "
            f"[parallel, n_jobs={n_jobs}]"
        )
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_visualize_single_subject)(
                subject,
                task,
                config,
                tfr_roi_only,
                tfr_topomaps_only,
                effective_deriv_root,
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

