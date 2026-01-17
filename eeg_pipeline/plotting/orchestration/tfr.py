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
    plot_temporal_topomaps_allbands_active,
)
from eeg_pipeline.plotting.tfr.band_evolution import visualize_band_evolution


# Configuration key constants
_CONFIG_KEY_TFR_ANALYSIS = "time_frequency_analysis"
_CONFIG_KEY_BASELINE_WINDOW = "baseline_window"
_CONFIG_KEY_ACTIVE_WINDOW = "active_window"
_CONFIG_KEY_PROJECT_TASK = "project.task"
_CONFIG_KEY_N_JOBS = "time_frequency_analysis.n_jobs"
_CONFIG_KEY_TOPOMAP_WINDOW_SIZE = "erp_analysis.topomap_windows.pain_nonpain_temporal_diff_allbands.window_size_ms"
_CONFIG_KEY_TOPOMAP_WINDOW_COUNT = "erp_analysis.topomap_windows.temporal_allbands_active.window_count"

# Default values
_DEFAULT_BASELINE_WINDOW = (-2.0, 0.0)
_DEFAULT_ACTIVE_WINDOW = (3.0, 10.5)
_DEFAULT_TASK = "thermalactive"
_DEFAULT_TOPOMAP_WINDOW_SIZE_MS = 100.0
_DEFAULT_TOPOMAP_WINDOW_COUNT = 5

# Plot type constants
_PLOT_SCALPMEAN = "scalpmean"
_PLOT_SCALPMEAN_CONTRAST = "scalpmean_contrast"
_PLOT_CHANNELS = "channels"
_PLOT_CHANNELS_CONTRAST = "channels_contrast"
_PLOT_ROIS = "rois"
_PLOT_ROIS_CONTRAST = "rois_contrast"
_PLOT_TOPOMAPS = "topomaps"
_PLOT_BAND_EVOLUTION = "band_evolution"

_ALL_PLOTS = [
    _PLOT_SCALPMEAN,
    _PLOT_SCALPMEAN_CONTRAST,
    _PLOT_CHANNELS,
    _PLOT_CHANNELS_CONTRAST,
    _PLOT_ROIS,
    _PLOT_ROIS_CONTRAST,
    _PLOT_TOPOMAPS,
    _PLOT_BAND_EVOLUTION,
]

_ROI_PLOTS = [_PLOT_ROIS, _PLOT_ROIS_CONTRAST]
_TOPOMAP_PLOTS = [_PLOT_TOPOMAPS]


def _get_tfr_windows(config, logger: logging.Logger) -> tuple[tuple[float, float], tuple[float, float]]:
    """Extract and validate baseline and active windows from config."""
    tfr_analysis = config.get(_CONFIG_KEY_TFR_ANALYSIS, {})
    baseline_window_raw = tfr_analysis.get(_CONFIG_KEY_BASELINE_WINDOW, _DEFAULT_BASELINE_WINDOW)
    baseline_window = validate_baseline_window_pre_stimulus(tuple(baseline_window_raw), logger=logger)
    active_window = tuple(tfr_analysis.get(_CONFIG_KEY_ACTIVE_WINDOW, _DEFAULT_ACTIVE_WINDOW))
    return baseline_window, active_window


def _determine_plots_to_run(
    plots: Optional[List[str]],
    tfr_roi_only: bool,
    tfr_topomaps_only: bool,
) -> List[str]:
    """Determine which plots to run based on flags and explicit plot list."""
    if plots is not None:
        return plots

    if tfr_roi_only:
        return _ROI_PLOTS

    if tfr_topomaps_only:
        return _TOPOMAP_PLOTS

    return _ALL_PLOTS


def _plot_topomaps(
    power,
    events_df,
    plots_dir: Path,
    config,
    baseline_window: tuple[float, float],
    active_window: tuple[float, float],
    logger: logging.Logger,
) -> None:
    """Plot all topomap visualizations."""
    plot_bands_pain_temp_contrasts(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        active_window=active_window,
        logger=logger,
    )

    plot_topomap_grid_baseline_temps(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        active_window=active_window,
        logger=logger,
    )

    window_size_ms = config.get(_CONFIG_KEY_TOPOMAP_WINDOW_SIZE, _DEFAULT_TOPOMAP_WINDOW_SIZE_MS)
    plot_pain_nonpain_temporal_topomaps_diff_allbands(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        active_window=active_window,
        window_size_ms=window_size_ms,
        logger=logger,
    )

    window_count = config.get(_CONFIG_KEY_TOPOMAP_WINDOW_COUNT, _DEFAULT_TOPOMAP_WINDOW_COUNT)
    plot_temporal_topomaps_allbands_active(
        power,
        events_df,
        plots_dir,
        config=config,
        baseline=baseline_window,
        active_window=active_window,
        window_count=window_count,
        logger=logger,
    )


###################################################################
# Subject-level visualization
###################################################################


def _load_subject_data(
    subject: str,
    task: str,
    config,
    effective_deriv_root: Path,
    logger: logging.Logger,
):
    """Load epochs and events for a subject."""
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
        return None, None

    if events_df is None:
        logger.warning("No events available; limited visualizations")
        events_df = epochs.metadata

    return epochs, events_df


def _plot_roi_visualizations(
    power,
    events_df,
    plots_dir: Path,
    config,
    baseline_window: tuple[float, float],
    plots_to_run: List[str],
    logger: logging.Logger,
) -> None:
    """Plot ROI-level TFR visualizations."""
    has_roi_plots = _PLOT_ROIS in plots_to_run or _PLOT_ROIS_CONTRAST in plots_to_run
    if not has_roi_plots:
        return

    roi_tfrs = extract_roi_tfrs(power, config, logger)
    if not roi_tfrs:
        return

    if _PLOT_ROIS in plots_to_run:
        logger.info("Plotting ROI-level TFR...")
        plot_rois_all_trials(roi_tfrs, plots_dir, config=config, baseline=baseline_window, logger=logger)

    if _PLOT_ROIS_CONTRAST in plots_to_run and events_df is not None:
        logger.info("Plotting ROI pain contrast...")
        contrast_pain_nonpain_rois(
            roi_tfrs, events_df, plots_dir, config=config, baseline=baseline_window, logger=logger
        )


def _plot_scalpmean_visualizations(
    power,
    events_df,
    plots_dir: Path,
    config,
    baseline_window: tuple[float, float],
    active_window: tuple[float, float],
    subject: str,
    task: str,
    plots_to_run: List[str],
    logger: logging.Logger,
) -> None:
    """Plot scalp-mean TFR visualizations."""
    if _PLOT_SCALPMEAN in plots_to_run:
        logger.info("Plotting scalp-mean TFR...")
        plot_scalpmean_all_trials(
            power,
            plots_dir,
            config=config,
            baseline=baseline_window,
            active_window=active_window,
            subject=subject,
            task=task,
            logger=logger,
        )

    if _PLOT_SCALPMEAN_CONTRAST in plots_to_run and events_df is not None:
        logger.info("Plotting pain contrast...")
        contrast_scalpmean_pain_nonpain(
            power,
            events_df,
            plots_dir,
            config=config,
            baseline=baseline_window,
            active_window=active_window,
            logger=logger,
            subject=subject,
        )


def _plot_channel_visualizations(
    power,
    events_df,
    plots_dir: Path,
    config,
    baseline_window: tuple[float, float],
    subject: str,
    task: str,
    plots_to_run: List[str],
    logger: logging.Logger,
) -> None:
    """Plot channel-level TFR visualizations."""
    if _PLOT_CHANNELS in plots_to_run:
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

    if _PLOT_CHANNELS_CONTRAST in plots_to_run and events_df is not None:
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


def _execute_plots(
    power,
    events_df,
    plots_dir: Path,
    config,
    baseline_window: tuple[float, float],
    active_window: tuple[float, float],
    subject: str,
    task: str,
    plots_to_run: List[str],
    logger: logging.Logger,
) -> None:
    """Execute all requested plot visualizations."""
    _plot_scalpmean_visualizations(
        power, events_df, plots_dir, config, baseline_window, active_window, subject, task, plots_to_run, logger
    )

    _plot_channel_visualizations(
        power, events_df, plots_dir, config, baseline_window, subject, task, plots_to_run, logger
    )

    _plot_roi_visualizations(power, events_df, plots_dir, config, baseline_window, plots_to_run, logger)

    if _PLOT_TOPOMAPS in plots_to_run and events_df is not None:
        logger.info("Plotting topomaps...")
        _plot_topomaps(power, events_df, plots_dir, config, baseline_window, active_window, logger)

    if _PLOT_BAND_EVOLUTION in plots_to_run and events_df is not None:
        logger.info("Plotting band power evolution...")
        visualize_band_evolution(
            power,
            events_df,
            plots_dir,
            config=config,
            baseline=baseline_window,
            active_window=active_window,
            logger=logger,
        )


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
    """Visualize TFR for a single subject."""
    logger.info(f"Visualizing TFR for sub-{subject}...")

    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)
    plots_dir = deriv_plots_path(effective_deriv_root, subject, subdir="tfr")
    ensure_dir(plots_dir)

    epochs, events_df = _load_subject_data(subject, task, config, effective_deriv_root, logger)
    if epochs is None:
        return

    baseline_window, active_window = _get_tfr_windows(config, logger)
    power = compute_tfr_for_visualization(epochs, config, logger)
    plots_to_run = _determine_plots_to_run(plots, tfr_roi_only, tfr_topomaps_only)

    if tfr_roi_only:
        logger.info("Computing and plotting ROI-level TFR only...")
        _plot_roi_visualizations(power, events_df, plots_dir, config, baseline_window, plots_to_run, logger)
        logger.info(f"TFR ROI visualizations saved to {plots_dir}")
        return

    if tfr_topomaps_only:
        logger.info("Plotting topomaps only...")
        if events_df is None:
            logger.warning("Topomaps require events_df; skipping.")
            return
        _plot_topomaps(power, events_df, plots_dir, config, baseline_window, active_window, logger)
        logger.info(f"TFR topomap visualizations saved to {plots_dir}")
        return

    _execute_plots(
        power, events_df, plots_dir, config, baseline_window, active_window, subject, task, plots_to_run, logger
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
    plots: Optional[List[str]],
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
            plots=plots,
            deriv_root=deriv_root,
        )
        return subject
    except (ValueError, FileNotFoundError, OSError) as e:
        logger.error(f"Failed to visualize sub-{subject}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error visualizing sub-{subject}: {e}")
        return None


def _get_visualization_mode(tfr_roi_only: bool, tfr_topomaps_only: bool) -> str:
    """Determine visualization mode string."""
    if tfr_roi_only:
        return "ROI-only"
    if tfr_topomaps_only:
        return "topomaps-only"
    return "full"


def _process_subjects_sequentially(
    subjects: List[str],
    task: str,
    config,
    effective_deriv_root: Path,
    logger: logging.Logger,
    tfr_roi_only: bool,
    tfr_topomaps_only: bool,
    plots: Optional[List[str]],
    mode_str: str,
) -> None:
    """Process subjects sequentially."""
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
            plots=plots,
            deriv_root=effective_deriv_root,
        )


def _process_subjects_parallel(
    subjects: List[str],
    task: str,
    config,
    effective_deriv_root: Path,
    logger: logging.Logger,
    tfr_roi_only: bool,
    tfr_topomaps_only: bool,
    plots: Optional[List[str]],
    n_jobs: int,
    mode_str: str,
) -> None:
    """Process subjects in parallel."""
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
            plots,
            effective_deriv_root,
        )
        for subject in subjects
    )
    successful = [r for r in results if r is not None]
    logger.info(f"Completed {len(successful)}/{len(subjects)} subjects")


def visualize_tfr_for_subjects(
    subjects: List[str],
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    config=None,
    logger: Optional[logging.Logger] = None,
    tfr_roi_only: bool = False,
    tfr_topomaps_only: bool = False,
    plots: Optional[List[str]] = None,
    n_jobs: Optional[int] = None,
) -> None:
    """Batch process TFR visualizations for multiple subjects."""
    if not subjects:
        raise ValueError("No subjects specified")

    if tfr_roi_only and tfr_topomaps_only:
        raise ValueError("Cannot specify both tfr_roi_only and tfr_topomaps_only")

    if config is None:
        from eeg_pipeline.utils.config.loader import load_config

        config = load_config()

    from eeg_pipeline.plotting.io.figures import setup_matplotlib
    from eeg_pipeline.infra.logging import get_logger

    setup_matplotlib(config)

    task = task or config.get(_CONFIG_KEY_PROJECT_TASK, _DEFAULT_TASK)
    effective_deriv_root = resolve_deriv_root(deriv_root=deriv_root, config=config)

    if logger is None:
        logger = get_logger(__name__)

    if n_jobs is None:
        n_jobs = get_n_jobs(config, config_path=_CONFIG_KEY_N_JOBS)

    mode_str = _get_visualization_mode(tfr_roi_only, tfr_topomaps_only)
    use_parallel = n_jobs > 1 and len(subjects) > 1

    if use_parallel:
        _process_subjects_parallel(
            subjects, task, config, effective_deriv_root, logger, tfr_roi_only, tfr_topomaps_only, plots, n_jobs, mode_str
        )
    else:
        _process_subjects_sequentially(
            subjects, task, config, effective_deriv_root, logger, tfr_roi_only, tfr_topomaps_only, plots, mode_str
        )

    logger.info("TFR visualization complete")


__all__ = [
    "visualize_subject_tfr",
    "visualize_tfr_for_subjects",
]
