from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eeg_pipeline.utils.io.general import save_fig
from eeg_pipeline.plotting.config import get_plot_config

logger = logging.getLogger(__name__)


###################################################################
# Helper Functions (imported from helpers module)
###################################################################

from eeg_pipeline.plotting.decoding.helpers import _despine


###################################################################
# Time Generalization Matrix Plots
###################################################################

def plot_time_generalization_matrix(
    tg_matrix: np.ndarray,
    window_centers: np.ndarray,
    save_path: Path,
    metric: str = "r",
    config: Optional[Any] = None,
) -> None:
    """
    Plot train×test time-generalization matrix for regression decoding.
    tg_matrix should be shape (n_train_windows, n_test_windows); window_centers in seconds.
    """
    if tg_matrix is None or tg_matrix.size == 0:
        logger.warning("Time-generalization matrix is empty; skipping plot")
        return
    if window_centers is None or len(window_centers) == 0:
        logger.warning("Window centers missing for time-generalization plot")
        return

    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("square", plot_type="decoding")

    train_times = np.asarray(window_centers, dtype=float)
    test_times = train_times

    vmax = np.nanmax(np.abs(tg_matrix))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    cmap = "RdBu_r" if metric.lower() == "r" else "magma"
    vmin = -vmax if metric.lower() == "r" else 0.0

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(
        tg_matrix,
        origin="lower",
        aspect="auto",
        extent=[float(test_times[0]), float(test_times[-1]), float(train_times[0]), float(train_times[-1])],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")

    if train_times[0] <= 0 <= train_times[-1]:
        ax.axhline(0, color=plot_cfg.style.colors.black, linestyle="--", linewidth=plot_cfg.style.line.width_thin, alpha=0.6)
    if test_times[0] <= 0 <= test_times[-1]:
        ax.axvline(0, color=plot_cfg.style.colors.black, linestyle="--", linewidth=plot_cfg.style.line.width_thin, alpha=0.6)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())

    ax.set_title("Time-generalization")
    _despine(ax)

    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info("Saved time-generalization matrix to %s", save_path)


def plot_time_generalization_with_null(
    tg_matrix: np.ndarray,
    null_matrix: Optional[np.ndarray],
    window_centers: np.ndarray,
    save_path: Path,
    metric: str = "r",
    config: Optional[Any] = None,
) -> None:
    """
    Plot train×test time-generalization matrix with null distribution significance masking.
    tg_matrix should be shape (n_train_windows, n_test_windows); window_centers in seconds.
    null_matrix can be 2D (same shape as tg_matrix) or 3D (n_permutations, n_train_windows, n_test_windows).
    """
    if tg_matrix is None or tg_matrix.size == 0:
        logger.warning("Time-generalization matrix is empty; skipping plot")
        return
    if window_centers is None or len(window_centers) == 0:
        logger.warning("Window centers missing for time-generalization plot")
        return

    sig_mask = None
    if null_matrix is not None and null_matrix.size > 0:
        if null_matrix.ndim == 3:
            max_null_per_perm = np.nanmax(np.abs(null_matrix), axis=(1, 2))
            max_null_threshold = np.nanpercentile(max_null_per_perm, 95)
            sig_mask = np.abs(tg_matrix) > max_null_threshold
        else:
            vmax_null = np.nanmax(np.abs(null_matrix))
            if np.isfinite(vmax_null) and vmax_null > 0:
                sig_mask = np.abs(tg_matrix) > np.nanpercentile(np.abs(null_matrix), 95, axis=0)

    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("square", plot_type="decoding")

    train_times = np.asarray(window_centers, dtype=float)
    test_times = train_times

    vmax = np.nanmax(np.abs(tg_matrix))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    cmap = "RdBu_r" if metric.lower() == "r" else "magma"
    vmin = -vmax if metric.lower() == "r" else 0.0

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(
        tg_matrix,
        origin="lower",
        aspect="auto",
        extent=[float(test_times[0]), float(test_times[-1]), float(train_times[0]), float(train_times[-1])],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")

    if sig_mask is not None:
        X, Y = np.meshgrid(test_times, train_times)
        ax.contour(X, Y, np.asarray(sig_mask, dtype=float), levels=[0.5], colors="k", linewidths=0.6)

    if train_times[0] <= 0 <= train_times[-1]:
        ax.axhline(0, color=plot_cfg.style.colors.black, linestyle="--", linewidth=plot_cfg.style.line.width_thin, alpha=0.6)
    if test_times[0] <= 0 <= test_times[-1]:
        ax.axvline(0, color=plot_cfg.style.colors.black, linestyle="--", linewidth=plot_cfg.style.line.width_thin, alpha=0.6)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())

    ax.set_title("Time-generalization (empirical vs null)")
    _despine(ax)

    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats)
    logger.info("Saved time-generalization (with null) to %s", save_path)

