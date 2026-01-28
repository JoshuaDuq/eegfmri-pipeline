from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.machine_learning.helpers import despine

logger = logging.getLogger(__name__)


# Constants
SIGNIFICANCE_PERCENTILE = 95
CONTOUR_LEVEL = 0.5
ZERO_LINE_ALPHA = 0.6
DEFAULT_VMAX = 1.0
CORRELATION_METRIC = "r"


def _validate_time_generalization_inputs(
    tg_matrix: np.ndarray,
    window_centers: np.ndarray,
) -> bool:
    """Validate inputs for time-generalization plotting."""
    if tg_matrix.size == 0:
        logger.warning("Time-generalization matrix is empty; skipping plot")
        return False
    if len(window_centers) == 0:
        logger.warning("Window centers missing for time-generalization plot")
        return False
    return True


def _compute_significance_mask(
    tg_matrix: np.ndarray,
    null_matrix: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Compute significance mask from null distribution."""
    if null_matrix is None or null_matrix.size == 0:
        return None

    if null_matrix.ndim == 3:
        max_null_per_permutation = np.nanmax(np.abs(null_matrix), axis=(1, 2))
        null_threshold = np.nanpercentile(max_null_per_permutation, SIGNIFICANCE_PERCENTILE)
        return np.abs(tg_matrix) > null_threshold

    null_threshold = np.nanpercentile(np.abs(null_matrix), SIGNIFICANCE_PERCENTILE, axis=0)
    if not np.any(np.isfinite(null_threshold)):
        return None

    return np.abs(tg_matrix) > null_threshold


def _compute_colormap_limits(
    tg_matrix: np.ndarray,
    metric: str,
) -> tuple[float, float, str]:
    """Compute colormap limits and colormap name for time-generalization matrix."""
    vmax = np.nanmax(np.abs(tg_matrix))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = DEFAULT_VMAX

    is_correlation = metric.lower() == CORRELATION_METRIC
    cmap = "RdBu_r" if is_correlation else "magma"
    vmin = -vmax if is_correlation else 0.0

    return vmin, vmax, cmap


def _add_zero_reference_lines(
    ax: Any,
    train_times: np.ndarray,
    test_times: np.ndarray,
    plot_cfg: Any,
) -> None:
    """Add horizontal and vertical zero reference lines if zero is within time range."""
    zero_line_style = {
        "color": plot_cfg.style.colors.black,
        "linestyle": "--",
        "linewidth": plot_cfg.style.line.width_thin,
        "alpha": ZERO_LINE_ALPHA,
    }

    train_contains_zero = train_times[0] <= 0 <= train_times[-1]
    if train_contains_zero:
        ax.axhline(0, **zero_line_style)

    test_contains_zero = test_times[0] <= 0 <= test_times[-1]
    if test_contains_zero:
        ax.axvline(0, **zero_line_style)


def _create_time_generalization_plot(
    tg_matrix: np.ndarray,
    window_centers: np.ndarray,
    metric: str,
    plot_cfg: Any,
    sig_mask: Optional[np.ndarray] = None,
) -> tuple[Any, Any]:
    """Create time-generalization matrix plot with optional significance contours."""
    fig_size = plot_cfg.get_figure_size("square", plot_type="machine_learning")
    train_times = np.asarray(window_centers, dtype=float)
    test_times = train_times

    vmin, vmax, cmap = _compute_colormap_limits(tg_matrix, metric)

    fig, ax = plt.subplots(figsize=fig_size)
    extent = [
        test_times[0],
        test_times[-1],
        train_times[0],
        train_times[-1],
    ]
    im = ax.imshow(
        tg_matrix,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Test time (s)")
    ax.set_ylabel("Train time (s)")

    if sig_mask is not None:
        test_grid, train_grid = np.meshgrid(test_times, train_times)
        ax.contour(
            test_grid,
            train_grid,
            sig_mask,
            levels=[CONTOUR_LEVEL],
            colors="k",
            linewidths=0.6,
        )

    _add_zero_reference_lines(ax, train_times, test_times, plot_cfg)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.upper())

    return fig, ax


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

    Parameters
    ----------
    tg_matrix : np.ndarray
        Time-generalization matrix of shape (n_train_windows, n_test_windows).
    null_matrix : Optional[np.ndarray]
        Null distribution matrix. Can be 2D (same shape as tg_matrix) or
        3D (n_permutations, n_train_windows, n_test_windows).
    window_centers : np.ndarray
        Window center times in seconds.
    save_path : Path
        Path to save the figure.
    metric : str, default="r"
        Metric name for colorbar label.
    config : Optional[Any], default=None
        Plot configuration object.
    """
    if not _validate_time_generalization_inputs(tg_matrix, window_centers):
        return

    sig_mask = _compute_significance_mask(tg_matrix, null_matrix)
    plot_cfg = get_plot_config(config)
    fig, ax = _create_time_generalization_plot(
        tg_matrix, window_centers, metric, plot_cfg, sig_mask
    )

    ax.set_title("Time-generalization (empirical vs null)")
    despine(ax)

    plt.tight_layout()
    save_fig(fig, save_path, formats=plot_cfg.formats, config=config)
    logger.info("Saved time-generalization (with null) to %s", save_path)

