from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import (
    get_behavior_footer as _get_behavior_footer,
    save_fig,
)


def _get_behavioral_config(plot_cfg: Any) -> dict:
    """Extract behavioral plotting configuration from plot config."""
    if plot_cfg is None:
        return {}
    return plot_cfg.plot_type_configs.get("behavioral", {})


def _create_rating_distribution_plot(
    y: pd.Series,
    config: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create histogram plot of rating distribution with statistics."""
    plot_cfg = get_plot_config(config)
    behavioral_config = _get_behavioral_config(plot_cfg)
    rating_config = behavioral_config.get("rating_distribution", {})

    figsize_width = rating_config.get("figsize_width", 8)
    figsize_height = rating_config.get("figsize_height", 6)
    bins = rating_config.get("bins", 20)
    alpha = rating_config.get("alpha", 0.7)
    edgecolor = rating_config.get("edgecolor", "black")
    grid_alpha = rating_config.get("grid_alpha", 0.3)
    text_box_alpha = rating_config.get("text_box_alpha", 0.8)

    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    ax.hist(y.dropna(), bins=bins, alpha=alpha, edgecolor=edgecolor)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    ax.set_title("Rating Distribution")
    ax.grid(True, alpha=grid_alpha)

    mean_rating = y.mean()
    std_rating = y.std()
    stats_text = f"Mean: {mean_rating:.2f} ± {std_rating:.2f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=text_box_alpha),
    )

    return fig, ax


def plot_behavioral_response_patterns(
    y: pd.Series,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot and save rating distribution for a subject."""
    if config is None:
        raise ValueError("config is required for behavioral plotting")

    plot_cfg = get_plot_config(config)
    fig, _ = _create_rating_distribution_plot(y, config)

    plt.tight_layout()
    output_path = save_dir / f"sub-{subject}_rating_distribution"
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        config=config,
    )
    plt.close(fig)
