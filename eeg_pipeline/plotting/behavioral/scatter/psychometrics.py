from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import PlotConfig, get_plot_config
from eeg_pipeline.plotting.behavioral.builders import generate_correlation_scatter
from eeg_pipeline.utils.data import _pick_first_column
from eeg_pipeline.infra.paths import deriv_plots_path, ensure_dir, _load_events_df
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.infra.logging import get_subject_logger


def _load_and_validate_psychometric_data(
    events: pd.DataFrame,
    temperature_columns: list[str],
    rating_columns: list[str],
    logger: logging.Logger,
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Load and validate temperature and rating data from events DataFrame."""
    temperature_column = _pick_first_column(events, temperature_columns)
    if temperature_column is None:
        return None, None

    rating_column = _pick_first_column(events, rating_columns)
    temperature = pd.to_numeric(events[temperature_column], errors="coerce")

    valid_mask = temperature.notna()
    if rating_column is not None:
        rating = pd.to_numeric(events[rating_column], errors="coerce")
        valid_mask = valid_mask & rating.notna()
    else:
        rating = None

    temperature_valid = temperature[valid_mask]
    rating_valid = rating[valid_mask] if rating is not None else None

    return temperature_valid, rating_valid


def _plot_temperature_rating_correlation(
    temperature: pd.Series,
    rating: pd.Series,
    subject: str,
    output_dir: Path,
    plot_config: PlotConfig,
    config,
    logger: logging.Logger,
) -> None:
    """Generate scatter plot of temperature vs rating with correlation statistics."""
    behavioral_config = plot_config.get_behavioral_config()
    rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = np.random.default_rng(rng_seed)

    output_path = output_dir / f"psychometrics_temp_vs_rating_sub-{subject}"

    generate_correlation_scatter(
        x_data=temperature,
        y_data=rating,
        x_label="Temperature (°C)",
        y_label="Rating",
        title_prefix=f"Psychometrics — Temperature vs Rating — sub-{subject}",
        band_color=get_band_color("alpha", config),
        output_path=output_path,
        rng=rng,
        logger=logger,
        config=config,
    )


def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    """Generate psychometric plots for temperature and rating data."""
    if config is None:
        raise ValueError("config is required for psychometrics plotting")

    logger = get_subject_logger("behavior_analysis", subject)
    plot_config = get_plot_config(config)
    behavioral_config = plot_config.get_behavioral_config()

    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    ensure_dir(plots_dir)

    events = _load_events_df(subject, task, config=config)
    if events is None or len(events) == 0:
        logger.warning(f"No events for psychometrics: sub-{subject}")
        return

    temperature_columns = config.get("event_columns.temperature", [])
    rating_columns = config.get("event_columns.rating", [])

    temperature_valid, rating_valid = _load_and_validate_psychometric_data(
        events,
        temperature_columns,
        rating_columns,
        logger,
    )

    if temperature_valid is None:
        logger.warning(
            f"Psychometrics: no temperature column found; skipping for sub-{subject}."
        )
        return

    n_valid = len(temperature_valid)
    min_samples_for_plot = plot_config.validation.get("min_samples_for_plot", 5)
    if n_valid < min_samples_for_plot:
        logger.warning(
            f"Insufficient valid data for psychometrics (n={n_valid} < {min_samples_for_plot}); "
            f"skipping for sub-{subject}"
        )
        return

    psychometrics_dir = plots_dir / "psychometrics"
    ensure_dir(psychometrics_dir)

    if rating_valid is not None:
        _plot_temperature_rating_correlation(
            temperature_valid,
            rating_valid,
            subject,
            psychometrics_dir,
            plot_config,
            config,
            logger,
        )

    logger.info(f"Completed psychometrics plotting for sub-{subject}")
