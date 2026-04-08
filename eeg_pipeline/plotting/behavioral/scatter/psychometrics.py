from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import PlotConfig, get_plot_config
from eeg_pipeline.plotting.behavioral.builders import generate_correlation_scatter
from eeg_pipeline.utils.data.manipulation import find_column
from eeg_pipeline.infra.paths import deriv_plots_path, ensure_dir, _load_events_df
from eeg_pipeline.utils.analysis.stats.validation import (
    assert_predictor_type_continuous,
    assert_continuous_predictor,
)
from eeg_pipeline.plotting.io.figures import get_band_color
from eeg_pipeline.infra.logging import get_subject_logger


def _load_and_validate_psychometric_data(
    events: pd.DataFrame,
    predictor_column: str,
    outcome_column: str,
    logger: logging.Logger,
) -> tuple[Optional[pd.Series], Optional[pd.Series], int]:
    """Load and validate predictor and rating data from events DataFrame."""
    if predictor_column not in events.columns:
        return None, None, 0
    predictor = pd.to_numeric(events[predictor_column], errors="coerce")

    valid_mask = predictor.notna()
    if outcome_column in events.columns:
        rating = pd.to_numeric(events[outcome_column], errors="coerce")
        valid_mask = valid_mask & rating.notna()
    else:
        rating = None

    predictor_valid = predictor[valid_mask]
    rating_valid = rating[valid_mask] if rating is not None else None

    return predictor_valid, rating_valid, int(valid_mask.sum())


def _resolve_psychometric_columns(
    events: pd.DataFrame,
    config,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve psychometrics columns with plot-specific overrides first."""
    psychometrics_config = config.get("plotting.plots.behavior.psychometrics", {}) or {}

    predictor_override = str(psychometrics_config.get("predictor_column") or "").strip()
    outcome_override = str(psychometrics_config.get("outcome_column") or "").strip()

    predictor_candidates = [predictor_override] if predictor_override else list(
        config.get("event_columns.predictor", []) or []
    )
    outcome_candidates = [outcome_override] if outcome_override else list(
        config.get("event_columns.outcome", []) or []
    )

    predictor_column = find_column(events, predictor_candidates) if predictor_candidates else None
    outcome_column = find_column(events, outcome_candidates) if outcome_candidates else None
    return predictor_column, outcome_column


def _plot_predictor_rating_correlation(
    predictor: pd.Series,
    rating: pd.Series,
    subject: str,
    output_dir: Path,
    plot_config: PlotConfig,
    config,
    logger: logging.Logger,
    predictor_label: str,
    outcome_label: str,
) -> None:
    """Generate scatter plot of predictor vs rating with correlation statistics."""
    behavioral_config = plot_config.get_behavioral_config()
    rng_seed = behavioral_config.get("default_rng_seed", 42)
    rng = np.random.default_rng(rng_seed)

    safe_predictor = predictor_label.lower().replace(" ", "_")
    safe_outcome = outcome_label.lower().replace(" ", "_")
    output_path = output_dir / f"psychometrics_{safe_predictor}_vs_{safe_outcome}_sub-{subject}"

    generate_correlation_scatter(
        x_data=predictor,
        y_data=rating,
        x_label=predictor_label,
        y_label=outcome_label,
        title_prefix=f"Psychometrics: {predictor_label} vs {outcome_label} - sub-{subject}",
        band_color=get_band_color("alpha", config),
        output_path=output_path,
        rng=rng,
        logger=logger,
        config=config,
    )


def plot_psychometrics(subject: str, deriv_root: Path, task: str, config) -> None:
    """Generate psychometric plots for predictor vs. outcome.

    Psychometric plots assume a continuous physical predictor on an ordered
    scale (e.g., stimulus intensity). They are not meaningful for binary or
    categorical predictors.

    Raises
    ------
    ValueError
        If predictor_type is not 'continuous' or has < 5 unique values.
    """
    if config is None:
        raise ValueError("config is required for psychometrics plotting")

    logger = get_subject_logger("behavior_analysis", subject)
    plot_config = get_plot_config(config)
    behavioral_config = plot_config.get_behavioral_config()

    plot_subdir = behavioral_config.get("plot_subdir", "behavior")
    plots_dir = deriv_plots_path(deriv_root, subject, subdir=plot_subdir)
    ensure_dir(plots_dir)

    assert_predictor_type_continuous(config, context="psychometrics")

    events = _load_events_df(subject, task, config=config)
    if events is None or len(events) == 0:
        logger.warning(f"No events for psychometrics: sub-{subject}")
        return

    predictor_column, outcome_column = _resolve_psychometric_columns(events, config)

    if predictor_column is None:
        logger.warning(
            f"Psychometrics: no predictor column found; skipping for sub-{subject}."
        )
        return
    if outcome_column is None:
        logger.warning(
            f"Psychometrics: no outcome column found; skipping for sub-{subject}."
        )
        return

    predictor_valid, rating_valid, n_valid = _load_and_validate_psychometric_data(
        events,
        predictor_column,
        outcome_column,
        logger,
    )

    if predictor_valid is None:
        logger.warning(
            f"Psychometrics: no predictor column found; skipping for sub-{subject}."
        )
        return

    assert_continuous_predictor(predictor_valid, config, context="psychometrics")

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
        _plot_predictor_rating_correlation(
            predictor_valid,
            rating_valid,
            subject,
            psychometrics_dir,
            plot_config,
            config,
            logger,
            predictor_column,
            outcome_column,
        )

    logger.info(f"Completed psychometrics plotting for sub-{subject}")

