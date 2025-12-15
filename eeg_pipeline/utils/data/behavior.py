from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config.loader import get_frequency_band_names
from eeg_pipeline.io.paths import resolve_deriv_root
from eeg_pipeline.io.columns import pick_target_column
from eeg_pipeline.io.tsv import read_tsv
from .epochs_loading import load_epochs_for_analysis


def load_behavior_plot_features(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
) -> tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame], Optional[List[str]]]:
    effective_deriv_root = resolve_deriv_root(config=config)
    epochs, _aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=False,
        deriv_root=effective_deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )

    from .loading import _load_features_and_targets

    _, pow_df, _, y, info = _load_features_and_targets(
        subject, task, effective_deriv_root, config, epochs=epochs
    )

    if pow_df is None or y is None:
        logger.error("Features or targets not found for sub-%s", subject)
        return None, None, None, None

    power_bands = get_frequency_band_names(config)
    return pow_df, y, info, power_bands


def load_stats_file_with_fallbacks(
    stats_dir: Path,
    patterns: List[str],
) -> Optional[pd.DataFrame]:
    for pattern in patterns:
        filepath = stats_dir / pattern
        if filepath.exists():
            df = read_tsv(filepath)
            if df is not None and not df.empty:
                return df
    return None


def load_behavior_stats_files(
    stats_dir: Path,
    logger: logging.Logger,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    rating_stats = None
    temp_stats = None

    rating_candidates = [
        stats_dir / "corr_stats_pow_roi_vs_rating.tsv",
        stats_dir / "corr_stats_power_roi_vs_rating.tsv",
    ]
    temp_candidates = [
        stats_dir / "corr_stats_pow_roi_vs_temp.tsv",
        stats_dir / "corr_stats_power_roi_vs_temp.tsv",
    ]

    rating_path = next((p for p in rating_candidates if p.exists()), None)
    temp_path = next((p for p in temp_candidates if p.exists()), None)

    if rating_path is not None:
        try:
            rating_stats = read_tsv(rating_path)
        except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
            logger.warning("Failed to read rating stats: %s", exc)

    if temp_path is not None:
        try:
            temp_stats = read_tsv(temp_path)
        except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
            logger.warning("Failed to read temp stats: %s", exc)

    return rating_stats, temp_stats


def load_subject_data_for_summary(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict, Dict, Dict, Dict, bool]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if not subjects:
        return {}, {}, {}, {}, False

    rating_x = {}
    rating_y = {}
    temp_x = {}
    temp_y = {}
    has_temperature = False

    for subject in subjects:
        try:
            from .loading import _load_features_and_targets

            _temporal_df, power_df, _conn_df, ratings, _info = _load_features_and_targets(
                subject, task, deriv_root, config
            )
            ratings = pd.to_numeric(ratings, errors="coerce")

            epochs, aligned = load_epochs_for_analysis(
                subject,
                task,
                align="strict",
                preload=False,
                deriv_root=deriv_root,
                bids_root=config.bids_root,
                config=config,
            )
            if epochs is None:
                continue
        except (FileNotFoundError, ValueError) as e:
            logger.debug(f"Skipping subject {subject} due to expected error: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error loading data for subject {subject}: {e}", exc_info=True)
            raise

        temperature = None
        if aligned is not None:
            temp_columns = config.get("event_columns.temperature")
            temp_col = next((c for c in temp_columns if c in aligned.columns), None)
            if temp_col:
                temperature = pd.to_numeric(aligned[temp_col], errors="coerce")
                has_temperature = True

        power_bands = get_frequency_band_names(config)
        for band in power_bands:
            band_str = str(band)
            power_cols = [
                col
                for col in power_df.columns
                if str(col).startswith(f"power_plateau_{band_str}_ch_")
            ]
            if not power_cols:
                power_cols = [col for col in power_df.columns if str(col).startswith(f"pow_{band_str}_")]
            if not power_cols:
                continue

            band_power = (
                power_df[power_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
            )
            rating_x.setdefault(band, []).append(band_power)
            rating_y.setdefault(band, []).append(ratings.to_numpy())

            if temperature is not None:
                temp_x.setdefault(band, []).append(band_power)
                temp_y.setdefault(band, []).append(temperature.to_numpy())

    return rating_x, rating_y, temp_x, temp_y, has_temperature


__all__ = [
    "load_behavior_plot_features",
    "load_stats_file_with_fallbacks",
    "load_behavior_stats_files",
    "load_subject_data_for_summary",
]
