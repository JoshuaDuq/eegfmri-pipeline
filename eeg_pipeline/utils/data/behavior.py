from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config.loader import get_frequency_band_names
from eeg_pipeline.infra.paths import resolve_deriv_root
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.utils.data.features import get_power_columns_by_band
from eeg_pipeline.utils.data.feature_io import _load_features_and_targets
from .epochs import load_epochs_for_analysis


# Constants
ALIGN_MODE_STRICT = "strict"
NUMERIC_CONVERSION_ERRORS = "coerce"
TEMPERATURE_CONFIG_KEY = "event_columns.temperature"


def load_stats_file_with_fallbacks(
    stats_dir: Path,
    patterns: List[str],
) -> Optional[pd.DataFrame]:
    """Load first available stats file matching any of the given patterns."""
    for pattern in patterns:
        filepath = stats_dir / pattern
        if filepath.exists():
            df = read_tsv(filepath)
            if df is not None and not df.empty:
                return df
    return None


def _build_stats_file_patterns(
    base_name: str,
    method_label: Optional[str],
) -> List[str]:
    """Build file patterns with and without method suffix."""
    method_suffix = f"_{method_label}" if method_label else ""
    patterns_with_suffix = [
        f"corr_stats_pow_roi_vs_{base_name}{method_suffix}.tsv",
        f"corr_stats_power_roi_vs_{base_name}{method_suffix}.tsv",
    ]
    
    if method_label:
        patterns_without_suffix = [
            f"corr_stats_pow_roi_vs_{base_name}.tsv",
            f"corr_stats_power_roi_vs_{base_name}.tsv",
        ]
        return patterns_with_suffix + patterns_without_suffix
    
    return patterns_with_suffix


def load_behavior_stats_files(
    stats_dir: Path,
    logger: logging.Logger,
    method_label: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load rating and temperature correlation stats files with fallback patterns."""
    rating_patterns = _build_stats_file_patterns("rating", method_label)
    temp_patterns = _build_stats_file_patterns("temp", method_label)
    
    rating_stats = load_stats_file_with_fallbacks(stats_dir, rating_patterns)
    temp_stats = load_stats_file_with_fallbacks(stats_dir, temp_patterns)
    
    return rating_stats, temp_stats


def load_behavior_plot_features(
    subject: str,
    task: str,
    config,
    logger: logging.Logger,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame], Optional[List[str]]]:
    """Load power features, targets, and metadata for behavioral plotting."""
    effective_deriv_root = resolve_deriv_root(config=config)
    
    epochs, _aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align=ALIGN_MODE_STRICT,
        preload=False,
        deriv_root=effective_deriv_root,
        bids_root=config.bids_root,
        config=config,
        logger=logger,
    )

    _, power_df, _, targets, info = _load_features_and_targets(
        subject, task, effective_deriv_root, config, epochs=epochs
    )

    if power_df is None or targets is None:
        logger.error("Features or targets not found for sub-%s", subject)
        return None, None, None, None

    power_bands = get_frequency_band_names(config)
    return power_df, targets, info, power_bands


def _extract_temperature_from_aligned_events(
    aligned_events: pd.DataFrame,
    config,
) -> Optional[pd.Series]:
    """Extract temperature column from aligned events dataframe."""
    if aligned_events is None:
        return None
    
    temperature_columns = config.get(TEMPERATURE_CONFIG_KEY)
    if not temperature_columns:
        return None
    
    temperature_column = next(
        (col for col in temperature_columns if col in aligned_events.columns),
        None
    )
    
    if temperature_column is None:
        return None
    
    return pd.to_numeric(
        aligned_events[temperature_column],
        errors=NUMERIC_CONVERSION_ERRORS
    )


def _compute_band_power_mean(
    power_df: pd.DataFrame,
    power_columns: List[str],
) -> pd.Series:
    """Compute mean power across specified columns for each row."""
    return (
        power_df[power_columns]
        .apply(pd.to_numeric, errors=NUMERIC_CONVERSION_ERRORS)
        .mean(axis=1)
    )


def _process_subject_band_data(
    power_df: pd.DataFrame,
    ratings: pd.Series,
    temperature: Optional[pd.Series],
    power_bands: List[str],
    power_cols_by_band: Dict[str, List[str]],
) -> Tuple[Dict, Dict, Dict, Dict, bool]:
    """Process power and behavioral data for a single subject across frequency bands."""
    rating_x = {}
    rating_y = {}
    temp_x = {}
    temp_y = {}
    has_temperature = temperature is not None

    for band in power_bands:
        band_str = str(band)
        power_columns = (
            power_cols_by_band.get(band_str)
            or power_cols_by_band.get(band_str.lower())
        )
        
        if not power_columns:
            continue

        band_power = _compute_band_power_mean(power_df, power_columns).to_numpy()
        ratings_array = ratings.to_numpy()
        
        rating_x.setdefault(band, []).append(band_power)
        rating_y.setdefault(band, []).append(ratings_array)

        if has_temperature:
            temperature_array = temperature.to_numpy()
            temp_x.setdefault(band, []).append(band_power)
            temp_y.setdefault(band, []).append(temperature_array)

    return rating_x, rating_y, temp_x, temp_y, has_temperature


def _load_subject_features_and_epochs(
    subject: str,
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Load features, targets, and temperature for a single subject."""
    _, power_df, _, ratings, _ = _load_features_and_targets(
        subject, task, deriv_root, config
    )
    ratings = pd.to_numeric(ratings, errors=NUMERIC_CONVERSION_ERRORS)

    epochs, aligned_events = load_epochs_for_analysis(
        subject,
        task,
        align=ALIGN_MODE_STRICT,
        preload=False,
        deriv_root=deriv_root,
        bids_root=config.bids_root,
        config=config,
    )
    
    if epochs is None:
        raise ValueError(f"Failed to load epochs for subject {subject}")

    temperature = _extract_temperature_from_aligned_events(aligned_events, config)
    
    return power_df, ratings, temperature


def load_subject_data_for_summary(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict, Dict, Dict, Dict, bool]:
    """Load and aggregate power and behavioral data across subjects for summary plots."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if not subjects:
        return {}, {}, {}, {}, False

    aggregated_rating_x = {}
    aggregated_rating_y = {}
    aggregated_temp_x = {}
    aggregated_temp_y = {}
    has_temperature = False

    power_bands = get_frequency_band_names(config)
    band_strings = [str(band) for band in power_bands]

    for subject in subjects:
        try:
            power_df, ratings, temperature = _load_subject_features_and_epochs(
                subject, task, deriv_root, config, logger
            )
            
            if temperature is not None:
                has_temperature = True

            power_cols_by_band = get_power_columns_by_band(
                power_df,
                bands=band_strings
            )
            
            subject_rating_x, subject_rating_y, subject_temp_x, subject_temp_y, _ = (
                _process_subject_band_data(
                    power_df,
                    ratings,
                    temperature,
                    power_bands,
                    power_cols_by_band,
                )
            )

            for band in power_bands:
                if band in subject_rating_x:
                    aggregated_rating_x.setdefault(band, []).extend(subject_rating_x[band])
                    aggregated_rating_y.setdefault(band, []).extend(subject_rating_y[band])
                
                if band in subject_temp_x:
                    aggregated_temp_x.setdefault(band, []).extend(subject_temp_x[band])
                    aggregated_temp_y.setdefault(band, []).extend(subject_temp_y[band])

        except (FileNotFoundError, ValueError) as e:
            logger.debug("Skipping subject %s due to expected error: %s", subject, e)
            continue
        except Exception as e:
            logger.error(
                "Unexpected error loading data for subject %s: %s",
                subject,
                e,
                exc_info=True
            )
            raise

    return (
        aggregated_rating_x,
        aggregated_rating_y,
        aggregated_temp_x,
        aggregated_temp_y,
        has_temperature,
    )


__all__ = [
    "load_behavior_plot_features",
    "load_stats_file_with_fallbacks",
    "load_behavior_stats_files",
    "load_subject_data_for_summary",
]
