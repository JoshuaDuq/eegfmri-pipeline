from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from ..config.loader import ConfigDict
from eeg_pipeline.infra.paths import find_clean_epochs_path, load_events_df
from .alignment import align_events_to_epochs, validate_alignment

EEGConfig = ConfigDict


###################################################################
# Epoch Processing Utilities
###################################################################

def apply_baseline(
    epochs: mne.Epochs,
    baseline_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    config: Optional[EEGConfig] = None,
) -> mne.Epochs:
    from ..analysis.tfr import validate_baseline_indices

    times = np.asarray(epochs.times)
    validate_baseline_indices(times, baseline_window, logger=logger, config=config)
    baseline_start = float(baseline_window[0])
    baseline_end = float(baseline_window[1])
    baseline_end_clamped = min(baseline_end, 0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return epochs.copy().apply_baseline((baseline_start, baseline_end_clamped))


def crop_epochs(
    epochs: mne.Epochs,
    crop_tmin: Optional[float],
    crop_tmax: Optional[float],
    include_tmax: bool,
    logger: Optional[logging.Logger] = None,
) -> mne.Epochs:
    if crop_tmin is None and crop_tmax is None:
        return epochs

    time_min = epochs.tmin if crop_tmin is None else float(crop_tmin)
    time_max = epochs.tmax if crop_tmax is None else float(crop_tmax)

    if time_max <= time_min:
        raise ValueError(f"Invalid crop window: tmin={time_min}, tmax={time_max}")

    epochs_copy = epochs.copy()
    is_preloaded = getattr(epochs_copy, "preload", False)
    if not is_preloaded:
        epochs_copy.load_data()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return epochs_copy.crop(tmin=time_min, tmax=time_max, include_tmax=include_tmax)


def process_temperature_levels(
    epochs: mne.Epochs,
    temperature_column: str,
) -> Tuple[List, Dict, bool]:
    temperature_series = epochs.metadata[temperature_column]
    numeric_values = pd.to_numeric(temperature_series, errors="coerce")
    all_numeric = numeric_values.notna().all()

    if all_numeric:
        levels = np.sort(numeric_values.unique())
        labels = {
            value: str(int(value)) if float(value).is_integer() else str(value)
            for value in levels
        }
        return levels, labels, True

    levels = sorted(temperature_series.astype(str).unique())
    return levels, {}, False


def build_epoch_query_string(
    column: str,
    level: Any,
    is_numeric: bool,
    labels: Optional[Dict[Any, str]] = None,
) -> Tuple[str, str]:
    if labels is None:
        labels = {}

    if is_numeric:
        value = _convert_to_numeric_value(level)
        label = _get_numeric_label(level, value, labels)
        query = f"{column} == {value}"
        return query, label

    value_str = str(level)
    label = labels.get(level, value_str)
    query = f"{column} == '{value_str}'"
    return query, str(label)


def _convert_to_numeric_value(level: Any) -> float:
    try:
        return float(level)
    except (TypeError, ValueError):
        return level


def _get_numeric_label(level: Any, value: Any, labels: Dict[Any, str]) -> str:
    label = labels.get(level)
    if label is not None:
        return str(label)
    try:
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    except (TypeError, ValueError):
        return str(level)


def select_epochs_by_value(epochs: mne.Epochs, column: str, value: Any) -> mne.Epochs:
    if epochs.metadata is None or column not in epochs.metadata.columns:
        raise ValueError(f"Column '{column}' not found in epochs.metadata")

    value_expression = value if isinstance(value, (int, float)) else f"'{value}'"
    query_string = f"{column} == {value_expression}"
    selected = epochs[query_string]
    
    if len(selected) == 0:
        available_values = sorted(epochs.metadata[column].unique().tolist())
        raise ValueError(
            f"No epochs found for {column} == {value}. "
            f"Available values: {available_values}"
        )
    return selected


###################################################################
# Epoch Loading and Alignment
###################################################################

def _validate_event_columns(
    events_df: pd.DataFrame, config: EEGConfig, logger: logging.Logger
) -> None:
    if events_df is None or events_df.empty:
        return

    event_cols_config = config.get("event_columns", {})
    if not event_cols_config:
        logger.warning("No event_columns found in config; skipping validation")
        return

    missing_columns = _find_missing_event_columns(events_df, event_cols_config)
    if missing_columns:
        available = list(events_df.columns)
        error_msg = (
            f"Required event columns not found in events DataFrame. "
            f"Missing: {', '.join(missing_columns)}. "
            f"Available columns: {available}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _find_missing_event_columns(
    events_df: pd.DataFrame, event_cols_config: Dict[str, Any]
) -> List[str]:
    missing_columns = []
    for logical_name, candidates in event_cols_config.items():
        if not isinstance(candidates, (list, tuple)):
            continue
        found = any(col in events_df.columns for col in candidates)
        if not found:
            missing_columns.append(f"event_columns.{logical_name} (tried: {candidates})")
    return missing_columns


def _validate_load_epochs_params(align: str, config: Optional[Any]) -> None:
    valid_align_modes = ("strict", "warn", "none")
    if align not in valid_align_modes:
        raise ValueError(
            f"align must be one of {valid_align_modes}, got '{align}'"
        )
    if config is None:
        raise ValueError("config is required for load_epochs_for_analysis")


def _load_epochs_and_events(
    subject: str,
    task: str,
    deriv_root: Optional[Path],
    bids_root: Optional[Path],
    preload: bool,
    config: EEGConfig,
    constants: Optional[Any],
    logger: logging.Logger,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    epochs_path = find_clean_epochs_path(
        subject, task, deriv_root=deriv_root, config=config, constants=constants
    )
    if epochs_path is None or not epochs_path.exists():
        logger.error(
            f"Could not find cleaned epochs file for sub-{subject}, task-{task}"
        )
        return None, None

    logger.info(f"Loading epochs: {epochs_path}")
    epochs = mne.read_epochs(epochs_path, preload=preload, verbose=False)

    events_df = load_events_df(
        subject,
        task,
        bids_root=bids_root,
        config=config,
        constants=constants,
    )
    return epochs, events_df


def _handle_alignment_mismatch(
    aligned_events: pd.DataFrame,
    epochs: mne.Epochs,
    subject: str,
    task: str,
    allow_trim: bool,
    min_alignment_samples: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    n_events = len(aligned_events)
    n_epochs = len(epochs)

    if n_events == n_epochs:
        return aligned_events

    mismatch_count = abs(n_events - n_epochs)
    can_trim = allow_trim and mismatch_count <= min_alignment_samples

    if can_trim:
        return _trim_aligned_events(
            aligned_events, n_events, n_epochs, subject, task, mismatch_count, logger
        )

    _raise_alignment_mismatch_error(
        n_events, n_epochs, mismatch_count, allow_trim, min_alignment_samples, subject, task, logger
    )
    return aligned_events


def _trim_aligned_events(
    aligned_events: pd.DataFrame,
    n_events: int,
    n_epochs: int,
    subject: str,
    task: str,
    mismatch_count: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    from eeg_pipeline.infra.logging import log_and_raise_error

    logger.warning(
        f"Alignment length mismatch for sub-{subject}, task-{task}: "
        f"events={n_events}, epochs={n_epochs}, diff={mismatch_count}. "
        f"Trimming enabled."
    )
    if n_events > n_epochs:
        return aligned_events.iloc[:n_epochs].reset_index(drop=True)

    error_msg = (
        f"Alignment length mismatch for sub-{subject}, task-{task}: "
        f"events={n_events}, epochs={n_epochs}, diff={mismatch_count}. "
        f"Cannot trim when events < epochs."
    )
    log_and_raise_error(logger, error_msg)
    return aligned_events


def _raise_alignment_mismatch_error(
    n_events: int,
    n_epochs: int,
    mismatch_count: int,
    allow_trim: bool,
    min_alignment_samples: int,
    subject: str,
    task: str,
    logger: logging.Logger,
) -> None:
    from eeg_pipeline.infra.logging import log_and_raise_error

    if not allow_trim:
        reason = "allow_misaligned_trim=False"
    else:
        reason = (
            f"mismatch ({mismatch_count}) exceeds "
            f"max_tolerable_mismatch ({min_alignment_samples})"
        )
    error_msg = (
        f"Alignment length mismatch for sub-{subject}, task-{task}: "
        f"events={n_events}, epochs={n_epochs}, diff={mismatch_count}. "
        f"Cannot proceed: {reason}"
    )
    log_and_raise_error(logger, error_msg)


def load_epochs_for_analysis(
    subject: str,
    task: str,
    align: str = "strict",
    preload: bool = False,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[EEGConfig] = None,
    constants: Optional[Any] = None,
    use_cache: bool = True,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    if logger is None:
        logger = logging.getLogger(__name__)

    _validate_load_epochs_params(align, config)
    if config is None:
        raise ValueError("config is required")

    alignment_config = _extract_alignment_config(config)
    epochs, events_df = _load_epochs_and_events(
        subject, task, deriv_root, bids_root, preload, config, constants, logger
    )
    if epochs is None:
        return None, None

    if events_df is None:
        return _handle_missing_events(epochs, align, subject, task, logger)

    logger.info(f"Loaded events: {len(events_df)} rows")
    logger.debug(
        f"Alignment parameters: allow_misaligned_trim={alignment_config['allow_trim']}, "
        f"min_alignment_samples={alignment_config['min_samples']}"
    )

    _validate_event_columns(events_df, config, logger)
    aligned_events = _attempt_alignment(
        events_df, epochs, align, subject, task, alignment_config, logger
    )

    if aligned_events is None:
        return _handle_alignment_failure(epochs, events_df, align, subject, task, alignment_config)

    aligned_events = _handle_alignment_mismatch(
        aligned_events,
        epochs,
        subject,
        task,
        alignment_config["allow_trim"],
        alignment_config["min_samples"],
        logger,
    )

    _verify_final_alignment_length(aligned_events, epochs, subject, task, logger)
    validation_passed = validate_alignment(
        aligned_events, epochs, logger, strict=(align == "strict")
    )

    if not validation_passed:
        return _handle_validation_failure(epochs, align, subject, task, logger)

    if use_cache:
        epochs._behavioral = aligned_events  # type: ignore[attr-defined]
    return epochs, aligned_events


def _extract_alignment_config(config: EEGConfig) -> Dict[str, Any]:
    return {
        "allow_trim": bool(config.get("alignment.allow_misaligned_trim")),
        "min_samples": int(config.get("alignment.min_alignment_samples")),
    }


def _handle_missing_events(
    epochs: mne.Epochs,
    align: str,
    subject: str,
    task: str,
    logger: logging.Logger,
) -> Tuple[mne.Epochs, None]:
    if align == "strict":
        raise ValueError(
            f"Events TSV not found for sub-{subject}, task-{task}. "
            f"Required when align='strict'"
        )
    logger.warning("Events TSV not found; metadata will not be set.")
    return epochs, None


def _attempt_alignment(
    events_df: pd.DataFrame,
    epochs: mne.Epochs,
    align: str,
    subject: str,
    task: str,
    alignment_config: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    try:
        return align_events_to_epochs(events_df, epochs, logger=logger)
    except ValueError as err:
        if align == "strict":
            raise ValueError(
                f"Alignment failed for sub-{subject}, task-{task} in strict mode: {err}. "
                f"Alignment parameters: allow_misaligned_trim={alignment_config['allow_trim']}, "
                f"min_alignment_samples={alignment_config['min_samples']}"
            ) from err
        if align == "warn":
            logger.warning(f"Alignment failed: {err}")
        return None


def _handle_alignment_failure(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    align: str,
    subject: str,
    task: str,
    alignment_config: Dict[str, Any],
) -> Tuple[mne.Epochs, None]:
    if align == "strict":
        raise ValueError(
            f"Could not align events to epochs for sub-{subject}, task-{task}. "
            f"Events have {len(events_df)} rows, epochs have {len(epochs)} epochs. "
            f"Alignment parameters: allow_misaligned_trim={alignment_config['allow_trim']}, "
            f"min_alignment_samples={alignment_config['min_samples']}. "
            f"This is required when align='strict'."
        )
    return epochs, None


def _verify_final_alignment_length(
    aligned_events: pd.DataFrame,
    epochs: mne.Epochs,
    subject: str,
    task: str,
    logger: logging.Logger,
) -> None:
    if len(aligned_events) != len(epochs):
        error_msg = (
            f"Failed to align events to epochs for sub-{subject}, task-{task}: "
            f"final length mismatch (events={len(aligned_events)}, epochs={len(epochs)}). "
            f"Cannot guarantee alignment."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _handle_validation_failure(
    epochs: mne.Epochs,
    align: str,
    subject: str,
    task: str,
    logger: logging.Logger,
) -> Tuple[mne.Epochs, None]:
    if align == "strict":
        raise ValueError(
            f"Alignment validation failed for sub-{subject}, task-{task} with strict mode."
        )
    logger.warning("Alignment validation failed; returning None for aligned events")
    return epochs, None


def load_epochs_with_aligned_events(
    subject: str,
    task: str,
    config: EEGConfig,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    preload: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    epochs, aligned = load_epochs_for_analysis(
        subject,
        task,
        align="strict",
        preload=preload,
        deriv_root=deriv_root,
        bids_root=bids_root,
        logger=logger,
        config=config,
    )
    if epochs is None or aligned is None:
        return None, None
    return epochs, aligned


###################################################################
# Column Resolution Utilities
###################################################################

def pick_event_columns(
    df: pd.DataFrame, config: EEGConfig
) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {
        "rating": None,
        "temperature": None,
        "pain_binary": None,
    }
    column_mappings = {
        "rating": config.get("event_columns.rating"),
        "temperature": config.get("event_columns.temperature"),
        "pain_binary": config.get("event_columns.pain_binary"),
    }

    for key, candidate_columns in column_mappings.items():
        result[key] = _find_first_matching_column(df, candidate_columns)

    return result


def _find_first_matching_column(
    df: pd.DataFrame, candidate_columns: Any
) -> Optional[str]:
    """Find first matching column from candidates.
    
    Uses the canonical find_column function from data.manipulation.
    """
    if not isinstance(candidate_columns, (list, tuple)):
        return None
    from eeg_pipeline.utils.data.manipulation import find_column
    return find_column(df, list(candidate_columns))


def resolve_columns(
    df: pd.DataFrame,
    config: Optional[EEGConfig] = None,
    deriv_root: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if config is None:
        if deriv_root is None:
            raise ValueError("Either config or deriv_root must be provided to resolve_columns")
        from ..config.loader import load_config

        config = load_config()

    cols = pick_event_columns(df, config)
    return cols["pain_binary"], cols["temperature"], cols["rating"]


__all__ = [
    "apply_baseline",
    "crop_epochs",
    "process_temperature_levels",
    "build_epoch_query_string",
    "select_epochs_by_value",
    "load_epochs_for_analysis",
    "load_epochs_with_aligned_events",
    "pick_event_columns",
    "resolve_columns",
]
