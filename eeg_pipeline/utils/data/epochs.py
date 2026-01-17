from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mne
import pandas as pd

from ..config.loader import ConfigDict
from eeg_pipeline.infra.paths import find_clean_epochs_path, load_events_df
from .alignment import align_events_to_epochs, validate_alignment

EEGConfig = ConfigDict


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
) -> list[str]:
    missing_columns = []
    for logical_name, candidates in event_cols_config.items():
        if not isinstance(candidates, (list, tuple)):
            continue
        found = any(col in events_df.columns for col in candidates)
        if not found:
            missing_columns.append(
                f"event_columns.{logical_name} (tried: {candidates})"
            )
    return missing_columns


def _validate_align_mode(align: str) -> None:
    valid_align_modes = ("strict", "warn", "none")
    if align not in valid_align_modes:
        raise ValueError(
            f"align must be one of {valid_align_modes}, got '{align}'"
        )


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

    if config is None:
        raise ValueError("config is required for load_epochs_for_analysis")

    _validate_align_mode(align)
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
        return _handle_alignment_failure(
            epochs, events_df, align, subject, task, alignment_config
        )

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


__all__ = [
    "load_epochs_for_analysis",
]
