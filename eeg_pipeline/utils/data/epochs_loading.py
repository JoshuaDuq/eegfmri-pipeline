from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mne
import pandas as pd

from ..config.loader import ConfigDict
from eeg_pipeline.io.paths import _find_clean_epochs_path, _load_events_df
from .alignment import align_events_to_epochs, validate_alignment


EEGConfig = ConfigDict


def _validate_event_columns(events_df: pd.DataFrame, config: EEGConfig, logger: logging.Logger) -> None:
    if events_df is None or events_df.empty:
        return

    event_cols_config = config.get("event_columns", {})
    if not event_cols_config:
        logger.warning("No event_columns found in config; skipping validation")
        return

    missing_columns = []
    for logical_name, candidates in event_cols_config.items():
        if not isinstance(candidates, (list, tuple)):
            continue
        found = any(col in events_df.columns for col in candidates)
        if not found:
            missing_columns.append(f"event_columns.{logical_name} (tried: {candidates})")

    if missing_columns:
        available = list(events_df.columns)
        error_msg = (
            f"Required event columns not found in events DataFrame. "
            f"Missing: {', '.join(missing_columns)}. "
            f"Available columns: {available}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _validate_load_epochs_params(align: str, config: Any) -> None:
    if align not in ("strict", "warn", "none"):
        raise ValueError(f"align must be one of 'strict', 'warn', 'none', got '{align}'")
    if config is None:
        raise ValueError("config is required for load_epochs_for_analysis")


def _load_epochs_and_events(
    subject: str,
    task: str,
    deriv_root: Optional[Path],
    bids_root: Optional[Path],
    preload: bool,
    config,
    constants,
    logger: logging.Logger,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    epochs_path = _find_clean_epochs_path(
        subject, task, deriv_root=deriv_root, config=config, constants=constants
    )
    if epochs_path is None or not epochs_path.exists():
        logger.error(f"Could not find cleaned epochs file for sub-{subject}, task-{task}")
        return None, None

    logger.info(f"Loading epochs: {epochs_path}")
    epochs = mne.read_epochs(epochs_path, preload=preload, verbose=False)

    events_df = _load_events_df(
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

    diff = abs(n_events - n_epochs)

    if allow_trim and diff <= min_alignment_samples:
        logger.warning(
            f"Alignment length mismatch for sub-{subject}, task-{task}: "
            f"events={n_events}, epochs={n_epochs}, diff={diff}. "
            f"Trimming enabled (max_tolerable_mismatch={min_alignment_samples})."
        )
        if n_events > n_epochs:
            return aligned_events.iloc[:n_epochs].reset_index(drop=True)
        else:
            from eeg_pipeline.io.logging import log_and_raise_error

            error_msg = (
                f"Alignment length mismatch for sub-{subject}, task-{task}: "
                f"events={n_events}, epochs={n_epochs}, diff={diff}. "
                f"Cannot trim when events < epochs."
            )
            log_and_raise_error(logger, error_msg)
    else:
        from eeg_pipeline.io.logging import log_and_raise_error

        reason = (
            "allow_misaligned_trim=False"
            if not allow_trim
            else f"mismatch ({diff}) exceeds max_tolerable_mismatch ({min_alignment_samples})"
        )
        error_msg = (
            f"Alignment length mismatch for sub-{subject}, task-{task}: "
            f"events={n_events}, epochs={n_epochs}, diff={diff}. "
            f"Cannot proceed: {reason}"
        )
        log_and_raise_error(logger, error_msg)

    return aligned_events


def load_epochs_for_analysis(
    subject: str,
    task: str,
    align: str = "strict",
    preload: bool = False,
    deriv_root: Optional[Path] = None,
    bids_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
    constants=None,
    use_cache: bool = True,
) -> Tuple[Optional[mne.Epochs], Optional[pd.DataFrame]]:
    if logger is None:
        logger = logging.getLogger(__name__)

    _validate_load_epochs_params(align, config)

    allow_trim = bool(config.get("alignment.allow_misaligned_trim"))
    min_alignment_samples = int(config.get("alignment.min_alignment_samples"))

    epochs, events_df = _load_epochs_and_events(
        subject, task, deriv_root, bids_root, preload, config, constants, logger
    )
    if epochs is None:
        return None, None

    if events_df is None:
        if align == "strict":
            raise ValueError(
                f"Events TSV not found for sub-{subject}, task-{task}. Required when align='strict'"
            )
        logger.warning("Events TSV not found; metadata will not be set.")
        return epochs, None

    logger.info(f"Loaded events: {len(events_df)} rows")
    logger.debug(
        f"Alignment parameters: allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}"
    )

    _validate_event_columns(events_df, config, logger)

    try:
        aligned_events = align_events_to_epochs(events_df, epochs, logger=logger)
    except ValueError as err:
        if align == "strict":
            raise ValueError(
                f"Alignment failed for sub-{subject}, task-{task} in strict mode: {err}. "
                f"Alignment parameters: allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}"
            ) from err
        if align == "warn":
            logger.warning(f"Alignment failed: {err}")
        aligned_events = None

    if aligned_events is None:
        if align == "strict":
            raise ValueError(
                f"Could not align events to epochs for sub-{subject}, task-{task}. "
                f"Events have {len(events_df)} rows, epochs have {len(epochs)} epochs. "
                f"Alignment parameters: allow_misaligned_trim={allow_trim}, min_alignment_samples={min_alignment_samples}. "
                f"This is required when align='strict'."
            )
        return epochs, None

    aligned_events = _handle_alignment_mismatch(
        aligned_events, epochs, subject, task, allow_trim, min_alignment_samples, logger
    )

    if len(aligned_events) != len(epochs):
        error_msg = (
            f"Failed to align events to epochs for sub-{subject}, task-{task}: "
            f"final length mismatch (events={len(aligned_events)}, epochs={len(epochs)}). "
            f"Cannot guarantee alignment."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    validation_result = validate_alignment(
        aligned_events, epochs, logger, strict=(align == "strict"), config=config
    )
    if not validation_result:
        if align == "strict":
            raise ValueError(
                f"Alignment validation failed for sub-{subject}, task-{task} with strict mode."
            )
        logger.warning("Alignment validation failed; returning None for aligned events")
        return epochs, None

    if use_cache:
        epochs._behavioral = aligned_events  # type: ignore[attr-defined]
    return epochs, aligned_events


def load_epochs_with_aligned_events(
    subject: str,
    task: str,
    config,
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


def pick_event_columns(df: pd.DataFrame, config) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"rating": None, "temperature": None, "pain_binary": None}
    rating_cols = config.get("event_columns.rating")
    temp_cols = config.get("event_columns.temperature")
    pain_cols = config.get("event_columns.pain_binary")

    for cand in rating_cols:
        if cand in df.columns:
            out["rating"] = cand
            break
    for cand in temp_cols:
        if cand in df.columns:
            out["temperature"] = cand
            break
    for cand in pain_cols:
        if cand in df.columns:
            out["pain_binary"] = cand
            break
    return out


def resolve_columns(
    df: pd.DataFrame,
    config: Optional[EEGConfig] = None,
    deriv_root: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if config is None:
        if deriv_root is None:
            raise ValueError("Either config or deriv_root must be provided to resolve_columns")
        from ..config.loader import load_settings

        config = load_settings()

    cols = pick_event_columns(df, config)
    return cols["pain_binary"], cols["temperature"], cols["rating"]


__all__ = [
    "_validate_event_columns",
    "load_epochs_for_analysis",
    "load_epochs_with_aligned_events",
    "pick_event_columns",
    "resolve_columns",
]
