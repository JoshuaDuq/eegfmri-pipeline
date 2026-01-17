"""
Event Alignment Utilities
=========================

Functions for aligning behavioral events with EEG epochs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mne
import pandas as pd

from eeg_pipeline.infra.paths import load_events_df


def _align_by_selection(
    events_df: pd.DataFrame,
    epochs: mne.Epochs,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Align events when epochs were selected by index/metadata."""
    n_epochs = len(epochs)
    n_events = len(events_df)

    if n_epochs == n_events:
        logger.info(
            f"Counts match directly ({n_epochs}), assuming 1:1 alignment"
        )
        return events_df.copy()

    if hasattr(epochs, "selection"):
        selection = epochs.selection
        is_valid_selection = (
            len(selection) == n_epochs and max(selection) < n_events
        )
        if is_valid_selection:
            logger.info("Aligning using epochs.selection indices")
            return events_df.iloc[selection].reset_index(drop=True)

    return events_df


def align_events_to_epochs(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Align behavioral events DataFrame to Epochs object.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events containing separate trial behavioral data
    epochs : mne.Epochs
        Epochs object
    logger : logging.Logger
        Logger

    Returns
    -------
    pd.DataFrame
        Aligned events DataFrame with same length as epochs
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if events_df is None:
        logger.warning("No events DataFrame provided for alignment")
        return pd.DataFrame(index=range(len(epochs)))

    if len(events_df) == len(epochs):
        return events_df.reset_index(drop=True)

    aligned = _align_by_selection(events_df, epochs, logger)
    if len(aligned) == len(epochs):
        return aligned

    n_events = len(events_df)
    n_epochs = len(epochs)
    logger.warning(
        f"Could not align events ({n_events}) to epochs ({n_epochs}) "
        "automatically. Returning original events (potential mismatch)."
    )
    return events_df


def trim_behavioral_to_events_strict(
    behavioral_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Trim behavioral data to match events DataFrame length.

    Strictly assumes linear matching and warns on mismatch.
    When behavioral data is shorter than events, pads with NaN values.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_behavioral_rows = len(behavioral_df)
    n_event_rows = len(events_df)

    if n_behavioral_rows == n_event_rows:
        return behavioral_df

    if n_behavioral_rows > n_event_rows:
        logger.info(
            f"Trimming behavioral ({n_behavioral_rows}) "
            f"to match events ({n_event_rows})"
        )
        return behavioral_df.iloc[:n_event_rows].reset_index(drop=True)

    logger.warning(
        f"Behavioral data ({n_behavioral_rows}) "
        f"shorter than events ({n_event_rows}). Padding with NaN."
    )
    return behavioral_df.reindex(range(n_event_rows))


def validate_alignment(
    aligned_events: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> bool:
    """Validate that aligned events match epochs structure."""
    if logger is None:
        logger = logging.getLogger(__name__)

    if aligned_events is None:
        message = "Aligned events DataFrame is None"
        if strict:
            raise ValueError(message)
        logger.warning(message)
        return False

    n_events = len(aligned_events)
    n_epochs = len(epochs)
    if n_events != n_epochs:
        message = (
            f"Length mismatch: events ({n_events}) != epochs ({n_epochs})"
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)
        return False

    return True


def get_aligned_events(
    epochs: mne.Epochs,
    subject: str,
    task: str,
    *,
    strict: bool = True,
    logger: Optional[logging.Logger] = None,
    bids_root: Optional[Path] = None,
    config=None,
    constants=None,
) -> Optional[pd.DataFrame]:
    """
    Load and align events for a subject/task.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    subject : str
        Subject ID
    task : str
        Task name
    strict : bool
        If True, raise error on failure
    logger : Logger
        Logger
    bids_root : Path
        BIDS root directory
    config
        Configuration object
    constants
        Constants object

    Returns
    -------
    pd.DataFrame or None
        Aligned events
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if config is None:
        raise ValueError("config is required for get_aligned_events")

    events_df = load_events_df(
        subject, task, bids_root=bids_root, config=config, constants=constants
    )
    if events_df is None:
        message = (
            f"Events TSV not found for sub-{subject}, task-{task}. "
            "Required when strict=True"
        )
        if strict:
            raise ValueError(message)
        logger.warning(
            f"Events TSV not found for sub-{subject}, task-{task}"
        )
        return None

    try:
        aligned_events = align_events_to_epochs(
            events_df, epochs, logger=logger
        )
    except ValueError as err:
        message = (
            f"Alignment failed for sub-{subject}, task-{task} "
            f"in strict mode: {err}"
        )
        if strict:
            raise ValueError(message) from err
        logger.warning(
            f"Alignment failed for sub-{subject}, task-{task}: {err}"
        )
        return None

    validate_alignment(aligned_events, epochs, logger, strict=strict)
    return aligned_events


__all__ = [
    "align_events_to_epochs",
    "trim_behavioral_to_events_strict",
    "validate_alignment",
    "get_aligned_events",
]
