"""
Event Alignment Utilities
=========================

Functions for aligning behavioral events with EEG epochs.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
import mne


from eeg_pipeline.infra.paths import load_events_df
from eeg_pipeline.infra.tsv import read_tsv


###################################################################
# Event / Epoch Alignment
###################################################################


def _align_by_selection(events_df: pd.DataFrame, epochs: mne.Epochs, logger: logging.Logger) -> pd.DataFrame:
    """Align events when epochs were selected by index/metadata."""
    # Check if we have selection in metadata/drop_log
    n_epochs = len(epochs)
    n_events = len(events_df)
    
    if n_epochs == n_events:
        logger.info(f"Counts match directly ({n_epochs}), assuming 1:1 alignment")
        return events_df.copy()
        
    # If epochs has 'selection' attribute (standard MNE)
    if hasattr(epochs, "selection"):
        selection = epochs.selection
        if len(selection) == n_epochs and max(selection) < n_events:
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
        
    # Try selection based alignment
    aligned = _align_by_selection(events_df, epochs, logger)
    if len(aligned) == len(epochs):
        return aligned
        
    logger.warning(
        f"Could not align events ({len(events_df)}) to epochs ({len(epochs)}) "
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
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    n_behav = len(behavioral_df)
    n_events = len(events_df)
    
    if n_behav == n_events:
        return behavioral_df
    
    if n_behav > n_events:
        logger.info(f"Trimming behavioral ({n_behav}) to match events ({n_events})")
        return behavioral_df.iloc[:n_events].reset_index(drop=True)
        
    logger.warning(f"Behavioral data ({n_behav}) shorter than events ({n_events})")
    # Pad with NaNs? Or return as is?
    # For strict behavior, we usually want to error or warn loudly
    return behavioral_df.reindex(range(n_events))


def _handle_validation_error(msg: str, strict: bool, logger: logging.Logger):
    if strict:
        raise ValueError(msg)
    logger.warning(msg)


def validate_alignment(
    aligned_events: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
    config: Optional[Any] = None,
) -> bool:
    """Validate that aligned events match epochs structure."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if aligned_events is None:
        _handle_validation_error("Aligned events DataFrame is None", strict, logger)
        return False
        
    if len(aligned_events) != len(epochs):
        _handle_validation_error(
            f"Length mismatch: events ({len(aligned_events)}) != epochs ({len(epochs)})", 
            strict, 
            logger
        )
        return False
        
    return True


def align_or_raise(
    events_df: Optional[pd.DataFrame],
    epochs: mne.Epochs,
    config,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Align events and raise error if validation fails."""
    aligned = align_events_to_epochs(events_df, epochs, logger)
    validate_alignment(aligned, epochs, logger=logger, strict=True, config=config)
    return aligned


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
        
    Returns
    -------
    pd.DataFrame or None
        Aligned events
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if config is None:
        raise ValueError("config is required for get_aligned_events")

    events_df = load_events_df(subject, task, bids_root=bids_root, config=config, constants=constants)
    if events_df is None:
        if strict:
            raise ValueError(f"Events TSV not found for sub-{subject}, task-{task}. Required when strict=True")
        logger.warning(f"Events TSV not found for sub-{subject}, task-{task}")
        return None

    try:
        aligned_events = align_events_to_epochs(events_df, epochs, logger=logger)
    except ValueError as err:
        if strict:
            raise ValueError(
                f"Alignment failed for sub-{subject}, task-{task} in strict mode: {err}"
            ) from err
        logger.warning(f"Alignment failed for sub-{subject}, task-{task}: {err}")
        return None

    if aligned_events is None:
        if strict:
            raise ValueError(
                f"Could not align events to epochs for sub-{subject}, task-{task}. "
                f"This is required when strict=True."
            )
        return None

    if len(aligned_events) != len(epochs):
        msg = (
            f"Alignment length mismatch for sub-{subject}, task-{task}: "
            f"aligned_events ({len(aligned_events)} rows) != epochs ({len(epochs)} epochs)"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    validate_alignment(aligned_events, epochs, logger, strict=strict, config=config)

    return aligned_events


###################################################################
# Index Reconstruction (migrated from general.py)
###################################################################


def reconstruct_kept_indices(dropped_trials_path: Path, n_events: int) -> np.ndarray:
    """Reconstruct which trial indices were kept after artifact rejection."""
    if not dropped_trials_path.exists():
        return np.arange(n_events)
    
    dropped_df = read_tsv(dropped_trials_path)
    if "original_index" not in dropped_df.columns or len(dropped_df) == 0:
        return np.arange(n_events)
    
    dropped_indices_raw = pd.to_numeric(dropped_df["original_index"], errors="coerce").dropna()
    if len(dropped_indices_raw) == 0:
        return np.arange(n_events)
    
    dropped_indices = set(dropped_indices_raw.astype(int).tolist())
    kept_indices = np.array([i for i in range(n_events) if i not in dropped_indices])
    return kept_indices


__all__ = [
    "align_events_to_epochs",
    "trim_behavioral_to_events_strict",
    "validate_alignment",
    "align_or_raise",
    "get_aligned_events",
    "reconstruct_kept_indices",
]
