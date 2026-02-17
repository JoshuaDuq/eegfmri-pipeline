"""
Clean Events Loading Utilities
===============================

Functions for loading clean events.tsv (already aligned to epochs).
"""

from __future__ import annotations

import logging
from typing import Optional

import mne
import pandas as pd



def get_aligned_events(
    epochs: mne.Epochs,
    subject: str,
    task: str,
    *,
    strict: bool = True,
    logger: Optional[logging.Logger] = None,
    config=None,
    constants=None,
) -> Optional[pd.DataFrame]:
    """
    Load clean events.tsv (already aligned, no alignment needed).

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
    config
        Configuration object
    constants
        Constants object

    Returns
    -------
    pd.DataFrame or None
        Clean events (already aligned)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if config is None:
        raise ValueError("config is required for get_aligned_events")

    from eeg_pipeline.infra.paths import _find_clean_events_path, _resolve_deriv_root
    
    try:
        deriv_root = _resolve_deriv_root(None, config, constants)
    except Exception:
        deriv_root = None
    
    if deriv_root is None:
        if strict:
            raise ValueError(
                f"Could not resolve deriv_root for sub-{subject}, task-{task}"
            )
        logger.warning(f"Could not resolve deriv_root for sub-{subject}, task-{task}")
        return None
    
    clean_events_path = _find_clean_events_path(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        config=config,
        constants=constants,
    )
    
    if clean_events_path is None or not clean_events_path.exists():
        message = (
            f"Clean events.tsv not found for sub-{subject}, task-{task}. "
            "Required when strict=True"
        )
        if strict:
            raise ValueError(message)
        logger.warning(
            f"Clean events.tsv not found for sub-{subject}, task-{task}"
        )
        return None

    events_df = pd.read_csv(clean_events_path, sep="\t")
    
    if len(events_df) != len(epochs):
        message = (
            f"Clean events.tsv length mismatch for sub-{subject}, task-{task}: "
            f"events={len(events_df)}, epochs={len(epochs)}"
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)
        return None
    
    logger.info(f"Loaded clean events.tsv: {len(events_df)} rows (already aligned)")
    return events_df.reset_index(drop=True)


__all__ = [
    "get_aligned_events",
]
