"""Reading marker trains out of a run.

Four report panels are driven by annotations — volume markers, R markers, and BAD spans —
and each one needs the same thing: the onsets carrying one description, measured from the
start of the run. Doing that inline each time invited the inconsistency it produced,
where some call sites subtracted ``first_time`` and others did not.
"""

from __future__ import annotations

import mne
import numpy as np


def annotation_onsets(raw: mne.io.BaseRaw, description: str) -> np.ndarray:
    """Return sorted onsets of one annotation description, in seconds from the run start.

    Annotation onsets are stored against the recording's absolute timeline, so
    ``first_time`` is subtracted to give the offset into this run.
    """
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    onsets = np.asarray(raw.annotations.onset, dtype=float)[descriptions == description]
    return np.sort(onsets) - raw.first_time


def onset_events(raw: mne.io.BaseRaw, onsets_s: np.ndarray) -> np.ndarray:
    """Convert run-relative onsets into an MNE event array.

    ``Epochs`` indexes events against the absolute sample timeline, so ``first_samp`` is
    added back to the sample each onset falls on.
    """
    samples = np.round(onsets_s * float(raw.info["sfreq"])).astype(int) + raw.first_samp
    return np.column_stack([samples, np.zeros_like(samples), np.ones_like(samples)])


__all__ = ["annotation_onsets", "onset_events"]
