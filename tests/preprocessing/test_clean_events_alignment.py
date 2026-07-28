"""Clean events must line up with the epochs that survived rejection.

These use real ``mne.Epochs`` rather than a stub because the whole point is the semantics
of ``drop_log`` and ``selection`` on an event array that also contains the scanner-volume
and pulse markers this dataset records.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.utils.data.preprocessing import (
    _build_epoch_event_mask,
    _kept_event_mask,
    _matches_condition,
)


def _raw_with_markers() -> mne.io.RawArray:
    """A recording whose annotations are mostly scanner markers, as the real ones are."""
    sfreq = 100.0
    info = mne.create_info(["C3", "Cz", "C4"], sfreq, "eeg")
    raw = mne.io.RawArray(np.zeros((3, int(60 * sfreq))), info, verbose=False)

    onsets = []
    descriptions = []
    # One trial per 10 s, with nine volume markers interleaved between each pair.
    for index, trial_onset in enumerate([5.0, 15.0, 25.0, 35.0, 45.0]):
        onsets.append(trial_onset)
        descriptions.append("painful" if index % 2 == 0 else "neutral")
        for step in range(9):
            onsets.append(trial_onset + 0.5 + step * 0.5)
            descriptions.append("Volume")

    raw.set_annotations(mne.Annotations(onsets, np.zeros(len(onsets)), descriptions))
    return raw


def _epochs_with_conditions() -> mne.Epochs:
    raw = _raw_with_markers()
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    conditions = {name: code for name, code in event_id.items() if name != "Volume"}
    return mne.Epochs(
        raw,
        events,
        event_id=conditions,
        tmin=-0.2,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )


def test_kept_mask_ignores_the_volume_markers_that_dominate_the_event_array() -> None:
    epochs = _epochs_with_conditions()

    mask = _kept_event_mask(epochs, target_count=5)

    assert mask.tolist() == [True] * 5
    # The event array is dominated by markers that are not trials; a mapping built from
    # `selection` would index a five-row events table with these values.
    assert max(epochs.selection) > 5


def test_kept_mask_marks_exactly_the_dropped_trials() -> None:
    epochs = _epochs_with_conditions()
    epochs.drop([1, 3], reason="USER", verbose="ERROR")

    mask = _kept_event_mask(epochs, target_count=5)

    assert mask.tolist() == [True, False, True, False, True]
    assert mask.sum() == len(epochs)


def test_a_condition_count_mismatch_is_an_error_not_a_silent_shift() -> None:
    epochs = _epochs_with_conditions()

    with pytest.raises(ValueError, match="do not select the same events"):
        _kept_event_mask(epochs, target_count=4)


def test_conditions_match_mne_tag_semantics_not_plain_prefixes() -> None:
    assert _matches_condition("pain/high", "pain")
    assert _matches_condition("pain", "pain")
    # A plain prefix match would pull this in and misalign the table against the epochs.
    assert not _matches_condition("painless", "pain")


def test_event_mask_does_not_select_a_longer_similarly_named_condition() -> None:
    events = pd.DataFrame({"trial_type": ["pain", "painless", "pain/high", "neutral"]})

    mask, column = _build_epoch_event_mask(events, ["pain"])

    assert column == "trial_type"
    assert mask.tolist() == [True, False, True, False]
