from __future__ import annotations

from datetime import datetime, timezone

import mne
import numpy as np


class _RawWithAnnotations:
    # An uncropped raw: first_time is 0.0, so onsets need no rebasing.
    first_time = 0.0

    def __init__(self) -> None:
        self.annotations = mne.Annotations(
            onset=[0.0],
            duration=[1.0],
            description=["stim"],
        )

    def set_annotations(self, annotations: mne.Annotations) -> None:
        self.annotations = annotations


def test_mark_breaks_bad_adds_bad_break_annotations(monkeypatch) -> None:
    from eeg_pipeline.preprocessing.pipeline import preprocess

    raw = _RawWithAnnotations()
    detected = mne.Annotations(
        onset=[10.0, 30.0],
        duration=[5.0, 7.5],
        description=["BAD_break", "BAD_break"],
    )

    def fake_annotate_break(**kwargs):
        assert kwargs["raw"] is raw
        assert kwargs["min_break_duration"] == 20
        assert kwargs["t_start_after_previous"] == 2
        assert kwargs["t_stop_before_next"] == 2
        return detected

    monkeypatch.setattr(mne.preprocessing, "annotate_break", fake_annotate_break)

    annotations, removed_duration = preprocess._mark_breaks_bad(
        raw=raw,
        breaks_min_length=20,
        t_start_after_previous=2,
        t_stop_before_next=2,
    )

    assert annotations is detected
    assert removed_duration == 12.5
    assert list(raw.annotations.description) == ["stim", "BAD_break", "BAD_break"]


def _cropped_raw_with_stims(meas_date) -> mne.io.BaseRaw:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 12_000)), info, verbose=False)
    if meas_date is not None:
        raw.set_meas_date(meas_date)
    raw.set_annotations(
        mne.Annotations(
            onset=[5.0, 95.0],
            duration=[0.0, 0.0],
            description=["stim", "stim"],
            orig_time=raw.info["meas_date"],
        )
    )
    raw.crop(tmin=4.0)
    return raw


def test_mark_breaks_bad_leaves_existing_onsets_put_on_a_cropped_raw() -> None:
    from eeg_pipeline.preprocessing.pipeline import preprocess

    raw = _cropped_raw_with_stims(meas_date=None)
    assert raw.first_time == 4.0

    preprocess._mark_breaks_bad(raw, 20, 2, 2)

    # annotate_break reports break onsets relative to the data start when there is
    # no meas_date, but raw.annotations.onset is on the absolute timeline. Merging
    # the two without rebasing shifts the stims by the 4.0s crop offset.
    assert raw.annotations.onset.tolist() == [5.0, 7.0, 95.0, 97.0]
    assert raw.annotations.description.tolist() == [
        "stim",
        "BAD_break",
        "stim",
        "BAD_break",
    ]


def test_mark_breaks_bad_matches_the_meas_date_path() -> None:
    from eeg_pipeline.preprocessing.pipeline import preprocess

    dated = _cropped_raw_with_stims(datetime(2026, 1, 1, tzinfo=timezone.utc))
    undated = _cropped_raw_with_stims(meas_date=None)

    preprocess._mark_breaks_bad(dated, 20, 2, 2)
    preprocess._mark_breaks_bad(undated, 20, 2, 2)

    assert undated.annotations.onset.tolist() == dated.annotations.onset.tolist()
    assert undated.annotations.duration.tolist() == dated.annotations.duration.tolist()
