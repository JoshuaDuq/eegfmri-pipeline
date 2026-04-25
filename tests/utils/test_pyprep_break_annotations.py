from __future__ import annotations

import mne


class _RawWithAnnotations:
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
