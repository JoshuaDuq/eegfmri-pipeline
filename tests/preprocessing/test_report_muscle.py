"""MNE muscle screening is visible, scoped, and never applied implicitly."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.muscle import (  # noqa: E402
    add_muscle_review,
    compute_muscle_review,
    muscle_review_html,
    plot_muscle_review,
)


def _raw() -> mne.io.RawArray:
    info = mne.create_info(["F7", "F8", "Cz"], 500.0, "eeg")
    return mne.io.RawArray(np.zeros((3, 2_000)), info, verbose="ERROR")


def test_mne_detector_receives_the_recorded_method(monkeypatch) -> None:
    calls = []

    def fake_detector(raw, **kwargs):
        calls.append((raw, kwargs))
        annotations = mne.Annotations(onset=[1.0], duration=[0.4], description=["BAD_muscle"])
        scores = np.linspace(-1.0, 6.0, raw.n_times)
        return annotations, scores

    monkeypatch.setattr(mne.preprocessing, "annotate_muscle_zscore", fake_detector)
    raw = _raw()

    review = compute_muscle_review(
        raw,
        recording_id="sub-0001_task-pain_run-1",
        filter_freq_hz=(70.0, 90.0),
        threshold=4.5,
        min_length_good_s=0.2,
    )

    assert calls == [
        (
            raw,
            {
                "ch_type": "eeg",
                "threshold": 4.5,
                "min_length_good": 0.2,
                "filter_freq": (70.0, 90.0),
                "n_jobs": 1,
                "verbose": "ERROR",
            },
        )
    ]
    assert review.spans_s == ((1.0, 1.4),)
    assert review.artifact_fraction == pytest.approx(0.1)
    assert raw.annotations.description.tolist() == []


def test_known_bad_eeg_channels_are_excluded_from_muscle_score(monkeypatch) -> None:
    detector_inputs = []

    def fake_detector(raw, **_kwargs):
        detector_inputs.append(raw)
        return mne.Annotations([], [], []), np.zeros(raw.n_times)

    monkeypatch.setattr(mne.preprocessing, "annotate_muscle_zscore", fake_detector)
    raw = _raw()
    raw.info["bads"] = ["F7"]

    review = compute_muscle_review(
        raw,
        recording_id="run-1",
        filter_freq_hz=(70.0, 90.0),
        threshold=4.0,
        min_length_good_s=0.1,
    )

    assert detector_inputs[0].ch_names == ["F8", "Cz"]
    assert review.excluded_bad_channels == ("F7",)
    assert "F7" in muscle_review_html([review])


def test_muscle_score_requires_a_non_bad_eeg_channel() -> None:
    raw = _raw()
    raw.info["bads"] = raw.ch_names

    with pytest.raises(ValueError, match="non-bad EEG channel"):
        compute_muscle_review(
            raw,
            recording_id="run-1",
            filter_freq_hz=(70.0, 90.0),
            threshold=4.0,
            min_length_good_s=0.1,
        )


def test_filter_band_must_exist_in_the_recorded_signal() -> None:
    raw = _raw().resample(100.0)

    with pytest.raises(ValueError, match="muscle screening band.*Nyquist"):
        compute_muscle_review(
            raw,
            recording_id="run-1",
            filter_freq_hz=(70.0, 90.0),
            threshold=4.0,
            min_length_good_s=0.1,
        )


def test_report_calls_the_result_screening_not_rejection(monkeypatch) -> None:
    monkeypatch.setattr(
        mne.preprocessing,
        "annotate_muscle_zscore",
        lambda raw, **_kwargs: (
            mne.Annotations([1.0], [0.4], ["BAD_muscle"]),
            np.linspace(-1.0, 6.0, raw.n_times),
        ),
    )
    review = compute_muscle_review(
        _raw(),
        recording_id="sub-0001_task-pain_run-1",
        filter_freq_hz=(70.0, 90.0),
        threshold=4.0,
        min_length_good_s=0.1,
    )

    document = muscle_review_html([review])
    figure = plot_muscle_review([review])
    report = mne.Report(title="subject", verbose="ERROR")
    add_muscle_review(report=report, reviews=[review])

    assert "diagnostic only" in document.lower()
    assert "not added" in document.lower()
    assert "70–90 Hz" in document
    assert "Recording flagged (%)" in document
    assert "Candidate spans" not in document
    assert "Maximum score" not in document
    assert figure.axes
    assert {item.section for item in report._content} == {"Muscle artifact screening"}
    assert len(report._content) == 2
