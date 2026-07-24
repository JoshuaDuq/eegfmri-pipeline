from __future__ import annotations

import mne
import numpy as np
import pytest

from eeg_pipeline.preprocessing.pulse_artifact_qc import (
    PulseMarkerCriteria,
    validate_pulse_marker_recordings,
    validate_pulse_markers,
)


def _raw_with_pulse_markers(onsets: np.ndarray, *, duration_seconds: float = 100.0):
    sfreq = 100.0
    info = mne.create_info(["Cz", "ECG"], sfreq=sfreq, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(
        np.zeros((2, int(duration_seconds * sfreq))),
        info,
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=onsets,
            duration=np.zeros(len(onsets)),
            description=["Pulse Artifact/R"] * len(onsets),
        )
    )
    return raw


def test_validate_pulse_markers_reports_physiological_run_metrics() -> None:
    onsets = np.arange(5.0, 100.0, 1.0)

    metrics = validate_pulse_markers(
        _raw_with_pulse_markers(onsets),
        PulseMarkerCriteria(
            minimum_bpm=45.0,
            maximum_bpm=80.0,
            minimum_marker_fraction=0.8,
            minimum_recording_coverage=0.8,
        ),
        recording_id="sub-0001_run-1",
    )

    assert metrics.recording_id == "sub-0001_run-1"
    assert metrics.marker_count == 95
    assert metrics.median_bpm == pytest.approx(60.0)
    assert metrics.marker_fraction == pytest.approx(1.0)
    assert metrics.recording_coverage == pytest.approx(0.94)


def test_validate_pulse_markers_rejects_marker_dropout_fraction() -> None:
    complete_onsets = np.arange(2.0, 98.0, 0.8)
    retained_onsets = np.delete(complete_onsets, np.arange(2, len(complete_onsets), 3))

    with pytest.raises(ValueError, match="marker fraction"):
        validate_pulse_markers(
            _raw_with_pulse_markers(retained_onsets),
            PulseMarkerCriteria(
                minimum_bpm=45.0,
                maximum_bpm=80.0,
                minimum_marker_fraction=0.8,
                minimum_recording_coverage=0.8,
            ),
            recording_id="sub-0001_run-1",
        )


@pytest.mark.parametrize(
    ("onsets", "message"),
    [
        (np.array([1.0, 2.0, 3.0, 4.0]), "marker count"),
        (np.arange(5.0, 100.0, 2.0), "heart rate"),
        (np.arange(50.0, 100.0, 0.75), "recording coverage"),
    ],
)
def test_validate_pulse_markers_rejects_invalid_analyzer_output(
    onsets: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_pulse_markers(
            _raw_with_pulse_markers(onsets),
            PulseMarkerCriteria(
                minimum_bpm=45.0,
                maximum_bpm=80.0,
                minimum_marker_fraction=0.8,
                minimum_recording_coverage=0.8,
            ),
            recording_id="sub-0001_run-1",
        )


def test_validate_pulse_markers_rejects_duplicate_onsets() -> None:
    raw = _raw_with_pulse_markers(np.arange(5.0, 100.0, 1.0))
    pulse_annotations = raw.annotations.copy()
    pulse_annotations.append(10.0, 0.0, "Pulse Artifact/R")
    raw.set_annotations(pulse_annotations)

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_pulse_markers(
            raw,
            PulseMarkerCriteria(
                minimum_bpm=45.0,
                maximum_bpm=80.0,
                minimum_marker_fraction=0.8,
                minimum_recording_coverage=0.8,
            ),
            recording_id="sub-0001_run-1",
        )


def test_validate_recordings_writes_qc_before_raising_for_invalid_run(tmp_path) -> None:
    criteria = PulseMarkerCriteria(
        minimum_bpm=45.0,
        maximum_bpm=80.0,
        minimum_marker_fraction=0.8,
        minimum_recording_coverage=0.8,
    )
    output_path = tmp_path / "pulse_qc.tsv"

    with pytest.raises(ValueError, match="sub-0001_run-2"):
        validate_pulse_marker_recordings(
            [
                ("sub-0001_run-1", _raw_with_pulse_markers(np.arange(5.0, 100.0, 1.0))),
                ("sub-0001_run-2", _raw_with_pulse_markers(np.array([1.0, 2.0, 3.0]))),
            ],
            criteria,
            output_path=output_path,
        )

    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split("\t") == [
        "recording_id",
        "marker_count",
        "median_bpm",
        "marker_fraction",
        "recording_coverage",
        "status",
        "error",
    ]
    assert "sub-0001_run-1\t95\t60.0\t1.0\t0.94\tpass\t" in rows[1]
    assert "sub-0001_run-2\t\t\t\t\tfail\t" in rows[2]
