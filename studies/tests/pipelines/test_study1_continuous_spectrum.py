from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_discover_final_clean_runs_filters_numbered_requested_subjects(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.continuous_spectrum import (
        discover_final_clean_runs,
    )

    expected = _touch_run(tmp_path, "sub-0001", run=2)
    _touch_run(tmp_path, "sub-pilot", run=1)
    _touch_run(tmp_path, "sub-0002", run=1)

    paths = discover_final_clean_runs(
        tmp_path,
        task="thermalactive",
        excluded_subjects=("sub-0002",),
        requested_subjects=("sub-0001",),
    )

    assert paths == (expected,)


def test_discover_final_clean_runs_rejects_empty_selection(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.continuous_spectrum import (
        discover_final_clean_runs,
    )

    with pytest.raises(FileNotFoundError, match="No numbered-participant final-clean FIF"):
        discover_final_clean_runs(
            tmp_path,
            task="thermalactive",
            excluded_subjects=(),
        )


def test_parse_final_clean_filename_rejects_malformed_name() -> None:
    from studies.pain_study.study1.figures.continuous_spectrum import (
        parse_final_clean_filename,
    )

    with pytest.raises(ValueError, match="Invalid final-clean EEG filename"):
        parse_final_clean_filename(Path("sub-01_task-pain_raw.fif"))


def test_bad_annotation_duration_uses_clipped_interval_union() -> None:
    from studies.pain_study.study1.figures.continuous_spectrum import (
        bad_annotation_duration_s,
    )

    duration = bad_annotation_duration_s(
        onsets_s=np.asarray([-1.0, 2.0, 3.0, 8.0]),
        durations_s=np.asarray([2.0, 3.0, 2.0, 5.0]),
        descriptions=("BAD_edge", "BAD_motion", "BAD_motion", "stimulus"),
        recording_start_s=0.0,
        recording_duration_s=10.0,
    )

    assert duration == pytest.approx(4.0)


def test_continuous_spectrum_specification_rejects_invalid_overlap() -> None:
    from studies.pain_study.study1.figures.continuous_spectrum import (
        ContinuousSpectrumSpecification,
    )

    with pytest.raises(ValueError, match="n_overlap must be smaller than n_fft"):
        ContinuousSpectrumSpecification(
            frequency_range_hz=(1.0, 90.0),
            n_fft=8192,
            n_overlap=8192,
            sampling_frequency_hz=500.0,
        )


def test_estimate_continuous_run_spectrum_uses_linear_channel_median(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mne

    from studies.pain_study.study1.figures.continuous_spectrum import (
        estimate_continuous_run_spectrum,
    )

    frequencies = np.asarray([1.0, 10.0, 90.0])
    channel_psd = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [8.0, 16.0, 24.0],
        ]
    )
    raw = _FakeRaw(_FakeSpectrum(frequencies, channel_psd))
    monkeypatch.setattr(mne.io, "read_raw_fif", lambda *args, **kwargs: raw)

    result = estimate_continuous_run_spectrum(
        _touch_run(tmp_path, "sub-0001", run=3),
        _specification(),
    )

    assert result.subject_id == "sub-0001"
    assert result.run_id == "3"
    assert result.median_psd_v2_hz == pytest.approx([2.0, 4.0, 6.0])
    assert result.recording_duration_s == pytest.approx(100.0)
    assert result.bad_annotation_duration_s == pytest.approx(3.0)
    assert result.analyzed_duration_s == pytest.approx(97.0)
    assert raw.compute_kwargs == {
        "method": "welch",
        "fmin": 1.0,
        "fmax": 90.0,
        "n_fft": 8192,
        "n_per_seg": 8192,
        "n_overlap": 4096,
        "picks": "eeg",
        "reject_by_annotation": True,
        "verbose": False,
    }


def test_estimate_continuous_run_spectrum_rejects_short_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mne

    from studies.pain_study.study1.figures.continuous_spectrum import (
        estimate_continuous_run_spectrum,
    )

    raw = _FakeRaw(_FakeSpectrum(np.asarray([1.0]), np.asarray([[1.0]])))
    raw.n_times = 8_191
    monkeypatch.setattr(mne.io, "read_raw_fif", lambda *args, **kwargs: raw)

    with pytest.raises(ValueError, match="fewer samples than n_fft"):
        estimate_continuous_run_spectrum(
            _touch_run(tmp_path, "sub-0001", run=1),
            _specification(),
        )


def _specification():
    from studies.pain_study.study1.figures.continuous_spectrum import (
        ContinuousSpectrumSpecification,
    )

    return ContinuousSpectrumSpecification(
        frequency_range_hz=(1.0, 90.0),
        n_fft=8192,
        n_overlap=4096,
        sampling_frequency_hz=500.0,
    )


class _FakeSpectrum:
    def __init__(self, frequencies: np.ndarray, psd: np.ndarray) -> None:
        self.freqs = frequencies
        self._psd = psd

    def get_data(self) -> np.ndarray:
        return self._psd


class _FakeAnnotations:
    onset = np.asarray([1.0, 2.0, 20.0])
    duration = np.asarray([2.0, 2.0, 1.0])
    description = np.asarray(["BAD_motion", "BAD_muscle", "stimulus"])


class _FakeRaw:
    def __init__(self, spectrum: _FakeSpectrum) -> None:
        self.info = {"sfreq": 500.0}
        self.n_times = 50_000
        self.first_time = 0.0
        self.annotations = _FakeAnnotations()
        self._spectrum = spectrum
        self.compute_kwargs: dict[str, object] = {}

    def compute_psd(self, **kwargs):
        self.compute_kwargs = kwargs
        return self._spectrum


def _touch_run(root: Path, subject: str, *, run: int) -> Path:
    path = (
        root
        / subject
        / "eeg"
        / f"{subject}_task-thermalactive_run-{run}_proc-clean_raw.fif"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path
