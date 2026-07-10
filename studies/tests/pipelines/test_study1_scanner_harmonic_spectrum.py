from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_scanner_harmonic_specification_reads_fixed_qc_config() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        scanner_harmonic_specification,
    )

    specification = scanner_harmonic_specification(load_study1_config())

    assert specification.frequency_range_hz == (15.0, 90.0)
    assert specification.n_fft == 8192
    assert specification.n_overlap == 4096
    assert specification.harmonic_orders == (18, 37, 55, 74)
    assert specification.excluded_subjects == ("sub-0006",)
    assert len(specification.harmonic_windows) == 4


def test_discover_final_clean_runs_uses_numbered_nonexcluded_participants(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        discover_final_clean_runs,
    )

    expected = [
        _touch_run(tmp_path, "sub-0001", task="thermalactive", run=1),
        _touch_run(tmp_path, "sub-0003", task="thermalactive", run=2),
    ]
    _touch_run(tmp_path, "sub-0006", task="thermalactive", run=1)
    _touch_run(tmp_path, "sub-pilot001", task="thermalactive", run=1)
    _touch_run(tmp_path, "sub-0004", task="other", run=1)

    paths = discover_final_clean_runs(
        tmp_path,
        task="thermalactive",
        excluded_subjects=("sub-0006",),
    )

    assert paths == tuple(expected)


def test_discover_final_clean_runs_restricts_requested_participants(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        discover_final_clean_runs,
    )

    _touch_run(tmp_path, "sub-0001", task="thermalactive", run=1)
    expected = _touch_run(tmp_path, "sub-0003", task="thermalactive", run=2)

    paths = discover_final_clean_runs(
        tmp_path,
        task="thermalactive",
        excluded_subjects=(),
        requested_subjects=("sub-0003",),
    )

    assert paths == (expected,)


def test_discover_final_clean_runs_rejects_empty_selection(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        discover_final_clean_runs,
    )

    with pytest.raises(FileNotFoundError, match="No numbered-participant final-clean FIF"):
        discover_final_clean_runs(
            tmp_path,
            task="thermalactive",
            excluded_subjects=(),
        )


def test_select_scanner_harmonic_peaks_uses_strongest_prominent_peak() -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        select_scanner_harmonic_peaks,
    )

    specification = _specification()
    frequencies = np.arange(15.0, 90.0, 500.0 / 8192.0)
    spectrum_db = np.zeros(frequencies.size, dtype=float)
    expected_frequencies = (20.0, 41.0, 61.0, 82.0)
    for amplitude, frequency in enumerate(expected_frequencies, start=5):
        spectrum_db[np.argmin(np.abs(frequencies - frequency))] = float(amplitude)
    weaker_peak = np.argmin(np.abs(frequencies - 40.0))
    spectrum_db[weaker_peak] = 2.0

    peaks = select_scanner_harmonic_peaks(
        frequencies,
        spectrum_db,
        specification,
    )

    assert [peak.window_name for peak in peaks] == [
        "scanner_18_23",
        "scanner_38_43",
        "scanner_56_67",
        "scanner_77_85",
    ]
    assert [peak.peak_frequency_hz for peak in peaks] == pytest.approx(
        expected_frequencies,
        abs=500.0 / 8192.0,
    )
    assert all(peak.prominence_db >= 5.0 for peak in peaks)


def test_select_scanner_harmonic_peaks_rejects_window_without_peak() -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        select_scanner_harmonic_peaks,
    )

    frequencies = np.arange(15.0, 90.0, 500.0 / 8192.0)

    with pytest.raises(ValueError, match="No qualifying spectral peak"):
        select_scanner_harmonic_peaks(
            frequencies,
            np.zeros(frequencies.size, dtype=float),
            _specification(),
        )


def _specification():
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        ScannerHarmonicSpecification,
    )

    return ScannerHarmonicSpecification(
        frequency_range_hz=(15.0, 90.0),
        n_fft=8192,
        n_overlap=4096,
        sampling_frequency_hz=500.0,
        peak_prominence_db=1.0,
        peak_distance_bins=4,
        volume_repetition_time_s=0.9,
        harmonic_orders=(18, 37, 55, 74),
        excluded_subjects=("sub-0006",),
    )


def _touch_run(root: Path, subject: str, *, task: str, run: int) -> Path:
    path = (
        root
        / subject
        / "eeg"
        / f"{subject}_task-{task}_run-{run}_proc-clean_raw.fif"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path
