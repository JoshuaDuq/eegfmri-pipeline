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


def test_validity_bootstrap_specification_reads_participant_bootstrap_config() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        validity_bootstrap_specification,
    )

    specification = validity_bootstrap_specification(load_study1_config())

    assert specification.iterations == 10_000
    assert specification.confidence_level == 0.95
    assert specification.seed == 42


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


def test_paired_participant_bootstrap_matches_seeded_median_resampling() -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        paired_participant_bootstrap,
    )

    values = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 40.0],
            [8.0, 90.0],
        ]
    )
    iterations = 200
    seed = 42
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    expected_bootstrap = np.median(values[indices], axis=1)

    median, ci_low, ci_high = paired_participant_bootstrap(
        values,
        iterations=iterations,
        confidence_level=0.95,
        seed=seed,
    )

    assert median == pytest.approx(np.median(values, axis=0))
    assert ci_low == pytest.approx(np.quantile(expected_bootstrap, 0.025, axis=0))
    assert ci_high == pytest.approx(np.quantile(expected_bootstrap, 0.975, axis=0))


def test_build_scanner_harmonic_summary_weights_participants_equally() -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        ParticipantBootstrapSpecification,
        build_scanner_harmonic_summary,
    )

    frequencies = np.asarray([15.0, 30.0, 45.0])
    run_spectra = tuple(
        [
            _run_spectrum("sub-01", run, frequencies, [0.0, 10.0, 20.0])
            for run in range(1, 7)
        ]
        + [
            _run_spectrum("sub-02", 1, frequencies, [100.0, 140.0, 180.0]),
            _run_spectrum("sub-02", 2, frequencies, [120.0, 160.0, 200.0]),
        ]
    )

    summary = build_scanner_harmonic_summary(
        run_spectra,
        _specification(),
        bootstrap=ParticipantBootstrapSpecification(
            iterations=200,
            confidence_level=0.95,
            seed=42,
        ),
    )

    participant = summary.participant_spectra.pivot(
        index="subject_id",
        columns="frequency_hz",
        values="relative_psd_db",
    )
    expected_sub_01 = np.asarray([-10.0, 0.0, 10.0])
    expected_sub_02 = np.asarray([-40.0, 0.0, 40.0])
    assert participant.loc["sub-01"].to_numpy() == pytest.approx(expected_sub_01)
    assert participant.loc["sub-02"].to_numpy() == pytest.approx(expected_sub_02)
    assert summary.cohort_spectrum["median_relative_psd_db"].to_numpy() == pytest.approx(
        np.median(np.vstack([expected_sub_01, expected_sub_02]), axis=0)
    )
    assert summary.n_subjects == 2
    assert summary.n_runs == 8
    assert summary.participant_audit.set_index("subject_id")["n_runs"].to_dict() == {
        "sub-01": 6,
        "sub-02": 2,
    }


def test_build_scanner_harmonic_summary_pairs_peak_offset_bootstrap() -> None:
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        ParticipantBootstrapSpecification,
        build_scanner_harmonic_summary,
    )

    frequencies = np.asarray([15.0, 30.0, 45.0])
    run_spectra = tuple(
        _run_spectrum(
            subject,
            1,
            frequencies,
            [0.0, 1.0, 2.0],
            peak_shift_hz=shift,
        )
        for subject, shift in (("sub-01", -0.03), ("sub-02", 0.01), ("sub-03", 0.05))
    )
    bootstrap = ParticipantBootstrapSpecification(
        iterations=200,
        confidence_level=0.95,
        seed=42,
    )

    summary = build_scanner_harmonic_summary(
        run_spectra,
        _specification(),
        bootstrap=bootstrap,
    )

    offset_matrix = summary.participant_offsets.pivot(
        index="subject_id",
        columns="window_name",
        values="offset_hz",
    ).to_numpy()
    rng = np.random.default_rng(bootstrap.seed)
    indices = rng.integers(
        0,
        offset_matrix.shape[0],
        size=(bootstrap.iterations, offset_matrix.shape[0]),
    )
    sampled = np.median(offset_matrix[indices], axis=1)
    cohort = summary.cohort_offsets.set_index("window_name").loc[
        summary.participant_offsets["window_name"].drop_duplicates()
    ]
    assert cohort["ci_low_offset_hz"].to_numpy() == pytest.approx(
        np.quantile(sampled, 0.025, axis=0)
    )
    assert cohort["ci_high_offset_hz"].to_numpy() == pytest.approx(
        np.quantile(sampled, 0.975, axis=0)
    )


def test_estimate_run_spectrum_uses_channel_median_linear_psd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import mne

    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        estimate_run_spectrum,
    )

    frequencies = np.arange(15.0, 90.0, 500.0 / 8192.0)
    median_db = np.zeros(frequencies.size)
    for amplitude, frequency in enumerate((20.0, 41.0, 61.0, 82.0), start=5):
        median_db[np.argmin(np.abs(frequencies - frequency))] = float(amplitude)
    median_linear = 10.0 ** (median_db / 10.0)
    channel_psd = np.vstack((median_linear * 0.5, median_linear, median_linear * 2.0))
    spectrum = _FakeSpectrum(frequencies, channel_psd)
    raw = _FakeRaw(spectrum)
    monkeypatch.setattr(mne.io, "read_raw_fif", lambda *args, **kwargs: raw)
    path = _touch_run(tmp_path, "sub-0001", task="thermalactive", run=3)

    result = estimate_run_spectrum(path, _specification())

    assert result.subject_id == "sub-0001"
    assert result.run_id == "3"
    assert result.n_channels == 3
    assert result.n_samples == 50_000
    assert result.median_psd_db == pytest.approx(median_db)
    assert raw.compute_kwargs == {
        "method": "welch",
        "fmin": 15.0,
        "fmax": 90.0,
        "n_fft": 8192,
        "n_per_seg": 8192,
        "n_overlap": 4096,
        "picks": "eeg",
        "reject_by_annotation": True,
        "verbose": False,
    }


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


def _run_spectrum(
    subject_id: str,
    run: int,
    frequencies: np.ndarray,
    median_psd_db: list[float],
    *,
    peak_shift_hz: float = 0.0,
):
    from studies.pain_study.study1.figures.scanner_harmonic_spectrum import (
        HarmonicPeak,
        RunSpectrum,
    )

    specification = _specification()
    predicted = np.asarray(specification.harmonic_orders) / specification.volume_repetition_time_s
    peaks = tuple(
        HarmonicPeak(
            window_name=window.name,
            peak_frequency_hz=float(frequency + peak_shift_hz),
            prominence_db=float(10.0 + window_index),
        )
        for window_index, (window, frequency) in enumerate(
            zip(specification.harmonic_windows, predicted, strict=True)
        )
    )
    return RunSpectrum(
        subject_id=subject_id,
        run_id=str(run),
        source_file=Path(f"{subject_id}_run-{run}.fif"),
        frequencies_hz=frequencies,
        median_psd_db=np.asarray(median_psd_db, dtype=float),
        n_channels=55,
        sampling_frequency_hz=500.0,
        n_samples=50_000,
        peaks=peaks,
    )


class _FakeSpectrum:
    def __init__(self, frequencies: np.ndarray, psd: np.ndarray) -> None:
        self.freqs = frequencies
        self._psd = psd

    def get_data(self) -> np.ndarray:
        return self._psd


class _FakeRaw:
    def __init__(self, spectrum: _FakeSpectrum) -> None:
        self.info = {"sfreq": 500.0}
        self.n_times = 50_000
        self.first_time = 0.0
        self.annotations = _EmptyAnnotations()
        self._spectrum = spectrum
        self.compute_kwargs: dict[str, object] = {}

    def compute_psd(self, **kwargs):
        self.compute_kwargs = kwargs
        return self._spectrum


class _EmptyAnnotations:
    onset = np.asarray([], dtype=float)
    duration = np.asarray([], dtype=float)
    description = np.asarray([], dtype=str)


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
