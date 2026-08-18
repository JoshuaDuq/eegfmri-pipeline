"""Per-run evidence must be gathered in one pass and degrade section by section."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.run_evidence import (  # noqa: E402
    add_run_evidence_review,
    measure_runs,
)
from eeg_pipeline.preprocessing.report.settings import ReportSettings  # noqa: E402

SFREQ = 500.0
TR = 0.9
DURATION = 90.0


def _write_run(
    tmp_path,
    name,
    *,
    with_markers=True,
    with_beats=True,
    n_channels=8,
    volume_description="Volume/V  1",
    pulse_description="Pulse Artifact/R",
):
    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(n_channels)], SFREQ, "eeg")
    n_samples = int(DURATION * SFREQ)
    times = np.arange(n_samples) / SFREQ
    data = rng.normal(0, 1e-5, (n_channels, n_samples))
    for order in range(18, 50):
        data[0] += 3e-6 * np.sin(2 * np.pi * (order / TR) * times + rng.uniform(0, 6))
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    annotations = mne.Annotations([], [], [])
    if with_markers:
        onsets = np.arange(0.0, DURATION - TR, TR)
        annotations += mne.Annotations(
            onset=onsets, duration=0.0, description=[volume_description] * len(onsets)
        )
    if with_beats:
        beats = np.cumsum(rng.normal(0.85, 0.02, 120))
        beats = beats[beats < DURATION]
        annotations += mne.Annotations(
            onset=beats, duration=0.0, description=[pulse_description] * len(beats)
        )
    if len(annotations):
        raw.set_annotations(annotations)

    path = tmp_path / name
    raw.save(path, verbose="ERROR")
    return path, raw


def _ica(raw):
    ica = mne.preprocessing.ICA(n_components=4, random_state=0, max_iter=200)
    ica.fit(raw, verbose="ERROR")
    ica.exclude = [0]
    return ica


def test_every_per_run_measurement_is_gathered(tmp_path) -> None:
    path, raw = _write_run(tmp_path, "sub-0001_task-x_run-1_proc-filt_raw.fif")

    evidence = measure_runs(filtered_raw_paths=[path], ica=_ica(raw), settings=ReportSettings())

    assert len(evidence.spectra) == 1
    assert len(evidence.continuity) == 1
    assert len(evidence.rr_intervals) == 1
    assert evidence.spectra[0].recording_id == "sub-0001_task-x_run-1"




def test_configured_aperiodic_range_reaches_the_spectral_fit(tmp_path) -> None:
    path, raw = _write_run(tmp_path, "sub-0001_task-x_run-1_proc-filt_raw.fif")
    settings = ReportSettings(aperiodic_fit_range_hz=(5.0, 35.0))

    evidence = measure_runs(filtered_raw_paths=[path], ica=_ica(raw), settings=settings)

    assert evidence.spectra[0].after.aperiodic.fit_range_hz == (5.0, 35.0)


def test_the_ica_is_applied_once_per_run_not_once_per_panel(tmp_path) -> None:
    """Four panels need the cleaned run; applying the ICA four times is the cost."""
    path, raw = _write_run(tmp_path, "sub-0001_task-x_run-1_proc-filt_raw.fif")
    ica = _ica(raw)
    calls = {"count": 0}
    original = ica.apply

    def counting_apply(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    ica.apply = counting_apply

    measure_runs(filtered_raw_paths=[path], ica=ica, settings=ReportSettings())

    assert calls["count"] == 1


def test_a_recording_without_scanner_markers_keeps_the_other_sections(tmp_path) -> None:
    path, raw = _write_run(
        tmp_path, "sub-0001_task-rest_run-1_proc-filt_raw.fif", with_markers=False
    )

    evidence = measure_runs(filtered_raw_paths=[path], ica=_ica(raw), settings=ReportSettings())

    assert len(evidence.spectra) == 1
    assert len(evidence.continuity) == 1
    assert not evidence.continuity[0].has_volume_markers


def test_a_recording_without_beats_keeps_the_other_sections(tmp_path) -> None:
    path, raw = _write_run(tmp_path, "sub-0001_task-x_run-1_proc-filt_raw.fif", with_beats=False)

    evidence = measure_runs(filtered_raw_paths=[path], ica=_ica(raw), settings=ReportSettings())

    assert evidence.rr_intervals == []


def test_the_report_gains_every_available_section(tmp_path) -> None:
    path, raw = _write_run(tmp_path, "sub-0001_task-x_run-1_proc-filt_raw.fif")
    report = mne.Report(title="subject", verbose="ERROR")

    add_run_evidence_review(
        report=report,
        filtered_raw_paths=[path],
        ica=_ica(raw),
        settings=ReportSettings(spectra_line_frequency=60.0),
    )

    sections = {element.section for element in report._content}
    assert sections == {
        "Sensor spectra before and after ICA",
        "Data quality over time",
        "Cardiac rhythm",
    }


def test_rebuilding_replaces_rather_than_accumulates(tmp_path) -> None:
    path, raw = _write_run(tmp_path, "sub-0001_task-x_run-1_proc-filt_raw.fif")
    ica = _ica(raw)
    report = mne.Report(title="subject", verbose="ERROR")

    add_run_evidence_review(
        report=report, filtered_raw_paths=[path], ica=ica, settings=ReportSettings()
    )
    first = len(report._content)
    add_run_evidence_review(
        report=report, filtered_raw_paths=[path], ica=ica, settings=ReportSettings()
    )

    assert len(report._content) == first


def test_no_runs_is_a_programming_error() -> None:
    """An unfitted ICA suffices: the guard must fire before anything is read."""
    ica = mne.preprocessing.ICA(n_components=2, random_state=0)

    with pytest.raises(ValueError, match="at least one filtered run"):
        measure_runs(filtered_raw_paths=[], ica=ica, settings=ReportSettings())


def test_non_finite_ica_output_is_rejected_before_measurement(tmp_path) -> None:
    path, _ = _write_run(tmp_path, "sub-0001_task-x_run-1_proc-filt_raw.fif")

    class NonFiniteIca:
        exclude = []

        @staticmethod
        def apply(raw, *, exclude, verbose):
            raw._data[0, 0] = np.nan
            return raw

    with pytest.raises(ValueError, match="non-finite values in ICA output"):
        measure_runs(
            filtered_raw_paths=[path],
            ica=NonFiniteIca(),
            settings=ReportSettings(),
        )
