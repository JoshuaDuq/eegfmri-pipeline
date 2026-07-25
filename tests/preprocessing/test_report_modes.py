"""The report must build for EEG-only and resting-state datasets, not just EEG-fMRI."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.analyzer_qc import (  # noqa: E402
    add_analyzer_correction_review,
)
from eeg_pipeline.preprocessing.report.coverage import add_coverage_review  # noqa: E402
from eeg_pipeline.preprocessing.report.provenance import add_provenance_review  # noqa: E402
from eeg_pipeline.preprocessing.report.run_evidence import (  # noqa: E402
    add_run_evidence_review,
    measure_runs,
)
from eeg_pipeline.preprocessing.report.settings import ReportSettings  # noqa: E402


def _raw(tmp_path, name="sub-0001_task-rest_run-1_proc-filt_raw.fif", sfreq=250.0):
    info = mne.create_info([f"C{index}" for index in range(8)], sfreq, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(0, 1e-5, (8, int(sfreq * 60))),
        info,
        verbose="ERROR",
    )
    path = tmp_path / name
    raw.save(path, verbose="ERROR")
    return path, raw


def test_scanner_sections_are_absent_without_scanner_data(tmp_path) -> None:
    """An EEG-only dataset gets no Analyzer or coverage section, not an empty one."""
    report = mne.Report(title="eeg-only", verbose="ERROR")

    analyzer = add_analyzer_correction_review(
        report=report, qc_dir=tmp_path, task="rest", subject="0001"
    )
    coverage = add_coverage_review(
        report=report,
        deriv_eeg_root=tmp_path,
        task="rest",
        subject="0001",
        settings=ReportSettings(),
    )

    assert analyzer is None
    assert coverage is None
    assert report._content == []


def test_provenance_builds_on_an_otherwise_empty_report() -> None:
    """Moving content to the front must not require something to sit in front of."""
    report = mne.Report(title="eeg-only", verbose="ERROR")

    add_provenance_review(report=report, config=None)

    assert len(report._content) == 1
    assert "provenance" in report._content[0].tags


def test_spectra_build_for_continuous_resting_state_data(tmp_path) -> None:
    """Resting state has no events or epochs; the spectra must not depend on them."""
    path, raw = _raw(tmp_path)
    ica = mne.preprocessing.ICA(n_components=5, random_state=0, max_iter=200)
    ica.fit(raw, verbose="ERROR")
    ica.exclude = [0]
    report = mne.Report(title="rest", verbose="ERROR")

    evidence = add_run_evidence_review(
        report=report,
        filtered_raw_paths=[path],
        ica=ica,
        settings=ReportSettings(),
    )

    assert len(evidence.spectra) == 1
    assert evidence.spectra[0].frequencies[-1] < raw.info["sfreq"] / 2
    # A recording made outside a scanner has no gradient or beat evidence, and the
    # sections that depend on them must be absent rather than empty.
    assert not evidence.has_scanner_evidence
    assert evidence.rr_intervals == []
    assert {element.section for element in report._content} == {
        "Sensor spectra before and after ICA",
        "Data quality over time",
    }


def test_placement_without_its_anchor_keeps_the_content(tmp_path) -> None:
    """A partial pipeline has no anchor section; that must not fail the report."""
    from eeg_pipeline.preprocessing.report.organize import move_tagged_content_before

    report = mne.Report(title="partial", verbose="ERROR")
    report.add_html(html="<p>x</p>", title="only", section="Only", tags=("keep",))

    move_tagged_content_before(report, tag="keep", anchor=lambda element: False)

    assert len(report._content) == 1


def test_missing_tagged_content_is_still_an_error() -> None:
    """Placing content that was never added is a programming error, not a data case."""
    from eeg_pipeline.preprocessing.report.organize import move_tagged_content_before

    report = mne.Report(title="empty", verbose="ERROR")

    with pytest.raises(ValueError, match="no content tagged"):
        move_tagged_content_before(report, tag="absent", anchor=lambda element: True)


def test_spectra_respect_a_low_sampling_rate(tmp_path) -> None:
    """A 100 Hz recording has no 120 Hz harmonic to plot."""
    path, raw = _raw(tmp_path, name="sub-0001_task-rest_run-2_proc-filt_raw.fif", sfreq=100.0)
    ica = mne.preprocessing.ICA(n_components=4, random_state=0, max_iter=200)
    ica.fit(raw, verbose="ERROR")

    evidence = measure_runs(
        filtered_raw_paths=[path],
        ica=ica,
        settings=ReportSettings(spectra_line_frequency=60.0),
    )

    assert evidence.spectra[0].frequencies[-1] <= 49.0
