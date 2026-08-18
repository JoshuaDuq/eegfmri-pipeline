"""The report must build for EEG-only and resting-state datasets, not just EEG-fMRI."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

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
    """An EEG-only dataset gets no coverage section, not an empty one."""
    report = mne.Report(title="eeg-only", verbose="ERROR")

    coverage = add_coverage_review(
        report=report,
        deriv_eeg_root=tmp_path,
        task="rest",
        subject="0001",
        settings=ReportSettings(),
    )

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
    # A recording made outside a scanner has no beat evidence, and the sections that
    # depend on it must be absent rather than empty.
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


class _Config:
    """A config that knows the difference between "set to false" and "never set"."""

    def __init__(self, values):
        self._values = dict(values)

    def get(self, key, default=None):
        return self._values.get(key, default)


def test_settings_the_dataset_never_configured_are_omitted() -> None:
    """An EEG-only report must not carry a row about scanner correction.

    A key absent from the configuration influenced nothing, so listing it as "not set"
    spends a row telling the reader about equipment they do not have. A key present and
    false is different: that was a decision, and reproducing the report needs it.
    """
    from eeg_pipeline.preprocessing.report.provenance import provenance_html

    document = provenance_html(
        _Config(
            {
                "eeg.reference": "average",
                "preprocessing.l_freq": 0.1,
                "ica.use_icalabel": False,
            }
        )
    )

    assert "not set" not in document
    assert "EEG reference" in document
    # Present and false: a recorded decision, so it stays.
    assert "ICLabel used" in document


def test_a_beat_marker_setting_is_reported_even_when_empty() -> None:
    """A named beat label is what decides whether the cardiac panels searched at all."""
    from eeg_pipeline.preprocessing.report.provenance import provenance_html

    document = provenance_html(_Config({"ica.cardiac_review.beat_source": "ecg"}))

    assert "ECG beat source" in document


def test_a_report_whose_settings_are_all_absent_says_so() -> None:
    """An empty table would read as a report with no settings rather than no record."""
    from eeg_pipeline.preprocessing.report.provenance import provenance_html

    document = provenance_html(_Config({}))

    assert "<table" not in document
    assert "no recorded settings" in document.lower()


def test_measurement_defining_report_settings_are_provenance() -> None:
    from eeg_pipeline.preprocessing.report.provenance import provenance_html

    document = provenance_html(
        _Config(
            {
                "report.analysis.aperiodic_fit_range_hz": [2.0, 45.0],
                "report.analysis.response_window_s": [0.0, 1.0],
                "report.analysis.alpha_band_hz": [7.0, 14.0],
                "report.acquisition.posterior_channel_pattern": "^O",
                "report.display.continuity_window_seconds": 1.0,
            }
        )
    )

    for label in (
        "Aperiodic fit range",
        "Split-half response window",
        "Posterior rhythm band",
        "Posterior channel pattern",
        "Continuity window",
    ):
        assert label in document


def _runless_raw(tmp_path, sfreq=250.0):
    """A single baseline acquisition: no run entity, no scanner, no ECG."""
    return _raw(tmp_path, name="sub-0001_task-baseline_proc-filt_raw.fif", sfreq=sfreq)


def test_a_runless_recording_needs_no_run_entity_to_be_measured(tmp_path) -> None:
    """BIDS omits ``run-`` when there is nothing to tell apart, which is the baseline case."""
    path, raw = _runless_raw(tmp_path)
    ica = mne.preprocessing.ICA(n_components=5, random_state=0, max_iter=200)
    ica.fit(raw, verbose="ERROR")

    evidence = measure_runs(filtered_raw_paths=[path], ica=ica, settings=ReportSettings())

    assert len(evidence.spectra) == 1


def test_a_runless_table_is_not_keyed_by_the_subject(tmp_path) -> None:
    """Every per-run table is keyed by ``run_label``, so a runless dataset repeated the
    subject and task in every row of every one of them — identical in each, and already
    the title of the report they sit in.

    Asserted on the table cells only. A figure caption naming the recording it was drawn
    from is provenance and belongs there.
    """
    import re

    from eeg_pipeline.preprocessing.report.style import run_label

    path, raw = _runless_raw(tmp_path)
    ica = mne.preprocessing.ICA(n_components=5, random_state=0, max_iter=200)
    ica.fit(raw, verbose="ERROR")
    report = mne.Report(title="baseline", verbose="ERROR")

    add_run_evidence_review(
        report=report, filtered_raw_paths=[path], ica=ica, settings=ReportSettings()
    )

    rendered = "".join(
        str(element.html) for element in report._content if getattr(element, "html", None)
    )
    cells = re.findall(r"<td[^>]*>(.*?)</td>", rendered, re.S)
    assert cells, "the run evidence should have rendered at least one table"
    assert not [cell for cell in cells if "sub-0001" in cell]
    assert run_label("sub-0001_task-baseline") in cells


def test_the_filter_section_claims_no_upstream_removal_without_a_manifest() -> None:
    """``unavailable_intervals_by_recording`` is empty for a dataset no comb removal ran on."""
    from eeg_pipeline.preprocessing.report.filtering import (
        describe_filter,
        filter_response_html,
    )

    html = filter_response_html(
        describe_filter(sfreq=250.0, l_freq=1.0, h_freq=40.0),
        unavailable_intervals_by_recording={},
        subject="0001",
    )

    assert "upstream" not in html.lower()
    assert "Stopbands" not in html


def test_the_cardiac_guide_says_nothing_about_a_detector_it_never_ran() -> None:
    """A montage with no ECG lead reviews no beats, so the guide states no source."""
    from eeg_pipeline.preprocessing import ica_cardiac_report
    from eeg_pipeline.preprocessing.ica_cardiac_review import CardiacReviewSettings

    html = ica_cardiac_report._cardiac_review_guide_html(
        CardiacReviewSettings.from_mapping({}), beat_sources=()
    )

    assert "Analyzer" not in html.split("find_bads_ecg")[0]
    assert "does not depend on BrainVision Analyzer" not in html
