from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.analyzer_qc import (  # noqa: E402
    analyzer_qc_html,
    load_analyzer_qc,
    plot_analyzer_qc,
)


def _write_qc(directory, task="pain", subject="0001", *, markers=True, attenuation=True):
    directory.mkdir(parents=True, exist_ok=True)
    if markers:
        pd.DataFrame(
            {
                "recording_id": [
                    f"sub-{subject}_task-{task}_run-1_eeg",
                    f"sub-{subject}_task-{task}_run-2_eeg",
                    "sub-9999_task-pain_run-1_eeg",
                ],
                "marker_count": [500, 0, 500],
                "median_bpm": [65.0, None, 70.0],
                "status": ["pass", "fail", "pass"],
                "error": ["", "no markers", ""],
            }
        ).to_csv(directory / f"task-{task}_desc-pulsemarkers_qc.tsv", sep="\t", index=False)
    if attenuation:
        pd.DataFrame(
            {
                "recording_id": [
                    f"sub-{subject}_task-{task}_run-1",
                    f"sub-{subject}_task-{task}_run-2",
                ],
                "before_rms_uv": [0.6, 12.0],
                "after_rms_uv": [0.3, 6.0],
                "is_fallback": [False, True],
            }
        ).to_csv(directory / f"task-{task}_desc-cardiacattenuation_qc.tsv", sep="\t", index=False)


def test_only_the_requested_subject_is_loaded(tmp_path) -> None:
    _write_qc(tmp_path)

    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    assert list(qc.runs["run"]) == ["1", "2"]
    assert "9999" not in "".join(str(value) for value in qc.runs.to_numpy().ravel())


def test_attenuation_in_decibels_is_derived(tmp_path) -> None:
    _write_qc(tmp_path)

    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    # Halving amplitude is 6.02 dB.
    assert qc.runs["attenuation_db"].iloc[0] == pytest.approx(6.0206, abs=1e-3)


def test_failed_and_fallback_runs_are_identified(tmp_path) -> None:
    _write_qc(tmp_path)

    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    assert qc.runs_outside_bounds == ("2",)
    assert qc.fallback_runs == ("2",)


def test_absent_qc_tables_produce_no_section(tmp_path) -> None:
    """A dataset corrected outside Analyzer must not get an empty section."""
    assert load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001") is None


def test_a_subject_absent_from_the_tables_produces_no_section(tmp_path) -> None:
    _write_qc(tmp_path)

    assert load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0002") is None


def test_only_one_table_present_still_loads(tmp_path) -> None:
    _write_qc(tmp_path, attenuation=False)

    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    assert list(qc.runs["run"]) == ["1", "2"]
    assert "attenuation_db" not in qc.runs.columns


def test_html_states_which_runs_are_affected_without_grading_them(tmp_path) -> None:
    """The report names the affected runs; judging them is the reviewer's job."""
    _write_qc(tmp_path)
    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    document = analyzer_qc_html(qc)

    assert "run(s) 2" in document.lower()
    assert "automated detection" in document
    for verdict in ("probably never", "very likely", "not trustworthy", "&#9888;"):
        assert verdict not in document
    # The bounds are reference values recorded beside the measurements. Wording that
    # grades a run against them ("did not pass") invites the reader to skip the number,
    # which is the whole failure mode this section exists to avoid.
    for grade in ("did not pass", "failed", "invalid", "unacceptable"):
        assert grade not in document.lower()


def test_figure_uses_a_log_amplitude_axis(tmp_path) -> None:
    """Residual amplitude differs by more than an order of magnitude between runs."""
    _write_qc(tmp_path)
    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    figure = plot_analyzer_qc(qc)

    assert figure.axes[0].get_yscale() == "log"
    # The fallback highlight is explained in the attenuation panel's legend, beside the
    # bars it marks, rather than in a title line above them.
    legend = figure.axes[1].get_legend()
    assert legend is not None
    assert any("fallback" in text.get_text() for text in legend.get_texts())


def test_figure_requires_amplitudes(tmp_path) -> None:
    _write_qc(tmp_path, attenuation=False)
    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    with pytest.raises(ValueError, match="before and after"):
        plot_analyzer_qc(qc)


# --------------------------------------------------------------------------------------
# Beat-locked residual: what the upstream pulse correction left behind
# --------------------------------------------------------------------------------------


def _cardiac_raw(
    *,
    residual_uv: float,
    with_markers: bool = True,
    with_qrs: bool = True,
    marked_until_s: float | None = None,
):
    """A run carrying a beat-locked EEG deflection of known amplitude.

    The deflection is identical on every EEG channel, so the across-channel RMS of the
    beat-locked average equals its amplitude and the test can assert on a known number.

    ``marked_until_s`` marks only the beats before that time, leaving the rest of the run
    beating but unmarked -- the shape of a run whose pulse correction ran for part of the
    recording and then lost the trace.
    """
    import mne
    import numpy as np

    sfreq = 200.0
    duration = 60.0
    times = np.arange(int(duration * sfreq)) / sfreq
    beats = np.arange(1.0, duration - 1.0, 1.0)
    eeg = np.zeros_like(times)
    ecg = np.zeros_like(times)
    for beat in beats:
        if residual_uv:
            eeg += residual_uv * 1e-6 * np.exp(-0.5 * ((times - beat - 0.2) / 0.03) ** 2)
        if with_qrs:
            ecg += 1.0e-3 * np.exp(-0.5 * ((times - beat) / 0.02) ** 2)
    info = mne.create_info(["Cz", "Pz", "Oz", "ECG"], sfreq, ["eeg", "eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(np.vstack([eeg, eeg, eeg, ecg]), info, verbose=False)
    if with_markers:
        marked = beats if marked_until_s is None else beats[beats < marked_until_s]
        raw.set_annotations(
            mne.Annotations(marked, np.zeros(marked.size), ["Pulse Artifact/R"] * marked.size)
        )
    return raw


def test_the_residual_recovers_an_injected_beat_locked_deflection() -> None:
    """The panel's whole claim is that it measures how much artifact is left."""
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    measured = compute_cardiac_residual(_cardiac_raw(residual_uv=20.0), recording_id="run-1")

    assert measured.residual_uv == pytest.approx(20.0, rel=0.25)


def test_a_corrected_run_measures_far_less_than_an_uncorrected_one() -> None:
    """The comparison the worklist is ordered by has to survive the estimator."""
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    corrected = compute_cardiac_residual(_cardiac_raw(residual_uv=1.0), recording_id="run-1")
    uncorrected = compute_cardiac_residual(_cardiac_raw(residual_uv=20.0), recording_id="run-2")

    assert uncorrected.residual_uv > 5 * corrected.residual_uv


def test_the_marker_train_is_recorded_as_the_beat_source() -> None:
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    measured = compute_cardiac_residual(_cardiac_raw(residual_uv=10.0), recording_id="run-1")

    assert measured.beat_source == "analyzer-markers"
    assert measured.marker_count == 58


def test_a_partially_covered_beat_train_reports_the_share_of_the_run_it_covers() -> None:
    """The residual is only as informative as the share of the run it was measured over.

    sub-0009 r3 scored 1.33 uV on the 44 markers Analyzer wrote and 19.76 uV on its true
    beat train: the correction worked at the beats it found and nowhere else. Read without
    the coverage beside it, the first number says the run is clean.
    """
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    measured = compute_cardiac_residual(
        _cardiac_raw(residual_uv=1.0, marked_until_s=35.0), recording_id="run-1"
    )

    assert measured.beat_train_coverage is not None
    assert measured.beat_train_coverage < 0.7


def test_a_fully_covered_beat_train_reports_near_complete_coverage() -> None:
    """Otherwise a low coverage would say nothing -- every run would carry one."""
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    measured = compute_cardiac_residual(_cardiac_raw(residual_uv=1.0), recording_id="run-1")

    assert measured.beat_train_coverage > 0.9


def test_coverage_is_reported_even_when_the_train_is_too_short_to_average() -> None:
    """The run with almost no markers is exactly the one whose coverage has to be visible.

    Below ``MINIMUM_RESIDUAL_BEATS`` the residual is withheld, and if coverage went with it
    the worklist would show a run with neither a number nor a reason.
    """
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    measured = compute_cardiac_residual(
        _cardiac_raw(residual_uv=20.0, marked_until_s=6.0), recording_id="run-1"
    )

    assert measured.residual_uv is None
    assert measured.beat_train_coverage is not None
    assert measured.beat_train_coverage < 0.2


def test_a_run_without_markers_falls_back_to_the_channel_and_says_so() -> None:
    """33 of 90 runs in this dataset have no markers; they still get a residual."""
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    measured = compute_cardiac_residual(
        _cardiac_raw(residual_uv=20.0, with_markers=False), recording_id="run-1"
    )

    assert measured.marker_count == 0
    assert measured.beat_source == "ecg-channel"
    assert measured.residual_uv == pytest.approx(20.0, rel=0.3)


def test_a_run_with_no_beat_train_at_all_measures_nothing_and_does_not_raise() -> None:
    """sub-0000 r5: no markers and no detectable QRS. The report-review stage must survive.

    Reported as a run with no measurement rather than as a run with no residual: an
    unmeasurable artifact and an absent one are not the same finding.
    """
    from eeg_pipeline.preprocessing.report.analyzer_qc import compute_cardiac_residual

    measured = compute_cardiac_residual(
        _cardiac_raw(residual_uv=0.0, with_markers=False, with_qrs=False),
        recording_id="run-5",
    )

    assert measured.residual_uv is None
    assert measured.marker_count == 0
