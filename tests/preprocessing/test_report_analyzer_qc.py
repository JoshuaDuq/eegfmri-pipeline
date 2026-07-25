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

    assert qc.failed_runs == ("2",)
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

    assert "run(s) 2" in document
    assert "automated detection" in document
    for verdict in ("probably never", "very likely", "not trustworthy", "&#9888;"):
        assert verdict not in document


def test_figure_uses_a_log_amplitude_axis(tmp_path) -> None:
    """Residual amplitude differs by more than an order of magnitude between runs."""
    _write_qc(tmp_path)
    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    figure = plot_analyzer_qc(qc)

    assert figure.axes[0].get_yscale() == "log"
    assert "fallback" in figure.axes[1].get_title()


def test_figure_requires_amplitudes(tmp_path) -> None:
    _write_qc(tmp_path, attenuation=False)
    qc = load_analyzer_qc(qc_dir=tmp_path, task="pain", subject="0001")

    with pytest.raises(ValueError, match="before and after"):
        plot_analyzer_qc(qc)
