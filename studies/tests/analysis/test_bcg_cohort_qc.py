from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from studies.pain_study.analysis.bcg.cohort_qc import (  # noqa: E402
    MINIMUM_R_MARKERS_PER_VOLUME,
    cohort_marker_html,
    plot_cohort_markers,
    summarize_cohort_markers,
)


def _runs(counts):
    return pd.DataFrame(
        [
            {"subject": subject, "run": run, "n_r_markers": markers, "n_volumes": 500}
            for subject, run, markers in counts
        ]
    )


def test_a_run_without_markers_is_flagged_broken() -> None:
    qc = summarize_cohort_markers(_runs([("0001", 1, 450), ("0001", 2, 5)]))

    assert list(qc.runs["is_broken"]) == [False, True]
    assert qc.broken_fraction == pytest.approx(0.5)


def test_threshold_is_applied_relative_to_recording_length() -> None:
    """Marker count alone cannot judge a run; a long run needs proportionally more."""
    runs = pd.DataFrame(
        [
            {"subject": "0001", "run": 1, "n_r_markers": 200, "n_volumes": 200},
            {"subject": "0001", "run": 2, "n_r_markers": 200, "n_volumes": 900},
        ]
    )

    qc = summarize_cohort_markers(runs)

    assert list(qc.runs["is_broken"]) == [False, True]


def test_subjects_with_every_run_usable_are_identified() -> None:
    qc = summarize_cohort_markers(
        _runs([("0001", 1, 450), ("0001", 2, 450), ("0002", 1, 450), ("0002", 2, 3)])
    )

    assert qc.subjects_fully_usable() == ["0001"]


def test_missing_columns_fail_fast() -> None:
    with pytest.raises(ValueError, match="requires columns"):
        summarize_cohort_markers(pd.DataFrame({"subject": ["0001"], "run": [1]}))


def test_zero_volumes_fail_fast() -> None:
    runs = pd.DataFrame([{"subject": "0001", "run": 1, "n_r_markers": 10, "n_volumes": 0}])

    with pytest.raises(ValueError, match="positive volume count"):
        summarize_cohort_markers(runs)


def test_html_reports_the_cohort_fraction_and_affected_subjects() -> None:
    qc = summarize_cohort_markers(_runs([("0001", 1, 450), ("0002", 1, 5), ("0002", 2, 5)]))

    document = cohort_marker_html(qc)

    assert "3" in document
    assert "67%" in document
    assert "sub-0002" in document
    assert "sub-0001" not in document.split("Affected runs per subject")[-1]


def test_figure_marks_the_usability_threshold() -> None:
    qc = summarize_cohort_markers(_runs([("0001", 1, 450), ("0001", 2, 5)]))

    figure = plot_cohort_markers(qc)

    lines = figure.axes[1].get_lines()
    assert any(line.get_xdata()[0] == pytest.approx(MINIMUM_R_MARKERS_PER_VOLUME) for line in lines)
    assert "2 runs" in figure.axes[0].get_title() or "of 2 runs" in figure.axes[0].get_title()


def test_the_roll_up_can_be_restricted_to_a_processing_batch() -> None:
    """Mixing processed subjects with ones awaiting correction describes neither."""
    runs = _runs([("0001", 1, 450), ("0002", 1, 5), ("0003", 1, 5)])

    qc = summarize_cohort_markers(runs, subjects=["0001", "0002"])

    assert sorted(qc.runs["subject"]) == ["0001", "0002"]
    assert qc.broken_fraction == pytest.approx(0.5)


def test_requesting_an_absent_subject_fails_fast() -> None:
    runs = _runs([("0001", 1, 450)])

    with pytest.raises(ValueError, match="No runs found"):
        summarize_cohort_markers(runs, subjects=["0001", "9999"])


def test_an_empty_selection_fails_fast() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        summarize_cohort_markers(
            _runs([]).reindex(columns=["subject", "run", "n_r_markers", "n_volumes"])
        )
