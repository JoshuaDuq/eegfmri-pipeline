"""Marker agreement and cardiac physiology, pooled without inventing a threshold.

Three things are pinned. Agreement is a rate over the session, so a short run cannot
outvote a long one. A participant recorded outside a scanner has no pulse correction to
describe and must not appear in the denominator at all. And a session that agrees well
overall while one run does not has a broken run, which the pooled figure is precisely what
hides -- so the worst run travels beside the pooled value.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.analyzer import (  # noqa: E402
    PLAUSIBLE_BPM,
    analyzer_cohort,
    analyzer_table,
    implausible_note,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
)


def _participant(
    subject: str,
    *,
    in_scanner: bool = True,
    matched: list[float] | None = None,
    markers: list[float] | None = None,
    bpm: list[float] | None = None,
    beats: list[float] | None = None,
    dropouts: list[float] | None = None,
) -> SubjectSidecar:
    matched = [0.97, 0.96] if matched is None else matched
    markers = [600.0, 600.0] if markers is None else markers
    n_runs = len(markers)
    frame = pd.DataFrame(
        {
            "run": [f"sub-{subject}_task-x_run-{index + 1}" for index in range(n_runs)],
            "n_channels": [63] * n_runs,
            "duration_s": [600.0] * n_runs,
            "flagged_fraction": [0.02] * n_runs,
            "continuity_median_db": [0.2] * n_runs,
            "continuity_max_db": [5.0] * n_runs,
            "marker_matched_fraction": matched,
            "n_markers": markers,
            "median_bpm": bpm if bpm is not None else [62.0] * n_runs,
            "n_beats": beats if beats is not None else [600.0] * n_runs,
            "beat_dropouts": dropouts if dropouts is not None else [3.0] * n_runs,
        }
    )
    if in_scanner:
        frame["n_volumes"] = [300] * n_runs
        frame["repetition_time_s"] = [2.0] * n_runs
        frame["volume_locked_corrected_uv"] = [0.7] * n_runs
        frame["volume_locked_noise_floor_uv"] = [0.3] * n_runs
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=(
            AcquisitionContext.IN_SCANNER if in_scanner else AcquisitionContext.OUT_OF_SCANNER
        ),
        paradigm=Paradigm.TASK,
        runs=frame,
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


def test_agreement_is_a_rate_over_the_session() -> None:
    """A short run cannot outvote a long one; every presented marker counts once."""
    analyzer = analyzer_cohort(
        _cohort(_participant("0014", matched=[0.0, 1.0], markers=[10.0, 990.0]))
    )

    row = analyzer.frame.iloc[0]

    assert row["marker_agreement"] == 990.0 / 1000.0
    # A median of the per-run fractions would have reported 50%.
    assert row["marker_agreement"] > 0.9


def test_the_worst_run_travels_beside_the_pooled_figure() -> None:
    """A session that pools to 96% with one run at 40% has a broken run."""
    analyzer = analyzer_cohort(
        _cohort(_participant("0014", matched=[0.4, 0.99], markers=[20.0, 980.0]))
    )

    row = analyzer.frame.iloc[0]

    assert row["marker_agreement"] > 0.95
    assert row["worst_run_agreement"] == 0.4


def test_a_participant_outside_a_scanner_is_not_in_the_denominator() -> None:
    """A recording made outside a bore has no pulse correction to describe."""
    analyzer = analyzer_cohort(
        _cohort(_participant("0014"), _participant("0015", in_scanner=False))
    )

    assert list(analyzer.frame["subject"]) == ["0014"]
    assert analyzer.n_participants == 1


def test_a_cohort_entirely_outside_a_scanner_has_no_section() -> None:
    assert analyzer_cohort(_cohort(_participant("0014", in_scanner=False))) is None


def test_a_participant_without_an_ecg_channel_carries_no_agreement() -> None:
    """One detector rather than two is nothing to reconcile, not a total disagreement."""
    analyzer = analyzer_cohort(
        _cohort(
            _participant("0014"),
            _participant("0015", matched=[np.nan, np.nan], markers=[np.nan, np.nan]),
        )
    )

    assert analyzer.n_participants == 2
    assert len(analyzer.contributors("marker_agreement")) == 1
    assert np.isnan(analyzer.frame.set_index("subject").loc["0015", "marker_agreement"])


def test_an_impossible_heart_rate_is_stated_as_a_detector_failure() -> None:
    """Algebraic rather than empirical: that interval series is not a heart rate."""
    analyzer = analyzer_cohort(
        _cohort(
            _participant("0014"),
            _participant("0015", bpm=[PLAUSIBLE_BPM[1] + 40.0] * 2),
        )
    )

    note = implausible_note(analyzer)

    assert "0015" in note
    assert "0014" not in note
    assert "about the detector" in note


def test_a_plausible_cohort_says_so_without_grading_anyone() -> None:
    note = implausible_note(analyzer_cohort(_cohort(_participant("0014"))))

    assert "inside a physiologically possible" in note


def test_no_threshold_is_applied_to_agreement() -> None:
    """Participants are placed in the distribution, never graded against a cutoff."""
    html = analyzer_table(
        analyzer_cohort(
            _cohort(_participant("0014"), _participant("0015", matched=[0.2, 0.2]))
        )
    )

    assert "0015" in html
    for word in ("fail", "pass", "poor", "acceptable", "warning"):
        assert word not in html.lower()


def test_dropouts_are_pooled_over_the_beats_that_were_detected() -> None:
    analyzer = analyzer_cohort(
        _cohort(_participant("0014", beats=[100.0, 900.0], dropouts=[10.0, 0.0]))
    )

    assert analyzer.frame.iloc[0]["dropout_fraction"] == 10.0 / 1000.0


def test_the_section_carries_no_figure_because_it_would_restate_the_table() -> None:
    """Every quantity here is one number per participant.

    A strip plot of one number per participant is the sorted table drawn with dots, and it
    carries less: the table also holds the worst run, the dropout rate and the run count.
    """
    from eeg_pipeline.preprocessing.report.cohort import analyzer as module

    assert not hasattr(module, "plot_analyzer")


# --------------------------------------------------------------------------------------
# Re-export worklist: which runs the upstream pulse correction never ran on
# --------------------------------------------------------------------------------------


def _residual_participant(subject: str, runs: dict[int, tuple[int, float | None]]):
    """One in-scanner participant, ``run -> (marker_count, residual_uv)``."""
    import pandas as pd
    from eeg_pipeline.preprocessing.report.cohort.sidecar import (
        AcquisitionContext,
        Paradigm,
        SubjectSidecar,
    )

    order = sorted(runs)
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=AcquisitionContext.IN_SCANNER,
        paradigm=Paradigm.TASK,
        runs=pd.DataFrame(
            {
                "run": [f"sub-{subject}_task-thermalactive_run-{r}" for r in order],
                "n_channels": [63] * len(order),
                "duration_s": [600.0] * len(order),
                "flagged_fraction": [0.01] * len(order),
                "continuity_median_db": [0.2] * len(order),
                "continuity_max_db": [4.0] * len(order),
                "pulse_marker_count": [runs[r][0] for r in order],
                "beat_source": [
                    "analyzer-markers" if runs[r][0] else "ecg-channel" for r in order
                ],
                "bcg_residual_uv": [runs[r][1] for r in order],
            }
        ),
    )


def test_runs_without_a_marker_train_are_listed_for_re_export() -> None:
    """No markers means no subtraction was possible, whatever the residual came out at."""
    from eeg_pipeline.preprocessing.report.cohort.analyzer import uncorrected_runs
    from eeg_pipeline.preprocessing.report.cohort.collect import Cohort

    cohort = Cohort(
        participants=(
            _residual_participant("0014", {1: (498, 0.9), 6: (0, 20.2)}),
            _residual_participant("0010", {1: (0, 36.1), 2: (409, 9.6)}),
        )
    )

    listed = uncorrected_runs(cohort)

    assert {(r.subject, r.run) for r in listed} == {("0014", "6"), ("0010", "1")}
    # Ordered worst first, because that is the order the work gets done in.
    assert [r.subject for r in listed] == ["0010", "0014"]


def test_a_run_whose_residual_could_not_be_measured_is_still_listed() -> None:
    """sub-0000 r5: no markers and no beat train. It needs re-export most of all."""
    from eeg_pipeline.preprocessing.report.cohort.analyzer import uncorrected_runs
    from eeg_pipeline.preprocessing.report.cohort.collect import Cohort

    cohort = Cohort(
        participants=(_residual_participant("0000", {1: (681, 0.9), 5: (0, None)}),)
    )

    listed = uncorrected_runs(cohort)

    assert [(r.subject, r.run) for r in listed] == [("0000", "5")]
    assert listed[0].residual_uv is None


def test_consecutive_affected_runs_read_as_a_range() -> None:
    """A lead that came off at run 2 and stayed off is a different story from three
    scattered dropouts, and the compact range is what carries that in a table."""
    from eeg_pipeline.preprocessing.report.cohort.analyzer import participant_correction_rows
    from eeg_pipeline.preprocessing.report.cohort.collect import Cohort

    cohort = Cohort(
        participants=(
            _residual_participant(
                "0006", {1: (442, 0.8), 2: (0, 24.1), 3: (0, 22.0), 4: (0, 21.0),
                          5: (0, 20.0), 6: (0, 19.0)}
            ),
            _residual_participant(
                "0009", {1: (349, 1.0), 2: (0, 15.0), 3: (0, 14.0), 4: (514, 1.1),
                          5: (0, 13.0), 6: (325, 1.2)}
            ),
        )
    )

    rows = {r.subject: r for r in participant_correction_rows(cohort)}

    assert rows["0006"].affected_runs == "2-6"
    assert rows["0009"].affected_runs == "2, 3, 5"


def test_a_fully_corrected_cohort_has_no_worklist() -> None:
    from eeg_pipeline.preprocessing.report.cohort.analyzer import uncorrected_runs
    from eeg_pipeline.preprocessing.report.cohort.collect import Cohort

    cohort = Cohort(participants=(_residual_participant("0005", {1: (457, 0.8)}),))

    assert uncorrected_runs(cohort) == []


def test_a_cohort_with_no_markers_anywhere_still_gets_its_worklist() -> None:
    """The worst case must not be the one that renders nothing.

    With no marker train on any run there is no agreement to measure and no heart rate, so
    the agreement panels have nothing to say -- but "none of these runs was corrected" is
    the most important thing the section could report, and it was being skipped with them.
    """
    import mne

    from eeg_pipeline.preprocessing.report.cohort.analyzer import add_analyzer_section
    from eeg_pipeline.preprocessing.report.cohort.collect import Cohort

    cohort = Cohort(
        participants=(
            _residual_participant("0012", {1: (0, 30.0), 2: (0, 28.0)}),
            _residual_participant("0013", {1: (0, 26.0), 2: (0, 24.0)}),
        )
    )
    report = mne.Report(title="cohort", verbose="ERROR")

    add_analyzer_section(report=report, cohort=cohort)

    document = " ".join(element.html for element in report._content if element.html)
    assert "no Analyzer R-marker" in document
    assert "0012" in document and "0013" in document


def _rest_participant(subject: str, *, marker_count: int, residual_uv: float | None):
    """An in-scanner resting-state participant, whose recording carries no run entity.

    Baseline and resting recordings go in the bore too, and BIDS omits the run entity where
    there is nothing to enumerate. The pulse correction fails on them exactly as it does on
    task runs, so they belong in the worklist.
    """
    import pandas as pd
    from eeg_pipeline.preprocessing.report.cohort.sidecar import (
        AcquisitionContext,
        Paradigm,
        SubjectSidecar,
    )

    recording = f"sub-{subject}_task-rest_eeg"
    return SubjectSidecar(
        subject=subject,
        task="rest",
        context=AcquisitionContext.IN_SCANNER,
        paradigm=Paradigm.REST,
        runs=pd.DataFrame(
            {
                "run": [recording],
                "n_channels": [63],
                "duration_s": [600.0],
                "flagged_fraction": [0.01],
                "continuity_median_db": [0.2],
                "continuity_max_db": [4.0],
                "pulse_marker_count": [marker_count],
                "beat_source": ["analyzer-markers" if marker_count else "ecg-channel"],
                "bcg_residual_uv": [residual_uv],
            }
        ),
    )


def test_an_in_scanner_resting_state_recording_is_listed_for_re_export() -> None:
    """Rest in the bore has the same ballistocardiogram and the same failure mode."""
    from eeg_pipeline.preprocessing.report.cohort.analyzer import uncorrected_runs
    from eeg_pipeline.preprocessing.report.cohort.collect import Cohort

    cohort = Cohort(
        participants=(
            _rest_participant("0004", marker_count=0, residual_uv=27.0),
            _rest_participant("0005", marker_count=451, residual_uv=0.8),
        )
    )

    listed = uncorrected_runs(cohort)

    assert [row.subject for row in listed] == ["0004"]
    # Named by what identifies the recording, not by a run entity the dataset lacks.
    assert not listed[0].run.startswith("run-")
    assert "sub-0004" not in listed[0].run
    assert "rest" in listed[0].run


def test_an_out_of_scanner_participant_is_not_listed() -> None:
    """There is no ballistocardiogram outside a bore and no pulse correction to check."""
    from eeg_pipeline.preprocessing.report.cohort.analyzer import (
        participant_correction_rows,
        uncorrected_runs,
    )
    from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
    from eeg_pipeline.preprocessing.report.cohort.sidecar import (
        AcquisitionContext,
        Paradigm,
        SubjectSidecar,
    )
    import pandas as pd

    outside = SubjectSidecar(
        subject="0099",
        task="rest",
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=Paradigm.REST,
        runs=pd.DataFrame(
            {
                "run": ["sub-0099_task-rest_eeg"],
                "n_channels": [63],
                "duration_s": [600.0],
                "flagged_fraction": [0.01],
                "continuity_median_db": [0.2],
                "continuity_max_db": [4.0],
            }
        ),
    )

    cohort = Cohort(participants=(outside,))

    assert uncorrected_runs(cohort) == []
    assert participant_correction_rows(cohort) == []
