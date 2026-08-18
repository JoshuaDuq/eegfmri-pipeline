"""The participant-by-run grid, and the two ways it can lie.

A run a participant never contributed must not be drawn as a clean one, and a run's
position in the session has to come from its own label rather than from the order the runs
happened to be read -- otherwise a participant whose first run was dropped upstream gets
their second run aligned against everybody else's first, which manufactures exactly the
ordering effect the grid exists to detect.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.continuity import (  # noqa: E402
    add_continuity_section,
    cohort_continuity,
    flagged_audit,
    participant_totals,
    plot_flagged_time,
    run_position_table,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    Paradigm,
    SubjectSidecar,
)
from eeg_pipeline.preprocessing.report.tables import MISSING  # noqa: E402


def _participant(
    subject: str,
    *,
    flagged: dict[str, float],
    durations: dict[str, float] | None = None,
    excursion: dict[str, float] | None = None,
) -> SubjectSidecar:
    runs = sorted(flagged)
    durations = durations or {name: 600.0 for name in runs}
    excursion = excursion or {name: 5.0 + index for index, name in enumerate(runs)}
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        paradigm=Paradigm.TASK,
        runs=pd.DataFrame(
            {
                "run": [f"sub-{subject}_task-thermalactive_run-{name}" for name in runs],
                "n_channels": [63] * len(runs),
                "duration_s": [durations[name] for name in runs],
                "flagged_fraction": [flagged[name] for name in runs],
                "continuity_median_db": [0.2] * len(runs),
                "continuity_max_db": [excursion[name] for name in runs],
            }
        ),
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


def test_the_grid_is_participants_by_run_position() -> None:
    continuity = cohort_continuity(
        _cohort(
            _participant("0014", flagged={"1": 0.02, "2": 0.05}),
            _participant("0015", flagged={"1": 0.10, "2": 0.20}),
        )
    )

    assert continuity is not None
    assert list(continuity.grid.index) == ["0014", "0015"]
    assert continuity.run_positions == ("1", "2")
    assert continuity.n_runs == 4


def test_a_run_a_participant_never_ran_is_absent_rather_than_clean() -> None:
    """An unrecorded run and a spotless one are the same colour in any zero-filled grid."""
    continuity = cohort_continuity(
        _cohort(
            _participant("0014", flagged={"1": 0.02, "2": 0.05, "3": 0.30}),
            _participant("0015", flagged={"1": 0.10}),
        )
    )

    assert continuity is not None
    assert continuity.is_ragged
    assert np.isnan(continuity.grid.loc["0015", "3"])
    # The absent cells are not counted as runs that were measured.
    assert continuity.n_runs == 4


def test_a_run_keeps_its_own_position_when_an_earlier_one_is_missing() -> None:
    """A dropped first run must not slide the second into everybody else's first column."""
    continuity = cohort_continuity(
        _cohort(
            _participant("0014", flagged={"1": 0.02, "2": 0.40}),
            _participant("0015", flagged={"2": 0.45}),
        )
    )

    assert continuity is not None
    assert np.isnan(continuity.grid.loc["0015", "1"])
    assert continuity.grid.loc["0015", "2"] == 0.45


def test_run_positions_sort_numerically() -> None:
    continuity = cohort_continuity(
        _cohort(_participant("0014", flagged={"1": 0.01, "2": 0.02, "10": 0.03}))
    )

    assert continuity is not None
    assert continuity.run_positions == ("1", "2", "10")


def test_a_participant_total_weights_every_second_once() -> None:
    """A thirty-second run entirely flagged must not outvote a clean twelve-minute one."""
    totals = participant_totals(
        _cohort(
            _participant(
                "0014",
                flagged={"1": 1.0, "2": 0.0},
                durations={"1": 30.0, "2": 720.0},
            )
        )
    )

    assert totals["0014"] == 30.0 / 750.0
    # A median of the per-run fractions would have reported half the session lost.
    assert totals["0014"] < 0.05


def test_each_run_position_carries_its_own_participant_count() -> None:
    """A late run exists only for whoever ran one, and is scored against them alone."""
    continuity = cohort_continuity(
        _cohort(
            _participant("0014", flagged={"1": 0.02, "2": 0.05, "3": 0.30}),
            _participant("0015", flagged={"1": 0.10, "2": 0.20}),
        )
    )

    counts = {name: int(continuity.grid[name].notna().sum()) for name in continuity.run_positions}

    assert counts == {"1": 2, "2": 2, "3": 1}
    assert "run-3" in run_position_table(continuity, gates=BandGates())


def test_no_median_is_printed_below_the_gate() -> None:
    """A panel with too few contributors withholds the median rather than inventing one."""
    table = run_position_table(
        cohort_continuity(_cohort(_participant("0014", flagged={"1": 0.02, "2": 0.06}))),
        gates=BandGates(),
    )

    assert MISSING in table  # the median cell, withheld
    assert "2.0%" in table  # the range, which one participant does support


def test_a_grid_whose_every_cell_agrees_is_not_drawn_as_a_grid() -> None:
    """Flagged time counts BAD annotations, and a pipeline that writes none reports zero.

    Drawn anyway, that is one flat colour under a scale spanning nothing, which reads as a
    pristine cohort rather than as a column nothing populated. The measured excursion is
    taken for every run whatever was annotated, so the grid falls back to it.
    """
    cohort = _cohort(
        _participant("0014", flagged={"1": 0.0, "2": 0.0}, excursion={"1": 4.0, "2": 9.0}),
        _participant("0015", flagged={"1": 0.0, "2": 0.0}, excursion={"1": 5.0, "2": 6.0}),
    )

    continuity = cohort_continuity(cohort)

    assert continuity is not None
    assert continuity.measure == "continuity_max_db"
    assert not continuity.is_fraction
    assert continuity.grid.loc["0014", "2"] == 9.0


def test_flagged_time_is_preferred_when_anything_was_flagged() -> None:
    cohort = _cohort(
        _participant("0014", flagged={"1": 0.0, "2": 0.3}, excursion={"1": 4.0, "2": 9.0})
    )

    continuity = cohort_continuity(cohort)

    assert continuity.measure == "flagged_fraction"
    assert continuity.is_fraction


def test_a_cohort_with_nothing_that_varies_has_no_grid_at_all() -> None:
    cohort = _cohort(
        _participant("0014", flagged={"1": 0.0, "2": 0.0}, excursion={"1": 5.0, "2": 5.0})
    )

    assert cohort_continuity(cohort) is None


def test_the_audit_table_holds_exactly_the_drawn_cells() -> None:
    continuity = cohort_continuity(
        _cohort(
            _participant("0014", flagged={"1": 0.02, "2": 0.05}),
            _participant("0015", flagged={"1": 0.10}),
        )
    )

    audit = flagged_audit(continuity)

    assert list(audit.columns) == ["subject", "run", "flagged_fraction"]
    assert len(audit) == 3


def test_a_cohort_that_measured_nothing_produces_no_grid() -> None:
    empty = SubjectSidecar(
        subject="0014",
        task="thermalactive",
        paradigm=Paradigm.TASK,
    )

    assert cohort_continuity(_cohort(empty)) is None


def test_the_figure_draws_without_a_complete_grid() -> None:
    figure = plot_flagged_time(
        cohort_continuity(
            _cohort(
                _participant("0014", flagged={"1": 0.02, "2": 0.05, "3": 0.30}),
                _participant("0015", flagged={"1": 0.10}),
            )
        )
    )

    assert figure is not None
    matplotlib.pyplot.close("all")


def test_a_bare_run_label_does_not_collapse_every_run_into_one_column() -> None:
    """The grid silently kept only the last run when the id was not a full BIDS path.

    ``run_label`` looks for ``_run-``, so a bare ``run-2`` came back unrecognised and every
    run landed in a single column. A participant with six runs then read as one, and the
    ordering effect the grid exists to show was averaged away before it could be drawn.
    """
    participant = SubjectSidecar(
        subject="0014",
        task="thermalactive",
        paradigm=Paradigm.TASK,
        runs=pd.DataFrame(
            {
                "run": ["run-1", "run-2", "run-3"],
                "n_channels": [63] * 3,
                "duration_s": [600.0] * 3,
                "flagged_fraction": [0.01, 0.05, 0.20],
                "continuity_median_db": [0.2] * 3,
                "continuity_max_db": [4.0, 5.0, 6.0],
            }
        ),
    )

    continuity = cohort_continuity(_cohort(participant))

    assert continuity.run_positions == ("1", "2", "3")
    assert continuity.n_runs == 3


def _rest_participant(subject: str, recordings: dict[str, float]) -> SubjectSidecar:
    """A resting-state participant, whose recordings carry no ``run`` entity.

    BIDS omits the run entity where there is nothing to enumerate, which is the ordinary
    case for rest and for single-run EEG-only acquisitions.
    """
    names = sorted(recordings)
    return SubjectSidecar(
        subject=subject,
        task="rest",
        paradigm=Paradigm.REST,
        runs=pd.DataFrame(
            {
                "run": names,
                "n_channels": [63] * len(names),
                "duration_s": [600.0] * len(names),
                "flagged_fraction": [recordings[name] for name in names],
                "continuity_median_db": [0.2] * len(names),
                "continuity_max_db": [4.0 + index for index, _ in enumerate(names)],
            }
        ),
    )


def test_a_recording_with_no_run_entity_is_not_labelled_as_a_run() -> None:
    """"run-sub-01_task-rest_eeg" names nothing a reader or a filesystem recognises.

    ``_run_position`` gives up on an id with no run token and returns the id itself, which
    the panel then prefixed with "run-" regardless. Rest and single-run EEG-only datasets
    are exactly the ones BIDS lets omit the entity, so this is their ordinary path.
    """
    continuity = cohort_continuity(
        _cohort(
            _rest_participant(
                "0014",
                {"sub-0014_task-rest_acq-a_eeg": 0.02, "sub-0014_task-rest_acq-b_eeg": 0.30},
            ),
            _rest_participant(
                "0015",
                {"sub-0015_task-rest_acq-a_eeg": 0.05, "sub-0015_task-rest_acq-b_eeg": 0.40},
            ),
        )
    )

    assert continuity is not None
    figure = plot_flagged_time(continuity)
    labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]

    assert not any(label.startswith("run-sub-") for label in labels)
    assert "run-3" not in run_position_table(continuity, gates=BandGates())
    matplotlib.pyplot.close(figure)


def test_a_single_position_grid_is_reported_as_a_table_rather_than_a_grid() -> None:
    """One scalar per participant is a table; drawn as an image it is a strip plot.

    A rest cohort has one recording per participant, so the grid is one column wide. The
    section's whole argument is that reading *down* a column asks about the session design
    and reading *across* a row asks about a participant, and neither question exists here.
    """
    continuity = cohort_continuity(
        _cohort(
            _participant("0014", flagged={"1": 0.02}),
            _participant("0015", flagged={"1": 0.10}),
        )
    )

    assert continuity is not None
    assert continuity.is_single_position


def test_a_multi_run_grid_is_still_drawn_as_a_grid() -> None:
    continuity = cohort_continuity(
        _cohort(
            _participant("0014", flagged={"1": 0.02, "2": 0.05}),
            _participant("0015", flagged={"1": 0.10, "2": 0.20}),
        )
    )

    assert not continuity.is_single_position


def test_a_single_position_cohort_keeps_its_tables_and_drops_the_figure() -> None:
    """The section still has something to say; what it has is not a grid."""

    class FakeReport:
        def __init__(self) -> None:
            self.figures: list[str] = []
            self.html: list[str] = []

        def add_figure(self, *, fig, title, **kwargs) -> None:
            self.figures.append(title)

        def add_html(self, *, html, title, **kwargs) -> None:
            self.html.append(html)

    report = FakeReport()
    add_continuity_section(
        report=report,
        cohort=_cohort(
            _participant("0014", flagged={"1": 0.02}),
            _participant("0015", flagged={"1": 0.10}),
        ),
        gates=BandGates(),
    )

    assert report.figures == []
    assert report.html
    assert "10.0%" in "".join(report.html)


def test_run_less_recordings_share_a_column_rather_than_one_each() -> None:
    """Keyed on the recording id, the grid came out a diagonal matrix.

    A recording id carries the subject label, so ``sub-0014_task-rest_eeg`` and
    ``sub-0015_task-rest_eeg`` are different strings naming the same position in each
    participant's session. Keyed on them, every participant occupied a column of its own
    and the grid had exactly one filled cell per row -- which cannot be read down a column,
    which is the only reason the grid exists.
    """
    continuity = cohort_continuity(
        _cohort(
            _rest_participant("0014", {"sub-0014_task-rest_eeg": 0.02}),
            _rest_participant("0015", {"sub-0015_task-rest_eeg": 0.10}),
        )
    )

    assert continuity is not None
    assert continuity.grid.shape == (2, 1)
    assert continuity.is_single_position
    assert continuity.grid.notna().to_numpy().all()


def test_run_less_recordings_are_positioned_by_their_order_in_the_session() -> None:
    """Two recordings each, no run entity: two columns, both filled for both participants."""
    continuity = cohort_continuity(
        _cohort(
            _rest_participant(
                "0014",
                {"sub-0014_task-rest_acq-a_eeg": 0.02, "sub-0014_task-rest_acq-b_eeg": 0.30},
            ),
            _rest_participant(
                "0015",
                {"sub-0015_task-rest_acq-a_eeg": 0.05, "sub-0015_task-rest_acq-b_eeg": 0.40},
            ),
        )
    )

    assert continuity is not None
    assert continuity.grid.shape == (2, 2)
    assert continuity.grid.notna().to_numpy().all()
    assert not any(heading.startswith("run-") for heading in continuity.position_headings)
