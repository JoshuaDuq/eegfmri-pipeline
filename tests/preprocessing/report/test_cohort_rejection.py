"""Retention, and the failure a total cannot show.

The point of the section is that thirty percent lost overall and thirty percent lost from
one condition print the same headline and mean different things. These tests pin that the
per-condition breakdown survives into the panel, that a participant with no breakdown is
still counted in the totals, and that drop reasons are tallied with denominators rather
than turned into rates over a population that changes by row.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.rejection import (  # noqa: E402
    REASON_PREFIX,
    _has_condition_spread,
    cohort_rejection,
    condition_table,
    drop_reasons,
    plot_retention,
    reason_totals,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    Paradigm,
    SubjectSidecar,
)


def _conditions(counts: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"condition": name, "n_total": total, "n_kept": kept}
            for name, (total, kept) in sorted(counts.items())
        ]
    )


def _participant(
    subject: str,
    *,
    total: float | None = 120,
    kept: float | None = 100,
    conditions: dict[str, tuple[float, float]] | None = None,
    reasons: dict[str, int] | None = None,
    paradigm: Paradigm = Paradigm.TASK,
) -> SubjectSidecar:
    measurements: dict = {}
    if total is not None:
        measurements["epochs_total"] = total
    if kept is not None:
        measurements["epochs_kept"] = kept
    for reason, count in (reasons or {}).items():
        measurements[f"{REASON_PREFIX}{reason}"] = count
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        paradigm=paradigm,
        measurements=measurements,
        conditions=_conditions(conditions) if conditions else pd.DataFrame(),
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


def test_retention_is_reported_per_participant() -> None:
    rejection = cohort_rejection(_cohort(_participant("0014", total=120, kept=90)))

    assert rejection.participants.iloc[0]["retained"] == 0.75


def test_uneven_rejection_across_conditions_reaches_the_panel() -> None:
    """The failure a total cannot show: same headline, different experiment."""
    rejection = cohort_rejection(
        _cohort(
            _participant(
                "0014",
                total=120,
                kept=100,
                conditions={"painful": (60, 42), "neutral": (60, 58)},
            )
        )
    )

    row = rejection.participants.iloc[0]

    assert row["n_conditions"] == 2
    assert row["condition_range"] == (58 / 60) - (42 / 60)


def test_evenly_rejected_conditions_leave_a_small_range() -> None:
    rejection = cohort_rejection(
        _cohort(
            _participant(
                "0014", conditions={"painful": (60, 50), "neutral": (60, 50)}
            )
        )
    )

    assert rejection.participants.iloc[0]["condition_range"] == 0.0


def test_a_single_condition_has_no_range_rather_than_a_zero_one() -> None:
    """A design with one condition cannot be unbalanced, and did not measure that it was."""
    rejection = cohort_rejection(
        _cohort(_participant("0014", conditions={"painful": (120, 100)}))
    )

    assert np.isnan(rejection.participants.iloc[0]["condition_range"])


def test_a_participant_with_no_breakdown_still_counts_in_the_totals() -> None:
    rejection = cohort_rejection(
        _cohort(
            _participant("0014", conditions={"painful": (60, 50), "neutral": (60, 50)}),
            _participant("0015", conditions=None),
        )
    )

    assert set(rejection.participants["subject"]) == {"0014", "0015"}
    assert set(rejection.conditions["subject"]) == {"0014"}


def test_a_participant_without_recorded_totals_is_placed_from_its_conditions() -> None:
    rejection = cohort_rejection(
        _cohort(
            _participant(
                "0014",
                total=None,
                kept=None,
                conditions={"painful": (60, 42), "neutral": (60, 58)},
            )
        )
    )

    row = rejection.participants.iloc[0]

    assert row["n_total"] == 120
    assert row["retained"] == 100 / 120


def test_a_resting_state_cohort_has_no_section() -> None:
    """A recording with no trials has nothing to retain, and the section is absent."""
    assert cohort_rejection(_cohort(_participant("0014", paradigm=Paradigm.REST))) is None


def test_drop_reasons_are_read_rather_than_recomputed() -> None:
    participant = _participant("0014", reasons={"AUTOREJECT": 12, "BAD_boundary": 3})

    assert drop_reasons(participant) == {"AUTOREJECT": 12, "BAD_boundary": 3}


def test_a_reason_nobody_dropped_on_is_not_a_reason() -> None:
    assert drop_reasons(_participant("0014", reasons={"AUTOREJECT": 0})) == {}


def test_reasons_are_counted_with_their_own_participant_denominator() -> None:
    """A reason only one pipeline can produce is not one the others scored zero on."""
    totals = reason_totals(
        _cohort(
            _participant("0014", reasons={"AUTOREJECT": 12}),
            _participant("0015", reasons={"AUTOREJECT": 8, "BAD_boundary": 4}),
        )
    )

    indexed = totals.set_index("reason")

    assert indexed.loc["AUTOREJECT", "n_epochs"] == 20
    assert indexed.loc["AUTOREJECT", "n_participants"] == 2
    assert indexed.loc["BAD_boundary", "n_participants"] == 1


def test_each_condition_is_scored_against_who_recorded_it() -> None:
    html = condition_table(
        cohort_rejection(
            _cohort(
                _participant("0014", conditions={"painful": (60, 50), "warm": (60, 30)}),
                _participant("0015", conditions={"painful": (60, 54)}),
            )
        ),
        gates=BandGates(),
    )

    assert "painful" in html
    assert "warm" in html


def test_the_figure_draws_with_and_without_conditions() -> None:
    figure = plot_retention(
        cohort_rejection(
            _cohort(
                _participant("0014", conditions={"painful": (60, 42), "neutral": (60, 58)}),
                _participant("0015", conditions=None),
            )
        )
    )

    assert figure is not None
    matplotlib.pyplot.close("all")


def test_the_figure_is_withheld_where_there_is_no_spread_to_draw() -> None:
    """One condition has no inside for the conditions to sit in.

    The panel exists to show where a participant's conditions fall within their own range.
    With a single condition every row is one dot at the retained fraction, which is the
    table's own column drawn sideways.
    """
    single = cohort_rejection(
        _cohort(
            _participant("0014", conditions={"painful": (120, 100)}),
            _participant("0015", conditions={"painful": (120, 90)}),
        )
    )
    several = cohort_rejection(
        _cohort(_participant("0014", conditions={"painful": (60, 42), "neutral": (60, 58)}))
    )

    assert not _has_condition_spread(single)
    assert _has_condition_spread(several)


def test_a_cohort_with_no_conditions_at_all_draws_nothing_either() -> None:
    assert not _has_condition_spread(cohort_rejection(_cohort(_participant("0014"))))
