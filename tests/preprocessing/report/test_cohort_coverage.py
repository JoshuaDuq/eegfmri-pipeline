"""Coverage separates a broken cap from a difficult participant, or it is worthless.

The failures are all denominators. An electrode only half the cohort ever recorded, scored
against everybody, looks reliable. A topography drawn from a guessed montage looks correct
and is not. Both produce a figure a reader has no reason to doubt.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.coverage import (  # noqa: E402
    cohort_coverage,
    participant_table,
    plot_failure_topography,
    ranked_table,
)
from eeg_pipeline.preprocessing.report.cohort.record import channel_table  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    Paradigm,
    SubjectSidecar,
)

#: A small ring of electrodes with plausible head-surface positions, in metres.
POSITIONS = {
    "Fp1": (-0.03, 0.08, 0.02),
    "Fp2": (0.03, 0.08, 0.02),
    "C3": (-0.05, 0.0, 0.08),
    "C4": (0.05, 0.0, 0.08),
    "O1": (-0.03, -0.08, 0.02),
    "O2": (0.03, -0.08, 0.02),
}


def _channels(bad: dict[str, int] | None = None, *, names=None, shift: float = 0.0):
    chosen = {name: POSITIONS[name] for name in (names or POSITIONS)}
    moved = {
        name: (x + shift, y, z) for name, (x, y, z) in chosen.items()
    }
    rows = []
    for name, (x, y, z) in moved.items():
        rows.append(
            {
                "channel": name,
                "x": x,
                "y": y,
                "z": z,
                "n_runs_bad": (bad or {}).get(name, 0),
            }
        )
    return pd.DataFrame(rows)


def _participant(subject: str, channels: pd.DataFrame) -> SubjectSidecar:
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        paradigm=Paradigm.REST,
        channels=channels,
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


# --------------------------------------------------------------------------------------
# The channel table, at write time
# --------------------------------------------------------------------------------------


def test_a_channel_bad_in_two_runs_is_counted_twice() -> None:
    """A channel bad once in six runs and one bad in all six are different facts."""
    table = channel_table(
        positions={"Cz": (0.0, 0.0, 0.1), "Pz": (0.0, -0.03, 0.09)},
        bad_by_run={"run-1": ["Cz"], "run-2": ["Cz"], "run-3": []},
    )

    assert int(table.loc[table["channel"] == "Cz", "n_runs_bad"].iloc[0]) == 2
    assert int(table.loc[table["channel"] == "Pz", "n_runs_bad"].iloc[0]) == 0


def test_a_channel_without_a_finite_position_is_dropped() -> None:
    """An electrode of unknown position cannot be drawn and must not be invented."""
    table = channel_table(
        positions={"Cz": (0.0, 0.0, 0.1), "Ghost": (np.nan, np.nan, np.nan)},
        bad_by_run={},
    )

    assert table["channel"].tolist() == ["Cz"]


def test_no_positions_gives_an_empty_table_rather_than_an_error() -> None:
    assert channel_table(positions=None).empty


# --------------------------------------------------------------------------------------
# Denominators
# --------------------------------------------------------------------------------------


def test_each_electrode_is_scored_against_the_participants_that_recorded_it() -> None:
    """The failure this guards: a late-added electrode looking reliable."""
    coverage = cohort_coverage(
        _cohort(
            _participant("0014", _channels(names=["Fp1", "C3", "O1"])),
            _participant("0015", _channels({"O2": 3}, names=["Fp1", "C3", "O1", "O2"])),
        )
    )

    assert coverage is not None
    o2 = coverage.channels[coverage.channels["channel"] == "O2"].iloc[0]
    # Only one participant ever recorded O2, and it was bad for that one.
    assert int(o2["n_participants"]) == 1
    assert float(o2["bad_fraction"]) == pytest.approx(1.0)
    fp1 = coverage.channels[coverage.channels["channel"] == "Fp1"].iloc[0]
    assert int(fp1["n_participants"]) == 2


def test_an_electrode_bad_for_everybody_ranks_first() -> None:
    coverage = cohort_coverage(
        _cohort(
            _participant("0014", _channels({"O1": 6, "C3": 1})),
            _participant("0015", _channels({"O1": 6})),
            _participant("0016", _channels({"O1": 2})),
        )
    )

    assert coverage.channels.iloc[0]["channel"] == "O1"
    assert float(coverage.channels.iloc[0]["bad_fraction"]) == pytest.approx(1.0)


def test_a_cohort_with_no_channel_tables_has_no_section() -> None:
    bare = SubjectSidecar(
        subject="0014",
        task="rest",
        paradigm=Paradigm.REST,
    )

    assert cohort_coverage(_cohort(bare)) is None


def test_the_ranked_table_names_the_per_electrode_denominator() -> None:
    coverage = cohort_coverage(
        _cohort(_participant("0014", _channels({"O1": 4})), _participant("0015", _channels()))
    )

    html = ranked_table(coverage)

    assert "Recorded by" in html
    assert "not against the cohort" in html


def test_a_cohort_with_no_bad_channels_says_so_rather_than_ranking_nothing() -> None:
    coverage = cohort_coverage(
        _cohort(_participant("0014", _channels()), _participant("0015", _channels()))
    )

    assert "No electrode was marked bad" in ranked_table(coverage)


# --------------------------------------------------------------------------------------
# The topography, and when it is refused
# --------------------------------------------------------------------------------------


def test_agreeing_positions_support_a_topography() -> None:
    coverage = cohort_coverage(
        _cohort(_participant("0014", _channels({"O1": 3})), _participant("0015", _channels()))
    )

    assert coverage.has_positions
    assert coverage.disagreeing_subjects == ()


def test_a_participant_whose_sensors_sit_elsewhere_is_named_and_the_figure_refused() -> None:
    """A cohort that cannot be drawn correctly is not drawn."""
    coverage = cohort_coverage(
        _cohort(
            _participant("0014", _channels()),
            _participant("0015", _channels(shift=0.04)),
        )
    )

    assert not coverage.has_positions
    assert coverage.disagreeing_subjects == ("0015",)


def test_rounding_differences_do_not_count_as_disagreement() -> None:
    coverage = cohort_coverage(
        _cohort(
            _participant("0014", _channels()),
            _participant("0015", _channels(shift=1e-4)),
        )
    )

    assert coverage.disagreeing_subjects == ()
    assert coverage.has_positions


def test_the_topography_is_drawn_on_the_recorded_positions() -> None:
    coverage = cohort_coverage(
        _cohort(_participant("0014", _channels({"O1": 3})), _participant("0015", _channels()))
    )

    figure = plot_failure_topography(coverage)

    assert "per electrode" in figure.axes[0].get_title()
    matplotlib.pyplot.close(figure)


# --------------------------------------------------------------------------------------
# Per participant
# --------------------------------------------------------------------------------------


def test_participants_are_sorted_by_how_many_channels_they_lost() -> None:
    html = participant_table(
        _cohort(
            _participant("0014", _channels({"O1": 1})),
            _participant("0015", _channels({"O1": 1, "O2": 2, "C3": 1})),
        )
    )

    assert html.index("0015") < html.index("0014")


def test_a_participant_without_a_channel_table_is_absent_from_the_list() -> None:
    bare = SubjectSidecar(
        subject="0016",
        task="rest",
        paradigm=Paradigm.REST,
    )

    html = participant_table(_cohort(_participant("0014", _channels()), bare))

    assert "0016" not in html
