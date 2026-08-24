"""Counting extremes, with the denominator that makes the count readable.

The table is only defensible if three things hold. It does not render where a decile does
not exist. A participant is scored only on metrics they actually had, so never having had a
chance is not the same as having passed. And a metric everyone agreed on is not scored at
all, because it has no outer decile and scoring it would make every participant extreme on
a measurement that separated nobody.
"""

from __future__ import annotations

import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import BandGates
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.multiplicity import (
    CHANNEL_FAMILY,
    ICA_FAMILY,
    METRIC_SOURCES,
    PHYSIOLOGY_FAMILY,
    _outer_members,
    multiplicity,
    multiplicity_table,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    Paradigm,
    SubjectSidecar,
)


def _participant(
    subject: str,
    *,
    variance: float,
    components: float = 62.0,
    flagged: float = 0.02,
    bpm: float | None = None,
) -> SubjectSidecar:
    runs = pd.DataFrame(
        {
            "run": [f"sub-{subject}_task-x_run-1"],
            "n_channels": [63],
            "duration_s": [600.0],
            "flagged_fraction": [flagged],
            "continuity_median_db": [0.2],
            "continuity_max_db": [5.0 + flagged],
        }
    )
    # A participant with no ECG lead has no physiology metric to be scored on, which is
    # the case the counting has to distinguish from having been measured and passed.
    if bpm is not None:
        runs["median_bpm"] = [bpm]
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        paradigm=Paradigm.TASK,
        measurements={
            "variance_removed": variance,
            "n_components": components,
            "n_excluded": components / 2.0,
            "retained_dimensions": components / 3.0,
        },
        runs=runs,
    )


def _spread_cohort(n: int = 12, **overrides) -> Cohort:
    """A cohort whose metrics vary, so a decile exists to be inside or outside of."""
    participants = []
    for index in range(n):
        subject = f"{index:04d}"
        participants.append(
            _participant(
                subject,
                variance=0.5 + index * 0.02,
                components=60.0 + index,
                flagged=0.01 + index * 0.005,
                **overrides,
            )
        )
    return Cohort(participants=tuple(participants))


def test_nothing_renders_below_the_outer_band_gate() -> None:
    """With nine participants the tenth percentile lies outside the observed values."""
    assert multiplicity(_spread_cohort(n=9)) is None


def test_the_table_renders_once_the_gate_is_met() -> None:
    result = multiplicity(_spread_cohort(n=10))

    assert result is not None
    assert result.n_participants == 10


def test_each_cell_is_a_count_over_the_metrics_that_family_had() -> None:
    result = multiplicity(_spread_cohort())

    per_family = result.counts.groupby("family")["n_metrics"].max()

    assert per_family[CHANNEL_FAMILY] >= 1
    assert per_family[ICA_FAMILY] >= 1
    assert (result.counts["n_extreme"] <= result.counts["n_metrics"]).all()


def test_the_extremes_are_the_participants_at_the_ends() -> None:
    result = multiplicity(_spread_cohort())
    counted = result.counts.groupby("subject")["n_extreme"].sum()

    # The cohort is built as a monotone ramp, so the ends hold every extreme and the
    # middle holds none.
    assert counted["0000"] > 0
    assert counted["0011"] > 0
    assert counted["0006"] == 0


def test_a_metric_everyone_agreed_on_is_not_scored() -> None:
    """It has no outer decile, and scoring it would make everybody extreme."""
    identical = Cohort(
        participants=tuple(
            _participant(f"{index:04d}", variance=0.8, components=62.0, flagged=0.02)
            for index in range(12)
        )
    )

    assert multiplicity(identical) is None


def test_a_participant_is_not_scored_on_a_metric_it_never_had() -> None:
    """Never having had the chance is not the same as having passed."""
    mixed = Cohort(
        participants=tuple(
            _participant(
                f"{index:04d}",
                variance=0.5 + index * 0.02,
                components=60.0 + index,
                flagged=0.01 + index * 0.005,
                bpm=None if index >= 10 else 55.0 + index * 2.0,
            )
            for index in range(12)
        )
    )

    result = multiplicity(mixed)
    physiology_rows = result.counts[result.counts["family"] == PHYSIOLOGY_FAMILY]

    assert set(physiology_rows["subject"]) == {f"{index:04d}" for index in range(10)}


def test_the_table_reads_as_a_count_over_a_denominator() -> None:
    html = multiplicity_table(multiplicity(_spread_cohort()))

    assert " / " in html
    for word in ("outlier", "fail", "suspicious", "warning"):
        assert word not in html.lower()


def test_a_stricter_gate_withholds_the_table() -> None:
    assert (
        multiplicity(_spread_cohort(n=12), gates=BandGates(min_subjects_for_outer_band=20)) is None
    )


def test_a_metric_most_of_the_cohort_ties_on_does_not_flag_most_of_the_cohort() -> None:
    """The regression this table exists to avoid, reproduced in the table itself.

    Comparing against the tenth percentile puts that percentile *at* the tied value when
    most participants share it, and every one of them then reads as "in the outer decile".
    On a bad-channel count where eight of twelve participants have none, that flagged ten
    of the twelve -- a panel written to stop manufactured suspicion manufacturing it
    wholesale.
    """
    measured = {f"{index:04d}": value for index, value in enumerate([0] * 8 + [1, 2, 3, 4])}

    outer = _outer_members(measured)

    assert len(outer) <= 2
    assert outer == {"0010", "0011"}


def test_the_count_per_tail_is_bounded_by_the_decile_itself() -> None:
    measured = {f"{index:04d}": float(index) for index in range(20)}

    outer = _outer_members(measured)

    # ceil(0.1 * 20) = 2 at each end, never more.
    assert len(outer) == 4


def test_a_tied_group_too_large_for_the_budget_is_excluded_entirely() -> None:
    """Admitting half of a tied group would be a statement about sort order."""
    measured = {
        f"{index:04d}": value for index, value in enumerate([1, 1, 1, 4, 5, 6, 7, 8, 9, 10])
    }

    outer = _outer_members(measured)

    assert "0000" not in outer
    assert "0009" in outer


def test_the_repetition_time_is_not_a_quality_metric() -> None:
    """A participant scanned under another protocol is a design fact, not an extreme."""
    assert all(source.key != "repetition_time_s" for source in METRIC_SOURCES)


def test_marker_agreement_pools_primitive_counts_across_runs() -> None:
    source = next(source for source in METRIC_SOURCES if source.key == "marker_matched_fraction")
    participant = SubjectSidecar(
        subject="0001",
        task="x",
        paradigm=Paradigm.TASK,
        runs=pd.DataFrame(
            {
                "marker_matched_fraction": [1.0, 0.5],
                "n_matched_beats": [1, 50],
                "n_detected_beats": [1, 100],
            }
        ),
    )

    assert source.read(participant) == 51 / 101
