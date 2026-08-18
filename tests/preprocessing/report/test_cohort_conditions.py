"""The conditions table: what the sidecar carries about a task design.

Two counts per condition, presented and retained, and the failure this pins is that they
must not be inferred from one another. A retained count with no denominator is not a
fraction, and a condition that lost every trial has to survive into the table rather than
vanishing with its zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.report.cohort.record import (
    build_subject_sidecar,
    condition_table,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    CONDITION_COLUMNS,
    Paradigm,
    read_sidecar,
    sidecar_paths,
    write_sidecar,
)
from eeg_pipeline.preprocessing.report.continuity import RunContinuity
from eeg_pipeline.preprocessing.report.spectra import RunSpectra, StageSpectrum

RUN = "sub-0014_task-thermalactive_run-1"
FREQUENCIES = np.asarray([1.0, 2.0, 3.0])


def _stage(level: float) -> StageSpectrum:
    return StageSpectrum(
        median_db=np.full(3, level),
        spread_low_db=np.full(3, level - 2),
        spread_high_db=np.full(3, level + 2),
        max_db=np.full(3, level + 5),
        aperiodic=None,
    )


def _spectra(recording_id: str = RUN) -> RunSpectra:
    return RunSpectra(
        recording_id=recording_id,
        frequencies=FREQUENCIES,
        before=_stage(-10.0),
        after=_stage(-14.0),
        n_channels=63,
        fmax_reason="low-pass",
    )


def _continuity(recording_id: str = RUN, *, events=(1.0, 2.0)) -> RunContinuity:
    return RunContinuity(
        recording_id=recording_id,
        window_seconds=1.0,
        times_s=np.arange(10.0),
        channel_names=("Cz",),
        relative_db=np.zeros((1, 10)) + 3.0,
        bad_spans=((0.0, 30.0),),
        duration_s=600.0,
        event_onsets=tuple(events),
    )


def _sidecar(**kwargs):
    return build_subject_sidecar(
        subject="0014",
        task="thermalactive",
        spectra=[_spectra()],
        continuity=[_continuity()],
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------------------


def test_both_counts_travel_together() -> None:
    frame = condition_table(
        total_by_condition={"painful": 60, "neutral": 60},
        kept_by_condition={"painful": 41, "neutral": 57},
    )

    assert list(frame.columns) == list(CONDITION_COLUMNS)
    painful = frame[frame["condition"] == "painful"].iloc[0]
    assert painful["n_total"] == 60
    assert painful["n_kept"] == 41


def test_a_condition_that_lost_every_trial_keeps_its_row() -> None:
    """The cell a cohort most needs to see must not vanish with its zero."""
    frame = condition_table(
        total_by_condition={"painful": 60, "neutral": 60},
        kept_by_condition={"painful": 60, "neutral": 0},
    )

    neutral = frame[frame["condition"] == "neutral"].iloc[0]
    assert neutral["n_total"] == 60
    assert neutral["n_kept"] == 0


def test_a_condition_named_by_only_one_side_still_gets_a_row() -> None:
    frame = condition_table(total_by_condition={"painful": 60}, kept_by_condition={"warm": 12})

    assert set(frame["condition"]) == {"painful", "warm"}
    assert np.isnan(frame[frame["condition"] == "painful"].iloc[0]["n_kept"])
    assert np.isnan(frame[frame["condition"] == "warm"].iloc[0]["n_total"])


def test_no_conditions_produces_the_shape_rather_than_nothing() -> None:
    """A consumer filters an empty frame instead of testing for one."""
    frame = condition_table()

    assert frame.empty
    assert list(frame.columns) == list(CONDITION_COLUMNS)


# --------------------------------------------------------------------------------------
# Through the sidecar
# --------------------------------------------------------------------------------------


def test_a_task_participant_carries_its_conditions(tmp_path) -> None:
    report = tmp_path / "sub-0014_report.h5"
    written = _sidecar(
        trials_by_condition={"painful": 60, "neutral": 60},
        retained_by_condition={"painful": 41, "neutral": 57},
    )

    write_sidecar(report, written)
    read = read_sidecar(report)

    assert read.paradigm is Paradigm.TASK
    assert read.has_condition_evidence
    assert set(read.conditions["condition"]) == {"painful", "neutral"}


def test_a_resting_state_participant_carries_none(tmp_path) -> None:
    """Conditions belong to a paradigm that has them, whatever the caller passed."""
    report = tmp_path / "sub-0014_report.h5"
    written = build_subject_sidecar(
        subject="0014",
        task="rest",
        spectra=[_spectra("sub-0014_task-rest_run-1")],
        continuity=[_continuity("sub-0014_task-rest_run-1", events=())],
        trials_by_condition={"painful": 60},
    )

    assert written.paradigm is Paradigm.REST
    assert written.conditions.empty
    write_sidecar(report, written)
    assert not sidecar_paths(report).conditions.exists()
    assert not read_sidecar(report).has_condition_evidence


def test_a_participant_without_conditions_writes_no_file(tmp_path) -> None:
    """A zero-row file on disk must not be confused with a design that had conditions."""
    report = tmp_path / "sub-0014_report.h5"

    write_sidecar(report, _sidecar())

    assert not sidecar_paths(report).conditions.exists()
    assert read_sidecar(report).conditions.empty


def test_a_conditions_file_missing_a_column_is_refused(tmp_path) -> None:
    report = tmp_path / "sub-0014_report.h5"
    write_sidecar(report, _sidecar(trials_by_condition={"painful": 60}))
    paths = sidecar_paths(report)
    pd.DataFrame({"condition": ["painful"]}).to_csv(paths.conditions, sep="\t", index=False)

    with pytest.raises(ValueError, match="n_total"):
        read_sidecar(report)
