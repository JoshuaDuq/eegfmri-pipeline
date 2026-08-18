"""Pooling harmonics by frequency across participants scanned at different repetition
times silently mixes one participant's third harmonic with another's fourth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    Paradigm,
    SubjectSidecar,
)
from studies.pain_study.analysis.gradient.cohort import (
    cohort_comb,
    comb_attenuation,
    comb_audit,
    participant_comb,
)

INDICES = np.asarray([30, 31, 32, 33])


def _comb_curves(
    *,
    n_runs: int,
    before_db: float,
    after_db: float,
    repetition_time_s: float,
    indices: np.ndarray = INDICES,
) -> pd.DataFrame:
    frames = []
    for run in range(n_runs):
        frames.append(
            pd.DataFrame(
                {
                    "run": f"run-{run + 1}",
                    "harmonic_index": indices,
                    "harmonic_hz": indices / repetition_time_s,
                    "notched": False,
                    "before_excess_db_median": np.full(indices.size, before_db),
                    "before_excess_db_max": np.full(indices.size, before_db + 6.0),
                    "after_excess_db_median": np.full(indices.size, after_db),
                    "after_excess_db_max": np.full(indices.size, after_db + 3.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _runs(*, n_runs: int, repetition_time_s: float, jitter_s: float = 0.002) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": [f"run-{index + 1}" for index in range(n_runs)],
            "n_channels": [63] * n_runs,
            "duration_s": [600.0] * n_runs,
            "flagged_fraction": [0.01] * n_runs,
            "continuity_median_db": [0.2] * n_runs,
            "continuity_max_db": [4.0] * n_runs,
            "n_volumes": [300] * n_runs,
            "repetition_time_s": [repetition_time_s] * n_runs,
            "volume_jitter_s": [jitter_s] * n_runs,
            "volume_locked_rms_before_uv": [np.sqrt(4.0 + 0.25)] * n_runs,
            "volume_locked_floor_before_uv": [0.5] * n_runs,
            "volume_locked_excess_power_before_uv2": [4.0] * n_runs,
            "volume_locked_resolved_before": [True] * n_runs,
            "volume_locked_rms_after_uv": [np.sqrt(0.64 + 0.09)] * n_runs,
            "volume_locked_floor_after_uv": [0.3] * n_runs,
            "volume_locked_excess_power_after_uv2": [0.64] * n_runs,
            "volume_locked_resolved_after": [True] * n_runs,
        }
    )


def _participant(
    subject: str,
    *,
    before_db: float = 14.0,
    after_db: float = 2.0,
    repetition_time_s: float = 2.0,
    n_runs: int = 2,
    jitter_s: float = 0.002,
    indices: np.ndarray = INDICES,
    with_comb: bool = True,
) -> SubjectSidecar:
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        paradigm=Paradigm.TASK,
        runs=_runs(n_runs=n_runs, repetition_time_s=repetition_time_s, jitter_s=jitter_s),
        comb_curves=(
            _comb_curves(
                n_runs=n_runs,
                before_db=before_db,
                after_db=after_db,
                repetition_time_s=repetition_time_s,
                indices=indices,
            )
            if with_comb
            else pd.DataFrame()
        ),
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


# --------------------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------------------


def test_a_participants_runs_are_pooled_before_it_joins_the_cohort() -> None:
    pooled = participant_comb(_participant("0014", n_runs=6))

    assert len(pooled) == INDICES.size
    assert pooled["after_excess_db_median"].tolist() == [2.0] * INDICES.size


def test_a_participant_without_a_resolved_comb_contributes_nothing() -> None:
    assert participant_comb(_participant("0014", with_comb=False)) is None


def test_a_cohort_where_nobody_resolved_a_comb_has_no_panel() -> None:
    """An ordinary outcome: the measurement declines on frequency resolution."""
    assert cohort_comb(_cohort(_participant("0014", with_comb=False))) is None


def test_a_shared_repetition_time_is_drawn_against_frequency() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert comb is not None
    assert comb.on_frequency_axis
    assert comb.harmonic_hz == pytest.approx(INDICES / 2.0)


def test_mixed_repetition_times_fall_back_to_the_harmonic_index() -> None:
    """The failure this guards: pooling one participant's third harmonic with another's."""
    comb = cohort_comb(
        _cohort(
            _participant("0014", repetition_time_s=2.0),
            _participant("0015", repetition_time_s=1.5),
        )
    )

    assert comb is not None
    assert not comb.on_frequency_axis
    assert comb.harmonic_hz is None
    assert comb.harmonic_index == pytest.approx(INDICES)


def test_participants_covering_different_harmonics_are_restricted_to_the_shared_set() -> None:
    comb = cohort_comb(
        _cohort(
            _participant("0014", indices=np.asarray([30, 31, 32, 33])),
            _participant("0015", indices=np.asarray([32, 33, 34, 35])),
        )
    )

    assert comb is not None
    assert comb.harmonic_index == pytest.approx(np.asarray([32.0, 33.0]))


def test_every_participant_is_kept_beneath_the_summary() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert set(comb.after.per_subject) == {"0014", "0015"}


def test_two_participants_get_no_cohort_median() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert comb.after.median is None
    assert comb.after.denominator.n_subjects == 2


# --------------------------------------------------------------------------------------
# Attenuation
# --------------------------------------------------------------------------------------


def test_attenuation_is_a_within_participant_difference() -> None:
    attenuation = comb_attenuation(
        cohort_comb(
            _cohort(
                _participant("0014", before_db=14.0, after_db=2.0),
                _participant("0015", before_db=20.0, after_db=11.0),
            )
        )
    )

    assert attenuation["0014"] == pytest.approx(12.0)
    assert attenuation["0015"] == pytest.approx(9.0)


def test_a_participant_whose_comb_survived_shows_little_attenuation() -> None:
    attenuation = comb_attenuation(
        cohort_comb(_cohort(_participant("0014", before_db=14.0, after_db=13.0)))
    )

    assert attenuation["0014"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------


def test_the_audit_holds_every_plotted_value() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    audit = comb_audit(comb)

    assert len(audit) == INDICES.size
    assert "before_0014_db" in audit.columns
    assert "after_0015_db" in audit.columns
    assert audit["n_subjects"].tolist() == [2] * INDICES.size


def test_the_audit_records_no_median_where_none_was_drawn() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert comb_audit(comb)["after_median_db"].isna().all()
