"""Measured evidence to a pooled cohort statistic, through every seam in between.

Each module is tested on its own elsewhere. This file is the one that would catch a
mismatch between them -- a column the converter names one thing and the schema another, a
context derived at write time and re-derived differently at read time, a denominator that
counts runs where it should count participants.

Two participants, because that is what is preprocessed today and because it exercises the
regime where the report must refuse to summarise.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    BandRegime,
    Contribution,
    align_grids,
    pool_participants_curve,
    pool_participants_scalar,
    pool_runs_rate,
)
from eeg_pipeline.preprocessing.report.cohort.collect import collect_cohort
from eeg_pipeline.preprocessing.report.cohort.record import build_subject_sidecar
from eeg_pipeline.preprocessing.report.cohort.sidecar import (
    Paradigm,
    write_sidecar,
)
from eeg_pipeline.preprocessing.report.continuity import RunContinuity
from eeg_pipeline.preprocessing.report.spectra import RunSpectra, StageSpectrum
from studies.pain_study.analysis.gradient.comb import VolumeTiming
from studies.pain_study.analysis.gradient.locked import VolumeLockedAverage

FREQUENCIES = np.arange(1.0, 41.0, 1.0)


def _stage(level: float) -> StageSpectrum:
    return StageSpectrum(
        median_db=np.full(FREQUENCIES.size, level),
        spread_low_db=np.full(FREQUENCIES.size, level - 2),
        spread_high_db=np.full(FREQUENCIES.size, level + 2),
        max_db=np.full(FREQUENCIES.size, level + 5),
        aperiodic=None,
    )


def _run_evidence(subject: str, run_index: int, *, level: float, flagged_s: float):
    recording_id = f"sub-{subject}_task-thermalactive_run-{run_index}"
    spectra = RunSpectra(
        recording_id=recording_id,
        frequencies=FREQUENCIES,
        before=_stage(level + 4.0),
        after=_stage(level),
        n_channels=63,
        fmax_reason="low-pass",
    )
    continuity = RunContinuity(
        recording_id=recording_id,
        window_seconds=1.0,
        times_s=np.arange(10.0),
        channel_names=("Cz",),
        relative_db=np.zeros((1, 10)),
        bad_spans=((0.0, flagged_s),),
        duration_s=600.0,
        event_onsets=(1.0, 2.0),
    )
    locked = VolumeLockedAverage(
        recording_id=recording_id,
        times_s=np.arange(5.0),
        before_rms_uv=np.full(5, 3.0),
        after_rms_uv=np.full(5, 1.0),
        n_volumes=300,
        before_locked_rms_uv=2.8,
        after_locked_rms_uv=1.0,
        before_noise_floor_uv=0.5,
        after_noise_floor_uv=0.4,
        before_excess_power_uv2=2.8**2 - 0.5**2,
        after_excess_power_uv2=(level / -20.0) ** 2,
    )
    return spectra, continuity, locked


def _write_participant(root, subject: str, *, level: float, n_runs: int, flagged_s: float):
    evidence = [
        _run_evidence(subject, index + 1, level=level, flagged_s=flagged_s)
        for index in range(n_runs)
    ]
    timing = VolumeTiming(n_volumes=300, repetition_time_s=2.0, interval_jitter_s=0.004)
    sidecar = build_subject_sidecar(
        subject=subject,
        task="thermalactive",
        spectra=[item[0] for item in evidence],
        continuity=[item[1] for item in evidence],
        measurements={"variance_removed": 0.80 + float(subject[-1]) / 100.0},
        versions={"mne": "1.12.1"},
    )
    report = root / f"sub-{subject}" / "eeg" / f"sub-{subject}_report.h5"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.touch()
    write_sidecar(report, sidecar)


@pytest.fixture
def cohort_root(tmp_path):
    """Two in-scanner task participants with unequal run counts."""
    _write_participant(tmp_path, "0014", level=-30.0, n_runs=6, flagged_s=30.0)
    _write_participant(tmp_path, "0015", level=-20.0, n_runs=2, flagged_s=0.0)
    return tmp_path


def test_the_whole_chain_produces_a_readable_cohort(cohort_root) -> None:
    cohort = collect_cohort(cohort_root, task="thermalactive")

    assert cohort.subjects == ("0014", "0015")
    assert cohort.not_aggregated == ()
    assert cohort.paradigms == (Paradigm.TASK,)


def test_the_paradigm_derived_at_write_time_survives_the_round_trip(cohort_root) -> None:
    """Derived once from the evidence, then read, never re-derived."""
    cohort = collect_cohort(cohort_root)

    for participant in cohort.participants:
        assert participant.paradigm is Paradigm.TASK
        assert participant.has_comb_evidence is False  # no comb resolved in this fixture


def test_two_participants_are_below_the_gate_and_get_no_summary(cohort_root) -> None:
    """The regime the report is in today, and it must say so rather than invent a median."""
    cohort = collect_cohort(cohort_root)
    pooled = pool_participants_scalar(
        [
            Contribution(
                subject=participant.subject,
                value=np.asarray([participant.measurements["variance_removed"]]),
                n_runs=participant.n_runs,
            )
            for participant in cohort.participants
        ]
    )

    assert pooled.regime is BandRegime.INDIVIDUALS
    assert pooled.median is None
    assert set(pooled.per_subject) == {"0014", "0015"}
    assert pooled.denominator.n_subjects == 2
    assert pooled.denominator.n_runs == 8


def test_a_participant_with_three_times_the_runs_does_not_weigh_three_times(
    cohort_root,
) -> None:
    """The denominator records the runs; the statistic does not weight by them."""
    cohort = collect_cohort(cohort_root)
    by_subject = {participant.subject: participant for participant in cohort.participants}

    assert by_subject["0014"].n_runs == 6
    assert by_subject["0015"].n_runs == 2

    curves = []
    for participant in cohort.participants:
        after = participant.spectrum_curves[participant.spectrum_curves["stage"] == "after"]
        # Runs first: each run is an independent estimate of the same spectrum.
        per_run = after.groupby("freq_hz")["median_db"].median()
        curves.append(
            Contribution(
                subject=participant.subject,
                value=per_run.to_numpy(),
                n_runs=participant.n_runs,
            )
        )

    pooled = pool_participants_curve(curves, grid=FREQUENCIES)

    # Midway between -30 and -20, not dragged toward the participant with six runs.
    assert pooled.per_subject["0014"][0] == pytest.approx(-30.0)
    assert pooled.per_subject["0015"][0] == pytest.approx(-20.0)
    assert pooled.regime is BandRegime.INDIVIDUALS


def test_flagged_time_pools_over_the_session_not_across_run_medians(cohort_root) -> None:
    """The participant with six runs flagged 30 s in each; the other flagged none."""
    cohort = collect_cohort(cohort_root)
    by_subject = {participant.subject: participant for participant in cohort.participants}

    runs = by_subject["0014"].runs
    flagged = pool_runs_rate(
        runs["flagged_fraction"] * runs["duration_s"],
        runs["duration_s"],
    )

    assert flagged == pytest.approx(30.0 / 600.0)


def test_the_participants_share_a_frequency_grid(cohort_root) -> None:
    """Pooling spectra is only legitimate when the bins are identical."""
    cohort = collect_cohort(cohort_root)
    grids = {
        participant.subject: np.sort(participant.spectrum_curves["freq_hz"].unique())
        for participant in cohort.participants
    }

    alignment = align_grids(grids)

    assert alignment.grid == pytest.approx(FREQUENCIES)
    assert alignment.limiting_subjects == ()
