"""Residual scanner gradient comb, pooled across a cohort.

The comb view is the excess of each line over the background beside it, before and after
the ICA exclusions, pooled participant-first: before and after are two measurements of one
participant, so their comparison is paired rather than drawn as independent distributions,
which would carry the larger between-participant spread instead of what cleaning removed.

The axis is chosen rather than assumed. Harmonics sit at multiples of the volume rate, so
a cohort scanned at one repetition time can be drawn against frequency, and a cohort that
mixes repetition times cannot: one participant's third harmonic and another's fourth would
land in the same bin. Where they differ, the axis becomes the harmonic index, which is the
thing the participants actually share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    DEFAULT_GATES,
    BandGates,
    CohortCurve,
    Contribution,
    align_grids,
    paired_differences,
    pool_participants_curve,
)
from eeg_pipeline.preprocessing.report.cohort.collect import Cohort
from eeg_pipeline.preprocessing.report.cohort.sidecar import SubjectSidecar

#: Repetition times within this of each other are treated as one rate, so that ordinary
#: measurement scatter in the marker train does not force a cohort onto the index axis.
# Was report.thresholds.repetition_time_tolerance_s in the core config. It describes a
# scanner's marker train, so it moved here with the rest of the gradient settings;
# scripts/gradient/config.yaml carries the same value for the workflow side.
REPETITION_TIME_TOLERANCE_S = 1e-3


@dataclass(frozen=True)
class CohortComb:
    """The comb either side of the exclusions, on an axis the participants share."""

    before: CohortCurve
    after: CohortCurve
    harmonic_index: np.ndarray
    #: Harmonic frequencies, when every contributing participant shares a repetition
    #: time. ``None`` when they do not, in which case the index is the only honest axis.
    harmonic_hz: np.ndarray | None
    repetition_times_s: tuple[float, ...]

    @property
    def on_frequency_axis(self) -> bool:
        return self.harmonic_hz is not None


def participant_comb(participant: SubjectSidecar) -> pd.DataFrame | None:
    """One participant's comb, pooled across its runs.

    An unweighted median across runs: each run is an independent estimate of the same
    comb, and a longer run does not measure a participant's gradient residual more
    correctly, only more precisely.
    """
    curves = participant.comb_curves
    if curves.empty:
        return None
    numeric = curves.copy()
    for column in (
        "harmonic_index",
        "harmonic_hz",
        "before_excess_db_median",
        "after_excess_db_median",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    # A harmonic inside the notch stopband measures the filter rather than the residual,
    # at something like -25 dB. Left in, it is the deepest excursion in the figure, sets
    # the whole vertical scale, and reads as the correction having worked spectacularly at
    # exactly the frequency where it was never tested. The subject panel already withholds
    # these; the cohort has to withhold the same ones.
    if "notched" in numeric.columns:
        keep = ~numeric["notched"].astype(str).str.lower().isin({"true", "1", "1.0"})
        numeric = numeric[keep]
    if numeric.empty:
        return None
    pooled = numeric.groupby("harmonic_index", as_index=False).median(numeric_only=True)
    return pooled.sort_values("harmonic_index").reset_index(drop=True)


def _repetition_times(cohort: Cohort) -> tuple[float, ...]:
    values: list[float] = []
    for participant in cohort.participants:
        if "repetition_time_s" not in participant.runs.columns:
            continue
        rates = pd.to_numeric(participant.runs["repetition_time_s"], errors="coerce").dropna()
        if not rates.empty:
            values.append(float(rates.median()))
    return tuple(values)


def cohort_comb(
    cohort: Cohort,
    *,
    gates: BandGates = DEFAULT_GATES,
    repetition_time_tolerance_s: float = REPETITION_TIME_TOLERANCE_S,
) -> CohortComb | None:
    """Pool the comb across participants, choosing the axis they can share.

    Returns ``None`` when no participant resolved a comb, which is an ordinary outcome:
    the measurement declines where the frequency resolution cannot separate a line from
    its background, and that is a property of the volume rate and the run length.
    """
    combs = {
        participant.subject: pooled
        for participant in cohort.participants
        if (pooled := participant_comb(participant)) is not None
    }
    if not combs:
        return None

    alignment = align_grids(
        {subject: frame["harmonic_index"].to_numpy(dtype=float) for subject, frame in combs.items()}
    )
    runs_by_subject = {
        participant.subject: participant.n_runs for participant in cohort.participants
    }

    def _contributions(column: str) -> list[Contribution]:
        return [
            Contribution(
                subject=subject,
                value=frame.loc[alignment.masks[subject], column].to_numpy(dtype=float),
                n_runs=runs_by_subject.get(subject, 1),
            )
            for subject, frame in combs.items()
        ]

    rates = _repetition_times(cohort)
    shared_rate = bool(rates) and float(np.ptp(rates)) <= repetition_time_tolerance_s
    harmonic_hz = None
    if shared_rate:
        reference = combs[sorted(combs)[0]]
        harmonic_hz = reference.loc[
            alignment.masks[sorted(combs)[0]], "harmonic_hz"
        ].to_numpy(dtype=float)

    return CohortComb(
        before=pool_participants_curve(
            _contributions("before_excess_db_median"), grid=alignment.grid, gates=gates
        ),
        after=pool_participants_curve(
            _contributions("after_excess_db_median"), grid=alignment.grid, gates=gates
        ),
        harmonic_index=alignment.grid,
        harmonic_hz=harmonic_hz,
        repetition_times_s=rates,
    )


def comb_attenuation(comb: CohortComb) -> dict[str, float]:
    """Median excess removed per participant, paired within participant."""
    before = {
        subject: float(np.median(values)) for subject, values in comb.before.per_subject.items()
    }
    after = {
        subject: float(np.median(values)) for subject, values in comb.after.per_subject.items()
    }
    # Negated: the difference is after minus before, and what a reader wants named is how
    # much was taken out.
    return {subject: -value for subject, value in paired_differences(before, after).items()}


def comb_audit(comb: CohortComb) -> pd.DataFrame:
    """The plotted values, so a figure can be checked without being re-derived."""
    rows: list[dict[str, object]] = []
    for index, harmonic in enumerate(comb.harmonic_index):
        row: dict[str, object] = {
            "harmonic_index": int(harmonic),
            "harmonic_hz": (
                None if comb.harmonic_hz is None else float(comb.harmonic_hz[index])
            ),
            "n_subjects": comb.after.denominator.n_subjects,
        }
        for stage, curve in (("before", comb.before), ("after", comb.after)):
            row[f"{stage}_median_db"] = (
                None if curve.median is None else float(curve.median[index])
            )
            for subject, values in curve.per_subject.items():
                row[f"{stage}_{subject}_db"] = float(values[index])
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "REPETITION_TIME_TOLERANCE_S",
    "CohortComb",
    "comb_attenuation",
    "comb_audit",
    "cohort_comb",
    "participant_comb",
]
