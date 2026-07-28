"""How runs become participants and participants become a cohort.

This module holds every pooling decision the cohort report makes, and nothing else. It
imports no MNE and draws nothing, so each statistical claim the document rests on can be
checked as arithmetic rather than inspected in a figure.

Three decisions shape everything here.

The unit of inference is the participant. Runs are pooled within a participant first, and
every participant then carries equal weight regardless of how many runs it contributed. A
cohort statistic that weighted by run count would describe whoever sat in the scanner
longest.

The rule for pooling runs depends on what kind of quantity it is, because one rule is
wrong for at least one of them. Curves are independent estimates of the same thing and are
pooled by an unweighted median. Fractions are defined over the whole session and are
pooled as a sum over a sum, so that a thirty-second run cannot outvote a twelve-minute
one. Counts are pooled as a union under the policy the preprocessing recorded.

Nothing here computes a confidence interval, and nothing bootstraps. A confidence interval
on a cohort median answers how precisely the group's central value is known, which is not
the question a quality-control document asks; the question is where a participant sits
among the others, and the empirical distribution answers it directly. Spread is therefore
reported as quantile bands, and only for quantiles the participant count can support --
see :func:`quantile_is_supported`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Quantiles the bands are drawn at. The inner pair needs three participants to sit inside
#: the observed order statistics, the outer pair needs nine; :class:`BandGates` refuses any
#: configuration that would extrapolate either.
QUARTILES = (0.25, 0.75)
DECILES = (0.10, 0.90)

#: Tolerance for deciding that two frequency grids hold the same bin. Spectra are never
#: interpolated onto a common grid, so this compares bins that should already be identical
#: and only absorbs the last bits of a floating-point round trip.
GRID_TOLERANCE_HZ = 1e-9


class BandRegime(Enum):
    """What a panel is entitled to draw at the participant count it has.

    The regime is carried on the result rather than decided at the drawing site, so that
    two panels built from the same participants cannot disagree about whether a median was
    justified.
    """

    #: Too few participants for any summary. Individual values only.
    INDIVIDUALS = "individuals"
    #: Median and interquartile band.
    MEDIAN_IQR = "median_iqr"
    #: Median, interquartile band, and the tenth-to-ninetieth band.
    MEDIAN_IQR_DECILES = "median_iqr_deciles"


def quantile_is_supported(probability: float, n_participants: int) -> bool:
    """Whether a quantile lies inside the observed order statistics.

    Under Weibull plotting positions the k-th of n ordered values estimates the k/(n+1)
    quantile, so a probability is interior to the sample when ``1/(n+1) <= p <= n/(n+1)``.
    Outside that range the "quantile" is an extrapolation beyond the most extreme
    participant observed, which is a statement about an assumed distribution rather than
    about the cohort in front of the reader.
    """
    if n_participants < 1:
        return False
    lower = 1.0 / (n_participants + 1)
    return lower <= probability <= 1.0 - lower


@dataclass(frozen=True)
class BandGates:
    """Participant counts at which each band becomes drawable.

    Configurable, because a site may want to be more conservative than the arithmetic
    minimum. Not configurable below it: a gate that would extrapolate a quantile is
    rejected at construction, so no setting can produce a band the sample cannot support.
    """

    min_subjects_for_median: int = 5
    min_subjects_for_outer_band: int = 10

    def __post_init__(self) -> None:
        if not quantile_is_supported(QUARTILES[0], self.min_subjects_for_median):
            raise ValueError(
                f"A quartile band needs enough participants to sit inside the sample; "
                f"min_subjects_for_median={self.min_subjects_for_median} would extrapolate it."
            )
        if not quantile_is_supported(DECILES[0], self.min_subjects_for_outer_band):
            raise ValueError(
                f"A decile band needs enough participants to sit inside the sample; "
                f"min_subjects_for_outer_band={self.min_subjects_for_outer_band} would "
                f"extrapolate it."
            )
        if self.min_subjects_for_outer_band < self.min_subjects_for_median:
            raise ValueError(
                "The outer band cannot become available before the interquartile band."
            )

    def regime_for(self, n_participants: int) -> BandRegime:
        if n_participants < self.min_subjects_for_median:
            return BandRegime.INDIVIDUALS
        if n_participants < self.min_subjects_for_outer_band:
            return BandRegime.MEDIAN_IQR
        return BandRegime.MEDIAN_IQR_DECILES


DEFAULT_GATES = BandGates()


@dataclass(frozen=True)
class Contribution:
    """One participant's contribution to a cohort statistic.

    ``value`` is always an array so that scalars and curves travel the same path: a
    scalar is a single-element array. ``n_runs`` does not weight anything; it is carried
    so the denominator a panel prints describes the evidence behind it.
    """

    subject: str
    value: np.ndarray
    n_runs: int


@dataclass(frozen=True)
class Denominator:
    """The evidence behind one number, travelling with it.

    Panels compute their own denominators rather than inheriting the report's, because a
    participant can contribute to the spectra and not to the marker agreement. A document
    whose header says twenty while a panel silently used fourteen is the failure this
    exists to make impossible.
    """

    n_subjects: int
    n_runs: int
    subjects: tuple[str, ...]


@dataclass(frozen=True)
class CohortScalar:
    """One measurement pooled across participants."""

    per_subject: Mapping[str, float]
    denominator: Denominator
    regime: BandRegime
    median: float | None = None
    quartiles: tuple[float, float] | None = None
    deciles: tuple[float, float] | None = None


@dataclass(frozen=True)
class CohortCurve:
    """One curve pooled pointwise across participants."""

    grid: np.ndarray
    per_subject: Mapping[str, np.ndarray]
    denominator: Denominator
    regime: BandRegime
    median: np.ndarray | None = None
    quartiles: tuple[np.ndarray, np.ndarray] | None = None
    deciles: tuple[np.ndarray, np.ndarray] | None = None


# --------------------------------------------------------------------------------------
# Run to participant
# --------------------------------------------------------------------------------------


def pool_runs_curve(values: Sequence[np.ndarray]) -> np.ndarray:
    """Pool per-run curves into one curve for the participant.

    An unweighted median: each run is an independent estimate of the same quantity, and a
    longer run does not estimate a participant's spectrum any more correctly, only more
    precisely. Weighting by duration would let one long run define the participant.
    """
    if not values:
        raise ValueError("Pooling runs into a participant needs at least one run.")
    stacked = np.vstack([np.asarray(value, dtype=float) for value in values])
    return np.median(stacked, axis=0)


def pool_runs_rate(numerators: Iterable[float], denominators: Iterable[float]) -> float:
    """Pool a per-run fraction over the whole session.

    A sum over a sum, not a median of the per-run fractions. The quantity is defined over
    the session -- what fraction of the recorded time was flagged, what fraction of the
    presented trials survived -- so every second and every trial counts once. Taking the
    median of per-run fractions instead lets a thirty-second run that was entirely flagged
    carry the same weight as a clean twelve-minute one, and reports half a session lost
    where four percent was.
    """
    total_numerator = float(np.sum(np.asarray(list(numerators), dtype=float)))
    total_denominator = float(np.sum(np.asarray(list(denominators), dtype=float)))
    if total_denominator <= 0.0:
        raise ValueError("A rate needs a positive total denominator.")
    return total_numerator / total_denominator


def pool_runs_union(per_run: Iterable[Sequence[str]]) -> tuple[str, ...]:
    """Pool per-run channel sets into the participant's union.

    Matches the ``bad_channel_sync_policy`` the preprocessing already records: a channel
    unusable in any run is unusable for the session.
    """
    union: set[str] = set()
    for names in per_run:
        union.update(names)
    return tuple(sorted(union))


# --------------------------------------------------------------------------------------
# Participant to cohort
# --------------------------------------------------------------------------------------


def _validated(contributions: Sequence[Contribution]) -> list[Contribution]:
    """Order contributions and reject the three ways a denominator can lie."""
    if not contributions:
        raise ValueError("A cohort statistic needs at least one participant.")
    seen: dict[str, int] = {}
    for contribution in contributions:
        seen[contribution.subject] = seen.get(contribution.subject, 0) + 1
    duplicated = sorted(name for name, count in seen.items() if count > 1)
    if duplicated:
        raise ValueError(
            f"Participants appear more than once and would be weighted twice: "
            f"{', '.join(duplicated)}."
        )
    # A missing value poisons every statistic it touches: ``np.median`` over a stack with
    # one NaN returns NaN at that point, so a single participant with a gap at one
    # frequency erases the cohort median at that frequency while the denominator beside it
    # still says how many participants contributed. That is precisely the failure
    # :class:`Denominator` exists to make impossible, and it cannot be prevented by
    # convention at each of a dozen drawing sites.
    #
    # Raised rather than dropped, because deciding what to do about a participant that
    # cannot contribute is policy and this module is arithmetic. The caller drops it and
    # shrinks the denominator it prints; see :func:`cohort_spectra`.
    incomplete = sorted(
        contribution.subject
        for contribution in contributions
        if not np.all(np.isfinite(np.asarray(contribution.value, dtype=float)))
    )
    if incomplete:
        raise ValueError(
            f"Participants contributed a missing value and would erase the pooled "
            f"statistic wherever it sits, without shrinking the denominator beside it: "
            f"{', '.join(incomplete)}. Drop them at the call site so the panel reports the "
            f"contributors it actually had."
        )
    return sorted(contributions, key=lambda contribution: contribution.subject)


def _denominator(contributions: Sequence[Contribution]) -> Denominator:
    return Denominator(
        n_subjects=len(contributions),
        n_runs=int(sum(contribution.n_runs for contribution in contributions)),
        subjects=tuple(contribution.subject for contribution in contributions),
    )


#: A median, an inner band and an outer band, any of which the regime may withhold.
_Bands = tuple[
    np.ndarray | None,
    tuple[np.ndarray, np.ndarray] | None,
    tuple[np.ndarray, np.ndarray] | None,
]


#: Quantile estimator, and it must be the one :func:`quantile_is_supported` reasons about.
#:
#: The gate is justified by Weibull plotting positions -- the k-th of n ordered values
#: estimates the k/(n+1) quantile -- and concludes that p = 0.10 needs nine participants.
#: NumPy's default ``method="linear"`` uses (k-1)/(n-1) instead, under which the p = 0.10
#: "quantile" of ten participants is an interpolation a tenth of the way from the smallest
#: to the second smallest, and is defined for as few as two. Estimating with one convention
#: while gating on another means the gate does not describe what the band actually is, so
#: the estimator is pinned to the convention the gate was argued from.
QUANTILE_METHOD = "weibull"


def _bands(stacked: np.ndarray, regime: BandRegime) -> _Bands:
    """Summary statistics the regime permits, pointwise down the participant axis.

    Withheld statistics are absent rather than merely undrawn. A panel cannot print a
    median it is not entitled to if the median was never computed, which makes the gate
    structural instead of a convention every drawing site has to remember.
    """
    if regime is BandRegime.INDIVIDUALS:
        return None, None, None
    median = np.median(stacked, axis=0)
    low, high = np.quantile(stacked, QUARTILES, axis=0, method=QUANTILE_METHOD)
    quartiles = (low, high)
    if regime is BandRegime.MEDIAN_IQR:
        return median, quartiles, None
    outer_low, outer_high = np.quantile(stacked, DECILES, axis=0, method=QUANTILE_METHOD)
    return median, quartiles, (outer_low, outer_high)


def pool_participants_scalar(
    contributions: Sequence[Contribution],
    *,
    gates: BandGates = DEFAULT_GATES,
) -> CohortScalar:
    """Pool one measurement across participants, with equal weight each.

    Whatever scale the caller pools in is the scale the result describes. For power the
    caller is expected to pass decibels: inter-participant power is approximately
    log-normal, so decibels are the scale on which the spread is symmetric and a quantile
    band means something. The median is additionally near-indifferent to that choice --
    it commutes with the transform exactly at odd participant counts, and at even counts
    differs only by the averaging of the two central values -- which is not true of a
    mean, and is why this pools medians.
    """
    ordered = _validated(contributions)
    for contribution in ordered:
        if np.asarray(contribution.value).size != 1:
            raise ValueError(
                f"Participant {contribution.subject} contributed "
                f"{np.asarray(contribution.value).size} values where a scalar was expected."
            )
    stacked = np.vstack([np.asarray(c.value, dtype=float).reshape(1) for c in ordered])
    regime = gates.regime_for(len(ordered))
    median, quartiles, deciles = _bands(stacked, regime)
    return CohortScalar(
        per_subject={c.subject: float(np.asarray(c.value).reshape(1)[0]) for c in ordered},
        denominator=_denominator(ordered),
        regime=regime,
        median=None if median is None else float(median[0]),
        quartiles=None if quartiles is None else (float(quartiles[0][0]), float(quartiles[1][0])),
        deciles=None if deciles is None else (float(deciles[0][0]), float(deciles[1][0])),
    )


def pool_participants_curve(
    contributions: Sequence[Contribution],
    *,
    grid: np.ndarray,
    gates: BandGates = DEFAULT_GATES,
) -> CohortCurve:
    """Pool one curve pointwise across participants, with equal weight each.

    Every participant must already sit on ``grid``; see :func:`align_grids` for how the
    shared grid is established. Pooling pointwise means the resulting curve is not any
    participant's curve, and a feature narrow enough to fall in different bins for
    different participants is flattened rather than averaged -- which is the honest
    outcome, and the reason participant traces stay drawn beneath the cohort median.
    """
    ordered = _validated(contributions)
    grid = np.asarray(grid, dtype=float)
    mismatched = [
        contribution.subject
        for contribution in ordered
        if np.asarray(contribution.value).shape != grid.shape
    ]
    if mismatched:
        raise ValueError(
            f"Participants do not sit on the pooled grid of {grid.size} points: "
            f"{', '.join(mismatched)}."
        )
    stacked = np.vstack([np.asarray(c.value, dtype=float) for c in ordered])
    regime = gates.regime_for(len(ordered))
    median, quartiles, deciles = _bands(stacked, regime)
    return CohortCurve(
        grid=grid,
        per_subject={c.subject: np.asarray(c.value, dtype=float) for c in ordered},
        denominator=_denominator(ordered),
        regime=regime,
        median=median,
        quartiles=quartiles,
        deciles=deciles,
    )


# --------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridAlignment:
    """The shared abscissa of a set of participants, and who narrowed it."""

    grid: np.ndarray
    #: Participants whose own range is narrower than the widest available, and which
    #: therefore set the extent of the cohort figure. Named on the panel so a reader can
    #: see that the axis stops where it does because of a participant, not a method.
    limiting_subjects: tuple[str, ...]
    #: Boolean selection into each participant's own grid.
    masks: Mapping[str, np.ndarray]


def align_grids(
    grids: Mapping[str, np.ndarray],
    *,
    tolerance: float = GRID_TOLERANCE_HZ,
) -> GridAlignment:
    """Restrict participants to their shared range and require identical bins there.

    Restriction, never interpolation. Resampling a spectrum onto a foreign grid smears
    exactly the narrow features -- a line-noise tooth, a gradient harmonic, an alpha peak
    -- that the cohort figure exists to show, and it does so invisibly. Where the bins
    genuinely differ, that is a fact about the acquisitions and is raised rather than
    smoothed over.
    """
    if not grids:
        raise ValueError("Aligning grids needs at least one participant.")
    arrays = {name: np.asarray(grid, dtype=float) for name, grid in grids.items()}
    for name, grid in arrays.items():
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError(f"Participant {name} has no one-dimensional grid to align.")

    minima = {name: float(grid[0]) for name, grid in arrays.items()}
    maxima = {name: float(grid[-1]) for name, grid in arrays.items()}
    low, high = max(minima.values()), min(maxima.values())
    if high <= low:
        raise ValueError("Participants share no common range and cannot be pooled.")

    limiting = sorted(
        name
        for name in arrays
        if minima[name] > min(minima.values()) or maxima[name] < max(maxima.values())
    )
    masks = {
        name: (grid >= low - tolerance) & (grid <= high + tolerance)
        for name, grid in arrays.items()
    }

    reference_name = sorted(arrays)[0]
    reference = arrays[reference_name][masks[reference_name]]
    differing = sorted(
        name
        for name, grid in arrays.items()
        if grid[masks[name]].shape != reference.shape
        or not np.allclose(grid[masks[name]], reference, atol=tolerance, rtol=0.0)
    )
    if differing:
        raise ValueError(
            f"Pooling spectra needs identical bins over the shared range; "
            f"{', '.join(differing)} differ from {reference_name}. "
            f"Interpolating onto a common grid would smear the peaks the figure exists "
            f"to show, so this is raised rather than resampled."
        )
    return GridAlignment(grid=reference, limiting_subjects=tuple(limiting), masks=masks)


# --------------------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------------------


def paired_differences(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    """Difference each participant against itself.

    Before and after ICA are two measurements of one participant, not two samples. Drawing
    them as independent distributions discards the pairing and inflates the spread by the
    between-participant variance, which is the larger of the two and is not what the
    comparison is about.

    A participant measured on only one side breaks the pairing rather than shrinking the
    sample quietly, so it is raised.
    """
    unpaired = sorted(set(before) ^ set(after))
    if unpaired:
        raise ValueError(
            f"A paired comparison needs both measurements for every participant; "
            f"missing one side for {', '.join(unpaired)}."
        )
    return {subject: float(after[subject]) - float(before[subject]) for subject in sorted(before)}


# --------------------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------------------


def project_reliability(
    reliability: float,
    *,
    observed_trials: int,
    reference_trials: int,
) -> float:
    """Project a reliability to the trial count the cohort is compared at.

    Reliability grows with test length, so a participant measured on forty retained trials
    is not comparable with one measured on a hundred and twenty: the second scores higher
    for having more trials, whatever the recording quality. The Spearman-Brown prophecy
    steps every participant to a common reference length so the panel compares recordings
    rather than trial counts.

    A non-positive reliability has no signal to project and is returned unchanged; the
    same applies where the projection's denominator is not positive, which a strongly
    negative reliability stepped upward can reach.
    """
    if observed_trials <= 0 or reference_trials <= 0:
        raise ValueError("A reliability projection needs a positive trial count on both sides.")
    if reliability <= 0.0:
        return float(reliability)
    factor = float(reference_trials) / float(observed_trials)
    denominator = 1.0 + (factor - 1.0) * float(reliability)
    if denominator <= 0.0:
        return float(reliability)
    return float(factor * float(reliability) / denominator)


__all__ = [
    "BandGates",
    "BandRegime",
    "CohortCurve",
    "CohortScalar",
    "Contribution",
    "DECILES",
    "DEFAULT_GATES",
    "Denominator",
    "GridAlignment",
    "QUARTILES",
    "align_grids",
    "paired_differences",
    "pool_participants_curve",
    "pool_participants_scalar",
    "pool_runs_curve",
    "pool_runs_rate",
    "pool_runs_union",
    "project_reliability",
    "quantile_is_supported",
]
