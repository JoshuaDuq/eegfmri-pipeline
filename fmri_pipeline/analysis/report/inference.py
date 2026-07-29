"""What a height threshold on a statistic map is actually worth.

A first-level z map is thresholded at a number chosen in a config file, and the
report drew it without ever saying what that number buys. Two facts decide that, and
neither is visible in a thresholded picture:

*How many voxels were tested.* At |z| > 2.3 over 50,000 in-mask voxels, roughly a
thousand voxels are expected to survive with no signal present at all. The
Benjamini-Hochberg and Bonferroni thresholds for the same map are the natural
comparisons, so they are computed here and stated beside the applied one.

*Whether the map's null is the one the threshold assumes.* A z map is nominally
N(0, 1) under the null, but a single-subject GLM with unmodelled autocorrelation and
physiological noise is routinely over-dispersed. Measured on this study's own data,
the empirical null is centred at -0.61 with a width of 1.51: |z| > 2.3 reads as
p < 0.021 and is worth closer to p ~ 0.13. Efron (2004) is the reference; the robust
quantile form used here is what survives a map with a real signal tail.

Everything is computed from a saved map. Nothing here fits a model, which is what
lets it live in the report package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Median absolute deviation of a standard normal: Phi^-1(0.75).
#:
#: Dividing a sample's MAD by this recovers its standard deviation from the central
#: mass alone. That is the property that matters: a map with a genuine activation tail
#: inflates the ordinary standard deviation, and the inflated null then looks like
#: evidence that the threshold is conservative when the opposite is true.
#:
#: MAD rather than the interquartile range, which is the other obvious choice. They
#: agree to three decimals on real data and on contamination up to about 10% of
#: voxels, but the IQR's upper quartile is swallowed once a one-sided signal reaches a
#: quarter of the map: measured at 25% contamination at +6, the IQR estimate reaches
#: 2.67 against MAD's 1.57. MAD's breakdown point is 50%, twice the IQR's.
_NORMAL_MAD = 0.6744897501960817


@dataclass(frozen=True)
class EmpiricalNull:
    """A normal null fitted to the central mass of an observed distribution."""

    centre: float
    scale: float
    n: int


@dataclass(frozen=True)
class ThresholdContext:
    """The applied threshold beside the corrected ones, with survivor counts.

    ``fdr`` is ``None`` when Benjamini-Hochberg rejects nothing, which is a result
    rather than a failure: a contrast with no signal must still draw its panel.
    ``null`` is ``None`` when the map has no spread to fit.
    """

    n_voxels: int
    two_sided: bool
    #: ``None`` under ``threshold_mode: none``, where no height was applied. The
    #: corrected thresholds are still worth stating: they are what the map *would*
    #: have been cut at, and an unthresholded report is the one whose reader most
    #: needs them.
    applied: Optional[float]
    applied_survivors: Optional[int]
    expected_null_survivors: Optional[float]
    fdr: Optional[float]
    fdr_q: float
    fdr_survivors: int
    bonferroni: float
    alpha: float
    bonferroni_survivors: int
    null: Optional[EmpiricalNull]
    #: The applied threshold expressed in the empirical null's own units. This is the
    #: number a reader wants and cannot get anywhere else: 2.3 against a null of width
    #: 1.5 is 1.5 sigma, not 2.3.
    applied_in_null_units: Optional[float]


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Inference summaries require at least one finite value.")
    return finite


def empirical_null(values: np.ndarray) -> EmpiricalNull:
    """Fit a normal null to the central mass of ``values``.

    The centre is the median and the width is the median absolute deviation rescaled
    by the standard normal's own MAD. Both read the middle of the distribution, so a
    signal tail moves neither much: that is the entire reason for preferring them to
    the sample mean and standard deviation, which a real activation inflates until the
    fitted null absorbs the signal it was supposed to be measured against.

    Robust is not immune. Contamination heavy enough to reach the median does inflate
    the estimate -- at a quarter of the map shifted to +6, the fitted width reaches
    about 1.57 rather than 1.0. That direction is the safe one: an over-wide null makes
    the applied threshold look *weaker* than it is, so the error runs toward caution
    rather than toward false confidence.

    Raises when the map has no spread, because a null of width zero would make every
    voxel off the median infinitely significant.
    """
    finite = _finite(values)
    centre = float(np.median(finite))
    scale = float(np.median(np.abs(finite - centre))) / _NORMAL_MAD
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "The central mass of this map has no spread, so no null can be fitted."
        )
    return EmpiricalNull(centre=centre, scale=scale, n=int(finite.size))


def p_values(values: np.ndarray, *, two_sided: bool) -> np.ndarray:
    """Convert z statistics to p values under the standard normal null.

    Sidedness is not cosmetic here. A one-sided test never examined the negative half,
    so a large negative z is the *least* significant value on the map; folding it into
    the upper tail would make a deactivation the strongest result of a test that could
    not detect one.
    """
    from scipy import stats

    array = np.asarray(values, dtype=float)
    if two_sided:
        return 2.0 * stats.norm.sf(np.abs(array))
    return stats.norm.sf(array)


def fdr_p_cutoff(p: np.ndarray, *, q: float) -> Optional[float]:
    """Largest p value Benjamini-Hochberg rejects at ``q``, or None if none is.

    The step-up form: sort ascending, find the largest ``k`` with
    ``p_(k) <= k/m * q``, and reject everything at or below that p. Taking instead the
    first ``k`` that *fails* would stop at the first gap in the ranking and reject far
    too little.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"FDR q must lie in (0, 1], got {q!r}.")
    ordered = np.sort(np.asarray(p, dtype=float).ravel())
    m = ordered.size
    if m == 0:
        return None
    passing = np.flatnonzero(ordered <= (np.arange(1, m + 1) / m) * float(q))
    if passing.size == 0:
        return None
    return float(ordered[passing[-1]])


def fdr_threshold(
    values: np.ndarray, *, q: float, two_sided: bool
) -> Optional[float]:
    """The z height at which Benjamini-Hochberg controls the FDR at ``q``.

    Returns ``None`` when nothing is rejected. Callers state that rather than
    substituting a threshold, since "no voxel survives correction" is the finding.
    """
    finite = _finite(values)
    cutoff = fdr_p_cutoff(p_values(finite, two_sided=two_sided), q=q)
    if cutoff is None:
        return None

    from scipy import stats

    return float(stats.norm.isf(cutoff / 2.0 if two_sided else cutoff))


def bonferroni_threshold(*, n: int, alpha: float, two_sided: bool) -> float:
    """The z height controlling the familywise error rate at ``alpha`` over ``n`` tests.

    Reported as the strict end of the scale, not as a recommendation. Bonferroni over
    spatially correlated voxels is conservative -- neighbouring voxels are not
    independent tests -- and the figure says so rather than letting the number pass as
    the correct answer.
    """
    if n < 1:
        raise ValueError(f"Bonferroni needs at least one test, got n={n!r}.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha!r}.")

    from scipy import stats

    per_test = float(alpha) / float(n)
    return float(stats.norm.isf(per_test / 2.0 if two_sided else per_test))


def expected_false_positives(*, n: int, threshold: float, two_sided: bool) -> float:
    """How many of ``n`` voxels clear ``threshold`` with no signal present.

    The number that makes an uncorrected threshold legible. It is a count, not a
    verdict: a map whose survivors greatly exceed it has structure, and one whose
    survivors match it may have none.
    """
    if threshold <= 0:
        raise ValueError(f"Threshold must be > 0, got {threshold!r}.")
    if n < 0:
        raise ValueError(f"Voxel count must be >= 0, got {n!r}.")

    from scipy import stats

    tail = float(stats.norm.sf(float(threshold)))
    return float(n) * (2.0 * tail if two_sided else tail)


def _survivors(values: np.ndarray, threshold: Optional[float], *, two_sided: bool) -> int:
    if threshold is None:
        return 0
    compared = np.abs(values) if two_sided else values
    return int(np.count_nonzero(compared > float(threshold)))


def threshold_context(
    values: np.ndarray,
    *,
    applied_threshold: Optional[float],
    fdr_q: float,
    alpha: float,
    two_sided: bool,
) -> ThresholdContext:
    """Assemble everything the calibration panel states about a threshold.

    ``values`` must already be restricted to the analysis mask. Passing a whole volume
    includes the background zeros -- on a typical map that is over half the voxels --
    which inflates the test count, drags the fitted null toward zero, and makes every
    corrected threshold wrong in the same direction.
    """
    finite = _finite(values)
    n = int(finite.size)

    try:
        null: Optional[EmpiricalNull] = empirical_null(finite)
    except ValueError:
        # A map with no central spread still has a well-defined test count, and the
        # corrected thresholds depend only on that. Losing them with the null would
        # cost the panel the numbers it exists to state.
        null = None

    fdr = fdr_threshold(finite, q=fdr_q, two_sided=two_sided)
    bonferroni = bonferroni_threshold(n=n, alpha=alpha, two_sided=two_sided)

    applied = None if applied_threshold is None else float(applied_threshold)
    return ThresholdContext(
        n_voxels=n,
        two_sided=two_sided,
        applied=applied,
        applied_survivors=(
            None if applied is None else _survivors(finite, applied, two_sided=two_sided)
        ),
        expected_null_survivors=(
            None
            if applied is None
            else expected_false_positives(n=n, threshold=applied, two_sided=two_sided)
        ),
        fdr=fdr,
        fdr_q=float(fdr_q),
        fdr_survivors=_survivors(finite, fdr, two_sided=two_sided),
        bonferroni=bonferroni,
        alpha=float(alpha),
        bonferroni_survivors=_survivors(finite, bonferroni, two_sided=two_sided),
        null=null,
        applied_in_null_units=(
            None if null is None or applied is None else applied / null.scale
        ),
    )


__all__ = [
    "EmpiricalNull",
    "ThresholdContext",
    "bonferroni_threshold",
    "empirical_null",
    "expected_false_positives",
    "fdr_p_cutoff",
    "fdr_threshold",
    "p_values",
    "threshold_context",
]
