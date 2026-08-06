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
p < 0.021 and is worth p = 0.027 upward and p = 0.13 downward. Efron (2004) is the
reference; the robust quantile form used here is what survives a map with a real
signal tail.

The correction is *applied*, not only noted. Every theoretical-null quantity has an
empirical-null counterpart -- expected survivors, tail probabilities, an FDR rejection
region -- because on real data they disagree by nearly an order of magnitude in both
directions, and the theoretical figures alone let a reader conclude the opposite of
what the map shows. Stating a fitted null beside a threshold while continuing to
report only what the threshold is worth under N(0, 1) leaves the reader to integrate
a normal tail by eye off a log axis.

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
class EmpiricalCalibration:
    """What the applied height and an FDR correction are worth against a fitted null.

    Every number here has a counterpart computed under N(0, 1), and the pair is the
    point: on this study's own data the two differ by nearly an order of magnitude in
    both directions, and nothing in a thresholded map or in the theoretical numbers
    alone reveals it.
    """

    #: Voxels expected to clear the applied height if every voxel were drawn from the
    #: *fitted* null. Compare against ``ThresholdContext.expected_null_survivors``,
    #: which assumes N(0, 1). ``None`` when no height was applied.
    expected_survivors: Optional[float]
    #: Tail probability of the applied height under the fitted null, upper and lower.
    #: A shifted null makes a symmetric ``|z| > c`` cut asymmetric in evidence, and one
    #: number cannot say so. ``lower_tail_p`` is ``None`` under one-sided inference,
    #: where the lower tail was never examined; both are ``None`` when no height was
    #: applied.
    upper_tail_p: Optional[float]
    lower_tail_p: Optional[float]
    #: Raw-z heights at which Benjamini-Hochberg controls the FDR when p values are
    #: computed under the fitted null rather than N(0, 1) -- Efron's (2004) correction.
    #: Asymmetric whenever the null is shifted, hence two bounds rather than one
    #: height. Either is ``None`` when that tail rejects nothing.
    fdr_upper: Optional[float]
    fdr_lower: Optional[float]
    fdr_survivors: int


@dataclass(frozen=True)
class SignFlipSummary:
    """A familywise height from run exchangeability, and what it is worth.

    The report-side view of the null enumerated during analysis. Only the scalars a
    panel needs, so the report never imports the fitting package to draw this.

    Why it belongs beside Bonferroni and FDR rather than replacing them: those two ask
    what a threshold is worth if every voxel is N(0, 1), and this map's fitted null is
    measurably not that. This one assumes nothing about the distribution -- it asks how
    large a maximum the same data produces when the only thing changed is which runs
    were labelled positive.

    ``p_floor`` travels with ``global_p`` because the two are not independent. The
    unflipped pattern is always a member of the null and always ties the observed
    maximum, so ``global_p`` can never fall below ``p_floor``. Printed alone, a p of
    0.061 from six runs reads as a near-miss when it is the smallest value the test can
    return.
    """

    height: float
    survivors: int
    global_p: float
    p_floor: float
    n_runs: int
    n_patterns: int
    observed_max: float

    @property
    def floor_limited(self) -> bool:
        """Whether no map-level p below 0.05 is reachable with this many runs."""
        return self.p_floor > 0.05


def sign_flip_p_floor(n_runs: int) -> float:
    """Smallest attainable global p for a run sign-flip test over ``n_runs`` runs.

    ``2 / (2**(n_runs-1) + 1)``: the numerator is 2 rather than 1 because the
    unflipped pattern is itself a member of the null and ties the observed maximum,
    so it is counted on both sides of the ratio.
    """
    if n_runs < 2:
        raise ValueError(f"A sign-flip null needs at least two runs, got {n_runs!r}.")
    return 2.0 / (2 ** (n_runs - 1) + 1)


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
    #: The random-field familywise height over the mask's resel count, and how many
    #: voxels clear it. ``None`` when no smoothness could be estimated, and when the
    #: search volume is too small for the approximation to admit a solution.
    #:
    #: Reported *beside* Bonferroni rather than in place of it. Both are valid
    #: familywise bounds and which one is tighter is a property of the smoothness:
    #: at this study's roughly ten voxels per resel the Euler-characteristic form
    #: overshoots and Bonferroni is the tighter of the two, while on heavily smoothed
    #: data the ranking reverses. Quoting either alone as "the" corrected height
    #: hands the reader whichever bound happens to be looser.
    rft: Optional[float] = None
    rft_survivors: int = 0
    n_resels: Optional[float] = None
    null: Optional[EmpiricalNull] = None
    #: Every threshold re-read against the map's own null, or ``None`` when no null
    #: could be fitted.
    #:
    #: This replaced a single "applied threshold is N x the null's width" figure, which
    #: was true and useless: it divides by the null's scale and ignores its centre, so
    #: on a null centred at -0.61 it reads as a sigma count that neither tail actually
    #: has. The tail probabilities below are what that number was reaching for.
    calibration: Optional[EmpiricalCalibration] = None
    #: The run sign-flip null's summary, or ``None`` for a single-run contrast and for
    #: a manifest written before the null existed.
    sign_flip: Optional[SignFlipSummary] = None


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


#: Peak of the 3D Euler-characteristic density, at z = sqrt(3).
#:
#: ``rho_3`` carries a factor of ``(z**2 - 1)``, so it is negative below z = 1, rises
#: to a maximum here, and decays monotonically above it. Only the decaying arm is a
#: threshold: the root the solver wants is the one to the right of this peak, and
#: bracketing from here is what keeps it from returning the spurious low-z crossing.
_EC_DENSITY_PEAK = float(np.sqrt(3.0))


def _ec_density_3d(z: float) -> float:
    """Worsley's Euler-characteristic density for a 3D Gaussian field, per resel."""
    return (
        (4.0 * np.log(2.0)) ** 1.5
        / (2.0 * np.pi) ** 2
        * (float(z) ** 2 - 1.0)
        * float(np.exp(-(float(z) ** 2) / 2.0))
    )


def rft_voxel_threshold(*, n_resels: float, alpha: float, two_sided: bool) -> float:
    """The height at which a Gaussian field of ``n_resels`` yields a max above it
    with probability ``alpha``.

    The familywise correction this pipeline had the inputs for and did not perform.
    Bonferroni divides alpha across voxels, and after 6 mm of smoothing on a 3 mm grid
    neighbouring voxels are not separate tests: on this study's own contrast that is
    50,626 tests charged for a family of about 5,000, and the resulting height
    (|z| > 4.89) is stricter than the data warrant.

    Random field theory charges for the resels instead. The expected Euler
    characteristic of the excursion set above ``z`` is ``R * rho_3(z)``, and at the
    heights that matter it approximates the probability that the field's maximum
    exceeds ``z`` -- so setting it equal to alpha and solving gives the corrected
    height. Worsley et al. (1996) is the reference.

    Only the 3D term is carried. The full expansion adds the lower-dimensional resel
    counts, whose contribution is negligible for a search volume of thousands of
    resels and which would require the mask's intrinsic volumes rather than one
    number. This is the same approximation SPM's single-resel-count form makes.

    Two-sided inference splits alpha between the tails, which are asymptotically
    independent for a smooth field.

    Raises when the search volume is too small for the approximation to admit a
    solution, which is a statement about applicability rather than a failure: below
    roughly one resel per unit of alpha the excursion set's expected Euler
    characteristic never reaches alpha at all, and no RFT height exists to return.
    """
    if not np.isfinite(n_resels) or n_resels <= 0:
        raise ValueError(f"A search volume needs at least one resel, got {n_resels!r}.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha!r}.")

    from scipy import optimize

    target = float(alpha) / 2.0 if two_sided else float(alpha)
    resels = float(n_resels)

    def excess(z: float) -> float:
        return resels * _ec_density_3d(z) - target

    if excess(_EC_DENSITY_PEAK) <= 0.0:
        raise ValueError(
            f"A search volume of {resels:.3g} resels never reaches an expected Euler "
            f"characteristic of {target:.3g}, so no random-field height exists for it."
        )

    # The density decays to zero, so a bracket wide enough to cross the target always
    # exists; 50 is far past any height a z map reaches.
    height = float(optimize.brentq(excess, _EC_DENSITY_PEAK, 50.0, xtol=1e-12))

    # ``E[EC] ~= P(max > z)`` holds in the tail, and a small enough search volume puts
    # the root back down where it does not. The tell is a familywise height that is
    # more permissive than running one uncorrected test at the same alpha, which no
    # correction can honestly be: at half a resel the solver returns 1.91 against an
    # uncorrected 1.96. Rejecting there keeps a nonsense row out of the table.
    from scipy import stats

    uncorrected = float(stats.norm.isf(target))
    if height <= uncorrected:
        raise ValueError(
            f"A search volume of {resels:.3g} resels puts the random-field height at "
            f"{height:.3g}, at or below the uncorrected {uncorrected:.3g}; the "
            f"Euler-characteristic approximation does not hold there."
        )
    return height


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


def expected_false_positives_under(
    *, n: int, threshold: float, null: EmpiricalNull, two_sided: bool
) -> float:
    """How many of ``n`` voxels clear ``threshold`` if every one is drawn from ``null``.

    The counterpart of :func:`expected_false_positives` for a null that is not
    N(0, 1), and the number that decides whether a survivor count is a finding. On this
    study's own contrast the two disagree by a factor of seven: 1,086 voxels expected
    under the theoretical null against 8,001 under the fitted one, out of 8,463
    observed. The theoretical figure alone invites reading eightfold enrichment into a
    map that produced almost exactly what its own noise predicts.

    Both tails are counted under two-sided inference even when the null is shifted, so
    the total is directly comparable with the observed survivor count.
    """
    if threshold <= 0:
        raise ValueError(f"Threshold must be > 0, got {threshold!r}.")
    if n < 0:
        raise ValueError(f"Voxel count must be >= 0, got {n!r}.")

    from scipy import stats

    upper = float(stats.norm.sf((float(threshold) - null.centre) / null.scale))
    if not two_sided:
        return float(n) * upper
    lower = float(stats.norm.cdf((-float(threshold) - null.centre) / null.scale))
    return float(n) * (upper + lower)


def empirical_calibration(
    values: np.ndarray,
    *,
    null: EmpiricalNull,
    applied_threshold: Optional[float],
    fdr_q: float,
    two_sided: bool,
) -> EmpiricalCalibration:
    """Re-read the applied height and an FDR correction against ``null``.

    This is Efron's (2004) empirical-null correction, applied where it changes the
    answer rather than only mentioned. Standardising each voxel by the fitted null and
    running Benjamini-Hochberg on the resulting p values is the whole of it, and the
    consequence is large: on this study's contrast the theoretical-null FDR rejects
    4,469 voxels at q = 0.05 and the empirical-null FDR rejects 82.

    The rejection region is returned as raw-z bounds rather than one height, because a
    shifted null makes it asymmetric -- here, raw z above 5.35 or below -6.57. Reporting
    a single ``|z| >`` height for it would be the same error this function exists to
    correct.
    """
    from scipy import stats

    finite = _finite(values)
    standardised = (finite - null.centre) / null.scale
    cutoff = fdr_p_cutoff(
        p_values(standardised, two_sided=two_sided), q=fdr_q
    )

    fdr_upper: Optional[float] = None
    fdr_lower: Optional[float] = None
    survivors = 0
    if cutoff is not None:
        height = float(stats.norm.isf(cutoff / 2.0 if two_sided else cutoff))
        fdr_upper = null.centre + height * null.scale
        rejected = finite > fdr_upper
        if two_sided:
            fdr_lower = null.centre - height * null.scale
            rejected = rejected | (finite < fdr_lower)
        survivors = int(np.count_nonzero(rejected))

    expected: Optional[float] = None
    upper_tail: Optional[float] = None
    lower_tail: Optional[float] = None
    if applied_threshold is not None:
        applied = float(applied_threshold)
        expected = expected_false_positives_under(
            n=int(finite.size), threshold=applied, null=null, two_sided=two_sided
        )
        upper_tail = float(stats.norm.sf((applied - null.centre) / null.scale))
        if two_sided:
            lower_tail = float(stats.norm.cdf((-applied - null.centre) / null.scale))

    return EmpiricalCalibration(
        expected_survivors=expected,
        upper_tail_p=upper_tail,
        lower_tail_p=lower_tail,
        fdr_upper=fdr_upper,
        fdr_lower=fdr_lower,
        fdr_survivors=survivors,
    )


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
    sign_flip: Optional[SignFlipSummary] = None,
    n_resels: Optional[float] = None,
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

    calibration = (
        None
        if null is None
        else empirical_calibration(
            finite,
            null=null,
            applied_threshold=applied_threshold,
            fdr_q=fdr_q,
            two_sided=two_sided,
        )
    )

    fdr = fdr_threshold(finite, q=fdr_q, two_sided=two_sided)
    bonferroni = bonferroni_threshold(n=n, alpha=alpha, two_sided=two_sided)

    rft: Optional[float] = None
    if n_resels is not None:
        try:
            rft = rft_voxel_threshold(
                n_resels=float(n_resels), alpha=alpha, two_sided=two_sided
            )
        except ValueError:
            # A search volume too small for the Euler-characteristic approximation is
            # a statement about this mask, not a reason to lose the other rows.
            rft = None

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
        rft=rft,
        rft_survivors=_survivors(finite, rft, two_sided=two_sided),
        n_resels=None if n_resels is None else float(n_resels),
        null=null,
        calibration=calibration,
        sign_flip=sign_flip,
    )


__all__ = [
    "EmpiricalCalibration",
    "EmpiricalNull",
    "SignFlipSummary",
    "ThresholdContext",
    "sign_flip_p_floor",
    "bonferroni_threshold",
    "empirical_calibration",
    "empirical_null",
    "expected_false_positives",
    "expected_false_positives_under",
    "fdr_p_cutoff",
    "fdr_threshold",
    "p_values",
    "rft_voxel_threshold",
    "threshold_context",
]
