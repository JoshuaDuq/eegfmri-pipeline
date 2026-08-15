"""Estimators for diagnosing narrowband scanner-linked contamination across a cohort.

The functions here take arrays and return arrays. Reading recordings, resolving cohort
paths and writing reports belong to the runner script, not to this module.

Two conventions run through the whole module and are worth stating once.

**TR-commensurate frequency grids.** A segment holding an integer number of volume
repetition times puts every volume-comb line ``k / TR`` exactly on a DFT bin centre.
The on-comb/off-comb partition is then exact rather than a nearest-bin approximation,
and scanner lines suffer no scalloping loss. :func:`tr_commensurate_length` picks such a
segment; :func:`comb_index` reports where an arbitrary frequency falls relative to the
comb.

**Local prominence, not absolute power.** A line is only interpretable against the
background beside it. :func:`local_background_db` estimates that background with a
running median wide enough to span several comb lines, so the few line bins inside the
window cannot move it. Prominence is the bin minus that background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

TR_SECONDS = 0.9
"""Volume repetition time of the thermal-pain acquisition, in seconds."""


@dataclass(frozen=True)
class CombPosition:
    """Where one frequency sits relative to the volume comb ``k / TR``."""

    frequency_hz: float
    harmonic_index: int
    offset_hz: float

    @property
    def on_comb(self) -> bool:
        """True when the frequency coincides with a comb line to within 1 uHz."""
        return abs(self.offset_hz) < 1e-6


@dataclass(frozen=True)
class CombFit:
    """Least-squares fit of an arithmetic comb ``f_n = intercept + n * spacing``."""

    intercept_hz: float
    spacing_hz: float
    indices: tuple[int, ...]
    residuals_hz: tuple[float, ...]

    @property
    def rmse_hz(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.residuals_hz))))

    @property
    def max_abs_residual_hz(self) -> float:
        return float(np.max(np.abs(self.residuals_hz)))


def tr_commensurate_length(n_times: int, sfreq: float, tr: float = TR_SECONDS) -> int:
    """Return the longest prefix of a segment spanning a whole number of TRs.

    Raises when one TR does not land on an integer sample count, because a
    non-integer count would defeat the purpose of the commensurate grid.
    """
    if n_times <= 0:
        raise ValueError("n_times must be positive.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a finite positive number.")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("tr must be a finite positive number.")

    samples_per_tr = tr * sfreq
    rounded = int(round(samples_per_tr))
    if abs(samples_per_tr - rounded) > 1e-9:
        raise ValueError(
            f"TR {tr} s at {sfreq} Hz spans {samples_per_tr} samples, which is not an integer."
        )
    n_tr = n_times // rounded
    if n_tr < 1:
        raise ValueError(f"Segment of {n_times} samples is shorter than one TR ({rounded}).")
    return int(n_tr * rounded)


def hann_periodogram(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided power spectral density of Hann-windowed segments.

    ``data`` may carry any number of leading dimensions; the last axis is time. The
    result is scaled so that summing over frequency and multiplying by the bin width
    recovers the mean square of the windowed signal (SciPy's ``density`` scaling).
    """
    array = np.asarray(data, dtype=float)
    if array.ndim < 1 or array.shape[-1] < 2:
        raise ValueError("data must have at least two samples along the last axis.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a finite positive number.")
    if not np.all(np.isfinite(array)):
        raise ValueError("data must contain only finite values.")

    n_times = array.shape[-1]
    window = np.hanning(n_times)
    normalisation = sfreq * float(np.sum(window**2))
    spectrum = np.fft.rfft(array * window, axis=-1)
    psd = np.abs(spectrum) ** 2 / normalisation
    # Fold negative-frequency power onto the positive bins, leaving DC and Nyquist alone.
    if psd.shape[-1] > 2:
        psd[..., 1:-1] *= 2.0
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    return freqs, psd


def to_db(psd: np.ndarray) -> np.ndarray:
    """Convert power to decibels, guarding against log of zero."""
    array = np.asarray(psd, dtype=float)
    if np.any(array < 0):
        raise ValueError("psd must be non-negative.")
    return 10.0 * np.log10(np.maximum(array, np.finfo(float).tiny))


def local_background_db(
    spectrum_db: Sequence[float],
    *,
    half_width_bins: int,
    core_bins: int = 1,
) -> np.ndarray:
    """Running median of a decibel spectrum, excluding a core around each bin.

    The window must be wide enough to hold many background bins beside the few line
    bins it inevitably contains, so that the median is set by the background. The
    ``core_bins`` exclusion keeps a line out of its own background estimate; with a Hann
    window a pure tone occupies three bins, so ``core_bins=1`` is the natural choice.

    Bins closer to either edge than ``half_width_bins`` get no symmetric window and are
    returned as NaN rather than estimated from a lopsided one. Under a convex 1/f
    background the symmetric median sits slightly above the centre bin, which biases
    prominence downward; the estimator is conservative in that direction by design.
    """
    values = np.asarray(spectrum_db, dtype=float)
    if values.ndim != 1:
        raise ValueError("spectrum_db must be one-dimensional.")
    if half_width_bins < 1:
        raise ValueError("half_width_bins must be at least 1.")
    if core_bins < 0:
        raise ValueError("core_bins must be non-negative.")
    if core_bins >= half_width_bins:
        raise ValueError("core_bins must be smaller than half_width_bins.")
    width = 2 * half_width_bins + 1
    if values.size < width:
        raise ValueError(f"spectrum_db needs at least {width} bins for this window.")

    windows = np.lib.stride_tricks.sliding_window_view(values, width)
    keep = np.ones(width, dtype=bool)
    keep[half_width_bins - core_bins : half_width_bins + core_bins + 1] = False
    background = np.full(values.size, np.nan)
    background[half_width_bins : values.size - half_width_bins] = np.median(
        windows[:, keep], axis=1
    )
    return background


def prominence_db(
    spectrum_db: Sequence[float],
    *,
    half_width_bins: int,
    core_bins: int = 1,
) -> np.ndarray:
    """Decibel spectrum minus its local background. NaN where no window fits."""
    values = np.asarray(spectrum_db, dtype=float)
    background = local_background_db(values, half_width_bins=half_width_bins, core_bins=core_bins)
    return values - background


def robust_null(values: Sequence[float]) -> tuple[float, float]:
    """Estimate location and scale of a null contaminated only in its upper tail.

    Prominence is zero-centred by construction wherever there is no line, so the bulk of
    a prominence spectrum is null and the lines sit in the upper tail. Fitting the scale
    from the lower half keeps the lines themselves from inflating it: for a Gaussian,
    the gap between the median and the 15.87th percentile is exactly one sigma.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size < 32:
        raise ValueError("robust_null needs at least 32 finite values.")
    location = float(np.median(finite))
    scale = location - float(np.percentile(finite, 15.865525393145702))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Could not estimate a positive scale from the lower tail.")
    return location, scale


def upper_tail_pvalues(values: Sequence[float]) -> np.ndarray:
    """One-sided Gaussian p-values against a null fitted to the lower tail."""
    array = np.asarray(values, dtype=float)
    location, scale = robust_null(array)
    from scipy.stats import norm

    return np.asarray(norm.sf((array - location) / scale), dtype=float)


def fdr_bh(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values, ordered as the input."""
    array = np.asarray(pvalues, dtype=float)
    if array.ndim != 1:
        raise ValueError("pvalues must be one-dimensional.")
    if array.size == 0:
        raise ValueError("pvalues must be non-empty.")
    if np.any(~np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise ValueError("pvalues must be finite and inside [0, 1].")

    n = array.size
    order = np.argsort(array)
    ranked = array[order] * n / np.arange(1, n + 1)
    qvalues = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(qvalues, 1.0)
    return out


def cluster_peaks(
    significant: Sequence[bool],
    values: Sequence[float],
    *,
    join_gap_bins: int = 1,
) -> list[int]:
    """Reduce runs of significant bins to one representative bin each.

    A Hann-windowed line occupies three bins and a strong one drags its shoulders above
    threshold too, so significance arrives in clusters rather than at single bins.
    Neighbouring clusters separated by no more than ``join_gap_bins`` quiet bins are
    treated as one line, and each surviving cluster is represented by its largest bin.
    """
    flags = np.asarray(significant, dtype=bool)
    magnitudes = np.asarray(values, dtype=float)
    if flags.shape != magnitudes.shape:
        raise ValueError("significant and values must have the same shape.")
    if flags.ndim != 1:
        raise ValueError("significant must be one-dimensional.")
    if join_gap_bins < 0:
        raise ValueError("join_gap_bins must be non-negative.")

    indices = np.flatnonzero(flags)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > join_gap_bins + 1)
    groups = np.split(indices, breaks + 1)
    return [int(group[int(np.argmax(magnitudes[group]))]) for group in groups]


def refine_peak_frequency(
    freqs: Sequence[float],
    spectrum_db: Sequence[float],
    index: int,
) -> float:
    """Sub-bin peak location from a quadratic through three decibel samples.

    A Hann-windowed tone has a near-parabolic log-magnitude peak, so interpolating the
    apex recovers the frequency to a small fraction of a bin. The correction is clipped
    to half a bin; a wider excursion means the bin is not a local maximum and the bin
    centre is returned instead.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    values = np.asarray(spectrum_db, dtype=float)
    if frequency_array.shape != values.shape:
        raise ValueError("freqs and spectrum_db must have the same shape.")
    if not 0 < index < values.size - 1:
        raise ValueError("index must have a neighbour on either side.")

    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * centre + right
    if denominator == 0:
        return float(frequency_array[index])
    shift = 0.5 * (left - right) / denominator
    if not np.isfinite(shift) or abs(shift) > 0.5:
        return float(frequency_array[index])
    bin_width = float(frequency_array[1] - frequency_array[0])
    return float(frequency_array[index] + shift * bin_width)


def hann_resolution_hz(segment_seconds: float) -> float:
    """Half-power width of a Hann-windowed pure tone: the narrowest peak measurable."""
    if not np.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("segment_seconds must be a finite positive number.")
    return 1.4382 / float(segment_seconds)


def spectral_linewidth_hz(
    freqs: Sequence[float],
    spectrum_db: Sequence[float],
    index: int,
    *,
    drop_db: float = 3.0,
    max_search_bins: int = 200,
) -> float:
    """Half-power width of the peak at ``index``, interpolated between bins.

    This is what separates an instrument line from a brain rhythm. A monochromatic
    source is as narrow as the window allows -- 1.44/T -- while an alpha peak is a
    biological resonance one to two hertz wide. Returns NaN when the peak does not fall
    by ``drop_db`` on both sides within the search range, which is itself informative:
    the feature is broader than the search window.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    values = np.asarray(spectrum_db, dtype=float)
    if frequency_array.shape != values.shape:
        raise ValueError("freqs and spectrum_db must have the same shape.")
    if not 0 <= index < values.size:
        raise ValueError("index must be inside the spectrum.")
    if drop_db <= 0:
        raise ValueError("drop_db must be positive.")

    target = values[index] - drop_db
    bin_width = float(frequency_array[1] - frequency_array[0])

    def crossing(step: int) -> float | None:
        previous = values[index]
        for offset in range(1, max_search_bins + 1):
            position = index + step * offset
            if not 0 <= position < values.size:
                return None
            current = values[position]
            if current <= target:
                span = previous - current
                fraction = (previous - target) / span if span > 0 else 0.0
                return (offset - 1 + fraction) * bin_width
            previous = current
        return None

    left, right = crossing(-1), crossing(1)
    if left is None or right is None:
        return float("nan")
    return float(left + right)


def comb_index(frequency_hz: float, tr: float = TR_SECONDS) -> CombPosition:
    """Locate one frequency relative to the volume comb ``k / TR``."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be a finite positive number.")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("tr must be a finite positive number.")
    exact = frequency_hz * tr
    harmonic = int(round(exact))
    return CombPosition(
        frequency_hz=float(frequency_hz),
        harmonic_index=harmonic,
        offset_hz=float(frequency_hz - harmonic / tr),
    )


def comb_phase(frequencies: Sequence[float], tr: float = TR_SECONDS) -> np.ndarray:
    """Fractional position of each frequency between adjacent comb lines, in [0, 1).

    A set of frequencies unrelated to the comb spreads this value uniformly; a set drawn
    from the comb piles it up at zero. :func:`comb_uniformity_pvalue` tests the
    difference.
    """
    array = np.asarray(frequencies, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("frequencies must be finite and positive.")
    return np.mod(array * tr, 1.0)


def comb_uniformity_pvalue(frequencies: Sequence[float], tr: float = TR_SECONDS) -> float:
    """Rayleigh p-value for clustering of comb phase at zero.

    Mapping the fractional comb position onto the circle turns "are these frequencies
    on the comb?" into a directional-statistics question, which avoids binning.
    """
    phases = 2.0 * np.pi * comb_phase(frequencies, tr=tr)
    return rayleigh_test(phases)[1]


def fit_arithmetic_comb(frequencies: Sequence[float]) -> CombFit:
    """Fit ``f_n = intercept + n * spacing`` to an ordered family of lines.

    Indices come from rounding each line's offset from the first against the median
    adjacent spacing, so a family with a missing member still fits correctly.
    """
    array = np.sort(np.asarray(frequencies, dtype=float))
    if array.size < 3:
        raise ValueError("fit_arithmetic_comb needs at least three frequencies.")
    if np.any(~np.isfinite(array)):
        raise ValueError("frequencies must be finite.")

    steps = np.diff(array)
    if np.any(steps <= 0):
        raise ValueError("frequencies must be distinct.")
    unit = float(np.median(steps))
    indices = np.rint((array - array[0]) / unit).astype(int)
    if np.unique(indices).size != indices.size:
        raise ValueError("Frequencies do not separate onto distinct comb indices.")

    design = np.column_stack([np.ones(indices.size), indices.astype(float)])
    (intercept, spacing), *_ = np.linalg.lstsq(design, array, rcond=None)
    residuals = array - (intercept + spacing * indices)
    return CombFit(
        intercept_hz=float(intercept),
        spacing_hz=float(spacing),
        indices=tuple(int(i) for i in indices),
        residuals_hz=tuple(float(r) for r in residuals),
    )


def pairwise_differences(frequencies: Sequence[float], *, max_difference_hz: float) -> np.ndarray:
    """All positive pairwise gaps up to a ceiling, for spotting a repeated spacing."""
    array = np.asarray(frequencies, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("frequencies must be a 1D array with at least two entries.")
    if not np.isfinite(max_difference_hz) or max_difference_hz <= 0:
        raise ValueError("max_difference_hz must be a finite positive number.")
    outer = array[None, :] - array[:, None]
    gaps = outer[np.triu_indices(array.size, k=1)]
    gaps = gaps[gaps > 0]
    return np.sort(gaps[gaps <= max_difference_hz])


def dominant_spacing(
    frequencies: Sequence[float],
    *,
    max_difference_hz: float,
    tolerance_hz: float,
) -> tuple[float, int]:
    """Most frequently repeated pairwise gap, and how many pairs support it.

    Each observed gap is scored by how many other gaps sit within ``tolerance_hz`` of
    it, and the winner is refined to the mean of its supporters.
    """
    gaps = pairwise_differences(frequencies, max_difference_hz=max_difference_hz)
    if gaps.size == 0:
        raise ValueError("No pairwise differences below the ceiling.")
    if not np.isfinite(tolerance_hz) or tolerance_hz <= 0:
        raise ValueError("tolerance_hz must be a finite positive number.")

    counts = np.array([np.sum(np.abs(gaps - gap) <= tolerance_hz) for gap in gaps])
    best = int(np.argmax(counts))
    supporters = gaps[np.abs(gaps - gaps[best]) <= tolerance_hz]
    return float(np.mean(supporters)), int(supporters.size)


def comb_members(
    frequencies: Sequence[float],
    fundamental: float,
    *,
    tolerance_hz: float,
) -> np.ndarray:
    """Boolean mask of frequencies within tolerance of an integer multiple."""
    array = np.asarray(frequencies, dtype=float)
    if not np.isfinite(fundamental) or fundamental <= 0:
        raise ValueError("fundamental must be a finite positive number.")
    return np.abs(array - np.rint(array / fundamental) * fundamental) <= tolerance_hz


_SPACING_SEARCH_FRACTION = 0.02
"""How far either side of a supplied gap the fundamental is looked for.

Two percent is a refinement of a pair spacing, not a search for a period: the gap this
site's diagnosis hands over is 1.206 Hz against a true 1.2, which is 0.5%.
"""


def refine_comb_fundamental(
    frequencies: Sequence[float],
    spacing: float,
    *,
    tolerance_hz: float,
    max_divisor: int = 6,
    min_gain: float = 0.2,
) -> tuple[float, np.ndarray]:
    """Reduce a repeated spacing to the fundamental that actually generates the comb.

    The commonest gap between detected lines is not the fundamental. A comb missing some
    of its members has more pairs two harmonics apart than adjacent ones, which makes the
    most-supported gap a multiple of the true period. Dividing recovers it.

    Subdividing can never explain fewer lines, so a plain "explains more" rule would
    subdivide without end. A divisor is therefore accepted only when it explains at least
    ``min_gain`` more in relative terms; a genuine halving of the period roughly doubles
    the membership, while spurious subdivision picks up a stray line or two.

    Two corrections make that rule behave as described.

    The gap is refined before it is used. It arrives as a pair spacing, good to a few
    millihertz, while membership is tested against an absolute tolerance -- and a spacing
    error grows as ``k * error``, so 6 mHz at harmonic 78 is 0.47 Hz, eight times a typical
    0.06 Hz tolerance. Measured on sub-0008's baseline, a gap of 1.206 Hz explained 2 of 62
    narrow lines where 1.199 explained 42, which left the divisor sweep a base of 2 to beat
    and handed it the sixth subharmonic.

    And the comparison is on membership *in excess of chance*, not raw membership. A grid
    of spacing ``s`` admits any line within ``tolerance_hz`` of a multiple, which is
    ``2 * tolerance_hz / s`` of the axis -- 60% for a 0.2 Hz grid against 10% for a 1.2 Hz
    one. Subdivision therefore bought membership simply by getting denser: on that same
    recording 0.200 Hz "explained" 54 lines of which about 37 are chance, against 42 for
    1.199 of which about 6 are. Excess leaves a real halving untouched, since halving
    doubles the chance term and the observed count alike.
    """
    array = np.asarray(frequencies, dtype=float)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a finite positive number.")
    if max_divisor < 1:
        raise ValueError("max_divisor must be at least 1.")

    # Every divisor is taken from the gap as supplied and refined afterwards, never from an
    # already-refined or already-accepted spacing. The gap is frequently a multiple that
    # explains nothing on its own -- a 1.2 Hz comb missing every third member has a
    # dominant gap of 3.6 Hz and no line on that grid at all -- so refining it first only
    # lets it wander, and dividing the wandered value misses the exact submultiple that
    # dividing the original hits.
    best = _spacing_refined_locally(array, spacing, tolerance_hz)
    best_excess, best_members = _comb_excess(array, best, tolerance_hz)
    for divisor in range(2, max_divisor + 1):
        candidate = _spacing_refined_locally(array, spacing / divisor, tolerance_hz)
        excess, members = _comb_excess(array, candidate, tolerance_hz)
        threshold = best_excess * (1.0 + min_gain) if best_excess > 0 else 0.0
        if excess >= threshold and excess > 0:
            best, best_excess, best_members = candidate, excess, members

    # Searching over spacing can manufacture membership, so the excess has to pay for it.
    # A grid fine enough admits everything, and with the spacing free rather than given, a
    # few lines can always be made to look periodic: sub-0008's cleaned baseline left three
    # 57 Hz cluster peaks that were fitted at 0.167 Hz -- a spacing unrelated to the 1.2 Hz
    # comb -- and that spurious family failed the recording's verification.
    #
    # Chance membership is binomial: each of ``n`` lines falls inside the grid's covered
    # fraction independently, so its spread is the usual sqrt(n p (1-p)). Requiring the
    # excess to clear two of those keeps the real comb (35.8 against 4.7 on that
    # recording) and drops the cluster (0.85 against 1.6).
    covered = min(1.0, 2.0 * tolerance_hz / best)
    deviation = float(np.sqrt(array.size * covered * (1.0 - covered)))
    if best_excess < 2.0 * deviation:
        return float(best), np.zeros(array.size, dtype=bool)
    return float(best), best_members


def _comb_excess(
    array: np.ndarray, spacing: float, tolerance_hz: float
) -> tuple[float, np.ndarray]:
    """Members of the grid, and how many more that is than chance would supply.

    A grid of this spacing accepts any line within ``tolerance_hz`` of a multiple, which is
    ``2 * tolerance_hz / spacing`` of the frequency axis. Scoring the raw count instead
    rewards a spacing for nothing but being dense.
    """
    members = comb_members(array, spacing, tolerance_hz=tolerance_hz)
    covered = min(1.0, 2.0 * tolerance_hz / spacing)
    return float(members.sum() - array.size * covered), members


def _spacing_refined_locally(array: np.ndarray, spacing: float, tolerance_hz: float) -> float:
    """Best spacing within a fraction of a percent of the one supplied.

    Only a refinement: the search spans ``_SPACING_SEARCH_FRACTION`` either side, so it
    corrects the pair gap's own error and cannot wander onto an unrelated period. The step
    keeps the highest harmonic moving well under the tolerance, because that harmonic is
    where a small spacing error first shows.
    """
    highest = float(np.max(np.abs(array))) if array.size else 0.0
    harmonics = max(1.0, highest / spacing)
    step = tolerance_hz / harmonics / 4.0
    span = spacing * _SPACING_SEARCH_FRACTION
    if not np.isfinite(step) or step <= 0.0 or span <= 0.0:
        return float(spacing)
    # The supplied spacing is always a candidate. An arange over the span need not land on
    # it, and a spacing that is already exact must not be nudged onto the nearest grid
    # point -- 122 uHz is nothing until harmonic 49 multiplies it into 6 mHz.
    candidates = np.arange(spacing - span, spacing + span + step, step)
    candidates = np.append(candidates[candidates > 0.0], float(spacing))
    if candidates.size == 0:
        return float(spacing)

    # Ranked on membership and then on fit, deliberately not on excess-over-chance. The
    # chance term is what makes different divisors comparable; inside one scan it only
    # varies as 1/spacing, which would tilt every tie towards the widest grid still holding
    # its members and leave the offset the highest harmonic multiplies back up.
    #
    # And ties are the normal case here: on a comb the grid already explains completely,
    # every spacing for some way either side explains it just as completely. Among equal
    # membership the closest fit is the fundamental.
    scored = []
    for candidate in candidates:
        members = comb_members(array, float(candidate), tolerance_hz=tolerance_hz)
        if members.any():
            selected = array[members]
            residual = selected - np.rint(selected / candidate) * candidate
            misfit = float(np.sqrt(np.mean(residual**2)))
        else:
            misfit = np.inf
        scored.append((-int(members.sum()), misfit, float(candidate)))
    return min(scored)[2]


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    statistic: str = "median",
) -> tuple[float, float, float]:
    """Percentile bootstrap over the sampling unit. Returns (point, low, high)."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise ValueError("bootstrap_ci needs at least two finite values.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be inside (0, 1).")
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'.")

    reducer = np.median if statistic == "median" else np.mean
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_resamples, array.size))
    replicates = reducer(array[draws], axis=1)
    low, high = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(reducer(array)), float(low), float(high)


def rayleigh_test(phases: Sequence[float]) -> tuple[float, float]:
    """Rayleigh test of circular uniformity. Returns (mean resultant length, p).

    Uses Zar's approximation to the exact p-value, which is accurate for n above
    roughly ten and conservative below it.
    """
    array = np.asarray(phases, dtype=float)
    array = array[np.isfinite(array)]
    n = array.size
    if n < 2:
        raise ValueError("rayleigh_test needs at least two phases.")
    resultant = float(np.abs(np.sum(np.exp(1j * array))))
    mean_resultant = resultant / n
    pvalue = np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n**2 - resultant**2)) - (1.0 + 2.0 * n))
    return mean_resultant, float(min(max(pvalue, 0.0), 1.0))


def volume_locked_phases(
    coefficients: Sequence[complex],
    volume_offsets_s: Sequence[float],
    frequency_hz: float,
) -> np.ndarray:
    """Rotate per-segment DFT phases onto a shared volume-grid origin.

    Each segment starts at an arbitrary point in the volume cycle. Undoing that offset
    puts every segment's phase in the scanner's own frame, where a deterministic
    volume-locked residual is a fixed phase and anything else is not. The rotation only
    has meaning at frequencies that complete a whole number of cycles per TR, since only
    those have a phase that repeats from one volume to the next.
    """
    coefficient_array = np.asarray(coefficients, dtype=complex)
    offset_array = np.asarray(volume_offsets_s, dtype=float)
    if coefficient_array.shape != offset_array.shape:
        raise ValueError("coefficients and volume_offsets_s must have the same shape.")
    if coefficient_array.size == 0:
        raise ValueError("coefficients must be non-empty.")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be a finite positive number.")
    rotation = np.exp(2j * np.pi * frequency_hz * offset_array)
    return np.angle(coefficient_array * rotation)


def one_way_variance_components(
    values: Sequence[float],
    groups: Sequence[object],
) -> dict[str, float]:
    """Split variance into between-group and within-group parts.

    Uses the balanced one-way random-effects mean squares, with the group size taken as
    the harmonic-style effective size when groups differ slightly. A negative between
    estimate is reported as zero, which is the usual convention and means the data carry
    no evidence of a group effect.
    """
    value_array = np.asarray(values, dtype=float)
    group_array = np.asarray(groups)
    if value_array.shape != group_array.shape:
        raise ValueError("values and groups must have the same shape.")
    finite = np.isfinite(value_array)
    value_array, group_array = value_array[finite], group_array[finite]

    labels = np.unique(group_array)
    if labels.size < 2:
        raise ValueError("one_way_variance_components needs at least two groups.")
    sizes = np.array([np.sum(group_array == label) for label in labels], dtype=float)
    if np.any(sizes < 2):
        raise ValueError("Every group needs at least two observations.")

    grand_mean = float(np.mean(value_array))
    group_means = np.array([np.mean(value_array[group_array == label]) for label in labels])
    total = value_array.size
    ss_between = float(np.sum(sizes * (group_means - grand_mean) ** 2))
    ss_within = float(
        np.sum(
            [
                np.sum((value_array[group_array == label] - mean) ** 2)
                for label, mean in zip(labels, group_means)
            ]
        )
    )
    df_between = labels.size - 1
    df_within = total - labels.size
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    n_effective = (total - np.sum(sizes**2) / total) / df_between
    between = max((ms_between - ms_within) / n_effective, 0.0)
    icc = between / (between + ms_within) if (between + ms_within) > 0 else 0.0
    return {
        "variance_between": float(between),
        "variance_within": float(ms_within),
        "icc": float(icc),
        "n_groups": float(labels.size),
        "n_observations": float(total),
    }


def band_power_db(
    freqs: Sequence[float],
    psd: Sequence[float],
    *,
    low_hz: float,
    high_hz: float,
    excluded_hz: Iterable[tuple[float, float]] = (),
) -> float:
    """Integrate power across a band, in decibels, optionally dropping intervals.

    Integration happens in the power domain and the decibel conversion comes last, so
    the result is band power. Averaging decibels instead would take a geometric mean and
    would understate exactly the narrow high-amplitude bins this analysis is about.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    psd_array = np.asarray(psd, dtype=float)
    if frequency_array.shape != psd_array.shape:
        raise ValueError("freqs and psd must have the same shape.")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz.")

    mask = (frequency_array >= low_hz) & (frequency_array <= high_hz)
    for start, stop in excluded_hz:
        mask &= ~((frequency_array >= start) & (frequency_array <= stop))
    if not np.any(mask):
        raise ValueError("No frequency bins remain inside the band after exclusions.")

    bin_width = float(frequency_array[1] - frequency_array[0])
    power = float(np.sum(psd_array[mask]) * bin_width)
    return float(10.0 * np.log10(max(power, np.finfo(float).tiny)))


def line_excess_fraction(
    freqs: Sequence[float],
    psd: Sequence[float],
    *,
    low_hz: float,
    high_hz: float,
    line_freqs: Sequence[float],
    half_width_bins: int,
    line_half_width_hz: float = 0.15,
) -> float:
    """Fraction of a band's power that sits *above the local background* at the lines.

    This is the honest way to ask how much of a band is artifact, and it is not what
    dropping the line bins and comparing band powers measures. Excluding bins removes
    their background along with their line, so the difference counts ordinary spectrum as
    contamination: on a flat spectrum, masking a fifth of the bins reads as a fifth of the
    power being artifact when none of it is. Over the comb that inflation is large -- the
    gamma share it produced was 47% against a true 35%.

    Here each line bin contributes only its excess over the running-median background,
    clipped at zero so a bin the removal has dug below its surroundings cannot count as
    negative artifact.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(psd, dtype=float)
    if frequency_array.shape != spectrum.shape:
        raise ValueError("freqs and psd must have the same shape.")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz.")
    if line_half_width_hz <= 0:
        raise ValueError("line_half_width_hz must be positive.")

    inside = (frequency_array >= low_hz) & (frequency_array <= high_hz)
    if not np.any(inside):
        raise ValueError("No frequency bin falls inside the band.")

    bin_width = float(frequency_array[1] - frequency_array[0])
    total = float(np.sum(spectrum[inside])) * bin_width
    if total <= 0:
        raise ValueError("The band carries no power.")

    at_a_line = np.zeros(frequency_array.size, dtype=bool)
    for frequency in line_freqs:
        if low_hz <= frequency <= high_hz:
            at_a_line |= np.abs(frequency_array - frequency) <= line_half_width_hz
    if not np.any(at_a_line & inside):
        return 0.0

    # A background is only needed where a line is. Bands sitting too close to DC for a
    # symmetric window -- delta, here -- hold no lines anyway and must not fail for it.
    background = 10.0 ** (
        local_background_db(to_db(spectrum), half_width_bins=half_width_bins) / 10.0
    )
    selected = at_a_line & inside & np.isfinite(background)
    if not np.any(selected):
        raise ValueError("Lines fall inside the band but none has a usable background.")
    excess = float(np.sum(np.clip(spectrum[selected] - background[selected], 0.0, None)))
    return excess * bin_width / total


def line_exclusion_windows(
    line_freqs: Sequence[float],
    *,
    half_width_hz: float,
) -> tuple[tuple[float, float], ...]:
    """Symmetric exclusion intervals around each line, merged where they overlap."""
    array = np.sort(np.asarray(line_freqs, dtype=float))
    if array.size == 0:
        return ()
    if not np.isfinite(half_width_hz) or half_width_hz <= 0:
        raise ValueError("half_width_hz must be a finite positive number.")

    merged: list[list[float]] = []
    for centre in array:
        start, stop = centre - half_width_hz, centre + half_width_hz
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return tuple((float(start), float(stop)) for start, stop in merged)
