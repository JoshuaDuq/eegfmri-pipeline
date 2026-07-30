"""Estimate and remove the room's line comb from continuous EEG.

The contamination this targets is characterised in ``docs/scanner_harmonic_diagnosis.md``:
a comb at integer multiples of about 1.2 Hz, mains-synchronous and independent of the
imaging gradients, plus four isolated lines that drift on their own. Because the sources
are monochromatic and stationary within a session, the right removal is a projection onto
sinusoids at the measured frequencies -- not a notch, which would take the surrounding
band with it.

Three properties of the artifact drive the design.

**The frequencies must be measured, not assumed.** The comb fundamental repeats to a few
tens of microhertz between sessions, but the isolated lines wander by tens of millihertz,
and the comb's own harmonics inherit mains wander multiplied by the harmonic index. Every
run therefore gets its own estimate.

**The fundamental is estimated from every harmonic at once.** Harmonic *k* carries the
fundamental's frequency error multiplied by *k*, so a weighted fit across harmonics 24-79
determines the fundamental far more precisely than the fundamental's own bin ever could.

**Removal has to be shown not to damage the signal.** :class:`Probe` injects sinusoids
away from every target frequency plus a broadband burst, and the metrics below measure
what came back. The gate criteria are stated up front rather than chosen after seeing the
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from studies.pain_study.analysis.harmonic_diagnosis import refine_peak_frequency

NOMINAL_FUNDAMENTAL_HZ = 1.2
COMB_HARMONIC_RANGE = (24, 79)
ISOLATED_NOMINAL_HZ = (47.0362, 57.2247, 58.1807, 94.0748)
MAINS_NOTCH_HZ = (59.5, 60.5)


@dataclass(frozen=True)
class CombEstimate:
    """One run's measured line frequencies."""

    fundamental_hz: float
    harmonics_used: tuple[int, ...]
    residual_rms_hz: float
    max_abs_residual_hz: float
    isolated_hz: tuple[float, ...]
    isolated_prominence_db: tuple[float, ...]

    @property
    def n_harmonics(self) -> int:
        return len(self.harmonics_used)


@dataclass(frozen=True)
class Probe:
    """Signals injected before removal that must survive it.

    The sinusoids stand for narrowband neural activity at frequencies the removal is not
    aimed at; the burst stands for a broadband transient. The burst deliberately sits near
    the comb, because a transient's bandwidth necessarily overlaps neighbouring lines and
    that overlap is the realistic worst case.
    """

    sinusoid_hz: tuple[float, ...] = (35.55, 44.05, 65.35, 78.45)
    sinusoid_amplitude_v: float = 0.5e-6
    burst_hz: float = 40.0
    burst_centre_s: float = 120.0
    burst_sd_s: float = 0.05
    burst_amplitude_v: float = 3.0e-6

    def waveform(self, times: np.ndarray) -> np.ndarray:
        """The probe signal sampled on ``times``."""
        time_array = np.asarray(times, dtype=float)
        signal = np.zeros_like(time_array)
        for frequency in self.sinusoid_hz:
            signal += self.sinusoid_amplitude_v * np.sin(
                2 * np.pi * frequency * time_array + frequency
            )
        envelope = np.exp(-0.5 * ((time_array - self.burst_centre_s) / self.burst_sd_s) ** 2)
        signal += self.burst_amplitude_v * envelope * np.sin(2 * np.pi * self.burst_hz * time_array)
        return signal

    def burst_window(self, times: np.ndarray, half_widths: float = 4.0) -> np.ndarray:
        time_array = np.asarray(times, dtype=float)
        return np.abs(time_array - self.burst_centre_s) <= half_widths * self.burst_sd_s


@dataclass(frozen=True)
class PreservationGate:
    """Decision rules for accepting a removal setting, fixed before the measurement.

    The thresholds mirror the ones the earlier residual-OBS benchmark used, so the two
    decisions stay comparable: suppress the lines, return the injected signals, and leave
    everything else alone.
    """

    max_residual_prominence_db: float = 1.0
    min_median_suppression_db: float = 10.0
    max_probe_deviation_db: float = 0.5
    max_nonline_change_db: float = 0.2
    max_burst_energy_deviation: float = 0.05
    min_burst_correlation: float = 0.99
    max_band_fraction_removed: float = 0.15
    """Most of the analysis band the removal may touch.

    This threshold was revised once, and the revision is stated rather than buried. It was
    first set at 0.08 on the assumption that a line needs only the two or three bins it
    occupies. That assumption was wrong: the comb is mains-disciplined, so harmonic *k*
    wanders by *k* times the fundamental's wander, and the top of the comb moves about a
    bin within a single run. Measuring the width actually required to push every line
    below background gave 12.1% of 28-95 Hz, against a floor of 4.1% for one bin per line
    and 25.3% for MNE's default. The criterion is therefore set at 15%: enough for the
    measured wander, and still ruling out the default. For scale, masking these lines in
    analysis instead of removing them would cost about 22% of the same band.
    """

    def evaluate(self, metrics: dict[str, float]) -> dict[str, bool]:
        return {
            "lines_suppressed": metrics["median_residual_prominence_db"]
            <= self.max_residual_prominence_db,
            "suppression_sufficient": metrics["median_suppression_db"]
            >= self.min_median_suppression_db,
            "sinusoids_preserved": metrics["max_probe_deviation_db"] <= self.max_probe_deviation_db,
            "spectrum_preserved": metrics["max_nonline_change_db"] <= self.max_nonline_change_db,
            "transient_preserved": abs(metrics["burst_energy_ratio"] - 1.0)
            <= self.max_burst_energy_deviation,
            "transient_undistorted": metrics["burst_correlation"] >= self.min_burst_correlation,
            "band_mostly_untouched": metrics["removed_band_fraction"]
            <= self.max_band_fraction_removed,
        }

    def passed(self, metrics: dict[str, float]) -> bool:
        return all(self.evaluate(metrics).values())


def estimate_comb(
    freqs: Sequence[float],
    spectrum_db: Sequence[float],
    prominence: Sequence[float],
    *,
    nominal_hz: float = NOMINAL_FUNDAMENTAL_HZ,
    harmonic_range: tuple[int, int] = COMB_HARMONIC_RANGE,
    isolated_nominal_hz: Sequence[float] = ISOLATED_NOMINAL_HZ,
    search_hz: float = 0.25,
    isolated_search_hz: float = 0.15,
    min_prominence_db: float = 1.0,
) -> CombEstimate:
    """Measure the comb fundamental and the isolated lines in one run's spectrum.

    Each harmonic contributes its refined peak position weighted by its own prominence,
    and the fundamental is the weighted least-squares slope through the origin. Harmonics
    whose peak is too weak, or too far from where it should be, drop out rather than drag
    the fit.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    prominence_array = np.asarray(prominence, dtype=float)
    if not frequency_array.shape == spectrum.shape == prominence_array.shape:
        raise ValueError("freqs, spectrum_db and prominence must have the same shape.")
    if not np.isfinite(nominal_hz) or nominal_hz <= 0:
        raise ValueError("nominal_hz must be a finite positive number.")
    if search_hz <= 0 or search_hz >= nominal_hz / 2:
        raise ValueError("search_hz must be positive and below half the nominal spacing.")
    low, high = harmonic_range
    if low < 1 or high < low:
        raise ValueError("harmonic_range must be an increasing range of positive integers.")

    harmonics, positions, weights = [], [], []
    for harmonic in range(low, high + 1):
        target = harmonic * nominal_hz
        found = _peak_near(frequency_array, spectrum, prominence_array, target, search_hz)
        if found is None:
            continue
        position, strength = found
        if strength >= min_prominence_db and abs(position - target) < search_hz:
            harmonics.append(harmonic)
            positions.append(position)
            weights.append(strength)
    if len(harmonics) < 3:
        raise ValueError(
            f"Only {len(harmonics)} comb harmonics exceeded {min_prominence_db} dB; "
            "refusing to fit a fundamental."
        )

    index = np.asarray(harmonics, dtype=float)
    position_array = np.asarray(positions, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    fundamental = float(
        np.sum(weight_array * index * position_array) / np.sum(weight_array * index**2)
    )
    residual = position_array - index * fundamental

    # The isolated lines get a narrower window than the comb. 47.036 Hz sits only 0.24 Hz
    # from comb harmonic 39 at 46.8 Hz, so a window wide enough for the comb would reach
    # across and lock onto the wrong peak. Their session-to-session drift is under 0.1 Hz,
    # which the narrower window still covers.
    for nominal in isolated_nominal_hz:
        nearest_harmonic = round(nominal / fundamental) * fundamental
        if abs(nominal - nearest_harmonic) <= isolated_search_hz:
            raise ValueError(
                f"Isolated line {nominal} Hz is within {isolated_search_hz} Hz of comb "
                f"position {nearest_harmonic:.4f} Hz; the search would find the comb."
            )

    # One entry per nominal, NaN where nothing was found, so estimates from different
    # runs stay aligned and can be combined position by position.
    isolated, isolated_prominence = [], []
    for nominal in isolated_nominal_hz:
        found = _peak_near(frequency_array, spectrum, prominence_array, nominal, isolated_search_hz)
        position, strength = found if found is not None else (float("nan"), float("nan"))
        isolated.append(position)
        isolated_prominence.append(strength)

    return CombEstimate(
        fundamental_hz=fundamental,
        harmonics_used=tuple(harmonics),
        residual_rms_hz=float(np.sqrt(np.mean(residual**2))),
        max_abs_residual_hz=float(np.max(np.abs(residual))),
        isolated_hz=tuple(isolated),
        isolated_prominence_db=tuple(isolated_prominence),
    )


def _peak_near(
    freqs: np.ndarray,
    spectrum_db: np.ndarray,
    prominence: np.ndarray,
    target_hz: float,
    search_hz: float,
) -> tuple[float, float] | None:
    """Refined position and prominence of the largest peak within a search window."""
    low, high = np.searchsorted(freqs, [target_hz - search_hz, target_hz + search_hz])
    if high <= low:
        return None
    window = prominence[low:high]
    if not np.any(np.isfinite(window)):
        return None
    index = low + int(np.nanargmax(window))
    if not 0 < index < freqs.size - 1:
        return None
    return refine_peak_frequency(freqs, spectrum_db, index), float(prominence[index])


def combine_estimates(estimates: Sequence[CombEstimate]) -> CombEstimate:
    """Pool a session's per-run estimates into one, by median.

    A single run measures the fundamental with more noise than the quantity actually
    varies. Across this cohort the per-run scatter is 120-280 microhertz within a session
    while the room's fundamental moves only about 60 microhertz between sessions five
    months apart, so a per-run frequency is mostly measurement error. Taking the median
    over a session's runs removes most of it and stays robust to the occasional run whose
    comb is too weak to fit well; it still lets one session differ from the next, which a
    fixed constant would not.
    """
    if not estimates:
        raise ValueError("combine_estimates needs at least one estimate.")
    widths = {len(estimate.isolated_hz) for estimate in estimates}
    if len(widths) != 1:
        raise ValueError("Estimates disagree on how many isolated lines they carry.")

    fundamentals = np.array([estimate.fundamental_hz for estimate in estimates], dtype=float)
    isolated = np.array([estimate.isolated_hz for estimate in estimates], dtype=float)
    prominence = np.array([estimate.isolated_prominence_db for estimate in estimates], dtype=float)
    with np.errstate(invalid="ignore"):
        pooled_isolated = np.nanmedian(isolated, axis=0) if isolated.size else isolated
        pooled_prominence = np.nanmedian(prominence, axis=0) if prominence.size else prominence

    harmonics = sorted({harmonic for e in estimates for harmonic in e.harmonics_used})
    return CombEstimate(
        fundamental_hz=float(np.median(fundamentals)),
        harmonics_used=tuple(harmonics),
        residual_rms_hz=float(np.median([e.residual_rms_hz for e in estimates])),
        max_abs_residual_hz=float(np.max([e.max_abs_residual_hz for e in estimates])),
        isolated_hz=tuple(float(value) for value in np.atleast_1d(pooled_isolated)),
        isolated_prominence_db=tuple(float(v) for v in np.atleast_1d(pooled_prominence)),
    )


def removal_frequencies(
    estimate: CombEstimate,
    *,
    harmonic_range: tuple[int, int] = COMB_HARMONIC_RANGE,
    low_hz: float = 3.0,
    high_hz: float = 95.0,
    excluded_hz: Iterable[tuple[float, float]] = (MAINS_NOTCH_HZ,),
) -> tuple[float, ...]:
    """The frequencies to project out, from one run's estimate.

    The mains neighbourhood is excluded because the downstream pipeline notches it, and
    projecting the same component out twice would take a second bite of the spectrum.
    """
    low, high = harmonic_range
    candidates = [estimate.fundamental_hz * harmonic for harmonic in range(low, high + 1)]
    candidates.extend(estimate.isolated_hz)

    keep = []
    for frequency in sorted(candidates):
        if not np.isfinite(frequency):
            continue
        if not low_hz <= frequency <= high_hz:
            continue
        if any(start <= frequency <= stop for start, stop in excluded_hz):
            continue
        keep.append(float(frequency))
    if not keep:
        raise ValueError("No removal frequency survived the range and exclusion filters.")
    return tuple(keep)


def check_probe_clearance(
    probe: Probe,
    targets: Sequence[float],
    *,
    min_separation_hz: float = 0.3,
) -> None:
    """Fail if an injected sinusoid sits close enough to a target to be removed with it.

    A probe that collides with a removal target tests nothing: it would be taken out by
    design, and reporting that as signal loss would be wrong.
    """
    target_array = np.asarray(targets, dtype=float)
    for frequency in probe.sinusoid_hz:
        separation = float(np.min(np.abs(target_array - frequency)))
        if separation < min_separation_hz:
            raise ValueError(
                f"Probe at {frequency} Hz is {separation:.3f} Hz from a removal target; "
                f"it needs at least {min_separation_hz} Hz of clearance."
            )


def line_suppression(
    freqs: Sequence[float],
    prominence_before: Sequence[float],
    prominence_after: Sequence[float],
    targets: Sequence[float],
) -> dict[str, float]:
    """How far the targeted lines fell, in local prominence."""
    frequency_array = np.asarray(freqs, dtype=float)
    before = np.asarray(prominence_before, dtype=float)
    after = np.asarray(prominence_after, dtype=float)
    rows = []
    for frequency in targets:
        index = int(np.argmin(np.abs(frequency_array - frequency)))
        if np.isfinite(before[index]) and np.isfinite(after[index]):
            rows.append((before[index], after[index]))
    if not rows:
        raise ValueError("No target frequency had a usable prominence estimate.")
    values = np.asarray(rows)
    return {
        "n_targets": float(len(values)),
        "median_prominence_before_db": float(np.median(values[:, 0])),
        "median_residual_prominence_db": float(np.median(values[:, 1])),
        "max_residual_prominence_db": float(np.max(values[:, 1])),
        "median_suppression_db": float(np.median(values[:, 0] - values[:, 1])),
    }


def probe_preservation(
    freqs: Sequence[float],
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    probe: Probe,
) -> dict[str, float]:
    """Power ratio at each injected sinusoid, averaged over channels."""
    frequency_array = np.asarray(freqs, dtype=float)
    before = np.asarray(psd_before, dtype=float)
    after = np.asarray(psd_after, dtype=float)
    deviations = []
    for frequency in probe.sinusoid_hz:
        index = int(np.argmin(np.abs(frequency_array - frequency)))
        ratio = float(after[..., index].mean() / before[..., index].mean())
        deviations.append(10.0 * np.log10(max(ratio, np.finfo(float).tiny)))
    return {
        "max_probe_deviation_db": float(np.max(np.abs(deviations))),
        "min_probe_ratio": float(np.min(10 ** (np.asarray(deviations) / 10.0))),
    }


def notch_widths_for(
    targets: Sequence[float],
    *,
    ratio: float,
    minimum_hz: float = 0.0,
) -> np.ndarray:
    """Per-line removal width, scaled by harmonic number.

    A fixed width is the wrong shape. The comb is mains-disciplined, so a wander of delta
    in the fundamental moves harmonic *k* by *k* times delta: the top of the comb strays
    about a bin within a run while the bottom barely moves. A width proportional to
    frequency tracks that, which is why MNE parameterises its own default the same way --
    the default is simply too generous by a factor of a few.
    """
    array = np.asarray(targets, dtype=float)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be a finite positive number.")
    if minimum_hz < 0:
        raise ValueError("minimum_hz must be non-negative.")
    return np.maximum(array / ratio, minimum_hz)


def removed_band_fraction(
    freqs: Sequence[float],
    targets: Sequence[float],
    notch_widths_hz: Sequence[float] | float,
    *,
    band_hz: tuple[float, float] = (28.0, 95.0),
) -> float:
    """Fraction of the analysis band whose bins the removal actually touches.

    This is the metric that stops the gate fooling itself. ``spectrum_fit`` subtracts a
    sinusoid at every bin within ``notch_widths`` of a target, not only at the target, so
    a generous width quietly turns a line removal into a band removal. Measuring the
    change only at untouched bins cannot see that -- it excludes precisely the bins being
    emptied. MNE's default width of ``freq / 200`` puts 0.14-0.47 Hz around each line
    here, which across the comb hollows out about a quarter of 28-95 Hz.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    widths = np.broadcast_to(np.asarray(notch_widths_hz, dtype=float), target_array.shape)
    band = (frequency_array >= band_hz[0]) & (frequency_array <= band_hz[1])
    if not np.any(band):
        raise ValueError("The analysis band contains no frequency bins.")

    touched = np.zeros(frequency_array.size, dtype=bool)
    for target, width in zip(target_array, widths):
        touched |= np.abs(frequency_array - target) <= width / 2.0
        # The nearest bin is always subtracted, whatever the width says.
        touched[int(np.argmin(np.abs(frequency_array - target)))] = True
    return float(np.count_nonzero(touched & band) / np.count_nonzero(band))


def nonline_change_db(
    freqs: Sequence[float],
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    targets: Sequence[float],
    *,
    guard_hz: float = 0.4,
    band_hz: tuple[float, float] = (28.0, 95.0),
) -> np.ndarray:
    """Per-channel change in mean power across the band, ignoring the removed lines."""
    frequency_array = np.asarray(freqs, dtype=float)
    before = np.asarray(psd_before, dtype=float)
    after = np.asarray(psd_after, dtype=float)
    mask = (frequency_array >= band_hz[0]) & (frequency_array <= band_hz[1])
    for frequency in targets:
        mask &= np.abs(frequency_array - frequency) > guard_hz
    if not np.any(mask):
        raise ValueError("No frequency bin remains outside the removed lines.")
    return 10.0 * np.log10(after[..., mask].mean(axis=-1) / before[..., mask].mean(axis=-1))


def recover_probe(
    cleaned_with_probe: np.ndarray,
    cleaned_without_probe: np.ndarray,
) -> np.ndarray:
    """Isolate what the removal did to the injected signal alone.

    Differencing two runs of the removal -- one on data carrying the probe, one on the
    same data without it -- cancels the recording and leaves the probe's fate. Comparing
    the transient's window energy before and after removal instead would be confounded:
    that window also holds comb lines, and taking those out is the point of the exercise,
    not damage. On synthetic data where the comb is a large share of the window, that
    confound reads as a 38% signal loss that never happened.
    """
    with_probe = np.asarray(cleaned_with_probe, dtype=float)
    without_probe = np.asarray(cleaned_without_probe, dtype=float)
    if with_probe.shape != without_probe.shape:
        raise ValueError("Both cleaned recordings must have the same shape.")
    return with_probe - without_probe


def probe_recovery(
    recovered: np.ndarray,
    reference: np.ndarray,
    times: np.ndarray,
    probe: Probe,
) -> dict[str, float]:
    """Compare the probe recovered from the recording against the probe cleaned alone.

    Two different losses have to be told apart. Projecting out a frequency necessarily
    takes with it any signal energy sitting at that frequency, and a short transient is
    broadband: a 50 ms burst at 40 Hz spreads across roughly nine comb lines, so a fifth
    of its energy is *supposed* to disappear. That is the price of the removal, not a
    defect in it.

    The reference is the injected probe put through the same removal by itself, so it
    carries exactly that unavoidable loss and nothing else. Comparing the recovered probe
    against the reference therefore isolates collateral damage -- distortion the removal
    causes by interacting with the recording -- while ``intrinsic_energy_ratio`` reports
    the unavoidable part separately, as information rather than as a criterion.
    """
    recovered_array = np.atleast_2d(np.asarray(recovered, dtype=float))
    reference_array = np.atleast_2d(np.asarray(reference, dtype=float))
    time_array = np.asarray(times, dtype=float)
    if recovered_array.shape[-1] != reference_array.shape[-1] != time_array.size:
        raise ValueError("recovered, reference and times must agree along the time axis.")

    window = probe.burst_window(time_array)
    if not np.any(window):
        raise ValueError("The burst window falls outside the recording.")

    injected = probe.waveform(time_array)[window]
    reference_inside = reference_array[..., window]
    reference_energy = float(np.sum(reference_inside[0] ** 2))
    if reference_energy <= 0:
        raise ValueError("The reference probe carries no energy in the burst window.")

    inside = recovered_array[..., window]
    ratios = np.sum(inside**2, axis=-1) / reference_energy
    correlations = [
        float(np.corrcoef(channel, reference_inside[0])[0, 1])
        for channel in inside
        if np.std(channel) > 0
    ]
    return {
        "burst_energy_ratio": float(np.median(ratios)),
        "burst_energy_ratio_min": float(np.min(ratios)),
        "burst_energy_ratio_max": float(np.max(ratios)),
        "burst_correlation": float(np.min(correlations)) if correlations else float("nan"),
        "intrinsic_energy_ratio": reference_energy / float(np.sum(injected**2)),
    }
