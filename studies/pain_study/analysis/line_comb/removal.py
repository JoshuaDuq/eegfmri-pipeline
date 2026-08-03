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
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from studies.pain_study.analysis.line_comb.diagnosis import refine_peak_frequency

NOMINAL_FUNDAMENTAL_HZ = 1.2
COMB_HARMONIC_RANGE = (24, 79)
"""Harmonics used to fit the fundamental: the span the cohort fit was built on.

Only well-determined harmonics belong here. Including weak ones would let a poorly
localised peak pull the fundamental, which every other harmonic then inherits.
"""
REMOVAL_HARMONIC_RANGE = (22, 82)
"""Harmonics actually projected out, which reaches further than the fit does at both ends.

The diagnosis detected comb membership down to harmonic 22 (26.40 Hz) and 23 (27.60 Hz),
below the span the fundamental was fitted on. Both sit within 5 mHz of their comb position
-- a coincidence that precise has probability under one per cent per detection -- so they
are lines, and their rhythm-like measured width is the low-amplitude broadening that
half-power width shows at any low signal-to-noise ratio.

Harmonic 11 (13.23 Hz) is deliberately left in place. It sits 30 mHz off its comb position
rather than 5, appears in only two of fifteen participants, and lands at the alpha-beta
boundary where real rhythms live; removing it would risk taking signal for an artifact
that may not be there.

At the top it now reaches harmonics 80-82 (96.0, 97.2, 98.4 Hz), which the earlier 95 Hz
ceiling left in the delivered data. 97.2 Hz is present in all fifteen participants with no
measurable frequency scatter. These sit above the bands this study analyses, so removing
them is hygiene rather than a result.
"""
ISOLATED_NOMINAL_HZ = (47.0362, 57.2247, 57.3485, 58.1807, 58.3442, 94.0748)
MAINS_NOTCH_HZ = (59.5, 60.5)

#: How much spectrum a resolved isolated line claims for itself, so that a nominal with an
#: overlapping window does not report the same peak again. Set to the 0.109 Hz half-power
#: width the diagnosis measured: wide enough to cover a line and the skirt that makes it
#: the tallest thing nearby, and narrower than the 0.124 Hz separating the closest pair of
#: nominals in use, so a line that is really there can still be found beside a claimed one.
_LINE_CLAIM_HZ = 0.109

#: Fewest harmonics that may carry a fundamental which then authorises the removal grid.
#:
#: Three was the old floor, and the fit it produces licenses removing every harmonic from
#: 22 to 83 -- sixty-one targets from three peaks. Across ninety real runs the fit uses
#: 52-56 harmonics, so this floor is far below anything genuine and only rules out a fit
#: with no evidence behind it.
MIN_HARMONICS_FOR_FIT = 20

#: Most the fitted harmonics may scatter about their arithmetic grid, in Hz RMS.
#:
#: A comb is an arithmetic series; peaks that do not lie on one are not a comb, however
#: many there are. Across ninety real runs the scatter is 0.030-0.156 Hz with a 99th
#: percentile of 0.111; three mutually inconsistent peaks produced 0.228 Hz and still
#: generated the full grid. The bound sits at 0.20 Hz -- a sixth of the 1.2 Hz spacing,
#: above every real run and below the constructed failure.
MAX_FIT_RESIDUAL_RMS_HZ = 0.20

#: How far either side of a target a residual is still that target's responsibility.
#:
#: The notch's own width is the wrong region to search. The failure being looked for is a
#: target that missed, and a missed line then sits just outside what the notch claimed --
#: precisely where the claimed width cannot see. This is the frequency-uncertainty scale of
#: the estimate instead: comb harmonics wander with the fundamental times the harmonic
#: index, and the isolated lines are searched over the same 0.15 Hz. Kept well inside the
#: 0.6 Hz half-spacing so one target is never charged with the next harmonic's line.
RESIDUAL_SEARCH_HZ = 0.15


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

    max_residual_excess_db: float = 1.0
    """How far the worst residual may stand above a blind control of the same search.

    Replaces a fixed bound on the residual itself. Searching a window around every target
    to catch a displaced line means taking the maximum of roughly a thousand bins, whose
    noise floor is several dB before any line survives. Control windows of the same width,
    placed where no target is, measure that floor on the same data, so what is gated is the
    excess over it -- a criterion that means the same thing at any search width or noise
    level.
    """
    max_residual_prominence_db: float = 1.0
    min_median_suppression_db: float = 10.0
    max_probe_deviation_db: float = 0.5
    max_nonline_change_db: float = 0.2
    max_burst_energy_deviation: float = 0.05
    min_burst_correlation: float = 0.99
    max_intrinsic_energy_ratio: float = 1.05
    """Most of the injected transient's energy that may come back.

    A floor alone cannot catch a removal that *adds* energy where the transient was, which
    is as much a defect as one that takes it away. The superseded gate covered this and the
    replacement has to keep it.
    """
    min_intrinsic_energy_ratio: float = 0.85
    """Least of the injected transient's window energy that must survive removal.

    This is the criterion the transient gates were missing. They compared the removal's
    effect on data+probe against its effect on the probe alone, and spectrum_fit is linear
    in the data -- measured here at 7.6e-14 relative -- so those two quantities are equal
    by construction. Both gates read exactly 1.0 on every run of every benchmark, for any
    settings, and could not fail. ``intrinsic_energy_ratio`` was computed all along and
    reported "as information rather than as a criterion"; on the delivered benchmark it
    ranged 0.899 to 0.960, so it is the quantity that actually varies.

    The floor is derived rather than read off those numbers. A 50 ms Gaussian burst at
    40 Hz has a spectral width of 1/(2*pi*0.05) = 3.2 Hz, so it spans about 13 Hz and
    crosses roughly eleven comb lines at 1.2 Hz spacing. Each line is subtracted over
    freq/450 = 0.089 Hz there, giving an expected loss near 11 * 0.089 / 13 = 7.5%. The
    floor sits at twice that expected loss: enough headroom for a transient that lands
    less favourably, and still failing anything that loses a sixth of its energy.
    """
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
            # The maximum against a blind control, not the median against a constant.
            # Gating the median let half a run's targets stand above the threshold -- the
            # 90-run manifest passed every gate carrying nineteen residuals over 1 dB and a
            # worst of +13.90 dB -- while a constant bound cannot survive widening the
            # search to where a displaced line actually sits.
            "lines_suppressed": metrics["residual_excess_db"] <= self.max_residual_excess_db,
            "suppression_sufficient": metrics["median_suppression_db"]
            >= self.min_median_suppression_db,
            "sinusoids_preserved": metrics["max_probe_deviation_db"] <= self.max_probe_deviation_db,
            "spectrum_preserved": metrics["max_nonline_change_db"] <= self.max_nonline_change_db,
            "transient_preserved": (
                self.min_intrinsic_energy_ratio
                <= metrics["intrinsic_energy_ratio"]
                <= self.max_intrinsic_energy_ratio
            ),
            # Kept because it can still catch a genuinely non-linear failure -- a filter
            # length that makes the removal state-dependent, say -- but on a linear
            # operator it is an invariant, not a test. test_removal_gates.py pins why.
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
    min_harmonics: int = MIN_HARMONICS_FOR_FIT,
    max_residual_rms_hz: float = MAX_FIT_RESIDUAL_RMS_HZ,
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
    if len(harmonics) < min_harmonics:
        raise ValueError(
            f"Only {len(harmonics)} comb harmonics exceeded {min_prominence_db} dB, below "
            f"the {min_harmonics} required; refusing to fit a fundamental that would then "
            "authorise removing the whole grid."
        )

    index = np.asarray(harmonics, dtype=float)
    position_array = np.asarray(positions, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    fundamental = float(
        np.sum(weight_array * index * position_array) / np.sum(weight_array * index**2)
    )
    residual = position_array - index * fundamental
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    if residual_rms > max_residual_rms_hz:
        raise ValueError(
            f"Fitted harmonics scatter {residual_rms:.3f} Hz RMS about their grid, above "
            f"the {max_residual_rms_hz} Hz bound; these peaks do not describe one comb and "
            "the fit must not authorise a removal grid."
        )

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
    #
    # Nominals closer together than the search half-width have overlapping windows, and
    # 57.2247 and 57.3485 are 0.124 Hz apart. On the recordings the first is about 17 dB
    # the stronger, so its skirt is the tallest thing in the second's window too, and a
    # plain largest-peak search hands one line to both nominals. Widths cannot separate
    # them: the strong line drifts up to 147 mHz across the cohort, so its window has to
    # stay wide enough to follow it.
    #
    # Each line is therefore claimed once. Nominals are resolved strongest first, and a
    # later one skips the neighbourhood of a line already taken, which leaves it looking
    # at the spectrum its own line would occupy.
    claims = []
    for order, nominal in enumerate(isolated_nominal_hz):
        found = _peak_near(frequency_array, spectrum, prominence_array, nominal, isolated_search_hz)
        strength = found[1] if found is not None else float("-inf")
        claims.append((strength, order, nominal))

    isolated = [float("nan")] * len(isolated_nominal_hz)
    isolated_prominence = [float("nan")] * len(isolated_nominal_hz)
    taken: list[float] = []
    for _, order, nominal in sorted(claims, key=lambda item: -item[0]):
        found = _peak_near(
            frequency_array,
            spectrum,
            prominence_array,
            nominal,
            isolated_search_hz,
            excluded_hz=taken,
        )
        if found is None:
            continue
        position, strength = found
        # The prominence floor that admits a comb harmonic to the fit applies here too,
        # and for a sharper reason. The isolated list is a cohort-level seed, and these
        # lines are carried by some participants and not others, so "absent" is an
        # ordinary outcome rather than a fault. The search returns the largest bin in its
        # window whatever is in it, so without the floor a participant who lacks a line
        # contributes its noise maximum as a removal target -- and the removal then digs a
        # notch into clean spectrum. NaN keeps that position out of `removal_frequencies`.
        if not np.isfinite(strength) or strength < min_prominence_db:
            continue
        isolated[order] = position
        isolated_prominence[order] = strength
        taken.append(position)

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
    excluded_hz: Sequence[float] = (),
) -> tuple[float, float] | None:
    """Refined position and prominence of the largest peak within a search window.

    ``excluded_hz`` names lines another nominal has already claimed. Their neighbourhoods
    are masked out, so an overlapping window looks past a peak that is already spoken for
    rather than reporting it a second time.
    """
    low, high = np.searchsorted(freqs, [target_hz - search_hz, target_hz + search_hz])
    if high <= low:
        return None
    window = np.array(prominence[low:high], dtype=float)
    for claimed in excluded_hz:
        window[np.abs(freqs[low:high] - claimed) <= _LINE_CLAIM_HZ] = np.nan
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


#: Widest a peak may be, measured 3 dB down from its own summit, to count as a line.
#: The diagnosis measured real lines at a 0.109 Hz half-power width; alpha and beta rhythms
#: are whole hertz wide. At equal height the two differ by a factor of about eighteen here,
#: so this threshold does not need to be delicate -- it needs to exist.
LINE_WIDTH_CEILING_HZ = 0.25

#: How much clear space a detected line needs from the comb positions the comb pass already
#: removes. Inside this distance a peak cannot be told apart from a harmonic's sideband, and
#: targeting it would take the same spectrum twice.
COMB_CLEARANCE_HZ = 0.20

#: Clear space required from any tone passed in ``probe_hz``. Nothing is protected by
#: default, and that default is the point.
#:
#: This was 0.35 Hz around the benchmark's five probe tones, on the reasoning that
#: detecting them would manufacture the proof that signal survives. The reasoning was
#: wrong: ``benchmark_run`` chooses targets from the raw recording and injects the probe
#: afterwards, so those tones are not in the spectrum detection sees. The exclusion
#: protected nothing and left five permanent 0.7 Hz blind spots in delivered data --
#: five 30 dB lines placed on 35.55, 40, 44.05, 65.35 and 78.45 Hz were all rejected.
#:
#: ``check_probe_clearance`` is the guard that actually matters and it stays: if a real
#: line ever does sit on a probe tone it raises, which says move the probe rather than
#: stop looking.
PROBE_CLEARANCE_HZ = 0.35

#: How far a peak must stand above the comb harmonic beside it to count as its own line
#: rather than that harmonic's sideband.
#:
#: The physics does the work: a sideband is weaker than the carrier it modulates, and a
#: peak sitting on a harmonic has no excess over itself. Measured on sub-0001, whose peak
#: 0.139 Hz from harmonic 78 stands 17.1 dB above it -- far too strong to be its sideband,
#: and declining it left that participant worse after cleaning than before. Against the
#: cases this must still reject: sub-0011's peak 0.001 Hz from harmonic 17 has no excess,
#: and the +/-0.11 Hz sideband population sits below its carriers by construction.
CARRIER_MARGIN_DB = 6.0

#: Most lines one run may contribute. A cap bounds how much spectrum removal can claim
#: however noisy a recording is; the cohort has needed at most twelve.
MAX_ISOLATED_LINES = 16

#: Prominence a peak needs to be treated as a line, calibrated on the fifteen uncleaned
#: run-1 recordings rather than chosen.
#:
#: Detections below this are a noise population, not lines: of 889 peaks clearing 6 dB,
#: three quarters sat at or below 8.3 dB and the median was 7.4 dB, and lowering the
#: threshold to 6 dB raised the count from 7.2 to 59.3 peaks per participant without
#: adding a single frequency that recurs across the cohort. At 10 dB what survives
#: clusters on the known lines and nothing else: 23.776 Hz in 10 participants with 0.022 Hz
#: of scatter, 29.684 in 8, 47.046 in 13, 57.234 in 15, 58.193 in 14, 94.091 in 14. Noise
#: does not land on the same frequency in fourteen people.
LINE_PROMINENCE_FLOOR_DB = 10.0


def removed_isolated_lines(
    manifest_path,
    *,
    fallback: Sequence[float] = (),
    merge_hz: float = 0.30,
) -> tuple[float, ...]:
    """The isolated lines the removal actually acted on, read from its manifest.

    An audit that keeps its own copy of the line list drifts from the removal, and two of
    them had: one still masked 61.0353 Hz, dropped for sitting 0.128 Hz from comb harmonic
    51, while masking nothing near 94 Hz. With detection the copy cannot be kept correct at
    all, because the lines are resolved per session and no static list names them.

    Positions of one line across recordings are collapsed, since the manifest records each
    session's own refined position and those differ by design -- the 94 Hz line spans
    0.595 Hz across this cohort. ``merge_hz`` is therefore wider than the per-run claim
    width: the question here is which line a position belongs to, not whether two nearby
    lines are distinct.

    ``fallback`` is returned when there is no manifest to read, so an audit still works
    before any apply has run.
    """
    import pandas as pd

    path = Path(manifest_path)
    if not path.exists():
        return tuple(float(f) for f in fallback)
    frame = pd.read_csv(path, sep="\t")
    if "isolated_hz" not in frame.columns:
        return tuple(float(f) for f in fallback)

    positions: list[float] = []
    for cell in frame["isolated_hz"]:
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            continue
        for piece in str(cell).split(";"):
            piece = piece.strip()
            if not piece:
                continue
            try:
                value = float(piece)
            except ValueError:
                continue
            if np.isfinite(value):
                positions.append(value)

    merged: list[float] = []
    for value in sorted(positions):
        if merged and value - merged[-1] <= merge_hz:
            continue
        merged.append(value)
    return tuple(merged)


def detect_isolated_lines(
    freqs: Sequence[float],
    spectrum_db: Sequence[float],
    prominence: Sequence[float],
    *,
    fundamental_hz: float,
    harmonic_range: tuple[int, int],
    min_prominence_db: float = LINE_PROMINENCE_FLOOR_DB,
    low_hz: float = 20.0,
    high_hz: float = 100.0,
    comb_clearance_hz: float = COMB_CLEARANCE_HZ,
    carrier_margin_db: float = CARRIER_MARGIN_DB,
    probe_clearance_hz: float = PROBE_CLEARANCE_HZ,
    probe_hz: Sequence[float] | None = None,  # nothing protected unless asked
    max_line_width_hz: float = LINE_WIDTH_CEILING_HZ,
    claim_hz: float = _LINE_CLAIM_HZ,
    max_lines: int = MAX_ISOLATED_LINES,
) -> tuple[float, ...]:
    """Find this run's isolated lines in its own spectrum, without a cohort list.

    A listed nominal is wrong for somebody by construction. Measured across the cohort
    these lines scatter 0.19 Hz to 0.595 Hz while one seed window reaches 0.30 Hz, so the
    94 Hz line was caught in 11 of 15 participants and left standing at +20 to +28 dB in
    the rest. Reading each run's own spectrum removes the class of error rather than adding
    seeds until it is covered.

    The detector is deliberately conservative, because its failure mode is removing signal:

    * peaks within ``comb_clearance_hz`` of a removed comb harmonic are left alone, since
      inside that distance a line and a harmonic's sideband are indistinguishable and the
      comb pass takes that spectrum anyway;
    * peaks within ``probe_clearance_hz`` of a benchmark probe tone are left alone;
    * peaks broader than ``max_line_width_hz``, measured 3 dB down, are left alone, which
      is what keeps a tall alpha or beta rhythm from being removed as an artifact;
    * ``max_lines`` bounds the total, spent strongest-first.

    Returns the accepted positions in ascending order.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum_db, dtype=float)
    prominence_array = np.asarray(prominence, dtype=float)
    if not frequency_array.shape == spectrum.shape == prominence_array.shape:
        raise ValueError("freqs, spectrum_db and prominence must have the same shape.")
    if not np.isfinite(fundamental_hz) or fundamental_hz <= 0:
        raise ValueError("fundamental_hz must be a finite positive number.")
    if low_hz >= high_hz:
        raise ValueError("low_hz must be below high_hz.")
    if max_lines < 0:
        raise ValueError("max_lines must not be negative.")
    for name, value in (
        ("comb_clearance_hz", comb_clearance_hz),
        ("probe_clearance_hz", probe_clearance_hz),
        ("max_line_width_hz", max_line_width_hz),
        ("claim_hz", claim_hz),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number.")

    protected = np.asarray(list(probe_hz or ()), dtype=float)

    # Clearance is owed to every comb position in the scanned range, not only to the
    # harmonics the comb pass removes. A peak 1 mHz from harmonic 17 is the comb whether or
    # not the removal range reaches harmonic 17; if it ought to be removed, that range is
    # what should say so. Harmonic 11 at 13.23 Hz is excluded from removal deliberately,
    # for landing where real rhythms live, and offering it here as an "isolated" line would
    # route around that decision instead of revisiting it.
    low_harmonic, high_harmonic = harmonic_range
    first = max(1, int(np.floor((low_hz - comb_clearance_hz) / fundamental_hz)))
    last = int(np.ceil((high_hz + comb_clearance_hz) / fundamental_hz))
    comb_positions = (
        np.arange(min(first, low_harmonic), max(last, high_harmonic) + 1, dtype=float)
        * fundamental_hz
    )

    # Only a summit can be a line. Testing every bin instead lets a rejected peak reappear
    # at the edge of its own exclusion: the skirt of a strong comb-adjacent peak still
    # clears the threshold a clearance-width away, and would be taken as a separate line
    # sitting exactly where the clearance ends.
    summit = np.zeros(prominence_array.shape, dtype=bool)
    summit[1:-1] = (prominence_array[1:-1] > prominence_array[:-2]) & (
        prominence_array[1:-1] >= prominence_array[2:]
    )

    inside = (frequency_array >= low_hz) & (frequency_array <= high_hz)
    candidate = (
        summit & inside & np.isfinite(prominence_array) & (prominence_array >= min_prominence_db)
    )
    if comb_positions.size:
        distance = np.abs(frequency_array[:, None] - comb_positions[None, :])
        near_comb = distance.min(axis=1) <= comb_clearance_hz
        nearest = distance.argmin(axis=1)
        harmonic_strength = np.array(
            [
                prominence_array[int(np.argmin(np.abs(frequency_array - position)))]
                for position in comb_positions
            ]
        )
        # Proximity alone does not make a peak part of the comb. A sideband cannot exceed
        # its own carrier, and a peak sitting on a harmonic has no excess over itself, so
        # a peak that clears the harmonic beside it by this margin is a line that happens
        # to land nearby. sub-0001 carries one 0.139 Hz from harmonic 78 and 17.1 dB above
        # it; declining that as a sideband left the participant worse after cleaning than
        # before, its neighbours removed and it not.
        #
        # The comparison only means anything beyond one line width. Closer than that, the
        # strength sampled at the comb position is the candidate's own skirt, so it
        # outranks itself: sub-0001 has a peak 0.064 Hz from harmonic 39 that would be
        # handed back as an isolated line and targeted twice. Separations here are cleanly
        # bimodal -- 0.002 and 0.064 Hz for peaks that are the comb, 0.147 Hz for the line
        # that is not.
        outranks = prominence_array - harmonic_strength[nearest] >= carrier_margin_db
        resolvable = distance.min(axis=1) > claim_hz
        candidate &= ~(near_comb & ~(resolvable & outranks))
    if protected.size:
        near_probe = (
            np.abs(frequency_array[:, None] - protected[None, :]).min(axis=1)
            <= probe_clearance_hz
        )
        candidate &= ~near_probe

    indices = np.flatnonzero(candidate)
    if indices.size == 0:
        return ()

    # Strongest first, frequency breaking ties, so the result does not depend on how the
    # spectrum happened to be ordered.
    order = sorted(indices, key=lambda i: (-prominence_array[i], frequency_array[i]))

    accepted: list[float] = []
    for index in order:
        position = float(frequency_array[index])
        if any(abs(position - taken) <= claim_hz for taken in accepted):
            continue
        if _peak_width_hz(frequency_array, prominence_array, index) > max_line_width_hz:
            continue
        accepted.append(position)
        if len(accepted) >= max_lines:
            break

    return tuple(sorted(accepted))


def _peak_width_hz(
    frequency_array: np.ndarray,
    prominence_array: np.ndarray,
    index: int,
    drop_db: float = 3.0,
) -> float:
    """Width of the peak at ``index``, measured ``drop_db`` below its own summit.

    A sinusoid is as narrow as the spectral resolution allows; a rhythm is not. Walking
    outward from the summit rather than fitting a shape keeps this honest on the asymmetric
    peaks that sit on a rhythm's shoulder.
    """
    floor = prominence_array[index] - drop_db
    left = index
    while left > 0 and prominence_array[left - 1] >= floor:
        left -= 1
    right = index
    last = prominence_array.size - 1
    while right < last and prominence_array[right + 1] >= floor:
        right += 1
    return float(frequency_array[right] - frequency_array[left])


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
    widths: Sequence[float] | None = None,
    search_hz: float = RESIDUAL_SEARCH_HZ,
) -> dict[str, float]:
    """How far the targeted lines fell, in local prominence.

    The residual is the worst bin left anywhere in the window the removal claimed, not the
    value at the target's centre. Reading the centre alone misses a line that moved: a
    target at 50 Hz whose centre falls to -10 dB while a residual at 50.05 Hz still stands
    at +15 dB was reported as -10 dB. That is precisely the failure this removal keeps
    producing -- taking a line out exposes or displaces its neighbour -- so the centre is
    the one place the evidence will not be.

    ``widths`` are the per-target notch widths; without them the search falls back to the
    centre bin and the old blind spot returns, so callers that have widths should pass them.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    before = np.asarray(prominence_before, dtype=float)
    after = np.asarray(prominence_after, dtype=float)
    target_array = np.asarray(list(targets), dtype=float)
    width_array = (
        np.asarray(list(widths), dtype=float)
        if widths is not None
        else np.zeros(target_array.size)
    )
    if width_array.size != target_array.size:
        raise ValueError("targets and widths must have the same length.")

    rows = []
    for frequency, width in zip(target_array, width_array):
        centre = int(np.argmin(np.abs(frequency_array - frequency)))
        if not (np.isfinite(before[centre]) and np.isfinite(after[centre])):
            continue
        reach = max(max(width, 0.0) / 2.0, search_hz)
        inside = np.abs(frequency_array - frequency) <= reach
        inside[centre] = True
        window_after = after[inside]
        finite = window_after[np.isfinite(window_after)]
        rows.append((before[centre], float(np.max(finite)) if finite.size else after[centre]))
    if not rows:
        raise ValueError("No target frequency had a usable prominence estimate.")
    # A blind control for the same search. Windows of the same width, as many of them,
    # placed where no target is: the largest background bin they hold is the floor this
    # search has by construction. Without it a fixed threshold cannot tell a surviving line
    # from the maximum of a thousand noise bins, which is what a +/-0.15 Hz window around
    # sixty-odd targets amounts to.
    control = _control_maximum(frequency_array, after, target_array, search_hz)

    values = np.asarray(rows)
    max_residual = float(np.max(values[:, 1]))
    return {
        "n_targets": float(len(values)),
        "median_prominence_before_db": float(np.median(values[:, 0])),
        "median_residual_prominence_db": float(np.median(values[:, 1])),
        "max_residual_prominence_db": max_residual,
        "control_max_prominence_db": control,
        "residual_excess_db": max_residual - control,
        "median_suppression_db": float(np.median(values[:, 0] - values[:, 1])),
    }


def _control_maximum(
    frequency_array: np.ndarray,
    after: np.ndarray,
    targets: np.ndarray,
    search_hz: float,
) -> float:
    """Largest prominence in as many target-free windows as there are targets."""
    if targets.size == 0:
        return float("-inf")
    away = np.ones(frequency_array.size, dtype=bool)
    for frequency in targets:
        away &= np.abs(frequency_array - frequency) > 2.0 * search_hz
    away &= np.isfinite(after)
    candidates = np.flatnonzero(away)
    if candidates.size == 0:
        return float("-inf")

    step = max(1, candidates.size // max(int(targets.size), 1))
    maxima = [
        float(np.max(after[candidates[start : start + step]]))
        for start in range(0, candidates.size - step + 1, step)
    ]
    return float(np.max(maxima)) if maxima else float(np.max(after[candidates]))


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
