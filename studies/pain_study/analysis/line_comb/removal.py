"""Estimate and remove the room's line comb from continuous EEG.

The contamination this targets is characterised in ``docs/scanner_harmonic_diagnosis.md``:
a comb at integer multiples of about 1.2 Hz, mains-synchronous and independent of the
imaging gradients, plus isolated narrow lines that drift independently and can be
intermittent. Because the sources are monochromatic, the right removal is a projection
onto sinusoids at automatically measured frequencies -- not a broad notch, which would
take the surrounding band with it.

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
from functools import lru_cache
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
#: 22 to 83 -- sixty-two targets from three peaks. Across all whole-run and adaptive
#: estimates in the current 90 recordings, the weakest accepted 54-second window has
#: twenty mutually consistent harmonics. Requiring those twenty prevents a sparse chance
#: grid from authorising a broad removal while retaining that independently validated
#: window; the residual-RMS and uncertainty checks still apply separately.
MIN_HARMONICS_FOR_FIT = 20

#: Largest individual deviation permitted from the fitted arithmetic grid.
MAX_HARMONIC_RESIDUAL_HZ = 0.06
"""Largest residual a peak may have and still support the arithmetic comb model.

The cohort diagnosis measured a 6.6 mHz RMS residual and a 26 mHz maximum on the
well-resolved comb.  A 54-second adaptive spectrum has 18.5 mHz bins, so 60 mHz leaves
more than one bin of localization headroom while excluding nearby independent lines.
"""

MAX_FIT_RESIDUAL_RMS_HZ = 0.04
"""Largest RMS scatter permitted about the fitted arithmetic grid.

A comb is an arithmetic series; peaks that do not lie on one are not a comb, however many
there are. After robust membership fitting the 90-run maximum RMS residual is 0.0341 Hz,
so 0.04 Hz separates every observed fit from an inconsistent grid.
"""

#: How far either side of a target a residual is still that target's responsibility.
#:
#: The notch's own width is the wrong region to search. The failure being looked for is a
#: target that missed, and a missed line then sits just outside what the notch claimed --
#: precisely where the claimed width cannot see. This is the frequency-uncertainty scale of
#: the estimate instead: comb harmonics wander with the fundamental times the harmonic
#: index, and the isolated lines are searched over the same 0.15 Hz. Kept well inside the
#: 0.6 Hz half-spacing so one target is never charged with the next harmonic's line.
RESIDUAL_SEARCH_HZ = 0.15


@lru_cache(maxsize=32)
def _thomson_tapers(
    n_times: int,
    sampling_frequency_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """Return the immutable DPSS basis shared by equal-length detection windows."""
    import warnings

    from scipy.signal.windows import dpss

    half_time_bandwidth = bandwidth_hz * n_times / (2.0 * sampling_frequency_hz)
    n_tapers = int(2.0 * half_time_bandwidth)
    if n_tapers < 2:
        raise ValueError("The time-bandwidth product supplies fewer than two tapers.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*matmul", category=RuntimeWarning)
        return dpss(
            n_times,
            half_time_bandwidth,
            Kmax=n_tapers,
            sym=False,
            norm=2,
        )


def thomson_f_statistics(
    data: np.ndarray,
    *,
    sampling_frequency_hz: float,
    bandwidth_hz: float,
    family_alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Thomson multitaper sinusoid test used by CleanLine and MNE spectrum-fit.

    The Bonferroni family is the complete time-series frequency grid, matching MNE's
    automatic ``spectrum_fit`` detector. Statistics remain channel-specific so a focal
    electrical line never authorises subtraction from channels where it is absent.

    Returns the frequency grid, the statistic, the Bonferroni critical value that
    detection compares against, and the uncorrected probabilities. The probabilities come
    from here rather than from a caller because this is where the taper count, and so the
    denominator degrees of freedom, is known.
    """
    from scipy.stats import f as f_distribution

    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("Thomson F statistics require channel-by-time data.")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be finite and positive.")
    if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be finite and positive.")
    if not np.isfinite(family_alpha) or not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie strictly between zero and one.")

    n_times = values.shape[1]
    tapers = _thomson_tapers(n_times, sampling_frequency_hz, bandwidth_hz)
    n_tapers = tapers.shape[0]

    odd_tapers = np.arange(0, n_tapers, 2)
    even_tapers = np.arange(1, n_tapers, 2)
    taper_sums = np.sum(tapers[odd_tapers], axis=1)
    taper_sum_squares = float(np.sum(taper_sums**2))
    if not np.isfinite(taper_sum_squares) or taper_sum_squares <= 0.0:
        raise ValueError("The multitaper sinusoid basis is degenerate.")

    frequencies = np.fft.rfftfreq(n_times, 1.0 / sampling_frequency_hz)
    statistic = np.empty((values.shape[0], frequencies.size), dtype=float)
    for channel_index, channel in enumerate(values):
        channel = channel - np.mean(channel)
        spectra = np.fft.rfft(tapers * channel, axis=-1)
        spectra[:, 0] /= np.sqrt(2.0)
        if n_times % 2 == 0:
            spectra[:, -1] /= np.sqrt(2.0)
        coefficient = (
            np.sum(
                spectra[odd_tapers] * taper_sums[:, np.newaxis],
                axis=0,
            )
            / taper_sum_squares
        )
        fitted = coefficient[np.newaxis, :] * taper_sums[:, np.newaxis]
        numerator = (n_tapers - 1) * np.abs(coefficient) ** 2 * taper_sum_squares
        denominator = np.sum(np.abs(spectra[odd_tapers] - fitted) ** 2, axis=0)
        denominator += np.sum(np.abs(spectra[even_tapers]) ** 2, axis=0)
        denominator[denominator == 0.0] = np.inf
        statistic[channel_index] = numerator / denominator

    threshold = float(
        f_distribution.ppf(
            1.0 - family_alpha / n_times,
            2,
            2 * n_tapers - 2,
        )
    )
    return frequencies, statistic, threshold, thomson_f_p_values(statistic, n_tapers=n_tapers)


def thomson_f_p_values(statistic: np.ndarray, *, n_tapers: int) -> np.ndarray:
    """Uncorrected right-tail probabilities of the Thomson F statistic."""
    from scipy.stats import f as f_distribution

    values = np.asarray(statistic, dtype=float)
    if n_tapers < 2:
        raise ValueError("The F statistic needs at least two tapers.")
    return np.asarray(f_distribution.sf(values, 2, 2 * n_tapers - 2), dtype=float)


def benjamini_hochberg_discoveries(
    p_values: Sequence[float],
    *,
    false_discovery_rate: float = 0.05,
) -> int:
    """Hypotheses the Benjamini-Hochberg step-up procedure rejects."""
    values = np.sort(np.asarray(p_values, dtype=float))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty vector.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must be finite probabilities.")
    if not np.isfinite(false_discovery_rate) or not 0.0 < false_discovery_rate < 1.0:
        raise ValueError("false_discovery_rate must lie strictly between zero and one.")
    ranks = np.arange(1, values.size + 1)
    below = np.flatnonzero(values <= false_discovery_rate * ranks / values.size)
    return int(below[-1] + 1) if below.size else 0


def run_residual_sinusoid_p_value(p_values: Sequence[float]) -> float:
    """One recording's evidence that any sinusoid survived, over everything it searched.

    The smallest probability in the family, corrected for the size of that family. One
    number per recording, because the cohort decision has to be made across recordings
    rather than inside them -- see ``residual_sinusoid_verdict``.
    """
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty vector.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must be finite probabilities.")
    return float(min(1.0, float(np.min(values)) * values.size))


def residual_sinusoid_verdict(
    run_p_values: Sequence[float],
    *,
    false_discovery_rate: float = 0.05,
) -> dict[str, float | bool]:
    """Cohort decision on surviving sinusoids, made over recordings rather than inside them.

    Deliberately not a per-run gate, for the same reason the seam criterion is not one.
    Requiring zero significant residuals within every recording rejects a clean cohort at
    the test's own error rate: a 5% family-wise rate over ninety recordings puts the
    expected number of false failures near four or five, which is what was measured before
    this replaced it. Each recording contributes one corrected probability instead, and
    Benjamini-Hochberg over those decides whether any recording is genuinely unclean.
    """
    values = np.asarray(run_p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one recording's residual probability is required.")
    discoveries = benjamini_hochberg_discoveries(
        values,
        false_discovery_rate=false_discovery_rate,
    )
    return {
        "n_runs": float(values.size),
        "n_discoveries": float(discoveries),
        "min_run_p_value": float(np.min(values)),
        "false_discovery_rate": float(false_discovery_rate),
        "passed": discoveries == 0,
    }


@dataclass(frozen=True)
class ResidualDetection:
    """What licenses removing a line that survived the first pass.

    Deliberately separate from ``PreservationGate``. A detector parameterised by the
    acceptance tolerance removes precisely what the gate would flag, which leaves a gate
    that can only fail where the search's own subtraction fell short -- never because a
    line was missed. It stops being a test of the removal and becomes the search's
    stopping rule. Nothing here may be derived from an acceptance threshold, and
    test_removal.py fails if the two ever share a field.

    The criterion is Thomson's multitaper F test, the statistic behind MNE's automatic
    ``spectrum_fit`` detection and CleanLine, measured on what the first pass produced --
    the raw data with the already-modelled component accounted for, which is how a line
    hidden under a stronger neighbour's skirt becomes visible. A residual carrying power
    without being a resolvable sinusoid is therefore left in place for the gate to
    report, rather than subtracted because it was inconvenient.
    """

    family_alpha: float = 0.05
    """Family-wise error rate over one channel's complete frequency search.

    The family is the whole frequency grid, matching the Bonferroni correction
    ``thomson_f_statistics`` applies and MNE's own detector.
    """

    min_shared_channel_fraction: float = 0.5
    """Share of channels carrying the sinusoid before it is fitted across the array.

    A routing rule rather than an evidence threshold: every channel counted here cleared
    the F test on its own. Below it the line is subtracted only from the channels that
    evidence it. At or above it the channels are fitted jointly, which conditions one
    estimate on the array instead of on each electrode's noise, and in exchange subtracts
    from the minority that did not evidence it -- a cost the preservation gates measure.
    """

    def __post_init__(self) -> None:
        if not np.isfinite(self.family_alpha) or not 0.0 < self.family_alpha < 1.0:
            raise ValueError("family_alpha must lie strictly between zero and one.")
        if (
            not np.isfinite(self.min_shared_channel_fraction)
            or not 0.0 < self.min_shared_channel_fraction <= 1.0
        ):
            raise ValueError("min_shared_channel_fraction must lie in (0, 1].")


def focal_residual_line_candidates(
    freqs: Sequence[float],
    statistic: np.ndarray,
    *,
    threshold: float,
    targets_hz: Sequence[float],
    widths_hz: Sequence[float],
    responsibility_hz: float,
) -> tuple[tuple[float, ...], ...]:
    """Significant channel-specific sinusoids inside authorised artifact regions."""
    frequency_array = np.asarray(freqs, dtype=float)
    values = np.asarray(statistic, dtype=float)
    if values.ndim != 2 or values.shape[1] != frequency_array.size:
        raise ValueError("statistic must have channel and frequency axes.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive.")
    authorised = authorised_residual_bins(
        frequency_array,
        targets_hz,
        widths_hz,
        responsibility_hz,
    )

    results = []
    for channel_statistic in values:
        indices = np.flatnonzero(
            authorised & np.isfinite(channel_statistic) & (channel_statistic > threshold)
        )
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        candidates = []
        for group in groups:
            if group.size:
                index = int(group[np.argmax(channel_statistic[group])])
                candidates.append(float(frequency_array[index]))
        results.append(tuple(candidates))
    return tuple(results)


def shared_residual_line_candidates(
    freqs: Sequence[float],
    statistic: np.ndarray,
    *,
    threshold: float,
    targets_hz: Sequence[float],
    widths_hz: Sequence[float],
    responsibility_hz: float,
    min_channel_fraction: float,
) -> tuple[float, ...]:
    """Sinusoids enough of the array evidences to fit jointly rather than per channel.

    Agreement across channels is what distinguishes an array-wide electrical line from a
    channel-local one; it is not what makes either of them real. Every channel counted
    here already cleared the F test on its own.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    values = np.asarray(statistic, dtype=float)
    if values.ndim != 2 or values.shape[1] != frequency_array.size:
        raise ValueError("statistic must have channel and frequency axes.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive.")
    if not np.isfinite(min_channel_fraction) or not 0.0 < min_channel_fraction <= 1.0:
        raise ValueError("min_channel_fraction must lie in (0, 1].")
    authorised = authorised_residual_bins(
        frequency_array,
        targets_hz,
        widths_hz,
        responsibility_hz,
    )

    significant = np.isfinite(values) & (values > threshold)
    share = significant.mean(axis=0)
    strength = np.median(np.where(significant, values, 0.0), axis=0)
    indices = np.flatnonzero(authorised & (share >= min_channel_fraction))
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)

    candidates = []
    for group in groups:
        if not group.size:
            continue
        # Widest agreement wins the group, the strongest median statistic breaks a tie.
        chosen = int(group[np.lexsort((strength[group], share[group]))[-1]])
        candidates.append(float(frequency_array[chosen]))
    return tuple(candidates)


def authorised_residual_bins(
    frequency_array: np.ndarray,
    targets_hz: Sequence[float],
    widths_hz: Sequence[float],
    responsibility_hz: float,
) -> np.ndarray:
    """Bins a residual search may reach: each target's own width, or its uncertainty."""
    targets = np.asarray(targets_hz, dtype=float)
    widths = np.asarray(widths_hz, dtype=float)
    if targets.shape != widths.shape or targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets_hz and widths_hz must be matching non-empty vectors.")
    if not np.isfinite(responsibility_hz) or responsibility_hz <= 0.0:
        raise ValueError("responsibility_hz must be finite and positive.")
    reaches = np.maximum(widths / 2.0, responsibility_hz)
    return np.any(
        np.abs(frequency_array[:, np.newaxis] - targets[np.newaxis, :]) <= reaches[np.newaxis, :],
        axis=1,
    )


@dataclass(frozen=True)
class CombEstimate:
    """One run's measured line frequencies."""

    fundamental_hz: float
    harmonics_used: tuple[int, ...]
    harmonic_positions_hz: tuple[float, ...]
    residual_rms_hz: float
    max_abs_residual_hz: float
    fundamental_jackknife_se_hz: float
    isolated_hz: tuple[float, ...]
    isolated_prominence_db: tuple[float, ...]

    @property
    def n_harmonics(self) -> int:
        return len(self.harmonics_used)

    def __post_init__(self) -> None:
        if len(self.harmonics_used) != len(self.harmonic_positions_hz):
            raise ValueError("Each supported harmonic must retain one measured position.")


@dataclass(frozen=True)
class AdaptiveCombModel:
    """A run represented by independently supported overlapping-window estimates."""

    whole_estimate: CombEstimate
    window_estimates: tuple[CombEstimate, ...]
    window_fundamental_hz: tuple[float, ...]
    fundamental_range_hz: float
    max_adjacent_shift_hz: float


@dataclass(frozen=True)
class Probe:
    """Signals injected before removal that must survive it.

    The sinusoids stand for narrowband neural activity at frequencies the removal is not
    aimed at; the burst stands for a broadband transient. The burst deliberately sits near
    the comb, because a transient's bandwidth necessarily overlaps neighbouring lines and
    that overlap is the realistic worst case.
    """

    sinusoid_hz: tuple[float, ...] = (35.40, 43.80, 65.40, 78.60)
    """Injected tones, placed midway between comb harmonics: ``(k + 0.5) * 1.2``.

    The positions are chosen rather than arbitrary, because the previous set was arbitrary
    and it broke. 44.05 Hz sat 0.350 Hz from harmonic 37 on the nominal grid -- clearing
    the 0.3 Hz requirement by 49 mHz, which is fine for one static fundamental and not fine
    for a model that fits one per window. Harmonic k moves by k times the fundamental's
    wander, so harmonic 37 travels about 67 mHz within a run and ate that margin: the
    90-recording benchmark aborted at recording 24 with harmonic 37 at 44.3479 Hz, 0.298 Hz
    away. 78.45 Hz was next in line, clearing by only 76 mHz measured across the cohort.

    A midpoint is the unique position that maximises the distance to both neighbours, at
    0.6 Hz. Measured against every fitted plan in the cohort the four tones clear by
    0.507-0.558 Hz, against 0.298-0.488 Hz for the set they replace.
    """
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
    max_focal_residual_excess_db: float = 1.0
    """Worst channel-window residual above its matched multiple-search control."""
    max_boundary_discontinuity_ratio: float = 1.0
    """Largest seam jump relative to the 95th percentile of matched maximum jumps."""
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
    max_band_fraction_removed: float = 0.18
    """Total opportunity-cost ceiling after every evidenced expansion.

    The adaptive model fits 17 or more overlapping windows. A two-standard-error interval
    covers each fitted target's position uncertainty, while the independent residual gate
    catches any line that nevertheless falls outside it. Isolated targets start at their
    physical line width and expand only with observed support or residual evidence. The earlier 90 continuous plans
    reached 17.017% of the analysis band. The 18% ceiling is applied only to the total
    transform: a separate cap on the base-width component had no scientific meaning, while
    this total still rejects MNE's 25% default and catches a pathological removal that
    empties a large fraction of the band. Exact-window and channel-local transforms are now
    included directly in each benchmark measurement.
    """

    @staticmethod
    def _within_band_budget(value: float, limit: float, bin_size: float) -> bool:
        """Compare a discrete Fourier-bin count with a continuous fraction limit."""
        if not np.all(np.isfinite((value, limit, bin_size))) or bin_size <= 0.0:
            raise ValueError("Band fractions, limits, and bin sizes must be finite and positive.")
        return value <= limit + bin_size / 2.0

    def evaluate(self, metrics: dict[str, float]) -> dict[str, bool]:
        return {
            # The maximum against a blind control, not the median against a constant.
            # Gating the median let half a run's targets stand above the threshold -- the
            # 90-run manifest passed every gate carrying nineteen residuals over 1 dB and a
            # worst of +13.90 dB -- while a constant bound cannot survive widening the
            # search to where a displaced line actually sits.
            "lines_suppressed": metrics["residual_excess_db"] <= self.max_residual_excess_db,
            "no_focal_residual": metrics["focal_residual_excess_db"]
            <= self.max_focal_residual_excess_db,
            "study_lines_suppressed": metrics["study_residual_excess_db"]
            <= self.max_residual_excess_db,
            "study_no_focal_residual": metrics["study_focal_residual_excess_db"]
            <= self.max_focal_residual_excess_db,
            # The seam criterion is deliberately absent from the per-run gate. It compares
            # against the second largest of 40 matched controls, so under the null it fails
            # 2/41 of runs by construction and an all-90-must-pass rule rejects a perfect
            # cohort about 99% of the time. It is decided over the cohort instead, by
            # seam_randomization_verdict. max_boundary_discontinuity_ratio is still measured and
            # reported per run, and still feeds that decision.
            "sinusoids_preserved": metrics["max_probe_deviation_db"] <= self.max_probe_deviation_db,
            "spectrum_preserved": metrics["max_nonline_change_db"] <= self.max_nonline_change_db,
            "study_sinusoids_preserved": metrics["study_max_probe_deviation_db"]
            <= self.max_probe_deviation_db,
            "study_spectrum_preserved": metrics["study_max_nonline_change_db"]
            <= self.max_nonline_change_db,
            "transient_preserved": (
                self.min_intrinsic_energy_ratio
                <= metrics["intrinsic_energy_ratio"]
                <= self.max_intrinsic_energy_ratio
            ),
            # Kept because it can still catch a genuinely non-linear failure -- a filter
            # length that makes the removal state-dependent, say -- but on a linear
            # operator it is an invariant, not a test. test_removal_gates.py pins why.
            "transient_undistorted": metrics["burst_correlation"] >= self.min_burst_correlation,
            "band_mostly_untouched": self._within_band_budget(
                metrics["removed_band_fraction"],
                self.max_band_fraction_removed,
                metrics["band_fraction_bin_size"],
            ),
        }

    def passed(self, metrics: dict[str, float]) -> bool:
        return all(self.evaluate(metrics).values())


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median with deterministic ordering and strictly positive weights."""
    if values.ndim != 1 or values.shape != weights.shape or values.size == 0:
        raise ValueError("values and weights must be matching non-empty vectors.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("values and weights must be finite.")
    if np.any(weights <= 0.0):
        raise ValueError("weights must be strictly positive.")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left"))
    return float(values[order[index]])


def _fit_consistent_harmonics(
    harmonics: np.ndarray,
    positions_hz: np.ndarray,
    weights: np.ndarray,
    *,
    min_harmonics: int,
    max_harmonic_residual_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Robustly fit the slope through the mutually consistent harmonic candidates."""
    seed = _weighted_median(positions_hz / harmonics, weights)
    keep = np.abs(positions_hz - harmonics * seed) <= max_harmonic_residual_hz
    visited: set[bytes] = set()
    while True:
        membership = keep.tobytes()
        if membership in visited:
            raise RuntimeError("Robust comb membership entered a cycle.")
        visited.add(membership)
        if np.count_nonzero(keep) < min_harmonics:
            raise ValueError(
                f"Only {np.count_nonzero(keep)} mutually consistent comb harmonics remain; "
                "the candidate peaks scatter across incompatible grids."
            )
        selected_harmonics = harmonics[keep]
        selected_positions = positions_hz[keep]
        selected_weights = weights[keep]
        fundamental = float(
            np.sum(selected_weights * selected_harmonics * selected_positions)
            / np.sum(selected_weights * selected_harmonics**2)
        )
        updated = np.abs(positions_hz - harmonics * fundamental) <= max_harmonic_residual_hz
        if np.array_equal(updated, keep):
            return selected_harmonics, selected_positions, selected_weights, fundamental
        keep = updated


def estimate_comb(
    freqs: Sequence[float],
    spectrum_db: Sequence[float],
    prominence: Sequence[float],
    *,
    nominal_hz: float = NOMINAL_FUNDAMENTAL_HZ,
    harmonic_range: tuple[int, int] = COMB_HARMONIC_RANGE,
    isolated_nominal_hz: Sequence[float] = (),
    search_hz: float = 0.25,
    isolated_search_hz: float = 0.15,
    min_prominence_db: float = 1.0,
    min_harmonics: int = MIN_HARMONICS_FOR_FIT,
    max_harmonic_residual_hz: float = MAX_HARMONIC_RESIDUAL_HZ,
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

    index, position_array, weight_array, fundamental = _fit_consistent_harmonics(
        np.asarray(harmonics, dtype=float),
        np.asarray(positions, dtype=float),
        np.asarray(weights, dtype=float),
        min_harmonics=min_harmonics,
        max_harmonic_residual_hz=max_harmonic_residual_hz,
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
        # The replicated session catalogue authorises the frequency. This window only
        # confirms that the source is present; letting the same window move the target
        # would make the transform and its residual audit select the same peak.
        isolated[order] = float(nominal)
        isolated_prominence[order] = strength
        taken.append(position)

    return CombEstimate(
        fundamental_hz=fundamental,
        harmonics_used=tuple(int(harmonic) for harmonic in index),
        harmonic_positions_hz=tuple(float(position) for position in position_array),
        residual_rms_hz=float(np.sqrt(np.mean(residual**2))),
        max_abs_residual_hz=float(np.max(np.abs(residual))),
        fundamental_jackknife_se_hz=_fundamental_jackknife_se(index, position_array, weight_array),
        isolated_hz=tuple(isolated),
        isolated_prominence_db=tuple(isolated_prominence),
    )


def _fundamental_jackknife_se(
    harmonics: np.ndarray,
    positions_hz: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Delete-one-harmonic standard error of the fitted fundamental."""
    count = harmonics.size
    if count < 3:
        raise ValueError("At least three harmonics are required for jackknife uncertainty.")
    estimates = np.empty(count, dtype=float)
    for omitted in range(count):
        keep = np.arange(count) != omitted
        numerator = np.sum(weights[keep] * harmonics[keep] * positions_hz[keep])
        denominator = np.sum(weights[keep] * harmonics[keep] ** 2)
        estimates[omitted] = numerator / denominator
    centre = float(np.mean(estimates))
    return float(np.sqrt((count - 1) / count * np.sum((estimates - centre) ** 2)))


def build_adaptive_comb_model(
    whole_estimate: CombEstimate,
    window_estimates: Sequence[CombEstimate],
) -> AdaptiveCombModel:
    """Validate that every adaptive window independently supports its removal grid."""
    estimates = tuple(window_estimates)
    if len(estimates) < 2:
        raise ValueError("At least two overlapping adaptive windows are required.")
    for index, estimate in enumerate(estimates):
        if estimate.n_harmonics < MIN_HARMONICS_FOR_FIT:
            raise ValueError(
                f"Adaptive window {index} has only {estimate.n_harmonics} supported "
                f"harmonics; at least {MIN_HARMONICS_FOR_FIT} are required."
            )
        if not (
            np.isfinite(estimate.fundamental_hz)
            and np.isfinite(estimate.fundamental_jackknife_se_hz)
            and estimate.fundamental_jackknife_se_hz > 0.0
        ):
            raise ValueError(f"Adaptive window {index} has an invalid fundamental or uncertainty.")

    frequencies = np.asarray(
        [estimate.fundamental_hz for estimate in estimates],
        dtype=float,
    )
    return AdaptiveCombModel(
        whole_estimate=whole_estimate,
        window_estimates=estimates,
        window_fundamental_hz=tuple(float(value) for value in frequencies),
        fundamental_range_hz=float(np.ptp(frequencies)),
        max_adjacent_shift_hz=float(np.max(np.abs(np.diff(frequencies)))),
    )


def uncertainty_aware_notch_widths(
    estimate: CombEstimate,
    targets: Sequence[float],
    *,
    ratio: float,
    minimum_hz: float,
    confidence_z: float,
    isolated_minimum_hz: float,
) -> np.ndarray:
    """Widths covering comb uncertainty and the audited isolated-line neighborhood."""
    if not np.isfinite(confidence_z) or confidence_z <= 0:
        raise ValueError("confidence_z must be a finite positive number.")
    if not np.isfinite(isolated_minimum_hz) or isolated_minimum_hz <= 0.0:
        raise ValueError("isolated_minimum_hz must be a finite positive number.")
    target_array = np.asarray(targets, dtype=float)
    widths = notch_widths_for(target_array, ratio=ratio, minimum_hz=minimum_hz)
    fundamental = estimate.fundamental_hz
    harmonic = np.rint(target_array / fundamental).astype(int)
    measured = dict(zip(estimate.harmonics_used, estimate.harmonic_positions_hz))
    comb_position = np.asarray(
        [measured.get(int(index), int(index) * fundamental) for index in harmonic],
        dtype=float,
    )
    on_comb = np.isclose(target_array, comb_position, rtol=0.0, atol=1e-8)
    half_uncertainty = confidence_z * harmonic * estimate.fundamental_jackknife_se_hz
    comb_widths = widths + 2.0 * half_uncertainty
    isolated_widths = np.maximum(widths, isolated_minimum_hz)
    return np.where(on_comb, comb_widths, isolated_widths)


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


#: Widest a peak may be, measured 3 dB down from its own summit, to count as a line.
#: The diagnosis measured real lines at a 0.109 Hz half-power width; alpha and beta rhythms
#: are whole hertz wide. At equal height the two differ by a factor of about eighteen here,
#: so this threshold does not need to be delicate -- it needs to exist.
LINE_WIDTH_CEILING_HZ = 0.25

#: How far a peak may deviate from the arithmetic grid and still belong to the comb.
#:
#: This must match the robust membership tolerance. A wider detector exclusion delegated
#: resolved peaks to a comb model that explicitly rejected them, leaving the 27.519 Hz line
#: in sub-0011 untouched beside harmonic 23.
COMB_CLEARANCE_HZ = MAX_HARMONIC_RESIDUAL_HZ

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
    subject: str | None = None,
    merge_hz: float = 0.02,
) -> tuple[float, ...]:
    """The isolated lines the removal actually acted on, read from its manifest.

    An audit that keeps its own copy of the line list drifts from the removal, and two of
    them had: one still masked 61.0353 Hz, dropped for sitting 0.128 Hz from comb harmonic
    51, while masking nothing near 94 Hz. With detection the copy cannot be kept correct at
    all, because the lines are resolved per session and no static list names them.

    Pass ``subject`` when scoring participant data. A cohort-wide union would mask clean
    frequencies merely because a different participant carried an artifact there.
    Near-identical positions are collapsed only within the 54-second spectral resolution;
    real between-participant drift remains represented when a cohort union is requested.

    Missing or malformed provenance is an error. Substituting a historical frequency list
    would make an audit score frequencies that the participant-specific transform did not
    necessarily remove.
    """
    import pandas as pd

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Line-removal manifest not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    if "isolated_hz" not in frame.columns:
        raise ValueError(f"Line-removal manifest has no isolated_hz column: {path}")
    if not np.isfinite(merge_hz) or merge_hz <= 0.0:
        raise ValueError("merge_hz must be finite and positive.")
    if subject is not None:
        if "recording" not in frame.columns:
            raise ValueError(f"Line-removal manifest has no recording column: {path}")
        frame = frame[frame["recording"].astype(str).str.startswith(f"{subject}_")]
        if frame.empty:
            raise ValueError(f"Line-removal manifest has no rows for {subject}: {path}")

    positions: list[float] = []
    for cell in frame["isolated_hz"]:
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            continue
        for piece in str(cell).split(";"):
            piece = piece.strip()
            if not piece:
                continue
            value = float(piece)
            if not np.isfinite(value):
                raise ValueError(f"Non-finite isolated frequency in {path}: {piece!r}")
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
    probe_clearance_hz: float = PROBE_CLEARANCE_HZ,
    probe_hz: Sequence[float] | None = None,  # nothing protected unless asked
    max_line_width_hz: float = LINE_WIDTH_CEILING_HZ,
    claim_hz: float = _LINE_CLAIM_HZ,
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
        candidate &= distance.min(axis=1) > comb_clearance_hz
    if protected.size:
        near_probe = (
            np.abs(frequency_array[:, None] - protected[None, :]).min(axis=1) <= probe_clearance_hz
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
    left_hz, right_hz = _peak_support_bounds_hz(
        frequency_array,
        prominence_array,
        index,
        drop_db=drop_db,
    )
    return right_hz - left_hz


def _peak_support_bounds_hz(
    frequency_array: np.ndarray,
    prominence_array: np.ndarray,
    index: int,
    drop_db: float = 3.0,
) -> tuple[float, float]:
    """Frequency-bin centres spanning a summit down to its requested drop."""
    if frequency_array.shape != prominence_array.shape or frequency_array.ndim != 1:
        raise ValueError("Peak-support arrays must be matching one-dimensional vectors.")
    if not 0 <= index < frequency_array.size:
        raise IndexError("Peak-support index lies outside the spectrum.")
    if not np.isfinite(drop_db) or drop_db <= 0.0:
        raise ValueError("drop_db must be finite and positive.")

    floor = prominence_array[index] - drop_db
    left = index
    while left > 0 and prominence_array[left - 1] >= floor:
        left -= 1
    right = index
    last = prominence_array.size - 1
    while right < last and prominence_array[right + 1] >= floor:
        right += 1
    return float(frequency_array[left]), float(frequency_array[right])


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
    measured = dict(zip(estimate.harmonics_used, estimate.harmonic_positions_hz))
    candidates = [
        measured.get(harmonic, estimate.fundamental_hz * harmonic)
        for harmonic in range(low, high + 1)
    ]
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


@dataclass(frozen=True)
class BoundaryDiscontinuityEvidence:
    """Observed adaptive-boundary jump and its 40 synchronized blind controls."""

    observed_max: float
    control_maxima: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray((self.observed_max, *self.control_maxima), dtype=float)
        if len(self.control_maxima) != 40:
            raise ValueError("Exactly 40 boundary controls are required.")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Boundary discontinuity evidence must be finite and non-negative.")

    @property
    def ratio(self) -> float:
        scale = float(np.quantile(self.control_maxima, 0.95, method="higher"))
        epsilon = np.finfo(float).eps * max(1.0, self.observed_max, *self.control_maxima)
        return self.observed_max / max(scale, epsilon)


def seam_randomization_verdict(
    evidence: Sequence[BoundaryDiscontinuityEvidence],
    *,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Exact synchronized-shift tests for widespread and single-run seam defects."""
    rows = tuple(evidence)
    if not rows:
        raise ValueError("At least one recording of boundary evidence is required.")
    values = np.asarray(
        [(row.observed_max, *row.control_maxima) for row in rows],
        dtype=float,
    )
    maxima = np.empty(values.shape[1], dtype=float)
    counts = np.empty(values.shape[1], dtype=int)
    for candidate_index in range(values.shape[1]):
        references = np.delete(values, candidate_index, axis=1)
        scales = np.quantile(references, 0.95, axis=1, method="higher")
        epsilon = np.finfo(float).eps * np.maximum(1.0, np.max(values, axis=1))
        ratios = values[:, candidate_index] / np.maximum(scales, epsilon)
        maxima[candidate_index] = float(np.max(ratios))
        counts[candidate_index] = int(np.count_nonzero(ratios > 1.0))

    max_p_value = float(np.mean(maxima >= maxima[0]))
    count_p_value = float(np.mean(counts >= counts[0]))
    endpoint_alpha = alpha / 2.0
    return {
        "n_runs": float(values.shape[0]),
        "n_exceeding": float(counts[0]),
        "max_ratio": float(maxima[0]),
        "max_p_value": max_p_value,
        "count_p_value": count_p_value,
        "passed": bool(max_p_value >= endpoint_alpha and count_p_value >= endpoint_alpha),
    }


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
    values, null_maxima, _, _ = _suppression_components(
        np.asarray(freqs, dtype=float),
        np.asarray(prominence_before, dtype=float),
        np.asarray(prominence_after, dtype=float),
        np.asarray(targets, dtype=float),
        np.zeros(len(targets), dtype=float) if widths is None else np.asarray(widths, dtype=float),
        search_hz,
    )
    return _summarize_suppression(values, null_maxima)


def adaptive_line_suppression(
    freqs: Sequence[float],
    prominence_before: np.ndarray,
    prominence_after: np.ndarray,
    targets: Sequence[Sequence[float]],
    widths: Sequence[Sequence[float]],
    search_hz: float = RESIDUAL_SEARCH_HZ,
) -> dict[str, float]:
    """Suppression across all adaptive windows with one matched multiple-search null."""
    frequency_array = np.asarray(freqs, dtype=float)
    before = np.asarray(prominence_before, dtype=float)
    after = np.asarray(prominence_after, dtype=float)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError(
            "Adaptive prominence arrays must be matching window-by-frequency matrices."
        )
    if before.shape[0] != len(targets) or len(targets) != len(widths):
        raise ValueError("Every adaptive spectrum needs one target and width sequence.")

    row_groups = []
    null_groups = []
    target_metadata = []
    residual_positions = []
    for window_index, (
        before_window,
        after_window,
        window_targets,
        window_widths,
    ) in enumerate(
        zip(
            before,
            after,
            targets,
            widths,
        )
    ):
        values, null_maxima, usable_targets, window_residual_positions = _suppression_components(
            frequency_array,
            before_window,
            after_window,
            np.asarray(window_targets, dtype=float),
            np.asarray(window_widths, dtype=float),
            search_hz,
        )
        row_groups.append(values)
        null_groups.append(null_maxima)
        target_metadata.extend((window_index, float(target)) for target in usable_targets)
        residual_positions.extend(window_residual_positions)
    combined_null = np.max(np.stack(null_groups), axis=0)
    combined_values = np.concatenate(row_groups)
    result = _summarize_suppression(combined_values, combined_null)
    worst_index = int(np.argmax(combined_values[:, 1]))
    worst_window, worst_target = target_metadata[worst_index]
    result.update(
        {
            "worst_residual_window": float(worst_window),
            "worst_residual_target_hz": worst_target,
            "worst_residual_frequency_hz": float(residual_positions[worst_index]),
            "worst_residual_before_db": float(combined_values[worst_index, 0]),
        }
    )
    return result


def _suppression_components(
    frequency_array: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    targets: np.ndarray,
    widths: np.ndarray,
    search_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if before.shape != frequency_array.shape or after.shape != frequency_array.shape:
        raise ValueError("Prominence arrays must match the frequency grid.")
    if widths.shape != targets.shape:
        raise ValueError("targets and widths must have the same shape.")
    narrow_candidates = _narrow_peak_mask(frequency_array, before) | _narrow_peak_mask(
        frequency_array, after
    )
    rows = []
    reaches = []
    usable_targets = []
    residual_positions = []
    for frequency, width in zip(targets, widths):
        centre = int(np.argmin(np.abs(frequency_array - frequency)))
        if not (np.isfinite(before[centre]) and np.isfinite(after[centre])):
            continue
        reach = max(max(float(width), 0.0) / 2.0, search_hz)
        inside = np.abs(frequency_array - frequency) <= reach
        inside[centre] = True
        finite_indices = np.flatnonzero(inside & np.isfinite(after) & narrow_candidates)
        if finite_indices.size:
            residual_index = int(finite_indices[np.argmax(after[finite_indices])])
            residual_value = float(after[residual_index])
        else:
            residual_index = centre
            residual_value = 0.0
        rows.append((before[centre], residual_value))
        reaches.append(reach)
        usable_targets.append(frequency)
        residual_positions.append(float(frequency_array[residual_index]))
    if not rows:
        raise ValueError("No target frequency had a usable prominence estimate.")
    null_maxima = _matched_null_maxima(
        frequency_array,
        after,
        np.asarray(usable_targets, dtype=float),
        np.asarray(reaches, dtype=float),
        eligible=narrow_candidates,
    )
    return (
        np.asarray(rows),
        null_maxima,
        np.asarray(usable_targets, dtype=float),
        np.asarray(residual_positions, dtype=float),
    )


def _narrow_peak_mask(
    frequency_array: np.ndarray,
    prominence: np.ndarray,
    *,
    max_line_width_hz: float = LINE_WIDTH_CEILING_HZ,
) -> np.ndarray:
    """Bins at summits whose 3 dB width is consistent with a monochromatic line."""
    if prominence.shape != frequency_array.shape:
        raise ValueError("prominence must match the frequency grid.")
    summit = np.zeros(prominence.shape, dtype=bool)
    summit[1:-1] = (
        np.isfinite(prominence[1:-1])
        & (prominence[1:-1] > prominence[:-2])
        & (prominence[1:-1] >= prominence[2:])
    )
    accepted = np.zeros(prominence.shape, dtype=bool)
    for index in np.flatnonzero(summit):
        accepted[index] = (
            _peak_width_hz(frequency_array, prominence, int(index)) <= max_line_width_hz
        )
    return accepted


def _summarize_suppression(values: np.ndarray, null_maxima: np.ndarray) -> dict[str, float]:
    null_max_95 = float(np.quantile(null_maxima, 0.95, method="higher"))
    max_residual = float(np.max(values[:, 1]))
    return {
        "n_targets": float(len(values)),
        "median_prominence_before_db": float(np.median(values[:, 0])),
        "median_residual_prominence_db": float(np.median(values[:, 1])),
        "max_residual_prominence_db": max_residual,
        "null_max_95_db": null_max_95,
        "residual_excess_db": max_residual - null_max_95,
        "median_suppression_db": float(np.median(values[:, 0] - values[:, 1])),
    }


def spatiotemporal_target_prominence(
    freqs: Sequence[float],
    background_spectrum_db: np.ndarray,
    peak_spectrum_db: np.ndarray,
    targets: Sequence[float],
    widths: Sequence[float],
    *,
    background_half_width_hz: float,
    search_hz: float = RESIDUAL_SEARCH_HZ,
) -> np.ndarray:
    """Target prominence against an immutable pre-clean spectral background.

    The peak is measured after cleaning, while its local floor is measured before
    cleaning. Recomputing both from the cleaned spectrum lets nearby notches lower the
    floor and manufacture an apparent residual that was not present in absolute power.
    """
    frequency_array = np.asarray(freqs, dtype=float)
    background_spectra = np.asarray(background_spectrum_db, dtype=float)
    peak_spectra = np.asarray(peak_spectrum_db, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    width_array = np.asarray(widths, dtype=float)
    if background_spectra.shape != peak_spectra.shape:
        raise ValueError("Background and peak spectra must have the same shape.")
    if background_spectra.shape[-1] != frequency_array.size:
        raise ValueError("The final spectrum axis must match freqs.")
    if target_array.shape != width_array.shape:
        raise ValueError("targets and widths must have the same shape.")
    if background_half_width_hz <= search_hz:
        raise ValueError("The background window must be wider than the target search.")

    values = []
    for target, width in zip(target_array, width_array):
        reach = max(max(float(width), 0.0) / 2.0, search_hz)
        distance = np.abs(frequency_array - target)
        inside = distance <= reach
        background = (distance > reach) & (distance <= background_half_width_hz)
        if not np.any(inside) or np.count_nonzero(background) < 32:
            raise ValueError(f"Insufficient spectrum around target {target:.6g} Hz.")
        local_floor = np.median(background_spectra[..., background], axis=-1)
        local_peak = np.max(peak_spectra[..., inside], axis=-1)
        values.append(local_peak - local_floor)
    if not values:
        raise ValueError("At least one target is required.")
    return np.stack(values, axis=-1)


def adaptive_spatiotemporal_suppression(
    freqs: Sequence[float],
    background_spectrum_db: np.ndarray,
    peak_spectrum_db: np.ndarray,
    targets: Sequence[Sequence[float]],
    widths: Sequence[Sequence[float]],
    *,
    background_half_width_hz: float,
    search_hz: float = RESIDUAL_SEARCH_HZ,
) -> dict[str, float]:
    """Focal residual evidence against a matched channel-window search control."""
    frequency_array = np.asarray(freqs, dtype=float)
    background_spectra = np.asarray(background_spectrum_db, dtype=float)
    peak_spectra = np.asarray(peak_spectrum_db, dtype=float)
    if background_spectra.shape != peak_spectra.shape:
        raise ValueError("Background and peak spectra must have the same shape.")
    if background_spectra.ndim != 3 or background_spectra.shape[-1] != frequency_array.size:
        raise ValueError("Adaptive spectra must have channel, window, and frequency axes.")
    if background_spectra.shape[1] != len(targets) or len(targets) != len(widths):
        raise ValueError("Every adaptive spectrum needs one target and width sequence.")

    target_groups = []
    target_metadata = []
    null_groups: list[list[float]] | None = None
    for window_index, (window_targets, window_widths) in enumerate(zip(targets, widths)):
        target_array = np.asarray(window_targets, dtype=float)
        width_array = np.asarray(window_widths, dtype=float)
        reaches = np.maximum(np.maximum(width_array, 0.0) / 2.0, search_hz)
        background_window = background_spectra[:, window_index, :]
        peak_window = peak_spectra[:, window_index, :]
        target_values = spatiotemporal_target_prominence(
            frequency_array,
            background_window,
            peak_window,
            target_array,
            width_array,
            background_half_width_hz=background_half_width_hz,
            search_hz=search_hz,
        )
        target_groups.append(target_values.ravel())
        for channel_index in range(peak_window.shape[0]):
            for target, reach in zip(target_array, reaches):
                indices = np.flatnonzero(np.abs(frequency_array - target) <= reach)
                peak_index = int(indices[np.argmax(peak_window[channel_index, indices])])
                target_metadata.append(
                    (
                        window_index,
                        channel_index,
                        float(target),
                        float(frequency_array[peak_index]),
                    )
                )
        placements = _matched_null_centres(
            frequency_array,
            np.all(np.isfinite(background_window) & np.isfinite(peak_window), axis=0),
            target_array,
            reaches,
            edge_margin_hz=background_half_width_hz,
        )
        if null_groups is None:
            null_groups = [[] for _ in placements]
        if len(placements) != len(null_groups):
            raise ValueError("Adaptive windows produced inconsistent matched-null counts.")
        for placement_index, control_targets in enumerate(placements):
            control = spatiotemporal_target_prominence(
                frequency_array,
                background_window,
                peak_window,
                control_targets,
                width_array,
                background_half_width_hz=background_half_width_hz,
                search_hz=search_hz,
            )
            null_groups[placement_index].append(float(np.max(control)))

    if null_groups is None:
        raise ValueError("At least one adaptive window is required.")
    target_values = np.concatenate(target_groups)
    null_maxima = np.asarray([max(group) for group in null_groups], dtype=float)
    null_max_95 = float(np.quantile(null_maxima, 0.95, method="higher"))
    maximum = float(np.max(target_values))
    worst_index = int(np.argmax(target_values))
    worst_window, worst_channel, worst_target, worst_frequency = target_metadata[worst_index]
    return {
        "max_channel_block_residual_prominence_db": maximum,
        "p99_channel_block_residual_prominence_db": float(np.quantile(target_values, 0.99)),
        "focal_null_max_95_db": null_max_95,
        "focal_residual_excess_db": maximum - null_max_95,
        "worst_focal_window": float(worst_window),
        "worst_focal_channel_index": float(worst_channel),
        "worst_focal_target_hz": worst_target,
        "worst_focal_frequency_hz": worst_frequency,
    }


def _matched_null_maxima(
    frequency_array: np.ndarray,
    after: np.ndarray,
    targets: np.ndarray,
    reaches: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Maxima from repeated target-free searches matched to all target widths.

    Every null placement contains one window with the same reach as every target window.
    This preserves the multiple-comparisons burden exactly instead of comparing the target
    maximum with a maximum over an unrelated number of background bins.
    """
    if targets.size == 0:
        raise ValueError("A matched null requires at least one target.")
    if reaches.shape != targets.shape or np.any(reaches <= 0.0):
        raise ValueError("reaches must be positive and match targets.")

    finite = np.isfinite(after)
    eligible_mask = finite if eligible is None else np.asarray(eligible, dtype=bool)
    if eligible_mask.shape != after.shape:
        raise ValueError("eligible must match the scored spectrum.")
    placements = _matched_null_centres(
        frequency_array,
        finite,
        targets,
        reaches,
    )
    maxima = []
    for centres in placements:
        windows = [
            np.abs(frequency_array - centre) <= reach for centre, reach in zip(centres, reaches)
        ]
        searched = np.logical_or.reduce(windows)
        candidates = searched & finite & eligible_mask
        maxima.append(float(np.max(after[candidates])) if np.any(candidates) else 0.0)
    return np.asarray(maxima, dtype=float)


def _matched_null_centres(
    frequency_array: np.ndarray,
    finite: np.ndarray,
    targets: np.ndarray,
    reaches: np.ndarray,
    *,
    edge_margin_hz: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Complete target-free placements preserving every target search width."""
    if finite.shape != frequency_array.shape:
        raise ValueError("finite must match the frequency grid.")
    if targets.shape != reaches.shape or targets.ndim != 1 or targets.size == 0:
        raise ValueError("targets and reaches must be matching non-empty vectors.")
    if np.any(reaches <= 0.0) or not np.all(np.isfinite(reaches)):
        raise ValueError("reaches must be finite and positive.")
    if not np.isfinite(edge_margin_hz) or edge_margin_hz < 0.0:
        raise ValueError("edge_margin_hz must be finite and non-negative.")
    candidate_pools = []
    for reach in reaches:
        margin = max(float(reach), edge_margin_hz)
        inside_edges = frequency_array >= frequency_array[0] + margin
        inside_edges &= frequency_array <= frequency_array[-1] - margin
        candidate_pools.append(np.flatnonzero(finite & inside_edges))

    phases = np.linspace(0.31, 0.59, 20)
    placements = []
    for phase in phases:
        for direction in (-1.0, 1.0):
            selected: list[tuple[float, float]] = []
            centres = np.empty(targets.size, dtype=float)
            # Place the broadest windows first because they have the fewest valid centres.
            for index in np.argsort(-reaches):
                reach = float(reaches[index])
                preferred = float(targets[index] + direction * phase)
                centre_index = _nearest_matched_null_index(
                    frequency_array,
                    candidate_pools[index],
                    preferred,
                    targets,
                    reaches,
                    reach,
                    selected,
                )
                if centre_index is None:
                    selected = []
                    break
                centre = float(frequency_array[centre_index])
                selected.append((centre, reach))
                centres[index] = centre
            if len(selected) == targets.size:
                placements.append(centres)
    expected = 2 * len(phases)
    if len(placements) != expected:
        raise ValueError(
            f"Could not construct all {expected} complete target-free matched-null searches."
        )
    return tuple(placements)


def _nearest_matched_null_index(
    frequency_array: np.ndarray,
    candidate_indices: np.ndarray,
    preferred: float,
    targets: np.ndarray,
    target_reaches: np.ndarray,
    reach: float,
    selected: Sequence[tuple[float, float]],
) -> int | None:
    """Nearest valid grid point without materializing a full mask per target."""
    candidate_frequencies = frequency_array[candidate_indices]
    right = int(np.searchsorted(candidate_frequencies, preferred))
    left = right - 1
    while left >= 0 or right < candidate_indices.size:
        left_distance = abs(float(candidate_frequencies[left]) - preferred) if left >= 0 else np.inf
        right_distance = (
            abs(float(candidate_frequencies[right]) - preferred)
            if right < candidate_indices.size
            else np.inf
        )
        if left_distance <= right_distance:
            candidate_position = left
            left -= 1
        else:
            candidate_position = right
            right += 1

        candidate = float(candidate_frequencies[candidate_position])
        if np.any(np.abs(candidate - targets) <= reach + target_reaches):
            continue
        if any(
            abs(candidate - centre) <= reach + selected_reach for centre, selected_reach in selected
        ):
            continue
        return int(candidate_indices[candidate_position])
    return None


def probe_preservation(
    freqs: Sequence[float],
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    probe: Probe,
) -> dict[str, float]:
    """Worst channel-by-frequency power change at the injected sinusoids."""
    frequency_array = np.asarray(freqs, dtype=float)
    before = np.atleast_2d(np.asarray(psd_before, dtype=float))
    after = np.atleast_2d(np.asarray(psd_after, dtype=float))
    if before.shape != after.shape or before.shape[-1] != frequency_array.size:
        raise ValueError("Probe PSD arrays must match each other and the frequency grid.")
    deviations = []
    for frequency in probe.sinusoid_hz:
        index = int(np.argmin(np.abs(frequency_array - frequency)))
        ratios = after[:, index] / before[:, index]
        deviations.extend(10.0 * np.log10(np.maximum(ratios, np.finfo(float).tiny)))
    return {
        "max_probe_deviation_db": float(np.max(np.abs(deviations))),
        "min_probe_ratio": float(np.min(10 ** (np.asarray(deviations) / 10.0))),
    }


def sinusoid_waveform(
    times: Sequence[float],
    frequencies_hz: Sequence[float],
    amplitude_v: float,
) -> np.ndarray:
    """Equal-amplitude tones, each given its own phase so they do not sum coherently."""
    time_array = np.asarray(times, dtype=float)
    if not np.isfinite(amplitude_v) or amplitude_v <= 0.0:
        raise ValueError("amplitude_v must be finite and positive.")
    signal = np.zeros_like(time_array)
    for frequency in frequencies_hz:
        signal += amplitude_v * np.sin(2 * np.pi * frequency * time_array + frequency)
    return signal


def in_band_probe_frequencies(
    targets_hz: Sequence[float],
    *,
    count: int = 4,
) -> tuple[float, ...]:
    """Probe positions taken from the plan's own targets, spread across the removed set.

    Every other probe in this benchmark sits where nothing is removed, so it measures the
    removal away from its own targets and cannot report a loss. This one sits on the
    targets and measures the opposite quantity: how much of a narrowband signal that
    coincides with an artifact does not survive. Signal exactly at an artifact frequency is
    not separable from the artifact, so this is a reported cost and never a pass or fail.

    Positions come from the fitted plan rather than a frequency list, so the measurement
    means the same thing at a site whose lines sit somewhere else entirely.
    """
    unique = np.unique(np.asarray(targets_hz, dtype=float))
    if unique.size == 0:
        raise ValueError("At least one target is required to place an in-band probe.")
    if not np.all(np.isfinite(unique)):
        raise ValueError("targets_hz must be finite.")
    if count < 1:
        raise ValueError("count must be positive.")
    if unique.size <= count:
        return tuple(float(value) for value in unique)
    positions = np.linspace(0, unique.size - 1, count)
    return tuple(float(unique[int(round(position))]) for position in positions)


def in_band_probe_survival(
    freqs: Sequence[float],
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    frequencies_hz: Sequence[float],
) -> dict[str, float]:
    """Fraction of each in-band probe tone's power still present after removal."""
    frequency_array = np.asarray(freqs, dtype=float)
    before = np.atleast_2d(np.asarray(psd_before, dtype=float))
    after = np.atleast_2d(np.asarray(psd_after, dtype=float))
    if before.shape != after.shape or before.shape[-1] != frequency_array.size:
        raise ValueError("Probe PSD arrays must match each other and the frequency grid.")
    if not len(tuple(frequencies_hz)):
        raise ValueError("At least one in-band probe frequency is required.")
    survivals = []
    for frequency in frequencies_hz:
        index = int(np.argmin(np.abs(frequency_array - frequency)))
        floor = np.maximum(before[:, index], np.finfo(float).tiny)
        survivals.extend(after[:, index] / floor)
    return {
        "min_in_band_probe_survival": float(np.min(survivals)),
        "median_in_band_probe_survival": float(np.median(survivals)),
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


def boundary_discontinuity_evidence(
    original: np.ndarray,
    cleaned: np.ndarray,
    boundaries: Sequence[int],
) -> BoundaryDiscontinuityEvidence:
    """Measure the seam maximum and retain every synchronized blind control."""
    original_array = np.atleast_2d(np.asarray(original, dtype=float))
    cleaned_array = np.atleast_2d(np.asarray(cleaned, dtype=float))
    if original_array.shape != cleaned_array.shape:
        raise ValueError("original and cleaned arrays must have the same shape.")
    boundary_indices = np.asarray(tuple(boundaries), dtype=int) - 1
    if boundary_indices.size == 0:
        raise ValueError("At least one interior adaptive boundary is required.")
    if np.any(boundary_indices < 0) or np.any(boundary_indices >= original_array.shape[-1] - 1):
        raise ValueError("Adaptive boundaries must lie inside the time axis.")

    correction_steps = np.abs(np.diff(cleaned_array - original_array, axis=-1))
    step_count = correction_steps.shape[-1]
    control_maxima = []
    observed_controls: set[tuple[int, ...]] = set()
    for fraction in np.linspace(0.07, 0.93, 160):
        offset = max(int(round(fraction * step_count)), 1)
        control_indices = tuple(sorted(((boundary_indices + offset) % step_count).tolist()))
        if (
            control_indices in observed_controls
            or np.intersect1d(
                control_indices,
                boundary_indices,
            ).size
        ):
            continue
        observed_controls.add(control_indices)
        control_maxima.append(float(np.max(correction_steps[:, control_indices])))
        if len(control_maxima) == 40:
            break
    if len(control_maxima) != 40:
        raise ValueError("Could not construct 40 matched adaptive-boundary controls.")

    boundary_jump = float(np.max(correction_steps[:, boundary_indices]))
    return BoundaryDiscontinuityEvidence(boundary_jump, tuple(control_maxima))


def boundary_discontinuity_ratio(
    original: np.ndarray,
    cleaned: np.ndarray,
    boundaries: Sequence[int],
) -> float:
    """Seam maximum relative to the run's 95th-percentile blind control."""
    return boundary_discontinuity_evidence(original, cleaned, boundaries).ratio


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
    if not (recovered_array.shape[-1] == reference_array.shape[-1] == time_array.size):
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
