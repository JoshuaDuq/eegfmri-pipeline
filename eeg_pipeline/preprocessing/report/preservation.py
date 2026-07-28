"""Evidence that brain signal survived preprocessing.

Every other panel in this report measures removal: variance excluded, amplitude
attenuated, epochs dropped, components rejected. None of them can distinguish a
recording that was cleaned from one that was emptied, because both score well on all of
them. Inside a scanner, where removing most of the sensor variance is normal and
expected, that asymmetry is the report's blind spot.

Two measurements are made, and which one applies depends on the paradigm.

For a task, split-half reliability of the evoked response. Odd and even trials are
averaged separately and correlated. The measurement needs no assumption about the
response's shape or timing: if a stimulus-locked response survived preprocessing, two
halves of the same trials must agree about it, and if preprocessing removed it, they
cannot.

For rest, the posterior alpha rhythm. It is the one EEG feature that is reliably present
in a healthy waking recording, spatially specific to posterior sensors, and spectrally
narrow enough to measure against its own neighbourhood.

Neither is graded. A weak result has many innocent explanations — a paradigm with no
early evoked response, an eyes-open recording, a genuinely low-alpha participant — and
the report's job is to put the number in front of someone who knows which applies.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.aperiodic import aperiodic_line_db, fit_aperiodic
from eeg_pipeline.preprocessing.report.spectra import (
    MICROVOLT_REFERENCE_DB,
    POWER_UNIT_LABEL,
)
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    PRIMARY_COLOR,
)
from eeg_pipeline.preprocessing.report.tables import Metric, metric_table

#: Band searched for the posterior alpha peak.
ALPHA_BAND_HZ = (7.0, 14.0)

#: Neighbourhood the alpha peak is measured against, so that a peak is scored by how far
#: it stands above the local background rather than by absolute power.
ALPHA_REFERENCE_BAND_HZ = (3.0, 25.0)

#: Regular expression matching the posterior sensors alpha is expected over.
POSTERIOR_PATTERN = r"^(P[0-9z]|PO[0-9z]|O[0-9z]|Oz|POz|Pz)"

#: Trials needed before a split-half average is an average rather than a few trials.
MINIMUM_TRIALS_FOR_SPLIT_HALF = 20

#: How often a spectrum with no rhythm may be credited with one.
#:
#: A method constant rather than a threshold on the data, and stated as a rate because the
#: quantity it governs is a *maximum*: the prominence is the largest excess over the fitted
#: background across every bin in the alpha band, so the question is not "is this bin
#: unusual" but "is the largest of seventy bins unusual". Those have very different
#: answers, and a criterion phrased in fixed multiples of the per-bin scatter answers the
#: first while being read as the second.
#:
#: A fixed multiple was used here previously and was calibrated for the first question. At
#: the resolution this pipeline computes spectra with, the alpha band holds of the order of
#: seventy bins, whose largest value exceeds two standard deviations about eighty per cent
#: of the time under pure noise. The test meant to keep fabricated peaks out of the cohort
#: histogram admitted them four times in five.
RESOLVABLE_PEAK_FALSE_POSITIVE_RATE = 0.05

#: Bins below which the extreme-value approximation is not used.
#:
#: The Gumbel limit needs ``log(log(n))`` and is poor for a handful of bins. Below this a
#: plain multiple is used, which is the right shape when there is barely a search to
#: correct for.
_MINIMUM_BINS_FOR_EXTREME_VALUE = 8
_SMALL_SEARCH_FACTOR = 2.0

#: How much larger the in-band excess runs than the residual it is scored against.
#:
#: The residual is measured where the aperiodic line was *fitted*; the excess is measured
#: inside the alpha window, which was excluded from that fit and where the line is
#: therefore extrapolated. Extrapolating a two-parameter fit across a gap costs accuracy in
#: the gap, so the excess carries a scale the fitted residual does not see, and a threshold
#: in plain multiples of the residual is calibrated against the wrong yardstick.
#:
#: Measured rather than assumed. On rhythm-free simulated recordings the ratio
#: ``prominence / (gumbel(n_bins) * residual)`` is stable across sampling rate, run length,
#: noise colour and bin count -- median 1.12, 95th percentile 1.57, 99th 1.74 over 200
#: recordings spanning 200-500 Hz, 120-500 s, white and 1/f. Set at the 95th percentile, so
#: the whole criterion carries the false-positive rate it names. For scale, a real
#: recording with an unambiguous 11 dB alpha peak sits at 4.57 on the same statistic.
_EXTRAPOLATION_INFLATION = 1.57


def resolvable_prominence_threshold(
    n_search_bins: int, *, false_positive_rate: float = RESOLVABLE_PEAK_FALSE_POSITIVE_RATE
) -> float:
    """Multiples of the background scatter the largest of ``n_search_bins`` must clear.

    The maximum of ``n`` roughly independent normal residuals converges to a Gumbel
    distribution with location ``b_n`` and scale ``1 / a_n``:

        a_n = sqrt(2 ln n)
        b_n = a_n - (ln ln n + ln 4pi) / (2 a_n)

    so the level exceeded with probability ``rate`` by pure noise is
    ``b_n - ln(-ln(1 - rate)) / a_n``. Requiring the observed prominence to clear that
    level makes the criterion scale with how wide a band was searched and how finely it was
    resolved, instead of being accidentally strict on a narrow band and vacuous on a wide
    one.

    The result is in multiples of the *aperiodic fit residual*, so it carries
    :data:`_EXTRAPOLATION_INFLATION` as well: the residual and the excess are measured on
    either side of the excluded window and are not the same scale.

    Neighbouring Welch bins are correlated, so the effective number of independent looks is
    somewhat below the bin count and the multiplicity part is mildly conservative.
    Conservative in the direction that matters: it withholds a peak frequency rather than
    inventing one.
    """
    if n_search_bins < _MINIMUM_BINS_FOR_EXTREME_VALUE:
        return _SMALL_SEARCH_FACTOR * _EXTRAPOLATION_INFLATION
    scale = math.sqrt(2.0 * math.log(n_search_bins))
    location = scale - (math.log(math.log(n_search_bins)) + math.log(4.0 * math.pi)) / (
        2.0 * scale
    )
    gumbel = location - math.log(-math.log(1.0 - false_positive_rate)) / scale
    return gumbel * _EXTRAPOLATION_INFLATION

#: Window the split halves are correlated over, in seconds from stimulus onset.
#:
#: The window matters more than it looks. This pipeline epochs from -7 to +15 s to give
#: time-frequency baselines room, so correlating across the whole epoch would average one
#: second of response into twenty-one seconds of baseline and return a near-zero
#: reliability for a perfectly good dataset. Restricting to the interval where an evoked
#: response actually lives is what makes the number mean anything.
#:
#: A paradigm whose response falls outside this window must set its own; the window used
#: is reported alongside the correlation so the figure cannot be read without it.
DEFAULT_RESPONSE_WINDOW_S = (0.0, 1.0)


@dataclass(frozen=True)
class SplitHalfReliability:
    """Agreement between two independent halves of the evoked response."""

    n_trials: int
    times_s: np.ndarray
    #: Global field power of the odd- and even-trial averages, in microvolts.
    #:
    #: Global field power, not the across-channel mean. These data are average
    #: referenced, so the mean across channels is zero by construction and plotting it
    #: shows floating-point cancellation noise rather than a response. Global field
    #: power is the spatial standard deviation, which the reference does not cancel and
    #: which peaks exactly where the two halves are being asked to agree.
    odd_gfp_uv: np.ndarray
    even_gfp_uv: np.ndarray
    #: Correlation between the two halves over channels and time.
    correlation: float
    #: Same quantity corrected to the full trial count by Spearman-Brown.
    corrected_correlation: float
    #: Interval the correlation was computed over. Reported because the number is
    #: meaningless without it.
    response_window_s: tuple[float, float]


def compute_split_half_reliability(
    epochs: mne.BaseEpochs,
    *,
    response_window_s: tuple[float, float] = DEFAULT_RESPONSE_WINDOW_S,
) -> SplitHalfReliability | None:
    """Correlate the evoked response of odd against even trials.

    Trials are split by alternating position rather than at the midpoint, so that slow
    drift in attention, impedance, or arousal falls equally on both halves. A midpoint
    split would confound reliability with whatever changed over the session.

    The correlation is taken over ``response_window_s`` rather than the whole epoch. See
    :data:`DEFAULT_RESPONSE_WINDOW_S` for why that choice dominates the result.

    Returns ``None`` when too few trials survive to average, or when the epoch does not
    overlap the requested window.
    """
    picked = epochs.copy().pick("eeg")
    if len(picked) < MINIMUM_TRIALS_FOR_SPLIT_HALF:
        return None

    low = max(float(response_window_s[0]), float(picked.times[0]))
    high = min(float(response_window_s[1]), float(picked.times[-1]))
    if high <= low:
        return None
    picked = picked.crop(tmin=low, tmax=high)

    odd = picked[1::2].average().get_data()
    even = picked[0::2].average().get_data()
    correlation = float(np.corrcoef(odd.ravel(), even.ravel())[0, 1])
    # Spearman-Brown steps the two half-length averages up to the reliability the full
    # trial count supports, which is the quantity the analysis actually runs on.
    denominator = 1.0 + correlation
    corrected = (2.0 * correlation / denominator) if denominator > 0 else 0.0
    return SplitHalfReliability(
        n_trials=len(picked),
        times_s=np.asarray(picked.times, dtype=float),
        odd_gfp_uv=odd.std(axis=0) * 1e6,
        even_gfp_uv=even.std(axis=0) * 1e6,
        correlation=correlation,
        corrected_correlation=float(corrected),
        response_window_s=(low, high),
    )


@dataclass(frozen=True)
class PosteriorAlpha:
    """Posterior alpha peak measured against its own spectral neighbourhood."""

    channel_names: tuple[str, ...]
    frequencies_hz: np.ndarray
    #: Posterior-channel mean spectrum, in decibels.
    power_db: np.ndarray
    peak_frequency_hz: float
    #: Height of the peak over the aperiodic background interpolated beneath it.
    prominence_db: float
    #: RMS roughness of the background the prominence is measured against, in decibels.
    #:
    #: The scale that decides whether a prominence means anything. Carried from the
    #: aperiodic fit rather than recomputed, so the peak and the background it is scored
    #: against cannot come from two different estimates.
    background_residual_db: float = 0.0
    #: Bins the peak was the maximum over.
    #:
    #: Carried because the prominence is a maximum, and what counts as a large maximum
    #: depends on how many values it was the maximum of. Without it the resolvability test
    #: cannot know whether it is correcting for a search over ten bins or a hundred.
    n_search_bins: int = 0
    #: Whether the peak is an interior local maximum rather than the edge of the search.
    #:
    #: An argmax at the boundary of a window is not established as a peak: it is the
    #: highest point of whatever was inside, and the spectrum may go on rising outside.
    #: Reporting one as a peak frequency puts the edge of the band into a cohort histogram
    #: and calls it a rhythm.
    is_interior: bool = True
    #: The next-highest peak in the band, and how far below the winner it sits.
    #:
    #: A band with two comparable bumps has no single peak frequency: the argmax picks
    #: whichever is momentarily higher, and a change too small to matter anywhere else
    #: moves the reported frequency by several hertz. Seen on real data -- one participant's
    #: peak sat at 13.4 Hz before cleaning and 10.0 Hz after, with the two bumps half a
    #: decibel apart, which is a coin toss reported as a shift. Carried so a cohort can say
    #: the frequency was contested instead of recording a bistable number as a property of
    #: the participant.
    runner_up_frequency_hz: float = float("nan")
    runner_up_gap_db: float = float("inf")

    @property
    def peak_is_contested(self) -> bool:
        """Whether a rival bump sits within the spectrum's own roughness of the winner.

        Scored against the same background scatter the prominence is scored against, so
        "contested" means the two are not separated by more than the noise that produced
        them -- not that they are close on some absolute scale.
        """
        if not np.isfinite(self.runner_up_gap_db):
            return False
        if self.background_residual_db <= 0.0:
            return False
        return self.runner_up_gap_db < self.background_residual_db

    @property
    def has_peak(self) -> bool:
        return self.prominence_db > 0.0

    def is_resolvable(
        self, *, false_positive_rate: float = RESOLVABLE_PEAK_FALSE_POSITIVE_RATE
    ) -> bool:
        """Whether the peak stands out from the spectrum's own roughness.

        :attr:`has_peak` is nearly always true, because the largest bin in a band sits
        above the fitted line by *something* and the argmax of noise is still an argmax.
        A recording with no rhythm therefore carries a plausible-looking
        :attr:`peak_frequency_hz`, and a cohort histogram built from those frequencies is
        structure that was manufactured rather than observed.

        This is the question a cohort has to ask instead, and it has to ask it of the
        maximum rather than of a single bin: see
        :func:`resolvable_prominence_threshold`. A participant that fails it is counted as
        having no resolvable peak and contributes no frequency, which is a measurement in
        its own right.
        """
        if not self.is_interior:
            # The highest point of a window is not a peak unless the spectrum comes back
            # down inside it. This one may go on rising past the band edge.
            return False
        if self.background_residual_db <= 0.0:
            return self.prominence_db > 0.0
        threshold = resolvable_prominence_threshold(
            self.n_search_bins, false_positive_rate=false_positive_rate
        )
        return self.prominence_db > threshold * self.background_residual_db


#: Separation below which two local maxima are one peak with a notch in it.
#:
#: A posterior alpha peak is one to two hertz wide, so maxima closer together than this are
#: structure within a single bump rather than two candidates for where the rhythm sits.
_RIVAL_SEPARATION_HZ = 1.0


def _runner_up(
    frequencies: np.ndarray, excess: np.ndarray, *, peak: int
) -> tuple[float, float]:
    """The next-highest separate peak in the band, and how far below the winner it is.

    Local maxima rather than bins: every bin beside the winner is lower than it, so a
    bin-wise second place would always be the winner's own shoulder. Separated by
    :data:`_RIVAL_SEPARATION_HZ`, so a notched single bump is not read as two peaks.

    Returns a missing frequency and an infinite gap where the band holds only one peak,
    which is the uncontested case and the ordinary one.
    """
    if excess.size < 3:
        return float("nan"), float("inf")
    interior = excess[1:-1]
    is_maximum = np.r_[
        False, (interior > excess[:-2]) & (interior >= excess[2:]), False
    ]
    candidates = [
        index
        for index in np.flatnonzero(is_maximum)
        if abs(float(frequencies[index]) - float(frequencies[peak]))
        >= _RIVAL_SEPARATION_HZ
    ]
    if not candidates:
        return float("nan"), float("inf")
    rival = max(candidates, key=lambda index: excess[index])
    return float(frequencies[rival]), float(excess[peak] - excess[rival])


def compute_posterior_alpha(
    inst: mne.io.BaseRaw | mne.BaseEpochs,
    *,
    band_hz: tuple[float, float] = ALPHA_BAND_HZ,
    reference_band_hz: tuple[float, float] = ALPHA_REFERENCE_BAND_HZ,
    pattern: str = POSTERIOR_PATTERN,
) -> PosteriorAlpha | None:
    """Measure the posterior alpha peak against the background beneath it.

    The peak is scored by prominence over a line fitted to the surrounding spectrum, not
    by absolute power. Absolute alpha power varies by an order of magnitude between
    participants and with electrode impedance, so it cannot distinguish "this recording
    has a rhythm" from "this recording is loud".
    """
    posterior = mne.pick_channels_regexp(inst.ch_names, pattern)
    eeg = set(mne.pick_types(inst.info, eeg=True, exclude="bads").tolist())
    picks = [index for index in posterior if index in eeg]
    if not picks:
        return None

    spectrum = inst.compute_psd(
        method="welch",
        fmin=reference_band_hz[0],
        fmax=reference_band_hz[1],
        picks=picks,
        verbose="ERROR",
    )
    power = np.asarray(spectrum.get_data(), dtype=float)
    # Epochs return (n_epochs, n_channels, n_freqs); collapse everything but frequency.
    power = power.reshape(-1, power.shape[-1])
    frequencies = np.asarray(spectrum.freqs, dtype=float)
    # Same reference as the sensor-spectra panel. Prominence is a difference and does not
    # care, but the level is drawn, and one report quoting power in two scales leaves a
    # reader unable to carry a number from one section to the next.
    power_db = (
        10.0 * np.log10(np.maximum(power.mean(axis=0), np.finfo(float).tiny))
        + MICROVOLT_REFERENCE_DB
    )

    background = fit_aperiodic(
        frequencies,
        power_db,
        fit_range_hz=reference_band_hz,
        excluded_windows=(band_hz,),
    )
    if background is None:
        return None

    in_band = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    if not in_band.any():
        return None
    band_frequencies = frequencies[in_band]
    excess = power_db[in_band] - aperiodic_line_db(background, band_frequencies)
    peak = int(np.argmax(excess))
    rival_hz, rival_gap = _runner_up(band_frequencies, excess, peak=peak)
    return PosteriorAlpha(
        channel_names=tuple(inst.ch_names[index] for index in picks),
        frequencies_hz=frequencies,
        power_db=power_db,
        background_residual_db=background.residual_db,
        peak_frequency_hz=float(frequencies[in_band][peak]),
        prominence_db=float(excess[peak]),
        n_search_bins=int(in_band.sum()),
        is_interior=bool(0 < peak < excess.size - 1),
        runner_up_frequency_hz=rival_hz,
        runner_up_gap_db=rival_gap,
    )


#: Basis stated when the measurement is made on the epochs that survived cleaning.
#:
#: Stated even though it is the default. The panel is also rendered provisionally, before
#: the exclusions are approved, and a reader who meets only one of the two has no way to
#: tell which unless every version names its own basis.
FINAL_ANALYSIS_STATUS = "Measured on the retained epochs"


def preservation_html(
    *,
    reliability: SplitHalfReliability | None = None,
    alpha: PosteriorAlpha | None = None,
    analysis_status: str = FINAL_ANALYSIS_STATUS,
) -> str:
    """Render whatever preservation evidence the paradigm supports."""
    if reliability is None and alpha is None:
        raise ValueError("The preservation section requires at least one measurement.")
    document = (
        f"<p><strong>{html.escape(analysis_status)}</strong>. Every other measurement in "
        "this report describes what preprocessing removed. A recording that was cleaned "
        "and a recording that was emptied score the same on all of them, so this section "
        "asks the opposite question.</p>"
    )
    if reliability is not None:
        document += (
            metric_table(
                [
                    ("Trials averaged", reliability.n_trials),
                    (
                        "Correlated over",
                        f"{reliability.response_window_s[0]:.2f} to "
                        f"{reliability.response_window_s[1]:.2f} s",
                    ),
                    ("Odd-vs-even correlation", f"{reliability.correlation:.3f}"),
                    Metric(
                        "Spearman-Brown corrected",
                        f"{reliability.corrected_correlation:.3f}",
                        emphasis=True,
                    ),
                ]
            )
            + "<p>Odd and even trials are averaged separately and correlated over channels "
            "and time. Splitting by alternating position rather than at the midpoint "
            "keeps slow drift in arousal or impedance on both halves equally. The "
            "measurement assumes nothing about the response's shape: if a stimulus-locked "
            "response survived, two halves of the same trials have to agree about it.</p>"
            "<p>The window matters as much as the correlation. Epochs here run well "
            "past the response to give time-frequency baselines room, and correlating "
            "across all of that would average one second of response into twenty of "
            "baseline and return near zero for a sound dataset. A paradigm whose "
            "response falls outside the window above needs the window changed, not the "
            "result interpreted.</p>"
        )
    if alpha is not None:
        document += (
            metric_table(
                [
                    (
                        "Posterior channels",
                        f"{len(alpha.channel_names)} ({', '.join(alpha.channel_names)})",
                    ),
                    ("Peak frequency", f"{alpha.peak_frequency_hz:.1f} Hz"),
                    Metric(
                        "Prominence over background",
                        f"{alpha.prominence_db:.1f} dB",
                        emphasis=True,
                    ),
                ]
            )
            + "<p>Prominence is the peak's height above the aperiodic background fitted "
            "through the surrounding spectrum, so it measures the presence of a rhythm "
            "rather than the loudness of the recording. A low value has innocent "
            "explanations — an eyes-open recording, a genuinely low-alpha participant — "
            "and is a prompt to look, not a verdict.</p>"
        )
    return document


def _draw_split_half(axis: plt.Axes, reliability: SplitHalfReliability) -> None:
    axis.plot(
        reliability.times_s,
        reliability.odd_gfp_uv,
        color=BEFORE_COLOR,
        linewidth=1.0,
        label="Odd trials",
    )
    axis.plot(
        reliability.times_s,
        reliability.even_gfp_uv,
        color=AFTER_COLOR,
        linewidth=1.0,
        label="Even trials",
    )
    axis.axvline(0.0, color=GUIDE_COLOR, linewidth=0.8)
    # Global field power is a spatial standard deviation and cannot go below zero, so
    # the axis starts there rather than floating on the data's own minimum.
    axis.set_ylim(bottom=0.0)
    window_start, window_stop = reliability.response_window_s
    axis.set(
        title=(
            f"Split-half evoked response · r = {reliability.correlation:.3f} "
            f"({reliability.corrected_correlation:.3f} corrected)\n"
            f"correlated over {window_start:.2f} to {window_stop:.2f} s"
        ),
        xlabel="Time (s)",
        ylabel="Global field power (µV)",
    )
    # The traces are global field power; the correlation above is taken over every
    # channel and time point, not over these two curves. Saying so on the panel stops
    # the figure being read as "r between the two lines shown".
    #
    # It goes below the axis rather than inside it. Global field power is at its most
    # variable in the upper half of the panel, which is exactly where an annotation
    # pinned to the top of the axes lands: on sub-0015 the caveat crossed both traces.
    axis.set_xlabel("Time (s)\nr is over all channels × times, not over these two traces")
    axis.legend(frameon=False, fontsize=8)


def _draw_posterior_alpha(axis: plt.Axes, alpha: PosteriorAlpha) -> None:
    axis.plot(alpha.frequencies_hz, alpha.power_db, color=PRIMARY_COLOR, linewidth=1.2)
    axis.axvspan(*ALPHA_BAND_HZ, color=GUIDE_COLOR, alpha=0.10, linewidth=0)
    axis.annotate(
        f"{alpha.peak_frequency_hz:.1f} Hz\n{alpha.prominence_db:.1f} dB over background",
        xy=(alpha.peak_frequency_hz, alpha.power_db.max()),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=7.5,
        color=GUIDE_COLOR,
    )
    axis.set(
        title=f"Posterior spectrum · {len(alpha.channel_names)} channels",
        xlabel="Frequency (Hz)",
        ylabel=f"PSD ({POWER_UNIT_LABEL})",
    )


def plot_preservation(
    *,
    reliability: SplitHalfReliability | None = None,
    alpha: PosteriorAlpha | None = None,
) -> plt.Figure:
    """Plot whichever preservation measurements the paradigm supported."""
    panels = []
    if reliability is not None:
        panels.append(lambda axis: _draw_split_half(axis, reliability))
    if alpha is not None:
        panels.append(lambda axis: _draw_posterior_alpha(axis, alpha))
    if not panels:
        raise ValueError("The preservation figure requires at least one measurement.")

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.6 * len(panels), 3.8),
        squeeze=False,
        layout="constrained",
    )
    for axis, draw in zip(axes[0], panels, strict=True):
        draw(axis)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


def _add_preservation_section(
    *,
    report: mne.Report,
    reliability: SplitHalfReliability | None,
    alpha: PosteriorAlpha | None,
    section: str,
    analysis_status: str = FINAL_ANALYSIS_STATUS,
) -> None:
    """Append whichever measurements were made, or nothing when none were."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_epoch_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if reliability is None and alpha is None:
        return
    remove_tagged_content(report, tag="signal-preservation")
    report.add_html(
        html=preservation_html(
            reliability=reliability,
            alpha=alpha,
            analysis_status=analysis_status,
        ),
        title="Evidence that signal survived preprocessing",
        section=section,
        tags=("epochs", "signal-preservation"),
        replace=True,
    )
    report.add_figure(
        fig=plot_preservation(reliability=reliability, alpha=alpha),
        title="Split-half response and posterior spectrum",
        section=section,
        tags=("epochs", "signal-preservation"),
        image_format=report_image_format(),
        replace=True,
    )
    move_tagged_content_before(
        report,
        tag="signal-preservation",
        anchor=before_epoch_sections,
    )


def add_task_preservation_review(
    *,
    report: mne.Report,
    epochs: mne.BaseEpochs,
    section: str = "Signal preservation",
    analysis_status: str = FINAL_ANALYSIS_STATUS,
) -> tuple[SplitHalfReliability | None, PosteriorAlpha | None]:
    """Append preservation evidence for a stimulus-locked paradigm.

    ``analysis_status`` names the epochs the measurement was made on. It exists because
    this section is rendered twice: provisionally at ICA review, where it is the only
    counterweight to a page of removal measurements, and finally after rejection. The
    tag-scoped removal in :func:`_add_preservation_section` means the second supersedes
    the first rather than sitting beside it.

    Returns the measurements so a caller can log them.
    """
    reliability = compute_split_half_reliability(epochs)
    alpha = compute_posterior_alpha(epochs)
    _add_preservation_section(
        report=report,
        reliability=reliability,
        alpha=alpha,
        section=section,
        analysis_status=analysis_status,
    )
    return reliability, alpha


def add_rest_preservation_review(
    *,
    report: mne.Report,
    epochs: mne.BaseEpochs,
    section: str = "Signal preservation",
    analysis_status: str = FINAL_ANALYSIS_STATUS,
) -> PosteriorAlpha | None:
    """Append preservation evidence for a resting-state recording.

    Rest has no stimulus to lock to, so split-half reliability of an evoked response is
    undefined rather than merely weak, and only the posterior rhythm is measured.
    """
    alpha = compute_posterior_alpha(epochs)
    _add_preservation_section(
        report=report,
        reliability=None,
        alpha=alpha,
        section=section,
        analysis_status=analysis_status,
    )
    return alpha


__all__ = [
    "ALPHA_BAND_HZ",
    "ALPHA_REFERENCE_BAND_HZ",
    "DEFAULT_RESPONSE_WINDOW_S",
    "FINAL_ANALYSIS_STATUS",
    "MINIMUM_TRIALS_FOR_SPLIT_HALF",
    "POSTERIOR_PATTERN",
    "PosteriorAlpha",
    "SplitHalfReliability",
    "add_rest_preservation_review",
    "add_task_preservation_review",
    "compute_posterior_alpha",
    "compute_split_half_reliability",
    "plot_preservation",
    "preservation_html",
]
