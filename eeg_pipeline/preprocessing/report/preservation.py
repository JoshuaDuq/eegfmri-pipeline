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
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.aperiodic import aperiodic_line_db, fit_aperiodic
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    PRIMARY_COLOR,
)

#: Band searched for the posterior alpha peak.
ALPHA_BAND_HZ = (7.0, 14.0)

#: Neighbourhood the alpha peak is measured against, so that a peak is scored by how far
#: it stands above the local background rather than by absolute power.
ALPHA_REFERENCE_BAND_HZ = (3.0, 25.0)

#: Regular expression matching the posterior sensors alpha is expected over.
POSTERIOR_PATTERN = r"^(P[0-9z]|PO[0-9z]|O[0-9z]|Oz|POz|Pz)"

#: Trials needed before a split-half average is an average rather than a few trials.
MINIMUM_TRIALS_FOR_SPLIT_HALF = 20

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
    #: Across-channel mean of the odd- and even-trial averages, in microvolts.
    odd_uv: np.ndarray
    even_uv: np.ndarray
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
        odd_uv=odd.mean(axis=0) * 1e6,
        even_uv=even.mean(axis=0) * 1e6,
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

    @property
    def has_peak(self) -> bool:
        return self.prominence_db > 0.0


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
    power_db = 10.0 * np.log10(np.maximum(power.mean(axis=0), np.finfo(float).tiny))

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
    excess = power_db[in_band] - aperiodic_line_db(background, frequencies[in_band])
    peak = int(np.argmax(excess))
    return PosteriorAlpha(
        channel_names=tuple(inst.ch_names[index] for index in picks),
        frequencies_hz=frequencies,
        power_db=power_db,
        peak_frequency_hz=float(frequencies[in_band][peak]),
        prominence_db=float(excess[peak]),
    )


def preservation_html(
    *,
    reliability: SplitHalfReliability | None = None,
    alpha: PosteriorAlpha | None = None,
) -> str:
    """Render whatever preservation evidence the paradigm supports."""
    if reliability is None and alpha is None:
        raise ValueError("The preservation section requires at least one measurement.")
    document = (
        "<p>Every other measurement in this report describes what preprocessing "
        "removed. A recording that was cleaned and a recording that was emptied score "
        "the same on all of them, so this section asks the opposite question.</p>"
    )
    if reliability is not None:
        document += (
            "<table><tbody>"
            f"<tr><td>Trials averaged</td><td>{reliability.n_trials}</td></tr>"
            f"<tr><td>Correlated over</td>"
            f"<td>{reliability.response_window_s[0]:.2f} to "
            f"{reliability.response_window_s[1]:.2f} s</td></tr>"
            f"<tr><td>Odd-vs-even correlation</td>"
            f"<td>{reliability.correlation:.3f}</td></tr>"
            f"<tr><td><strong>Spearman-Brown corrected</strong></td>"
            f"<td><strong>{reliability.corrected_correlation:.3f}</strong></td></tr>"
            "</tbody></table>"
            "<p>Odd and even trials are averaged separately and correlated over channels "
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
            "<table><tbody>"
            f"<tr><td>Posterior channels</td>"
            f"<td>{len(alpha.channel_names)} "
            f"({html.escape(', '.join(alpha.channel_names))})</td></tr>"
            f"<tr><td>Peak frequency</td><td>{alpha.peak_frequency_hz:.1f} Hz</td></tr>"
            f"<tr><td><strong>Prominence over background</strong></td>"
            f"<td><strong>{alpha.prominence_db:.1f} dB</strong></td></tr>"
            "</tbody></table>"
            "<p>Prominence is the peak's height above the aperiodic background fitted "
            "through the surrounding spectrum, so it measures the presence of a rhythm "
            "rather than the loudness of the recording. A low value has innocent "
            "explanations — an eyes-open recording, a genuinely low-alpha participant — "
            "and is a prompt to look, not a verdict.</p>"
        )
    return document


def _draw_split_half(axis: plt.Axes, reliability: SplitHalfReliability) -> None:
    axis.plot(
        reliability.times_s,
        reliability.odd_uv,
        color=BEFORE_COLOR,
        linewidth=1.0,
        label="Odd trials",
    )
    axis.plot(
        reliability.times_s,
        reliability.even_uv,
        color=AFTER_COLOR,
        linewidth=1.0,
        label="Even trials",
    )
    axis.axvline(0.0, color=GUIDE_COLOR, linewidth=0.8)
    axis.axhline(0.0, color=GUIDE_COLOR, linewidth=0.8)
    window_start, window_stop = reliability.response_window_s
    axis.set(
        title=(
            f"Split-half evoked response · r = {reliability.correlation:.3f} "
            f"({reliability.corrected_correlation:.3f} corrected)\n"
            f"correlated over {window_start:.2f} to {window_stop:.2f} s"
        ),
        xlabel="Time (s)",
        ylabel="Mean across channels (µV)",
    )
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
        ylabel="PSD (dB)",
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
        html=preservation_html(reliability=reliability, alpha=alpha),
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
) -> tuple[SplitHalfReliability | None, PosteriorAlpha | None]:
    """Append preservation evidence for a stimulus-locked paradigm.

    Returns the measurements so a caller can log them.
    """
    reliability = compute_split_half_reliability(epochs)
    alpha = compute_posterior_alpha(epochs)
    _add_preservation_section(report=report, reliability=reliability, alpha=alpha, section=section)
    return reliability, alpha


def add_rest_preservation_review(
    *,
    report: mne.Report,
    epochs: mne.BaseEpochs,
    section: str = "Signal preservation",
) -> PosteriorAlpha | None:
    """Append preservation evidence for a resting-state recording.

    Rest has no stimulus to lock to, so split-half reliability of an evoked response is
    undefined rather than merely weak, and only the posterior rhythm is measured.
    """
    alpha = compute_posterior_alpha(epochs)
    _add_preservation_section(report=report, reliability=None, alpha=alpha, section=section)
    return alpha


__all__ = [
    "ALPHA_BAND_HZ",
    "ALPHA_REFERENCE_BAND_HZ",
    "DEFAULT_RESPONSE_WINDOW_S",
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
