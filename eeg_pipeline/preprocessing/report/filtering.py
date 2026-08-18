"""What the configured filter actually is, as opposed to what was asked for.

The provenance table records ``l_freq``, ``h_freq`` and ``notch_freq``, which is what a
reader needs to reproduce the run. It is not what a reader needs to judge the run. Those
three numbers are a request; MNE turns them into a finite impulse response whose length,
transition widths, and settling time follow from the sampling rate and are nowhere in the
configuration. At 500 Hz a 0.1 Hz high-pass is a 33-second filter — longer than the
22-second epochs it is applied to — and that relationship is the one that decides whether
a slow drift in one trial can reach the next.

Nothing here grades the filter. The panel states the response that was built and marks
the analysis window on it; whether the two are compatible is a judgement about the effect
being measured, which this module cannot see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import freqz

from eeg_pipeline.preprocessing.report.style import (
    FLAG_COLOR,
    GUIDE_COLOR,
    PRIMARY_COLOR,
    REPORT_IMAGE_FORMAT,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import (
    Align,
    Column,
    Metric,
    grid_table,
    metric_table,
)

#: Attenuation defining a filter corner, matching the convention MNE reports its own
#: filters with, so the numbers here and in MNE's log are the same quantity.
CUTOFF_DB = -6.0

#: Half-width of the band a notch filter is treated as having removed, in hertz.
#:
#: A notch cuts a narrow trough to the numerical floor. Every measurement that reads
#: power at a frequency has to know which bins are the filter rather than the data, or it
#: reports the filter as a finding: the aperiodic fit tilts on the trough, the spectral
#: axis stretches 40 dB to contain it, and a narrowband measurement scores its deepest line
#: where the notch happens to fall on a harmonic.
NOTCH_EXCLUSION_HALF_WIDTH_HZ = 2.0


def notch_windows(
    line_frequency: float | None,
    *,
    fmax: float,
    half_width: float = NOTCH_EXCLUSION_HALF_WIDTH_HZ,
    unavailable_intervals: Sequence[tuple[float, float]] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return the bands a filter removed, from every source that recorded one.

    Shared by every section that measures power at a frequency, so that they cannot
    disagree about which bins the filter owns.

    A configured ``line_frequency`` contributes a harmonic grid padded by ``half_width``,
    because the width of that fixed notch is not recorded anywhere. ``unavailable_intervals``
    are measured bands that already span their stopband and FIR transitions, so they are
    used as given; padding them would claim bins the filter never touched.
    """
    windows: list[tuple[float, float]] = []

    if line_frequency:
        harmonics = np.arange(line_frequency, fmax + line_frequency, line_frequency)
        windows.extend(
            (float(harmonic - half_width), float(harmonic + half_width))
            for harmonic in harmonics
            if harmonic - half_width < fmax
        )

    for low, high in unavailable_intervals or ():
        if float(low) >= fmax:
            continue
        windows.append((float(low), min(float(high), float(fmax))))

    return tuple(sorted(windows))


def in_notch(frequencies: np.ndarray, windows: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Return a mask marking the frequencies that fall inside a notch window."""
    inside = np.zeros(np.shape(frequencies), dtype=bool)
    for low, high in windows:
        inside |= (np.asarray(frequencies) >= low) & (np.asarray(frequencies) <= high)
    return inside


#: Resolution of the frequency response. Enough bins that a 0.1 Hz corner at 500 Hz is
#: resolved rather than interpolated across from DC.
_RESPONSE_BINS = 16384


@dataclass(frozen=True)
class FilterDescription:
    """The realised response of one band-pass configuration."""

    sfreq: float
    l_freq: float | None
    h_freq: float | None
    notch_freq: float | None
    #: Impulse response MNE builds for this configuration.
    taps: np.ndarray
    frequencies_hz: np.ndarray
    magnitude_db: np.ndarray
    #: Measured -6 dB corners, which sit outside the requested passband edges.
    low_cutoff_hz: float | None
    high_cutoff_hz: float | None

    @property
    def n_taps(self) -> int:
        return int(self.taps.size)

    @property
    def duration_s(self) -> float:
        """Length of the impulse response in seconds, comparable with an epoch."""
        return float(self.taps.size) / float(self.sfreq)

    @property
    def edge_support_s(self) -> float:
        """One-sided support of this linear-phase FIR around an output sample.

        MNE's zero-phase FIR is centred, so an output sample depends on this much input
        on either side. This measured property, rather than a time constant inferred from
        a cutoff frequency, defines the run edge that continuity statistics exclude.
        """
        return float(self.taps.size - 1) / (2.0 * float(self.sfreq))


def _cutoff(frequencies: np.ndarray, magnitude_db: np.ndarray, *, edge: str) -> float | None:
    """Return the frequency where the response crosses :data:`CUTOFF_DB`.

    ``None`` when the response never drops below the threshold on that side, which is the
    honest answer for a band that was not filtered: there is no corner to report.
    """
    passband = magnitude_db > CUTOFF_DB
    if not passband.any():
        return None
    indices = np.flatnonzero(passband)
    if edge == "low":
        # A response that is already in the passband at the first analysed bin has no
        # measurable lower corner: the high-pass, if any, turns over below the resolution.
        return float(frequencies[indices[0]]) if indices[0] > 0 else None
    return float(frequencies[indices[-1]]) if indices[-1] < frequencies.size - 1 else None


def describe_filter(
    *,
    sfreq: float,
    l_freq: float | None,
    h_freq: float | None,
    notch_freq: float | None = None,
) -> FilterDescription:
    """Build the filter MNE would build and measure the response it produces.

    The filter is constructed with MNE's own defaults rather than a reimplementation, so
    the panel describes the filter the pipeline applies and not one that resembles it.
    """
    if l_freq is None and h_freq is None:
        raise ValueError("A filter description needs at least one of l_freq and h_freq.")
    taps = mne.filter.create_filter(
        data=None,
        sfreq=float(sfreq),
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        verbose="ERROR",
    )
    frequencies, response = freqz(taps, worN=_RESPONSE_BINS, fs=float(sfreq))
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    # The DC bin is not a meaningful corner candidate and would otherwise be read as one
    # for a low-pass-only configuration.
    frequencies, magnitude_db = frequencies[1:], magnitude_db[1:]
    return FilterDescription(
        sfreq=float(sfreq),
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=notch_freq,
        taps=np.asarray(taps, dtype=float),
        frequencies_hz=frequencies,
        magnitude_db=magnitude_db,
        low_cutoff_hz=_cutoff(frequencies, magnitude_db, edge="low") if l_freq else None,
        high_cutoff_hz=_cutoff(frequencies, magnitude_db, edge="high") if h_freq else None,
    )


def describe_configured_filter(config: object, *, sfreq: float) -> FilterDescription | None:
    """Describe the filter a configuration asks for, or ``None`` when it asks for none.

    The keys are the ones the provenance table records, so the panel and the configuration
    table cannot end up describing different filters.

    ``sfreq`` fixes the tap count but not the filter's length in seconds: MNE sets the
    transition bandwidth in hertz, so the response lasts the same time whatever rate it is
    sampled at. Passing the rate of the data on disk therefore gives the right duration
    even where filtering happened before resampling, and the tap count is stated against
    the rate it was computed for.
    """
    l_freq = config.get("preprocessing.l_freq", None)
    h_freq = config.get("preprocessing.h_freq", None)
    if l_freq is None and h_freq is None:
        return None
    return describe_filter(
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=config.get("preprocessing.notch_freq", None),
    )


def _upstream_stopband_html(
    unavailable_intervals_by_recording: Mapping[str, Sequence[tuple[float, float]]],
    subject: str | None,
) -> str:
    """Account for frequencies a stage before this pipeline removed from the recordings.

    The response above describes the filter *this configuration* builds. Where a line- or
    comb-removal stage ran upstream, the delivered data also carries its stopbands, and
    nothing in the configuration records them: a report whose filter section shows a flat
    passband from 0.06 to 112 Hz over data with eighty notches in it is describing a
    recording that does not exist.

    The intervals are the same ones the comb measurement masks its harmonics with, so the
    two sections cannot disagree about what was removed.
    """
    # These settings hold every participant in the study, and a run label drops the
    # subject: rendered whole, a six-run session showed ninety-six rows cycling through
    # run-1..run-6, each subject's intervals presented as this one's.
    prefix = f"sub-{subject}_" if subject else None
    rows = []
    for recording_id, intervals in sorted(unavailable_intervals_by_recording.items()):
        if prefix is not None and not recording_id.startswith(prefix):
            continue
        usable = [(float(low), float(high)) for low, high in intervals if high > low]
        if not usable:
            continue
        removed = sum(high - low for low, high in usable)
        span = (min(low for low, _ in usable), max(high for _, high in usable))
        rows.append(
            [
                run_label(recording_id),
                len(usable),
                f"{removed:.1f}",
                f"{span[0]:.1f}–{span[1]:.1f}",
            ]
        )
    if not rows:
        return ""
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Stopbands"),
        Column("Bandwidth removed (Hz)"),
        Column("Between (Hz)", align=Align.TEXT),
    )
    return (
        "<p>Frequencies an upstream stage removed before this pipeline read the "
        "recordings. They are not part of the response above, which describes only the "
        "filter this configuration builds — but they are part of the data every spectrum "
        "in this report is measured on, and no configuration key records them. A trough "
        "in a spectrum at one of these frequencies is that removal, not the recording.</p>"
        + grid_table(columns, rows)
        + "<p>Bandwidth removed is the total width of the stopbands, which is not the "
        "same as the span they fall between: a comb of narrow notches and one wide one "
        "can remove the same bandwidth over very different ranges. These are the "
        "intervals a narrowband measurement excludes its own lines with, so the two "
        "sections describe one set of removals.</p>"
    )


def filter_response_html(
    description: FilterDescription,
    unavailable_intervals_by_recording: Mapping[str, Sequence[tuple[float, float]]]
    | None = None,
    subject: str | None = None,
) -> str:
    """Render the filter's measured properties beside the settings that requested them."""

    def frequency(value: float | None) -> str:
        return "&mdash;" if value is None else f"{value:.3g} Hz"

    rows = [
        ("Sampling rate", f"{description.sfreq:g} Hz"),
        ("Requested high-pass", frequency(description.l_freq)),
        ("Requested low-pass", frequency(description.h_freq)),
        ("Notch", frequency(description.notch_freq)),
        (
            f"Measured {CUTOFF_DB:g} dB corners",
            f"{frequency(description.low_cutoff_hz)} – {frequency(description.high_cutoff_hz)}",
        ),
        ("Filter length", f"{description.n_taps} taps"),
        Metric("Filter length in time", f"{description.duration_s:.1f} s", emphasis=True),
        Metric("Edge support on each side", f"{description.edge_support_s:.1f} s"),
    ]
    return (
        "<p>The filter this configuration produces, built with the same MNE defaults the "
        "pipeline applies. The requested frequencies sit inside the passband: MNE places "
        f"the {CUTOFF_DB:g} dB point outside each of them, so the corner a reader "
        "measures from a spectrum is not the number in the configuration table.</p>"
        f"{metric_table(rows)}"
        "<p>Filter length is given in seconds because that is the unit the analysis "
        "window is in. A high-pass filter is long in inverse proportion to its cutoff, so "
        "a low cutoff on a fast recording produces an impulse response that can be longer "
        "than a single epoch. Whether that matters depends on the effect being measured "
        "&mdash; a slow evoked response and a burst-rate estimate are not equally exposed "
        "to it &mdash; so the step response below is drawn against the epoch window "
        "rather than judged here.</p>"
        + _upstream_stopband_html(unavailable_intervals_by_recording or {}, subject)
    )


def plot_filter_response(
    description: FilterDescription,
    *,
    epoch_window_s: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot the magnitude response and the step response of the configured filter.

    Two panels because the two questions are different. The magnitude response says which
    frequencies survive, which is what the spectra sections are read against. The step
    response says how long the filter takes to settle after a transition, which is what
    decides whether the filter is still ringing when the epoch of interest starts.
    """
    figure, (magnitude_axis, step_axis) = plt.subplots(
        2, 1, figsize=(7.6, 6.0), layout="constrained"
    )

    magnitude_axis.semilogx(
        description.frequencies_hz,
        description.magnitude_db,
        color=PRIMARY_COLOR,
        linewidth=1.2,
    )
    magnitude_axis.axhline(
        CUTOFF_DB,
        color=GUIDE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"{CUTOFF_DB:g} dB",
    )
    # Distinct dash patterns rather than two identical dotted lines: the markers sit at
    # opposite ends of a log axis, so the legend is never beside the line it names and
    # colour alone left the pair separable only by legend order.
    for value, style, label in (
        (description.l_freq, ":", "requested high-pass"),
        (description.h_freq, "-.", "requested low-pass"),
    ):
        if value:
            magnitude_axis.axvline(
                float(value), color=FLAG_COLOR, linestyle=style, linewidth=1.0, label=label
            )
    magnitude_axis.set(
        title="Magnitude response of the configured filter",
        xlabel="Frequency (Hz)",
        ylabel="Gain (dB)",
        ylim=(-80.0, 5.0),
        xlim=(float(description.frequencies_hz[0]), description.sfreq / 2.0),
    )
    magnitude_axis.grid(alpha=0.2, which="both")
    magnitude_axis.legend(frameon=False, fontsize=8, loc="lower center", ncol=3)

    # Step rather than impulse: a step is what a drift or a shifted baseline looks like to
    # the filter, and its recovery is directly readable as a settling time.
    step = np.cumsum(description.taps)
    times_s = (np.arange(step.size) - step.size // 2) / description.sfreq
    step_axis.plot(times_s, step, color=PRIMARY_COLOR, linewidth=1.0)
    step_axis.axhline(0.0, color=GUIDE_COLOR, linewidth=0.8)
    if epoch_window_s is not None:
        for edge in epoch_window_s:
            step_axis.axvline(
                float(edge),
                color=FLAG_COLOR,
                linestyle="--",
                linewidth=1.0,
            )
        step_axis.set_xlim(
            min(times_s[0], float(epoch_window_s[0]) * 1.2),
            max(times_s[-1], float(epoch_window_s[1]) * 1.2),
        )
        epoch_note = " · dashed lines mark the epoch window"
    else:
        epoch_note = ""
    step_axis.set(
        title=f"Step response{epoch_note}",
        xlabel="Time relative to the step (s)",
        ylabel="Response",
    )
    step_axis.grid(alpha=0.2)
    for axis in (magnitude_axis, step_axis):
        axis.spines[["top", "right"]].set_visible(False)
    plt.close(figure)
    return figure


def add_filter_review(
    *,
    report: mne.Report,
    description: FilterDescription,
    epoch_window_s: tuple[float, float] | None = None,
    unavailable_intervals_by_recording: Mapping[str, Sequence[tuple[float, float]]]
    | None = None,
    subject: str | None = None,
    section: str = "Filter response",
) -> None:
    """Add the realised filter response, ahead of the spectra it is read against."""
    from eeg_pipeline.preprocessing.report.organize import (
        remove_tagged_content,
    )

    remove_tagged_content(report, tag="filter-response")
    tags = ("raw", "filter-response")
    report.add_html(
        html=filter_response_html(
            description,
            unavailable_intervals_by_recording=unavailable_intervals_by_recording,
            subject=subject,
        ),
        title="What the configured filter actually is",
        section=section,
        tags=tags,
        replace=True,
    )
    report.add_figure(
        fig=plot_filter_response(description, epoch_window_s=epoch_window_s),
        title="Magnitude and step response",
        section=section,
        tags=tags,
        image_format=REPORT_IMAGE_FORMAT,
        replace=True,
    )


__all__ = [
    "NOTCH_EXCLUSION_HALF_WIDTH_HZ",
    "in_notch",
    "notch_windows",
    "CUTOFF_DB",
    "FilterDescription",
    "add_filter_review",
    "describe_configured_filter",
    "describe_filter",
    "filter_response_html",
    "plot_filter_response",
]
