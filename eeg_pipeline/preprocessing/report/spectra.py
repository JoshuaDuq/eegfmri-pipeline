"""Sensor-space power spectra before and after ICA cleaning.

The spectrum is the most information-dense single view of an EEG recording: line noise,
drift, muscle, and residual scanner harmonics each have a recognisable signature in it,
and comparing the same run before and after cleaning shows what the pipeline removed
across the whole frequency range rather than one component at a time.

Three properties of this panel are deliberate.

The across-channel *spread* is drawn, not only the median. Residual gradient and pulse
artifact are spatially focal — peripheral sensors, large lead loops, the temporal
chain — and a median across sixty channels stays clean while two channels are ruined.
The median alone answers "is the montage typical"; the spread answers "is any channel
unusable", which is the question that decides whether a region survives.

The frequency axis stops where the data stops. These spectra are computed on the
filtered recording, so everything above the configured low-pass is filter roll-off.
Plotting it to Nyquist presents sixty percent of the axis as measurement when it is
attenuation.

The aperiodic background is fitted and drawn. Every other number in the report measures
what cleaning removed; the slope and offset measure what it left behind.

Nothing here is judged. Frequencies of interest are marked so they can be found, and the
reviewer decides what the spectra mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.ticker import ScalarFormatter

from eeg_pipeline.preprocessing.report.aperiodic import (
    DEFAULT_FIT_RANGE_HZ,
    AperiodicFit,
    aperiodic_line_db,
    fit_aperiodic,
)
from eeg_pipeline.preprocessing.report.filtering import notch_windows
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

#: Welch segment length in seconds. Long enough to resolve a narrow line-noise peak.
WELCH_SECONDS = 4.0

#: Percentiles of the across-channel distribution drawn as a shaded band.
SPREAD_PERCENTILES = (10.0, 90.0)



@dataclass(frozen=True)
class StageSpectrum:
    """One stage of one run, collapsed across channels.

    The maximum is carried alongside the median because they answer different questions:
    the median describes the montage, the maximum describes its worst sensor, and a focal
    residual moves only the second.
    """

    median_db: np.ndarray
    spread_low_db: np.ndarray
    spread_high_db: np.ndarray
    max_db: np.ndarray
    aperiodic: AperiodicFit | None

    @property
    def worst_channel_gap_db(self) -> float:
        """Widest gap between the worst channel and the median, over the whole band."""
        return float(np.max(self.max_db - self.median_db))


def summarize_stage(
    frequencies: np.ndarray,
    power_db: np.ndarray,
    *,
    excluded_windows: Sequence[tuple[float, float]],
) -> StageSpectrum:
    """Collapse per-channel decibels and fit the aperiodic background of the median."""
    low, high = np.percentile(power_db, SPREAD_PERCENTILES, axis=0)
    median = np.median(power_db, axis=0)
    return StageSpectrum(
        median_db=median,
        spread_low_db=low,
        spread_high_db=high,
        max_db=np.max(power_db, axis=0),
        aperiodic=fit_aperiodic(
            frequencies,
            median,
            fit_range_hz=DEFAULT_FIT_RANGE_HZ,
            excluded_windows=excluded_windows,
        ),
    )


@dataclass(frozen=True)
class RunSpectra:
    """Across-channel sensor spectrum for one run, before and after ICA."""

    recording_id: str
    frequencies: np.ndarray
    before: StageSpectrum
    after: StageSpectrum
    n_channels: int
    #: Upper edge of the plotted band and why it sits there.
    fmax_reason: str

    @property
    def exponent_change(self) -> float | None:
        """Change in aperiodic exponent across ICA, when both fits succeeded."""
        if self.before.aperiodic is None or self.after.aperiodic is None:
            return None
        return self.after.aperiodic.exponent - self.before.aperiodic.exponent


#: Power reference the spectra are expressed against, as a decibel offset from V²/Hz.
#:
#: MNE returns EEG power in V²/Hz, which puts an ordinary spectrum between about -150 and
#: -120 dB. Those numbers are right and unreadable: EEG is quoted in microvolts
#: everywhere, so no published reference value lands in that scale and a reviewer has
#: nothing to read the level against. 1 V² is 1e12 µV², hence a fixed 120 dB. It is a
#: shift, not a rescaling, so every shape in the figure — slope, peak prominence, the
#: before/after difference — is unchanged.
MICROVOLT_REFERENCE_DB = 120.0

#: Unit string for the power axis and any table column carrying one of these levels.
POWER_UNIT_LABEL = "dB re 1 µV²/Hz"


def _channel_spectra_db(
    raw: mne.io.BaseRaw,
    *,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frequency grid and per-channel power in decibels re 1 µV²/Hz.

    Bad channels are excluded: ``picks="eeg"`` drops ``info["bads"]``, so a sensor the
    pipeline already rejected cannot inflate the spread it is not part of.

    The reference conversion happens here rather than at each drawing site so that the
    aperiodic fit, the table, and the figure cannot end up quoting different units for
    the same quantity.
    """
    segment = min(int(round(WELCH_SECONDS * raw.info["sfreq"])), raw.n_times)
    spectrum = raw.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks="eeg",
        n_fft=segment,
        n_per_seg=segment,
        n_overlap=segment // 2,
        verbose="ERROR",
    )
    power = np.asarray(spectrum.get_data(), dtype=float)
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    return np.asarray(spectrum.freqs, dtype=float), power_db + MICROVOLT_REFERENCE_DB


def _set_power_limits(
    axis: plt.Axes,
    spectra: "RunSpectra",
    *,
    line_frequency: float | None,
) -> None:
    """Bound the power axis by the spectrum rather than by the notch it contains.

    A notch filter drives its band to the numerical floor, which for this data is tens of
    decibels below anything else in the run. Included in the limits it stretched the axis
    by 40 dB and left the spectrum the panel exists to show squeezed into the top third.

    The notch bands are the same ones the aperiodic fit already excludes, so the figure and
    the fit agree about which bins are filter rather than signal. The trace is still drawn
    through them and simply leaves the axis, which reads as a filtered band rather than as
    missing data.
    """
    windows = notch_windows(line_frequency, fmax=float(spectra.frequencies[-1]))
    keep = np.ones_like(spectra.frequencies, dtype=bool)
    for low, high in windows:
        keep &= (spectra.frequencies < low) | (spectra.frequencies > high)
    if not keep.any():
        return
    stacked = np.concatenate(
        [
            stage_values[keep]
            for stage in (spectra.before, spectra.after)
            for stage_values in (stage.spread_low_db, stage.max_db)
        ]
    )
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return
    low, high = float(np.min(finite)), float(np.max(finite))
    margin = max(1.0, 0.05 * (high - low))
    axis.set_ylim(low - margin, high + margin)


def _resolve_ceiling(sfreq: float, fmax: float | None) -> tuple[float, str]:
    """Return the upper edge of the plotted band, and why it sits there."""
    nyquist = sfreq / 2.0
    highest_measurable = nyquist - 1.0
    if fmax is None or fmax >= highest_measurable:
        return highest_measurable, f"Nyquist ({nyquist:.0f} Hz)"
    return float(fmax), f"configured low-pass ({fmax:g} Hz)"


#: Half-width of a withheld harmonic, in frequency bins either side of the tooth.
#:
#: One bin is not enough: a tooth leaks into its neighbours through the Welch window, so
#: the bins beside it are part of the peak rather than part of the background. Three is not
#: better: at a short repetition time the harmonics are close together and a wide skirt
#: would withhold the whole range, leaving the fit nothing to sit on.
_HARMONIC_SKIRT_BINS = 1.5


def gradient_windows(
    fundamental_hz: float | None,
    *,
    frequencies: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    """Frequency windows covering the gradient comb, for exclusion from a fit.

    Empty where there is no volume rate, which is the ordinary case outside a scanner.

    Withheld rather than trimmed, because the peak-residual trim inside
    :func:`fit_aperiodic` finds outliers against a line that the comb has already tilted.
    Naming the harmonics is possible here and guessing is not: the volume rate is measured.
    """
    if not fundamental_hz or fundamental_hz <= 0.0:
        return ()
    grid = np.asarray(frequencies, dtype=float)
    if grid.size < 2:
        return ()
    skirt = _HARMONIC_SKIRT_BINS * float(np.median(np.diff(grid)))
    highest = float(grid[-1])
    orders = range(1, int(highest / fundamental_hz) + 1)
    return tuple(
        (order * fundamental_hz - skirt, order * fundamental_hz + skirt) for order in orders
    )


def compute_run_spectra(
    raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    *,
    recording_id: str,
    fmin: float = 1.0,
    fmax: float | None = None,
    line_frequency: float | None = None,
    gradient_fundamental_hz: float | None = None,
) -> RunSpectra:
    """Compute the across-channel sensor spectrum before and after ICA.

    ``cleaned`` is the ICA-applied recording, passed in rather than derived here so a
    caller measuring several things about one run applies the decomposition once —
    ``ICA.apply`` on a long run is the expensive step in this module.

    ``fmax`` should be the configured low-pass. Above it the filter, not the recording,
    determines the trace, so plotting further presents roll-off as data.

    ``gradient_fundamental_hz`` is the volume rate of an in-scanner recording. Its
    harmonics are withheld from the aperiodic fit: the comb runs straight through the fit
    range, and a line fitted across a forest of narrow peaks is a line fitted partly to the
    scanner. Absent for a recording made outside a bore, which has no comb.
    """
    upper, reason = _resolve_ceiling(float(raw.info["sfreq"]), fmax)
    if upper <= fmin:
        raise ValueError(f"{recording_id}: no frequency range between {fmin} and {upper} Hz.")

    frequencies, before_channels = _channel_spectra_db(raw, fmin=fmin, fmax=upper)
    _, after_channels = _channel_spectra_db(cleaned, fmin=fmin, fmax=upper)
    excluded = tuple(notch_windows(line_frequency, fmax=upper)) + gradient_windows(
        gradient_fundamental_hz, frequencies=frequencies
    )
    return RunSpectra(
        recording_id=recording_id,
        frequencies=frequencies,
        before=summarize_stage(frequencies, before_channels, excluded_windows=excluded),
        after=summarize_stage(frequencies, after_channels, excluded_windows=excluded),
        n_channels=before_channels.shape[0],
        fmax_reason=reason,
    )


def _draw_stage(axis: plt.Axes, spectra: RunSpectra, stage: StageSpectrum, color: str, label: str):
    """Draw one stage's median, across-channel spread, worst channel, and fitted slope."""
    axis.fill_between(
        spectra.frequencies,
        stage.spread_low_db,
        stage.spread_high_db,
        color=color,
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(spectra.frequencies, stage.median_db, color=color, linewidth=1.2, label=label)
    # The worst channel is the point of the panel: a focal residual leaves the median
    # untouched and shows up here alone.
    axis.plot(
        spectra.frequencies,
        stage.max_db,
        color=color,
        linewidth=0.8,
        linestyle=":",
        alpha=0.9,
    )
    if stage.aperiodic is None:
        return
    low, high = stage.aperiodic.fit_range_hz
    band = (spectra.frequencies >= low) & (spectra.frequencies <= high)
    axis.plot(
        spectra.frequencies[band],
        aperiodic_line_db(stage.aperiodic, spectra.frequencies[band]),
        color=color,
        linewidth=1.6,
        linestyle="--",
        alpha=0.55,
    )


def plot_run_spectra(
    spectra: RunSpectra,
    *,
    line_frequency: float | None = None,
    marked_frequencies: Sequence[float] = (),
) -> plt.Figure:
    """Plot the before/after spectra, the across-channel spread, and their difference.

    The frequency axis is logarithmic: on a linear axis a 1-100 Hz range gives the delta
    and theta bands a few pixels while the high-frequency tail, which is mostly noise
    floor, takes most of the width.
    """
    figure, (level_axis, difference_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.2),
        height_ratios=(2, 1),
        sharex=True,
        layout="constrained",
    )
    _draw_stage(level_axis, spectra, spectra.before, BEFORE_COLOR, "Before ICA")
    _draw_stage(level_axis, spectra, spectra.after, AFTER_COLOR, "After ICA")

    low_percentile, high_percentile = SPREAD_PERCENTILES
    level_axis.set(
        title=(
            f"{spectra.recording_id} · {spectra.n_channels} good EEG channels\n"
            f"median (solid), {low_percentile:g}-{high_percentile:g}th percentile across "
            "channels (shaded), worst channel (dotted), aperiodic fit (dashed)"
        ),
        ylabel=f"PSD ({POWER_UNIT_LABEL})",
    )
    _set_power_limits(level_axis, spectra, line_frequency=line_frequency)
    level_axis.legend(frameon=False, fontsize=8)

    difference_axis.plot(
        spectra.frequencies,
        spectra.after.median_db - spectra.before.median_db,
        color=GUIDE_COLOR,
        linewidth=1.0,
        label="median channel",
    )
    difference_axis.plot(
        spectra.frequencies,
        spectra.after.max_db - spectra.before.max_db,
        color=GUIDE_COLOR,
        linewidth=0.8,
        linestyle=":",
        label="worst channel",
    )
    difference_axis.axhline(0.0, color="black", linewidth=0.8)
    difference_axis.set(
        xlabel=f"Frequency (Hz) · upper edge set by the {spectra.fmax_reason}",
        ylabel="After − before (dB)",
        xscale="log",
    )
    difference_axis.legend(frameon=False, fontsize=7, ncol=2)
    ticks = [
        tick
        for tick in (1, 2, 5, 10, 20, 50, 100, 200)
        if spectra.frequencies[0] <= tick <= spectra.frequencies[-1]
    ]
    difference_axis.set_xticks(ticks)
    difference_axis.xaxis.set_major_formatter(ScalarFormatter())

    marks = list(marked_frequencies)
    if line_frequency:
        marks.extend(np.arange(line_frequency, spectra.frequencies[-1], line_frequency))
    for axis in (level_axis, difference_axis):
        for mark in marks:
            if spectra.frequencies[0] <= mark <= spectra.frequencies[-1]:
                axis.axvline(mark, color=GUIDE_COLOR, linestyle=":", linewidth=0.7, alpha=0.7)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    notes = []
    if marks:
        notes.append(
            "dotted vertical lines mark line-noise harmonics, gradient harmonics, "
            "and configured frequencies of interest"
        )
    # The power axis is bounded excluding the notch bands, so the trace dives off the
    # bottom of the panel at each one. Undeclared, a trace leaving the axis is
    # indistinguishable from data that stops, and a reader has no way to tell which.
    if line_frequency:
        notes.append(
            f"the trace leaves the axis at each {line_frequency:g} Hz notch: "
            "the stopband is excluded from the limits, not from the data"
        )
    if notes:
        difference_axis.annotate(
            "\n".join(notes),
            xy=(0.5, 1.02),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=GUIDE_COLOR,
        )
    plt.close(figure)
    return figure


def _aperiodic_values(fit: AperiodicFit | None) -> list[object]:
    if fit is None:
        return [None, None]
    return [f"{fit.exponent:.2f}", f"{fit.offset_db:.1f}"]


def spectra_summary_html(spectra: Sequence[RunSpectra]) -> str:
    """Render the aperiodic fits and the across-channel spread, per run."""
    if not spectra:
        raise ValueError("The spectra summary requires at least one run.")
    columns = (
        Column("Run", align=Align.TEXT),
        Column("exponent", group="Before ICA"),
        Column("offset (dB re 1 µV²/Hz)", group="Before ICA"),
        Column("exponent", group="After ICA"),
        Column("offset (dB re 1 µV²/Hz)", group="After ICA"),
        Column("Δ exponent"),
        Column("Worst channel above median (dB)"),
    )
    rows = [
        [
            run_label(run.recording_id),
            *_aperiodic_values(run.before.aperiodic),
            *_aperiodic_values(run.after.aperiodic),
            None if run.exponent_change is None else format(run.exponent_change, "+.2f"),
            f"{run.after.worst_channel_gap_db:.1f}",
        ]
        for run in spectra
    ]
    fit_low, fit_high = DEFAULT_FIT_RANGE_HZ
    return (
        "<p>The aperiodic background is a robust line through the spectrum in log-log "
        f"coordinates over {fit_low:g}-{fit_high:g} Hz, after dropping the bins that sit "
        "in the upper quartile of the residuals so that oscillatory peaks do not tilt "
        "it. The exponent is the tilt and the offset is the level at 1 Hz.</p>"
        + grid_table(columns, rows)
        + "<p>Broadband artifact raises the offset and flattens the exponent; removing "
        "components takes the offset down with it. A large exponent change across ICA "
        "means cleaning altered the background the analysis sits on, not only the "
        "artifact on top of it. The last column is the widest across-channel gap after "
        "cleaning, which locates a focal residual the median trace cannot show.</p>"
    )


def add_spectra_section(
    *,
    report: mne.Report,
    spectra: Sequence[RunSpectra],
    line_frequency: float | None = None,
    marked_frequencies: Sequence[float] = (),
    section: str = "Sensor spectra before and after ICA",
) -> None:
    """Append already-computed per-run spectra to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_ica_component_review,
        drop_replaced_filtered_spectrum,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not spectra:
        raise ValueError("Sensor spectra require at least one filtered run.")
    figures = [
        plot_run_spectra(
            run,
            line_frequency=line_frequency,
            marked_frequencies=marked_frequencies,
        )
        for run in spectra
    ]
    remove_tagged_content(report, tag="sensor-spectra")
    # Dropped beside its replacement, so a report built without this section keeps MNE's
    # full-bandwidth panel rather than losing every view of the filtered spectrum.
    drop_replaced_filtered_spectrum(report)
    report.add_html(
        html=spectra_summary_html(spectra),
        title="Spectral background before and after ICA",
        section=section,
        tags=("raw", "sensor-spectra"),
        replace=True,
    )
    report.add_figure(
        fig=figures,
        title=f"Median sensor spectrum by run — {len(figures)} figures, use the slider",
        caption=[run.recording_id for run in spectra],
        section=section,
        tags=("raw", "sensor-spectra"),
        image_format=report_image_format(is_figure_list=True),
        replace=True,
    )
    move_tagged_content_before(
        report,
        tag="sensor-spectra",
        anchor=before_ica_component_review,
    )


__all__ = [
    "SPREAD_PERCENTILES",
    "WELCH_SECONDS",
    "RunSpectra",
    "StageSpectrum",
    "add_spectra_section",
    "compute_run_spectra",
    "plot_run_spectra",
    "spectra_summary_html",
    "summarize_stage",
]
