"""Per-subject residual scanner-gradient evidence for the subject report.

Gradient correction happens upstream, in BrainVision Analyzer, and the pipeline already
measures what survives it — but only as a cohort figure. A cohort median cannot tell the
person reviewing one subject whether *that* subject's correction worked, and residual
gradient artifact is the failure that most easily passes for data: it is periodic, it is
broadband, and after averaging across channels it disappears into the noise floor.

Two measurements are made here, both per run.

The comb residual works in the frequency domain. Gradient switching repeats once per
volume, so its residual appears as a comb of narrow lines at integer multiples of the
volume rate. Comparing the power in each line against the background *between* the lines
isolates the periodic residual from whatever else occupies the same band, which a plain
spectrum cannot do.

The volume-locked average works in the time domain and is the canonical Allen/Niazy
view. Averaging the recording time-locked to the volume marker cancels everything that
is not phase-locked to the gradient, so what remains is the residual artifact waveform
itself, at the amplitude it actually reaches.

Neither measurement is graded. The volume rate is measured from the markers rather than
configured, so the comb cannot be pointed at the wrong frequencies by a stale setting.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np

from eeg_pipeline.preprocessing.report.annotations import annotation_onsets, onset_events
from eeg_pipeline.preprocessing.report.style import AFTER_COLOR, BEFORE_COLOR, GUIDE_COLOR

#: Annotation written by BrainVision at each scanner volume. Matches the description
#: :mod:`eeg_pipeline.preprocessing.residual_gradient` requires, so both read the same
#: marker rather than two spellings of it.
VOLUME_MARKER_DESCRIPTION = "Volume/V  1"

#: Welch window for the comb. Longer than the sensor-spectra window because the comb
#: has to be resolved *between* its teeth, not merely detected.
COMB_WELCH_SECONDS = 8.0

#: Half-width of the peak window around each harmonic, as a fraction of the harmonic
#: spacing. Wide enough to catch a line that sits between two bins, narrow enough not to
#: reach into the background.
HARMONIC_PEAK_FRACTION = 0.15

#: Band, again as a fraction of the harmonic spacing, whose power defines the local
#: background. It sits between consecutive harmonics, away from both.
BACKGROUND_FRACTION_RANGE = (0.30, 0.50)

#: Frequency bins needed per harmonic spacing before the comb can be separated from its
#: background at all. Below this the peak and background windows would overlap and the
#: excess would be an artifact of the window layout rather than a measurement.
MINIMUM_BINS_PER_HARMONIC = 6.0

#: Volumes needed before a marker train defines a rate worth reporting.
MINIMUM_VOLUMES = 20


@dataclass(frozen=True)
class VolumeTiming:
    """Scanner volume rate measured from the markers in one run."""

    n_volumes: int
    repetition_time_s: float
    #: Largest absolute departure of a single interval from the median, in seconds.
    interval_jitter_s: float

    @property
    def fundamental_hz(self) -> float:
        return 1.0 / self.repetition_time_s

    def harmonics(self, *, fmin: float, fmax: float) -> tuple[float, ...]:
        """Return the comb frequencies falling inside a band."""
        if fmax <= fmin:
            raise ValueError(f"Harmonic band ({fmin}, {fmax}) is empty or reversed.")
        first = int(np.ceil(fmin / self.fundamental_hz))
        last = int(np.floor(fmax / self.fundamental_hz))
        return tuple(float(order * self.fundamental_hz) for order in range(max(first, 1), last + 1))


def measure_volume_timing(
    raw: mne.io.BaseRaw,
    *,
    description: str = VOLUME_MARKER_DESCRIPTION,
) -> VolumeTiming | None:
    """Measure the volume rate from a run's markers.

    Returns ``None`` when the run carries no usable volume marker train, so a recording
    made outside a scanner has no gradient section rather than an empty one.
    """
    onsets = annotation_onsets(raw, description)
    if onsets.size < MINIMUM_VOLUMES:
        return None
    intervals = np.diff(onsets)
    repetition_time = float(np.median(intervals))
    if repetition_time <= 0:
        raise ValueError(
            f"{description!r} markers are not strictly increasing: the median interval "
            f"is {repetition_time:g} s."
        )
    return VolumeTiming(
        n_volumes=int(onsets.size),
        repetition_time_s=repetition_time,
        interval_jitter_s=float(np.max(np.abs(intervals - repetition_time))),
    )


@dataclass(frozen=True)
class CombResidual:
    """Power at the gradient comb against its local background, before and after ICA.

    Every excess array is ``(n_channels, n_harmonics)``. Keeping the channel axis is the
    point: a gradient residual confined to a handful of peripheral sensors is the common
    failure, and any summary that averages over channels first cannot see it.
    """

    recording_id: str
    timing: VolumeTiming
    harmonic_frequencies_hz: np.ndarray
    channel_names: tuple[str, ...]
    before_excess_db: np.ndarray
    after_excess_db: np.ndarray

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def before_typical_db(self) -> np.ndarray:
        """Across-channel median excess at each harmonic."""
        return np.median(self.before_excess_db, axis=0)

    @property
    def after_typical_db(self) -> np.ndarray:
        return np.median(self.after_excess_db, axis=0)

    @property
    def before_worst_db(self) -> np.ndarray:
        """Excess at each harmonic in the channel that carries the most of it."""
        return np.max(self.before_excess_db, axis=0)

    @property
    def after_worst_db(self) -> np.ndarray:
        return np.max(self.after_excess_db, axis=0)

    @property
    def median_before_excess_db(self) -> float:
        return float(np.median(self.before_excess_db))

    @property
    def median_after_excess_db(self) -> float:
        return float(np.median(self.after_excess_db))

    @property
    def worst_channel(self) -> str:
        """Channel with the largest median surviving comb across harmonics."""
        return self.channel_names[int(np.argmax(np.median(self.after_excess_db, axis=1)))]

    @property
    def worst_harmonic_hz(self) -> float:
        """Frequency of the largest surviving comb line, in any channel."""
        _, harmonic = np.unravel_index(
            int(np.argmax(self.after_excess_db)), self.after_excess_db.shape
        )
        return float(self.harmonic_frequencies_hz[harmonic])

    @property
    def worst_excess_db(self) -> float:
        return float(np.max(self.after_excess_db))


def _channel_spectrum(
    raw: mne.io.BaseRaw,
    *,
    fmin: float,
    fmax: float,
    welch_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frequency grid and per-channel power in linear units.

    Channels are kept separate on purpose. Residual gradient artifact is focal — it
    concentrates in the sensors with the largest lead loops — so collapsing to an
    across-channel median before measuring the comb would hide the very recordings this
    section exists to catch.
    """
    segment = int(round(welch_seconds * raw.info["sfreq"]))
    if segment > raw.n_times:
        raise ValueError(
            f"The comb needs {welch_seconds:g} s of data to resolve its background, but "
            f"the run holds {raw.n_times / raw.info['sfreq']:.1f} s."
        )
    spectrum = raw.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks="eeg",
        n_fft=segment,
        n_per_seg=segment,
        n_overlap=segment // 2,
        average="median",
        verbose="ERROR",
    )
    return np.asarray(spectrum.freqs, dtype=float), np.asarray(spectrum.get_data(), dtype=float)


def _excess_db_per_channel(
    frequencies: np.ndarray,
    power: np.ndarray,
    harmonics: np.ndarray,
    *,
    spacing: float,
) -> np.ndarray:
    """Return each channel's comb excess over its own local background, in decibels.

    ``power`` is ``(n_channels, n_frequencies)``. Each channel is measured against its
    own background so that a noisy sensor does not register a comb it does not have:
    the quantity of interest is the *contrast* between the line and the spectrum beside
    it, not the absolute level.
    """
    peak_half_width = HARMONIC_PEAK_FRACTION * spacing
    background_low, background_high = (fraction * spacing for fraction in BACKGROUND_FRACTION_RANGE)

    excess = np.empty((power.shape[0], harmonics.size), dtype=float)
    tiny = np.finfo(float).tiny
    for index, harmonic in enumerate(harmonics):
        offset = np.abs(frequencies - harmonic)
        peak_window = offset <= peak_half_width
        background_window = (offset >= background_low) & (offset <= background_high)
        if not peak_window.any() or not background_window.any():
            raise ValueError(
                f"The {harmonic:.2f} Hz harmonic has no bins in its peak or background "
                "window; the frequency resolution is too coarse for this comb."
            )
        # The peak is the maximum because a line landing between two bins splits its
        # power; the background is the median because it must resist a neighbouring
        # line leaking into the window.
        peaks = np.max(power[:, peak_window], axis=1)
        backgrounds = np.median(power[:, background_window], axis=1)
        excess[:, index] = 10.0 * np.log10(np.maximum(peaks, tiny) / np.maximum(backgrounds, tiny))
    return excess


def compute_comb_residual(
    raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    *,
    timing: VolumeTiming,
    recording_id: str,
    band_hz: tuple[float, float] = (15.0, 90.0),
    welch_seconds: float = COMB_WELCH_SECONDS,
) -> CombResidual | None:
    """Measure the gradient comb against its background, before and after ICA.

    ``cleaned`` is passed in rather than derived here so that a caller measuring several
    things about the same run applies the ICA once.

    Returns ``None`` when the frequency resolution cannot separate the comb from its
    background, which is a property of the volume rate and the run length rather than of
    the data quality, and so is reported as an absent measurement rather than a bad one.
    """
    fmin, fmax = band_hz
    fmax = min(fmax, float(raw.info["sfreq"]) / 2.0 - 1.0)
    spacing = timing.fundamental_hz
    if fmax <= fmin or spacing * welch_seconds < MINIMUM_BINS_PER_HARMONIC:
        return None

    frequencies, before_power = _channel_spectrum(
        raw, fmin=fmin, fmax=fmax, welch_seconds=welch_seconds
    )
    _, after_power = _channel_spectrum(cleaned, fmin=fmin, fmax=fmax, welch_seconds=welch_seconds)

    # Harmonics within a background window of the band edge have no background on one
    # side, so they are dropped rather than measured against a one-sided estimate.
    margin = BACKGROUND_FRACTION_RANGE[1] * spacing
    lowest, highest = frequencies[0] + margin, frequencies[-1] - margin
    if highest <= lowest:
        return None
    harmonics = np.asarray(timing.harmonics(fmin=lowest, fmax=highest), dtype=float)
    if harmonics.size == 0:
        return None

    good = mne.pick_types(raw.info, eeg=True)
    return CombResidual(
        recording_id=recording_id,
        timing=timing,
        harmonic_frequencies_hz=harmonics,
        channel_names=tuple(raw.ch_names[index] for index in good),
        before_excess_db=_excess_db_per_channel(
            frequencies, before_power, harmonics, spacing=spacing
        ),
        after_excess_db=_excess_db_per_channel(
            frequencies, after_power, harmonics, spacing=spacing
        ),
    )


@dataclass(frozen=True)
class VolumeLockedAverage:
    """Gradient-locked average waveform for one run, before and after ICA."""

    recording_id: str
    times_s: np.ndarray
    #: Across-channel RMS of the volume-locked average, in microvolts.
    before_rms_uv: np.ndarray
    after_rms_uv: np.ndarray
    n_volumes: int

    @property
    def before_peak_to_peak_uv(self) -> float:
        return float(np.ptp(self.before_rms_uv))

    @property
    def after_peak_to_peak_uv(self) -> float:
        return float(np.ptp(self.after_rms_uv))


def _locked_average_rms_uv(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    *,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("The volume-locked average needs at least one volume epoch.")
    average = epochs.average().get_data()
    return epochs.times, np.sqrt(np.mean(average**2, axis=0)) * 1e6


def compute_volume_locked_average(
    raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    *,
    timing: VolumeTiming,
    recording_id: str,
    description: str = VOLUME_MARKER_DESCRIPTION,
) -> VolumeLockedAverage | None:
    """Average one run time-locked to the scanner volume marker, before and after ICA.

    Everything not phase-locked to gradient switching averages away, so the surviving
    waveform is the residual artifact at the amplitude it reaches in the data.
    """
    onsets = annotation_onsets(raw, description)
    if onsets.size < MINIMUM_VOLUMES:
        return None

    events = onset_events(raw, onsets)
    # One volume period, minus one sample so consecutive epochs do not overlap.
    tmax = timing.repetition_time_s - 1.0 / float(raw.info["sfreq"])
    times, before = _locked_average_rms_uv(raw, events, tmax=tmax)
    _, after = _locked_average_rms_uv(cleaned, events, tmax=tmax)
    return VolumeLockedAverage(
        recording_id=recording_id,
        times_s=np.asarray(times, dtype=float),
        before_rms_uv=before,
        after_rms_uv=after,
        n_volumes=int(onsets.size),
    )


_COMB_INTRO = (
    "<p>Gradient switching repeats once per scanner volume, so whatever survives "
    "upstream correction appears as a comb of narrow lines at multiples of the volume "
    "rate. Each line is compared with the background measured between it and its "
    "neighbours, which separates periodic gradient residual from everything else "
    "occupying the same band. The volume rate is measured from the markers in each run, "
    "not read from configuration.</p>"
)

_COMB_NOTE = (
    "<p>Excess is the power at a comb line minus the background beside it, measured "
    "within each channel separately: 0 dB means the line is indistinguishable from the "
    "surrounding spectrum. The median and the largest surviving line are reported "
    "together because gradient residual is focal — it concentrates in the sensors with "
    "the largest lead loops — so a median across the montage can sit near zero while "
    "individual channels are unusable. Marker jitter smears the comb across neighbouring "
    "bins, which lowers the measured excess without the residual itself having changed.</p>"
)

_LOCKED_NOTE = (
    "<p>The locked residual is the peak-to-peak amplitude of the volume-locked average, "
    "which is the artifact waveform itself rather than a spectral summary of it.</p>"
)


def _comb_table(combs: Sequence[CombResidual]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(comb.recording_id)}</td>"
        f"<td>{comb.timing.repetition_time_s:.4f}</td>"
        f"<td>{comb.timing.n_volumes}</td>"
        f"<td>{comb.timing.interval_jitter_s * 1e3:.1f}</td>"
        f"<td>{comb.harmonic_frequencies_hz.size}</td>"
        f"<td>{comb.median_before_excess_db:.1f}</td>"
        f"<td>{comb.median_after_excess_db:.1f}</td>"
        f"<td>{comb.worst_excess_db:.1f} at {comb.worst_harmonic_hz:.1f} Hz "
        f"({html.escape(comb.worst_channel)})</td></tr>"
        for comb in combs
    )
    return (
        "<table><thead><tr><th>Run</th><th>TR (s)</th><th>Volumes</th>"
        "<th>Marker jitter (ms)</th><th>Harmonics</th>"
        "<th>Median excess before (dB)</th><th>Median excess after (dB)</th>"
        "<th>Largest surviving line</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _locked_table(averages: Sequence[VolumeLockedAverage]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(locked.recording_id)}</td>"
        f"<td>{locked.n_volumes}</td>"
        f"<td>{locked.before_peak_to_peak_uv:.2f}</td>"
        f"<td>{locked.after_peak_to_peak_uv:.2f}</td></tr>"
        for locked in averages
    )
    return (
        "<table><thead><tr><th>Run</th><th>Volumes</th>"
        "<th>Locked residual before ICA (µV p-p)</th>"
        "<th>Locked residual after ICA (µV p-p)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def scanner_residual_html(
    combs: Sequence[CombResidual],
    averages: Sequence[VolumeLockedAverage],
) -> str:
    """Render whichever of the two gradient measurements the runs supported."""
    sections = []
    if combs:
        sections.append(_COMB_INTRO + _comb_table(combs) + _COMB_NOTE)
    if averages:
        sections.append(_locked_table(averages) + _LOCKED_NOTE)
    if not sections:
        raise ValueError("The gradient section requires at least one measured run.")
    return "".join(sections)


def plot_comb_residual(combs: Sequence[CombResidual]) -> plt.Figure:
    """Plot comb excess against frequency for every run, before and after ICA."""
    if not combs:
        raise ValueError("The comb figure requires at least one measured run.")
    figure, axes = plt.subplots(
        len(combs),
        1,
        figsize=(10.0, 2.6 * len(combs) + 0.8),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    for axis, comb in zip(axes[:, 0], combs, strict=True):
        for typical, worst, color, label in (
            (comb.before_typical_db, comb.before_worst_db, BEFORE_COLOR, "Before ICA"),
            (comb.after_typical_db, comb.after_worst_db, AFTER_COLOR, "After ICA"),
        ):
            axis.plot(
                comb.harmonic_frequencies_hz,
                typical,
                color=color,
                marker="o",
                markersize=2.5,
                linewidth=0.9,
                label=f"{label}, median channel",
            )
            # A residual confined to a few peripheral sensors leaves the median flat, so
            # the worst channel is the trace that actually carries the failure.
            axis.plot(
                comb.harmonic_frequencies_hz,
                worst,
                color=color,
                linewidth=0.8,
                linestyle=":",
                label=f"{label}, worst channel",
            )
        axis.axhline(0.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.0)
        axis.annotate(
            "indistinguishable from background",
            xy=(comb.harmonic_frequencies_hz[-1], 0.0),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            fontsize=6.5,
            color=GUIDE_COLOR,
        )
        axis.set(
            title=(
                f"{comb.recording_id} · TR {comb.timing.repetition_time_s:.4f} s "
                f"({comb.timing.fundamental_hz:.3f} Hz) · median excess "
                f"{comb.median_before_excess_db:.1f} → {comb.median_after_excess_db:.1f} dB · "
                f"worst line {comb.worst_excess_db:.1f} dB at {comb.worst_harmonic_hz:.1f} Hz "
                f"({comb.worst_channel})"
            ),
            ylabel="Excess over\nbackground (dB)",
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("Frequency (Hz) · one point per gradient harmonic")
    plt.close(figure)
    return figure


def plot_volume_locked_average(averages: Sequence[VolumeLockedAverage]) -> plt.Figure:
    """Plot the gradient-locked residual waveform per run, before and after ICA."""
    if not averages:
        raise ValueError("The volume-locked figure requires at least one measured run.")
    figure, axes = plt.subplots(
        1,
        len(averages),
        figsize=(4.2 * len(averages), 3.6),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )
    for axis, locked in zip(axes[0], averages, strict=True):
        axis.plot(
            locked.times_s,
            locked.before_rms_uv,
            color=BEFORE_COLOR,
            linewidth=1.0,
            label="Before ICA",
        )
        axis.plot(
            locked.times_s,
            locked.after_rms_uv,
            color=AFTER_COLOR,
            linewidth=1.0,
            label="After ICA",
        )
        axis.set(
            title=(
                f"{locked.recording_id}\n{locked.n_volumes} volumes · "
                f"{locked.before_peak_to_peak_uv:.2f} → {locked.after_peak_to_peak_uv:.2f} µV p-p"
            ),
            xlabel="Time within volume (s)",
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("RMS across channels of the\nvolume-locked average (µV)")
    axes[0][0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Residual gradient waveform: everything not locked to the volume marker "
        "averages away, so what remains is the artifact itself",
        fontsize=9,
    )
    plt.close(figure)
    return figure


def add_scanner_residual_section(
    *,
    report: mne.Report,
    combs: Sequence[CombResidual],
    averages: Sequence[VolumeLockedAverage],
    section: str = "Residual scanner gradient",
) -> None:
    """Append the per-run gradient residual evidence to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        before_raw_sections,
        move_tagged_content_before,
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not combs and not averages:
        raise ValueError("The gradient section requires at least one measured run.")
    remove_tagged_content(report, tag="scanner-residual")
    report.add_html(
        html=scanner_residual_html(combs, averages),
        title="Gradient residual by run",
        section=section,
        tags=("raw", "scanner-residual"),
        replace=True,
    )
    if combs:
        report.add_figure(
            fig=plot_comb_residual(combs),
            title="Gradient comb against its local background",
            section=section,
            tags=("raw", "scanner-residual"),
            image_format=report_image_format(),
            replace=True,
        )
    if averages:
        report.add_figure(
            fig=plot_volume_locked_average(averages),
            title="Volume-locked residual waveform",
            section=section,
            tags=("raw", "scanner-residual"),
            image_format=report_image_format(),
            replace=True,
        )
    # This describes what upstream correction left behind, so it belongs with the other
    # input-quality evidence ahead of the raw sections.
    move_tagged_content_before(report, tag="scanner-residual", anchor=before_raw_sections)


__all__ = [
    "BACKGROUND_FRACTION_RANGE",
    "COMB_WELCH_SECONDS",
    "HARMONIC_PEAK_FRACTION",
    "MINIMUM_BINS_PER_HARMONIC",
    "MINIMUM_VOLUMES",
    "VOLUME_MARKER_DESCRIPTION",
    "CombResidual",
    "VolumeLockedAverage",
    "VolumeTiming",
    "add_scanner_residual_section",
    "compute_comb_residual",
    "compute_volume_locked_average",
    "measure_volume_timing",
    "plot_comb_residual",
    "plot_volume_locked_average",
    "scanner_residual_html",
]
