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
is not phase-locked to the gradient, so what remains is the residual artifact at the
amplitude it actually reaches. It is reported as the across-channel RMS of that average,
which is an envelope rather than the waveform: rectifying across channels discards
polarity, so the trace carries magnitude over time and not shape.

Neither measurement is graded. The volume rate is measured from the markers rather than
configured, so the comb cannot be pointed at the wrong frequencies by a stale setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.lines import Line2D

from eeg_pipeline.preprocessing.report.annotations import annotation_onsets, onset_events
from eeg_pipeline.preprocessing.report.cohort.noise_floor import measure_locked_average
from eeg_pipeline.preprocessing.report.filtering import in_notch, notch_windows
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

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
    #: Harmonics falling inside a notch stopband, excluded from every statistic below.
    #:
    #: A notch drives its band to the numerical floor, so a comb line that lands in one
    #: measures −25 dB of excess: the filter, reported as though the correction had
    #: removed the artifact. On a 0.9 s repetition time the 54th harmonic sits at 60 Hz,
    #: which made the deepest excursion in the whole figure an artifact of the pipeline's
    #: own notch. Those harmonics are still drawn, hollow, because a gap in the comb is
    #: itself worth seeing; they are simply not scored.
    notched: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __post_init__(self) -> None:
        if self.notched.size == 0:
            object.__setattr__(
                self, "notched", np.zeros(self.harmonic_frequencies_hz.size, dtype=bool)
            )
        if self.notched.shape != self.harmonic_frequencies_hz.shape:
            raise ValueError(
                f"The notch mask describes {self.notched.size} harmonic(s) but the comb "
                f"has {self.harmonic_frequencies_hz.size}."
            )

    @property
    def scored(self) -> np.ndarray:
        """Mask of the harmonics that carry a measurement rather than a filter."""
        return ~self.notched

    @property
    def has_scored_harmonics(self) -> bool:
        return bool(self.scored.any())

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
        return float(np.median(self.before_excess_db[:, self.scored]))

    @property
    def median_after_excess_db(self) -> float:
        return float(np.median(self.after_excess_db[:, self.scored]))

    @property
    def worst_channel(self) -> str:
        """Channel with the largest median surviving comb across scored harmonics."""
        scored = self.after_excess_db[:, self.scored]
        return self.channel_names[int(np.argmax(np.median(scored, axis=1)))]

    @property
    def worst_harmonic_hz(self) -> float:
        """Frequency of the largest surviving comb line, in any channel."""
        return float(self.harmonic_frequencies_hz[self.scored][self._worst_index[1]])

    @property
    def worst_excess_db(self) -> float:
        scored = self.after_excess_db[:, self.scored]
        return float(scored[self._worst_index])

    @property
    def _worst_index(self) -> tuple[int, int]:
        """Index of the largest surviving line, within the scored harmonics."""
        scored = self.after_excess_db[:, self.scored]
        channel, harmonic = np.unravel_index(int(np.argmax(scored)), scored.shape)
        return int(channel), int(harmonic)


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
    line_frequency: float | None = None,
) -> CombResidual | None:
    """Measure the gradient comb against its background, before and after ICA.

    ``cleaned`` is passed in rather than derived here so that a caller measuring several
    things about the same run applies the ICA once.

    ``line_frequency`` is the configured notch. Harmonics landing in its stopband are
    marked and excluded from the reported statistics, because in that band the spectrum
    describes the filter and not the correction. Left unset, every harmonic is scored,
    which is correct for a recording that was never notched.

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

    notched = in_notch(harmonics, notch_windows(line_frequency, fmax=float(harmonics[-1])))
    if notched.all():
        # Every line sits in a stopband, so there is no comb left to measure. Reported as
        # an absent measurement rather than a table of filter depths.
        return None

    good = mne.pick_types(raw.info, eeg=True)
    return CombResidual(
        recording_id=recording_id,
        timing=timing,
        harmonic_frequencies_hz=harmonics,
        notched=notched,
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
    #: Locked amplitude with the averaging noise floor removed, in microvolts.
    #:
    #: The waveform above carries a floor of sigma / sqrt(n_volumes), so its amplitude
    #: falls as a session lengthens whether or not the correction improved. These are the
    #: same measurement with that floor subtracted, which makes them comparable between
    #: runs of different lengths and between participants. See
    #: :mod:`eeg_pipeline.preprocessing.report.cohort.noise_floor`.
    before_amplitude_uv: float | None = None
    after_amplitude_uv: float | None = None
    #: The floor itself, so the panel can say how much of the waveform above is it.
    before_noise_floor_uv: float | None = None
    after_noise_floor_uv: float | None = None

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
    """Average time-locked to the volume marker and reduce to an across-channel RMS.

    Each epoch has its own mean removed. Gradient switching runs continuously, so there
    is no artifact-free interval inside a volume period to use as a baseline in the usual
    sense; what the whole-epoch mean removes is the level each channel happens to sit at,
    which is not part of the volume-locked waveform but does enter a peak-to-peak taken
    on a non-negative RMS trace. On sub-0015 it accounted for roughly 40% of every
    reported figure — run-1 fell from 1.08 to 0.66 µV p-p and run-2 from 0.97 to 0.54 —
    so the column was reporting the level and the waveform together under the waveform's
    name.

    Removing a constant per epoch cannot change what is phase-locked to the marker, which
    is why this is a correction to the measurement rather than to the artifact.
    """
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=tmax,
        baseline=(None, None),
        picks="eeg",
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise ValueError("The volume-locked average needs at least one volume epoch.")
    # ``copy=False`` because the epochs are preloaded and this only reads them: a copy
    # here is a few hundred megabytes to produce an array identical to the one beside it.
    measured = measure_locked_average(epochs.get_data(copy=False))
    return (
        epochs.times,
        np.sqrt(np.mean(measured.average**2, axis=0)) * 1e6,
        measured,
    )


def compute_volume_locked_average(
    raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    *,
    timing: VolumeTiming,
    recording_id: str,
    description: str = VOLUME_MARKER_DESCRIPTION,
) -> VolumeLockedAverage | None:
    """Average one run time-locked to the scanner volume marker, before and after ICA.

    Everything not phase-locked to gradient switching averages away, so what survives is
    the residual artifact at the amplitude it reaches in the data. It is reduced to the
    across-channel RMS, which is an envelope: the rectification discards polarity, so a
    channel whose residual opposes its neighbours' raises the trace just as one that
    agrees with them does.
    """
    onsets = annotation_onsets(raw, description)
    if onsets.size < MINIMUM_VOLUMES:
        return None

    events = onset_events(raw, onsets)
    # One volume period, minus one sample so consecutive epochs do not overlap.
    tmax = timing.repetition_time_s - 1.0 / float(raw.info["sfreq"])
    times, before, before_measured = _locked_average_rms_uv(raw, events, tmax=tmax)
    _, after, after_measured = _locked_average_rms_uv(cleaned, events, tmax=tmax)
    return VolumeLockedAverage(
        recording_id=recording_id,
        times_s=np.asarray(times, dtype=float),
        before_rms_uv=before,
        after_rms_uv=after,
        n_volumes=int(onsets.size),
        before_amplitude_uv=before_measured.corrected_amplitude_uv,
        after_amplitude_uv=after_measured.corrected_amplitude_uv,
        before_noise_floor_uv=before_measured.noise_floor_uv,
        after_noise_floor_uv=after_measured.noise_floor_uv,
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

#: Title of the volume-locked panel.
#:
#: "Envelope", not "waveform". What is plotted is the across-channel RMS of the
#: volume-locked average, which is non-negative by construction: rectifying across
#: channels discards polarity, so the trace has a magnitude over time and no shape in the
#: sense a waveform does. The previous title, and the note beneath it, invited reading
#: excursions as deflections with a direction.
VOLUME_LOCKED_TITLE = "Volume-locked residual envelope"


def volume_locked_note_html() -> str:
    """Explain what the volume-locked figures measure, and what they do not."""
    return (
        "<p>The locked residual is the peak-to-peak range of the across-channel "
        "<abbr title='root mean square'>RMS</abbr> of the volume-locked average. "
        "Averaging on the volume marker keeps whatever repeats at the volume rate and "
        "averages away what does not, so this is the residual artifact at the amplitude "
        "it reaches in the data rather than a spectral summary of it.</p>"
        "<p>It is an envelope, not the artifact waveform: the RMS across channels is "
        "non-negative, so the trace carries magnitude over time and not polarity. A "
        "channel whose residual is large but opposite in sign to its neighbours' raises "
        "this trace exactly as one that agrees with them.</p>"
        "<p>Each volume epoch has its own mean removed before averaging. Gradient "
        "switching never stops, so there is no artifact-free interval inside a volume "
        "period to baseline against; what is removed is the level each channel sits at, "
        "which is not part of the volume-locked waveform but does enter a peak-to-peak "
        "taken on a non-negative trace.</p>"
        "<p>The after figure is not guaranteed to be the smaller of the two, and on some "
        "recordings it is not. ICA is fitted to maximise independence over the whole "
        "recording, not to minimise what repeats at the volume rate, so a decomposition "
        "can remove a great deal of sensor variance and leave more volume-locked residual "
        "than it started with. A run whose locked residual rises across ICA is reporting "
        "that, not a fault in the measurement, and is worth reading beside the comb table "
        "above rather than on its own.</p>"
    )


_LOCKED_NOTE = volume_locked_note_html()


def _comb_table(combs: Sequence[CombResidual]) -> str:
    columns = (
        Column("Run", align=Align.TEXT),
        Column("TR (s)"),
        Column("Volumes"),
        Column("Marker jitter (ms)"),
        Column("Harmonics"),
        Column("Median excess before (dB)"),
        Column("Median excess after (dB)"),
        Column("Largest surviving line", align=Align.TEXT),
    )
    rows = [
        [
            run_label(comb.recording_id),
            f"{comb.timing.repetition_time_s:.4f}",
            comb.timing.n_volumes,
            f"{comb.timing.interval_jitter_s * 1e3:.1f}",
            comb.harmonic_frequencies_hz.size,
            f"{comb.median_before_excess_db:.1f}",
            f"{comb.median_after_excess_db:.1f}",
            f"{comb.worst_excess_db:.1f} at {comb.worst_harmonic_hz:.1f} Hz "
            f"({comb.worst_channel})",
        ]
        for comb in combs
    ]
    return grid_table(columns, rows)


def _locked_table(averages: Sequence[VolumeLockedAverage]) -> str:
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Volumes"),
        Column("Locked residual before ICA (µV p-p)"),
        Column("Locked residual after ICA (µV p-p)"),
    )
    rows = [
        [
            run_label(locked.recording_id),
            locked.n_volumes,
            f"{locked.before_peak_to_peak_uv:.2f}",
            f"{locked.after_peak_to_peak_uv:.2f}",
        ]
        for locked in averages
    ]
    return grid_table(columns, rows)


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


def _comb_run_label(recording_id: str) -> str:
    """Return the run-identifying tail of a recording id, or the id when it has none."""
    return run_label(recording_id)


def plot_comb_residual(combs: Sequence[CombResidual]) -> plt.Figure:
    """Plot comb excess against frequency for every run, before and after ICA.

    The runs are laid out as a grid on one shared pair of axes rather than as a column
    of independently scaled panels. A session's runs differ from each other by a
    fraction of a decibel, so a stack in which every panel repeated the subject, task
    and repetition time in its title and then chose its own y limits made the runs look
    both more distinct and less comparable than they are.
    """
    if not combs:
        raise ValueError("The comb figure requires at least one measured run.")
    columns = 2 if len(combs) > 3 else 1
    rows = math.ceil(len(combs) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.6 * columns, 2.5 * rows + 1.0),
        squeeze=False,
        sharex=True,
        # Shared limits are the point of the grid: an eye moving between panels should
        # be comparing residuals, not silently recalibrating to each panel's own scale.
        sharey=True,
        layout="constrained",
    )
    flat = axes.ravel()
    for axis, comb in zip(flat, combs, strict=False):
        for typical, worst, color, label in (
            (comb.before_typical_db, comb.before_worst_db, BEFORE_COLOR, "Before ICA"),
            (comb.after_typical_db, comb.after_worst_db, AFTER_COLOR, "After ICA"),
        ):
            axis.plot(
                comb.harmonic_frequencies_hz,
                typical,
                color=color,
                linewidth=0.9,
                label=f"{label}, median channel",
            )
            # Filled markers on the harmonics that carry a measurement, hollow on the
            # ones inside the notch stopband. Drawing them alike let the notch's −25 dB
            # trough read as the deepest correction in the figure.
            for mask, facecolor in ((comb.scored, color), (comb.notched, "white")):
                if not mask.any():
                    continue
                axis.plot(
                    comb.harmonic_frequencies_hz[mask],
                    typical[mask],
                    linestyle="none",
                    marker="o",
                    markersize=2.5,
                    markerfacecolor=facecolor,
                    markeredgecolor=color,
                    markeredgewidth=0.6,
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
        # The zero line is explained in the legend rather than by a caption pinned to
        # the line itself, which landed on top of the traces in every panel.
        axis.axhline(
            0.0,
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="0 dB: indistinguishable from background",
        )
        axis.set_title(
            f"{_comb_run_label(comb.recording_id)} · median "
            f"{comb.median_before_excess_db:.1f} → {comb.median_after_excess_db:.1f} dB · "
            f"worst {comb.worst_excess_db:.1f} dB at {comb.worst_harmonic_hz:.1f} Hz "
            f"({comb.worst_channel})",
            fontsize=8,
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    # Limits from the scored harmonics alone. A notch drives its harmonic tens of
    # decibels below background, and that trough -- the pipeline's own filter, excluded
    # from every reported statistic -- was setting the scale for the residual it is not
    # part of: on sub-0012 a -25 dB stopband compressed the ±10 dB the comb lives in
    # into the top fifth of each panel. The trough stays drawn and now runs off the axis,
    # which is the honest picture of a value this figure does not measure.
    scored_values = np.concatenate(
        [
            trace[comb.scored]
            for comb in combs
            for trace in (
                comb.before_typical_db,
                comb.after_typical_db,
                comb.before_worst_db,
                comb.after_worst_db,
            )
        ]
    )
    finite = scored_values[np.isfinite(scored_values)]
    if finite.size:
        margin = max(0.05 * float(np.ptp(finite)), 1.0)
        flat[0].set_ylim(float(finite.min()) - margin, float(finite.max()) + margin)
    for axis in flat[len(combs) :]:
        axis.remove()
    for row in range(rows):
        flat[row * columns].set_ylabel("Excess over\nbackground (dB)")
    # The bottom-most surviving panel in each column carries the frequency axis. When
    # the last row is short, ``sharex`` has already hidden the tick labels of the panel
    # above the removed slot, so they are turned back on explicitly.
    for column in range(columns):
        present = [index for index in range(len(combs)) if index % columns == column]
        if not present:
            continue
        axis = flat[present[-1]]
        axis.set_xlabel("Frequency (Hz) · one point per gradient harmonic")
        axis.tick_params(axis="x", labelbottom=True)

    # Repetition time belongs to the acquisition, not to a run, so it is stated once.
    # A session whose runs disagree about it is a finding in itself and is named as one.
    repetition_times = {round(comb.timing.repetition_time_s, 6) for comb in combs}
    if len(repetition_times) == 1:
        timing = combs[0].timing
        timing_text = (
            f"TR {timing.repetition_time_s:.4f} s ({timing.fundamental_hz:.3f} Hz fundamental)"
        )
    else:
        timing_text = "runs differ in repetition time: " + ", ".join(
            f"{_comb_run_label(comb.recording_id)} {comb.timing.repetition_time_s:.4f} s"
            for comb in combs
        )
    figure.suptitle(f"Gradient comb against its local background · {timing_text}", fontsize=10)
    handles, labels = flat[0].get_legend_handles_labels()
    # The hollow marker is a legend concept, so it is explained in the legend rather than
    # in a footnote below it. As free-floating figure text the explanation sat in
    # coordinates ``constrained_layout`` never reads, and printed through this very
    # legend; as an entry it also puts the symbol beside the sentence describing it.
    if any(comb.notched.any() for comb in combs):
        notched_count = max(int(comb.notched.sum()) for comb in combs)
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=2.5,
                markerfacecolor="white",
                markeredgecolor=GUIDE_COLOR,
                markeredgewidth=0.6,
            )
        )
        labels.append(
            f"inside the notch stopband: {notched_count} harmonic(s) drawn but not scored"
        )
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(labels), 3),
        frameon=False,
        fontsize=7,
    )
    plt.close(figure)
    return figure


def plot_volume_locked_average(averages: Sequence[VolumeLockedAverage]) -> plt.Figure:
    """Plot the gradient-locked residual envelope per run, before and after ICA.

    Laid out as a grid on the same rule as :func:`plot_comb_residual`. A row of one panel
    per run put six panels across 1814 points, four times the width of the report column
    the figure sits in, so the browser scaled or clipped it and each waveform ended up a
    couple of hundred pixels wide.
    """
    if not averages:
        raise ValueError("The volume-locked figure requires at least one measured run.")
    columns = 2 if len(averages) > 3 else 1
    rows = math.ceil(len(averages) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.6 * columns, 2.5 * rows + 1.0),
        squeeze=False,
        sharex=True,
        # The runs are being compared against each other, so they cannot each rescale.
        sharey=True,
        layout="constrained",
    )
    flat = axes.ravel()
    for axis, locked in zip(flat, averages, strict=False):
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
        # The floor this trace has to clear to be artifact rather than averaging noise.
        # Averaging n_volumes epochs suppresses everything not locked to the marker by
        # sqrt(n) and no further, so the residual sits on a floor of sigma / sqrt(n) that
        # the odd-even split measures exactly. Without it drawn, a reported 0.17 µV and a
        # floor of 0.15 µV are the same picture.
        if locked.after_noise_floor_uv is not None:
            axis.axhline(
                locked.after_noise_floor_uv,
                color=GUIDE_COLOR,
                linestyle="--",
                linewidth=1.0,
                label="Averaging noise floor (odd–even split)",
            )
        title = (
            f"{_comb_run_label(locked.recording_id)} · {locked.n_volumes} volumes · "
            f"{locked.before_peak_to_peak_uv:.2f} → {locked.after_peak_to_peak_uv:.2f} µV p-p"
        )
        if locked.after_noise_floor_uv is not None:
            title += f" · floor {locked.after_noise_floor_uv:.2f} µV"
        axis.set_title(title, fontsize=8)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in flat[len(averages) :]:
        axis.remove()
    for row in range(rows):
        flat[row * columns].set_ylabel("RMS across channels of the\nvolume-locked average (µV)")
    # The bottom-most surviving panel in each column carries the time axis. When the last
    # row is short, ``sharex`` has already hidden the tick labels of the panel above the
    # removed slot, so they are turned back on explicitly.
    for column in range(columns):
        present = [index for index in range(len(averages)) if index % columns == column]
        if not present:
            continue
        axis = flat[present[-1]]
        axis.set_xlabel("Time within volume (s)")
        axis.tick_params(axis="x", labelbottom=True)
    handles, labels = flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(len(labels), 3),
        frameon=False,
        fontsize=7,
    )
    figure.suptitle(
        "Residual gradient envelope: everything not locked to the volume marker averages "
        "down by √n, so what clears the dashed floor is locked to the marker",
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
            title=VOLUME_LOCKED_TITLE,
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
    "VOLUME_LOCKED_TITLE",
    "VOLUME_MARKER_DESCRIPTION",
    "CombResidual",
    "VolumeLockedAverage",
    "VolumeTiming",
    "add_scanner_residual_section",
    "compute_comb_residual",
    "compute_volume_locked_average",
    "measure_volume_timing",
    "plot_comb_residual",
    "volume_locked_note_html",
    "plot_volume_locked_average",
    "scanner_residual_html",
]
