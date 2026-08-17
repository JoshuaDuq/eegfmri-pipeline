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
from matplotlib.ticker import MaxNLocator

from eeg_pipeline.preprocessing.report.annotations import annotation_onsets, onset_events
from eeg_pipeline.preprocessing.report.cohort.noise_floor import measure_locked_average
from eeg_pipeline.preprocessing.report.filtering import (
    NOTCH_EXCLUSION_HALF_WIDTH_HZ,
    in_notch,
    notch_windows,
)
from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    GUIDE_COLOR,
    run_label,
)
from eeg_pipeline.preprocessing.report.tables import Align, Column, grid_table

#: Annotation written by BrainVision at each scanner volume. Spelled exactly as the
#: recording carries it, so a run whose markers were never sanitized is not silently
#: read as having no volumes.
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
        """Channelwise maximum excess at each harmonic."""
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
        """Channel carrying :attr:`worst_excess_db`."""
        return self.channel_names[self._worst_index[0]]

    @property
    def persistent_worst_channel(self) -> str:
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


@dataclass(frozen=True)
class CombNotMeasured:
    """A run the comb measurement declined, and why.

    Returned instead of ``None`` because the four reasons below are not interchangeable
    and none of them is a property of the data quality. A run dropped without a reason
    leaves a table holding only the runs that still had a comb, which reads as though the
    absent ones had been measured and found clean -- on sub-0001 that described four runs
    of six, all of them with a residual the section never showed.
    """

    recording_id: str
    #: Sentence fragment completing "not measured: ...".
    reason: str
    #: Harmonics the band contained, and how many sat inside a stopband. Zero where the
    #: run was declined before the harmonics were known.
    n_harmonics: int = 0
    n_notched: int = 0


def compute_comb_residual(
    raw: mne.io.BaseRaw,
    cleaned: mne.io.BaseRaw,
    *,
    timing: VolumeTiming,
    recording_id: str,
    band_hz: tuple[float, float] = (15.0, 90.0),
    welch_seconds: float = COMB_WELCH_SECONDS,
    line_frequency: float | None = None,
    notch_half_width_hz: float = NOTCH_EXCLUSION_HALF_WIDTH_HZ,
    unavailable_intervals: Sequence[tuple[float, float]] | None = None,
) -> CombResidual | CombNotMeasured:
    """Measure the gradient comb against its background, before and after ICA.

    ``cleaned`` is passed in rather than derived here so that a caller measuring several
    things about the same run applies the ICA once.

    ``line_frequency`` is the configured notch. Harmonics landing in its stopband are
    marked and excluded from the reported statistics, because in that band the spectrum
    describes the filter and not the correction. Left unset, every harmonic is scored,
    which is correct for a recording that was never notched.

    Returns a :class:`CombNotMeasured` naming the reason where the measurement cannot be
    made. Each reason is a property of the volume rate, the run length or the upstream
    filtering rather than of the data quality, so it is reported as an absent measurement
    rather than a bad one -- and reported rather than dropped, so a reader can tell an
    absent measurement from a clean run.
    """
    fmin, fmax = band_hz
    fmax = min(fmax, float(raw.info["sfreq"]) / 2.0 - 1.0)
    spacing = timing.fundamental_hz
    if fmax <= fmin or spacing * welch_seconds < MINIMUM_BINS_PER_HARMONIC:
        return CombNotMeasured(
            recording_id=recording_id,
            reason=(
                f"the frequency resolution of a {welch_seconds:g} s window cannot separate "
                f"harmonics {spacing:.3f} Hz apart from the background between them"
            ),
        )

    frequencies, before_power = _channel_spectrum(
        raw, fmin=fmin, fmax=fmax, welch_seconds=welch_seconds
    )
    _, after_power = _channel_spectrum(cleaned, fmin=fmin, fmax=fmax, welch_seconds=welch_seconds)

    # Harmonics within a background window of the band edge have no background on one
    # side, so they are dropped rather than measured against a one-sided estimate.
    margin = BACKGROUND_FRACTION_RANGE[1] * spacing
    lowest, highest = frequencies[0] + margin, frequencies[-1] - margin
    if highest <= lowest:
        return CombNotMeasured(
            recording_id=recording_id,
            reason=(
                f"the {fmin:g}–{fmax:g} Hz band is narrower than the background windows "
                "each harmonic has to be compared against"
            ),
        )
    harmonics = np.asarray(timing.harmonics(fmin=lowest, fmax=highest), dtype=float)
    if harmonics.size == 0:
        return CombNotMeasured(
            recording_id=recording_id,
            reason=f"no multiple of the {spacing:.3f} Hz volume rate falls inside the band",
        )

    notched = in_notch(
        harmonics,
        notch_windows(
            line_frequency,
            fmax=float(harmonics[-1]),
            half_width=notch_half_width_hz,
            unavailable_intervals=unavailable_intervals,
        ),
    )
    if notched.all():
        # Every line sits in a stopband, so there is no comb left to measure. Reported as
        # an absent measurement rather than a table of filter depths -- and named, because
        # the run is not clean, it is unmeasurable on this grid. What survives upstream
        # removal of every k/TR line repeats over two volumes rather than one, so the
        # volume-locked panel beside this one is the evidence that remains.
        return CombNotMeasured(
            recording_id=recording_id,
            reason=(
                f"all {harmonics.size} harmonic(s) in the band sit inside an upstream "
                "stopband, so no comb line survives to score"
            ),
            n_harmonics=int(harmonics.size),
            n_notched=int(notched.sum()),
        )

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
    #: Across-channel, across-latency RMS and the floor it was measured against.
    before_locked_rms_uv: float | None = None
    after_locked_rms_uv: float | None = None
    before_noise_floor_uv: float | None = None
    after_noise_floor_uv: float | None = None
    #: Signed floor-adjusted power. Negative means unresolved, not zero artifact.
    before_excess_power_uv2: float | None = None
    after_excess_power_uv2: float | None = None
    #: Correlation between the odd-epoch and even-epoch averages the floor was taken from.
    #: See :attr:`LockedAverage.half_correlation`; carried here so the table can report the
    #: condition the floor was measured under beside the floor itself.
    before_half_correlation: float | None = None
    after_half_correlation: float | None = None

    @property
    def before_is_resolved(self) -> bool:
        return self.before_excess_power_uv2 is not None and self.before_excess_power_uv2 > 0.0

    @property
    def after_is_resolved(self) -> bool:
        return self.after_excess_power_uv2 is not None and self.after_excess_power_uv2 > 0.0

    @property
    def before_resolved_amplitude_uv(self) -> float | None:
        if not self.before_is_resolved:
            return None
        return float(np.sqrt(self.before_excess_power_uv2))

    @property
    def after_resolved_amplitude_uv(self) -> float | None:
        if not self.after_is_resolved:
            return None
        return float(np.sqrt(self.after_excess_power_uv2))

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
        before_locked_rms_uv=before_measured.locked_rms_uv,
        after_locked_rms_uv=after_measured.locked_rms_uv,
        before_noise_floor_uv=before_measured.noise_floor_uv,
        after_noise_floor_uv=after_measured.noise_floor_uv,
        before_excess_power_uv2=before_measured.excess_power_uv2,
        after_excess_power_uv2=after_measured.excess_power_uv2,
        before_half_correlation=before_measured.half_correlation,
        after_half_correlation=after_measured.half_correlation,
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
    "within each channel separately: 0 dB means the peak-window maximum equals the "
    "background-window median. It is not a calibrated statistical null. The median and "
    "the largest surviving line are reported "
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
        "<p>Observed locked RMS is the root mean square over channels and latencies of "
        "the volume-locked average. It still contains finite-average noise. The "
        "odd–even split estimates that noise floor, and signed excess power is observed "
        "RMS squared minus floor squared. Only a positive excess supports a resolved "
        "floor-adjusted amplitude; a negative value is reported as unresolved rather "
        "than clipped to zero.</p>"
        "<p>Halves agree is the correlation between the odd-epoch and even-epoch averages "
        "the floor was taken from, and it says which of two very different situations an "
        "unresolved row describes. The split estimates noise only where the locked "
        "waveform <em>cancels</em> between the halves, which requires that it be the same "
        "waveform in both. Near +1 it is, and the floor is what it claims to be; near 0 "
        "there is no locked waveform for the halves to share. Near −1 the halves are "
        "mirror images, so the waveform cancels in the average and doubles in the "
        "difference: the reported floor is then the residual itself and the excess is "
        "negative by construction. That is what a residual repeating over two volume "
        "periods rather than one looks like, which is what removing every integer harmonic "
        "of the volume rate upstream leaves behind — and it makes an unresolved row mean "
        "the opposite of an absent artifact. No threshold is applied to this column.</p>"
        "<p>It is an envelope, not the artifact waveform: the RMS across channels is "
        "non-negative, so the trace carries magnitude over time and not polarity. A "
        "channel whose residual is large but opposite in sign to its neighbours' raises "
        "this trace exactly as one that agrees with them. Its peak-to-peak range describes "
        "the envelope's variation; it is not the floor-adjusted residual amplitude.</p>"
        "<p>Each volume epoch has its own mean removed before averaging. Gradient "
        "switching never stops, so there is no artifact-free interval inside a volume "
        "period to baseline against; what is removed is the level each channel sits at, "
        "which is not part of the volume-locked waveform but does enter a peak-to-peak "
        "taken on a non-negative trace.</p>"
        "<p>The after figure is not guaranteed to be the smaller of the two, and on some "
        "recordings it is not. ICA is fitted to maximise independence over the whole "
        "recording, not to minimise what repeats at the volume rate, so the resolved "
        "amplitude can rise across ICA. That result is worth reading beside the comb "
        "table above rather than on its own.</p>"
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


def _locked_stage_measurements(
    locked: VolumeLockedAverage,
) -> tuple[tuple[float, float, float, float | None], ...]:
    """Return complete before/after locked estimates or fail on an incomplete record."""
    values = (
        (
            locked.before_locked_rms_uv,
            locked.before_noise_floor_uv,
            locked.before_excess_power_uv2,
            locked.before_resolved_amplitude_uv,
        ),
        (
            locked.after_locked_rms_uv,
            locked.after_noise_floor_uv,
            locked.after_excess_power_uv2,
            locked.after_resolved_amplitude_uv,
        ),
    )
    if any(value is None for stage in values for value in stage[:3]):
        raise ValueError(
            f"{locked.recording_id} has an incomplete volume-locked noise-floor estimate."
        )
    return tuple(
        (float(observed), float(floor), float(excess), resolved)
        for observed, floor, excess, resolved in values
    )


def _locked_table(averages: Sequence[VolumeLockedAverage]) -> str:
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Stage", align=Align.TEXT),
        Column("Volumes"),
        Column("Observed locked RMS (µV)"),
        Column("Noise floor (µV)"),
        Column("Signed excess power (µV²)"),
        Column("Halves agree (r)", align=Align.TEXT),
        Column("Floor-adjusted amplitude (µV)", align=Align.TEXT),
    )
    rows = []
    for locked in averages:
        before, after = _locked_stage_measurements(locked)
        correlations = (locked.before_half_correlation, locked.after_half_correlation)
        for (stage, observed_rms, noise_floor, excess_power, resolved_amplitude), agreement in zip(
            (("Before ICA", *before), ("After ICA", *after)),
            correlations,
            strict=True,
        ):
            amplitude = "unresolved" if resolved_amplitude is None else f"{resolved_amplitude:.2f}"
            rows.append(
                [
                    run_label(locked.recording_id),
                    stage,
                    locked.n_volumes,
                    f"{observed_rms:.2f}",
                    f"{noise_floor:.2f}",
                    f"{excess_power:+.3f}",
                    "—" if agreement is None or not np.isfinite(agreement) else f"{agreement:+.2f}",
                    amplitude,
                ]
            )
    return grid_table(columns, rows)


def _declined_table(declined: Sequence[CombNotMeasured]) -> str:
    """Account for the runs the comb measurement could not be made on."""
    columns = (
        Column("Run", align=Align.TEXT),
        Column("Harmonics in band"),
        Column("Inside a stopband"),
        Column("Not measured because", align=Align.TEXT),
    )
    rows = [
        [
            run_label(item.recording_id),
            item.n_harmonics if item.n_harmonics else "—",
            item.n_notched if item.n_harmonics else "—",
            item.reason,
        ]
        for item in declined
    ]
    return (
        "<p>These runs carry volume markers but no comb measurement. They are listed "
        "because a run missing from the table above has not been measured and found "
        "clean — it has not been measured. Where the reason is upstream filtering, the "
        "residual that survives it no longer repeats once per volume, so the "
        "volume-locked table below is where the evidence for these runs is.</p>"
        + grid_table(columns, rows)
    )


def scanner_residual_html(
    combs: Sequence[CombResidual],
    averages: Sequence[VolumeLockedAverage],
    declined: Sequence[CombNotMeasured] = (),
) -> str:
    """Render whichever of the two gradient measurements the runs supported."""
    sections = []
    if combs:
        sections.append(_COMB_INTRO + _comb_table(combs) + _COMB_NOTE)
    if declined:
        sections.append(_declined_table(declined))
    if averages:
        sections.append(_locked_table(averages) + _LOCKED_NOTE)
    if not sections:
        raise ValueError("The gradient section requires at least one measured run.")
    return "".join(sections)


def _comb_run_label(recording_id: str) -> str:
    """Return the run-identifying tail of a recording id, or the id when it has none."""
    return run_label(recording_id)


#: Width of one character at a given font size, as a fraction of that size. Matplotlib's
#: default sans face averages close to this over lower-case prose, which is what these
#: labels are.
_CHARACTER_WIDTH_RATIO = 0.62

#: Points a legend entry spends on its handle and the padding either side of it.
_LEGEND_HANDLE_POINTS = 34.0


def legend_columns(figure: plt.Figure, labels: Sequence[str], *, fontsize: float) -> int:
    """Columns that keep the widest legend entry inside ``figure``.

    A fixed column count sets the legend's width from the number of entries and ignores
    the figure it has to fit in. On a single-column panel layout that ran the first and
    last of six entries off opposite edges, cut mid-word, so the key explaining which
    trace was which could not be read at all.

    Estimated from the label text rather than measured from a render, because the figure
    is built without a canvas and drawing one here to place a legend would pay a full
    render on every panel. The estimate only has to be good enough to choose between one,
    two and three columns.
    """
    if not labels:
        return 1
    widest = max(len(label) for label in labels)
    entry_points = widest * fontsize * _CHARACTER_WIDTH_RATIO + _LEGEND_HANDLE_POINTS
    figure_points = figure.get_figwidth() * 72.0
    return max(1, min(len(labels), int(figure_points // entry_points)))


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
        # The trailing inches are the legend's, not the panels'. Six entries that no
        # longer fit on one line need two or three rows beneath the axes, and taking that
        # space out of the panels instead pushed the tick labels of vertically adjacent
        # panels into each other and into the y-axis label.
        figsize=(5.6 * columns, 2.5 * rows + 1.8),
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
                label=label,
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
            # A residual confined to a few peripheral sensors leaves the median flat. This
            # is an envelope whose contributing channel may change between harmonics, not
            # the trace of one sensor.
            axis.plot(
                comb.harmonic_frequencies_hz,
                worst,
                color=color,
                linewidth=0.8,
                linestyle=":",
            )
        # The zero line is explained in the legend rather than by a caption pinned to
        # the line itself, which landed on top of the traces in every panel.
        axis.axhline(
            0.0,
            color=GUIDE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="0 dB: peak equals background",
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
    # Few enough ticks that the bottom label of one panel and the top label of the panel
    # below it cannot meet. Stacked panels share an edge, so the default density puts two
    # numbers within a few points of each other and both become hard to read.
    flat[0].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    for axis in flat[len(combs) :]:
        axis.remove()
    for row in range(rows):
        # Sized with the panel titles rather than at the default. Rotated, two lines of
        # default-size text stand almost as tall as a panel, so the label of one row
        # reached the bottom tick label of the row above it.
        flat[row * columns].set_ylabel("Excess over\nbackground (dB)", fontsize=8)
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
    # Colour carries the stage and line style carries the statistic, so the legend states
    # each of those once instead of spelling out all four combinations. As four sentences
    # the widest entry was 39 characters, which no arrangement fits across a single-column
    # panel layout: the key ran off both edges of the figure, cut mid-word.
    for linestyle, linewidth, name in (
        ("-", 0.9, "median channel"),
        (":", 0.8, "channelwise maximum envelope"),
    ):
        handles.append(
            Line2D([], [], color=GUIDE_COLOR, linestyle=linestyle, linewidth=linewidth)
        )
        labels.append(name)
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
            f"notch stopband: {notched_count} drawn, not scored"
        )
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=legend_columns(figure, labels, fontsize=7),
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
        before, after = _locked_stage_measurements(locked)
        for values, floor, color, label in (
            (locked.before_rms_uv, before[1], BEFORE_COLOR, "Before ICA"),
            (locked.after_rms_uv, after[1], AFTER_COLOR, "After ICA"),
        ):
            axis.plot(
                locked.times_s,
                values,
                color=color,
                linewidth=1.0,
                label=label,
            )
            # Each trace has its own floor. ICA changes the non-locked variance as well as
            # the average, so applying the after-ICA floor to the before trace can turn an
            # unresolved estimate into an apparently resolved one.
            axis.axhline(
                floor,
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.75,
                label=f"{label} noise floor",
            )
        before_summary = "unresolved" if before[3] is None else f"{before[3]:.2f} µV resolved"
        after_summary = "unresolved" if after[3] is None else f"{after[3]:.2f} µV resolved"
        title = (
            f"{_comb_run_label(locked.recording_id)} · {locked.n_volumes} volumes · "
            f"before {before_summary} → after {after_summary}"
        )
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
        ncol=min(len(labels), 4),
        frameon=False,
        fontsize=7,
    )
    figure.suptitle(
        "Residual gradient envelope: each trace has its own odd–even noise floor; only "
        "positive signed excess power supports a resolved amplitude",
        fontsize=9,
    )
    plt.close(figure)
    return figure


def add_scanner_residual_section(
    *,
    report: mne.Report,
    combs: Sequence[CombResidual],
    averages: Sequence[VolumeLockedAverage],
    declined: Sequence[CombNotMeasured] = (),
    section: str = "Residual scanner gradient",
) -> None:
    """Append the per-run gradient residual evidence to a subject report."""
    from eeg_pipeline.preprocessing.report.organize import (
        remove_tagged_content,
    )
    from eeg_pipeline.preprocessing.report.style import report_image_format

    if not combs and not averages:
        raise ValueError("The gradient section requires at least one measured run.")
    remove_tagged_content(report, tag="scanner-residual")
    report.add_html(
        html=scanner_residual_html(combs, averages, declined=declined),
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


__all__ = [
    "BACKGROUND_FRACTION_RANGE",
    "COMB_WELCH_SECONDS",
    "CombNotMeasured",
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
