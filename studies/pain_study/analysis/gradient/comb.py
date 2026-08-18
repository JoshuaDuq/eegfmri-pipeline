"""Per-run comb residual measurement for the scanner gradient.

Gradient correction happens upstream, in BrainVision Analyzer, and the pipeline already
measures what survives it — but only as a cohort figure. A cohort median cannot tell the
person reviewing one subject whether *that* subject's correction worked, and residual
gradient artifact is the failure that most easily passes for data: it is periodic, it is
broadband, and after averaging across channels it disappears into the noise floor.

The comb residual works in the frequency domain. Gradient switching repeats once per
volume, so its residual appears as a comb of narrow lines at integer multiples of the
volume rate. Comparing the power in each line against the background *between* the lines
isolates the periodic residual from whatever else occupies the same band, which a plain
spectrum cannot do.

The measurement is not graded. The volume rate is measured from the markers rather than
configured, so the comb cannot be pointed at the wrong frequencies by a stale setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import mne
import numpy as np

from eeg_pipeline.preprocessing.report.annotations import annotation_onsets
from eeg_pipeline.preprocessing.report.filtering import (
    NOTCH_EXCLUSION_HALF_WIDTH_HZ,
    in_notch,
    notch_windows,
)

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


__all__ = [
    "BACKGROUND_FRACTION_RANGE",
    "COMB_WELCH_SECONDS",
    "CombNotMeasured",
    "HARMONIC_PEAK_FRACTION",
    "MINIMUM_BINS_PER_HARMONIC",
    "MINIMUM_VOLUMES",
    "VOLUME_MARKER_DESCRIPTION",
    "CombResidual",
    "VolumeTiming",
    "compute_comb_residual",
    "measure_volume_timing",
]
