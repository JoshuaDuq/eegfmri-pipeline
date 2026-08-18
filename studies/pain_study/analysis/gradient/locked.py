"""Per-run volume-locked residual average for the scanner gradient.

The volume-locked average works in the time domain and is the canonical Allen/Niazy
view. Averaging the recording time-locked to the volume marker cancels everything that
is not phase-locked to the gradient, so what remains is the residual artifact at the
amplitude it actually reaches. It is reported as the across-channel RMS of that average,
which is an envelope rather than the waveform: rectifying across channels discards
polarity, so the trace carries magnitude over time and not shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np

from eeg_pipeline.preprocessing.report.annotations import annotation_onsets, onset_events
from studies.pain_study.analysis.gradient.comb import (
    MINIMUM_VOLUMES,
    VOLUME_MARKER_DESCRIPTION,
    VolumeTiming,
)
from studies.pain_study.analysis.noise_floor import measure_locked_average


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


__all__ = [
    "VolumeLockedAverage",
    "compute_volume_locked_average",
]
