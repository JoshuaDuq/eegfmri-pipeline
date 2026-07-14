"""Stage-wise scanner-harmonic quality control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mne
import numpy as np

from eeg_pipeline.analysis.qc.scanner_harmonics import (
    DEFAULT_HARMONIC_WINDOWS,
    summarize_scanner_harmonics,
)

CARDIAC_QC_TMIN_SECONDS = -0.2
CARDIAC_QC_TMAX_SECONDS = 0.6
CARDIAC_QC_BASELINE_STOP_SECONDS = -0.05
CARDIAC_QC_MINIMUM_EPOCHS = 3


@dataclass(frozen=True)
class HarmonicSpectrum:
    """Median Welch spectrum used for scanner-harmonic QC."""

    frequencies_hz: np.ndarray
    median_power_db: np.ndarray


@dataclass(frozen=True)
class HarmonicStageQc:
    """Numerical scanner-line metrics and their analyzed spectrum."""

    summary: dict[str, object]
    spectrum: HarmonicSpectrum


@dataclass(frozen=True)
class CardiacLockedSummary:
    """Robust EEG pulse-artifact summary aligned to accepted R peaks."""

    channel_names: tuple[str, ...]
    times: np.ndarray
    median_evoked: np.ndarray
    valid_epoch_count: int
    median_evoked_rms: float
    median_evoked_peak_to_peak: float
    channel_rms: np.ndarray
    channel_peak_to_peak: np.ndarray


@dataclass(frozen=True)
class CardiacLockedComparison:
    """Cardiac-locked EEG amplitude before and after pulse correction."""

    before: CardiacLockedSummary
    after: CardiacLockedSummary
    rms_attenuation_db: float
    peak_to_peak_attenuation_db: float


def _immutable(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result


def _validate_qrs_times(qrs_times: np.ndarray) -> np.ndarray:
    times = np.asarray(qrs_times, dtype=float)
    if times.ndim != 1:
        raise ValueError("QRS times must be one-dimensional")
    if not np.all(np.isfinite(times)):
        raise ValueError("QRS times must contain only finite values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("QRS times must be strictly increasing")
    return times


def summarize_cardiac_locked_eeg(
    raw: mne.io.BaseRaw,
    *,
    qrs_times: np.ndarray,
) -> CardiacLockedSummary:
    """Summarize baseline-corrected EEG epochs from -0.2 to 0.6 seconds around R peaks."""
    if not raw.preload:
        raise ValueError("Cardiac-locked QC requires a preloaded Raw object")
    times = _validate_qrs_times(qrs_times)
    channel_names = tuple(
        channel_name
        for channel_name, channel_type in zip(
            raw.ch_names,
            raw.get_channel_types(),
            strict=True,
        )
        if channel_type == "eeg"
    )
    if not channel_names:
        raise ValueError("Cardiac-locked QC requires at least one EEG channel")

    sampling_frequency = float(raw.info["sfreq"])
    start_offset = int(round(CARDIAC_QC_TMIN_SECONDS * sampling_frequency))
    stop_offset = int(round(CARDIAC_QC_TMAX_SECONDS * sampling_frequency))
    epoch_times = np.arange(start_offset, stop_offset, dtype=float) / sampling_frequency
    baseline = epoch_times < CARDIAC_QC_BASELINE_STOP_SECONDS
    if not np.any(baseline):
        raise ValueError("Cardiac-locked QC baseline contains no samples")

    data = raw.get_data(picks=list(channel_names))
    qrs_samples = np.rint(times * sampling_frequency).astype(int)
    valid_samples = qrs_samples[
        (qrs_samples + start_offset >= 0) & (qrs_samples + stop_offset <= raw.n_times)
    ]
    if valid_samples.size < CARDIAC_QC_MINIMUM_EPOCHS:
        raise ValueError(
            f"Cardiac-locked QC requires at least {CARDIAC_QC_MINIMUM_EPOCHS} valid epochs, "
            f"found {valid_samples.size}"
        )
    epochs = np.stack(
        [data[:, sample + start_offset : sample + stop_offset] for sample in valid_samples]
    )
    baseline_means = np.mean(epochs[:, :, baseline], axis=-1, keepdims=True)
    median_evoked = np.median(epochs - baseline_means, axis=0)
    channel_rms = np.sqrt(np.mean(median_evoked**2, axis=1))
    channel_peak_to_peak = np.ptp(median_evoked, axis=1)
    return CardiacLockedSummary(
        channel_names=channel_names,
        times=_immutable(epoch_times),
        median_evoked=_immutable(median_evoked),
        valid_epoch_count=int(valid_samples.size),
        median_evoked_rms=float(np.median(channel_rms)),
        median_evoked_peak_to_peak=float(np.median(channel_peak_to_peak)),
        channel_rms=_immutable(channel_rms),
        channel_peak_to_peak=_immutable(channel_peak_to_peak),
    )


def _amplitude_attenuation_db(before: float, after: float) -> float:
    if not np.isfinite(before) or before <= 0:
        raise ValueError("Before-correction cardiac amplitude must be finite and positive")
    if not np.isfinite(after) or after <= 0:
        raise ValueError("After-correction cardiac amplitude must be finite and positive")
    return float(20.0 * np.log10(before / after))


def compare_cardiac_locked_summaries(
    before: CardiacLockedSummary,
    after: CardiacLockedSummary,
) -> CardiacLockedComparison:
    """Compare like-for-like cardiac-locked EEG summaries."""
    if before.channel_names != after.channel_names:
        raise ValueError("Cardiac-locked channel names do not match")
    if not np.array_equal(before.times, after.times):
        raise ValueError("Cardiac-locked times do not match")
    return CardiacLockedComparison(
        before=before,
        after=after,
        rms_attenuation_db=_amplitude_attenuation_db(
            before.median_evoked_rms,
            after.median_evoked_rms,
        ),
        peak_to_peak_attenuation_db=_amplitude_attenuation_db(
            before.median_evoked_peak_to_peak,
            after.median_evoked_peak_to_peak,
        ),
    )


def reference_harmonic_frequencies(summary: Mapping[str, object]) -> dict[str, float]:
    """Extract raw-stage scanner-line frequencies for like-for-like comparisons."""
    return {
        window.column_prefix: float(summary[f"{window.column_prefix}_peak_hz"])
        for window in DEFAULT_HARMONIC_WINDOWS
    }


def _add_reference_metrics(
    summary: dict[str, object],
    frequencies: np.ndarray,
    median_power_db: np.ndarray,
    reference_frequencies: Mapping[str, float],
) -> None:
    for prefix, reference_frequency in reference_frequencies.items():
        reference_index = int(np.argmin(np.abs(frequencies - reference_frequency)))
        local_distance = np.abs(frequencies - reference_frequency)
        background = (local_distance >= 0.35) & (local_distance <= 2.0)
        if np.count_nonzero(background) < 2:
            raise ValueError(f"Insufficient neighboring PSD bins around {reference_frequency} Hz")
        reference_power_db = float(median_power_db[reference_index])
        summary[f"{prefix}_reference_hz"] = float(frequencies[reference_index])
        summary[f"{prefix}_reference_power_db"] = reference_power_db
        summary[f"{prefix}_reference_local_prominence_db"] = reference_power_db - float(
            np.median(median_power_db[background])
        )


def summarize_raw_harmonics(
    raw: mne.io.BaseRaw,
    *,
    stage: str,
    channels: Sequence[str],
    welch_duration_seconds: float,
    minimum_duration_seconds: float,
    reference_frequencies: Mapping[str, float] | None = None,
) -> HarmonicStageQc:
    """Compute the established scanner-harmonic summary for one in-memory stage."""
    sampling_frequency = float(raw.info["sfreq"])
    minimum_samples = int(round(minimum_duration_seconds * sampling_frequency))
    if raw.n_times < minimum_samples:
        raise ValueError(
            f"Scanner-harmonic QC requires {minimum_samples} samples, found {raw.n_times}"
        )
    missing = sorted(set(channels) - set(raw.ch_names))
    if missing:
        raise ValueError(f"Scanner-harmonic QC channels are missing: {missing}")

    requested_segment_samples = int(round(welch_duration_seconds * sampling_frequency))
    segment_samples = min(requested_segment_samples, raw.n_times)
    spectrum = raw.compute_psd(
        method="welch",
        n_fft=segment_samples,
        n_per_seg=segment_samples,
        n_overlap=segment_samples // 2,
        picks=list(channels),
        verbose=False,
    )
    frequencies = np.asarray(spectrum.freqs, dtype=float)
    power = np.asarray(spectrum.get_data(), dtype=float)
    if np.isclose(frequencies[-1], sampling_frequency / 2.0):
        frequencies[-1] = sampling_frequency / 2.0
    summary = summarize_scanner_harmonics(
        freqs=frequencies,
        psd=power,
        source_file=stage,
        sfreq=sampling_frequency,
        n_samples=raw.n_times,
        channel_names=channels,
    )
    summary["stage"] = stage
    if reference_frequencies is None:
        reference_frequencies = reference_harmonic_frequencies(summary)
    median_power_db = 10.0 * np.log10(np.maximum(np.median(power, axis=0), np.finfo(float).tiny))
    _add_reference_metrics(
        summary,
        frequencies,
        median_power_db,
        reference_frequencies,
    )
    return HarmonicStageQc(
        summary=summary,
        spectrum=HarmonicSpectrum(
            frequencies_hz=_immutable(frequencies),
            median_power_db=_immutable(median_power_db),
        ),
    )
