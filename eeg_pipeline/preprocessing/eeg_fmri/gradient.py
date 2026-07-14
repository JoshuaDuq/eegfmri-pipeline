"""Scanner-gradient artifact correction for synchronized EEG-fMRI recordings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class GradientArtifactParameters:
    """Fixed parameters for phase-aligned adaptive average subtraction."""

    repetition_time_seconds: float
    moving_average_volumes: int = 21
    alignment_upsampling: int = 4
    maximum_alignment_shift_samples: float = 2.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.repetition_time_seconds):
            raise ValueError("repetition_time_seconds must be finite")
        if self.repetition_time_seconds <= 0:
            raise ValueError("repetition_time_seconds must be positive")
        if self.moving_average_volumes < 3:
            raise ValueError("moving_average_volumes must be at least 3")
        if self.moving_average_volumes % 2 == 0:
            raise ValueError("moving_average_volumes must be odd")
        if self.alignment_upsampling < 2:
            raise ValueError("alignment_upsampling must be at least 2")
        if not np.isfinite(self.maximum_alignment_shift_samples):
            raise ValueError("maximum_alignment_shift_samples must be finite")
        if self.maximum_alignment_shift_samples <= 0:
            raise ValueError("maximum_alignment_shift_samples must be positive")


@dataclass(frozen=True)
class GradientCorrectionResult:
    """Corrected samples and the estimated scanner-phase displacement per volume."""

    data: np.ndarray
    volume_shifts_samples: np.ndarray


@dataclass(frozen=True)
class VolumeBoundary:
    """Canonical full-volume samples and an explicit terminal crop boundary."""

    complete_volume_samples: np.ndarray
    marker_offsets_samples: np.ndarray
    crop_stop_sample: int | None
    discarded_terminal_samples: int


def _samples_per_volume(
    sampling_frequency: float,
    repetition_time_seconds: float,
) -> int:
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    sample_count = sampling_frequency * repetition_time_seconds
    rounded_sample_count = round(sample_count)
    if not np.isclose(sample_count, rounded_sample_count, rtol=0.0, atol=1e-9):
        raise ValueError(
            "repetition_time_seconds must map to an integer number of acquisition samples"
        )
    return int(rounded_sample_count)


def validate_volume_samples(
    volume_samples: Sequence[int] | np.ndarray,
    *,
    n_samples: int,
    sampling_frequency: float,
    repetition_time_seconds: float,
) -> int:
    """Validate exact scanner-volume timing and return samples per volume."""
    samples = np.asarray(volume_samples)
    if samples.ndim != 1:
        raise ValueError("volume_samples must be one-dimensional")
    if not np.issubdtype(samples.dtype, np.integer):
        raise TypeError("volume_samples must contain integer sample indices")
    if samples.size < 2:
        raise ValueError("At least two volume markers are required")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if samples[0] < 0:
        raise ValueError("volume_samples cannot contain negative indices")
    if np.any(np.diff(samples) <= 0):
        raise ValueError("volume_samples must be strictly increasing")

    sample_count = _samples_per_volume(sampling_frequency, repetition_time_seconds)
    intervals = np.diff(samples)
    if np.any(intervals != sample_count):
        unexpected = np.unique(intervals[intervals != sample_count]).tolist()
        raise ValueError(
            f"Volume-marker interval differs from {sample_count} samples: {unexpected}"
        )
    if samples[-1] + sample_count > n_samples:
        raise ValueError("The last volume marker does not have a complete volume interval")
    return sample_count


def resolve_volume_boundary(
    volume_samples: Sequence[int] | np.ndarray,
    *,
    n_samples: int,
    sampling_frequency: float,
    repetition_time_seconds: float,
    maximum_marker_deviation_samples: int,
) -> VolumeBoundary:
    """Canonicalize marker quantization and isolate an incomplete final volume."""
    samples = np.asarray(volume_samples)
    if samples.ndim != 1:
        raise ValueError("volume_samples must be one-dimensional")
    if not np.issubdtype(samples.dtype, np.integer):
        raise TypeError("volume_samples must contain integer sample indices")
    if samples.size < 2:
        raise ValueError("At least two volume markers are required")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if maximum_marker_deviation_samples < 0:
        raise ValueError("maximum_marker_deviation_samples must be non-negative")
    if samples[0] < 0 or np.any(np.diff(samples) <= 0):
        raise ValueError("volume_samples must be non-negative and strictly increasing")
    if samples[-1] >= n_samples:
        raise ValueError("The final volume marker must occur before the recording ends")

    samples_per_volume = _samples_per_volume(
        sampling_frequency,
        repetition_time_seconds,
    )
    canonical_samples = samples[0] + np.arange(samples.size) * samples_per_volume
    marker_offsets = samples - canonical_samples
    if np.any(np.abs(marker_offsets) > maximum_marker_deviation_samples):
        deviations = np.unique(
            marker_offsets[np.abs(marker_offsets) > maximum_marker_deviation_samples]
        ).tolist()
        raise ValueError(
            "Volume-marker deviation exceeds the configured tolerance: " f"{deviations} samples"
        )

    terminal_sample_count = n_samples - int(canonical_samples[-1])
    if terminal_sample_count >= samples_per_volume:
        complete_samples = canonical_samples
        crop_stop_sample = None
        discarded_terminal_samples = 0
    else:
        complete_samples = canonical_samples[:-1]
        crop_stop_sample = int(canonical_samples[-1])
        discarded_terminal_samples = terminal_sample_count
    if complete_samples.size < 2:
        raise ValueError("At least two complete scanner volumes are required")
    return VolumeBoundary(
        complete_volume_samples=complete_samples.astype(int, copy=False),
        marker_offsets_samples=marker_offsets.astype(int, copy=False),
        crop_stop_sample=crop_stop_sample,
        discarded_terminal_samples=int(discarded_terminal_samples),
    )


def _validate_data(data: np.ndarray) -> np.ndarray:
    samples = np.asarray(data)
    if samples.ndim != 2:
        raise ValueError("data must have shape (channels, samples)")
    if samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("data must contain at least one channel and one sample")
    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError("data must use a floating-point dtype")
    if not np.all(np.isfinite(samples)):
        raise ValueError("data contains non-finite samples")
    return samples


def _validate_picks(picks: Sequence[int] | np.ndarray, n_channels: int) -> np.ndarray:
    indices = np.asarray(picks)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("alignment_picks must be a non-empty one-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("alignment_picks must contain integer channel indices")
    if np.unique(indices).size != indices.size:
        raise ValueError("alignment_picks must not contain duplicates")
    if np.any(indices < 0) or np.any(indices >= n_channels):
        raise ValueError("alignment_picks contains an out-of-range channel index")
    return indices.astype(int, copy=False)


def _extract_alignment_epochs(
    data: np.ndarray,
    volume_samples: np.ndarray,
    samples_per_volume: int,
    picks: np.ndarray,
) -> np.ndarray:
    channel_energy = np.zeros(picks.size, dtype=float)
    for start in volume_samples:
        segment = data[picks, start : start + samples_per_volume]
        channel_energy += np.sum(np.square(segment), axis=1)
    reference_pick = picks[int(np.argmax(channel_energy))]
    return np.stack(
        [data[reference_pick, start : start + samples_per_volume] for start in volume_samples]
    )


def _normalized_correlation(observed: np.ndarray, template: np.ndarray) -> float:
    observed_centered = observed - np.mean(observed)
    template_centered = template - np.mean(template)
    denominator = np.linalg.norm(observed_centered) * np.linalg.norm(template_centered)
    if denominator == 0:
        raise ValueError("Cannot align a constant scanner-artifact reference")
    return float(np.dot(observed_centered, template_centered) / denominator)


def _correlation_at_shift(
    observed: np.ndarray,
    template: np.ndarray,
    shift: int,
) -> float:
    if shift > 0:
        return _normalized_correlation(observed[shift:], template[:-shift])
    if shift < 0:
        return _normalized_correlation(observed[:shift], template[-shift:])
    return _normalized_correlation(observed, template)


def _parabolic_peak_offset(scores: np.ndarray, peak_index: int) -> float:
    if peak_index == 0 or peak_index == scores.size - 1:
        return 0.0
    left, center, right = scores[peak_index - 1 : peak_index + 2]
    curvature = left - 2.0 * center + right
    if curvature == 0:
        return 0.0
    offset = 0.5 * (left - right) / curvature
    return float(np.clip(offset, -0.5, 0.5))


def estimate_volume_shifts(
    data: np.ndarray,
    volume_samples: np.ndarray,
    *,
    samples_per_volume: int,
    alignment_picks: np.ndarray,
    upsampling: int,
    maximum_shift_samples: float,
) -> np.ndarray:
    """Estimate a shared fractional scanner-phase shift for each volume."""
    reference_epochs = _extract_alignment_epochs(
        data,
        volume_samples,
        samples_per_volume,
        alignment_picks,
    )
    reference_template = np.mean(reference_epochs, axis=0)
    upsampled_template = resample_poly(reference_template, upsampling, 1)
    maximum_shift = int(round(maximum_shift_samples * upsampling))
    candidate_shifts = np.arange(-maximum_shift, maximum_shift + 1)

    shifts = np.empty(volume_samples.size, dtype=float)
    for index, reference_epoch in enumerate(reference_epochs):
        upsampled_epoch = resample_poly(reference_epoch, upsampling, 1)
        scores = np.array(
            [
                _correlation_at_shift(upsampled_epoch, upsampled_template, int(shift))
                for shift in candidate_shifts
            ]
        )
        peak_index = int(np.argmax(scores))
        peak_shift = candidate_shifts[peak_index] + _parabolic_peak_offset(scores, peak_index)
        shifts[index] = peak_shift / upsampling
    return shifts


def _shift_to_canonical(waveform: np.ndarray, shift_samples: float) -> np.ndarray:
    sample_positions = np.arange(waveform.size, dtype=float)
    return np.interp(
        sample_positions + shift_samples,
        sample_positions,
        waveform,
        left=waveform[0],
        right=waveform[-1],
    )


def _window_bounds(index: int, count: int, window: int) -> tuple[int, int]:
    start = min(max(index - window // 2, 0), count - window)
    return start, start + window


def _leave_one_out_templates(aligned_epochs: np.ndarray, window: int) -> np.ndarray:
    templates = np.empty_like(aligned_epochs)
    for index in range(aligned_epochs.shape[0]):
        start, stop = _window_bounds(index, aligned_epochs.shape[0], window)
        window_sum = np.sum(aligned_epochs[start:stop], axis=0)
        templates[index] = (window_sum - aligned_epochs[index]) / (window - 1)
    return templates


def _fit_artifact(epoch: np.ndarray, template: np.ndarray) -> np.ndarray:
    centered_epoch = epoch - np.mean(epoch)
    centered_template = template - np.mean(template)
    temporal_derivative = np.gradient(centered_template)
    basis = np.column_stack((centered_template, temporal_derivative))
    if np.linalg.matrix_rank(basis) != basis.shape[1]:
        raise ValueError("Cannot fit a constant gradient-artifact template")
    coefficients, _, _, _ = np.linalg.lstsq(basis, centered_epoch, rcond=None)
    return basis[:, 0] * coefficients[0] + basis[:, 1] * coefficients[1]


def subtract_gradient_average(
    data: np.ndarray,
    volume_samples: np.ndarray,
    *,
    samples_per_volume: int,
    volume_shifts_samples: np.ndarray,
    moving_average_volumes: int,
) -> np.ndarray:
    """Subtract a leave-one-out, phase-aligned moving artifact average."""
    corrected = np.array(data, dtype=float, copy=True)
    for channel_index in range(data.shape[0]):
        epochs = np.stack(
            [data[channel_index, start : start + samples_per_volume] for start in volume_samples]
        )
        aligned_epochs = np.stack(
            [
                _shift_to_canonical(epoch, shift)
                for epoch, shift in zip(epochs, volume_shifts_samples, strict=True)
            ]
        )
        templates = _leave_one_out_templates(aligned_epochs, moving_average_volumes)
        for index, start in enumerate(volume_samples):
            observed_template = _shift_to_canonical(
                templates[index],
                -volume_shifts_samples[index],
            )
            artifact = _fit_artifact(epochs[index], observed_template)
            corrected[channel_index, start : start + samples_per_volume] -= artifact
    return corrected


def correct_gradient_artifact(
    data: np.ndarray,
    volume_samples: Sequence[int] | np.ndarray,
    *,
    sampling_frequency: float,
    alignment_picks: Sequence[int] | np.ndarray,
    parameters: GradientArtifactParameters,
) -> GradientCorrectionResult:
    """Apply synchronized, phase-aligned adaptive AAS to continuous samples."""
    sample_data = _validate_data(data)
    samples = np.asarray(volume_samples)
    samples_per_volume = validate_volume_samples(
        samples,
        n_samples=sample_data.shape[1],
        sampling_frequency=sampling_frequency,
        repetition_time_seconds=parameters.repetition_time_seconds,
    )
    picks = _validate_picks(alignment_picks, sample_data.shape[0])
    if samples.size < parameters.moving_average_volumes:
        raise ValueError(
            f"At least {parameters.moving_average_volumes} volume markers are required, "
            f"found {samples.size}"
        )

    shifts = estimate_volume_shifts(
        sample_data,
        samples,
        samples_per_volume=samples_per_volume,
        alignment_picks=picks,
        upsampling=parameters.alignment_upsampling,
        maximum_shift_samples=parameters.maximum_alignment_shift_samples,
    )
    corrected = subtract_gradient_average(
        sample_data,
        samples,
        samples_per_volume=samples_per_volume,
        volume_shifts_samples=shifts,
        moving_average_volumes=parameters.moving_average_volumes,
    )
    if not np.all(np.isfinite(corrected)):
        raise FloatingPointError("Gradient correction produced non-finite samples")
    return GradientCorrectionResult(data=corrected, volume_shifts_samples=shifts)
