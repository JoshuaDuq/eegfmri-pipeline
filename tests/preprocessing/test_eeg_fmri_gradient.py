from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.preprocessing.eeg_fmri.gradient import (
    GradientArtifactParameters,
    correct_gradient_artifact,
    resolve_volume_boundary,
    validate_volume_samples,
)


def _fractional_delay(waveform: np.ndarray, delay_samples: float) -> np.ndarray:
    samples = np.arange(waveform.size, dtype=float)
    return np.interp(
        samples - delay_samples,
        samples,
        waveform,
        left=waveform[0],
        right=waveform[-1],
    )


def _synthetic_recording() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sampling_frequency = 1_000.0
    samples_per_volume = 100
    volume_count = 31
    first_volume_sample = 25
    volume_samples = first_volume_sample + np.arange(volume_count) * samples_per_volume
    n_samples = int(volume_samples[-1] + samples_per_volume + 25)

    time = np.arange(n_samples) / sampling_frequency
    neural = np.vstack(
        (
            2.0 * np.sin(2 * np.pi * 13.7 * time),
            1.5 * np.cos(2 * np.pi * 8.3 * time + 0.4),
            0.8 * np.sin(2 * np.pi * 1.2 * time),
        )
    )
    within_volume = np.arange(samples_per_volume)
    artifact_template = 80.0 * np.sin(
        2 * np.pi * 3 * within_volume / samples_per_volume
    ) + 35.0 * np.cos(2 * np.pi * 7 * within_volume / samples_per_volume)
    channel_scales = np.array([1.0, -0.7, 0.4])
    contaminated = neural.copy()
    for volume_index, start in enumerate(volume_samples):
        delay = 0.45 * np.sin(2 * np.pi * volume_index / 9)
        amplitude = 1.0 + 0.08 * np.cos(2 * np.pi * volume_index / 7)
        artifact = amplitude * _fractional_delay(artifact_template, delay)
        stop = start + samples_per_volume
        contaminated[:, start:stop] += channel_scales[:, np.newaxis] * artifact

    return contaminated, neural, volume_samples, sampling_frequency


def test_validate_volume_samples_requires_exact_fixed_repetition_time() -> None:
    samples = np.array([10, 110, 210, 311])

    with pytest.raises(ValueError, match="interval"):
        validate_volume_samples(
            samples,
            n_samples=500,
            sampling_frequency=1_000.0,
            repetition_time_seconds=0.1,
        )


def test_validate_volume_samples_requires_complete_last_volume() -> None:
    with pytest.raises(ValueError, match="last volume"):
        validate_volume_samples(
            np.array([10, 110, 210]),
            n_samples=305,
            sampling_frequency=1_000.0,
            repetition_time_seconds=0.1,
        )


def test_resolve_volume_boundary_crops_an_incomplete_terminal_volume() -> None:
    boundary = resolve_volume_boundary(
        np.array([10, 110, 210]),
        n_samples=305,
        sampling_frequency=1_000.0,
        repetition_time_seconds=0.1,
        maximum_marker_deviation_samples=0,
    )

    np.testing.assert_array_equal(boundary.complete_volume_samples, np.array([10, 110]))
    np.testing.assert_array_equal(boundary.marker_offsets_samples, np.zeros(3, dtype=int))
    assert boundary.crop_stop_sample == 210
    assert boundary.discarded_terminal_samples == 95


def test_resolve_volume_boundary_canonicalizes_one_sample_marker_quantization() -> None:
    boundary = resolve_volume_boundary(
        np.array([10, 110, 211, 310]),
        n_samples=420,
        sampling_frequency=1_000.0,
        repetition_time_seconds=0.1,
        maximum_marker_deviation_samples=1,
    )

    np.testing.assert_array_equal(
        boundary.complete_volume_samples,
        np.array([10, 110, 210, 310]),
    )
    np.testing.assert_array_equal(
        boundary.marker_offsets_samples,
        np.array([0, 0, 1, 0]),
    )
    assert boundary.crop_stop_sample is None


def test_resolve_volume_boundary_rejects_marker_deviation_above_tolerance() -> None:
    with pytest.raises(ValueError, match="deviation"):
        resolve_volume_boundary(
            np.array([10, 110, 212, 310]),
            n_samples=420,
            sampling_frequency=1_000.0,
            repetition_time_seconds=0.1,
            maximum_marker_deviation_samples=1,
        )


def test_gradient_correction_suppresses_jittered_artifact_and_preserves_boundaries() -> None:
    contaminated, neural, volume_samples, sampling_frequency = _synthetic_recording()
    parameters = GradientArtifactParameters(
        repetition_time_seconds=0.1,
        moving_average_volumes=11,
        alignment_upsampling=4,
        maximum_alignment_shift_samples=1.0,
    )

    result = correct_gradient_artifact(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        parameters=parameters,
    )

    scan_slice = slice(volume_samples[0], volume_samples[-1] + 100)
    raw_error = contaminated[:, scan_slice] - neural[:, scan_slice]
    corrected_error = result.data[:, scan_slice] - neural[:, scan_slice]
    assert np.sqrt(np.mean(corrected_error[:2] ** 2)) < 0.03 * np.sqrt(np.mean(raw_error[:2] ** 2))
    assert np.sqrt(np.mean(corrected_error[2] ** 2)) < 0.12 * np.sqrt(np.mean(raw_error[2] ** 2))
    np.testing.assert_array_equal(result.data[:, : volume_samples[0]], contaminated[:, :25])
    np.testing.assert_array_equal(
        result.data[:, volume_samples[-1] + 100 :],
        contaminated[:, volume_samples[-1] + 100 :],
    )
    assert np.max(np.abs(result.volume_shifts_samples)) <= 1.0


def test_gradient_correction_preserves_neural_projection() -> None:
    contaminated, neural, volume_samples, sampling_frequency = _synthetic_recording()
    parameters = GradientArtifactParameters(
        repetition_time_seconds=0.1,
        moving_average_volumes=11,
        alignment_upsampling=4,
        maximum_alignment_shift_samples=1.0,
    )

    result = correct_gradient_artifact(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        parameters=parameters,
    )

    scan_slice = slice(volume_samples[0], volume_samples[-1] + 100)
    for channel_index in range(neural.shape[0]):
        retained_projection = np.dot(
            result.data[channel_index, scan_slice],
            neural[channel_index, scan_slice],
        ) / np.dot(
            neural[channel_index, scan_slice],
            neural[channel_index, scan_slice],
        )
        assert 0.85 < retained_projection < 1.15


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("moving_average_volumes", 10, "odd"),
        ("alignment_upsampling", 1, "at least 2"),
        ("maximum_alignment_shift_samples", 0.0, "positive"),
    ],
)
def test_gradient_parameters_reject_invalid_values(
    field: str,
    value: int | float,
    message: str,
) -> None:
    values: dict[str, int | float] = {
        "repetition_time_seconds": 0.9,
        "moving_average_volumes": 21,
        "alignment_upsampling": 4,
        "maximum_alignment_shift_samples": 2.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        GradientArtifactParameters(**values)
