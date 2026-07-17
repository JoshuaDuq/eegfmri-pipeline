from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.preprocessing.eeg_fmri.gradient import (
    GradientArtifactParameters,
    apply_fitted_residual_obs,
    correct_gradient_average,
    correct_gradient_artifact,
    fit_cross_fitted_residual_obs,
    resolve_volume_boundary,
    resolve_volume_segments,
    subtract_gradient_average,
    validate_volume_samples,
)
from eeg_pipeline.preprocessing.eeg_fmri.sequence import MultibandSliceSchedule


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


def _multiband_recording() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    MultibandSliceSchedule,
]:
    sampling_frequency = 1_000.0
    samples_per_volume = 100
    volume_count = 41
    volume_samples = 20 + np.arange(volume_count) * samples_per_volume
    n_samples = int(volume_samples[-1] + samples_per_volume + 20)
    schedule = MultibandSliceSchedule(
        repetition_time_seconds=0.1,
        slice_times_seconds=np.repeat([0.0, 0.025, 0.05, 0.075], 2),
        multiband_factor=2,
    )

    time = np.arange(n_samples) / sampling_frequency
    neural = np.vstack(
        (
            1.2 * np.sin(2 * np.pi * 13.7 * time + 0.3),
            0.9 * np.cos(2 * np.pi * 8.3 * time),
            1.5 * np.sin(2 * np.pi * 5.1 * time + 0.2),
        )
    )
    group_samples = schedule.group_boundaries_samples(sampling_frequency)
    contaminated = neural.copy()
    for volume_index, volume_start in enumerate(volume_samples):
        for group_index, (group_start, group_stop) in enumerate(
            zip(group_samples[:-1], group_samples[1:], strict=True)
        ):
            sample_count = group_stop - group_start
            phase = np.linspace(0.0, 1.0, sample_count, endpoint=False)
            template = 70.0 * np.sin(2 * np.pi * (group_index + 2) * phase) + 25.0 * np.cos(
                2 * np.pi * (group_index + 5) * phase
            )
            residual_shape = 12.0 * np.sin(2 * np.pi * 3 * phase + 0.7)
            residual_weight = np.sin(2 * np.pi * volume_index / 7 + group_index)
            delay = 0.35 * np.sin(2 * np.pi * volume_index / 11 + group_index)
            amplitude = 1.0 + 0.12 * np.cos(2 * np.pi * volume_index / 9 + group_index)
            artifact = amplitude * _fractional_delay(template, delay)
            artifact += residual_weight * residual_shape
            start = volume_start + group_start
            stop = volume_start + group_stop
            contaminated[0, start:stop] += artifact
            contaminated[1, start:stop] -= 0.65 * artifact
            contaminated[2, start:stop] += 0.4 * artifact
    return contaminated, neural, volume_samples, sampling_frequency, schedule


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


def test_resolve_volume_segments_preserves_internal_scanner_restart() -> None:
    segments = resolve_volume_segments(
        np.array([10, 110, 210, 500, 600, 700]),
        n_samples=850,
        sampling_frequency=1_000.0,
        repetition_time_seconds=0.1,
        maximum_marker_deviation_samples=0,
    )

    assert len(segments) == 2
    np.testing.assert_array_equal(
        segments[0].complete_volume_samples,
        np.array([10, 110, 210]),
    )
    np.testing.assert_array_equal(
        segments[1].complete_volume_samples,
        np.array([500, 600, 700]),
    )
    assert all(segment.crop_stop_sample is None for segment in segments)


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
        residual_obs_picks=np.array([0, 1, 2]),
        slice_schedule=MultibandSliceSchedule(
            repetition_time_seconds=0.1,
            slice_times_seconds=np.array([0.0]),
            multiband_factor=1,
        ),
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
        residual_obs_picks=np.array([0, 1, 2]),
        slice_schedule=MultibandSliceSchedule(
            repetition_time_seconds=0.1,
            slice_times_seconds=np.array([0.0]),
            multiband_factor=1,
        ),
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


def test_optional_cross_fitted_obs_removes_group_locked_residuals() -> None:
    contaminated, neural, volume_samples, sampling_frequency, schedule = _multiband_recording()
    without_obs = GradientArtifactParameters(
        repetition_time_seconds=0.1,
        moving_average_volumes=11,
        alignment_upsampling=4,
        maximum_alignment_shift_samples=1.0,
        residual_obs_components=0,
        residual_obs_folds=5,
        residual_obs_seed=42,
    )
    with_obs = GradientArtifactParameters(
        repetition_time_seconds=0.1,
        moving_average_volumes=11,
        alignment_upsampling=4,
        maximum_alignment_shift_samples=1.0,
        residual_obs_components=2,
        residual_obs_folds=5,
        residual_obs_seed=42,
    )

    aas = correct_gradient_artifact(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        residual_obs_picks=np.array([0, 1]),
        slice_schedule=schedule,
        parameters=without_obs,
    )
    obs = correct_gradient_artifact(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        residual_obs_picks=np.array([0, 1]),
        slice_schedule=schedule,
        parameters=with_obs,
    )

    scan = slice(volume_samples[0], volume_samples[-1] + 100)
    aas_error = np.sqrt(np.mean((aas.data[:2, scan] - neural[:2, scan]) ** 2))
    obs_error = np.sqrt(np.mean((obs.data[:2, scan] - neural[:2, scan]) ** 2))
    assert obs_error < 0.6 * aas_error
    np.testing.assert_allclose(obs.data[2], aas.data[2], atol=1e-12)
    assert obs.group_shifts_samples.shape == (volume_samples.size, 4)
    assert obs.residual_obs_removed_rms > 0.0


def test_gradient_average_exposes_the_reusable_pre_obs_boundary() -> None:
    contaminated, _, volume_samples, sampling_frequency, schedule = _multiband_recording()
    parameters = GradientArtifactParameters(
        repetition_time_seconds=0.1,
        moving_average_volumes=11,
        alignment_upsampling=4,
        maximum_alignment_shift_samples=1.0,
        residual_obs_components=0,
        residual_obs_folds=5,
        residual_obs_seed=42,
    )

    average = correct_gradient_average(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        slice_schedule=schedule,
        parameters=parameters,
    )
    complete = correct_gradient_artifact(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        residual_obs_picks=np.array([0, 1]),
        slice_schedule=schedule,
        parameters=parameters,
    )

    np.testing.assert_allclose(average.data, complete.data, atol=1e-12)
    expected = subtract_gradient_average(
        contaminated,
        volume_samples,
        samples_per_volume=100,
        volume_shifts_samples=average.volume_shifts_samples,
        moving_average_volumes=parameters.moving_average_volumes,
    )
    np.testing.assert_allclose(average.data, expected, atol=1e-12)
    np.testing.assert_allclose(
        average.group_shifts_samples,
        complete.group_shifts_samples,
        atol=1e-12,
    )


def test_cross_fitted_obs_preserves_non_scanner_locked_neural_projection() -> None:
    contaminated, neural, volume_samples, sampling_frequency, schedule = _multiband_recording()
    parameters = GradientArtifactParameters(
        repetition_time_seconds=0.1,
        moving_average_volumes=11,
        alignment_upsampling=4,
        maximum_alignment_shift_samples=1.0,
        residual_obs_components=1,
        residual_obs_folds=5,
        residual_obs_seed=42,
    )

    result = correct_gradient_artifact(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        residual_obs_picks=np.array([0, 1]),
        slice_schedule=schedule,
        parameters=parameters,
    )

    scan = slice(volume_samples[0], volume_samples[-1] + 100)
    for channel_index in (0, 1):
        retained_projection = np.dot(
            result.data[channel_index, scan], neural[channel_index, scan]
        ) / np.dot(neural[channel_index, scan], neural[channel_index, scan])
        assert 0.85 < retained_projection < 1.15


def test_fitted_obs_model_produces_nested_component_candidates() -> None:
    contaminated, _, volume_samples, sampling_frequency, schedule = _multiband_recording()
    parameters = GradientArtifactParameters(
        repetition_time_seconds=0.1,
        moving_average_volumes=11,
        alignment_upsampling=4,
        maximum_alignment_shift_samples=1.0,
        residual_obs_components=0,
        residual_obs_folds=5,
        residual_obs_seed=42,
    )
    average = correct_gradient_average(
        contaminated,
        volume_samples,
        sampling_frequency=sampling_frequency,
        alignment_picks=np.array([0, 1]),
        slice_schedule=schedule,
        parameters=parameters,
    )
    model = fit_cross_fitted_residual_obs(
        average.data,
        volume_samples,
        group_boundaries=average.group_boundaries_samples,
        picks=np.array([0, 1]),
        maximum_components=4,
        n_folds=5,
        seed=42,
    )

    one_component, one_removed_rms = apply_fitted_residual_obs(
        average.data,
        model=model,
        n_components=1,
    )
    two_components, two_removed_rms = apply_fitted_residual_obs(
        average.data,
        model=model,
        n_components=2,
    )

    assert one_component.shape == average.data.shape
    two_component_difference = np.linalg.norm(two_components - one_component)
    assert two_component_difference > 0
    assert two_removed_rms >= one_removed_rms > 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("moving_average_volumes", 10, "odd"),
        ("alignment_upsampling", 1, "at least 2"),
        ("maximum_alignment_shift_samples", 0.0, "positive"),
        ("residual_obs_components", 5, "between 0 and 4"),
        ("residual_obs_folds", 1, "at least 2"),
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
        "residual_obs_components": 2,
        "residual_obs_folds": 5,
        "residual_obs_seed": 42,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        GradientArtifactParameters(**values)
