"""The volume-locked residual must not measure how long the participant was scanned.

An average over N epochs suppresses everything not locked to the marker by sqrt(N), so the
residual a subject report prints falls as the session lengthens even when the correction
has not improved at all. Pooling that number would rank participants by volume count.

The floor is measured by splitting the epochs odd against even, where the locked waveform
cancels exactly and only noise survives. These tests pin that the estimator recovers a
known artifact independently of epoch count, and returns nothing where there is nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.preprocessing.report.cohort.noise_floor import (
    LockedAverage,
    measure_locked_average,
)

N_CHANNELS = 8
N_TIMES = 100
ARTIFACT_AMPLITUDE_V = 2e-6
NOISE_V = 20e-6

#: RMS of a full-cycle sine of the amplitude above, in microvolts.
TRUTH_UV = ARTIFACT_AMPLITUDE_V / np.sqrt(2.0) * 1e6


def _epochs(n_epochs: int, *, amplitude_v=ARTIFACT_AMPLITUDE_V, seed: int = 0) -> np.ndarray:
    """Epochs holding an identical locked waveform buried in independent noise."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_epochs, N_CHANNELS, N_TIMES)) * NOISE_V
    latency = np.arange(N_TIMES)
    data += amplitude_v * np.sin(2 * np.pi * latency / N_TIMES)
    return data - data.mean(axis=2, keepdims=True)


def test_the_resolved_amplitude_does_not_depend_on_the_number_of_epochs() -> None:
    """The property the whole correction exists for."""
    short = measure_locked_average(_epochs(50))
    long = measure_locked_average(_epochs(400))

    # The raw averaged amplitude falls substantially with epoch count: that is the
    # confound, and it is present in the number the subject report prints. It does not
    # fall by the full sqrt(8) between these counts precisely because part of it is the
    # artifact, which does not fall at all -- which is the whole difficulty.
    assert short.locked_rms_uv > 1.7 * long.locked_rms_uv

    # The corrected amplitude recovers the injected artifact at both counts.
    assert short.resolved_amplitude_uv == pytest.approx(TRUTH_UV, rel=0.25)
    assert long.resolved_amplitude_uv == pytest.approx(TRUTH_UV, rel=0.10)


def test_a_recording_below_the_floor_is_unresolved_not_zero() -> None:
    """A negative noisy estimate is censored evidence, not an exact zero."""
    measured = measure_locked_average(_epochs(400, amplitude_v=0.0, seed=2))

    assert measured.excess_power_uv2 < 0.0
    assert not measured.is_resolved
    assert measured.resolved_amplitude_uv is None

    # The raw figure, by contrast, is the noise floor wearing the units of an artifact:
    # a reader given it has no way to see the recording contains none at all.
    assert measured.locked_rms_uv == pytest.approx(NOISE_V * 1e6 / np.sqrt(400), rel=0.1)


def test_the_measured_floor_falls_as_the_root_of_the_epoch_count() -> None:
    """The floor is measured, not assumed, so it must show the known behaviour."""
    short = measure_locked_average(_epochs(50, amplitude_v=0.0, seed=3))
    long = measure_locked_average(_epochs(200, amplitude_v=0.0, seed=3))

    assert short.noise_floor_uv / long.noise_floor_uv == pytest.approx(2.0, rel=0.15)


def test_the_measured_floor_matches_the_analytic_one() -> None:
    """An odd-even split estimates sigma / sqrt(N) and nothing else."""
    measured = measure_locked_average(_epochs(200, amplitude_v=0.0, seed=4))

    assert measured.noise_floor_uv == pytest.approx(NOISE_V * 1e6 / np.sqrt(200), rel=0.1)


def test_the_locked_waveform_cancels_in_the_split_however_large_it_is() -> None:
    """A huge artifact must not inflate the floor, or it would subtract itself away."""
    quiet = measure_locked_average(_epochs(200, amplitude_v=0.0, seed=5))
    loud = measure_locked_average(_epochs(200, amplitude_v=200e-6, seed=5))

    assert loud.noise_floor_uv == pytest.approx(quiet.noise_floor_uv, rel=0.05)
    assert loud.resolved_amplitude_uv == pytest.approx(200e-6 / np.sqrt(2.0) * 1e6, rel=0.02)


def test_the_estimator_is_deterministic() -> None:
    """No random draws, so two calls on one input cannot differ."""
    data = _epochs(60)

    assert measure_locked_average(data) == measure_locked_average(data)


def test_an_odd_epoch_count_pairs_all_but_one() -> None:
    """Equal halves keep the algebra exact; the dropped epoch is reported."""
    measured = measure_locked_average(_epochs(61))

    assert measured.n_epochs == 61
    assert measured.n_paired_epochs == 60
    assert measured.resolved_amplitude_uv == pytest.approx(TRUTH_UV, rel=0.25)


def test_an_odd_count_scales_the_paired_floor_to_the_full_average() -> None:
    data = np.asarray([1.0, -1.0, 0.0], dtype=float).reshape(3, 1, 1) * 1e-6

    measured = measure_locked_average(data)

    assert measured.n_paired_epochs == 2
    assert measured.noise_floor_uv == pytest.approx(np.sqrt(2.0 / 3.0))


def test_a_single_epoch_cannot_be_corrected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        measure_locked_average(_epochs(1))


def test_detectability_is_reported_separately_and_is_not_the_pooled_quantity() -> None:
    """It grows with epoch count by construction, so a cohort cannot pool it."""
    short = measure_locked_average(_epochs(50))
    long = measure_locked_average(_epochs(400))

    assert long.detectability > short.detectability


def test_the_average_is_returned_for_the_caller_to_draw() -> None:
    """Returned rather than recomputed: a second pass over the epochs is the cost this
    module exists to avoid."""
    measured = measure_locked_average(_epochs(20))

    assert isinstance(measured, LockedAverage)
    assert measured.average.shape == (N_CHANNELS, N_TIMES)


def test_a_two_dimensional_array_is_refused() -> None:
    with pytest.raises(ValueError, match="epochs-by-channels-by-times"):
        measure_locked_average(np.zeros((8, 100)))
