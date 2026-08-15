from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from eeg_pipeline.spectral_availability import (
    EpochSpectralAvailability,
    FrequencyInterval,
    RecordingExclusions,
    RecordingKey,
    merge_frequency_intervals,
)


def _availability() -> EpochSpectralAvailability:
    first_run = RecordingKey(subject="0001", task="thermalactive", run="1")
    second_run = RecordingKey(subject="0001", task="thermalactive", run="2")
    return EpochSpectralAvailability(
        recording_keys=(first_run, second_run),
        exclusions_by_epoch=(
            (FrequencyInterval(59.0, 61.0),),
            (FrequencyInterval(39.0, 41.0),),
        ),
    )


def test_recording_key_accepts_canonical_bids_values_and_is_frozen() -> None:
    key = RecordingKey(
        subject="0001",
        task="thermalactive",
        run="1",
        session="baseline2",
    )

    assert key.subject == "0001"
    assert key.session == "baseline2"
    with pytest.raises(FrozenInstanceError):
        key.run = "2"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["subject", "task", "run", "session"])
@pytest.mark.parametrize("value", ["", " ", "run-1", "run_1", "1.0"])
def test_recording_key_rejects_noncanonical_bids_values(field: str, value: str) -> None:
    values = {"subject": "0001", "task": "thermalactive", "run": "1"}
    values[field] = value

    with pytest.raises(ValueError, match=field):
        RecordingKey(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["subject", "task", "run", "session"])
def test_recording_key_rejects_non_string_values(field: str) -> None:
    values: dict[str, object] = {
        "subject": "0001",
        "task": "thermalactive",
        "run": "1",
    }
    values[field] = 1

    with pytest.raises(TypeError, match=field):
        RecordingKey(**values)  # type: ignore[arg-type]


def test_frequency_interval_accepts_finite_nonnegative_edges_and_is_frozen() -> None:
    interval = FrequencyInterval(59, 61)

    assert interval == FrequencyInterval(59.0, 61.0)
    with pytest.raises(FrozenInstanceError):
        interval.low_hz = 58.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("low_hz", "high_hz"),
    [
        (-1.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (np.nan, 1.0),
        (1.0, np.inf),
    ],
)
def test_frequency_interval_rejects_invalid_geometry(
    low_hz: float,
    high_hz: float,
) -> None:
    with pytest.raises(ValueError):
        FrequencyInterval(low_hz, high_hz)


@pytest.mark.parametrize("edge", [True, "59"])
def test_frequency_interval_rejects_non_real_edges(edge: object) -> None:
    with pytest.raises(TypeError):
        FrequencyInterval(edge, 61.0)  # type: ignore[arg-type]


def test_merge_frequency_intervals_sorts_and_merges_closed_intervals() -> None:
    merged = merge_frequency_intervals(
        (
            FrequencyInterval(61.0, 63.0),
            FrequencyInterval(59.0, 61.0),
            FrequencyInterval(60.0, 62.0),
            FrequencyInterval(59.0, 61.0),
            FrequencyInterval(39.0, 41.0),
        )
    )

    assert merged == (
        FrequencyInterval(39.0, 41.0),
        FrequencyInterval(59.0, 63.0),
    )


def test_merge_frequency_intervals_rejects_non_interval_values() -> None:
    with pytest.raises(TypeError, match="FrequencyInterval"):
        merge_frequency_intervals((object(),))  # type: ignore[arg-type]


def test_recording_exclusions_normalize_intervals() -> None:
    key = RecordingKey(subject="0001", task="thermalactive", run="1")

    exclusions = RecordingExclusions(
        key=key,
        intervals=(
            FrequencyInterval(60.0, 62.0),
            FrequencyInterval(59.0, 61.0),
        ),
    )

    assert exclusions.intervals == (FrequencyInterval(59.0, 62.0),)


def test_epoch_availability_requires_equal_epoch_axes() -> None:
    key = RecordingKey(subject="0001", task="thermalactive", run="1")

    with pytest.raises(ValueError, match="same length"):
        EpochSpectralAvailability(
            recording_keys=(key,),
            exclusions_by_epoch=(),
        )


def test_epoch_availability_normalizes_each_epochs_intervals() -> None:
    key = RecordingKey(subject="0001", task="thermalactive", run="1")

    availability = EpochSpectralAvailability(
        recording_keys=(key,),
        exclusions_by_epoch=(
            (
                FrequencyInterval(60.0, 62.0),
                FrequencyInterval(59.0, 61.0),
            ),
        ),
    )

    assert availability.exclusions_by_epoch == ((FrequencyInterval(59.0, 62.0),),)


def test_valid_frequency_mask_is_recording_specific() -> None:
    valid = _availability().valid_frequency_mask(
        np.array([10.0, 40.0, 60.0]),
        half_support_hz=np.array([0.5, 0.5, 0.5]),
    )

    assert valid.dtype == np.bool_
    assert valid.tolist() == [[True, True, False], [True, False, True]]


def test_valid_frequency_mask_treats_touching_closed_edges_as_invalid() -> None:
    key = RecordingKey(subject="0001", task="thermalactive", run="1")
    availability = EpochSpectralAvailability(
        recording_keys=(key,),
        exclusions_by_epoch=((FrequencyInterval(59.0, 61.0),),),
    )

    valid = availability.valid_frequency_mask(
        np.array([58.0, 58.5, 61.5, 62.0]),
        half_support_hz=0.5,
    )

    assert valid.tolist() == [[True, False, False, True]]


@pytest.mark.parametrize(
    "frequencies",
    [
        np.array([]),
        np.array([[10.0, 20.0]]),
        np.array([10.0, 10.0]),
        np.array([20.0, 10.0]),
        np.array([10.0, np.nan]),
        np.array([10.0, np.inf]),
    ],
)
def test_valid_frequency_mask_rejects_invalid_frequency_grids(
    frequencies: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="centre_frequencies"):
        _availability().valid_frequency_mask(frequencies, half_support_hz=0.0)


@pytest.mark.parametrize(
    "half_support_hz",
    [
        -0.1,
        np.nan,
        np.inf,
        np.array([0.1, 0.2]),
        np.array([[0.1, 0.2, 0.3]]),
        np.array([0.1, np.nan, 0.3]),
    ],
)
def test_valid_frequency_mask_rejects_invalid_half_support(
    half_support_hz: float | np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="half_support_hz"):
        _availability().valid_frequency_mask(
            np.array([10.0, 20.0, 30.0]),
            half_support_hz,
        )


def test_contiguous_band_eligibility_is_recording_specific() -> None:
    availability = _availability()

    assert availability.contiguous_band_eligible(8.0, 13.0).tolist() == [True, True]
    assert availability.contiguous_band_eligible(35.0, 45.0).tolist() == [True, False]
    assert availability.contiguous_band_eligible(61.0, 65.0).tolist() == [False, True]


def test_intersections_returns_intervals_touching_the_closed_band() -> None:
    assert _availability().intersections(41.0, 59.0) == (
        (FrequencyInterval(59.0, 61.0),),
        (FrequencyInterval(39.0, 41.0),),
    )


@pytest.mark.parametrize(
    ("fmin", "fmax"),
    [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0), (1.0, np.inf)],
)
def test_band_operations_reject_invalid_geometry(fmin: float, fmax: float) -> None:
    with pytest.raises(ValueError):
        _availability().contiguous_band_eligible(fmin, fmax)
    with pytest.raises(ValueError):
        _availability().intersections(fmin, fmax)


def test_retained_bandwidth_sums_valid_positive_weights_per_epoch() -> None:
    retained = _availability().retained_bandwidth(
        frequencies=np.array([10.0, 20.0, 40.0, 60.0]),
        weights=np.array([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_array_equal(retained, np.array([6.0, 7.0]))


def test_retained_bandwidth_surfaces_an_epoch_with_zero_support() -> None:
    key = RecordingKey(subject="0001", task="thermalactive", run="1")
    availability = EpochSpectralAvailability(
        recording_keys=(key,),
        exclusions_by_epoch=((FrequencyInterval(0.0, 100.0),),),
    )

    retained = availability.retained_bandwidth(
        frequencies=np.array([10.0, 20.0]),
        weights=np.array([1.0, 2.0]),
    )

    np.testing.assert_array_equal(retained, np.array([0.0]))


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0, 4.0]]),
        np.array([1.0, 0.0, 3.0, 4.0]),
        np.array([1.0, -1.0, 3.0, 4.0]),
        np.array([1.0, np.nan, 3.0, 4.0]),
        np.array([1.0, np.inf, 3.0, 4.0]),
    ],
)
def test_retained_bandwidth_rejects_invalid_weights(weights: np.ndarray) -> None:
    with pytest.raises(ValueError, match="weights"):
        _availability().retained_bandwidth(
            frequencies=np.array([10.0, 20.0, 40.0, 60.0]),
            weights=weights,
        )
