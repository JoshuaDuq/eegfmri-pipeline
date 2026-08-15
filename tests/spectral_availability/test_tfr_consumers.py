from __future__ import annotations

import mne
import numpy as np
import pytest
from mne.time_frequency import AverageTFRArray, EpochsTFRArray

from eeg_pipeline.spectral_availability import (
    EpochSpectralAvailability,
    FrequencyInterval,
    RecordingKey,
    morlet_half_support,
)
from eeg_pipeline.utils.analysis.tfr import (
    apply_tfr_availability,
    store_tfr_geometry,
    tfr_geometry,
)
from eeg_pipeline.utils.config.loader import ensure_config

SFREQ = 200.0
FREQS = np.array([8.0, 10.0, 40.0, 60.0])
N_CYCLES = np.array([3.0, 3.0, 5.0, 7.0])


def _availability(*exclusions: tuple[tuple[float, float], ...]) -> EpochSpectralAvailability:
    return EpochSpectralAvailability(
        recording_keys=tuple(
            RecordingKey(subject="0001", task="thermalactive", run=str(index + 1))
            for index in range(len(exclusions))
        ),
        exclusions_by_epoch=tuple(
            tuple(FrequencyInterval(low, high) for low, high in intervals)
            for intervals in exclusions
        ),
    )


def _epochs_tfr(n_epochs: int = 2, n_channels: int = 2, n_times: int = 16) -> EpochsTFRArray:
    info = mne.create_info([f"E{index}" for index in range(n_channels)], SFREQ, ch_types="eeg")
    data = np.arange(
        n_epochs * n_channels * FREQS.size * n_times,
        dtype=float,
    ).reshape(n_epochs, n_channels, FREQS.size, n_times)
    times = np.arange(n_times, dtype=float) / SFREQ
    tfr = EpochsTFRArray(info, data, times, FREQS, method="morlet")
    store_tfr_geometry(tfr, FREQS, N_CYCLES)
    return tfr


def test_tfr_geometry_round_trips_the_exact_morlet_parameters() -> None:
    tfr = _epochs_tfr()

    freqs, n_cycles = tfr_geometry(tfr)

    assert np.array_equal(freqs, FREQS)
    assert np.array_equal(n_cycles, N_CYCLES)


def test_tfr_geometry_rejects_a_tfr_without_stored_parameters() -> None:
    info = mne.create_info(["E0"], SFREQ, ch_types="eeg")
    data = np.zeros((1, 1, FREQS.size, 8))
    tfr = EpochsTFRArray(info, data, np.arange(8) / SFREQ, FREQS, method="morlet")

    with pytest.raises(ValueError, match="n_cycles"):
        tfr_geometry(tfr)


def test_tfr_geometry_recomputes_cycles_after_mne_rebuilds_the_object() -> None:
    from eeg_pipeline.utils.analysis.tfr import compute_adaptive_n_cycles

    config = ensure_config(None)
    tfr = _epochs_tfr()
    store_tfr_geometry(tfr, FREQS, compute_adaptive_n_cycles(FREQS, config=config))

    copied = tfr.copy().crop(tmin=0.01, tmax=0.05)
    freqs, n_cycles = tfr_geometry(copied, config=config)

    assert np.array_equal(freqs, FREQS)
    assert np.array_equal(n_cycles, compute_adaptive_n_cycles(FREQS, config=config))


def test_apply_tfr_availability_without_availability_is_a_no_op() -> None:
    tfr = _epochs_tfr()
    baseline = tfr.data.copy()

    assert apply_tfr_availability(tfr, None) is None
    assert np.array_equal(tfr.data, baseline)


def test_apply_tfr_availability_masks_only_unavailable_cells() -> None:
    tfr = _epochs_tfr()
    baseline = tfr.data.copy()
    availability = _availability(((59.0, 61.0),), ((39.0, 41.0),))

    result = apply_tfr_availability(tfr, availability)

    expected_support = morlet_half_support(FREQS, N_CYCLES)
    expected_mask = availability.valid_frequency_mask(FREQS, expected_support)

    assert np.array_equal(result.half_support_hz, expected_support)
    assert np.array_equal(result.valid_frequency_mask, expected_mask)
    assert result.valid_frequency_mask.tolist() == [
        [True, True, True, False],
        [True, True, False, True],
    ]
    assert result.eligible_epoch_counts.tolist() == [2, 2, 1, 1]

    assert tfr.data.shape == baseline.shape
    valid = np.broadcast_to(expected_mask[:, None, :, None], tfr.data.shape)
    assert np.all(np.isnan(tfr.data[~valid]))
    assert np.array_equal(tfr.data[valid], baseline[valid])


def test_apply_tfr_availability_masks_complex_tfr_data() -> None:
    tfr = _epochs_tfr()
    tfr.data = tfr.data.astype(complex)
    availability = _availability(((59.0, 61.0),), ((39.0, 41.0),))

    apply_tfr_availability(tfr, availability)

    assert np.iscomplexobj(tfr.data)
    assert np.all(np.isnan(tfr.data[0, :, 3, :]))
    assert np.all(np.isfinite(tfr.data[0, :, 2, :]))


def test_apply_tfr_availability_rejects_averaged_tfr_data() -> None:
    info = mne.create_info(["E0", "E1"], SFREQ, ch_types="eeg")
    data = np.zeros((2, FREQS.size, 8))
    tfr = AverageTFRArray(info, data, np.arange(8) / SFREQ, FREQS, nave=2, method="morlet")
    store_tfr_geometry(tfr, FREQS, N_CYCLES)

    with pytest.raises(ValueError, match="epoch"):
        apply_tfr_availability(tfr, _availability(((59.0, 61.0),), ((39.0, 41.0),)))


def test_apply_tfr_availability_rejects_epoch_count_mismatch() -> None:
    tfr = _epochs_tfr()

    with pytest.raises(ValueError, match="epoch"):
        apply_tfr_availability(tfr, _availability(((59.0, 61.0),)))


def test_apply_tfr_availability_leaves_a_fully_excluded_frequency_unavailable() -> None:
    tfr = _epochs_tfr()
    availability = _availability(((59.0, 61.0),), ((59.0, 61.0),))

    result = apply_tfr_availability(tfr, availability)

    assert result.eligible_epoch_counts.tolist() == [2, 2, 2, 0]
    assert np.all(np.isnan(tfr.data[:, :, 3, :]))
    assert np.all(np.isfinite(tfr.data[:, :, :3, :]))
