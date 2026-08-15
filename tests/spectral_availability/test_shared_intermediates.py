from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.context.features import FeatureContext
from eeg_pipeline.spectral_availability import (
    EpochSpectralAvailability,
    FrequencyInterval,
    RecordingKey,
    multitaper_half_support,
    welch_half_support,
)
from eeg_pipeline.types import BandData, PrecomputedData, PSDData, TimeWindows
from eeg_pipeline.utils.analysis import spectral
from eeg_pipeline.utils.analysis.spectral import (
    compute_band_data,
    compute_psd,
    compute_psd_bandpower,
)

SFREQ = 200.0
N_TIMES = 512


class _Config:
    def __init__(self, values: dict | None = None) -> None:
        self._values = values or {}

    def get(self, key: str, default=None):
        return self._values.get(key, default)


class _EpochStub:
    def __init__(self, n_epochs: int) -> None:
        self._n_epochs = n_epochs
        self.info = {"sfreq": SFREQ}
        self.times = np.arange(N_TIMES, dtype=float) / SFREQ

    def __len__(self) -> int:
        return self._n_epochs


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


def _signal(n_epochs: int = 2, n_channels: int = 2) -> np.ndarray:
    generator = np.random.default_rng(11)
    return generator.normal(size=(n_epochs, n_channels, N_TIMES))


###################################################################
# Shared containers
###################################################################


def _precomputed(**overrides) -> PrecomputedData:
    times = np.arange(N_TIMES, dtype=float) / SFREQ
    defaults = {
        "data": np.zeros((2, 2, times.size), dtype=float),
        "times": times,
        "sfreq": SFREQ,
        "ch_names": ["C3", "C4"],
        "picks": np.arange(2),
        "windows": TimeWindows(
            masks={"active": np.ones(times.size, dtype=bool)},
            ranges={"active": (float(times[0]), float(times[-1]))},
            times=times,
            name="active",
        ),
    }
    defaults.update(overrides)
    return PrecomputedData(**defaults)


def test_precomputed_data_rejects_availability_with_wrong_epoch_count() -> None:
    with pytest.raises(ValueError, match="spectral_availability"):
        _precomputed(spectral_availability=_availability(()))


def test_precomputed_crop_preserves_availability_and_clears_psd() -> None:
    availability = _availability(((59.0, 61.0),), ((39.0, 41.0),))
    precomputed = _precomputed(spectral_availability=availability)
    precomputed.psd_data = PSDData(
        freqs=np.array([10.0, 11.0]),
        psd=np.zeros((2, 2, 2)),
    )
    precomputed.band_data["alpha"] = BandData(
        band="alpha",
        fmin=8.0,
        fmax=13.0,
        filtered=np.zeros((2, 2, N_TIMES)),
        analytic=np.zeros((2, 2, N_TIMES), dtype=complex),
        envelope=np.zeros((2, 2, N_TIMES)),
        phase=np.zeros((2, 2, N_TIMES)),
        power=np.zeros((2, 2, N_TIMES)),
        eligible_epochs=np.array([True, False]),
    )

    cropped = precomputed.crop(0.1, 0.9)

    assert cropped.spectral_availability is availability
    assert cropped.psd_data is None
    assert cropped.band_data["alpha"].eligible_epochs.tolist() == [True, False]


def test_precomputed_with_windows_preserves_availability() -> None:
    availability = _availability(((59.0, 61.0),), ((39.0, 41.0),))
    precomputed = _precomputed(spectral_availability=availability)

    updated = precomputed.with_windows(precomputed.windows)

    assert updated.spectral_availability is availability


def test_band_data_rejects_eligible_epochs_with_wrong_length() -> None:
    with pytest.raises(ValueError, match="eligible_epochs"):
        BandData(
            band="alpha",
            fmin=8.0,
            fmax=13.0,
            filtered=np.zeros((2, 2, 4)),
            analytic=np.zeros((2, 2, 4), dtype=complex),
            envelope=np.zeros((2, 2, 4)),
            phase=np.zeros((2, 2, 4)),
            power=np.zeros((2, 2, 4)),
            eligible_epochs=np.array([True]),
        )


def test_psd_data_rejects_mask_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match="valid_frequency_mask"):
        PSDData(
            freqs=np.array([10.0, 11.0]),
            psd=np.zeros((2, 2, 2)),
            valid_frequency_mask=np.ones((2, 3), dtype=bool),
        )


def test_feature_context_carries_spectral_availability() -> None:
    availability = _availability(((59.0, 61.0),), ((39.0, 41.0),))

    context = FeatureContext(
        subject="0001",
        task="thermalactive",
        config=_Config(),
        deriv_root=Path("."),
        logger=logging.getLogger("spectral-availability-context"),
        epochs=_EpochStub(2),
        aligned_events=pd.DataFrame({"run_id": [1.0, 2.0]}),
        spectral_availability=availability,
    )

    assert context.spectral_availability is availability


###################################################################
# Welch PSD masking
###################################################################


def test_welch_psd_without_availability_is_unchanged() -> None:
    data = _signal()

    baseline = compute_psd(data, SFREQ)
    repeated = compute_psd(data, SFREQ, spectral_availability=None)

    assert repeated.valid_frequency_mask is None
    assert repeated.half_support_hz is None
    assert np.array_equal(repeated.psd, baseline.psd)
    assert np.array_equal(repeated.freqs, baseline.freqs)


def test_welch_psd_masks_only_unavailable_cells() -> None:
    data = _signal()
    availability = _availability(((59.0, 61.0),), ((39.0, 41.0),))

    baseline = compute_psd(data, SFREQ)
    masked = compute_psd(data, SFREQ, spectral_availability=availability)

    assert np.array_equal(masked.freqs, baseline.freqs)
    n_fft = min(N_TIMES, int(2.0 * SFREQ))
    expected_support = welch_half_support(SFREQ, n_fft, "hann")
    expected_mask = availability.valid_frequency_mask(baseline.freqs, expected_support)

    assert masked.half_support_hz == pytest.approx(expected_support)
    assert masked.valid_frequency_mask.shape == (2, baseline.freqs.size)
    assert np.array_equal(masked.valid_frequency_mask, expected_mask)

    valid = np.broadcast_to(expected_mask[:, None, :], masked.psd.shape)
    assert np.all(np.isnan(masked.psd[~valid]))
    assert np.array_equal(masked.psd[valid], baseline.psd[valid])


def test_welch_psd_rejects_availability_with_wrong_epoch_count() -> None:
    data = _signal()

    with pytest.raises(ValueError, match="epoch"):
        compute_psd(data, SFREQ, spectral_availability=_availability(((59.0, 61.0),)))


###################################################################
# Retained-bandwidth band power
###################################################################

_STUB_FREQS = np.array([8.0, 9.0, 10.0, 12.0, 13.0])


def _stub_multitaper(monkeypatch: pytest.MonkeyPatch, n_epochs: int = 2) -> None:
    psds = np.broadcast_to(
        _STUB_FREQS,
        (n_epochs, 1, _STUB_FREQS.size),
    ).astype(float)

    def _fake(data, **kwargs):
        return psds.copy(), _STUB_FREQS.copy()

    monkeypatch.setattr(spectral, "psd_array_multitaper", _fake)


def test_bandpower_without_availability_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_multitaper(monkeypatch)
    data = _signal(n_channels=1)

    result = compute_psd_bandpower(
        data,
        SFREQ,
        {"alpha": (8.0, 13.0)},
        bandwidth=2.0,
    )

    weights = np.gradient(_STUB_FREQS)
    expected = float(np.sum(_STUB_FREQS * weights) / np.sum(weights))
    assert result["alpha"] == pytest.approx(np.full((2, 1), expected))


def test_bandpower_normalizes_by_retained_width_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_multitaper(monkeypatch)
    data = _signal(n_channels=1)
    availability = _availability((), ((11.5, 12.5),))

    result = compute_psd_bandpower(
        data,
        SFREQ,
        {"alpha": (8.0, 13.0)},
        bandwidth=2.0,
        spectral_availability=availability,
    )

    assert multitaper_half_support(2.0) == pytest.approx(1.0)
    # Epoch 2 loses the 12 Hz and 13 Hz bins to the +/- 1 Hz multitaper support.
    assert result["alpha"][0, 0] == pytest.approx(63.0 / 6.0)
    assert result["alpha"][1, 0] == pytest.approx(32.0 / 3.5)


def test_bandpower_combines_configured_line_mask_with_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_multitaper(monkeypatch)
    data = _signal(n_channels=1)
    availability = _availability((), ((11.5, 12.5),))

    result = compute_psd_bandpower(
        data,
        SFREQ,
        {"alpha": (8.0, 13.0)},
        bandwidth=2.0,
        exclude_line_noise=True,
        line_freqs=[10.0],
        line_width=0.5,
        n_harmonics=1,
        spectral_availability=availability,
    )

    assert result["alpha"][0, 0] == pytest.approx(48.0 / 4.5)
    assert result["alpha"][1, 0] == pytest.approx(17.0 / 2.0)


def test_bandpower_marks_exhausted_epoch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_multitaper(monkeypatch)
    data = _signal(n_channels=1)
    availability = _availability((), ((7.0, 14.0),))

    result = compute_psd_bandpower(
        data,
        SFREQ,
        {"alpha": (8.0, 13.0)},
        bandwidth=2.0,
        spectral_availability=availability,
    )

    assert result["alpha"][0, 0] == pytest.approx(63.0 / 6.0)
    assert np.isnan(result["alpha"][1, 0])


def test_bandpower_raises_when_no_epoch_retains_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_multitaper(monkeypatch)
    data = _signal(n_channels=1)
    availability = _availability(((7.0, 14.0),), ((7.0, 14.0),))

    with pytest.raises(ValueError, match="alpha"):
        compute_psd_bandpower(
            data,
            SFREQ,
            {"alpha": (8.0, 13.0)},
            bandwidth=2.0,
            spectral_availability=availability,
        )


def test_bandpower_rejects_availability_with_wrong_epoch_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_multitaper(monkeypatch)
    data = _signal(n_channels=1)

    with pytest.raises(ValueError, match="epoch"):
        compute_psd_bandpower(
            data,
            SFREQ,
            {"alpha": (8.0, 13.0)},
            bandwidth=2.0,
            spectral_availability=_availability(((7.0, 14.0),)),
        )


###################################################################
# Contiguous band data
###################################################################


def test_band_data_without_availability_is_unchanged() -> None:
    data = _signal()

    baseline = compute_band_data(data, SFREQ, "alpha", 8.0, 13.0)
    repeated = compute_band_data(data, SFREQ, "alpha", 8.0, 13.0, spectral_availability=None)

    assert repeated.eligible_epochs is None
    assert np.array_equal(repeated.filtered, baseline.filtered)


def test_band_data_matches_unmasked_when_every_epoch_is_eligible() -> None:
    data = _signal()
    availability = _availability(((59.0, 61.0),), ((39.0, 41.0),))

    baseline = compute_band_data(data, SFREQ, "alpha", 8.0, 13.0)
    filtered = compute_band_data(
        data,
        SFREQ,
        "alpha",
        8.0,
        13.0,
        spectral_availability=availability,
    )

    assert filtered.eligible_epochs.tolist() == [True, True]
    assert np.array_equal(filtered.filtered, baseline.filtered)
    assert np.array_equal(filtered.envelope, baseline.envelope)


def test_band_data_leaves_ineligible_epochs_unavailable() -> None:
    data = _signal()
    availability = _availability(((59.0, 61.0),), ((9.0, 11.0),))

    baseline = compute_band_data(data, SFREQ, "alpha", 8.0, 13.0)
    filtered = compute_band_data(
        data,
        SFREQ,
        "alpha",
        8.0,
        13.0,
        spectral_availability=availability,
    )

    assert filtered.eligible_epochs.tolist() == [True, False]
    assert np.array_equal(filtered.filtered[0], baseline.filtered[0])
    assert np.array_equal(filtered.envelope[0], baseline.envelope[0])
    for array in (
        filtered.filtered,
        filtered.analytic,
        filtered.envelope,
        filtered.phase,
        filtered.power,
    ):
        assert np.all(np.isnan(array[1]))


def test_band_data_raises_when_no_epoch_is_eligible() -> None:
    data = _signal()
    availability = _availability(((9.0, 11.0),), ((9.0, 11.0),))

    with pytest.raises(ValueError, match="alpha"):
        compute_band_data(
            data,
            SFREQ,
            "alpha",
            8.0,
            13.0,
            spectral_availability=availability,
        )


def test_band_data_rejects_availability_with_wrong_epoch_count() -> None:
    data = _signal()

    with pytest.raises(ValueError, match="epoch"):
        compute_band_data(
            data,
            SFREQ,
            "alpha",
            8.0,
            13.0,
            spectral_availability=_availability(((59.0, 61.0),)),
        )


###################################################################
# Threading through precomputation
###################################################################


def test_precompute_threads_availability_into_shared_intermediates() -> None:
    import mne

    from eeg_pipeline.analysis.features.preparation import precompute_data

    info = mne.create_info(["C3", "C4"], SFREQ, ch_types="eeg")
    epochs = mne.EpochsArray(_signal(), info, verbose=False)
    availability = _availability(((59.0, 61.0),), ((9.0, 11.0),))
    config = _Config(
        {
            "feature_engineering.task_is_rest": True,
            "preprocessing.task_is_rest": True,
        }
    )

    precomputed = precompute_data(
        epochs,
        ["alpha"],
        config,
        logging.getLogger("spectral-availability-precompute"),
        frequency_bands_override={"alpha": [8.0, 13.0]},
        spectral_availability=availability,
    )

    assert precomputed.spectral_availability is availability
    assert precomputed.band_data["alpha"].eligible_epochs.tolist() == [True, False]
    assert precomputed.psd_data.valid_frequency_mask is not None
