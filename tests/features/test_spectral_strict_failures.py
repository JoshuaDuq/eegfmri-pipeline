from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.utils.analysis.spectral import (
    bandpass_filter_epochs,
    compute_band_data,
    compute_psd,
    compute_psd_bandpower,
)


def test_compute_band_data_raises_on_invalid_shape() -> None:
    data = np.ones((2, 128), dtype=float)

    with pytest.raises(ValueError, match="Expected 3D data"):
        compute_band_data(data, sfreq=100.0, band="alpha", fmin=8.0, fmax=12.0)


def test_compute_psd_raises_when_epoch_is_too_short() -> None:
    data = np.ones((2, 3, 12), dtype=float)

    with pytest.raises(ValueError, match="PSD requires at least"):
        compute_psd(data, sfreq=100.0, min_samples=64)


def test_compute_psd_bandpower_raises_when_epoch_is_too_short() -> None:
    data = np.ones((2, 3, 12), dtype=float)

    with pytest.raises(ValueError, match="PSD bandpower requires at least"):
        compute_psd_bandpower(
            data,
            sfreq=100.0,
            band_ranges={"alpha": (8.0, 12.0)},
        )


def test_bandpass_filter_epochs_raises_on_invalid_frequency_range() -> None:
    data = np.ones((2, 3, 128), dtype=float)

    with pytest.raises(ValueError, match="Invalid frequency range"):
        bandpass_filter_epochs(data, sfreq=100.0, fmin=30.0, fmax=10.0)
