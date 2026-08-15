import numpy as np
import pytest

import eeg_pipeline.spectral_availability.estimators as estimator_support
from eeg_pipeline.spectral_availability import (
    morlet_half_support,
    multitaper_half_support,
    multitaper_tfr_half_support,
    welch_half_support,
)


def test_multitaper_half_support_is_half_the_full_bandwidth() -> None:
    assert multitaper_half_support(2.0) == pytest.approx(1.0)


@pytest.mark.parametrize("full_bandwidth_hz", [0.0, -1.0, np.nan, np.inf])
def test_multitaper_half_support_rejects_invalid_bandwidth(
    full_bandwidth_hz: float,
) -> None:
    with pytest.raises(ValueError):
        multitaper_half_support(full_bandwidth_hz)


@pytest.mark.parametrize("full_bandwidth_hz", [True, "2"])
def test_multitaper_half_support_rejects_non_real_bandwidth(
    full_bandwidth_hz: object,
) -> None:
    with pytest.raises(TypeError):
        multitaper_half_support(full_bandwidth_hz)  # type: ignore[arg-type]


def test_morlet_half_support_uses_gaussian_power_half_height() -> None:
    support = morlet_half_support(
        frequencies=np.array([10.0]),
        n_cycles=np.array([5.0]),
    )

    expected = np.array([2.0 * np.sqrt(np.log(2.0))])
    assert support == pytest.approx(expected)


@pytest.mark.parametrize(
    ("frequencies", "n_cycles"),
    [
        (np.array([]), np.array([])),
        (np.array([[10.0]]), np.array([5.0])),
        (np.array([10.0, 20.0]), np.array([5.0])),
        (np.array([0.0]), np.array([5.0])),
        (np.array([-10.0]), np.array([5.0])),
        (np.array([np.nan]), np.array([5.0])),
        (np.array([np.inf]), np.array([5.0])),
        (np.array([10.0]), np.array([0.0])),
        (np.array([10.0]), np.array([-5.0])),
        (np.array([10.0]), np.array([np.nan])),
        (np.array([10.0]), np.array([np.inf])),
    ],
)
def test_morlet_half_support_rejects_invalid_arrays(
    frequencies: np.ndarray,
    n_cycles: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        morlet_half_support(frequencies, n_cycles)


def test_multitaper_tfr_half_support_uses_time_bandwidth_geometry() -> None:
    support = multitaper_tfr_half_support(
        frequencies=np.array([10.0]),
        n_cycles=np.array([5.0]),
        time_bandwidth=4.0,
    )

    assert support == pytest.approx(np.array([4.0]))


@pytest.mark.parametrize("time_bandwidth", [0.0, -1.0, 1.99, np.nan, np.inf])
def test_multitaper_tfr_half_support_rejects_invalid_time_bandwidth(
    time_bandwidth: float,
) -> None:
    with pytest.raises(ValueError):
        multitaper_tfr_half_support(
            frequencies=np.array([10.0]),
            n_cycles=np.array([5.0]),
            time_bandwidth=time_bandwidth,
        )


def test_multitaper_tfr_half_support_rejects_invalid_frequency_geometry() -> None:
    with pytest.raises(ValueError):
        multitaper_tfr_half_support(
            frequencies=np.array([10.0, 20.0]),
            n_cycles=np.array([5.0]),
            time_bandwidth=4.0,
        )


def test_welch_hann_half_support_is_sub_hertz_for_requested_geometry() -> None:
    support = welch_half_support(sfreq=500.0, n_per_seg=1000, window="hann")

    assert 0.0 < support < 1.0


@pytest.mark.parametrize(
    ("window", "expected_half_width_bins"),
    [
        ("boxcar", 0.44294647),
        ("hann", 0.72029100),
    ],
)
def test_welch_half_support_matches_known_periodic_window_half_widths(
    window: str,
    expected_half_width_bins: float,
) -> None:
    sfreq = 1024.0
    n_per_seg = 1024

    support_hz = welch_half_support(sfreq, n_per_seg, window)
    normalized_support = support_hz / (sfreq / n_per_seg)

    assert normalized_support == pytest.approx(
        expected_half_width_bins,
        rel=5e-5,
    )


def test_welch_half_support_accepts_equivalent_kaiser_window_forms() -> None:
    tuple_support = welch_half_support(500.0, 1000, ("kaiser", 8.0))
    float_support = welch_half_support(500.0, 1000, 8.0)

    assert float_support == pytest.approx(tuple_support, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("sfreq", [0.0, -500.0, np.nan, np.inf])
def test_welch_half_support_rejects_invalid_sampling_frequency(sfreq: float) -> None:
    with pytest.raises(ValueError):
        welch_half_support(sfreq=sfreq, n_per_seg=1000, window="hann")


@pytest.mark.parametrize("n_per_seg", [1, 0, -1])
def test_welch_half_support_rejects_fewer_than_two_samples(n_per_seg: int) -> None:
    with pytest.raises(ValueError):
        welch_half_support(sfreq=500.0, n_per_seg=n_per_seg, window="hann")


@pytest.mark.parametrize("n_per_seg", [True, 1000.0])
def test_welch_half_support_rejects_non_integer_segment_lengths(
    n_per_seg: object,
) -> None:
    with pytest.raises(TypeError):
        welch_half_support(
            sfreq=500.0,
            n_per_seg=n_per_seg,  # type: ignore[arg-type]
            window="hann",
        )


def test_welch_half_support_rejects_unsupported_windows() -> None:
    with pytest.raises(ValueError):
        welch_half_support(
            sfreq=500.0,
            n_per_seg=1000,
            window="not-a-scipy-window",
        )


def test_welch_half_support_rejects_finite_window_with_zero_dc_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        estimator_support,
        "get_window",
        lambda window, n_per_seg, *, fftbins: np.array([1.0, -1.0]),
    )

    with pytest.raises(
        ValueError,
        match="^window has no finite positive zero-frequency power$",
    ):
        welch_half_support(500.0, 2, "ignored")


def test_welch_half_support_rejects_window_without_half_power_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        estimator_support,
        "get_window",
        lambda window, n_per_seg, *, fftbins: np.array([1.0, 0.0]),
    )

    with pytest.raises(
        ValueError,
        match="^window has no measurable half-power main-lobe crossing$",
    ):
        welch_half_support(500.0, 2, "ignored")
