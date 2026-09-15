"""Windowed power must come from inside the window it is named after."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from eeg_pipeline.analysis.features.spectral import (  # noqa: E402
    _compute_frequency_weighted_power,
)
from eeg_pipeline.spectral_availability import morlet_temporal_half_support  # noqa: E402
from eeg_pipeline.utils.analysis.tfr import (  # noqa: E402
    compute_tfr_morlet,
    support_restricted_time_mask,
    tfr_geometry,
)
from tests.utils.pipelines_test_utils import DotConfig  # noqa: E402

SFREQ = 500.0
EPOCH_TMIN = -7.0
EPOCH_TMAX = 15.0
ALPHA_HZ = 9.4267


def _config() -> DotConfig:
    return DotConfig(
        {
            "time_frequency_analysis": {
                "tfr": {
                    "freq_min": 8.0,
                    "freq_max": 12.9,
                    "n_freqs": 4,
                    "n_cycles_factor": 2.0,
                    "min_cycles": 3.0,
                    "max_cycles": 1000.0,
                    "decim_power": 1,
                    "picks": "eeg",
                    "workers": 1,
                }
            }
        }
    )


def _epochs_with_step(post_gain: float, step_time_s: float) -> "mne.EpochsArray":
    """One stationary alpha tone whose amplitude changes only at ``step_time_s``."""
    times = np.arange(EPOCH_TMIN, EPOCH_TMAX, 1.0 / SFREQ)
    envelope = np.where(times >= step_time_s, post_gain, 1.0)
    data = (np.sin(2.0 * np.pi * ALPHA_HZ * times) * envelope)[None, None, :] * 1e-6
    info = mne.create_info(["Cz"], SFREQ, "eeg")
    return mne.EpochsArray(data, info, tmin=EPOCH_TMIN, verbose=False)


def _window_power_db(
    *,
    post_gain: float,
    step_time_s: float,
    window: tuple[float, float],
    restrict: bool,
) -> float:
    config = _config()
    tfr = compute_tfr_morlet(
        _epochs_with_step(post_gain, step_time_s),
        config,
        freqs=np.array([ALPHA_HZ]),
    )
    times = np.asarray(tfr.times)
    freqs, n_cycles = tfr_geometry(tfr, config=config)

    if restrict:
        time_mask = support_restricted_time_mask(
            times=times,
            window=window,
            freqs=freqs,
            n_cycles=n_cycles,
        ).time_mask
    else:
        time_mask = (times >= window[0]) & (times < window[1])

    power = _compute_frequency_weighted_power(
        np.asarray(tfr.data),
        np.ones(freqs.size, dtype=bool),
        time_mask,
        freqs,
    )
    return 10.0 * np.log10(float(power[0, 0]))


@pytest.mark.parametrize(
    ("window", "step_time_s"),
    [
        ((-5.0, -0.01), 0.0),  # wide prestimulus vs stimulus onset
        ((0.0, 3.0), 3.0),  # ramp-up vs plateau onset
    ],
)
def test_window_power_ignores_activity_outside_the_window(
    window: tuple[float, float],
    step_time_s: float,
) -> None:
    """A tenfold rise strictly outside the window must not move the window's power.

    Selecting coefficients by time after a full-epoch transform does not restrict what
    those coefficients were computed from, so without support restriction this leaks.
    """
    flat = _window_power_db(post_gain=1.0, step_time_s=step_time_s, window=window, restrict=True)
    stepped = _window_power_db(
        post_gain=10.0, step_time_s=step_time_s, window=window, restrict=True
    )

    assert abs(stepped - flat) < 0.01

    leaked_flat = _window_power_db(
        post_gain=1.0, step_time_s=step_time_s, window=window, restrict=False
    )
    leaked_stepped = _window_power_db(
        post_gain=10.0, step_time_s=step_time_s, window=window, restrict=False
    )
    assert leaked_stepped - leaked_flat > 0.5


def test_a_window_narrower_than_the_kernel_is_left_unmeasured() -> None:
    """Alpha cannot be read from 0.19 s of data, and a number here would be a fiction."""
    times = np.arange(EPOCH_TMIN, EPOCH_TMAX, 1.0 / SFREQ)
    freqs = np.array([ALPHA_HZ])
    n_cycles = freqs / 2.0

    support = support_restricted_time_mask(
        times=times,
        window=(-0.2, -0.01),
        freqs=freqs,
        n_cycles=n_cycles,
    )

    assert bool(support.unmeasurable_frequencies[0])
    assert not support.time_mask.any()

    value = _compute_frequency_weighted_power(
        np.ones((1, 1, 1, times.size)),
        np.ones(1, dtype=bool),
        support.time_mask,
        freqs,
    )
    assert np.isnan(value[0, 0])


def test_half_support_matches_the_kernel_mne_actually_builds() -> None:
    freqs = np.array([ALPHA_HZ, 30.0, 60.0])
    n_cycles = freqs / 2.0

    expected = morlet_temporal_half_support(freqs, n_cycles)
    for freq, cycles, half in zip(freqs, n_cycles, expected):
        wavelet = mne.time_frequency.morlet(sfreq=SFREQ, freqs=freq, n_cycles=cycles)
        built_half_s = (len(wavelet) - 1) / 2.0 / SFREQ
        assert abs(built_half_s - half) <= 1.0 / SFREQ


def test_additional_half_support_widens_the_excluded_margin() -> None:
    times = np.arange(-5.0, 5.0, 1.0 / SFREQ)
    freqs = np.array([ALPHA_HZ])
    n_cycles = freqs / 2.0

    without = support_restricted_time_mask(
        times=times, window=(-3.0, 3.0), freqs=freqs, n_cycles=n_cycles
    )
    with_filter = support_restricted_time_mask(
        times=times,
        window=(-3.0, 3.0),
        freqs=freqs,
        n_cycles=n_cycles,
        additional_half_support_s=0.5,
    )

    assert with_filter.time_mask.sum() < without.time_mask.sum()
    assert with_filter.half_support_s[0] == pytest.approx(without.half_support_s[0] + 0.5)


def test_extract_power_features_restricts_the_window_it_is_given(monkeypatch) -> None:
    """The restriction has to reach the public extraction path, not just the helper."""
    from eeg_pipeline.analysis.features import spectral

    seen: list = []
    original = spectral._restrict_time_mask_to_window_support

    def _observe(**kwargs):
        mask, support = original(**kwargs)
        seen.append((kwargs["segment_name"], support))
        return mask, support

    monkeypatch.setattr(spectral, "_restrict_time_mask_to_window_support", _observe)

    config = _config()
    config["feature_engineering"] = {"power": {"require_baseline": False}}
    config["frequency_bands"] = {"alpha": [8.0, 12.9]}
    config["rois"] = {}

    tfr = compute_tfr_morlet(_epochs_with_step(1.0, 0.0), config, freqs=np.array([ALPHA_HZ]))

    class _Windows:
        ranges = {"active": (3.0, 10.5)}

    ctx = SimpleNamespace(
        results={"tfr": tfr},
        config=config,
        windows=_Windows(),
        logger=logging.getLogger(__name__),
        spatial_modes=["global"],
        frequency_bands={"alpha": [8.0, 12.9]},
        name="active",
        baseline_df=None,
    )

    features, columns = spectral.extract_power_features(ctx, ["alpha"])

    assert columns
    assert seen and seen[0][0] == "active"
    support = seen[0][1]
    assert support is not None
    assert support.half_support_s[0] == pytest.approx(0.39788736, abs=1e-6)
    assert features.attrs["support_restricted_window"] is True
