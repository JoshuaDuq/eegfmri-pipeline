from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.preprocessing.band_ica_report import (
    BandIcaDefinition,
    BandIcaReportSettings,
    _fieldtrip_tfr,
)
from eeg_pipeline.spectral_availability import (
    EpochSpectralAvailability,
    FrequencyInterval,
    RecordingKey,
)

SFREQ = 200.0
GAMMA = BandIcaDefinition("gamma", "Gamma (30-100 Hz)", 30.0, 100.0)


def _settings() -> BandIcaReportSettings:
    return BandIcaReportSettings(
        time_min_s=-2.0,
        time_max_s=2.0,
        time_step_s=0.1,
        baseline_tmin_s=-2.0,
        baseline_tmax_s=-1.5,
    )


def _data(n_epochs: int = 4, n_components: int = 2, n_times: int = 1200) -> np.ndarray:
    generator = np.random.default_rng(3)
    return generator.normal(size=(n_epochs, n_components, n_times))


def _times(n_times: int = 1200) -> np.ndarray:
    return np.arange(n_times, dtype=float) / SFREQ - 3.0


def _availability(*exclusions: tuple[tuple[float, float], ...]) -> EpochSpectralAvailability:
    """One recording per supplied exclusion tuple, two epochs each."""
    keys = []
    per_epoch = []
    for index, intervals in enumerate(exclusions):
        key = RecordingKey(subject="0001", task="thermalactive", run=str(index + 1))
        for _ in range(2):
            keys.append(key)
            per_epoch.append(tuple(FrequencyInterval(lo, hi) for lo, hi in intervals))
    return EpochSpectralAvailability(
        recording_keys=tuple(keys),
        exclusions_by_epoch=tuple(per_epoch),
    )


def _run(availability=None):
    return _fieldtrip_tfr(
        data=_data(),
        sfreq=SFREQ,
        times=_times(),
        band=GAMMA,
        settings=_settings(),
        epoch_availability=availability,
    )


def test_without_availability_the_panel_is_unchanged() -> None:
    baseline_freqs, baseline_times, baseline_power, baseline_counts = _run()
    repeat_freqs, repeat_times, repeat_power, repeat_counts = _run(availability=None)

    assert baseline_counts is None
    assert repeat_counts is None
    assert np.array_equal(baseline_freqs, repeat_freqs)
    assert np.array_equal(baseline_times, repeat_times)
    assert np.array_equal(baseline_power, repeat_power)
    assert np.all(np.isfinite(baseline_power))


def test_availability_with_no_exclusions_matches_the_unmasked_panel() -> None:
    _f, _t, baseline_power, _c = _run()
    freqs, _times_out, power, counts = _run(availability=_availability((), ()))

    assert counts.tolist() == [4] * freqs.size
    assert np.allclose(power, baseline_power)


def test_a_frequency_excluded_in_one_recording_averages_only_the_other() -> None:
    # Run 2 loses 60 Hz; run 1 keeps it. The +/- 5 Hz gamma smoothing widens the reach.
    freqs, _times_out, power, counts = _run(availability=_availability((), ((59.0, 61.0),)))

    reached = (freqs + 5.0 >= 59.0) & (freqs - 5.0 <= 61.0)
    assert np.any(reached)
    assert np.all(counts[reached] == 2)
    assert np.all(counts[~reached] == 4)
    assert np.all(np.isfinite(power[:, reached, :]))


def test_a_frequency_excluded_everywhere_stays_unavailable() -> None:
    freqs, _times_out, power, counts = _run(
        availability=_availability(((59.0, 61.0),), ((59.0, 61.0),))
    )

    reached = (freqs + 5.0 >= 59.0) & (freqs - 5.0 <= 61.0)
    assert np.all(counts[reached] == 0)
    assert np.all(np.isnan(power[:, reached, :]))
    assert np.all(np.isfinite(power[:, ~reached, :]))


def test_partial_exclusion_equals_the_average_of_the_eligible_recordings() -> None:
    availability = _availability((), ((59.0, 61.0),))
    freqs, _t, masked_power, _counts = _run(availability=availability)

    # The eligible-only average must equal a panel built from run 1's epochs alone.
    first_run_only = _fieldtrip_tfr(
        data=_data()[:2],
        sfreq=SFREQ,
        times=_times(),
        band=GAMMA,
        settings=_settings(),
        epoch_availability=None,
    )[2]

    reached = (freqs + 5.0 >= 59.0) & (freqs - 5.0 <= 61.0)
    assert np.allclose(masked_power[:, reached, :], first_run_only[:, reached, :])


def test_availability_must_cover_every_epoch() -> None:
    with pytest.raises(ValueError, match="epoch"):
        _run(availability=_availability(()))
