"""The aperiodic fit must recover a known slope and refuse an undefined one."""

from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.preprocessing.report.aperiodic import (
    MINIMUM_FIT_BINS,
    aperiodic_line_db,
    fit_aperiodic,
)


def _spectrum(exponent: float, offset_db: float = -120.0, fmax: float = 60.0):
    frequencies = np.arange(1.0, fmax, 0.25)
    power_db = offset_db - 10.0 * exponent * np.log10(frequencies)
    return frequencies, power_db


def test_fit_recovers_a_known_exponent() -> None:
    frequencies, power_db = _spectrum(exponent=1.7)

    fit = fit_aperiodic(frequencies, power_db)

    assert fit is not None
    assert fit.exponent == pytest.approx(1.7, abs=0.01)
    assert fit.r_squared > 0.999


def test_fit_recovers_the_offset_at_one_hertz() -> None:
    frequencies, power_db = _spectrum(exponent=1.0, offset_db=-95.0)

    fit = fit_aperiodic(frequencies, power_db)

    assert fit is not None
    assert fit.offset_db == pytest.approx(-95.0, abs=0.1)
    assert aperiodic_line_db(fit, np.array([1.0]))[0] == pytest.approx(-95.0, abs=0.1)


def test_an_oscillatory_peak_does_not_tilt_the_slope() -> None:
    """A large alpha bump must be excluded, or the fitted background follows it."""
    frequencies, power_db = _spectrum(exponent=1.5)
    alpha = (frequencies >= 8.0) & (frequencies <= 12.0)
    contaminated = power_db.copy()
    contaminated[alpha] += 12.0

    fit = fit_aperiodic(frequencies, contaminated)

    assert fit is not None
    assert fit.exponent == pytest.approx(1.5, abs=0.1)
    assert fit.n_bins_used < fit.n_bins_available


def test_a_notch_can_be_excluded_by_window() -> None:
    """A notch is a downward deviation, which peak removal is not designed to catch."""
    frequencies, power_db = _spectrum(exponent=1.0, fmax=80.0)
    notch = (frequencies >= 58.0) & (frequencies <= 62.0)
    contaminated = power_db.copy()
    contaminated[notch] -= 25.0

    without = fit_aperiodic(frequencies, contaminated, fit_range_hz=(2.0, 75.0))
    excluded = fit_aperiodic(
        frequencies,
        contaminated,
        fit_range_hz=(2.0, 75.0),
        excluded_windows=((58.0, 62.0),),
    )

    assert without is not None and excluded is not None
    assert excluded.exponent == pytest.approx(1.0, abs=0.01)
    # The notch drags the uncorrected fit away from the truth it would otherwise hit.
    assert abs(without.exponent - 1.0) > abs(excluded.exponent - 1.0)


def test_too_few_bins_returns_no_fit_rather_than_a_fabricated_one() -> None:
    frequencies = np.linspace(2.0, 3.0, MINIMUM_FIT_BINS - 1)
    power_db = np.full(frequencies.shape, -100.0)

    assert fit_aperiodic(frequencies, power_db) is None


def test_a_reversed_fit_range_is_rejected() -> None:
    frequencies, power_db = _spectrum(exponent=1.0)

    with pytest.raises(ValueError, match="empty or reversed"):
        fit_aperiodic(frequencies, power_db, fit_range_hz=(40.0, 10.0))


def test_mismatched_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="matching frequency and power"):
        fit_aperiodic(np.arange(10.0), np.arange(9.0))


def test_the_line_is_undefined_at_zero_frequency() -> None:
    frequencies, power_db = _spectrum(exponent=1.0)
    fit = fit_aperiodic(frequencies, power_db)

    with pytest.raises(ValueError, match="undefined at or below zero"):
        aperiodic_line_db(fit, np.array([0.0, 10.0]))


def test_a_notch_stopband_does_not_drag_the_slope() -> None:
    """The fit trims oscillatory peaks from the upper tail of its residuals. Keeping
    everything below that threshold retained -- preferentially -- bins sitting far under
    the background, and a notch stopband is an absence of signal rather than a
    measurement of it.

    Ground truth here: a 1/f background with a known slope, and a four-bin hole cut into
    it of the depth line-cleaned data actually carries.
    """
    import numpy as np

    from eeg_pipeline.preprocessing.report.aperiodic import fit_aperiodic

    frequencies = np.arange(2.0, 45.0, 0.25)
    truth = -12.0
    clean = truth * np.log10(frequencies) + 30.0

    notched = clean.copy()
    hole = (frequencies >= 28.0) & (frequencies <= 28.75)
    notched[hole] -= 28.0

    recovered = fit_aperiodic(frequencies=frequencies, power_db=notched, fit_range_hz=(2.0, 45.0))

    assert recovered is not None
    assert abs(recovered.slope_db_per_decade - truth) < 0.5


def test_an_oscillatory_peak_is_still_trimmed() -> None:
    """Making the trim two-sided must not cost it what it was written for."""
    import numpy as np

    from eeg_pipeline.preprocessing.report.aperiodic import fit_aperiodic

    frequencies = np.arange(2.0, 45.0, 0.25)
    truth = -12.0
    values = truth * np.log10(frequencies) + 30.0
    alpha = (frequencies >= 9.0) & (frequencies <= 12.0)
    values[alpha] += 8.0

    recovered = fit_aperiodic(frequencies=frequencies, power_db=values, fit_range_hz=(2.0, 45.0))

    assert recovered is not None
    assert abs(recovered.slope_db_per_decade - truth) < 0.5
