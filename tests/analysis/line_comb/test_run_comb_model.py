"""An adaptive removal grid must carry evidence for every time window."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr


def _estimate(
    fundamental_hz: float,
    *,
    jackknife_se_hz: float,
    harmonics: tuple[int, ...] = tuple(range(24, 80)),
) -> lr.CombEstimate:
    return lr.CombEstimate(
        fundamental_hz=fundamental_hz,
        harmonics_used=harmonics,
        harmonic_positions_hz=tuple(fundamental_hz * harmonic for harmonic in harmonics),
        residual_rms_hz=0.05,
        max_abs_residual_hz=0.10,
        fundamental_jackknife_se_hz=jackknife_se_hz,
        isolated_hz=(),
        isolated_prominence_db=(),
    )


def test_nonstationary_windows_produce_an_adaptive_model():
    whole = _estimate(1.2, jackknife_se_hz=8e-5)
    windows = tuple(
        _estimate(value, jackknife_se_hz=2e-4) for value in (1.1990, 1.2008, 1.1992, 1.2009)
    )

    model = lr.build_adaptive_comb_model(whole, windows)

    assert model.whole_estimate is whole
    assert model.window_fundamental_hz == pytest.approx(
        tuple(estimate.fundamental_hz for estimate in windows)
    )
    assert model.fundamental_range_hz == pytest.approx(0.0019)
    assert model.max_adjacent_shift_hz == pytest.approx(0.0018)


def test_a_window_without_enough_harmonic_support_is_refused():
    whole = _estimate(1.2, jackknife_se_hz=8e-5)
    unsupported = _estimate(
        1.2,
        jackknife_se_hz=2e-4,
        harmonics=tuple(range(24, 30)),
    )

    with pytest.raises(ValueError, match="window 1"):
        lr.build_adaptive_comb_model(whole, (whole, unsupported))


def test_each_window_width_uses_its_own_fundamental_uncertainty():
    precise = _estimate(1.2, jackknife_se_hz=5e-5)
    uncertain = _estimate(1.2, jackknife_se_hz=3e-4)
    targets = (28.8, 94.8, 57.25)

    precise_widths = lr.uncertainty_aware_notch_widths(
        precise,
        targets,
        ratio=450.0,
        minimum_hz=0.05,
        confidence_z=3.0,
        isolated_minimum_hz=0.3,
    )
    uncertain_widths = lr.uncertainty_aware_notch_widths(
        uncertain,
        targets,
        ratio=450.0,
        minimum_hz=0.05,
        confidence_z=3.0,
        isolated_minimum_hz=0.3,
    )

    assert uncertain_widths[1] > precise_widths[1]
    assert uncertain_widths[0] >= 28.8 / 450.0 + 2 * 24 * 3.0e-4
    assert uncertain_widths[2] == pytest.approx(0.3)


def test_comb_estimate_reports_delete_one_harmonic_uncertainty():
    freqs = np.arange(1.0, 110.0, 0.002)
    spectrum = np.zeros_like(freqs)
    prominence = np.zeros_like(freqs)
    rng = np.random.default_rng(4)
    for harmonic in range(24, 80):
        frequency = harmonic * 1.2 + rng.normal(0.0, 0.015)
        index = int(np.argmin(np.abs(freqs - frequency)))
        spectrum[index] = 20.0
        prominence[index] = 20.0

    estimate = lr.estimate_comb(freqs, spectrum, prominence, isolated_nominal_hz=())

    assert np.isfinite(estimate.fundamental_jackknife_se_hz)
    assert estimate.fundamental_jackknife_se_hz > 0
