"""The preservation gates have to be able to fail.

Two of them could not. ``lines_suppressed`` was named for a maximum and evaluated a
median, so half a run's lines could stand above the threshold and still pass. The transient
gates compared the removal's effect on data+probe against its effect on the probe alone --
quantities that are identical for a linear operator, which spectrum_fit is, so they read
1.0 by construction on any data with any settings.
"""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr


def _metrics(**overrides):
    base = {
        "median_residual_prominence_db": -5.0,
        "max_residual_prominence_db": -3.0,
        "median_suppression_db": 20.0,
        "max_probe_deviation_db": 0.0,
        "max_nonline_change_db": 0.0,
        "burst_energy_ratio": 1.0,
        "burst_correlation": 1.0,
        "intrinsic_energy_ratio": 0.95,
        "removed_band_fraction": 0.12,
    }
    base.update(overrides)
    return base


def test_a_run_with_a_line_still_standing_does_not_pass():
    """The 90-run manifest had a worst residual of +13.90 dB while every gate passed."""
    gate = lr.PreservationGate()
    metrics = _metrics(median_residual_prominence_db=-5.0, max_residual_prominence_db=13.9)
    assert not gate.evaluate(metrics)["lines_suppressed"], (
        "a line 13.9 dB above background survived and the gate called it suppressed"
    )


def test_the_median_alone_cannot_carry_the_line_gate():
    """Half the targets above threshold is not 'lines suppressed'."""
    gate = lr.PreservationGate()
    assert not gate.passed(_metrics(max_residual_prominence_db=3.47))


def test_a_clean_run_still_passes():
    assert lr.PreservationGate().passed(_metrics())


def test_the_transient_gate_reads_the_measurement_that_can_fail():
    """intrinsic_energy_ratio is the one that varies; it was reported but never gated.

    On the delivered benchmark it ranged 0.899 to 0.960 while burst_energy_ratio was
    exactly 1.0 on every row.
    """
    gate = lr.PreservationGate()
    assert not gate.evaluate(_metrics(intrinsic_energy_ratio=0.40))["transient_preserved"], (
        "a transient reduced to 40% of its injected energy passed the transient gate"
    )
    assert gate.evaluate(_metrics(intrinsic_energy_ratio=0.95))["transient_preserved"]


def test_a_residual_displaced_by_a_bin_is_not_missed():
    """Suppression was read at the nearest bin only, so a line that moved slipped through.

    Reproduced: a target at 50 Hz whose centre falls to -10 dB while a residual at
    50.05 Hz still stands at +15 dB was reported as a maximum residual of -10 dB. That is
    the exact shape of the failure this work has been chasing -- removing a line exposes or
    displaces its neighbour -- so measuring only the centre cannot see it.
    """
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -10.0
    after[np.argmin(np.abs(freqs - 50.05))] = 15.0

    result = lr.line_suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] >= 15.0 - 1e-6, (
        f"the displaced residual was invisible: {result['max_residual_prominence_db']}"
    )


def test_suppression_still_reads_the_centre_when_nothing_moved():
    freqs = np.arange(45.0, 55.0, 0.01)
    before = np.full_like(freqs, -20.0)
    after = np.full_like(freqs, -20.0)
    before[np.argmin(np.abs(freqs - 50.0))] = 25.0
    after[np.argmin(np.abs(freqs - 50.0))] = -12.0

    result = lr.line_suppression(freqs, before, after, [50.0], widths=[0.2])
    assert result["max_residual_prominence_db"] == pytest.approx(-12.0)
