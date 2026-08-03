"""A comb fit has to be good enough to justify the deletion grid it authorises.

Three harmonics were enough to fit a fundamental, with no criterion on how well they
agreed -- and the fit then authorises removing every harmonic from 22 to 83, sixty-one
targets. Three mutually inconsistent peaks produced a 0.228 Hz RMS fit error and still
generated the full grid. Across ninety real runs the RMS error is 0.030-0.156 Hz on 52-56
harmonics, so the two cases are far apart and the fail-safe does not need to be delicate.
"""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import removal as lr


def _spectrum(positions, df=0.002, high=110.0, height=20.0):
    freqs = np.arange(1.0, high, df)
    spectrum = np.zeros_like(freqs)
    sigma = 0.109 / 2.355
    for centre in positions:
        spectrum[:] = np.maximum(spectrum, height * np.exp(-0.5 * ((freqs - centre) / sigma) ** 2))
    return freqs, spectrum, spectrum.copy()


def test_a_consistent_comb_still_fits():
    positions = [k * 1.2 for k in range(24, 80)]
    freqs, spec, prom = _spectrum(positions)
    estimate = lr.estimate_comb(freqs, spec, prom, isolated_nominal_hz=())
    assert estimate.fundamental_hz == pytest.approx(1.2, abs=1e-4)


def test_a_fit_from_too_few_harmonics_is_refused():
    """Sixty-one targets must not rest on three peaks."""
    freqs, spec, prom = _spectrum([24 * 1.2, 40 * 1.2, 60 * 1.2])
    with pytest.raises(ValueError, match="harmonics"):
        lr.estimate_comb(freqs, spec, prom, isolated_nominal_hz=())


def test_a_fit_whose_harmonics_disagree_is_refused():
    """Peaks that do not lie on one arithmetic grid are not a comb."""
    # Alternating near the edge of the search window: every harmonic is still found, so
    # this isolates the scatter bound from the harmonic-count floor.
    positions = [k * 1.2 + (0.24 if k % 2 else -0.24) for k in range(24, 80)]
    freqs, spec, prom = _spectrum(positions)
    with pytest.raises(ValueError, match="scatter"):
        lr.estimate_comb(freqs, spec, prom, isolated_nominal_hz=())
