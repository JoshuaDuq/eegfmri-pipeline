"""The audit must see delta, and must not confuse removal with contamination."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis import band_audit as ba


def _flat_spectrum(bin_width: float = 0.0462963, high_hz: float = 100.0):
    freqs = np.arange(0.0, high_hz, bin_width)
    psd = np.full(freqs.size, 1e-12)
    return freqs, psd


class TestAdaptiveBackground:
    def test_is_finite_in_delta_where_the_wide_window_is_not(self):
        freqs, psd = _flat_spectrum()
        spectrum_db = ba.hd.to_db(psd)
        wide = ba.hd.local_background_db(spectrum_db, half_width_bins=100)
        adaptive = ba.adaptive_background_db(spectrum_db, freqs)

        delta = (freqs >= 1.0) & (freqs <= 4.0)
        assert not np.any(np.isfinite(wide[delta])), "the 4.63 Hz window should not reach delta"
        assert np.all(np.isfinite(adaptive[delta])), "the adaptive window must reach delta"

    def test_uses_the_wide_window_above_the_switch(self):
        freqs, psd = _flat_spectrum()
        spectrum_db = ba.hd.to_db(psd)
        wide = ba.hd.local_background_db(spectrum_db, half_width_bins=100)
        adaptive = ba.adaptive_background_db(spectrum_db, freqs)

        upper = freqs >= 20.0
        assert np.allclose(adaptive[upper], wide[upper], equal_nan=True)

    def test_tracks_a_one_over_f_slope_without_reading_it_as_a_line(self):
        bin_width = 0.0462963
        freqs = np.arange(bin_width, 100.0, bin_width)
        psd = 1e-12 / freqs  # pure 1/f, no lines anywhere
        adaptive = ba.adaptive_background_db(ba.hd.to_db(psd), freqs)
        excess = ba.hd.to_db(psd) - adaptive

        delta = np.isfinite(excess) & (freqs >= 1.5) & (freqs <= 4.0)
        assert (
            np.nanmax(np.abs(excess[delta])) < 0.5
        ), "a running median over a monotone background must not manufacture a line"


class TestBandCosts:
    def test_a_clean_flat_band_costs_nothing(self):
        freqs, psd = _flat_spectrum()
        costs = ba.band_costs(freqs, psd, low_hz=30.1, high_hz=45.0, independent_hz=[], comb_hz=[])
        assert costs["independent_pct"] == pytest.approx(0.0, abs=1e-6)
        assert costs["sideband_pct"] == pytest.approx(0.0, abs=1e-6)
        assert costs["hole_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_an_added_line_is_charged_to_independent_not_to_sidebands(self):
        freqs, psd = _flat_spectrum()
        psd = psd.copy()
        peak = int(round(40.0 / (freqs[1] - freqs[0])))
        psd[peak] *= 100.0

        costs = ba.band_costs(
            freqs, psd, low_hz=30.1, high_hz=45.0, independent_hz=[40.0], comb_hz=[]
        )
        assert costs["independent_pct"] > 1.0
        assert costs["sideband_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_a_nulled_bin_is_charged_to_holes_not_to_lines(self):
        freqs, psd = _flat_spectrum()
        psd = psd.copy()
        notch = int(round(40.0 / (freqs[1] - freqs[0])))
        psd[notch] *= 0.01  # 20 dB below background: removal, not contamination

        costs = ba.band_costs(
            freqs, psd, low_hz=30.1, high_hz=45.0, independent_hz=[40.0], comb_hz=[]
        )
        assert costs["hole_pct"] > 0.0
        assert costs["independent_pct"] == pytest.approx(
            0.0, abs=1e-6
        ), "a bin dug below background must never count as positive artifact"

    def test_comb_sidebands_exclude_the_nulled_centre_bin(self):
        freqs, psd = _flat_spectrum()
        psd = psd.copy()
        centre = int(round(82.2222 / (freqs[1] - freqs[0])))
        psd[centre] *= 0.01  # comb centre nulled by the gradient correction
        psd[centre + 3] *= 50.0  # shoulder carrying the residual

        costs = ba.band_costs(
            freqs, psd, low_hz=62.0, high_hz=95.0, independent_hz=[], comb_hz=[82.2222]
        )
        assert costs["sideband_pct"] > 0.0, "the shoulder must be counted"
        assert costs["hole_pct"] > 0.0, "the nulled centre must be counted as a hole"

    def test_rejects_a_band_with_no_usable_background(self):
        freqs, psd = _flat_spectrum()
        with pytest.raises(ValueError, match="no bin with a usable background"):
            ba.band_costs(freqs, psd, low_hz=0.0, high_hz=0.2, independent_hz=[], comb_hz=[])
