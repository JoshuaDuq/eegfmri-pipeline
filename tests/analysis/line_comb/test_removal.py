"""Tests for line-comb estimation and removal."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import diagnosis as hd
from studies.pain_study.analysis.line_comb import removal as lr


def synthetic_spectrum(fundamental=1.2, harmonics=range(24, 80), extra=(), amplitude_db=12.0):
    """A 1/f spectrum with comb lines planted at integer multiples."""
    freqs = np.arange(0, 110.0 + 1e-9, 1 / 21.6)
    spectrum = 10 ** ((-100.0 - 12.0 * np.log10(np.maximum(freqs, 0.5))) / 10.0)
    for harmonic in harmonics:
        index = int(np.argmin(np.abs(freqs - fundamental * harmonic)))
        spectrum[index] *= 10 ** (amplitude_db / 10.0)
    for frequency in extra:
        spectrum[int(np.argmin(np.abs(freqs - frequency)))] *= 10 ** (amplitude_db / 10.0)
    db = hd.to_db(spectrum)
    return freqs, db, hd.prominence_db(db, half_width_bins=100)


class TestEstimateComb:
    def test_recovers_a_planted_fundamental(self):
        freqs, db, prom = synthetic_spectrum(fundamental=1.19993)
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=())
        assert est.fundamental_hz == pytest.approx(1.19993, abs=2e-4)
        assert est.n_harmonics > 40
        assert est.residual_rms_hz < 0.02

    def test_uses_many_harmonics_to_beat_the_bin_width(self):
        # The fundamental is recovered far more precisely than one bin of the grid.
        freqs, db, prom = synthetic_spectrum(fundamental=1.19993)
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=())
        assert abs(est.fundamental_hz - 1.19993) < (freqs[1] - freqs[0]) / 10

    def test_finds_the_isolated_lines(self):
        freqs, db, prom = synthetic_spectrum(extra=(57.22, 47.04))
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(57.2247, 47.0362))
        assert len(est.isolated_hz) == 2
        assert est.isolated_hz[0] == pytest.approx(57.22, abs=0.03)
        assert est.isolated_hz[1] == pytest.approx(47.04, abs=0.03)

    def test_does_not_lock_onto_the_comb_line_beside_an_isolated_one(self):
        # 47.036 Hz sits 0.24 Hz from comb harmonic 39 at 46.8 Hz. With only the comb
        # present, the narrow isolated window must not report the comb line as the
        # isolated one. Reporting nothing is the stronger outcome and the one the
        # prominence floor now produces, so the check is that 46.8 never comes back.
        freqs, db, prom = synthetic_spectrum()
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(47.0362,))
        position = est.isolated_hz[0]
        assert np.isnan(position) or abs(position - 46.8) > 0.1

    def test_rejects_an_isolated_nominal_that_collides_with_the_comb(self):
        freqs, db, prom = synthetic_spectrum()
        with pytest.raises(ValueError, match="the search would find the comb"):
            lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(46.82,))

    def test_reports_no_position_for_an_isolated_line_that_is_not_there(self):
        # The isolated list is a cohort-level seed, so a participant who simply does not
        # carry one of its lines is ordinary. Without a floor the search still returns the
        # largest bin in its window, and that position becomes a removal target -- which
        # digs a notch into clean spectrum on the strength of noise.
        freqs, db, prom = synthetic_spectrum(extra=(57.22,))
        est = lr.estimate_comb(
            freqs, db, prom, isolated_nominal_hz=(57.2247, 42.6), min_prominence_db=3.0
        )
        assert est.isolated_hz[0] == pytest.approx(57.22, abs=0.03)
        assert np.isnan(est.isolated_hz[1])

    def test_a_weaker_nominal_does_not_take_a_stronger_ones_line(self):
        # 57.2247 and 57.3485 sit 0.124 Hz apart, closer than the search half-width, so
        # their windows overlap. On real data the 57.14 line is ~17 dB the stronger of the
        # two, and a plain largest-peak search hands it to both nominals. The weaker
        # nominal must look past a line another one has already claimed and find its own.
        freqs, db, prom = synthetic_spectrum(extra=(57.14, 57.40), amplitude_db=12.0)
        # Make the first line dominate, as it does in the recordings.
        index = int(np.argmin(np.abs(freqs - 57.14)))
        db = db.copy()
        db[index] += 17.0
        prom = hd.prominence_db(db, half_width_bins=100)
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(57.2247, 57.3485))
        assert est.isolated_hz[0] == pytest.approx(57.14, abs=0.04)
        assert est.isolated_hz[1] == pytest.approx(57.40, abs=0.04)

    def test_reports_nothing_for_a_nominal_whose_only_peak_is_already_claimed(self):
        # One line, two nominals reaching for it: the second has nothing of its own, and
        # saying so is the honest outcome. Removing the claimed line twice would widen the
        # notch around it while telling the reader a second line was found.
        freqs, db, prom = synthetic_spectrum(extra=(57.28,))
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(57.2247, 57.3485))
        assert np.isfinite(est.isolated_hz[0])
        assert np.isnan(est.isolated_hz[1])

    def test_allows_two_isolated_nominals_that_find_their_own_lines(self):
        freqs, db, prom = synthetic_spectrum(extra=(57.14, 57.40))
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(57.1432, 57.3485))
        assert est.isolated_hz[0] == pytest.approx(57.14, abs=0.04)
        assert est.isolated_hz[1] == pytest.approx(57.40, abs=0.04)

    def test_an_absent_isolated_line_never_becomes_a_removal_target(self):
        freqs, db, prom = synthetic_spectrum(extra=(57.22,))
        est = lr.estimate_comb(
            freqs, db, prom, isolated_nominal_hz=(57.2247, 42.6), min_prominence_db=3.0
        )
        targets = lr.removal_frequencies(est, harmonic_range=(24, 30))
        assert not any(abs(t - 42.6) < 0.2 for t in targets)

    def test_ignores_harmonics_below_the_prominence_floor(self):
        freqs, db, prom = synthetic_spectrum(harmonics=range(24, 50))
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(), min_prominence_db=3.0)
        assert max(est.harmonics_used) < 55
        assert est.fundamental_hz == pytest.approx(1.2, abs=1e-3)

    def test_refuses_to_fit_when_almost_nothing_is_there(self):
        freqs, db, prom = synthetic_spectrum(harmonics=range(24, 26))
        with pytest.raises(ValueError, match="refusing to fit"):
            lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(), min_prominence_db=3.0)

    def test_rejects_a_search_window_wider_than_half_the_spacing(self):
        freqs, db, prom = synthetic_spectrum()
        with pytest.raises(ValueError, match="below half the nominal spacing"):
            lr.estimate_comb(freqs, db, prom, search_hz=0.8)

    def test_rejects_mismatched_inputs(self):
        freqs, db, prom = synthetic_spectrum()
        with pytest.raises(ValueError, match="same shape"):
            lr.estimate_comb(freqs, db[:-1], prom)


class TestRemovalFrequencies:
    def _estimate(self, **kw):
        return lr.CombEstimate(
            fundamental_hz=kw.get("fundamental", 1.2),
            harmonics_used=(24, 25),
            residual_rms_hz=0.0,
            max_abs_residual_hz=0.0,
            isolated_hz=kw.get("isolated", (57.22,)),
            isolated_prominence_db=(10.0,),
        )

    def test_covers_the_harmonic_range_and_the_isolated_lines(self):
        targets = lr.removal_frequencies(self._estimate(), harmonic_range=(24, 30))
        assert min(targets) == pytest.approx(28.8)
        assert max(targets) == pytest.approx(57.22)
        assert len(targets) == 8  # seven harmonics plus one isolated line

    def test_excludes_the_mains_neighbourhood(self):
        targets = lr.removal_frequencies(self._estimate(), harmonic_range=(24, 79))
        assert not any(59.5 <= f <= 60.5 for f in targets)
        assert not any(f == pytest.approx(60.0) for f in targets)

    def test_respects_the_analysis_range(self):
        targets = lr.removal_frequencies(self._estimate(), harmonic_range=(1, 79), low_hz=30.0)
        assert min(targets) >= 30.0

    def test_is_sorted_and_unique(self):
        targets = lr.removal_frequencies(self._estimate(), harmonic_range=(24, 79))
        assert list(targets) == sorted(targets)

    def test_raises_when_everything_is_filtered_out(self):
        with pytest.raises(ValueError, match="No removal frequency"):
            lr.removal_frequencies(
                self._estimate(isolated=()), harmonic_range=(24, 30), low_hz=200.0, high_hz=300.0
            )


class TestCombineEstimates:
    def _estimate(self, fundamental, isolated=(57.2, 47.0)):
        return lr.CombEstimate(
            fundamental_hz=fundamental,
            harmonics_used=(24, 25, 26),
            residual_rms_hz=0.01,
            max_abs_residual_hz=0.02,
            isolated_hz=tuple(isolated),
            isolated_prominence_db=(10.0,) * len(isolated),
        )

    def test_median_ignores_a_single_bad_run(self):
        # sub-0000 run-1 estimated 1.199338 while its session sat at 1.19998.
        runs = [
            self._estimate(f) for f in (1.199338, 1.200054, 1.200043, 1.199770, 1.200000, 1.199965)
        ]
        pooled = lr.combine_estimates(runs)
        assert pooled.fundamental_hz == pytest.approx(1.199982, abs=1e-6)

    def test_pooling_reduces_scatter_towards_the_true_value(self):
        rng = np.random.default_rng(0)
        truth = 1.2
        scatter, pooled = [], []
        for _ in range(200):
            runs = [self._estimate(truth + rng.normal(scale=2.8e-4)) for _ in range(6)]
            scatter.append(runs[0].fundamental_hz - truth)
            pooled.append(lr.combine_estimates(runs).fundamental_hz - truth)
        assert np.std(pooled) < np.std(scatter) / 2

    def test_isolated_lines_pool_position_by_position(self):
        runs = [
            self._estimate(1.2, isolated=(57.1, 47.0)),
            self._estimate(1.2, isolated=(57.3, 47.2)),
            self._estimate(1.2, isolated=(57.2, 47.1)),
        ]
        pooled = lr.combine_estimates(runs)
        assert pooled.isolated_hz[0] == pytest.approx(57.2)
        assert pooled.isolated_hz[1] == pytest.approx(47.1)

    def test_a_missing_line_in_one_run_does_not_poison_the_pool(self):
        runs = [
            self._estimate(1.2, isolated=(57.1, float("nan"))),
            self._estimate(1.2, isolated=(57.2, 47.0)),
            self._estimate(1.2, isolated=(57.3, 47.2)),
        ]
        pooled = lr.combine_estimates(runs)
        assert pooled.isolated_hz[1] == pytest.approx(47.1)

    def test_rejects_estimates_of_differing_width(self):
        with pytest.raises(ValueError, match="how many isolated lines"):
            lr.combine_estimates([self._estimate(1.2, (57.2,)), self._estimate(1.2, (57.2, 47.0))])

    def test_rejects_an_empty_session(self):
        with pytest.raises(ValueError, match="at least one estimate"):
            lr.combine_estimates([])

    def test_a_nan_line_is_dropped_from_the_removal_list(self):
        estimate = self._estimate(1.2, isolated=(57.2, float("nan")))
        targets = lr.removal_frequencies(estimate, harmonic_range=(24, 26))
        assert all(np.isfinite(t) for t in targets)
        assert len(targets) == 4


class TestProbe:
    def test_waveform_contains_every_sinusoid(self):
        probe = lr.Probe(burst_amplitude_v=0.0)
        sfreq, n = 1000.0, 240_000
        times = np.arange(n) / sfreq
        freqs, psd = hd.hann_periodogram(probe.waveform(times), sfreq)
        for frequency in probe.sinusoid_hz:
            index = int(np.argmin(np.abs(freqs - frequency)))
            assert psd[index] > 100 * np.median(psd)

    def test_burst_is_localised_in_time(self):
        probe = lr.Probe(sinusoid_amplitude_v=0.0)
        times = np.arange(240_000) / 1000.0
        wave = probe.waveform(times)
        inside = probe.burst_window(times)
        assert np.abs(wave[inside]).max() > 100 * np.abs(wave[~inside]).max()

    def test_burst_window_brackets_the_centre(self):
        probe = lr.Probe()
        times = np.arange(240_000) / 1000.0
        window = probe.burst_window(times)
        assert times[window].min() == pytest.approx(probe.burst_centre_s - 0.2, abs=0.01)
        assert times[window].max() == pytest.approx(probe.burst_centre_s + 0.2, abs=0.01)


class TestProbeClearance:
    def test_accepts_well_separated_probes(self):
        lr.check_probe_clearance(lr.Probe(), [1.2 * k for k in range(24, 80)])

    def test_rejects_a_probe_sitting_on_a_target(self):
        probe = lr.Probe(sinusoid_hz=(36.0,))
        with pytest.raises(ValueError, match="clearance"):
            lr.check_probe_clearance(probe, [1.2 * k for k in range(24, 80)])

    def test_the_default_probes_clear_the_real_comb(self):
        targets = [1.2 * k for k in range(24, 80)] + list(lr.ISOLATED_NOMINAL_HZ)
        lr.check_probe_clearance(lr.Probe(), targets)


class TestMetrics:
    def test_line_suppression_measures_the_drop(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        before = np.zeros_like(freqs)
        after = np.zeros_like(freqs)
        targets = [30.0, 45.0]
        for frequency in targets:
            index = int(np.argmin(np.abs(freqs - frequency)))
            before[index], after[index] = 20.0, 1.0
        result = lr.line_suppression(freqs, before, after, targets)
        assert result["median_suppression_db"] == pytest.approx(19.0)
        assert result["median_residual_prominence_db"] == pytest.approx(1.0)
        assert result["n_targets"] == 2

    def test_probe_preservation_detects_an_untouched_probe(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        psd = np.ones((4, freqs.size))
        result = lr.probe_preservation(freqs, psd, psd.copy(), lr.Probe())
        assert result["max_probe_deviation_db"] == pytest.approx(0.0)

    def test_probe_preservation_detects_a_removed_probe(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        before = np.ones((4, freqs.size))
        after = before.copy()
        probe = lr.Probe()
        after[:, int(np.argmin(np.abs(freqs - probe.sinusoid_hz[0])))] = 0.5
        result = lr.probe_preservation(freqs, before, after, probe)
        assert result["max_probe_deviation_db"] == pytest.approx(3.01, abs=0.02)

    def test_nonline_change_ignores_the_removed_bins(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        before = np.ones((3, freqs.size))
        after = before.copy()
        target = 54.0
        near = np.abs(freqs - target) <= 0.2
        after[:, near] = 1e-6  # the line is gone; that must not count as a loss
        change = lr.nonline_change_db(freqs, before, after, [target])
        assert np.max(np.abs(change)) < 1e-9

    def test_nonline_change_sees_a_broadband_loss(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        before = np.ones((3, freqs.size))
        change = lr.nonline_change_db(freqs, before, before * 0.5, [54.0])
        assert np.allclose(change, -3.0103, atol=1e-3)

    def test_a_probe_matching_its_reference_scores_one(self):
        times = np.arange(240_000) / 1000.0
        probe = lr.Probe()
        wave = probe.waveform(times)
        result = lr.probe_recovery(np.tile(wave, (3, 1)), wave, times, probe)
        assert result["burst_energy_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert result["burst_correlation"] == pytest.approx(1.0, abs=1e-9)
        assert result["intrinsic_energy_ratio"] == pytest.approx(1.0, abs=1e-9)

    def test_inevitable_loss_is_reported_separately_and_not_charged_as_damage(self):
        # The reference already lost a fifth of its energy to the removal; a recovered
        # probe that matches it exactly must still score a ratio of one.
        times = np.arange(240_000) / 1000.0
        probe = lr.Probe()
        reference = probe.waveform(times) * np.sqrt(0.8)
        result = lr.probe_recovery(np.tile(reference, (3, 1)), reference, times, probe)
        assert result["burst_energy_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert result["intrinsic_energy_ratio"] == pytest.approx(0.8, abs=1e-9)

    def test_a_halved_probe_shows_a_quarter_of_the_energy(self):
        times = np.arange(240_000) / 1000.0
        probe = lr.Probe()
        wave = probe.waveform(times)
        result = lr.probe_recovery(np.tile(wave * 0.5, (3, 1)), wave, times, probe)
        assert result["burst_energy_ratio"] == pytest.approx(0.25, abs=1e-9)

    def test_a_distorted_probe_loses_correlation(self):
        times = np.arange(240_000) / 1000.0
        probe = lr.Probe()
        rng = np.random.default_rng(0)
        recovered = np.tile(rng.normal(size=times.size) * 1e-6, (3, 1))
        result = lr.probe_recovery(recovered, probe.waveform(times), times, probe)
        assert result["burst_correlation"] < 0.5

    def test_recover_probe_differences_the_two_passes(self):
        with_probe = np.array([[3.0, 4.0], [5.0, 6.0]])
        without = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert np.allclose(lr.recover_probe(with_probe, without), [[2.0, 3.0], [4.0, 5.0]])

    def test_recover_probe_rejects_a_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            lr.recover_probe(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_probe_recovery_rejects_an_empty_window(self):
        times = np.arange(1000) / 1000.0  # one second, burst centre is at 120 s
        with pytest.raises(ValueError, match="outside the recording"):
            lr.probe_recovery(np.ones((2, 1000)), np.ones(1000), times, lr.Probe())


class TestRemovedBandFraction:
    def _grid(self):
        return np.arange(0, 120, 0.05)

    def test_single_bin_per_line_costs_about_four_percent(self):
        targets = [1.2 * k for k in range(24, 80) if not 59.5 <= 1.2 * k <= 60.5]
        fraction = lr.removed_band_fraction(self._grid(), targets, 0.0)
        assert 0.03 < fraction < 0.05

    def test_mne_default_width_would_empty_a_quarter_of_the_band(self):
        targets = np.array([1.2 * k for k in range(24, 80) if not 59.5 <= 1.2 * k <= 60.5])
        fraction = lr.removed_band_fraction(self._grid(), targets, targets / 200.0)
        assert fraction > 0.2

    def test_the_chosen_width_stays_inside_the_gate(self):
        targets = np.array([1.2 * k for k in range(24, 80) if not 59.5 <= 1.2 * k <= 60.5])
        widths = lr.notch_widths_for(targets, ratio=450.0, minimum_hz=0.05)
        fraction = lr.removed_band_fraction(self._grid(), targets, widths)
        assert fraction <= lr.PreservationGate().max_band_fraction_removed
        assert fraction > 0.10  # and it really is wider than one bin per line


class TestNotchWidthsFor:
    def test_wider_settings_touch_more_of_the_band(self):
        targets = [1.2 * k for k in range(24, 80)]
        grid = np.arange(0, 120, 0.05)
        fractions = [lr.removed_band_fraction(grid, targets, w) for w in (0.0, 0.1, 0.3, 0.6)]
        assert fractions == sorted(fractions)

    def test_scales_with_frequency(self):
        widths = lr.notch_widths_for([30.0, 60.0, 90.0], ratio=450.0)
        assert widths.tolist() == pytest.approx([30 / 450, 60 / 450, 90 / 450])

    def test_applies_the_floor_to_low_harmonics(self):
        widths = lr.notch_widths_for([9.0, 90.0], ratio=450.0, minimum_hz=0.05)
        assert widths[0] == pytest.approx(0.05)
        assert widths[1] == pytest.approx(0.2)

    def test_rejects_a_non_positive_ratio(self):
        with pytest.raises(ValueError, match="finite positive"):
            lr.notch_widths_for([30.0], ratio=0.0)

    def test_rejects_a_negative_floor(self):
        with pytest.raises(ValueError, match="non-negative"):
            lr.notch_widths_for([30.0], ratio=450.0, minimum_hz=-1.0)

    def test_rejects_an_empty_band(self):
        with pytest.raises(ValueError, match="no frequency bins"):
            lr.removed_band_fraction(np.arange(0, 10, 0.05), [5.0], 0.1, band_hz=(200.0, 300.0))


class TestPreservationGate:
    def _metrics(self, **overrides):
        base = {
            "median_residual_prominence_db": -15.0,
            "median_suppression_db": 28.0,
            "max_probe_deviation_db": 0.01,
            "max_nonline_change_db": 0.001,
            "burst_energy_ratio": 0.998,
            "burst_correlation": 0.999,
            "removed_band_fraction": 0.06,
        }
        base.update(overrides)
        return base

    def test_a_good_removal_passes_every_criterion(self):
        gate = lr.PreservationGate()
        assert gate.passed(self._metrics())
        assert all(gate.evaluate(self._metrics()).values())

    def test_leftover_lines_fail(self):
        gate = lr.PreservationGate()
        verdict = gate.evaluate(self._metrics(median_residual_prominence_db=6.0))
        assert not verdict["lines_suppressed"]

    def test_a_removed_probe_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(max_probe_deviation_db=1.2))["sinusoids_preserved"]

    def test_broadband_damage_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(max_nonline_change_db=0.9))["spectrum_preserved"]

    def test_a_flattened_transient_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(burst_energy_ratio=0.80))["transient_preserved"]

    def test_an_inflated_transient_also_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(burst_energy_ratio=1.20))["transient_preserved"]

    def test_emptying_the_band_fails_even_when_every_line_is_gone(self):
        # The failure mode that hid behind a guard band: lines suppressed, probes intact,
        # untouched bins unchanged -- because a quarter of the band was simply gone.
        gate = lr.PreservationGate()
        verdict = gate.evaluate(self._metrics(removed_band_fraction=0.27))
        assert verdict["lines_suppressed"]
        assert verdict["spectrum_preserved"]
        assert not verdict["band_mostly_untouched"]
        assert not gate.passed(self._metrics(removed_band_fraction=0.27))

    def test_a_distorted_transient_fails_even_at_the_right_energy(self):
        gate = lr.PreservationGate()
        verdict = gate.evaluate(self._metrics(burst_correlation=0.80))
        assert verdict["transient_preserved"]
        assert not verdict["transient_undistorted"]


class TestEndToEndOnSyntheticData:
    def test_removal_takes_out_the_comb_and_leaves_the_probes(self):
        """The whole chain on a synthetic recording with a known comb."""
        import mne

        mne.set_log_level("ERROR")
        sfreq, duration = 500.0, 200.0
        n_times = int(sfreq * duration)
        times = np.arange(n_times) / sfreq
        rng = np.random.default_rng(0)

        fundamental = 1.19994
        signal = rng.normal(scale=5e-6, size=(4, n_times))
        for harmonic in range(24, 40):
            signal += 1e-6 * np.sin(2 * np.pi * fundamental * harmonic * times + harmonic)
        probe = lr.Probe(burst_centre_s=100.0)
        signal += probe.waveform(times)[None, :]

        info = mne.create_info([f"E{i}" for i in range(4)], sfreq, "eeg")
        raw = mne.io.RawArray(signal, info)

        nperseg = int(60 * 0.9 * sfreq)
        freqs, psd_before = hd.hann_periodogram(
            raw.get_data()[:, : (n_times // nperseg) * nperseg].reshape(4, -1, nperseg), sfreq
        )
        psd_before = psd_before.mean(axis=1)
        db = hd.to_db(np.median(psd_before, axis=0))
        prom = hd.prominence_db(db, half_width_bins=int(round(4.63 / (freqs[1] - freqs[0]))))

        estimate = lr.estimate_comb(
            freqs, db, prom, harmonic_range=(24, 39), isolated_nominal_hz=()
        )
        assert estimate.fundamental_hz == pytest.approx(fundamental, abs=5e-4)

        targets = lr.removal_frequencies(estimate, harmonic_range=(24, 39), low_hz=3.0)
        lr.check_probe_clearance(probe, targets)
        widths = lr.notch_widths_for(targets, ratio=450.0, minimum_hz=0.05)

        def clean(data):
            # RawArray does not copy, and notch_filter works in place, so without an
            # explicit copy the first call would clean the caller's array underneath it.
            copy = mne.io.RawArray(np.array(data, copy=True), info, verbose="ERROR")
            return copy.notch_filter(
                freqs=list(targets),
                method="spectrum_fit",
                filter_length="20s",
                mt_bandwidth=0.6,
                notch_widths=widths,
                verbose="ERROR",
            )

        cleaned = clean(signal)
        cleaned_bare = clean(signal - probe.waveform(times)[None, :])

        _, psd_after = hd.hann_periodogram(
            cleaned.get_data()[:, : (n_times // nperseg) * nperseg].reshape(4, -1, nperseg), sfreq
        )
        psd_after = psd_after.mean(axis=1)
        prom_after = hd.prominence_db(
            hd.to_db(np.median(psd_after, axis=0)),
            half_width_bins=int(round(4.63 / (freqs[1] - freqs[0]))),
        )
        metrics = {
            **lr.line_suppression(freqs, prom, prom_after, targets),
            **lr.probe_preservation(freqs, psd_before, psd_after, probe),
            "max_nonline_change_db": float(
                np.max(np.abs(lr.nonline_change_db(freqs, psd_before, psd_after, targets)))
            ),
            **lr.probe_recovery(
                lr.recover_probe(cleaned.get_data(), cleaned_bare.get_data()),
                clean(probe.waveform(times)[None, :].repeat(4, axis=0)).get_data()[0],
                times,
                probe,
            ),
            "removed_band_fraction": lr.removed_band_fraction(
                freqs, targets, widths, band_hz=(28.0, 48.0)
            ),
        }
        verdict = lr.PreservationGate().evaluate(metrics)
        assert all(verdict.values()), (verdict, metrics)
