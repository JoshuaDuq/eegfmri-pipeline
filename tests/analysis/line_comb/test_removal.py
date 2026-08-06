"""Tests for line-comb estimation and removal."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis.line_comb import diagnosis as hd
from studies.pain_study.analysis.line_comb import removal as lr


def test_thomson_f_test_identifies_a_channel_specific_stationary_line():
    sampling_frequency_hz = 200.0
    times = np.arange(2_000) / sampling_frequency_hz
    rng = np.random.default_rng(42)
    data = rng.normal(size=(2, times.size))
    data[1] += 4.0 * np.sin(2.0 * np.pi * 33.0 * times)

    freqs, statistic, threshold, p_values = lr.thomson_f_statistics(
        data,
        sampling_frequency_hz=sampling_frequency_hz,
        bandwidth_hz=1.2,
    )

    line = int(np.argmin(np.abs(freqs - 33.0)))
    assert statistic[1, line] > threshold
    assert statistic[0, line] < threshold
    assert p_values[1, line] < p_values[0, line]
    assert p_values.shape == statistic.shape


def test_focal_residual_candidates_stay_inside_authorised_neighborhoods():
    freqs = np.arange(20.0, 30.0, 0.05)
    statistic = np.zeros((2, freqs.size))
    statistic[0, np.argmin(np.abs(freqs - 25.10))] = 100.0
    statistic[1, np.argmin(np.abs(freqs - 27.00))] = 100.0

    candidates = lr.focal_residual_line_candidates(
        freqs,
        statistic,
        threshold=10.0,
        targets_hz=(25.0,),
        widths_hz=(0.1,),
        responsibility_hz=0.15,
    )

    assert candidates[0] == pytest.approx((25.10,))
    assert candidates[1] == ()


def test_shared_residual_candidates_need_agreement_across_the_array():
    freqs = np.arange(20.0, 30.0, 0.05)
    statistic = np.zeros((4, freqs.size))
    shared = int(np.argmin(np.abs(freqs - 25.10)))
    lone = int(np.argmin(np.abs(freqs - 24.95)))
    statistic[:3, shared] = 100.0
    statistic[0, lone] = 100.0

    candidates = lr.shared_residual_line_candidates(
        freqs,
        statistic,
        threshold=10.0,
        targets_hz=(25.0,),
        widths_hz=(0.1,),
        responsibility_hz=0.15,
        min_channel_fraction=0.5,
    )

    assert candidates == pytest.approx((25.10,))


def test_shared_residual_candidates_stay_inside_authorised_neighborhoods():
    freqs = np.arange(20.0, 30.0, 0.05)
    statistic = np.zeros((2, freqs.size))
    statistic[:, np.argmin(np.abs(freqs - 27.00))] = 100.0

    candidates = lr.shared_residual_line_candidates(
        freqs,
        statistic,
        threshold=10.0,
        targets_hz=(25.0,),
        widths_hz=(0.1,),
        responsibility_hz=0.15,
        min_channel_fraction=0.5,
    )

    assert candidates == ()


def test_shared_residual_candidates_do_not_reject_within_window_drift():
    freqs = np.arange(20.0, 30.0, 0.05)
    statistic = np.zeros((2, freqs.size))
    statistic[:, np.abs(freqs - 25.0) <= 0.15] = 100.0

    candidates = lr.shared_residual_line_candidates(
        freqs,
        statistic,
        threshold=10.0,
        targets_hz=(25.0,),
        widths_hz=(0.1,),
        responsibility_hz=0.15,
        min_channel_fraction=0.5,
    )

    assert len(candidates) == 1
    assert abs(candidates[0] - 25.0) <= 0.15


def test_benjamini_hochberg_rejects_the_step_up_set():
    p_values = [0.001, 0.008, 0.039, 0.041, 0.900]

    assert lr.benjamini_hochberg_discoveries(p_values, false_discovery_rate=0.05) == 2


def test_benjamini_hochberg_steps_up_past_a_p_below_its_own_rank():
    """0.020 fails its own rank threshold and is still rejected, which is the procedure."""
    p_values = [0.020, 0.021, 0.040]

    assert lr.benjamini_hochberg_discoveries(p_values, false_discovery_rate=0.05) == 3


def test_benjamini_hochberg_makes_no_discovery_under_the_null():
    p_values = np.linspace(0.05, 0.95, 90)

    assert lr.benjamini_hochberg_discoveries(p_values, false_discovery_rate=0.05) == 0


def test_run_residual_sinusoid_p_value_corrects_for_the_family_searched():
    """One uncorrected 0.01 among 200 tests is not evidence of a surviving line."""
    p_values = np.full(200, 0.5)
    p_values[7] = 0.01

    assert lr.run_residual_sinusoid_p_value(p_values) == pytest.approx(1.0)
    assert lr.run_residual_sinusoid_p_value([0.01, 0.5]) == pytest.approx(0.02)


def test_residual_sinusoid_verdict_decides_over_runs_not_inside_them():
    """A nominal-rate detection in a few runs is not a cohort defect.

    Demanding zero significant residuals inside every run rejects a clean cohort at the
    test's own error rate. The cohort decision is made across runs instead, the same way
    the seam criterion is.
    """
    clean_cohort = [0.04, 0.045, 0.048, *np.linspace(0.2, 0.99, 87)]
    real_defect = [1e-9, 1e-8, 1e-7, *np.linspace(0.2, 0.99, 87)]

    assert lr.residual_sinusoid_verdict(clean_cohort)["passed"]
    assert not lr.residual_sinusoid_verdict(real_defect)["passed"]


def test_in_band_probe_frequencies_come_from_the_fitted_targets():
    """Positions are read off the plan, so the measurement transfers to another site."""
    targets = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)

    frequencies = lr.in_band_probe_frequencies(targets, count=4)

    assert len(frequencies) == 4
    assert set(frequencies) <= set(targets)
    assert frequencies == tuple(sorted(frequencies))
    assert frequencies[0] == 10.0
    assert frequencies[-1] == 80.0


def test_in_band_probe_frequencies_never_exceed_the_targets_available():
    frequencies = lr.in_band_probe_frequencies((25.0, 50.0), count=4)

    assert frequencies == (25.0, 50.0)


def test_in_band_probe_survival_measures_what_is_left_at_the_target():
    freqs = np.arange(0.0, 100.0, 0.5)
    before = np.zeros((1, freqs.size))
    after = np.zeros((1, freqs.size))
    before[0, np.argmin(np.abs(freqs - 25.0))] = 1.0
    after[0, np.argmin(np.abs(freqs - 25.0))] = 0.25
    before[0, np.argmin(np.abs(freqs - 50.0))] = 1.0
    after[0, np.argmin(np.abs(freqs - 50.0))] = 1.0

    survival = lr.in_band_probe_survival(freqs, before, after, (25.0, 50.0))

    assert survival["min_in_band_probe_survival"] == pytest.approx(0.25)
    assert survival["median_in_band_probe_survival"] == pytest.approx(0.625)


def test_residual_detection_carries_no_acceptance_tolerance():
    """The detector's thresholds are its own, so the gate stays able to fail.

    A detector parameterised by the gate's tolerance removes precisely what the gate
    would flag; this pins that the two objects share no field.
    """
    detection = set(lr.ResidualDetection().__dataclass_fields__)
    acceptance = set(lr.PreservationGate().__dataclass_fields__)

    assert detection & acceptance == set()


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


def test_robust_harmonic_membership_converges_past_four_updates():
    """Edge candidates may leave the fit gradually; convergence is data-dependent."""
    harmonics = np.arange(10, 30, dtype=float)
    positions = np.array(
        [
            11.93473118,
            13.24327174,
            14.55756896,
            15.57223142,
            16.83811310,
            18.05974891,
            19.11977912,
            20.35723348,
            21.54901720,
            22.73810076,
            24.04443747,
            25.23857092,
            26.37201215,
            27.55570849,
            28.73462484,
            30.10186407,
            31.09025268,
            32.38091373,
            33.52333588,
            34.86104043,
        ]
    )
    weights = np.array(
        [
            4.631908,
            14.913439,
            5.416903,
            1.854552,
            9.997043,
            5.935572,
            5.230769,
            8.857639,
            13.852629,
            5.746439,
            5.409724,
            14.598954,
            11.144800,
            14.357612,
            9.104984,
            5.630445,
            1.222079,
            13.837141,
            1.899266,
            9.494959,
        ]
    )

    used, used_positions, _, fundamental = lr._fit_consistent_harmonics(
        harmonics,
        positions,
        weights,
        min_harmonics=3,
        max_harmonic_residual_hz=0.06,
    )

    assert len(used) >= 3
    assert np.max(np.abs(used_positions - used * fundamental)) <= 0.06


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

    def test_validated_harmonics_retain_their_measured_peak_positions(self):
        freqs, db, prom = synthetic_spectrum(fundamental=1.19993)

        estimate = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=())
        measured = dict(zip(estimate.harmonics_used, estimate.harmonic_positions_hz))
        targets = lr.removal_frequencies(estimate, harmonic_range=(45, 45), excluded_hz=())

        assert targets == (measured[45],)

    def test_a_strong_off_grid_peak_cannot_pull_the_comb_fundamental(self):
        """One nearby machine line is not evidence that every harmonic moved."""
        freqs, db, prom = synthetic_spectrum(fundamental=1.2, extra=(61.05,))
        displaced = int(np.argmin(np.abs(freqs - 61.05)))
        db = db.copy()
        db[displaced] += 25.0
        prom = hd.prominence_db(db, half_width_bins=100)

        estimate = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=())

        assert estimate.fundamental_hz == pytest.approx(1.2, abs=2e-4)
        assert estimate.max_abs_residual_hz <= lr.MAX_HARMONIC_RESIDUAL_HZ
        assert 51 not in estimate.harmonics_used

    def test_finds_the_isolated_lines(self):
        freqs, db, prom = synthetic_spectrum(extra=(57.22, 47.04))
        est = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(57.2247, 47.0362))
        assert len(est.isolated_hz) == 2
        assert est.isolated_hz[0] == pytest.approx(57.22, abs=0.03)
        assert est.isolated_hz[1] == pytest.approx(47.04, abs=0.03)

    def test_an_isolated_target_stays_at_its_replicated_session_nominal(self):
        """A window may confirm a source but cannot move the deletion target itself."""
        nominal = 57.2247
        freqs, db, prom = synthetic_spectrum(extra=(57.20,))

        estimate = lr.estimate_comb(freqs, db, prom, isolated_nominal_hz=(nominal,))

        assert estimate.isolated_hz == (nominal,)

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
        assert est.isolated_hz == (57.2247, 57.3485)
        assert np.all(np.isfinite(est.isolated_prominence_db))

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
        assert est.isolated_hz == (57.1432, 57.3485)
        assert np.all(np.isfinite(est.isolated_prominence_db))

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
            harmonic_positions_hz=(28.8, 30.0),
            residual_rms_hz=0.0,
            max_abs_residual_hz=0.0,
            fundamental_jackknife_se_hz=1e-4,
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

    def test_keeps_mains_when_the_exclusion_is_turned_off(self):
        targets = lr.removal_frequencies(
            self._estimate(isolated=(57.22, 60.0)), harmonic_range=(24, 79), excluded_hz=()
        )
        assert any(
            f == pytest.approx(60.0) for f in targets
        ), "with the FIR notch disabled the removal must take mains itself"

    def test_a_nan_line_is_dropped_from_the_removal_list(self):
        estimate = self._estimate(isolated=(57.2, float("nan")))
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
        targets = [1.2 * k for k in range(24, 80)]
        lr.check_probe_clearance(lr.Probe(), targets)

    def test_the_default_probes_clear_the_comb_wherever_the_fundamental_sits(self):
        """A static grid is not the grid the adaptive model actually removes on.

        The check above plants the comb at exactly 1.2 Hz, so it only ever asks whether the
        probes clear one nominal grid. The adaptive model fits a fundamental per 54-second
        window, and harmonic k moves by k times that wander -- so the grid the probes have
        to clear is a band, not a line. On 2026-08-03 the 90-recording benchmark aborted at
        recording 24: window 4 of sub-0004 run 6 fitted 1.1985926 Hz, which put harmonic 37
        at 44.3479 Hz, 0.298 Hz from the 44.05 Hz probe and inside the 0.3 Hz requirement.

        The envelope below spans every per-window fundamental measured across the cohort
        (lowest 1.198593, highest 1.200334) with room to spare.
        """
        for fundamental in np.linspace(1.198, 1.202, 81):
            targets = [fundamental * k for k in range(22, 84)]
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

    def test_probe_preservation_cannot_average_away_one_failed_channel(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        before = np.ones((64, freqs.size))
        after = before.copy()
        probe = lr.Probe()
        index = int(np.argmin(np.abs(freqs - probe.sinusoid_hz[0])))
        after[0, index] = 0.01

        result = lr.probe_preservation(freqs, before, after, probe)

        assert result["max_probe_deviation_db"] == pytest.approx(20.0)

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

    def test_probe_recovery_rejects_a_time_axis_mismatch(self):
        probe = lr.Probe(burst_centre_s=0.5)
        recovered = np.ones((2, 100))
        reference = np.ones(100)

        with pytest.raises(ValueError, match="time axis"):
            lr.probe_recovery(recovered, reference, np.arange(99) / 100.0, probe)

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
            "max_residual_prominence_db": -12.0,
            "null_max_95_db": -8.0,
            "residual_excess_db": -4.0,
            "focal_residual_excess_db": -2.0,
            "study_residual_excess_db": -4.0,
            "study_focal_residual_excess_db": -2.0,
            "study_significant_focal_residual_count": 0,
            "max_boundary_discontinuity_ratio": 1.0,
            "median_suppression_db": 28.0,
            "intrinsic_energy_ratio": 0.95,
            "max_probe_deviation_db": 0.01,
            "max_nonline_change_db": 0.001,
            "study_max_probe_deviation_db": 0.01,
            "study_max_nonline_change_db": 0.001,
            "burst_energy_ratio": 0.998,
            "burst_correlation": 0.999,
            "removed_band_fraction": 0.06,
            "base_removed_band_fraction": 0.06,
            "base_band_fraction_bin_size": 1.0 / 1810.0,
            "band_fraction_bin_size": 1.0 / 1810.0,
        }
        base.update(overrides)
        return base

    def test_a_good_removal_passes_every_criterion(self):
        gate = lr.PreservationGate()
        assert gate.passed(self._metrics())
        assert all(gate.evaluate(self._metrics()).values())

    def test_leftover_lines_fail(self):
        gate = lr.PreservationGate()
        verdict = gate.evaluate(self._metrics(residual_excess_db=6.0))
        assert not verdict["lines_suppressed"]

    def test_a_removed_probe_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(max_probe_deviation_db=1.2))["sinusoids_preserved"]

    def test_broadband_damage_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(max_nonline_change_db=0.9))["spectrum_preserved"]

    def test_a_flattened_transient_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(intrinsic_energy_ratio=0.80))["transient_preserved"]

    def test_an_inflated_transient_also_fails(self):
        gate = lr.PreservationGate()
        assert not gate.evaluate(self._metrics(intrinsic_energy_ratio=1.20))["transient_preserved"]

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
        # 24-79, as the cohort fit uses. At 24-39 this fixture carried sixteen
        # harmonics -- below the floor that now refuses to build a removal grid on
        # evidence that thin, which is the fixture being unrepresentative, not the
        # floor being wrong.
        for harmonic in range(24, 80):
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
            freqs, db, prom, harmonic_range=(24, 79), isolated_nominal_hz=()
        )
        assert estimate.fundamental_hz == pytest.approx(fundamental, abs=5e-4)

        targets = lr.removal_frequencies(estimate, harmonic_range=(24, 79), low_hz=3.0)
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
            "focal_residual_excess_db": -2.0,
            "study_residual_excess_db": -2.0,
            "study_focal_residual_excess_db": -2.0,
            "study_significant_focal_residual_count": 0,
            "max_boundary_discontinuity_ratio": 1.0,
            **lr.probe_preservation(freqs, psd_before, psd_after, probe),
            "max_nonline_change_db": float(
                np.max(np.abs(lr.nonline_change_db(freqs, psd_before, psd_after, targets)))
            ),
            "study_max_probe_deviation_db": 0.01,
            "study_max_nonline_change_db": 0.001,
            **lr.probe_recovery(
                lr.recover_probe(cleaned.get_data(), cleaned_bare.get_data()),
                clean(probe.waveform(times)[None, :].repeat(4, axis=0)).get_data()[0],
                times,
                probe,
            ),
            "removed_band_fraction": lr.removed_band_fraction(
                freqs, targets, widths, band_hz=(28.0, 48.0)
            ),
            "base_removed_band_fraction": lr.removed_band_fraction(
                freqs, targets, widths, band_hz=(28.0, 48.0)
            ),
            "base_band_fraction_bin_size": 1.0
            / np.count_nonzero((freqs >= 28.0) & (freqs <= 48.0)),
            "band_fraction_bin_size": 1.0 / np.count_nonzero((freqs >= 28.0) & (freqs <= 48.0)),
        }
        verdict = lr.PreservationGate().evaluate(metrics)
        assert all(verdict.values()), (verdict, metrics)
