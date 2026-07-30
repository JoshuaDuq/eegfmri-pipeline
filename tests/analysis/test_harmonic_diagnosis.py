"""Tests for the scanner-harmonic diagnosis estimators."""

from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.analysis import harmonic_diagnosis as hd


def _tone(frequency_hz: float, sfreq: float, n_times: int, amplitude: float, phase: float = 0.0):
    times = np.arange(n_times) / sfreq
    return amplitude * np.sin(2 * np.pi * frequency_hz * times + phase)


class TestTrCommensurateLength:
    def test_trims_to_whole_volumes(self):
        assert hd.tr_commensurate_length(11001, 500.0) == 10800

    def test_exact_multiple_is_unchanged(self):
        assert hd.tr_commensurate_length(10800, 500.0) == 10800

    def test_five_kilohertz_grid(self):
        assert hd.tr_commensurate_length(60000, 5000.0) == 58500

    def test_rejects_non_integer_samples_per_tr(self):
        with pytest.raises(ValueError, match="not an integer"):
            hd.tr_commensurate_length(10000, 333.0)

    def test_rejects_segment_shorter_than_one_tr(self):
        with pytest.raises(ValueError, match="shorter than one TR"):
            hd.tr_commensurate_length(100, 500.0)


class TestHannPeriodogram:
    def test_recovers_tone_frequency(self):
        sfreq, n_times = 500.0, 10800
        signal = _tone(57.1759, sfreq, n_times, amplitude=3.0)
        freqs, psd = hd.hann_periodogram(signal, sfreq)
        assert freqs[int(np.argmax(psd))] == pytest.approx(57.1759, abs=freqs[1])

    def test_parseval_scaling_matches_windowed_mean_square(self):
        rng = np.random.default_rng(0)
        sfreq, n_times = 500.0, 10800
        signal = rng.normal(size=n_times)
        freqs, psd = hd.hann_periodogram(signal, sfreq)
        window = np.hanning(n_times)
        expected = np.sum((signal * window) ** 2) / (sfreq * np.sum(window**2))
        assert np.sum(psd) * freqs[1] == pytest.approx(
            expected * sfreq * freqs[1] / freqs[1], rel=0.02
        )

    def test_handles_leading_dimensions(self):
        sfreq, n_times = 500.0, 900
        data = np.stack([_tone(20.0, sfreq, n_times, 1.0), _tone(40.0, sfreq, n_times, 1.0)])
        freqs, psd = hd.hann_periodogram(data, sfreq)
        assert psd.shape == (2, freqs.size)
        assert freqs[int(np.argmax(psd[0]))] == pytest.approx(20.0, abs=freqs[1])
        assert freqs[int(np.argmax(psd[1]))] == pytest.approx(40.0, abs=freqs[1])

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            hd.hann_periodogram(np.array([1.0, np.nan, 2.0]), 500.0)


class TestLocalBackground:
    def test_flat_spectrum_gives_zero_prominence(self):
        spectrum = np.full(500, -110.0)
        prom = hd.prominence_db(spectrum, half_width_bins=50)
        assert np.allclose(prom[50:-50], 0.0)

    def test_edges_are_nan(self):
        spectrum = np.full(500, -110.0)
        background = hd.local_background_db(spectrum, half_width_bins=50)
        assert np.all(np.isnan(background[:50]))
        assert np.all(np.isnan(background[-50:]))

    def test_isolated_line_keeps_its_prominence(self):
        spectrum = np.full(500, -110.0)
        spectrum[250] = -95.0
        prom = hd.prominence_db(spectrum, half_width_bins=50)
        assert prom[250] == pytest.approx(15.0)

    def test_core_exclusion_keeps_line_out_of_its_own_background(self):
        spectrum = np.full(41, -110.0)
        spectrum[18:23] = -90.0
        with_core = hd.prominence_db(spectrum, half_width_bins=20, core_bins=2)
        assert with_core[20] == pytest.approx(20.0)

    def test_survives_several_lines_inside_the_window(self):
        spectrum = np.full(500, -110.0)
        for index in range(200, 300, 24):
            spectrum[index] = -90.0
        prom = hd.prominence_db(spectrum, half_width_bins=50)
        assert prom[248] == pytest.approx(20.0)

    def test_rejects_core_wider_than_window(self):
        with pytest.raises(ValueError, match="core_bins must be smaller"):
            hd.local_background_db(np.zeros(100), half_width_bins=5, core_bins=5)


class TestRobustNull:
    def test_recovers_gaussian_scale(self):
        rng = np.random.default_rng(1)
        values = rng.normal(loc=0.0, scale=2.0, size=20000)
        location, scale = hd.robust_null(values)
        assert location == pytest.approx(0.0, abs=0.1)
        assert scale == pytest.approx(2.0, rel=0.05)

    def test_upper_tail_contamination_does_not_inflate_scale(self):
        rng = np.random.default_rng(2)
        values = np.concatenate([rng.normal(scale=1.0, size=5000), rng.normal(loc=30, size=200)])
        _, scale = hd.robust_null(values)
        assert scale == pytest.approx(1.0, rel=0.1)

    def test_rejects_tiny_sample(self):
        with pytest.raises(ValueError, match="at least 32"):
            hd.robust_null(np.zeros(10))


class TestFdr:
    def test_uniform_pvalues_yield_few_discoveries(self):
        rng = np.random.default_rng(3)
        qvalues = hd.fdr_bh(rng.uniform(size=5000))
        assert np.sum(qvalues < 0.05) == 0

    def test_strong_signal_survives(self):
        pvalues = np.concatenate([np.full(10, 1e-12), np.linspace(0.05, 1.0, 990)])
        qvalues = hd.fdr_bh(pvalues)
        assert np.sum(qvalues < 0.05) == 10

    def test_monotone_in_the_sorted_order(self):
        rng = np.random.default_rng(4)
        pvalues = rng.uniform(size=200)
        qvalues = hd.fdr_bh(pvalues)
        order = np.argsort(pvalues)
        assert np.all(np.diff(qvalues[order]) >= -1e-12)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="inside"):
            hd.fdr_bh([0.5, 1.5])


class TestClusterPeaks:
    def test_one_cluster_yields_its_largest_bin(self):
        flags = np.zeros(20, dtype=bool)
        flags[5:9] = True
        values = np.zeros(20)
        values[5:9] = [1.0, 4.0, 2.0, 1.0]
        assert hd.cluster_peaks(flags, values) == [6]

    def test_separated_clusters_stay_separate(self):
        flags = np.zeros(30, dtype=bool)
        flags[3:5] = True
        flags[20:22] = True
        values = np.arange(30, dtype=float)
        assert hd.cluster_peaks(flags, values) == [4, 21]

    def test_single_quiet_bin_does_not_split_a_line(self):
        flags = np.array([False, True, False, True, False])
        values = np.array([0.0, 3.0, 0.0, 9.0, 0.0])
        assert hd.cluster_peaks(flags, values, join_gap_bins=1) == [3]

    def test_zero_gap_keeps_them_apart(self):
        flags = np.array([False, True, False, True, False])
        values = np.array([0.0, 3.0, 0.0, 9.0, 0.0])
        assert hd.cluster_peaks(flags, values, join_gap_bins=0) == [1, 3]

    def test_nothing_significant_gives_no_lines(self):
        assert hd.cluster_peaks(np.zeros(10, dtype=bool), np.zeros(10)) == []


class TestRefinePeakFrequency:
    def test_recovers_off_bin_tone(self):
        sfreq, n_times = 500.0, 10800
        true_hz = 57.1759 + 0.017
        freqs, psd = hd.hann_periodogram(_tone(true_hz, sfreq, n_times, 1.0), sfreq)
        db = hd.to_db(psd)
        peak = int(np.argmax(db))
        refined = hd.refine_peak_frequency(freqs, db, peak)
        assert abs(refined - true_hz) < abs(freqs[peak] - true_hz)
        assert refined == pytest.approx(true_hz, abs=0.005)

    def test_on_bin_tone_is_left_alone(self):
        sfreq, n_times = 500.0, 10800
        true_hz = 24 / 0.9  # exactly on a bin of the commensurate grid
        freqs, psd = hd.hann_periodogram(_tone(true_hz, sfreq, n_times, 1.0), sfreq)
        db = hd.to_db(psd)
        refined = hd.refine_peak_frequency(freqs, db, int(np.argmax(db)))
        assert refined == pytest.approx(true_hz, abs=1e-3)


class TestSpectralLinewidth:
    def test_pure_tone_is_as_narrow_as_the_window_allows(self):
        sfreq, n_times = 500.0, 10800
        freqs, psd = hd.hann_periodogram(_tone(57.2, sfreq, n_times, 1.0), sfreq)
        db = hd.to_db(psd)
        width = hd.spectral_linewidth_hz(freqs, db, int(np.argmax(db)))
        assert width == pytest.approx(hd.hann_resolution_hz(n_times / sfreq), rel=0.15)

    def test_a_broad_resonance_is_much_wider(self):
        # A 2 Hz-wide Gaussian bump standing in for an alpha peak.
        freqs = np.arange(0, 100, 1 / 21.6)
        db = -120 + 12 * np.exp(-0.5 * ((freqs - 10.0) / 0.85) ** 2)
        width = hd.spectral_linewidth_hz(freqs, db, int(np.argmin(np.abs(freqs - 10.0))))
        # 2 * sigma * sqrt(2 ln(4/3)) for a 12 dB Gaussian, and 19 window widths across.
        assert width == pytest.approx(1.289, abs=0.02)
        assert width > 15 * hd.hann_resolution_hz(21.6)

    def test_returns_nan_when_the_peak_never_falls_away(self):
        freqs = np.arange(0, 100, 1 / 21.6)
        db = np.full(freqs.size, -110.0)
        assert np.isnan(hd.spectral_linewidth_hz(freqs, db, 500))

    def test_hann_resolution_scales_inversely_with_duration(self):
        assert hd.hann_resolution_hz(21.6) == pytest.approx(hd.hann_resolution_hz(10.8) / 2)


class TestCombIndex:
    def test_volume_harmonic_is_on_comb(self):
        position = hd.comb_index(55 / 0.9)
        assert position.harmonic_index == 55
        assert position.on_comb

    def test_off_comb_line_reports_its_offset(self):
        position = hd.comb_index(57.1759)
        assert position.harmonic_index == 51
        assert not position.on_comb
        assert position.offset_hz == pytest.approx(57.1759 - 51 / 0.9, abs=1e-9)

    def test_sixty_hertz_coincides_with_harmonic_54(self):
        position = hd.comb_index(60.0)
        assert position.harmonic_index == 54
        assert position.on_comb


class TestCombPhaseAndUniformity:
    def test_comb_lines_cluster_at_zero(self):
        freqs = np.array([18, 27, 37, 45, 55, 62, 74, 80, 88, 95]) / 0.9
        assert np.allclose(hd.comb_phase(freqs), 0.0, atol=1e-9)
        assert hd.comb_uniformity_pvalue(freqs) < 1e-3

    def test_four_perfect_comb_lines_are_only_weak_evidence(self):
        # Directional statistics on a handful of lines cannot be decisive, and the test
        # should not pretend otherwise.
        freqs = np.array([18, 37, 55, 74]) / 0.9
        assert 1e-3 < hd.comb_uniformity_pvalue(freqs) < 0.05

    def test_unrelated_frequencies_do_not_cluster(self):
        rng = np.random.default_rng(5)
        freqs = rng.uniform(30, 90, size=40)
        assert hd.comb_uniformity_pvalue(freqs) > 0.05


class TestFitArithmeticComb:
    def test_recovers_known_spacing(self):
        spacing = 2.40741
        freqs = 51.574 + spacing * np.arange(5)
        fit = hd.fit_arithmetic_comb(freqs)
        assert fit.spacing_hz == pytest.approx(spacing, abs=1e-6)
        assert fit.intercept_hz == pytest.approx(51.574, abs=1e-6)
        assert fit.rmse_hz < 1e-9

    def test_tolerates_a_missing_member(self):
        spacing = 2.40741
        freqs = 51.574 + spacing * np.array([0, 1, 3, 4])
        fit = hd.fit_arithmetic_comb(freqs)
        assert fit.indices == (0, 1, 3, 4)
        assert fit.spacing_hz == pytest.approx(spacing, abs=1e-6)

    def test_reports_residuals_for_an_imperfect_family(self):
        freqs = np.array([51.574, 53.981, 56.389, 58.796])
        fit = hd.fit_arithmetic_comb(freqs)
        assert fit.spacing_hz == pytest.approx(2.4074, abs=1e-3)
        assert fit.max_abs_residual_hz < 1e-3

    def test_rejects_too_few_lines(self):
        with pytest.raises(ValueError, match="at least three"):
            hd.fit_arithmetic_comb([1.0, 2.0])


class TestDominantSpacing:
    def test_finds_the_repeated_gap(self):
        freqs = np.concatenate([51.574 + 2.4074 * np.arange(5), [37.1, 47.0]])
        spacing, support = hd.dominant_spacing(freqs, max_difference_hz=12.0, tolerance_hz=0.05)
        assert spacing == pytest.approx(2.4074, abs=0.01)
        assert support >= 4


class TestRefineCombFundamental:
    def test_recovers_the_fundamental_when_the_commonest_gap_is_a_multiple(self):
        # A 1.2 Hz comb with every third member missing: two-apart pairs outnumber
        # adjacent ones, so the most-supported gap is 2.4 Hz.
        freqs = np.array([1.2 * k for k in range(22, 60) if k % 3])
        spacing, _ = hd.dominant_spacing(freqs, max_difference_hz=12.0, tolerance_hz=0.06)
        fundamental, members = hd.refine_comb_fundamental(freqs, spacing, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-6)
        assert members.all()

    def test_leaves_a_true_fundamental_alone(self):
        freqs = np.array([1.2 * k for k in range(22, 50)])
        fundamental, members = hd.refine_comb_fundamental(freqs, 1.2, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-9)
        assert members.all()

    def test_does_not_subdivide_for_one_stray_line(self):
        freqs = np.concatenate([[1.2 * k for k in range(22, 50)], [58.2]])
        fundamental, _ = hd.refine_comb_fundamental(freqs, 1.2, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-9)

    def test_excludes_lines_that_are_not_multiples(self):
        freqs = np.array([1.2 * k for k in range(22, 50)] + [57.2247, 47.0362])
        fundamental, members = hd.refine_comb_fundamental(freqs, 2.4, tolerance_hz=0.06)
        assert fundamental == pytest.approx(1.2, abs=1e-6)
        assert not members[-1] and not members[-2]

    def test_rejects_a_non_positive_spacing(self):
        with pytest.raises(ValueError, match="finite positive"):
            hd.refine_comb_fundamental([1.0, 2.0], 0.0, tolerance_hz=0.1)


class TestBootstrapCi:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(6)
        values = rng.normal(loc=10.0, scale=2.0, size=15)
        point, low, high = hd.bootstrap_ci(values, n_resamples=2000, seed=7)
        assert low <= point <= high

    def test_is_deterministic_for_a_fixed_seed(self):
        values = np.arange(15, dtype=float)
        assert hd.bootstrap_ci(values, seed=11) == hd.bootstrap_ci(values, seed=11)

    def test_mean_statistic_is_available(self):
        values = np.array([1.0, 2.0, 30.0])
        assert hd.bootstrap_ci(values, statistic="mean")[0] == pytest.approx(11.0)


class TestRayleigh:
    def test_uniform_phases_are_not_significant(self):
        phases = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        resultant, pvalue = hd.rayleigh_test(phases)
        assert resultant < 1e-9
        assert pvalue > 0.9

    def test_concentrated_phases_are_significant(self):
        rng = np.random.default_rng(8)
        phases = rng.normal(loc=1.0, scale=0.2, size=66)
        resultant, pvalue = hd.rayleigh_test(phases)
        assert resultant > 0.9
        assert pvalue < 1e-10


class TestVolumeLockedPhases:
    def test_deterministic_comb_line_locks_after_rotation(self):
        sfreq, n_times, tr = 500.0, 10800, 0.9
        frequency = 55 / tr
        rng = np.random.default_rng(9)
        offsets = rng.uniform(0, tr, size=40)
        coefficients = []
        for offset in offsets:
            # A residual whose phase is fixed in the scanner's frame, seen from a segment
            # whose start precedes the volume marker by `offset` seconds.
            times = np.arange(n_times) / sfreq
            signal = np.sin(2 * np.pi * frequency * (times - offset) + 0.7)
            spectrum = np.fft.rfft(signal * np.hanning(n_times))
            coefficients.append(spectrum[int(round(frequency * n_times / sfreq))])
        phases = hd.volume_locked_phases(coefficients, offsets, frequency)
        assert hd.rayleigh_test(phases)[0] > 0.99

    def test_any_volume_marker_in_the_segment_gives_the_same_answer(self):
        # Phase advances by a whole number of cycles per TR at a comb frequency, so it
        # cannot matter which of the segment's volume markers is used as the origin.
        rng = np.random.default_rng(14)
        frequency = 55 / 0.9
        coefficients = np.exp(1j * rng.uniform(0, 2 * np.pi, size=30))
        offsets = rng.uniform(0, 0.9, size=30)
        first = hd.volume_locked_phases(coefficients, offsets, frequency)
        later = hd.volume_locked_phases(coefficients, offsets + 7 * 0.9, frequency)
        assert np.allclose(np.exp(1j * first), np.exp(1j * later), atol=1e-9)

    def test_random_phase_source_does_not_lock(self):
        rng = np.random.default_rng(10)
        coefficients = np.exp(1j * rng.uniform(0, 2 * np.pi, size=200))
        offsets = rng.uniform(0, 0.9, size=200)
        phases = hd.volume_locked_phases(coefficients, offsets, 55 / 0.9)
        assert hd.rayleigh_test(phases)[1] > 0.05

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="same shape"):
            hd.volume_locked_phases([1 + 0j, 1 + 0j], [0.1], 20.0)


class TestVarianceComponents:
    def test_pure_between_group_variance(self):
        values = np.repeat([0.0, 10.0, 20.0], 6)
        groups = np.repeat(["a", "b", "c"], 6)
        result = hd.one_way_variance_components(values, groups)
        assert result["variance_within"] == pytest.approx(0.0)
        assert result["icc"] == pytest.approx(1.0)

    def test_pure_within_group_variance(self):
        rng = np.random.default_rng(12)
        groups = np.repeat(list("abcdefghij"), 40)
        values = rng.normal(size=groups.size)
        result = hd.one_way_variance_components(values, groups)
        assert result["icc"] < 0.1

    def test_recovers_a_known_split(self):
        rng = np.random.default_rng(13)
        n_groups, per_group = 40, 12
        group_effects = rng.normal(scale=3.0, size=n_groups)
        groups = np.repeat(np.arange(n_groups), per_group)
        values = group_effects[groups] + rng.normal(scale=1.0, size=groups.size)
        result = hd.one_way_variance_components(values, groups)
        assert result["variance_between"] == pytest.approx(9.0, rel=0.4)
        assert result["variance_within"] == pytest.approx(1.0, rel=0.2)

    def test_requires_two_groups(self):
        with pytest.raises(ValueError, match="at least two groups"):
            hd.one_way_variance_components([1.0, 2.0], ["a", "a"])


class TestBandPower:
    def test_integrates_in_the_power_domain(self):
        freqs = np.arange(0, 100.5, 0.5)
        psd = np.ones_like(freqs)
        psd[freqs == 50.0] = 1e6
        with_line = hd.band_power_db(freqs, psd, low_hz=30.0, high_hz=80.0)
        without = hd.band_power_db(
            freqs, psd, low_hz=30.0, high_hz=80.0, excluded_hz=[(49.5, 50.5)]
        )
        assert with_line - without > 30.0

    def test_excluding_everything_is_an_error(self):
        freqs = np.arange(0, 100.5, 0.5)
        with pytest.raises(ValueError, match="No frequency bins remain"):
            hd.band_power_db(
                freqs,
                np.ones_like(freqs),
                low_hz=30.0,
                high_hz=40.0,
                excluded_hz=[(20.0, 50.0)],
            )


class TestLineExclusionWindows:
    def test_merges_overlapping_windows(self):
        windows = hd.line_exclusion_windows([57.10, 57.18, 57.22], half_width_hz=0.15)
        assert windows == ((56.95, 57.37),)

    def test_keeps_separated_windows_apart(self):
        windows = hd.line_exclusion_windows([51.57, 57.18], half_width_hz=0.15)
        assert len(windows) == 2

    def test_empty_input_gives_no_windows(self):
        assert hd.line_exclusion_windows([], half_width_hz=0.15) == ()
