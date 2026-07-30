"""Tests for the scanner-harmonic diagnosis runner.

The heavy stages read a multi-gigabyte drive, so what is exercised here is the analysis
logic that turns cached spectra into the reported statistics. Synthetic spectra with
lines planted at known frequencies stand in for the cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.analysis.line_comb import diagnosis as hd
from studies.pain_study.scripts.line_comb import diagnose as ds

TR = hd.TR_SECONDS


def synthetic_cohort(line_freqs, *, n_subjects=15, amplitude_db=15.0, seed=0):
    """A 1/f cohort with narrow lines planted at the given frequencies."""
    freqs = np.arange(0, 110.0 + 1e-9, 1 / 21.6)
    rng = np.random.default_rng(seed)
    background = 10 ** ((-100.0 - 12.0 * np.log10(np.maximum(freqs, 0.5))) / 10.0)
    spectra = []
    for _ in range(n_subjects):
        spectrum = background * 10 ** (rng.normal(0, 0.05, freqs.size))
        for frequency in line_freqs:
            index = int(np.argmin(np.abs(freqs - frequency)))
            spectrum[index] *= 10 ** (amplitude_db / 10.0)
        spectra.append(spectrum)
    return freqs, np.stack(spectra)


class TestHalfWidthBins:
    def test_high_resolution_grid_gives_one_hundred_bins(self):
        freqs = np.arange(0, 110, 1 / 21.6)
        assert ds.half_width_bins(freqs) == 100

    def test_matched_grid_gives_a_proportionally_smaller_window(self):
        freqs = np.arange(0, 110, 1 / 3.6)
        assert ds.half_width_bins(freqs) == 17


class TestDetectionMask:
    def test_covers_the_analysis_range(self):
        freqs = np.arange(0, 110, 1 / 21.6)
        mask = ds.detection_mask(freqs)
        assert not mask[freqs < 3.0].any()
        assert not mask[freqs > 95.0].any()
        assert mask[np.argmin(np.abs(freqs - 45.0))]

    def test_excludes_the_mains_notch(self):
        freqs = np.arange(0, 110, 1 / 21.6)
        mask = ds.detection_mask(freqs)
        assert not mask[np.argmin(np.abs(freqs - 60.0))]
        assert mask[np.argmin(np.abs(freqs - 58.0))]


class TestChannelMedian:
    def test_drops_bad_channels(self):
        psd = np.array([[1.0, 1.0], [1.0, 1.0], [500.0, 500.0]])
        result = ds.channel_median(psd, ["Cz", "Pz", "FC2"], ["FC2"])
        assert np.allclose(result, 1.0)

    def test_keeps_everything_when_nothing_is_bad(self):
        psd = np.array([[1.0], [3.0], [5.0]])
        assert ds.channel_median(psd, ["a", "b", "c"], [])[0] == pytest.approx(3.0)


class TestDetectCohortLines:
    def test_recovers_planted_lines(self):
        planted = [37.0, 51.5740, 57.1759, 74.0]
        freqs, spectra = synthetic_cohort(planted)
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        found = lines["refined_hz"].to_numpy()
        assert len(found) == len(planted)
        for frequency in planted:
            assert np.min(np.abs(found - frequency)) < 0.05

    def test_reports_no_lines_when_the_spectrum_is_smooth(self):
        freqs, spectra = synthetic_cohort([])
        with pytest.raises(RuntimeError, match="No line survived"):
            ds.detect_cohort_lines(ds.build_grid(freqs, spectra))

    def test_flags_comb_membership_correctly(self):
        on_comb = 55 / TR
        off_comb = 57.1759
        freqs, spectra = synthetic_cohort([on_comb, off_comb])
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        flags = dict(zip(np.round(lines["refined_hz"], 2), lines["on_comb"]))
        assert flags[round(on_comb, 2)]
        assert not flags[round(off_comb, 2)]

    def test_ignores_a_line_inside_the_mains_notch(self):
        freqs, spectra = synthetic_cohort([60.0, 51.574])
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        assert np.min(np.abs(lines["refined_hz"].to_numpy() - 60.0)) > 0.5

    def test_prevalence_counts_every_participant_when_the_line_is_universal(self):
        freqs, spectra = synthetic_cohort([57.1759], amplitude_db=25.0)
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        assert int(lines["n_subjects_detected"].iloc[0]) == 15

    def test_confidence_interval_brackets_the_point_estimate(self):
        freqs, spectra = synthetic_cohort([57.1759])
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        row = lines.iloc[0]
        assert row["ci_low_db"] <= row["cohort_median_prominence_db"] <= row["ci_high_db"]


class TestBandImpact:
    def test_line_inside_a_band_inflates_it(self):
        freqs, spectra = synthetic_cohort([51.574], amplitude_db=30.0)
        grid = ds.build_grid(freqs, spectra)
        lines = ds.detect_cohort_lines(grid)
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        impact = ds.band_impact(grid, [f"sub-{i:04d}" for i in range(15)], classified)
        mid = impact.loc[impact["band"] == "gamma_mid_clean", "artifact_share_percent"]
        assert mid.min() > 5.0

    def test_band_without_lines_is_untouched(self):
        freqs, spectra = synthetic_cohort([51.574])
        grid = ds.build_grid(freqs, spectra)
        lines = ds.detect_cohort_lines(grid)
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        impact = ds.band_impact(grid, [f"sub-{i:04d}" for i in range(15)], classified)
        alpha = impact.loc[impact["band"] == "alpha"]
        assert (alpha["n_lines_inside"] == 0).all()
        assert np.allclose(alpha["artifact_share"], 0.0)

    def test_a_band_of_pure_background_reports_no_artifact(self):
        # The estimator error this replaced: dropping line bins made an empty band look
        # contaminated in proportion to how many bins were dropped.
        freqs, spectra = synthetic_cohort([51.574])
        grid = ds.build_grid(freqs, spectra)
        lines = ds.detect_cohort_lines(grid)
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        impact = ds.band_impact(grid, [f"sub-{i:04d}" for i in range(15)], classified)
        theta = impact.loc[impact["band"] == "theta", "artifact_share"]
        assert np.allclose(theta, 0.0)


class TestCombStructure:
    def test_recovers_a_planted_fundamental(self):
        fundamental = 1.2
        planted = [fundamental * k for k in range(22, 40)]
        freqs, spectra = synthetic_cohort(planted)
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        row = ds.comb_structure(lines).set_index("family").loc["narrow_comb"]
        assert row["fundamental_hz"] == pytest.approx(fundamental, abs=1e-3)
        assert row["harmonic_min"] == 22
        assert row["harmonic_max"] == 39
        assert row["rmse_hz"] < 0.02

    def test_separates_lines_that_do_not_join_the_comb(self):
        planted = [1.2 * k for k in range(22, 40)] + [57.2247]
        freqs, spectra = synthetic_cohort(planted)
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        structure = ds.comb_structure(lines).set_index("family")
        assert int(structure.loc["narrow_comb", "n_lines"]) == 18
        assert int(structure.loc["narrow_off_comb", "n_lines"]) == 1

    def test_widely_spaced_lines_report_no_comb_rather_than_failing(self):
        freqs, spectra = synthetic_cohort([20.0, 45.0, 88.0])
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        structure = ds.comb_structure(lines)
        assert "narrow_comb" not in set(structure["family"])
        assert int(structure.loc[structure["family"] == "narrow_off_comb", "n_lines"].iloc[0]) == 3

    def test_per_subject_fundamental_agrees_across_participants(self):
        planted = [1.2 * k for k in range(22, 40)]
        freqs, spectra = synthetic_cohort(planted, amplitude_db=25.0)
        grid = ds.build_grid(freqs, spectra)
        lines = ds.detect_cohort_lines(grid)
        subjects = [f"sub-{i:04d}" for i in range(15)]
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        estimates = ds.per_subject_fundamental(grid, subjects, classified, 1.2)
        assert len(estimates) == 15
        assert estimates["fundamental_hz"].std(ddof=1) < 1e-3


class TestClassifyLines:
    def test_comb_members_are_labelled_regardless_of_measured_width(self):
        planted = [1.2 * k for k in range(24, 60)]
        freqs, spectra = synthetic_cohort(planted)
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        assert (classified["kind"] == "comb").sum() >= 30
        assert classified.loc[classified["kind"] == "comb", "comb_harmonic_1p2"].min() >= 24

    def test_a_narrow_line_off_the_comb_is_isolated(self):
        planted = [1.2 * k for k in range(24, 60)] + [57.2247]
        freqs, spectra = synthetic_cohort(planted)
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        row = classified.loc[np.isclose(classified["refined_hz"], 57.2247, atol=0.05)].iloc[0]
        assert row["kind"] == "isolated"
        assert row["comb_harmonic_1p2"] == -1

    def test_enrichment_is_overwhelming_for_a_real_comb(self):
        planted = [1.2 * k for k in range(24, 60)]
        freqs, spectra = synthetic_cohort(planted)
        lines = ds.detect_cohort_lines(ds.build_grid(freqs, spectra))
        classified = ds.classify_lines(lines, ds.comb_structure(lines))
        result = ds.comb_enrichment(classified, 1.2)
        assert result["chance_rate"] == pytest.approx(0.1)
        assert result["binomial_p"] < 1e-20


class TestVarianceComponents:
    def test_separates_a_participant_effect_from_run_noise(self):
        rng = np.random.default_rng(3)
        rows = []
        for index in range(15):
            level = 5.0 * index
            for run in range(1, 7):
                rows.append(
                    {
                        "subject": f"sub-{index:04d}",
                        "run": run,
                        "frequency_hz": 57.1759,
                        "prominence_db": level + rng.normal(0, 0.5),
                    }
                )
        components = ds.variance_components(pd.DataFrame(rows))
        assert components["icc"].iloc[0] > 0.95

    def test_pure_run_noise_gives_a_low_icc(self):
        rng = np.random.default_rng(4)
        rows = [
            {
                "subject": f"sub-{index:04d}",
                "run": run,
                "frequency_hz": 57.1759,
                "prominence_db": rng.normal(0, 1.0),
            }
            for index in range(15)
            for run in range(1, 7)
        ]
        assert ds.variance_components(pd.DataFrame(rows))["icc"].iloc[0] < 0.3


class TestFrequencyStability:
    def test_zero_spread_when_every_participant_shares_a_line(self):
        freqs, spectra = synthetic_cohort([57.1759], amplitude_db=30.0)
        grid = ds.build_grid(freqs, spectra)
        lines = ds.detect_cohort_lines(grid)
        subjects = [f"sub-{i:04d}" for i in range(15)]
        dates = {s: f"2026-0{1 + i % 6}-15" for i, s in enumerate(subjects)}
        stability = ds.frequency_stability(subjects, lines, grid, dates)
        assert stability["sd_hz"].iloc[0] < 0.01
