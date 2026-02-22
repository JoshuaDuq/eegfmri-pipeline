from __future__ import annotations

import unittest

import numpy as np

from eeg_pipeline.analysis.features.aperiodic import _parse_line_noise_config
from eeg_pipeline.analysis.features.erp import (
    _compute_auc,
    _compute_peak_pair_metrics,
)
from eeg_pipeline.analysis.features.precomputed.extras import _get_psd_config
from eeg_pipeline.analysis.features.quality import _extract_quality_config
from eeg_pipeline.analysis.features.quality import (
    _compute_muscle_ratio_from_psd,
    _compute_snr_from_psd,
)
from eeg_pipeline.analysis.features.spectral import (
    _resolve_line_noise_freqs,
    _robust_aperiodic_fit,
    compute_peak_frequency,
)
from tests.pipelines_test_utils import DotConfig


class TestScientificValidityIssues(unittest.TestCase):
    def test_erp_peak_to_peak_uses_absolute_magnitude(self):
        neg_vals = np.array([[1.2, 0.4]], dtype=float)
        pos_vals = np.array([[0.5, 0.1]], dtype=float)
        neg_times = np.array([[0.15, 0.16]], dtype=float)
        pos_times = np.array([[0.27, 0.29]], dtype=float)

        ptp, lat_diff = _compute_peak_pair_metrics(
            neg_vals=neg_vals,
            pos_vals=pos_vals,
            neg_times=neg_times,
            pos_times=pos_times,
        )

        np.testing.assert_allclose(ptp, np.array([[0.7, 0.3]], dtype=float), atol=1e-12)
        np.testing.assert_allclose(lat_diff, np.array([[0.12, 0.13]], dtype=float), atol=1e-12)
        self.assertTrue(np.all(ptp >= 0))

    def test_erp_auc_does_not_zero_impute_nan_gaps(self):
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        data = np.array([[[1.0, 1.0, np.nan, 1.0, 1.0]]], dtype=float)

        auc = _compute_auc(data, times)
        self.assertEqual(auc.shape, (1, 1))
        # Two finite contiguous segments: [0..1] and [3..4], each area=1.
        self.assertAlmostEqual(float(auc[0, 0]), 2.0, places=7)

    def test_quality_line_noise_defaults_to_preprocessing_line_freq(self):
        cfg = DotConfig(
            {
                "preprocessing": {"line_freq": 60.0},
                "feature_engineering": {"quality": {"exclude_line_noise": True}},
            }
        )
        quality_cfg = _extract_quality_config(cfg)
        self.assertEqual(quality_cfg.get("line_noise_freqs"), [60.0])

    def test_aperiodic_line_noise_defaults_to_preprocessing_line_freq(self):
        cfg = DotConfig(
            {
                "preprocessing": {"line_freq": 60.0},
                "feature_engineering": {"aperiodic": {"exclude_line_noise": True}},
            }
        )
        line_cfg = _parse_line_noise_config(cfg)
        self.assertEqual(line_cfg.frequencies, [60.0])

    def test_extras_psd_line_noise_defaults_to_preprocessing_line_freq(self):
        cfg = DotConfig(
            {
                "preprocessing": {"line_freq": 60.0},
                "feature_engineering": {"spectral": {"exclude_line_noise": True}},
            }
        )
        psd_cfg = _get_psd_config(cfg, sfreq=500.0)
        self.assertEqual(psd_cfg["line_freqs"], [60.0])

    def test_spectral_line_noise_defaults_to_preprocessing_line_freq(self):
        cfg = DotConfig({"preprocessing": {"line_freq": 60.0}})
        spec_cfg = {}
        resolved = _resolve_line_noise_freqs(spec_cfg, cfg)
        self.assertEqual(resolved, [60.0])

    def test_peak_metrics_align_with_reported_cog_frequency(self):
        freqs = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 20.0, 30.0, 40.0], dtype=float)
        psd = np.array([20.0, 10.0, 6.0, 1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 1.0], dtype=float)

        peak_freq, peak_power, peak_ratio, peak_residual = compute_peak_frequency(
            psd,
            freqs,
            fmin=8.0,
            fmax=12.0,
            aperiodic_adjusted=False,
            smoothing_hz=0.0,
            min_prominence=1e6,  # Force center-of-gravity fallback.
        )

        self.assertGreater(peak_freq, 10.0)
        self.assertLess(peak_freq, 12.0)
        self.assertTrue(np.isfinite(peak_ratio))
        self.assertTrue(np.isfinite(peak_residual))

        mask = (freqs >= 8.0) & (freqs <= 12.0)
        freqs_band = freqs[mask]
        closest_idx = int(np.argmin(np.abs(freqs_band - peak_freq)))
        global_peak_idx = int(np.where(mask)[0][closest_idx])

        log_f = np.log10(np.maximum(freqs, 1e-6))
        log_p = np.log10(np.maximum(psd, 1e-20))
        fit_mask = (freqs >= 2.0) & (freqs <= 40.0) & np.isfinite(log_p)
        slope, intercept = _robust_aperiodic_fit(log_f, log_p, fit_mask)
        if slope is None or intercept is None:
            slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
        aperiodic_fit = 10 ** (intercept + slope * log_f)

        expected_ratio = float(peak_power / aperiodic_fit[global_peak_idx])
        expected_residual = float(np.log10(peak_power) - np.log10(aperiodic_fit[global_peak_idx]))
        self.assertAlmostEqual(peak_ratio, expected_ratio, places=10)
        self.assertAlmostEqual(peak_residual, expected_residual, places=10)

    def test_quality_snr_uses_bandwidth_weighted_power_density(self):
        # Flat spectrum should yield ~0 dB SNR regardless of uneven bin counts.
        freqs = np.array([1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=float)
        psds = np.ones((1, freqs.size), dtype=float)
        cfg = {
            "snr_signal_band": [1.0, 30.0],
            "snr_noise_band": [40.0, 80.0],
        }
        snr_db = _compute_snr_from_psd(psds, freqs, cfg)
        self.assertEqual(snr_db.shape, (1,))
        self.assertAlmostEqual(float(snr_db[0]), 0.0, places=10)

    def test_quality_muscle_ratio_uses_frequency_width_weighting(self):
        freqs = np.array([1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=float)
        # Non-flat PSD so weighted integration matters.
        psds = np.array([[1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0]], dtype=float)
        cfg = {"muscle_band": [30.0, 80.0]}
        ratio = _compute_muscle_ratio_from_psd(psds, freqs, cfg)

        df = np.gradient(freqs)
        muscle_mask = (freqs >= 30.0) & (freqs <= 80.0)
        muscle_power = float(np.sum(psds[0, muscle_mask] * df[muscle_mask]))
        total_power = float(np.sum(psds[0] * df))
        expected = muscle_power / total_power

        self.assertEqual(ratio.shape, (1,))
        self.assertAlmostEqual(float(ratio[0]), expected, places=10)


if __name__ == "__main__":
    unittest.main()
