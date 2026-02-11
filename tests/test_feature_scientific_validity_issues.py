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
from eeg_pipeline.analysis.features.spectral import _resolve_line_noise_freqs
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


if __name__ == "__main__":
    unittest.main()
