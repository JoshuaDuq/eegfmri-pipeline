from __future__ import annotations

import unittest

import numpy as np

from eeg_pipeline.utils.analysis.signal_metrics import (
    compute_multiscale_entropy,
    compute_sample_entropy,
)


class TestSignalMetricsComplexityEntropy(unittest.TestCase):
    def test_sample_entropy_returns_finite_for_variable_signal(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(500)
        value = compute_sample_entropy(x, order=2, r=0.2)
        self.assertTrue(np.isfinite(value))

    def test_multiscale_entropy_returns_requested_scale_keys(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(1000)
        scales = [1, 2, 3, 4, 5]
        mse = compute_multiscale_entropy(x, scales=scales, order=2, r=0.2)
        self.assertEqual(sorted(mse.keys()), scales)
        self.assertTrue(np.isfinite(mse[1]))

    def test_multiscale_entropy_marks_insufficient_scales_as_nan(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 60))
        mse = compute_multiscale_entropy(x, scales=[1, 10, 20], order=2, r=0.2)
        self.assertTrue(np.isfinite(mse[1]))
        self.assertTrue(np.isnan(mse[20]))


if __name__ == "__main__":
    unittest.main()
