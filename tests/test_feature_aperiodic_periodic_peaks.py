from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

import numpy as np

from eeg_pipeline.analysis.features.aperiodic import extract_aperiodic_from_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.types import PrecomputedData, TimeWindows
from tests.pipelines_test_utils import DotConfig


class TestAperiodicPeriodicPeaks(unittest.TestCase):
    def _build_precomputed(self) -> PrecomputedData:
        rng = np.random.default_rng(7)
        sfreq = 200.0
        times = np.arange(0.0, 2.5, 1.0 / sfreq)
        n_epochs = 10
        n_channels = 4

        data = np.zeros((n_epochs, n_channels, times.size), dtype=float)
        for epoch_idx in range(n_epochs):
            phase_jitter = rng.uniform(0.0, 2.0 * np.pi)
            for ch_idx in range(n_channels):
                alpha_amp = 6.0 + 0.5 * ch_idx
                beta_amp = 3.5 + 0.3 * ch_idx
                trend = 0.2 * np.sin(2.0 * np.pi * 1.0 * times + 0.25 * ch_idx)
                alpha = alpha_amp * np.sin(2.0 * np.pi * 10.0 * times + phase_jitter)
                beta = beta_amp * np.sin(2.0 * np.pi * 20.0 * times + 0.4 * phase_jitter)
                noise = 0.5 * rng.standard_normal(times.size)
                data[epoch_idx, ch_idx, :] = trend + alpha + beta + noise

        mask = np.ones(times.size, dtype=bool)
        windows = TimeWindows(
            baseline_mask=~mask,
            active_mask=mask,
            masks={"active": mask},
            ranges={"active": (float(times[0]), float(times[-1] + (1.0 / sfreq)))},
            times=times,
        )

        config = DotConfig(
            {
                "feature_engineering": {
                    "constants": {
                        "min_epochs_for_features": 2,
                    },
                    "aperiodic": {
                        "model": "fixed",
                        "fmin": 2.0,
                        "fmax": 40.0,
                        "psd_method": "multitaper",
                        "peak_rejection_z": 2.5,
                        "min_fit_points": 8,
                        "min_r2": 0.0,
                        "min_segment_sec": 1.0,
                    },
                    "parallel": {
                        "n_jobs_aperiodic": 1,
                    },
                },
                "feature_categories": {
                    "spatial_modes": ["global"],
                },
                "frequency_bands": {
                    "alpha": [8.0, 13.0],
                    "beta": [13.0, 30.0],
                    "theta": [4.0, 8.0],
                },
            }
        )

        return PrecomputedData(
            data=data,
            times=times,
            sfreq=sfreq,
            ch_names=["F3", "F4", "C3", "C4"],
            picks=np.arange(n_channels),
            windows=windows,
            config=config,
            logger=logging.getLogger("test-aperiodic-periodic-peaks"),
            spatial_modes=["global"],
        )

    def test_naming_schema_parses_peak_height_stat(self):
        name = NamingSchema.build(
            "aperiodic",
            "active",
            "alpha",
            "ch",
            "peak_height",
            channel="C3",
        )
        parsed = NamingSchema.parse(name)
        self.assertTrue(parsed.get("valid"))
        self.assertEqual(parsed.get("stat"), "peak_height")

    def test_extracts_periodic_peak_metrics_per_band(self):
        precomputed = self._build_precomputed()

        df, cols, qc = extract_aperiodic_from_precomputed(precomputed, ["alpha", "beta"])

        expected = [
            NamingSchema.build("aperiodic", "active", "alpha", "global", "center_freq"),
            NamingSchema.build("aperiodic", "active", "alpha", "global", "bandwidth"),
            NamingSchema.build("aperiodic", "active", "alpha", "global", "peak_height"),
            NamingSchema.build("aperiodic", "active", "beta", "global", "center_freq"),
            NamingSchema.build("aperiodic", "active", "beta", "global", "bandwidth"),
            NamingSchema.build("aperiodic", "active", "beta", "global", "peak_height"),
        ]

        for col in expected:
            self.assertIn(col, cols)
            self.assertIn(col, df.columns)
            self.assertTrue(np.isfinite(pd_values := np.asarray(df[col], dtype=float)).any(), msg=col)

        alpha_cf = np.nanmean(np.asarray(df[expected[0]], dtype=float))
        alpha_bw = np.nanmean(np.asarray(df[expected[1]], dtype=float))
        alpha_h = np.nanmean(np.asarray(df[expected[2]], dtype=float))
        beta_cf = np.nanmean(np.asarray(df[expected[3]], dtype=float))
        beta_bw = np.nanmean(np.asarray(df[expected[4]], dtype=float))
        beta_h = np.nanmean(np.asarray(df[expected[5]], dtype=float))

        self.assertGreaterEqual(alpha_cf, 8.0)
        self.assertLessEqual(alpha_cf, 13.0)
        self.assertGreater(alpha_bw, 0.0)
        self.assertGreater(alpha_h, 0.0)

        self.assertGreaterEqual(beta_cf, 13.0)
        self.assertLessEqual(beta_cf, 30.0)
        self.assertGreater(beta_bw, 0.0)
        self.assertGreater(beta_h, 0.0)

        self.assertIn("segments", qc)
        self.assertIn("active", qc["segments"])

    def test_trial_ml_safe_requires_train_mask_for_subtract_evoked(self):
        precomputed = self._build_precomputed()
        precomputed.config["feature_engineering"]["aperiodic"]["subtract_evoked"] = True
        precomputed.config["feature_engineering"]["analysis_mode"] = "trial_ml_safe"
        precomputed.train_mask = None

        with self.assertRaisesRegex(ValueError, "trial_ml_safe mode without train_mask"):
            extract_aperiodic_from_precomputed(precomputed, ["alpha"])

    def test_subtract_evoked_uses_train_mask_in_precomputed_mode(self):
        precomputed = self._build_precomputed()
        precomputed.config["feature_engineering"]["aperiodic"]["subtract_evoked"] = True
        precomputed.config["feature_engineering"]["analysis_mode"] = "trial_ml_safe"
        precomputed.train_mask = np.array([True, True, True, True, True, False, False, False, False, False], dtype=bool)

        seen_masks = []

        def _fake_subtract_evoked(data, condition_labels=None, train_mask=None, min_trials_per_condition=2):
            del condition_labels, min_trials_per_condition
            seen_masks.append(None if train_mask is None else np.asarray(train_mask, dtype=bool).copy())
            return data

        with patch("eeg_pipeline.utils.analysis.spectral.subtract_evoked", new=_fake_subtract_evoked):
            df, cols, _qc = extract_aperiodic_from_precomputed(precomputed, ["alpha"])

        self.assertFalse(df.empty)
        self.assertTrue(cols)
        self.assertTrue(seen_masks)
        np.testing.assert_array_equal(seen_masks[0], precomputed.train_mask)


if __name__ == "__main__":
    unittest.main()
