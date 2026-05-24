from __future__ import annotations

import logging
import unittest

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.features.complexity import extract_complexity_from_precomputed
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.types import BandData, PrecomputedData, TimeWindows
from tests.pipelines_test_utils import DotConfig


class TestComplexityEntropyFeatures(unittest.TestCase):
    def _build_precomputed(self) -> PrecomputedData:
        rng = np.random.default_rng(13)
        sfreq = 250.0
        times = np.arange(-0.4, 1.2, 1.0 / sfreq)
        n_epochs = 8
        n_channels = 4
        n_times = len(times)

        filtered = np.zeros((n_epochs, n_channels, n_times), dtype=float)
        for ep in range(n_epochs):
            for ch in range(n_channels):
                sig = (
                    np.sin(2 * np.pi * 10.0 * times + 0.2 * ch)
                    + 0.5 * np.sin(2 * np.pi * 22.0 * times + 0.1 * ep)
                    + 0.2 * rng.standard_normal(n_times)
                )
                filtered[ep, ch, :] = sig

        envelope = np.abs(filtered)
        analytic = filtered.astype(np.complex128) + 0.0j
        power = envelope**2

        band_data = {
            "alpha": BandData(
                band="alpha",
                fmin=8.0,
                fmax=12.9,
                filtered=filtered,
                analytic=analytic,
                envelope=envelope,
                phase=np.angle(analytic),
                power=power,
            )
        }

        baseline_mask = (times >= -0.3) & (times < 0.0)
        active_mask = (times >= 0.0) & (times <= 1.0)
        windows = TimeWindows(
            baseline_mask=baseline_mask,
            active_mask=active_mask,
            masks={"baseline": baseline_mask, "active": active_mask},
            ranges={"baseline": (-0.3, 0.0), "active": (0.0, 1.0)},
            times=times,
        )

        config = DotConfig(
            {
                "feature_engineering": {
                    "complexity": {
                        "signal_basis": "filtered",
                        "pe_order": 3,
                        "pe_delay": 1,
                        "sampen_order": 2,
                        "sampen_r": 0.2,
                        "mse_scale_min": 1,
                        "mse_scale_max": 4,
                        "zscore": True,
                        "min_segment_sec": 0.5,
                        "min_samples": 80,
                    },
                    "parallel": {"n_jobs_complexity": 1},
                }
            }
        )

        return PrecomputedData(
            data=filtered,
            times=times,
            sfreq=sfreq,
            ch_names=["C3", "C4", "CP3", "CP4"],
            picks=np.arange(n_channels),
            windows=windows,
            band_data=band_data,
            config=config,
            logger=logging.getLogger("test-complexity-entropy"),
            spatial_modes=["global"],
            metadata=pd.DataFrame({"trial_type": ["a"] * n_epochs}),
        )

    def test_extracts_sample_entropy_and_mse_scales(self):
        precomputed = self._build_precomputed()
        df, cols = extract_complexity_from_precomputed(precomputed, n_jobs=1)

        expected_cols = [
            NamingSchema.build("comp", "active", "alpha", "global", "sampen"),
            NamingSchema.build("comp", "active", "alpha", "global", "mse01"),
            NamingSchema.build("comp", "active", "alpha", "global", "mse04"),
        ]
        for col in expected_cols:
            self.assertIn(col, cols)
            self.assertIn(col, df.columns)
            self.assertTrue(np.isfinite(np.asarray(df[col], dtype=float)).any(), msg=col)

    def test_extracts_requested_baseline_segment(self):
        precomputed = self._build_precomputed()
        precomputed.windows.name = "baseline"
        precomputed.config["feature_engineering"]["complexity"]["min_segment_sec"] = 0.2
        precomputed.config["feature_engineering"]["complexity"]["min_samples"] = 50
        precomputed.config["feature_engineering"]["complexity"]["mse_scale_max"] = 1

        df, cols = extract_complexity_from_precomputed(precomputed, n_jobs=1)

        expected_col = NamingSchema.build("comp", "baseline", "alpha", "global", "sampen")
        self.assertIn(expected_col, cols)
        self.assertIn(expected_col, df.columns)
        self.assertTrue(np.isfinite(np.asarray(df[expected_col], dtype=float)).any())

    def test_resting_state_rejects_empty_target_window(self):
        precomputed = self._build_precomputed()
        times = precomputed.times
        analysis_mask = (times >= 0.0) & (times <= 1.0)
        empty_mask = np.zeros(times.shape, dtype=bool)
        precomputed.windows = TimeWindows(
            masks={"analysis": analysis_mask, "active": empty_mask},
            ranges={"analysis": (0.0, 1.0), "active": (0.0, 1.0)},
            times=times,
            name="active",
        )
        precomputed.config = DotConfig(
            {
                "feature_engineering": {
                    "task_is_rest": True,
                    "complexity": {
                        "signal_basis": "filtered",
                        "pe_order": 3,
                        "pe_delay": 1,
                        "sampen_order": 2,
                        "sampen_r": 0.2,
                        "mse_scale_min": 1,
                        "mse_scale_max": 4,
                        "zscore": True,
                        "min_segment_sec": 0.5,
                        "min_samples": 80,
                    },
                    "parallel": {"n_jobs_complexity": 1},
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError, "target window 'active' does not contain valid samples"
        ):
            extract_complexity_from_precomputed(precomputed, n_jobs=1)

    def test_complexity_global_features_reject_partial_channel_support(self):
        precomputed = self._build_precomputed()
        active_mask = precomputed.windows.masks["active"]
        precomputed.band_data["alpha"].filtered[0, 0, active_mask] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite complexity values"):
            extract_complexity_from_precomputed(precomputed, n_jobs=1)

    def test_invalid_complexity_config_raises(self):
        precomputed = self._build_precomputed()
        precomputed.config = DotConfig(
            {
                "feature_engineering": {
                    "complexity": {
                        "signal_basis": "bad-basis",
                        "pe_order": 1,
                        "pe_delay": 0,
                        "sampen_order": 0,
                        "sampen_r": 0.0,
                        "mse_scale_min": 3,
                        "mse_scale_max": 2,
                        "zscore": True,
                        "min_segment_sec": 0.5,
                        "min_samples": 80,
                    },
                    "parallel": {"n_jobs_complexity": 1},
                }
            }
        )

        with self.assertRaises(ValueError):
            extract_complexity_from_precomputed(precomputed, n_jobs=1)


if __name__ == "__main__":
    unittest.main()
