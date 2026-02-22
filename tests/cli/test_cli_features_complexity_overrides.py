from __future__ import annotations

import argparse
import unittest

from eeg_pipeline.cli.commands.features_helpers import _apply_complexity_overrides


class TestCliComplexityOverrides(unittest.TestCase):
    def test_applies_sample_entropy_and_mse_overrides(self):
        args = argparse.Namespace(
            pe_order=5,
            pe_delay=2,
            complexity_signal_basis="envelope",
            complexity_min_segment_sec=1.25,
            complexity_min_samples=160,
            complexity_zscore=False,
            complexity_sampen_order=3,
            complexity_sampen_r=0.25,
            complexity_mse_scale_min=2,
            complexity_mse_scale_max=12,
        )

        config = {"feature_engineering": {"complexity": {}}}
        _apply_complexity_overrides(args, config)

        comp_cfg = config["feature_engineering"]["complexity"]
        self.assertEqual(comp_cfg["pe_order"], 5)
        self.assertEqual(comp_cfg["pe_delay"], 2)
        self.assertEqual(comp_cfg["signal_basis"], "envelope")
        self.assertEqual(comp_cfg["min_segment_sec"], 1.25)
        self.assertEqual(comp_cfg["min_samples"], 160)
        self.assertEqual(comp_cfg["zscore"], False)
        self.assertEqual(comp_cfg["sampen_order"], 3)
        self.assertEqual(comp_cfg["sampen_r"], 0.25)
        self.assertEqual(comp_cfg["mse_scale_min"], 2)
        self.assertEqual(comp_cfg["mse_scale_max"], 12)


if __name__ == "__main__":
    unittest.main()
