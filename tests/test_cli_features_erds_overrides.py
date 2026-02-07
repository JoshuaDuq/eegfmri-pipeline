from __future__ import annotations

import argparse
import unittest

from eeg_pipeline.cli.commands.features import _apply_erds_overrides


class TestCliERDSPainOverrides(unittest.TestCase):
    def test_applies_pain_marker_override_values(self):
        args = argparse.Namespace(
            erds_use_log_ratio=True,
            erds_min_baseline_power=1e-10,
            erds_min_active_power=1e-11,
            erds_min_segment_sec=0.75,
            erds_bands=["alpha"],
            erds_onset_threshold_sigma=1.5,
            erds_onset_min_duration_ms=40.0,
            erds_rebound_min_latency_ms=150.0,
            erds_infer_contralateral=True,
        )

        config = {"feature_engineering": {"erds": {}}}
        _apply_erds_overrides(args, config)

        erds_cfg = config["feature_engineering"]["erds"]
        self.assertEqual(erds_cfg["use_log_ratio"], True)
        self.assertEqual(erds_cfg["min_baseline_power"], 1e-10)
        self.assertEqual(erds_cfg["min_active_power"], 1e-11)
        self.assertEqual(erds_cfg["min_segment_sec"], 0.75)
        self.assertEqual(erds_cfg["bands"], ["alpha"])
        self.assertEqual(erds_cfg["onset_threshold_sigma"], 1.5)
        self.assertEqual(erds_cfg["onset_min_duration_ms"], 40.0)
        self.assertEqual(erds_cfg["rebound_min_latency_ms"], 150.0)
        self.assertEqual(erds_cfg["infer_contralateral_when_missing"], True)


if __name__ == "__main__":
    unittest.main()
