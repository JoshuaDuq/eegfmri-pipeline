from __future__ import annotations

import argparse
import unittest

from eeg_pipeline.cli.commands.features_helpers import _apply_connectivity_overrides


class TestCliConnectivityDynamicOverrides(unittest.TestCase):
    def test_applies_dynamic_connectivity_overrides(self):
        args = argparse.Namespace(
            conn_dynamic_enabled=True,
            conn_dynamic_measures=["wpli", "aec"],
            conn_dynamic_autocorr_lag=2,
            conn_dynamic_min_windows=5,
            conn_dynamic_include_roi_pairs=False,
            conn_dynamic_state_enabled=True,
            conn_dynamic_state_n_states=4,
            conn_dynamic_state_min_windows=10,
            conn_dynamic_state_random_state=17,
        )

        config = {"feature_engineering": {"connectivity": {}}}
        _apply_connectivity_overrides(args, config)

        conn_cfg = config["feature_engineering"]["connectivity"]
        self.assertEqual(conn_cfg["dynamic_enabled"], True)
        self.assertEqual(conn_cfg["dynamic_measures"], ["wpli", "aec"])
        self.assertEqual(conn_cfg["dynamic_autocorr_lag"], 2)
        self.assertEqual(conn_cfg["dynamic_min_windows"], 5)
        self.assertEqual(conn_cfg["dynamic_include_roi_pairs"], False)
        self.assertEqual(conn_cfg["dynamic_state_enabled"], True)
        self.assertEqual(conn_cfg["dynamic_state_n_states"], 4)
        self.assertEqual(conn_cfg["dynamic_state_min_windows"], 10)
        self.assertEqual(conn_cfg["dynamic_state_random_state"], 17)


if __name__ == "__main__":
    unittest.main()
