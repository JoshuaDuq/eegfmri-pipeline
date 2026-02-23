import argparse
import unittest

from eeg_pipeline.cli.commands.behavior_config import _configure_behavior_compute_mode
from eeg_pipeline.cli.commands.behavior_parser import setup_behavior
from eeg_pipeline.utils.config.loader import ConfigDict


class TestBehaviorCliTemporalOptions(unittest.TestCase):
    def _parse_behavior_compute_args(self, extra_args):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_behavior(subparsers)
        argv = ["behavior", "compute", *extra_args]
        return parser.parse_args(argv)

    def test_temporal_flags_map_to_temporal_config(self):
        args = self._parse_behavior_compute_args(
            [
                "--no-temporal-include-tf-grid",
                "--temporal-time-resolution-ms",
                "80",
                "--temporal-freqs-hz",
                "4",
                "8",
                "13",
            ]
        )
        config = ConfigDict({"project": {"task": "task"}})

        _configure_behavior_compute_mode(args, config)

        self.assertFalse(config.get("behavior_analysis.temporal.include_tf_grid", True))
        self.assertEqual(config.get("behavior_analysis.temporal.time_resolution_ms"), 80)
        self.assertEqual(config.get("behavior_analysis.temporal.freqs_hz"), [4.0, 8.0, 13.0])


if __name__ == "__main__":
    unittest.main()
