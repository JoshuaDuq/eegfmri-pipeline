import argparse
import unittest

from eeg_pipeline.cli.commands.behavior import setup_behavior


class TestCliBehaviorHelp(unittest.TestCase):
    def test_behavior_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_behavior(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["behavior", "--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_behavior_compute_mode_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_behavior(subparsers)

        args = parser.parse_args(["behavior", "compute", "--subject", "0001"])
        self.assertEqual(args.command, "behavior")
        self.assertEqual(args.mode, "compute")


if __name__ == "__main__":
    unittest.main()
