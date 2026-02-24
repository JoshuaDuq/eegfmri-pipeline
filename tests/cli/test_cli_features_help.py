import argparse
import unittest

from eeg_pipeline.cli.commands.features import setup_features


class TestCliFeaturesHelp(unittest.TestCase):
    def test_features_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_features(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["features", "--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_features_compute_mode_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_features(subparsers)

        args = parser.parse_args(["features", "compute", "--subject", "0001"])
        self.assertEqual(args.command, "features")
        self.assertEqual(args.mode, "compute")

    def test_features_connectivity_measures_accept_imcoh(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_features(subparsers)

        args = parser.parse_args(
            [
                "features",
                "compute",
                "--subject",
                "0001",
                "--connectivity-measures",
                "imcoh",
                "wpli",
            ]
        )
        self.assertEqual(args.connectivity_measures, ["imcoh", "wpli"])


if __name__ == "__main__":
    unittest.main()
