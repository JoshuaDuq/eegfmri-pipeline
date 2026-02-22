import argparse
import unittest

from eeg_pipeline.cli.commands.machine_learning import setup_ml


class TestCliMachineLearningHelp(unittest.TestCase):
    def test_ml_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_ml(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["ml", "--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_ml_default_mode_is_regression(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_ml(subparsers)

        args = parser.parse_args(["ml"])
        self.assertEqual(args.command, "ml")
        self.assertEqual(args.mode, "regression")


if __name__ == "__main__":
    unittest.main()
