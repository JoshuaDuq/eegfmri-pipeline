import argparse
import unittest

from eeg_pipeline.cli.commands.utilities import setup_utilities


class TestCliUtilitiesHelp(unittest.TestCase):
    def test_utilities_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_utilities(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["utilities", "--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_utilities_mode_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_utilities(subparsers)

        args = parser.parse_args(["utilities", "clean", "--subject", "0001"])
        self.assertEqual(args.command, "utilities")
        self.assertEqual(args.mode, "clean")


if __name__ == "__main__":
    unittest.main()
