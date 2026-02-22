import argparse
import unittest

from eeg_pipeline.cli.commands.validate import setup_validate


class TestCliValidateHelp(unittest.TestCase):
    def test_validate_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_validate(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["validate", "--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_validate_default_mode_is_quick(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_validate(subparsers)

        args = parser.parse_args(["validate"])
        self.assertEqual(args.command, "validate")
        self.assertEqual(args.mode, "quick")


if __name__ == "__main__":
    unittest.main()
