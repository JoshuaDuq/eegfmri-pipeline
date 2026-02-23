import argparse
import unittest

from eeg_pipeline.cli.commands.info import setup_info


class TestCliInfoHelp(unittest.TestCase):
    def test_info_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_info(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["info", "--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_info_subjects_mode_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_info(subparsers)

        args = parser.parse_args(["info", "subjects"])
        self.assertEqual(args.command, "info")
        self.assertEqual(args.mode, "subjects")
        self.assertEqual(args.source, "all")


if __name__ == "__main__":
    unittest.main()
