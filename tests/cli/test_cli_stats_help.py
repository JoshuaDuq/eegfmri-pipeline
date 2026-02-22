import argparse
import unittest

from eeg_pipeline.cli.commands.stats import setup_stats


class TestCliStatsHelp(unittest.TestCase):
    def test_stats_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_stats(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["stats", "--help"])

        self.assertEqual(exc.exception.code, 0)

    def test_stats_default_mode_is_summary(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_stats(subparsers)

        args = parser.parse_args(["stats"])
        self.assertEqual(args.command, "stats")
        self.assertEqual(args.mode, "summary")


if __name__ == "__main__":
    unittest.main()
