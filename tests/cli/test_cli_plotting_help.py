import argparse
import unittest

from eeg_pipeline.cli.commands.plotting import setup_plotting


class TestCliPlottingHelp(unittest.TestCase):
    def test_plotting_visualize_help_renders(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_plotting(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(["plotting", "visualize", "--help"])

        self.assertEqual(exc.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
