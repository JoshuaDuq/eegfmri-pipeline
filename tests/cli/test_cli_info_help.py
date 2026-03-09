import argparse
import json
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from eeg_pipeline.cli.commands.info import setup_info
from eeg_pipeline.cli.commands.info_helpers import _handle_fmri_conditions_mode


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

    def test_fmri_conditions_requires_explicit_or_configured_column(self):
        args = SimpleNamespace(
            task="pain",
            subject="0001",
            condition_column=None,
            output_json=True,
        )

        with patch(
            "eeg_pipeline.cli.commands.info_helpers._discover_fmri_events_columns_and_values",
            return_value={
                "columns": ["trial_label", "stimulus_family"],
                "values": {
                    "trial_label": ["pain", "rest"],
                    "stimulus_family": ["thermal"],
                },
                "subject": "0001",
                "task": "pain",
            },
        ), patch(
            "eeg_pipeline.utils.config.loader.get_condition_column_candidates",
            return_value=[],
        ), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            _handle_fmri_conditions_mode(args, config={})

        payload = json.loads(stdout.getvalue())
        self.assertIsNone(payload["condition_column"])
        self.assertIn("Could not resolve an fMRI condition column", payload["error"])


if __name__ == "__main__":
    unittest.main()
