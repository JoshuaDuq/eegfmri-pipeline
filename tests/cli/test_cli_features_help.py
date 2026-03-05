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

    def test_features_rejects_removed_source_fmri_window_flags(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_features(subparsers)

        with self.assertRaises(SystemExit) as exc:
            parser.parse_args(
                [
                    "features",
                    "compute",
                    "--subject",
                    "0001",
                    "--source-fmri-window-a-name",
                    "baseline",
                ]
            )
        self.assertEqual(exc.exception.code, 2)

    def test_features_accepts_source_fmri_output_space_flag(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_features(subparsers)

        args = parser.parse_args(
            [
                "features",
                "compute",
                "--subject",
                "0001",
                "--source-fmri-output-space",
                "atlas",
            ]
        )
        self.assertEqual(args.source_fmri_output_space, "atlas")

    def test_features_accepts_source_contrast_and_stc_flags(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_features(subparsers)

        args = parser.parse_args(
            [
                "features",
                "compute",
                "--subject",
                "0001",
                "--source-save-stc",
                "--source-contrast",
                "--source-contrast-condition-column",
                "stimulus_temp",
                "--source-contrast-condition-a",
                "44.3",
                "--source-contrast-condition-b",
                "49.3",
                "--source-contrast-welch-stats",
                "--source-fmri-condition-scope-column",
                "stim_phase",
            ]
        )
        self.assertTrue(bool(args.source_save_stc))
        self.assertTrue(bool(args.source_contrast_enabled))
        self.assertEqual(args.source_contrast_condition_column, "stimulus_temp")
        self.assertEqual(args.source_contrast_condition_a, "44.3")
        self.assertEqual(args.source_contrast_condition_b, "49.3")
        self.assertTrue(bool(args.source_contrast_welch_stats))
        self.assertEqual(args.source_fmri_condition_scope_column, "stim_phase")


if __name__ == "__main__":
    unittest.main()
