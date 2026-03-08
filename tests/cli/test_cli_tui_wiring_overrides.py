from __future__ import annotations

import argparse
import pathlib
import re
import unittest

from eeg_pipeline.cli.commands.behavior_config import _configure_behavior_compute_mode
from eeg_pipeline.cli.commands.behavior_parser import setup_behavior
from eeg_pipeline.cli.commands.features_helpers import (
    _apply_erds_overrides,
    _apply_itpc_overrides,
    _apply_microstates_overrides,
    _apply_pac_overrides,
    _apply_spatial_transform_overrides,
)
from eeg_pipeline.cli.commands.machine_learning_overrides import _update_model_config
from eeg_pipeline.cli.commands.machine_learning_parser import setup_ml
from eeg_pipeline.cli.commands.preprocessing_overrides import (
    _update_alignment_event_config,
    _update_epochs_config,
    _update_pyprep_config,
    _update_preprocessing_config,
)
from eeg_pipeline.cli.commands.preprocessing_parser import setup_preprocessing
from eeg_pipeline.utils.config.loader import ConfigDict


class TestPreprocessingTUIWiring(unittest.TestCase):
    def test_unwired_preprocessing_flags_update_config(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_preprocessing(subparsers)
        args = parser.parse_args(
            [
                "preprocessing",
                "full",
                "--ecg-channels",
                "ECG1,ECG2",
                "--autoreject-n-interpolate",
                "4",
                "8",
                "16",
                "--allow-misaligned-trim",
                "--min-alignment-samples",
                "7",
                "--trim-to-first-volume",
                "--fmri-onset-reference",
                "scanner_trigger",
                "--event-col-predictor",
                "temperature",
                "stim_temp",
                "--event-col-outcome",
                "rating",
                "--event-col-binary-outcome",
                "binary_outcome",
                "--condition-preferred-prefixes",
                "Trig_",
                "Stim_",
                "--rename-anot-dict",
                "{\"BAD boundary\":\"BAD_boundary\"}",
                "--custom-bad-dict",
                "{\"task\":{\"0001\":[\"TP8\"]}}",
            ]
        )
        config = ConfigDict({})

        _update_preprocessing_config(args, config)
        _update_pyprep_config(args, config)
        _update_epochs_config(args, config)
        _update_alignment_event_config(args, config)

        self.assertEqual(config.get("eeg.ecg_channels"), ["ECG1", "ECG2"])
        self.assertEqual(config.get("epochs.autoreject_n_interpolate"), [4, 8, 16])
        self.assertTrue(config.get("alignment.allow_misaligned_trim", False))
        self.assertEqual(config.get("alignment.min_alignment_samples"), 7)
        self.assertTrue(config.get("alignment.trim_to_first_volume", False))
        self.assertEqual(config.get("alignment.fmri_onset_reference"), "first_stim_start")
        self.assertEqual(config.get("event_columns.predictor"), ["temperature", "stim_temp"])
        self.assertEqual(config.get("event_columns.outcome"), ["rating"])
        self.assertEqual(config.get("event_columns.binary_outcome"), ["binary_outcome"])
        self.assertEqual(config.get("preprocessing.condition_preferred_prefixes"), ["Trig_", "Stim_"])
        self.assertEqual(config["pyprep"]["rename_anot_dict"]["BAD boundary"], "BAD_boundary")
        self.assertEqual(config["pyprep"]["custom_bad_dict"]["task"]["0001"], ["TP8"])


class TestFeaturesTUIWiring(unittest.TestCase):
    def test_unwired_feature_overrides_are_applied(self):
        args = argparse.Namespace(
            pac_phase_range=None,
            pac_amp_range=None,
            pac_method=None,
            pac_min_epochs=None,
            pac_pairs=None,
            pac_source=None,
            pac_normalize=None,
            pac_n_surrogates=None,
            pac_allow_harmonic_overlap=None,
            pac_max_harmonic=None,
            pac_harmonic_tolerance_hz=None,
            pac_compute_waveform_qc=None,
            pac_waveform_offset_ms=None,
            pac_random_seed=None,
            pac_min_segment_sec=1.5,
            pac_min_cycles_at_fmin=4.0,
            pac_surrogate_method="trial_shuffle",
            itpc_method=None,
            itpc_allow_unsafe_loo=None,
            itpc_baseline_correction=None,
            itpc_condition_column=None,
            itpc_condition_values=None,
            itpc_min_trials_per_condition=None,
            itpc_n_jobs=None,
            itpc_min_segment_sec=1.25,
            itpc_min_cycles_at_fmin=3.5,
            microstates_n_states=None,
            microstates_min_peak_distance_ms=None,
            microstates_max_gfp_peaks_per_epoch=None,
            microstates_min_duration_ms=None,
            microstates_gfp_peak_prominence=None,
            microstates_random_state=None,
            microstates_assign_from_gfp_peaks=False,
            erds_use_log_ratio=None,
            erds_min_baseline_power=None,
            erds_min_active_power=None,
            erds_min_segment_sec=None,
            erds_bands=None,
            erds_onset_threshold_sigma=None,
            erds_onset_min_duration_ms=None,
            erds_rebound_min_latency_ms=None,
            erds_infer_contralateral=None,
            erds_condition_marker_bands=["alpha"],
            erds_laterality_columns=["stim_side"],
            erds_somatosensory_left_channels=["C3"],
            erds_somatosensory_right_channels=["C4"],
            erds_onset_min_threshold_percent=12.5,
            erds_rebound_threshold_sigma=1.2,
            erds_rebound_min_threshold_percent=15.0,
            spatial_transform=None,
            spatial_transform_lambda2=None,
            spatial_transform_stiffness=None,
            spatial_transform_connectivity="csd",
            spatial_transform_itpc="laplacian",
            spatial_transform_pac=None,
            spatial_transform_power=None,
            spatial_transform_aperiodic=None,
            spatial_transform_bursts=None,
            spatial_transform_erds=None,
            spatial_transform_complexity=None,
            spatial_transform_ratios=None,
            spatial_transform_asymmetry=None,
            spatial_transform_spectral=None,
            spatial_transform_erp=None,
            spatial_transform_quality=None,
            spatial_transform_microstates="none",
        )
        config = ConfigDict({"feature_engineering": {}})

        _apply_pac_overrides(args, config)
        _apply_itpc_overrides(args, config)
        _apply_microstates_overrides(args, config)
        _apply_erds_overrides(args, config)
        _apply_spatial_transform_overrides(args, config)

        self.assertEqual(config.get("feature_engineering.pac.min_segment_sec"), 1.5)
        self.assertEqual(config.get("feature_engineering.pac.min_cycles_at_fmin"), 4.0)
        self.assertEqual(config.get("feature_engineering.pac.surrogate_method"), "trial_shuffle")
        self.assertEqual(config.get("feature_engineering.itpc.min_segment_sec"), 1.25)
        self.assertEqual(config.get("feature_engineering.itpc.min_cycles_at_fmin"), 3.5)
        self.assertFalse(config.get("feature_engineering.microstates.assign_from_gfp_peaks", True))
        self.assertEqual(config.get("feature_engineering.erds.laterality_marker_bands"), ["alpha"])
        self.assertEqual(config.get("feature_engineering.erds.laterality_columns"), ["stim_side"])
        self.assertEqual(config.get("feature_engineering.erds.onset_min_threshold_percent"), 12.5)
        self.assertEqual(config.get("feature_engineering.spatial_transform_per_family.connectivity"), "csd")
        self.assertEqual(config.get("feature_engineering.spatial_transform_per_family.itpc"), "laplacian")
        self.assertEqual(config.get("feature_engineering.spatial_transform_per_family.microstates"), "none")


class TestBehaviorTUIWiring(unittest.TestCase):
    def test_unwired_behavior_flags_update_config(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_behavior(subparsers)
        args = parser.parse_args(
            [
                "behavior",
                "compute",
                "--predictor-column",
                "stim_temp",
                "--outcome-column",
                "vas_custom",
                "--perm-scheme",
                "circular_shift",
                "--stats-predictor-control",
                "spline",
                "--stats-allow-iid-trials",
                "--feature-registry-files-json",
                "{\"power\":\"features_power.parquet\"}",
                "--feature-registry-patterns-json",
                "{\"erds\":\"^erds_.*$\"}",
                "--feature-registry-classifiers-json",
                "[{\"label\":\"power\",\"startswith\":[\"power_\"]}]",
                "--group-level-target",
                "predictor_residual",
                "--group-level-control-predictor",
                "--no-group-level-control-trial-order",
                "--no-group-level-control-run-effects",
                "--group-level-max-run-dummies",
                "15",
                "--correlations-min-runs",
                "5",
                "--correlations-prefer-predictor-residual",
                "--correlations-permutations",
                "111",
                "--condition-primary-unit",
                "run_mean",
                "--condition-compare-labels",
                "low",
                "high",
                "--regression-primary-unit",
                "run_mean",
                "--temporal-correction-method",
                "cluster",
                "--icc-unit-columns",
                "predictor",
                "trial_type",
            ]
        )
        config = ConfigDict({"project": {"task": "task"}})
        _configure_behavior_compute_mode(args, config)

        self.assertEqual(config.get("behavior_analysis.predictor_column"), "stim_temp")
        self.assertEqual(config.get("behavior_analysis.outcome_column"), "vas_custom")
        self.assertEqual(config.get("behavior_analysis.permutation.scheme"), "circular_shift")
        self.assertEqual(config.get("behavior_analysis.statistics.predictor_control"), "spline")
        self.assertEqual(config.get("behavior_analysis.statistics.base_seed"), 42)
        self.assertTrue(config.get("behavior_analysis.statistics.allow_iid_trials", False))
        self.assertEqual(
            config.get("behavior_analysis.feature_registry.files"),
            {"power": "features_power.parquet"},
        )
        self.assertEqual(
            config.get("behavior_analysis.feature_registry.feature_patterns"),
            {"erds": "^erds_.*$"},
        )
        self.assertEqual(
            config.get("behavior_analysis.feature_registry.feature_classifiers"),
            [{"label": "power", "startswith": ["power_"]}],
        )
        self.assertEqual(
            config.get("behavior_analysis.group_level.multilevel_correlations.target"),
            "predictor_residual",
        )
        self.assertTrue(
            config.get("behavior_analysis.group_level.multilevel_correlations.control_predictor", False)
        )
        self.assertFalse(
            config.get("behavior_analysis.group_level.multilevel_correlations.control_trial_order", True)
        )
        self.assertFalse(
            config.get("behavior_analysis.group_level.multilevel_correlations.control_run_effects", True)
        )
        self.assertEqual(
            config.get("behavior_analysis.group_level.multilevel_correlations.max_run_dummies"),
            15,
        )
        self.assertEqual(config.get("behavior_analysis.correlations.min_runs"), 5)
        self.assertTrue(config.get("behavior_analysis.correlations.prefer_predictor_residual", False))
        self.assertEqual(config.get("behavior_analysis.correlations.permutation.n_permutations"), 111)
        self.assertEqual(config.get("behavior_analysis.condition.primary_unit"), "run_mean")
        self.assertEqual(config.get("behavior_analysis.condition.compare_labels"), ["low", "high"])
        self.assertEqual(config.get("behavior_analysis.regression.primary_unit"), "run_mean")
        self.assertEqual(config.get("behavior_analysis.temporal.correction_method"), "cluster")
        self.assertEqual(config.get("behavior_analysis.icc.unit_columns"), ["predictor", "trial_type"])

    def test_behavior_parser_accepts_explicit_none_and_loso_disable(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_behavior(subparsers)
        args = parser.parse_args(
            [
                "behavior",
                "compute",
                "--perm-scheme",
                "shuffle",
                "--stats-predictor-control",
                "none",
                "--no-loso-stability",
            ]
        )
        config = ConfigDict({"project": {"task": "task"}})
        _configure_behavior_compute_mode(args, config)

        self.assertEqual(config.get("behavior_analysis.permutation.scheme"), "shuffle")
        self.assertEqual(config.get("behavior_analysis.statistics.predictor_control"), "none")
        self.assertFalse(config.get("behavior_analysis.correlations.loso_stability", True))

    def test_behavior_parser_updates_temporal_topomap_window(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_behavior(subparsers)
        args = parser.parse_args(
            [
                "behavior",
                "compute",
                "--temporal-topomap-window-ms",
                "750",
            ]
        )
        config = ConfigDict({"project": {"task": "task"}})
        _configure_behavior_compute_mode(args, config)

        self.assertEqual(
            config.get("behavior_analysis.temporal_correlation_topomaps.window_size_ms"),
            750,
        )

    def test_correlations_target_column_sets_single_target_list(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_behavior(subparsers)
        args = parser.parse_args(
            [
                "behavior",
                "compute",
                "--correlations-target-column",
                "vas_custom",
            ]
        )
        config = ConfigDict({"project": {"task": "task"}})
        _configure_behavior_compute_mode(args, config)

        self.assertEqual(config.get("behavior_analysis.correlations.target_column"), "vas_custom")

    def test_behavior_cli_override_paths_have_tui_hydration(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        config_text = (root / "eeg_pipeline/cli/commands/behavior_config.py").read_text()
        hydration_text = (
            root / "eeg_pipeline/cli/tui/views/wizard/model_config_hydration.go"
        ).read_text()

        override_paths = {
            match[1]
            for match in re.findall(
                r'ConfigOverrideRule\("([^"]+)",\s*"([^"]+)"',
                config_text,
            )
        }
        hydration_paths = set(re.findall(r'\{key: "([^"]+)"', hydration_text))

        missing_paths = []
        for path in sorted(override_paths):
            if not path.startswith("behavior_analysis."):
                continue
            if path in hydration_paths:
                continue
            if any(
                hydration.startswith(path + ".") or path.startswith(hydration + ".")
                for hydration in hydration_paths
            ):
                continue
            missing_paths.append(path)

        self.assertEqual(missing_paths, [])


class TestMLTUIWiring(unittest.TestCase):
    def test_unwired_ml_flags_update_config(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_ml(subparsers)
        args = parser.parse_args(
            [
                "ml",
                "classify",
                "--imputer",
                "mean",
                "--power-transformer-method",
                "yeo-johnson",
                "--no-power-transformer-standardize",
                "--pca-enabled",
                "--pca-n-components",
                "0.9",
                "--pca-whiten",
                "--pca-svd-solver",
                "full",
                "--pca-rng-seed",
                "13",
                "--svm-kernel",
                "linear",
                "--svm-c-grid",
                "0.1",
                "1.0",
                "--svm-gamma-grid",
                "scale",
                "0.01",
                "--rf-min-samples-split-grid",
                "2",
                "5",
                "--rf-min-samples-leaf-grid",
                "1",
                "2",
                "--no-rf-bootstrap",
                "--cv-permutation-scheme",
                "within_subject",
                "--cv-min-valid-perm-fraction",
                "0.7",
                "--eval-ci-method",
                "fixed_effects",
                "--eval-bootstrap-iterations",
                "500",
                "--class-min-subjects-for-auc",
                "3",
                "--class-max-failed-fold-fraction",
                "0.2",
                "--no-strict-regression-continuous",
            ]
        )
        config = ConfigDict({})
        _update_model_config(args, config)

        self.assertEqual(config.get("machine_learning.preprocessing.imputer_strategy"), "mean")
        self.assertEqual(config.get("machine_learning.preprocessing.power_transformer_method"), "yeo-johnson")
        self.assertFalse(config.get("machine_learning.preprocessing.power_transformer_standardize", True))
        self.assertTrue(config.get("machine_learning.preprocessing.pca.enabled", False))
        self.assertEqual(config.get("machine_learning.preprocessing.pca.n_components"), 0.9)
        self.assertTrue(config.get("machine_learning.preprocessing.pca.whiten", False))
        self.assertEqual(config.get("machine_learning.preprocessing.pca.svd_solver"), "full")
        self.assertEqual(config.get("machine_learning.preprocessing.pca.random_state"), 13)
        self.assertEqual(config.get("machine_learning.models.svm.kernel"), "linear")
        self.assertEqual(config.get("machine_learning.models.random_forest.min_samples_split_grid"), [2, 5])
        self.assertFalse(config.get("machine_learning.models.random_forest.bootstrap", True))
        self.assertEqual(config.get("machine_learning.cv.permutation_scheme"), "within_subject")
        self.assertEqual(config.get("machine_learning.evaluation.ci_method"), "fixed_effects")
        self.assertEqual(config.get("machine_learning.classification.min_subjects_with_auc_for_inference"), 3)
        self.assertEqual(config.get("machine_learning.classification.max_failed_fold_fraction"), 0.2)
        self.assertFalse(config.get("machine_learning.targets.strict_regression_target_continuous", True))


if __name__ == "__main__":
    unittest.main()
