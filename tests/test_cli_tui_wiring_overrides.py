from __future__ import annotations

import argparse
import unittest

from eeg_pipeline.cli.commands.behavior import _configure_behavior_compute_mode, setup_behavior
from eeg_pipeline.cli.commands.features import (
    _apply_erds_overrides,
    _apply_itpc_overrides,
    _apply_microstates_overrides,
    _apply_pac_overrides,
    _apply_spatial_transform_overrides,
)
from eeg_pipeline.cli.commands.machine_learning import _update_model_config, setup_ml
from eeg_pipeline.cli.commands.preprocessing import (
    _update_alignment_event_config,
    _update_epochs_config,
    _update_pyprep_config,
    _update_preprocessing_config,
    setup_preprocessing,
)
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
                "--event-col-temperature",
                "temperature",
                "stim_temp",
                "--event-col-rating",
                "rating",
                "--event-col-pain-binary",
                "pain_binary",
                "--rename-anot-dict",
                "{\"BAD boundary\":\"BAD_boundary\"}",
                "--custom-bad-dict",
                "{\"thermalactive\":{\"0001\":[\"TP8\"]}}",
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
        self.assertEqual(config.get("event_columns.temperature"), ["temperature", "stim_temp"])
        self.assertEqual(config.get("event_columns.rating"), ["rating"])
        self.assertEqual(config.get("event_columns.pain_binary"), ["pain_binary"])
        self.assertEqual(config["pyprep"]["rename_anot_dict"]["BAD boundary"], "BAD_boundary")
        self.assertEqual(config["pyprep"]["custom_bad_dict"]["thermalactive"]["0001"], ["TP8"])


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
            erds_pain_marker_bands=["alpha"],
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
        self.assertEqual(config.get("feature_engineering.erds.pain_marker_bands"), ["alpha"])
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
                "--global-n-bootstrap",
                "2500",
                "--perm-scheme",
                "circular_shift",
                "--stats-temp-control",
                "spline",
                "--stats-allow-iid-trials",
                "--group-level-target",
                "pain_residual",
                "--group-level-control-temperature",
                "--no-group-level-control-trial-order",
                "--no-group-level-control-run-effects",
                "--group-level-max-run-dummies",
                "15",
                "--models-primary-unit",
                "run_mean",
                "--models-force-trial-iid-asymptotic",
                "--cluster-correction-enabled",
                "--cluster-correction-alpha",
                "0.01",
                "--cluster-correction-min-cluster-size",
                "4",
                "--cluster-correction-tail",
                "1",
                "--validation-min-epochs",
                "30",
                "--validation-min-channels",
                "16",
                "--validation-max-amplitude-uv",
                "400",
            ]
        )
        config = ConfigDict({"project": {"task": "thermalactive"}})
        _configure_behavior_compute_mode(args, config)

        self.assertEqual(config.get("behavior_analysis.statistics.default_n_bootstrap"), 2500)
        self.assertEqual(config.get("behavior_analysis.permutation.scheme"), "circular_shift")
        self.assertEqual(config.get("behavior_analysis.statistics.temperature_control"), "spline")
        self.assertTrue(config.get("behavior_analysis.statistics.allow_iid_trials", False))
        self.assertEqual(
            config.get("behavior_analysis.group_level.multilevel_correlations.target"),
            "pain_residual",
        )
        self.assertTrue(
            config.get("behavior_analysis.group_level.multilevel_correlations.control_temperature", False)
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
        self.assertEqual(config.get("behavior_analysis.models.primary_unit"), "run_mean")
        self.assertTrue(config.get("behavior_analysis.models.force_trial_iid_asymptotic", False))
        self.assertTrue(config.get("behavior_analysis.cluster_correction.enabled", False))
        self.assertEqual(config.get("behavior_analysis.cluster_correction.alpha"), 0.01)
        self.assertEqual(config.get("behavior_analysis.cluster_correction.min_cluster_size"), 4)
        self.assertEqual(config.get("behavior_analysis.cluster_correction.tail"), 1)
        self.assertEqual(config.get("validation.min_epochs"), 30)
        self.assertEqual(config.get("validation.min_channels"), 16)
        self.assertEqual(config.get("validation.max_amplitude_uv"), 400.0)

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
        config = ConfigDict({"project": {"task": "thermalactive"}})
        _configure_behavior_compute_mode(args, config)

        self.assertEqual(config.get("behavior_analysis.correlations.target_column"), "vas_custom")
        self.assertEqual(config.get("behavior_analysis.correlations.targets"), ["vas_custom"])


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
