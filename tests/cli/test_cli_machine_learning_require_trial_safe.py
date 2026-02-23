import argparse
import unittest
from unittest.mock import patch

from eeg_pipeline.cli.commands.machine_learning import run_ml
from eeg_pipeline.utils.config.loader import ConfigDict


class TestMachineLearningCliSafetyOverrides(unittest.TestCase):
    def test_require_trial_ml_safe_forces_trial_analysis_mode(self):
        args = argparse.Namespace(
            list_stages=False,
            dry_run=False,
            task="task",
            bids_root=None,
            deriv_root=None,
            mode="regression",
            cv_scope="subject",
            n_perm=0,
            inner_splits=2,
            outer_jobs=1,
            rng_seed=123,
            model="ridge",
            uncertainty_alpha=0.1,
            perm_n_repeats=10,
            classification_model=None,
            target=None,
            binary_threshold=None,
            feature_families=None,
            feature_bands=None,
            feature_segments=None,
            feature_scopes=None,
            feature_stats=None,
            feature_harmonization=None,
            baseline_predictors=None,
            covariates=None,
            require_trial_ml_safe=True,
            elasticnet_alpha_grid=None,
            elasticnet_l1_ratio_grid=None,
            rf_n_estimators=None,
            rf_max_depth_grid=None,
            ridge_alpha_grid=None,
            variance_threshold_grid=None,
            fmri_signature_method=None,
            fmri_signature_contrast_name=None,
            fmri_signature_name=None,
            fmri_signature_metric=None,
            fmri_signature_normalization=None,
            fmri_signature_round_decimals=None,
            ml_plots=None,
            ml_plot_formats=None,
            ml_plot_dpi=None,
            ml_plot_top_n_features=None,
            ml_plot_diagnostics=None,
        )
        config = ConfigDict(
            {
                "project": {"task": "task", "random_state": 7},
                "feature_engineering": {"analysis_mode": "group_stats"},
                "machine_learning": {"data": {"require_trial_ml_safe": False}},
            }
        )

        with patch("eeg_pipeline.pipelines.machine_learning.MLPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run_batch.return_value = None
            run_ml(args, ["0001", "0002", "0003"], config)

        self.assertTrue(config.get("machine_learning.data.require_trial_ml_safe", False))
        self.assertEqual(config.get("feature_engineering.analysis_mode"), "trial_ml_safe")

    def test_ml_defaults_to_trial_ml_safe(self):
        args = argparse.Namespace(
            list_stages=False,
            dry_run=False,
            task="task",
            bids_root=None,
            deriv_root=None,
            mode="regression",
            cv_scope="subject",
            n_perm=0,
            inner_splits=2,
            outer_jobs=1,
            rng_seed=123,
            model="ridge",
            uncertainty_alpha=0.1,
            perm_n_repeats=10,
            classification_model=None,
            target=None,
            binary_threshold=None,
            feature_families=None,
            feature_bands=None,
            feature_segments=None,
            feature_scopes=None,
            feature_stats=None,
            feature_harmonization=None,
            baseline_predictors=None,
            covariates=None,
            require_trial_ml_safe=False,
            elasticnet_alpha_grid=None,
            elasticnet_l1_ratio_grid=None,
            rf_n_estimators=None,
            rf_max_depth_grid=None,
            ridge_alpha_grid=None,
            variance_threshold_grid=None,
            fmri_signature_method=None,
            fmri_signature_contrast_name=None,
            fmri_signature_name=None,
            fmri_signature_metric=None,
            fmri_signature_normalization=None,
            fmri_signature_round_decimals=None,
            ml_plots=None,
            ml_plot_formats=None,
            ml_plot_dpi=None,
            ml_plot_top_n_features=None,
            ml_plot_diagnostics=None,
        )
        config = ConfigDict(
            {
                "project": {"task": "task", "random_state": 7},
                "feature_engineering": {"analysis_mode": "group_stats"},
                "machine_learning": {"data": {"require_trial_ml_safe": False}},
            }
        )

        with patch("eeg_pipeline.pipelines.machine_learning.MLPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run_batch.return_value = None
            run_ml(args, ["0001", "0002", "0003"], config)

        self.assertTrue(config.get("machine_learning.data.require_trial_ml_safe", False))
        self.assertEqual(config.get("feature_engineering.analysis_mode"), "trial_ml_safe")

    def test_ml_plot_overrides_update_config(self):
        args = argparse.Namespace(
            list_stages=False,
            dry_run=False,
            task="task",
            bids_root=None,
            deriv_root=None,
            mode="shap",
            cv_scope="group",
            n_perm=0,
            inner_splits=2,
            outer_jobs=1,
            rng_seed=123,
            model="ridge",
            uncertainty_alpha=0.1,
            perm_n_repeats=10,
            classification_model=None,
            target=None,
            binary_threshold=None,
            feature_families=None,
            feature_bands=None,
            feature_segments=None,
            feature_scopes=None,
            feature_stats=None,
            feature_harmonization=None,
            baseline_predictors=None,
            covariates=None,
            require_trial_ml_safe=False,
            elasticnet_alpha_grid=None,
            elasticnet_l1_ratio_grid=None,
            rf_n_estimators=None,
            rf_max_depth_grid=None,
            ridge_alpha_grid=None,
            variance_threshold_grid=None,
            fmri_signature_method=None,
            fmri_signature_contrast_name=None,
            fmri_signature_name=None,
            fmri_signature_metric=None,
            fmri_signature_normalization=None,
            fmri_signature_round_decimals=None,
            ml_plots=False,
            ml_plot_formats=["png", "pdf"],
            ml_plot_dpi=450,
            ml_plot_top_n_features=25,
            ml_plot_diagnostics=False,
        )
        config = ConfigDict(
            {
                "project": {"task": "task", "random_state": 7},
                "feature_engineering": {"analysis_mode": "group_stats"},
                "machine_learning": {"data": {"require_trial_ml_safe": False}},
            }
        )

        with patch("eeg_pipeline.pipelines.machine_learning.MLPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run_batch.return_value = None
            run_ml(args, ["0001", "0002", "0003"], config)

        self.assertFalse(config.get("machine_learning.plotting.enabled", True))
        self.assertEqual(config.get("machine_learning.plotting.formats"), ["png", "pdf"])
        self.assertEqual(config.get("machine_learning.plotting.dpi"), 450)
        self.assertEqual(config.get("machine_learning.plotting.top_n_features"), 25)
        self.assertFalse(config.get("machine_learning.plotting.include_diagnostics", True))
