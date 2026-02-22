import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from tests.pipelines_test_utils import DotConfig


class TestDoseResponsePlotting(unittest.TestCase):
    def test_stat_matching_logratio_does_not_match_db_mean(self):
        from eeg_pipeline.plotting.behavioral.dose_response import _stat_matches_request

        self.assertTrue(_stat_matches_request(requested="logratio", feature_stat="logratio", scope="ch"))
        self.assertTrue(_stat_matches_request(requested="logratio", feature_stat="logratio_mean", scope="global"))
        self.assertFalse(_stat_matches_request(requested="logratio", feature_stat="db_mean", scope="global"))

    def test_visualize_dose_response_runs_power_category_when_requested(self):
        from eeg_pipeline.plotting.behavioral.dose_response import visualize_dose_response

        deriv_root = Path(tempfile.mkdtemp())
        trials = pd.DataFrame(
            {
                "stimulus_temp": [44.0, 45.0, 46.0],
                "power_active_alpha_roi_central_mean": [0.1, 0.2, 0.3],
            }
        )
        config = DotConfig(
            {
                "plotting": {
                    "plots": {
                        "behavior": {
                            "dose_response": {
                                "dose_column": "stimulus_temp",
                                "response_column": "power",
                                "segment": "active",
                                "stat": "mean",
                            }
                        }
                    }
                }
            }
        )
        logger = Mock()
        fake_plot_cfg = Mock()
        fake_plot_cfg.get_behavioral_config.return_value = {}

        with patch(
            "eeg_pipeline.plotting.behavioral.dose_response.get_plot_config",
            return_value=fake_plot_cfg,
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response._load_trial_table",
            return_value=(trials, Path("/tmp/trials_power.parquet")),
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response.get_frequency_band_names",
            return_value=["alpha"],
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response.get_rois",
            return_value={},
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response._plot_category_features_vs_dose_single_subject",
            return_value={"power_scope-global_id-all_band-alpha_dose_response": Path("/tmp/power_plot")},
        ) as cat_plot:
            saved = visualize_dose_response(
                subject="0000",
                deriv_root=deriv_root,
                task="thermalactive",
                config=config,
                logger=logger,
            )

        self.assertIn("power_scope-global_id-all_band-alpha_dose_response", saved)
        self.assertEqual(cat_plot.call_count, 1)
        self.assertEqual(cat_plot.call_args.kwargs["category"], "power")

    def test_visualize_dose_response_still_runs_power_category_when_power_column_exists(self):
        from eeg_pipeline.plotting.behavioral.dose_response import visualize_dose_response

        deriv_root = Path(tempfile.mkdtemp())
        trials = pd.DataFrame(
            {
                "stimulus_temp": [44.0, 45.0, 46.0],
                "power": [1.0, 2.0, 3.0],
                "power_active_alpha_global_mean": [0.11, 0.21, 0.31],
            }
        )
        config = DotConfig(
            {
                "plotting": {
                    "plots": {
                        "behavior": {
                            "dose_response": {
                                "dose_column": "stimulus_temp",
                                "response_column": "power",
                                "segment": "active",
                                "stat": "mean",
                            }
                        }
                    }
                }
            }
        )
        logger = Mock()
        fake_plot_cfg = Mock()
        fake_plot_cfg.get_behavioral_config.return_value = {}

        with patch(
            "eeg_pipeline.plotting.behavioral.dose_response.get_plot_config",
            return_value=fake_plot_cfg,
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response._load_trial_table",
            return_value=(trials, Path("/tmp/trials_power.parquet")),
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response.get_frequency_band_names",
            return_value=["alpha"],
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response.get_rois",
            return_value={},
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response._plot_xy_mean_sem",
            return_value=None,
        ), patch(
            "eeg_pipeline.plotting.behavioral.dose_response._plot_category_features_vs_dose_single_subject",
            return_value={"power_scope-global_id-global_band-alpha_dose_response": Path("/tmp/power_global_plot")},
        ) as cat_plot:
            saved = visualize_dose_response(
                subject="0000",
                deriv_root=deriv_root,
                task="thermalactive",
                config=config,
                logger=logger,
            )

        self.assertIn("power_scope-global_id-global_band-alpha_dose_response", saved)
        self.assertEqual(cat_plot.call_count, 1)
        self.assertEqual(cat_plot.call_args.kwargs["category"], "power")


if __name__ == "__main__":
    unittest.main()
