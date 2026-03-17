from __future__ import annotations

import unittest

import pandas as pd

from eeg_pipeline.utils.analysis.stats.trialwise_regression import (
    run_trialwise_feature_regressions,
)
from eeg_pipeline.utils.data.columns import (
    require_outcome_column,
    require_predictor_column,
    resolve_outcome_column,
    resolve_predictor_column,
)
from eeg_pipeline.utils.data.epochs import _find_missing_event_columns
from tests.pipelines_test_utils import DotConfig


class TestEventColumnValidation(unittest.TestCase):
    def test_outcome_required_accepts_configured_numeric_column(self):
        events_df = pd.DataFrame(
            {
                "trial_type": ["stim", "stim"],
                "vas_final_coded_rating": [12.0, 37.5],
            }
        )
        config = DotConfig(
            {
                "event_columns": {
                    "required": ["outcome"],
                    "outcome": ["vas_final_coded_rating", "rating"],
                }
            }
        )

        missing = _find_missing_event_columns(
            events_df,
            config["event_columns"],
            required_groups=config.get("event_columns.required"),
            config=config,
        )
        self.assertEqual(missing, [])

    def test_outcome_required_rejects_non_numeric_configured_column(self):
        events_df = pd.DataFrame(
            {
                "trial_type": ["stim", "stim"],
                "vas_final_coded_rating": ["low", "high"],
            }
        )
        config = DotConfig(
            {
                "event_columns": {
                    "required": ["outcome"],
                    "outcome": ["vas_final_coded_rating", "rating"],
                }
            }
        )

        missing = _find_missing_event_columns(
            events_df,
            config["event_columns"],
            required_groups=config.get("event_columns.required"),
            config=config,
        )
        self.assertEqual(
            missing,
            ["event_columns.outcome (tried: ['vas_final_coded_rating', 'rating'])"],
        )

    def test_behavior_resolution_does_not_fallback_to_unconfigured_outcome_column(self):
        events_df = pd.DataFrame(
            {
                "outcome": [1.0, 2.0],
                "custom_score": [3.0, 4.0],
            }
        )
        config = DotConfig(
            {
                "event_columns": {
                    "outcome": ["custom_score"],
                }
            }
        )

        self.assertEqual(resolve_outcome_column(events_df, config), "custom_score")

        config_without_match = DotConfig(
            {
                "event_columns": {
                    "outcome": ["vas_rating"],
                }
            }
        )
        self.assertIsNone(resolve_outcome_column(events_df, config_without_match))
        with self.assertRaisesRegex(ValueError, "Could not resolve a numeric behavior outcome column"):
            require_outcome_column(events_df, config_without_match)

    def test_behavior_resolution_does_not_fallback_to_unconfigured_predictor_column(self):
        events_df = pd.DataFrame(
            {
                "predictor": [1.0, 2.0],
                "stim_temp": [45.0, 46.0],
            }
        )
        config = DotConfig(
            {
                "event_columns": {
                    "predictor": ["stim_temp"],
                }
            }
        )

        self.assertEqual(resolve_predictor_column(events_df, config), "stim_temp")

        config_without_match = DotConfig(
            {
                "event_columns": {
                    "predictor": ["dose"],
                }
            }
        )
        self.assertIsNone(resolve_predictor_column(events_df, config_without_match))
        with self.assertRaisesRegex(ValueError, "Could not resolve a numeric behavior predictor column"):
            require_predictor_column(events_df, config_without_match)

    def test_trialwise_regression_requires_configured_predictor_when_enabled(self):
        trial_df = pd.DataFrame(
            {
                "rating": [10.0, 11.0, 12.0, 13.0],
                "predictor": [44.0, 45.0, 46.0, 47.0],
                "power_alpha": [0.2, 0.3, 0.4, 0.5],
            }
        )
        config = DotConfig(
            {
                "event_columns": {
                    "outcome": ["rating"],
                    "predictor": ["stimulus_temp"],
                },
                "behavior_analysis": {
                    "regression": {
                        "outcome": "outcome",
                        "include_predictor": True,
                        "include_interaction": False,
                        "include_trial_order": False,
                        "include_run_block": False,
                        "min_samples": 3,
                    }
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "Could not resolve a numeric behavior predictor column"):
            run_trialwise_feature_regressions(
                trial_df,
                feature_cols=["power_alpha"],
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
