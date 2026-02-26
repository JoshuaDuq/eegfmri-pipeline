from __future__ import annotations

import unittest

import pandas as pd

from eeg_pipeline.utils.data.epochs import _find_missing_event_columns
from tests.pipelines_test_utils import DotConfig


class TestEventColumnValidation(unittest.TestCase):
    def test_outcome_required_accepts_vas_like_numeric_column(self):
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
                    "outcome": ["outcome", "rating"],
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

    def test_outcome_required_rejects_non_numeric_vas_like_column(self):
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
                    "outcome": ["outcome", "rating"],
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
            ["event_columns.outcome (tried: ['outcome', 'rating'])"],
        )


if __name__ == "__main__":
    unittest.main()
