from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.features.complexity import (
    _get_complexity_columns,
    _plot_column_comparison,
)
from eeg_pipeline.plotting.features.utils import _get_displayed_multi_window_pairs


class TestPlottingComplexity(unittest.TestCase):
    def test_get_complexity_columns_all_requires_true_global_columns(self):
        roi_only_df = pd.DataFrame(
            {
                NamingSchema.build("comp", "active", "alpha", "roi", "lzc", channel="Frontal"): [0.1, 0.2],
                NamingSchema.build("comp", "active", "alpha", "roi", "lzc", channel="Posterior"): [0.3, 0.4],
            }
        )

        self.assertEqual(_get_complexity_columns(roi_only_df, "active", "alpha", "lzc", "all"), [])

    def test_plot_column_comparison_passes_metric_specific_feature_keys_and_roi(self):
        features_df = pd.DataFrame(
            {
                NamingSchema.build("comp", "active", "alpha", "roi", "lzc", channel="Frontal"): [0.1, 0.2, 0.3, 0.4],
                NamingSchema.build("comp", "active", "beta", "roi", "lzc", channel="Frontal"): [0.2, 0.3, 0.4, 0.5],
            }
        )
        events_df = pd.DataFrame({"condition": ["cool", "cool", "hot", "hot"]})
        config = {
            "plotting": {
                "comparisons": {
                    "compare_columns": True,
                    "comparison_column": "condition",
                    "comparison_values": ["cool", "hot"],
                    "comparison_labels": ["Cool", "Hot"],
                    "comparison_segment": "active",
                }
            }
        }

        captured: dict[str, object] = {}

        def fake_compute_or_load_column_stats(*, stats_dir, feature_type, feature_keys, cell_data, config, logger, roi_name="all", require_precomputed_stats=False):
            captured["feature_type"] = feature_type
            captured["feature_keys"] = list(feature_keys)
            captured["roi_name"] = roi_name
            return {}, 0, False

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "eeg_pipeline.plotting.features.complexity.compute_or_load_column_stats",
                side_effect=fake_compute_or_load_column_stats,
            ), patch(
                "eeg_pipeline.plotting.features.complexity._plot_single_band_comparison",
                return_value=None,
            ), patch(
                "eeg_pipeline.plotting.features.complexity.save_fig",
                return_value=None,
            ):
                _plot_column_comparison(
                    features_df=features_df,
                    events_df=events_df,
                    bands=["alpha", "beta"],
                    metrics=["lzc"],
                    roi_names=["Frontal"],
                    subject="01",
                    save_dir=Path(temp_dir),
                    config=config,
                    logger=None,
                    stats_dir=None,
                )

        self.assertEqual(captured["feature_type"], "complexity")
        self.assertEqual(
            captured["feature_keys"],
            [
                NamingSchema.build("comp", "active", "alpha", "roi", "lzc", channel="Frontal"),
                NamingSchema.build("comp", "active", "beta", "roi", "lzc", channel="Frontal"),
            ],
        )
        self.assertEqual(captured["roi_name"], "Frontal")

    def test_get_displayed_multi_window_pairs_returns_all_pairwise_comparisons(self):
        segments = ["baseline", "active", "recovery"]

        self.assertEqual(
            _get_displayed_multi_window_pairs(segments),
            [
                ("baseline", "active"),
                ("baseline", "recovery"),
                ("active", "recovery"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
