from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.features.aperiodic import plot_aperiodic_by_condition


def test_plot_aperiodic_by_condition_passes_metric_specific_multigroup_match_terms(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    features_df = pd.DataFrame(
        {
            NamingSchema.build("aperiodic", "active", "alpha", "global", "peak_height"): [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ]
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0, 1, 1, 2, 2]})
    config = {
        "plotting": {
            "comparisons": {
                "compare_windows": False,
                "compare_columns": True,
                "comparison_windows": ["baseline", "active"],
                "comparison_segment": "active",
                "comparison_column": "condition",
                "comparison_values": [0, 1, 2],
                "comparison_labels": ["Cool", "Warm", "Hot"],
                "comparison_rois": ["all"],
            }
        }
    }

    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.utils.resolve_complete_multigroup_plot_groups",
        lambda *args, **kwargs: (
            {
                "Cool": pd.Series([True, True, False, False, False, False]).to_numpy(),
                "Warm": pd.Series([False, False, True, True, False, False]).to_numpy(),
                "Hot": pd.Series([False, False, False, False, True, True]).to_numpy(),
            },
            ["Cool", "Warm", "Hot"],
        ),
    )
    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.utils.load_multigroup_stats",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "feature": ["aperiodic_alpha_peak_height_roi-all"],
                "identifier": ["aperiodic_alpha_peak_height_roi-all"],
                "group1": ["Cool"],
                "group2": ["Warm"],
                "q_value": [0.2],
                "significant_fdr": [False],
            }
        ),
    )

    def fake_plot_multi_group_column_comparison(**kwargs) -> None:
        captured["stats_match_terms"] = kwargs["stats_match_terms"]

    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.utils.plot_multi_group_column_comparison",
        fake_plot_multi_group_column_comparison,
    )

    plot_aperiodic_by_condition(
        features_df=features_df,
        events_df=events_df,
        subject="01",
        save_dir=tmp_path,
        logger=Mock(),
        config=config,
        stats_dir=tmp_path,
    )

    assert captured["stats_match_terms"] == {
        "alpha peak height": ("alpha", "peak_height"),
    }
