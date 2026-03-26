from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import eeg_pipeline.plotting.features.erds as erds_plots
import eeg_pipeline.plotting.features.utils as plotting_utils
from eeg_pipeline.domain.features.naming import NamingSchema


def test_get_erds_columns_for_roi_uses_single_scope_and_percent_stat() -> None:
    features_df = pd.DataFrame(
        {
            NamingSchema.build("erds", "active", "alpha", "global", "percent_mean"): [1.0, 2.0],
            NamingSchema.build("erds", "active", "alpha", "global", "db_mean"): [0.1, 0.2],
            NamingSchema.build(
                "erds",
                "active",
                "alpha",
                "roi",
                "percent_mean",
                channel="frontal",
            ): [3.0, 4.0],
            NamingSchema.build(
                "erds",
                "active",
                "alpha",
                "roi",
                "db_mean",
                channel="frontal",
            ): [0.3, 0.4],
            NamingSchema.build(
                "erds",
                "active",
                "alpha",
                "roi",
                "percent_mean",
                channel="parietal",
            ): [5.0, 6.0],
            NamingSchema.build("erds", "active", "alpha", "ch", "percent", channel="C3"): [7.0, 8.0],
            NamingSchema.build("erds", "active", "alpha", "ch", "db", channel="C3"): [0.7, 0.8],
            NamingSchema.build("erds", "active", "alpha", "ch", "percent", channel="Pz"): [9.0, 10.0],
        }
    )
    rois = {"frontal": ["C3"], "parietal": ["Pz"]}
    all_channels = ["C3", "Pz"]

    assert erds_plots._get_erds_columns_for_roi(
        features_df,
        "active",
        "alpha",
        "all",
        all_channels,
        rois,
    ) == [NamingSchema.build("erds", "active", "alpha", "global", "percent_mean")]

    assert erds_plots._get_erds_columns_for_roi(
        features_df,
        "active",
        "alpha",
        "frontal",
        all_channels,
        rois,
    ) == [
        NamingSchema.build(
            "erds",
            "active",
            "alpha",
            "roi",
            "percent_mean",
            channel="frontal",
        )
    ]


def test_get_erds_columns_for_roi_falls_back_to_matching_channel_percent_columns() -> None:
    features_df = pd.DataFrame(
        {
            NamingSchema.build("erds", "active", "alpha", "global", "percent_mean"): [1.0, 2.0],
            NamingSchema.build("erds", "active", "alpha", "ch", "percent", channel="C3"): [3.0, 4.0],
            NamingSchema.build("erds", "active", "alpha", "ch", "db", channel="C3"): [0.3, 0.4],
            NamingSchema.build("erds", "active", "alpha", "ch", "percent", channel="Pz"): [5.0, 6.0],
        }
    )
    rois = {"frontal": ["C3"], "parietal": ["Pz"]}
    all_channels = ["C3", "Pz"]

    assert erds_plots._get_erds_columns_for_roi(
        features_df,
        "active",
        "alpha",
        "frontal",
        all_channels,
        rois,
    ) == [NamingSchema.build("erds", "active", "alpha", "ch", "percent", channel="C3")]


def test_plot_erds_by_condition_requires_precomputed_stats_for_window_comparison(
    tmp_path: Path,
) -> None:
    features_df = pd.DataFrame(
        {
            NamingSchema.build("erds", "baseline", "alpha", "global", "percent_mean"): [1.0, 1.1, 1.2, 1.3],
            NamingSchema.build("erds", "plateau", "alpha", "global", "percent_mean"): [2.0, 2.1, 2.2, 2.3],
        }
    )
    events_df = pd.DataFrame({"condition": ["A", "A", "B", "B"]})
    config = {
        "plotting": {
            "overwrite": False,
            "comparisons": {
                "compare_windows": True,
                "compare_columns": False,
                "comparison_windows": ["baseline", "plateau"],
            }
        }
    }

    with pytest.raises(ValueError, match="pre-computed statistics"):
        erds_plots.plot_erds_by_condition(
            features_df=features_df,
            events_df=events_df,
            subject="0000",
            save_dir=tmp_path,
            logger=logging.getLogger("test.plotting.erds.windows"),
            config=config,
            stats_dir=None,
        )


def test_plot_erds_by_condition_requires_precomputed_stats_for_column_comparison(
    tmp_path: Path,
) -> None:
    features_df = pd.DataFrame(
        {
            NamingSchema.build("erds", "plateau", "alpha", "global", "percent_mean"): [1.0, 1.1, 1.2, 1.3],
        }
    )
    events_df = pd.DataFrame({"condition": ["A", "A", "B", "B"]})
    config = {
        "plotting": {
            "overwrite": False,
            "comparisons": {
                "compare_windows": False,
                "compare_columns": True,
                "comparison_column": "condition",
                "comparison_values": ["A", "B"],
                "comparison_windows": ["baseline", "plateau"],
                "comparison_segment": "plateau",
            }
        }
    }

    with pytest.raises(ValueError, match="pre-computed statistics"):
        erds_plots.plot_erds_by_condition(
            features_df=features_df,
            events_df=events_df,
            subject="0000",
            save_dir=tmp_path,
            logger=logging.getLogger("test.plotting.erds.columns"),
            config=config,
            stats_dir=None,
        )


def test_plot_multi_window_comparison_reports_only_displayed_window_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_save_fig(_fig, _save_path, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("eeg_pipeline.plotting.io.figures.save_fig", fake_save_fig)

    precomputed_stats = pd.DataFrame(
        {
            "identifier": ["alpha_all", "alpha_all", "alpha_all"],
            "window1": ["baseline", "baseline", "early"],
            "window2": ["early", "late", "late"],
            "p_value": [0.01, 0.02, 0.03],
            "q_value": [0.01, 0.02, 0.03],
            "effect_size_d": [0.5, 0.6, 0.7],
            "significant_fdr": [True, True, True],
        }
    )

    plotting_utils.plot_multi_window_comparison(
        data_by_band={
            "alpha": {
                "baseline": pd.Series([1.0, 1.1, 1.2, 1.3]).values,
                "early": pd.Series([2.0, 2.1, 2.2, 2.3]).values,
                "late": pd.Series([3.0, 3.1, 3.2, 3.3]).values,
            }
        },
        subject="0000",
        save_path=tmp_path / "erds_multiwindow",
        feature_label="ERDS",
        segments=["baseline", "early", "late"],
        roi_name="all",
        precomputed_stats=precomputed_stats,
    )

    footer = str(captured["footer"])
    assert "Displayed comparisons: baseline vs early, baseline vs late, early vs late" in footer
    assert "FDR: 3/3 significant" in footer
