from __future__ import annotations

import json
import logging
from pathlib import Path

import mne
import pandas as pd
import pytest

import eeg_pipeline.plotting.features.phase as phase_plots
import eeg_pipeline.plotting.features.utils as plotting_utils
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.features.context import FeaturePlotContext


def _build_context(features_dir: Path, plots_dir: Path) -> FeaturePlotContext:
    return FeaturePlotContext(
        subject="0000",
        plots_dir=plots_dir,
        features_dir=features_dir,
        logger=logging.getLogger("test.plotting.phase"),
    )


def test_load_feature_set_restores_attrs_from_metadata_sidecar(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    plots_dir = tmp_path / "plots"
    itpc_dir = features_dir / "itpc"
    metadata_dir = itpc_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    table_path = itpc_dir / "features_itpc.parquet"
    table_path.touch()
    metadata_path = metadata_dir / "features_itpc.json"
    metadata_path.write_text(
        json.dumps(
            {
                "provenance": {
                    "file_attrs": {
                        "feature_granularity": "subject",
                        "itpc_method": "fold_global",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    context = _build_context(features_dir, plots_dir)
    context._safe_read_table = lambda path: pd.DataFrame(  # type: ignore[method-assign]
        {"itpc_baseline_alpha_ch_Fz_val": [0.25, 0.25]}
    )

    loaded = context._load_feature_set([table_path], mode="wide", stem="features_itpc")

    assert loaded is not None
    assert loaded.attrs["feature_granularity"] == "subject"
    assert loaded.attrs["itpc_method"] == "fold_global"


def test_plot_itpc_topomaps_rejects_condition_level_broadcast_features(tmp_path: Path) -> None:
    itpc_df = pd.DataFrame({"itpc_baseline_alpha_ch_Fz_val": [0.1, 0.1, 0.2, 0.2]})
    itpc_df.attrs["feature_granularity"] = "condition"
    itpc_df.attrs["itpc_method"] = "condition"

    info = mne.create_info(["Fz"], sfreq=100.0, ch_types=["eeg"])

    with pytest.raises(ValueError, match="condition-level"):
        phase_plots.plot_itpc_topomaps(
            itpc_df=itpc_df,
            info=info,
            subject="0000",
            save_dir=tmp_path,
            logger=logging.getLogger("test.plotting.phase.topomaps"),
            config={},
        )


def test_plot_itpc_by_condition_requires_precomputed_stats_for_broadcast_itpc(
    tmp_path: Path,
) -> None:
    itpc_df = pd.DataFrame(
        {
            "itpc_baseline_alpha_ch_Fz_val": [0.3, 0.3, 0.3, 0.3],
            "itpc_plateau_alpha_ch_Fz_val": [0.5, 0.5, 0.5, 0.5],
        }
    )
    itpc_df.attrs["feature_granularity"] = "subject"
    itpc_df.attrs["itpc_method"] = "fold_global"
    events_df = pd.DataFrame({"condition": ["A", "A", "B", "B"]})
    config = {
        "plotting": {
            "comparisons": {
                "compare_windows": True,
                "compare_columns": False,
                "comparison_windows": ["baseline", "plateau"],
            }
        }
    }

    with pytest.raises(ValueError, match="pre-computed statistics"):
        phase_plots.plot_itpc_by_condition(
            itpc_df=itpc_df,
            events_df=events_df,
            subject="0000",
            save_dir=tmp_path,
            logger=logging.getLogger("test.plotting.phase.by_condition"),
            config=config,
            stats_dir=None,
        )


def test_compute_or_load_column_stats_uses_roi_specific_precomputed_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precomputed = pd.DataFrame(
        {
            "identifier": ["alpha_frontal", "alpha_occipital"],
            "p_value": [0.01, 0.4],
            "q_value": [0.02, 0.5],
            "effect_size_d": [0.8, 0.1],
            "significant_fdr": [True, False],
        }
    )
    monkeypatch.setattr(
        plotting_utils,
        "load_precomputed_paired_stats",
        lambda **kwargs: precomputed,
    )

    qvalues, n_significant, use_precomputed = plotting_utils.compute_or_load_column_stats(
        stats_dir=Path("/tmp/fake"),
        feature_type="itpc",
        feature_keys=["alpha"],
        cell_data={},
        roi_name="frontal",
    )

    assert use_precomputed is True
    assert n_significant == 1
    assert qvalues[0] == (0.01, 0.02, 0.8, True)


def test_get_precomputed_qvalues_prefers_roi_specific_row_over_global_exact_match() -> None:
    precomputed = pd.DataFrame(
        {
            "identifier": ["alpha", "alpha_frontal"],
            "p_value": [0.9, 0.01],
            "q_value": [0.9, 0.02],
            "effect_size_d": [0.0, 0.8],
            "significant_fdr": [False, True],
        }
    )

    qvalues = plotting_utils.get_precomputed_qvalues(
        precomputed,
        feature_keys=["alpha"],
        roi_name="frontal",
    )

    assert qvalues["alpha"] == (0.01, 0.02, 0.8, True)


def test_get_pac_columns_for_roi_uses_single_scope_and_val_stat() -> None:
    pac_df = pd.DataFrame(
        {
            NamingSchema.build("pac", "active", "theta_gamma", "global", "val"): [0.1, 0.2],
            NamingSchema.build("pac", "active", "theta_gamma", "global", "z"): [1.1, 1.2],
            NamingSchema.build("pac", "active", "theta_gamma", "roi", "val", channel="frontal"): [0.3, 0.4],
            NamingSchema.build("pac", "active", "theta_gamma", "roi", "val", channel="parietal"): [0.5, 0.6],
            NamingSchema.build("pac", "active", "theta_gamma", "ch", "val", channel="C3"): [0.7, 0.8],
            NamingSchema.build("pac", "active", "theta_gamma", "ch", "val", channel="Pz"): [0.9, 1.0],
            NamingSchema.build(
                "pac",
                "active",
                "theta_gamma",
                "roi",
                "lf_sharpness_ratio",
                channel="frontal",
            ): [2.0, 2.0],
        }
    )
    rois = {"frontal": ["C3"], "parietal": ["Pz"]}

    assert phase_plots._get_pac_columns_for_roi(
        pac_df,
        "active",
        "theta_gamma",
        "all",
        rois,
        ["C3", "Pz"],
    ) == [NamingSchema.build("pac", "active", "theta_gamma", "global", "val")]

    assert phase_plots._get_pac_columns_for_roi(
        pac_df,
        "active",
        "theta_gamma",
        "frontal",
        rois,
        ["C3", "Pz"],
    ) == [NamingSchema.build("pac", "active", "theta_gamma", "roi", "val", channel="frontal")]


def test_plot_pac_by_condition_requires_precomputed_stats_for_window_comparison(
    tmp_path: Path,
) -> None:
    pac_trials_df = pd.DataFrame(
        {
            NamingSchema.build("pac", "baseline", "theta_gamma", "global", "val"): [0.1, 0.2, 0.3, 0.4],
            NamingSchema.build("pac", "plateau", "theta_gamma", "global", "val"): [0.2, 0.3, 0.4, 0.5],
        }
    )
    events_df = pd.DataFrame({"condition": ["A", "A", "B", "B"]})
    config = {
        "plotting": {
            "comparisons": {
                "compare_windows": True,
                "compare_columns": False,
                "comparison_windows": ["baseline", "plateau"],
            }
        }
    }

    with pytest.raises(ValueError, match="pre-computed statistics"):
        phase_plots.plot_pac_by_condition(
            pac_trials_df=pac_trials_df,
            events_df=events_df,
            subject="0000",
            save_dir=tmp_path,
            logger=logging.getLogger("test.plotting.phase.pac_windows"),
            config=config,
            stats_dir=None,
        )


def test_plot_pac_by_condition_requires_precomputed_stats_for_column_comparison(
    tmp_path: Path,
) -> None:
    pac_trials_df = pd.DataFrame(
        {
            NamingSchema.build("pac", "plateau", "theta_gamma", "global", "val"): [0.1, 0.2, 0.3, 0.4],
        }
    )
    events_df = pd.DataFrame({"condition": ["A", "A", "B", "B"]})
    config = {
        "plotting": {
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
        phase_plots.plot_pac_by_condition(
            pac_trials_df=pac_trials_df,
            events_df=events_df,
            subject="0000",
            save_dir=tmp_path,
            logger=logging.getLogger("test.plotting.phase.pac_columns"),
            config=config,
            stats_dir=None,
        )


def test_apply_stats_filters_matches_window_alias_columns() -> None:
    stats_df = pd.DataFrame(
        {
            "identifier": ["alpha_all"],
            "comparison_type": ["window"],
            "window1": ["baseline"],
            "window2": ["plateau"],
        }
    )

    filtered = plotting_utils._apply_stats_filters(
        stats_df,
        feature_type="itpc",
        comparison_type="window",
        condition1="baseline",
        condition2="plateau",
        roi_name="all",
    )

    assert filtered is not None
    assert len(filtered) == 1
