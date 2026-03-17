from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.plotting.features.power import (
    _build_topomap_panel,
    _compute_group_band_summary_stats,
    _compute_column_effect_summary,
    _compute_group_curve_significance_mask,
    _compute_group_paired_effect_forest_data,
    _compute_group_paired_effect_summary,
    _compute_group_paired_sample_count_summary,
    _compute_group_timecourse_significance_mask,
    _compute_paired_effect_matrix,
    _compute_shared_topomap_vlim,
    _compute_window_mean_series,
    _compute_window_effect_summary,
    _draw_curve_significance_strip,
    _draw_group_subject_traces,
    _format_condition_display_label,
    _format_triptych_condition_label,
    _format_topomap_condition_title_label,
)
from eeg_pipeline.plotting.features.utils import (
    _format_count_range,
    _compute_paired_wilcoxon_stats,
    _compute_paired_differences,
    _summarize_multi_window_sample_counts,
    _summarize_paired_sample_counts,
    _plot_single_band_comparison,
)
from eeg_pipeline.utils.analysis.stats.paired_comparisons import compute_paired_cohens_d


def test_compute_window_effect_summary_returns_roi_band_matrices() -> None:
    power_df = pd.DataFrame(
        {
            "power_baseline_alpha_ch_Fz_logratio": [0.10, 0.12, 0.11, 0.09],
            "power_baseline_alpha_ch_Cz_logratio": [0.08, 0.07, 0.09, 0.08],
            "power_active_alpha_ch_Fz_logratio": [0.35, 0.33, 0.32, 0.31],
            "power_active_alpha_ch_Cz_logratio": [0.28, 0.27, 0.29, 0.26],
            "power_baseline_beta_ch_Fz_logratio": [0.02, 0.03, 0.01, 0.02],
            "power_active_beta_ch_Fz_logratio": [0.05, 0.06, 0.04, 0.05],
        }
    )
    config = {"statistics": {"fdr_alpha": 0.05}}
    bands = ["alpha", "beta"]
    roi_names = ["all", "Frontal"]
    rois = {"Frontal": [r"^Fz$"]}
    all_channels = ["Fz", "Cz"]

    effect_df, qvalue_df = _compute_window_effect_summary(
        power_df=power_df,
        bands=bands,
        segments=["baseline", "active"],
        roi_names=roi_names,
        rois=rois,
        all_channels=all_channels,
        config=config,
    )

    assert list(effect_df.index) == roi_names
    assert list(effect_df.columns) == bands
    assert effect_df.loc["all", "alpha"] > 0
    assert effect_df.loc["Frontal", "alpha"] > 0
    assert np.isfinite(qvalue_df.loc["all", "alpha"])


def test_compute_column_effect_summary_uses_configured_masks() -> None:
    power_df = pd.DataFrame(
        {
            "power_active_alpha_ch_Fz_logratio": [0.10, 0.11, 0.12, 0.34, 0.35, 0.36],
            "power_active_alpha_ch_Cz_logratio": [0.09, 0.10, 0.11, 0.30, 0.31, 0.32],
            "power_active_beta_ch_Fz_logratio": [0.03, 0.04, 0.03, 0.05, 0.06, 0.05],
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0, 0, 1, 1, 1]})
    config = {
        "statistics": {"fdr_alpha": 0.05},
        "plotting": {
            "comparisons": {
                "compare_columns": True,
                "comparison_column": "condition",
                "comparison_values": [0, 1],
                "comparison_labels": ["Cool", "Hot"],
            }
        },
    }
    bands = ["alpha", "beta"]
    roi_names = ["all", "Frontal"]
    rois = {"Frontal": [r"^Fz$"]}
    all_channels = ["Fz", "Cz"]

    effect_df, qvalue_df, label1, label2 = _compute_column_effect_summary(
        power_df=power_df,
        events_df=events_df,
        bands=bands,
        seg_name="active",
        roi_names=roi_names,
        rois=rois,
        all_channels=all_channels,
        config=config,
    )

    assert (label1, label2) == ("Cool", "Hot")
    assert effect_df.loc["all", "alpha"] > 0
    assert np.isfinite(qvalue_df.loc["all", "alpha"])


def test_compute_group_paired_effect_summary_returns_roi_band_matrices() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.3},
                "sub-02": {"baseline": 0.9, "plateau": 1.2},
                "sub-03": {"baseline": 1.1, "plateau": 1.4},
            },
            "beta": {
                "sub-01": {"baseline": 0.8, "plateau": 0.9},
                "sub-02": {"baseline": 0.7, "plateau": 0.8},
                "sub-03": {"baseline": 0.9, "plateau": 1.0},
            },
        },
        "Frontal": {
            "alpha": {
                "sub-01": {"baseline": 1.1, "plateau": 1.5},
                "sub-02": {"baseline": 1.0, "plateau": 1.4},
                "sub-03": {"baseline": 1.2, "plateau": 1.6},
            }
        },
    }

    effect_df, qvalue_df = _compute_group_paired_effect_summary(
        subject_values=subject_values,
        bands=["alpha", "beta"],
        roi_names=["all", "Frontal"],
        labels=("baseline", "plateau"),
        config={"statistics": {"fdr_alpha": 0.05}},
    )

    assert list(effect_df.index) == ["all", "Frontal"]
    assert list(effect_df.columns) == ["alpha", "beta"]
    assert effect_df.loc["all", "alpha"] > 0
    assert effect_df.loc["Frontal", "alpha"] > 0
    assert np.isfinite(qvalue_df.loc["all", "alpha"])


def test_compute_group_paired_effect_summary_handles_zero_differences() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.0},
                "sub-02": {"baseline": 1.2, "plateau": 1.2},
                "sub-03": {"baseline": 0.8, "plateau": 0.8},
            }
        }
    }

    effect_df, qvalue_df = _compute_group_paired_effect_summary(
        subject_values=subject_values,
        bands=["alpha"],
        roi_names=["all"],
        labels=("baseline", "plateau"),
        config={"statistics": {"fdr_alpha": 0.05}},
    )

    assert effect_df.loc["all", "alpha"] == 0.0
    assert qvalue_df.loc["all", "alpha"] == 1.0


def test_compute_group_paired_sample_count_summary_counts_complete_pairs() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.2},
                "sub-02": {"baseline": 0.9, "plateau": 1.1},
                "sub-03": {"baseline": 1.1},
            },
            "beta": {
                "sub-01": {"baseline": 0.8, "plateau": 0.9},
                "sub-02": {"baseline": 0.7, "plateau": 0.8},
            },
        }
    }

    count_df = _compute_group_paired_sample_count_summary(
        subject_values=subject_values,
        bands=["alpha", "beta"],
        roi_names=["all"],
        labels=("baseline", "plateau"),
    )

    assert count_df.loc["all", "alpha"] == 2
    assert count_df.loc["all", "beta"] == 2


def test_compute_group_paired_effect_forest_data_returns_ci_and_qvalues() -> None:
    subject_values = {
        "all": {
            "alpha": {
                "sub-01": {"baseline": 1.0, "plateau": 1.3},
                "sub-02": {"baseline": 0.9, "plateau": 1.2},
                "sub-03": {"baseline": 1.1, "plateau": 1.4},
            },
            "beta": {
                "sub-01": {"baseline": 0.8, "plateau": 0.9},
                "sub-02": {"baseline": 0.7, "plateau": 0.8},
                "sub-03": {"baseline": 0.9, "plateau": 1.0},
            },
        }
    }

    forest_df = _compute_group_paired_effect_forest_data(
        subject_values=subject_values,
        bands=["alpha", "beta"],
        roi_names=["all"],
        labels=("baseline", "plateau"),
        config={"statistics": {"fdr_alpha": 0.05}},
    )

    assert set(forest_df["band"]) == {"alpha", "beta"}
    assert set(forest_df["roi_name"]) == {"all"}
    assert np.all(np.isfinite(forest_df["effect_size"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(forest_df["q_value"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(forest_df["ci_low"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(forest_df["ci_high"].to_numpy(dtype=float)))


def test_compute_shared_topomap_vlim_uses_descriptive_range_for_positive_data() -> None:
    arrays = [
        np.array([0.08, 0.10, 0.12, 0.14]),
        np.array([0.09, 0.11, 0.13, 0.15]),
    ]

    vmin, vmax = _compute_shared_topomap_vlim(arrays, config={}, symmetric=False)

    assert vmin >= 0
    assert vmax > vmin


def test_format_topomap_condition_title_label_expands_raw_value() -> None:
    config = {
        "plotting": {
            "comparisons": {
                "comparison_column": "pain_binary_coded",
                "comparison_values": [0.0, 1.0],
            }
        }
    }

    assert _format_topomap_condition_title_label("1.0", config) == "pain_binary_coded=1.0"


def test_format_condition_display_label_preserves_configured_label() -> None:
    config = {
        "plotting": {
            "comparisons": {
                "comparison_column": "pain_binary_coded",
                "comparison_values": [0.0, 1.0],
                "comparison_labels": ["Low pain", "High pain"],
            }
        }
    }

    assert _format_condition_display_label("High pain", config) == "High pain"


def test_format_triptych_condition_label_keeps_raw_comparison_value_compact() -> None:
    config = {
        "plotting": {
            "comparisons": {
                "comparison_column": "pain_binary_coded",
                "comparison_values": [0.0, 1.0],
            }
        }
    }

    assert _format_triptych_condition_label("1.0", config) == "1.0"


def test_compute_group_timecourse_significance_mask_flags_consistent_effect() -> None:
    condition1 = np.ones((8, 5), dtype=float)
    condition2 = np.ones((8, 5), dtype=float)
    condition2[:, 2:] += 0.5

    significant_mask = _compute_group_timecourse_significance_mask(condition1, condition2, config={})

    assert significant_mask.shape == (5,)
    assert not bool(significant_mask[0])
    assert bool(significant_mask[-1])


def test_compute_group_curve_significance_mask_flags_consistent_effect() -> None:
    condition1 = np.ones((6, 4), dtype=float)
    condition2 = np.ones((6, 4), dtype=float)
    condition2[:, 1:] += 0.4

    significant_mask = _compute_group_curve_significance_mask(condition1, condition2, config={})

    assert significant_mask.shape == (4,)
    assert not bool(significant_mask[0])
    assert bool(significant_mask[-1])


def test_compute_group_band_summary_stats_flags_band_level_effect() -> None:
    freqs = np.array([2.0, 4.0, 8.0, 16.0], dtype=float)
    frequency_bands = {
        "delta": (1.0, 4.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
    }
    condition1 = np.ones((6, 4), dtype=float)
    condition2 = np.ones((6, 4), dtype=float)
    condition2[:, 2] += 0.5

    band_stats = _compute_group_band_summary_stats(
        condition1,
        condition2,
        freqs,
        frequency_bands,
        config={},
    )

    assert set(band_stats) == {"delta", "alpha", "beta"}
    assert not bool(band_stats["delta"]["significant"])
    assert bool(band_stats["alpha"]["significant"])
    assert band_stats["alpha"]["effect_size"] > 0


def test_build_topomap_panel_requires_more_than_minimum_channels() -> None:
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq=250.0, ch_types="eeg")

    assert _build_topomap_panel({"Fz": 1.0, "Cz": 2.0, "Pz": 3.0}, info) is None

    panel = _build_topomap_panel({"Fz": 1.0, "Cz": 2.0, "Pz": 3.0, "Oz": 4.0}, info)

    assert panel is not None
    data, panel_info = panel
    assert data.shape == (4,)
    assert len(panel_info.ch_names) == 4


def test_draw_group_subject_traces_requires_matching_time_axis() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="time axis length"):
            _draw_group_subject_traces(
                ax,
                np.array([0.0, 1.0, 2.0], dtype=float),
                np.ones((3, 2), dtype=float),
                color="tab:blue",
            )
    finally:
        plt.close(fig)


def test_draw_curve_significance_strip_requires_matching_axis_shape() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="must match the axis shape"):
            _draw_curve_significance_strip(
                ax,
                np.array([1.0, 2.0, 4.0], dtype=float),
                np.array([True, False], dtype=bool),
            )
    finally:
        plt.close(fig)


def test_compute_window_mean_series_returns_one_value_per_sample() -> None:
    sample_matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=float,
    )
    times = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)

    window_mean = _compute_window_mean_series(
        sample_matrix,
        times,
        (0.0, 1.0),
        context="test window summary",
    )

    np.testing.assert_allclose(window_mean, np.array([2.5, 5.0], dtype=float))


def test_compute_paired_effect_matrix_uses_condition2_minus_condition1() -> None:
    condition1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    condition2 = np.array([[1.5, 1.0], [4.0, 6.0]], dtype=float)

    effect_matrix = _compute_paired_effect_matrix(
        condition1,
        condition2,
        context="test effect matrix",
    )

    np.testing.assert_allclose(
        effect_matrix,
        np.array([[0.5, -1.0], [1.0, 2.0]], dtype=float),
    )


def test_compute_paired_differences_uses_condition2_minus_condition1() -> None:
    condition1 = np.array([1.0, 2.0, 3.0], dtype=float)
    condition2 = np.array([1.5, 1.0, 4.0], dtype=float)

    differences = _compute_paired_differences(condition1, condition2)

    np.testing.assert_allclose(differences, np.array([0.5, -1.0, 1.0], dtype=float))


def test_compute_paired_cohens_d_returns_infinite_for_constant_nonzero_shift() -> None:
    before = np.array([1.0, 2.0, 3.0], dtype=float)
    after = np.array([1.5, 2.5, 3.5], dtype=float)

    effect_size = compute_paired_cohens_d(before, after)

    assert np.isposinf(effect_size)


def test_compute_paired_wilcoxon_stats_returns_rank_biserial_for_constant_shift() -> None:
    before = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    after = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)

    _, effect_size = _compute_paired_wilcoxon_stats(before, after)

    assert effect_size == pytest.approx(1.0)


def test_sample_count_summaries_report_exact_values_and_ranges() -> None:
    paired_summary = _summarize_paired_sample_counts(
        {
            "alpha": (np.array([1.0, 2.0]), np.array([1.1, 2.1])),
            "beta": (np.array([1.0, 2.0, 3.0]), np.array([1.2, 2.2, 3.2])),
        },
        "trials",
    )
    multi_window_summary = _summarize_multi_window_sample_counts(
        {
            "alpha": {
                "baseline": np.array([1.0, 2.0]),
                "plateau": np.array([1.1, 2.1, 3.1]),
            },
            "beta": {
                "baseline": np.array([0.5, 0.6, 0.7]),
            },
        },
        "trials",
    )

    assert _format_count_range([4, 4, 4]) == "4"
    assert _format_count_range([2, 4, 3]) == "2-4"
    assert paired_summary == "N: 2-3 trials"
    assert multi_window_summary == "N per window: 2-3 trials"


def test_plot_single_band_comparison_adds_delta_column_for_paired_data() -> None:
    fig, (ax, delta_ax) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [5, 1], "wspace": 0.1},
    )
    plot_cfg = SimpleNamespace(font=SimpleNamespace(title=12, small=8))

    try:
        _plot_single_band_comparison(
            ax=ax,
            delta_ax=delta_ax,
            condition1_values=np.array([1.0, 1.2, 1.4], dtype=float),
            condition2_values=np.array([1.3, 1.5, 1.8], dtype=float),
            band="alpha",
            label1="Low",
            label2="High",
            band_color="#ff7f0e",
            condition1_color="#1f77b4",
            condition2_color="#d62728",
            q_value=0.02,
            effect_size=0.8,
            is_significant=True,
            plot_cfg=plot_cfg,
            config={},
        )

        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert tick_labels == ["Low", "High"]
        assert ax.get_xlim()[1] <= 1.4
        assert len(fig.axes) == 2
        delta_tick_labels = [tick.get_text() for tick in delta_ax.get_xticklabels()]
        assert delta_tick_labels == ["Δ"]
        assert delta_ax.get_position().x0 > ax.get_position().x1
        assert ax.get_ylim()[0] > 0.8
        assert delta_ax.get_ylim()[0] < 0.0 < delta_ax.get_ylim()[1]
    finally:
        plt.close(fig)
