from __future__ import annotations

from eeg_pipeline.cli.commands.plotting_definition_helpers import (
    collect_plot_definitions,
    map_plot_id_to_plotters,
)


def test_map_plot_id_to_plotters_returns_expected_tokens() -> None:
    result = map_plot_id_to_plotters("power_by_condition", ["power"])
    assert result == ["power.plot_power_condition_comparison"]

    timecourse_result = map_plot_id_to_plotters("power_timecourse", ["power"])
    assert timecourse_result == ["power.plot_power_timecourse_visualization"]

    correlation_result = map_plot_id_to_plotters("cross_frequency_power_correlation", ["power"])
    assert correlation_result == ["power.plot_power_cross_frequency_correlation"]


def test_map_plot_id_to_plotters_returns_none_for_unknown_plot_id() -> None:
    assert map_plot_id_to_plotters("unknown_plot", ["power"]) is None


def test_collect_plot_definitions_extracts_modes_and_plotters() -> None:
    (
        feature_categories,
        feature_plot_patterns,
        behavior_plots,
        tfr_plots,
        erp_plots,
        feature_plotters,
    ) = collect_plot_definitions(["power_by_condition", "connectivity_by_condition"])

    assert "power" in feature_categories
    assert "connectivity" in feature_categories
    assert "power.plot_power_condition_comparison" in feature_plotters
    assert "connectivity.plot_connectivity_condition" in feature_plotters
    assert isinstance(feature_plot_patterns, set)
    assert behavior_plots == []
    assert tfr_plots == []
    assert erp_plots == []


def test_collect_plot_definitions_includes_cross_frequency_power_plot() -> None:
    (
        feature_categories,
        feature_plot_patterns,
        _behavior_plots,
        _tfr_plots,
        _erp_plots,
        feature_plotters,
    ) = collect_plot_definitions(["cross_frequency_power_correlation"])

    assert feature_categories == {"power"}
    assert "cross_frequency_power_correlation" in feature_plot_patterns
    assert feature_plotters == ["power.plot_power_cross_frequency_correlation"]


def test_collect_plot_definitions_includes_connectivity_circle_summary_plot() -> None:
    (
        feature_categories,
        feature_plot_patterns,
        _behavior_plots,
        _tfr_plots,
        _erp_plots,
        feature_plotters,
    ) = collect_plot_definitions(["connectivity_circle"])

    assert feature_categories == {"connectivity"}
    assert "*_circle" in feature_plot_patterns
    assert feature_plotters == ["connectivity.plot_connectivity_mne_suite"]
