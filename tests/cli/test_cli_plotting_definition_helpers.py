from __future__ import annotations

from eeg_pipeline.cli.commands.plotting_definition_helpers import (
    collect_plot_definitions,
    map_plot_id_to_plotters,
)


def test_map_plot_id_to_plotters_returns_expected_tokens() -> None:
    result = map_plot_id_to_plotters("power_by_condition", ["power"])
    assert result == ["power.plot_power_condition_comparison"]


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
