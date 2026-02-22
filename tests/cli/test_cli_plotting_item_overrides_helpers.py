from __future__ import annotations

import pytest

from eeg_pipeline.cli.commands.plotting_item_overrides import (
    apply_plot_item_overrides,
    parse_bool,
    parse_plot_item_configs,
    validate_plot_item_configs,
)


def test_parse_bool_handles_expected_tokens() -> None:
    assert parse_bool("true") is True
    assert parse_bool("No") is False
    assert parse_bool("maybe") is None


def test_parse_plot_item_configs_groups_entries() -> None:
    raw = [
        ["plot_a", "compare_windows", "true"],
        ["plot_a", "comparison_windows", "baseline", "active"],
        ["plot_b", "comparison_column", "condition"],
    ]
    parsed = parse_plot_item_configs(raw)
    assert parsed["plot_a"]["compare_windows"] == ["true"]
    assert parsed["plot_a"]["comparison_windows"] == ["baseline", "active"]
    assert parsed["plot_b"]["comparison_column"] == ["condition"]


def test_validate_plot_item_configs_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        validate_plot_item_configs(
            {"plot_a": {"not_a_key": ["x"]}},
            {"plot_a"},
        )


def test_apply_plot_item_overrides_sets_expected_paths() -> None:
    config: dict[str, object] = {}
    apply_plot_item_overrides(
        config,
        {
            "compare_windows": ["true"],
            "comparison_column": ["condition"],
            "tfr_topomap_window_count": ["4"],
        },
    )
    assert config["plotting.comparisons.compare_windows"] is True
    assert config["plotting.comparisons.comparison_column"] == "condition"
    assert config["time_frequency_analysis.topomap.temporal.window_count"] == 4
