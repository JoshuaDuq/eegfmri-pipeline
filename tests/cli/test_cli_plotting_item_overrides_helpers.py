from __future__ import annotations

import pytest

from eeg_pipeline.cli.commands.plotting_item_overrides import (
    apply_plot_item_overrides,
    parse_bool,
    parse_plot_item_configs,
    validate_plot_item_configs,
)
from eeg_pipeline.cli.commands.plotting_runner_helpers import (
    _collect_shared_group_overrides,
    _resolve_plot_overrides,
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
            "source_segment": ["plateau"],
            "source_condition": ["1.0"],
            "tfr_topomap_window_count": ["4"],
        },
    )
    assert config["plotting.comparisons.compare_windows"] is True
    assert config["plotting.comparisons.comparison_column"] == "condition"
    assert config["plotting.plots.features.sourcelocalization.segment"] == "plateau"
    assert config["plotting.plots.features.sourcelocalization.condition"] == "1.0"
    assert config["time_frequency_analysis.topomap.temporal.window_count"] == 4


def test_collect_shared_group_overrides_propagates_power_comparison_settings() -> None:
    plot_ids = ["power_by_condition", "power_timecourse"]
    plot_item_configs = {
        "power_by_condition": {
            "comparison_column": ["condition"],
            "comparison_values": ["0", "1"],
        }
    }

    shared = _collect_shared_group_overrides(plot_ids, plot_item_configs)
    resolved = _resolve_plot_overrides("power_timecourse", plot_item_configs, shared)

    assert resolved["comparison_column"] == ["condition"]
    assert resolved["comparison_values"] == ["0", "1"]


def test_collect_shared_group_overrides_rejects_conflicting_group_settings() -> None:
    plot_ids = ["power_by_condition", "power_spectral_density"]
    plot_item_configs = {
        "power_by_condition": {"comparison_column": ["condition_a"]},
        "power_spectral_density": {"comparison_column": ["condition_b"]},
    }

    with pytest.raises(ValueError, match="Conflicting --plot-item-config values"):
        _collect_shared_group_overrides(plot_ids, plot_item_configs)
