from __future__ import annotations

import argparse

from eeg_pipeline.cli.commands.plotting_selection import resolve_plot_ids, unique_in_order


def test_resolve_plot_ids_uses_plots_groups_and_defaults() -> None:
    args = argparse.Namespace(
        plots=["plot_b"],
        groups=["group_x"],
        all_plots=False,
    )
    result = resolve_plot_ids(
        args,
        plot_ids={"plot_a", "plot_b", "plot_c"},
        plot_groups={"group_x": ["plot_c"]},
    )
    assert result == ["plot_b", "plot_c"]


def test_resolve_plot_ids_falls_back_to_all_when_empty() -> None:
    args = argparse.Namespace(
        plots=None,
        groups=None,
        all_plots=False,
    )
    result = resolve_plot_ids(
        args,
        plot_ids={"plot_z", "plot_a"},
        plot_groups={},
    )
    assert result == ["plot_a", "plot_z"]


def test_unique_in_order_preserves_first_occurrence() -> None:
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
