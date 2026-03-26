from __future__ import annotations

import ast
import logging
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from eeg_pipeline.plotting.features import connectivity as connectivity_plots

CONNECTIVITY_PLOTTING_PATH = (
    Path(__file__).resolve().parents[2]
    / "eeg_pipeline"
    / "plotting"
    / "features"
    / "connectivity.py"
)
PLOTTING_UTILS_PATH = (
    Path(__file__).resolve().parents[2]
    / "eeg_pipeline"
    / "plotting"
    / "features"
    / "utils.py"
)


def _get_function_def(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found")


def _keyword_is_none(call: ast.Call, keyword_name: str) -> bool:
    for keyword in call.keywords:
        if keyword.arg == keyword_name:
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is None
    return False


def test_normalize_condition_effects_maps_available_effect_size_columns() -> None:
    source = PLOTTING_UTILS_PATH.read_text()

    assert 'if "cohens_d" in result.columns:' in source
    assert 'elif "hedges_g" in result.columns:' in source
    assert 'result["effect_size_d"] = pd.to_numeric(result["cohens_d"], errors="coerce")' in source
    assert 'result["effect_size_d"] = pd.to_numeric(result["hedges_g"], errors="coerce")' in source


def test_window_connectivity_comparison_ignores_precomputed_stats() -> None:
    tree = ast.parse(CONNECTIVITY_PLOTTING_PATH.read_text())
    function_def = _get_function_def(tree, "_plot_window_comparison_connectivity")

    paired_calls = []
    multi_window_calls = []
    for node in ast.walk(function_def):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "plot_paired_comparison":
            paired_calls.append(node)
        if node.func.id == "plot_multi_window_comparison":
            multi_window_calls.append(node)

    assert paired_calls
    assert multi_window_calls
    assert all(_keyword_is_none(call, "stats_dir") for call in paired_calls)
    assert all(_keyword_is_none(call, "stats_dir") for call in multi_window_calls)


def test_column_connectivity_comparison_ignores_precomputed_stats() -> None:
    source = CONNECTIVITY_PLOTTING_PATH.read_text()

    assert "compute_connectivity_column_stats(cell_data)" in source
    assert "needs_centering = observed_span / reference_scale < 0.05" in source
    assert 'center_label = f"Centered at {center_value:.3e}"' in source
    assert "ax.axhline(0.0" in source
    assert 'annotation_text = f"q={qvalue:.3f}{sig_marker}\\nr={effect_size:.2f}"' in source
    assert "transform=ax.transAxes" in source
    assert 'return f"N: {format_range(counts1)} vs {format_range(counts2)} trials"' in source
    assert "yrange = ymax - ymin if ymax > ymin else 0.1" not in source
    assert "compute_or_load_column_stats(" not in source
    assert "plot_multi_group_column_comparison(" in source
    assert "stats_dir=None" in source


def test_connectivity_plot_defaults_exclude_unsupported_coherence() -> None:
    import yaml

    config = yaml.safe_load(
        CONNECTIVITY_PLOTTING_PATH.parents[3]
        .joinpath("eeg_pipeline", "utils", "config", "eeg_config.yaml")
        .read_text()
    )
    measures = config["plotting"]["plots"]["features"]["connectivity"]["measures"]

    assert measures == ["aec", "wpli", "pli", "plv", "imcoh"]
    assert "coherence" not in measures


def test_connectivity_plot_validates_requested_segments_and_measures() -> None:
    source = CONNECTIVITY_PLOTTING_PATH.read_text()

    assert "segments = _detect_segments_from_data(features_df, config, logger) if compare_windows else []" in source
    assert "_validate_connectivity_plot_request(" in source
    assert "Use extractor-supported measures only: aec, wpli, pli, plv, imcoh." in source


def test_compute_top_fraction_threshold_uses_only_unique_upper_triangle_weights() -> None:
    adjacency = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.8],
            [0.4, 0.8, 0.0],
        ],
        dtype=float,
    )

    threshold = connectivity_plots._compute_top_fraction_threshold(
        adjacency,
        top_fraction=1.0 / 3.0,
        config_path="plotting.plots.features.connectivity.network_top_fraction",
    )

    unique_edge_weights = np.array([0.2, 0.4, 0.8], dtype=float)
    expected = float(np.percentile(unique_edge_weights, (1.0 - (1.0 / 3.0)) * 100.0))
    wrong = float(
        np.percentile(
            np.triu(np.abs(adjacency), k=1)[np.isfinite(np.triu(np.abs(adjacency), k=1))],
            (1.0 - (1.0 / 3.0)) * 100.0,
        )
    )

    assert threshold == pytest.approx(expected)
    assert threshold != pytest.approx(wrong)


def test_compute_top_fraction_threshold_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        connectivity_plots._compute_top_fraction_threshold(
            np.array([0.1, 0.2, 0.3], dtype=float),
            top_fraction=1.5,
            config_path="plotting.plots.features.connectivity.network_top_fraction",
        )


def test_plot_connectivity_heatmap_uses_configured_segment_for_topology_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features_df = pd.DataFrame(
        {
            "conn_baseline_alpha_chpair_F3-F4_wpli": [0.1, 0.2, 0.3],
            "conn_active_alpha_chpair_F3-F4_wpli": [0.7, 0.8, 0.9],
        }
    )
    info = mne.create_info(["F3", "F4"], sfreq=100.0, ch_types="eeg")
    config = {
        "plotting": {
            "comparisons": {
                "comparison_segment": "active",
            }
        }
    }
    captured: dict[str, object] = {}

    def fake_build_adjacency_from_edges(
        frame: pd.DataFrame,
        edge_cols: list[str],
        channel_order: list[str],
        edges: list[tuple[str, str]],
    ) -> np.ndarray:
        captured["edge_cols"] = list(edge_cols)
        return np.array([[0.0, 0.5], [0.5, 0.0]], dtype=float)

    monkeypatch.setattr(
        connectivity_plots,
        "build_adjacency_from_edges",
        fake_build_adjacency_from_edges,
    )
    monkeypatch.setattr(
        connectivity_plots,
        "compute_significant_edges",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(connectivity_plots, "save_fig", lambda *args, **kwargs: None)

    connectivity_plots.plot_connectivity_heatmap(
        features_df=features_df,
        info=info,
        subject="01",
        save_dir=tmp_path,
        logger=logging.getLogger("test_connectivity"),
        config=config,
        measure="wpli",
        band="alpha",
        events_df=None,
    )

    assert captured["edge_cols"] == ["conn_active_alpha_chpair_F3-F4_wpli"]


def test_plot_column_comparison_connectivity_computes_multigroup_stats_for_connectivity_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features_df = pd.DataFrame(
        {
            "conn_active_alpha_chpair_F3-F4_wpli": [
                0.10,
                0.12,
                0.14,
                0.40,
                0.42,
                0.44,
                0.70,
                0.72,
                0.74,
            ],
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0, 0, 1, 1, 1, 2, 2, 2]})
    config = {
        "plotting": {
            "comparisons": {
                "compare_columns": True,
                "comparison_column": "condition",
                "comparison_values": [0, 1, 2],
                "comparison_labels": ["Cool", "Warm", "Hot"],
                "comparison_segment": "active",
            }
        },
        "statistics": {"fdr_alpha": 0.05},
    }
    captured: dict[str, object] = {}

    def fake_multigroup_plot(**kwargs) -> None:
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "eeg_pipeline.plotting.features.utils.plot_multi_group_column_comparison",
        fake_multigroup_plot,
    )

    connectivity_plots._plot_column_comparison_connectivity(
        features_df=features_df,
        events_df=events_df,
        measures=["wpli"],
        bands=["alpha"],
        roi_names=["all"],
        roi_definitions={},
        subject="01",
        save_dir=tmp_path,
        config=config,
        logger=logging.getLogger("test_connectivity"),
        stats_dir=None,
    )

    kwargs = captured["kwargs"]
    multigroup_stats = kwargs["multigroup_stats"]
    assert multigroup_stats is not None
    assert not multigroup_stats.empty
    assert set(multigroup_stats["group1"]) == {"Cool", "Cool", "Warm"}
    assert set(multigroup_stats["group2"]) == {"Warm", "Hot", "Hot"}


def test_plot_connectivity_circle_by_condition_propagates_render_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features_df = pd.DataFrame(
        {
            "conn_active_alpha_chpair_F3-F4_wpli": [0.1, 0.2, 0.3, 0.4],
        }
    )
    events_df = pd.DataFrame({"condition": [0, 0, 1, 1]})
    info = mne.create_info(["F3", "F4"], sfreq=100.0, ch_types="eeg")
    config = {
        "plotting": {
            "comparisons": {
                "compare_columns": True,
                "comparison_column": "condition",
                "comparison_values": [0, 1],
                "comparison_labels": ["Cool", "Hot"],
                "comparison_segment": "active",
            },
            "plots": {
                "features": {
                    "connectivity": {
                        "circle_top_fraction": 0.1,
                        "circle_min_lines": 1,
                    }
                }
            },
            "overwrite": True,
        }
    }

    def raise_render_error(*args, **kwargs):
        raise RuntimeError("circle render failed")

    monkeypatch.setattr(connectivity_plots, "plot_connectivity_circle", raise_render_error)

    with pytest.raises(RuntimeError, match="circle render failed"):
        connectivity_plots.plot_connectivity_circle_by_condition(
            features_df=features_df,
            events_df=events_df,
            info=info,
            subject="01",
            save_dir=tmp_path,
            logger=logging.getLogger("test_connectivity"),
            config=config,
            measure="wpli",
            band="alpha",
        )


def test_plot_connectivity_circle_renders_rest_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    features_df = pd.DataFrame(
        {
            "conn_baseline_alpha_chpair_F3-F4_wpli": [0.1, 0.2, 0.3],
            "conn_active_alpha_chpair_F3-F4_wpli": [0.7, 0.8, 0.9],
        }
    )
    info = mne.create_info(["F3", "F4"], sfreq=100.0, ch_types="eeg")
    config = {
        "plotting": {
            "comparisons": {
                "comparison_segment": "active",
            },
            "plots": {
                "features": {
                    "connectivity": {
                        "circle_top_fraction": 1.0,
                        "circle_min_lines": 1,
                    }
                }
            },
            "overwrite": True,
        }
    }
    captured: dict[str, object] = {}

    def fake_circle(
        matrix,
        node_names,
        *,
        n_lines,
        ax,
        title,
        show,
        vmin,
        vmax,
        colorbar,
        colormap,
    ) -> None:
        captured["matrix"] = np.asarray(matrix, dtype=float)
        captured["node_names"] = list(node_names)
        captured["n_lines"] = n_lines

    def fake_save_fig(_fig, output_name, **_kwargs) -> None:
        captured["output_name"] = str(output_name)

    monkeypatch.setattr(connectivity_plots, "plot_connectivity_circle", fake_circle)
    monkeypatch.setattr(connectivity_plots, "save_fig", fake_save_fig)

    connectivity_plots.plot_connectivity_circle_summary(
        features_df=features_df,
        info=info,
        subject="01",
        save_dir=tmp_path,
        logger=logging.getLogger("test_connectivity"),
        config=config,
        measure="wpli",
        band="alpha",
    )

    assert captured["node_names"] == ["F3", "F4"]
    assert captured["n_lines"] == 1
    assert np.asarray(captured["matrix"]).shape == (2, 2)
    assert np.asarray(captured["matrix"])[0, 1] == pytest.approx(0.8)
    assert str(captured["output_name"]).endswith("sub-01_connectivity_wpli_alpha_circle")
