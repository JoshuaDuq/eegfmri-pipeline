from __future__ import annotations

import ast
from pathlib import Path

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
