"""Parsing, validation, and application for --plot-item-config overrides."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


def parse_bool(value: str) -> Optional[bool]:
    value_lower = str(value).strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def parse_plot_item_configs(raw: Optional[List[List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    configs: Dict[str, Dict[str, List[str]]] = {}
    if not raw:
        return configs
    for entry in raw:
        if not entry or len(entry) < 3:
            continue
        plot_id = entry[0]
        key = entry[1]
        values = entry[2:]
        configs.setdefault(plot_id, {})[key] = values
    return configs


def _apply_config_override(config: Any, path: str, value: Any) -> None:
    config[path] = value


def apply_plot_item_overrides(config: Any, overrides: Dict[str, List[str]]) -> None:
    for key, values in overrides.items():
        if key == "compare_windows" and values:
            parsed = parse_bool(values[0])
            if parsed is not None:
                _apply_config_override(config, "plotting.comparisons.compare_windows", parsed)
        elif key == "comparison_windows" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_windows", list(values))
        elif key == "compare_columns" and values:
            parsed = parse_bool(values[0])
            if parsed is not None:
                _apply_config_override(config, "plotting.comparisons.compare_columns", parsed)
        elif key == "comparison_segment" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_segment", values[0])
        elif key == "comparison_column" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_column", values[0])
        elif key == "comparison_values" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_values", list(values))
        elif key == "comparison_labels" and len(values) >= 2:
            _apply_config_override(config, "plotting.comparisons.comparison_labels", [values[0], values[1]])
        elif key == "comparison_rois" and values:
            _apply_config_override(config, "plotting.comparisons.comparison_rois", list(values))
        elif key == "temporal_stats_feature_folder" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.temporal_topomaps.stats_feature_folder",
                values[0],
            )
        elif key == "topomap_windows" and values:
            _apply_config_override(config, "plotting.plots.features.power.topomap_windows", list(values))
        elif key == "topomap_window" and values:
            _apply_config_override(config, "plotting.plots.features.power.topomap_windows", [values[0]])
        elif key == "source_segment" and values:
            _apply_config_override(config, "plotting.plots.features.sourcelocalization.segment", values[0])
        elif key == "source_subjects_dir" and values:
            _apply_config_override(
                config,
                "plotting.plots.features.sourcelocalization.subjects_dir",
                values[0],
            )
        elif key == "connectivity_circle_top_fraction" and values:
            try:
                _apply_config_override(config, "plotting.plots.features.connectivity.circle_top_fraction", float(values[0]))
            except ValueError:
                pass
        elif key == "connectivity_circle_min_lines" and values:
            try:
                _apply_config_override(config, "plotting.plots.features.connectivity.circle_min_lines", int(values[0]))
            except ValueError:
                pass
        elif key == "connectivity_network_top_fraction" and values:
            try:
                _apply_config_override(config, "plotting.plots.features.connectivity.network_top_fraction", float(values[0]))
            except ValueError:
                pass
        elif key == "itpc_shared_colorbar" and values:
            parsed = parse_bool(values[0])
            if parsed is not None:
                _apply_config_override(config, "plotting.plots.itpc.shared_colorbar", parsed)
        elif key == "tfr_topomap_active_window" and values:
            try:
                parts = values[0].split()
                if len(parts) == 2:
                    tmin = float(parts[0])
                    tmax = float(parts[1])
                    _apply_config_override(
                        config,
                        "time_frequency_analysis.active_window",
                        [tmin, tmax]
                    )
            except (ValueError, IndexError):
                pass
        elif key == "tfr_topomap_window_size_ms" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.window_size_ms",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_window_count" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.window_count",
                    int(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_label_x_position" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.label_x_position",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_label_y_position_bottom" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.label_y_position_bottom",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_label_y_position" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.label_y_position",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_title_y" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.title_y",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_title_pad" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.title_pad",
                    int(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_subplots_right" and values:
            try:
                _apply_config_override(
                    config,
                    "plotting.plots.tfr.topomap.subplots_right",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_temporal_hspace" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.single_subject.hspace",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "tfr_topomap_temporal_wspace" and values:
            try:
                _apply_config_override(
                    config,
                    "time_frequency_analysis.topomap.temporal.single_subject.wspace",
                    float(values[0])
                )
            except ValueError:
                pass
        elif key == "scatter_features" and values:
            _apply_config_override(config, "plotting.plots.behavior.scatter.features", list(values))
        elif key == "scatter_columns" and values:
            _apply_config_override(config, "plotting.plots.behavior.scatter.columns", list(values))
        elif key == "scatter_aggregation_modes" and values:
            _apply_config_override(config, "plotting.plots.behavior.scatter.aggregation_modes", list(values))
        elif key == "scatter_segment" and values:
            _apply_config_override(config, "plotting.plots.behavior.scatter.segment", values[0])
        elif key == "dose_response_dose_column" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.dose_column",
                values[0],
            )
        elif key == "dose_response_response_column" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.response_column",
                list(values),
            )
        elif key == "dose_response_binary_outcome_column" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.binary_outcome_column",
                values[0],
            )
        elif key == "dose_response_segment" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.segment",
                values[0],
            )
        elif key == "dose_response_bands" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.bands",
                list(values),
            )
        elif key == "dose_response_rois" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.rois",
                list(values),
            )
        elif key == "dose_response_scopes" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.scopes",
                list(values),
            )
        elif key == "dose_response_stat" and values:
            _apply_config_override(
                config,
                "plotting.plots.behavior.dose_response.stat",
                values[0],
            )


PLOT_ITEM_CONFIG_KEYS: Dict[str, str] = {
    "compare_windows": "plotting.comparisons.compare_windows",
    "comparison_windows": "plotting.comparisons.comparison_windows",
    "compare_columns": "plotting.comparisons.compare_columns",
    "comparison_segment": "plotting.comparisons.comparison_segment",
    "comparison_column": "plotting.comparisons.comparison_column",
    "comparison_values": "plotting.comparisons.comparison_values",
    "comparison_labels": "plotting.comparisons.comparison_labels",
    "comparison_rois": "plotting.comparisons.comparison_rois",
    "temporal_stats_feature_folder": "plotting.plots.behavior.temporal_topomaps.stats_feature_folder",
    "topomap_windows": "plotting.plots.features.power.topomap_windows",
    "topomap_window": "plotting.plots.features.power.topomap_windows",
    "source_segment": "plotting.plots.features.sourcelocalization.segment",
    "source_subjects_dir": "plotting.plots.features.sourcelocalization.subjects_dir",
    "tfr_topomap_active_window": "time_frequency_analysis.active_window",
    "tfr_topomap_window_size_ms": "time_frequency_analysis.topomap.temporal.window_size_ms",
    "tfr_topomap_window_count": "time_frequency_analysis.topomap.temporal.window_count",
    "tfr_topomap_label_x_position": "plotting.plots.tfr.topomap.label_x_position",
    "tfr_topomap_label_y_position_bottom": "plotting.plots.tfr.topomap.label_y_position_bottom",
    "tfr_topomap_label_y_position": "plotting.plots.tfr.topomap.label_y_position",
    "tfr_topomap_title_y": "plotting.plots.tfr.topomap.title_y",
    "tfr_topomap_title_pad": "plotting.plots.tfr.topomap.title_pad",
    "tfr_topomap_subplots_right": "plotting.plots.tfr.topomap.subplots_right",
    "tfr_topomap_temporal_hspace": "time_frequency_analysis.topomap.temporal.single_subject.hspace",
    "tfr_topomap_temporal_wspace": "time_frequency_analysis.topomap.temporal.single_subject.wspace",
    "connectivity_circle_top_fraction": "plotting.plots.features.connectivity.circle_top_fraction",
    "connectivity_circle_min_lines": "plotting.plots.features.connectivity.circle_min_lines",
    "connectivity_network_top_fraction": "plotting.plots.features.connectivity.network_top_fraction",
    "itpc_shared_colorbar": "plotting.plots.itpc.shared_colorbar",
    "scatter_features": "plotting.plots.behavior.scatter.features",
    "scatter_columns": "plotting.plots.behavior.scatter.columns",
    "scatter_aggregation_modes": "plotting.plots.behavior.scatter.aggregation_modes",
    "scatter_segment": "plotting.plots.behavior.scatter.segment",
    "dose_response_dose_column": "plotting.plots.behavior.dose_response.dose_column",
    "dose_response_response_column": "plotting.plots.behavior.dose_response.response_column",
    "dose_response_binary_outcome_column": "plotting.plots.behavior.dose_response.binary_outcome_column",
    "dose_response_segment": "plotting.plots.behavior.dose_response.segment",
    "dose_response_bands": "plotting.plots.behavior.dose_response.bands",
    "dose_response_rois": "plotting.plots.behavior.dose_response.rois",
    "dose_response_scopes": "plotting.plots.behavior.dose_response.scopes",
    "dose_response_stat": "plotting.plots.behavior.dose_response.stat",
}


def validate_plot_item_configs(
    configs: Dict[str, Dict[str, List[str]]],
    valid_plot_ids: Set[str],
) -> None:
    errors: List[str] = []
    for plot_id, overrides in configs.items():
        if plot_id not in valid_plot_ids:
            errors.append(f"Unknown plot_id '{plot_id}'. See --plots choices or use --all-plots.")
            continue

        for key, values in overrides.items():
            if key not in PLOT_ITEM_CONFIG_KEYS:
                allowed = ", ".join(sorted(PLOT_ITEM_CONFIG_KEYS.keys()))
                errors.append(f"Unknown key '{key}' for plot_id '{plot_id}'. Allowed keys: {allowed}.")
                continue

            if key in {"compare_windows", "compare_columns"}:
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects a boolean value (true/false).")
                    continue
                parsed = parse_bool(values[0])
                if parsed is None:
                    errors.append(
                        f"plot_id '{plot_id}': {key} expects true/false, got: {values[0]!r}."
                    )
                continue

            if key in {
                "comparison_segment",
                "comparison_column",
                "scatter_segment",
                "dose_response_dose_column",
                "dose_response_response_column",
                "dose_response_binary_outcome_column",
                "dose_response_segment",
                "dose_response_bands",
                "dose_response_rois",
                "dose_response_scopes",
                "dose_response_stat",
                "connectivity_circle_top_fraction",
                "connectivity_circle_min_lines",
                "connectivity_network_top_fraction",
                "temporal_stats_feature_folder",
                "source_segment",
                "source_subjects_dir",
            }:
                if not values or not str(values[0]).strip():
                    errors.append(f"plot_id '{plot_id}': {key} expects a non-empty value.")
                continue

            if key == "topomap_windows":
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects one or more values.")
                continue

            if key == "topomap_window":
                if not values or not str(values[0]).strip():
                    errors.append(f"plot_id '{plot_id}': {key} expects a non-empty value.")
                continue

            if key == "connectivity_circle_top_fraction":
                try:
                    val = float(values[0])
                    if not (0.0 <= val <= 1.0):
                        errors.append(f"plot_id '{plot_id}': {key} must be between 0.0 and 1.0.")
                except (ValueError, IndexError):
                    errors.append(f"plot_id '{plot_id}': {key} must be a number between 0.0 and 1.0.")
                continue

            if key == "connectivity_circle_min_lines":
                try:
                    val = int(values[0])
                    if val < 0:
                        errors.append(f"plot_id '{plot_id}': {key} must be a non-negative integer.")
                except (ValueError, IndexError):
                    errors.append(f"plot_id '{plot_id}': {key} must be a non-negative integer.")
                continue

            if key == "connectivity_network_top_fraction":
                try:
                    val = float(values[0])
                    if not (0.0 <= val <= 1.0):
                        errors.append(f"plot_id '{plot_id}': {key} must be between 0.0 and 1.0.")
                except (ValueError, IndexError):
                    errors.append(f"plot_id '{plot_id}': {key} must be a number between 0.0 and 1.0.")
                continue

            if key == "itpc_shared_colorbar":
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects a boolean value (true/false).")
                else:
                    val_str = str(values[0]).lower()
                    if val_str not in {"true", "false"}:
                        errors.append(f"plot_id '{plot_id}': {key} must be 'true' or 'false'.")
                continue

            if key in {"comparison_windows", "comparison_values", "comparison_rois",
                       "scatter_features", "scatter_columns", "scatter_aggregation_modes",
                       }:
                if not values:
                    errors.append(f"plot_id '{plot_id}': {key} expects one or more values.")

            if key == "comparison_labels":
                if len(values) != 2:
                    errors.append(f"plot_id '{plot_id}': {key} expects exactly 2 values (label1 label2).")

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Invalid --plot-item-config overrides:\n  - {joined}")
