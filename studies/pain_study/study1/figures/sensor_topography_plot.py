"""Publication rendering for Study 1 sensor topographies."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.sensor_cluster_inference import SensorClusterResult
from studies.pain_study.study1.figures.validity_style import (
    figure_size_inches,
    publication_style,
)

FIGURE_CONFIG_PATH = "study1.figures.sensor_topographies"
CONSTRUCT_ESTIMANDS = ("temperature", "intensity")
SIGNATURE_ESTIMANDS = ("NPS", "SIIPS1")
EXPECTED_BAND_COUNT = 5
MASK_PARAMETERS = {
    "marker": "o",
    "markeredgecolor": "#222222",
    "markeredgewidth": 0.9,
    "markerfacecolor": "none",
    "markersize": 4.6,
}


@dataclass(frozen=True)
class SensorTopographyPlotMap:
    """One ordered unthresholded map and its corrected sensor overlay."""

    estimand: str
    band: str
    display_values: tuple[float, ...]
    corrected_sensors: tuple[str, ...]


@dataclass(frozen=True)
class SensorTopographyPlotSummary:
    """Immutable rendering-only boundary for one ten-map figure."""

    participant_order: tuple[str, ...]
    map_order: tuple[tuple[str, str], ...]
    sensor_order: tuple[str, ...]
    positions_xy: tuple[tuple[float, float], ...]
    maps: tuple[SensorTopographyPlotMap, ...]

    @classmethod
    def from_cluster_result(
        cls,
        result: SensorClusterResult,
        *,
        positions_xy: Sequence[Sequence[float]],
    ) -> SensorTopographyPlotSummary:
        """Adapt Task 4 results and Task 2 positions without recomputing statistics."""

        if not isinstance(result, SensorClusterResult):
            raise TypeError("Sensor topography plotting requires a SensorClusterResult.")
        try:
            positions = np.asarray(positions_xy, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Sensor topography x-y positions must be numeric.") from error
        normalized_positions = tuple(
            tuple(float(coordinate) for coordinate in position) for position in positions
        )
        maps = tuple(
            SensorTopographyPlotMap(
                estimand=map_result.estimand,
                band=map_result.band,
                display_values=map_result.cohort_values,
                corrected_sensors=map_result.significant_sensors,
            )
            for map_result in result.map_results
        )
        return cls(
            participant_order=result.participant_order,
            map_order=result.map_order,
            sensor_order=result.sensor_order,
            positions_xy=normalized_positions,
            maps=maps,
        )


def build_sensor_topography_figure(
    summary: SensorTopographyPlotSummary,
    config: Any,
) -> Figure:
    """Render fixed two-by-five unthresholded maps and corrected sensor rings."""

    figure_config = _figure_config(config)
    band_specs = _validate_summary(summary, figure_config)
    family = _figure_family(summary.map_order)
    limits = _display_limits(summary, family)
    color_map = _diverging_color_map()
    dimensions = figure_config["dimensions_mm"]

    with publication_style(config):
        figure = plt.figure(
            figsize=figure_size_inches(dimensions),
            facecolor="white",
        )
        figure.set_size_inches(*figure_size_inches(dimensions), forward=False)
        grid = figure.add_gridspec(
            2,
            EXPECTED_BAND_COUNT,
            left=0.105,
            right=0.985,
            bottom=0.21,
            top=0.79,
            hspace=0.24,
            wspace=0.10,
        )
        axes = [
            figure.add_subplot(grid[row, column])
            for row in range(2)
            for column in range(EXPECTED_BAND_COUNT)
        ]
        for map_index, (axis, plot_map) in enumerate(zip(axes, summary.maps, strict=True)):
            row = map_index // EXPECTED_BAND_COUNT
            axis.set_gid(f"topomap-{row}-{plot_map.band}")
            _draw_topomap(
                axis,
                plot_map,
                sensor_order=summary.sensor_order,
                positions_xy=summary.positions_xy,
                color_map=color_map,
                limit=limits[row],
            )
            if row == 0:
                axis.set_title(_band_title(band_specs[map_index]), pad=4.0)

        _add_labels(figure, family=family, n_participants=len(summary.participant_order))
        _add_colorbars(figure, family=family, color_map=color_map, limits=limits)
    return figure


def _draw_topomap(
    axis: mpl.axes.Axes,
    plot_map: SensorTopographyPlotMap,
    *,
    sensor_order: tuple[str, ...],
    positions_xy: tuple[tuple[float, float], ...],
    color_map: LinearSegmentedColormap,
    limit: float,
) -> None:
    corrected = set(plot_map.corrected_sensors)
    arguments: dict[str, object] = {
        "axes": axis,
        "show": False,
        "sensors": True,
        "contours": 0,
        "cmap": color_map,
        "vlim": (-limit, limit),
        "extrapolate": "local",
        "border": "mean",
        "image_interp": "cubic",
        "res": 160,
    }
    if corrected:
        arguments["mask"] = np.asarray(
            [sensor in corrected for sensor in sensor_order],
            dtype=bool,
        )
        arguments["mask_params"] = MASK_PARAMETERS
    mne.viz.plot_topomap(
        np.asarray(plot_map.display_values, dtype=float),
        np.asarray(positions_xy, dtype=float),
        **arguments,
    )


def _add_labels(figure: Figure, *, family: str, n_participants: int) -> None:
    if family == "construct":
        title = "EEG sensor power construct effects"
        row_labels = ("Delivered temperature", "Subjective intensity beyond temperature")
    else:
        title = "EEG sensor power signature associations"
        row_labels = ("NPS association", "SIIPS1 association")
    figure.text(
        0.5,
        0.965,
        f"{title} (n = {n_participants})",
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
    )
    for panel, panel_y, row_y, row_label in zip(
        ("a", "b"),
        (0.835, 0.515),
        (0.65, 0.35),
        row_labels,
        strict=True,
    ):
        figure.text(
            0.018,
            panel_y,
            panel,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )
        figure.text(
            0.057,
            row_y,
            row_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.1,
            fontweight="bold",
        )


def _add_colorbars(
    figure: Figure,
    *,
    family: str,
    color_map: LinearSegmentedColormap,
    limits: tuple[float, float],
) -> None:
    if family == "construct":
        specifications = (
            (
                "temperature",
                limits[0],
                "Mean power slope (dB/°C)",
                (0.105, 0.075, 0.38, 0.018),
            ),
            (
                "intensity",
                limits[1],
                "Fisher-mean partial correlation, r",
                (0.585, 0.075, 0.38, 0.018),
            ),
        )
    else:
        specifications = (
            (
                "signature",
                limits[0],
                "Fisher-mean partial correlation, r",
                (0.30, 0.075, 0.40, 0.018),
            ),
        )
    for name, limit, label, bounds in specifications:
        axis = figure.add_axes(bounds)
        axis.set_gid(f"colorbar-{name}")
        colorbar = figure.colorbar(
            mpl.cm.ScalarMappable(
                norm=Normalize(vmin=-limit, vmax=limit),
                cmap=color_map,
            ),
            cax=axis,
            orientation="horizontal",
            ticks=(-limit, 0.0, limit),
        )
        colorbar.set_label(label, labelpad=1.5, fontsize=5.7)
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(length=1.8, width=0.5, pad=1.2, labelsize=5.2)


def _validate_summary(
    summary: SensorTopographyPlotSummary,
    figure_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(summary, SensorTopographyPlotSummary):
        raise TypeError("Sensor topography rendering requires SensorTopographyPlotSummary.")
    _unique_names(summary.participant_order, "Participant order")
    sensors = _unique_names(summary.sensor_order, "Sensor order")
    if len(sensors) < 3:
        raise ValueError("Sensor topography rendering requires at least three sensors.")
    positions = np.asarray(summary.positions_xy, dtype=float)
    if positions.shape != (len(sensors), 2) or not np.isfinite(positions).all():
        raise ValueError("Sensor topography positions must be finite x-y pairs in sensor order.")
    if len(np.unique(positions, axis=0)) != len(sensors):
        raise ValueError("Sensor topography positions must be unique.")

    band_specs = _band_specs(figure_config)
    band_order = tuple(str(spec["name"]) for spec in band_specs)
    estimands = tuple(dict.fromkeys(estimand for estimand, _band in summary.map_order))
    expected_order = tuple((estimand, band) for estimand in estimands for band in band_order)
    _figure_family(summary.map_order)
    if summary.map_order != expected_order:
        raise ValueError("Sensor topography maps must use configured bands in row-major order.")
    observed_order = tuple((plot_map.estimand, plot_map.band) for plot_map in summary.maps)
    if observed_order != summary.map_order:
        raise ValueError("Sensor topography plot maps do not match the declared map order.")

    sensor_set = set(sensors)
    for plot_map in summary.maps:
        values = np.asarray(plot_map.display_values, dtype=float)
        if values.shape != (len(sensors),):
            raise ValueError("Each sensor topography requires one display value per sensor.")
        corrected = plot_map.corrected_sensors
        if len(corrected) != len(set(corrected)) or not set(corrected).issubset(sensor_set):
            raise ValueError("Corrected sensor masks must contain unique analyzed sensors.")
    return band_specs


def _display_limits(
    summary: SensorTopographyPlotSummary,
    family: str,
) -> tuple[float, float]:
    values = np.asarray([plot_map.display_values for plot_map in summary.maps], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sensor topography display ranges must be finite and non-zero.")
    if family == "signature":
        limit = float(np.abs(values).max())
        limits = (limit, limit)
    else:
        limits = tuple(
            float(np.abs(row_values).max()) for row_values in np.split(values, 2, axis=0)
        )
    if any(not np.isfinite(limit) or limit <= 0.0 for limit in limits):
        raise ValueError("Sensor topography display ranges must be finite and non-zero.")
    return limits


def _figure_family(map_order: Sequence[tuple[str, str]]) -> str:
    if len(map_order) != 2 * EXPECTED_BAND_COUNT:
        raise ValueError("Sensor topography rendering requires exactly ten maps.")
    estimands = tuple(dict.fromkeys(estimand for estimand, _band in map_order))
    if estimands == CONSTRUCT_ESTIMANDS:
        return "construct"
    if estimands == SIGNATURE_ESTIMANDS:
        return "signature"
    raise ValueError("Sensor topography maps must form a construct or signature family.")


def _figure_config(config: Any) -> Mapping[str, Any]:
    figure_config = require_config_value(config, FIGURE_CONFIG_PATH)
    if not isinstance(figure_config, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_PATH} must be a mapping.")
    dimensions = figure_config.get("dimensions_mm")
    if not isinstance(dimensions, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_PATH}.dimensions_mm must be a mapping.")
    dimension_values = tuple(dimensions.get(name) for name in ("width", "height"))
    if any(
        isinstance(value, bool)
        or not isinstance(value, (Integral, float))
        or not np.isfinite(float(value))
        or float(value) <= 0.0
        for value in dimension_values
    ):
        raise ValueError("Sensor topography dimensions must be finite and positive.")
    return figure_config


def _band_specs(figure_config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_specs = figure_config.get("bands")
    if not isinstance(raw_specs, list) or len(raw_specs) != EXPECTED_BAND_COUNT:
        raise ValueError("Sensor topography configuration requires exactly five bands.")
    specifications = tuple(raw_specs)
    for specification in specifications:
        if not isinstance(specification, Mapping):
            raise ValueError("Each sensor topography band specification must be a mapping.")
        if (
            not str(specification.get("name", "")).strip()
            or not str(specification.get("label", "")).strip()
        ):
            raise ValueError("Sensor topography bands require non-empty names and labels.")
        frequencies = specification.get("frequency_hz")
        if not isinstance(frequencies, list) or len(frequencies) != 2:
            raise ValueError("Sensor topography bands require two frequency limits.")
        frequency_values = np.asarray(frequencies, dtype=float)
        if (
            not np.isfinite(frequency_values).all()
            or frequency_values[0] <= 0.0
            or frequency_values[0] >= frequency_values[1]
        ):
            raise ValueError("Sensor topography band frequencies must be finite and increasing.")
    names = tuple(str(specification["name"]) for specification in specifications)
    if len(names) != len(set(names)):
        raise ValueError("Sensor topography band names must be unique.")
    return specifications


def _band_title(specification: Mapping[str, Any]) -> str:
    low, high = (float(value) for value in specification["frequency_hz"])
    return f"{specification['label']}\n{low:g}–{high:g} Hz"


def _unique_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of names.")
    names = tuple(str(value).strip() for value in values)
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{label} must contain non-empty unique names.")
    return names


def _diverging_color_map() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "study1_sensor_diverging",
        ("#0072B2", "#F7F7F7", "#D55E00"),
        N=256,
    )


__all__ = [
    "SensorTopographyPlotMap",
    "SensorTopographyPlotSummary",
    "build_sensor_topography_figure",
]
