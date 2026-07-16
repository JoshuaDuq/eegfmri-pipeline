from __future__ import annotations

from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.sensor_cluster_inference import (
    ClusterResult,
    SensorClusterResult,
    SensorMapResult,
)

BANDS = (
    "alpha",
    "beta",
    "gamma_low_clean",
    "gamma_mid_clean",
    "gamma_high_clean",
)
BAND_TITLES = (
    "Alpha\n8–12.9 Hz",
    "Beta\n13–30 Hz",
    "Low gamma\n30.1–38 Hz",
    "Mid gamma\n43–56 Hz",
    "High gamma\n67–77 Hz",
)
SENSORS = ("Fz", "Cz", "Pz", "Oz")
POSITIONS_XY = ((0.0, 0.8), (0.0, 0.2), (-0.4, -0.3), (0.4, -0.5))


def test_construct_figure_uses_ordered_maps_row_scales_and_corrected_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures.sensor_topography_plot import (
        SensorTopographyPlotSummary,
        build_sensor_topography_figure,
    )

    result = _cluster_result(
        estimands=("temperature", "intensity"),
        row_limits=(1.2, 0.45),
        significant_map_index=0,
    )
    summary = SensorTopographyPlotSummary.from_cluster_result(
        result,
        positions_xy=POSITIONS_XY,
    )
    calls = _capture_topomap_calls(monkeypatch)

    figure = build_sensor_topography_figure(summary, load_study1_config())

    assert len(calls) == 10
    assert [tuple(call["data"]) for call in calls] == [
        map_result.cohort_values for map_result in result.map_results
    ]
    assert all(np.array_equal(call["pos"], np.asarray(POSITIONS_XY)) for call in calls)
    assert [call["vlim"] for call in calls] == [(-1.2, 1.2)] * 5 + [(-0.45, 0.45)] * 5
    assert all(call["contours"] == 0 for call in calls)
    assert all(call["show"] is False for call in calls)
    assert all(call["sensors"] is True for call in calls)

    assert np.array_equal(calls[0]["mask"], [False, True, False, True])
    assert calls[0]["mask_params"] == {
        "marker": "o",
        "markeredgecolor": "#222222",
        "markeredgewidth": 0.9,
        "markerfacecolor": "none",
        "markersize": 4.6,
    }
    assert all("mask" not in call and "mask_params" not in call for call in calls[1:])

    topomap_axes = figure.axes[:10]
    assert [axis.get_gid() for axis in topomap_axes] == [
        f"topomap-{row}-{band}" for row in range(2) for band in BANDS
    ]
    assert [axis.get_title() for axis in topomap_axes[:5]] == list(BAND_TITLES)
    assert [axis.get_title() for axis in topomap_axes[5:]] == [""] * 5
    assert [axis.get_gid() for axis in figure.axes[10:]] == [
        "colorbar-temperature",
        "colorbar-intensity",
    ]
    assert [axis.get_xlabel() for axis in figure.axes[10:]] == [
        "Mean power slope (dB/°C)",
        "Fisher-mean partial correlation, r",
    ]
    assert figure.get_size_inches() == pytest.approx((183.0 / 25.4, 86.0 / 25.4))
    text = {artist.get_text() for artist in figure.texts}
    assert "EEG sensor power construct effects (n = 7)" in text
    assert "a" in text
    assert "b" in text
    assert "Delivered temperature" in text
    assert "Subjective intensity beyond temperature" in text
    assert all("Preliminary" not in value and "ready" not in value.lower() for value in text)
    plt.close(figure)


def test_signature_figure_uses_one_partial_r_scale_for_all_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures.sensor_topography_plot import (
        SensorTopographyPlotSummary,
        build_sensor_topography_figure,
    )

    result = _cluster_result(estimands=("NPS", "SIIPS1"), row_limits=(0.35, 0.8))
    summary = SensorTopographyPlotSummary.from_cluster_result(
        result,
        positions_xy=POSITIONS_XY,
    )
    calls = _capture_topomap_calls(monkeypatch)

    figure = build_sensor_topography_figure(summary, load_study1_config())

    assert [call["vlim"] for call in calls] == [(-0.8, 0.8)] * 10
    assert [axis.get_gid() for axis in figure.axes[10:]] == ["colorbar-signature"]
    assert figure.axes[10].get_xlabel() == "Fisher-mean partial correlation, r"
    text = {artist.get_text() for artist in figure.texts}
    assert "EEG sensor power signature associations (n = 7)" in text
    assert "NPS association" in text
    assert "SIIPS1 association" in text
    plt.close(figure)


def test_figure_omits_significance_mask_when_no_corrected_cluster_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures.sensor_topography_plot import (
        SensorTopographyPlotSummary,
        build_sensor_topography_figure,
    )

    result = _cluster_result(estimands=("NPS", "SIIPS1"), row_limits=(0.35, 0.8))
    summary = SensorTopographyPlotSummary.from_cluster_result(
        result,
        positions_xy=POSITIONS_XY,
    )
    calls = _capture_topomap_calls(monkeypatch)

    figure = build_sensor_topography_figure(summary, load_study1_config())

    assert all("mask" not in call and "mask_params" not in call for call in calls)
    plt.close(figure)


def test_figure_renders_single_participant_descriptive_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures.sensor_topography_plot import (
        SensorTopographyPlotSummary,
        build_sensor_topography_figure,
    )

    result = replace(
        _cluster_result(estimands=("NPS", "SIIPS1"), row_limits=(0.35, 0.8)),
        participant_order=("sub-0001",),
    )
    summary = SensorTopographyPlotSummary.from_cluster_result(
        result,
        positions_xy=POSITIONS_XY,
    )
    calls = _capture_topomap_calls(monkeypatch)

    figure = build_sensor_topography_figure(summary, load_study1_config())

    assert len(calls) == 10
    assert "EEG sensor power signature associations (n = 1)" in {
        artist.get_text() for artist in figure.texts
    }
    plt.close(figure)


@pytest.mark.parametrize("invalid_value", [0.0, float("nan"), float("inf")])
def test_figure_rejects_zero_or_nonfinite_display_range(
    invalid_value: float,
) -> None:
    from studies.pain_study.study1.figures.sensor_topography_plot import (
        SensorTopographyPlotSummary,
        build_sensor_topography_figure,
    )

    result = _cluster_result(estimands=("temperature", "intensity"), row_limits=(1.2, 0.45))
    first_map = replace(
        result.map_results[0],
        cohort_values=(invalid_value,) * len(SENSORS),
    )
    result = replace(result, map_results=(first_map, *result.map_results[1:]))
    if invalid_value == 0.0:
        result = replace(
            result,
            map_results=tuple(
                (
                    replace(map_result, cohort_values=(0.0,) * len(SENSORS))
                    if map_result.estimand == "temperature"
                    else map_result
                )
                for map_result in result.map_results
            ),
        )
    summary = SensorTopographyPlotSummary.from_cluster_result(
        result,
        positions_xy=POSITIONS_XY,
    )

    with pytest.raises(ValueError, match="finite and non-zero"):
        build_sensor_topography_figure(summary, load_study1_config())


def _capture_topomap_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def capture(data: np.ndarray, pos: np.ndarray, **kwargs: Any) -> tuple[object, object]:
        calls.append({"data": np.asarray(data), "pos": np.asarray(pos), **kwargs})
        return object(), object()

    monkeypatch.setattr("mne.viz.plot_topomap", capture)
    return calls


def _cluster_result(
    *,
    estimands: tuple[str, str],
    row_limits: tuple[float, float],
    significant_map_index: int | None = None,
) -> SensorClusterResult:
    map_order = tuple((estimand, band) for estimand in estimands for band in BANDS)
    map_results = []
    for map_index, (estimand, band) in enumerate(map_order):
        row_index = map_index // len(BANDS)
        limit = row_limits[row_index]
        values = (-0.25 * limit, 0.5 * limit, -0.75 * limit, limit)
        significant_sensors = ("Cz", "Oz") if map_index == significant_map_index else ()
        clusters = (
            (
                ClusterResult(
                    cluster_id=1,
                    sign="positive",
                    sensors=significant_sensors,
                    extent=2,
                    mass=8.0,
                    corrected_p_value=0.03,
                    significant=True,
                ),
            )
            if significant_sensors
            else ()
        )
        map_results.append(
            SensorMapResult(
                estimand=estimand,
                band=band,
                cohort_values=values,
                t_statistics=(1.0, 2.0, -1.0, -2.0),
                clusters=clusters,
                significant_sensors=significant_sensors,
            )
        )
    participants = tuple(f"sub-{index:04d}" for index in range(1, 8))
    return SensorClusterResult(
        participant_order=participants,
        map_order=map_order,
        sensor_order=SENSORS,
        adjacency=tuple(tuple(left == right for right in range(4)) for left in range(4)),
        map_results=tuple(map_results),
        null_max_cluster_masses=(1.0,),
        sign_patterns=((1,) * len(participants),),
        cluster_forming_p=0.01,
        family_alpha=0.05,
        positive_threshold=3.0,
        negative_threshold=-3.0,
        requested_max_null_draws=10_000,
        sampled_null_draws=1,
        total_exact_patterns=64,
        observed_included=True,
        exact_enumeration=True,
        seed=20_260_715,
    )
