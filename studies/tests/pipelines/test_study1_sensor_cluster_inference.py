from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from studies.pain_study.study1.figures.sensor_topography_estimands import ParticipantEffects

MAP_ORDER = tuple(
    (estimand, band)
    for estimand in ("temperature", "intensity")
    for band in ("alpha", "beta", "gamma_low_clean", "gamma_mid_clean", "gamma_high_clean")
)
SENSORS = ("A", "B", "C", "D")
POSITIONS_XY = ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.7, 0.7))


def test_build_delaunay_adjacency_preserves_order_and_triangle_edges() -> None:
    from studies.pain_study.study1.figures.sensor_cluster_inference import (
        build_delaunay_adjacency,
    )

    adjacency = build_delaunay_adjacency(
        ("A", "B", "C", "D", "E"),
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)),
    )

    expected = np.asarray(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert adjacency.dtype == np.bool_
    np.testing.assert_array_equal(adjacency, expected)


def test_build_delaunay_adjacency_rejects_invalid_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    from studies.pain_study.study1.figures import sensor_cluster_inference
    from studies.pain_study.study1.figures.sensor_cluster_inference import (
        build_delaunay_adjacency,
    )

    with pytest.raises(ValueError, match="unique top-view"):
        build_delaunay_adjacency(("A", "B", "C"), ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)))
    with pytest.raises(ValueError, match="Delaunay triangulation failed"):
        build_delaunay_adjacency(("A", "B", "C"), ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))

    class DisconnectedTriangulation:
        simplices = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=int)

    monkeypatch.setattr(
        sensor_cluster_inference,
        "Delaunay",
        lambda _positions: DisconnectedTriangulation(),
    )
    with pytest.raises(ValueError, match="connected"):
        build_delaunay_adjacency(
            tuple("ABCDEF"),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 0.0), (4.0, 0.0), (3.0, 1.0)),
        )


def test_find_sensor_clusters_separates_sign_and_sums_absolute_t() -> None:
    from studies.pain_study.study1.figures.sensor_cluster_inference import find_sensor_clusters

    adjacency = _chain_adjacency(6)
    clusters = find_sensor_clusters(
        np.asarray((3.0, 2.5, -3.5, -2.25, 0.0, 2.1)),
        threshold=2.0,
        adjacency=adjacency,
        sensor_order=tuple("ABCDEF"),
    )

    assert [
        (cluster.sign, cluster.sensors, cluster.extent, cluster.mass) for cluster in clusters
    ] == [
        ("positive", ("A", "B"), 2, 5.5),
        ("negative", ("C", "D"), 2, 5.75),
        ("positive", ("F",), 1, 2.1),
    ]


def test_exact_joint_inference_matches_synchronized_six_participant_reference() -> None:
    from studies.pain_study.study1.figures.sensor_cluster_inference import (
        compute_sensor_cluster_inference,
    )

    tensor = _effect_tensor(6, seed=17)
    tensor[:, 0, :2] += 2.2
    tensor[:, 7, 2:] -= 1.8
    effects = _participant_effects(tensor)
    cluster_forming_p = 0.20
    alpha = 0.05

    result = compute_sensor_cluster_inference(
        effects=effects,
        positions_xy=POSITIONS_XY,
        config=_config(
            cluster_forming_p=cluster_forming_p,
            family_alpha=alpha,
            max_null_draws=64,
        ),
    )

    threshold = stats.t.ppf(1.0 - cluster_forming_p / 2.0, df=5)
    assert result.positive_threshold == pytest.approx(threshold, rel=0.0, abs=0.0)
    assert result.negative_threshold == pytest.approx(-threshold, rel=0.0, abs=0.0)
    expected_maxima = _reference_null_maxima(tensor, threshold, np.asarray(result.adjacency))
    np.testing.assert_allclose(result.null_max_cluster_masses, expected_maxima)
    assert result.total_exact_patterns == 32
    assert result.sampled_null_draws == 32
    assert result.observed_included is True
    assert result.exact_enumeration is True
    assert result.sign_patterns[0] == (1, 1, 1, 1, 1, 1)

    observed_clusters = [
        cluster for map_result in result.map_results for cluster in map_result.clusters
    ]
    assert observed_clusters
    for cluster in observed_clusters:
        expected_p = np.mean(expected_maxima >= cluster.mass)
        assert cluster.corrected_p_value == pytest.approx(expected_p)
        assert cluster.significant is bool(expected_p <= alpha)

    sensor_frame = result.sensor_frame()
    cluster_frame = result.cluster_frame()
    family_frame = result.family_frame()
    assert tuple(sensor_frame.columns) == (
        "estimand",
        "band",
        "sensor",
        "sensor_index",
        "cohort_value",
        "t_statistic",
        "significant_corrected",
        "significant_cluster_ids",
    )
    assert tuple(cluster_frame.columns) == (
        "estimand",
        "band",
        "cluster_id",
        "sign",
        "sensors",
        "extent",
        "mass",
        "corrected_p_value",
        "significant",
    )
    assert family_frame.loc[0, "sampled_null_draws"] == 32
    significant_sensors = {
        sensor
        for map_result in result.map_results
        for cluster in map_result.clusters
        if cluster.corrected_p_value <= alpha
        for sensor in cluster.sensors
    }
    assert set(sensor_frame.loc[sensor_frame["significant_corrected"], "sensor"]) == (
        significant_sensors
    )


def test_sampled_joint_inference_is_unique_deterministic_and_uses_plus_one() -> None:
    from studies.pain_study.study1.figures.sensor_cluster_inference import (
        compute_sensor_cluster_inference,
    )

    tensor = _effect_tensor(15, seed=41)
    tensor[:, 0, :2] += 1.5
    effects = _participant_effects(tensor)
    config = _config(
        cluster_forming_p=0.20,
        family_alpha=0.05,
        max_null_draws=64,
        seed=91,
    )

    first = compute_sensor_cluster_inference(
        effects=effects,
        positions_xy=POSITIONS_XY,
        config=config,
    )
    second = compute_sensor_cluster_inference(
        effects=effects,
        positions_xy=POSITIONS_XY,
        config=config,
    )
    changed_seed = _config(
        cluster_forming_p=0.20,
        family_alpha=0.05,
        max_null_draws=64,
        seed=92,
    )
    third = compute_sensor_cluster_inference(
        effects=effects,
        positions_xy=POSITIONS_XY,
        config=changed_seed,
    )

    assert first.sign_patterns == second.sign_patterns
    assert first.sign_patterns != third.sign_patterns
    assert len(first.sign_patterns) == len(set(first.sign_patterns)) == 64
    assert all(pattern[0] == 1 for pattern in first.sign_patterns)
    assert (1,) * 15 not in first.sign_patterns
    assert first.total_exact_patterns == 2**14
    assert first.sampled_null_draws == 64
    assert first.observed_included is False
    assert first.exact_enumeration is False
    for cluster in (cluster for result in first.map_results for cluster in result.clusters):
        exceedances = np.count_nonzero(np.asarray(first.null_max_cluster_masses) >= cluster.mass)
        assert cluster.corrected_p_value == pytest.approx((1 + exceedances) / 65)


def test_inference_rejects_incomplete_or_semantically_wrong_effects() -> None:
    from studies.pain_study.study1.figures.sensor_cluster_inference import (
        compute_sensor_cluster_inference,
    )

    tensor = _effect_tensor(6, seed=3)
    effects = _participant_effects(tensor)
    incomplete = _replace_effect_rows(effects, effects.effects.iloc[:-1].copy())
    with pytest.raises(ValueError, match="complete tensor"):
        compute_sensor_cluster_inference(
            effects=incomplete,
            positions_xy=POSITIONS_XY,
            config=_config(),
        )

    wrong_fisher = effects.effects.copy()
    correlation = wrong_fisher["estimand"].eq("intensity")
    wrong_fisher.loc[correlation, "fisher_z"] += 0.1
    with pytest.raises(ValueError, match="Fisher-z"):
        compute_sensor_cluster_inference(
            effects=_replace_effect_rows(effects, wrong_fisher),
            positions_xy=POSITIONS_XY,
            config=_config(),
        )

    wrong_slope = effects.effects.copy()
    temperature = wrong_slope["estimand"].eq("temperature")
    wrong_slope.loc[temperature, "effect_value"] += 0.1
    with pytest.raises(ValueError, match="temperature slopes"):
        compute_sensor_cluster_inference(
            effects=_replace_effect_rows(effects, wrong_slope),
            positions_xy=POSITIONS_XY,
            config=_config(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("cluster_forming_p", True, "numeric"),
        ("cluster_forming_p", 0.0, r"\(0, 1\)"),
        ("cluster_forming_p", 1.0, r"\(0, 1\)"),
        ("family_alpha", False, "numeric"),
        ("family_alpha", np.nan, r"\(0, 1\)"),
        ("max_null_draws", True, "positive integer"),
        ("max_null_draws", 10.5, "positive integer"),
        ("seed", False, "nonnegative integer"),
        ("seed", -1, "nonnegative integer"),
    ),
)
def test_inference_settings_are_strict(field: str, value: object, message: str) -> None:
    from studies.pain_study.study1.figures.sensor_cluster_inference import (
        compute_sensor_cluster_inference,
    )

    config = _config()
    config["study1"]["figures"]["sensor_topographies"]["inference"][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        compute_sensor_cluster_inference(
            effects=_participant_effects(_effect_tensor(6, seed=5)),
            positions_xy=POSITIONS_XY,
            config=config,
        )


def test_inference_requires_family_alpha_resolution() -> None:
    from studies.pain_study.study1.figures.sensor_cluster_inference import (
        compute_sensor_cluster_inference,
    )

    with pytest.raises(ValueError, match="resolve family_alpha"):
        compute_sensor_cluster_inference(
            effects=_participant_effects(_effect_tensor(5, seed=7)),
            positions_xy=POSITIONS_XY,
            config=_config(family_alpha=0.05, max_null_draws=100),
        )

    with pytest.raises(ValueError, match="resolve family_alpha"):
        compute_sensor_cluster_inference(
            effects=_participant_effects(_effect_tensor(10, seed=9)),
            positions_xy=POSITIONS_XY,
            config=_config(family_alpha=0.01, max_null_draws=50),
        )


def _participant_effects(tensor: np.ndarray) -> ParticipantEffects:
    participants = tuple(f"sub-{index + 1:04d}" for index in range(tensor.shape[0]))
    records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for participant_index, participant in enumerate(participants):
        for map_index, (estimand, band) in enumerate(MAP_ORDER):
            for sensor_index, sensor in enumerate(SENSORS):
                inference_value = float(tensor[participant_index, map_index, sensor_index])
                is_temperature = estimand == "temperature"
                records.append(
                    {
                        "subject_id": participant,
                        "estimand": estimand,
                        "band": band,
                        "channel": sensor,
                        "effect_value": (
                            inference_value if is_temperature else float(np.tanh(inference_value))
                        ),
                        "partial_r": (
                            np.nan if is_temperature else float(np.tanh(inference_value))
                        ),
                        "fisher_z": np.nan if is_temperature else inference_value,
                        "inference_value": inference_value,
                    }
                )
    for map_index, (estimand, band) in enumerate(MAP_ORDER):
        for sensor_index, sensor in enumerate(SENSORS):
            mean_value = float(tensor[:, map_index, sensor_index].mean())
            summary_records.append(
                {
                    "estimand": estimand,
                    "band": band,
                    "channel": sensor,
                    "mean_inference_value": mean_value,
                    "display_value": (
                        mean_value if estimand == "temperature" else float(np.tanh(mean_value))
                    ),
                    "n_participants": tensor.shape[0],
                }
            )
    return ParticipantEffects(
        effects=pd.DataFrame.from_records(records),
        summary=pd.DataFrame.from_records(summary_records),
        exclusions=pd.DataFrame(),
        sensitivity_effects=None,
        sensitivity_summary=None,
        map_order=MAP_ORDER,
        sensor_order=SENSORS,
        participant_order=participants,
    )


def _replace_effect_rows(effects: ParticipantEffects, rows: pd.DataFrame) -> ParticipantEffects:
    return ParticipantEffects(
        effects=rows,
        summary=effects.summary,
        exclusions=effects.exclusions,
        sensitivity_effects=effects.sensitivity_effects,
        sensitivity_summary=effects.sensitivity_summary,
        map_order=effects.map_order,
        sensor_order=effects.sensor_order,
        participant_order=effects.participant_order,
        inference_value_column=effects.inference_value_column,
    )


def _effect_tensor(n_participants: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.7, size=(n_participants, len(MAP_ORDER), len(SENSORS)))


def _config(
    *,
    cluster_forming_p: float = 0.20,
    family_alpha: float = 0.05,
    max_null_draws: int = 64,
    seed: int = 20260715,
) -> dict[str, object]:
    return {
        "study1": {
            "figures": {
                "sensor_topographies": {
                    "inference": {
                        "cluster_forming_p": cluster_forming_p,
                        "family_alpha": family_alpha,
                        "max_null_draws": max_null_draws,
                        "seed": seed,
                    }
                }
            }
        }
    }


def _chain_adjacency(n_sensors: int) -> np.ndarray:
    adjacency = np.eye(n_sensors, dtype=bool)
    indices = np.arange(n_sensors - 1)
    adjacency[indices, indices + 1] = True
    adjacency[indices + 1, indices] = True
    return adjacency


def _reference_null_maxima(
    tensor: np.ndarray,
    threshold: float,
    adjacency: np.ndarray,
) -> np.ndarray:
    patterns = ((1, *tail) for tail in product((1, -1), repeat=tensor.shape[0] - 1))
    maxima = []
    for pattern in patterns:
        signed = tensor * np.asarray(pattern, dtype=float)[:, None, None]
        t_values = stats.ttest_1samp(signed, popmean=0.0, axis=0).statistic
        masses = [_reference_max_cluster_mass(map_t, threshold, adjacency) for map_t in t_values]
        maxima.append(max(masses))
    return np.asarray(maxima)


def _reference_max_cluster_mass(
    t_values: np.ndarray,
    threshold: float,
    adjacency: np.ndarray,
) -> float:
    masses = []
    for mask in (t_values >= threshold, t_values <= -threshold):
        remaining = set(np.flatnonzero(mask))
        while remaining:
            start = remaining.pop()
            component = {start}
            stack = [start]
            while stack:
                sensor = stack.pop()
                neighbors = set(np.flatnonzero(adjacency[sensor] & mask)) & remaining
                remaining.difference_update(neighbors)
                component.update(neighbors)
                stack.extend(neighbors)
            masses.append(float(np.abs(t_values[list(component)]).sum()))
    return max(masses, default=0.0)
