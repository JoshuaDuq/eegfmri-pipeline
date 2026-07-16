"""Joint sensor-cluster inference for Study 1 topographies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import Delaunay, QhullError

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.sensor_topography_estimands import ParticipantEffects

INFERENCE_CONFIG_PATH = "study1.figures.sensor_topographies.inference"
INFERENCE_SETTING_NAMES = frozenset({"cluster_forming_p", "family_alpha", "max_null_draws", "seed"})
EXPECTED_MAP_COUNT = 10


@dataclass(frozen=True)
class ClusterResult:
    """One sign-specific connected sensor cluster."""

    cluster_id: int
    sign: str
    sensors: tuple[str, ...]
    extent: int
    mass: float
    corrected_p_value: float = 1.0
    significant: bool = False


@dataclass(frozen=True)
class SensorMapResult:
    """Observed statistics and corrected clusters for one ordered map."""

    estimand: str
    band: str
    cohort_values: tuple[float, ...]
    t_statistics: tuple[float, ...]
    clusters: tuple[ClusterResult, ...]
    significant_sensors: tuple[str, ...]


@dataclass(frozen=True)
class SensorClusterResult:
    """Immutable joint-family inference result and audit contract."""

    participant_order: tuple[str, ...]
    map_order: tuple[tuple[str, str], ...]
    sensor_order: tuple[str, ...]
    adjacency: tuple[tuple[bool, ...], ...]
    map_results: tuple[SensorMapResult, ...]
    null_max_cluster_masses: tuple[float, ...]
    sign_patterns: tuple[tuple[int, ...], ...]
    cluster_forming_p: float
    family_alpha: float
    positive_threshold: float
    negative_threshold: float
    requested_max_null_draws: int
    sampled_null_draws: int
    total_exact_patterns: int
    observed_included: bool
    exact_enumeration: bool
    seed: int
    inference_available: bool = True
    inference_reason: str = ""

    def sensor_frame(self) -> pd.DataFrame:
        """Return one tidy observed-result row per map and sensor."""

        records: list[dict[str, object]] = []
        for map_result in self.map_results:
            for sensor_index, sensor in enumerate(self.sensor_order):
                cluster_ids = tuple(
                    cluster.cluster_id
                    for cluster in map_result.clusters
                    if cluster.significant and sensor in cluster.sensors
                )
                records.append(
                    {
                        "estimand": map_result.estimand,
                        "band": map_result.band,
                        "sensor": sensor,
                        "sensor_index": sensor_index,
                        "cohort_value": map_result.cohort_values[sensor_index],
                        "t_statistic": map_result.t_statistics[sensor_index],
                        "significant_corrected": bool(cluster_ids),
                        "significant_cluster_ids": ",".join(map(str, cluster_ids)),
                    }
                )
        return pd.DataFrame.from_records(records, columns=_sensor_frame_columns())

    def cluster_frame(self) -> pd.DataFrame:
        """Return one tidy audit row per observed cluster."""

        records = [
            {
                "estimand": map_result.estimand,
                "band": map_result.band,
                "cluster_id": cluster.cluster_id,
                "sign": cluster.sign,
                "sensors": ",".join(cluster.sensors),
                "extent": cluster.extent,
                "mass": cluster.mass,
                "corrected_p_value": cluster.corrected_p_value,
                "significant": cluster.significant,
            }
            for map_result in self.map_results
            for cluster in map_result.clusters
        ]
        return pd.DataFrame.from_records(records, columns=_cluster_frame_columns())

    def family_frame(self) -> pd.DataFrame:
        """Return the single-row joint-family permutation audit."""

        return pd.DataFrame.from_records(
            [
                {
                    "participant_order": ",".join(self.participant_order),
                    "n_participants": len(self.participant_order),
                    "n_maps": len(self.map_order),
                    "n_sensors": len(self.sensor_order),
                    "degrees_of_freedom": len(self.participant_order) - 1,
                    "inference_available": self.inference_available,
                    "inference_reason": self.inference_reason,
                    "cluster_forming_p": self.cluster_forming_p,
                    "positive_threshold": self.positive_threshold,
                    "negative_threshold": self.negative_threshold,
                    "family_alpha": self.family_alpha,
                    "requested_max_null_draws": self.requested_max_null_draws,
                    "sampled_null_draws": self.sampled_null_draws,
                    "total_exact_patterns": self.total_exact_patterns,
                    "observed_included": self.observed_included,
                    "exact_enumeration": self.exact_enumeration,
                    "seed": self.seed,
                    "permutation_method": "synchronized_participant_sign_flip",
                    "correction_method": "joint_max_cluster_mass",
                    "cluster_mass": "sum_absolute_t",
                }
            ]
        )


@dataclass(frozen=True)
class _InferenceSettings:
    cluster_forming_p: float
    family_alpha: float
    max_null_draws: int
    seed: int


@dataclass(frozen=True)
class _SignPatternSelection:
    patterns: tuple[tuple[int, ...], ...]
    total_exact_patterns: int
    observed_included: bool
    exact_enumeration: bool


def build_delaunay_adjacency(
    sensor_order: Sequence[str],
    positions_xy: Sequence[Sequence[float]],
) -> np.ndarray:
    """Build exact triangle-edge adjacency from ordered top-view sensor positions."""

    sensors = _validated_names(sensor_order, "Sensor order")
    if len(sensors) < 3:
        raise ValueError("Delaunay sensor adjacency requires at least three sensors.")
    try:
        positions = np.asarray(positions_xy, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Sensor top-view x-y positions must be numeric.") from error
    if positions.shape != (len(sensors), 2):
        raise ValueError(
            "Sensor top-view x-y positions must have shape "
            f"({len(sensors)}, 2), got {positions.shape}."
        )
    if not np.isfinite(positions).all():
        raise ValueError("Sensor top-view x-y positions must be finite.")
    if len(np.unique(positions, axis=0)) != len(sensors):
        raise ValueError("Sensors require unique top-view x-y positions.")
    try:
        simplices = Delaunay(positions).simplices
    except QhullError as error:
        raise ValueError("Delaunay triangulation failed for sensor top-view positions.") from error
    adjacency = _adjacency_from_simplices(simplices, len(sensors))
    if not _is_connected(adjacency):
        raise ValueError("Delaunay sensor adjacency must form one connected graph.")
    return adjacency


def find_sensor_clusters(
    t_statistics: Sequence[float],
    *,
    threshold: float,
    adjacency: np.ndarray,
    sensor_order: Sequence[str],
) -> tuple[ClusterResult, ...]:
    """Find positive and negative thresholded components in sensor order."""

    sensors = _validated_names(sensor_order, "Sensor order")
    t_values = np.asarray(t_statistics, dtype=float)
    if t_values.shape != (len(sensors),) or not np.isfinite(t_values).all():
        raise ValueError("Sensor t-statistics must be one finite value per ordered sensor.")
    cluster_threshold = _open_probability_complement(threshold, "Cluster t threshold")
    graph = _validated_adjacency(adjacency, len(sensors))
    specifications = _cluster_specifications(t_values, cluster_threshold, graph)
    return tuple(
        ClusterResult(
            cluster_id=cluster_id,
            sign=sign,
            sensors=tuple(sensors[index] for index in indices),
            extent=len(indices),
            mass=mass,
        )
        for cluster_id, (sign, indices, mass) in enumerate(specifications, start=1)
    )


def compute_sensor_cluster_inference(
    *,
    effects: ParticipantEffects,
    positions_xy: Sequence[Sequence[float]],
    config: Any,
) -> SensorClusterResult:
    """Run synchronized ten-map sensor-cluster inference."""

    tensor, cohort_values = _validated_effect_tensor(effects)
    settings = _inference_settings(config)
    adjacency = build_delaunay_adjacency(effects.sensor_order, positions_xy)
    if tensor.shape[0] == 1:
        return _descriptive_sensor_result(
            effects=effects,
            cohort_values=cohort_values,
            adjacency=adjacency,
            settings=settings,
        )
    selection = _select_sign_patterns(
        n_participants=tensor.shape[0],
        settings=settings,
    )
    degrees_of_freedom = tensor.shape[0] - 1
    positive_threshold = float(
        stats.t.ppf(1.0 - settings.cluster_forming_p / 2.0, degrees_of_freedom)
    )
    negative_threshold = -positive_threshold

    observed_t = _one_sample_t(tensor)
    null_maxima = tuple(
        _joint_max_cluster_mass(
            tensor * np.asarray(pattern, dtype=float)[:, None, None],
            positive_threshold,
            adjacency,
        )
        for pattern in selection.patterns
    )
    map_results = _map_results(
        effects=effects,
        cohort_values=cohort_values,
        observed_t=observed_t,
        threshold=positive_threshold,
        adjacency=adjacency,
        null_maxima=null_maxima,
        settings=settings,
        exact_enumeration=selection.exact_enumeration,
    )
    return SensorClusterResult(
        participant_order=effects.participant_order,
        map_order=effects.map_order,
        sensor_order=effects.sensor_order,
        adjacency=tuple(tuple(bool(value) for value in row) for row in adjacency),
        map_results=map_results,
        null_max_cluster_masses=null_maxima,
        sign_patterns=selection.patterns,
        cluster_forming_p=settings.cluster_forming_p,
        family_alpha=settings.family_alpha,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        requested_max_null_draws=settings.max_null_draws,
        sampled_null_draws=len(selection.patterns),
        total_exact_patterns=selection.total_exact_patterns,
        observed_included=selection.observed_included,
        exact_enumeration=selection.exact_enumeration,
        seed=settings.seed,
    )


def _validated_effect_tensor(effects: ParticipantEffects) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(effects, ParticipantEffects):
        raise TypeError("Sensor-cluster inference requires ParticipantEffects.")
    participants = _validated_names(effects.participant_order, "Participant order")
    sensors = _validated_names(effects.sensor_order, "Sensor order")
    map_order = _validated_map_order(effects.map_order)
    if len(sensors) < 3:
        raise ValueError("Sensor-cluster inference requires at least three sensors.")
    if effects.inference_value_column != "inference_value":
        raise ValueError("ParticipantEffects must use the Task 3 inference_value column.")

    required = {
        "subject_id",
        "estimand",
        "band",
        "channel",
        "effect_value",
        "fisher_z",
        "inference_value",
    }
    missing = sorted(required.difference(effects.effects.columns))
    if missing:
        raise ValueError(f"Participant effects are missing inference columns: {missing}.")
    expected_rows = len(participants) * len(map_order) * len(sensors)
    if len(effects.effects) != expected_rows:
        raise ValueError("Participant effects do not form the declared complete tensor.")
    tensor = effects.inference_tensor()
    expected_shape = (len(participants), len(map_order), len(sensors))
    if tensor.shape != expected_shape or not np.isfinite(tensor).all():
        raise ValueError("Participant effects do not form a finite complete tensor.")
    _validate_inference_values(effects.effects)
    cohort_values = _validated_summary(effects, map_order, sensors, len(participants))
    return tensor, cohort_values


def _validated_map_order(
    map_order: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(map_order, (str, bytes)):
        raise ValueError("Map order must contain ten ordered estimand-band pairs.")
    normalized: list[tuple[str, str]] = []
    for item in map_order:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Map order must contain ten ordered estimand-band pairs.")
        estimand, band = (str(value).strip() for value in item)
        if not estimand or not band:
            raise ValueError("Map order estimands and bands must be non-empty names.")
        normalized.append((estimand, band))
    ordered = tuple(normalized)
    if len(ordered) != EXPECTED_MAP_COUNT or len(set(ordered)) != len(ordered):
        raise ValueError("Sensor-cluster inference requires ten unique ordered maps.")
    estimands = tuple(dict.fromkeys(estimand for estimand, _band in ordered))
    if estimands not in (("temperature", "intensity"), ("NPS", "SIIPS1")):
        raise ValueError("Map order must be a Task 3 construct or signature family.")
    band_orders = tuple(
        tuple(band for estimand, band in ordered if estimand == expected) for expected in estimands
    )
    if len(band_orders[0]) != 5 or band_orders[0] != band_orders[1]:
        raise ValueError("Both estimands must use the same five ordered bands.")
    return ordered


def _validate_inference_values(effects: pd.DataFrame) -> None:
    inference_values = effects["inference_value"].to_numpy(dtype=float)
    temperature = effects["estimand"].eq("temperature").to_numpy()
    if temperature.any():
        slopes = effects.loc[temperature, "effect_value"].to_numpy(dtype=float)
        if not np.isfinite(slopes).all() or not np.array_equal(
            inference_values[temperature], slopes
        ):
            raise ValueError("Temperature inference values must be Task 3 temperature slopes.")
    correlation = ~temperature
    fisher_z = effects.loc[correlation, "fisher_z"].to_numpy(dtype=float)
    if not np.isfinite(fisher_z).all() or not np.array_equal(
        inference_values[correlation], fisher_z
    ):
        raise ValueError("Correlation inference values must be Task 3 Fisher-z values.")


def _validated_summary(
    effects: ParticipantEffects,
    map_order: tuple[tuple[str, str], ...],
    sensors: tuple[str, ...],
    n_participants: int,
) -> np.ndarray:
    required = {
        "estimand",
        "band",
        "channel",
        "mean_inference_value",
        "display_value",
        "n_participants",
    }
    missing = sorted(required.difference(effects.summary.columns))
    if missing:
        raise ValueError(f"Participant-effects summary is missing columns: {missing}.")
    keys = [(estimand, band, sensor) for estimand, band in map_order for sensor in sensors]
    index = effects.summary.set_index(["estimand", "band", "channel"])
    if index.index.has_duplicates or len(index) != len(keys):
        raise ValueError("Participant-effects summary requires exact unique map-sensor rows.")
    ordered = index.reindex(pd.MultiIndex.from_tuples(keys, names=index.index.names))
    if ordered.isna().any().any():
        raise ValueError("Participant-effects summary does not match the declared map order.")
    counts = ordered["n_participants"].to_numpy()
    if not np.equal(counts, n_participants).all():
        raise ValueError("Participant-effects summary participant counts disagree with the tensor.")
    mean_values = ordered["mean_inference_value"].to_numpy(dtype=float)
    display_values = ordered["display_value"].to_numpy(dtype=float)
    expected_display = np.asarray(
        [
            mean_value if estimand == "temperature" else np.tanh(mean_value)
            for (estimand, _band, _sensor), mean_value in zip(keys, mean_values)
        ]
    )
    if not np.isfinite(display_values).all() or not np.allclose(
        display_values, expected_display, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Participant-effects summary violates the Task 3 display-value contract.")
    return display_values.reshape(len(map_order), len(sensors))


def _inference_settings(config: Any) -> _InferenceSettings:
    raw = require_config_value(config, INFERENCE_CONFIG_PATH)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{INFERENCE_CONFIG_PATH} must be a mapping.")
    if set(raw) != INFERENCE_SETTING_NAMES:
        raise ValueError(
            f"{INFERENCE_CONFIG_PATH} requires exactly {sorted(INFERENCE_SETTING_NAMES)}."
        )
    return _InferenceSettings(
        cluster_forming_p=_probability(raw["cluster_forming_p"], "cluster_forming_p"),
        family_alpha=_probability(raw["family_alpha"], "family_alpha"),
        max_null_draws=_positive_integer(raw["max_null_draws"], "max_null_draws"),
        seed=_nonnegative_integer(raw["seed"], "seed"),
    )


def _select_sign_patterns(
    *,
    n_participants: int,
    settings: _InferenceSettings,
) -> _SignPatternSelection:
    total_exact_patterns = 2 ** (n_participants - 1)
    exact_enumeration = total_exact_patterns <= settings.max_null_draws
    if exact_enumeration:
        patterns = tuple((1, *tail) for tail in product((1, -1), repeat=n_participants - 1))
        return _SignPatternSelection(
            patterns=patterns,
            total_exact_patterns=total_exact_patterns,
            observed_included=True,
            exact_enumeration=True,
        )
    return _SignPatternSelection(
        patterns=_sample_sign_patterns(
            n_participants=n_participants,
            n_patterns=settings.max_null_draws,
            seed=settings.seed,
        ),
        total_exact_patterns=total_exact_patterns,
        observed_included=False,
        exact_enumeration=False,
    )


def _descriptive_sensor_result(
    *,
    effects: ParticipantEffects,
    cohort_values: np.ndarray,
    adjacency: np.ndarray,
    settings: _InferenceSettings,
) -> SensorClusterResult:
    map_results = tuple(
        SensorMapResult(
            estimand=estimand,
            band=band,
            cohort_values=tuple(float(value) for value in cohort_values[index]),
            t_statistics=tuple(float("nan") for _sensor in effects.sensor_order),
            clusters=(),
            significant_sensors=(),
        )
        for index, (estimand, band) in enumerate(effects.map_order)
    )
    return SensorClusterResult(
        participant_order=effects.participant_order,
        map_order=effects.map_order,
        sensor_order=effects.sensor_order,
        adjacency=tuple(tuple(bool(value) for value in row) for row in adjacency),
        map_results=map_results,
        null_max_cluster_masses=(),
        sign_patterns=(),
        cluster_forming_p=settings.cluster_forming_p,
        family_alpha=settings.family_alpha,
        positive_threshold=float("nan"),
        negative_threshold=float("nan"),
        requested_max_null_draws=settings.max_null_draws,
        sampled_null_draws=0,
        total_exact_patterns=1,
        observed_included=False,
        exact_enumeration=False,
        seed=settings.seed,
        inference_available=False,
        inference_reason="At least two participants are required for one-sample inference.",
    )


def _sample_sign_patterns(
    *,
    n_participants: int,
    n_patterns: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = np.random.default_rng(seed)
    observed = (1,) * n_participants
    selected: list[tuple[int, ...]] = []
    selected_set: set[tuple[int, ...]] = set()
    while len(selected) < n_patterns:
        remaining = n_patterns - len(selected)
        draws = rng.integers(0, 2, size=(max(32, 2 * remaining), n_participants - 1))
        for draw in draws:
            tail = tuple(1 if bit == 0 else -1 for bit in draw)
            pattern = (1, *tail)
            if pattern == observed or pattern in selected_set:
                continue
            selected.append(pattern)
            selected_set.add(pattern)
            if len(selected) == n_patterns:
                break
    return tuple(selected)


def _one_sample_t(tensor: np.ndarray) -> np.ndarray:
    t_statistics = np.asarray(
        stats.ttest_1samp(tensor, popmean=0.0, axis=0).statistic,
        dtype=float,
    )
    if not np.isfinite(t_statistics).all():
        raise ValueError("One-sample t-statistics require positive finite standard errors.")
    return t_statistics


def _joint_max_cluster_mass(
    signed_tensor: np.ndarray,
    threshold: float,
    adjacency: np.ndarray,
) -> float:
    t_statistics = _one_sample_t(signed_tensor)
    return max(
        (
            mass
            for map_t in t_statistics
            for _sign, _indices, mass in _cluster_specifications(map_t, threshold, adjacency)
        ),
        default=0.0,
    )


def _map_results(
    *,
    effects: ParticipantEffects,
    cohort_values: np.ndarray,
    observed_t: np.ndarray,
    threshold: float,
    adjacency: np.ndarray,
    null_maxima: tuple[float, ...],
    settings: _InferenceSettings,
    exact_enumeration: bool,
) -> tuple[SensorMapResult, ...]:
    null_values = np.asarray(null_maxima)
    results = []
    for map_index, (estimand, band) in enumerate(effects.map_order):
        uncorrected = find_sensor_clusters(
            observed_t[map_index],
            threshold=threshold,
            adjacency=adjacency,
            sensor_order=effects.sensor_order,
        )
        corrected = tuple(
            _correct_cluster(
                cluster,
                null_values=null_values,
                family_alpha=settings.family_alpha,
                exact_enumeration=exact_enumeration,
            )
            for cluster in uncorrected
        )
        significant = {
            sensor for cluster in corrected if cluster.significant for sensor in cluster.sensors
        }
        results.append(
            SensorMapResult(
                estimand=estimand,
                band=band,
                cohort_values=tuple(float(value) for value in cohort_values[map_index]),
                t_statistics=tuple(float(value) for value in observed_t[map_index]),
                clusters=corrected,
                significant_sensors=tuple(
                    sensor for sensor in effects.sensor_order if sensor in significant
                ),
            )
        )
    return tuple(results)


def _correct_cluster(
    cluster: ClusterResult,
    *,
    null_values: np.ndarray,
    family_alpha: float,
    exact_enumeration: bool,
) -> ClusterResult:
    exceedances = int(np.count_nonzero(null_values >= cluster.mass))
    if exact_enumeration:
        corrected_p_value = exceedances / len(null_values)
    else:
        corrected_p_value = (1 + exceedances) / (len(null_values) + 1)
    return ClusterResult(
        cluster_id=cluster.cluster_id,
        sign=cluster.sign,
        sensors=cluster.sensors,
        extent=cluster.extent,
        mass=cluster.mass,
        corrected_p_value=corrected_p_value,
        significant=corrected_p_value <= family_alpha,
    )


def _cluster_specifications(
    t_statistics: np.ndarray,
    threshold: float,
    adjacency: np.ndarray,
) -> tuple[tuple[str, tuple[int, ...], float], ...]:
    specifications = []
    for sign, mask in (
        ("positive", t_statistics >= threshold),
        ("negative", t_statistics <= -threshold),
    ):
        for indices in _connected_components(mask, adjacency):
            mass = float(np.abs(t_statistics[list(indices)]).sum())
            specifications.append((sign, indices, mass))
    return tuple(sorted(specifications, key=lambda item: item[1][0]))


def _connected_components(
    mask: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    remaining = set(np.flatnonzero(mask).tolist())
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        stack = [start]
        while stack:
            sensor = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[sensor] & mask).tolist()) & remaining
            remaining.difference_update(neighbors)
            component.update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(components)


def _adjacency_from_simplices(simplices: np.ndarray, n_sensors: int) -> np.ndarray:
    triangles = np.asarray(simplices)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] < 1:
        raise ValueError("Delaunay triangulation must contain triangular simplices.")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("Delaunay triangle indices must be integers.")
    if np.any(triangles < 0) or np.any(triangles >= n_sensors):
        raise ValueError("Delaunay triangle indices are outside the sensor order.")
    adjacency = np.eye(n_sensors, dtype=bool)
    for triangle in triangles:
        for left, right in combinations(triangle.tolist(), 2):
            adjacency[left, right] = True
            adjacency[right, left] = True
    return adjacency


def _validated_adjacency(adjacency: np.ndarray, n_sensors: int) -> np.ndarray:
    graph = np.asarray(adjacency, dtype=bool)
    if graph.shape != (n_sensors, n_sensors):
        raise ValueError("Sensor adjacency must be square with one row per ordered sensor.")
    if not np.array_equal(graph, graph.T):
        raise ValueError("Sensor adjacency must be undirected and symmetric.")
    if not np.diag(graph).all():
        raise ValueError("Sensor adjacency must include a true diagonal.")
    if not _is_connected(graph):
        raise ValueError("Sensor adjacency must form one connected graph.")
    return graph


def _is_connected(adjacency: np.ndarray) -> bool:
    visited = {0}
    stack = [0]
    while stack:
        sensor = stack.pop()
        neighbors = set(np.flatnonzero(adjacency[sensor]).tolist()) - visited
        visited.update(neighbors)
        stack.extend(neighbors)
    return len(visited) == adjacency.shape[0]


def _validated_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of names.")
    names = tuple(str(value).strip() for value in values)
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{label} must contain non-empty unique names.")
    return names


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric.")
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"{label} must be in (0, 1).")
    return probability


def _open_probability_complement(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be numeric.")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"{label} must be finite and positive.")
    return threshold


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer.")
    return int(value)


def _sensor_frame_columns() -> tuple[str, ...]:
    return (
        "estimand",
        "band",
        "sensor",
        "sensor_index",
        "cohort_value",
        "t_statistic",
        "significant_corrected",
        "significant_cluster_ids",
    )


def _cluster_frame_columns() -> tuple[str, ...]:
    return (
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


__all__ = [
    "ClusterResult",
    "SensorClusterResult",
    "SensorMapResult",
    "build_delaunay_adjacency",
    "compute_sensor_cluster_inference",
    "find_sensor_clusters",
]
