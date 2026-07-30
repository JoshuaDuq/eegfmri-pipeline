"""Group-level source-map inference for Study 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from studies.pain_study.study2.statistics import plus_one_p_value


@dataclass(frozen=True)
class SourceCluster:
    sign: str
    vertices: tuple[int, ...]
    mass: float
    p_value: float


@dataclass(frozen=True)
class GroupSourceInferenceResult:
    t_values: np.ndarray
    p_values: np.ndarray
    threshold: float
    clusters: tuple[SourceCluster, ...]
    null_max_cluster_masses: np.ndarray
    n_subjects: int
    n_permutations: int


def compute_group_source_inference(
    *,
    observed_maps: np.ndarray,
    null_maps: np.ndarray,
    adjacency: np.ndarray,
    cluster_forming_p: float,
) -> GroupSourceInferenceResult:
    """Run one-band group cluster inference from subject maps and null maps."""
    observed = np.asarray(observed_maps, dtype=float)
    null = np.asarray(null_maps, dtype=float)
    adjacency_arr = np.asarray(adjacency, dtype=bool)
    _validate_inputs(observed, null, adjacency_arr, cluster_forming_p)

    t_values, p_values = _one_sample_t(observed)
    threshold = float(stats.t.ppf(1.0 - cluster_forming_p / 2.0, observed.shape[0] - 1))
    null_max_masses = np.asarray(
        [_max_cluster_mass(_one_sample_t(draw)[0], threshold, adjacency_arr) for draw in null],
        dtype=float,
    )
    clusters = tuple(
        SourceCluster(
            sign=sign,
            vertices=vertices,
            mass=mass,
            p_value=plus_one_p_value(mass, null_max_masses),
        )
        for sign, vertices, mass in _clusters(t_values, threshold, adjacency_arr)
    )
    return GroupSourceInferenceResult(
        t_values=t_values,
        p_values=p_values,
        threshold=threshold,
        clusters=clusters,
        null_max_cluster_masses=null_max_masses,
        n_subjects=int(observed.shape[0]),
        n_permutations=int(null.shape[0]),
    )


def _validate_inputs(
    observed: np.ndarray,
    null: np.ndarray,
    adjacency: np.ndarray,
    cluster_forming_p: float,
) -> None:
    if observed.ndim != 2:
        raise ValueError(f"Study 2 observed_maps must be 2D, got shape {observed.shape}.")
    if observed.shape[0] < 2:
        raise ValueError("Study 2 group source inference requires at least two subjects.")
    if null.ndim != 3:
        raise ValueError(f"Study 2 null_maps must be 3D, got shape {null.shape}.")
    if null.shape[1:] != observed.shape:
        raise ValueError("Study 2 null_maps draws must match observed_maps shape.")
    if null.shape[0] < 1:
        raise ValueError("Study 2 group source inference requires at least one null draw.")
    if not np.all(np.isfinite(observed)):
        raise ValueError("Study 2 observed_maps contains non-finite values.")
    if not np.all(np.isfinite(null)):
        raise ValueError("Study 2 null_maps contains non-finite values.")
    if adjacency.shape != (observed.shape[1], observed.shape[1]):
        raise ValueError("Study 2 adjacency must be square with one row per vertex.")
    if adjacency.shape[0] == 0:
        raise ValueError("Study 2 adjacency must not be empty.")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Study 2 adjacency must be symmetric.")
    if isinstance(cluster_forming_p, bool):
        raise TypeError("Study 2 cluster_forming_p must be numeric.")
    p_value = float(cluster_forming_p)
    if not np.isfinite(p_value) or p_value <= 0.0 or p_value >= 1.0:
        raise ValueError("Study 2 cluster_forming_p must be in (0, 1).")


def _one_sample_t(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(maps, axis=0)
    sd = np.std(maps, axis=0, ddof=1)
    standard_error = sd / np.sqrt(maps.shape[0])
    t_values = np.zeros(maps.shape[1], dtype=float)
    nonzero = standard_error > 0.0
    t_values[nonzero] = means[nonzero] / standard_error[nonzero]
    df = maps.shape[0] - 1
    p_values = np.ones(maps.shape[1], dtype=float)
    p_values[nonzero] = 2.0 * stats.t.sf(np.abs(t_values[nonzero]), df)
    return t_values, p_values


def _max_cluster_mass(t_values: np.ndarray, threshold: float, adjacency: np.ndarray) -> float:
    clusters = _clusters(t_values, threshold, adjacency)
    if not clusters:
        return 0.0
    return float(max(mass for _sign, _vertices, mass in clusters))


def _clusters(
    t_values: np.ndarray,
    threshold: float,
    adjacency: np.ndarray,
) -> list[tuple[str, tuple[int, ...], float]]:
    positive = _connected_components(t_values >= threshold, adjacency)
    negative = _connected_components(t_values <= -threshold, adjacency)
    output: list[tuple[str, tuple[int, ...], float]] = []
    for vertices in positive:
        mass = float(np.sum(np.abs(t_values[list(vertices)])))
        output.append(("positive", vertices, mass))
    for vertices in negative:
        mass = float(np.sum(np.abs(t_values[list(vertices)])))
        output.append(("negative", vertices, mass))
    return sorted(output, key=lambda item: item[1][0])


def _connected_components(mask: np.ndarray, adjacency: np.ndarray) -> list[tuple[int, ...]]:
    remaining = set(np.flatnonzero(mask).tolist())
    components: list[tuple[int, ...]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            vertex = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[vertex] & mask).tolist())
            new_neighbors = neighbors & remaining
            remaining.difference_update(new_neighbors)
            component.update(new_neighbors)
            stack.extend(sorted(new_neighbors))
        components.append(tuple(sorted(component)))
    return sorted(components)


__all__ = [
    "GroupSourceInferenceResult",
    "SourceCluster",
    "compute_group_source_inference",
]
