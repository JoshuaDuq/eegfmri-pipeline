from __future__ import annotations

import numpy as np
import pytest


def test_compute_source_family_inference_holm_corrects_band_cluster_p_values() -> None:
    from studies.pain_study.study2.source_family import compute_source_family_inference

    observed_maps_by_band = {
        "alpha": _clustered_subject_maps(),
        "beta": np.zeros((4, 5), dtype=float),
        "gamma": np.zeros((4, 5), dtype=float),
    }
    null_maps_by_band = {band: np.zeros((99, 4, 5), dtype=float) for band in observed_maps_by_band}

    result = compute_source_family_inference(
        observed_maps_by_band=observed_maps_by_band,
        null_maps_by_band=null_maps_by_band,
        adjacency=_chain_adjacency(5),
        cluster_forming_p=0.05,
        alpha=0.05,
    )

    assert result.bands == ("alpha", "beta", "gamma")
    assert result.band_results["alpha"].min_cluster_p_value == pytest.approx(0.01)
    assert result.band_results["alpha"].holm_q_value == pytest.approx(0.03)
    assert result.band_results["alpha"].significant is True
    assert result.band_results["beta"].min_cluster_p_value == 1.0
    assert result.band_results["beta"].holm_q_value == 1.0
    assert result.band_results["beta"].significant is False
    assert result.band_results["gamma"].inference.n_subjects == 4


def test_compute_source_family_inference_requires_matching_band_keys() -> None:
    from studies.pain_study.study2.source_family import compute_source_family_inference

    with pytest.raises(ValueError, match="matching band keys"):
        compute_source_family_inference(
            observed_maps_by_band={"alpha": _clustered_subject_maps()},
            null_maps_by_band={"beta": np.zeros((3, 4, 5), dtype=float)},
            adjacency=_chain_adjacency(5),
            cluster_forming_p=0.05,
        )


def _clustered_subject_maps() -> np.ndarray:
    return np.asarray(
        [
            [2.0, 2.1, 0.0, -1.4, -1.5],
            [2.2, 2.0, 0.1, -1.5, -1.4],
            [1.9, 2.2, -0.1, -1.3, -1.6],
            [2.1, 2.3, 0.0, -1.6, -1.7],
        ],
        dtype=float,
    )


def _chain_adjacency(n_vertices: int) -> np.ndarray:
    adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)
    for vertex in range(n_vertices - 1):
        adjacency[vertex, vertex + 1] = True
        adjacency[vertex + 1, vertex] = True
    return adjacency
