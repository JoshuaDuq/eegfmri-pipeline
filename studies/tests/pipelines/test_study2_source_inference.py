from __future__ import annotations

import numpy as np
import pytest


def _chain_adjacency(n_vertices: int) -> np.ndarray:
    adjacency = np.zeros((n_vertices, n_vertices), dtype=bool)
    for vertex in range(n_vertices - 1):
        adjacency[vertex, vertex + 1] = True
        adjacency[vertex + 1, vertex] = True
    return adjacency


def test_group_source_inference_extracts_sign_preserving_clusters() -> None:
    from studies.pain_study.study2.source_inference import compute_group_source_inference

    observed_maps = np.asarray(
        [
            [2.0, 2.1, 0.0, -1.4, -1.5],
            [2.2, 2.0, 0.1, -1.5, -1.4],
            [1.9, 2.2, -0.1, -1.3, -1.6],
            [2.1, 2.3, 0.0, -1.6, -1.7],
        ],
        dtype=float,
    )
    null_maps = np.zeros((5, *observed_maps.shape), dtype=float)

    result = compute_group_source_inference(
        observed_maps=observed_maps,
        null_maps=null_maps,
        adjacency=_chain_adjacency(observed_maps.shape[1]),
        cluster_forming_p=0.05,
    )

    assert result.n_subjects == 4
    assert result.n_permutations == 5
    assert result.threshold > 0.0
    assert len(result.clusters) == 2
    assert result.clusters[0].sign == "positive"
    assert result.clusters[0].vertices == (0, 1)
    assert result.clusters[0].p_value == pytest.approx(1 / 6)
    assert result.clusters[1].sign == "negative"
    assert result.clusters[1].vertices == (3, 4)


def test_group_source_inference_rejects_invalid_adjacency_shape() -> None:
    from studies.pain_study.study2.source_inference import compute_group_source_inference

    with pytest.raises(ValueError, match="adjacency"):
        compute_group_source_inference(
            observed_maps=np.ones((4, 3), dtype=float),
            null_maps=np.zeros((2, 4, 3), dtype=float),
            adjacency=np.ones((2, 2), dtype=bool),
            cluster_forming_p=0.05,
        )
