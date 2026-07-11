from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_common_source_vertex_manifest_round_trips_exact_hemisphere_vertices(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study2.source_vertex_manifest import (
        ensure_common_source_vertices,
        load_common_source_vertices,
    )

    array_path = tmp_path / "common_source_vertices.npz"
    metadata_path = tmp_path / "common_source_vertices.json"
    expected = ensure_common_source_vertices(
        array_path=array_path,
        metadata_path=metadata_path,
        vertices=(np.array([2, 5]), np.array([1, 4, 8])),
        common_subject="fsaverage",
        spacing="oct6",
    )
    loaded = load_common_source_vertices(
        array_path=array_path,
        metadata_path=metadata_path,
    )

    np.testing.assert_array_equal(expected.lh_vertices, [2, 5])
    np.testing.assert_array_equal(expected.rh_vertices, [1, 4, 8])
    np.testing.assert_array_equal(loaded.lh_vertices, expected.lh_vertices)
    np.testing.assert_array_equal(loaded.rh_vertices, expected.rh_vertices)
    assert loaded.common_subject == "fsaverage"
    assert loaded.spacing == "oct6"
    assert loaded.n_vertices == 5
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["n_vertices"] == 5


def test_common_source_vertex_manifest_rejects_incompatible_subsequent_vertices(
    tmp_path: Path,
) -> None:
    from studies.pain_study.study2.source_vertex_manifest import ensure_common_source_vertices

    array_path = tmp_path / "common_source_vertices.npz"
    metadata_path = tmp_path / "common_source_vertices.json"
    arguments = {
        "array_path": array_path,
        "metadata_path": metadata_path,
        "vertices": (np.array([2, 5]), np.array([1, 4, 8])),
        "common_subject": "fsaverage",
        "spacing": "oct6",
    }
    ensure_common_source_vertices(**arguments)

    with pytest.raises(ValueError, match="does not match"):
        ensure_common_source_vertices(
            **{**arguments, "vertices": (np.array([2, 6]), np.array([1, 4, 8]))}
        )


@pytest.mark.parametrize(
    "vertices",
    [
        (np.array([], dtype=int), np.array([1], dtype=int)),
        (np.array([2, 2], dtype=int), np.array([1], dtype=int)),
        (np.array([-1], dtype=int), np.array([1], dtype=int)),
    ],
)
def test_common_source_vertex_manifest_rejects_invalid_vertices(
    tmp_path: Path,
    vertices: tuple[np.ndarray, np.ndarray],
) -> None:
    from studies.pain_study.study2.source_vertex_manifest import ensure_common_source_vertices

    with pytest.raises(ValueError, match="vertices"):
        ensure_common_source_vertices(
            array_path=tmp_path / "vertices.npz",
            metadata_path=tmp_path / "vertices.json",
            vertices=vertices,
            common_subject="fsaverage",
            spacing="oct6",
        )
