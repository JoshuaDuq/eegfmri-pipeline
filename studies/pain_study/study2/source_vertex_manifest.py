"""Exact common-space vertex identity for Study 2 source arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


@dataclass(frozen=True)
class CommonSourceVertices:
    lh_vertices: np.ndarray
    rh_vertices: np.ndarray
    common_subject: str
    spacing: str

    @property
    def n_vertices(self) -> int:
        return int(self.lh_vertices.size + self.rh_vertices.size)

    @property
    def vertices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lh_vertices, self.rh_vertices


def ensure_common_source_vertices(
    *,
    array_path: Path,
    metadata_path: Path,
    vertices: tuple[np.ndarray, np.ndarray],
    common_subject: str,
    spacing: str,
) -> CommonSourceVertices:
    """Create the common vertex manifest once, then require exact agreement."""

    candidate = _validated_manifest(vertices, common_subject=common_subject, spacing=spacing)
    array_exists = array_path.exists()
    metadata_exists = metadata_path.exists()
    if array_exists != metadata_exists:
        raise FileNotFoundError(
            "Study 2 common source vertex manifest is incomplete; both the NPZ and JSON "
            "artifacts are required."
        )
    if array_exists:
        saved = load_common_source_vertices(
            array_path=array_path,
            metadata_path=metadata_path,
        )
        _require_matching_manifest(saved, candidate)
        return saved

    array_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _write_arrays(array_path, candidate)
    _write_metadata(metadata_path, candidate)
    return candidate


def load_common_source_vertices(
    *,
    array_path: Path,
    metadata_path: Path,
) -> CommonSourceVertices:
    """Load and validate a paired common-space vertex manifest."""

    if not array_path.is_file():
        raise FileNotFoundError(f"Study 2 source vertex array not found: {array_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Study 2 source vertex metadata not found: {metadata_path}")

    with np.load(array_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"lh_vertices", "rh_vertices"}:
            raise ValueError(
                "Study 2 source vertex NPZ must contain only lh_vertices and rh_vertices."
            )
        vertices = (arrays["lh_vertices"], arrays["rh_vertices"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "common_subject",
        "spacing",
        "n_lh_vertices",
        "n_rh_vertices",
        "n_vertices",
    }
    if set(metadata) != required:
        raise ValueError("Study 2 source vertex metadata has an invalid schema.")
    if metadata["schema_version"] != 1:
        raise ValueError("Study 2 source vertex metadata schema_version must be 1.")

    manifest = _validated_manifest(
        vertices,
        common_subject=metadata["common_subject"],
        spacing=metadata["spacing"],
    )
    expected_counts = (
        int(metadata["n_lh_vertices"]),
        int(metadata["n_rh_vertices"]),
        int(metadata["n_vertices"]),
    )
    observed_counts = (
        manifest.lh_vertices.size,
        manifest.rh_vertices.size,
        manifest.n_vertices,
    )
    if expected_counts != observed_counts:
        raise ValueError("Study 2 source vertex metadata counts do not match the NPZ arrays.")
    return manifest


def _validated_manifest(
    vertices: tuple[np.ndarray, np.ndarray],
    *,
    common_subject: object,
    spacing: object,
) -> CommonSourceVertices:
    if not isinstance(vertices, (tuple, list)) or len(vertices) != 2:
        raise ValueError("Study 2 common source vertices must contain left and right arrays.")
    lh_vertices = _validated_vertices(vertices[0], hemisphere="left")
    rh_vertices = _validated_vertices(vertices[1], hemisphere="right")
    subject = _non_empty_string(common_subject, name="common subject")
    spacing_name = _non_empty_string(spacing, name="source spacing")
    return CommonSourceVertices(
        lh_vertices=lh_vertices,
        rh_vertices=rh_vertices,
        common_subject=subject,
        spacing=spacing_name,
    )


def _validated_vertices(values: np.ndarray, *, hemisphere: str) -> np.ndarray:
    vertices = np.asarray(values)
    if vertices.ndim != 1 or vertices.size == 0:
        raise ValueError(f"Study 2 {hemisphere} vertices must be a non-empty 1D array.")
    if not np.issubdtype(vertices.dtype, np.integer):
        raise ValueError(f"Study 2 {hemisphere} vertices must be integers.")
    vertices = vertices.astype(np.int64, copy=True)
    if np.any(vertices < 0):
        raise ValueError(f"Study 2 {hemisphere} vertices must be non-negative.")
    if np.unique(vertices).size != vertices.size:
        raise ValueError(f"Study 2 {hemisphere} vertices must be unique.")
    vertices.setflags(write=False)
    return vertices


def _non_empty_string(value: object, *, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Study 2 {name} must be a non-empty string.")
    return text


def _require_matching_manifest(
    saved: CommonSourceVertices,
    candidate: CommonSourceVertices,
) -> None:
    matches = (
        saved.common_subject == candidate.common_subject
        and saved.spacing == candidate.spacing
        and np.array_equal(saved.lh_vertices, candidate.lh_vertices)
        and np.array_equal(saved.rh_vertices, candidate.rh_vertices)
    )
    if not matches:
        raise ValueError(
            "Study 2 morphed source vertex identity does not match the saved common manifest."
        )


def _write_arrays(path: Path, manifest: CommonSourceVertices) -> None:
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        np.savez(
            handle,
            lh_vertices=manifest.lh_vertices,
            rh_vertices=manifest.rh_vertices,
        )
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_metadata(path: Path, manifest: CommonSourceVertices) -> None:
    payload = {
        "schema_version": 1,
        "common_subject": manifest.common_subject,
        "spacing": manifest.spacing,
        "n_lh_vertices": int(manifest.lh_vertices.size),
        "n_rh_vertices": int(manifest.rh_vertices.size),
        "n_vertices": manifest.n_vertices,
    }
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".json",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "CommonSourceVertices",
    "ensure_common_source_vertices",
    "load_common_source_vertices",
]
