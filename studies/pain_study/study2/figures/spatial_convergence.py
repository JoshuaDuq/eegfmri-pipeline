"""Strict artifact reader for the Study 2 spatial-convergence figure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from studies.pain_study.study2 import paths
from studies.pain_study.study2.source_stage_design import contribution_bands
from studies.pain_study.study2.source_vertex_manifest import (
    CommonSourceVertices,
    load_common_source_vertices,
)
from studies.pain_study.study2.spatial_comparison import (
    SpatialCorrespondenceResult,
    compute_spatial_correspondence,
)
from studies.pain_study.study2.statistics import holm_adjusted_p_values
from studies.pain_study.study2.table_io import parse_bool
from studies.pain_study.study2.validation import (
    require_config_float,
    require_config_int,
    require_config_string,
)

SPATIAL_BANDS = ("alpha", "beta", "gamma")
SUMMARY_COLUMNS = (
    "band",
    "spatial_r",
    "p_value",
    "meaningful",
    "holm_adjusted_p_value",
    "holm_significant",
)
AUDIT_COLUMNS = (*SUMMARY_COLUMNS, "null_ci_low", "null_ci_high")
_METADATA_KEYS = {
    "schema_version",
    "method",
    "n_surrogates",
    "common_subject",
    "common_source_space_spacing",
    "source_vertex_manifest_sha256",
    "analysis_mask_sha256",
    "bands",
}
_BAND_METADATA_KEYS = {
    "seed",
    "masked_vertices",
    "fmri_map_sha256",
    "surrogate_maps_sha256",
}


@dataclass(frozen=True)
class SpatialConvergenceBand:
    band: str
    eeg_map: np.ndarray
    surrogate_r: np.ndarray
    spatial_r: float
    p_value: float
    meaningful: bool
    holm_adjusted_p_value: float
    holm_significant: bool


@dataclass(frozen=True)
class SpatialConvergenceSummary:
    bands: tuple[str, ...]
    fmri_map: np.ndarray
    mask: np.ndarray
    band_results: dict[str, SpatialConvergenceBand]
    vertices_manifest: CommonSourceVertices
    audit: pd.DataFrame
    family_alpha: float
    source_paths: tuple[Path, ...]

    @property
    def n_vertices(self) -> int:
        return self.vertices_manifest.n_vertices

    @property
    def n_surrogates(self) -> int:
        return self.band_results[self.bands[0]].surrogate_r.size


def load_spatial_convergence(config: Any) -> SpatialConvergenceSummary:
    """Load and reconcile the complete three-band spatial artifact family."""

    bands = contribution_bands(config)
    if bands != SPATIAL_BANDS:
        raise ValueError(
            "Study 2 spatial-convergence figure requires configured bands alpha, beta, gamma."
        )

    manifest_path = paths.source_vertex_manifest_path(config)
    manifest_metadata_path = paths.source_vertex_metadata_path(config)
    manifest = load_common_source_vertices(
        array_path=manifest_path,
        metadata_path=manifest_metadata_path,
    )
    common_subject, spacing = _validate_manifest_config(manifest, config)

    mask_path = paths.spatial_mask_path(config)
    mask = _load_array(mask_path, label="analysis mask")
    _validate_mask(mask, n_vertices=manifest.n_vertices)

    n_surrogates = require_config_int(
        config,
        "study2.spatial_comparison.brainsmash_surrogates",
    )
    if n_surrogates < 1:
        raise ValueError("Study 2 BrainSMASH surrogate count must be positive.")
    family_alpha = require_config_float(config, "study2.spatial_comparison.holm_alpha")
    if family_alpha <= 0.0 or family_alpha >= 1.0:
        raise ValueError("Study 2 spatial Holm alpha must lie strictly between zero and one.")

    metadata_path = paths.spatial_surrogate_metadata_path(config)
    metadata = _load_metadata(metadata_path)
    _validate_metadata_family(
        metadata,
        bands=bands,
        common_subject=common_subject,
        spacing=spacing,
        n_surrogates=n_surrogates,
        masked_vertices=int(mask.sum()),
        base_seed=require_config_int(config, "project.random_state"),
        manifest_path=manifest_path,
        mask_path=mask_path,
    )

    eeg_maps: dict[str, np.ndarray] = {}
    fmri_maps: dict[str, np.ndarray] = {}
    surrogate_maps: dict[str, np.ndarray] = {}
    source_paths: list[Path] = [
        manifest_path,
        manifest_metadata_path,
        mask_path,
        metadata_path,
        paths.spatial_correspondence_summary_path(config),
    ]
    for band in bands:
        eeg_path = paths.spatial_eeg_map_path(config, band=band)
        fmri_path = paths.spatial_fmri_map_path(config, band=band)
        surrogate_path = paths.spatial_surrogate_maps_path(config, band=band)
        source_paths.extend([eeg_path, fmri_path, surrogate_path])

        eeg_map = _load_array(eeg_path, label=f"{band} EEG map")
        fmri_map = _load_array(fmri_path, label=f"{band} fMRI map")
        surrogates = _load_array(surrogate_path, label=f"{band} surrogate maps")
        _validate_eeg_map(eeg_map, band=band, n_vertices=manifest.n_vertices)
        _validate_map(fmri_map, label=f"{band} fMRI map", n_vertices=manifest.n_vertices)
        _validate_surrogates(
            surrogates,
            band=band,
            n_surrogates=n_surrogates,
            n_vertices=manifest.n_vertices,
        )
        _validate_band_metadata_hashes(
            metadata,
            band=band,
            fmri_path=fmri_path,
            surrogate_path=surrogate_path,
        )
        eeg_maps[band] = eeg_map
        fmri_maps[band] = fmri_map
        surrogate_maps[band] = surrogates

    fmri_map = fmri_maps[bands[0]]
    for band in bands[1:]:
        if not np.allclose(fmri_maps[band], fmri_map, rtol=0.0, atol=1.0e-12):
            raise ValueError("Study 2 band-specific fMRI maps do not share one reference map.")

    computed = {
        band: compute_spatial_correspondence(
            eeg_map=eeg_maps[band],
            fmri_map=fmri_maps[band],
            surrogate_maps=surrogate_maps[band],
            mask=mask,
            config=config,
        )
        for band in bands
    }
    adjusted_p_values = holm_adjusted_p_values({band: computed[band].p_value for band in bands})
    saved_summary = _load_summary(paths.spatial_correspondence_summary_path(config))
    _reconcile_summary(
        saved_summary,
        bands=bands,
        computed=computed,
        adjusted_p_values=adjusted_p_values,
        family_alpha=family_alpha,
    )

    band_results = {
        band: SpatialConvergenceBand(
            band=band,
            eeg_map=eeg_maps[band],
            surrogate_r=computed[band].surrogate_r,
            spatial_r=computed[band].spatial_r,
            p_value=computed[band].p_value,
            meaningful=computed[band].meaningful,
            holm_adjusted_p_value=adjusted_p_values[band],
            holm_significant=adjusted_p_values[band] <= family_alpha,
        )
        for band in bands
    }
    audit = _build_audit(bands, band_results)
    return SpatialConvergenceSummary(
        bands=bands,
        fmri_map=fmri_map,
        mask=mask,
        band_results=band_results,
        vertices_manifest=manifest,
        audit=audit,
        family_alpha=family_alpha,
        source_paths=tuple(source_paths),
    )


def _load_array(path: Path, *, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Study 2 {label} artifact not found: {path}")
    return np.load(path, allow_pickle=False)


def _validate_manifest_config(
    manifest: CommonSourceVertices,
    config: Any,
) -> tuple[str, str]:
    common_subject = require_config_string(config, "study2.source_modeling.common_subject")
    spacing = require_config_string(
        config,
        "study2.source_modeling.common_source_space_spacing",
    )
    if manifest.common_subject != common_subject or manifest.spacing != spacing:
        raise ValueError("Study 2 source vertex manifest does not match the configured space.")
    return common_subject, spacing


def _validate_mask(mask: np.ndarray, *, n_vertices: int) -> None:
    if mask.dtype != np.bool_:
        raise ValueError("Study 2 spatial analysis mask must have boolean dtype.")
    if mask.ndim != 1 or mask.size != n_vertices:
        raise ValueError("Study 2 spatial analysis mask does not match the source vertex manifest.")
    if not mask.any():
        raise ValueError("Study 2 spatial analysis mask must not be empty.")
    if int(mask.sum()) < 2:
        raise ValueError("Study 2 spatial analysis mask must include at least two vertices.")


def _validate_map(array: np.ndarray, *, label: str, n_vertices: int) -> None:
    _validate_real_values(array, label=label)
    if array.ndim != 1 or array.size != n_vertices:
        raise ValueError(f"Study 2 {label} does not match the source vertex manifest.")
    if not np.isfinite(array).all():
        raise ValueError(f"Study 2 {label} contains non-finite values.")


def _validate_eeg_map(array: np.ndarray, *, band: str, n_vertices: int) -> None:
    label = f"{band} EEG map"
    _validate_map(array, label=label, n_vertices=n_vertices)
    if np.any(np.abs(array) >= 1.0):
        raise ValueError(f"Study 2 {label} values must lie strictly within (-1, 1).")


def _validate_surrogates(
    array: np.ndarray,
    *,
    band: str,
    n_surrogates: int,
    n_vertices: int,
) -> None:
    label = f"{band} surrogate maps"
    _validate_real_values(array, label=label)
    if array.ndim != 2 or array.shape != (n_surrogates, n_vertices):
        raise ValueError(
            f"Study 2 {band} surrogate maps must have shape ({n_surrogates}, {n_vertices})."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Study 2 {band} surrogate maps contain non-finite values.")


def _validate_real_values(array: np.ndarray, *, label: str) -> None:
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise ValueError(f"Study 2 {label} must be real-valued.")


def _load_metadata(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Study 2 spatial surrogate metadata not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_KEYS:
        raise ValueError("Study 2 spatial surrogate metadata has an invalid schema.")
    return metadata


def _validate_metadata_family(
    metadata: Mapping[str, object],
    *,
    bands: tuple[str, ...],
    common_subject: str,
    spacing: str,
    n_surrogates: int,
    masked_vertices: int,
    base_seed: int,
    manifest_path: Path,
    mask_path: Path,
) -> None:
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ValueError("Study 2 spatial surrogate metadata has an invalid schema.")
    if metadata["method"] != "BrainSMASH Base":
        raise ValueError("Study 2 spatial surrogate metadata has an invalid schema.")
    if type(metadata["n_surrogates"]) is not int or metadata["n_surrogates"] != n_surrogates:
        raise ValueError("Study 2 spatial surrogate metadata has the wrong draw count.")
    if (
        metadata["common_subject"] != common_subject
        or metadata["common_source_space_spacing"] != spacing
    ):
        raise ValueError("Study 2 spatial surrogate metadata names the wrong source space.")
    if metadata["source_vertex_manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("Study 2 spatial surrogate metadata names the wrong vertex manifest.")
    if metadata["analysis_mask_sha256"] != _sha256(mask_path):
        raise ValueError("Study 2 spatial surrogate metadata names the wrong analysis mask.")

    band_metadata = metadata["bands"]
    if not isinstance(band_metadata, Mapping) or tuple(band_metadata) != bands:
        raise ValueError("Study 2 spatial surrogate metadata has the wrong band family.")
    for band_index, band in enumerate(bands):
        entry = band_metadata[band]
        if not isinstance(entry, Mapping) or set(entry) != _BAND_METADATA_KEYS:
            raise ValueError("Study 2 spatial surrogate band metadata has an invalid schema.")
        if type(entry["seed"]) is not int or entry["seed"] != base_seed + band_index:
            raise ValueError(f"Study 2 {band} spatial surrogate seed is inconsistent.")
        if type(entry["masked_vertices"]) is not int or entry["masked_vertices"] != masked_vertices:
            raise ValueError(f"Study 2 {band} spatial masked-vertex count is inconsistent.")


def _validate_band_metadata_hashes(
    metadata: Mapping[str, object],
    *,
    band: str,
    fmri_path: Path,
    surrogate_path: Path,
) -> None:
    band_metadata = cast(Mapping[str, object], metadata["bands"])
    entry = cast(Mapping[str, object], band_metadata[band])
    if entry["fmri_map_sha256"] != _sha256(fmri_path):
        raise ValueError(f"Study 2 {band} spatial metadata names the wrong fMRI map.")
    if entry["surrogate_maps_sha256"] != _sha256(surrogate_path):
        raise ValueError(f"Study 2 {band} spatial metadata names the wrong surrogate maps.")


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Study 2 spatial correspondence summary not found: {path}")
    summary = pd.read_csv(path, sep="\t")
    if tuple(summary.columns) != SUMMARY_COLUMNS or summary.empty:
        raise ValueError("Study 2 spatial correspondence summary has an invalid schema.")
    return summary


def _reconcile_summary(
    saved: pd.DataFrame,
    *,
    bands: tuple[str, ...],
    computed: Mapping[str, SpatialCorrespondenceResult],
    adjusted_p_values: Mapping[str, float],
    family_alpha: float,
) -> None:
    if tuple(saved["band"].astype(str)) != bands:
        raise ValueError("Study 2 spatial correspondence summary has the wrong band order.")
    for row in saved.to_dict("records"):
        band = str(row["band"])
        result = computed[band]
        expected_values = (
            result.spatial_r,
            result.p_value,
            adjusted_p_values[band],
        )
        saved_values = (
            float(row["spatial_r"]),
            float(row["p_value"]),
            float(row["holm_adjusted_p_value"]),
        )
        if not np.allclose(saved_values, expected_values, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("Study 2 saved spatial summary does not match recomputed inference.")
        if parse_bool(row["meaningful"]) != result.meaningful:
            raise ValueError("Study 2 saved spatial summary does not match recomputed inference.")
        expected_significant = adjusted_p_values[band] <= family_alpha
        if parse_bool(row["holm_significant"]) != expected_significant:
            raise ValueError("Study 2 saved spatial summary does not match recomputed inference.")


def _build_audit(
    bands: tuple[str, ...],
    band_results: Mapping[str, SpatialConvergenceBand],
) -> pd.DataFrame:
    records = []
    for band in bands:
        result = band_results[band]
        null_ci_low, null_ci_high = np.percentile(result.surrogate_r, [2.5, 97.5])
        records.append(
            {
                "band": band,
                "spatial_r": result.spatial_r,
                "p_value": result.p_value,
                "meaningful": result.meaningful,
                "holm_adjusted_p_value": result.holm_adjusted_p_value,
                "holm_significant": result.holm_significant,
                "null_ci_low": float(null_ci_low),
                "null_ci_high": float(null_ci_high),
            }
        )
    return pd.DataFrame.from_records(records, columns=AUDIT_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AUDIT_COLUMNS",
    "SpatialConvergenceBand",
    "SpatialConvergenceSummary",
    "load_spatial_convergence",
]
