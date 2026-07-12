"""Strict statistical reader for the Study 2 primary cortical figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from studies.pain_study.study2 import paths
from studies.pain_study.study2.source_family import (
    SourceFamilyInferenceResult,
    compute_source_family_inference,
    summarize_source_family,
)
from studies.pain_study.study2.source_stage_design import contribution_bands
from studies.pain_study.study2.source_vertex_manifest import (
    CommonSourceVertices,
    load_common_source_vertices,
)
from studies.pain_study.study2.table_io import parse_bool, require_columns
from studies.pain_study.study2.validation import (
    require_config_float,
    require_config_int,
    require_config_string,
)

PRIMARY_BANDS = ("alpha", "beta", "gamma")
VERTEX_COLUMNS = (
    "band",
    "hemisphere",
    "vertex_number",
    "effect_r",
    "t_value",
    "cluster_id",
    "corrected_contour",
)
CLUSTER_COLUMNS = (
    "band",
    "cluster_id",
    "sign",
    "n_vertices",
    "cluster_mass",
    "max_cluster_p_value",
    "band_holm_adjusted_p_value",
    "corrected_contour",
)
SUMMARY_COLUMNS = (
    "band",
    "n_subjects",
    "n_vertices",
    "n_permutations",
    "n_clusters",
    "n_corrected_clusters",
    "min_cluster_p_value",
    "holm_adjusted_p_value",
    "significant",
    "cluster_forming_p",
    "cluster_threshold",
    "family_alpha",
    "display_limit",
)


@dataclass(frozen=True)
class PrimarySourceBand:
    band: str
    partial_r_maps: np.ndarray
    fisher_z_maps: np.ndarray
    effect_r: np.ndarray
    t_values: np.ndarray
    cluster_ids: np.ndarray
    corrected_vertex_mask: np.ndarray
    min_cluster_p_value: float
    holm_adjusted_p_value: float
    significant: bool
    n_permutations: int


@dataclass(frozen=True)
class PrimarySourceAssociations:
    bands: tuple[str, ...]
    subject_ids: tuple[str, ...]
    band_results: dict[str, PrimarySourceBand]
    vertices_manifest: CommonSourceVertices
    vertices: pd.DataFrame
    clusters: pd.DataFrame
    summary: pd.DataFrame
    display_limit: float
    cluster_forming_p: float
    family_alpha: float
    source_paths: tuple[Path, ...]

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_vertices(self) -> int:
        return self.vertices_manifest.n_vertices


def load_primary_source_associations(config: object) -> PrimarySourceAssociations:
    """Load, validate, and reconcile all primary source-map artifacts."""

    bands = contribution_bands(config)
    if bands != PRIMARY_BANDS:
        raise ValueError(
            "Study 2 primary source figure requires configured bands alpha, beta, gamma."
        )
    cluster_forming_p = require_config_float(
        config,
        "study2.source_inference.primary_cluster_forming_p",
    )
    family_alpha = require_config_float(config, "study2.source_inference.family_alpha")
    manifest = load_common_source_vertices(
        array_path=paths.source_vertex_manifest_path(config),
        metadata_path=paths.source_vertex_metadata_path(config),
    )
    _validate_manifest_config(manifest, config)

    partial_maps: dict[str, np.ndarray] = {}
    fisher_maps: dict[str, np.ndarray] = {}
    null_maps: dict[str, np.ndarray] = {}
    cohorts: dict[str, tuple[str, ...]] = {}
    source_paths: list[Path] = [
        paths.source_vertex_manifest_path(config),
        paths.source_vertex_metadata_path(config),
        paths.source_adjacency_path(config),
        paths.source_family_summary_path(config),
    ]
    for band in bands:
        partial_path = paths.source_stage_partial_r_path(config, band=band)
        fisher_path = paths.source_stage_fisher_z_path(config, band=band)
        qc_path = paths.source_stage_qc_path(config, band=band)
        null_path = paths.null_source_maps_path(config, band=band)
        source_paths.extend([partial_path, fisher_path, qc_path, null_path])

        partial = _load_array(partial_path, label=f"{band} partial-r maps")
        fisher = _load_array(fisher_path, label=f"{band} Fisher-z maps")
        _validate_observed_maps(
            partial,
            fisher,
            band=band,
            n_vertices=manifest.n_vertices,
        )
        cohort = _valid_subject_ids(_read_table(qc_path, label=f"{band} QC"), band=band)
        if len(cohort) != partial.shape[0]:
            raise ValueError(
                f"Study 2 {band} map rows do not match its source-valid QC participants."
            )
        null = _load_array(null_path, label=f"{band} target-retrained null maps")
        _validate_null_maps(null, observed_shape=fisher.shape, band=band)
        partial_maps[band] = partial
        fisher_maps[band] = fisher
        null_maps[band] = null
        cohorts[band] = cohort

    subject_ids = cohorts[bands[0]]
    if any(cohorts[band] != subject_ids for band in bands[1:]):
        raise ValueError(
            "Study 2 primary bands must use the same valid participant cohort and order."
        )
    minimum_subjects = require_config_int(
        config,
        "study2.source_stage.min_source_valid_subjects",
    )
    if len(subject_ids) < minimum_subjects:
        raise ValueError(
            "Study 2 primary source figure requires at least "
            f"{minimum_subjects} source-valid participants; found {len(subject_ids)}."
        )

    adjacency = _load_array(paths.source_adjacency_path(config), label="source adjacency")
    if adjacency.dtype != np.bool_:
        raise ValueError("Study 2 source adjacency must use boolean values.")
    family = compute_source_family_inference(
        observed_maps_by_band=fisher_maps,
        null_maps_by_band=null_maps,
        adjacency=adjacency,
        cluster_forming_p=cluster_forming_p,
        alpha=family_alpha,
    )
    saved_summary = _read_table(
        paths.source_family_summary_path(config),
        label="source-family summary",
    )
    _reconcile_family_summary(saved_summary, family)

    band_results, vertices, clusters = _build_outputs(
        bands=bands,
        partial_maps=partial_maps,
        fisher_maps=fisher_maps,
        family=family,
        manifest=manifest,
        family_alpha=family_alpha,
    )
    display_limit = _display_limit(band_results)
    summary = _summary_table(
        bands=bands,
        band_results=band_results,
        family=family,
        n_vertices=manifest.n_vertices,
        cluster_forming_p=cluster_forming_p,
        family_alpha=family_alpha,
        display_limit=display_limit,
    )
    return PrimarySourceAssociations(
        bands=bands,
        subject_ids=subject_ids,
        band_results=band_results,
        vertices_manifest=manifest,
        vertices=vertices,
        clusters=clusters,
        summary=summary,
        display_limit=display_limit,
        cluster_forming_p=cluster_forming_p,
        family_alpha=family_alpha,
        source_paths=tuple(source_paths),
    )


def _load_array(path: Path, *, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Study 2 {label} artifact not found: {path}")
    return np.load(path, allow_pickle=False)


def _read_table(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Study 2 {label} artifact not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        raise ValueError(f"Study 2 {label} artifact is empty: {path}")
    return frame


def _validate_manifest_config(manifest: CommonSourceVertices, config: object) -> None:
    common_subject = require_config_string(config, "study2.source_modeling.common_subject")
    spacing = require_config_string(
        config,
        "study2.source_modeling.common_source_space_spacing",
    )
    if manifest.common_subject != common_subject or manifest.spacing != spacing:
        raise ValueError("Study 2 source vertex manifest does not match the configured space.")


def _validate_observed_maps(
    partial: np.ndarray,
    fisher: np.ndarray,
    *,
    band: str,
    n_vertices: int,
) -> None:
    if partial.ndim != 2 or fisher.ndim != 2:
        raise ValueError(f"Study 2 {band} observed maps must be two-dimensional.")
    if partial.shape != fisher.shape:
        raise ValueError(f"Study 2 {band} partial-r and Fisher-z map shapes differ.")
    if partial.shape[1] != n_vertices:
        raise ValueError(f"Study 2 {band} maps do not match the source vertex manifest.")
    if not np.isfinite(partial).all() or not np.isfinite(fisher).all():
        raise ValueError(f"Study 2 {band} observed maps contain non-finite values.")
    if np.any(np.abs(partial) >= 1.0):
        raise ValueError(f"Study 2 {band} partial-r values must lie within (-1, 1).")
    if not np.allclose(np.arctanh(partial), fisher, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"Study 2 {band} partial-r and Fisher-z maps are inconsistent.")


def _validate_null_maps(null: np.ndarray, *, observed_shape: tuple[int, int], band: str) -> None:
    if null.ndim != 3 or null.shape[1:] != observed_shape:
        raise ValueError(f"Study 2 {band} null maps must be draws by participants by vertices.")
    if null.shape[0] < 1 or not np.isfinite(null).all():
        raise ValueError(f"Study 2 {band} null maps must contain finite draws.")


def _valid_subject_ids(qc: pd.DataFrame, *, band: str) -> tuple[str, ...]:
    require_columns(
        qc,
        ("subject_id", "source_stage_criteria_met"),
        name=f"Study 2 {band} source-stage QC",
    )
    subject_ids = qc["subject_id"].astype(str)
    if subject_ids.duplicated().any():
        raise ValueError(f"Study 2 {band} source-stage QC has duplicate participant IDs.")
    valid = qc["source_stage_criteria_met"].map(parse_bool).to_numpy(dtype=bool)
    return tuple(subject_ids.loc[valid])


def _reconcile_family_summary(
    saved: pd.DataFrame,
    family: SourceFamilyInferenceResult,
) -> None:
    recomputed = summarize_source_family(family)
    required = tuple(recomputed.columns)
    require_columns(saved, required, name="Study 2 saved source-family summary")
    saved = saved.loc[:, list(required)].reset_index(drop=True)
    if saved["band"].astype(str).tolist() != recomputed["band"].tolist():
        raise ValueError("Study 2 saved source-family summary does not match recomputed inference.")
    for column in ("n_subjects", "n_permutations", "n_clusters"):
        if not np.array_equal(
            pd.to_numeric(saved[column], errors="raise").to_numpy(dtype=int),
            recomputed[column].to_numpy(dtype=int),
        ):
            raise ValueError(
                "Study 2 saved source-family summary does not match recomputed inference."
            )
    for column in ("min_cluster_p_value", "holm_adjusted_p_value"):
        if not np.allclose(
            pd.to_numeric(saved[column], errors="raise").to_numpy(dtype=float),
            recomputed[column].to_numpy(dtype=float),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Study 2 saved source-family summary does not match recomputed inference."
            )
    saved_significant = saved["significant"].map(parse_bool).to_numpy(dtype=bool)
    if not np.array_equal(saved_significant, recomputed["significant"].to_numpy(dtype=bool)):
        raise ValueError("Study 2 saved source-family summary does not match recomputed inference.")


def _build_outputs(
    *,
    bands: tuple[str, ...],
    partial_maps: dict[str, np.ndarray],
    fisher_maps: dict[str, np.ndarray],
    family: SourceFamilyInferenceResult,
    manifest: CommonSourceVertices,
    family_alpha: float,
) -> tuple[dict[str, PrimarySourceBand], pd.DataFrame, pd.DataFrame]:
    band_results: dict[str, PrimarySourceBand] = {}
    vertex_records: list[dict[str, object]] = []
    cluster_records: list[dict[str, object]] = []
    hemispheres = np.repeat(
        ("left", "right"),
        (manifest.lh_vertices.size, manifest.rh_vertices.size),
    )
    vertex_numbers = np.concatenate(manifest.vertices)

    for band in bands:
        family_band = family.band_results[band]
        inference = family_band.inference
        cluster_ids = np.full(manifest.n_vertices, "", dtype=object)
        corrected_mask = np.zeros(manifest.n_vertices, dtype=bool)
        for cluster_index, cluster in enumerate(inference.clusters, start=1):
            cluster_id = f"{band}_{cluster.sign}_{cluster_index:02d}"
            cluster_vertices = np.asarray(cluster.vertices, dtype=int)
            cluster_ids[cluster_vertices] = cluster_id
            corrected = bool(
                cluster.p_value <= family_alpha
                and family_band.holm_adjusted_p_value <= family_alpha
            )
            if corrected:
                corrected_mask[cluster_vertices] = True
            cluster_records.append(
                {
                    "band": band,
                    "cluster_id": cluster_id,
                    "sign": cluster.sign,
                    "n_vertices": len(cluster.vertices),
                    "cluster_mass": cluster.mass,
                    "max_cluster_p_value": cluster.p_value,
                    "band_holm_adjusted_p_value": family_band.holm_adjusted_p_value,
                    "corrected_contour": corrected,
                }
            )
        effect_r = np.tanh(np.mean(fisher_maps[band], axis=0))
        for vertex_index in range(manifest.n_vertices):
            vertex_records.append(
                {
                    "band": band,
                    "hemisphere": hemispheres[vertex_index],
                    "vertex_number": int(vertex_numbers[vertex_index]),
                    "effect_r": effect_r[vertex_index],
                    "t_value": inference.t_values[vertex_index],
                    "cluster_id": cluster_ids[vertex_index],
                    "corrected_contour": bool(corrected_mask[vertex_index]),
                }
            )
        band_results[band] = PrimarySourceBand(
            band=band,
            partial_r_maps=partial_maps[band],
            fisher_z_maps=fisher_maps[band],
            effect_r=effect_r,
            t_values=inference.t_values,
            cluster_ids=cluster_ids,
            corrected_vertex_mask=corrected_mask,
            min_cluster_p_value=family_band.min_cluster_p_value,
            holm_adjusted_p_value=family_band.holm_adjusted_p_value,
            significant=family_band.significant,
            n_permutations=inference.n_permutations,
        )
    return (
        band_results,
        pd.DataFrame.from_records(vertex_records, columns=VERTEX_COLUMNS),
        pd.DataFrame.from_records(cluster_records, columns=CLUSTER_COLUMNS),
    )


def _display_limit(band_results: dict[str, PrimarySourceBand]) -> float:
    limit = float(max(np.max(np.abs(result.effect_r)) for result in band_results.values()))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Study 2 primary source effects must contain a finite non-zero value.")
    return limit


def _summary_table(
    *,
    bands: tuple[str, ...],
    band_results: dict[str, PrimarySourceBand],
    family: SourceFamilyInferenceResult,
    n_vertices: int,
    cluster_forming_p: float,
    family_alpha: float,
    display_limit: float,
) -> pd.DataFrame:
    records = []
    for band in bands:
        result = band_results[band]
        inference = family.band_results[band].inference
        records.append(
            {
                "band": band,
                "n_subjects": result.fisher_z_maps.shape[0],
                "n_vertices": n_vertices,
                "n_permutations": result.n_permutations,
                "n_clusters": len(inference.clusters),
                "n_corrected_clusters": sum(
                    cluster.p_value <= family_alpha
                    and result.holm_adjusted_p_value <= family_alpha
                    for cluster in inference.clusters
                ),
                "min_cluster_p_value": result.min_cluster_p_value,
                "holm_adjusted_p_value": result.holm_adjusted_p_value,
                "significant": result.significant,
                "cluster_forming_p": cluster_forming_p,
                "cluster_threshold": inference.threshold,
                "family_alpha": family_alpha,
                "display_limit": display_limit,
            }
        )
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


__all__ = [
    "CLUSTER_COLUMNS",
    "PRIMARY_BANDS",
    "SUMMARY_COLUMNS",
    "VERTEX_COLUMNS",
    "PrimarySourceAssociations",
    "PrimarySourceBand",
    "load_primary_source_associations",
]
