"""Build frozen cortical ROI labels for EEG-BOLD coupling from published signature maps."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import mne
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.surface import vol_to_surf
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from eeg_pipeline.infra.paths import ensure_dir, resolve_deriv_root
from eeg_pipeline.utils.config.loader import get_config_value
from fmri_pipeline.analysis.multivariate_signatures import discover_signature_files
from fmri_pipeline.utils.signature_paths import discover_signature_root_and_specs


LOGGER = logging.getLogger(__name__)
_SURFACE_DEPTH_FRACTIONS = tuple(np.linspace(0.0, 1.0, 7))


@dataclass(frozen=True)
class ROIIncludeSpec:
    signature: str
    sign: str


@dataclass(frozen=True)
class ROIBuilderItem:
    name: str
    definition: str
    hemisphere: str
    atlas_labels: Tuple[str, ...]
    include: Tuple[ROIIncludeSpec, ...]
    min_vertices: int
    exclude_discordant_vertices: bool
    connected_components: bool
    component_selection: str
    min_component_vertices: int


@dataclass(frozen=True)
class ROIBuilderConfig:
    enabled: bool
    output_dir: Optional[Path]
    template_subject: str
    parcellation: str
    min_vertices_default: int
    abs_weight_threshold: float
    exclude_discordant_vertices: bool
    items: Tuple[ROIBuilderItem, ...]

    @classmethod
    def from_config(cls, config: Any) -> "ROIBuilderConfig":
        raw = get_config_value(config, "eeg_bold_coupling.roi_builder", {})
        if not isinstance(raw, dict):
            raise ValueError("eeg_bold_coupling.roi_builder must be a mapping.")
        items_raw = raw.get("items", [])
        if not isinstance(items_raw, list):
            raise ValueError("eeg_bold_coupling.roi_builder.items must be a list.")
        min_vertices_default = int(raw.get("min_vertices_default", 25))
        component_min_vertices_default = int(
            raw.get("min_component_vertices_default", min_vertices_default)
        )
        default_exclude = bool(raw.get("exclude_discordant_vertices", True))
        default_connected_components = bool(raw.get("connected_components", True))
        default_component_selection = str(
            raw.get("component_selection", "largest")
        ).strip().lower()
        if default_component_selection not in {"largest", "all"}:
            raise ValueError(
                "eeg_bold_coupling.roi_builder.component_selection must be 'largest' or 'all'."
            )
        items: List[ROIBuilderItem] = []
        for entry in items_raw:
            if not isinstance(entry, dict):
                raise ValueError("Each roi_builder item must be a mapping.")
            definition = str(
                entry.get("definition", "anatomical")
            ).strip().lower()
            if definition not in {"anatomical", "signature_intersection"}:
                raise ValueError(
                    "roi_builder definition must be 'anatomical' or 'signature_intersection'."
                )
            include_raw = entry.get("include", [])
            include_specs: List[ROIIncludeSpec] = []
            if definition == "signature_intersection":
                if not isinstance(include_raw, list) or not include_raw:
                    raise ValueError(
                        "signature_intersection ROI builder items must define a non-empty include list."
                    )
                for include in include_raw:
                    if not isinstance(include, dict):
                        raise ValueError("Each roi_builder include entry must be a mapping.")
                    signature = str(include.get("signature", "")).strip()
                    sign = str(include.get("sign", "")).strip().lower()
                    if sign not in {"positive", "negative"}:
                        raise ValueError("roi_builder include sign must be 'positive' or 'negative'.")
                    if not signature:
                        raise ValueError("roi_builder include signature is required.")
                    include_specs.append(ROIIncludeSpec(signature=signature, sign=sign))
            elif include_raw not in (None, [], ()):
                raise ValueError(
                    "anatomical ROI builder items must not define include entries."
                )
            atlas_labels = tuple(
                str(label).strip()
                for label in entry.get("atlas_labels", [])
                if str(label).strip()
            )
            if not atlas_labels:
                raise ValueError("Each roi_builder item must define atlas_labels.")
            name = str(entry.get("name", "")).strip()
            hemisphere = str(entry.get("hemisphere", "")).strip().lower()
            if not name:
                raise ValueError("Each roi_builder item needs a name.")
            if hemisphere not in {"lh", "rh", "both"}:
                raise ValueError(
                    "roi_builder hemisphere must be 'lh', 'rh', or 'both'."
                )
            component_selection = str(
                entry.get("component_selection", default_component_selection)
            ).strip().lower()
            if component_selection not in {"largest", "all"}:
                raise ValueError(
                    f"ROI {name!r} has invalid component_selection={component_selection!r}; expected 'largest' or 'all'."
                )
            items.append(
                ROIBuilderItem(
                    name=name,
                    definition=definition,
                    hemisphere=hemisphere,
                    atlas_labels=atlas_labels,
                    include=tuple(include_specs),
                    min_vertices=int(entry.get("min_vertices", min_vertices_default)),
                    exclude_discordant_vertices=bool(
                        entry.get("exclude_discordant_vertices", default_exclude)
                    ),
                    connected_components=bool(
                        entry.get("connected_components", default_connected_components)
                    ),
                    component_selection=component_selection,
                    min_component_vertices=int(
                        entry.get(
                            "min_component_vertices",
                            component_min_vertices_default,
                        )
                    ),
                )
            )
        output_dir_raw = raw.get("output_dir")
        output_dir = None
        if output_dir_raw is not None and str(output_dir_raw).strip():
            output_dir = Path(str(output_dir_raw)).expanduser().resolve()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            output_dir=output_dir,
            template_subject=str(raw.get("template_subject", "fsaverage")).strip(),
            parcellation=str(raw.get("parcellation", "aparc.a2009s")).strip(),
            min_vertices_default=min_vertices_default,
            abs_weight_threshold=float(raw.get("abs_weight_threshold", 0.0)),
            exclude_discordant_vertices=default_exclude,
            items=tuple(items),
        )


@dataclass(frozen=True)
class BuiltROI:
    name: str
    hemisphere: str
    label_paths: Tuple[Path, ...]
    provenance_path: Path
    atlas_labels: Tuple[str, ...]
    include: Tuple[ROIIncludeSpec, ...]
    n_vertices: int


@dataclass(frozen=True)
class ROIComponentResult:
    kept_vertices: np.ndarray
    component_sizes: Tuple[int, ...]
    kept_component_sizes: Tuple[int, ...]


def _resolve_subjects_dir(config: Any) -> Path:
    value = get_config_value(config, "eeg_bold_coupling.eeg.subjects_dir", None)
    if value is None:
        value = get_config_value(config, "paths.freesurfer_dir", None)
    if value is None or not str(value).strip():
        raise ValueError(
            "eeg_bold_coupling.eeg.subjects_dir or paths.freesurfer_dir is required."
        )
    return Path(str(value)).expanduser().resolve()


def _resolve_output_dir(
    *,
    cfg: ROIBuilderConfig,
    deriv_root: Path,
) -> Path:
    if cfg.output_dir is not None:
        return cfg.output_dir
    return deriv_root / "roi_library" / "eeg_bold_coupling"


def _signature_surface_values(
    *,
    signature_paths: Mapping[str, Path],
    subjects_dir: Path,
    template_subject: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    surf_dir = subjects_dir / template_subject / "surf"
    surfaces = {
        "lh": {
            "pial": surf_dir / "lh.pial",
            "white": surf_dir / "lh.white",
        },
        "rh": {
            "pial": surf_dir / "rh.pial",
            "white": surf_dir / "rh.white",
        },
    }
    for hemi, hemi_paths in surfaces.items():
        for path in hemi_paths.values():
            if not path.exists():
                raise FileNotFoundError(f"Missing template surface file: {path}")

    values: Dict[str, Dict[str, np.ndarray]] = {}
    for name, path in signature_paths.items():
        img = nib.load(str(path))
        hemi_values: Dict[str, np.ndarray] = {}
        for hemi, hemi_paths in surfaces.items():
            hemi_values[hemi] = np.asarray(
                vol_to_surf(
                    img,
                    surf_mesh=str(hemi_paths["pial"]),
                    inner_mesh=str(hemi_paths["white"]),
                    kind="depth",
                    depth=_SURFACE_DEPTH_FRACTIONS,
                ),
                dtype=float,
            )
        values[name] = hemi_values
    return values


def _read_template_surface_triangles(
    *,
    subjects_dir: Path,
    template_subject: str,
    hemisphere: str,
) -> np.ndarray:
    surface_path = subjects_dir / template_subject / "surf" / f"{hemisphere}.white"
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing template surface: {surface_path}")
    _rr, tris = mne.read_surface(str(surface_path), verbose=False)
    triangles = np.asarray(tris, dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Invalid surface triangles in {surface_path}")
    return triangles


def _atlas_label_map(
    *,
    subjects_dir: Path,
    template_subject: str,
    parcellation: str,
) -> Dict[str, Any]:
    labels = mne.read_labels_from_annot(
        subject=template_subject,
        parc=parcellation,
        subjects_dir=str(subjects_dir),
        verbose=False,
    )
    label_map = {str(label.name): label for label in labels}
    if not label_map:
        raise ValueError(
            f"No labels found for {template_subject}/{parcellation}."
        )
    return label_map


def _candidate_vertices_for_hemi(
    *,
    item: ROIBuilderItem,
    atlas_labels: Mapping[str, Any],
    hemisphere: str,
) -> np.ndarray:
    vertices: List[np.ndarray] = []
    for name in item.atlas_labels:
        if name not in atlas_labels:
            raise ValueError(f"Atlas label {name!r} not found in template parcellation.")
        label = atlas_labels[name]
        hemi = str(label.hemi).strip().lower()
        if hemi != hemisphere:
            if item.hemisphere == "both":
                continue
            raise ValueError(
                f"Atlas label {name!r} hemisphere {hemi!r} does not match ROI hemisphere {item.hemisphere!r}."
            )
        vertices.append(np.asarray(label.vertices, dtype=int))
    if not vertices:
        raise ValueError(
            f"ROI {item.name!r} has no atlas vertices in hemisphere {hemisphere!r}."
        )
    return np.unique(np.concatenate(vertices))


def _include_vertices_for_signature(
    *,
    values: np.ndarray,
    vertices: np.ndarray,
    sign: str,
    abs_weight_threshold: float,
) -> np.ndarray:
    candidate_values = np.asarray(values[vertices], dtype=float)
    magnitude_mask = np.abs(candidate_values) > float(abs_weight_threshold)
    if sign == "positive":
        sign_mask = candidate_values > 0
    else:
        sign_mask = candidate_values < 0
    return vertices[magnitude_mask & sign_mask]


def _discordant_vertices_for_signature(
    *,
    values: np.ndarray,
    vertices: np.ndarray,
    sign: str,
    abs_weight_threshold: float,
) -> np.ndarray:
    candidate_values = np.asarray(values[vertices], dtype=float)
    magnitude_mask = np.abs(candidate_values) > float(abs_weight_threshold)
    if sign == "positive":
        sign_mask = candidate_values < 0
    else:
        sign_mask = candidate_values > 0
    return vertices[magnitude_mask & sign_mask]


def _component_adjacency(
    *,
    selected_vertices: np.ndarray,
    triangles: np.ndarray,
) -> sparse.csr_matrix:
    if selected_vertices.size == 0:
        return sparse.csr_matrix((0, 0), dtype=bool)
    selected_set = set(int(v) for v in selected_vertices.tolist())
    remap = {int(vertex): idx for idx, vertex in enumerate(selected_vertices.tolist())}
    edges: List[Tuple[int, int]] = []
    for tri in np.asarray(triangles, dtype=int):
        tri_vertices = [int(v) for v in tri.tolist() if int(v) in selected_set]
        if len(tri_vertices) < 2:
            continue
        unique_vertices = sorted(set(tri_vertices))
        for idx, left in enumerate(unique_vertices[:-1]):
            for right in unique_vertices[idx + 1:]:
                edges.append((remap[left], remap[right]))
                edges.append((remap[right], remap[left]))
    if not edges:
        return sparse.eye(selected_vertices.size, format="csr", dtype=bool)
    rows = np.asarray([edge[0] for edge in edges], dtype=int)
    cols = np.asarray([edge[1] for edge in edges], dtype=int)
    data = np.ones(rows.shape[0], dtype=bool)
    adjacency = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(selected_vertices.size, selected_vertices.size),
        dtype=bool,
    ).tocsr()
    adjacency = adjacency + sparse.eye(selected_vertices.size, format="csr", dtype=bool)
    return adjacency


def _prune_connected_components(
    *,
    item: ROIBuilderItem,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> ROIComponentResult:
    if vertices.size == 0:
        return ROIComponentResult(
            kept_vertices=np.asarray([], dtype=int),
            component_sizes=tuple(),
            kept_component_sizes=tuple(),
        )
    if not item.connected_components:
        return ROIComponentResult(
            kept_vertices=np.asarray(vertices, dtype=int),
            component_sizes=(int(vertices.size),),
            kept_component_sizes=(int(vertices.size),),
        )

    adjacency = _component_adjacency(
        selected_vertices=np.asarray(vertices, dtype=int),
        triangles=triangles,
    )
    n_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    all_component_sizes = tuple(
        int(np.sum(labels == component_id))
        for component_id in range(int(n_components))
    )
    kept_components: List[np.ndarray] = []
    for component_id in range(int(n_components)):
        component_vertices = np.asarray(vertices, dtype=int)[labels == component_id]
        if component_vertices.size < int(item.min_component_vertices):
            continue
        kept_components.append(np.asarray(component_vertices, dtype=int))

    if not kept_components:
        return ROIComponentResult(
            kept_vertices=np.asarray([], dtype=int),
            component_sizes=all_component_sizes,
            kept_component_sizes=tuple(),
        )

    if item.component_selection == "largest":
        kept_components = [max(kept_components, key=lambda arr: int(arr.size))]

    kept_sizes = tuple(int(component.size) for component in kept_components)
    kept_vertices = np.unique(np.concatenate(kept_components))
    return ROIComponentResult(
        kept_vertices=np.asarray(kept_vertices, dtype=int),
        component_sizes=all_component_sizes,
        kept_component_sizes=kept_sizes,
    )


def _build_single_hemi_roi(
    *,
    item: ROIBuilderItem,
    hemisphere: str,
    surface_values: Mapping[str, Mapping[str, np.ndarray]],
    atlas_labels: Mapping[str, Any],
    template_subject: str,
    abs_weight_threshold: float,
    triangles: np.ndarray,
) -> Tuple[mne.Label, Dict[str, Any]]:
    candidate_vertices = _candidate_vertices_for_hemi(
        item=item,
        atlas_labels=atlas_labels,
        hemisphere=hemisphere,
    )
    include_details: List[Dict[str, Any]] = []
    if item.definition == "anatomical":
        merged_vertices = np.asarray(candidate_vertices, dtype=int)
        merged_excluded = np.asarray([], dtype=int)
    else:
        included: List[np.ndarray] = []
        excluded: List[np.ndarray] = []
        for include in item.include:
            if include.signature not in surface_values:
                raise ValueError(
                    f"Signature {include.signature!r} required by ROI {item.name!r} was not resolved."
                )
            hemi_values = surface_values[include.signature][hemisphere]
            included_vertices = _include_vertices_for_signature(
                values=hemi_values,
                vertices=candidate_vertices,
                sign=include.sign,
                abs_weight_threshold=abs_weight_threshold,
            )
            included.append(included_vertices)
            detail: Dict[str, Any] = {
                "signature": include.signature,
                "sign": include.sign,
                "n_included_vertices": int(included_vertices.size),
            }
            if item.exclude_discordant_vertices:
                excluded_vertices = _discordant_vertices_for_signature(
                    values=hemi_values,
                    vertices=candidate_vertices,
                    sign=include.sign,
                    abs_weight_threshold=abs_weight_threshold,
                )
                excluded.append(excluded_vertices)
                detail["n_discordant_vertices"] = int(excluded_vertices.size)
            include_details.append(detail)
        merged_vertices = (
            np.unique(np.concatenate(included))
            if included
            else np.asarray([], dtype=int)
        )
        if excluded:
            merged_excluded = np.unique(np.concatenate(excluded))
            merged_vertices = merged_vertices[~np.isin(merged_vertices, merged_excluded)]
        else:
            merged_excluded = np.asarray([], dtype=int)

    component_result = _prune_connected_components(
        item=item,
        vertices=np.asarray(merged_vertices, dtype=int),
        triangles=triangles,
    )
    merged_vertices = component_result.kept_vertices

    if int(merged_vertices.size) < int(item.min_vertices):
        raise ValueError(
            f"ROI {item.name!r} retained {int(merged_vertices.size)} vertices, below min_vertices={item.min_vertices}."
        )

    label = mne.Label(
        vertices=np.asarray(merged_vertices, dtype=int),
        hemi=hemisphere,
        name=item.name,
        subject=template_subject,
    )
    details = {
        "hemisphere": hemisphere,
        "definition": item.definition,
        "candidate_vertex_count": int(candidate_vertices.size),
        "post_sign_vertex_count": int(
            np.unique(np.concatenate(included)).size if included else 0
        ),
        "discordant_vertex_count": int(merged_excluded.size),
        "component_sizes": list(component_result.component_sizes),
        "kept_component_sizes": list(component_result.kept_component_sizes),
        "final_vertex_count": int(merged_vertices.size),
        "final_vertices": [
            int(vertex) for vertex in np.asarray(merged_vertices, dtype=int).tolist()
        ],
        "connected_components": bool(item.connected_components),
        "component_selection": item.component_selection,
        "min_component_vertices": int(item.min_component_vertices),
        "include_details": include_details,
    }
    return label, details


def _build_single_roi(
    *,
    item: ROIBuilderItem,
    surface_values: Mapping[str, Mapping[str, np.ndarray]],
    atlas_labels: Mapping[str, Any],
    template_subject: str,
    abs_weight_threshold: float,
    triangles: Mapping[str, np.ndarray],
) -> Tuple[Any, Dict[str, Any]]:
    if item.hemisphere in {"lh", "rh"}:
        label, details = _build_single_hemi_roi(
            item=item,
            hemisphere=item.hemisphere,
            surface_values=surface_values,
            atlas_labels=atlas_labels,
            template_subject=template_subject,
            abs_weight_threshold=abs_weight_threshold,
            triangles=triangles[item.hemisphere],
        )
        details["per_hemisphere"] = {item.hemisphere: details.copy()}
        return label, details

    hemi_results: Dict[str, Tuple[mne.Label, Dict[str, Any]]] = {}
    for hemisphere in ("lh", "rh"):
        hemi_results[hemisphere] = _build_single_hemi_roi(
            item=item,
            hemisphere=hemisphere,
            surface_values=surface_values,
            atlas_labels=atlas_labels,
            template_subject=template_subject,
            abs_weight_threshold=abs_weight_threshold,
            triangles=triangles[hemisphere],
        )
    label = hemi_results["lh"][0] + hemi_results["rh"][0]
    per_hemi_details = {
        hemisphere: details
        for hemisphere, (_label, details) in hemi_results.items()
    }
    merged_details = {
        "candidate_vertex_count": int(
            sum(detail["candidate_vertex_count"] for detail in per_hemi_details.values())
        ),
        "post_sign_vertex_count": int(
            sum(detail["post_sign_vertex_count"] for detail in per_hemi_details.values())
        ),
        "discordant_vertex_count": int(
            sum(detail["discordant_vertex_count"] for detail in per_hemi_details.values())
        ),
        "component_sizes": [],
        "kept_component_sizes": [],
        "final_vertex_count": int(
            len(label.lh.vertices) + len(label.rh.vertices)
        ),
        "connected_components": bool(item.connected_components),
        "component_selection": item.component_selection,
        "min_component_vertices": int(item.min_component_vertices),
        "include_details": [],
        "per_hemisphere": per_hemi_details,
    }
    for detail in per_hemi_details.values():
        merged_details["component_sizes"].extend(detail["component_sizes"])
        merged_details["kept_component_sizes"].extend(detail["kept_component_sizes"])
        merged_details["include_details"].extend(detail["include_details"])
    return label, merged_details


def build_eeg_bold_rois(
    *,
    config: Any,
    logger: Optional[logging.Logger] = None,
) -> List[BuiltROI]:
    roi_logger = logger or LOGGER
    cfg = ROIBuilderConfig.from_config(config)
    if not cfg.enabled:
        return []
    if not cfg.items:
        raise ValueError("ROI builder is enabled but roi_builder.items is empty.")

    deriv_root = resolve_deriv_root(config=config)
    output_dir = _resolve_output_dir(cfg=cfg, deriv_root=deriv_root)
    ensure_dir(output_dir)
    subjects_dir = _resolve_subjects_dir(config)

    requires_signatures = any(
        item.definition == "signature_intersection"
        for item in cfg.items
    )
    signature_paths: Dict[str, Path] = {}
    surface_values: Dict[str, Dict[str, np.ndarray]] = {}
    if requires_signatures:
        signature_root, signature_specs = discover_signature_root_and_specs(config, deriv_root)
        if signature_root is None or not signature_specs:
            raise ValueError(
                "signature_intersection ROI builder items require paths.signature_dir and paths.signature_maps."
            )
        signature_paths = discover_signature_files(signature_root, signature_specs)
        surface_values = _signature_surface_values(
            signature_paths=signature_paths,
            subjects_dir=subjects_dir,
            template_subject=cfg.template_subject,
        )
    atlas_labels = _atlas_label_map(
        subjects_dir=subjects_dir,
        template_subject=cfg.template_subject,
        parcellation=cfg.parcellation,
    )
    hemisphere_triangles = {
        "lh": _read_template_surface_triangles(
            subjects_dir=subjects_dir,
            template_subject=cfg.template_subject,
            hemisphere="lh",
        ),
        "rh": _read_template_surface_triangles(
            subjects_dir=subjects_dir,
            template_subject=cfg.template_subject,
            hemisphere="rh",
        ),
    }

    label_dir = output_dir / cfg.template_subject / "label"
    ensure_dir(label_dir)
    provenance_dir = output_dir / "provenance"
    ensure_dir(provenance_dir)

    built: List[BuiltROI] = []
    manifest_rows: List[Dict[str, Any]] = []
    signature_path_map = {
        name: str(path)
        for name, path in sorted(signature_paths.items())
    }
    for item in cfg.items:
        label, roi_details = _build_single_roi(
            item=item,
            surface_values=surface_values,
            atlas_labels=atlas_labels,
            template_subject=cfg.template_subject,
            abs_weight_threshold=cfg.abs_weight_threshold,
            triangles=hemisphere_triangles,
        )
        if item.hemisphere == "both":
            label_paths = tuple(
                label_dir / f"{hemisphere}.{item.name}.label"
                for hemisphere in ("lh", "rh")
            )
            for hemisphere, label_path in zip(("lh", "rh"), label_paths):
                hemi_vertices = np.asarray(
                    roi_details["per_hemisphere"][hemisphere]["final_vertices"],
                    dtype=int,
                )
                hemi_label = mne.Label(
                    vertices=hemi_vertices,
                    hemi=hemisphere,
                    name=item.name,
                    subject=cfg.template_subject,
                )
                hemi_label.save(str(label_path))
        else:
            label_path = label_dir / f"{item.hemisphere}.{item.name}.label"
            label.save(str(label_path))
            label_paths = (label_path,)
        provenance_path = provenance_dir / f"{item.name}.json"
        provenance_payload = {
            "roi_name": item.name,
            "definition": item.definition,
            "template_subject": cfg.template_subject,
            "parcellation": cfg.parcellation,
            "hemisphere": item.hemisphere,
            "atlas_labels": list(item.atlas_labels),
            "signature_paths": signature_path_map,
            "abs_weight_threshold": float(cfg.abs_weight_threshold),
            "exclude_discordant_vertices": bool(item.exclude_discordant_vertices),
            "min_vertices": int(item.min_vertices),
            "label_paths": [str(path) for path in label_paths],
            **roi_details,
        }
        provenance_path.write_text(
            json.dumps(provenance_payload, indent=2),
            encoding="utf-8",
        )
        built.append(
            BuiltROI(
                name=item.name,
                hemisphere=item.hemisphere,
                label_paths=label_paths,
                provenance_path=provenance_path,
                atlas_labels=item.atlas_labels,
                include=item.include,
                n_vertices=int(roi_details["final_vertex_count"]),
            )
        )
        manifest_rows.append(
            {
                "name": item.name,
                "definition": item.definition,
                "hemisphere": item.hemisphere,
                "label_paths": [str(path) for path in label_paths],
                "provenance_path": str(provenance_path),
                "n_vertices": int(roi_details["final_vertex_count"]),
                "candidate_vertex_count": int(roi_details["candidate_vertex_count"]),
                "post_sign_vertex_count": int(roi_details["post_sign_vertex_count"]),
                "discordant_vertex_count": int(roi_details["discordant_vertex_count"]),
                "component_sizes": list(roi_details["component_sizes"]),
                "kept_component_sizes": list(roi_details["kept_component_sizes"]),
                "atlas_labels": list(item.atlas_labels),
                "include": [asdict(include) for include in item.include],
            }
        )
        roi_logger.info(
            "Built ROI %s (%s): %d vertices -> %s",
            item.name,
            item.hemisphere,
            int(roi_details["final_vertex_count"]),
            ", ".join(str(path) for path in label_paths),
        )

    manifest_path = output_dir / "roi_manifest.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    metadata_path = output_dir / "roi_builder_config.json"
    metadata_path.write_text(
        json.dumps(asdict(cfg), indent=2, default=str),
        encoding="utf-8",
    )
    return built


def built_rois_as_runtime_specs(
    built_rois: Sequence[BuiltROI],
    *,
    template_subject: str,
) -> List[Dict[str, Any]]:
    """Convert built ROI labels into runtime specs for the coupling analysis."""
    return [
        {
            "name": roi.name,
            "template_subject": template_subject,
            "parcellation": None,
            "annot_labels": [],
            "label_files": [str(path) for path in roi.label_paths],
        }
        for roi in built_rois
    ]


__all__ = [
    "ROIBuilderConfig",
    "BuiltROI",
    "build_eeg_bold_rois",
    "built_rois_as_runtime_specs",
]
