"""
Source Localization Visualization
=================================

Plotting functions for 3D source localization brain maps.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import re
import sys
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names


def _normalize_optional_path(value: Any) -> Optional[str]:
    """Normalize optional path-like config values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return str(Path(text).expanduser())


def _auto_detect_subjects_dir_from_deriv_root(config: Any) -> Optional[str]:
    """Auto-detect FreeSurfer SUBJECTS_DIR from configured derivatives root."""
    deriv_root_raw = get_config_value(config, "paths.deriv_root", None)
    deriv_root_str = _normalize_optional_path(deriv_root_raw)
    if deriv_root_str is None:
        return None

    deriv_root = Path(deriv_root_str)
    candidates = [
        deriv_root / "preprocessed" / "fmri" / "sourcedata" / "freesurfer",
        deriv_root / "preprocessed" / "fmri" / "freesurfer",
        deriv_root / "freesurfer",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None


_SOURCE_VIEW_ALIASES: dict[str, str] = {
    "lat": "lat",
    "lateral": "lat",
    "med": "med",
    "medial": "med",
    "fro": "fro",
    "frontal": "fro",
    "cau": "cau",
    "caudal": "cau",
    "dor": "dor",
    "dorsal": "dor",
    "par": "par",
    "parietal": "par",
    "ros": "ros",
    "rostral": "ros",
    "ven": "ven",
    "ventral": "ven",
}


def _resolve_source_views(raw_views: Any) -> list[str]:
    """Resolve source-plot views to MNE-compatible short view codes."""
    if isinstance(raw_views, str):
        tokens = [tok for tok in re.split(r"[\s,]+", raw_views.strip()) if tok]
    elif isinstance(raw_views, (list, tuple)):
        tokens = [str(item).strip() for item in raw_views if str(item).strip()]
    else:
        raise TypeError(
            f"source views must be string/list/tuple, got {type(raw_views).__name__}."
        )

    if not tokens:
        raise ValueError("source views list cannot be empty.")

    resolved: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        key = token.strip().lower()
        mapped = _SOURCE_VIEW_ALIASES.get(key)
        if mapped is None:
            unknown.append(token)
            continue
        if mapped not in resolved:
            resolved.append(mapped)

    if unknown:
        allowed = ", ".join(sorted(_SOURCE_VIEW_ALIASES.keys()))
        raise ValueError(
            f"Invalid source view(s): {', '.join(unknown)}. Allowed values: {allowed}."
        )

    return resolved


def _resolve_source_plot_subjects_dir(config: Any, logger: logging.Logger) -> Optional[str]:
    """Resolve subjects_dir for source plotting with deterministic priority."""
    configured_candidates = [
        get_config_value(config, "plotting.plots.features.sourcelocalization.subjects_dir", None),
        get_config_value(config, "feature_engineering.sourcelocalization.subjects_dir", None),
        get_config_value(config, "paths.freesurfer_dir", None),
        get_config_value(config, "fmri_preprocessing.fmriprep.fs_subjects_dir", None),
    ]
    for raw_value in configured_candidates:
        resolved = _normalize_optional_path(raw_value)
        if resolved is not None:
            return resolved

    detected = _auto_detect_subjects_dir_from_deriv_root(config)
    if detected is not None:
        logger.info("Auto-detected subjects_dir for source plotting: %s", detected)
    return detected


def _resolve_fs_subject_name(subject: str, subjects_dir: Optional[str]) -> str:
    """Resolve FreeSurfer subject folder name from pipeline subject ID."""
    subject_clean = str(subject).strip()
    if not subject_clean:
        raise ValueError("Subject identifier is required for source plotting.")
    if subjects_dir is None:
        return subject_clean

    fs_root = Path(subjects_dir)
    if not fs_root.exists():
        return subject_clean

    candidates: List[str] = [subject_clean]
    if subject_clean.startswith("sub-"):
        candidates.append(subject_clean[4:])
    else:
        candidates.append(f"sub-{subject_clean}")

    for candidate in candidates:
        if (fs_root / candidate).is_dir():
            return candidate
    return subject_clean


def _build_3d_backend_error_message() -> str:
    """Create a precise install hint for MNE 3D backend issues."""
    python_exe = sys.executable

    def _is_importable(module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
            return True
        except Exception:
            return False

    qt_bindings = [name for name in ("PyQt6", "PySide6", "PyQt5", "PySide2") if _is_importable(name)]
    qt_bindings_str = ", ".join(qt_bindings) if qt_bindings else "none"
    qtpy_present = _is_importable("qtpy")
    pyvistaqt_present = _is_importable("pyvistaqt")

    install_cmd = f"{python_exe} -m pip install pyvista pyvistaqt qtpy PySide6"
    return (
        "No usable MNE 3D backend for source plotting.\n"
        f"Python executable: {python_exe}\n"
        f"Detected qtpy: {'yes' if qtpy_present else 'no'}\n"
        f"Detected pyvistaqt: {'yes' if pyvistaqt_present else 'no'}\n"
        f"Detected Qt bindings: {qt_bindings_str}\n"
        f"Install into this interpreter: `{install_cmd}`"
    )


def _parse_stc_identifiers(stc_path: Path) -> tuple[str, str, str]:
    """Parse subject, task, and method from STC filename."""
    match = re.match(
        r"^sub-(?P<subject>[^_]+)_task-(?P<task>[^_]+)(?:_seg-.+?)?_cond-[^_]+_band-[^_]+_(?P<method>[^_]+)-[a-z]{2}\.stc$",
        stc_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected STC filename format: {stc_path.name}")
    return match.group("subject"), match.group("task"), match.group("method")


def _parse_stc_plot_metadata(stc_path: Path) -> dict[str, Optional[str]]:
    """Parse detailed STC plot metadata from filename."""
    match = re.match(
        r"^sub-(?P<subject>[^_]+)_task-(?P<task>[^_]+)(?:_seg-(?P<segment>.+?))?_cond-(?P<condition>[^_]+)_band-(?P<band>[^_]+)_(?P<method>[^_]+)-[a-z]{2}\.stc$",
        stc_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected STC filename format: {stc_path.name}")
    return {
        "subject": match.group("subject"),
        "task": match.group("task"),
        "segment": match.group("segment"),
        "condition": match.group("condition"),
        "band": match.group("band"),
        "method": match.group("method"),
    }


def _filter_stc_files_by_segment(stc_files: List[Path], segment: str) -> List[Path]:
    """Filter STC files to one segment, raising if segment is unavailable."""
    segment_label = str(segment).strip()
    if not segment_label:
        return stc_files

    filtered: List[Path] = []
    available_segments: set[str] = set()
    for stc_path in stc_files:
        metadata = _parse_stc_plot_metadata(stc_path)
        stc_segment = metadata["segment"]
        if stc_segment is None:
            continue
        available_segments.add(str(stc_segment))
        if stc_segment == segment_label:
            filtered.append(stc_path)

    if filtered:
        return filtered

    available_text = ", ".join(sorted(available_segments)) if available_segments else "<none>"
    raise ValueError(
        f"No STC files found for requested source segment '{segment_label}'. "
        f"Available segments: {available_text}."
    )


def _resolve_source_plot_condition_pair(
    config: Any,
    available_conditions: set[str],
    logger: Optional[logging.Logger] = None,
) -> tuple[str, str]:
    """Resolve condition labels for contrast plotting."""
    cond_a = str(
        get_config_value(config, "feature_engineering.sourcelocalization.contrast.condition_a", "0.0")
    ).strip()
    cond_b = str(
        get_config_value(config, "feature_engineering.sourcelocalization.contrast.condition_b", "1.0")
    ).strip()
    if not cond_a or not cond_b:
        raise ValueError("Source plot contrast conditions must be non-empty.")
    if cond_a != cond_b:
        return cond_a, cond_b

    ordered_conditions: list[str]
    try:
        ordered_conditions = sorted(available_conditions, key=lambda value: float(value))
    except Exception:
        ordered_conditions = sorted(available_conditions)
    if len(ordered_conditions) == 2:
        inferred_a, inferred_b = ordered_conditions
        if logger is not None:
            logger.warning(
                "Source plot contrast conditions were identical in config ('%s'). "
                "Using inferred pair from STC files: '%s' vs '%s'.",
                cond_a,
                inferred_a,
                inferred_b,
            )
        return inferred_a, inferred_b

    available_text = ", ".join(ordered_conditions) if ordered_conditions else "<none>"
    raise ValueError(
        "Source plot contrast conditions must differ. "
        f"Configured values were both '{cond_a}'. Available STC conditions: {available_text}."
    )


def _build_discrete_condition_differences(
    stc_map: dict[Path, Any],
    condition_a: str,
    condition_b: str,
) -> list[dict[str, Any]]:
    """Build condition-difference STCs for each (task, segment, band, method)."""
    grouped: dict[tuple[str, Optional[str], str, str], dict[str, Any]] = {}
    for path, stc in stc_map.items():
        metadata = _parse_stc_plot_metadata(path)
        key = (
            str(metadata["task"]),
            metadata["segment"],
            str(metadata["band"]),
            str(metadata["method"]),
        )
        slot = grouped.setdefault(key, {})
        cond = str(metadata["condition"])
        if cond in slot:
            raise ValueError(
                f"Duplicate STC for condition '{cond}' in group {key}: {path.name}"
            )
        slot[cond] = stc

    diff_records: list[dict[str, Any]] = []
    for (task, segment, band, method), cond_map in grouped.items():
        if condition_a not in cond_map or condition_b not in cond_map:
            continue
        stc_a = cond_map[condition_a]
        stc_b = cond_map[condition_b]
        if stc_a.data.shape != stc_b.data.shape:
            raise ValueError(
                f"Condition STC shape mismatch for band '{band}' segment '{segment}': "
                f"{stc_a.data.shape} vs {stc_b.data.shape}"
            )
        if len(stc_a.vertices) != len(stc_b.vertices):
            raise ValueError(
                f"Condition STC vertex structure mismatch for band '{band}' segment '{segment}'."
            )
        for verts_a, verts_b in zip(stc_a.vertices, stc_b.vertices):
            if not np.array_equal(np.asarray(verts_a), np.asarray(verts_b)):
                raise ValueError(
                    f"Condition STC vertices mismatch for band '{band}' segment '{segment}'."
                )

        diff_stc = stc_b.copy()
        diff_stc.data = np.asarray(stc_b.data, dtype=float) - np.asarray(stc_a.data, dtype=float)
        diff_records.append(
            {
                "task": task,
                "segment": segment,
                "band": band,
                "method": method,
                "stc": diff_stc,
            }
        )

    return diff_records


def _discrete_group_key(task: str, segment: Optional[str], band: str, method: str) -> tuple[str, Optional[str], str, str]:
    """Build a stable grouping key for discrete STC comparisons."""
    return (str(task), segment, str(band), str(method))


def _compute_discrete_group_limits(stc_map: dict[Path, Any]) -> dict[tuple[str, Optional[str], str, str], tuple[float, float]]:
    """Compute per-group color limits for discrete condition maps."""
    grouped_values: dict[tuple[str, Optional[str], str, str], list[np.ndarray]] = {}
    for path, stc in stc_map.items():
        metadata = _parse_stc_plot_metadata(path)
        key = _discrete_group_key(
            task=str(metadata["task"]),
            segment=metadata["segment"],
            band=str(metadata["band"]),
            method=str(metadata["method"]),
        )
        grouped_values.setdefault(key, []).append(_summarize_stc_values_for_discrete_plot(stc))

    limits: dict[tuple[str, Optional[str], str, str], tuple[float, float]] = {}
    for key, value_arrays in grouped_values.items():
        vmin = float(min(np.nanmin(values) for values in value_arrays))
        vmax = float(max(np.nanmax(values) for values in value_arrays))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError(
                f"Invalid non-finite source values encountered for group {key}."
            )
        if vmax <= vmin:
            raise ValueError(
                f"Degenerate source color scale for group {key}; STC values do not vary."
            )
        limits[key] = (vmin, vmax)
    return limits


def _compute_discrete_difference_absmax(
    diff_records: list[dict[str, Any]],
) -> dict[tuple[str, Optional[str], str, str], float]:
    """Compute per-group symmetric limits for discrete condition-difference maps."""
    diff_absmax: dict[tuple[str, Optional[str], str, str], float] = {}
    for rec in diff_records:
        key = _discrete_group_key(
            task=str(rec["task"]),
            segment=rec["segment"],
            band=str(rec["band"]),
            method=str(rec["method"]),
        )
        values = _summarize_stc_values_for_discrete_plot(rec["stc"])
        abs_max = float(np.nanmax(np.abs(values)))
        if not np.isfinite(abs_max) or abs_max <= 0.0:
            raise ValueError(
                f"Condition-difference STC map has no dynamic range for group {key}."
            )
        diff_absmax[key] = abs_max
    return diff_absmax


def _detect_stc_kind(stc_files: List[Path]) -> str:
    """Detect STC rendering kind by inspecting the first non-empty STC file.

    Returns one of 'surface', 'volume', or 'discrete'.
    Using the STC object type is authoritative; we never rely on the
    separately-saved src.fif kind, which can be stale or mismatched.
    """
    vol_files = [p for p in stc_files if p.name.endswith("-vl.stc")]
    probe_path = vol_files[0] if vol_files else stc_files[0]
    stc = mne.read_source_estimate(str(probe_path))
    stc_type = type(stc).__name__  # 'SourceEstimate', 'VolSourceEstimate', 'MixedSourceEstimate'
    if stc_type == "SourceEstimate":
        return "surface"
    return "volume"  # VolSourceEstimate; discrete is a sub-variant we verify via src.fif


def _load_volume_source_space(stc_files: List[Path]) -> Optional[Any]:
    """Load source space from the saved *-src.fif, only for volume/discrete rendering."""
    vol_files = [p for p in stc_files if p.name.endswith("-vl.stc")]
    if not vol_files:
        return None
    subject, task, method = _parse_stc_identifiers(vol_files[0])
    src_path = vol_files[0].parent / f"sub-{subject}_task-{task}_{method}-src.fif"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Missing source space for volumetric STC plotting: {src_path}. "
            "Recompute source features with --source-save-stc so '*-src.fif' is emitted."
        )
    return mne.read_source_spaces(str(src_path))


def _summarize_stc_values_for_discrete_plot(stc: Any) -> np.ndarray:
    """Reduce STC time dimension to one value per source for point-cloud plotting."""
    data = np.asarray(stc.data, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("STC data must be a non-empty 2D array for discrete source plotting.")
    if data.shape[1] == 1:
        return data[:, 0]
    return np.nanmax(np.abs(data), axis=1)


def _plot_discrete_stc_3d_points(
    stc: Any,
    src: Any,
    save_path: Path,
    title: str,
    mesh_rr_mm: np.ndarray,
    mesh_tris: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    """Render fMRI-constrained discrete STC points across canonical 3D views."""
    if len(src) != 1:
        raise ValueError(f"Discrete source space must contain exactly one volume block (got {len(src)}).")

    vertices = np.asarray(stc.vertices[0], dtype=int)
    rr = np.asarray(src[0]["rr"], dtype=float)
    if vertices.size == 0:
        raise ValueError("No vertices available in STC for discrete source plotting.")
    if np.max(vertices) >= rr.shape[0]:
        raise ValueError("STC vertex index exceeds available source-space coordinates.")

    coords_mm = rr[vertices] * 1000.0
    values = _summarize_stc_values_for_discrete_plot(stc)
    if values.shape[0] != vertices.size:
        raise ValueError(
            f"STC/source mismatch for discrete plotting: values={values.shape[0]}, vertices={vertices.size}."
        )

    order = np.argsort(np.abs(values))
    coords_mm = coords_mm[order]
    values = values[order]

    mesh_min = np.min(mesh_rr_mm, axis=0)
    mesh_max = np.max(mesh_rr_mm, axis=0)
    mesh_center = 0.5 * (mesh_min + mesh_max)
    mesh_radius = 0.5 * float(np.max(mesh_max - mesh_min))
    if mesh_radius <= 0:
        mesh_radius = 80.0

    fig = plt.figure(figsize=(14, 10))
    view_specs = [
        ("Left lateral", 10, 180),
        ("Right lateral", 10, 0),
        ("Dorsal", 90, -90),
        ("Posterior", 5, 90),
    ]
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    scatter = None
    for ax, (label, elev, azim) in zip(axes, view_specs):
        ax.plot_trisurf(
            mesh_rr_mm[:, 0],
            mesh_rr_mm[:, 1],
            mesh_rr_mm[:, 2],
            triangles=mesh_tris,
            color="#bcbcbc",
            alpha=0.11,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        scatter = ax.scatter(
            coords_mm[:, 0],
            coords_mm[:, 1],
            coords_mm[:, 2],
            c=values,
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            s=16.0,
            alpha=0.95,
            linewidths=0.0,
        )
        ax.set_xlim(mesh_center[0] - mesh_radius, mesh_center[0] + mesh_radius)
        ax.set_ylim(mesh_center[1] - mesh_radius, mesh_center[1] + mesh_radius)
        ax.set_zlim(mesh_center[2] - mesh_radius, mesh_center[2] + mesh_radius)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.set_axis_off()
        ax.set_title(label, fontsize=10, pad=2)

    if scatter is None:
        raise RuntimeError("Discrete source plotting failed to create a scatter artist.")
    peak_index = int(np.argmax(values))
    peak_coord = coords_mm[peak_index]
    fig.colorbar(scatter, ax=axes, fraction=0.03, pad=0.03, label=colorbar_label)
    fig.suptitle(
        f"{title}\n(fMRI-constrained discrete source space)",
        fontsize=13,
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        f"Peak coordinate (MRI RAS, mm): x={peak_coord[0]:.1f}, y={peak_coord[1]:.1f}, z={peak_coord[2]:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.03, 0.95, 0.95))
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_discrete_stc_orthogonal_projections(
    stc: Any,
    src: Any,
    save_path: Path,
    title: str,
    mesh_rr_mm: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    """Render orthogonal slice-style projections for a discrete source space."""
    if len(src) != 1:
        raise ValueError(
            f"Discrete source space must contain exactly one volume block (got {len(src)})."
        )

    vertices = np.asarray(stc.vertices[0], dtype=int)
    rr = np.asarray(src[0]["rr"], dtype=float)
    if vertices.size == 0:
        raise ValueError("No vertices available in STC for discrete orthogonal plotting.")
    if np.max(vertices) >= rr.shape[0]:
        raise ValueError("STC vertex index exceeds available source-space coordinates.")

    coords_mm = rr[vertices] * 1000.0
    values = _summarize_stc_values_for_discrete_plot(stc)
    if values.shape[0] != vertices.size:
        raise ValueError(
            f"STC/source mismatch for discrete orthogonal plotting: values={values.shape[0]}, vertices={vertices.size}."
        )

    peak_idx = int(np.argmax(values))
    peak = coords_mm[peak_idx]
    slab_thickness_mm = 4.0

    fig, all_axes = plt.subplots(
        1,
        4,
        figsize=(16, 5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.04]},
    )
    axes = all_axes[:3]
    colorbar_ax = all_axes[3]
    view_specs = [
        ("Sagittal", 0, ("y", "z"), "x"),
        ("Coronal", 1, ("x", "z"), "y"),
        ("Axial", 2, ("x", "y"), "z"),
    ]
    label_to_idx = {"x": 0, "y": 1, "z": 2}
    scatter_ref = None

    for ax, (label, fixed_idx, (axis_a, axis_b), fixed_name) in zip(axes, view_specs):
        mesh_mask = np.abs(mesh_rr_mm[:, fixed_idx] - peak[fixed_idx]) <= slab_thickness_mm
        mesh_slice = mesh_rr_mm[mesh_mask]
        if mesh_slice.size > 0:
            ax.scatter(
                mesh_slice[:, label_to_idx[axis_a]],
                mesh_slice[:, label_to_idx[axis_b]],
                s=0.2,
                color="#b8b8b8",
                alpha=0.28,
                linewidths=0.0,
                zorder=1,
            )

        src_mask = np.abs(coords_mm[:, fixed_idx] - peak[fixed_idx]) <= slab_thickness_mm
        if np.any(src_mask):
            scatter_ref = ax.scatter(
                coords_mm[src_mask, label_to_idx[axis_a]],
                coords_mm[src_mask, label_to_idx[axis_b]],
                c=values[src_mask],
                cmap=cmap,
                vmin=float(vmin),
                vmax=float(vmax),
                s=18.0,
                alpha=0.95,
                linewidths=0.0,
                zorder=2,
            )
        else:
            scatter_ref = ax.scatter([], [], c=[], cmap=cmap, vmin=float(vmin), vmax=float(vmax))

        ax.set_xlabel(f"{axis_a.upper()} (mm)")
        ax.set_ylabel(f"{axis_b.upper()} (mm)")
        ax.set_title(
            f"{label}\n({fixed_name}={peak[fixed_idx]:.1f} mm, +/-{slab_thickness_mm:.1f} mm)"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    if scatter_ref is None:
        raise RuntimeError("Discrete orthogonal plotting failed to create a scatter artist.")

    fig.colorbar(scatter_ref, cax=colorbar_ax, label=colorbar_label)
    fig.suptitle(
        f"{title}\n(fMRI-constrained discrete source space: orthogonal projections)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93))
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_discrete_context_mesh(subjects_dir: str, fs_subject: str) -> tuple[np.ndarray, np.ndarray]:
    """Load cortical pial mesh as anatomical context for discrete-source plots."""
    surf_dir = Path(subjects_dir) / fs_subject / "surf"
    lh_path = surf_dir / "lh.pial"
    rh_path = surf_dir / "rh.pial"
    if not lh_path.exists() or not rh_path.exists():
        raise FileNotFoundError(
            "Discrete source plotting requires FreeSurfer pial surfaces at: "
            f"{lh_path} and {rh_path}"
        )

    lh_rr, lh_tris = mne.read_surface(str(lh_path))
    rh_rr, rh_tris = mne.read_surface(str(rh_path))
    lh_rr = np.asarray(lh_rr, dtype=float)
    lh_tris = np.asarray(lh_tris, dtype=int)
    rh_rr = np.asarray(rh_rr, dtype=float)
    rh_tris = np.asarray(rh_tris, dtype=int)
    if lh_rr.shape[1] != 3 or rh_rr.shape[1] != 3:
        raise ValueError(
            f"Invalid pial mesh vertex geometry (lh={lh_rr.shape}, rh={rh_rr.shape})."
        )
    if lh_tris.shape[1] != 3 or rh_tris.shape[1] != 3:
        raise ValueError(
            f"Invalid pial mesh triangle geometry (lh={lh_tris.shape}, rh={rh_tris.shape})."
        )

    rr = np.vstack([lh_rr, rh_rr])
    tris = np.vstack([lh_tris, rh_tris + lh_rr.shape[0]])
    if np.nanmax(np.abs(rr)) < 1.0:
        rr = rr * 1000.0
    return rr, tris


def _plot_surface_stc_canonical(
    stc: Any,
    *,
    subject: str,
    subjects_dir: Optional[str],
    size: tuple,
) -> Any:
    """Render a surface STC as a publication-quality 2×2 brain grid.

    Uses MNE's compute_source_morph to perform geodesic smoothing along the
    cortical mesh, spreading sparse source-space values (~4k vertices) to the
    full cortical surface (~160k vertices). This is the same algorithm MNE
    uses internally in stc.plot() and is considered the gold standard for
    source estimate visualization. Layout:

        LH lateral  |  RH lateral
        LH medial   |  RH medial
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from nilearn import plotting as nipl
    except ImportError:
        raise RuntimeError(
            "nilearn is required for quality surface STC rendering. "
            "Install with: pip install nilearn"
        )

    fs_dir = Path(subjects_dir) / subject if subjects_dir else None
    if fs_dir is None or not fs_dir.exists():
        raise FileNotFoundError(f"FreeSurfer directory not found: {fs_dir}")

    # Geodesic smoothing: morph sparse STC to full-resolution cortical surface.
    # subject_from == subject_to with spacing=None expands to all vertices
    # while smooth=10 performs 10 iterations of nearest-neighbor averaging
    # along the cortical mesh, respecting sulcal boundaries.
    morph = mne.compute_source_morph(
        stc,
        subject_from=subject,
        subject_to=subject,
        subjects_dir=subjects_dir,
        smooth=10,
        spacing=None,
        verbose=False,
    )
    stc_full = morph.apply(stc)

    # Mean absolute source power across time
    vals = np.abs(
        stc_full.data.mean(axis=1) if stc_full.data.shape[1] > 1 else stc_full.data[:, 0]
    )
    n_lh = len(stc_full.vertices[0])

    # Percentile-based thresholds: show top ~35% of activation
    active = vals[np.isfinite(vals) & (vals > 0)]
    vmin = float(np.percentile(active, 65)) if active.size else 0.0
    vmax = float(np.percentile(active, 99.5)) if active.size else 1.0
    if vmax <= vmin:
        vmax = vmin * 2.0 + 1e-30

    lh_tex = vals[:n_lh]
    rh_tex = vals[n_lh:]

    # (texture, mesh, sulc, hemi, view, label)
    panels_cfg = [
        (lh_tex, "lh.inflated", "lh.sulc", "left",  "lateral", "LH — Lateral"),
        (rh_tex, "rh.inflated", "rh.sulc", "right", "lateral", "RH — Lateral"),
        (lh_tex, "lh.inflated", "lh.sulc", "left",  "medial",  "LH — Medial"),
        (rh_tex, "rh.inflated", "rh.sulc", "right", "medial",  "RH — Medial"),
    ]

    imgs: list[np.ndarray] = []
    for tex, mesh_f, sulc_f, hemi, view, label in panels_cfg:
        pfig = nipl.plot_surf_stat_map(
            surf_mesh=str(fs_dir / "surf" / mesh_f),
            stat_map=tex,
            bg_map=str(fs_dir / "surf" / sulc_f),
            hemi=hemi,
            view=view,
            colorbar=False,
            cmap="hot",
            vmax=vmax,
            threshold=vmin,
            bg_on_data=True,
            darkness=0.5,
            title=label,
            title_font_size=12,
        )
        pfig.set_facecolor("white")
        buf = io.BytesIO()
        pfig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        imgs.append(plt.imread(buf))
        plt.close(pfig)

    # Pad all panels to identical dimensions before concatenation
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)

    def _pad(img: np.ndarray) -> np.ndarray:
        ph, pw = max_h - img.shape[0], max_w - img.shape[1]
        return np.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)),
                      mode="constant", constant_values=1.0)

    imgs = [_pad(img) for img in imgs]
    grid = np.concatenate([
        np.concatenate(imgs[:2], axis=1),
        np.concatenate(imgs[2:], axis=1),
    ], axis=0)

    # Build output figure with shared colorbar
    dpi = 150
    w_in = grid.shape[1] / dpi
    h_in = grid.shape[0] / dpi
    out_fig = plt.figure(figsize=(w_in + 1.0, h_in), dpi=dpi, facecolor="white")
    gs = out_fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0.03)

    ax_img = out_fig.add_subplot(gs[0])
    ax_img.imshow(grid, interpolation="lanczos")
    ax_img.axis("off")

    ax_cb = out_fig.add_subplot(gs[1])
    cb = out_fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="hot"),
        cax=ax_cb,
    )
    cb.set_label("Source power (a.u.)", fontsize=9, labelpad=8)
    cb.ax.tick_params(labelsize=8)

    out_fig.tight_layout(pad=0.4)
    return out_fig


def plot_source_stc_3d(
    subject: str,
    stc_files: List[Path],
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    subjects_dir: Optional[str] = None
) -> None:
    """Plot 3D brain maps from saved STC files."""
    if not stc_files:
        logger.info("No STC files provided for source plotting.")
        return

    source_segment_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.segment", None
    )
    source_segment = "" if source_segment_raw is None else str(source_segment_raw).strip()
    if source_segment:
        stc_files = _filter_stc_files_by_segment(stc_files, source_segment)
        logger.info(
            "Filtered source STCs to segment '%s' (%d files).",
            source_segment,
            len(stc_files),
        )

    if subjects_dir is None:
        subjects_dir = _resolve_source_plot_subjects_dir(config, logger)
    
    if subjects_dir is None:
        logger.warning("subjects_dir is not configured. 3D source plotting may fail if fsaverage or subject MRI is missing.")

    fs_subject = _resolve_fs_subject_name(subject, subjects_dir)
    if subjects_dir is not None:
        logger.info(
            "Using FreeSurfer subject '%s' from subjects_dir '%s'.",
            fs_subject,
            subjects_dir,
        )

    out_dir = save_dir / "3d_brains"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Plotting 3D brains for {len(stc_files)} STC combinations...")

    # Detect rendering mode from the STC object type rather than the src.fif kind,
    # which can be stale or mismatched with actual STC data.
    vol_stc_files = [p for p in stc_files if p.name.endswith("-vl.stc")]
    stc_kind = _detect_stc_kind(stc_files) if vol_stc_files else "surface"
    if stc_kind == "surface":
        logger.info(
            "Using canonical surface rendering for EEG-only source STCs "
            "(hemi=both, view=lat, cortex=classic)."
        )
    else:
        hemi = "both"
        views_list = ["lat"]
        cortex = "classic"

    volume_src: Optional[Any] = None
    discrete_mesh_rr_mm: Optional[np.ndarray] = None
    discrete_mesh_tris: Optional[np.ndarray] = None
    discrete_stc_map: dict[Path, Any] = {}
    discrete_condition_limits: dict[tuple[str, Optional[str], str, str], tuple[float, float]] = {}
    discrete_diff_records: list[dict[str, Any]] = []
    condition_a: Optional[str] = None
    condition_b: Optional[str] = None
    discrete_diff_absmax: dict[tuple[str, Optional[str], str, str], float] = {}

    if stc_kind == "volume":
        # Load src.fif and check whether it describes a discrete (fMRI-constrained)
        # or regular volumetric source space.
        volume_src = _load_volume_source_space(stc_files)
        if volume_src is not None:
            src_space_kind = str(volume_src.kind)
        else:
            src_space_kind = "volume"

        if src_space_kind == "surface":
            # VolSourceEstimate paired with a surface src.fif — these are stale artifacts
            # from an incompatible earlier run. They cannot be rendered correctly.
            logger.warning(
                "Source estimates are VolSourceEstimate but the saved source space (src.fif) is a "
                "surface source space — these files are stale from an incompatible pipeline run. "
                "Rerun 'features compute --source-save-stc' to regenerate compatible STC files. "
                "Skipping all volume STC files for this subject."
            )
            stc_files = [p for p in stc_files if not p.name.endswith("-vl.stc")]
            stc_kind = "surface"  # redirect remaining surface files through surface path
        elif src_space_kind == "discrete":
            if subjects_dir is None:
                raise ValueError(
                    "Discrete source plotting requires a valid subjects_dir for anatomical context."
                )
            discrete_mesh_rr_mm, discrete_mesh_tris = _load_discrete_context_mesh(
                subjects_dir=subjects_dir,
                fs_subject=fs_subject,
            )
            logger.warning(
                "source_localization_3d is rendering a discrete fMRI-constrained source space. "
                "Only constrained source points are shown; this is not a dense whole-brain volumetric map."
            )
            discrete_stc_map = {path: mne.read_source_estimate(str(path)) for path in stc_files}
            discrete_condition_limits = _compute_discrete_group_limits(discrete_stc_map)

            available_conditions = {
                str(_parse_stc_plot_metadata(path)["condition"])
                for path in discrete_stc_map.keys()
            }
            condition_a, condition_b = _resolve_source_plot_condition_pair(
                config,
                available_conditions=available_conditions,
                logger=logger,
            )
            discrete_diff_records = _build_discrete_condition_differences(
                discrete_stc_map,
                condition_a=condition_a,
                condition_b=condition_b,
            )
            if discrete_diff_records:
                discrete_diff_absmax = _compute_discrete_difference_absmax(discrete_diff_records)
                logger.info(
                    "Rendering %d condition-difference maps (%s minus %s) with per-(segment,band) limits.",
                    len(discrete_diff_records),
                    condition_b,
                    condition_a,
                )
            else:
                logger.warning(
                    "No matched condition pairs found for difference plotting (%s vs %s).",
                    condition_a,
                    condition_b,
                )
            logger.info("Using per-(segment,band) condition color limits for discrete STC maps.")
            logger.info("Using discrete source-space point-cloud rendering for volumetric STCs.")
        else:
            # Regular volumetric source space — needs MNE 3D backend in offscreen mode.
            backend = mne.viz.get_3d_backend()
            if not backend:
                raise RuntimeError(_build_3d_backend_error_message())

    failures: List[str] = []
    for stc_path in stc_files:
        try:
            # We assume the STC path stem has a specific naming convention we set during feature computation
            # E.g., sub-0000_task-..._cond-..._band-..._lcmv
            stc = (
                discrete_stc_map.get(stc_path)
                if stc_kind == "volume" and src_space_kind == "discrete"
                else mne.read_source_estimate(str(stc_path))
            )
            if stc is None:
                raise RuntimeError(f"Discrete STC cache missing for {stc_path.name}.")

            # Volume STCs (-vl.stc) are rendered differently by source-space kind:
            # regular volume source spaces use MNE 3D brain rendering, while
            # discrete source spaces are shown as 3D point clouds.
            if stc_path.name.endswith("-vl.stc") and stc_kind == "volume":
                image_name = stc_path.name.replace("-vl.stc", "")
                save_path = out_dir / f"{image_name}_3d.png"
                if src_space_kind == "discrete":
                    if discrete_mesh_rr_mm is None or discrete_mesh_tris is None:
                        raise RuntimeError("Discrete source plotting mesh was not initialized.")
                    metadata = _parse_stc_plot_metadata(stc_path)
                    group_key = _discrete_group_key(
                        task=str(metadata["task"]),
                        segment=metadata["segment"],
                        band=str(metadata["band"]),
                        method=str(metadata["method"]),
                    )
                    if group_key not in discrete_condition_limits:
                        raise RuntimeError(
                            f"Missing condition scale for STC group {group_key}."
                        )
                    discrete_condition_vmin, discrete_condition_vmax = discrete_condition_limits[group_key]
                    _plot_discrete_stc_3d_points(
                        stc=stc,
                        src=volume_src,
                        save_path=save_path,
                        title=image_name,
                        mesh_rr_mm=discrete_mesh_rr_mm,
                        mesh_tris=discrete_mesh_tris,
                        cmap="hot",
                        vmin=discrete_condition_vmin,
                        vmax=discrete_condition_vmax,
                        colorbar_label="Source amplitude (a.u.)",
                    )
                    orth_save_path = out_dir / f"{image_name}_orthogonal.png"
                    _plot_discrete_stc_orthogonal_projections(
                        stc=stc,
                        src=volume_src,
                        save_path=orth_save_path,
                        title=image_name,
                        mesh_rr_mm=discrete_mesh_rr_mm,
                        cmap="hot",
                        vmin=discrete_condition_vmin,
                        vmax=discrete_condition_vmax,
                        colorbar_label="Source amplitude (a.u.)",
                    )
                    if logger:
                        logger.debug(f"Saved 3D source plot: {save_path.name}")
                        logger.debug(f"Saved orthogonal source plot: {orth_save_path.name}")
                    continue
                brain = stc.plot_3d(
                    subject=fs_subject,
                    subjects_dir=subjects_dir,
                    src=volume_src,
                    hemi=hemi,
                    views=views_list,
                    background="white",
                    time_viewer=False,
                    show_traces=False,
                    size=(1200, 600),
                    cortex=cortex,
                    brain_kwargs={"show": False},
                )
            else:
                brain = _plot_surface_stc_canonical(
                    stc,
                    subject=fs_subject,
                    subjects_dir=subjects_dir,
                    size=(1200, 600),
                )

            image_name = stc_path.name.replace("-lh.stc", "").replace("-rh.stc", "").replace("-vl.stc", "")
            save_path = out_dir / f"{image_name}_3d.png"

            import matplotlib.figure as _mpl_fig
            import matplotlib.pyplot as _plt
            if isinstance(brain, _mpl_fig.Figure):
                brain.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="white")
                _plt.close(brain)
            else:
                brain.save_image(str(save_path))
                brain.close()

            
            if logger:
                logger.debug(f"Saved 3D source plot: {save_path.name}")
                
        except Exception as exc:
            failures.append(stc_path.name)
            if logger:
                logger.error(f"Failed to plot STC 3D brain for {stc_path.name}: {exc}")

    if stc_kind == "volume" and src_space_kind == "discrete" and discrete_diff_records:
        if condition_a is None or condition_b is None:
            raise RuntimeError("Discrete contrast plotting state is incomplete.")
        for diff_record in discrete_diff_records:
            try:
                task = str(diff_record["task"])
                segment = diff_record["segment"]
                band = str(diff_record["band"])
                method = str(diff_record["method"])
                diff_stc = diff_record["stc"]
                segment_token = f"_seg-{segment}" if segment else ""
                image_name = (
                    f"sub-{subject}_task-{task}{segment_token}_contrast-cond-{condition_b}-minus-{condition_a}_"
                    f"band-{band}_{method}"
                )
                group_key = _discrete_group_key(task=task, segment=segment, band=band, method=method)
                if group_key not in discrete_diff_absmax:
                    raise RuntimeError(
                        f"Missing contrast scale for STC difference group {group_key}."
                    )
                diff_absmax = discrete_diff_absmax[group_key]

                save_path = out_dir / f"{image_name}_3d.png"
                _plot_discrete_stc_3d_points(
                    stc=diff_stc,
                    src=volume_src,
                    save_path=save_path,
                    title=image_name,
                    mesh_rr_mm=discrete_mesh_rr_mm,
                    mesh_tris=discrete_mesh_tris,
                    cmap="RdBu_r",
                    vmin=-diff_absmax,
                    vmax=diff_absmax,
                    colorbar_label=f"Source difference ({condition_b} - {condition_a}, a.u.)",
                )
                orth_save_path = out_dir / f"{image_name}_orthogonal.png"
                _plot_discrete_stc_orthogonal_projections(
                    stc=diff_stc,
                    src=volume_src,
                    save_path=orth_save_path,
                    title=image_name,
                    mesh_rr_mm=discrete_mesh_rr_mm,
                    cmap="RdBu_r",
                    vmin=-diff_absmax,
                    vmax=diff_absmax,
                    colorbar_label=f"Source difference ({condition_b} - {condition_a}, a.u.)",
                )
                if logger:
                    logger.debug(f"Saved 3D contrast plot: {save_path.name}")
                    logger.debug(f"Saved orthogonal contrast plot: {orth_save_path.name}")
            except Exception as exc:
                label = (
                    f"{diff_record.get('task')}/{diff_record.get('segment')}/{diff_record.get('band')}"
                )
                failures.append(f"contrast:{label}")
                if logger:
                    logger.error("Failed to render discrete contrast map (%s): %s", label, exc)

    if failures:
        joined = ", ".join(failures[:3])
        suffix = "" if len(failures) <= 3 else f", ... ({len(failures)} total)"
        raise RuntimeError(f"Failed to render STC 3D plots for: {joined}{suffix}")
