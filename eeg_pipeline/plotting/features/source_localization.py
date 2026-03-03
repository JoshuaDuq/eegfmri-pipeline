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


def _filter_stc_files_by_condition(stc_files: List[Path], condition: str) -> List[Path]:
    """Filter STC files to one condition, raising if condition is unavailable."""
    condition_label = str(condition).strip()
    if not condition_label:
        return stc_files

    filtered: List[Path] = []
    available_conditions: set[str] = set()
    for stc_path in stc_files:
        metadata = _parse_stc_plot_metadata(stc_path)
        stc_condition = metadata["condition"]
        if stc_condition is None:
            continue
        available_conditions.add(str(stc_condition))
        if stc_condition == condition_label:
            filtered.append(stc_path)

    if filtered:
        return filtered

    available_text = ", ".join(sorted(available_conditions)) if available_conditions else "<none>"
    raise ValueError(
        f"No STC files found for requested source condition '{condition_label}'. "
        f"Available conditions: {available_text}."
    )


def _filter_stc_files_by_bands(
    stc_files: List[Path],
    bands: List[str],
    logger: Optional[logging.Logger] = None,
) -> List[Path]:
    """Filter STC files to only those whose band matches any entry in *bands*.

    Filtering is case-insensitive and skips files with unparseable names.
    Logs a warning (rather than raising) when no files survive the filter so
    that the pipeline degrades gracefully for a mistyped band name.
    """
    requested = {b.strip().lower() for b in bands if b.strip()}
    if not requested:
        return stc_files

    filtered: List[Path] = []
    available_bands: set[str] = set()
    for stc_path in stc_files:
        try:
            metadata = _parse_stc_plot_metadata(stc_path)
        except ValueError:
            filtered.append(stc_path)  # keep unparseable files unfiltered
            continue
        stc_band = metadata.get("band")
        if stc_band is None:
            continue
        available_bands.add(str(stc_band))
        if str(stc_band).lower() in requested:
            filtered.append(stc_path)

    if not filtered and logger is not None:
        available_text = ", ".join(sorted(available_bands)) if available_bands else "<none>"
        logger.warning(
            "No STC files matched requested source bands %s. "
            "Available bands: %s. Falling back to all files.",
            list(requested),
            available_text,
        )
        return stc_files

    return filtered


def _has_contrast_conditions(config: Any) -> bool:
    """Return True if both condition_a and condition_b are configured."""
    raw_a = get_config_value(
        config, "feature_engineering.sourcelocalization.contrast.condition_a", None
    )
    raw_b = get_config_value(
        config, "feature_engineering.sourcelocalization.contrast.condition_b", None
    )
    cond_a = str(raw_a).strip() if raw_a is not None else ""
    cond_b = str(raw_b).strip() if raw_b is not None else ""
    return bool(cond_a and cond_b and cond_a.lower() != "none" and cond_b.lower() != "none" and cond_a != cond_b)


def _should_skip_absolute_plot(stc_condition: str, config_source_condition: str, has_contrasts: bool) -> bool:
    """Determine if an absolute (single-condition) plot should be skipped.
    
    Rules:
    - If user explicitly asked for one absolute condition (`config_source_condition`), only plot that one.
    - If user didn't ask for an absolute condition, but DID ask for contrasts, skip all absolute plots.
    - If user didn't ask for anything, plot all conditions absolutely (fallback).
    """
    if config_source_condition:
        return stc_condition != config_source_condition
    if has_contrasts:
        return True
    return False


def _resolve_source_plot_condition_pair(
    config: Any,
    available_conditions: set[str],
    logger: Optional[logging.Logger] = None,
) -> tuple[str, str]:
    """Resolve condition labels for contrast plotting.

    Reads condition_a and condition_b from config.
    Both must be explicitly configured, non-empty, and distinct.
    """
    raw_a = get_config_value(
        config, "feature_engineering.sourcelocalization.contrast.condition_a", None
    )
    raw_b = get_config_value(
        config, "feature_engineering.sourcelocalization.contrast.condition_b", None
    )
    cond_a = str(raw_a).strip() if raw_a is not None else ""
    cond_b = str(raw_b).strip() if raw_b is not None else ""

    is_missing = not cond_a or not cond_b or cond_a.lower() == "none" or cond_b.lower() == "none"
    if is_missing or cond_a == cond_b:
        available_text = ", ".join(sorted(available_conditions)) if available_conditions else "<none>"
        raise ValueError(
            "Source plot contrast conditions must be explicitly configured and distinct. "
            f"Got condition_a={cond_a!r}, condition_b={cond_b!r}. "
            f"Available STC conditions: {available_text}. "
            "Configure via --plot-item-config <plot_id> "
            "source_condition_a <value> source_condition_b <value>, "
            "or set them in the TUI under the plot's per-item config."
        )

    return cond_a, cond_b


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


def _discrete_stc_to_nifti(
    stc: Any,
    src: Any,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
    voxel_size_mm: float = 2.0,
) -> Any:
    """Build a NIfTI from a discrete source estimate.

    MNE's ``stc.as_volume()`` raises ``AssertionError`` on discrete source
    spaces.  This function manually snaps each source coordinate onto a
    regular voxel grid in MRI-RAS space, producing a sparse 3-D NIfTI that
    nilearn can render with ``plot_stat_map``.

    Returns a ``nibabel.Nifti1Image`` with non-zero voxels clamped to
    ``[vmin, vmax]`` (zeros = transparent background in nilearn).
    """
    import nibabel as nib

    if len(src) != 1:
        raise ValueError(
            f"Discrete source space must contain exactly one volume block (got {len(src)})."
        )

    vertices = np.asarray(stc.vertices[0], dtype=int)
    rr_m = np.asarray(src[0]["rr"], dtype=float)
    coords_mm = rr_m[vertices] * 1000.0  # MRI-RAS, millimeters

    # Collapse time dimension: mean across time points.
    data = np.asarray(stc.data, dtype=float)
    values = data.mean(axis=1) if data.ndim == 2 and data.shape[1] > 1 else data[:, 0]

    # Define a bounding box with some padding and build the voxel grid.
    pad = 10.0  # mm of padding around the cluster
    origin = coords_mm.min(axis=0) - pad
    extent = coords_mm.max(axis=0) + pad - origin
    grid_shape = np.ceil(extent / voxel_size_mm).astype(int) + 1

    # Affine: maps (i, j, k) voxel indices → MRI-RAS mm.
    affine = np.eye(4)
    np.fill_diagonal(affine[:3, :3], voxel_size_mm)
    affine[:3, 3] = origin

    # Map source coordinates to voxel indices and fill the volume.
    ijk = np.round((coords_mm - origin) / voxel_size_mm).astype(int)
    vol = np.zeros(tuple(grid_shape), dtype=float)
    for idx, (i, j, k) in enumerate(ijk):
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1] and 0 <= k < grid_shape[2]:
            # If multiple sources map to the same voxel, keep the maximum.
            vol[i, j, k] = max(vol[i, j, k], values[idx])

    # Clamp non-zero voxels; zeros stay zero (nilearn background).
    nonzero = vol != 0.0
    vol[nonzero] = np.clip(vol[nonzero], vmin, vmax)

    return nib.Nifti1Image(vol, affine)

def _plot_discrete_stc_volumetric(
    stc: Any,
    src: Any,
    save_path: Path,
    title: str,
    subjects_dir: str,
    fs_subject: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    symmetric_cbar: bool = False,
) -> None:
    """Render a discrete fMRI-constrained STC as a dense volumetric stat map.

    Uses stc.as_volume(src) to produce a NIfTI image in MRI voxel space, then
    renders publication-quality orthogonal slices (axial, sagittal, coronal) via
    nilearn.plotting.plot_stat_map on the subject's T1.mgz anatomical background.

    The result looks identical to an fMRI activation map: only fMRI-cluster voxels
    carry EEG source values; the rest of the volume is zero (transparent in nilearn).
    """
    try:
        from nilearn import plotting as nipl
        import nibabel as nib
    except ImportError:
        raise RuntimeError(
            "nilearn and nibabel are required for volumetric STC rendering. "
            "Install with: pip install nilearn nibabel"
        )

    # MNE's as_volume() does not support discrete source spaces (AssertionError
    # in _interpolate_data). Build a sparse NIfTI manually instead:
    # snap each source coordinate onto a regular 2 mm grid.
    nii_clamped = _discrete_stc_to_nifti(stc, src, vmin=vmin, vmax=vmax)

    # Use the subject's T1.mgz as anatomical background if available.
    t1_path = Path(subjects_dir) / fs_subject / "mri" / "T1.mgz"
    bg_img = str(t1_path) if t1_path.exists() else "MNI152"

    # Find the peak voxel to center the ortho slices.
    vol_data = np.asarray(nii_clamped.dataobj, dtype=float)
    abs_vol = np.abs(vol_data)
    peak_flat = int(np.argmax(abs_vol))
    peak_ijk = np.unravel_index(peak_flat, vol_data.shape)
    peak_xyz = (nii_clamped.affine @ np.array([*peak_ijk, 1.0]))[:3]
    cut_coords = tuple(float(v) for v in peak_xyz)

    threshold = float(vmin) if vmin > 0 else None

    display = nipl.plot_stat_map(
        nii_clamped,
        bg_img=bg_img,
        display_mode="ortho",
        cut_coords=cut_coords,
        colorbar=True,
        cmap=cmap,
        vmax=float(vmax),
        threshold=threshold,
        symmetric_cbar=symmetric_cbar,
        title=title,
        draw_cross=True,
    )
    display.savefig(str(save_path), dpi=150)
    display.close()


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
    subjects_dir: str,
    fs_subject: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    symmetric_cbar: bool = False,
) -> None:
    """Render publication-quality multi-slice views for a discrete fMRI-constrained STC.

    Produces a two-row nilearn figure:
      - Top row : 6 equidistant axial slices spanning the cluster volume.
      - Bottom row : 3-plane ortho view (sagittal / coronal / axial) at peak voxel.

    Background is the subject's T1.mgz (falls back to MNI152 if absent).
    """
    try:
        from nilearn import plotting as nipl
        import nibabel as nib
        import io
    except ImportError:
        raise RuntimeError(
            "nilearn and nibabel are required for orthogonal STC rendering. "
            "Install with: pip install nilearn nibabel"
        )

    if len(src) != 1:
        raise ValueError(
            f"Discrete source space must contain exactly one volume block (got {len(src)})."
        )

    # MNE's as_volume() does not support discrete source spaces.
    # Build a sparse NIfTI manually via a regular 2 mm grid.
    nii_clamped = _discrete_stc_to_nifti(stc, src, vmin=vmin, vmax=vmax)

    t1_path = Path(subjects_dir) / fs_subject / "mri" / "T1.mgz"
    bg_img = str(t1_path) if t1_path.exists() else "MNI152"

    # Peak voxel in world coordinates ---------------------------------------
    vol_data = np.asarray(nii_clamped.dataobj, dtype=float)
    abs_vol = np.abs(vol_data)
    peak_ijk = np.unravel_index(int(np.argmax(abs_vol)), vol_data.shape)
    peak_xyz = tuple(float(v) for v in (nii_clamped.affine @ np.array([*peak_ijk, 1.0]))[:3])

    threshold = float(vmin) if vmin > 0 else None

    # Determine 6 axial z-cuts spanning the active cluster ------------------
    z_indices = np.where(np.any(vol_data != 0, axis=(0, 1)))[0]
    if z_indices.size >= 2:
        z_lo = float((nii_clamped.affine @ np.array([0, 0, int(z_indices[0]), 1.0]))[2])
        z_hi = float((nii_clamped.affine @ np.array([0, 0, int(z_indices[-1]), 1.0]))[2])
    else:
        z_lo, z_hi = peak_xyz[2] - 20.0, peak_xyz[2] + 20.0
    z_cuts = np.linspace(z_lo, z_hi, 6).tolist()

    # Render nilearn displays into RGBA arrays ------------------------------
    def _render(display_obj: Any) -> np.ndarray:
        buf = io.BytesIO()
        display_obj.savefig(buf, dpi=180)
        buf.seek(0)
        arr = plt.imread(buf)
        display_obj.close()
        return arr

    axial_arr = _render(nipl.plot_stat_map(
        nii_clamped,
        bg_img=bg_img,
        display_mode="z",
        cut_coords=z_cuts,
        colorbar=False,
        cmap=cmap,
        vmax=float(vmax),
        threshold=threshold,
        symmetric_cbar=symmetric_cbar,
        annotate=True,
        draw_cross=False,
    ))

    ortho_arr = _render(nipl.plot_stat_map(
        nii_clamped,
        bg_img=bg_img,
        display_mode="ortho",
        cut_coords=peak_xyz,
        colorbar=True,
        cmap=cmap,
        vmax=float(vmax),
        threshold=threshold,
        symmetric_cbar=symmetric_cbar,
        annotate=True,
        draw_cross=True,
    ))

    # Composite into a single figure ----------------------------------------
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, 8), facecolor="white")
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.08, height_ratios=[1.0, 1.1])

    ax_top = fig.add_subplot(gs[0])
    ax_top.imshow(axial_arr, interpolation="lanczos", aspect="equal")
    ax_top.axis("off")
    ax_top.set_title("Axial slices through cluster", fontsize=10, pad=4, color="#333333")

    ax_bot = fig.add_subplot(gs[1])
    ax_bot.imshow(ortho_arr, interpolation="lanczos", aspect="equal")
    ax_bot.axis("off")
    ax_bot.set_title(
        f"Orthogonal view at peak  "
        f"(x={peak_xyz[0]:.0f}, y={peak_xyz[1]:.0f}, z={peak_xyz[2]:.0f} mm)",
        fontsize=10, pad=4, color="#333333",
    )

    fig.suptitle(title, fontsize=11, y=0.998, color="#111111", fontweight="bold")
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
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


def _plot_surface_stc_contrast(
    diff_stc: Any,
    *,
    subject: str,
    subjects_dir: Optional[str],
    condition_a: str,
    condition_b: str,
    title: str,
) -> Any:
    """Render a condition-difference surface STC as a 2×2 diverging brain grid.

    Uses geodesic smoothing (compute_source_morph) and nilearn rendering,
    with a symmetric RdBu_r colormap centered at zero. Red indicates
    condition B > A; blue indicates A > B.
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

    # Geodesic smoothing to full cortical surface
    morph = mne.compute_source_morph(
        diff_stc,
        subject_from=subject,
        subject_to=subject,
        subjects_dir=subjects_dir,
        smooth=10,
        spacing=None,
        verbose=False,
    )
    stc_full = morph.apply(diff_stc)

    # Signed mean difference across time
    vals = stc_full.data.mean(axis=1) if stc_full.data.shape[1] > 1 else stc_full.data[:, 0]
    n_lh = len(stc_full.vertices[0])

    # Symmetric color limits for diverging map
    abs_max = float(np.percentile(np.abs(vals[np.isfinite(vals)]), 99.5))
    if abs_max <= 0:
        abs_max = 1e-30

    lh_tex = vals[:n_lh]
    rh_tex = vals[n_lh:]

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
            cmap="RdBu_r",
            vmax=abs_max,
            threshold=abs_max * 0.10,
            bg_on_data=True,
            darkness=0.5,
            symmetric_cbar=True,
            title=label,
            title_font_size=12,
        )
        pfig.set_facecolor("white")
        buf = io.BytesIO()
        pfig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        imgs.append(plt.imread(buf))
        plt.close(pfig)

    # Pad and assemble grid
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

    dpi = 150
    w_in = grid.shape[1] / dpi
    h_in = grid.shape[0] / dpi
    out_fig = plt.figure(figsize=(w_in + 1.0, h_in + 0.6), dpi=dpi, facecolor="white")
    gs = out_fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0.03)

    ax_img = out_fig.add_subplot(gs[0])
    ax_img.imshow(grid, interpolation="lanczos")
    ax_img.axis("off")

    ax_cb = out_fig.add_subplot(gs[1])
    cb = out_fig.colorbar(
        plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=-abs_max, vmax=abs_max), cmap="RdBu_r"
        ),
        cax=ax_cb,
    )
    cb.set_label(f"Source difference ({condition_b} − {condition_a}, a.u.)", fontsize=9, labelpad=8)
    cb.ax.tick_params(labelsize=8)

    out_fig.suptitle(title, fontsize=11, y=0.99)
    out_fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.96))
    return out_fig


def _build_surface_condition_contrasts(
    stc_files: List[Path],
    condition_a: str,
    condition_b: str,
) -> list[dict[str, Any]]:
    """Build condition-difference STCs for surface source estimates.

    Groups STC files by (task, segment, band, method), reads conditions A
    and B, and computes B − A. Returns a list of diff records.
    """
    grouped: dict[tuple, dict[str, Path]] = {}
    for path in stc_files:
        if not path.name.endswith("-lh.stc"):
            continue
        metadata = _parse_stc_plot_metadata(path)
        key = (
            str(metadata["task"]),
            metadata["segment"],
            str(metadata["band"]),
            str(metadata["method"]),
        )
        cond = str(metadata["condition"])
        grouped.setdefault(key, {})[cond] = path

    diff_records: list[dict[str, Any]] = []
    for (task, segment, band, method), cond_map in grouped.items():
        if condition_a not in cond_map or condition_b not in cond_map:
            continue
        stc_a = mne.read_source_estimate(str(cond_map[condition_a]))
        stc_b = mne.read_source_estimate(str(cond_map[condition_b]))

        if stc_a.data.shape != stc_b.data.shape:
            continue

        diff_stc = stc_b.copy()
        diff_stc.data = np.asarray(stc_b.data, dtype=float) - np.asarray(stc_a.data, dtype=float)
        diff_records.append({
            "task": task,
            "segment": segment,
            "band": band,
            "method": method,
            "stc": diff_stc,
        })

    return diff_records


###################################################################
# Glass Brain Projections
###################################################################


def plot_source_glass_brain(
    subject: str,
    stc_files: List[Path],
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    subjects_dir: Optional[str] = None,
) -> None:
    """Render nilearn glass-brain projections for source estimates.

    For fMRI-informed (discrete) STCs, uses _discrete_stc_to_nifti.
    For EEG-only (surface) STCs, renders each condition on an inflated surface.
    Produces per-condition glass brains and, if two conditions are configured,
    a B-A contrast glass brain.
    """
    try:
        from nilearn import plotting as nipl
    except ImportError:
        raise RuntimeError("nilearn is required for glass brain rendering.")

    if not stc_files:
        logger.info("No STC files for glass brain plotting.")
        return

    source_segment_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.segment", None
    )
    source_segment = "" if source_segment_raw is None else str(source_segment_raw).strip()
    if source_segment:
        stc_files = _filter_stc_files_by_segment(stc_files, source_segment)

    has_contrasts = _has_contrast_conditions(config)
    source_condition_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.condition", None
    )
    source_condition = "" if source_condition_raw is None else str(source_condition_raw).strip()

    source_bands_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.bands", None
    )
    if source_bands_raw is not None:
        if isinstance(source_bands_raw, str):
            source_bands = [b.strip() for b in source_bands_raw.split() if b.strip()]
        elif isinstance(source_bands_raw, (list, tuple)):
            source_bands = [str(b).strip() for b in source_bands_raw if str(b).strip()]
        else:
            source_bands = [str(source_bands_raw).strip()]
        stc_files = _filter_stc_files_by_bands(stc_files, source_bands, logger)

        subjects_dir = _resolve_source_plot_subjects_dir(config, logger)

    out_dir = save_dir / "glass_brain"
    out_dir.mkdir(parents=True, exist_ok=True)

    is_discrete = any(p.name.endswith("-vl.stc") for p in stc_files)

    if is_discrete:
        src = _load_volume_source_space(stc_files)
        if src is None:
            logger.warning("No src.fif found for glass brain rendering.")
            return

        for stc_path in stc_files:
            if not stc_path.name.endswith("-vl.stc"):
                continue
            cond = str(_parse_stc_plot_metadata(stc_path).get("condition", ""))
            if _should_skip_absolute_plot(cond, source_condition, has_contrasts):
                continue
            try:
                stc = mne.read_source_estimate(str(stc_path))
                nii = _discrete_stc_to_nifti(stc, src)
                image_name = stc_path.name.replace("-vl.stc", "")
                save_path = out_dir / f"{image_name}_glass_brain.png"
                display = nipl.plot_glass_brain(
                    nii,
                    display_mode="ortho",
                    colorbar=True,
                    cmap="hot",
                    threshold=None,
                    plot_abs=False,
                    black_bg=True,
                    alpha=0.8,
                    title=image_name.replace("_", " "),
                    draw_cross=True,
                )
                display.savefig(str(save_path), dpi=200)
                display.close()
                logger.debug("Saved glass brain: %s", save_path.name)
            except Exception as exc:
                logger.error("Glass brain failed for %s: %s", stc_path.name, exc)

        # Contrast glass brain (B-A)
        try:
            available_conditions = {
                str(_parse_stc_plot_metadata(p)["condition"])
                for p in stc_files if p.name.endswith("-vl.stc")
            }
            if len(available_conditions) >= 2:
                cond_a, cond_b = _resolve_source_plot_condition_pair(
                    config, available_conditions, logger,
                )
                stc_map = {
                    p: mne.read_source_estimate(str(p))
                    for p in stc_files if p.name.endswith("-vl.stc")
                }
                diff_records = _build_discrete_condition_differences(stc_map, cond_a, cond_b)
                for rec in diff_records:
                    try:
                        seg_t = f"_seg-{rec['segment']}" if rec["segment"] else ""
                        name = (
                            f"sub-{subject}_task-{rec['task']}{seg_t}"
                            f"_contrast-{cond_b}-minus-{cond_a}"
                            f"_band-{rec['band']}_{rec['method']}"
                        )
                        nii = _discrete_stc_to_nifti(rec["stc"], src, vmin=-1.0, vmax=1.0)
                        save_path = out_dir / f"{name}_glass_brain.png"
                        display = nipl.plot_glass_brain(
                            nii,
                            display_mode="ortho",
                            colorbar=True,
                            cmap="RdBu_r",
                            threshold=1e-15,
                            plot_abs=False,
                            black_bg=True,
                            alpha=0.8,
                            symmetric_cbar=True,
                            title=name.replace("_", " "),
                            draw_cross=True,
                        )
                        display.savefig(str(save_path), dpi=200)
                        display.close()
                        logger.debug("Saved contrast glass brain: %s", save_path.name)
                    except Exception as exc:
                        logger.error("Contrast glass brain failed (%s): %s", rec.get("band"), exc)
        except Exception as exc:
            logger.warning("Glass brain contrast computation failed: %s", exc)
    else:
        # Surface STC: render on fsaverage inflated cortex (4-view composite)
        fs_subject = _resolve_fs_subject_name(subject, subjects_dir)
        for stc_path in stc_files:
            if not stc_path.name.endswith("-lh.stc"):
                continue
            cond = str(_parse_stc_plot_metadata(stc_path).get("condition", ""))
            if _should_skip_absolute_plot(cond, source_condition, has_contrasts):
                continue
            try:
                import io as _io
                import nibabel as nib
                from nilearn import datasets as nids, plotting as nipl

                stc = mne.read_source_estimate(str(stc_path))
                morph = mne.compute_source_morph(
                    stc,
                    subject_from=fs_subject,
                    subject_to="fsaverage",
                    subjects_dir=subjects_dir,
                    smooth=10,
                    verbose=False,
                )
                stc_fs = morph.apply(stc)
                data = np.asarray(stc_fs.data, dtype=float)
                scalar = data.mean(axis=1) if data.ndim == 2 and data.shape[1] > 1 else data[:, 0]

                fsaverage = nids.fetch_surf_fsaverage()

                # Determine vertex counts from the actual mesh files.
                n_verts_lh = nib.load(fsaverage["infl_left"]).agg_data()[0].shape[0]
                n_verts_rh = nib.load(fsaverage["infl_right"]).agg_data()[0].shape[0]

                view_specs = [
                    ("left",  "lateral",  fsaverage["infl_left"],  "lh", n_verts_lh),
                    ("left",  "medial",   fsaverage["infl_left"],  "lh", n_verts_lh),
                    ("right", "lateral",  fsaverage["infl_right"], "rh", n_verts_rh),
                    ("right", "medial",   fsaverage["infl_right"], "rh", n_verts_rh),
                ]

                panels: list[np.ndarray] = []
                for hemi_side, view, surf_mesh, hemi_label, n_mesh_verts in view_specs:
                    hemi_idx = 0 if hemi_label == "lh" else 1
                    verts = stc_fs.vertices[hemi_idx]
                    offset = 0 if hemi_label == "lh" else len(stc_fs.vertices[0])
                    n_hemi = len(verts)
                    vals = scalar[offset : offset + n_hemi]

                    surf_data = np.zeros(n_mesh_verts)
                    surf_data[verts] = vals

                    threshold = (
                        np.percentile(np.abs(scalar[scalar != 0]), 1)
                        if np.any(scalar != 0)
                        else None
                    )
                    fig_tmp = nipl.plot_surf_stat_map(
                        surf_mesh,
                        stat_map=surf_data,
                        hemi=hemi_side,
                        view=view,
                        colorbar=False,
                        cmap="hot",
                        bg_map=fsaverage["sulc_" + hemi_side],
                        threshold=threshold,
                        engine="matplotlib",
                    )
                    buf = _io.BytesIO()
                    fig_tmp.savefig(buf, dpi=150, bbox_inches="tight", facecolor="white")
                    buf.seek(0)
                    panels.append(plt.imread(buf))
                    plt.close(fig_tmp)

                # Pad panels to equal width before concatenation.
                max_w = max(p.shape[1] for p in panels)
                padded = []
                for p in panels:
                    pw = max_w - p.shape[1]
                    if pw > 0:
                        p = np.pad(p, ((0, 0), (pw // 2, pw - pw // 2), (0, 0)),
                                   mode="constant", constant_values=1.0)
                    padded.append(p)

                # Pad rows to equal height before stacking.
                max_h_top = max(padded[0].shape[0], padded[1].shape[0])
                max_h_bot = max(padded[2].shape[0], padded[3].shape[0])

                def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
                    ph = target_h - img.shape[0]
                    if ph > 0:
                        return np.pad(img, ((ph // 2, ph - ph // 2), (0, 0), (0, 0)),
                                      mode="constant", constant_values=1.0)
                    return img

                top_row = np.concatenate([_pad_h(padded[0], max_h_top), _pad_h(padded[1], max_h_top)], axis=1)
                bot_row = np.concatenate([_pad_h(padded[2], max_h_bot), _pad_h(padded[3], max_h_bot)], axis=1)

                # Equalize widths of top and bot rows.
                final_w = max(top_row.shape[1], bot_row.shape[1])
                def _pad_row_w(row: np.ndarray) -> np.ndarray:
                    pw = final_w - row.shape[1]
                    if pw > 0:
                        return np.pad(row, ((0, 0), (pw // 2, pw - pw // 2), (0, 0)),
                                      mode="constant", constant_values=1.0)
                    return row

                grid = np.concatenate([_pad_row_w(top_row), _pad_row_w(bot_row)], axis=0)

                image_name = stc_path.name.replace("-lh.stc", "")
                save_path = out_dir / f"{image_name}_glass_brain.png"
                fig = plt.figure(figsize=(12, 8), facecolor="white")
                ax = fig.add_subplot(111)
                ax.imshow(grid, interpolation="lanczos", aspect="equal")
                ax.axis("off")
                fig.suptitle(
                    image_name.replace("_", " "),
                    fontsize=11, y=0.995, fontweight="bold",
                )
                fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                logger.debug("Saved surface glass brain: %s", save_path.name)
            except Exception as exc:
                logger.error("Surface glass brain failed for %s: %s", stc_path.name, exc)

        # Surface contrast glass brain (B-A)
        try:
            available_conditions = {
                str(_parse_stc_plot_metadata(p)["condition"])
                for p in stc_files if p.name.endswith("-lh.stc")
            }
            if len(available_conditions) >= 2:
                cond_a, cond_b = _resolve_source_plot_condition_pair(
                    config, available_conditions, logger,
                )
                diff_records = _build_surface_condition_contrasts(stc_files, cond_a, cond_b)
                fs_subject = _resolve_fs_subject_name(subject, subjects_dir)
                for rec in diff_records:
                    try:
                        import io as _io
                        import nibabel as nib
                        from nilearn import datasets as nids, plotting as nipl

                        diff_stc = rec["stc"]
                        morph = mne.compute_source_morph(
                            diff_stc,
                            subject_from=fs_subject,
                            subject_to="fsaverage",
                            subjects_dir=subjects_dir,
                            smooth=10,
                            verbose=False,
                        )
                        stc_fs = morph.apply(diff_stc)
                        data = np.asarray(stc_fs.data, dtype=float)
                        scalar = data.mean(axis=1) if data.ndim == 2 and data.shape[1] > 1 else data[:, 0]

                        fsaverage_m = nids.fetch_surf_fsaverage()
                        n_lh = nib.load(fsaverage_m["infl_left"]).agg_data()[0].shape[0]
                        n_rh = nib.load(fsaverage_m["infl_right"]).agg_data()[0].shape[0]

                        vmax = np.abs(scalar).max()
                        view_specs = [
                            ("left",  "lateral",  fsaverage_m["infl_left"],  "lh", n_lh),
                            ("left",  "medial",   fsaverage_m["infl_left"],  "lh", n_lh),
                            ("right", "lateral",  fsaverage_m["infl_right"], "rh", n_rh),
                            ("right", "medial",   fsaverage_m["infl_right"], "rh", n_rh),
                        ]

                        panels: list[np.ndarray] = []
                        for hemi_side, view, surf_mesh, hemi_label, n_mesh_verts in view_specs:
                            hemi_idx = 0 if hemi_label == "lh" else 1
                            verts = stc_fs.vertices[hemi_idx]
                            offset = 0 if hemi_label == "lh" else len(stc_fs.vertices[0])
                            vals = scalar[offset : offset + len(verts)]
                            surf_data = np.zeros(n_mesh_verts)
                            surf_data[verts] = vals

                            fig_tmp = nipl.plot_surf_stat_map(
                                surf_mesh,
                                stat_map=surf_data,
                                hemi=hemi_side,
                                view=view,
                                colorbar=False,
                                cmap="RdBu_r",
                                bg_map=fsaverage_m["sulc_" + hemi_side],
                                threshold=None,
                                vmax=vmax,
                                engine="matplotlib",
                            )
                            buf = _io.BytesIO()
                            fig_tmp.savefig(buf, dpi=150, bbox_inches="tight", facecolor="white")
                            buf.seek(0)
                            panels.append(plt.imread(buf))
                            plt.close(fig_tmp)

                        # Composite 2x2 with padding
                        max_w = max(p.shape[1] for p in panels)
                        padded = []
                        for p in panels:
                            pw = max_w - p.shape[1]
                            if pw > 0:
                                p = np.pad(p, ((0, 0), (pw // 2, pw - pw // 2), (0, 0)),
                                           mode="constant", constant_values=1.0)
                            padded.append(p)
                        max_h_top = max(padded[0].shape[0], padded[1].shape[0])
                        max_h_bot = max(padded[2].shape[0], padded[3].shape[0])

                        def _ph(img, th):
                            d = th - img.shape[0]
                            return np.pad(img, ((d // 2, d - d // 2), (0, 0), (0, 0)),
                                          mode="constant", constant_values=1.0) if d > 0 else img

                        top = np.concatenate([_ph(padded[0], max_h_top), _ph(padded[1], max_h_top)], axis=1)
                        bot = np.concatenate([_ph(padded[2], max_h_bot), _ph(padded[3], max_h_bot)], axis=1)
                        fw = max(top.shape[1], bot.shape[1])

                        def _pw(row):
                            d = fw - row.shape[1]
                            return np.pad(row, ((0, 0), (d // 2, d - d // 2), (0, 0)),
                                          mode="constant", constant_values=1.0) if d > 0 else row

                        grid = np.concatenate([_pw(top), _pw(bot)], axis=0)

                        seg_t = f"_seg-{rec['segment']}" if rec["segment"] else ""
                        image_name = (
                            f"sub-{subject}_task-{rec['task']}{seg_t}"
                            f"_contrast-{cond_b}-minus-{cond_a}"
                            f"_band-{rec['band']}_{rec['method']}"
                        )
                        save_path = out_dir / f"{image_name}_glass_brain.png"
                        fig = plt.figure(figsize=(12, 8), facecolor="white")
                        ax = fig.add_subplot(111)
                        ax.imshow(grid, interpolation="lanczos", aspect="equal")
                        ax.axis("off")
                        fig.suptitle(
                            f"Contrast: {cond_b} − {cond_a}\n{rec['band']} band",
                            fontsize=11, y=0.995, fontweight="bold",
                        )
                        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
                        plt.close(fig)
                        logger.debug("Saved surface contrast glass brain: %s", save_path.name)
                    except Exception as exc:
                        logger.error("Surface contrast glass brain failed (%s): %s", rec.get("band"), exc)
        except Exception as exc:
            logger.warning("Surface glass brain contrast computation failed: %s", exc)


###################################################################
# Band Comparison Panel
###################################################################


def plot_source_band_panel(
    subject: str,
    stc_files: List[Path],
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    subjects_dir: Optional[str] = None,
) -> None:
    """Render a multi-band comparison panel: one row per frequency band, same view.

    Groups STCs by (condition, segment), produces one figure per group with
    all frequency bands as rows. Uses glass-brain rendering for consistency.
    """
    try:
        from nilearn import plotting as nipl
        import io
    except ImportError:
        raise RuntimeError("nilearn is required for band comparison panel.")

    if not stc_files:
        logger.info("No STC files for band comparison panel.")
        return

    source_segment_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.segment", None
    )
    source_segment = "" if source_segment_raw is None else str(source_segment_raw).strip()
    if source_segment:
        stc_files = _filter_stc_files_by_segment(stc_files, source_segment)

    has_contrasts = _has_contrast_conditions(config)
    source_condition_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.condition", None
    )
    source_condition = "" if source_condition_raw is None else str(source_condition_raw).strip()

    source_bands_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.bands", None
    )
    if source_bands_raw is not None:
        if isinstance(source_bands_raw, str):
            source_bands = [b.strip() for b in source_bands_raw.split() if b.strip()]
        elif isinstance(source_bands_raw, (list, tuple)):
            source_bands = [str(b).strip() for b in source_bands_raw if str(b).strip()]
        else:
            source_bands = [str(source_bands_raw).strip()]
        stc_files = _filter_stc_files_by_bands(stc_files, source_bands, logger)

    if subjects_dir is None:
        subjects_dir = _resolve_source_plot_subjects_dir(config, logger)

    out_dir = save_dir / "band_panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    is_discrete = any(p.name.endswith("-vl.stc") for p in stc_files)
    suffix = "-vl.stc" if is_discrete else "-lh.stc"

    src = None
    if is_discrete:
        src = _load_volume_source_space(stc_files)
        if src is None:
            logger.warning("No src.fif found for band panel rendering.")
            return

    # Group STC files by (task, segment, condition) -> list of (band, path)
    grouped: dict[tuple[str, Optional[str], str], list[tuple[str, Path]]] = {}
    for stc_path in stc_files:
        if not stc_path.name.endswith(suffix):
            continue
        metadata = _parse_stc_plot_metadata(stc_path)
        cond = str(metadata.get("condition", ""))
        if _should_skip_absolute_plot(cond, source_condition, has_contrasts):
            continue
        key = (str(metadata["task"]), metadata["segment"], str(metadata["condition"]))
        grouped.setdefault(key, []).append((str(metadata["band"]), stc_path))

    for (task, segment, condition), band_files in grouped.items():
        band_files.sort(key=lambda x: x[0])
        n_bands = len(band_files)
        if n_bands == 0:
            continue

        seg_token = f"_seg-{segment}" if segment else ""
        figure_name = f"sub-{subject}_task-{task}{seg_token}_cond-{condition}_band-panel"

        band_images: list[tuple[str, np.ndarray]] = []
        for band_label, stc_path in band_files:
            try:
                stc = mne.read_source_estimate(str(stc_path))
                if is_discrete and src is not None:
                    nii = _discrete_stc_to_nifti(stc, src)
                    display = nipl.plot_glass_brain(
                        nii,
                        display_mode="ortho",
                        colorbar=False,
                        cmap="hot",
                        threshold=None,
                        plot_abs=False,
                        black_bg=True,
                        alpha=0.8,
                        title=band_label,
                        draw_cross=False,
                    )
                    buf = io.BytesIO()
                    display.savefig(buf, dpi=150)
                    buf.seek(0)
                    band_images.append((band_label, plt.imread(buf)))
                    display.close()
                else:
                    # Surface STC: morph to fsaverage, render left lateral view
                    import nibabel as nib
                    from nilearn import datasets as nids
                    fs_subject = _resolve_fs_subject_name(subject, subjects_dir)
                    morph = mne.compute_source_morph(
                        stc,
                        subject_from=fs_subject,
                        subject_to="fsaverage",
                        subjects_dir=subjects_dir,
                        smooth=10,
                        verbose=False,
                    )
                    stc_fs = morph.apply(stc)
                    data_fs = np.asarray(stc_fs.data, dtype=float)
                    scalar = data_fs.mean(axis=1) if data_fs.ndim == 2 and data_fs.shape[1] > 1 else data_fs[:, 0]

                    fsaverage_meshes = nids.fetch_surf_fsaverage()
                    n_mesh = nib.load(fsaverage_meshes["infl_left"]).agg_data()[0].shape[0]

                    # Left lateral view
                    verts_lh = stc_fs.vertices[0]
                    surf_data_lh = np.zeros(n_mesh)
                    surf_data_lh[verts_lh] = scalar[: len(verts_lh)]

                    threshold = (
                        np.percentile(np.abs(scalar[scalar != 0]), 1)
                        if np.any(scalar != 0)
                        else None
                    )
                    fig_tmp = nipl.plot_surf_stat_map(
                        fsaverage_meshes["infl_left"],
                        stat_map=surf_data_lh,
                        hemi="left",
                        view="lateral",
                        colorbar=False,
                        cmap="hot",
                        bg_map=fsaverage_meshes["sulc_left"],
                        threshold=threshold,
                        title=band_label,
                        engine="matplotlib",
                    )
                    buf = io.BytesIO()
                    fig_tmp.savefig(buf, dpi=150, bbox_inches="tight", facecolor="white")
                    buf.seek(0)
                    band_images.append((band_label, plt.imread(buf)))
                    plt.close(fig_tmp)
            except Exception as exc:
                logger.warning("Band panel: skipping %s: %s", band_label, exc)

        if not band_images:
            continue

        # Composite: vertical stack
        max_w = max(img.shape[1] for _, img in band_images)

        def _pad_w(img: np.ndarray) -> np.ndarray:
            pw = max_w - img.shape[1]
            if pw > 0:
                return np.pad(img, ((0, 0), (pw // 2, pw - pw // 2), (0, 0)),
                              mode="constant", constant_values=1.0)
            return img

        padded = [_pad_w(img) for _, img in band_images]
        grid = np.concatenate(padded, axis=0)

        fig = plt.figure(figsize=(12, 2.5 * n_bands), facecolor="white")
        ax = fig.add_subplot(111)
        ax.imshow(grid, interpolation="lanczos", aspect="equal")
        ax.axis("off")
        fig.suptitle(
            f"Source power by frequency band\n{condition} ({task}{seg_token})",
            fontsize=12, y=0.998, fontweight="bold",
        )
        save_path = out_dir / f"{figure_name}.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.debug("Saved band panel: %s", save_path.name)

    # Contrast band panel (B-A)
    try:
        available_conditions: set[str] = set()
        for stc_path in stc_files:
            if not stc_path.name.endswith(suffix):
                continue
            metadata = _parse_stc_plot_metadata(stc_path)
            cond = metadata["condition"]
            if cond is not None:
                available_conditions.add(str(cond))

        if len(available_conditions) >= 2:
            cond_a, cond_b = _resolve_source_plot_condition_pair(
                config, available_conditions, logger,
            )

            if is_discrete and src is not None:
                stc_map = {
                    p: mne.read_source_estimate(str(p))
                    for p in stc_files if p.name.endswith("-vl.stc")
                }
                diff_records = _build_discrete_condition_differences(stc_map, cond_a, cond_b)
            else:
                diff_records = _build_surface_condition_contrasts(stc_files, cond_a, cond_b)

            if diff_records:
                diff_records.sort(key=lambda r: str(r["band"]))
                n_bands = len(diff_records)

                band_images: list[tuple[str, np.ndarray]] = []
                for rec in diff_records:
                    try:
                        diff_stc = rec["stc"]
                        band_label = str(rec["band"])

                        if is_discrete and src is not None:
                            nii = _discrete_stc_to_nifti(diff_stc, src, vmin=-1.0, vmax=1.0)
                            display = nipl.plot_glass_brain(
                                nii,
                                display_mode="ortho",
                                colorbar=False,
                                cmap="RdBu_r",
                                threshold=1e-15,
                                plot_abs=False,
                                black_bg=True,
                                alpha=0.8,
                                symmetric_cbar=True,
                                title=band_label,
                                draw_cross=False,
                            )
                            buf = io.BytesIO()
                            display.savefig(buf, dpi=150)
                            buf.seek(0)
                            band_images.append((band_label, plt.imread(buf)))
                            display.close()
                        else:
                            import nibabel as nib
                            from nilearn import datasets as nids
                            fs_subject = _resolve_fs_subject_name(subject, subjects_dir)
                            morph = mne.compute_source_morph(
                                diff_stc,
                                subject_from=fs_subject,
                                subject_to="fsaverage",
                                subjects_dir=subjects_dir,
                                smooth=10,
                                verbose=False,
                            )
                            stc_fs = morph.apply(diff_stc)
                            data_fs = np.asarray(stc_fs.data, dtype=float)
                            scalar = data_fs.mean(axis=1) if data_fs.ndim == 2 and data_fs.shape[1] > 1 else data_fs[:, 0]

                            fsaverage_meshes = nids.fetch_surf_fsaverage()
                            n_mesh = nib.load(fsaverage_meshes["infl_left"]).agg_data()[0].shape[0]
                            verts_lh = stc_fs.vertices[0]
                            surf_data_lh = np.zeros(n_mesh)
                            surf_data_lh[verts_lh] = scalar[: len(verts_lh)]

                            fig_tmp = nipl.plot_surf_stat_map(
                                fsaverage_meshes["infl_left"],
                                stat_map=surf_data_lh,
                                hemi="left",
                                view="lateral",
                                colorbar=False,
                                cmap="RdBu_r",
                                bg_map=fsaverage_meshes["sulc_left"],
                                threshold=None,
                                vmax=np.abs(scalar).max(),
                                title=band_label,
                                engine="matplotlib",
                            )
                            buf = io.BytesIO()
                            fig_tmp.savefig(buf, dpi=150, bbox_inches="tight", facecolor="white")
                            buf.seek(0)
                            band_images.append((band_label, plt.imread(buf)))
                            plt.close(fig_tmp)
                    except Exception as exc:
                        logger.warning("Band panel contrast: skipping %s: %s", rec.get("band"), exc)

                if band_images:
                    max_w = max(img.shape[1] for _, img in band_images)

                    def _pad_contrast_w(img: np.ndarray) -> np.ndarray:
                        pw = max_w - img.shape[1]
                        if pw > 0:
                            return np.pad(img, ((0, 0), (pw // 2, pw - pw // 2), (0, 0)),
                                          mode="constant", constant_values=1.0)
                        return img

                    padded = [_pad_contrast_w(img) for _, img in band_images]
                    grid = np.concatenate(padded, axis=0)

                    # Use metadata from first record for naming
                    r0 = diff_records[0]
                    seg_t = f"_seg-{r0['segment']}" if r0["segment"] else ""
                    contrast_name = (
                        f"sub-{subject}_task-{r0['task']}{seg_t}"
                        f"_contrast-{cond_b}-minus-{cond_a}_band-panel"
                    )
                    fig = plt.figure(figsize=(12, 2.5 * n_bands), facecolor="white")
                    ax = fig.add_subplot(111)
                    ax.imshow(grid, interpolation="lanczos", aspect="equal")
                    ax.axis("off")
                    fig.suptitle(
                        f"Source contrast: {cond_b} − {cond_a}\nby frequency band",
                        fontsize=12, y=0.998, fontweight="bold",
                    )
                    save_path = out_dir / f"{contrast_name}.png"
                    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                    logger.debug("Saved contrast band panel: %s", save_path.name)
    except Exception as exc:
        logger.warning("Band panel contrast computation failed: %s", exc)


###################################################################
# Cluster Time Course
###################################################################


def plot_source_cluster_timecourse(
    subject: str,
    stc_files: List[Path],
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    subjects_dir: Optional[str] = None,
) -> None:
    """Plot source cluster comparison across conditions.

    Adapts to the STC time dimension:
    - **Single time point** (band-averaged power): renders a grouped bar chart
      comparing mean cluster amplitude across conditions for each frequency
      band. This is the typical output for band-power source localization.
    - **Multiple time points** (time-resolved): renders a trace of mean
      cluster amplitude over time, one line per condition.
    """
    if not stc_files:
        logger.info("No STC files for cluster time course plotting.")
        return

    source_segment_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.segment", None
    )
    source_segment = "" if source_segment_raw is None else str(source_segment_raw).strip()
    if source_segment:
        stc_files = _filter_stc_files_by_segment(stc_files, source_segment)

    source_condition_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.condition", None
    )
    source_condition = "" if source_condition_raw is None else str(source_condition_raw).strip()

    source_bands_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.bands", None
    )
    if source_bands_raw is not None:
        if isinstance(source_bands_raw, str):
            source_bands = [b.strip() for b in source_bands_raw.split() if b.strip()]
        elif isinstance(source_bands_raw, (list, tuple)):
            source_bands = [str(b).strip() for b in source_bands_raw if str(b).strip()]
        else:
            source_bands = [str(source_bands_raw).strip()]
        stc_files = _filter_stc_files_by_bands(stc_files, source_bands, logger)

    vol_stc_files = [p for p in stc_files if p.name.endswith("-vl.stc")]
    if not vol_stc_files:
        logger.info("Cluster time course requires volume (-vl.stc) STCs; none found.")
        return

    out_dir = save_dir / "cluster_timecourse"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by (task, segment, band, method) → {condition: stc}
    grouped: dict[tuple[str, Optional[str], str, str], dict[str, Any]] = {}
    for stc_path in vol_stc_files:
        try:
            metadata = _parse_stc_plot_metadata(stc_path)
            key = (
                str(metadata["task"]),
                metadata["segment"],
                str(metadata["band"]),
                str(metadata["method"]),
            )
            cond = str(metadata["condition"])
            if source_condition and cond != source_condition:
                continue
            stc = mne.read_source_estimate(str(stc_path))
            grouped.setdefault(key, {})[cond] = stc
        except Exception as exc:
            logger.warning("Cluster time course: skipping %s: %s", stc_path.name, exc)

    # Detect whether STCs are single-timepoint (band power) or multi-timepoint.
    sample_stc = next(iter(next(iter(grouped.values())).values()), None) if grouped else None
    is_single_tp = sample_stc is not None and sample_stc.data.shape[1] <= 1

    if is_single_tp:
        _plot_cluster_bar_comparison(subject, grouped, out_dir, config, logger)
    else:
        _plot_cluster_timeseries(subject, grouped, out_dir, logger)


def _plot_cluster_bar_comparison(
    subject: str,
    grouped: dict[tuple[str, Optional[str], str, str], dict[str, Any]],
    out_dir: Path,
    config: Any,
    logger: logging.Logger,
) -> None:
    """Grouped bar chart: mean cluster amplitude per condition, one group per band."""
    # Re-group by (task, segment, method) → {band: {condition: scalar}}
    by_context: dict[tuple[str, Optional[str], str], dict[str, dict[str, float]]] = {}
    for (task, segment, band, method), cond_stcs in grouped.items():
        ctx_key = (task, segment, method)
        band_data: dict[str, float] = {}
        for cond, stc in cond_stcs.items():
            band_data[cond] = float(np.abs(stc.data).mean())
        by_context.setdefault(ctx_key, {})[band] = band_data

    for (task, segment, method), band_cond_data in by_context.items():
        seg_token = f"_seg-{segment}" if segment else ""
        figure_name = f"sub-{subject}_task-{task}{seg_token}_{method}_cluster_comparison"

        bands = sorted(band_cond_data.keys())
        all_conditions = sorted({c for bd in band_cond_data.values() for c in bd})
        n_bands = len(bands)
        n_conds = len(all_conditions)

        if n_bands == 0 or n_conds == 0:
            continue

        x = np.arange(n_bands)
        width = 0.7 / n_conds
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_conds, 3)))

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_bands * n_conds), 5), facecolor="white")

        for i, cond in enumerate(all_conditions):
            values = [band_cond_data[b].get(cond, 0.0) for b in bands]
            offset = (i - (n_conds - 1) / 2) * width
            bars = ax.bar(x + offset, values, width * 0.9, label=cond,
                          color=colors[i], edgecolor="white", linewidth=0.5)
            # Value labels on bars
            for bar_rect, val in zip(bars, values):
                ax.text(
                    bar_rect.get_x() + bar_rect.get_width() / 2,
                    bar_rect.get_height(),
                    f"{val:.2e}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in bands], fontsize=10)
        ax.set_xlabel("Frequency Band", fontsize=11)
        ax.set_ylabel("Mean source amplitude (a.u.)", fontsize=11)
        ax.set_title(
            f"Cluster source power by condition\n{task}{seg_token} — {method}",
            fontsize=12, fontweight="bold",
        )
        ax.legend(title="Condition", fontsize=9, title_fontsize=10, framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()

        save_path = out_dir / f"{figure_name}.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.debug("Saved cluster bar comparison: %s", save_path.name)

        # Contrast bar chart (B-A difference)
        if n_conds >= 2:
            try:
                cond_a, cond_b = _resolve_source_plot_condition_pair(
                    config, set(all_conditions), logger,
                )
                diff_values = [
                    band_cond_data[b].get(cond_b, 0.0) - band_cond_data[b].get(cond_a, 0.0)
                    for b in bands
                ]
                bar_colors = ["#c0392b" if v > 0 else "#2980b9" for v in diff_values]

                fig_c, ax_c = plt.subplots(
                    figsize=(max(8, 1.5 * n_bands), 5), facecolor="white",
                )
                bars_c = ax_c.bar(
                    np.arange(n_bands), diff_values, 0.5,
                    color=bar_colors, edgecolor="white", linewidth=0.5,
                )
                for bar_rect, val in zip(bars_c, diff_values):
                    va = "bottom" if val >= 0 else "top"
                    ax_c.text(
                        bar_rect.get_x() + bar_rect.get_width() / 2,
                        bar_rect.get_height(),
                        f"{val:+.2e}",
                        ha="center", va=va, fontsize=8,
                    )
                ax_c.axhline(0, color="grey", linewidth=0.8, linestyle="--")
                ax_c.set_xticks(np.arange(n_bands))
                ax_c.set_xticklabels([b.capitalize() for b in bands], fontsize=10)
                ax_c.set_xlabel("Frequency Band", fontsize=11)
                ax_c.set_ylabel("Δ Mean amplitude (B − A)", fontsize=11)
                ax_c.set_title(
                    f"Cluster contrast: {cond_b} − {cond_a}\n{task}{seg_token} — {method}",
                    fontsize=12, fontweight="bold",
                )
                ax_c.spines["top"].set_visible(False)
                ax_c.spines["right"].set_visible(False)
                ax_c.tick_params(labelsize=9)
                fig_c.tight_layout()

                contrast_path = out_dir / f"{figure_name}_contrast.png"
                fig_c.savefig(str(contrast_path), dpi=200, bbox_inches="tight", facecolor="white")
                plt.close(fig_c)
                logger.debug("Saved cluster contrast bar: %s", contrast_path.name)
            except Exception as exc:
                logger.warning("Cluster contrast bar failed: %s", exc)


def _plot_cluster_timeseries(
    subject: str,
    grouped: dict[tuple[str, Optional[str], str, str], dict[str, Any]],
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Line plot: mean cluster amplitude over time, one line per condition."""
    for (task, segment, band, method), cond_stcs in grouped.items():
        if not cond_stcs:
            continue

        seg_token = f"_seg-{segment}" if segment else ""
        figure_name = f"sub-{subject}_task-{task}{seg_token}_band-{band}_{method}_timecourse"

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

        sorted_conditions = sorted(cond_stcs.keys())
        for cond_label in sorted_conditions:
            stc = cond_stcs[cond_label]
            mean_power = np.abs(stc.data).mean(axis=0)
            times_ms = stc.times * 1000.0
            ax.plot(times_ms, mean_power, linewidth=1.5, label=cond_label, alpha=0.85)

        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylabel("Mean source amplitude (a.u.)", fontsize=11)
        ax.set_title(
            f"Cluster time course — {band} band ({task}{seg_token})",
            fontsize=12, fontweight="bold",
        )
        ax.legend(title="Condition", fontsize=9, title_fontsize=10, framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()

        save_path = out_dir / f"{figure_name}.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.debug("Saved cluster time course: %s", save_path.name)


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

    has_contrasts = _has_contrast_conditions(config)
    source_condition_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.condition", None
    )
    source_condition = "" if source_condition_raw is None else str(source_condition_raw).strip()

    source_bands_raw = get_config_value(
        config, "plotting.plots.features.sourcelocalization.bands", None
    )
    if source_bands_raw is not None:
        if isinstance(source_bands_raw, str):
            source_bands = [b.strip() for b in source_bands_raw.split() if b.strip()]
        elif isinstance(source_bands_raw, (list, tuple)):
            source_bands = [str(b).strip() for b in source_bands_raw if str(b).strip()]
        else:
            source_bands = [str(source_bands_raw).strip()]
        if source_bands:
            stc_files = _filter_stc_files_by_bands(stc_files, source_bands, logger)
            logger.info(
                "Filtered source STCs to bands %s (%d files).",
                source_bands,
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
            if has_contrasts:
                try:
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
                except Exception as exc:
                    logger.warning("Contrast 3d brain setup failed: %s", exc)
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

            cond = str(_parse_stc_plot_metadata(stc_path).get("condition", ""))
            if _should_skip_absolute_plot(cond, source_condition, has_contrasts):
                continue

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
                        subjects_dir=subjects_dir,
                        fs_subject=fs_subject,
                        cmap="hot",
                        vmin=discrete_condition_vmin,
                        vmax=discrete_condition_vmax,
                        colorbar_label="Source amplitude (a.u.)",
                    )
                    vol_save_path = out_dir / f"{image_name}_volumetric.png"
                    try:
                        _plot_discrete_stc_volumetric(
                            stc=stc,
                            src=volume_src,
                            save_path=vol_save_path,
                            title=image_name.replace("_", " "),
                            subjects_dir=subjects_dir,
                            fs_subject=fs_subject,
                            cmap="hot",
                            vmin=float(discrete_condition_vmin),
                            vmax=float(discrete_condition_vmax),
                            colorbar_label="Source amplitude (a.u.)",
                        )
                        if logger:
                            logger.debug(f"Saved volumetric source plot: {vol_save_path.name}")
                    except Exception as vol_exc:
                        if logger:
                            logger.warning(
                                "Volumetric NIfTI rendering skipped for %s: %s",
                                image_name, vol_exc,
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

    # Surface condition contrasts (B − A)
    if stc_kind == "surface" and has_contrasts:
        try:
            available_conditions = {
                str(_parse_stc_plot_metadata(p)["condition"])
                for p in stc_files
                if p.name.endswith("-lh.stc")
            }
            if len(available_conditions) >= 2:
                cond_a, cond_b = _resolve_source_plot_condition_pair(
                    config,
                    available_conditions=available_conditions,
                    logger=logger,
                )
                surf_diff_records = _build_surface_condition_contrasts(
                    stc_files, condition_a=cond_a, condition_b=cond_b,
                )
                if surf_diff_records:
                    logger.info(
                        "Rendering %d surface condition-contrast maps (%s minus %s).",
                        len(surf_diff_records), cond_b, cond_a,
                    )
                    for rec in surf_diff_records:
                        try:
                            seg_token = f"_seg-{rec['segment']}" if rec["segment"] else ""
                            image_name = (
                                f"sub-{subject}_task-{rec['task']}{seg_token}"
                                f"_contrast-{cond_b}-minus-{cond_a}"
                                f"_band-{rec['band']}_{rec['method']}"
                            )
                            contrast_fig = _plot_surface_stc_contrast(
                                rec["stc"],
                                subject=fs_subject,
                                subjects_dir=subjects_dir,
                                condition_a=cond_a,
                                condition_b=cond_b,
                                title=image_name.replace("_", " "),
                            )
                            save_path = out_dir / f"{image_name}_contrast_3d.png"
                            contrast_fig.savefig(
                                str(save_path), dpi=150, bbox_inches="tight", facecolor="white",
                            )
                            plt.close(contrast_fig)
                            if logger:
                                logger.debug(f"Saved contrast plot: {save_path.name}")
                        except Exception as exc:
                            label = f"contrast:{rec.get('band')}/{rec.get('segment')}"
                            failures.append(label)
                            if logger:
                                logger.error("Failed to render surface contrast (%s): %s", label, exc)
                else:
                    logger.info("No matched condition pairs for surface contrast plotting.")
        except Exception as exc:
            logger.warning("Could not compute surface contrasts: %s", exc)

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
                    subjects_dir=subjects_dir,
                    fs_subject=fs_subject,
                    cmap="RdBu_r",
                    vmin=-diff_absmax,
                    vmax=diff_absmax,
                    colorbar_label=f"Source difference ({condition_b} - {condition_a}, a.u.)",
                    symmetric_cbar=True,
                )
                vol_save_path = out_dir / f"{image_name}_volumetric.png"
                try:
                    _plot_discrete_stc_volumetric(
                        stc=diff_stc,
                        src=volume_src,
                        save_path=vol_save_path,
                        title=image_name.replace("_", " "),
                        subjects_dir=subjects_dir,
                        fs_subject=fs_subject,
                        cmap="RdBu_r",
                        vmin=-diff_absmax,
                        vmax=diff_absmax,
                        colorbar_label=f"Source difference ({condition_b} − {condition_a}, a.u.)",
                        symmetric_cbar=True,
                    )
                    if logger:
                        logger.debug(f"Saved volumetric contrast plot: {vol_save_path.name}")
                except Exception as vol_exc:
                    if logger:
                        logger.warning(
                            "Volumetric NIfTI contrast rendering skipped for %s: %s",
                            image_name, vol_exc,
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
