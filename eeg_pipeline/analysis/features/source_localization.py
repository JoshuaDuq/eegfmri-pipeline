"""
Source Localization Feature Extraction
======================================

Extracts ROI-specific features using source localization methods:
- LCMV beamformer
- eLORETA inverse solution

Provides source-space power, connectivity, and time-course features
for anatomically-defined ROIs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.analysis.features.rest import (
    is_resting_state_feature_mode,
    select_single_rest_analysis_segment,
    valid_rest_analysis_segment_masks,
)
from eeg_pipeline.infra.paths import deriv_features_path
from eeg_pipeline.utils.analysis.windowing import get_segment_masks
from eeg_pipeline.utils.data.source_localization_paths import source_localization_estimates_dir
from eeg_pipeline.utils.config.loader import get_config_float, get_config_int, get_config_value, get_frequency_bands

if TYPE_CHECKING:
    import mne



def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    return Path(str(value)).expanduser()


def _valid_source_segment_masks(
    masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return valid_rest_analysis_segment_masks(masks)


def _resolve_source_segment(
    *,
    times: Optional[np.ndarray],
    windows: Any,
    target_name: Optional[str],
    config: Any,
    logger: Optional[logging.Logger],
    feature_name: str,
) -> Tuple[str, Optional[np.ndarray]]:
    if not target_name or windows is None:
        return "full", None

    mask = windows.get_mask(target_name)
    if mask is not None and np.any(mask):
        return str(target_name), np.asarray(mask, dtype=bool)

    if times is None:
        raise ValueError(
            "Source connectivity requires epochs.times when time-window masks are specified."
        )

    if not is_resting_state_feature_mode(config):
        if logger is not None:
            logger.error(
                "%s: targeted window '%s' has no valid mask; skipping.",
                feature_name,
                target_name,
            )
        return str(target_name), np.zeros(times.shape, dtype=bool)

    segment_name, segment_mask = select_single_rest_analysis_segment(
        get_segment_masks(times, windows, config),
        feature_name=feature_name,
        target_name=str(target_name),
    )
    if logger is not None:
        logger.info(
            "%s: resting-state mode found no valid target window '%s'; "
            "using available analysis segment '%s' instead.",
            feature_name,
            target_name,
            segment_name,
        )
    return str(segment_name), np.asarray(segment_mask, dtype=bool)


@dataclass(frozen=True)
class FMRIConstraintConfig:
    enabled: bool
    stats_map_path: Optional[Path]
    provenance: str  # "independent" | "same_dataset" | "unknown"
    require_provenance: bool
    allow_same_dataset_provenance: bool
    threshold: float
    tail: str  # "pos" or "abs"
    threshold_mode: str  # "z" or "fdr"
    fdr_q: float
    stat_type: str  # "z" only (required)
    cluster_min_voxels: int
    cluster_min_volume_mm3: Optional[float]
    max_clusters: int
    max_voxels_per_cluster: int
    max_total_voxels: int
    random_seed: int
    output_space: str = "dual"  # "cluster" | "atlas" | "dual"
    prethresholded_mask: bool = False


@dataclass(frozen=True)
class FMRIVoxelSelection:
    """Selected fMRI-constrained voxels and derived ROI groupings."""

    selected_voxels_ijk: np.ndarray  # shape (n_voxels, 3), MRI voxel indices
    selected_coords_m: np.ndarray  # shape (n_voxels, 3), surface RAS in meters
    cluster_indices: Dict[str, List[int]]
    cluster_voxel_counts: Dict[str, int]
    atlas_indices: Dict[str, List[int]]
    atlas_voxel_counts: Dict[str, int]
    cluster_to_atlas_counts: Dict[str, Dict[str, int]]
    dropped_unlabeled_voxels: int
    matched_reference_path: Path
    voxel_volume_mm3: Optional[float]


def _is_prethresholded_fmri_mask(stats_map_path: Optional[Path]) -> bool:
    if stats_map_path is None:
        return False

    sidecar = stats_map_path.with_suffix("").with_suffix(".json")
    if not sidecar.exists():
        return False

    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False

    return str(payload.get("artifact_type", "")).strip().lower() == "fmri_constraint_mask"


def _load_fmri_constraint_config(
    config: Any,
    bids_fmri_root: Optional[Path] = None,
    bids_derivatives: Optional[Path] = None,
    freesurfer_subjects_dir: Optional[Path] = None,
    subject: Optional[str] = None,
    task: str = "",
) -> FMRIConstraintConfig:
    """
    Load fMRI constraint configuration, building stats map if needed.

    If contrast.enabled is True and no stats_map_path exists, the contrast
    builder will be invoked to generate the stats map from BOLD data.
    """
    src_cfg = get_config_value(config, "feature_engineering.sourcelocalization", {}) or {}
    fmri_cfg = {}
    if isinstance(src_cfg, dict):
        fmri_cfg = src_cfg.get("fmri", {}) or {}
    elif hasattr(config, "get"):
        fmri_cfg = config.get("feature_engineering.sourcelocalization.fmri", {}) or {}

    enabled = bool(fmri_cfg.get("enabled", False))
    stats_map_path = _as_path(fmri_cfg.get("stats_map_path") or fmri_cfg.get("stats_map"))
    provenance = str(fmri_cfg.get("provenance", "unknown")).strip().lower()
    if provenance not in {"independent", "same_dataset", "unknown"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.provenance must be one of "
            "{'independent','same_dataset','unknown'} "
            f"(got '{provenance}')."
        )
    require_provenance = bool(fmri_cfg.get("require_provenance", True))
    allow_same_dataset_provenance = bool(
        fmri_cfg.get("allow_same_dataset_provenance", False)
    )

    contrast_cfg = fmri_cfg.get("contrast", {}) or {}
    contrast_enabled = bool(contrast_cfg.get("enabled", False))

    if (enabled or contrast_enabled) and stats_map_path is None and contrast_enabled:
        # Try to build contrast from BOLD data
        missing_paths = []
        if bids_fmri_root is None:
            missing_paths.append("bids_fmri_root (paths.bids_fmri_root or --bids-root)")
        if bids_derivatives is None:
            missing_paths.append("bids_derivatives (paths.deriv_root or --deriv-root)")
        if freesurfer_subjects_dir is None:
            missing_paths.append("freesurfer_subjects_dir (feature_engineering.sourcelocalization.subjects_dir or --source-subjects-dir)")
        if not subject:
            missing_paths.append("subject")
        if not str(task or "").strip():
            missing_paths.append("task (project.task or --task)")

        if missing_paths:
            raise ValueError(
                "Cannot build fMRI contrast: missing required inputs: "
                + ", ".join(missing_paths)
            )

        from fmri_pipeline.analysis.contrast_builder import ensure_fmri_stats_map

        built_path = ensure_fmri_stats_map(
            config=config,
            bids_fmri_root=bids_fmri_root,
            bids_derivatives=bids_derivatives,
            freesurfer_subjects_dir=freesurfer_subjects_dir,
            subject=subject,
            task=task,
        )
        if built_path is None:
            raise ValueError("fMRI contrast builder did not produce a stats map path.")
        stats_map_path = built_path
        enabled = True

    try:
        threshold = float(fmri_cfg.get("threshold", 3.1))
    except (TypeError, ValueError):
        raise ValueError(
            f"feature_engineering.sourcelocalization.fmri.threshold must be a float (got {fmri_cfg.get('threshold')})."
        )
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.threshold must be a finite float > 0 "
            f"(got {threshold})."
        )

    tail = str(fmri_cfg.get("tail", "pos")).strip().lower()
    if tail not in {"pos", "abs"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.tail must be one of {'pos','abs'} "
            f"(got '{tail}')."
        )

    thresholding_cfg = fmri_cfg.get("thresholding", {}) or {}
    threshold_mode = str(thresholding_cfg.get("mode", fmri_cfg.get("threshold_mode", "z"))).strip().lower()
    if threshold_mode not in {"z", "fdr"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.thresholding.mode must be one of {'z','fdr'} "
            f"(got '{threshold_mode}')."
        )

    try:
        fdr_q = float(thresholding_cfg.get("fdr_q", 0.05))
    except (TypeError, ValueError):
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.thresholding.fdr_q must be a float in (0, 1)."
        )
    if not np.isfinite(fdr_q) or fdr_q <= 0 or fdr_q >= 1:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.thresholding.fdr_q must be in (0, 1)."
        )

    stat_type = str(thresholding_cfg.get("stat_type", "z")).strip().lower()
    if stat_type not in {"z"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.thresholding.stat_type must be 'z' "
            f"(got '{stat_type}')."
        )

    cluster_min_volume_mm3 = fmri_cfg.get("cluster_min_volume_mm3", None)
    if cluster_min_volume_mm3 is not None:
        try:
            cluster_min_volume_mm3 = float(cluster_min_volume_mm3)
        except (TypeError, ValueError):
            raise ValueError(
                "feature_engineering.sourcelocalization.fmri.cluster_min_volume_mm3 must be a float > 0 or null."
            )
        if not np.isfinite(cluster_min_volume_mm3) or cluster_min_volume_mm3 <= 0:
            raise ValueError(
                "feature_engineering.sourcelocalization.fmri.cluster_min_volume_mm3 must be > 0 when provided."
            )

    # Explicitly reject stale, non-functional settings to avoid false assumptions.
    time_windows_cfg = fmri_cfg.get("time_windows", {}) or {}
    has_window_settings = False
    if isinstance(time_windows_cfg, dict):
        for window_cfg in time_windows_cfg.values():
            if isinstance(window_cfg, dict):
                if str(window_cfg.get("name", "")).strip():
                    has_window_settings = True
                    break
            elif str(window_cfg).strip():
                has_window_settings = True
                break
    elif bool(time_windows_cfg):
        has_window_settings = True
    if has_window_settings:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.time_windows is currently unsupported for "
            "source feature computation and can lead to misleading interpretation. Remove this block."
        )

    cluster_min_voxels = get_config_int(
        config,
        "feature_engineering.sourcelocalization.fmri.cluster_min_voxels",
        50,
    )
    max_clusters = get_config_int(
        config,
        "feature_engineering.sourcelocalization.fmri.max_clusters",
        20,
    )
    max_voxels_per_cluster = get_config_int(
        config,
        "feature_engineering.sourcelocalization.fmri.max_voxels_per_cluster",
        2000,
    )
    max_total_voxels = get_config_int(
        config,
        "feature_engineering.sourcelocalization.fmri.max_total_voxels",
        20000,
    )
    if cluster_min_voxels <= 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.cluster_min_voxels must be > 0."
        )
    if max_clusters <= 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.max_clusters must be > 0."
        )
    if max_voxels_per_cluster < 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.max_voxels_per_cluster must be >= 0."
        )
    if max_total_voxels < 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.max_total_voxels must be >= 0."
        )
    random_seed = int(
        get_config_int(
            config,
            "feature_engineering.sourcelocalization.fmri.random_seed",
            0,
        )
    )
    if random_seed < 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.random_seed must be >= 0 "
            f"(got {random_seed})."
        )

    output_space = str(fmri_cfg.get("output_space", "dual")).strip().lower()
    if output_space not in {"cluster", "atlas", "dual"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.output_space must be one of "
            "{'cluster','atlas','dual'} "
            f"(got '{output_space}')."
        )

    prethresholded_mask = _is_prethresholded_fmri_mask(stats_map_path)

    return FMRIConstraintConfig(
        enabled=enabled or (stats_map_path is not None),
        stats_map_path=stats_map_path,
        provenance=provenance,
        require_provenance=require_provenance,
        allow_same_dataset_provenance=allow_same_dataset_provenance,
        threshold=threshold,
        tail=tail,
        threshold_mode=threshold_mode,
        fdr_q=fdr_q,
        stat_type=stat_type,
        cluster_min_voxels=cluster_min_voxels,
        cluster_min_volume_mm3=cluster_min_volume_mm3,
        max_clusters=max_clusters,
        max_voxels_per_cluster=max_voxels_per_cluster,
        max_total_voxels=max_total_voxels,
        random_seed=random_seed,
        output_space=output_space,
        prethresholded_mask=prethresholded_mask,
    )

###################################################################
# Forward Model Setup
###################################################################


def _setup_forward_model(
    info: Any,
    subjects_dir: Optional[str] = None,
    subject: str = "fsaverage",
    spacing: str = "oct6",
    conductivity: Tuple[float, ...] = (0.3, 0.006, 0.3),
    mindist: float = 5.0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Any, Any]:
    """
    Set up forward model for source localization.
    
    Uses fsaverage template if no subject-specific MRI is available.
    
    Returns
    -------
    fwd : mne.Forward
        Forward solution
    src : mne.SourceSpaces
        Source space
    bem : str or mne.bem.ConductorModel
        BEM solution
    """
    import mne
    from mne.datasets import fetch_fsaverage
    
    if logger:
        logger.info(f"Setting up forward model using {subject} template")
    
    if subjects_dir is None:
        fs_dir = Path(fetch_fsaverage(verbose=False))
        subjects_dir = str(fs_dir.parent)
    else:
        fs_dir = Path(subjects_dir) / subject
    
    src = mne.setup_source_space(
        subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    
    bem = str(fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif")
    trans_path = fs_dir / "bem" / "fsaverage-trans.fif"
    trans = str(trans_path) if trans_path.exists() else "fsaverage"
    
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=mindist,
        verbose=False,
    )
    
    if logger:
        logger.info(f"Forward model: {fwd['nsource']} sources")
    
    return fwd, src, bem


def _setup_surface_forward_model_configured(
    info: Any,
    *,
    subject: str,
    subjects_dir: str,
    spacing: str,
    trans: str,
    bem: str,
    mindist_mm: float,
    logger: Optional[logging.Logger],
) -> Tuple[Any, Any]:
    import mne

    src = mne.setup_source_space(
        subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=float(mindist_mm),
        verbose=False,
    )
    fwd = mne.convert_forward_solution(
        fwd,
        surf_ori=True,
        copy=False,
        use_cps=True,
        verbose=False,
    )
    if logger:
        logger.info("Surface forward model: %d sources", int(fwd["nsource"]))
    return fwd, src


def _setup_volume_source_space_configured(
    info: Any,
    *,
    subject: str,
    subjects_dir: str,
    trans: str,
    bem: str,
    roi_coords_m: Dict[str, np.ndarray],
    mindist_mm: float,
    logger: Optional[logging.Logger],
) -> Tuple[Any, Any, Dict[str, List[int]]]:
    import mne

    all_coords_m: List[np.ndarray] = []
    roi_indices: Dict[str, List[int]] = {}
    idx = 0
    for roi_name, coords_m in roi_coords_m.items():
        coords_m = np.atleast_2d(np.asarray(coords_m, dtype=float))
        roi_indices[roi_name] = list(range(idx, idx + int(coords_m.shape[0])))
        all_coords_m.append(coords_m)
        idx += int(coords_m.shape[0])

    rr = np.vstack(all_coords_m) if all_coords_m else np.zeros((0, 3), dtype=float)
    pos = {
        "rr": rr,
        "nn": np.tile([0.0, 0.0, 1.0], (int(rr.shape[0]), 1)),
    }

    src = mne.setup_volume_source_space(
        subject,
        pos=pos,
        subjects_dir=subjects_dir,
        verbose=False,
    )

    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=float(mindist_mm),
        verbose=False,
    )

    if logger:
        logger.info("Volume forward model (fMRI constrained): %d sources", int(fwd["nsource"]))

    return fwd, src, roi_indices


def _setup_volume_source_space_from_points_configured(
    info: Any,
    *,
    subject: str,
    subjects_dir: str,
    trans: str,
    bem: str,
    coords_m: np.ndarray,
    mindist_mm: float,
    logger: Optional[logging.Logger],
) -> Tuple[Any, Any]:
    """Set up a discrete volume source space from explicit point coordinates."""
    import mne

    coords = np.atleast_2d(np.asarray(coords_m, dtype=float))
    pos = {
        "rr": coords,
        "nn": np.tile([0.0, 0.0, 1.0], (int(coords.shape[0]), 1)),
    }

    src = mne.setup_volume_source_space(
        subject,
        pos=pos,
        subjects_dir=subjects_dir,
        verbose=False,
    )
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=float(mindist_mm),
        verbose=False,
    )
    if logger:
        logger.info("Volume forward model (fMRI constrained): %d sources", int(fwd["nsource"]))
    return fwd, src


def _make_fmri_subsampling_rng(seed: int) -> np.random.Generator:
    """Create deterministic RNG for fMRI voxel subsampling."""
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError(f"fMRI random_seed must be >= 0 (got {seed_int}).")
    return np.random.default_rng(seed_int)


def _iter_subject_mri_reference_paths(subjects_dir: Path, subject: str) -> List[Path]:
    subject_mri_dir = Path(subjects_dir) / str(subject) / "mri"
    return [
        subject_mri_dir / "orig.mgz",
        subject_mri_dir / "T1.mgz",
    ]


def _iter_subject_aparcaseg_paths(subjects_dir: Path, subject: str) -> List[Path]:
    subject_mri_dir = Path(subjects_dir) / str(subject) / "mri"
    return [
        subject_mri_dir / "aparc+aseg.mgz",
        subject_mri_dir / "aparc+aseg.nii.gz",
        subject_mri_dir / "aparc+aseg.nii",
    ]


def _resolve_subject_aparcaseg_path(subjects_dir: Path, subject: str) -> Path:
    candidates = _iter_subject_aparcaseg_paths(subjects_dir, subject)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "fMRI output_space='atlas' or 'dual' requires aparc+aseg in subject MRI space. "
        f"Expected one of: {', '.join(str(path) for path in candidates)}"
    )


def _atlas_label_name(label_id: int) -> str:
    if label_id <= 0:
        raise ValueError(f"Atlas label id must be > 0 (got {label_id}).")
    return f"aparc_aseg_id{int(label_id)}"


def _validate_stats_map_grid_against_subject_mri(
    *,
    nib: Any,
    map_shape: Tuple[int, int, int],
    map_affine: np.ndarray,
    stats_map_path: Path,
    subjects_dir: Path,
    subject: str,
) -> Path:
    """Require exact voxel-grid match against FreeSurfer reference MRI."""
    ref_candidates = _iter_subject_mri_reference_paths(subjects_dir, subject)
    existing_refs = [path for path in ref_candidates if path.exists()]
    if not existing_refs:
        raise FileNotFoundError(
            "Cannot validate fMRI map alignment: no FreeSurfer reference MRI found. "
            f"Expected one of: {', '.join(str(p) for p in ref_candidates)}"
        )

    mismatch_notes: List[str] = []
    for ref_path in existing_refs:
        ref_img = nib.load(str(ref_path))
        ref_shape = tuple(int(v) for v in ref_img.shape[:3])
        ref_affine = np.asarray(ref_img.affine, dtype=float)
        shape_match = tuple(map_shape) == tuple(ref_shape)
        affine_match = np.allclose(map_affine, ref_affine, rtol=1e-5, atol=1e-3)
        if shape_match and affine_match:
            return ref_path

        max_affine_diff = float(np.max(np.abs(map_affine - ref_affine)))
        mismatch_notes.append(
            f"{ref_path.name}: shape={ref_shape}, max_affine_diff={max_affine_diff:.6g}"
        )

    raise ValueError(
        "fMRI stats map is not on the same voxel grid as the FreeSurfer subject MRI. "
        f"Map={stats_map_path} shape={map_shape}. Checked references: "
        + "; ".join(mismatch_notes)
        + ". Resample the map to subject space (orig.mgz or T1.mgz) and retry."
    )


def _compute_fmri_analysis_voxel_mask(
    stats_data: np.ndarray,
    subject_ref_data: np.ndarray,
) -> np.ndarray:
    """
    Select voxel universe for thresholding/FDR.

    Uses finite voxels in the subject reference MRI support (non-zero anatomy)
    so in-brain zero-stat voxels are retained for FDR while padded/background
    map voxels are excluded.
    """
    if stats_data.shape != subject_ref_data.shape:
        raise ValueError(
            "Subject MRI reference shape does not match fMRI stats map shape "
            f"(stats={stats_data.shape}, ref={subject_ref_data.shape})."
        )
    finite_stats = np.isfinite(stats_data)
    finite_ref = np.isfinite(subject_ref_data)
    in_reference_support = np.abs(subject_ref_data) > np.finfo(np.float32).eps
    analysis_voxels = finite_stats & finite_ref & in_reference_support
    if not np.any(analysis_voxels):
        raise ValueError(
            "fMRI stats map contains no finite voxels inside the subject MRI support; "
            "cannot compute thresholded ROIs."
        )
    return analysis_voxels


def _stats_map_sidecar_path(stats_map_path: Path) -> Path:
    return stats_map_path.with_suffix("").with_suffix(".json")


def _read_stats_map_sidecar(stats_map_path: Path) -> Dict[str, Any]:
    sidecar_path = _stats_map_sidecar_path(stats_map_path)
    if not sidecar_path.exists():
        return {}
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"fMRI stats map sidecar is not valid JSON: {sidecar_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"fMRI stats map sidecar is not a JSON object: {sidecar_path}"
        )
    return payload


def _is_z_type_value(value: Any) -> bool:
    token = str(value or "").strip().lower().replace("-", "_")
    return token in {
        "z",
        "zscore",
        "z_score",
        "zstat",
        "z_stat",
    }


def _assert_fdr_compatible_z_map(
    *,
    stats_map_path: Path,
    cfg: FMRIConstraintConfig,
    nib: Any,
    img: Any,
) -> None:
    """Require explicit evidence that FDR is being applied to a z-stat map."""
    if str(cfg.threshold_mode).lower() != "fdr":
        return

    if str(cfg.stat_type).lower() != "z":
        raise ValueError(
            "FDR thresholding requires feature_engineering.sourcelocalization.fmri.thresholding.stat_type='z'."
        )

    evidence: List[str] = []

    sidecar = _read_stats_map_sidecar(stats_map_path)
    for key in ("output_type_actual", "output_type_requested", "output_type", "stat_type", "StatisticType"):
        if key in sidecar and _is_z_type_value(sidecar.get(key)):
            evidence.append(f"sidecar:{key}")

    map_name = stats_map_path.name.lower().replace("-", "_")
    if any(token in map_name for token in ("_z_", "zmap", "z_score", "zstat", "stat_z")):
        evidence.append("filename")

    intent_code = int(
        getattr(getattr(img, "header", None), "get_intent", lambda: (0, (), ""))()[0]
        or 0
    )
    intent_code_map = (
        getattr(getattr(getattr(nib, "nifti1", None), "intent_codes", None), "code", {})
        or {}
    )
    z_intent_code = int(intent_code_map.get("z score", -1))
    if intent_code > 0 and intent_code == z_intent_code:
        evidence.append("nifti_intent")

    if evidence:
        return

    sidecar_path = _stats_map_sidecar_path(stats_map_path)
    raise ValueError(
        "FDR thresholding requires a z-statistics map, but explicit z-map evidence was not found. "
        f"Map={stats_map_path}. Checked filename, NIfTI intent code, and sidecar ({sidecar_path}). "
        "Provide a z-map and sidecar with output_type_actual='z_score' (or equivalent)."
    )


def _fmri_roi_coords_from_stats_map(
    stats_map_path: Path,
    cfg: FMRIConstraintConfig,
    logger: Optional[logging.Logger],
    *,
    subjects_dir: Path,
    subject: str,
) -> Dict[str, np.ndarray]:
    """Backward-compatible wrapper returning cluster ROI coordinates only."""
    selection = _select_fmri_constrained_voxels(
        stats_map_path=stats_map_path,
        cfg=cfg,
        logger=logger,
        subjects_dir=subjects_dir,
        subject=subject,
        include_atlas=False,
    )
    roi_coords_m: Dict[str, np.ndarray] = {}
    for roi_name, roi_idxs in selection.cluster_indices.items():
        if not roi_idxs:
            continue
        roi_coords_m[roi_name] = selection.selected_coords_m[np.asarray(roi_idxs, dtype=int)]
    if not roi_coords_m:
        raise ValueError("fMRI constraint produced no ROI coordinates after subsampling.")
    return roi_coords_m


def _map_selected_voxels_to_aparcaseg(
    *,
    nib: Any,
    selected_voxels_ijk: np.ndarray,
    subjects_dir: Path,
    subject: str,
) -> Tuple[Dict[str, List[int]], Dict[str, int], np.ndarray, int]:
    """
    Map selected voxels to subject-space aparc+aseg atlas labels.

    Returns
    -------
    atlas_indices
        Atlas label name -> selected-point indices.
    atlas_counts
        Atlas label name -> number of selected points.
    atlas_ids_per_point
        Integer atlas ID per selected point (0 means unlabeled/background).
    dropped_unlabeled
        Number of selected points with atlas ID <= 0.
    """
    atlas_path = _resolve_subject_aparcaseg_path(Path(subjects_dir), str(subject))
    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.asarray(atlas_img.get_fdata(dtype=np.float32))
    if atlas_data.ndim != 3:
        raise ValueError(
            f"Expected 3D aparc+aseg atlas, got shape={atlas_data.shape} from {atlas_path}."
        )

    selected_voxels = np.asarray(selected_voxels_ijk, dtype=int)
    if selected_voxels.ndim != 2 or selected_voxels.shape[1] != 3:
        raise ValueError(
            f"Selected voxels must have shape (n, 3), got {selected_voxels.shape}."
        )
    if np.any(selected_voxels < 0):
        raise ValueError("Selected voxel indices must be non-negative.")
    if (
        np.any(selected_voxels[:, 0] >= atlas_data.shape[0])
        or np.any(selected_voxels[:, 1] >= atlas_data.shape[1])
        or np.any(selected_voxels[:, 2] >= atlas_data.shape[2])
    ):
        raise ValueError(
            "Selected voxel indices exceed aparc+aseg bounds "
            f"(atlas_shape={atlas_data.shape})."
        )

    atlas_ids = atlas_data[
        selected_voxels[:, 0],
        selected_voxels[:, 1],
        selected_voxels[:, 2],
    ].astype(int, copy=False)

    atlas_indices: Dict[str, List[int]] = {}
    for point_idx, atlas_id in enumerate(atlas_ids.tolist()):
        if int(atlas_id) <= 0:
            continue
        label_name = _atlas_label_name(int(atlas_id))
        atlas_indices.setdefault(label_name, []).append(int(point_idx))

    atlas_counts = {name: int(len(idxs)) for name, idxs in atlas_indices.items()}
    dropped_unlabeled = int(np.sum(atlas_ids <= 0))
    return atlas_indices, atlas_counts, atlas_ids, dropped_unlabeled


def _select_fmri_constrained_voxels(
    *,
    stats_map_path: Path,
    cfg: FMRIConstraintConfig,
    logger: Optional[logging.Logger],
    subjects_dir: Path,
    subject: str,
    include_atlas: bool,
) -> FMRIVoxelSelection:
    """
    Select thresholded fMRI voxels and build cluster/atlas ROI index mappings.

    IMPORTANT: `stats_map_path` must be on the same voxel grid as the subject MRI.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "fMRI-constrained source localization requires nibabel. "
            "Install nibabel (e.g., `pip install nibabel`) and retry."
        ) from exc

    from scipy.ndimage import generate_binary_structure, label as cc_label

    img = nib.load(str(stats_map_path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    affine = np.asarray(img.affine, dtype=float)

    voxel_vol_mm3 = None
    try:
        zooms = img.header.get_zooms()
        if len(zooms) >= 3:
            voxel_vol_mm3 = float(abs(float(zooms[0]) * float(zooms[1]) * float(zooms[2])))
            if not np.isfinite(voxel_vol_mm3) or voxel_vol_mm3 <= 0:
                voxel_vol_mm3 = None
    except Exception:
        voxel_vol_mm3 = None

    if data.ndim != 3:
        raise ValueError(
            f"Expected a single-volume 3D NIfTI fMRI map, got shape={data.shape}. "
            "Provide a 3D contrast/stat map (not 4D)."
        )

    matched_ref_path = _validate_stats_map_grid_against_subject_mri(
        nib=nib,
        map_shape=tuple(int(v) for v in data.shape),
        map_affine=affine,
        stats_map_path=stats_map_path,
        subjects_dir=Path(subjects_dir),
        subject=str(subject),
    )
    ref_img = nib.load(str(matched_ref_path))
    subject_ref_data = np.asarray(ref_img.get_fdata(dtype=np.float32))
    if subject_ref_data.ndim != 3:
        raise ValueError(
            f"Expected 3D subject MRI reference, got shape={subject_ref_data.shape} "
            f"from {matched_ref_path}."
        )

    # MNE source spaces use FreeSurfer surface RAS (tkRAS), NOT scanner RAS.
    # The NIfTI affine maps voxels → scanner RAS, but the BEM surfaces live
    # in surface RAS. The vox2ras_tkr transform from the FreeSurfer reference
    # volume provides the correct voxel → surface RAS mapping.
    vox2ras_tkr = np.asarray(ref_img.header.get_vox2ras_tkr(), dtype=float)

    analysis_voxels = _compute_fmri_analysis_voxel_mask(
        data,
        subject_ref_data=subject_ref_data,
    )
    if cfg.prethresholded_mask:
        mask = analysis_voxels & np.isfinite(data) & (data > 0)
    else:
        thr = float(cfg.threshold)
        if not np.isfinite(thr) or thr <= 0:
            raise ValueError(
                "feature_engineering.sourcelocalization.fmri.threshold must be > 0 "
                f"(got {thr})."
            )

        if str(cfg.threshold_mode).lower() == "fdr":
            q = float(cfg.fdr_q)
            if not (np.isfinite(q) and 0 < q < 1):
                raise ValueError(
                    "feature_engineering.sourcelocalization.fmri.thresholding.fdr_q must be in (0, 1). "
                    f"(got {q})."
                )

            _assert_fdr_compatible_z_map(
                stats_map_path=stats_map_path,
                cfg=cfg,
                nib=nib,
                img=img,
            )

            zvals = data[analysis_voxels].astype(float, copy=False)
            if cfg.tail == "abs":
                pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(zvals)))
            else:
                pvals = 1.0 - stats.norm.cdf(zvals)
            pvals = np.clip(pvals, 0.0, 1.0)

            order = np.argsort(pvals)
            ranked = pvals[order]
            n = int(ranked.size)
            if n < 1:
                raise ValueError("No finite voxels found in fMRI stats map.")

            bh_thresh = (q * (np.arange(1, n + 1, dtype=float) / float(n)))
            below = ranked <= bh_thresh
            if not np.any(below):
                mask = np.zeros_like(data, dtype=bool)
            else:
                k = int(np.max(np.where(below)[0]))
                cutoff = float(ranked[k])
                keep = pvals <= cutoff
                mask = np.zeros_like(data, dtype=bool)
                mask[analysis_voxels] = keep
        else:
            if cfg.tail == "abs":
                mask = analysis_voxels & (np.abs(data) >= thr)
            else:
                mask = analysis_voxels & (data >= thr)

    if not np.any(mask):
        if cfg.prethresholded_mask:
            raise ValueError(
                "The prethresholded fMRI constraint mask contains no nonzero voxels on the subject MRI grid."
            )

        finite_data = data[analysis_voxels]
        if logger is not None and len(finite_data) > 0:
            logger.warning(
                f"fMRI stats map threshold produced empty mask (threshold={thr}, tail={cfg.tail}, mode={cfg.threshold_mode}). "
                f"Stats: min={np.nanmin(finite_data):.3f}, max={np.nanmax(finite_data):.3f}, "
                f"mean={np.nanmean(finite_data):.3f}, 95th percentile={np.nanpercentile(finite_data, 95):.3f}"
            )
        raise ValueError(
            f"fMRI stats map threshold produced empty mask (threshold={thr}, tail={cfg.tail}, mode={cfg.threshold_mode}). "
            "Consider lowering the threshold (mode='z') or increasing fdr_q (mode='fdr')."
        )

    structure = generate_binary_structure(3, 1)  # 6-connectivity
    labeled, n_clusters = cc_label(mask.astype(np.uint8), structure=structure)
    if n_clusters < 1:
        raise ValueError("No connected components found in fMRI-thresholded mask.")

    clusters: List[Tuple[int, int, float]] = []
    for cluster_id in range(1, int(n_clusters) + 1):
        vox = np.argwhere(labeled == cluster_id)
        if vox.size == 0:
            continue
        n_vox = int(vox.shape[0])
        if not cfg.prethresholded_mask:
            if cfg.cluster_min_volume_mm3 is not None and voxel_vol_mm3 is not None:
                vol = float(n_vox) * float(voxel_vol_mm3)
                if vol < float(cfg.cluster_min_volume_mm3):
                    continue
            else:
                if n_vox < int(cfg.cluster_min_voxels):
                    continue
        peak_val = float(np.nanmax(np.abs(data[labeled == cluster_id])))
        clusters.append((cluster_id, n_vox, peak_val))

    if not clusters:
        if cfg.prethresholded_mask:
            raise ValueError("The prethresholded fMRI constraint mask contains no connected components.")
        if cfg.cluster_min_volume_mm3 is not None and voxel_vol_mm3 is not None:
            raise ValueError(
                "All clusters were smaller than "
                f"cluster_min_volume_mm3={float(cfg.cluster_min_volume_mm3):g} (voxel_vol_mm3={float(voxel_vol_mm3):g})."
            )
        raise ValueError(f"All clusters were smaller than cluster_min_voxels={cfg.cluster_min_voxels}.")

    clusters.sort(key=lambda t: (t[2], t[1]), reverse=True)
    clusters = clusters[: max(1, int(cfg.max_clusters))]

    rng = _make_fmri_subsampling_rng(int(cfg.random_seed))

    selected_voxels_by_cluster: Dict[str, np.ndarray] = {}
    cluster_voxel_counts: Dict[str, int] = {}
    total_selected = 0
    for out_idx, (cluster_id, _, peak_val) in enumerate(clusters, start=1):
        vox = np.argwhere(labeled == cluster_id)
        if vox.size == 0:
            continue

        if int(cfg.max_voxels_per_cluster) > 0 and int(vox.shape[0]) > int(cfg.max_voxels_per_cluster):
            pick = rng.choice(int(vox.shape[0]), size=int(cfg.max_voxels_per_cluster), replace=False)
            vox = vox[pick]

        if int(cfg.max_total_voxels) > 0:
            remaining = int(cfg.max_total_voxels) - total_selected
            if remaining <= 0:
                break
            if int(vox.shape[0]) > remaining:
                pick = rng.choice(int(vox.shape[0]), size=int(remaining), replace=False)
                vox = vox[pick]

        roi_name = f"fmri_c{out_idx:02d}_peak{peak_val:.2f}".replace(".", "p")
        selected_voxels_by_cluster[roi_name] = np.asarray(vox, dtype=int)
        n_kept = int(vox.shape[0])
        cluster_voxel_counts[roi_name] = n_kept
        total_selected += n_kept

    if logger:
        logger.info(
            "fMRI constraint: %d clusters, %d voxels (map=%s)",
            len(selected_voxels_by_cluster),
            total_selected,
            str(stats_map_path),
        )

    if not selected_voxels_by_cluster:
        raise ValueError("fMRI constraint produced no ROI coordinates after subsampling.")

    selected_voxels = np.vstack(list(selected_voxels_by_cluster.values()))
    selected_voxels = np.asarray(selected_voxels, dtype=int)
    coords_mm = nib.affines.apply_affine(vox2ras_tkr, selected_voxels)
    selected_coords_m = np.asarray(coords_mm, dtype=float) / 1000.0

    cluster_indices: Dict[str, List[int]] = {}
    cluster_to_atlas_counts: Dict[str, Dict[str, int]] = {}
    atlas_indices: Dict[str, List[int]] = {}
    atlas_counts: Dict[str, int] = {}
    dropped_unlabeled = 0

    start = 0
    for cluster_name, vox in selected_voxels_by_cluster.items():
        n_points = int(vox.shape[0])
        cluster_indices[cluster_name] = list(range(start, start + n_points))
        cluster_to_atlas_counts[cluster_name] = {}
        start += n_points

    if include_atlas:
        (
            atlas_indices,
            atlas_counts,
            atlas_ids_per_point,
            dropped_unlabeled,
        ) = _map_selected_voxels_to_aparcaseg(
            nib=nib,
            selected_voxels_ijk=selected_voxels,
            subjects_dir=Path(subjects_dir),
            subject=str(subject),
        )
        for cluster_name, point_indices in cluster_indices.items():
            coverage: Dict[str, int] = {}
            for point_idx in point_indices:
                atlas_id = int(atlas_ids_per_point[int(point_idx)])
                if atlas_id <= 0:
                    continue
                atlas_name = _atlas_label_name(atlas_id)
                coverage[atlas_name] = int(coverage.get(atlas_name, 0) + 1)
            cluster_to_atlas_counts[cluster_name] = coverage

    return FMRIVoxelSelection(
        selected_voxels_ijk=selected_voxels,
        selected_coords_m=selected_coords_m,
        cluster_indices=cluster_indices,
        cluster_voxel_counts=cluster_voxel_counts,
        atlas_indices=atlas_indices,
        atlas_voxel_counts=atlas_counts,
        cluster_to_atlas_counts=cluster_to_atlas_counts,
        dropped_unlabeled_voxels=dropped_unlabeled,
        matched_reference_path=Path(matched_ref_path),
        voxel_volume_mm3=voxel_vol_mm3,
    )


def _resolve_surviving_roi_rows(
    *,
    stc_vertices: np.ndarray,
    roi_indices: Dict[str, List[int]],
) -> Tuple[Dict[str, List[int]], List[str], List[str]]:
    """Map requested ROI vertex indices onto surviving STC rows."""
    vert_to_idx = {v: i for i, v in enumerate(stc_vertices)}

    surviving_roi_names: List[str] = []
    dropped_rois: List[str] = []
    roi_row_indices: Dict[str, List[int]] = {}
    for roi_name, orig_idxs in roi_indices.items():
        if not orig_idxs:
            dropped_rois.append(roi_name)
            continue
        row_idxs = [vert_to_idx[v] for v in orig_idxs if v in vert_to_idx]
        if not row_idxs:
            dropped_rois.append(roi_name)
            continue
        surviving_roi_names.append(roi_name)
        roi_row_indices[roi_name] = row_idxs
    return roi_row_indices, surviving_roi_names, dropped_rois


def _extract_roi_timecourses_from_vertex_indices(
    stcs: List[Any],
    roi_indices: Dict[str, List[int]],
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract ROI timecourses from STCs using explicit vertex mapping to handle pruned sources."""
    n_epochs = len(stcs)
    if n_epochs == 0:
        return np.zeros((0, 0, 0)), []

    n_times = int(stcs[0].data.shape[1])

    # stcs[0].vertices is a list (one per source space). For volume, usually [vertno]
    stc_verts = np.asarray(stcs[0].vertices[0], dtype=int)
    roi_row_indices, surviving_roi_names, dropped_rois = _resolve_surviving_roi_rows(
        stc_vertices=stc_verts,
        roi_indices=roi_indices,
    )

    if dropped_rois and logger is not None:
        logger.warning(
            "fMRI-constrained ROI mapping dropped %d ROI(s) with no surviving vertices: %s",
            len(dropped_rois),
            ", ".join(dropped_rois),
        )

    if not surviving_roi_names:
        raise ValueError(
            "fMRI-constrained ROI mapping failed: no surviving vertices remained "
            "after forward/inverse modeling for any ROI."
        )

    out = _compute_roi_timecourses_from_row_indices(
        stcs=stcs,
        roi_row_indices=roi_row_indices,
        roi_names=surviving_roi_names,
    )
    return out, surviving_roi_names


def _compute_roi_timecourses_from_row_indices(
    *,
    stcs: List[Any],
    roi_row_indices: Dict[str, List[int]],
    roi_names: List[str],
) -> np.ndarray:
    """Extract ROI time-courses from pre-resolved STC row indices."""
    n_epochs = len(stcs)
    if n_epochs == 0:
        return np.zeros((0, 0, 0), dtype=float)

    n_times = int(stcs[0].data.shape[1])
    stc_data_list = [np.asarray(stc.data, dtype=float) for stc in stcs]

    n_rois = len(roi_names)
    out = np.full((n_epochs, n_rois, n_times), np.nan, dtype=float)

    for roi_idx, roi_name in enumerate(roi_names):
        row_idxs = roi_row_indices[roi_name]

        blocks: List[np.ndarray] = []
        for data in stc_data_list:
            block = data[row_idxs]
            if block.ndim == 3:
                # e.g., VectorSourceEstimate with 3 unconstrained orientations per region
                block = block.reshape(-1, block.shape[-1])
            blocks.append(block)

        block_stack = np.stack(blocks, axis=0)  # (n_epochs, n_verts, n_times)
        if block_stack.shape[1] == 0:
            out[:, roi_idx, :] = np.nan
            continue
        if block_stack.shape[1] == 1:
            out[:, roi_idx, :] = block_stack[:, 0, :]
            continue

        # Flatten to (n_verts, n_epochs * n_times) to compute stable cross-epoch covariance
        block_flat = np.nan_to_num(block_stack).transpose(1, 0, 2).reshape(block_stack.shape[1], -1)
        cov = block_flat @ block_flat.T
        u, _, _ = np.linalg.svd(cov, full_matrices=False)
        flip = np.sign(u[:, 0])
        flip_sign = np.sign(np.sum(flip))
        if flip_sign != 0:
            flip *= flip_sign
        out[:, roi_idx, :] = np.nanmean(block_stack * flip[np.newaxis, :, np.newaxis], axis=1)

    return out


###################################################################
# LCMV Beamformer
###################################################################


def _compute_lcmv_source_estimates(
    epochs_apply: "mne.Epochs",
    fwd: Any,
    *,
    epochs_fit: Optional["mne.Epochs"] = None,
    data_cov: Optional[Any] = None,
    noise_cov: Optional[Any] = None,
    reg: float = 0.05,
    pick_ori: str = "max-power",
    weight_norm: str = "unit-noise-gain",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], Any]:
    """
    Compute LCMV beamformer source estimates for epochs.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
    fwd : mne.Forward
        Forward solution
    data_cov : mne.Covariance, optional
        Data covariance (computed from epochs if None)
    noise_cov : mne.Covariance, optional
        Noise covariance for whitening
    reg : float
        Regularization parameter
    pick_ori : str
        Orientation picking strategy
    weight_norm : str
        Weight normalization method
        
    Returns
    -------
    stcs : list of mne.SourceEstimate
        Source estimates for each epoch
    filters : mne.beamformer.Beamformer
        LCMV spatial filters
    """
    import mne
    from mne.beamformer import make_lcmv, apply_lcmv_epochs
    
    if logger:
        logger.info("Computing LCMV beamformer source estimates")
    
    epochs_for_cov = epochs_fit if epochs_fit is not None else epochs_apply
    if data_cov is None:
        data_cov = mne.compute_covariance(
            epochs_for_cov,
            method="empirical",
            keep_sample_mean=False,
            verbose=False,
        )
    
    filters = make_lcmv(
        epochs_apply.info,
        fwd,
        data_cov,
        reg=reg,
        noise_cov=noise_cov,
        pick_ori=pick_ori,
        weight_norm=weight_norm,
        rank="info",
        verbose=False,
    )
    
    stcs = apply_lcmv_epochs(epochs_apply, filters, verbose=False)
    
    if logger:
        logger.info(f"LCMV: {len(stcs)} epochs, {stcs[0].data.shape[0]} sources")
    
    return stcs, filters


###################################################################
# eLORETA Inverse Solution
###################################################################


def _compute_eloreta_source_estimates(
    epochs: "mne.Epochs",
    fwd: Any,
    noise_cov: Optional[Any] = None,
    loose: float = 0.2,
    depth: float = 0.8,
    snr: float = 3.0,
    pick_ori: Optional[str] = "normal",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], Any]:
    """
    Compute eLORETA inverse solution source estimates.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
    fwd : mne.Forward
        Forward solution
    noise_cov : mne.Covariance, optional
        Noise covariance (identity if None)
    loose : float
        Loose orientation constraint (0-1)
    depth : float
        Depth weighting (0-1)
    snr : float
        Assumed SNR for regularization
    pick_ori : {'normal', None, 'vector'}
        Orientation selection for inverse estimates. Use ``None`` for
        volume/discrete source spaces.
        
    Returns
    -------
    stcs : list of mne.SourceEstimate
        Source estimates for each epoch
    inv : mne.minimum_norm.InverseOperator
        Inverse operator
    """
    src_spaces = fwd.get("src") if isinstance(fwd, dict) else None
    src_types: List[str] = []
    if isinstance(src_spaces, (list, tuple)):
        for src in src_spaces:
            if isinstance(src, dict):
                src_types.append(str(src.get("type", "")).strip().lower())
            else:
                try:
                    src_types.append(str(src["type"]).strip().lower())
                except Exception:
                    continue
    src_types = [t for t in src_types if t]
    surface_only = bool(src_types) and all(t == "surf" for t in src_types)
    has_nonsurface = bool(src_types) and not surface_only
    if pick_ori == "normal" and has_nonsurface:
        raise ValueError(
            "eLORETA pick_ori='normal' is only valid for cortical surface source spaces. "
            "For volume/discrete source spaces (including fMRI constraints), use pick_ori=None."
        )
    if has_nonsurface and float(loose) < 1.0:
        raise ValueError(
            "eLORETA with volume/discrete source spaces requires loose=1.0 "
            "(free orientation)."
        )

    import mne
    from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
    
    if logger:
        logger.info("Computing eLORETA source estimates")
    
    if noise_cov is None:
        noise_cov = mne.make_ad_hoc_cov(epochs.info, verbose=False)
    
    inv = make_inverse_operator(
        epochs.info,
        fwd,
        noise_cov,
        loose=loose,
        depth=depth,
        verbose=False,
    )
    
    lambda2 = 1.0 / snr ** 2
    
    stcs = apply_inverse_epochs(
        epochs,
        inv,
        lambda2=lambda2,
        method="eLORETA",
        pick_ori=pick_ori,
        verbose=False,
    )
    
    if logger:
        logger.info(f"eLORETA: {len(stcs)} epochs, {stcs[0].data.shape[0]} sources")
    
    return stcs, inv


###################################################################
# ROI Feature Extraction
###################################################################


def _extract_roi_timecourses(
    stcs: List[Any],
    labels: List[Any],
    src: Any,
    mode: str = "mean_flip",
) -> np.ndarray:
    """
    Extract ROI time courses from source estimates.
    
    Parameters
    ----------
    stcs : list of mne.SourceEstimate
        Source estimates
    labels : list of mne.Label
        ROI labels
    src : mne.SourceSpaces
        Source space used for the forward model
    mode : str
        Extraction mode: 'mean', 'mean_flip', 'pca_flip', 'max'
        
    Returns
    -------
    roi_data : ndarray, shape (n_epochs, n_rois, n_times)
        ROI time courses
    """
    import mne
    
    try:
        tcs = mne.extract_label_time_course(
            stcs,
            labels=labels,
            src=src,
            mode=mode,
            allow_empty=False,
            verbose=False,
        )
    except ValueError as exc:
        raise ValueError(
            "ROI extraction failed because one or more labels have no vertices in the source space. "
            "Check parcellation/source-space compatibility and do not use empty labels."
        ) from exc
        
    return np.asarray(tcs, dtype=float)


def _compute_roi_power(
    roi_data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Compute band power for ROI time courses.
    
    Parameters
    ----------
    roi_data : ndarray, shape (n_epochs, n_rois, n_times)
        ROI time courses
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency band limits
        
    Returns
    -------
    power : ndarray, shape (n_epochs, n_rois)
        Band power per epoch and ROI
    """
    from mne.time_frequency import psd_array_welch
    
    n_epochs, n_rois, n_times = roi_data.shape
    power = np.zeros((n_epochs, n_rois))

    nyquist = float(sfreq) / 2.0
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Sampling frequency must be > 0 for ROI power (got {sfreq}).")
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin <= 0 or fmin >= fmax:
        raise ValueError(f"Invalid band range for ROI power: fmin={fmin}, fmax={fmax}.")
    if float(fmax) >= nyquist:
        raise ValueError(
            f"ROI power band upper edge must be below Nyquist ({nyquist:.6g} Hz), got fmax={fmax}."
        )
    
    bad_mask = np.any(np.isnan(roi_data), axis=-1) | (np.std(roi_data, axis=-1) < 1e-12)
    valid_mask = ~bad_mask

    if not np.any(valid_mask):
        power[:] = np.nan
        return power
    
    valid_data = roi_data[valid_mask]
    nperseg = min(n_times, int(sfreq * 2))
    
    psds, freqs = psd_array_welch(
        valid_data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=nperseg,
        n_per_seg=nperseg,
        n_overlap=nperseg // 2,
        n_jobs=1,
        verbose=False,
    )
    
    if len(freqs) < 1:
        raise ValueError(
            "ROI power band has no Welch frequency bins; increase segment length or use a lower band. "
            f"(fmin={fmin}, fmax={fmax}, sfreq={sfreq}, n_times={n_times})"
        )
    
    # Integrate discrete PSD over frequency to compute total band power
    df = freqs[1] - freqs[0] if len(freqs) > 1 else sfreq / nperseg
    valid_power = np.sum(psds, axis=-1) * df
    
    power[valid_mask] = valid_power
    power[bad_mask] = np.nan
    
    return power


def _compute_roi_envelope(
    roi_data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Compute band-limited amplitude envelope for ROI time courses.
    
    Parameters
    ----------
    roi_data : ndarray, shape (n_epochs, n_rois, n_times)
        ROI time courses
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency band limits
        
    Returns
    -------
    envelope : ndarray, shape (n_epochs, n_rois, n_times)
        Amplitude envelope
    """
    from mne.filter import filter_data
    from scipy.signal import hilbert
    
    n_epochs, n_rois, n_times = roi_data.shape
    envelope = np.zeros_like(roi_data)
    
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Sampling frequency must be > 0 for ROI envelope (got {sfreq}).")
    nyq = sfreq / 2.0
    if fmin <= 0 or fmax >= nyq or fmin >= fmax:
        raise ValueError(f"Invalid bandpass range for ROI envelope: fmin={fmin}, fmax={fmax}, sfreq={sfreq}.")

    bad_mask = np.any(np.isnan(roi_data), axis=-1) | (np.std(roi_data, axis=-1) < 1e-12)
    valid_mask = ~bad_mask

    if not np.any(valid_mask):
        envelope[:] = np.nan
        return envelope
    
    valid_data = roi_data[valid_mask]
    
    # Use zero-phase IIR filtering to avoid FIR-kernel-length distortion on short windows.
    filtered_data = filter_data(
        valid_data,
        sfreq=sfreq,
        l_freq=fmin,
        h_freq=fmax,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        phase="zero",
        copy=True,
        verbose=False,
    )
    
    # Hilbert transform applies across last axis
    analytic = hilbert(filtered_data)
    valid_envelope = np.abs(analytic)
    
    envelope[valid_mask] = valid_envelope
    envelope[bad_mask] = np.nan
    
    return envelope


###################################################################
# Configuration Loading
###################################################################


@dataclass(frozen=True)
class SourceLocalizationConfig:
    """Source localization configuration parameters."""
    mode: str
    allow_template_fallback: bool
    method: str
    spacing: str
    parcellation: str
    subject: str
    subjects_dir: Optional[Path]
    trans_path: Optional[Path]
    bem_path: Optional[Path]
    mindist_mm: float
    lcmv_reg: float
    eloreta_snr: float
    eloreta_loose: float
    eloreta_depth: float
    fmri_cfg: FMRIConstraintConfig
    save_stc: bool = False


@dataclass(frozen=True)
class SourceContrastConfig:
    """Condition-contrast configuration for source-localized features."""

    enabled: bool
    condition_column: str
    condition_a: str
    condition_b: str
    min_trials_per_condition: int
    emit_welch_stats: bool


def _parse_bool_config(value: Any, key: str, *, default: bool) -> bool:
    """Parse a boolean config value with strict type validation."""
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{key} must be a boolean (got {value!r}).")


def _sanitize_feature_token(value: Any) -> str:
    token = str(value).strip()
    for symbol in (" ", "-", "/", "\\", ":", ";", ",", "|", "(", ")", "[", "]", "{", "}", "+", "="):
        token = token.replace(symbol, "_")
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError(f"Invalid empty token in source contrast naming (value={value!r}).")
    return token


def _cohen_d_unpaired(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    dof = int(a.size + b.size - 2)
    if dof <= 0:
        return float("nan")
    pooled_var = (((a.size - 1) * var_a) + ((b.size - 1) * var_b)) / float(dof)
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


def _load_source_contrast_config(config: Any) -> SourceContrastConfig:
    """Load and validate source contrast configuration."""
    src_cfg = get_config_value(config, "feature_engineering.sourcelocalization", {}) or {}
    if not isinstance(src_cfg, dict):
        src_cfg = {}
    contrast_cfg = src_cfg.get("contrast", {}) or {}
    if not isinstance(contrast_cfg, dict):
        raise ValueError(
            "feature_engineering.sourcelocalization.contrast must be a mapping."
        )

    enabled = _parse_bool_config(
        contrast_cfg.get("enabled", False),
        "feature_engineering.sourcelocalization.contrast.enabled",
        default=False,
    )

    condition_column = str(contrast_cfg.get("condition_column", "") or "").strip()
    condition_a = str(contrast_cfg.get("condition_a", "") or "").strip()
    condition_b = str(contrast_cfg.get("condition_b", "") or "").strip()
    min_trials_raw = contrast_cfg.get("min_trials_per_condition", 5)
    emit_welch_stats = _parse_bool_config(
        contrast_cfg.get("emit_welch_stats", False),
        "feature_engineering.sourcelocalization.contrast.emit_welch_stats",
        default=False,
    )
    if isinstance(min_trials_raw, bool):
        raise ValueError(
            "feature_engineering.sourcelocalization.contrast.min_trials_per_condition must be an integer >= 1."
        )

    try:
        min_trials_per_condition = int(min_trials_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "feature_engineering.sourcelocalization.contrast.min_trials_per_condition must be an integer >= 1."
        ) from exc
    if min_trials_per_condition < 1:
        raise ValueError(
            "feature_engineering.sourcelocalization.contrast.min_trials_per_condition must be >= 1."
        )

    if enabled:
        if is_resting_state_feature_mode(config):
            raise ValueError(
                "feature_engineering.sourcelocalization.contrast.enabled is not scientifically "
                "valid when feature_engineering.task_is_rest=true."
            )
        if not condition_column:
            raise ValueError(
                "feature_engineering.sourcelocalization.contrast.condition_column is required when "
                "feature_engineering.sourcelocalization.contrast.enabled=true."
            )
        if not condition_a:
            raise ValueError(
                "feature_engineering.sourcelocalization.contrast.condition_a is required when "
                "feature_engineering.sourcelocalization.contrast.enabled=true."
            )
        if not condition_b:
            raise ValueError(
                "feature_engineering.sourcelocalization.contrast.condition_b is required when "
                "feature_engineering.sourcelocalization.contrast.enabled=true."
            )
        if condition_a == condition_b:
            raise ValueError(
                "feature_engineering.sourcelocalization.contrast.condition_a and condition_b must differ."
            )
        if emit_welch_stats:
            raise ValueError(
                "feature_engineering.sourcelocalization.contrast.emit_welch_stats=true is not "
                "scientifically valid for within-subject trial rows. Keep source contrasts "
                "descriptive here and run inferential models at the subject/group level."
            )

    return SourceContrastConfig(
        enabled=enabled,
        condition_column=condition_column,
        condition_a=condition_a,
        condition_b=condition_b,
        min_trials_per_condition=min_trials_per_condition,
        emit_welch_stats=emit_welch_stats,
    )


def _preflight_fmri_trans_requirements(
    config: Any,
    *,
    mode: str,
    fmri_cfg_raw: Dict[str, Any],
    subjects_dir: Optional[Path],
    subject: str,
    trans_path: Optional[Path],
) -> None:
    """Fail fast when fMRI-informed source runs cannot satisfy trans requirements."""
    contrast_cfg = fmri_cfg_raw.get("contrast", {}) or {}
    fmri_needed = (
        str(mode).strip().lower() == "fmri_informed"
        or bool(fmri_cfg_raw.get("enabled", False))
        or bool(contrast_cfg.get("enabled", False))
    )
    if not fmri_needed:
        return
    if trans_path is not None:
        return
    if subjects_dir is None:
        return

    default_trans_path = Path(subjects_dir) / str(subject) / "bem" / f"{subject}-trans.fif"
    if default_trans_path.exists():
        return

    from fmri_pipeline.analysis.bem_generation import get_bem_generation_config

    bem_cfg = get_bem_generation_config(config)
    create_trans = bool(bem_cfg.get("create_trans", False))
    if create_trans:
        # When create_trans is enabled, generation can proceed from EEG
        # digitization/fiducials at runtime.
        return
    if not create_trans:
        raise ValueError(
            "fMRI-constrained source localization requires feature_engineering.sourcelocalization.trans "
            "(EEG↔MRI transform .fif). Set path or enable --source-create-trans to auto-generate."
        )


def _load_source_localization_config(
    ctx: Any,
    config: Any,
    method: str = "lcmv",
) -> SourceLocalizationConfig:
    """Load and validate source localization configuration."""
    src_cfg = get_config_value(config, "feature_engineering.sourcelocalization", {}) or {}
    if not isinstance(src_cfg, dict):
        src_cfg = {}

    mode = str(src_cfg.get("mode", "eeg_only")).strip().lower()
    if mode not in {"eeg_only", "fmri_informed"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.mode must be one of "
            "{'eeg_only','fmri_informed'} "
            f"(got '{mode}')."
        )

    method_use = str(src_cfg.get("method", method)).strip().lower()
    if method_use not in {"lcmv", "eloreta"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.method must be one of "
            "{'lcmv','eloreta'} "
            f"(got '{method_use}')."
        )

    spacing = str(src_cfg.get("spacing", "oct6")).strip()
    if spacing not in {"oct5", "oct6", "ico4", "ico5"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.spacing must be one of "
            "{'oct5','oct6','ico4','ico5'} "
            f"(got '{spacing}')."
        )

    parcellation = str(src_cfg.get("parcellation", src_cfg.get("parc", "aparc"))).strip()
    if parcellation not in {"aparc", "aparc.a2009s", "HCPMMP1"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.parcellation must be one of "
            "{'aparc','aparc.a2009s','HCPMMP1'} "
            f"(got '{parcellation}')."
        )

    subjects_dir_path = _as_path(src_cfg.get("subjects_dir"))
    if subjects_dir_path is None:
        subjects_dir_path = _as_path(get_config_value(config, "paths.freesurfer_dir", None))

    subject_from_cfg = src_cfg.get("subject")
    if subject_from_cfg is None or str(subject_from_cfg).strip().lower() in ("none", ""):
        subject = f"sub-{getattr(ctx, 'subject', '')}".strip() or "fsaverage"
    else:
        subject = str(subject_from_cfg).strip()

    trans_path = _as_path(src_cfg.get("trans"))
    bem_path = _as_path(src_cfg.get("bem"))

    mindist_mm = float(get_config_float(config, "feature_engineering.sourcelocalization.mindist_mm", 5.0))
    lcmv_reg = float(get_config_float(config, "feature_engineering.sourcelocalization.reg", 0.05))
    eloreta_snr = float(get_config_float(config, "feature_engineering.sourcelocalization.snr", 3.0))
    eloreta_loose = float(get_config_float(config, "feature_engineering.sourcelocalization.loose", 0.2))
    eloreta_depth = float(get_config_float(config, "feature_engineering.sourcelocalization.depth", 0.8))

    if not np.isfinite(mindist_mm) or mindist_mm < 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.mindist_mm must be a finite float >= 0 "
            f"(got {mindist_mm})."
        )
    if not np.isfinite(lcmv_reg) or lcmv_reg < 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.reg must be a finite float >= 0 "
            f"(got {lcmv_reg})."
        )
    if not np.isfinite(eloreta_snr) or eloreta_snr <= 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.snr must be a finite float > 0 "
            f"(got {eloreta_snr})."
        )
    if not np.isfinite(eloreta_loose) or eloreta_loose < 0 or eloreta_loose > 1:
        raise ValueError(
            "feature_engineering.sourcelocalization.loose must be a finite float in [0, 1] "
            f"(got {eloreta_loose})."
        )
    if not np.isfinite(eloreta_depth) or eloreta_depth < 0 or eloreta_depth > 1:
        raise ValueError(
            "feature_engineering.sourcelocalization.depth must be a finite float in [0, 1] "
            f"(got {eloreta_depth})."
        )

    bids_fmri_root = _as_path(get_config_value(config, "paths.bids_fmri_root", None))
    bids_derivatives = _as_path(get_config_value(config, "paths.deriv_root", None))
    if bids_fmri_root is None:
        bids_fmri_root = _as_path(get_config_value(config, "paths.bids_root", None))

    ctx_subject = getattr(ctx, "subject", None) or subject.replace("sub-", "")
    eeg_task = str(get_config_value(config, "project.task", "")).strip()
    fmri_task = eeg_task

    fmri_cfg_raw = src_cfg.get("fmri", {}) if isinstance(src_cfg, dict) else {}
    if not isinstance(fmri_cfg_raw, dict):
        fmri_cfg_raw = {}
    _preflight_fmri_trans_requirements(
        config,
        mode=mode,
        fmri_cfg_raw=fmri_cfg_raw,
        subjects_dir=subjects_dir_path,
        subject=subject,
        trans_path=trans_path,
    )

    fmri_cfg = _load_fmri_constraint_config(
        config,
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        freesurfer_subjects_dir=subjects_dir_path,
        subject=ctx_subject,
        task=fmri_task,
    )

    allow_template_fallback = bool(
        src_cfg.get("allow_template_fallback", False)
    )
    if mode == "fmri_informed" and not bool(fmri_cfg.enabled):
        raise ValueError(
            "feature_engineering.sourcelocalization.mode='fmri_informed' requires "
            "feature_engineering.sourcelocalization.fmri.enabled=true."
        )
    if bool(fmri_cfg.enabled) and mode != "fmri_informed":
        logger = getattr(ctx, "logger", None) or logging.getLogger(__name__)
        logger.warning(
            "fMRI constraint is enabled but mode='%s'; auto-promoting to 'fmri_informed' "
            "so STC outputs are saved in the correct directory.",
            mode,
        )
        mode = "fmri_informed"
    if (
        bool(fmri_cfg.enabled)
        and method_use == "eloreta"
        and not np.isclose(eloreta_loose, 1.0, atol=1e-12)
    ):
        raise ValueError(
            "fMRI-constrained eLORETA requires feature_engineering.sourcelocalization.loose=1.0 "
            "(free orientation for volume/discrete source spaces)."
        )

    return SourceLocalizationConfig(
        mode=mode,
        allow_template_fallback=allow_template_fallback,
        method=method_use,
        spacing=spacing,
        parcellation=parcellation,
        subject=subject,
        subjects_dir=subjects_dir_path,
        trans_path=trans_path,
        bem_path=bem_path,
        mindist_mm=mindist_mm,
        lcmv_reg=lcmv_reg,
        eloreta_snr=eloreta_snr,
        eloreta_loose=eloreta_loose,
        eloreta_depth=eloreta_depth,
        fmri_cfg=fmri_cfg,
        save_stc=bool(src_cfg.get("save_stc", False)),
    )


def extract_source_contrast_features(
    ctx: Any,
    source_df: Optional[pd.DataFrame],
    source_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute condition contrasts from source-localized per-trial features.

    Returns a one-row, subject-level table where each column is a contrast statistic
    for one source feature. This avoids pseudo-replication at the trial level.
    """
    logger = getattr(ctx, "logger", None) or logging.getLogger(__name__)
    contrast_cfg = _load_source_contrast_config(getattr(ctx, "config", None))
    if not contrast_cfg.enabled:
        return pd.DataFrame(), []

    if source_df is None or source_df.empty:
        raise ValueError(
            "Source contrast requested, but source localization features are empty. "
            "Enable and validate source localization first."
        )
    if not isinstance(getattr(ctx, "aligned_events", None), pd.DataFrame):
        raise ValueError(
            "Source contrast requires trial-aligned events in ctx.aligned_events."
        )
    events_df = ctx.aligned_events.reset_index(drop=True)
    if events_df.empty:
        raise ValueError("Source contrast requires non-empty aligned events.")
    if len(events_df) != len(source_df):
        raise ValueError(
            "Source contrast requires events and source features to be row-aligned "
            f"(events={len(events_df)}, source={len(source_df)})."
        )

    source_features = source_df.reset_index(drop=True)
    if source_cols:
        candidate_cols = [str(col) for col in source_cols if str(col) in source_features.columns]
    else:
        candidate_cols = [str(col) for col in source_features.columns]
    if not candidate_cols:
        raise ValueError("Source contrast has no candidate source feature columns.")

    lookup = {str(col).strip().lower(): str(col) for col in events_df.columns}
    resolved_condition_col = lookup.get(contrast_cfg.condition_column.lower())
    if resolved_condition_col is None:
        raise ValueError(
            "Source contrast condition column was not found in aligned events: "
            f"{contrast_cfg.condition_column!r}. Available columns: {list(events_df.columns)}"
        )

    condition_series = events_df[resolved_condition_col]
    normalized_conditions = condition_series.astype(str).str.strip()
    normalized_conditions = normalized_conditions.mask(condition_series.isna(), "")

    mask_a = normalized_conditions == contrast_cfg.condition_a
    mask_b = normalized_conditions == contrast_cfg.condition_b
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())

    if n_a < contrast_cfg.min_trials_per_condition or n_b < contrast_cfg.min_trials_per_condition:
        raise ValueError(
            "Source contrast requires minimum trials per condition, but got "
            f"{contrast_cfg.condition_a}={n_a}, {contrast_cfg.condition_b}={n_b}, "
            f"required>={contrast_cfg.min_trials_per_condition}."
        )

    numeric_features = source_features.loc[:, candidate_cols].apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
    numeric_features = numeric_features.loc[:, numeric_features.notna().any(axis=0)]
    if numeric_features.empty:
        raise ValueError("Source contrast found no numeric source feature columns.")

    cond_a_token = _sanitize_feature_token(contrast_cfg.condition_a)
    cond_b_token = _sanitize_feature_token(contrast_cfg.condition_b)
    if cond_a_token == cond_b_token:
        raise ValueError(
            "Source contrast condition names collide after sanitization. "
            "Use condition values that map to distinct tokens."
        )

    row: Dict[str, float] = {
        f"sourcecontrast_n_trials_{cond_a_token}": int(n_a),
        f"sourcecontrast_n_trials_{cond_b_token}": int(n_b),
    }
    feature_token_to_original: Dict[str, str] = {}

    for feature_name in numeric_features.columns:
        feature_token = _sanitize_feature_token(feature_name)
        previous_name = feature_token_to_original.get(feature_token)
        if previous_name is not None and previous_name != str(feature_name):
            raise ValueError(
                "Source contrast feature naming collision after sanitization: "
                f"{previous_name!r} and {feature_name!r} both map to token {feature_token!r}."
            )
        feature_token_to_original[feature_token] = str(feature_name)
        values_a = numeric_features.loc[mask_a, feature_name].dropna().to_numpy(dtype=float)
        values_b = numeric_features.loc[mask_b, feature_name].dropna().to_numpy(dtype=float)

        mean_a = float(np.nanmean(values_a)) if values_a.size else float("nan")
        mean_b = float(np.nanmean(values_b)) if values_b.size else float("nan")
        delta = mean_a - mean_b if np.isfinite(mean_a) and np.isfinite(mean_b) else float("nan")
        cohen_d = _cohen_d_unpaired(values_a, values_b)

        prefix = f"sourcecontrast_{feature_token}"
        row[f"{prefix}_mean_{cond_a_token}"] = mean_a
        row[f"{prefix}_mean_{cond_b_token}"] = mean_b
        row[f"{prefix}_delta_{cond_a_token}_minus_{cond_b_token}"] = delta
        row[f"{prefix}_cohen_d_{cond_a_token}_vs_{cond_b_token}"] = cohen_d

    contrast_df = pd.DataFrame([row])
    contrast_df.attrs["feature_granularity"] = "subject"
    contrast_df.attrs["condition_column"] = resolved_condition_col
    contrast_df.attrs["condition_a"] = contrast_cfg.condition_a
    contrast_df.attrs["condition_b"] = contrast_cfg.condition_b
    contrast_df.attrs["min_trials_per_condition"] = int(contrast_cfg.min_trials_per_condition)
    contrast_df.attrs["source_contrast_enabled"] = True
    contrast_df.attrs["statistical_scope"] = "within_subject_trial_level_descriptive_only"

    logger.info(
        "Source contrast: %d columns computed for '%s' vs '%s' (%s=%d, %s=%d).",
        int(contrast_df.shape[1]),
        contrast_cfg.condition_a,
        contrast_cfg.condition_b,
        contrast_cfg.condition_a,
        n_a,
        contrast_cfg.condition_b,
        n_b,
    )
    return contrast_df, list(contrast_df.columns)


###################################################################
# Main Extraction Functions
###################################################################


def _normalize_train_mask(
    train_mask: Optional[np.ndarray],
    n_epochs: int,
) -> Optional[np.ndarray]:
    """Return a boolean train mask aligned to epochs, or None if unavailable/invalid."""
    if train_mask is None:
        return None
    mask = np.asarray(train_mask, dtype=bool).ravel()
    if mask.size != int(n_epochs):
        return None
    return mask


def _require_lcmv_train_mask_if_trial_safe(
    *,
    analysis_mode: str,
    method: str,
    train_mask: Optional[np.ndarray],
    n_epochs: int,
    context_name: str,
) -> Optional[np.ndarray]:
    """Enforce leak-safe train-mask requirements for LCMV in trial_ml_safe mode."""
    mask = _normalize_train_mask(train_mask, n_epochs)
    method_l = str(method or "").strip().lower()
    mode_l = str(analysis_mode or "").strip().lower()
    if mode_l != "trial_ml_safe" or method_l != "lcmv":
        return mask

    if mask is None:
        raise ValueError(
            f"{context_name}: trial_ml_safe + LCMV requires a valid train_mask aligned to epochs. "
            "Without train-only covariance fitting, cross-trial leakage can bias results."
        )
    n_train = int(np.sum(mask))
    if n_train < 2:
        raise ValueError(
            f"{context_name}: trial_ml_safe + LCMV requires at least 2 training epochs in train_mask "
            f"(got {n_train})."
        )
    return mask


def _resolve_source_connectivity_min_cycles(config: Any) -> float:
    """Resolve minimum cycle count for source connectivity validity checks."""
    min_cycles = get_config_value(config, "feature_engineering.sourcelocalization.min_cycles_per_band", None)
    if min_cycles is None:
        min_cycles = get_config_value(config, "feature_engineering.connectivity.min_cycles_per_band", 3.0)
    try:
        min_cycles_f = float(min_cycles)
    except (TypeError, ValueError):
        min_cycles_f = 3.0
    if not np.isfinite(min_cycles_f) or min_cycles_f <= 0:
        min_cycles_f = 3.0
    return max(1.0, float(min_cycles_f))


def _validate_source_connectivity_duration(
    *,
    n_times: int,
    sfreq: float,
    fmin: float,
    min_cycles: float,
    band: str,
    method: str,
    logger: logging.Logger,
) -> bool:
    """Check whether per-epoch duration supports stable band-limited connectivity."""
    if n_times <= 0 or not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(fmin) or fmin <= 0:
        return False
    duration_sec = float(n_times) / float(sfreq)
    min_duration_sec = float(min_cycles) / float(fmin)
    if duration_sec < min_duration_sec:
        logger.warning(
            "Source connectivity (%s): epoch duration %.3fs is shorter than recommended %.3fs "
            "for band '%s' (%d cycles at %.2f Hz); skipping band.",
            method,
            duration_sec,
            min_duration_sec,
            band,
            int(round(min_cycles)),
            fmin,
        )
        return False
    return True


def _validate_source_localization_duration(
    *,
    n_times: int,
    sfreq: float,
    fmin: float,
    min_cycles: float,
    band: str,
    logger: logging.Logger,
) -> bool:
    """Check whether per-epoch duration supports stable band-limited source power/envelope."""
    if n_times <= 0 or not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(fmin) or fmin <= 0:
        return False
    duration_sec = float(n_times) / float(sfreq)
    min_duration_sec = float(min_cycles) / float(fmin)
    if duration_sec < min_duration_sec:
        logger.warning(
            "Source localization: epoch duration %.3fs is shorter than recommended %.3fs "
            "for band '%s' (%d cycles at %.2f Hz); skipping band.",
            duration_sec,
            min_duration_sec,
            band,
            int(round(min_cycles)),
            fmin,
        )
        return False
    return True


def _enforce_fmri_provenance_policy(
    fmri_cfg: FMRIConstraintConfig,
    logger: logging.Logger,
    context_name: str,
) -> None:
    """Validate provenance constraints for fMRI-informed source features."""
    if not bool(fmri_cfg.enabled):
        return

    if bool(fmri_cfg.require_provenance) and str(fmri_cfg.provenance) == "unknown":
        raise ValueError(
            f"{context_name}: fMRI constraint enabled but fmri.provenance is unknown. "
            "Set feature_engineering.sourcelocalization.fmri.provenance to "
            "'independent' (recommended) or 'same_dataset' with explicit override."
        )

    if str(fmri_cfg.provenance) != "same_dataset":
        return

    allow_same_dataset = bool(getattr(fmri_cfg, "allow_same_dataset_provenance", False))
    if not allow_same_dataset:
        raise ValueError(
            f"{context_name}: fmri.provenance='same_dataset' is blocked by default due circularity risk. "
            "To proceed intentionally, set "
            "feature_engineering.sourcelocalization.fmri.allow_same_dataset_provenance=true."
        )

    logger.warning(
        "%s: using fmri.provenance='same_dataset' with explicit override. "
        "Interpret downstream EEG-vs-condition analyses cautiously due double-dipping risk.",
        context_name,
    )


def _sanitize_segment_token(value: str) -> str:
    token = _sanitize_feature_token(value)
    return token.lower()


def _write_fmri_constraint_metadata_sidecar(
    *,
    ctx: Any,
    src_cfg: SourceLocalizationConfig,
    segment_label: str,
    payload: Dict[str, Any],
    logger: logging.Logger,
) -> Optional[Path]:
    """Write per-subject fMRI-constraint metadata sidecar for source features."""
    deriv_root = getattr(ctx, "deriv_root", None)
    subject = getattr(ctx, "subject", None)
    if deriv_root is None or subject is None:
        return None

    base_features_dir = deriv_features_path(Path(str(deriv_root)), str(subject))
    metadata_dir = base_features_dir / "sourcelocalization" / str(src_cfg.method) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    safe_segment = _sanitize_segment_token(str(segment_label or "full"))
    out_path = metadata_dir / f"fmri_constraint_{safe_segment}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved fMRI constraint metadata sidecar: %s", str(out_path))
    return out_path


def _normalize_fmri_cluster_name(roi_name: str) -> str:
    token = _sanitize_feature_token(str(roi_name)).lower()
    if token.startswith("fmri_"):
        token = token[len("fmri_") :]
    return token


def _require_finite_source_feature_matrix(
    values: np.ndarray,
    *,
    description: str,
    band: str,
    segment_label: str,
    method: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(
            f"Invalid source-localization {description} for band '{band}' "
            f"({segment_label}, {method}): expected a non-empty 2D matrix, got shape {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Found non-finite source-localization {description} for band '{band}' "
            f"({segment_label}, {method})."
        )
    return matrix


def _append_source_band_family_features(
    *,
    records: List[Dict[str, float]],
    feature_cols: List[str],
    n_epochs: int,
    roi_data: np.ndarray,
    label_names: List[str],
    sfreq: float,
    fmin: float,
    fmax: float,
    segment_label: str,
    method: str,
    band: str,
    family_prefix: str,
) -> None:
    power = _require_finite_source_feature_matrix(
        _compute_roi_power(roi_data, sfreq, fmin, fmax),
        description="power features",
        band=band,
        segment_label=segment_label,
        method=method,
    )
    for roi_idx, roi_name in enumerate(label_names):
        safe_name = _sanitize_feature_token(roi_name).lower()
        col_name = f"src_{segment_label}_{method}_{band}_{family_prefix}_{safe_name}_power"
        feature_cols.append(col_name)
        for epoch_idx in range(n_epochs):
            records[epoch_idx][col_name] = power[epoch_idx, roi_idx]

    global_power = np.nanmean(power, axis=1)
    col_name = f"src_{segment_label}_{method}_{band}_{family_prefix}_global_power"
    feature_cols.append(col_name)
    for epoch_idx in range(n_epochs):
        records[epoch_idx][col_name] = global_power[epoch_idx]

    envelope = _compute_roi_envelope(roi_data, sfreq, fmin, fmax)
    mean_env = _require_finite_source_feature_matrix(
        np.nanmean(envelope, axis=2),
        description="envelope features",
        band=band,
        segment_label=segment_label,
        method=method,
    )
    for roi_idx, roi_name in enumerate(label_names):
        safe_name = _sanitize_feature_token(roi_name).lower()
        col_name = f"src_{segment_label}_{method}_{band}_{family_prefix}_{safe_name}_envelope"
        feature_cols.append(col_name)
        for epoch_idx in range(n_epochs):
            records[epoch_idx][col_name] = mean_env[epoch_idx, roi_idx]


def extract_source_localization_features(
    ctx: Any,
    bands: List[str],
    method: str = "lcmv",
    roi_labels: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract source-localized ROI features from epochs.
    
    Parameters
    ----------
    ctx : FeatureContext
        Feature extraction context with epochs and config
    bands : list of str
        Frequency bands to analyze
    method : str
        Source localization method: 'lcmv' or 'eloreta'
    roi_labels : list of str, optional
        ROI label names to extract (uses aparc if None)
    n_jobs : int
        Number of parallel jobs (unused, kept for API compatibility)
        
    Returns
    -------
    features_df : pd.DataFrame
        Source-localized features per epoch
    feature_cols : list of str
        Feature column names
    """
    import mne
    
    epochs = ctx.epochs
    config = ctx.config
    logger = getattr(ctx, "logger", None) or logging.getLogger(__name__)
    analysis_mode = str(getattr(ctx, "analysis_mode", "") or "").strip().lower()
    train_mask_raw = getattr(ctx, "train_mask", None)

    src_cfg = _load_source_localization_config(ctx, config, method)
    fmri_cfg = src_cfg.fmri_cfg

    _enforce_fmri_provenance_policy(fmri_cfg, logger, "Source localization")

    n_epochs = len(epochs)
    if n_epochs < 2:
        logger.warning("Source localization requires at least 2 epochs")
        return pd.DataFrame(), []

    train_mask = _require_lcmv_train_mask_if_trial_safe(
        analysis_mode=analysis_mode,
        method=src_cfg.method,
        train_mask=train_mask_raw,
        n_epochs=n_epochs,
        context_name="Source localization",
    )
    epochs_fit = epochs[train_mask] if train_mask is not None and src_cfg.method == "lcmv" else None

    sfreq = epochs.info["sfreq"]
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)
    min_cycles_per_band = _resolve_source_connectivity_min_cycles(config)

    if logger:
        logger.info(f"Extracting source-localized features using {src_cfg.method.upper()}")

    fmri_family_series: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    fmri_metadata_payload: Optional[Dict[str, Any]] = None

    if fmri_cfg.enabled:
        if src_cfg.subjects_dir is None:
            raise ValueError(
                "fMRI-constrained source localization requires feature_engineering.sourcelocalization.subjects_dir "
                "(FreeSurfer SUBJECTS_DIR). Set via --source-subjects-dir or in config."
            )

        from fmri_pipeline.analysis.bem_generation import ensure_bem_and_trans_files
        trans_path, _, bem_path = ensure_bem_and_trans_files(
            subject=src_cfg.subject,
            subjects_dir=src_cfg.subjects_dir,
            config=config,
            logger_instance=logger,
            eeg_info=epochs.info,
        )

        if fmri_cfg.stats_map_path is None:
            raise ValueError(
                "fMRI constraint enabled but no stats map path provided. "
                "Set feature_engineering.sourcelocalization.fmri.stats_map_path or enable contrast builder."
            )
        if trans_path is None:
            raise ValueError(
                "fMRI-constrained source localization requires feature_engineering.sourcelocalization.trans "
                "(EEG↔MRI transform .fif). Set path or enable --source-create-trans to auto-generate."
            )
        if bem_path is None:
            raise ValueError(
                "fMRI-constrained source localization requires feature_engineering.sourcelocalization.bem "
                "(*-bem-sol.fif). Set path or enable --source-create-bem-model and --source-create-bem-solution to auto-generate."
            )
        if not fmri_cfg.stats_map_path.exists():
            raise FileNotFoundError(f"Missing fMRI stats map: {fmri_cfg.stats_map_path}")

        include_atlas = str(fmri_cfg.output_space) in {"atlas", "dual"}
        voxel_selection = _select_fmri_constrained_voxels(
            stats_map_path=fmri_cfg.stats_map_path,
            cfg=fmri_cfg,
            logger=logger,
            subjects_dir=Path(src_cfg.subjects_dir),
            subject=src_cfg.subject,
            include_atlas=include_atlas,
        )
        fwd, src = _setup_volume_source_space_from_points_configured(
            epochs.info,
            subject=src_cfg.subject,
            subjects_dir=str(src_cfg.subjects_dir),
            trans=str(trans_path),
            bem=str(bem_path),
            coords_m=voxel_selection.selected_coords_m,
            mindist_mm=src_cfg.mindist_mm,
            logger=logger,
        )
        if src_cfg.method == "lcmv":
            stcs, _ = _compute_lcmv_source_estimates(
                epochs,
                fwd,
                epochs_fit=epochs_fit,
                reg=src_cfg.lcmv_reg,
                logger=logger,
            )
        else:
            stcs, _ = _compute_eloreta_source_estimates(
                epochs,
                fwd,
                loose=src_cfg.eloreta_loose,
                depth=src_cfg.eloreta_depth,
                snr=src_cfg.eloreta_snr,
                pick_ori=None,
                logger=logger,
            )

        stc_verts = np.asarray(stcs[0].vertices[0], dtype=int)
        requested_roi_sets: Dict[str, Dict[str, List[int]]] = {}
        if str(fmri_cfg.output_space) in {"cluster", "dual"}:
            requested_roi_sets["fmri_cluster"] = {
                _normalize_fmri_cluster_name(name): list(indices)
                for name, indices in voxel_selection.cluster_indices.items()
            }
        if str(fmri_cfg.output_space) in {"atlas", "dual"}:
            if not voxel_selection.atlas_indices:
                raise ValueError(
                    "fMRI output_space includes atlas mapping but no non-background aparc+aseg "
                    "labels survived in the selected voxels."
                )
            requested_roi_sets["atlas"] = {
                str(name): list(indices)
                for name, indices in voxel_selection.atlas_indices.items()
            }
        if not requested_roi_sets:
            raise ValueError(
                f"Unsupported fMRI output_space '{fmri_cfg.output_space}'. "
                "Expected one of {'cluster','atlas','dual'}."
            )

        roi_survival_metadata: Dict[str, Dict[str, Any]] = {}
        for family_name, family_indices in requested_roi_sets.items():
            roi_row_indices, surviving_roi_names, dropped_rois = _resolve_surviving_roi_rows(
                stc_vertices=stc_verts,
                roi_indices=family_indices,
            )
            if dropped_rois:
                logger.warning(
                    "fMRI-constrained %s mapping dropped %d ROI(s) with no surviving vertices: %s",
                    family_name,
                    len(dropped_rois),
                    ", ".join(dropped_rois),
                )
            if not surviving_roi_names:
                raise ValueError(
                    f"fMRI-constrained {family_name} ROI mapping failed: no surviving vertices "
                    "remained after forward/inverse modeling."
                )
            roi_data_family = _compute_roi_timecourses_from_row_indices(
                stcs=stcs,
                roi_row_indices=roi_row_indices,
                roi_names=surviving_roi_names,
            )
            if roi_data_family.size == 0:
                raise ValueError(
                    f"fMRI-constrained {family_name} ROI mapping produced empty time-courses."
                )
            fmri_family_series[family_name] = (roi_data_family, surviving_roi_names)
            roi_survival_metadata[family_name] = {
                "requested_rois": sorted(family_indices.keys()),
                "surviving_rois": list(surviving_roi_names),
                "dropped_rois": list(dropped_rois),
                "surviving_vertices_per_roi": {
                    roi_name: int(len(roi_row_indices[roi_name]))
                    for roi_name in surviving_roi_names
                },
                "surviving_stc_rows_per_roi": {
                    roi_name: [int(idx) for idx in roi_row_indices[roi_name]]
                    for roi_name in surviving_roi_names
                },
            }

        if not fmri_family_series:
            raise ValueError(
                "fMRI-constrained source localization produced no surviving ROI families."
            )

        fmri_metadata_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "subject": str(src_cfg.subject),
            "method": str(src_cfg.method),
            "output_space": str(fmri_cfg.output_space),
            "stats_map_path": str(fmri_cfg.stats_map_path),
            "matched_reference_mri": str(voxel_selection.matched_reference_path),
            "selected_voxels_total": int(voxel_selection.selected_voxels_ijk.shape[0]),
            "voxel_volume_mm3": (
                float(voxel_selection.voxel_volume_mm3)
                if voxel_selection.voxel_volume_mm3 is not None
                else None
            ),
            "cluster_voxel_counts": dict(voxel_selection.cluster_voxel_counts),
            "atlas_voxel_counts": dict(voxel_selection.atlas_voxel_counts),
            "cluster_to_atlas_counts": {
                cluster_name: dict(counts)
                for cluster_name, counts in voxel_selection.cluster_to_atlas_counts.items()
            },
            "dropped_unlabeled_voxels": int(voxel_selection.dropped_unlabeled_voxels),
            "roi_survival": roi_survival_metadata,
        }
    else:
        trans_path = src_cfg.trans_path
        bem_path = src_cfg.bem_path

        # Auto-generate or discover BEM/trans if subjects_dir is set but paths aren't.
        # This makes --source-create-trans/bem work in EEG-only mode (no fMRI constraint needed).
        if src_cfg.subjects_dir is not None and (trans_path is None or bem_path is None):
            from fmri_pipeline.analysis.bem_generation import ensure_bem_and_trans_files
            resolved_trans, _, resolved_bem = ensure_bem_and_trans_files(
                subject=src_cfg.subject,
                subjects_dir=src_cfg.subjects_dir,
                config=config,
                logger_instance=logger,
                eeg_info=epochs.info,
            )
            trans_path = trans_path or resolved_trans
            bem_path = bem_path or resolved_bem

        if src_cfg.subjects_dir is not None and trans_path is not None and bem_path is not None:
            fwd, src = _setup_surface_forward_model_configured(
                epochs.info,
                subject=src_cfg.subject,
                subjects_dir=str(src_cfg.subjects_dir),
                spacing=src_cfg.spacing,
                trans=str(trans_path),
                bem=str(bem_path),
                mindist_mm=src_cfg.mindist_mm,
                logger=logger,
            )
            labels = mne.read_labels_from_annot(
                src_cfg.subject,
                parc=src_cfg.parcellation,
                subjects_dir=str(src_cfg.subjects_dir),
                verbose=False,
            )
        else:
            if not bool(getattr(src_cfg, "allow_template_fallback", False)):
                raise ValueError(
                    "Source localization template fallback is disabled "
                    "(feature_engineering.sourcelocalization.allow_template_fallback=false). "
                    "Provide subject-specific sources via sourcelocalization.subjects_dir, subject, trans, and bem."
                )
            fwd, src, _ = _setup_forward_model(epochs.info, logger=logger)
            logger.warning(
                "Source localization: using fsaverage/template forward model fallback. "
                "Interpret ROI anatomy cautiously; subject-specific MRI/trans/BEM is recommended."
            )
            labels = mne.read_labels_from_annot(
                "fsaverage",
                parc="aparc",
                subjects_dir=None,
                verbose=False,
            )

        labels = [l for l in labels if "unknown" not in l.name.lower()]
        if roi_labels is not None:
            labels = [l for l in labels if any(r in l.name for r in roi_labels)]
        if not labels:
            logger.warning("No ROI labels found for source localization")
            return pd.DataFrame(), []
        label_names = [l.name for l in labels]
        if logger:
            logger.info(f"Using {len(labels)} ROI labels")

        if src_cfg.method == "lcmv":
            stcs, _ = _compute_lcmv_source_estimates(
                epochs,
                fwd,
                epochs_fit=epochs_fit,
                reg=src_cfg.lcmv_reg,
                logger=logger,
            )
        else:
            stcs, _ = _compute_eloreta_source_estimates(
                epochs,
                fwd,
                loose=src_cfg.eloreta_loose,
                depth=src_cfg.eloreta_depth,
                snr=src_cfg.eloreta_snr,
                pick_ori="normal",
                logger=logger,
            )

        roi_data = _extract_roi_timecourses(stcs, labels, src, mode="mean_flip")
    
    segment_label, segment_mask = _resolve_source_segment(
        times=np.asarray(getattr(epochs, "times", None), dtype=float)
        if getattr(epochs, "times", None) is not None
        else None,
        windows=getattr(ctx, "windows", None),
        target_name=getattr(ctx, "name", None),
        config=config,
        logger=logger,
        feature_name="Source localization",
    )
    if segment_mask is not None and not np.any(segment_mask):
        return pd.DataFrame(), []

    if fmri_cfg.enabled:
        masked_family_series: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        for family_name, (family_roi_data, family_label_names) in fmri_family_series.items():
            family_roi_array = np.asarray(family_roi_data, dtype=float)
            if segment_mask is not None:
                if family_roi_array.shape[-1] != int(segment_mask.shape[0]):
                    raise ValueError(
                        "Source localization segment mask length does not match extracted ROI time-courses "
                        f"(mask={int(segment_mask.shape[0])}, roi_times={int(family_roi_array.shape[-1])})."
                    )
                family_roi_array = family_roi_array[..., segment_mask]
            masked_family_series[family_name] = (family_roi_array, family_label_names)
        fmri_family_series = masked_family_series
    elif segment_mask is not None:
        if roi_data.shape[-1] != int(segment_mask.shape[0]):
            raise ValueError(
                "Source localization segment mask length does not match extracted ROI time-courses "
                f"(mask={int(segment_mask.shape[0])}, roi_times={int(roi_data.shape[-1])})."
            )
        roi_data = roi_data[..., segment_mask]

    if fmri_metadata_payload is not None:
        sidecar_path = _write_fmri_constraint_metadata_sidecar(
            ctx=ctx,
            src_cfg=src_cfg,
            segment_label=str(segment_label),
            payload=fmri_metadata_payload,
            logger=logger,
        )
        if sidecar_path is not None:
            fmri_metadata_payload["metadata_sidecar_path"] = str(sidecar_path)

    records = [{} for _ in range(n_epochs)]
    feature_cols: List[str] = []

    if fmri_cfg.enabled:
        for band in bands:
            if band not in freq_bands:
                continue
            fmin, fmax = freq_bands[band]
            for family_name, (family_roi_data, family_label_names) in fmri_family_series.items():
                if not _validate_source_localization_duration(
                    n_times=int(family_roi_data.shape[-1]) if np.ndim(family_roi_data) >= 3 else 0,
                    sfreq=float(sfreq),
                    fmin=float(fmin),
                    min_cycles=min_cycles_per_band,
                    band=str(band),
                    logger=logger,
                ):
                    continue
                _append_source_band_family_features(
                    records=records,
                    feature_cols=feature_cols,
                    n_epochs=n_epochs,
                    roi_data=family_roi_data,
                    label_names=family_label_names,
                    sfreq=sfreq,
                    fmin=float(fmin),
                    fmax=float(fmax),
                    segment_label=str(segment_label),
                    method=str(src_cfg.method),
                    band=str(band),
                    family_prefix=str(family_name),
                )
    else:
        for band in bands:
            if band not in freq_bands:
                continue

            fmin, fmax = freq_bands[band]
            if not _validate_source_localization_duration(
                n_times=int(roi_data.shape[-1]) if np.ndim(roi_data) >= 3 else 0,
                sfreq=float(sfreq),
                fmin=float(fmin),
                min_cycles=min_cycles_per_band,
                band=str(band),
                logger=logger,
            ):
                continue

            power = _require_finite_source_feature_matrix(
                _compute_roi_power(roi_data, sfreq, fmin, fmax),
                description="power features",
                band=band,
                segment_label=str(segment_label),
                method=str(src_cfg.method),
            )

            for roi_idx, roi_name in enumerate(label_names):
                safe_name = roi_name.replace("-", "_").replace(" ", "_")
                col_name = f"src_{segment_label}_{src_cfg.method}_{band}_{safe_name}_power"
                feature_cols.append(col_name)
                for epoch_idx in range(n_epochs):
                    records[epoch_idx][col_name] = power[epoch_idx, roi_idx]

            global_power = np.nanmean(power, axis=1)
            col_name = f"src_{segment_label}_{src_cfg.method}_{band}_global_power"
            feature_cols.append(col_name)
            for epoch_idx in range(n_epochs):
                records[epoch_idx][col_name] = global_power[epoch_idx]

            envelope = _compute_roi_envelope(roi_data, sfreq, fmin, fmax)
            mean_env = _require_finite_source_feature_matrix(
                np.nanmean(envelope, axis=2),
                description="envelope features",
                band=band,
                segment_label=str(segment_label),
                method=str(src_cfg.method),
            )

            for roi_idx, roi_name in enumerate(label_names):
                safe_name = roi_name.replace("-", "_").replace(" ", "_")
                col_name = f"src_{segment_label}_{src_cfg.method}_{band}_{safe_name}_envelope"
                feature_cols.append(col_name)
                for epoch_idx in range(n_epochs):
                    records[epoch_idx][col_name] = mean_env[epoch_idx, roi_idx]
    
    if src_cfg.save_stc and hasattr(ctx, "aligned_events") and ctx.aligned_events is not None:
        from eeg_pipeline.utils.data.columns import get_condition_column_from_config

        contrast_cfg = _load_source_contrast_config(getattr(ctx, "config", None))
        cond_col = (
            contrast_cfg.condition_column
            if contrast_cfg and getattr(contrast_cfg, "condition_column", None)
            else None
        )
        if not cond_col:
            cond_col = get_condition_column_from_config(ctx.config, ctx.aligned_events)
        if not cond_col or cond_col not in ctx.aligned_events.columns:
            raise ValueError(
                "Source STC export requires an explicit condition column. "
                "Set feature_engineering.sourcelocalization.contrast.condition_column "
                "or configure event_columns.condition to match the aligned events table. "
                f"Available columns: {list(ctx.aligned_events.columns)}"
            )

        if logger:
            logger.info(
                "Computing and saving condition-averaged band power STCs based on column '%s'...",
                cond_col,
            )

        out_dir = source_localization_estimates_dir(
            features_dir=deriv_features_path(Path(ctx.deriv_root), str(ctx.subject)),
            method=str(src_cfg.method),
            mode=str(src_cfg.mode),
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        src_path = out_dir / f"sub-{ctx.subject}_task-{ctx.task}_{src_cfg.method}-src.fif"
        mne.write_source_spaces(str(src_path), src, overwrite=True)
        if logger:
            logger.info("Saved source space for STC plotting: %s", str(src_path))

        segment_times = np.asarray(epochs.times, dtype=float)
        if segment_mask is not None:
            segment_times = segment_times[np.asarray(segment_mask, dtype=bool)]

        conditions_list = ctx.aligned_events[cond_col].unique()
        for cond in conditions_list:
            if pd.isna(cond):
                continue

            cond_mask = (ctx.aligned_events[cond_col] == cond).values
            cond_indices = np.where(cond_mask)[0]
            if len(cond_indices) == 0:
                continue

            cond_stc_data = np.stack([stcs[idx].data for idx in cond_indices], axis=0)
            if segment_mask is not None:
                cond_stc_data = cond_stc_data[..., np.asarray(segment_mask, dtype=bool)]

            for band in bands:
                if band not in freq_bands:
                    continue
                fmin_b, fmax_b = freq_bands[band]
                if not _validate_source_localization_duration(
                    n_times=int(cond_stc_data.shape[-1]) if np.ndim(cond_stc_data) >= 3 else 0,
                    sfreq=float(sfreq),
                    fmin=float(fmin_b),
                    min_cycles=min_cycles_per_band,
                    band=str(band),
                    logger=logger,
                ):
                    continue

                # Compute power per epoch per vertex
                power_stc = _compute_roi_power(cond_stc_data, sfreq, fmin_b, fmax_b)
                mean_power = np.nanmean(power_stc, axis=0)

                out_stc = stcs[0].copy()
                out_stc.data = mean_power[:, np.newaxis]
                out_stc.tmin = 0.0
                out_stc.tstep = 1.0

                # MNE automatically appends appropriate extension like -vl.stc or -lh.stc
                safe_cond = str(cond).replace(" ", "_").replace("/", "_")
                safe_segment = _sanitize_feature_token(segment_label)
                stc_name = (
                    out_dir
                    / f"sub-{ctx.subject}_task-{ctx.task}_seg-{safe_segment}_cond-{safe_cond}_band-{band}_{src_cfg.method}"
                )
                out_stc.save(str(stc_name), overwrite=True)

                if fmri_cfg.enabled and fmri_family_series and "fmri_cluster" in fmri_family_series:
                    cluster_roi_data, cluster_roi_names = fmri_family_series["fmri_cluster"]
                    cond_cluster_data = cluster_roi_data[cond_indices]
                    
                    if len(cond_indices) > 0 and cond_cluster_data.shape[-1] > 100:
                        from mne.time_frequency import AverageTFR, tfr_array_morlet
                        
                        freqs = np.logspace(*np.log10([3.0, 100.0]), num=40)
                        n_cycles = freqs / 3.0
                        
                        try:
                            if logger:
                                logger.info(f"Computing source-level Morlet TFR for {len(cluster_roi_names)} clusters for condition '{cond}'...")
                            power_tfr = tfr_array_morlet(
                                cond_cluster_data,
                                sfreq=sfreq,
                                freqs=freqs,
                                n_cycles=n_cycles,
                                output='power',
                                n_jobs=getattr(ctx, "n_jobs", 1),
                                verbose=False
                            )
                            mean_power_tfr = np.nanmean(power_tfr, axis=0) # shape (n_rois, n_freqs, n_times)
                            
                            info = mne.create_info(
                                ch_names=list(cluster_roi_names),
                                sfreq=sfreq,
                                ch_types='eeg'
                            )
                            tfr_obj = AverageTFR(
                                info=info,
                                data=mean_power_tfr,
                                times=segment_times,
                                freqs=freqs,
                                nave=len(cond_indices),
                                comment=f"Source cluster TFR: {cond}"
                            )
                            
                            safe_cond = str(cond).replace(" ", "_").replace("/", "_")
                            safe_segment = _sanitize_feature_token(segment_label)
                            tfr_path = out_dir / f"sub-{ctx.subject}_task-{ctx.task}_seg-{safe_segment}_cond-{safe_cond}_{src_cfg.method}-tfr.h5"
                            tfr_obj.save(str(tfr_path), overwrite=True)
                            if logger:
                                logger.info(f"Saved cluster Source TFR to {tfr_path.name}")
                        except Exception as e:
                            if logger:
                                logger.warning(f"Failed to compute/save Source TFR for condition '{cond}': {e}")

    features_df = pd.DataFrame(records)
    feature_cols = list(features_df.columns)
    features_df.attrs["method"] = str(src_cfg.method)
    features_df.attrs["fmri_constraint_enabled"] = bool(fmri_cfg.enabled)
    features_df.attrs["fmri_provenance"] = str(fmri_cfg.provenance)
    features_df.attrs["fmri_output_space"] = (
        str(fmri_cfg.output_space) if bool(fmri_cfg.enabled) else "none"
    )
    if fmri_metadata_payload is not None:
        metadata_path = fmri_metadata_payload.get("metadata_sidecar_path")
        if metadata_path:
            features_df.attrs["fmri_constraint_metadata_sidecar"] = str(metadata_path)
    features_df.attrs["train_mask_used_for_covariance"] = bool(epochs_fit is not None and src_cfg.method == "lcmv")
    
    if logger:
        logger.info(f"Source localization: {len(feature_cols)} features extracted")
    
    return features_df, feature_cols


def extract_source_connectivity_features(
    ctx: Any,
    bands: List[str],
    method: str = "lcmv",
    connectivity_method: str = "aec",
    roi_labels: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract source-space connectivity features.
    
    Parameters
    ----------
    ctx : FeatureContext
        Feature extraction context
    bands : list of str
        Frequency bands
    method : str
        Source localization method: 'lcmv' or 'eloreta'
    connectivity_method : str
        Connectivity measure: 'aec', 'wpli', 'plv'
    roi_labels : list of str, optional
        ROI labels to use
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    features_df : pd.DataFrame
        Connectivity features per epoch
    feature_cols : list of str
        Feature column names
    """
    import mne
    from mne_connectivity import spectral_connectivity_epochs, envelope_correlation
    
    epochs = ctx.epochs
    config = getattr(ctx, "config", None)
    logger = getattr(ctx, "logger", None) or logging.getLogger(__name__)
    analysis_mode = str(getattr(ctx, "analysis_mode", "") or "").strip().lower()
    train_mask_raw = getattr(ctx, "train_mask", None)

    src_cfg = _load_source_localization_config(ctx, config, method)
    fmri_cfg = src_cfg.fmri_cfg

    _enforce_fmri_provenance_policy(fmri_cfg, logger, "Source connectivity")

    n_epochs = len(epochs)
    if n_epochs < 2:
        logger.warning("Source connectivity requires at least 2 epochs")
        return pd.DataFrame(), []

    train_mask = _require_lcmv_train_mask_if_trial_safe(
        analysis_mode=analysis_mode,
        method=src_cfg.method,
        train_mask=train_mask_raw,
        n_epochs=n_epochs,
        context_name="Source connectivity",
    )
    
    sfreq = epochs.info["sfreq"]
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)
    min_cycles_per_band = _resolve_source_connectivity_min_cycles(config)
    connectivity_method_l = str(connectivity_method).strip().lower()
    if connectivity_method_l not in {"aec", "wpli", "plv"}:
        raise ValueError(
            "source connectivity method must be one of {'aec','wpli','plv'} "
            f"(got '{connectivity_method}')."
        )
    
    if logger:
        logger.info(f"Extracting source-space {connectivity_method_l.upper()} connectivity")

    if fmri_cfg.enabled:
        if fmri_cfg.stats_map_path is None:
            raise ValueError(
                "fMRI constraint enabled but no stats map path provided. "
                "Set feature_engineering.sourcelocalization.fmri.stats_map_path or enable contrast builder."
            )
        if src_cfg.subjects_dir is None:
            raise ValueError(
                "fMRI-constrained source connectivity requires feature_engineering.sourcelocalization.subjects_dir "
                "(FreeSurfer SUBJECTS_DIR)."
            )

        from fmri_pipeline.analysis.bem_generation import ensure_bem_and_trans_files

        resolved_trans, _, resolved_bem = ensure_bem_and_trans_files(
            subject=src_cfg.subject,
            subjects_dir=src_cfg.subjects_dir,
            config=config,
            logger_instance=logger,
            eeg_info=epochs.info,
        )
        if resolved_trans is None:
            raise ValueError(
                "fMRI-constrained source connectivity requires feature_engineering.sourcelocalization.trans "
                "(EEG↔MRI transform .fif). Set path or enable --source-create-trans to auto-generate."
            )
        if resolved_bem is None:
            raise ValueError(
                "fMRI-constrained source connectivity requires feature_engineering.sourcelocalization.bem "
                "(*-bem-sol.fif). Set path or enable --source-create-bem-model and --source-create-bem-solution to auto-generate."
            )
        if not fmri_cfg.stats_map_path.exists():
            raise FileNotFoundError(f"Missing fMRI stats map: {fmri_cfg.stats_map_path}")

        roi_coords_m = _fmri_roi_coords_from_stats_map(
            fmri_cfg.stats_map_path,
            fmri_cfg,
            logger,
            subjects_dir=Path(src_cfg.subjects_dir),
            subject=src_cfg.subject,
        )
        fwd, src, roi_indices = _setup_volume_source_space_configured(
            epochs.info,
            subject=src_cfg.subject,
            subjects_dir=str(src_cfg.subjects_dir),
            trans=str(resolved_trans),
            bem=str(resolved_bem),
            roi_coords_m=roi_coords_m,
            mindist_mm=src_cfg.mindist_mm,
            logger=logger,
        )
        labels = None
        label_names = list(roi_indices.keys())
    else:
        trans_path = src_cfg.trans_path
        bem_path = src_cfg.bem_path

        # Auto-generate or discover BEM/trans if subjects_dir is set but paths aren't.
        if src_cfg.subjects_dir is not None and (trans_path is None or bem_path is None):
            from fmri_pipeline.analysis.bem_generation import ensure_bem_and_trans_files
            resolved_trans, _, resolved_bem = ensure_bem_and_trans_files(
                subject=src_cfg.subject,
                subjects_dir=src_cfg.subjects_dir,
                config=config,
                logger_instance=logger,
                eeg_info=epochs.info,
            )
            trans_path = trans_path or resolved_trans
            bem_path = bem_path or resolved_bem

        if src_cfg.subjects_dir is not None and trans_path is not None and bem_path is not None:
            fwd, src = _setup_surface_forward_model_configured(
                epochs.info,
                subject=src_cfg.subject,
                subjects_dir=str(src_cfg.subjects_dir),
                spacing=src_cfg.spacing,
                trans=str(trans_path),
                bem=str(bem_path),
                mindist_mm=src_cfg.mindist_mm,
                logger=logger,
            )
            labels = mne.read_labels_from_annot(
                src_cfg.subject,
                parc=src_cfg.parcellation,
                subjects_dir=str(src_cfg.subjects_dir),
                verbose=False,
            )
        else:
            if not bool(getattr(src_cfg, "allow_template_fallback", False)):
                raise ValueError(
                    "Source connectivity template fallback is disabled "
                    "(feature_engineering.sourcelocalization.allow_template_fallback=false). "
                    "Provide subject-specific sources via sourcelocalization.subjects_dir, subject, trans, and bem."
                )
            fwd, src, _ = _setup_forward_model(epochs.info, logger=logger)
            logger.warning(
                "Source connectivity: using fsaverage/template forward model fallback. "
                "Interpret ROI anatomy cautiously; subject-specific MRI/trans/BEM is recommended."
            )
            labels = mne.read_labels_from_annot(
                "fsaverage",
                parc="aparc",
                subjects_dir=None,
                verbose=False,
            )

        labels = [l for l in labels if "unknown" not in l.name.lower()]
        if roi_labels is not None:
            labels = [l for l in labels if any(r in l.name for r in roi_labels)]
        if len(labels) < 2:
            logger.warning("Need at least 2 ROIs for connectivity")
            return pd.DataFrame(), []
        label_names = [l.name for l in labels]


    n_rois = len(label_names)
    if not fmri_cfg.enabled and n_rois < 2:
        logger.warning("Need at least 2 ROIs for connectivity")
        return pd.DataFrame(), []

    segment_label, segment_mask = _resolve_source_segment(
        times=np.asarray(getattr(epochs, "times", None), dtype=float)
        if getattr(epochs, "times", None) is not None
        else None,
        windows=getattr(ctx, "windows", None),
        target_name=getattr(ctx, "name", None),
        config=config,
        logger=logger,
        feature_name="Source connectivity",
    )
    if segment_mask is not None and not np.any(segment_mask):
        return pd.DataFrame(), []
    
    records = [{} for _ in range(n_epochs)]
    feature_cols = []
    
    for band in bands:
        if band not in freq_bands:
            continue
        
        fmin, fmax = freq_bands[band]
        if connectivity_method_l == "aec":
            epochs_for_source = epochs.copy().filter(fmin, fmax, n_jobs=n_jobs, verbose=False)
        else:
            epochs_for_source = epochs.copy()
        
        if src_cfg.method == "lcmv":
            epochs_fit = None
            if analysis_mode == "trial_ml_safe" and train_mask is not None and np.any(train_mask):
                epochs_fit = epochs_for_source[train_mask]
            stcs, _ = _compute_lcmv_source_estimates(
                epochs_for_source,
                fwd,
                epochs_fit=epochs_fit,
                reg=src_cfg.lcmv_reg,
                logger=None,
            )
        else:
            stcs, _ = _compute_eloreta_source_estimates(
                epochs_for_source,
                fwd,
                loose=src_cfg.eloreta_loose,
                depth=src_cfg.eloreta_depth,
                snr=src_cfg.eloreta_snr,
                pick_ori=None if fmri_cfg.enabled else "normal",
                logger=None,
            )

        if fmri_cfg.enabled:
            roi_data, _mapped_label_names = _extract_roi_timecourses_from_vertex_indices(
                stcs,
                roi_indices,
                logger=logger,
            )
        else:
            roi_data = _extract_roi_timecourses(stcs, labels, src, mode="mean_flip")

        if segment_mask is not None:
            if roi_data.shape[-1] != int(segment_mask.shape[0]):
                raise ValueError(
                    "Source connectivity segment mask length does not match extracted ROI time-courses "
                    f"(mask={int(segment_mask.shape[0])}, roi_times={int(roi_data.shape[-1])})."
                )
            roi_data = roi_data[..., segment_mask]

        n_rois_band = int(roi_data.shape[1]) if np.ndim(roi_data) >= 2 else 0
        if n_rois_band < 2:
            logger.warning(
                "Need at least 2 surviving ROIs for source connectivity; got %d for band %s.",
                n_rois_band,
                band,
            )
            continue

        if not _validate_source_connectivity_duration(
            n_times=int(roi_data.shape[-1]) if np.ndim(roi_data) >= 3 else 0,
            sfreq=float(sfreq),
            fmin=float(fmin),
            min_cycles=min_cycles_per_band,
            band=str(band),
            method=connectivity_method_l,
            logger=logger,
        ):
            continue
        
        if connectivity_method_l == "aec":
            con = envelope_correlation(
                roi_data,
                orthogonalize="pairwise",
                verbose=False,
            )
            con_data = np.asarray(con.get_data(output="dense"), dtype=float)
            if con_data.ndim == 4:
                con_tensor = con_data[:, :, :, 0]
            elif con_data.ndim == 3:
                if con_data.shape[0] == n_epochs:
                    con_tensor = con_data
                else:
                    con_tensor = np.broadcast_to(
                        con_data[:, :, 0],
                        (n_epochs, n_rois_band, n_rois_band),
                    )
            else:
                raise ValueError(
                    "Envelope correlation returned an unexpected tensor shape "
                    f"{con_data.shape!r}; expected 3D or 4D dense output."
                )
            triu_idx = np.triu_indices(n_rois_band, k=1)

            col_name = f"src_{src_cfg.method}_{band}_aec_global"
            if col_name not in feature_cols:
                feature_cols.append(col_name)

            for epoch_idx in range(n_epochs):
                mean_conn = float(np.nanmean(con_tensor[epoch_idx][triu_idx]))
                records[epoch_idx][col_name] = mean_conn
                
        elif connectivity_method_l in {"wpli", "plv"}:
            roi_data_fit = roi_data
            if (
                analysis_mode == "trial_ml_safe"
                and train_mask is not None
                and train_mask.shape[0] == n_epochs
                and np.any(train_mask)
            ):
                roi_data_fit = roi_data[train_mask]
            if roi_data_fit.shape[0] < 2:
                logger.warning(
                    "Source connectivity (%s): insufficient epochs for stable cross-epoch estimate (%d); skipping band %s.",
                    connectivity_method_l,
                    int(roi_data_fit.shape[0]),
                    band,
                )
                continue

            edge_indices = np.triu_indices(n_rois_band, k=1)
            if edge_indices[0].size < 1:
                continue

            con = spectral_connectivity_epochs(
                roi_data_fit,
                method=connectivity_method_l,
                mode="multitaper",
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                faverage=True,
                indices=edge_indices,
                n_jobs=n_jobs,
                verbose=False,
            )

            con_data = np.asarray(con.get_data(), dtype=float)
            if (
                con_data.ndim >= 3
                and con_data.shape[0] == n_rois_band
                and con_data.shape[1] == n_rois_band
            ):
                edge_values = con_data[:, :, 0][edge_indices]
            elif con_data.ndim == 2:
                edge_values = con_data[:, 0]
            elif con_data.ndim == 1:
                edge_values = con_data
            else:
                edge_values = np.ravel(con_data)
            mean_conn = float(np.nanmean(edge_values)) if edge_values.size else np.nan

            for epoch_idx in range(n_epochs):
                col_name = f"src_{src_cfg.method}_{band}_{connectivity_method_l}_global"
                if col_name not in feature_cols:
                    feature_cols.append(col_name)
                records[epoch_idx][col_name] = mean_conn
    
    features_df = pd.DataFrame(records)
    feature_cols = list(features_df.columns)

    features_df.attrs["method"] = str(src_cfg.method)
    features_df.attrs["connectivity_method"] = connectivity_method_l
    features_df.attrs["segment_label"] = str(segment_label)
    features_df.attrs["fmri_constraint_enabled"] = bool(fmri_cfg.enabled)
    features_df.attrs["fmri_provenance"] = str(fmri_cfg.provenance)
    conn_method_l = connectivity_method_l
    if conn_method_l in {"wpli", "plv"}:
        features_df.attrs["feature_granularity"] = "subject"
        features_df.attrs["broadcast_warning"] = (
            f"Source connectivity method='{conn_method_l}' is estimated across epochs and broadcast to all rows. "
            "Treat rows as non-i.i.d.; aggregate before trial-level inference."
        )
        features_df.attrs["threshold_train_mask_used"] = bool(
            analysis_mode == "trial_ml_safe"
            and train_mask is not None
            and np.any(train_mask)
        )
    features_df.attrs["train_mask_used_for_covariance"] = bool(
        src_cfg.method == "lcmv"
        and analysis_mode == "trial_ml_safe"
        and train_mask is not None
        and np.any(train_mask)
    )
    
    if logger:
        logger.info(f"Source connectivity: {len(feature_cols)} features extracted")
    
    return features_df, feature_cols


def extract_source_localization_from_precomputed(
    precomputed: Any,
    method: str = "lcmv",
    bands: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract source-localized features from precomputed data.
    
    This is a wrapper that extracts source features when precomputed
    band-limited data is available.
    
    Parameters
    ----------
    precomputed : PrecomputedData
        Precomputed intermediate data
    method : str
        Source localization method
    bands : list of str, optional
        Bands to process
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    features_df : pd.DataFrame
        Source features
    feature_cols : list of str
        Feature column names
    """
    logger = getattr(precomputed, "logger", None) or logging.getLogger(__name__)
    
    if not hasattr(precomputed, "epochs") or precomputed.epochs is None:
        logger.warning("Source localization requires epochs in precomputed data")
        return pd.DataFrame(), []
    
    class MockContext:
        def __init__(self, precomputed):
            self.epochs = precomputed.epochs
            self.config = getattr(precomputed, "config", {})
            self.logger = logger
            self.frequency_bands = {}
            if hasattr(precomputed, "band_data") and precomputed.band_data:
                for band_name, band_info in precomputed.band_data.items():
                    if hasattr(band_info, "fmin") and hasattr(band_info, "fmax"):
                        self.frequency_bands[band_name] = (band_info.fmin, band_info.fmax)
    
    ctx = MockContext(precomputed)
    
    if bands is None:
        bands = list(ctx.frequency_bands.keys())
    
    return extract_source_localization_features(
        ctx,
        bands=bands,
        method=method,
        n_jobs=n_jobs,
    )
