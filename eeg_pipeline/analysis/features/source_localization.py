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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_config_float, get_config_int, get_config_value, get_frequency_bands

if TYPE_CHECKING:
    import mne



def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    return Path(str(value)).expanduser()


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
    """
    Convert a thresholded fMRI statistical map into discrete volume-source coordinates.

    IMPORTANT: For accurate EEG↔MRI alignment, `stats_map_path` must be in the SAME MRI
    space as your FreeSurfer subject (typically produced by resampling into `orig.mgz`
    or `T1.mgz` space and then converting to NIfTI).
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
    subject_ref_data = np.asarray(
        nib.load(str(matched_ref_path)).get_fdata(dtype=np.float32)
    )
    if subject_ref_data.ndim != 3:
        raise ValueError(
            f"Expected 3D subject MRI reference, got shape={subject_ref_data.shape} "
            f"from {matched_ref_path}."
        )

    thr = float(cfg.threshold)
    if not np.isfinite(thr) or thr <= 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.threshold must be > 0 "
            f"(got {thr})."
        )

    analysis_voxels = _compute_fmri_analysis_voxel_mask(
        data,
        subject_ref_data=subject_ref_data,
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
        # Log statistics about the contrast map to diagnose the issue
        finite_data = data[analysis_voxels]
        if len(finite_data) > 0:
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
        if cfg.cluster_min_volume_mm3 is not None and voxel_vol_mm3 is not None:
            raise ValueError(
                "All clusters were smaller than "
                f"cluster_min_volume_mm3={float(cfg.cluster_min_volume_mm3):g} (voxel_vol_mm3={float(voxel_vol_mm3):g})."
            )
        raise ValueError(f"All clusters were smaller than cluster_min_voxels={cfg.cluster_min_voxels}.")

    clusters.sort(key=lambda t: (t[2], t[1]), reverse=True)
    clusters = clusters[: max(1, int(cfg.max_clusters))]

    rng = _make_fmri_subsampling_rng(int(cfg.random_seed))

    roi_coords_m: Dict[str, np.ndarray] = {}
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

        coords_mm = nib.affines.apply_affine(affine, vox)
        coords_m = np.asarray(coords_mm, dtype=float) / 1000.0
        roi_name = f"fmri_c{out_idx:02d}_peak{peak_val:.2f}".replace(".", "p")
        roi_coords_m[roi_name] = coords_m
        total_selected += int(coords_m.shape[0])

    if logger:
        logger.info(
            "fMRI constraint: %d clusters, %d voxels (map=%s)",
            len(roi_coords_m),
            total_selected,
            str(stats_map_path),
        )

    if not roi_coords_m:
        raise ValueError("fMRI constraint produced no ROI coordinates after subsampling.")

    return roi_coords_m


def _extract_roi_timecourses_from_vertex_indices(
    stcs: List[Any],
    roi_indices: Dict[str, List[int]],
) -> Tuple[np.ndarray, List[str]]:
    """Extract ROI timecourses from STCs using explicit vertex mapping to handle pruned sources."""
    n_epochs = len(stcs)
    if n_epochs == 0:
        return np.zeros((0, 0, 0)), []
    
    roi_names = list(roi_indices.keys())
    n_rois = len(roi_names)
    n_times = int(stcs[0].data.shape[1])
    
    # stcs[0].vertices is a list (one per source space). For volume, usually [vertno]
    stc_verts = stcs[0].vertices[0]
    # Create mapping: original_vertex_index -> row_index_in_stc_data
    vert_to_idx = {v: i for i, v in enumerate(stc_verts)}

    missing_rois = [
        roi_name
        for roi_name, orig_idxs in roi_indices.items()
        if orig_idxs and not any(v in vert_to_idx for v in orig_idxs)
    ]
    if missing_rois:
        raise ValueError(
            "fMRI-constrained ROI mapping failed: some ROIs have no surviving vertices "
            f"after forward/inverse modeling: {', '.join(missing_rois)}"
        )
    
    out = np.full((n_epochs, n_rois, n_times), np.nan, dtype=float)
    for epoch_idx, stc in enumerate(stcs):
        data = np.asarray(stc.data, dtype=float)
        for roi_idx, roi_name in enumerate(roi_names):
            # Get original indices for this ROI
            orig_idxs = roi_indices.get(roi_name, [])
            if not orig_idxs:
                continue
            
            # Find which of these indices survived in the STC
            row_idxs = [vert_to_idx[v] for v in orig_idxs if v in vert_to_idx]
            if not row_idxs:
                continue
                
            block = data[row_idxs, :]
            out[epoch_idx, roi_idx, :] = np.nanmean(block, axis=0)
            
    return out, roi_names


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
    
    n_epochs = len(stcs)
    n_rois = len(labels)
    n_times = stcs[0].data.shape[1]
    
    roi_data = np.zeros((n_epochs, n_rois, n_times), dtype=float)
    
    for epoch_idx, stc in enumerate(stcs):
        try:
            tc = mne.extract_label_time_course(
                stc,
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
        roi_data[epoch_idx, :, :] = np.asarray(tc, dtype=float)
    
    return roi_data


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
    from scipy.signal import welch
    
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
    
    nperseg = min(n_times, int(sfreq * 2))
    
    for epoch_idx in range(n_epochs):
        for roi_idx in range(n_rois):
            tc = roi_data[epoch_idx, roi_idx, :]
            if np.any(np.isnan(tc)):
                power[epoch_idx, roi_idx] = np.nan
                continue
            
            freqs, psd = welch(tc, fs=sfreq, nperseg=nperseg)
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(freq_mask):
                raise ValueError(
                    "ROI power band has no Welch frequency bins; increase segment length or use a lower band. "
                    f"(fmin={fmin}, fmax={fmax}, sfreq={sfreq}, n_times={n_times})"
                )
            power[epoch_idx, roi_idx] = np.mean(psd[freq_mask])
    
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
    from scipy.signal import butter, filtfilt, hilbert
    
    n_epochs, n_rois, n_times = roi_data.shape
    envelope = np.zeros_like(roi_data)
    
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Sampling frequency must be > 0 for ROI envelope (got {sfreq}).")
    nyq = sfreq / 2.0
    low = fmin / nyq
    high = fmax / nyq
    
    if low >= high or low <= 0 or high >= 1:
        raise ValueError(f"Invalid bandpass range for ROI envelope: fmin={fmin}, fmax={fmax}, sfreq={sfreq}.")

    b, a = butter(4, [low, high], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if n_times <= padlen:
        raise ValueError(
            f"ROI envelope requires at least {padlen + 1} samples for filtfilt padding "
            f"(n_times={n_times}, sfreq={sfreq}, fmin={fmin}, fmax={fmax})."
        )
    
    for epoch_idx in range(n_epochs):
        for roi_idx in range(n_rois):
            tc = roi_data[epoch_idx, roi_idx, :]
            if np.any(np.isnan(tc)) or np.std(tc) < 1e-12:
                envelope[epoch_idx, roi_idx, :] = np.nan
                continue
            
            filtered = filtfilt(b, a, tc)
            analytic = hilbert(filtered)
            envelope[epoch_idx, roi_idx, :] = np.abs(analytic)
    
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
    )


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
            trans=str(trans_path),
            bem=str(bem_path),
            roi_coords_m=roi_coords_m,
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
        roi_data, label_names = _extract_roi_timecourses_from_vertex_indices(stcs, roi_indices)
        if roi_data.size == 0 or not label_names:
            logger.warning("fMRI-constrained source localization produced no ROI time courses.")
            return pd.DataFrame(), []
    else:
        if src_cfg.subjects_dir is not None and src_cfg.trans_path is not None and src_cfg.bem_path is not None:
            fwd, src = _setup_surface_forward_model_configured(
                epochs.info,
                subject=src_cfg.subject,
                subjects_dir=str(src_cfg.subjects_dir),
                spacing=src_cfg.spacing,
                trans=str(src_cfg.trans_path),
                bem=str(src_cfg.bem_path),
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
    
    segment_label = ctx.name or getattr(ctx.windows, "name", None) or "full"
    
    records = [{} for _ in range(n_epochs)]
    feature_cols = []
    
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
        
        power = _compute_roi_power(roi_data, sfreq, fmin, fmax)
        
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
        mean_env = np.nanmean(envelope, axis=2)
        
        for roi_idx, roi_name in enumerate(label_names):
            safe_name = roi_name.replace("-", "_").replace(" ", "_")
            col_name = f"src_{segment_label}_{src_cfg.method}_{band}_{safe_name}_envelope"
            feature_cols.append(col_name)
            for epoch_idx in range(n_epochs):
                records[epoch_idx][col_name] = mean_env[epoch_idx, roi_idx]
    
    features_df = pd.DataFrame(records)
    feature_cols = list(features_df.columns)
    features_df.attrs["method"] = str(src_cfg.method)
    features_df.attrs["fmri_constraint_enabled"] = bool(fmri_cfg.enabled)
    features_df.attrs["fmri_provenance"] = str(fmri_cfg.provenance)
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
        if src_cfg.trans_path is None:
            raise ValueError(
                "fMRI-constrained source connectivity requires feature_engineering.sourcelocalization.trans "
                "(EEG↔MRI transform .fif)."
            )
        if src_cfg.bem_path is None:
            raise ValueError(
                "fMRI-constrained source connectivity requires feature_engineering.sourcelocalization.bem "
                "(*-bem-sol.fif)."
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
            trans=str(src_cfg.trans_path),
            bem=str(src_cfg.bem_path),
            roi_coords_m=roi_coords_m,
            mindist_mm=src_cfg.mindist_mm,
            logger=logger,
        )
        labels = None
        label_names = list(roi_indices.keys())
    else:
        if src_cfg.subjects_dir is not None and src_cfg.trans_path is not None and src_cfg.bem_path is not None:
            fwd, src = _setup_surface_forward_model_configured(
                epochs.info,
                subject=src_cfg.subject,
                subjects_dir=str(src_cfg.subjects_dir),
                spacing=src_cfg.spacing,
                trans=str(src_cfg.trans_path),
                bem=str(src_cfg.bem_path),
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
    if n_rois < 2:
        logger.warning("Need at least 2 ROIs for connectivity")
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
            roi_data, _ = _extract_roi_timecourses_from_vertex_indices(stcs, roi_indices)
        else:
            roi_data = _extract_roi_timecourses(stcs, labels, src, mode="mean_flip")

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
            for epoch_idx in range(n_epochs):
                epoch_data = roi_data[epoch_idx:epoch_idx+1, :, :]
                
                con = envelope_correlation(
                    epoch_data,
                    orthogonalize="pairwise",
                    verbose=False,
                )
                con_matrix = con.combine().get_data(output="dense")[:, :, 0]
                
                triu_idx = np.triu_indices(n_rois, k=1)
                mean_conn = np.nanmean(con_matrix[triu_idx])
                
                col_name = f"src_{src_cfg.method}_{band}_aec_global"
                if col_name not in feature_cols:
                    feature_cols.append(col_name)
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

            edge_indices = np.triu_indices(n_rois, k=1)
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
            if con_data.ndim >= 3 and con_data.shape[0] == n_rois and con_data.shape[1] == n_rois:
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
