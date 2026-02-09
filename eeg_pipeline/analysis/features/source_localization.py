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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.utils.config.loader import get_frequency_bands, get_nested_value

if TYPE_CHECKING:
    import mne


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get") and not isinstance(config, dict):
        return config.get(key, default)
    return get_nested_value(config, key, default)


def _safe_float(value: Any, default: float) -> float:
    """Safely convert value to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int) -> int:
    """Safely convert value to int, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    return Path(str(value)).expanduser()


@dataclass(frozen=True)
class FMRITimeWindow:
    """fMRI-specific time window definition."""
    name: str
    tmin: float
    tmax: float


@dataclass(frozen=True)
class FMRIConstraintConfig:
    enabled: bool
    stats_map_path: Optional[Path]
    provenance: str  # "independent" | "same_dataset" | "unknown"
    require_provenance: bool
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
    # fMRI-specific time windows (independent of EEG feature extraction windows)
    window_a: Optional[FMRITimeWindow] = None
    window_b: Optional[FMRITimeWindow] = None


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
    src_cfg = _cfg_get(config, "feature_engineering.sourcelocalization", {}) or {}
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
    try:
        if cluster_min_volume_mm3 is not None:
            cluster_min_volume_mm3 = float(cluster_min_volume_mm3)
            if not np.isfinite(cluster_min_volume_mm3) or cluster_min_volume_mm3 <= 0:
                cluster_min_volume_mm3 = None
    except Exception:
        cluster_min_volume_mm3 = None

    # Parse fMRI-specific time windows
    time_windows_cfg = fmri_cfg.get("time_windows", {}) or {}
    window_a = None
    window_b = None
    
    window_a_cfg = time_windows_cfg.get("window_a", {}) or {}
    if window_a_cfg.get("name"):
        name = str(window_a_cfg.get("name", "")).strip()
        tmin = float(window_a_cfg.get("tmin", 0.0))
        tmax = float(window_a_cfg.get("tmax", 0.0))
        if not name:
            raise ValueError("fMRI window_a.name is empty.")
        if not (np.isfinite(tmin) and np.isfinite(tmax) and tmax > tmin):
            raise ValueError(f"Invalid fMRI window_a range: tmin={tmin}, tmax={tmax}.")
        window_a = FMRITimeWindow(name=name, tmin=tmin, tmax=tmax)
    
    window_b_cfg = time_windows_cfg.get("window_b", {}) or {}
    if window_b_cfg.get("name"):
        name = str(window_b_cfg.get("name", "")).strip()
        tmin = float(window_b_cfg.get("tmin", 0.0))
        tmax = float(window_b_cfg.get("tmax", 0.0))
        if not name:
            raise ValueError("fMRI window_b.name is empty.")
        if not (np.isfinite(tmin) and np.isfinite(tmax) and tmax > tmin):
            raise ValueError(f"Invalid fMRI window_b range: tmin={tmin}, tmax={tmax}.")
        window_b = FMRITimeWindow(name=name, tmin=tmin, tmax=tmax)

    return FMRIConstraintConfig(
        enabled=enabled or (stats_map_path is not None),
        stats_map_path=stats_map_path,
        provenance=provenance,
        require_provenance=require_provenance,
        threshold=threshold,
        tail=tail,
        threshold_mode=threshold_mode,
        fdr_q=fdr_q,
        stat_type=stat_type,
        cluster_min_voxels=_safe_int(fmri_cfg.get("cluster_min_voxels", 50), 50),
        cluster_min_volume_mm3=cluster_min_volume_mm3,
        max_clusters=_safe_int(fmri_cfg.get("max_clusters", 20), 20),
        max_voxels_per_cluster=_safe_int(fmri_cfg.get("max_voxels_per_cluster", 2000), 2000),
        max_total_voxels=_safe_int(fmri_cfg.get("max_total_voxels", 20000), 20000),
        random_seed=_safe_int(fmri_cfg.get("random_seed", 0), 0),
        window_a=window_a,
        window_b=window_b,
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


def _fmri_roi_coords_from_stats_map(
    stats_map_path: Path,
    cfg: FMRIConstraintConfig,
    logger: Optional[logging.Logger],
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

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI fMRI map, got shape={data.shape}.")

    thr = float(cfg.threshold)
    if not np.isfinite(thr) or thr <= 0:
        thr = 3.1

    finite = np.isfinite(data)
    if str(cfg.threshold_mode).lower() == "fdr":
        q = float(cfg.fdr_q)
        if not (np.isfinite(q) and 0 < q < 1):
            q = 0.05

        # FDR is implemented for z-maps (or maps treated as z).
        # If a user provides non-z maps, they should convert externally.
        zvals = data[finite].astype(float, copy=False)
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
            mask[finite] = keep
    else:
        if cfg.tail == "abs":
            mask = finite & (np.abs(data) >= thr)
        else:
            mask = finite & (data >= thr)

    if not np.any(mask):
        # Log statistics about the contrast map to diagnose the issue
        finite_data = data[finite]
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

    rng = np.random.default_rng(None if int(cfg.random_seed) == 0 else int(cfg.random_seed))

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
        
    Returns
    -------
    stcs : list of mne.SourceEstimate
        Source estimates for each epoch
    inv : mne.minimum_norm.InverseOperator
        Inverse operator
    """
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
        pick_ori="normal",
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
    
    roi_data = np.zeros((n_epochs, n_rois, n_times))
    
    for epoch_idx, stc in enumerate(stcs):
        for roi_idx, label in enumerate(labels):
            tc = mne.extract_label_time_course(
                stc,
                labels=[label],
                src=src,
                mode=mode,
                allow_empty=True,
                verbose=False,
            )
            roi_data[epoch_idx, roi_idx, :] = tc.squeeze()
    
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
    
    nperseg = min(n_times, int(sfreq * 2))
    
    for epoch_idx in range(n_epochs):
        for roi_idx in range(n_rois):
            tc = roi_data[epoch_idx, roi_idx, :]
            if np.any(np.isnan(tc)):
                power[epoch_idx, roi_idx] = np.nan
                continue
            
            freqs, psd = welch(tc, fs=sfreq, nperseg=nperseg)
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
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
    
    nyq = sfreq / 2.0
    low = fmin / nyq
    high = min(fmax / nyq, 0.99)
    
    if low >= high or low <= 0:
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
    src_cfg = _cfg_get(config, "feature_engineering.sourcelocalization", {}) or {}
    if not isinstance(src_cfg, dict):
        src_cfg = {}

    method_use = str(src_cfg.get("method", method)).strip().lower()
    if method_use not in {"lcmv", "eloreta"}:
        method_use = "lcmv"

    spacing = str(src_cfg.get("spacing", "oct6")).strip()
    if spacing not in {"oct5", "oct6", "ico4", "ico5"}:
        spacing = "oct6"

    parcellation = str(src_cfg.get("parcellation", src_cfg.get("parc", "aparc"))).strip()
    if parcellation not in {"aparc", "aparc.a2009s", "HCPMMP1"}:
        parcellation = "aparc"

    subjects_dir_path = _as_path(src_cfg.get("subjects_dir"))
    if subjects_dir_path is None:
        subjects_dir_path = _as_path(_cfg_get(config, "paths.freesurfer_dir", None))

    subject_from_cfg = src_cfg.get("subject")
    if subject_from_cfg is None or str(subject_from_cfg).strip().lower() in ("none", ""):
        subject = f"sub-{getattr(ctx, 'subject', '')}".strip() or "fsaverage"
    else:
        subject = str(subject_from_cfg).strip()

    trans_path = _as_path(src_cfg.get("trans"))
    bem_path = _as_path(src_cfg.get("bem"))

    mindist_mm = _safe_float(src_cfg.get("mindist_mm", 5.0), 5.0)
    lcmv_reg = _safe_float(src_cfg.get("reg", 0.05), 0.05)
    eloreta_snr = _safe_float(src_cfg.get("snr", 3.0), 3.0)
    eloreta_loose = _safe_float(src_cfg.get("loose", 0.2), 0.2)
    eloreta_depth = _safe_float(src_cfg.get("depth", 0.8), 0.8)

    bids_fmri_root = _as_path(_cfg_get(config, "paths.bids_fmri_root", None))
    bids_derivatives = _as_path(_cfg_get(config, "paths.deriv_root", None))
    if bids_fmri_root is None:
        bids_fmri_root = _as_path(_cfg_get(config, "paths.bids_root", None))

    ctx_subject = getattr(ctx, "subject", None) or subject.replace("sub-", "")
    eeg_task = str(_cfg_get(config, "project.task", "")).strip()
    fmri_task = eeg_task

    fmri_cfg = _load_fmri_constraint_config(
        config,
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        freesurfer_subjects_dir=subjects_dir_path,
        subject=ctx_subject,
        task=fmri_task,
    )

    return SourceLocalizationConfig(
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
    train_mask = getattr(ctx, "train_mask", None)
    epochs_fit = None
    if analysis_mode == "trial_ml_safe" and train_mask is not None:
        train_mask = np.asarray(train_mask, dtype=bool).ravel()
        if train_mask.size == len(epochs) and np.any(train_mask):
            epochs_fit = epochs[train_mask]

    src_cfg = _load_source_localization_config(ctx, config, method)
    fmri_cfg = src_cfg.fmri_cfg

    if fmri_cfg.enabled:
        if fmri_cfg.require_provenance and fmri_cfg.provenance == "unknown":
            raise ValueError(
                "Source localization: fMRI constraint enabled but fmri.provenance is unknown. "
                "Set feature_engineering.sourcelocalization.fmri.provenance to "
                "'independent' (recommended) or 'same_dataset' (circularity risk)."
            )
        if fmri_cfg.provenance == "same_dataset":
            logger.warning(
                "Source localization: fMRI constraint provenance is 'same_dataset'. "
                "This can create circularity/double-dipping if you test EEG features against the same labels/conditions."
            )

    n_epochs = len(epochs)
    if n_epochs < 2:
        logger.warning("Source localization requires at least 2 epochs")
        return pd.DataFrame(), []

    sfreq = epochs.info["sfreq"]
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)

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

        roi_coords_m = _fmri_roi_coords_from_stats_map(fmri_cfg.stats_map_path, fmri_cfg, logger)
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
    train_mask = getattr(ctx, "train_mask", None)
    train_mask = np.asarray(train_mask, dtype=bool).ravel() if train_mask is not None else None
    if train_mask is not None and train_mask.size != len(epochs):
        train_mask = None

    src_cfg = _load_source_localization_config(ctx, config, method)
    fmri_cfg = src_cfg.fmri_cfg

    if fmri_cfg.enabled:
        if fmri_cfg.require_provenance and fmri_cfg.provenance == "unknown":
            raise ValueError(
                "Source connectivity: fMRI constraint enabled but fmri.provenance is unknown. "
                "Set feature_engineering.sourcelocalization.fmri.provenance to "
                "'independent' (recommended) or 'same_dataset' (circularity risk)."
            )
        if fmri_cfg.provenance == "same_dataset":
            logger.warning(
                "Source connectivity: fMRI constraint provenance is 'same_dataset'. "
                "This can create circularity/double-dipping if you test EEG features against the same labels/conditions."
            )

    n_epochs = len(epochs)
    if n_epochs < 2:
        logger.warning("Source connectivity requires at least 2 epochs")
        return pd.DataFrame(), []
    
    sfreq = epochs.info["sfreq"]
    freq_bands = getattr(ctx, "frequency_bands", None) or get_frequency_bands(config)
    
    if logger:
        logger.info(f"Extracting source-space {connectivity_method.upper()} connectivity")

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

        roi_coords_m = _fmri_roi_coords_from_stats_map(fmri_cfg.stats_map_path, fmri_cfg, logger)
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
        
        epochs_band = epochs.copy().filter(fmin, fmax, n_jobs=n_jobs, verbose=False)
        
        if src_cfg.method == "lcmv":
            epochs_fit = None
            if analysis_mode == "trial_ml_safe" and train_mask is not None and np.any(train_mask):
                epochs_fit = epochs_band[train_mask]
            stcs, _ = _compute_lcmv_source_estimates(
                epochs_band,
                fwd,
                epochs_fit=epochs_fit,
                reg=src_cfg.lcmv_reg,
                logger=None,
            )
        else:
            stcs, _ = _compute_eloreta_source_estimates(
                epochs_band,
                fwd,
                loose=src_cfg.eloreta_loose,
                depth=src_cfg.eloreta_depth,
                snr=src_cfg.eloreta_snr,
                logger=None,
            )

        if fmri_cfg.enabled:
            roi_data, _ = _extract_roi_timecourses_from_vertex_indices(stcs, roi_indices)
        else:
            roi_data = _extract_roi_timecourses(stcs, labels, src, mode="mean_flip")
        
        if connectivity_method.lower() == "aec":
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
                
        elif connectivity_method.lower() in ("wpli", "plv"):
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
                    connectivity_method.lower(),
                    int(roi_data_fit.shape[0]),
                    band,
                )
                continue

            con = spectral_connectivity_epochs(
                roi_data_fit,
                method=connectivity_method.lower(),
                mode="multitaper",
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                faverage=True,
                n_jobs=n_jobs,
                verbose=False,
            )
            
            con_data = con.get_data()
            
            for epoch_idx in range(n_epochs):
                if con_data.ndim == 3:
                    epoch_con = con_data[epoch_idx, :, 0]
                else:
                    epoch_con = con_data[:, 0]
                
                mean_conn = np.nanmean(epoch_con)
                
                col_name = f"src_{src_cfg.method}_{band}_{connectivity_method}_global"
                if col_name not in feature_cols:
                    feature_cols.append(col_name)
                records[epoch_idx][col_name] = mean_conn
    
    features_df = pd.DataFrame(records)
    feature_cols = list(features_df.columns)

    features_df.attrs["method"] = str(src_cfg.method)
    features_df.attrs["connectivity_method"] = str(connectivity_method).lower()
    features_df.attrs["fmri_constraint_enabled"] = bool(fmri_cfg.enabled)
    features_df.attrs["fmri_provenance"] = str(fmri_cfg.provenance)
    conn_method_l = str(connectivity_method).strip().lower()
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
