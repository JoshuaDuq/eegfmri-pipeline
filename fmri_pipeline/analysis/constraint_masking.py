from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def _load_image(image_or_path: Any) -> Any:
    import nibabel as nib  # type: ignore

    if isinstance(image_or_path, (str, Path)):
        return nib.load(str(image_or_path))
    return image_or_path


def _align_mask_to_image(mask_img: Any, ref_img: Any) -> np.ndarray:
    mask = _load_image(mask_img)
    if tuple(getattr(mask, "shape", ())) != tuple(getattr(ref_img, "shape", ())):
        from nilearn.image import resample_to_img  # type: ignore

        mask = resample_to_img(
            mask,
            ref_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    mask_data = np.asarray(mask.get_fdata(), dtype=np.float32)
    if mask_data.shape != tuple(getattr(ref_img, "shape", ())):
        raise ValueError(
            "Constraint analysis mask does not match the z-map grid after alignment "
            f"(mask_shape={mask_data.shape}, z_map_shape={getattr(ref_img, 'shape', None)})."
        )
    return mask_data > 0


def build_thresholded_constraint_mask(
    z_img: Any,
    *,
    threshold_mode: str,
    z_threshold: float,
    fdr_q: float,
    tail: str,
    analysis_mask_img: Optional[Any],
    min_cluster_voxels: int,
    min_cluster_volume_mm3: Optional[float] = None,
) -> Any:
    import nibabel as nib  # type: ignore
    from scipy import stats as sps  # type: ignore
    from scipy.ndimage import generate_binary_structure, label as cc_label  # type: ignore

    z_map = _load_image(z_img)
    data = np.asarray(z_map.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Constraint masking expects a 3D z-map, got shape={data.shape}.")

    threshold_mode = str(threshold_mode or "z").strip().lower()
    if threshold_mode not in {"z", "fdr"}:
        raise ValueError(f"threshold_mode must be 'z' or 'fdr' (got {threshold_mode!r}).")

    tail = str(tail or "pos").strip().lower()
    if tail not in {"pos", "abs"}:
        raise ValueError(f"tail must be 'pos' or 'abs' (got {tail!r}).")

    if analysis_mask_img is None:
        analysis_voxels = np.isfinite(data)
    else:
        analysis_mask = _align_mask_to_image(analysis_mask_img, z_map)
        analysis_voxels = np.isfinite(data) & analysis_mask

    if not np.any(analysis_voxels):
        raise ValueError("Constraint masking found no finite voxels inside the analysis mask.")

    if threshold_mode == "fdr":
        q = float(fdr_q)
        if not np.isfinite(q) or not (0 < q < 1):
            raise ValueError(f"fdr_q must be in (0, 1), got {fdr_q!r}.")
        zvals = data[analysis_voxels].astype(float, copy=False)
        if tail == "abs":
            pvals = 2.0 * (1.0 - sps.norm.cdf(np.abs(zvals)))
        else:
            pvals = 1.0 - sps.norm.cdf(zvals)
        pvals = np.clip(pvals, 0.0, 1.0)
        order = np.argsort(pvals)
        ranked = pvals[order]
        bh = q * (np.arange(1, ranked.size + 1, dtype=float) / float(ranked.size))
        below = ranked <= bh
        seed = np.zeros_like(data, dtype=bool)
        if np.any(below):
            cutoff = float(ranked[int(np.max(np.where(below)[0]))])
            seed[analysis_voxels] = pvals <= cutoff
    else:
        thr = float(z_threshold)
        if not np.isfinite(thr) or thr <= 0:
            raise ValueError(f"z_threshold must be > 0, got {z_threshold!r}.")
        if tail == "abs":
            seed = analysis_voxels & (np.abs(data) >= thr)
        else:
            seed = analysis_voxels & (data >= thr)

    if not np.any(seed):
        return nib.Nifti1Image(np.zeros_like(data, dtype=np.uint8), z_map.affine, z_map.header)

    voxel_vol_mm3 = None
    zooms = z_map.header.get_zooms()
    if len(zooms) >= 3:
        voxel_vol_mm3 = float(abs(float(zooms[0]) * float(zooms[1]) * float(zooms[2])))
        if not np.isfinite(voxel_vol_mm3) or voxel_vol_mm3 <= 0:
            voxel_vol_mm3 = None

    structure = generate_binary_structure(3, 1)
    labeled, n_clusters = cc_label(seed.astype(np.uint8), structure=structure)
    keep = np.zeros_like(seed, dtype=bool)
    for cluster_id in range(1, int(n_clusters) + 1):
        voxels = labeled == cluster_id
        n_voxels = int(np.sum(voxels))
        if min_cluster_volume_mm3 is not None and voxel_vol_mm3 is not None:
            if float(n_voxels) * float(voxel_vol_mm3) < float(min_cluster_volume_mm3):
                continue
        elif n_voxels < int(min_cluster_voxels):
            continue
        keep |= voxels

    return nib.Nifti1Image(keep.astype(np.uint8), z_map.affine, z_map.header)
