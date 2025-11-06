from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


###################################################################
# Voxel Volume
###################################################################

def voxel_volume(img: nib.Nifti1Image) -> Optional[float]:
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        return None
    volume = float(np.prod(zooms[:3]))
    return volume if np.isfinite(volume) and volume > 0 else None


###################################################################
# Signature Weights
###################################################################

def load_signature_weights(weights_path: Path, signature_name: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    if not weights_path.exists():
        raise FileNotFoundError(f"{signature_name.upper()} weights not found: {weights_path}")
    weights_img = nib.load(str(weights_path))
    data = np.squeeze(weights_img.get_fdata())
    mask = np.isfinite(data) & (data != 0)
    return weights_img, mask


###################################################################
# Resampling
###################################################################

def resample_beta_to_signature_grid(beta_img: nib.Nifti1Image, signature_img: nib.Nifti1Image, interpolation: str = 'linear') -> nib.Nifti1Image:
    resampled = resample_to_img(
        beta_img, signature_img, 
        interpolation=interpolation, copy=True, 
        force_resample=True, copy_header=True
    )
    data = np.squeeze(resampled.get_fdata())
    if data.ndim != 3:
        raise ValueError(f"Expected 3D, got {data.ndim}D")
    
    n_invalid = np.sum(~np.isfinite(data))
    if n_invalid > 0:
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return nib.Nifti1Image(data.astype(np.float32), resampled.affine, resampled.header)


###################################################################
# Grid Validation
###################################################################

def validate_grid_match(beta_img: nib.Nifti1Image, signature_img: nib.Nifti1Image) -> Dict:
    beta_data = beta_img.get_fdata()
    sig_data = signature_img.get_fdata()
    beta_mask = np.isfinite(beta_data) & (beta_data != 0)
    sig_mask = np.isfinite(sig_data) & (sig_data != 0)
    
    shapes_match = beta_img.shape == signature_img.shape
    affine_max_diff = float(np.max(np.abs(beta_img.affine - signature_img.affine)))
    
    if np.any(sig_mask):
        overlap_pct = float(100.0 * np.sum(beta_mask & sig_mask) / np.sum(sig_mask))
    else:
        overlap_pct = 0.0
    
    return {
        'shapes_match': shapes_match,
        'affine_max_diff': affine_max_diff,
        'overlap_pct': overlap_pct
    }


###################################################################
# Analysis Mask
###################################################################

def build_analysis_mask(inventory: Dict, min_runs_threshold: Optional[int] = None) -> nib.Nifti1Image:
    from .logging_utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    mask_paths = [
        Path(run.get("files", {}).get("mask", {}).get("path", ""))
        for run in inventory.get("runs", {}).values()
    ]
    mask_paths = [p for p in mask_paths if p.exists()]
    if not mask_paths:
        raise FileNotFoundError("No run masks found")
    
    masks = [nib.load(str(p)) for p in mask_paths]
    reference = masks[0]
    
    for idx, img in enumerate(masks[1:], start=2):
        if img.shape != reference.shape or not np.allclose(img.affine, reference.affine):
            raise ValueError(f"Mask {idx} misaligned")
    
    threshold = len(masks) if min_runs_threshold is None else max(1, min(min_runs_threshold, len(masks)))
    
    mask_stack = np.stack([(img.get_fdata() > 0).astype(np.uint8) for img in masks], axis=0)
    mask_final = mask_stack.sum(axis=0) >= threshold
    
    if not np.any(mask_final):
        raise ValueError("Empty analysis mask")
    
    return nib.Nifti1Image(mask_final.astype(np.uint8), reference.affine, reference.header.copy())


###################################################################
# Beta Maps
###################################################################

def find_run_betas_for_temperature(subject_dir: Path, temp_label: str, runs: List[int]) -> List[Path]:
    beta_paths = []
    for run_num in runs:
        beta_path = subject_dir / f"run-{run_num:02d}_beta_{temp_label}.nii.gz"
        if beta_path.exists():
            beta_paths.append(beta_path)
    return beta_paths


def find_variance_maps(beta_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    from .logging_utils import get_log_function
    log, _ = get_log_function(Path(__file__).stem)
    
    paired = [(bp, bp.parent / bp.name.replace(".nii.gz", "_variance.nii.gz")) for bp in beta_paths]
    valid = [(bp, vp) for bp, vp in paired if vp.exists()]
    if not valid:
        raise FileNotFoundError("No variance maps found for any beta maps")
    return [bp for bp, _ in valid], [vp for _, vp in valid]

