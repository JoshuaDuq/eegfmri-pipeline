#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from utils import get_log_function


SCRIPT_NAME = Path(__file__).stem
log, _ = get_log_function(SCRIPT_NAME, console_level="INFO")


###################################################################
# Helper Functions
###################################################################


def _format_voxel_sizes(zooms) -> str:
    return "×".join(f"{float(v):.1f}" for v in zooms) + " mm"


def _count_finite_nonzero(data: np.ndarray) -> int:
    return int(np.sum(np.isfinite(data) & (data != 0)))


###################################################################
# Main Resampling
###################################################################


def resample_siips1_to_nps_grid() -> int:
    log("Resample SIIPS1")
    nps_path = Path("resources/weights_NSF_grouppred_cvpcr.nii.gz")
    siips1_source = Path("resources/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz")
    siips1_resampled = Path("resources/nonnoc_v11_4_137subjmap_weighted_mean_3mm.nii.gz")

    if not nps_path.exists():
        log(f"NPS: {nps_path}", "ERROR")
        return 1
    if not siips1_source.exists():
        log(f"SIIPS1: {siips1_source}", "ERROR")
        return 1

    nps_img = nib.load(str(nps_path))
    siips1_native = nib.load(str(siips1_source))
    siips1_native_data = siips1_native.get_fdata()

    resampled_img = resample_to_img(
        siips1_native,
        nps_img,
        interpolation="continuous",
        copy=True,
        force_resample=True,
    )
    resampled_data = resampled_img.get_fdata()

    if resampled_img.shape != nps_img.shape:
        log(f"Shape: {resampled_img.shape} vs {nps_img.shape}", "ERROR")
        return 1

    affine_diff = float(np.max(np.abs(resampled_img.affine - nps_img.affine)))
    if affine_diff > 1e-3:
        log(f"Affine: {affine_diff:.6f}", "ERROR")
        return 1

    nib.save(resampled_img, str(siips1_resampled))

    metadata = {
        "original_resolution_mm": 2.0,
        "target_resolution_mm": 3.0,
        "original_voxel_volume_mm3": 8.0,
        "target_voxel_volume_mm3": 27.0,
        "resampled_from_native": True,
        "scale_factor_already_applied": True,
        "note": (
            "SIIPS1 weights were resampled from native 2 mm resolution to the 3 mm NPS grid. "
            "Do not apply additional voxel-volume scale factors during scoring."
        ),
    }
    metadata_path = siips1_resampled.with_name(f"{siips1_resampled.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(resample_siips1_to_nps_grid())
