from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.study1.signature_manifest import (
    STUDY1_SIGNATURE_SPECS,
    build_signature_manifest,
)


def test_build_signature_manifest_records_map_identity(tmp_path) -> None:
    nib = pytest.importorskip("nibabel")

    nps_path = tmp_path / "NPS" / "weights_NSF_grouppred_cvpcr.nii.gz"
    siips_path = tmp_path / "SIIPS1" / "nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
    nps_path.parent.mkdir()
    siips_path.parent.mkdir()

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nps_data = np.array([[[1.0, -2.0], [0.0, np.nan]]], dtype=np.float32)
    siips_data = np.array([[[3.0, 0.0], [-4.0, 5.0]]], dtype=np.float32)
    nib.save(nib.Nifti1Image(nps_data, affine), nps_path)
    nib.save(nib.Nifti1Image(siips_data, affine), siips_path)

    manifest = build_signature_manifest(
        signature_root=tmp_path,
        signature_specs=STUDY1_SIGNATURE_SPECS,
        space="mni152nlin2009casym",
    )

    nps = manifest["signatures"]["NPS"]
    siips1 = manifest["signatures"]["SIIPS1"]
    assert nps["path"] == "NPS/weights_NSF_grouppred_cvpcr.nii.gz"
    assert nps["shape"] == [1, 2, 2]
    assert nps["affine"] == affine.tolist()
    assert nps["support"]["nonzero_voxels"] == 2
    assert nps["support"]["positive_voxels"] == 1
    assert nps["support"]["negative_voxels"] == 1
    assert nps["source_publication"]
    assert nps["source_repository_or_access_record"]
    assert siips1["support"]["nonzero_voxels"] == 3


def test_build_signature_manifest_requires_existing_maps(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NPS"):
        build_signature_manifest(
            signature_root=tmp_path,
            signature_specs=STUDY1_SIGNATURE_SPECS,
            space="mni152nlin2009casym",
        )
