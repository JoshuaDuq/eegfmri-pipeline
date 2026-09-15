from __future__ import annotations

import numpy as np
import pytest

from studies.pain_study.study1.signature_manifest import (
    build_signature_manifest,
)


def _signature_specs():
    return [
        {
            "name": name,
            "path": path,
            "space": "MNI152NLin2009cAsym",
            "source_space": "MNI152NLin2009cAsym",
            "spatial_reference": "Synthetic image created directly in the test reference grid",
            "source_publication": "test fixture",
            "source_repository_or_access_record": "test fixture",
        }
        for name, path in (
            ("NPS", "NPS/weights_NSF_grouppred_cvpcr.nii.gz"),
            ("SIIPS1", "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"),
        )
    ]


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
        signature_specs=_signature_specs(),
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
            signature_specs=_signature_specs(),
        )

@pytest.mark.parametrize("field", ["space", "source_space", "spatial_reference"])
def test_manifest_rejects_missing_spatial_provenance(tmp_path, field):
    nib = pytest.importorskip("nibabel")
    spec = _signature_specs()[0]
    path = tmp_path / spec["path"]
    path.parent.mkdir()
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), path)
    del spec[field]
    with pytest.raises(ValueError, match=field):
        build_signature_manifest(signature_root=tmp_path, signature_specs=[spec])


def test_manifest_rejects_space_change_without_transform(tmp_path):
    nib = pytest.importorskip("nibabel")
    spec = _signature_specs()[0]
    spec["source_space"] = "original_spm_template"
    path = tmp_path / spec["path"]
    path.parent.mkdir()
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), path)
    with pytest.raises(ValueError, match="transform"):
        build_signature_manifest(signature_root=tmp_path, signature_specs=[spec])


def test_original_siips_checksum_cannot_be_declared_2009c(tmp_path, monkeypatch):
    from studies.pain_study.study1 import targets

    cfg_entry = _signature_specs()[1]
    checksum = "da9992717887ed3ec038d3f87887a7f3d061384f382855dc717e6fafbc70198c"
    cfg_entry["sha256"] = checksum
    monkeypatch.setattr(targets, "_sha256", lambda path: checksum)
    with pytest.raises(ValueError, match="unmodified published SIIPS1"):
        targets._validate_manifest_entry(
            name="SIIPS1", entry=cfg_entry, configured_path=cfg_entry["path"],
            configured_space="mni152nlin2009casym", image_path=tmp_path / "original.nii.gz",
        )
