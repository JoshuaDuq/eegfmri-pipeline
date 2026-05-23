from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fmri_pipeline.analysis.multivariate_signatures import (
    _maybe_resample_to_img,
    compute_signature_expression,
    discover_signature_files,
)
from fmri_pipeline.utils.signature_paths import (
    discover_signature_root,
    get_signature_specs,
)


class _BadConfig:
    def get(self, *_args, **_kwargs):
        raise RuntimeError("bad config")


def test_discover_signature_root_prefers_existing_config_path(tmp_path: Path) -> None:
    configured = tmp_path / "configured_signatures"
    configured.mkdir(parents=True, exist_ok=True)
    config = {"paths": {"signature_dir": str(configured)}}

    discovered = discover_signature_root(config, tmp_path / "derivatives")
    assert discovered == configured


def test_discover_signature_root_rejects_missing_configured_path(tmp_path: Path) -> None:
    config = {"paths": {"signature_dir": str(tmp_path / "missing_signatures")}}

    with pytest.raises(FileNotFoundError, match="Configured paths.signature_dir does not exist"):
        discover_signature_root(config, tmp_path / "derivatives")


def test_discover_signature_root_returns_none_when_no_signatures_are_configured(
    tmp_path: Path,
) -> None:
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)

    discovered = discover_signature_root({}, deriv_root)
    assert discovered is None


def test_discover_signature_root_requires_configured_root_for_signature_maps(
    tmp_path: Path,
) -> None:
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)
    config = {"paths": {"signature_maps": [{"name": "NPS", "path": "nps.nii.gz"}]}}

    with pytest.raises(ValueError, match="paths.signature_dir must be set"):
        discover_signature_root(config, deriv_root)


def test_discover_signature_root_surfaces_invalid_config_getter() -> None:
    with pytest.raises(RuntimeError, match="bad config"):
        discover_signature_root(_BadConfig(), Path("/tmp/derivatives"))


def test_get_signature_specs_rejects_duplicate_names() -> None:
    config = {
        "paths": {
            "signature_maps": [
                {"name": "NPS", "path": "nps.nii.gz"},
                {"name": "NPS", "path": "nps_copy.nii.gz"},
            ]
        }
    }

    with pytest.raises(ValueError, match="Duplicate signature name"):
        get_signature_specs(config)


def test_discover_signature_files_rejects_missing_weight_map(tmp_path: Path) -> None:
    root = tmp_path / "signatures"
    root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Signature weight map not found"):
        discover_signature_files(
            root,
            [{"name": "NPS", "path": "nps.nii.gz"}],
        )


def test_compute_signature_expression_rejects_missing_requested_signature(tmp_path: Path) -> None:
    root = tmp_path / "signatures"
    root.mkdir(parents=True, exist_ok=True)
    (root / "nps.nii.gz").write_bytes(b"fake")

    with pytest.raises(FileNotFoundError, match="Requested signatures were not found"):
        compute_signature_expression(
            stat_or_effect_img=object(),
            signature_root=root,
            signature_specs=[{"name": "NPS", "path": "nps.nii.gz"}],
            signatures=["SIIPS1"],
        )


def test_resample_to_img_rejects_nonfinite_voxels_in_continuous_resampling() -> None:
    effect_data = np.ones((2, 2, 2), dtype=np.float32)
    effect_data[0, 0, 0] = np.nan
    effect_img = nib.Nifti1Image(effect_data, np.eye(4))

    target_affine = np.eye(4, dtype=float)
    target_affine[:3, 3] = 0.1
    target_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), target_affine)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match="continuous resampling does not support non-finite voxels",
        ):
            _maybe_resample_to_img(
                moving_img=effect_img,
                target_img=target_img,
                interpolation="continuous",
            )

    warning_messages = [str(w.message) for w in captured]
    assert not any(
        "NaNs or infinite values are present in the data passed to resample" in msg
        for msg in warning_messages
    )


def test_compute_signature_expression_raises_when_mask_data_cannot_be_read(tmp_path: Path) -> None:
    root = tmp_path / "signatures"
    root.mkdir(parents=True, exist_ok=True)
    weight_path = root / "nps.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), weight_path)

    effect_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))

    class BrokenMask:
        shape = (2, 2, 2)
        affine = np.eye(4)

        def get_fdata(self):
            raise RuntimeError("mask read failed")

    with pytest.raises(ValueError, match="Failed to compute signature expression"):
        compute_signature_expression(
            stat_or_effect_img=effect_img,
            signature_root=root,
            signature_specs=[{"name": "NPS", "path": "nps.nii.gz"}],
            mask_img=BrokenMask(),
        )


def test_compute_signature_expression_rejects_nonfinite_values_inside_fixed_mask(
    tmp_path: Path,
) -> None:
    root = tmp_path / "signatures"
    root.mkdir(parents=True, exist_ok=True)
    weight_path = root / "nps.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), weight_path)

    effect_data = np.ones((2, 2, 2), dtype=np.float32)
    effect_data[0, 0, 0] = np.nan
    effect_img = nib.Nifti1Image(effect_data, np.eye(4))

    with pytest.raises(ValueError, match="fixed signature mask"):
        compute_signature_expression(
            stat_or_effect_img=effect_img,
            signature_root=root,
            signature_specs=[{"name": "NPS", "path": "nps.nii.gz"}],
        )


def test_compute_signature_expression_resamples_masked_effect_with_nonfinite_background(
    tmp_path: Path,
) -> None:
    root = tmp_path / "signatures"
    root.mkdir(parents=True, exist_ok=True)
    weight_path = root / "nps.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)),
        weight_path,
    )

    effect_data = np.ones((2, 2, 2), dtype=np.float32)
    effect_data[0, 0, 0] = np.nan
    effect_img = nib.Nifti1Image(effect_data, np.eye(4))
    mask_data = np.ones((2, 2, 2), dtype=np.uint8)
    mask_data[0, 0, 0] = 0
    mask_img = nib.Nifti1Image(mask_data, np.eye(4))

    results = compute_signature_expression(
        stat_or_effect_img=effect_img,
        signature_root=root,
        signature_specs=[{"name": "NPS", "path": "nps.nii.gz"}],
        mask_img=mask_img,
    )

    assert results[0].name == "NPS"
    assert results[0].n_voxels > 0


def test_compute_signature_expression_records_scoring_mask_extent_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "signatures"
    root.mkdir(parents=True, exist_ok=True)
    weight_path = root / "nps.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), weight_path)

    effect_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    full_mask_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    partial_mask = np.ones((2, 2, 2), dtype=np.uint8)
    partial_mask[0, 0, 0] = 0
    partial_mask_img = nib.Nifti1Image(partial_mask, np.eye(4))

    full_result = compute_signature_expression(
        stat_or_effect_img=effect_img,
        signature_root=root,
        signature_specs=[{"name": "NPS", "path": "nps.nii.gz"}],
        mask_img=full_mask_img,
    )[0]
    partial_result = compute_signature_expression(
        stat_or_effect_img=effect_img,
        signature_root=root,
        signature_specs=[{"name": "NPS", "path": "nps.nii.gz"}],
        mask_img=partial_mask_img,
    )[0]

    assert full_result.scoring_mask_sha256
    assert partial_result.scoring_mask_sha256
    assert full_result.scoring_mask_sha256 != partial_result.scoring_mask_sha256


def test_compute_signature_expression_enforces_signature_support_thresholds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "signatures"
    root.mkdir(parents=True, exist_ok=True)
    weight_path = root / "nps.nii.gz"
    weights = np.ones((2, 2, 2), dtype=np.float32)
    weights[0, 0, 0] = -1.0
    nib.save(nib.Nifti1Image(weights, np.eye(4)), weight_path)

    effect_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 0
    mask_img = nib.Nifti1Image(mask, np.eye(4))

    with pytest.raises(ValueError, match="positive/negative signature support"):
        compute_signature_expression(
            stat_or_effect_img=effect_img,
            signature_root=root,
            signature_specs=[{"name": "NPS", "path": "nps.nii.gz"}],
            mask_img=mask_img,
            min_support_fraction=0.90,
            max_weight_mass_change_fraction=0.10,
        )
