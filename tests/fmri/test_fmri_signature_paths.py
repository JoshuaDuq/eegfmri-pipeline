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


def test_discover_signature_root_falls_back_to_external_sibling(tmp_path: Path) -> None:
    deriv_root = tmp_path / "derivatives"
    deriv_root.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)

    discovered = discover_signature_root({}, deriv_root)
    assert discovered == external


def test_discover_signature_root_surfaces_invalid_config_getter() -> None:
    with pytest.raises(RuntimeError, match="bad config"):
        discover_signature_root(_BadConfig(), Path("/tmp/derivatives"))


def test_discover_signature_root_surfaces_invalid_derivative_root_type() -> None:
    with pytest.raises(TypeError):
        discover_signature_root({}, object())


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


def test_resample_to_img_ignores_nonfinite_voxels_in_moving_image() -> None:
    effect_data = np.ones((2, 2, 2), dtype=np.float32)
    effect_data[0, 0, 0] = np.nan
    effect_img = nib.Nifti1Image(effect_data, np.eye(4))

    target_affine = np.eye(4, dtype=float)
    target_affine[:3, 3] = 0.1
    target_img = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), target_affine)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        resampled = _maybe_resample_to_img(
            moving_img=effect_img,
            target_img=target_img,
            interpolation="continuous",
        )

    resampled_data = resampled.get_fdata()
    assert np.isfinite(resampled_data).all()
    warning_messages = [str(w.message) for w in captured]
    assert not any(
        "NaNs or infinite values are present in the data passed to resample" in msg
        for msg in warning_messages
    )
