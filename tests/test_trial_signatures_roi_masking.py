import tempfile


def test_roi_mask_intersects_brain_mask():
    import numpy as np
    import nibabel as nib

    from fmri_pipeline.analysis.trial_signatures import (
        _build_roi_masks_from_atlas,
        _intersect_masks_to_target,
        _mask_hash_and_count,
    )

    affine = np.eye(4)
    atlas_data = np.ones((2, 2, 2), dtype=np.int16)  # label=1 everywhere
    brain_data = np.zeros((2, 2, 2), dtype=np.uint8)
    brain_data[0, 0, 0] = 1
    brain_data[1, 1, 1] = 1

    atlas_img = nib.Nifti1Image(atlas_data, affine)
    brain_img = nib.Nifti1Image(brain_data, affine)
    target_img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine)

    with tempfile.TemporaryDirectory() as td:
        atlas_path = f"{td}/atlas.nii.gz"
        nib.save(atlas_img, atlas_path)
        roi_masks = _build_roi_masks_from_atlas(atlas_path=atlas_path, labels=[(1, "ROI1")], target_img=target_img)

    roi_mask = roi_masks["ROI1"]
    _, n_roi = _mask_hash_and_count(roi_mask)
    assert n_roi == int(np.sum(atlas_data > 0))

    inter = _intersect_masks_to_target(roi_mask_img=roi_mask, brain_mask_img=brain_img, target_img=target_img)
    _, n_inter = _mask_hash_and_count(inter)
    assert n_inter == int(np.sum(brain_data > 0))
    assert n_inter <= n_roi

