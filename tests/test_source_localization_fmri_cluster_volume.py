from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_cfg(*, stats_map_path: Path, cluster_min_voxels: int, cluster_min_volume_mm3: float | None):
    from eeg_pipeline.analysis.features.source_localization import FMRIConstraintConfig

    return FMRIConstraintConfig(
        enabled=True,
        stats_map_path=stats_map_path,
        provenance="independent",
        require_provenance=False,
        threshold=1.0,
        tail="pos",
        threshold_mode="z",
        fdr_q=0.05,
        stat_type="z",
        cluster_min_voxels=int(cluster_min_voxels),
        cluster_min_volume_mm3=cluster_min_volume_mm3,
        max_clusters=20,
        max_voxels_per_cluster=0,
        max_total_voxels=0,
        random_seed=1,
        window_a=None,
        window_b=None,
    )


def test_fmri_cluster_min_volume_mm3_filters_clusters(tmp_path: Path) -> None:
    import nibabel as nib  # type: ignore

    from eeg_pipeline.analysis.features.source_localization import _fmri_roi_coords_from_stats_map

    # Build a simple z-map in 1mm isotropic voxels.
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[5:10, 5:10, 5:10] = 5.0  # 5x5x5 = 125 vox => 125 mm^3 at 1mm
    img = nib.Nifti1Image(data, affine=np.eye(4))
    p = tmp_path / "zmap.nii.gz"
    nib.save(img, str(p))

    # Require 400 mm^3 -> should reject this 125 mm^3 cluster.
    cfg = _make_cfg(stats_map_path=p, cluster_min_voxels=1, cluster_min_volume_mm3=400.0)
    with pytest.raises(ValueError, match="cluster_min_volume_mm3"):
        _fmri_roi_coords_from_stats_map(p, cfg, logger=None)

    # Require 100 mm^3 -> should keep the cluster.
    cfg2 = _make_cfg(stats_map_path=p, cluster_min_voxels=1, cluster_min_volume_mm3=100.0)
    coords = _fmri_roi_coords_from_stats_map(p, cfg2, logger=None)
    assert coords

