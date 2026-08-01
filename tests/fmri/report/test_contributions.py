"""What each run contributes to a contrast, as measurements rather than a picture.

Six runs and four numbers each is a table. Drawn as a chart it would be a table with
worse precision and more ink, so these are tested as values.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.report import contributions


def _run_maps(per_run_means, shape=(4, 4, 4)):
    data = np.stack(
        [np.full(shape, value, dtype=np.float32) for value in per_run_means], axis=-1
    )
    return nib.Nifti1Image(data, np.eye(4))


def _mask(shape=(4, 4, 4), keep=None):
    data = np.zeros(shape, dtype=np.uint8)
    if keep is None:
        data[:] = 1
    else:
        data[keep] = 1
    return nib.Nifti1Image(data, np.eye(4))


def test_offsets_are_the_whole_mask_mean_per_run():
    offsets = contributions.run_offsets(
        _run_maps([-0.083, 0.057, -0.037]), _mask()
    )
    np.testing.assert_allclose(offsets, [-0.083, 0.057, -0.037], atol=1e-6)


def test_only_mask_voxels_count():
    """Background dominates the volume, and dilutes a real offset toward zero."""
    shape = (4, 4, 4)
    data = np.zeros(shape + (1,), dtype=np.float32)
    data[:2, :, :, 0] = 0.5
    offsets = contributions.run_offsets(
        nib.Nifti1Image(data, np.eye(4)), _mask(shape, keep=(slice(0, 2),))
    )
    np.testing.assert_allclose(offsets, [0.5], atol=1e-6)


def test_a_mismatched_mask_is_refused_rather_than_broadcast():
    """A mask from another grid would silently select the wrong voxels."""
    offsets = contributions.run_offsets(_run_maps([0.5]), _mask((8, 8, 8)))
    # Falls back to non-zero voxels, which for this fixture is every voxel.
    np.testing.assert_allclose(offsets, [0.5], atol=1e-6)


def test_offsets_refuse_a_three_dimensional_map():
    assert (
        contributions.run_offsets(
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
        )
        is None
    )


def test_influence_is_read_by_run_label(tmp_path):
    path = tmp_path / "influence.tsv"
    pd.DataFrame(
        {
            "dropped_run": ["run-01", "run-02"],
            "survivors": [7108, 8003],
            "delta": [-1367, -472],
            "max_abs_z": [8.18, 7.68],
            "correlation": [0.941, 0.913],
        }
    ).to_csv(path, sep="\t", index=False)

    influence = contributions.read_run_influence(path)
    assert influence["run-02"]["delta"] == -472


def test_missing_influence_table_is_not_an_error():
    assert contributions.read_run_influence(None) == {}


def test_rows_key_on_the_run_label_not_the_position():
    """A table missing one run must empty that cell, not shift the rest up by one."""
    rows = contributions.contribution_rows(
        run_labels=("run-01", "run-02", "run-03"),
        offsets=[-0.08, 0.06, -0.04],
        influence={
            "run-01": {"survivors": 10, "delta": -5, "correlation": 0.9},
            "run-03": {"survivors": 30, "delta": 7, "correlation": 0.8},
        },
    )
    assert rows[1]["Run"] == "run-02"
    assert "Change" not in rows[1]
    assert rows[2]["Change"] == 7


def test_rows_carry_offsets_alone_when_no_influence_exists():
    rows = contributions.contribution_rows(
        run_labels=("run-01",), offsets=[-0.08], influence=None
    )
    assert rows[0]["Mean effect over mask"] == pytest.approx(-0.08)
    assert "Change" not in rows[0]


def test_the_measured_offsets_of_this_study_are_recovered():
    """Five runs negative and one positive is what centres the fitted null at -0.55."""
    offsets = contributions.run_offsets(
        _run_maps([-0.0834, -0.0124, 0.0570, -0.0422, -0.0676, -0.0367]), _mask()
    )
    assert sum(1 for value in offsets if value > 0) == 1
    assert float(np.mean(offsets)) < 0
