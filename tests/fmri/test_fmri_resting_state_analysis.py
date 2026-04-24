from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.resting_state import (
    RestingStateAnalysisConfig,
    _aggregate_connectivity_matrices,
    _discover_rest_runs,
    _prepare_confounds_and_sample_mask,
    _validate_filter_settings,
    _validate_roi_timeseries,
    run_resting_state_analysis_for_subject,
)


def test_prepare_confounds_and_sample_mask_builds_sample_mask_and_drops_scrub_columns() -> None:
    confounds_df = pd.DataFrame(
        {
            "trans_x": [0.1, 0.2, 0.3],
            "white_matter": [1.0, 1.1, 1.2],
            "motion_outlier00": [0, 1, 0],
            "non_steady_state_outlier00": [0, 0, 0],
        }
    )

    cleaned_confounds, sample_mask, scrub_columns = _prepare_confounds_and_sample_mask(
        confounds_df
    )

    assert cleaned_confounds is not None
    assert list(cleaned_confounds.columns) == ["trans_x", "white_matter"]
    assert scrub_columns == ["motion_outlier00", "non_steady_state_outlier00"]
    np.testing.assert_array_equal(sample_mask, np.array([0, 2], dtype=int))


def test_prepare_confounds_and_sample_mask_returns_none_when_only_scrub_columns_remain() -> None:
    confounds_df = pd.DataFrame(
        {
            "motion_outlier00": [0, 1, 0],
            "outlier01": [0, 0, 0],
        }
    )

    cleaned_confounds, sample_mask, scrub_columns = _prepare_confounds_and_sample_mask(
        confounds_df
    )

    assert cleaned_confounds is None
    assert scrub_columns == ["motion_outlier00", "outlier01"]
    np.testing.assert_array_equal(sample_mask, np.array([0, 2], dtype=int))


def test_prepare_confounds_and_sample_mask_rejects_nan_scrub_flags() -> None:
    confounds_df = pd.DataFrame(
        {
            "trans_x": [0.1, 0.2, 0.3],
            "motion_outlier00": [0, np.nan, 0],
        }
    )

    with pytest.raises(ValueError, match="scrub"):
        _prepare_confounds_and_sample_mask(confounds_df)


def test_validate_filter_settings_rejects_frequencies_at_or_above_nyquist() -> None:
    cfg = RestingStateAnalysisConfig(
        atlas_labels_img="/tmp/atlas.nii.gz",
        high_pass_hz=0.25,
        low_pass_hz=0.3,
    )

    with pytest.raises(ValueError, match="Nyquist"):
        _validate_filter_settings(2.0, cfg)


def test_resting_state_config_rejects_raw_bids_input_source() -> None:
    cfg = RestingStateAnalysisConfig(
        input_source="bids_raw",
        require_fmriprep=False,
        atlas_labels_img="/tmp/atlas.nii.gz",
    )

    with pytest.raises(ValueError, match="fMRIPrep derivatives are required"):
        cfg.normalized()


def test_validate_roi_timeseries_rejects_degenerate_roi_columns() -> None:
    timeseries = np.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="Degenerate ROI timeseries"):
        _validate_roi_timeseries(
            timeseries,
            ["roi_a", "roi_b"],
            subject="sub-0001",
            run_num=1,
        )


def test_aggregate_connectivity_matrices_uses_n_minus_3_fisher_weights() -> None:
    run_1 = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=float)
    run_2 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=float)

    connectivity, fisher_z = _aggregate_connectivity_matrices([run_1, run_2], [4, 6])

    expected_fisher = (math.atanh(0.1) + 3.0 * math.atanh(0.5)) / 4.0
    assert np.isclose(fisher_z[0, 1], expected_fisher)
    assert np.isclose(fisher_z[1, 0], expected_fisher)
    assert np.isclose(connectivity[0, 1], math.tanh(expected_fisher))
    assert np.isclose(connectivity[1, 0], math.tanh(expected_fisher))
    assert np.allclose(np.diag(connectivity), 1.0)
    assert np.allclose(np.diag(fisher_z), 0.0)


def test_aggregate_connectivity_matrices_rejects_multi_run_inputs_with_too_few_frames() -> None:
    run_1 = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=float)
    run_2 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="at least 4 retained frames"):
        _aggregate_connectivity_matrices([run_1, run_2], [3, 6])


def test_run_resting_state_analysis_uses_matching_brain_mask_for_roi_extraction(
    tmp_path: Path,
) -> None:
    bold_path = (
        tmp_path
        / "sub-0001_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    bold_path.write_bytes(b"")
    mask_path = (
        tmp_path
        / "sub-0001_task-rest_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )
    mask_path.write_bytes(b"")
    atlas_path = tmp_path / "atlas.nii.gz"
    atlas_path.write_bytes(b"")

    captured: dict[str, object] = {}

    class FakeMasker:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def fit_transform(self, *_args, **_kwargs) -> np.ndarray:
            return np.array(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0],
                ],
                dtype=float,
            )

    fake_maskers = types.ModuleType("nilearn.maskers")
    fake_maskers.NiftiLabelsMasker = FakeMasker

    cfg = RestingStateAnalysisConfig(atlas_labels_img=atlas_path)

    with patch.dict(sys.modules, {"nilearn.maskers": fake_maskers}):
        with patch(
            "fmri_pipeline.analysis.resting_state._discover_rest_runs",
            return_value=[(bold_path, 1)],
        ), patch(
            "fmri_pipeline.analysis.resting_state._load_rest_confounds",
            return_value=(
                pd.DataFrame({"trans_x": [0.0, 0.0, 0.0, 0.0]}),
                ["trans_x"],
            ),
        ), patch(
            "fmri_pipeline.analysis.resting_state.get_tr_from_bold",
            return_value=2.0,
        ):
            run_resting_state_analysis_for_subject(
                bids_fmri_root=tmp_path,
                bids_derivatives=tmp_path,
                deriv_root=tmp_path,
                subject="0001",
                task="rest",
                cfg=cfg,
                output_dir=tmp_path / "out",
            )

    assert captured["kwargs"]["mask_img"] == str(mask_path)


def test_discover_rest_runs_accepts_single_runless_bold_file(tmp_path: Path) -> None:
    func_dir = tmp_path / "sub-0001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    bold_path = func_dir / "sub-0001_task-rest_bold.nii.gz"
    bold_path.write_bytes(b"")

    cfg = RestingStateAnalysisConfig(
        input_source="bids_raw",
        require_fmriprep=False,
        atlas_labels_img="/tmp/atlas.nii.gz",
    )

    discovered = _discover_rest_runs(
        bids_fmri_root=tmp_path,
        bids_derivatives=None,
        subject="0001",
        task="rest",
        cfg=cfg,
    )

    assert discovered == [(bold_path, 1)]


def test_discover_rest_runs_rejects_multiple_runless_bold_files(tmp_path: Path) -> None:
    func_dir = tmp_path / "sub-0001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    (func_dir / "sub-0001_task-rest_acq-a_bold.nii.gz").write_bytes(b"")
    (func_dir / "sub-0001_task-rest_acq-b_bold.nii.gz").write_bytes(b"")

    cfg = RestingStateAnalysisConfig(
        input_source="bids_raw",
        require_fmriprep=False,
        atlas_labels_img="/tmp/atlas.nii.gz",
    )

    with pytest.raises(FileNotFoundError, match="without explicit run entities"):
        _discover_rest_runs(
            bids_fmri_root=tmp_path,
            bids_derivatives=None,
            subject="0001",
            task="rest",
            cfg=cfg,
        )
