from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fmri_pipeline.analysis.second_level import (
    FirstLevelMapRecord,
    SecondLevelConfig,
    prepare_second_level_input,
)


def _make_record(
    tmp_path: Path,
    subject: str,
    contrast_name: str,
    *,
    contrast_cfg: dict[str, object] | None = None,
    run_meta: dict[str, object] | None = None,
) -> FirstLevelMapRecord:
    map_path = tmp_path / f"sub-{subject}_{contrast_name}.nii.gz"
    sidecar_path = tmp_path / f"sub-{subject}_{contrast_name}.json"
    map_path.write_text("nii", encoding="utf-8")
    sidecar_path.write_text("{}", encoding="utf-8")
    default_run_meta = {
        "discovered_bold_paths": [
            str(tmp_path / f"sub-{subject}_task-pain_run-01_desc-preproc_bold.nii.gz"),
            str(tmp_path / f"sub-{subject}_task-pain_run-02_desc-preproc_bold.nii.gz"),
        ],
        "included_bold_paths": [
            str(tmp_path / f"sub-{subject}_task-pain_run-01_desc-preproc_bold.nii.gz"),
            str(tmp_path / f"sub-{subject}_task-pain_run-02_desc-preproc_bold.nii.gz"),
        ],
    }
    return FirstLevelMapRecord(
        subject=subject,
        subject_label=f"sub-{subject}",
        contrast_name=contrast_name,
        map_path=map_path,
        sidecar_path=sidecar_path,
        contrast_cfg=contrast_cfg or {"fmriprep_space": "MNI152NLin2009cAsym"},
        run_meta=run_meta or default_run_meta,
    )


def test_prepare_second_level_one_sample_builds_intercept_design(tmp_path: Path) -> None:
    records = {
        ("0001", "pain"): _make_record(tmp_path, "0001", "pain"),
        ("0002", "pain"): _make_record(tmp_path, "0002", "pain"),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(model="one-sample", contrast_names=("pain",)).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
        side_effect=_discover,
    ), patch("fmri_pipeline.analysis.second_level._validate_same_grid"):
        prepared = prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert list(prepared.design_matrix.columns) == ["intercept"]
    assert prepared.contrast_spec == "intercept"
    assert prepared.output_name == "pain_group_mean"
    assert len(prepared.image_paths) == 2


def test_prepare_second_level_two_sample_uses_group_columns_and_covariates(
    tmp_path: Path,
) -> None:
    records = {
        ("0001", "pain"): _make_record(tmp_path, "0001", "pain"),
        ("0002", "pain"): _make_record(tmp_path, "0002", "pain"),
        ("0003", "pain"): _make_record(tmp_path, "0003", "pain"),
        ("0004", "pain"): _make_record(tmp_path, "0004", "pain"),
    }
    covariates_path = tmp_path / "group.tsv"
    pd.DataFrame(
        {
            "subject": ["sub-0001", "sub-0002", "sub-0003", "sub-0004"],
            "group": ["control", "patient", "control", "patient"],
            "age": [20, 24, 26, 30],
        }
    ).to_csv(covariates_path, sep="\t", index=False)

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="two-sample",
        contrast_names=("pain",),
        covariates_file=str(covariates_path),
        subject_column="subject",
        covariate_columns=("age",),
        group_column="group",
        group_a_value="control",
        group_b_value="patient",
    ).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
        side_effect=_discover,
    ), patch("fmri_pipeline.analysis.second_level._validate_same_grid"):
        prepared = prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0002", "0003", "0004"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert list(prepared.design_matrix.columns) == [
        "group_control",
        "group_patient",
        "cov_age",
    ]
    assert prepared.contrast_spec == "group_patient - group_control"
    assert prepared.metadata["group_design_columns"]["group_a"] == "group_control"
    assert prepared.metadata["group_design_columns"]["group_b"] == "group_patient"


def test_prepare_second_level_repeated_measures_defaults_to_omnibus_f(
    tmp_path: Path,
) -> None:
    records = {
        ("0001", "low"): _make_record(tmp_path, "0001", "low"),
        ("0002", "low"): _make_record(tmp_path, "0002", "low"),
        ("0001", "med"): _make_record(tmp_path, "0001", "med"),
        ("0002", "med"): _make_record(tmp_path, "0002", "med"),
        ("0001", "high"): _make_record(tmp_path, "0001", "high"),
        ("0002", "high"): _make_record(tmp_path, "0002", "high"),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="repeated-measures",
        contrast_names=("low", "med", "high"),
    ).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
        side_effect=_discover,
    ), patch("fmri_pipeline.analysis.second_level._validate_same_grid"):
        prepared = prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
        )

    assert prepared.stat_type == "F"
    assert prepared.contrast_spec.shape == (2, len(prepared.design_matrix.columns))
    assert list(prepared.design_matrix.columns) == [
        "condition_low",
        "condition_med",
        "condition_high",
        "subject_0002",
    ]
    np.testing.assert_allclose(
        prepared.contrast_spec,
        np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
            ]
        ),
    )


def test_prepare_second_level_rejects_cross_contrast_model_mismatch(
    tmp_path: Path,
) -> None:
    low_cfg = {
        "fmriprep_space": "MNI152NLin2009cAsym",
        "hrf_model": "spm",
        "high_pass_hz": 0.008,
    }
    high_cfg = {
        "fmriprep_space": "MNI152NLin2009cAsym",
        "hrf_model": "spm",
        "high_pass_hz": 0.01,
    }
    records = {
        ("0001", "low"): _make_record(tmp_path, "0001", "low", contrast_cfg=low_cfg),
        ("0002", "low"): _make_record(tmp_path, "0002", "low", contrast_cfg=low_cfg),
        ("0001", "high"): _make_record(tmp_path, "0001", "high", contrast_cfg=high_cfg),
        ("0002", "high"): _make_record(tmp_path, "0002", "high", contrast_cfg=high_cfg),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="paired",
        contrast_names=("low", "high"),
    ).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
        side_effect=_discover,
    ), patch("fmri_pipeline.analysis.second_level._validate_same_grid"):
        with pytest.raises(ValueError, match="same first-level model settings"):
            prepare_second_level_input(
                config=cfg,
                subjects=["0001", "0002"],
                task="pain",
                deriv_root=tmp_path,
            )


def test_prepare_second_level_rejects_rank_deficient_design(
    tmp_path: Path,
) -> None:
    records = {
        ("0001", "pain"): _make_record(tmp_path, "0001", "pain"),
        ("0002", "pain"): _make_record(tmp_path, "0002", "pain"),
        ("0003", "pain"): _make_record(tmp_path, "0003", "pain"),
    }
    covariates_path = tmp_path / "group.tsv"
    pd.DataFrame(
        {
            "subject": ["sub-0001", "sub-0002", "sub-0003"],
            "age": [20, 24, 28],
            "age_copy": [30, 34, 38],
        }
    ).to_csv(covariates_path, sep="\t", index=False)

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(
        model="one-sample",
        contrast_names=("pain",),
        covariates_file=str(covariates_path),
        subject_column="subject",
        covariate_columns=("age", "age_copy"),
    ).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
        side_effect=_discover,
    ), patch("fmri_pipeline.analysis.second_level._validate_same_grid"):
        with pytest.raises(ValueError, match="rank-deficient"):
            prepare_second_level_input(
                config=cfg,
                subjects=["0001", "0002", "0003"],
                task="pain",
                deriv_root=tmp_path,
            )


def test_prepare_second_level_rejects_duplicate_subjects(tmp_path: Path) -> None:
    cfg = SecondLevelConfig(model="one-sample", contrast_names=("pain",)).normalized()

    with pytest.raises(ValueError, match="requires unique subjects"):
        prepare_second_level_input(
            config=cfg,
            subjects=["0001", "0001"],
            task="pain",
            deriv_root=tmp_path,
        )


def test_prepare_second_level_rejects_inconsistent_run_inclusion(tmp_path: Path) -> None:
    records = {
        ("0001", "pain"): _make_record(
            tmp_path,
            "0001",
            "pain",
            run_meta={
                "discovered_bold_paths": [
                    str(tmp_path / "sub-0001_task-pain_run-01_desc-preproc_bold.nii.gz"),
                    str(tmp_path / "sub-0001_task-pain_run-02_desc-preproc_bold.nii.gz"),
                ],
                "included_bold_paths": [
                    str(tmp_path / "sub-0001_task-pain_run-01_desc-preproc_bold.nii.gz"),
                    str(tmp_path / "sub-0001_task-pain_run-02_desc-preproc_bold.nii.gz"),
                ],
            },
        ),
        ("0002", "pain"): _make_record(
            tmp_path,
            "0002",
            "pain",
            run_meta={
                "discovered_bold_paths": [
                    str(tmp_path / "sub-0002_task-pain_run-01_desc-preproc_bold.nii.gz"),
                    str(tmp_path / "sub-0002_task-pain_run-02_desc-preproc_bold.nii.gz"),
                ],
                "included_bold_paths": [
                    str(tmp_path / "sub-0002_task-pain_run-01_desc-preproc_bold.nii.gz"),
                ],
            },
        ),
    }

    def _discover(*, subject: str, contrast_name: str, **_kwargs):
        return records[(subject, contrast_name)]

    cfg = SecondLevelConfig(model="one-sample", contrast_names=("pain",)).normalized()

    with patch(
        "fmri_pipeline.analysis.second_level._discover_first_level_effect_size_map",
        side_effect=_discover,
    ), patch("fmri_pipeline.analysis.second_level._validate_same_grid"):
        with pytest.raises(ValueError, match="same discovered and included runs"):
            prepare_second_level_input(
                config=cfg,
                subjects=["0001", "0002"],
                task="pain",
                deriv_root=tmp_path,
            )
