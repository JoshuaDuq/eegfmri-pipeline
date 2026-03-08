from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fmri_pipeline.analysis.second_level import (
    FirstLevelMapRecord,
    SecondLevelConfig,
    prepare_second_level_input,
)


def _make_record(tmp_path: Path, subject: str, contrast_name: str) -> FirstLevelMapRecord:
    map_path = tmp_path / f"sub-{subject}_{contrast_name}.nii.gz"
    sidecar_path = tmp_path / f"sub-{subject}_{contrast_name}.json"
    map_path.write_text("nii", encoding="utf-8")
    sidecar_path.write_text("{}", encoding="utf-8")
    return FirstLevelMapRecord(
        subject=subject,
        subject_label=f"sub-{subject}",
        contrast_name=contrast_name,
        map_path=map_path,
        sidecar_path=sidecar_path,
        contrast_cfg={"fmriprep_space": "MNI152NLin2009cAsym"},
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
    }
    covariates_path = tmp_path / "group.tsv"
    pd.DataFrame(
        {
            "subject": ["0001", "0002"],
            "group": ["control", "patient"],
            "age": [20, 24],
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
            subjects=["0001", "0002"],
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
