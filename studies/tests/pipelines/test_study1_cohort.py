"""Tests for studies.pain_study.study1.cohort path resolution and subject filtering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from studies.tests.test_support import DotConfig


def _config(deriv_root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": str(deriv_root)},
            "study1": {
                "outputs": {"root_name": "study1"},
                "cohort": {"min_subjects": 2},
                "targets": {"names": ["NPS", "SIIPS1"]},
            },
        }
    )


def _write_primary_targets(
    config: DotConfig,
    subjects: list[str] | None = None,
    task: str = "pain",
) -> Path:
    if subjects is None:
        subjects = ["sub-0001", "sub-0002", "sub-0003"]

    rows: list[dict] = []
    for subject_id in subjects:
        rows.append(
            {
                "subject_id": subject_id,
                "task": task,
                "run": 1,
                "trial_index": 1,
                "within_run_trial": 1,
                "onset": 10.0,
                "duration": 1.0,
                "NPS": 1.0,
                "SIIPS1": 2.0,
            }
        )

    from studies.pain_study.study1.cohort import primary_targets_parquet_path

    target_path = primary_targets_parquet_path(config)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target_path, index=False)
    return target_path


###################################################################
# Path helpers
###################################################################


def test_study1_output_root_uses_root_name(tmp_path) -> None:
    from studies.pain_study.study1.cohort import study1_output_root

    cfg = _config(tmp_path / "derivatives")
    assert study1_output_root(cfg) == tmp_path / "derivatives" / "group" / "multimodal" / "study1"


def test_study1_output_root_rejects_empty_root_name(tmp_path) -> None:
    from studies.pain_study.study1.cohort import study1_output_root

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["outputs"]["root_name"] = "   "

    with pytest.raises(ValueError, match="non-empty"):
        study1_output_root(cfg)


def test_study1_feature_table_path_structure(tmp_path) -> None:
    from studies.pain_study.study1.cohort import study1_feature_table_path

    cfg = _config(tmp_path / "derivatives")
    path = study1_feature_table_path(cfg, "sub-0001", "power")

    assert path.name == "features_power.parquet"
    assert "eeg" in path.parts
    assert "features" in path.parts


def test_study1_feature_metadata_path_structure(tmp_path) -> None:
    from studies.pain_study.study1.cohort import study1_feature_metadata_path

    cfg = _config(tmp_path / "derivatives")
    path = study1_feature_metadata_path(cfg, "sub-0001", "spectral")

    assert path.name == "extraction_config.json"
    assert "metadata" in path.parts


def test_study1_feature_family_dir_rejects_empty_family(tmp_path) -> None:
    from studies.pain_study.study1.cohort import study1_feature_family_dir

    cfg = _config(tmp_path / "derivatives")

    with pytest.raises(ValueError, match="non-empty"):
        study1_feature_family_dir(cfg, "sub-0001", "   ")


###################################################################
# Subject normalization
###################################################################


def test_normalize_subject_adds_prefix(tmp_path) -> None:
    from studies.pain_study.study1.cohort import _normalize_subject

    assert _normalize_subject("0001") == "sub-0001"
    assert _normalize_subject("sub-0001") == "sub-0001"


def test_normalize_subject_rejects_empty_string(tmp_path) -> None:
    from studies.pain_study.study1.cohort import _normalize_subject

    with pytest.raises(ValueError, match="non-empty"):
        _normalize_subject("   ")


###################################################################
# resolve_primary_subjects
###################################################################


def test_resolve_primary_subjects_returns_sorted_available(tmp_path) -> None:
    from studies.pain_study.study1.cohort import resolve_primary_subjects

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg, subjects=["sub-0003", "sub-0001", "sub-0002"])

    resolved = resolve_primary_subjects(requested_subjects=[], task="pain", config=cfg)

    assert resolved == ["sub-0001", "sub-0002", "sub-0003"]


def test_resolve_primary_subjects_filters_to_requested(tmp_path) -> None:
    from studies.pain_study.study1.cohort import resolve_primary_subjects

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg, subjects=["sub-0001", "sub-0002", "sub-0003"])

    resolved = resolve_primary_subjects(
        requested_subjects=["0001", "0003"], task="pain", config=cfg
    )

    assert resolved == ["sub-0001", "sub-0003"]


def test_resolve_primary_subjects_rejects_missing_requested(tmp_path) -> None:
    from studies.pain_study.study1.cohort import resolve_primary_subjects

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg, subjects=["sub-0001", "sub-0002"])

    with pytest.raises(ValueError, match="missing"):
        resolve_primary_subjects(requested_subjects=["0001", "9999"], task="pain", config=cfg)


def test_resolve_primary_subjects_enforces_min_subjects(tmp_path) -> None:
    from studies.pain_study.study1.cohort import resolve_primary_subjects

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["cohort"]["min_subjects"] = 5
    _write_primary_targets(cfg, subjects=["sub-0001", "sub-0002"])

    with pytest.raises(ValueError, match="at least 5 subjects"):
        resolve_primary_subjects(requested_subjects=[], task="pain", config=cfg)


def test_resolve_primary_subjects_rejects_wrong_task(tmp_path) -> None:
    from studies.pain_study.study1.cohort import resolve_primary_subjects

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg, task="pain")

    with pytest.raises(ValueError, match="no rows"):
        resolve_primary_subjects(requested_subjects=[], task="rest", config=cfg)


###################################################################
# subject_target_rows
###################################################################


def test_subject_target_rows_returns_subject_specific_rows(tmp_path) -> None:
    from studies.pain_study.study1.cohort import subject_target_rows

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg, subjects=["sub-0001", "sub-0002"])

    rows = subject_target_rows(subject_id="0001", task="pain", config=cfg)

    assert len(rows) == 1
    assert rows["subject_id"].iloc[0] == "sub-0001"


def test_subject_target_rows_rejects_missing_subject(tmp_path) -> None:
    from studies.pain_study.study1.cohort import subject_target_rows

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg, subjects=["sub-0001"])

    with pytest.raises(ValueError, match="no rows"):
        subject_target_rows(subject_id="9999", task="pain", config=cfg)


###################################################################
# resolve_primary_target_name
###################################################################


def test_resolve_primary_target_name_accepts_valid(tmp_path) -> None:
    from studies.pain_study.study1.cohort import resolve_primary_target_name

    cfg = _config(tmp_path / "derivatives")

    assert resolve_primary_target_name(cfg, "NPS") == "NPS"
    assert resolve_primary_target_name(cfg, "SIIPS1") == "SIIPS1"


def test_resolve_primary_target_name_rejects_unknown(tmp_path) -> None:
    from studies.pain_study.study1.cohort import resolve_primary_target_name

    cfg = _config(tmp_path / "derivatives")

    with pytest.raises(ValueError, match="Invalid"):
        resolve_primary_target_name(cfg, "PINES")


###################################################################
# load_primary_target_table
###################################################################


def test_load_primary_target_table_rejects_missing_file(tmp_path) -> None:
    from studies.pain_study.study1.cohort import load_primary_target_table

    cfg = _config(tmp_path / "derivatives")

    with pytest.raises(FileNotFoundError, match="primary_targets"):
        load_primary_target_table(cfg)


def test_load_primary_target_table_rejects_missing_columns(tmp_path) -> None:
    from studies.pain_study.study1.cohort import (
        load_primary_target_table,
        primary_targets_parquet_path,
    )

    cfg = _config(tmp_path / "derivatives")
    target_path = primary_targets_parquet_path(cfg)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_id": ["sub-0001"], "task": ["pain"]}).to_parquet(
        target_path, index=False
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_primary_target_table(cfg)


def test_load_primary_target_table_rejects_empty_table(tmp_path) -> None:
    from studies.pain_study.study1.cohort import (
        load_primary_target_table,
        primary_targets_parquet_path,
    )

    cfg = _config(tmp_path / "derivatives")
    target_path = primary_targets_parquet_path(cfg)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            col: pd.Series(dtype=str)
            for col in (
                "subject_id",
                "task",
                "run",
                "trial_index",
                "within_run_trial",
                "onset",
                "duration",
                "NPS",
                "SIIPS1",
            )
        }
    ).to_parquet(target_path, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_primary_target_table(cfg)
