from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from fmri_pipeline.utils.bold_discovery import (
    _parse_optional_positive_float_attr,
    build_first_level_model,
    coerce_condition_value,
    discover_fmriprep_preproc_bold,
    get_tr_from_bold,
    select_consistent_run_source,
    select_confound_columns,
    select_confounds,
    validate_design_matrices,
)


def test_discover_fmriprep_preproc_bold_accepts_zero_padded_and_non_padded_runs(
    tmp_path: Path,
) -> None:
    func_dir = tmp_path / "fmriprep" / "sub-0001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    bold_path = func_dir / "sub-0001_task-task_run-1_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")

    discovered = discover_fmriprep_preproc_bold(
        bids_derivatives=tmp_path,
        subject="0001",
        task="task",
        run_num=1,
        space=None,
    )
    assert discovered == bold_path


def test_discover_fmriprep_preproc_bold_does_not_match_runless_file_for_run(tmp_path: Path) -> None:
    func_dir = tmp_path / "fmriprep" / "sub-0001" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    runless_path = func_dir / "sub-0001_task-task_space-T1w_desc-preproc_bold.nii.gz"
    runless_path.write_bytes(b"")

    discovered = discover_fmriprep_preproc_bold(
        bids_derivatives=tmp_path,
        subject="0001",
        task="task",
        run_num=2,
        space="T1w",
    )

    assert discovered is None


def test_get_tr_from_bold_prefers_sidecar_repetition_time(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    sidecar.write_text(json.dumps({"RepetitionTime": "1.75"}), encoding="utf-8")

    class FakeHeader:
        @staticmethod
        def get_zooms() -> tuple[float, float, float, float]:
            return (2.0, 2.0, 2.0, 1.75)

    class FakeImage:
        header = FakeHeader()

    fake_nib = types.ModuleType("nibabel")
    fake_nib.load = lambda *_args, **_kwargs: FakeImage()

    with patch.dict(sys.modules, {"nibabel": fake_nib}):
        tr = get_tr_from_bold(bold_path)

    assert tr == 1.75


def test_get_tr_from_bold_raises_when_sidecar_is_not_mapping(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    sidecar.write_text("[1.75]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        get_tr_from_bold(bold_path)


def test_get_tr_from_bold_raises_when_sidecar_json_is_invalid(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    sidecar.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid BOLD sidecar JSON"):
        get_tr_from_bold(bold_path)


def test_get_tr_from_bold_raises_when_header_cannot_be_validated(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    sidecar.write_text(json.dumps({"RepetitionTime": 1.75}), encoding="utf-8")

    with pytest.raises(ValueError, match="Could not validate TR"):
        get_tr_from_bold(bold_path)


def test_get_tr_from_bold_raises_when_sidecar_and_header_disagree(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    sidecar.write_text(json.dumps({"RepetitionTime": 1.75}), encoding="utf-8")

    class FakeHeader:
        @staticmethod
        def get_zooms() -> tuple[float, float, float, float]:
            return (2.0, 2.0, 2.0, 2.5)

    class FakeImage:
        header = FakeHeader()

    fake_nib = types.ModuleType("nibabel")
    fake_nib.load = lambda *_args, **_kwargs: FakeImage()

    with patch.dict(sys.modules, {"nibabel": fake_nib}):
        with pytest.raises(ValueError, match="TR mismatch"):
            get_tr_from_bold(bold_path)


def test_build_first_level_model_coerces_and_filters_optional_float_settings() -> None:
    class FakeFirstLevelModel:
        def __init__(
            self,
            *,
            t_r: float,
            hrf_model: str,
            drift_model: str | None,
            high_pass: float | None,
            noise_model: str,
            standardize: bool,
            signal_scaling: int,
            minimize_memory: bool,
            low_pass: float | None = None,
            mask_img: str | None = None,
            smoothing_fwhm: float | None = None,
        ) -> None:
            self.params = {
                "t_r": t_r,
                "hrf_model": hrf_model,
                "drift_model": drift_model,
                "high_pass": high_pass,
                "noise_model": noise_model,
                "standardize": standardize,
                "signal_scaling": signal_scaling,
                "minimize_memory": minimize_memory,
                "low_pass": low_pass,
                "mask_img": mask_img,
                "smoothing_fwhm": smoothing_fwhm,
            }

    fake_first_level = types.ModuleType("nilearn.glm.first_level")
    fake_first_level.FirstLevelModel = FakeFirstLevelModel
    fake_glm = types.ModuleType("nilearn.glm")
    fake_glm.first_level = fake_first_level
    fake_nilearn = types.ModuleType("nilearn")
    fake_nilearn.glm = fake_glm

    cfg = SimpleNamespace(
        low_pass_hz="0.12",
        high_pass_hz="-0.5",
        hrf_model="spm",
        drift_model="cosine",
        smoothing_fwhm=4.0,
    )

    with patch.dict(
        sys.modules,
        {
            "nilearn": fake_nilearn,
            "nilearn.glm": fake_glm,
            "nilearn.glm.first_level": fake_first_level,
        },
    ):
        model = build_first_level_model(tr=2.0, cfg=cfg, mask_img="brain-mask")

    assert model.params["low_pass"] == 0.12
    assert model.params["high_pass"] is None
    assert model.params["mask_img"] == "brain-mask"
    assert model.params["standardize"] is False
    assert model.params["signal_scaling"] == 0


def test_build_first_level_model_rejects_unsupported_low_pass_setting() -> None:
    class FakeFirstLevelModel:
        def __init__(
            self,
            *,
            t_r: float,
            hrf_model: str,
            drift_model: str | None,
            high_pass: float | None,
            noise_model: str,
            standardize: bool,
            signal_scaling: int,
            minimize_memory: bool,
        ) -> None:
            self.params = {"t_r": t_r}

    fake_first_level = types.ModuleType("nilearn.glm.first_level")
    fake_first_level.FirstLevelModel = FakeFirstLevelModel
    fake_glm = types.ModuleType("nilearn.glm")
    fake_glm.first_level = fake_first_level
    fake_nilearn = types.ModuleType("nilearn")
    fake_nilearn.glm = fake_glm

    cfg = SimpleNamespace(
        low_pass_hz=0.12,
        high_pass_hz=0.008,
        hrf_model="spm",
        drift_model="cosine",
        smoothing_fwhm=None,
    )

    with patch.dict(
        sys.modules,
        {
            "nilearn": fake_nilearn,
            "nilearn.glm": fake_glm,
            "nilearn.glm.first_level": fake_first_level,
        },
    ):
        with pytest.raises(ValueError, match="low_pass_hz"):
            build_first_level_model(tr=2.0, cfg=cfg)


def test_parse_optional_positive_float_attr_surfaces_config_accessor_failures() -> None:
    class BadConfig:
        @property
        def low_pass_hz(self) -> float:
            raise RuntimeError("bad config")

    with pytest.raises(RuntimeError, match="bad config"):
        _parse_optional_positive_float_attr(BadConfig(), "low_pass_hz")


def test_coerce_condition_value_matches_series_dtype_best_effort() -> None:
    assert coerce_condition_value("10", pd.Series([1, 2, 3])) == 10
    assert coerce_condition_value("3.5", pd.Series([1.0, 2.0])) == 3.5
    assert coerce_condition_value("yes", pd.Series([True, False])) is True
    assert coerce_condition_value("not-a-number", pd.Series([1, 2, 3])) == "not-a-number"


def test_parse_optional_positive_float_attr_surfaces_unexpected_accessor_errors() -> None:
    class BrokenConfig:
        def __getattr__(self, _name: str) -> float:
            raise RuntimeError("broken config accessor")

    with pytest.raises(RuntimeError, match="broken config accessor"):
        _parse_optional_positive_float_attr(BrokenConfig(), "low_pass_hz")


def test_coerce_condition_value_surfaces_unexpected_series_accessor_errors() -> None:
    class BrokenSeries:
        @property
        def dtype(self) -> str:
            raise RuntimeError("broken dtype accessor")

    with pytest.raises(RuntimeError, match="broken dtype accessor"):
        coerce_condition_value("10", BrokenSeries())


def test_select_confounds_returns_empty_when_input_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tsv"
    confounds_df, columns = select_confounds(missing, strategy="auto")
    assert confounds_df is None
    assert columns == []


def test_select_confound_columns_uses_configured_compcor_count() -> None:
    confounds = pd.DataFrame(
        {
            "trans_x": [0.0, 0.1],
            "trans_y": [0.0, 0.1],
            "trans_z": [0.0, 0.1],
            "rot_x": [0.0, 0.1],
            "rot_y": [0.0, 0.1],
            "rot_z": [0.0, 0.1],
            "a_comp_cor_00": [0.0, 0.1],
            "a_comp_cor_01": [0.0, 0.1],
        }
    )

    selected = select_confound_columns(
        confounds,
        strategy="auto",
        auto_compcor_n=1,
    )

    assert selected is not None
    assert "a_comp_cor_00" in selected.columns
    assert "a_comp_cor_01" not in selected.columns


def test_select_confound_columns_rejects_missing_values_in_selected_confounds() -> None:
    confounds = pd.DataFrame(
        {
            "trans_x": [0.0, None],
            "trans_y": [0.0, 0.1],
            "trans_z": [0.0, 0.1],
            "rot_x": [0.0, 0.1],
            "rot_y": [0.0, 0.1],
            "rot_z": [0.0, 0.1],
        }
    )

    with pytest.raises(ValueError, match="missing values.*trans_x"):
        select_confound_columns(confounds, strategy="auto")


def test_validate_design_matrices_rejects_rank_deficient_designs() -> None:
    model = SimpleNamespace(
        design_matrices_=[
            pd.DataFrame(
                {
                    "intercept": [1.0, 1.0, 1.0],
                    "duplicate": [1.0, 1.0, 1.0],
                }
            )
        ]
    )

    with pytest.raises(ValueError, match="rank-deficient"):
        validate_design_matrices(model, context="unit-test")


def test_validate_design_matrices_rejects_unstable_condition_number() -> None:
    model = SimpleNamespace(
        design_matrices_=[
            pd.DataFrame(
                {
                    "target": [1.0, 0.0, 0.0, 0.0],
                    "nearly_target": [1.0, 1e-8, 0.0, 0.0],
                    "constant": [1.0, 1.0, 1.0, 1.0],
                }
            )
        ]
    )

    with pytest.raises(ValueError, match="condition number"):
        validate_design_matrices(
            model,
            context="unit-test",
            max_condition_number=100.0,
        )


def test_validate_design_matrices_rejects_low_target_design_efficiency() -> None:
    model = SimpleNamespace(
        design_matrices_=[
            pd.DataFrame(
                {
                    "target": [0.0, 0.001, 0.0, 0.001, 0.0],
                    "other": [0.0, 0.0, 1.0, 0.0, 1.0],
                    "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
                }
            )
        ]
    )

    with pytest.raises(ValueError, match="design efficiency"):
        validate_design_matrices(
            model,
            context="unit-test",
            target_columns=("target",),
            min_target_efficiency=0.1,
        )


def test_select_consistent_run_source_rejects_mixed_preproc_availability(tmp_path: Path) -> None:
    preproc_run_1 = tmp_path / "run-01_desc-preproc_bold.nii.gz"
    preproc_run_1.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="inconsistent across runs"):
        select_consistent_run_source(
            run_numbers=[1, 2],
            discover_preproc_bold=lambda run_num: preproc_run_1 if int(run_num) == 1 else None,
            require_fmriprep=False,
        )


def test_select_consistent_run_source_uses_raw_only_when_no_preproc_exists() -> None:
    source, preproc_by_run = select_consistent_run_source(
        run_numbers=[1, 2],
        discover_preproc_bold=lambda _run_num: None,
        require_fmriprep=False,
    )

    assert source == "bids_raw"
    assert preproc_by_run == {1: None, 2: None}
