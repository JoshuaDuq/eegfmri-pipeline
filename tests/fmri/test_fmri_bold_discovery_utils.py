from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from fmri_pipeline.utils.bold_discovery import (
    build_first_level_model,
    coerce_condition_value,
    discover_fmriprep_preproc_bold,
    get_tr_from_bold,
    select_confounds,
)


def test_discover_fmriprep_preproc_bold_accepts_zero_padded_and_non_padded_runs(tmp_path: Path) -> None:
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


def test_get_tr_from_bold_prefers_sidecar_repetition_time(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    sidecar.write_text(json.dumps({"RepetitionTime": "1.75"}), encoding="utf-8")

    tr = get_tr_from_bold(bold_path)
    assert tr == 1.75


def test_get_tr_from_bold_falls_back_to_nifti_when_sidecar_is_invalid(tmp_path: Path) -> None:
    bold_path = tmp_path / "sub-0001_task-task_run-01_desc-preproc_bold.nii.gz"
    bold_path.write_bytes(b"")
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    sidecar.write_text("{", encoding="utf-8")

    class FakeHeader:
        @staticmethod
        def get_zooms() -> tuple[float, float, float, float]:
            return (2.0, 2.0, 2.0, 2.5)

    class FakeImage:
        header = FakeHeader()

    fake_nib = types.ModuleType("nibabel")
    fake_nib.load = lambda *_args, **_kwargs: FakeImage()

    with patch.dict(sys.modules, {"nibabel": fake_nib}):
        tr = get_tr_from_bold(bold_path)

    assert tr == 2.5


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


def test_coerce_condition_value_matches_series_dtype_best_effort() -> None:
    assert coerce_condition_value("10", pd.Series([1, 2, 3])) == 10
    assert coerce_condition_value("3.5", pd.Series([1.0, 2.0])) == 3.5
    assert coerce_condition_value("yes", pd.Series([True, False])) is True
    assert coerce_condition_value("not-a-number", pd.Series([1, 2, 3])) == "not-a-number"


def test_select_confounds_returns_empty_when_input_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tsv"
    confounds_df, columns = select_confounds(missing, strategy="auto")
    assert confounds_df is None
    assert columns == []
