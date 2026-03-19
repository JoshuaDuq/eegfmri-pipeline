from __future__ import annotations

import argparse
import sys
import types

from eeg_pipeline.utils.config.loader import ConfigDict
from fmri_pipeline.cli.commands.fmri import run_fmri, setup_fmri
from fmri_pipeline.cli.commands.fmri_analysis import run_fmri_analysis, setup_fmri_analysis


class _CapturePreprocessingPipeline:
    last_config = None
    last_kwargs = None

    def __init__(self, config):
        type(self).last_config = config

    def run_batch(self, **kwargs):
        type(self).last_kwargs = kwargs


class _CaptureRestPipeline:
    last_config = None
    last_kwargs = None

    def __init__(self, config):
        type(self).last_config = config

    def run_batch(self, **kwargs):
        type(self).last_kwargs = kwargs


class _CaptureAnalysisPipeline:
    last_config = None
    last_kwargs = None

    def __init__(self, config):
        type(self).last_config = config

    def run_batch(self, **kwargs):
        type(self).last_kwargs = kwargs


class _CaptureTrialPipeline:
    last_config = None
    last_kwargs = None

    def __init__(self, config):
        type(self).last_config = config

    def run_batch(self, **kwargs):
        type(self).last_kwargs = kwargs


class _CaptureSecondLevelPipeline:
    last_config = None
    last_kwargs = None

    def __init__(self, config):
        type(self).last_config = config

    def run_batch(self, **kwargs):
        type(self).last_kwargs = kwargs


class _ContrastBuilderConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _TrialSignatureExtractionConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _SecondLevelPermutationConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _SecondLevelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def normalized(self):
        return self


class _RestingStateAnalysisConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def normalized(self):
        return self


def _build_args_for_fmri(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_fmri(subparsers)
    return parser.parse_args(argv)


def _build_args_for_fmri_analysis(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_fmri_analysis(subparsers)
    return parser.parse_args(argv)


def test_run_fmri_uses_fmri_yaml_as_runtime_source(tmp_path, monkeypatch) -> None:
    bids_root = tmp_path / "bids"
    (bids_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    fs_license = tmp_path / "license.txt"
    fs_license.write_text("ok", encoding="utf-8")

    fmri_cfg = tmp_path / "fmri_config.yaml"
    fmri_cfg.write_text(
        f"""
paths:
  bids_fmri_root: "{bids_root}"
fmri_preprocessing:
  engine: "apptainer"
  fmriprep:
    fs_license_file: "{fs_license}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEG_PIPELINE_FMRI_CONFIG", str(fmri_cfg))

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_preprocessing",
        types.SimpleNamespace(FmriPreprocessingPipeline=_CapturePreprocessingPipeline),
    )

    args = _build_args_for_fmri(["fmri", "preprocess", "--all-subjects", "--dry-run"])
    config = ConfigDict({"project": {"task": "task"}})
    run_fmri(args, [], config)

    assert _CapturePreprocessingPipeline.last_config.get("paths.bids_fmri_root") == str(bids_root)
    assert _CapturePreprocessingPipeline.last_config.get("fmri_preprocessing.engine") == "apptainer"
    assert _CapturePreprocessingPipeline.last_kwargs["subjects"] == ["0001"]


def test_run_fmri_precedence_is_yaml_then_cli_then_set(tmp_path, monkeypatch) -> None:
    bids_root = tmp_path / "bids"
    (bids_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    fs_license = tmp_path / "license.txt"
    fs_license.write_text("ok", encoding="utf-8")

    fmri_cfg = tmp_path / "fmri_config.yaml"
    fmri_cfg.write_text(
        f"""
paths:
  bids_fmri_root: "{bids_root}"
fmri_preprocessing:
  engine: "docker"
  fmriprep:
    fs_license_file: "{fs_license}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEG_PIPELINE_FMRI_CONFIG", str(fmri_cfg))

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_preprocessing",
        types.SimpleNamespace(FmriPreprocessingPipeline=_CapturePreprocessingPipeline),
    )

    args = _build_args_for_fmri(
        [
            "fmri",
            "preprocess",
            "--all-subjects",
            "--dry-run",
            "--engine",
            "apptainer",
            "--set",
            "fmri_preprocessing.engine=docker",
        ]
    )
    config = ConfigDict({"project": {"task": "task"}})
    run_fmri(args, [], config)

    assert _CapturePreprocessingPipeline.last_config.get("fmri_preprocessing.engine") == "docker"


def test_run_fmri_preprocess_uses_rest_root_for_subject_discovery(tmp_path, monkeypatch) -> None:
    task_root = tmp_path / "task_bids"
    task_root.mkdir(parents=True, exist_ok=True)
    rest_root = tmp_path / "rest_bids"
    (rest_root / "sub-0001").mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_preprocessing",
        types.SimpleNamespace(FmriPreprocessingPipeline=_CapturePreprocessingPipeline),
    )

    args = _build_args_for_fmri(
        [
            "fmri",
            "preprocess",
            "--all-subjects",
            "--dry-run",
            "--task-is-rest",
            "--bids-fmri-root",
            str(task_root),
            "--bids-rest-root",
            str(rest_root),
        ]
    )
    config = ConfigDict({"project": {"task": "task"}})
    run_fmri(args, [], config)

    assert _CapturePreprocessingPipeline.last_config.get("fmri_preprocessing.task_is_rest") is True
    assert _CapturePreprocessingPipeline.last_kwargs["subjects"] == ["0001"]


def test_run_fmri_analysis_loads_fmri_yaml_into_runtime_config(tmp_path, monkeypatch) -> None:
    bids_root = tmp_path / "bids"
    (bids_root / "sub-0001").mkdir(parents=True, exist_ok=True)

    fmri_cfg = tmp_path / "fmri_config.yaml"
    fmri_cfg.write_text(
        f"""
paths:
  bids_fmri_root: "{bids_root}"
fmri_contrast:
  name: "from_yaml"
  condition_a:
    column: "trial_type"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEG_PIPELINE_FMRI_CONFIG", str(fmri_cfg))

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.contrast_builder",
        types.SimpleNamespace(
            ContrastBuilderConfig=_ContrastBuilderConfig,
            load_contrast_config_section=lambda config: config.get("fmri_contrast"),
            validate_contrast_config_section=lambda _section: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.plotting_config",
        types.SimpleNamespace(build_fmri_plotting_config_from_args=lambda **kwargs: types.SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.smoothing",
        types.SimpleNamespace(normalize_smoothing_fwhm=lambda value: value),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_analysis",
        types.SimpleNamespace(FmriAnalysisPipeline=_CaptureAnalysisPipeline),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.trial_signatures",
        types.SimpleNamespace(TrialSignatureExtractionConfig=_TrialSignatureExtractionConfig),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_trial_signatures",
        types.SimpleNamespace(FmriTrialSignaturePipeline=_CaptureTrialPipeline),
    )

    args = _build_args_for_fmri_analysis(
        [
            "fmri-analysis",
            "first-level",
            "--all-subjects",
            "--cond-a-value",
            "pain",
            "--dry-run",
        ]
    )
    config = ConfigDict({"project": {"task": "task"}})
    run_fmri_analysis(args, [], config)

    assert _CaptureAnalysisPipeline.last_config.get("fmri_contrast.name") == "from_yaml"


def test_run_fmri_analysis_uses_yaml_defaults_for_first_level_cfg(tmp_path, monkeypatch) -> None:
    bids_root = tmp_path / "bids"
    (bids_root / "sub-0001").mkdir(parents=True, exist_ok=True)

    fmri_cfg = tmp_path / "fmri_config.yaml"
    fmri_cfg.write_text(
        f"""
paths:
  bids_fmri_root: "{bids_root}"
  freesurfer_dir: "{tmp_path / 'fs'}"
fmri_contrast:
  name: "yaml-defaults"
  condition_a:
    column: "binary_outcome"
    value: 1
  condition_b:
    column: "binary_outcome"
    value: 0
  smoothing_fwhm: 6.0
  output_type: "t-stat"
  resample_to_freesurfer: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEG_PIPELINE_FMRI_CONFIG", str(fmri_cfg))

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.contrast_builder",
        types.SimpleNamespace(
            ContrastBuilderConfig=_ContrastBuilderConfig,
            load_contrast_config_section=lambda config: config.get("fmri_contrast"),
            validate_contrast_config_section=lambda _section: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.plotting_config",
        types.SimpleNamespace(build_fmri_plotting_config_from_args=lambda **kwargs: types.SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.smoothing",
        types.SimpleNamespace(normalize_smoothing_fwhm=lambda value: value),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_analysis",
        types.SimpleNamespace(FmriAnalysisPipeline=_CaptureAnalysisPipeline),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.trial_signatures",
        types.SimpleNamespace(TrialSignatureExtractionConfig=_TrialSignatureExtractionConfig),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_trial_signatures",
        types.SimpleNamespace(FmriTrialSignaturePipeline=_CaptureTrialPipeline),
    )

    args = _build_args_for_fmri_analysis(
        [
            "fmri-analysis",
            "first-level",
            "--all-subjects",
            "--dry-run",
        ]
    )
    config = ConfigDict({"project": {"task": "task"}})
    run_fmri_analysis(args, [], config)

    contrast_cfg = _CaptureAnalysisPipeline.last_kwargs["contrast_cfg"]
    assert contrast_cfg.name == "yaml-defaults"
    assert contrast_cfg.condition_a_column == "binary_outcome"
    assert contrast_cfg.condition_a_value == "1"
    assert contrast_cfg.condition_b_value == "0"
    assert contrast_cfg.smoothing_fwhm == 6.0
    assert contrast_cfg.output_type == "t-stat"
    assert contrast_cfg.resample_to_freesurfer is True


def test_run_fmri_analysis_dispatches_second_level_mode(tmp_path, monkeypatch) -> None:
    bids_root = tmp_path / "bids"
    (bids_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    (bids_root / "sub-0002").mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.second_level",
        types.SimpleNamespace(
            SecondLevelConfig=_SecondLevelConfig,
            SecondLevelPermutationConfig=_SecondLevelPermutationConfig,
            load_second_level_config_section=lambda config: config.get("fmri_group_level", {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_second_level",
        types.SimpleNamespace(FmriSecondLevelPipeline=_CaptureSecondLevelPipeline),
    )

    args = _build_args_for_fmri_analysis(
        [
            "fmri-analysis",
            "second-level",
            "--all-subjects",
            "--group-model",
            "paired",
            "--group-contrast-names",
            "pain_low",
            "pain_high",
            "--contrast-name",
            "pain_high_minus_low",
            "--group-permutation-inference",
            "--group-n-permutations",
            "2500",
            "--group-one-sided",
            "--dry-run",
        ]
    )
    config = ConfigDict(
        {
            "project": {"task": "pain"},
            "paths": {"bids_fmri_root": str(bids_root)},
        }
    )
    run_fmri_analysis(args, [], config)

    second_level_cfg = _CaptureSecondLevelPipeline.last_kwargs["second_level_cfg"]
    assert second_level_cfg.model == "paired"
    assert second_level_cfg.contrast_names == ("pain_low", "pain_high")
    assert second_level_cfg.output_name == "pain_high_minus_low"
    assert second_level_cfg.permutation.enabled is True
    assert second_level_cfg.permutation.n_permutations == 2500
    assert second_level_cfg.permutation.two_sided is False


def test_run_fmri_analysis_dispatches_rest_mode(tmp_path, monkeypatch) -> None:
    task_root = tmp_path / "task_bids"
    task_root.mkdir(parents=True, exist_ok=True)
    rest_root = tmp_path / "rest_bids"
    (rest_root / "sub-0001").mkdir(parents=True, exist_ok=True)
    atlas_img = tmp_path / "atlas.nii.gz"
    atlas_img.write_bytes(b"atlas")

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.resting_state",
        types.SimpleNamespace(
            RestingStateAnalysisConfig=_RestingStateAnalysisConfig,
            load_resting_state_config_section=lambda config: config.get("fmri_resting_state", {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.pipelines.fmri_resting_state",
        types.SimpleNamespace(FmriRestingStatePipeline=_CaptureRestPipeline),
    )

    args = _build_args_for_fmri_analysis(
        [
            "fmri-analysis",
            "rest",
            "--all-subjects",
            "--dry-run",
            "--task-is-rest",
            "--bids-fmri-root",
            str(task_root),
            "--bids-rest-root",
            str(rest_root),
            "--atlas-labels-img",
            str(atlas_img),
        ]
    )
    config = ConfigDict({"project": {"task": "pain"}})
    run_fmri_analysis(args, [], config)

    assert _CaptureRestPipeline.last_config.get("fmri_resting_state.task_is_rest") is True
    assert _CaptureRestPipeline.last_kwargs["subjects"] == ["0001"]
    assert _CaptureRestPipeline.last_kwargs["task"] == "rest"
    assert _CaptureRestPipeline.last_kwargs["rest_cfg"].atlas_labels_img == str(atlas_img)
