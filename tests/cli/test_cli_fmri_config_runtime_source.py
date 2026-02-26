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


class _ContrastBuilderConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _TrialSignatureExtractionConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEG_PIPELINE_FMRI_CONFIG", str(fmri_cfg))

    monkeypatch.setitem(
        sys.modules,
        "fmri_pipeline.analysis.contrast_builder",
        types.SimpleNamespace(ContrastBuilderConfig=_ContrastBuilderConfig),
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
