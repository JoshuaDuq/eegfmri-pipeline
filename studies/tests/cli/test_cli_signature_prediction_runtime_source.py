from __future__ import annotations

import argparse

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.cli.command_registry import signature_prediction_command
from studies.pain_study.cli.signature_prediction import (
    run_signature_prediction,
    setup_signature_prediction,
)


class _CaptureSignaturePredictionRunner:
    last_config = None
    last_call = None

    def __init__(self, config):
        type(self).last_config = config

    def run(self, **kwargs):
        type(self).last_call = kwargs


def _build_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_signature_prediction(subparsers)
    return parser.parse_args(argv)


def test_signature_prediction_command_is_registered_from_study_package() -> None:
    command = signature_prediction_command()
    assert command.setup is setup_signature_prediction
    assert command.run is run_signature_prediction


def test_signature_prediction_parser_includes_prepare_features() -> None:
    args = _build_args(["signature-prediction", "prepare-features", "--subject", "0001"])
    assert args.mode == "prepare-features"


def test_run_signature_prediction_loads_study_yaml_into_runtime_config(
    tmp_path,
    monkeypatch,
) -> None:
    study_cfg = tmp_path / "study1.yaml"
    study_cfg.write_text(
        """
study1:
  targets:
    contrast_name: "custom_contrast"
  reference_power:
    primary_window: [-5.0, -0.01]
    sensitivity_windows:
      prestimulus_2s: [-2.0, -0.01]
      immediate_prestimulus: [-0.2, -0.01]
    unnormalized_active_power:
      feature_transform: "raw_log_power"
      feature_baseline_window: null
      active_window: [3.0, 10.5]
      reference_window: [-5.0, -0.01]
      reference_power_covariate: true
  feature_benchmark:
    n_perm: 1
    permutation_scheme: "within_subject"
    max_invalid_permutation_fraction: 0.2
  temporal_negative_controls:
    feature_transform: "raw_log_power"
    feature_baseline_window: null
    windows:
      prestimulus: [-1.0, -0.01]
    wrong_lag_windows:
      active: [1.0, 2.0]
    plateau_windows:
      early_plateau: [3.0, 5.5]
time_frequency_analysis:
  baseline_window: [-5.0, -0.01]
  active_window: [3.0, 10.5]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "studies.pain_study.cli.signature_prediction.SignaturePredictionRunner",
        _CaptureSignaturePredictionRunner,
    )

    args = _build_args(
        [
            "signature-prediction",
            "prepare-targets",
            "--subject",
            "0001",
            "--task",
            "pain",
            "--study1-config",
            str(study_cfg),
        ]
    )
    config = ConfigDict({})
    run_signature_prediction(args, ["0001"], config)

    assert _CaptureSignaturePredictionRunner.last_config.get("study1.targets.contrast_name") == (
        "custom_contrast"
    )
    assert _CaptureSignaturePredictionRunner.last_call == {
        "mode": "prepare-targets",
        "subjects": ["0001"],
        "task": "pain",
    }


def test_run_signature_prediction_all_subjects_prepare_targets_keeps_fmri_subjects(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "studies.pain_study.cli.signature_prediction.SignaturePredictionRunner",
        _CaptureSignaturePredictionRunner,
    )
    bids_fmri_root = tmp_path / "bids_fmri"
    (bids_fmri_root / "sub-0001" / "func").mkdir(parents=True)

    args = _build_args(
        [
            "signature-prediction",
            "prepare-targets",
            "--all-subjects",
            "--task",
            "pain",
        ]
    )
    config = ConfigDict({"paths": {"bids_fmri_root": str(bids_fmri_root)}})
    run_signature_prediction(args, ["0001", "0006eegonly"], config)

    assert _CaptureSignaturePredictionRunner.last_call == {
        "mode": "prepare-targets",
        "subjects": ["0001"],
        "task": "pain",
    }


def test_run_signature_prediction_all_subjects_later_stage_uses_target_table(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "studies.pain_study.cli.signature_prediction.SignaturePredictionRunner",
        _CaptureSignaturePredictionRunner,
    )

    args = _build_args(
        [
            "signature-prediction",
            "feature-benchmark",
            "--all-subjects",
            "--task",
            "pain",
        ]
    )
    run_signature_prediction(args, ["0001", "0006eegonly"], ConfigDict({}))

    assert _CaptureSignaturePredictionRunner.last_call == {
        "mode": "feature-benchmark",
        "subjects": [],
        "task": "pain",
    }


def test_run_signature_prediction_dry_run_does_not_dispatch(tmp_path, monkeypatch) -> None:
    _CaptureSignaturePredictionRunner.last_call = None
    monkeypatch.setattr(
        "studies.pain_study.cli.signature_prediction.SignaturePredictionRunner",
        _CaptureSignaturePredictionRunner,
    )
    bids_fmri_root = tmp_path / "bids_fmri"
    (bids_fmri_root / "sub-0001" / "func").mkdir(parents=True)

    args = _build_args(
        [
            "signature-prediction",
            "prepare-targets",
            "--all-subjects",
            "--task",
            "pain",
            "--dry-run",
        ]
    )
    config = ConfigDict({"paths": {"bids_fmri_root": str(bids_fmri_root)}})
    run_signature_prediction(args, ["0001"], config)

    assert _CaptureSignaturePredictionRunner.last_call is None
