from __future__ import annotations

import argparse

from eeg_pipeline.cli.commands import get_command
from eeg_pipeline.utils.config.loader import ConfigDict
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
    command = get_command("signature-prediction")
    assert command is not None
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
