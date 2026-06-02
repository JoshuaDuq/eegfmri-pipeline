from __future__ import annotations

import argparse

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.cli.command_registry import source_interpretation_command
from studies.pain_study.cli.source_interpretation import (
    run_source_interpretation,
    setup_source_interpretation,
)


class _CaptureStudy2Runner:
    last_config = None
    last_call = None

    def __init__(self, config):
        type(self).last_config = config

    def run(self, **kwargs):
        type(self).last_call = kwargs


def _build_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_source_interpretation(subparsers)
    return parser.parse_args(argv)


def test_source_interpretation_command_is_registered_from_study_package() -> None:
    command = source_interpretation_command()
    assert command.setup is setup_source_interpretation
    assert command.run is run_source_interpretation


def test_source_interpretation_parser_accepts_all_mode() -> None:
    args = _build_args(["source-interpretation", "all", "--task", "pain"])
    assert args.mode == "all"


def test_run_source_interpretation_dispatches_to_runner(monkeypatch) -> None:
    _CaptureStudy2Runner.last_call = None
    monkeypatch.setattr(
        "studies.pain_study.cli.source_interpretation.Study2Runner",
        _CaptureStudy2Runner,
    )

    args = _build_args(
        ["source-interpretation", "gate", "--subject", "0000", "--task", "pain"]
    )
    run_source_interpretation(args, ["sub-0000"], ConfigDict({}))

    assert _CaptureStudy2Runner.last_call == {
        "mode": "gate",
        "subjects": ["sub-0000"],
        "task": "pain",
    }


def test_run_source_interpretation_dry_run_does_not_dispatch(monkeypatch) -> None:
    _CaptureStudy2Runner.last_call = None
    monkeypatch.setattr(
        "studies.pain_study.cli.source_interpretation.Study2Runner",
        _CaptureStudy2Runner,
    )

    args = _build_args(
        ["source-interpretation", "gate", "--task", "pain", "--dry-run"]
    )
    run_source_interpretation(args, ["sub-0000"], ConfigDict({}))

    assert _CaptureStudy2Runner.last_call is None
