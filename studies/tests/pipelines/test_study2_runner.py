"""Tests for the Study 2 source-space stage dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from studies.pain_study.study2.runner import (
    STUDY2_STAGES,
    Study2Runner,
    Study2Stage,
    Study2StageContext,
)


def _config(tmp_path: Path) -> dict:
    return {"paths": {"deriv_root": str(tmp_path)}, "study2": {}}


def _recording_stage(name: str, calls: list[str], *, required: tuple[Path, ...] = ()) -> Study2Stage:
    def run(context: Study2StageContext) -> None:
        calls.append(name)

    return Study2Stage(name=name, run=run, required_inputs=lambda context: required)


def test_runner_dispatches_named_stage(tmp_path: Path) -> None:
    calls: list[str] = []
    stages = (_recording_stage("gate", calls), _recording_stage("source-stage", calls))
    runner = Study2Runner(config=_config(tmp_path), stages=stages)

    runner.run(mode="source-stage", subjects=["sub-0000"], task="pain")

    assert calls == ["source-stage"]


def test_runner_all_runs_every_stage_in_table_order(tmp_path: Path) -> None:
    calls: list[str] = []
    stages = (_recording_stage("gate", calls), _recording_stage("source-stage", calls))
    runner = Study2Runner(config=_config(tmp_path), stages=stages)

    runner.run(mode="all", subjects=["sub-0000"], task="pain")

    assert calls == ["gate", "source-stage"]


def test_runner_rejects_unknown_mode(tmp_path: Path) -> None:
    runner = Study2Runner(config=_config(tmp_path), stages=(_recording_stage("gate", []),))

    with pytest.raises(ValueError, match="Unsupported Study 2 mode: unknown-mode"):
        runner.run(mode="unknown-mode", subjects=[], task="pain")


def test_runner_fails_fast_on_missing_required_input(tmp_path: Path) -> None:
    calls: list[str] = []
    missing = tmp_path / "absent" / "source_power_alpha.npy"
    stages = (_recording_stage("source-stage", calls, required=(missing,)),)
    runner = Study2Runner(config=_config(tmp_path), stages=stages)

    with pytest.raises(FileNotFoundError, match="source-stage"):
        runner.run(mode="source-stage", subjects=["sub-0000"], task="pain")

    assert calls == []


def test_runner_runs_stage_when_required_inputs_exist(tmp_path: Path) -> None:
    calls: list[str] = []
    present = tmp_path / "present.npy"
    present.write_bytes(b"0")
    stages = (_recording_stage("source-stage", calls, required=(present,)),)
    runner = Study2Runner(config=_config(tmp_path), stages=stages)

    runner.run(mode="source-stage", subjects=["sub-0000"], task="pain")

    assert calls == ["source-stage"]


def test_default_stage_table_modes_are_unique_and_nonempty() -> None:
    names = [stage.name for stage in STUDY2_STAGES]

    assert names, "Study 2 stage table must not be empty."
    assert len(names) == len(set(names)), "Study 2 stage names must be unique."
