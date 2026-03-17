from __future__ import annotations

import argparse

from eeg_pipeline.cli.commands import get_command
from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.cli.coupling import run_coupling, setup_coupling


class _CaptureCouplingPipeline:
    last_config = None
    last_kwargs = None

    def __init__(self, config):
        type(self).last_config = config

    def run_batch(self, **kwargs):
        type(self).last_kwargs = kwargs
        return []


def _build_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_coupling(subparsers)
    return parser.parse_args(argv)


def test_coupling_command_is_registered_from_study_package() -> None:
    command = get_command("coupling")
    assert command is not None
    assert command.setup is setup_coupling
    assert command.run is run_coupling


def test_run_coupling_loads_study_yaml_into_runtime_config(tmp_path, monkeypatch) -> None:
    coupling_cfg = tmp_path / "coupling.yaml"
    coupling_cfg.write_text(
        """
eeg_bold_coupling:
  eeg:
    method: "eloreta"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "studies.pain_study.cli.coupling.EEGBOLDCouplingPipeline",
        _CaptureCouplingPipeline,
    )

    args = _build_args(
        [
            "coupling",
            "compute",
            "--subject",
            "0001",
            "--task",
            "pain",
            "--coupling-config",
            str(coupling_cfg),
        ]
    )
    config = ConfigDict({})
    run_coupling(args, ["0001"], config)

    assert _CaptureCouplingPipeline.last_config.get("eeg_bold_coupling.eeg.method") == "eloreta"
    assert _CaptureCouplingPipeline.last_kwargs["subjects"] == ["0001"]
    assert _CaptureCouplingPipeline.last_kwargs["task"] == "pain"
