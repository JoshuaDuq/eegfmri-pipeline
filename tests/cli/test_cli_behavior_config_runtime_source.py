from __future__ import annotations

import argparse
import sys
import types

import pytest

from eeg_pipeline.utils.config.loader import ConfigDict
from eeg_pipeline.cli.commands.behavior import run_behavior, setup_behavior


class _CaptureBehaviorPipeline:
    last_config = None
    last_kwargs = None

    def __init__(self, config, **_kwargs):
        type(self).last_config = config

    def run_batch(self, **kwargs):
        type(self).last_kwargs = kwargs


def _build_args_for_behavior(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_behavior(subparsers)
    return parser.parse_args(argv)


def test_run_behavior_uses_behavior_yaml_as_runtime_source(tmp_path, monkeypatch) -> None:
    behavior_cfg = tmp_path / "behavior_config.yaml"
    behavior_cfg.write_text(
        """
behavior_analysis:
  statistics:
    base_seed: 123
""".strip() + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEG_PIPELINE_BEHAVIOR_CONFIG", str(behavior_cfg))

    monkeypatch.setitem(
        sys.modules,
        "eeg_pipeline.pipelines.behavior",
        types.SimpleNamespace(BehaviorPipeline=_CaptureBehaviorPipeline),
    )

    args = _build_args_for_behavior(["behavior", "compute", "--all-subjects"])
    config = ConfigDict({"project": {"task": "task"}})
    run_behavior(args, ["0001"], config)

    assert _CaptureBehaviorPipeline.last_config.get("behavior_analysis.statistics.base_seed") == 123


def test_run_behavior_precedence_yaml_then_cli_then_set(tmp_path, monkeypatch) -> None:
    behavior_cfg = tmp_path / "behavior_config.yaml"
    behavior_cfg.write_text(
        """
behavior_analysis:
  statistics:
    predictor_control: "linear"
""".strip() + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EEG_PIPELINE_BEHAVIOR_CONFIG", str(behavior_cfg))

    monkeypatch.setitem(
        sys.modules,
        "eeg_pipeline.pipelines.behavior",
        types.SimpleNamespace(BehaviorPipeline=_CaptureBehaviorPipeline),
    )

    args = _build_args_for_behavior(
        [
            "behavior",
            "compute",
            "--all-subjects",
            "--stats-predictor-control",
            "spline",
            "--set",
            "behavior_analysis.statistics.predictor_control=none",
        ]
    )
    config = ConfigDict({"project": {"task": "task"}})
    run_behavior(args, ["0001"], config)

    assert (
        _CaptureBehaviorPipeline.last_config.get("behavior_analysis.statistics.predictor_control")
        == "none"
    )


def test_load_behavior_config_raises_for_missing_env_path(tmp_path, monkeypatch) -> None:
    from eeg_pipeline.utils.config.behavior_loader import load_behavior_config
    from eeg_pipeline.utils.config.loader import ConfigError

    missing = tmp_path / "missing_behavior_config.yaml"
    monkeypatch.setenv("EEG_PIPELINE_BEHAVIOR_CONFIG", str(missing))

    with pytest.raises(ConfigError, match="Behavior config file not found"):
        load_behavior_config()


def test_load_behavior_config_raises_for_non_mapping_yaml(tmp_path, monkeypatch) -> None:
    from eeg_pipeline.utils.config.behavior_loader import load_behavior_config
    from eeg_pipeline.utils.config.loader import ConfigError

    config_path = tmp_path / "behavior_config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    monkeypatch.setenv("EEG_PIPELINE_BEHAVIOR_CONFIG", str(config_path))

    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_behavior_config()


def test_load_behavior_config_raises_for_empty_yaml(tmp_path, monkeypatch) -> None:
    from eeg_pipeline.utils.config.behavior_loader import load_behavior_config
    from eeg_pipeline.utils.config.loader import ConfigError

    config_path = tmp_path / "behavior_config.yaml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("EEG_PIPELINE_BEHAVIOR_CONFIG", str(config_path))

    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_behavior_config()
