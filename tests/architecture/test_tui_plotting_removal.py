from __future__ import annotations

import os

import pytest

from tests import REPO_ROOT

os.environ["MNE_DONTWRITE_HOME"] = "true"


# There is no PROTECTED_PLOTS list any more, and its emptiness is the record of a
# reversed decision rather than an oversight. scanner_harmonic_comb.py was deliberately
# kept in core when the rest of the plotting tree was removed. It has now MOVED to
# studies/pain_study/scripts/gradient/scanner_harmonic_comb_plot.py -- not been deleted --
# because it draws a scanner measurement, and core describes EEG alone.

REMOVED_ENTRY_POINTS = (
    "eeg_pipeline/plotting/plot_catalog.json",
    "eeg_pipeline/cli/commands/plotting.py",
    "eeg_pipeline/cli/commands/plotting_parser.py",
    "eeg_pipeline/cli/commands/plotting_orchestrator.py",
)

REMOVED_BEHAVIOR_PLOTS = (
    "eeg_pipeline/plotting/behavioral",
    "eeg_pipeline/plotting/orchestration/behavior.py",
)

REMOVED_FEATURE_PLOTS = (
    "eeg_pipeline/plotting/features",
    "eeg_pipeline/plotting/erp",
    "eeg_pipeline/plotting/tfr",
    "eeg_pipeline/plotting/core",
    "eeg_pipeline/plotting/orchestration/features.py",
)


def test_top_level_plotting_command_is_removed() -> None:
    from eeg_pipeline.cli.commands import get_commands

    command_names = {command.name for command in get_commands()}
    commands_module = (REPO_ROOT / "eeg_pipeline/cli/commands/__init__.py").read_text(
        encoding="utf-8"
    )
    cli_main = (REPO_ROOT / "eeg_pipeline/cli/main.py").read_text(encoding="utf-8")

    assert "plotting" not in command_names
    assert "setup_plotting" not in commands_module
    assert "run_plotting" not in commands_module
    assert "python -m eeg_pipeline.cli.main plotting" not in cli_main


@pytest.mark.parametrize("relative_path", REMOVED_ENTRY_POINTS)
def test_dedicated_plotting_entry_point_is_removed(relative_path: str) -> None:
    assert not (REPO_ROOT / relative_path).exists()


def test_behavior_visualize_is_rejected_by_parser() -> None:
    from eeg_pipeline.cli.main import create_argument_parser

    parser = create_argument_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["behavior", "visualize", "--help"])

    assert error.value.code == 2


@pytest.mark.parametrize("relative_path", REMOVED_BEHAVIOR_PLOTS)
def test_behavior_plotting_root_is_removed(relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    assert not path.exists() if path.suffix else not any(path.rglob("*.py"))


def test_features_visualize_is_rejected_by_parser() -> None:
    from eeg_pipeline.cli.main import create_argument_parser

    parser = create_argument_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["features", "visualize", "--help"])

    assert error.value.code == 2


@pytest.mark.parametrize("relative_path", REMOVED_FEATURE_PLOTS)
def test_feature_plotting_root_is_removed(relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    assert not path.exists() if path.suffix else not any(path.rglob("*.py"))
