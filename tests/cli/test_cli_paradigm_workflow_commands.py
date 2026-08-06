"""The two paradigm artifact workflows must reach the CLI the same way every pipeline does.

They were previously loose ``python -m`` scripts. What makes them runnable like
``eeg-pipeline preprocessing`` is the entry point in ``studies/pyproject.toml`` plus a
parser that builds without dragging in the analysis stack. Both are easy to break by
accident -- an entry point is only re-read on install, and a stray module-level import
would silently put MNE back on the ``--help`` path. See issue #14 for why that matters.
"""

from __future__ import annotations

import argparse

import pytest

from studies.pain_study.cli import command_registry
from studies.pain_study.cli import line_comb as line_comb_cli

WORKFLOW_COMMANDS = {
    "line-comb": (
        command_registry.line_comb_command,
        ("diagnose", "plot", "benchmark", "apply", "verify", "report"),
    ),
    "cardiac-gaps": (
        command_registry.cardiac_gaps_command,
        ("report", "markers", "benchmark", "apply", "verify"),
    ),
}


@pytest.mark.parametrize("name", sorted(WORKFLOW_COMMANDS))
def test_registry_factory_returns_the_named_command(name: str) -> None:
    factory, _ = WORKFLOW_COMMANDS[name]

    command = factory()

    assert command.name == name
    # Both default to the whole cohort; --subjects narrows it. Requiring subjects would
    # make the common invocation an error.
    assert command.requires_subjects is False


@pytest.mark.parametrize("name", sorted(WORKFLOW_COMMANDS))
def test_declared_entry_point_resolves_to_the_factory(name: str) -> None:
    """The installed entry point, not just the module, is what the CLI reads."""
    from importlib import metadata

    entry_points = {
        entry.name: entry for entry in metadata.entry_points(group="eeg_pipeline.cli_commands")
    }
    assert name in entry_points, (
        f"{name} is not registered. Reinstall the studies package after editing "
        f"studies/pyproject.toml: pip install -e studies --no-deps"
    )

    command = entry_points[name].load()()

    assert command.name == name


@pytest.mark.parametrize("name", sorted(WORKFLOW_COMMANDS))
def test_parser_exposes_the_stages_in_order(name: str) -> None:
    factory, expected_modes = WORKFLOW_COMMANDS[name]
    command = factory()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    command.setup(subparsers)

    args = parser.parse_args([name, expected_modes[0]])
    assert args.mode == expected_modes[0]

    mode_action = next(
        action
        for action in subparsers.choices[name]._actions
        if getattr(action, "dest", None) == "mode"
    )
    assert tuple(mode_action.choices) == expected_modes


@pytest.mark.parametrize("name", sorted(WORKFLOW_COMMANDS))
def test_an_unknown_stage_is_rejected(name: str) -> None:
    factory, _ = WORKFLOW_COMMANDS[name]
    command = factory()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    command.setup(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args([name, "not-a-stage"])


@pytest.mark.parametrize("obsolete", ["--limit", "--fundamental-scope"])
def test_line_comb_parser_rejects_obsolete_benchmark_controls(obsolete: str) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    line_comb_cli.setup_line_comb(subparsers)
    value = "1" if obsolete == "--limit" else "session"

    with pytest.raises(SystemExit):
        parser.parse_args(["line-comb", "benchmark", obsolete, value])


def test_line_comb_removal_rejects_subject_subsets() -> None:
    args = argparse.Namespace(mode="benchmark", subjects=["sub-0001"])

    with pytest.raises(ValueError, match="all recordings"):
        line_comb_cli.run_line_comb(args, [], config=None)


@pytest.mark.parametrize("name", sorted(WORKFLOW_COMMANDS))
def test_building_the_parser_does_not_import_the_analysis_stack(name: str) -> None:
    """``--help`` must not pay for MNE. The run functions import; the parsers must not."""
    import subprocess
    import sys

    module = {"line-comb": "line_comb", "cardiac-gaps": "cardiac_gaps"}[name]
    code = (
        "import sys, argparse\n"
        f"from studies.pain_study.cli import {module} as m\n"
        "p = argparse.ArgumentParser(); s = p.add_subparsers()\n"
        f"m.setup_{module}(s)\n"
        "print('mne' in sys.modules or 'sklearn' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "parser construction pulled in the analysis stack"
