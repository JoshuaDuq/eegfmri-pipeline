"""A workflow's own config flag must not collide with the global ``--config``.

``extract_config_path`` pulls ``--config`` out of argv before argparse ever runs, wherever
in the line it appears, because that file decides what the whole run means. A subcommand
that also declared ``--config`` therefore could never receive one: the path went to the
core loader instead, and the run died on an unrelated ``deriv_root`` error rather than on
anything the user could connect to what they typed.

Both paradigm workflows declared exactly that flag, and both were unreachable.
"""

from __future__ import annotations

import argparse

import pytest

from eeg_pipeline.cli.main import extract_config_path
from studies.pain_study.cli.cardiac_gaps import setup_cardiac_gaps
from studies.pain_study.cli.line_comb import setup_line_comb

WORKFLOWS = {"line-comb": setup_line_comb, "cardiac-gaps": setup_cardiac_gaps}


def _parser(setup) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    setup(parser.add_subparsers(dest="command"))
    return parser


def _options(setup) -> set[str]:
    subparser = _parser(setup)._subparsers._group_actions[0].choices
    return {option for action in next(iter(subparser.values()))._actions for option in action.option_strings}


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_the_global_flag_leaves_the_workflow_flag_alone(name) -> None:
    remaining, config_path = extract_config_path(
        [name, "benchmark", "--workflow-config", "workflow.yaml"]
    )

    assert config_path is None
    assert remaining == [name, "benchmark", "--workflow-config", "workflow.yaml"]


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_the_workflow_flag_reaches_the_namespace(name) -> None:
    parser = _parser(WORKFLOWS[name])

    args = parser.parse_args([name, "benchmark", "--workflow-config", "workflow.yaml"])

    assert args.config == "workflow.yaml"


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_no_workflow_declares_the_unreachable_flag(name) -> None:
    """Declaring it is worse than omitting it: argparse advertises an option that the
    layer above has already consumed, so the value silently reconfigures the core run."""
    assert "--config" not in _options(WORKFLOWS[name])


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_the_global_flag_still_wins_when_it_is_the_one_typed(name) -> None:
    """Unchanged behaviour: --config is the core config, wherever it appears."""
    remaining, config_path = extract_config_path([name, "benchmark", "--config", "study.yaml"])

    assert config_path == "study.yaml"
    assert remaining == [name, "benchmark"]
