"""``cardiac-gaps`` path options must reach the workflow as paths.

The wrapper declares every path-like option as ``type=str`` while the workflow treats them
as ``Path`` -- ``_write_tsv`` calls ``destination.parent``, ``write_corrected_recording``
builds ``output_root / name``. Passing any of them from the CLI therefore raised
``AttributeError: 'str' object has no attribute 'parent'`` at the point of writing, after
the work had already been done.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from studies.pain_study.cli.cardiac_gaps import run_cardiac_gaps, setup_cardiac_gaps

PATH_OPTIONS = ("output", "output_root", "config", "uncorrected_root", "corrected_root")


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    setup_cardiac_gaps(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_path_options_reach_the_workflow_as_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, argparse.Namespace] = {}

    def fake_run(args: argparse.Namespace) -> None:
        seen["args"] = args

    from studies.pain_study.scripts.cardiac_gaps import correct

    monkeypatch.setattr(correct, "run", fake_run)

    args = _parse(
        [
            "cardiac-gaps",
            "report",
            "--output",
            "/tmp/report.tsv",
            "--output-root",
            "/tmp/corrected",
            # Not --config: the top-level CLI consumes that one before argparse sees it.
            "--workflow-config",
            "/tmp/config.yaml",
            "--uncorrected-root",
            "/tmp/step1",
            "--corrected-root",
            "/tmp/reference",
        ]
    )
    run_cardiac_gaps(args, subjects=[], config=None)

    passed = seen["args"]
    for option in PATH_OPTIONS:
        assert isinstance(getattr(passed, option), Path), f"{option} arrived as a str"


def test_unset_path_options_stay_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """None means 'take it from the config', which a Path would silently replace."""
    seen: dict[str, argparse.Namespace] = {}
    from studies.pain_study.scripts.cardiac_gaps import correct

    monkeypatch.setattr(correct, "run", lambda args: seen.setdefault("args", args))

    run_cardiac_gaps(_parse(["cardiac-gaps", "report"]), subjects=[], config=None)

    for option in PATH_OPTIONS:
        assert getattr(seen["args"], option) is None
