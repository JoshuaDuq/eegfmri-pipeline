from __future__ import annotations

import argparse

from eeg_pipeline.cli.commands.harmonics import setup_harmonics


def test_harmonics_help_renders() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_harmonics(subparsers)

    try:
        parser.parse_args(["harmonics", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected --help to exit.")


def test_harmonics_parser_requires_input_and_output_paths() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    setup_harmonics(subparsers)

    args = parser.parse_args(
        [
            "harmonics",
            "--input-root",
            "/tmp/brainvision",
            "--output-dir",
            "/tmp/qc",
            "--subject",
            "0008",
            "--subject",
            "0009",
        ]
    )

    assert args.command == "harmonics"
    assert args.input_root == "/tmp/brainvision"
    assert args.output_dir == "/tmp/qc"
    assert args.subject == ["0008", "0009"]
