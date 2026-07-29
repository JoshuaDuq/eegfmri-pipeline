"""``--config`` is accepted wherever the user puts it, and takes effect globally."""

from __future__ import annotations

import pytest

from eeg_pipeline.cli.main import extract_config_path


def test_the_flag_is_accepted_before_the_subcommand() -> None:
    remaining, config_path = extract_config_path(
        ["--config", "study.yaml", "preprocessing", "full"]
    )

    assert config_path == "study.yaml"
    assert remaining == ["preprocessing", "full"]


def test_the_flag_is_accepted_after_the_subcommand() -> None:
    """Which is where anyone actually types it. A global option declared on the
    top-level parser would only be accepted in the position above."""
    remaining, config_path = extract_config_path(
        ["preprocessing", "full", "--config", "study.yaml", "--all-subjects"]
    )

    assert config_path == "study.yaml"
    assert remaining == ["preprocessing", "full", "--all-subjects"]


def test_the_equals_form_is_accepted() -> None:
    remaining, config_path = extract_config_path(["preprocessing", "--config=study.yaml"])

    assert config_path == "study.yaml"
    assert remaining == ["preprocessing"]


def test_no_flag_leaves_the_arguments_untouched() -> None:
    argv = ["preprocessing", "full", "--all-subjects"]

    remaining, config_path = extract_config_path(list(argv))

    assert config_path is None
    assert remaining == argv


def test_a_flag_without_a_path_is_rejected() -> None:
    with pytest.raises(SystemExit):
        extract_config_path(["preprocessing", "--config"])


def test_validate_config_only_reports_without_reading_derivatives(tmp_path, capsys) -> None:
    """The check is meant to be cheap enough to ask before committing to a run, so it
    must not need derivatives — or subjects — to exist."""
    import argparse

    from eeg_pipeline.cli.commands.validate_orchestrator import run_validate
    from eeg_pipeline.utils.config.loader import load_config

    study = tmp_path / "study.yaml"
    study.write_text('extends: "eeg_only"\n', encoding="utf-8")

    args = argparse.Namespace(
        mode="config",
        config_only=True,
        output_json=False,
        subjects=None,
        task=None,
    )

    run_validate(args, [], load_config(study))

    # project.task is unset in the preset by design; that is the one thing to fix.
    assert "project.task" in capsys.readouterr().out
