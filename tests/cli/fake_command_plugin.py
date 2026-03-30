from __future__ import annotations

from eeg_pipeline.cli.commands import Command


def fake_external_command() -> Command:
    def _setup(_subparsers):
        raise NotImplementedError

    def _run(_args, _subjects, _config):
        raise NotImplementedError

    return Command(
        name="fake-external",
        setup=_setup,
        run=_run,
        requires_subjects=False,
    )
