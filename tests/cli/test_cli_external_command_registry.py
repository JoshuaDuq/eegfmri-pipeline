from __future__ import annotations

import importlib
import importlib.metadata
from importlib.metadata import EntryPoint

import eeg_pipeline.cli.commands as commands_module


CLI_COMMAND_GROUP = "eeg_pipeline.cli_commands"


def _reload_commands_module(monkeypatch, entry_points: list[EntryPoint]):
    def _entry_points(**kwargs):
        if kwargs.get("group") == CLI_COMMAND_GROUP:
            return entry_points
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", _entry_points)
    return importlib.reload(commands_module)


def test_external_commands_are_not_present_without_entry_point(monkeypatch) -> None:
    commands = _reload_commands_module(monkeypatch, entry_points=[])
    assert commands.get_command("fake-external") is None


def test_external_commands_are_loaded_from_entry_points(monkeypatch) -> None:
    commands = _reload_commands_module(
        monkeypatch,
        entry_points=[
            EntryPoint(
                name="fake-external",
                value="tests.cli.fake_command_plugin:fake_external_command",
                group=CLI_COMMAND_GROUP,
            )
        ],
    )

    command = commands.get_command("fake-external")

    assert command is not None
    assert command.name == "fake-external"
    assert command.requires_subjects is False
