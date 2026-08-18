"""
CLI Commands Package
=====================

Modular CLI command definitions. Each command module provides:
- setup function: Configure argparse parser
- run function: Execute the command

Commands are re-exported for registration in main.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any, Callable, List

CLI_COMMAND_GROUP = "eeg_pipeline.cli_commands"


@dataclass
class Command:
    """CLI command definition."""

    name: str
    setup: Callable[[argparse._SubParsersAction], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace, List[str], Any], None]
    requires_subjects: bool = True


class MissingCommandDependency(ImportError):
    """A command needs an analysis package that this environment does not have.

    Separate from a configuration error because the remedy is different and the user
    cannot tell them apart from the traceback: one is fixed by editing YAML, the other
    by installing something. Issue #14 asks for exactly this distinction.
    """


#: Import roots that are this repository rather than a dependency. A ModuleNotFoundError
#: naming one of these is a bug in the code, not a missing install, and must keep its
#: traceback instead of being reported as an environment problem.
_FIRST_PARTY_ROOTS = frozenset({"eeg_pipeline", "fmri_pipeline", "studies"})


def _deferred(module: str, attribute: str, command_name: str = "") -> Callable[..., Any]:
    """A stand-in for ``module.attribute`` that imports it when it is first called.

    Registering the twelve commands used to import all twelve, so every invocation paid
    for every command: ``validate --config-only``, which reads a YAML and reports what
    contradicts what, loaded MNE, scikit-learn, Nilearn and Seaborn to do it. In an
    environment where those are not installed — the environment of someone still setting
    the study up — the config checker failed on a plotting import, naming a package that
    has nothing to do with their config. See issue #14.

    The command that is actually invoked imports exactly what it needs, when it runs.
    """

    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            imported = import_module(module)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").partition(".")[0]
            if not missing or missing in _FIRST_PARTY_ROOTS:
                raise
            label = f"The '{command_name}' command" if command_name else f"{module}"
            raise MissingCommandDependency(
                f"{label} needs the '{missing}' package, which is not installed in this "
                f"environment. This is an environment problem rather than a "
                f"configuration one — 'eeg-pipeline validate --config-only' checks a "
                f"study config and does not need it. Install the analysis dependencies "
                f'with: pip install -e ".[dev]"'
            ) from exc
        return getattr(imported, attribute)(*args, **kwargs)

    call.__name__ = attribute
    call.__qualname__ = f"{module}.{attribute}"
    return call


#: ``name -> (parser module, setup attribute, orchestrator module, run attribute)``.
#: Split across two modules per command because building the parser and doing the work
#: need different things: argparse declarations are cheap, and the analysis imports live
#: behind the orchestrator, where only the command being run pays for them.
_EEG = "eeg_pipeline.cli.commands"
_FMRI = "fmri_pipeline.cli.commands"

_BUILTIN_COMMAND_SPECS: tuple[tuple[str, str, str, bool], ...] = (
    # (command name, module stem, setup/run suffix, requires_subjects)
    ("behavior", f"{_EEG}.behavior", "behavior", True),
    # The cohort-report default is every participant with a QC sidecar, so a missing
    # subject list is the ordinary case rather than a usage error.
    ("cohort-report", f"{_EEG}.cohort_report", "cohort_report", False),
    ("component-tfr", f"{_EEG}.component_tfr", "component_tfr", True),
    ("features", f"{_EEG}.features", "features", True),
    ("info", f"{_EEG}.info", "info", False),
    ("ml", f"{_EEG}.machine_learning", "ml", True),
    # Reports the shape of the dataset rather than acting on named subjects, and
    # which subjects are present is one of the things it reports.
    ("preflight", f"{_EEG}.preflight", "preflight", False),
    ("preprocessing", f"{_EEG}.preprocessing", "preprocessing", True),
    ("stats", f"{_EEG}.stats", "stats", False),
    ("validate", f"{_EEG}.validate", "validate", False),
)

#: The fMRI commands keep setup and run in one module rather than a parser/orchestrator
#: pair, so they are spelled out instead of following the stem convention above.
_FMRI_COMMAND_SPECS: tuple[tuple[str, str, str], ...] = (
    ("fmri", f"{_FMRI}.fmri", "fmri"),
    ("fmri-analysis", f"{_FMRI}.fmri_analysis", "fmri_analysis"),
)


def _builtin_commands() -> list[Command]:
    commands = [
        Command(
            name=name,
            setup=_deferred(f"{stem}_parser", f"setup_{suffix}", name),
            run=_deferred(f"{stem}_orchestrator", f"run_{suffix}", name),
            requires_subjects=requires_subjects,
        )
        for name, stem, suffix, requires_subjects in _BUILTIN_COMMAND_SPECS
    ]
    commands += [
        Command(
            name=name,
            setup=_deferred(module, f"setup_{suffix}", name),
            run=_deferred(module, f"run_{suffix}", name),
            requires_subjects=False,
        )
        for name, module, suffix in _FMRI_COMMAND_SPECS
    ]
    return commands


def _load_entry_point(entry_point: Any) -> Command:
    command = entry_point.load()()
    if not isinstance(command, Command):
        raise TypeError(
            f"CLI entry point {entry_point.value!r} must return "
            f"{Command.__module__}.{Command.__qualname__}"
        )
    return command


def _external_commands() -> list[Command]:
    return [
        _load_entry_point(entry_point)
        for entry_point in metadata.entry_points(group=CLI_COMMAND_GROUP)
    ]


def _validate_unique_names(commands: list[Command]) -> None:
    command_names = [command.name for command in commands]
    duplicates = {name for name in command_names if command_names.count(name) > 1}
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate CLI command names are not allowed: {duplicate_list}")


def get_commands() -> list[Command]:
    commands = _builtin_commands() + _external_commands()
    _validate_unique_names(commands)
    return commands


def get_command_names() -> list[str]:
    """Every registered command name, without importing any command module.

    Entry-point *names* are packaging metadata, so third-party commands can be named
    here without being loaded. This is what lets the CLI decide which single subparser
    it needs before paying to build any of them.
    """
    names = [name for name, _, _, _ in _BUILTIN_COMMAND_SPECS]
    names += [name for name, _, _ in _FMRI_COMMAND_SPECS]
    names += [entry_point.name for entry_point in metadata.entry_points(group=CLI_COMMAND_GROUP)]
    return names


def get_command(name: str) -> Command | None:
    """Get one command by name, building only that one.

    Resolving a name used to mean constructing all of them, which imported all of them.
    Only the requested command is built here, and its ``setup`` and ``run`` are deferred
    besides, so the modules that actually get imported are the ones the invocation uses.
    """
    for spec_name, stem, suffix, requires_subjects in _BUILTIN_COMMAND_SPECS:
        if spec_name == name:
            return Command(
                name=name,
                setup=_deferred(f"{stem}_parser", f"setup_{suffix}", name),
                run=_deferred(f"{stem}_orchestrator", f"run_{suffix}", name),
                requires_subjects=requires_subjects,
            )

    for spec_name, module, suffix in _FMRI_COMMAND_SPECS:
        if spec_name == name:
            return Command(
                name=name,
                setup=_deferred(module, f"setup_{suffix}", name),
                run=_deferred(module, f"run_{suffix}", name),
                requires_subjects=False,
            )

    for entry_point in metadata.entry_points(group=CLI_COMMAND_GROUP):
        if entry_point.name == name:
            return _load_entry_point(entry_point)

    return None


def __getattr__(attribute: str) -> Any:
    """Build the ``COMMANDS`` list on first access rather than at import.

    It was a module-level constant, so importing this package to reach ``Command`` — as
    the entry-point factories do — constructed every command and imported every command
    module. Nothing needs the whole list except ``--help``.
    """
    if attribute == "COMMANDS":
        return get_commands()
    raise AttributeError(f"module {__name__!r} has no attribute {attribute!r}")


__all__ = [
    "CLI_COMMAND_GROUP",
    "Command",
    "MissingCommandDependency",
    "COMMANDS",
    "get_command",
    "get_command_names",
    "get_commands",
]
