"""
Unified EEG Pipeline CLI
========================

Single entry point for all pipeline commands.

Usage:
    python -m eeg_pipeline.cli.main <command> [options]

    Or via console script (if installed):
    eeg-pipeline <command> [options]
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from typing import Any

from eeg_pipeline.utils.config.loader import ConfigError, load_config, set_default_config_path
from eeg_pipeline.utils.config.overrides import apply_runtime_overrides
from eeg_pipeline.utils.data.subjects import parse_subject_args
from eeg_pipeline.cli.common import get_deriv_root
from eeg_pipeline.cli.commands import (
    Command,
    MissingCommandDependency,
    get_command,
    get_command_names,
    get_commands,
)

os.environ["NUMPY_SKIP_MACOS_CHECK"] = "1"
warnings.filterwarnings(
    "ignore", message=".*found in sys.modules.*", category=RuntimeWarning, module="runpy"
)


EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_NO_SUBJECTS = 2


def setup_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_argument_parser(only: str | None = None) -> argparse.ArgumentParser:
    """Create the main parser, with every subcommand or with just the one being run.

    Registering a subcommand means importing the module that declares its arguments, so
    building all of them imported all of them — and ``validate --config-only``, which
    only reads a YAML, paid for MNE, scikit-learn, Nilearn and Seaborn before it started.
    See issue #14.

    ``only`` names the single subcommand to register. It is passed when the invocation
    already identifies one, and left None for ``--help`` and for an unrecognized command,
    where the full list is the answer being asked for.
    """
    parser = argparse.ArgumentParser(
        description="Unified EEG Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Behavior: compute correlations
  python -m eeg_pipeline.cli.main behavior compute --subject 0001

  # Features: compute feature tables
  python -m eeg_pipeline.cli.main features compute --subject 0001

  # Machine Learning: run analysis
  python -m eeg_pipeline.cli.main ml --subject 0001 --subject 0002

For detailed help on each subcommand:
  python -m eeg_pipeline.cli.main <subcommand> --help
        """,
    )

    # Declared for --help, but removed from argv before this parser sees it (see
    # extract_config_path) so that it works in either position.
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "Configuration YAML to run with. Defaults to $EEG_PIPELINE_CONFIG, then the "
            "packaged eeg_config.yaml. Accepted before or after the subcommand."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis type")

    if only is None:
        for command in get_commands():
            command.setup(subparsers)
    else:
        command = get_command(only)
        if command is None:
            raise ValueError(f"Unknown command: {only}")
        command.setup(subparsers)

    return parser


def find_requested_command(argv: list[str]) -> str | None:
    """The registered command name in ``argv``, if it names one.

    Read positionally rather than by parsing, because parsing is what this decides the
    cost of. Only names are compared, and names are available from packaging metadata
    without importing anything.
    """
    names = set(get_command_names())
    for argument in argv:
        if argument in names:
            return argument
    return None


def extract_config_path(argv: list[str]) -> tuple[list[str], str | None]:
    """Pull ``--config PATH`` out of argv, wherever in it the user put it.

    Handled before argparse rather than as an ordinary option because the config file
    decides what the rest of the run means, and because a global option declared on the
    top-level parser would only be accepted *before* the subcommand — which is not where
    anyone types it.

    Returns the remaining arguments and the requested path, if any.
    """
    remaining: list[str] = []
    config_path: str | None = None
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--config":
            if index + 1 >= len(argv):
                raise SystemExit("--config requires a path to a configuration YAML file.")
            config_path = argv[index + 1]
            index += 2
            continue
        if argument.startswith("--config="):
            config_path = argument.split("=", 1)[1]
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return remaining, config_path


def update_config_from_args(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Update configuration dictionary with values from command-line arguments."""
    apply_runtime_overrides(
        config,
        task=getattr(args, "task", None),
        task_is_rest=getattr(args, "task_is_rest", None),
        source_root=getattr(args, "source_root", None),
        bids_root=getattr(args, "bids_root", None),
        bids_rest_root=getattr(args, "bids_rest_root", None),
        bids_fmri_root=getattr(args, "bids_fmri_root", None),
        deriv_root=getattr(args, "deriv_root", None),
        deriv_rest_root=getattr(args, "deriv_rest_root", None),
        set_overrides=getattr(args, "set_overrides", None),
    )


def get_subjects_for_command(
    args: argparse.Namespace, config: dict[str, Any], deriv_root: Path
) -> list[str]:
    """Parse and validate subject arguments for commands that require them."""
    return parse_subject_args(args, config, task=getattr(args, "task", None), deriv_root=deriv_root)


def execute_command(
    command: Command, args: argparse.Namespace, subjects: list[str], config: dict[str, Any]
) -> int:
    """Execute a command with error handling. Returns exit code."""
    try:
        command.run(args, subjects, config)
        return EXIT_SUCCESS
    except MissingCommandDependency as exc:
        # Not a failure of the run: the code to do it was never importable here. One
        # line, no traceback, because the stack is the import machinery rather than
        # anything the user can act on.
        logging.error("%s", exc)
        return EXIT_ERROR
    except Exception as e:
        logging.error("Error running %s: %s", command.name, e, exc_info=True)
        return EXIT_ERROR


def main() -> int:
    """Main entry point for the CLI application."""
    setup_logging()

    argv, config_path = extract_config_path(sys.argv[1:])

    parser = create_argument_parser(only=find_requested_command(argv))
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return EXIT_ERROR

    # Set process-wide before anything loads configuration: most of the pipeline reaches
    # it through an argument-less load_config(), so passing the path here alone would
    # leave those call sites reading the packaged default.
    # A bad path, a broken 'extends', or unparseable YAML is a mistake in the invocation,
    # not a crash in the pipeline. Report it as one line rather than a traceback through
    # the loader's internals.
    try:
        set_default_config_path(config_path)
        config = load_config()
    except ConfigError as exc:
        logging.error("%s", exc)
        return EXIT_ERROR

    update_config_from_args(config, args)
    deriv_root = get_deriv_root(config, command=args.command)

    command = get_command(args.command)
    if not command:
        logging.error("Unknown command: %s", args.command)
        return EXIT_ERROR

    if not command.requires_subjects:
        return execute_command(command, args, [], config)

    subjects = get_subjects_for_command(args, config, deriv_root)
    if not subjects:
        logging.error(
            "No subjects provided. Use --group all|A,B,C, "
            "or --subject (repeatable), or --all-subjects."
        )
        return EXIT_NO_SUBJECTS

    return execute_command(command, args, subjects, config)


if __name__ == "__main__":
    sys.exit(main())
