"""
Unified EEG Pipeline CLI
========================

Single entry point for all pipeline commands.

Usage:
    python -m eeg_pipeline.cli.main <command> [options]
    
    Or via console script (if installed):
    eeg-pipeline <command> [options]

Commands:
    utilities      Raw-to-BIDS conversion and behavior merge
    behavior       Brain-behavior correlation analysis
    features       Feature extraction from epochs
    tfr            Time-frequency visualization
    ml             Machine learning-based prediction

Examples:
    python -m eeg_pipeline.cli.main utilities raw-to-bids --source-root data/source_data
    python -m eeg_pipeline.cli.main features compute --subject 0001
    python -m eeg_pipeline.cli.main behavior compute --all-subjects
    python -m eeg_pipeline.cli.main ml --subject 0001 --subject 0002
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import warnings
from typing import Any

from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.utils.data.subjects import parse_subject_args
from eeg_pipeline.cli.common import get_deriv_root
from eeg_pipeline.cli.commands import COMMANDS, get_command, Command


os.environ["NUMPY_SKIP_MACOS_CHECK"] = "1"
warnings.filterwarnings(
    "ignore",
    message=".*found in sys.modules.*",
    category=RuntimeWarning,
    module="runpy"
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


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="Unified EEG Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Behavior: compute correlations
  python -m eeg_pipeline.cli.main behavior compute --subject 0001

  # Features: extract and visualize
  python -m eeg_pipeline.cli.main features compute --subject 0001
  python -m eeg_pipeline.cli.main features visualize --subject 0001

  # TFR: visualize
  python -m eeg_pipeline.cli.main tfr visualize --subject 0001

  # Machine Learning: run analysis
  python -m eeg_pipeline.cli.main ml --subject 0001 --subject 0002

For detailed help on each subcommand:
  python -m eeg_pipeline.cli.main <subcommand> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Analysis type")
    
    for command in COMMANDS:
        command.setup(subparsers)
    
    return parser


def update_config_from_args(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Update configuration dictionary with values from command-line arguments."""
    paths = config.setdefault("paths", {})
    
    if getattr(args, "bids_root", None):
        paths["bids_root"] = args.bids_root
    if getattr(args, "source_root", None):
        paths["source_data"] = args.source_root
    if getattr(args, "deriv_root", None):
        paths["deriv_root"] = args.deriv_root


def get_subjects_for_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    deriv_root: Any
) -> list[str]:
    """Parse and validate subject arguments for commands that require them."""
    subjects = parse_subject_args(
        args,
        config,
        task=getattr(args, "task", None),
        deriv_root=deriv_root
    )
    
    if not subjects:
        logging.error(
            "No subjects provided. Use --group all|A,B,C, "
            "or --subject (repeatable), or --all-subjects."
        )
    
    return subjects


def execute_command(
    command: Command,
    args: argparse.Namespace,
    subjects: list[str],
    config: dict[str, Any]
) -> int:
    """Execute a command with error handling. Returns exit code."""
    try:
        command.run(args, subjects, config)
        return EXIT_SUCCESS
    except Exception as e:
        logging.error("Error running %s: %s", command.name, e, exc_info=True)
        return EXIT_ERROR


def main() -> int:
    """Main entry point for the CLI application."""
    setup_logging()
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return EXIT_ERROR
    
    config = load_config()
    update_config_from_args(config, args)
    deriv_root = get_deriv_root(config)
    
    command = get_command(args.command)
    if not command:
        logging.error("Unknown command: %s", args.command)
        return EXIT_ERROR
    
    if not command.requires_subjects:
        return execute_command(command, args, [], config)
    
    subjects = get_subjects_for_command(args, config, deriv_root)
    if not subjects:
        return EXIT_NO_SUBJECTS
    
    return execute_command(command, args, subjects, config)


if __name__ == "__main__":
    sys.exit(main())
