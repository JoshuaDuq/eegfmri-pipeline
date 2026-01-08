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
os.environ["NUMPY_SKIP_MACOS_CHECK"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*found in sys.modules.*", category=RuntimeWarning, module="runpy")

import sys
import logging
import argparse

from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.utils.data.subjects import parse_subject_args
from eeg_pipeline.cli.common import get_deriv_root
from eeg_pipeline.cli.commands import COMMANDS, get_command


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
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
    
    for cmd in COMMANDS:
        cmd.setup(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    config = load_config()

    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "source_root", None):
        config.setdefault("paths", {})["source_data"] = args.source_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root

    deriv_root = get_deriv_root(config)
    
    cmd = get_command(args.command)
    if not cmd:
        logging.error("Unknown command: %s", args.command)
        return 1
    
    if not cmd.requires_subjects:
        try:
            cmd.run(args, [], config)
            return 0
        except Exception as e:
            logging.error("Error running %s: %s", args.command, e, exc_info=True)
            return 1
    
    subjects = parse_subject_args(args, config, task=args.task, deriv_root=deriv_root)
    
    if not subjects:
        logging.error("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        return 2
    
    try:
        cmd.run(args, subjects, config)
        return 0
    except Exception as e:
        logging.error("Error running %s: %s", args.command, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
