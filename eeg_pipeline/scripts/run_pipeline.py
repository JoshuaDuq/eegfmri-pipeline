"""Unified EEG Pipeline CLI.

Usage:
    python run_pipeline.py <command> <mode> [options]

Commands:
    preprocessing  Raw-to-BIDS conversion and behavior merge
    behavior       Brain-behavior correlation analysis
    features       Feature extraction from epochs
    erp            Event-related potential analysis
    tfr            Time-frequency visualization
    decoding       ML-based prediction

Examples:
    python run_pipeline.py preprocessing raw-to-bids --source-root data/source_data
    python run_pipeline.py preprocessing merge-behavior --task thermalactive
    python run_pipeline.py features compute --subject 0001
    python run_pipeline.py behavior compute --all-subjects
    python run_pipeline.py decoding --subject 0001 --subject 0002

CLI subcommand implementations are in eeg_pipeline/cli/*.py
"""

import os
os.environ["NUMPY_SKIP_MACOS_CHECK"] = "1"

import sys
import logging
from pathlib import Path
from typing import List, Optional, Any
import argparse

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import parse_subject_args
from eeg_pipeline.cli.common import get_deriv_root

from eeg_pipeline.cli import (
    setup_behavior_parser,
    run_behavior,
    setup_features_parser,
    run_features,
    setup_erp_parser,
    run_erp,
    setup_tfr_parser,
    run_tfr,
    setup_decoding_parser,
    run_decoding,
    setup_preprocessing_parser,
    run_preprocessing,
)


###################################################################
# Main Entry Point
###################################################################

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(
        description="Unified EEG Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Behavior: compute correlations
  python run_pipeline.py behavior compute --subject 0001

  # Features: extract and visualize
  python run_pipeline.py features compute --subject 0001
  python run_pipeline.py features visualize --subject 0001

  # ERP: compute statistics
  python run_pipeline.py erp compute --subject 0001

  # TFR: visualize
  python run_pipeline.py tfr visualize --subject 0001

  # Decoding: run analysis
  python run_pipeline.py decoding --subject 0001 --subject 0002

For detailed help on each subcommand:
  python run_pipeline.py <subcommand> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Analysis type")
    
    setup_behavior_parser(subparsers)
    setup_features_parser(subparsers)
    setup_erp_parser(subparsers)
    setup_tfr_parser(subparsers)
    setup_decoding_parser(subparsers)
    setup_preprocessing_parser(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    config = load_settings()
    deriv_root = get_deriv_root(config)
    
    if args.command == "preprocessing":
        run_preprocessing(args, config)
        return 0
    
    subjects = parse_subject_args(args, config, task=args.task, deriv_root=deriv_root)
    
    if not subjects:
        logging.error("No subjects provided. Use --group all|A,B,C, or --subject (repeatable), or --all-subjects.")
        return 2
    
    command_handlers = {
        "behavior": run_behavior,
        "features": run_features,
        "erp": run_erp,
        "tfr": run_tfr,
        "decoding": run_decoding,
    }
    
    handler = command_handlers.get(args.command)
    if not handler:
        logging.error("Unknown command: %s", args.command)
        return 1
    
    try:
        handler(args, subjects, config)
        return 0
    except Exception as e:
        logging.error("Error running %s: %s", args.command, e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

