import argparse
from typing import Optional
from pathlib import Path

from ..config.loader import ConfigDict
from ..data.loading import parse_subject_args


###################################################################
# Argument Parser Helpers
###################################################################

def add_subject_args(parser: argparse.ArgumentParser, required: bool = False) -> argparse._MutuallyExclusiveGroup:
    """
    Add standard subject selection arguments to an argument parser.
    
    Args:
        parser: ArgumentParser instance
        required: Whether subject selection is required
        
    Returns:
        The mutually exclusive group containing subject arguments
    """
    subject_group = parser.add_mutually_exclusive_group(required=required)
    subject_group.add_argument(
        "--group",
        type=str,
        help="Group of subjects: 'all' or comma-separated list"
    )
    subject_group.add_argument(
        "--subject", "-s",
        type=str,
        action="append",
        help="Subject label(s) without 'sub-' prefix"
    )
    subject_group.add_argument(
        "--all-subjects",
        action="store_true",
        help="Process all available subjects"
    )
    return subject_group


def add_task_args(parser: argparse.ArgumentParser, default_help: str = "Task label (default from config)") -> None:
    """
    Add standard task argument to an argument parser.
    
    Args:
        parser: ArgumentParser instance
        default_help: Help text for the task argument
    """
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help=default_help
    )


###################################################################
# Argument Parsing Helpers
###################################################################

def parse_script_args(
    args: argparse.Namespace,
    config: ConfigDict,
    task: Optional[str] = None,
    deriv_root: Optional[Path] = None,
    error_message: Optional[str] = None,
) -> list[str]:
    """
    Parse subject arguments from parsed command-line arguments.
    
    Args:
        args: Parsed arguments from argparse
        config: Configuration dictionary
        task: Optional task override
        deriv_root: Optional derivatives root path
        error_message: Custom error message if no subjects found
        
    Returns:
        List of subject IDs
        
    Raises:
        SystemExit: If no subjects are provided
    """
    subjects = parse_subject_args(args, config, task=task, deriv_root=deriv_root)
    
    if not subjects:
        if error_message is None:
            error_message = (
                "No subjects provided. Use --group all|A,B,C, or --subject (repeatable), "
                "or --all-subjects."
            )
        print(error_message)
        raise SystemExit(2)
    
    return subjects

