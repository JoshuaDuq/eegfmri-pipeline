"""Info and discovery CLI command exports."""

from eeg_pipeline.cli.commands.info_orchestrator import run_info
from eeg_pipeline.cli.commands.info_parser import setup_info

__all__ = [
    "setup_info",
    "run_info",
]
