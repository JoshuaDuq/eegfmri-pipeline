"""Utilities CLI command (raw-to-bids, merge-psychopy, clean)."""

from eeg_pipeline.cli.commands.utilities_orchestrator import run_utilities
from eeg_pipeline.cli.commands.utilities_parser import setup_utilities

__all__ = ["setup_utilities", "run_utilities"]
