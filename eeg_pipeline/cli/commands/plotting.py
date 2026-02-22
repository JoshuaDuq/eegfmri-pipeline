"""Plotting orchestration CLI command."""

from eeg_pipeline.cli.commands.plotting_orchestrator import run_plotting
from eeg_pipeline.cli.commands.plotting_parser import setup_plotting

__all__ = [
    "setup_plotting",
    "run_plotting",
]
