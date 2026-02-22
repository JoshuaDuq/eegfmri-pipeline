"""Behavior analysis CLI command exports."""

from eeg_pipeline.cli.commands.behavior_orchestrator import run_behavior
from eeg_pipeline.cli.commands.behavior_parser import setup_behavior

__all__ = [
    "setup_behavior",
    "run_behavior",
]
