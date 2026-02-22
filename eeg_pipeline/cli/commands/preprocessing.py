"""Preprocessing CLI command exports."""

from eeg_pipeline.cli.commands.preprocessing_orchestrator import run_preprocessing
from eeg_pipeline.cli.commands.preprocessing_parser import setup_preprocessing

__all__ = [
    "setup_preprocessing",
    "run_preprocessing",
]
