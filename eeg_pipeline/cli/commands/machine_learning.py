"""Machine Learning CLI command exports."""

from eeg_pipeline.cli.commands.machine_learning_orchestrator import run_ml
from eeg_pipeline.cli.commands.machine_learning_parser import setup_ml

__all__ = [
    "setup_ml",
    "run_ml",
]
