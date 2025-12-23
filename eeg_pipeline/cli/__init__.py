"""
CLI Module
==========

Unified command-line interface for the EEG pipeline.

All commands are registered in cli/commands.py.
The main entry point is cli/main.py.

Available commands:
- behavior: Behavior correlation analysis
- features: Feature extraction
- erp: Event-related potential analysis
- tfr: Time-frequency visualization
- decoding: ML-based prediction
- preprocessing: Raw-to-BIDS and behavior merge
- plotting: Curated visualization suites
"""

from eeg_pipeline.cli.commands import COMMANDS, Command, get_command
from eeg_pipeline.cli.main import main

__all__ = [
    "COMMANDS",
    "Command",
    "get_command",
    "main",
]
