"""
CLI Module
==========

Unified command-line interface for the EEG pipeline.

All commands are registered in cli/commands/.
The main entry point is cli/main.py.

Available commands:
- behavior: Behavior correlation analysis
- features: Feature extraction
- fmri: fMRI preprocessing
- info: Dataset information and validation
- ml: Machine learning-based prediction
- plotting: Curated visualization suites
- preprocessing: Raw-to-BIDS and behavior merge
- stats: Statistical analysis
- utilities: Raw-to-BIDS conversion and behavior merge
- validate: BIDS validation
"""

from eeg_pipeline.cli.commands import COMMANDS, Command, get_command
from eeg_pipeline.cli.main import main

__all__ = [
    "COMMANDS",
    "Command",
    "get_command",
    "main",
]
