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
- preprocessing: EEG preprocessing (bad channels, ICA, epochs)
- stats: Statistical analysis
- validate: BIDS validation
"""

__all__ = ["COMMANDS", "Command", "get_command", "main"]


def __getattr__(name: str):
    # Lazy import to avoid circular imports when submodules import eeg_pipeline.cli.common.
    if name in {"COMMANDS", "Command", "get_command"}:
        from eeg_pipeline.cli.commands import COMMANDS, Command, get_command

        return {"COMMANDS": COMMANDS, "Command": Command, "get_command": get_command}[name]
    if name == "main":
        from eeg_pipeline.cli.main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
