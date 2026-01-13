"""
CLI Commands Package
=====================

Modular CLI command definitions. Each command module provides:
- setup function: Configure argparse parser
- run function: Execute the command

Commands are re-exported for registration in main.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, List


@dataclass
class Command:
    """CLI command definition."""
    name: str
    help: str
    description: str
    setup: Callable[[argparse._SubParsersAction], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace, List[str], Any], None]
    requires_subjects: bool = True


from eeg_pipeline.cli.commands.behavior import setup_behavior, run_behavior
from eeg_pipeline.cli.commands.features import setup_features, run_features
from eeg_pipeline.cli.commands.machine_learning import setup_ml, run_ml
from eeg_pipeline.cli.commands.preprocessing import setup_preprocessing, run_preprocessing
from fmri_pipeline.cli.commands.fmri import setup_fmri, run_fmri
from eeg_pipeline.cli.commands.utilities import setup_utilities, run_utilities
from eeg_pipeline.cli.commands.plotting import setup_plotting, run_plotting
from eeg_pipeline.cli.commands.info import setup_info, run_info
from eeg_pipeline.cli.commands.stats import setup_stats, run_stats
from eeg_pipeline.cli.commands.validate import setup_validate, run_validate

from eeg_pipeline.cli.commands.base import (
    detect_available_bands,
    detect_feature_availability,
    BEHAVIOR_COMPUTATIONS,
    FEATURE_VISUALIZE_CATEGORIES,
    BEHAVIOR_VISUALIZE_CATEGORIES,
    FREQUENCY_BANDS,
)


COMMANDS: List[Command] = [
    Command(
        name="behavior",
        help="Behavior analysis: compute correlations or visualize",
        description="Behavior pipeline: compute correlations or visualize",
        setup=setup_behavior,
        run=run_behavior,
    ),
    Command(
        name="features",
        help="Features analysis: extract features or visualize",
        description="Features pipeline: extract features or visualize",
        setup=setup_features,
        run=run_features,
    ),
    Command(
        name="ml",
        help="Machine learning: run LOSO regression and time-generalization",
        description="Run machine learning pipeline (LOSO regression + time-generalization)",
        setup=setup_ml,
        run=run_ml,
    ),
    Command(
        name="plotting",
        help="Plotting pipeline: curate visualization suites",
        description="Plotting pipeline: select and render visualization suites",
        setup=setup_plotting,
        run=run_plotting,
    ),
    Command(
        name="preprocessing",
        help="EEG preprocessing: bad channels, ICA, epochs",
        description="Run preprocessing: detect bad channels, fit ICA, create epochs",
        setup=setup_preprocessing,
        run=run_preprocessing,
    ),
    Command(
        name="fmri",
        help="fMRI preprocessing: fMRIPrep-style (containerized)",
        description="Run fMRIPrep-style preprocessing for a BIDS fMRI dataset",
        setup=setup_fmri,
        run=run_fmri,
        requires_subjects=False,
    ),
    Command(
        name="utilities",
        help="Utilities: raw-to-bids conversion, merge behavior, clean up disk space",
        description="Utilities pipeline: convert raw EEG to BIDS, merge behavioral data, or clean up disk space",
        setup=setup_utilities,
        run=run_utilities,
        requires_subjects=True,
    ),
    Command(
        name="info",
        help="Discovery and status: list subjects, features, config, discover columns",
        description="Show information about subjects, features, configuration, and discover available columns",
        setup=setup_info,
        run=run_info,
        requires_subjects=False,
    ),
    Command(
        name="stats",
        help="Project statistics: subjects, features, storage analytics",
        description="Show project-wide statistics and analytics",
        setup=setup_stats,
        run=run_stats,
        requires_subjects=False,
    ),
    Command(
        name="validate",
        help="Validate data integrity: epochs, features, BIDS compliance",
        description="Check data files for integrity and consistency issues",
        setup=setup_validate,
        run=run_validate,
        requires_subjects=False,
    ),
]


def get_command(name: str):
    """Get command by name."""
    for cmd in COMMANDS:
        if cmd.name == name:
            return cmd
    return None


__all__ = [
    "Command",
    "COMMANDS",
    "get_command",
    "detect_available_bands",
    "detect_feature_availability",
    "BEHAVIOR_COMPUTATIONS",
    "FEATURE_VISUALIZE_CATEGORIES",
    "BEHAVIOR_VISUALIZE_CATEGORIES",
    "FREQUENCY_BANDS",
]
