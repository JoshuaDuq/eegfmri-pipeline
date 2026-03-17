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
    setup: Callable[[argparse._SubParsersAction], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace, List[str], Any], None]
    requires_subjects: bool = True


from eeg_pipeline.cli.commands.behavior import setup_behavior, run_behavior
from eeg_pipeline.cli.commands.features import setup_features, run_features
from eeg_pipeline.cli.commands.info import setup_info, run_info
from eeg_pipeline.cli.commands.machine_learning import setup_ml, run_ml
from eeg_pipeline.cli.commands.plotting import setup_plotting, run_plotting
from eeg_pipeline.cli.commands.preprocessing import setup_preprocessing, run_preprocessing
from eeg_pipeline.cli.commands.stats import setup_stats, run_stats
from eeg_pipeline.cli.commands.validate import setup_validate, run_validate
from fmri_pipeline.cli.commands.fmri import setup_fmri, run_fmri
from fmri_pipeline.cli.commands.fmri_analysis import setup_fmri_analysis, run_fmri_analysis
from studies.pain_study.cli.coupling import setup_coupling, run_coupling


COMMANDS: List[Command] = [
    Command(
        name="behavior",
        setup=setup_behavior,
        run=run_behavior,
    ),
    Command(
        name="coupling",
        setup=setup_coupling,
        run=run_coupling,
    ),
    Command(
        name="features",
        setup=setup_features,
        run=run_features,
    ),
    Command(
        name="fmri",
        setup=setup_fmri,
        run=run_fmri,
        requires_subjects=False,
    ),
    Command(
        name="fmri-analysis",
        setup=setup_fmri_analysis,
        run=run_fmri_analysis,
        requires_subjects=False,
    ),
    Command(
        name="info",
        setup=setup_info,
        run=run_info,
        requires_subjects=False,
    ),
    Command(
        name="ml",
        setup=setup_ml,
        run=run_ml,
    ),
    Command(
        name="plotting",
        setup=setup_plotting,
        run=run_plotting,
    ),
    Command(
        name="preprocessing",
        setup=setup_preprocessing,
        run=run_preprocessing,
    ),
    Command(
        name="stats",
        setup=setup_stats,
        run=run_stats,
        requires_subjects=False,
    ),
    Command(
        name="validate",
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


__all__ = ["Command", "COMMANDS", "get_command"]
