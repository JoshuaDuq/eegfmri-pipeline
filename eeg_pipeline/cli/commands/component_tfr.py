"""ICA-component TFR CLI command exports."""

from eeg_pipeline.cli.commands.component_tfr_orchestrator import run_component_tfr
from eeg_pipeline.cli.commands.component_tfr_parser import setup_component_tfr

__all__ = ["run_component_tfr", "setup_component_tfr"]
