"""Cohort preprocessing QC report command exports."""

from eeg_pipeline.cli.commands.cohort_report_orchestrator import run_cohort_report
from eeg_pipeline.cli.commands.cohort_report_parser import setup_cohort_report

__all__ = [
    "setup_cohort_report",
    "run_cohort_report",
]
