"""Pipeline wrapper for trial-wise cortical EEG-BOLD coupling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from studies.pain_study.analysis.eeg_bold_coupling import (
    run_group_eeg_bold_coupling,
    run_subject_eeg_bold_coupling,
)
from eeg_pipeline.pipelines.base import PipelineBase


class EEGBOLDCouplingPipeline(PipelineBase):
    """Run subject-level and group-level EEG-BOLD coupling analyses."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="eeg_bold_coupling", config=config)

    def process_subject(self, subject: str, task: str, **kwargs: Any) -> None:
        subject_logger = self.get_subject_logger(subject)
        run_subject_eeg_bold_coupling(
            subject=subject,
            task=task,
            config=self.config,
            logger=subject_logger,
        )

    def run_group_level(self, subjects: List[str], task: str, **kwargs: Any) -> None:
        group_path = run_group_eeg_bold_coupling(
            subjects=subjects,
            task=task,
            config=self.config,
            logger=self.logger,
        )
        if group_path is not None:
            self.logger.info("Wrote group EEG-BOLD coupling results: %s", group_path)


def run_coupling_for_subjects(
    *,
    subjects: List[str],
    task: str,
    config: Optional[Any] = None,
    ledger_path: Optional[Path] = None,
    progress: Optional[Any] = None,
) -> List[dict[str, Any]]:
    """Convenience entry point for batch EEG-BOLD coupling runs."""
    pipeline = EEGBOLDCouplingPipeline(config=config)
    return pipeline.run_batch(
        subjects=subjects,
        task=task,
        ledger_path=ledger_path,
        progress=progress,
    )


__all__ = [
    "EEGBOLDCouplingPipeline",
    "run_coupling_for_subjects",
]
