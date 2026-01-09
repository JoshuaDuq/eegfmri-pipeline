"""
Base Pipeline Orchestration.

Provides a standardized base class for analysis pipelines, handling:
- Batch processing loops
- Logging setup
- Error handling
- Configuration loading
"""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Dict

from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.infra.logging import (
    get_logger,
    get_subject_logger,
)
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.infra.paths import (
    ensure_derivatives_dataset_description,
    resolve_deriv_root,
)
from eeg_pipeline.utils.progress import BatchProgress

_MIN_SUBJECTS_FOR_GROUP_ANALYSIS = 2


class PipelineBase(ABC):
    """Base class for analysis pipelines."""

    def __init__(self, name: str, config: Optional[Any] = None):
        self.name = name
        self.config = config or load_config()
        self.logger = get_logger(f"eeg_pipeline.pipelines.{name}")
        self._setup()

    def _setup(self) -> None:
        """Perform initial setup."""
        setup_matplotlib(self.config)
        self.deriv_root = resolve_deriv_root(config=self.config)
        ensure_derivatives_dataset_description(deriv_root=self.deriv_root)

    def _validate_batch_inputs(
        self, subjects: List[str], task: Optional[str]
    ) -> str:
        """Validate batch processing inputs and return resolved task."""
        if not subjects:
            raise ValueError("No subjects specified")

        resolved_task = task or self.config.get("project.task")
        if resolved_task is None:
            raise ValueError("Missing required config value: project.task")

        return resolved_task

    def _write_traceback(self, subject: str, ledger_dir: Path) -> Path:
        """Write traceback to file and return path."""
        traceback_path = ledger_dir / f"sub-{subject}_{self.name}_traceback.log"
        traceback_path.write_text(traceback.format_exc())
        return traceback_path

    def _write_ledger(
        self, entries: List[Dict[str, Any]], ledger_path: Path
    ) -> None:
        """Write batch processing ledger to TSV file."""
        header = "subject\tstatus\tduration_s\terror\ttraceback_path"
        rows = [header]

        for entry in entries:
            error_text = str(entry.get("error", "")).replace("\n", " | ")
            traceback_text = str(entry.get("traceback_path", "")).replace(
                "\n", " | "
            )
            subject_id = entry.get("subject", "")
            status = entry.get("status", "")
            duration = entry.get("duration_s", "")

            row = (
                f"{subject_id}\t{status}\t{duration}\t"
                f"{error_text}\t{traceback_text}"
            )
            rows.append(row)

        ledger_path.write_text("\n".join(rows))

    def _complete_progress(
        self, progress: Any, success: bool
    ) -> None:
        """Complete progress tracking if available."""
        if progress is not None and hasattr(progress, "complete"):
            progress.complete(success=success)

    def _process_single_subject(
        self,
        subject: str,
        task: str,
        start_time: float,
        ledger: List[Dict[str, Any]],
        ledger_dir: Path,
        **kwargs: Any,
    ) -> None:
        """Process a single subject and update ledger."""
        try:
            self.process_subject(subject, task=task, **kwargs)
            duration = round(time.time() - start_time, 3)
            ledger.append(
                {
                    "subject": subject,
                    "status": "success",
                    "duration_s": duration,
                }
            )
        except Exception as exc:
            traceback_path = self._write_traceback(subject, ledger_dir)
            self.logger.error(
                f"Failed sub-{subject}: {exc} (traceback -> {traceback_path})"
            )
            duration = round(time.time() - start_time, 3)
            ledger.append(
                {
                    "subject": subject,
                    "status": "failed",
                    "duration_s": duration,
                    "error": str(exc),
                    "traceback_path": traceback_path,
                }
            )
            raise

    def _handle_batch_failures(
        self,
        ledger: List[Dict[str, Any]],
        subjects: List[str],
        ledger_path: Path,
        progress: Any,
    ) -> List[str]:
        """Handle failed subjects and return list of failed subject IDs."""
        failed_subjects = [
            entry["subject"]
            for entry in ledger
            if entry.get("status") == "failed"
        ]

        if not failed_subjects:
            return failed_subjects

        self.logger.warning(
            f"{len(failed_subjects)}/{len(subjects)} subjects failed: "
            f"{failed_subjects} (ledger: {ledger_path})"
        )

        if len(failed_subjects) == len(subjects):
            self._complete_progress(progress, success=False)
            raise RuntimeError("All subjects failed; see ledger for details.")

        return failed_subjects

    def run_batch(
        self,
        subjects: List[str],
        task: Optional[str] = None,
        *,
        fail_fast: bool = False,
        ledger_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run the pipeline for a batch of subjects, returning a per-subject ledger."""
        resolved_task = self._validate_batch_inputs(subjects, task)
        progress = kwargs.get("progress")

        if progress is not None and hasattr(progress, "start"):
            progress.start(self.name, subjects)

        ledger: List[Dict[str, Any]] = []
        default_ledger_path = (
            self.deriv_root / "logs" / f"{self.name}_batch_ledger.tsv"
        )
        resolved_ledger_path = ledger_path or default_ledger_path
        ledger_dir = resolved_ledger_path.parent
        ledger_dir.mkdir(parents=True, exist_ok=True)

        try:
            with BatchProgress(
                subjects=subjects,
                logger=self.logger,
                desc=self.name.title(),
            ) as batch:
                for subject in subjects:
                    start_time = batch.start_subject(subject)
                    try:
                        self._process_single_subject(
                            subject,
                            resolved_task,
                            start_time,
                            ledger,
                            ledger_dir,
                            **kwargs,
                        )
                        batch.finish_subject(subject, start_time)
                    except Exception:
                        batch.finish_subject(subject, start_time)
                        if fail_fast:
                            self._complete_progress(progress, success=False)
                            raise
        finally:
            self._write_ledger(ledger, resolved_ledger_path)

        failed_subjects = self._handle_batch_failures(
            ledger, subjects, resolved_ledger_path, progress
        )

        if len(subjects) >= _MIN_SUBJECTS_FOR_GROUP_ANALYSIS:
            self.run_group_level(subjects, task=resolved_task, **kwargs)

        all_succeeded = len(failed_subjects) == 0
        self._complete_progress(progress, success=all_succeeded)

        return ledger

    @abstractmethod
    def process_subject(self, subject: str, task: str, **kwargs: Any) -> None:
        """Process a single subject. Must be implemented by subclasses."""
        pass

    def run_group_level(self, subjects: List[str], task: str, **kwargs: Any) -> None:
        """Run group-level analysis. Optional."""
        pass

    def get_subject_logger(self, subject: str, filename: Optional[str] = None) -> logging.Logger:
        """Get a subject-specific logger."""
        return get_subject_logger(
            self.name,
            subject,
            filename or f"{self.name}.log",
            config=self.config,
        )
