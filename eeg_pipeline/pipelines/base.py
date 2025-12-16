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
from typing import Any, List, Optional, TypeVar, Generic, Dict

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.infra.logging import (
    get_logger,
    get_subject_logger,
)
from eeg_pipeline.plotting.io.figures import setup_matplotlib
from eeg_pipeline.infra.paths import ensure_derivatives_dataset_description, resolve_deriv_root
from eeg_pipeline.utils.progress import BatchProgress

T = TypeVar("T")


class PipelineBase(ABC):
    """Base class for analysis pipelines."""

    def __init__(self, name: str, config: Optional[Any] = None):
        self.name = name
        self.config = config or load_settings()
        self.logger = get_logger(f"eeg_pipeline.pipelines.{name}")
        self._setup()

    def _setup(self) -> None:
        """Perform initial setup."""
        setup_matplotlib(self.config)
        self.deriv_root = resolve_deriv_root(config=self.config)
        ensure_derivatives_dataset_description(deriv_root=self.deriv_root)

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
        if not subjects:
            raise ValueError("No subjects specified")

        task = task or self.config.get("project.task")
        if task is None:
            raise ValueError("Missing required config value: project.task")

        ledger: List[Dict[str, Any]] = []
        ledger_path = ledger_path or (self.deriv_root / "logs" / f"{self.name}_batch_ledger.tsv")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        def _write_traceback(subject: str) -> Path:
            tb_path = ledger_path.parent / f"sub-{subject}_{self.name}_traceback.log"
            tb_path.write_text(traceback.format_exc())
            return tb_path

        def _write_ledger(entries: List[Dict[str, Any]]) -> None:
            rows = ["subject\tstatus\tduration_s\terror\ttraceback_path"]
            for entry in entries:
                err_text = str(entry.get("error", "")).replace("\n", " | ")
                tb_text = str(entry.get("traceback_path", "")).replace("\n", " | ")
                rows.append(
                    f"{entry.get('subject','')}\t{entry.get('status','')}\t"
                    f"{entry.get('duration_s','')}\t{err_text}\t{tb_text}"
                )
            ledger_path.write_text("\n".join(rows))

        with BatchProgress(subjects=subjects, logger=self.logger, desc=self.name.title()) as batch:
            for subject in subjects:
                start_time = batch.start_subject(subject)
                try:
                    self.process_subject(subject, task=task, **kwargs)
                    batch.finish_subject(subject, start_time)
                    ledger.append(
                        {
                            "subject": subject,
                            "status": "success",
                            "duration_s": round(time.time() - start_time, 3),
                        }
                    )
                except Exception as exc:
                    tb_path = _write_traceback(subject)
                    self.logger.error(f"Failed sub-{subject}: {exc} (traceback -> {tb_path})")
                    batch.finish_subject(subject, start_time)
                    ledger.append(
                        {
                            "subject": subject,
                            "status": "failed",
                            "duration_s": round(time.time() - start_time, 3),
                            "error": str(exc),
                            "traceback_path": tb_path,
                        }
                    )
                    if fail_fast:
                        _write_ledger(ledger)
                        raise

        _write_ledger(ledger)

        failed_subjects = [row["subject"] for row in ledger if row.get("status") == "failed"]
        if failed_subjects:
            self.logger.warning(
                f"{len(failed_subjects)}/{len(subjects)} subjects failed: {failed_subjects} "
                f"(ledger: {ledger_path})"
            )
            if len(failed_subjects) == len(subjects):
                raise RuntimeError("All subjects failed; see ledger for details.")

        if len(subjects) >= 2:
            self.run_group_level(subjects, task=task, **kwargs)

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
