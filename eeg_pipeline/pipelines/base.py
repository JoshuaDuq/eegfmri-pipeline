"""
Base Pipeline Orchestration.

Provides a standardized base class for analysis pipelines, handling:
- Batch processing loops
- Logging setup
- Error handling
- Configuration loading
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Dict
from uuid import uuid4

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
        self.deriv_root = self._resolve_pipeline_deriv_root()
        ensure_derivatives_dataset_description(deriv_root=self.deriv_root)

    def _resolve_pipeline_deriv_root(self) -> Path:
        """Resolve the derivatives root used by this pipeline."""
        return resolve_deriv_root(config=self.config)

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
            traceback_path = str(entry.get("traceback_path", ""))
            subject_id = entry.get("subject", "")
            status = entry.get("status", "")
            duration = entry.get("duration_s", "")

            row = (
                f"{subject_id}\t{status}\t{duration}\t"
                f"{error_text}\t{traceback_path}"
            )
            rows.append(row)

        ledger_path.write_text("\n".join(rows))

    def _complete_progress(self, progress: Any, success: bool) -> None:
        """Complete progress tracking if available."""
        if progress is not None:
            progress.complete(success=success)

    def _sanitize_metadata_value(self, value: Any, depth: int = 0) -> Any:
        """Convert arbitrary Python values to JSON-safe metadata payloads."""
        if depth > 8:
            return repr(value)

        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, Path):
            return str(value)

        if is_dataclass(value):
            try:
                return self._sanitize_metadata_value(asdict(value), depth + 1)
            except Exception:
                return repr(value)

        if isinstance(value, dict):
            return {
                str(k): self._sanitize_metadata_value(v, depth + 1)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_metadata_value(v, depth + 1) for v in value]

        # Compact fallback for arrays/dataframes/objects without serializing full data.
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            return {
                "__type__": type(value).__name__,
                "shape": list(shape) if shape is not None else None,
                "dtype": str(dtype) if dtype is not None else None,
            }

        if hasattr(value, "__dict__"):
            try:
                return self._sanitize_metadata_value(vars(value), depth + 1)
            except Exception:
                return repr(value)

        return repr(value)

    def _create_run_metadata_context(
        self,
        *,
        subjects: List[str],
        task: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare metadata context at pipeline run start."""
        started_at = datetime.now(timezone.utc)
        run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

        safe_specs = {
            key: value
            for key, value in kwargs.items()
            if key != "progress"
        }

        return {
            "run_id": run_id,
            "started_at": started_at,
            "task": task,
            "subjects": list(subjects),
            "specifications": self._sanitize_metadata_value(safe_specs),
        }

    def _write_run_metadata(
        self,
        run_context: Dict[str, Any],
        *,
        status: str,
        error: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist run-level reproducibility metadata as JSON."""
        pipeline_name = getattr(self, "name", type(self).__name__)
        deriv_root = getattr(self, "deriv_root", None)
        if deriv_root is None:
            raise RuntimeError(
                f"Cannot write run metadata for {pipeline_name}: deriv_root is not set."
            )

        started_at = run_context["started_at"]
        finished_at = datetime.now(timezone.utc)
        duration_seconds = round(
            (finished_at - started_at).total_seconds(),
            3,
        )

        metadata_dir = (
            Path(deriv_root)
            / "logs"
            / "run_metadata"
            / pipeline_name
        )
        metadata_dir.mkdir(parents=True, exist_ok=True)

        raw_config: Any = getattr(self, "config", {})
        if isinstance(raw_config, dict):
            raw_config = dict(raw_config)

        payload = {
            "schema_version": "1.0",
            "run_id": run_context["run_id"],
            "pipeline": pipeline_name,
            "status": status,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "task": run_context.get("task"),
            "subjects": run_context.get("subjects", []),
            "specifications": run_context.get("specifications", {}),
            "config": self._sanitize_metadata_value(raw_config),
            "environment": {
                "cwd": os.getcwd(),
                "argv": list(sys.argv),
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "outputs": self._sanitize_metadata_value(outputs or {}),
            "summary": self._sanitize_metadata_value(summary or {}),
        }
        if error:
            payload["error"] = str(error)

        out_path = metadata_dir / f"run_{run_context['run_id']}.json"
        out_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return out_path

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
        run_context = self._create_run_metadata_context(
            subjects=subjects,
            task=resolved_task,
            kwargs=kwargs,
        )
        run_status = "failed"
        run_error: Optional[str] = None
        caught_error: Optional[Exception] = None

        if progress is not None:
            progress.start(self.name, subjects)

        ledger: List[Dict[str, Any]] = []
        default_ledger_path = (
            self.deriv_root / "logs" / f"{self.name}_batch_ledger.tsv"
        )
        resolved_ledger_path = ledger_path or default_ledger_path
        ledger_dir = resolved_ledger_path.parent
        ledger_dir.mkdir(parents=True, exist_ok=True)

        try:
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
            run_status = "success" if all_succeeded else "partial_success"
            self._complete_progress(progress, success=all_succeeded)

        except Exception as exc:
            run_error = str(exc)
            caught_error = exc

        summary = {
            "n_subjects": len(subjects),
            "n_success": sum(
                1 for item in ledger if item.get("status") == "success"
            ),
            "n_failed": sum(
                1 for item in ledger if item.get("status") == "failed"
            ),
        }
        outputs = {"ledger_path": str(resolved_ledger_path)}

        metadata_error: Optional[Exception] = None
        try:
            self._write_run_metadata(
                run_context,
                status=run_status,
                error=run_error,
                outputs=outputs,
                summary=summary,
            )
        except Exception as exc:
            metadata_error = exc

        if caught_error is not None:
            if metadata_error is not None:
                caught_error.add_note(f"Run metadata writing also failed: {metadata_error}")
            raise caught_error
        if metadata_error is not None:
            raise metadata_error

        return ledger

    @abstractmethod
    def process_subject(self, subject: str, task: str, **kwargs: Any) -> None:
        """Process a single subject. Must be implemented by subclasses."""
        pass

    def run_group_level(self, subjects: List[str], task: str, **kwargs: Any) -> None:
        """Run group-level analysis. Optional."""
        pass

    def get_subject_logger(self, subject: str) -> logging.Logger:
        """Get a subject-specific logger."""
        return get_subject_logger(self.name, subject)
