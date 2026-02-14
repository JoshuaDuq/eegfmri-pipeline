"""
Utilities Pipeline
==================

Pipeline class and orchestration functions for utility tasks:
- Raw-to-BIDS conversion
- PsychoPy-to-events merge

Usage:
    pipeline = UtilityPipeline(config=config)
    pipeline.run_batch(["0001", "0002"], task="thermalactive")

Low-level helpers (file discovery, annotation filtering) are in utils/data/preprocessing.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.analysis.utilities.eeg_raw_to_bids import (
    run_raw_to_bids as _run_raw_to_bids,
)
from eeg_pipeline.analysis.utilities.merge_psychopy import (
    run_merge_psychopy as _run_merge_psychopy,
)


logger = logging.getLogger(__name__)


def run_raw_to_bids(
    source_root: Path,
    bids_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    montage: str = "easycap-M1",
    line_freq: float = 60.0,
    overwrite: bool = False,
    do_trim_to_first_volume: bool = False,
    event_prefixes: Optional[List[str]] = None,
    keep_all_annotations: bool = False,
) -> int:
    """Convert raw BrainVision files to BIDS format."""
    return _run_raw_to_bids(
        source_root=source_root,
        bids_root=bids_root,
        task=task,
        subjects=subjects,
        montage=montage,
        line_freq=line_freq,
        overwrite=overwrite,
        do_trim_to_first_volume=do_trim_to_first_volume,
        event_prefixes=event_prefixes,
        keep_all_annotations=keep_all_annotations,
        _logger=logger,
    )


def run_merge_psychopy(
    bids_root: Path,
    source_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
    allow_misaligned_trim: bool = False,
) -> int:
    """Merge PsychoPy behavioral data into BIDS events files."""
    return _run_merge_psychopy(
        bids_root=bids_root,
        source_root=source_root,
        task=task,
        subjects=subjects,
        event_prefixes=event_prefixes,
        event_types=event_types,
        dry_run=dry_run,
        allow_misaligned_trim=allow_misaligned_trim,
        _logger=logger,
    )


class UtilityPipeline(PipelineBase):
    """Pipeline for utility operations (raw-to-BIDS, merge-psychopy).

    Unlike other pipelines, these utilities operate on source data or
    BIDS raw data rather than per-subject derivatives.
    """

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="utilities", config=config)
        self.bids_root = Path(self.config.bids_root)
        default_source = "data/source_data"
        self.source_root = Path(
            self.config.get("paths.source_data", default_source)
        )

    def _extract_raw_to_bids_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and normalize kwargs for raw-to-BIDS conversion."""
        montage = kwargs.get("montage")
        if montage is None:
            montage = self.config.get("eeg.montage", "easycap-M1")
        
        line_freq = kwargs.get("line_freq")
        if line_freq is None:
            line_freq = self.config.get("preprocessing.line_freq", 60.0)
        
        return {
            "montage": montage,
            "line_freq": line_freq,
            "overwrite": kwargs.get("overwrite", False),
            "do_trim_to_first_volume": kwargs.get("do_trim_to_first_volume", False),
            "event_prefixes": kwargs.get("event_prefixes"),
            "keep_all_annotations": kwargs.get("keep_all_annotations", False),
        }

    def _extract_merge_psychopy_kwargs(
        self, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and normalize kwargs for merge-psychopy operation."""
        return {
            "event_prefixes": kwargs.get("event_prefixes"),
            "event_types": kwargs.get("event_types"),
            "dry_run": kwargs.get("dry_run", False),
            "allow_misaligned_trim": kwargs.get(
                "allow_misaligned_trim",
                bool(self.config.get("alignment.allow_misaligned_trim", False)),
            ),
        }

    def _resolve_task(self, task: Optional[str]) -> str:
        """Resolve task name from argument or config."""
        return task or self.config.get("project.task", "thermalactive")

    def process_subject(
        self, subject: str, task: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Process a single subject through raw-to-BIDS and merge-psychopy."""
        from eeg_pipeline.cli.common import ProgressReporter

        resolved_task = self._resolve_task(task)
        raw_to_bids_kwargs = self._extract_raw_to_bids_kwargs(kwargs)
        merge_kwargs = self._extract_merge_psychopy_kwargs(kwargs)

        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        subject_id = f"sub-{subject}"

        import time as _time

        self.logger.info("=== Utilities: %s, task-%s ===", subject_id, resolved_task)
        progress.subject_start(subject_id)

        progress.step("Converting to BIDS", current=1, total=2)
        t0 = _time.perf_counter()
        run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=resolved_task,
            subjects=[subject],
            **raw_to_bids_kwargs,
        )
        self.logger.info("Raw-to-BIDS complete (%.1fs)", _time.perf_counter() - t0)

        progress.step("Merging behavior", current=2, total=2)
        t1 = _time.perf_counter()
        run_merge_psychopy(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=resolved_task,
            **merge_kwargs,
        )
        self.logger.info("Merge-psychopy complete (%.1fs)", _time.perf_counter() - t1)

        self.logger.info(
            "Utilities done for %s (%.1fs total)",
            subject_id, _time.perf_counter() - t0,
        )
        progress.subject_done(subject_id, success=True)

    def run_batch(
        self, subjects: List[str], task: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Run utilities for multiple subjects."""
        from eeg_pipeline.cli.common import ProgressReporter

        resolved_task = self._resolve_task(task)
        raw_to_bids_kwargs = self._extract_raw_to_bids_kwargs(kwargs)
        merge_kwargs = self._extract_merge_psychopy_kwargs(kwargs)

        progress = kwargs.get("progress") or ProgressReporter(enabled=False)

        progress.start("utilities", subjects)

        import time as _time

        self.logger.info(
            "=== Batch utilities: %d subjects, task-%s ===",
            len(subjects), resolved_task,
        )

        progress.step("Converting to BIDS", current=1, total=2)
        t0 = _time.perf_counter()
        n_converted = run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=resolved_task,
            subjects=subjects,
            **raw_to_bids_kwargs,
        )
        self.logger.info(
            "Raw-to-BIDS: %d/%d subjects converted (%.1fs)",
            n_converted, len(subjects), _time.perf_counter() - t0,
        )

        progress.step("Merging behavior", current=2, total=2)
        t1 = _time.perf_counter()
        n_merged = run_merge_psychopy(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=resolved_task,
            subjects=subjects,
            **merge_kwargs,
        )
        self.logger.info(
            "Merge-psychopy: %d/%d subjects merged (%.1fs)",
            n_merged, len(subjects), _time.perf_counter() - t1,
        )

        progress.complete(success=True)

        return [
            {
                "subjects": subjects,
                "n_converted": n_converted,
                "n_merged": n_merged,
                "status": "success",
            }
        ]

    def run_raw_to_bids(
        self,
        task: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        montage: Optional[str] = None,
        line_freq: Optional[float] = None,
        overwrite: bool = False,
        do_trim_to_first_volume: bool = False,
        event_prefixes: Optional[List[str]] = None,
        keep_all_annotations: bool = False,
    ) -> int:
        """Convert raw BrainVision files to BIDS format."""
        resolved_task = self._resolve_task(task)
        kwargs = self._extract_raw_to_bids_kwargs({
            "montage": montage,
            "line_freq": line_freq,
            "overwrite": overwrite,
            "do_trim_to_first_volume": do_trim_to_first_volume,
            "event_prefixes": event_prefixes,
            "keep_all_annotations": keep_all_annotations,
        })
        return run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=resolved_task,
            subjects=subjects,
            **kwargs,
        )

    def run_merge_psychopy(
        self,
        task: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        event_prefixes: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> int:
        """Merge PsychoPy behavioral data into BIDS events files."""
        resolved_task = self._resolve_task(task)
        kwargs = self._extract_merge_psychopy_kwargs({
            "event_prefixes": event_prefixes,
            "event_types": event_types,
            "dry_run": dry_run,
        })
        return run_merge_psychopy(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=resolved_task,
            subjects=subjects,
            **kwargs,
        )


__all__ = [
    "UtilityPipeline",
    "run_raw_to_bids",
    "run_merge_psychopy",
]
