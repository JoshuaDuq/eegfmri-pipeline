"""
Utilities Pipeline
==================

Pipeline class and orchestration functions for utility tasks:
- Raw-to-BIDS conversion
- Behavioral data merge

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
from eeg_pipeline.analysis.preprocessing.orchestration import (
    merge_behavior_to_events as _merge_behavior_to_events,
    run_merge_behavior as _run_merge_behavior,
    run_raw_to_bids as _run_raw_to_bids,
)
from eeg_pipeline.analysis.behavior.orchestration import (
    run_combine_features_utility as _run_combine_features,
)

logger = logging.getLogger(__name__)


###################################################################
# Orchestration Functions
###################################################################


def merge_behavior_to_events(
    events_tsv: Path,
    source_root: Path,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
) -> bool:
    """Merge behavioral data into a single events.tsv file."""
    return _merge_behavior_to_events(
        events_tsv=events_tsv,
        source_root=source_root,
        event_prefixes=event_prefixes,
        event_types=event_types,
        dry_run=dry_run,
        _logger=logger,
    )


def run_raw_to_bids(
    source_root: Path,
    bids_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    montage: str = "easycap-M1",
    line_freq: float = 60.0,
    overwrite: bool = False,
    zero_base_onsets: bool = False,
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
        zero_base_onsets=zero_base_onsets,
        do_trim_to_first_volume=do_trim_to_first_volume,
        event_prefixes=event_prefixes,
        keep_all_annotations=keep_all_annotations,
        _logger=logger,
    )


def run_merge_behavior(
    bids_root: Path,
    source_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
) -> int:
    """Merge behavioral data into BIDS events files."""
    return _run_merge_behavior(
        bids_root=bids_root,
        source_root=source_root,
        task=task,
        subjects=subjects,
        event_prefixes=event_prefixes,
        event_types=event_types,
        dry_run=dry_run,
        _logger=logger,
    )


###################################################################
# Pipeline Class
###################################################################


class UtilityPipeline(PipelineBase):
    """Pipeline for utility operations (raw-to-BIDS, merge-behavior).
    
    Unlike other pipelines, these utilities operate on source data or 
    BIDS raw data rather than per-subject derivatives.
    """
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="utilities", config=config)
        self.bids_root = Path(self.config.bids_root)
        self.source_root = Path(self.config.get("paths.source_data", "data/source_data"))

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        """Process a single subject through raw-to-BIDS and merge-behavior."""
        from eeg_pipeline.cli.common import ProgressReporter
        
        task = task or self.config.get("project.task", "thermalactive")
        do_trim = kwargs.get(
            "do_trim_to_first_volume",
            kwargs.get("trim_to_first_volume", False),
        )
        
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        total_steps = 2
        
        progress.subject_start(f"sub-{subject}")
        
        progress.step("Converting to BIDS", current=1, total=total_steps)
        self.logger.info(f"Processing sub-{subject}: raw-to-BIDS")
        run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=task,
            subjects=[subject],
            montage=kwargs.get("montage", self.config.get("eeg.montage", "easycap-M1")),
            line_freq=kwargs.get("line_freq", self.config.get("preprocessing.line_freq", 60.0)),
            overwrite=kwargs.get("overwrite", False),
            zero_base_onsets=kwargs.get("zero_base_onsets", False),
            do_trim_to_first_volume=do_trim,
            event_prefixes=kwargs.get("event_prefixes"),
            keep_all_annotations=kwargs.get("keep_all_annotations", False),
        )
        
        progress.step("Merging behavior", current=2, total=total_steps)
        self.logger.info(f"Processing sub-{subject}: merge-behavior")
        run_merge_behavior(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=task,
            event_prefixes=kwargs.get("event_prefixes"),
            event_types=kwargs.get("event_types"),
            dry_run=kwargs.get("dry_run", False),
        )
        
        progress.subject_done(f"sub-{subject}", success=True)

    def run_combine_features(
        self,
        subjects: List[str],
        categories: List[str],
        **kwargs,
    ) -> int:
        """Run feature combination for multiple subjects."""
        from eeg_pipeline.infra.paths import resolve_deriv_root
        
        deriv_root = resolve_deriv_root(config=self.config)
        progress = kwargs.get("progress")
        
        return _run_combine_features(
            subjects=subjects,
            categories=categories,
            deriv_root=deriv_root,
            config=self.config,
            logger=self.logger,
            progress=progress,
        )

    def run_batch(self, subjects: List[str], task: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Run utilities for multiple subjects."""
        from eeg_pipeline.cli.common import ProgressReporter
        
        task = task or self.config.get("project.task", "thermalactive")
        do_trim = kwargs.get(
            "do_trim_to_first_volume",
            kwargs.get("trim_to_first_volume", False),
        )
        
        progress = kwargs.get("progress") or ProgressReporter(enabled=False)
        total_steps = 2
        
        progress.start("utilities", subjects)
        
        progress.step("Converting to BIDS", current=1, total=total_steps)
        self.logger.info(f"Running raw-to-BIDS for {len(subjects)} subjects")
        n_converted = run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=task,
            subjects=subjects,
            montage=kwargs.get("montage", self.config.get("eeg.montage", "easycap-M1")),
            line_freq=kwargs.get("line_freq", self.config.get("preprocessing.line_freq", 60.0)),
            overwrite=kwargs.get("overwrite", False),
            zero_base_onsets=kwargs.get("zero_base_onsets", False),
            do_trim_to_first_volume=do_trim,
            event_prefixes=kwargs.get("event_prefixes"),
            keep_all_annotations=kwargs.get("keep_all_annotations", False),
        )
        
        progress.step("Merging behavior", current=2, total=total_steps)
        self.logger.info(f"Running merge-behavior for {len(subjects)} subjects")
        n_merged = run_merge_behavior(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=task,
            subjects=subjects,
            event_prefixes=kwargs.get("event_prefixes"),
            event_types=kwargs.get("event_types"),
            dry_run=kwargs.get("dry_run", False),
        )
        
        progress.complete(success=True)
        
        return [{
            "subjects": subjects,
            "n_converted": n_converted,
            "n_merged": n_merged,
            "status": "success",
        }]

    def run_raw_to_bids(
        self,
        task: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        montage: str = "easycap-M1",
        line_freq: float = 60.0,
        overwrite: bool = False,
        zero_base_onsets: bool = False,
        do_trim_to_first_volume: bool = False,
        event_prefixes: Optional[List[str]] = None,
        keep_all_annotations: bool = False,
    ) -> int:
        """Convert raw BrainVision files to BIDS format."""
        task = task or self.config.get("project.task", "thermalactive")
        return run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=task,
            subjects=subjects,
            montage=montage or self.config.get("eeg.montage", "easycap-M1"),
            line_freq=line_freq or self.config.get("preprocessing.line_freq", 60.0),
            overwrite=overwrite,
            zero_base_onsets=zero_base_onsets,
            do_trim_to_first_volume=do_trim_to_first_volume,
            event_prefixes=event_prefixes,
            keep_all_annotations=keep_all_annotations,
        )

    def run_merge_behavior(
        self,
        task: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        event_prefixes: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> int:
        """Merge behavioral data into BIDS events files."""
        task = task or self.config.get("project.task", "thermalactive")
        return run_merge_behavior(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=task,
            subjects=subjects,
            event_prefixes=event_prefixes,
            event_types=event_types,
            dry_run=dry_run,
        )


###################################################################
# Exports
###################################################################

__all__ = [
    "UtilityPipeline",
    "run_raw_to_bids",
    "run_merge_behavior",
    "merge_behavior_to_events",
]
