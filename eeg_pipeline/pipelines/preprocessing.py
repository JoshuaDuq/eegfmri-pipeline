"""
Preprocessing Pipeline
======================

Pipeline class and orchestration functions for preprocessing EEG data:
- Raw-to-BIDS conversion
- Behavioral data merge

Usage:
    pipeline = PreprocessingPipeline(config=config)
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
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
) -> int:
    """Merge behavioral data into BIDS events files."""
    return _run_merge_behavior(
        bids_root=bids_root,
        source_root=source_root,
        task=task,
        event_prefixes=event_prefixes,
        event_types=event_types,
        dry_run=dry_run,
        _logger=logger,
    )


###################################################################
# Pipeline Class
###################################################################


class PreprocessingPipeline(PipelineBase):
    """Pipeline for preprocessing EEG data.
    
    Unlike other pipelines, preprocessing operates on source data rather than
    per-subject derivatives. Use run_raw_to_bids() and run_merge_behavior()
    instead of process_subject().
    """
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="preprocessing", config=config)
        self.bids_root = Path(self.config.bids_root)
        self.source_root = Path(self.config.get("paths.source_data", "data/source_data"))

    def process_subject(self, subject: str, task: Optional[str] = None, **kwargs) -> None:
        """Process a single subject through raw-to-BIDS and merge-behavior."""
        task = task or self.config.get("project.task", "thermalactive")
        do_trim = kwargs.get(
            "do_trim_to_first_volume",
            kwargs.get("trim_to_first_volume", False),
        )
        
        self.logger.info(f"Processing sub-{subject}: raw-to-BIDS")
        run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=task,
            subjects=[subject],
            montage=kwargs.get("montage", "easycap-M1"),
            line_freq=kwargs.get("line_freq", 60.0),
            overwrite=kwargs.get("overwrite", False),
            zero_base_onsets=kwargs.get("zero_base_onsets", False),
            do_trim_to_first_volume=do_trim,
            event_prefixes=kwargs.get("event_prefixes"),
            keep_all_annotations=kwargs.get("keep_all_annotations", False),
        )
        
        self.logger.info(f"Processing sub-{subject}: merge-behavior")
        run_merge_behavior(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=task,
            event_prefixes=kwargs.get("event_prefixes"),
            event_types=kwargs.get("event_types"),
            dry_run=kwargs.get("dry_run", False),
        )

    def run_batch(self, subjects: List[str], task: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Run preprocessing for multiple subjects."""
        task = task or self.config.get("project.task", "thermalactive")
        do_trim = kwargs.get(
            "do_trim_to_first_volume",
            kwargs.get("trim_to_first_volume", False),
        )
        
        self.logger.info(f"Running raw-to-BIDS for {len(subjects)} subjects")
        n_converted = run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=task,
            subjects=subjects,
            montage=kwargs.get("montage", "easycap-M1"),
            line_freq=kwargs.get("line_freq", 60.0),
            overwrite=kwargs.get("overwrite", False),
            zero_base_onsets=kwargs.get("zero_base_onsets", False),
            do_trim_to_first_volume=do_trim,
            event_prefixes=kwargs.get("event_prefixes"),
            keep_all_annotations=kwargs.get("keep_all_annotations", False),
        )
        
        self.logger.info(f"Running merge-behavior")
        n_merged = run_merge_behavior(
            bids_root=self.bids_root,
            source_root=self.source_root,
            task=task,
            event_prefixes=kwargs.get("event_prefixes"),
            event_types=kwargs.get("event_types"),
            dry_run=kwargs.get("dry_run", False),
        )
        
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
            montage=montage,
            line_freq=line_freq,
            overwrite=overwrite,
            zero_base_onsets=zero_base_onsets,
            do_trim_to_first_volume=do_trim_to_first_volume,
            event_prefixes=event_prefixes,
            keep_all_annotations=keep_all_annotations,
        )

    def run_merge_behavior(
        self,
        task: Optional[str] = None,
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
            event_prefixes=event_prefixes,
            event_types=event_types,
            dry_run=dry_run,
        )


###################################################################
# Exports
###################################################################

__all__ = [
    "PreprocessingPipeline",
    "run_raw_to_bids",
    "run_merge_behavior",
    "merge_behavior_to_events",
]
