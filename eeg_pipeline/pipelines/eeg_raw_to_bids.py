"""EEG raw (BrainVision) → BIDS pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from eeg_pipeline.analysis.utilities.eeg_raw_to_bids import run_raw_to_bids
from eeg_pipeline.pipelines.base import PipelineBase


class EEGRawToBidsPipeline(PipelineBase):
    """Convert EEG source data under `paths.source_data` into BIDS EEG (`paths.bids_root`)."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__(name="eeg_raw_to_bids", config=config)
        self.bids_root = Path(self.config.bids_root)
        default_source = "data/source_data"
        self.source_root = Path(self.config.get("paths.source_data", default_source))

    def run_group_level(self, subjects: List[str], task: str, **kwargs: Any) -> None:
        return

    def process_subject(self, subject: str, task: str, **kwargs: Any) -> None:
        run_raw_to_bids(
            source_root=self.source_root,
            bids_root=self.bids_root,
            task=task,
            subjects=[subject],
            montage=kwargs.get("montage", "easycap-M1"),
            line_freq=float(kwargs.get("line_freq", 60.0)),
            overwrite=bool(kwargs.get("overwrite", False)),
            do_trim_to_first_volume=bool(kwargs.get("do_trim_to_first_volume", False)),
            event_prefixes=kwargs.get("event_prefixes"),
            keep_all_annotations=bool(kwargs.get("keep_all_annotations", False)),
            _logger=self.logger,
        )

    def run_batch(
        self,
        subjects: List[str],
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        resolved_task = self._validate_batch_inputs(subjects, task)
        run_context = self._create_run_metadata_context(
            subjects=subjects,
            task=resolved_task,
            kwargs=kwargs,
        )
        run_status = "failed"
        run_error: Optional[str] = None
        n_converted = 0

        try:
            n_converted = run_raw_to_bids(
                source_root=self.source_root,
                bids_root=self.bids_root,
                task=resolved_task,
                subjects=subjects,
                montage=kwargs.get("montage", "easycap-M1"),
                line_freq=float(kwargs.get("line_freq", 60.0)),
                overwrite=bool(kwargs.get("overwrite", False)),
                do_trim_to_first_volume=bool(kwargs.get("do_trim_to_first_volume", False)),
                event_prefixes=kwargs.get("event_prefixes"),
                keep_all_annotations=bool(kwargs.get("keep_all_annotations", False)),
                _logger=self.logger,
            )
            run_status = "success"
            return n_converted
        except Exception as exc:
            run_error = str(exc)
            raise
        finally:
            self._write_run_metadata(
                run_context,
                status=run_status,
                error=run_error,
                outputs={},
                summary={
                    "n_subjects": len(subjects),
                    "n_converted": n_converted,
                },
            )
