from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from tests.pipelines_test_utils import DotConfig


def test_eeg_raw_to_bids_process_subject_forwards_logger_to_analysis_utility() -> None:
    from eeg_pipeline.pipelines.eeg_raw_to_bids import EEGRawToBidsPipeline

    pipeline = object.__new__(EEGRawToBidsPipeline)
    pipeline.source_root = Path("/tmp/source")
    pipeline.bids_root = Path("/tmp/bids")
    pipeline.logger = Mock()

    def _wrapper(
        source_root: Path,
        bids_root: Path,
        task: str,
        subjects=None,
        montage: str = "easycap-M1",
        line_freq: float = 60.0,
        overwrite: bool = False,
        do_trim_to_first_volume: bool = False,
        event_prefixes=None,
        keep_all_annotations: bool = False,
        _logger=None,
    ) -> int:
        assert _logger is pipeline.logger
        return 1

    with patch("eeg_pipeline.pipelines.eeg_raw_to_bids.run_raw_to_bids", side_effect=_wrapper):
        pipeline.process_subject("0001", "thermalactive")


def test_merge_psychopy_process_subject_forwards_logger_to_analysis_utility() -> None:
    from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

    pipeline = object.__new__(MergePsychopyPipeline)
    pipeline.source_root = Path("/tmp/source")
    pipeline.bids_root = Path("/tmp/bids")
    pipeline.logger = Mock()
    pipeline.config = DotConfig({"alignment": {"allow_misaligned_trim": False}})

    def _wrapper(
        bids_root: Path,
        source_root: Path,
        task: str,
        subjects=None,
        event_prefixes=None,
        event_types=None,
        dry_run: bool = False,
        allow_misaligned_trim: bool = False,
        _logger=None,
    ) -> int:
        assert _logger is pipeline.logger
        return 1

    with patch("eeg_pipeline.pipelines.merge_psychopy.run_merge_psychopy", side_effect=_wrapper):
        pipeline.process_subject("0001", "thermalactive", dry_run=True)
