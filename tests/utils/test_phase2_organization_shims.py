from __future__ import annotations

from eeg_pipeline.analysis.utilities.eeg_raw_to_bids import run_raw_to_bids as analysis_run_raw_to_bids
from eeg_pipeline.analysis.utilities.merge_psychopy import run_merge_psychopy as analysis_run_merge_psychopy
from eeg_pipeline.pipelines import eeg_raw_to_bids as raw_to_bids_module
from eeg_pipeline.pipelines import merge_psychopy as merge_psychopy_module


def test_eeg_raw_to_bids_pipeline_uses_analysis_utility_function() -> None:
    assert raw_to_bids_module.run_raw_to_bids is analysis_run_raw_to_bids


def test_merge_psychopy_pipeline_uses_analysis_utility_function() -> None:
    assert merge_psychopy_module.run_merge_psychopy is analysis_run_merge_psychopy
