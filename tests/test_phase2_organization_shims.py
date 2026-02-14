from __future__ import annotations

from eeg_pipeline.pipelines import eeg_raw_to_bids as raw_to_bids_module
from eeg_pipeline.pipelines import merge_psychopy as merge_psychopy_module
from eeg_pipeline.pipelines import utilities as utilities_module


def test_eeg_raw_to_bids_pipeline_uses_shared_utility_wrapper() -> None:
    assert raw_to_bids_module.run_raw_to_bids is utilities_module.run_raw_to_bids


def test_merge_psychopy_pipeline_uses_shared_utility_wrapper() -> None:
    assert merge_psychopy_module.run_merge_psychopy is utilities_module.run_merge_psychopy
