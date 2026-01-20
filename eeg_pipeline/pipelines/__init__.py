"""Pipeline orchestration modules.

This package provides high-level pipeline classes for running
complete analysis workflows on EEG data.

Pipeline Classes:
- FeaturePipeline: Feature extraction (TFR-based and precomputed)
- BehaviorPipeline: EEG-behavior correlation analysis
- MLPipeline: Machine learning pipeline (LOSO, time-generalization)
- PreprocessingPipeline: Bad channels, ICA, epochs
- EEGRawToBidsPipeline: BrainVision → BIDS EEG
- MergePsychopyPipeline: TrialSummary → events.tsv

Preprocessing Functions:
- run_raw_to_bids: Convert raw BrainVision to BIDS
- run_merge_behavior: Merge behavioral data into events
"""

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.pipelines.behavior import (
    BehaviorPipeline,
    BehaviorPipelineConfig,
    BehaviorPipelineResults,
)
from eeg_pipeline.pipelines.features import (
    FeaturePipeline,
    extract_all_features,
    extract_precomputed_features,
)
from eeg_pipeline.pipelines.machine_learning import MLPipeline
from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
from eeg_pipeline.pipelines.eeg_raw_to_bids import EEGRawToBidsPipeline
from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline
from eeg_pipeline.pipelines.utilities import (
    UtilityPipeline,
    run_merge_behavior,
    run_raw_to_bids,
)

__all__ = [
    # Base
    "PipelineBase",
    # Utilities
    "UtilityPipeline",
    "run_raw_to_bids",
    "run_merge_behavior",
    "EEGRawToBidsPipeline",
    "MergePsychopyPipeline",
    # Features
    "FeaturePipeline",
    "extract_all_features",
    "extract_precomputed_features",
    # Behavior
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
    # Machine Learning
    "MLPipeline",
    # Preprocessing
    "PreprocessingPipeline",
]
