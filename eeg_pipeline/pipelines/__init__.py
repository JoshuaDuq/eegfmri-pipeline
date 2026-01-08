"""Pipeline orchestration modules.

This package provides high-level pipeline classes for running
complete analysis workflows on EEG data.

Pipeline Classes:
- FeaturePipeline: Feature extraction (TFR-based and precomputed)
- BehaviorPipeline: EEG-behavior correlation analysis
- MLPipeline: Machine learning pipeline (LOSO, time-generalization)
- PreprocessingPipeline: Bad channels, ICA, epochs

Preprocessing Functions:
- run_raw_to_bids: Convert raw BrainVision to BIDS
- run_merge_behavior: Merge behavioral data into events
"""

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.pipelines.utilities import (
    run_raw_to_bids,
    run_merge_behavior,
)
from eeg_pipeline.pipelines.features import (
    FeaturePipeline,
    extract_all_features,
    extract_precomputed_features,
)
from eeg_pipeline.pipelines.behavior import (
    BehaviorPipeline,
    BehaviorPipelineConfig,
    BehaviorPipelineResults,
)
from eeg_pipeline.pipelines.machine_learning import MLPipeline
from eeg_pipeline.pipelines.utilities import UtilityPipeline
from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

__all__ = [
    # Base
    "PipelineBase",
    # Utilities
    "UtilityPipeline",
    "run_raw_to_bids",
    "run_merge_behavior",
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
