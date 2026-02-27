"""Pipeline orchestration modules.

This package provides high-level pipeline classes for running
complete analysis workflows on EEG data.

Pipeline Classes:
- FeaturePipeline: Feature extraction (TFR-based and precomputed)
- BehaviorPipeline: EEG-behavior correlation analysis
- MLPipeline: Machine learning pipeline (LOSO, time-generalization)
- PreprocessingPipeline: Bad channels, ICA, epochs

Note: Raw-to-BIDS conversion and event-log harmonization are external to this
package. Run those dataset-specific steps before this pipeline.
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

__all__ = [
    # Base
    "PipelineBase",
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
