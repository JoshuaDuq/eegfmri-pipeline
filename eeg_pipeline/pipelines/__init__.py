"""Pipeline orchestration modules.

This package provides high-level pipeline classes for running
complete analysis workflows on EEG data.

Pipeline Classes:
- FeaturePipeline: Feature extraction (TFR-based and precomputed)
- BehaviorPipeline: EEG-behavior correlation analysis
- DecodingPipeline: ML-based prediction (LOSO, time-generalization)
- ErpPipeline: Event-related potential analysis
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
from eeg_pipeline.pipelines.erp import (
    ErpPipeline,
    get_erp_config,
    load_and_prepare_epochs,
)
from eeg_pipeline.pipelines.features import (
    FeaturePipeline,
    extract_all_features,
    extract_precomputed_features,
    extract_fmri_prediction_features,
)
from eeg_pipeline.pipelines.behavior import (
    BehaviorPipeline,
    BehaviorPipelineConfig,
    BehaviorPipelineResults,
)
from eeg_pipeline.pipelines.decoding import DecodingPipeline
from eeg_pipeline.pipelines.utilities import UtilityPipeline
from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

__all__ = [
    # Base
    "PipelineBase",
    # Utilities
    "UtilityPipeline",
    "run_raw_to_bids",
    "run_merge_behavior",
    # ERP
    "ErpPipeline",
    "get_erp_config",
    "load_and_prepare_epochs",
    # Features
    "FeaturePipeline",
    "extract_all_features",
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
    # Behavior
    "BehaviorPipeline",
    "BehaviorPipelineConfig",
    "BehaviorPipelineResults",
    # Decoding
    "DecodingPipeline",
    # Preprocessing
    "PreprocessingPipeline",
]
