"""
EEG/fMRI Analysis Pipeline

A comprehensive pipeline for EEG data processing, feature extraction,
behavioral correlation analysis, and machine learning decoding.

Package Structure
-----------------
- analysis/: Core analysis modules (features, behavior, decoding, group)
- pipelines/: High-level pipeline orchestration
- plotting/: Visualization utilities
- preprocessing/: BIDS conversion and event processing
- types: Type definitions and protocols

Quick Start
-----------
>>> from eeg_pipeline.pipelines import extract_features_for_subjects
>>> extract_features_for_subjects(["0001", "0002"])

>>> from eeg_pipeline.pipelines import compute_behavior_correlations_for_subjects
>>> compute_behavior_correlations_for_subjects(["0001", "0002"])
"""

__version__ = "0.1.0"

# Expose key types
from eeg_pipeline.types import (
    EEGConfig,
    ConfigLike,
    FeatureResult,
    CorrelationResult,
    DecodingResult,
    ValidationResult,
    BandData,
    PSDData,
    TimeWindows,
    PrecomputedQC,
    PrecomputedData,
)

__all__ = [
    "__version__",
    # Types
    "EEGConfig",
    "ConfigLike",
    "FeatureResult",
    "CorrelationResult",
    "DecodingResult",
    "ValidationResult",
    # Precomputed data structures
    "BandData",
    "PSDData",
    "TimeWindows",
    "PrecomputedQC",
    "PrecomputedData",
]

