"""
EEG analysis modules.

This package contains the core analysis logic for the EEG pipeline:

Submodules:
- behavior: Behavioral correlation analysis (EEG-pain correlations)
- machine_learning: Machine learning pipeline (LOSO CV, time generalization)
- features: Feature extraction (power, connectivity, ERDS, ERP, etc.)
- preprocessing: BIDS conversion and event processing
"""

from eeg_pipeline.analysis import behavior
from eeg_pipeline.analysis import machine_learning
from eeg_pipeline.analysis import features
from eeg_pipeline.analysis import preprocessing

__all__ = [
    "behavior",
    "machine_learning",
    "features",
    "preprocessing",
]
