"""
EEG analysis modules.

This package contains the core analysis logic for the EEG pipeline:

Submodules:
- behavior: Behavioral correlation analysis (EEG-pain correlations)
- machine_learning: Machine learning pipeline (LOSO CV, time generalization)
- features: Feature extraction (power, connectivity, ERDS, ERP, etc.)
- utilities: BIDS conversion and event processing (raw-to-bids, merge-psychopy)
"""

from eeg_pipeline.analysis import behavior
from eeg_pipeline.analysis import machine_learning
from eeg_pipeline.analysis import features
from eeg_pipeline.analysis import utilities

__all__ = [
    "behavior",
    "machine_learning",
    "features",
    "utilities",
]
