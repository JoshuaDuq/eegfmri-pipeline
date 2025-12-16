"""
EEG analysis modules.

This package contains the core analysis logic for the EEG pipeline:

Submodules:
- behavior: Behavioral correlation analysis (EEG-pain correlations)
- decoding: Machine learning decoding (LOSO CV, time generalization)
- features: Feature extraction (power, connectivity, microstates, etc.)
- preprocessing: BIDS conversion and event processing
"""

from eeg_pipeline.analysis import behavior
from eeg_pipeline.analysis import decoding
from eeg_pipeline.analysis import features
from eeg_pipeline.analysis import preprocessing

__all__ = [
    "behavior",
    "decoding",
    "features",
    "preprocessing",
]
