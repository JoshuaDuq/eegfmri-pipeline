"""
EEG analysis modules.

Submodules:
- behavior: Behavioral correlation analysis
- decoding: Machine learning decoding
- features: Feature extraction
"""

from eeg_pipeline.analysis import behavior
from eeg_pipeline.analysis import decoding
from eeg_pipeline.analysis import features

__all__ = [
    "behavior",
    "decoding",
    "features",
]

