"""
EEG analysis modules.

Submodules:
- behavior: Behavioral correlation analysis
- decoding: Machine learning decoding
- features: Feature extraction
- group: Group-level analysis
"""

from eeg_pipeline.analysis import behavior
from eeg_pipeline.analysis import decoding
from eeg_pipeline.analysis import features
from eeg_pipeline.analysis import group

__all__ = [
    "behavior",
    "decoding",
    "features",
    "group",
]

