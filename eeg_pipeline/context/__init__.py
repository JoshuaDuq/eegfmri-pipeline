"""
Context objects for pipeline state management.

This package contains dataclasses that encapsulate the state and configuration
required for various stages of the pipeline (e.g., feature extraction, behavior analysis).
"""

from .features import FeatureContext
from .behavior import BehaviorContext, AnalysisConfig

__all__ = [
    "FeatureContext",
    "BehaviorContext",
    "AnalysisConfig",
]
