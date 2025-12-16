"""
Domain models for the EEG pipeline.

This package contains domain-specific models and business logic:
- features/: Feature naming conventions, registry, and constants
"""

from __future__ import annotations

from eeg_pipeline.domain.features import (
    NamingSchema,
    FeatureRegistry,
    FeatureRule,
    classify_feature,
    get_feature_registry,
)
from eeg_pipeline.domain.features.constants import (
    FEATURE_CATEGORIES,
    PRECOMPUTED_GROUP_CHOICES,
)

__all__ = [
    "NamingSchema",
    "FeatureRegistry",
    "FeatureRule",
    "classify_feature",
    "get_feature_registry",
    "FEATURE_CATEGORIES",
    "PRECOMPUTED_GROUP_CHOICES",
]
