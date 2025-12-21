from __future__ import annotations

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.registry import (
    FeatureRegistry,
    FeatureRule,
    classify_feature,
    get_feature_registry,
)


FEATURE_TYPES = (
    "power",
    "connectivity",
    "microstates",
    "aperiodic",
    "itpc",
    "pac",
    "complexity",
    "dynamics",
    "cfc",
)


__all__ = [
    "FEATURE_TYPES",
    "NamingSchema",
    "FeatureRegistry",
    "FeatureRule",
    "classify_feature",
    "get_feature_registry",
]
