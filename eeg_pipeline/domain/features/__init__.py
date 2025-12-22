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
    "aperiodic",
    "complexity",
    "itpc",
    "pac",
    "quality",
    "erds",
    "spectral",
    "ratios",
    "asymmetry",
    "temporal",
    "erp",
    "bursts",
)


__all__ = [
    "FEATURE_TYPES",
    "NamingSchema",
    "FeatureRegistry",
    "FeatureRule",
    "classify_feature",
    "get_feature_registry",
]
