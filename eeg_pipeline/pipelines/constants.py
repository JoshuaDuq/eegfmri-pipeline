"""
Pipeline Constants
==================

Centralized constants for pipeline options, feature categories, and computations.
Single source of truth for both Python CLI and Go TUI.

FEATURE_CATEGORIES is re-exported from domain.features.constants (canonical).
"""

from __future__ import annotations

from typing import List

from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES

__all__ = [
    "FEATURE_CATEGORIES",
    "FREQUENCY_BANDS",
    "BEHAVIOR_COMPUTATIONS",
    "FEATURE_VISUALIZE_CATEGORIES",
    "BEHAVIOR_VISUALIZE_CATEGORIES",
]



FREQUENCY_BANDS: List[str] = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
]


BEHAVIOR_COMPUTATIONS: List[str] = [
    "trial_table",
    "lag_features",
    "pain_residual",
    "temperature_models",
    "regression",
    "models",
    "stability",
    "consistency",
    "influence",
    "report",
    "correlations",
    "pain_sensitivity",
    "condition",
    "temporal",
    "cluster",
    "mediation",
    "moderation",
    "mixed_effects",
]


FEATURE_VISUALIZE_CATEGORIES: List[str] = [
    "power",
    "connectivity",
    "aperiodic",
    "itpc",
    "pac",
    "quality",
    "erds",
    "complexity",
    "spectral",
    "ratios",
    "asymmetry",
    "microstates",
]


BEHAVIOR_VISUALIZE_CATEGORIES: List[str] = [
    "psychometrics",
    "temperature_models",
    "stability",
    "power",
    "aperiodic",
    "connectivity",
    "itpc",
    "temporal",
    "dose_response",
]
