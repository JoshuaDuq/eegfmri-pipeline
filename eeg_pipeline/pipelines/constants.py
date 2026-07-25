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
    "predictor_residual",
    "regression",
    "icc",
    "correlations",
    "condition",
    "temporal",
    "cluster",
    "multilevel_correlations",
]
