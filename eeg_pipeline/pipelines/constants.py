"""
Pipeline Constants
==================

Centralized constants for pipeline options, feature categories, and computations.
Single source of truth for both Python CLI and Go TUI.
"""

from __future__ import annotations

from typing import List


FREQUENCY_BANDS: List[str] = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
]


FEATURE_CATEGORIES: List[str] = [
    "power",
    "connectivity",
    "directedconnectivity",
    "sourcelocalization",
    "aperiodic",
    "erp",
    "complexity",
    "bursts",
    "itpc",
    "pac",
    "quality",
    "erds",
    "spectral",
    "ratios",
    "asymmetry",
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
