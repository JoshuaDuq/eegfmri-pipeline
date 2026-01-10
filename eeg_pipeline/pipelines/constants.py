"""
Pipeline Constants
==================

Centralized constants for pipeline options, feature categories, and computations.
Single source of truth for both Python CLI and Go TUI.
"""

from __future__ import annotations

from typing import Dict, List


FREQUENCY_BANDS: List[str] = [
    "delta",
    "theta",
    "alpha",
    "beta",
    "gamma",
]

# Removed hardcoded FREQUENCY_BAND_RANGES. Ranges should be fetched from config.


MANDATORY_TIME_WINDOWS: List[str] = [
    "baseline",
    "active",
]


FEATURE_CATEGORIES: List[str] = [
    "power",
    "connectivity",
    "directed_connectivity",
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

FEATURE_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "power": "Band power features (theta, alpha, beta, gamma)",
    "connectivity": "Phase connectivity metrics",
    "directed_connectivity": "Directed connectivity metrics (PSI, DTF, PDC)",
    "aperiodic": "1/f spectral slope parameters",
    "erp": "ERP/LEP time-domain features",
    "complexity": "Signal complexity measures",
    "bursts": "Oscillatory burst dynamics",
    "itpc": "Inter-trial phase coherence",
    "pac": "Phase-amplitude coupling",
    "quality": "Data quality and artifact metrics",
    "erds": "Event-related (de)synchronization",
    "spectral": "Spectral edge and other freq metrics",
    "ratios": "Power ratios (e.g., theta/beta)",
    "asymmetry": "Inter-hemispheric asymmetry",
}


BEHAVIOR_COMPUTATIONS: List[str] = [
    "trial_table",
    "lag_features",
    "pain_residual",
    "temperature_models",
    "confounds",
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

BEHAVIOR_COMPUTATION_DESCRIPTIONS: Dict[str, str] = {
    "trial_table": "Export canonical per-trial analysis table",
    "lag_features": "Add lag/delta dynamics to trial table",
    "pain_residual": "Compute pain_residual = rating - f(temperature)",
    "temperature_models": "Model comparison and breakpoint detection on temp→rating",
    "confounds": "Audit QC confounds vs targets",
    "regression": "Trialwise regression/moderation models",
    "models": "Sensitivity model families (robust/quantile/logistic)",
    "stability": "Run/block stability diagnostics (non-gating)",
    "consistency": "Effect direction consistency across outcomes",
    "influence": "Influence diagnostics (Cook's distance/leverage)",
    "report": "Single-subject report (reproducible summary)",
    "correlations": "EEG-rating correlations",
    "pain_sensitivity": "Individual pain sensitivity analysis",
    "condition": "Compare conditions (e.g., ramp vs active)",
    "temporal": "Time-resolved correlation analysis",
    "cluster": "Cluster-based permutation tests",
    "mediation": "Path analysis and mediation models",
    "moderation": "Moderation analysis (interaction effects)",
    "mixed_effects": "Mixed-effects modeling",
}


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


PREPROCESSING_MODES: List[str] = [
    "full",
    "bad-channels",
    "ica",
    "epochs",
]

PREPROCESSING_MODE_DESCRIPTIONS: Dict[str, str] = {
    "full": "Full preprocessing pipeline",
    "bad-channels": "Bad channel detection only",
    "ica": "ICA fitting and labeling",
    "epochs": "Epoch creation only",
}


ML_MODES: List[str] = [
    "regression",
    "timegen",
    "classify",
]

ML_MODE_DESCRIPTIONS: Dict[str, str] = {
    "regression": "LOSO regression",
    "timegen": "Time generalization",
    "classify": "Binary classification",
}


def get_definitions_dict() -> Dict[str, List[Dict[str, str]]]:
    """Return all definitions as a dictionary for JSON export."""
    frequency_band_entries = [
        {
            "key": band,
            "name": band.title(),
            "description": f"{band.title()} frequency band",
        }
        for band in FREQUENCY_BANDS
    ]

    feature_category_entries = [
        {
            "key": category,
            "name": category.title(),
            "description": FEATURE_CATEGORY_DESCRIPTIONS.get(category, ""),
        }
        for category in FEATURE_CATEGORIES
    ]

    behavior_computation_entries = [
        {
            "key": computation,
            "name": computation.replace("_", " ").title(),
            "description": BEHAVIOR_COMPUTATION_DESCRIPTIONS.get(computation, ""),
        }
        for computation in BEHAVIOR_COMPUTATIONS
    ]

    preprocessing_mode_entries = [
        {
            "key": mode,
            "name": mode.replace("-", " ").title(),
            "description": PREPROCESSING_MODE_DESCRIPTIONS.get(mode, ""),
        }
        for mode in PREPROCESSING_MODES
    ]

    ml_mode_entries = [
        {
            "key": mode,
            "name": mode.title(),
            "description": ML_MODE_DESCRIPTIONS.get(mode, ""),
        }
        for mode in ML_MODES
    ]

    return {
        "frequency_bands": frequency_band_entries,
        "feature_categories": feature_category_entries,
        "behavior_computations": behavior_computation_entries,
        "preprocessing_modes": preprocessing_mode_entries,
        "ml_modes": ml_mode_entries,
    }
