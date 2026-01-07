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


DECODING_MODES: List[str] = [
    "regression",
    "timegen",
    "classify",
]

DECODING_MODE_DESCRIPTIONS: Dict[str, str] = {
    "regression": "LOSO regression",
    "timegen": "Time generalization",
    "classify": "Binary classification",
}


def get_definitions_dict() -> Dict:
    """Return all definitions as a dictionary for JSON export."""
    return {
        "frequency_bands": [
            {"key": band, "name": band.title(), "description": f"{band.title()} frequency band"}
            for band in FREQUENCY_BANDS
        ],
        "feature_categories": [
            {"key": cat, "name": cat.title(), "description": FEATURE_CATEGORY_DESCRIPTIONS.get(cat, "")}
            for cat in FEATURE_CATEGORIES
        ],
        "behavior_computations": [
            {"key": comp, "name": comp.replace("_", " ").title(), "description": BEHAVIOR_COMPUTATION_DESCRIPTIONS.get(comp, "")}
            for comp in BEHAVIOR_COMPUTATIONS
        ],
        "preprocessing_modes": [
            {"key": mode, "name": mode.replace("-", " ").title(), "description": PREPROCESSING_MODE_DESCRIPTIONS.get(mode, "")}
            for mode in PREPROCESSING_MODES
        ],
        "decoding_modes": [
            {"key": mode, "name": mode.title(), "description": DECODING_MODE_DESCRIPTIONS.get(mode, "")}
            for mode in DECODING_MODES
        ],
    }
