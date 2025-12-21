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

FREQUENCY_BAND_RANGES: Dict[str, tuple] = {
    "delta": (1.0, 3.9),
    "theta": (4.0, 7.9),
    "alpha": (8.0, 12.9),
    "beta": (13.0, 30.0),
    "gamma": (30.1, 80.0),
}


FEATURE_CATEGORIES: List[str] = [
    "power",
    "connectivity",
    "aperiodic",
    "dynamics",
    "complexity",
    "itpc",
    "microstates",
]

FEATURE_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "power": "Band power features (theta, alpha, beta, gamma)",
    "connectivity": "Phase connectivity metrics",
    "aperiodic": "1/f spectral slope parameters",
    "dynamics": "Temporal dynamics (GFP, bursts)",
    "complexity": "Signal complexity measures",
    "itpc": "Inter-trial phase coherence",
    "microstates": "Microstate features",
}


BEHAVIOR_COMPUTATIONS: List[str] = [
    "correlations",
    "pain_sensitivity",
    "condition",
    "temporal",
    "cluster",
    "mediation",
    "mixed_effects",
    "export",
]

BEHAVIOR_COMPUTATION_DESCRIPTIONS: Dict[str, str] = {
    "correlations": "EEG-rating correlations",
    "pain_sensitivity": "Individual pain sensitivity analysis",
    "condition": "Compare conditions (e.g., ramp vs plateau)",
    "temporal": "Time-resolved correlation analysis",
    "cluster": "Cluster-based permutation tests",
    "mediation": "Path analysis and mediation models",
    "mixed_effects": "Mixed-effects modeling",
    "export": "Export results",
}


FEATURE_VISUALIZE_CATEGORIES: List[str] = [
    "power",
    "connectivity",
    "microstates",
    "aperiodic",
    "itpc",
    "pac",
    "dynamics",
    "burst",
    "erds",
    "complexity",
]


BEHAVIOR_VISUALIZE_CATEGORIES: List[str] = [
    "psychometrics",
    "power",
    "dynamics",
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
            {"key": band, "name": band.title(), "description": f"{FREQUENCY_BAND_RANGES[band][0]}-{FREQUENCY_BAND_RANGES[band][1]} Hz"}
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
