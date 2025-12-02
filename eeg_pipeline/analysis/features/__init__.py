"""
EEG Feature Extraction Package
==============================

Efficient feature extraction for EEG analysis, optimized for pain research
and EEG-fMRI integration.

Quick Start
-----------
>>> from eeg_pipeline.analysis.features import extract_precomputed_features
>>> result = extract_precomputed_features(epochs, bands, config, logger)
>>> df = result.get_combined_df()

For fMRI prediction (optimized feature subset):
>>> from eeg_pipeline.analysis.features import extract_fmri_prediction_features
>>> result = extract_fmri_prediction_features(epochs, config, logger)

Feature Groups
--------------
Available groups for `extract_precomputed_features`:
    power, erds, spectral, gfp, connectivity, roi, temporal,
    complexity, aperiodic, ratios, asymmetry, itpc, pac, cfc,
    microstates, quality

Naming Convention
-----------------
Features follow: {domain}_{measure}_{band}_{location}_{time}_{statistic}
"""

from __future__ import annotations

# =============================================================================
# Core API - Primary Entry Points
# =============================================================================
from eeg_pipeline.analysis.features.pipeline import (
    extract_precomputed_features,
    extract_fmri_prediction_features,
    ExtractionResult,
    FeatureSet,
)

from eeg_pipeline.analysis.features.core import (
    PrecomputedData,
    precompute_data,
    ConfigLike,
)

# =============================================================================
# Quality & Normalization
# =============================================================================
from eeg_pipeline.analysis.features.quality import (
    FeatureQuality,
    FeatureQualityReport,
    compute_feature_quality,
    filter_quality_features,
)

from eeg_pipeline.analysis.features.normalization import (
    normalize_features,
    FeatureNormalizer,
)

from eeg_pipeline.analysis.features.reliability import (
    compute_feature_reliability,
    filter_reliable_features,
)

# =============================================================================
# Manifest & Output
# =============================================================================
from eeg_pipeline.analysis.features.manifest import (
    generate_manifest,
    save_features_organized,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Main extraction functions
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
    "ExtractionResult",
    "FeatureSet",
    # Core data structures
    "PrecomputedData",
    "precompute_data",
    "ConfigLike",
    # Quality assessment
    "FeatureQuality",
    "FeatureQualityReport",
    "compute_feature_quality",
    "filter_quality_features",
    # Normalization
    "normalize_features",
    "FeatureNormalizer",
    # Reliability
    "compute_feature_reliability",
    "filter_reliable_features",
    # Output
    "generate_manifest",
    "save_features_organized",
]

# =============================================================================
# Lazy Imports for Domain-Specific Functions
# =============================================================================
# These are available but not imported by default to reduce startup time.
# Access via: from eeg_pipeline.analysis.features import <name>

_LAZY_IMPORTS = {
    # Naming utilities
    "make_feature_name": "naming",
    "parse_feature_name": "manifest",  # Authoritative parser with FeatureMetadata
    
    # Core data structures
    "BandData": "core",
    "PSDData": "core",
    "TimeWindows": "core",
    "EPSILON_STD": "core",
    "build_roi_map": "core",
    "pick_eeg_channels": "core",
    "compute_gfp": "core",
    
    # Normalization
    "normalize_train_test": "normalization",
    "zscore_normalize": "normalization",
    "robust_normalize": "normalization",
    
    # Reliability
    "ReliabilityResult": "reliability",
    "compute_split_half_reliability": "reliability",
    "compute_icc": "reliability",
    
    # Quality
    "identify_correlated_features": "quality",
    "compute_trial_quality_metrics": "quality",
    
    # Microstates
    "extract_microstate_features": "microstates",
    
    # Cross-frequency coupling
    "extract_modulation_index_pac": "cfc",
    "extract_phase_phase_coupling": "cfc",
    
    # Aperiodic
    "extract_aperiodic_features": "aperiodic",
    
    # Complexity
    "extract_permutation_entropy_features": "complexity",
    "extract_hjorth_parameters": "complexity",
    
    # ML presets
    "get_feature_groups_for_ml": "pipeline",
}


def __getattr__(name: str):
    """Lazy import for secondary functions."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available names for tab completion."""
    return list(__all__) + list(_LAZY_IMPORTS.keys())
