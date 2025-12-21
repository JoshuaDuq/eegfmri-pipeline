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
# Canonical implementations live in eeg_pipeline.pipelines.features
# Imported lazily below to avoid circular dependencies.
from eeg_pipeline.analysis.features.results import (
    ExtractionResult,
    FeatureSet,
    FeatureExtractionResult,
)

from eeg_pipeline.types import PrecomputedData, ConfigLike, BandData, PSDData, TimeWindows
from eeg_pipeline.analysis.features.preparation import precompute_data

# =============================================================================
# Quality & Normalization
# =============================================================================
from eeg_pipeline.analysis.features.quality import extract_quality_features

from eeg_pipeline.analysis.features.normalization import (
    normalize_features,
    FeatureNormalizer,
)

from eeg_pipeline.utils.analysis.stats.reliability import (
    compute_dataframe_reliability as compute_feature_reliability,
    filter_reliable_features,
    ReliabilityResult,
)

# =============================================================================
# Manifest & Output
# =============================================================================
from eeg_pipeline.domain.features.naming import (
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
    "extract_all_features",
    "ExtractionResult",
    "FeatureSet",
    # Orchestration
    "FeaturePipeline",
    "process_subject",
    "extract_features_for_subjects",
    # Core data structures
    "PrecomputedData",
    "BandData",
    "PSDData",
    "TimeWindows",
    "precompute_data",
    "ConfigLike",
    # Quality assessment
    "extract_quality_features",
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
    "parse_feature_name": "eeg_pipeline.domain.features.naming",  # Authoritative parser with FeatureMetadata
    
    # Constants and utilities (imported from their actual modules)
    "build_roi_map": "eeg_pipeline.utils.analysis.channels",
    "pick_eeg_channels": "eeg_pipeline.utils.analysis.channels",
    "compute_gfp": "eeg_pipeline.utils.analysis.signal_metrics",
    
    # Normalization
    "normalize_train_test": "normalization",
    "zscore_normalize": "normalization",
    "robust_normalize": "normalization",
    
    # Reliability (canonical: utils.analysis.stats.reliability)
    "ReliabilityResult": "eeg_pipeline.utils.analysis.stats.reliability",
    "compute_split_half_reliability": "eeg_pipeline.utils.analysis.stats.reliability",
    "compute_icc": "eeg_pipeline.utils.analysis.stats.reliability",
    
    # Quality
    "compute_trial_quality_metrics": "quality",
    
    # Microstates
    "extract_microstate_features": "microstates",
    "extract_microstate_features_from_epochs": "microstates",

    # Connectivity
    "extract_connectivity_features": "connectivity",
    "extract_connectivity_from_precomputed": "connectivity",

    # Power
    "extract_power_features": "power",
    "extract_spectral_extras_from_precomputed": "precomputed.spectral",
    "extract_asymmetry_from_precomputed": "precomputed.extras",
    "extract_segment_power_from_precomputed": "precomputed.spectral",

    # Aperiodic
    "extract_aperiodic_features": "aperiodic",
    "extract_aperiodic_features_from_epochs": "aperiodic",

    # Phase / PAC / ITPC
    "extract_phase_features": "phase",
    "compute_pac_comodulograms": "phase",
    "extract_itpc_from_precomputed": "phase",
    
    # Cross-frequency coupling
    "extract_modulation_index_pac": "cfc",
    "extract_phase_phase_coupling": "cfc",
    "extract_pac_from_precomputed": "cfc",
    "extract_all_cfc_features": "cfc",
    
    # Complexity
    "extract_dynamics_features": "complexity",
    "extract_complexity_from_precomputed": "complexity",

    # Dynamics (precomputed)
    "extract_dynamics_from_precomputed": "dynamics",

    # ML presets

    # Pipeline orchestrators (lazy to prevent circular imports)
    "FeaturePipeline": "eeg_pipeline.pipelines.features",
    "process_subject": "eeg_pipeline.pipelines.features",
    "extract_features_for_subjects": "eeg_pipeline.pipelines.features",
    "extract_all_features": "eeg_pipeline.analysis.features.api",
    "extract_precomputed_features": "eeg_pipeline.analysis.features.api",
    "extract_fmri_prediction_features": "eeg_pipeline.analysis.features.api",
}


def __getattr__(name: str):
    """Lazy import for secondary functions."""
    if name in _LAZY_IMPORTS:
        import importlib
        module_path = _LAZY_IMPORTS[name]
        # Handle absolute module paths (starting with eeg_pipeline)
        if module_path.startswith("eeg_pipeline."):
            module = importlib.import_module(module_path)
        else:
            # Relative import within this package
            module = importlib.import_module(f".{module_path}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available names for tab completion."""
    return list(__all__) + list(_LAZY_IMPORTS.keys())
