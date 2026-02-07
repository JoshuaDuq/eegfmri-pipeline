"""
EEG Feature Extraction Package
==============================

Efficient feature extraction for EEG analysis, optimized for pain research.

Quick Start
-----------
>>> from eeg_pipeline.analysis.features import extract_precomputed_features
>>> result = extract_precomputed_features(epochs, bands, config, logger)
>>> df = result.get_combined_df()


Feature Groups
--------------
Available groups for `extract_precomputed_features`:
    power, erds, spectral, connectivity, complexity,
    aperiodic, ratios, asymmetry, itpc, pac, quality, microstates

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
    # Constants and utilities (imported from their actual modules)
    "pick_eeg_channels": "eeg_pipeline.utils.analysis.channels",
    # Normalization
    "normalize_train_test": "normalization",
    "zscore_normalize": "normalization",
    "robust_normalize": "normalization",
    # Reliability (canonical: utils.analysis.stats.reliability)
    "compute_icc": "eeg_pipeline.utils.analysis.stats.reliability",
    # Quality
    "compute_trial_quality_metrics": "quality",
    # Microstates
    "extract_microstate_features": "microstates",
    # Connectivity
    "extract_connectivity_features": "connectivity",
    "extract_connectivity_from_precomputed": "connectivity",
    "extract_directed_connectivity_features": "connectivity",
    "extract_directed_connectivity_from_precomputed": "connectivity",

    # Source Localization
    "extract_source_localization_features": "source_localization",
    "extract_source_connectivity_features": "source_localization",
    "extract_source_localization_from_precomputed": "source_localization",

    # Power
    "extract_power_features": "spectral",
    "extract_asymmetry_from_precomputed": "precomputed.extras",

    # Aperiodic
    "extract_aperiodic_features": "aperiodic",

    # Phase / PAC / ITPC
    "extract_phase_features": "phase",
    "compute_pac_comodulograms": "phase",
    "extract_itpc_from_precomputed": "phase",
    # Complexity
    "extract_complexity_from_precomputed": "complexity",
    # Pipeline orchestrators (lazy to prevent circular imports)
    "FeaturePipeline": "eeg_pipeline.pipelines.features",
    "process_subject": "eeg_pipeline.pipelines.features",
    "extract_features_for_subjects": "eeg_pipeline.pipelines.features",
    "extract_all_features": "eeg_pipeline.analysis.features.api",
    "extract_precomputed_features": "eeg_pipeline.analysis.features.api",
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
