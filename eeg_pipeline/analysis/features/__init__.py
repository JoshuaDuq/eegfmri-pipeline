"""
EEG Feature Extraction Package
==============================

A comprehensive, modular feature extraction pipeline for EEG analysis,
optimized for pain research and EEG-fMRI integration.

Module Organization
-------------------

**Core Infrastructure:**
    - core: Data containers, precomputation, shared utilities
    - pipeline: Main orchestration and batch extraction
    - naming: Standardized feature naming convention
    - manifest: Feature metadata and organized output

**Frequency-Domain Features:**
    - power: Band power extraction (time-resolved)
    - spectral: IAF, entropy, ratios, peak frequencies
    - aperiodic: 1/f slope, offset (FOOOF-like)

**Time-Domain Features:**
    - temporal: Statistical moments, amplitude, waveform
    - erds: Event-related desynchronization/synchronization
    - complexity: Permutation entropy, Hjorth, Lempel-Ziv

**Connectivity Features:**
    - connectivity: wPLI, PLV, AEC, graph metrics

**Phase Features:**
    - phase: ITPC, PAC (phase-amplitude coupling)

**Spatial Features:**
    - roi_features: ROI-averaged power, asymmetry
    - global_features: GFP, global synchrony

**Advanced Features:**
    - microstates: Microstate dynamics and transitions

Quick Start
-----------
>>> from eeg_pipeline.analysis.features import extract_precomputed_features
>>> result = extract_precomputed_features(epochs, bands, config, logger)
>>> df = result.get_combined_df()  # All features with condition labels

For fMRI prediction:
>>> from eeg_pipeline.analysis.features import extract_fmri_prediction_features
>>> result = extract_fmri_prediction_features(epochs, config, logger)

Feature Groups
--------------
Available groups for `extract_precomputed_features`:
    - power: Time-resolved band power (log-ratio normalized)
    - erds: ERD/ERS with temporal dynamics
    - spectral: PSD-based spectral features
    - gfp: Global field power
    - connectivity: Phase and amplitude connectivity
    - roi: ROI-averaged features
    - temporal: Time-domain statistics
    - complexity: Nonlinear dynamics
    - aperiodic: 1/f spectral slope
    - ratios: Cross-band power ratios
    - asymmetry: Hemispheric asymmetry
    - itpc: Inter-trial phase coherence

Naming Convention
-----------------
Features follow: {domain}_{measure}_{band}_{location}_{time}_{statistic}

Examples:
    - power_alpha_Cz_early_logratio
    - erds_beta_global_full_percent
    - conn_wpli_theta_t3_mean
    - phase_itpc_gamma_global_late_mean
"""

from __future__ import annotations

# =============================================================================
# Core Infrastructure (always imported)
# =============================================================================
from eeg_pipeline.analysis.features.core import (
    # Data containers
    PrecomputedData,
    BandData,
    PSDData,
    TimeWindows,
    FeatureResult,
    # Main functions
    precompute_data,
    build_roi_map,
    # Constants
    EPSILON_STD,
    EPSILON_PSD,
    EPSILON_AMP,
)

from eeg_pipeline.analysis.features.pipeline import (
    # Main extraction functions
    extract_precomputed_features,
    extract_fmri_prediction_features,
    # Result containers
    ExtractionResult,
    FeatureSet,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # --- Core Infrastructure ---
    "PrecomputedData",
    "BandData",
    "PSDData", 
    "TimeWindows",
    "FeatureResult",
    "precompute_data",
    "build_roi_map",
    "EPSILON_STD",
    "EPSILON_PSD",
    "EPSILON_AMP",
    # --- Pipeline ---
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
    "ExtractionResult",
    "FeatureSet",
]

# =============================================================================
# Feature Domain Modules (available via lazy import)
# =============================================================================
_LAZY_IMPORTS = {
    # -------------------------------------------------------------------------
    # Naming & Manifest
    # -------------------------------------------------------------------------
    "FeatureName": "naming",
    "make_feature_name": "naming",
    "make_power_name": "naming",
    "make_erds_name": "naming",
    "make_conn_name": "naming",
    "make_phase_name": "naming",
    "parse_feature_name": "naming",
    "FeatureMetadata": "manifest",
    "generate_manifest": "manifest",
    "save_manifest": "manifest",
    "save_features_organized": "manifest",
    
    # -------------------------------------------------------------------------
    # Power & Spectral
    # -------------------------------------------------------------------------
    "extract_band_power_features": "power",
    "extract_baseline_power_features": "power",
    "extract_individual_alpha_frequency": "spectral",
    "extract_relative_band_power": "spectral",
    "extract_band_power_ratios": "spectral",
    "extract_spectral_entropy_features": "spectral",
    "extract_peak_frequencies": "spectral",
    
    # -------------------------------------------------------------------------
    # ERD/ERS
    # -------------------------------------------------------------------------
    "extract_erds_features": "erds",
    "extract_erds_temporal_features": "erds",
    "extract_erds_slope_features": "erds",
    
    # -------------------------------------------------------------------------
    # Connectivity
    # -------------------------------------------------------------------------
    "extract_connectivity_features": "connectivity",
    "compute_sliding_connectivity_features": "connectivity",
    
    # -------------------------------------------------------------------------
    # Phase (ITPC, PAC)
    # -------------------------------------------------------------------------
    "extract_itpc_features": "phase",
    "extract_trialwise_itpc_features": "phase",
    "compute_pac_comodulograms": "phase",
    
    # -------------------------------------------------------------------------
    # Temporal & Complexity
    # -------------------------------------------------------------------------
    "extract_statistical_features": "temporal",
    "extract_amplitude_features": "temporal",
    "extract_waveform_features": "temporal",
    "extract_percentile_features": "temporal",
    "extract_derivative_features": "temporal",
    "extract_all_temporal_features": "temporal",
    "extract_band_statistical_features": "temporal",
    "extract_band_amplitude_features": "temporal",
    "extract_band_waveform_features": "temporal",
    "extract_all_band_temporal_features": "temporal",
    "extract_permutation_entropy_features": "complexity",
    "extract_sample_entropy_features": "complexity",
    "extract_hjorth_parameters": "complexity",
    "extract_lempel_ziv_complexity": "complexity",
    "extract_all_complexity_features": "complexity",
    "extract_band_permutation_entropy_features": "complexity",
    "extract_band_hjorth_parameters": "complexity",
    "extract_band_lempel_ziv_complexity": "complexity",
    "extract_all_band_complexity_features": "complexity",
    
    # -------------------------------------------------------------------------
    # Aperiodic (1/f)
    # -------------------------------------------------------------------------
    "extract_aperiodic_features": "aperiodic",
    
    # -------------------------------------------------------------------------
    # ROI & Global
    # -------------------------------------------------------------------------
    "extract_roi_power_features": "roi_features",
    "extract_roi_asymmetry_features": "roi_features",
    "extract_roi_connectivity_features": "roi_features",
    "extract_pain_roi_features": "roi_features",
    "extract_gfp_features": "global_features",
    "extract_gfp_band_features": "global_features",
    "extract_global_synchrony_features": "global_features",
    "extract_variance_explained_features": "global_features",
    "extract_all_global_features": "global_features",
    
    # -------------------------------------------------------------------------
    # Microstates
    # -------------------------------------------------------------------------
    "extract_microstate_features": "microstates",
    "zscore_maps": "microstates",
    "compute_gfp_with_floor": "microstates",
    "corr_maps": "microstates",
    "label_timecourse": "microstates",
    "extract_templates_from_trials": "microstates",
    "MicrostateTransitionStats": "microstates",
    "MicrostateDurationStat": "microstates",
    
    # -------------------------------------------------------------------------
    # Core Utilities (additional)
    # -------------------------------------------------------------------------
    "get_band_power_in_window": "core",
    "get_psd_band_power": "core",
    "compute_gfp_with_peaks": "core",
    "compute_band_envelope_fast": "core",
    "compute_band_phase_fast": "core",
    "match_channels_to_pattern": "core",
    "pick_eeg_channels": "core",
    "bandpass_filter_epochs": "core",
    "compute_gfp": "core",
    
    # -------------------------------------------------------------------------
    # Plateau (legacy)
    # -------------------------------------------------------------------------
    "build_plateau_features": "plateau",
}


def __getattr__(name: str):
    """Lazy import for feature extraction functions."""
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available names for tab completion."""
    return list(__all__) + list(_LAZY_IMPORTS.keys())

