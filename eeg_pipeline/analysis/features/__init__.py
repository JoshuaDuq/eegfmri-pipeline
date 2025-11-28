"""
EEG feature extraction modules.

Submodules:
- power: Band power extraction
- connectivity: wPLI, AEC, graph metrics
- microstates: Microstate dynamics
- aperiodic: 1/f spectral features
- phase: ITPC, PAC
- spectral: IAF, entropy, ratios
- complexity: PE, Hjorth, LZC
- temporal: Statistical moments
- erds: ERD/ERS
- roi_features: ROI-averaged features
- global_features: GFP, synchrony
- core: Shared utilities and data containers
- pipeline: Feature orchestration
"""

# Core data containers (commonly used)
from eeg_pipeline.analysis.features.core import (
    PrecomputedData,
    FeatureResult,
    precompute_data,
    build_roi_map,
)

# Main extraction functions (commonly used)
from eeg_pipeline.analysis.features.power import extract_band_power_features
from eeg_pipeline.analysis.features.connectivity import extract_connectivity_features
from eeg_pipeline.analysis.features.microstates import extract_microstate_features
from eeg_pipeline.analysis.features.aperiodic import extract_aperiodic_features
from eeg_pipeline.analysis.features.phase import extract_itpc_features, compute_pac_comodulograms
from eeg_pipeline.analysis.features.pipeline import (
    extract_precomputed_features,
    extract_fmri_prediction_features,
    ExtractionResult,
    FeatureSet,
)
__all__ = [
    # Core
    "PrecomputedData",
    "FeatureResult",
    "precompute_data",
    "build_roi_map",
    # Pipeline containers
    "ExtractionResult",
    "FeatureSet",
    # Main extractors
    "extract_band_power_features",
    "extract_connectivity_features",
    "extract_microstate_features",
    "extract_aperiodic_features",
    "extract_itpc_features",
    "compute_pac_comodulograms",
    "extract_precomputed_features",
    "extract_fmri_prediction_features",
]


def __getattr__(name: str):
    """Lazy import for all feature extraction functions."""
    _module_map = {
        # Power
        "extract_baseline_power_features": "power",
        # Connectivity
        "compute_sliding_connectivity_features": "connectivity",
        # Microstates
        "zscore_maps": "microstates",
        "compute_gfp_with_floor": "microstates",
        "corr_maps": "microstates",
        "label_timecourse": "microstates",
        "extract_templates_from_trials": "microstates",
        "MicrostateTransitionStats": "microstates",
        "MicrostateDurationStat": "microstates",
        # Phase
        "extract_trialwise_itpc_features": "phase",
        # Plateau
        "build_plateau_features": "plateau",
        # Spectral
        "extract_individual_alpha_frequency": "spectral",
        "extract_relative_band_power": "spectral",
        "extract_band_power_ratios": "spectral",
        "extract_spectral_entropy_features": "spectral",
        "extract_peak_frequencies": "spectral",
        # Complexity
        "extract_permutation_entropy_features": "complexity",
        "extract_sample_entropy_features": "complexity",
        "extract_hjorth_parameters": "complexity",
        "extract_lempel_ziv_complexity": "complexity",
        "extract_all_complexity_features": "complexity",
        # Complexity (band-filtered)
        "extract_band_permutation_entropy_features": "complexity",
        "extract_band_hjorth_parameters": "complexity",
        "extract_band_lempel_ziv_complexity": "complexity",
        "extract_all_band_complexity_features": "complexity",
        # Temporal
        "extract_statistical_features": "temporal",
        "extract_amplitude_features": "temporal",
        "extract_waveform_features": "temporal",
        "extract_percentile_features": "temporal",
        "extract_derivative_features": "temporal",
        "extract_all_temporal_features": "temporal",
        # Temporal (band-filtered)
        "extract_band_statistical_features": "temporal",
        "extract_band_amplitude_features": "temporal",
        "extract_band_waveform_features": "temporal",
        "extract_all_band_temporal_features": "temporal",
        # ERDS
        "extract_erds_features": "erds",
        "extract_erds_temporal_features": "erds",
        "extract_erds_slope_features": "erds",
        # ROI
        "extract_roi_power_features": "roi_features",
        "extract_roi_asymmetry_features": "roi_features",
        "extract_roi_connectivity_features": "roi_features",
        "extract_pain_roi_features": "roi_features",
        # Global
        "extract_gfp_features": "global_features",
        "extract_gfp_band_features": "global_features",
        "extract_global_synchrony_features": "global_features",
        "extract_variance_explained_features": "global_features",
        "extract_all_global_features": "global_features",
        # Core
        "BandData": "core",
        "PSDData": "core",
        "TimeWindows": "core",
        "get_band_power_in_window": "core",
        "get_psd_band_power": "core",
        "compute_gfp_with_peaks": "core",
        "compute_band_envelope_fast": "core",
        "compute_band_phase_fast": "core",
        "match_channels_to_pattern": "core",
        "pick_eeg_channels": "core",
        "bandpass_filter_epochs": "core",
        "compute_gfp": "core",
        "EPSILON_STD": "core",
        "EPSILON_PSD": "core",
        "EPSILON_AMP": "core",
        # Pipeline
        "extract_fmri_prediction_features": "pipeline",
        "get_feature_groups_for_ml": "pipeline",
        "ExtractionResult": "pipeline",
        "FeatureSet": "pipeline",
    }

    if name in _module_map:
        import importlib
        module = importlib.import_module(f".{_module_map[name]}", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

