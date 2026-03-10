"""Helper functions for features extraction CLI command."""

from __future__ import annotations

import argparse
import re
from typing import Any, List

from eeg_pipeline.pipelines.constants import FEATURE_CATEGORIES
from eeg_pipeline.cli.commands.base import FEATURE_VISUALIZE_CATEGORIES
from eeg_pipeline.utils.parsing import (
    parse_frequency_band_definitions,
    parse_roi_definitions,
)

FEATURE_CATEGORY_CHOICES = FEATURE_CATEGORIES + [
    category for category in FEATURE_VISUALIZE_CATEGORIES if category not in FEATURE_CATEGORIES
]

_COMPONENT_RANGE_RE = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*-\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$"
)

def _split_list_tokens(tokens: List[str]) -> List[str]:
    parts: List[str] = []
    for token in tokens:
        for chunk in re.split(r"[;,]", str(token)):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
    return parts


def _parse_pair_tokens(tokens: List[str], *, label: str) -> List[List[str]]:
    pairs: List[List[str]] = []
    for token in _split_list_tokens(tokens):
        sep = None
        for candidate in (":", "-", "/", "|"):
            if candidate in token:
                sep = candidate
                break
        if sep is None:
            raise ValueError(f"Invalid {label} pair token {token!r}; expected e.g. A:B or A-B")
        left, right = token.split(sep, 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Invalid {label} pair token {token!r}; expected e.g. A:B")
        pairs.append([left, right])
    return pairs


def _parse_erp_components(tokens: List[str]) -> List[dict]:
    components: List[dict] = []
    for token in _split_list_tokens(tokens):
        if "=" in token:
            name, rest = token.split("=", 1)
        elif ":" in token:
            name, rest = token.split(":", 1)
        else:
            raise ValueError(
                f"Invalid ERP component token {token!r}; expected e.g. n2=0.20-0.35"
            )
        name = name.strip().lower()
        rest = rest.strip()
        if not name:
            raise ValueError(f"Invalid ERP component token {token!r}; missing name")
        match = _COMPONENT_RANGE_RE.match(rest)
        if not match:
            raise ValueError(
                f"Invalid ERP component range {rest!r}; expected start-end (seconds), e.g. 0.20-0.35"
            )
        start = float(match.group(1))
        end = float(match.group(2))
        if not (start < end):
            raise ValueError(f"Invalid ERP component range {rest!r}; expected start < end")
        components.append({"name": name, "start": start, "end": end})
    return components


def _apply_connectivity_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply connectivity-related config overrides."""
    if getattr(args, "connectivity_measures", None) is not None:
        config["feature_engineering.connectivity.measures"] = args.connectivity_measures
    if getattr(args, "conn_output_level", None) is not None:
        config["feature_engineering.connectivity.output_level"] = args.conn_output_level
    if getattr(args, "conn_graph_metrics", None) is not None:
        config["feature_engineering.connectivity.enable_graph_metrics"] = args.conn_graph_metrics
    if getattr(args, "conn_aec_mode", None) is not None:
        config["feature_engineering.connectivity.aec_mode"] = args.conn_aec_mode
    if getattr(args, "conn_graph_prop", None) is not None:
        config["feature_engineering.connectivity.graph_top_prop"] = args.conn_graph_prop
    if getattr(args, "aec_output", None) is not None:
        config["feature_engineering.connectivity.aec_output"] = args.aec_output
    if getattr(args, "conn_force_within_epoch_for_ml", None) is not None:
        config["feature_engineering.connectivity.force_within_epoch_for_ml"] = args.conn_force_within_epoch_for_ml
    
    conn_cfg = config.setdefault("feature_engineering", {}).setdefault("connectivity", {})
    if getattr(args, "conn_window_len", None) is not None:
        conn_cfg["sliding_window_len"] = args.conn_window_len
    if getattr(args, "conn_window_step", None) is not None:
        conn_cfg["sliding_window_step"] = args.conn_window_step
    if getattr(args, "conn_granularity", None) is not None:
        conn_cfg["granularity"] = args.conn_granularity
    if getattr(args, "conn_condition_column", None) is not None:
        conn_cfg["condition_column"] = args.conn_condition_column
    if getattr(args, "conn_condition_values", None) is not None:
        conn_cfg["condition_values"] = args.conn_condition_values
    if getattr(args, "conn_min_epochs_per_group", None) is not None:
        conn_cfg["min_epochs_per_group"] = args.conn_min_epochs_per_group
    if getattr(args, "conn_min_cycles_per_band", None) is not None:
        conn_cfg["min_cycles_per_band"] = args.conn_min_cycles_per_band
    if getattr(args, "conn_warn_no_spatial_transform", None) is not None:
        conn_cfg["warn_if_no_spatial_transform"] = args.conn_warn_no_spatial_transform
    if getattr(args, "conn_phase_estimator", None) is not None:
        conn_cfg["phase_estimator"] = args.conn_phase_estimator
    if getattr(args, "conn_min_segment_sec", None) is not None:
        conn_cfg["min_segment_sec"] = args.conn_min_segment_sec
    if getattr(args, "conn_mode", None) is not None:
        conn_cfg["mode"] = args.conn_mode
    if getattr(args, "conn_aec_absolute", None) is not None:
        conn_cfg["aec_absolute"] = args.conn_aec_absolute
    if getattr(args, "conn_n_freqs_per_band", None) is not None:
        conn_cfg["n_freqs_per_band"] = args.conn_n_freqs_per_band
    if getattr(args, "conn_n_cycles", None) is not None:
        conn_cfg["n_cycles"] = args.conn_n_cycles
    if getattr(args, "conn_decim", None) is not None:
        conn_cfg["decim"] = args.conn_decim
    if getattr(args, "conn_min_segment_samples", None) is not None:
        conn_cfg["min_segment_samples"] = args.conn_min_segment_samples
    if getattr(args, "conn_small_world_n_rand", None) is not None:
        conn_cfg["small_world_n_rand"] = args.conn_small_world_n_rand
    if getattr(args, "conn_enable_aec", None) is not None:
        conn_cfg["enable_aec"] = args.conn_enable_aec
    if getattr(args, "conn_dynamic_enabled", None) is not None:
        conn_cfg["dynamic_enabled"] = args.conn_dynamic_enabled
    if getattr(args, "conn_dynamic_measures", None) is not None:
        conn_cfg["dynamic_measures"] = args.conn_dynamic_measures
    if getattr(args, "conn_dynamic_autocorr_lag", None) is not None:
        conn_cfg["dynamic_autocorr_lag"] = args.conn_dynamic_autocorr_lag
    if getattr(args, "conn_dynamic_min_windows", None) is not None:
        conn_cfg["dynamic_min_windows"] = args.conn_dynamic_min_windows
    if getattr(args, "conn_dynamic_include_roi_pairs", None) is not None:
        conn_cfg["dynamic_include_roi_pairs"] = args.conn_dynamic_include_roi_pairs
    if getattr(args, "conn_dynamic_state_enabled", None) is not None:
        conn_cfg["dynamic_state_enabled"] = args.conn_dynamic_state_enabled
    if getattr(args, "conn_dynamic_state_n_states", None) is not None:
        conn_cfg["dynamic_state_n_states"] = args.conn_dynamic_state_n_states
    if getattr(args, "conn_dynamic_state_min_windows", None) is not None:
        conn_cfg["dynamic_state_min_windows"] = args.conn_dynamic_state_min_windows
    if getattr(args, "conn_dynamic_state_random_state", None) is not None:
        conn_cfg["dynamic_state_random_state"] = args.conn_dynamic_state_random_state


def _apply_directedconnectivity_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply directed connectivity-related config overrides (PSI, DTF, PDC)."""
    dconn_cfg = config.setdefault("feature_engineering", {}).setdefault("directedconnectivity", {})
    
    if getattr(args, "directed_connectivity_measures", None) is not None:
        measures = args.directed_connectivity_measures
        dconn_cfg["enable_psi"] = "psi" in measures
        dconn_cfg["enable_dtf"] = "dtf" in measures
        dconn_cfg["enable_pdc"] = "pdc" in measures
    if getattr(args, "directed_conn_output_level", None) is not None:
        dconn_cfg["output_level"] = args.directed_conn_output_level
    if getattr(args, "directed_conn_mvar_order", None) is not None:
        dconn_cfg["mvar_order"] = args.directed_conn_mvar_order
    if getattr(args, "directed_conn_n_freqs", None) is not None:
        dconn_cfg["n_freqs"] = args.directed_conn_n_freqs
    if getattr(args, "directed_conn_min_segment_samples", None) is not None:
        dconn_cfg["min_segment_samples"] = args.directed_conn_min_segment_samples
    if getattr(args, "directed_conn_min_samples_per_mvar_param", None) is not None:
        dconn_cfg["min_samples_per_mvar_parameter"] = args.directed_conn_min_samples_per_mvar_param


def _apply_sourcelocalization_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply source localization-related config overrides (LCMV, eLORETA)."""
    src_cfg = config.setdefault("feature_engineering", {}).setdefault("sourcelocalization", {})
    
    if getattr(args, "source_method", None) is not None:
        src_cfg["method"] = args.source_method
    if getattr(args, "source_spacing", None) is not None:
        src_cfg["spacing"] = args.source_spacing
    if getattr(args, "source_reg", None) is not None:
        src_cfg["reg"] = args.source_reg
    if getattr(args, "source_snr", None) is not None:
        src_cfg["snr"] = args.source_snr
    if getattr(args, "source_loose", None) is not None:
        src_cfg["loose"] = args.source_loose
    if getattr(args, "source_depth", None) is not None:
        src_cfg["depth"] = args.source_depth
    if getattr(args, "source_parc", None) is not None:
        src_cfg["parcellation"] = args.source_parc
    if getattr(args, "source_connectivity_method", None) is not None:
        src_cfg["connectivity_method"] = args.source_connectivity_method
    if getattr(args, "source_subject", None) is not None:
        src_cfg["subject"] = args.source_subject
    if getattr(args, "source_subjects_dir", None) is not None:
        src_cfg["subjects_dir"] = args.source_subjects_dir
    if getattr(args, "source_trans", None) is not None:
        src_cfg["trans"] = args.source_trans
    if getattr(args, "source_bem", None) is not None:
        src_cfg["bem"] = args.source_bem
    if getattr(args, "source_mindist_mm", None) is not None:
        src_cfg["mindist_mm"] = args.source_mindist_mm
    if getattr(args, "source_save_stc", None) is not None:
        src_cfg["save_stc"] = args.source_save_stc

    contrast_cfg = src_cfg.setdefault("contrast", {})
    if getattr(args, "source_contrast_enabled", None) is not None:
        contrast_cfg["enabled"] = args.source_contrast_enabled
    if getattr(args, "source_contrast_condition_column", None) is not None:
        contrast_cfg["condition_column"] = args.source_contrast_condition_column
    if getattr(args, "source_contrast_condition_a", None) is not None:
        contrast_cfg["condition_a"] = args.source_contrast_condition_a
    if getattr(args, "source_contrast_condition_b", None) is not None:
        contrast_cfg["condition_b"] = args.source_contrast_condition_b
    if getattr(args, "source_contrast_min_trials_per_condition", None) is not None:
        contrast_cfg["min_trials_per_condition"] = args.source_contrast_min_trials_per_condition
    if getattr(args, "source_contrast_welch_stats", None) is not None:
        contrast_cfg["emit_welch_stats"] = args.source_contrast_welch_stats

    bem_cfg = src_cfg.setdefault("bem_generation", {})
    if getattr(args, "source_create_trans", None) is not None:
        bem_cfg["create_trans"] = args.source_create_trans
    if getattr(args, "source_create_bem_model", None) is not None:
        bem_cfg["create_model"] = args.source_create_bem_model
    if getattr(args, "source_create_bem_solution", None) is not None:
        bem_cfg["create_solution"] = args.source_create_bem_solution
    if getattr(args, "source_allow_identity_trans", None) is not None:
        bem_cfg["allow_identity_trans"] = args.source_allow_identity_trans

    fmri_cfg = src_cfg.setdefault("fmri", {})
    if getattr(args, "source_fmri_enabled", None) is not None:
        fmri_cfg["enabled"] = args.source_fmri_enabled
        # Sync mode with fMRI enabled/disabled state so that STC output
        # directories (eeg_only/ vs fmri_informed/) match the actual
        # computation mode.
        if args.source_fmri_enabled:
            src_cfg["mode"] = "fmri_informed"
        else:
            src_cfg["mode"] = "eeg_only"
    if getattr(args, "source_fmri_stats_map", None) is not None:
        fmri_cfg["stats_map_path"] = args.source_fmri_stats_map
    if getattr(args, "source_fmri_provenance", None) is not None:
        fmri_cfg["provenance"] = args.source_fmri_provenance
    if getattr(args, "source_fmri_require_provenance", None) is not None:
        fmri_cfg["require_provenance"] = bool(args.source_fmri_require_provenance)
    if getattr(args, "source_fmri_threshold", None) is not None:
        fmri_cfg["threshold"] = args.source_fmri_threshold
    if getattr(args, "source_fmri_tail", None) is not None:
        fmri_cfg["tail"] = args.source_fmri_tail
    if getattr(args, "source_fmri_cluster_min_voxels", None) is not None:
        fmri_cfg["cluster_min_voxels"] = args.source_fmri_cluster_min_voxels
        # If the user explicitly sets a voxel threshold, clear the volume threshold unless they also set it.
        if getattr(args, "source_fmri_cluster_min_mm3", None) is None:
            fmri_cfg["cluster_min_volume_mm3"] = None
    if getattr(args, "source_fmri_cluster_min_mm3", None) is not None:
        fmri_cfg["cluster_min_volume_mm3"] = args.source_fmri_cluster_min_mm3
    if getattr(args, "source_fmri_max_clusters", None) is not None:
        fmri_cfg["max_clusters"] = args.source_fmri_max_clusters
    if getattr(args, "source_fmri_max_voxels_per_cluster", None) is not None:
        fmri_cfg["max_voxels_per_cluster"] = args.source_fmri_max_voxels_per_cluster
    if getattr(args, "source_fmri_max_total_voxels", None) is not None:
        fmri_cfg["max_total_voxels"] = args.source_fmri_max_total_voxels
    if getattr(args, "source_fmri_random_seed", None) is not None:
        fmri_cfg["random_seed"] = args.source_fmri_random_seed
    if getattr(args, "source_fmri_output_space", None) is not None:
        fmri_cfg["output_space"] = args.source_fmri_output_space

    contrast_cfg = fmri_cfg.setdefault("contrast", {})
    if getattr(args, "source_fmri_contrast_enabled", None) is not None:
        contrast_cfg["enabled"] = args.source_fmri_contrast_enabled
    if getattr(args, "source_fmri_contrast_type", None) is not None:
        contrast_cfg["type"] = args.source_fmri_contrast_type
    
    cond_a_cfg = contrast_cfg.setdefault("condition_a", {})
    if getattr(args, "source_fmri_cond_a_column", None) is not None:
        cond_a_cfg["column"] = args.source_fmri_cond_a_column
    if getattr(args, "source_fmri_cond_a_value", None) is not None:
        cond_a_cfg["value"] = args.source_fmri_cond_a_value
    
    cond_b_cfg = contrast_cfg.setdefault("condition_b", {})
    if getattr(args, "source_fmri_cond_b_column", None) is not None:
        cond_b_cfg["column"] = args.source_fmri_cond_b_column
    if getattr(args, "source_fmri_cond_b_value", None) is not None:
        cond_b_cfg["value"] = args.source_fmri_cond_b_value
    
    if getattr(args, "source_fmri_contrast_formula", None) is not None:
        contrast_cfg["formula"] = args.source_fmri_contrast_formula
    if getattr(args, "source_fmri_contrast_name", None) is not None:
        contrast_cfg["name"] = args.source_fmri_contrast_name
    if getattr(args, "source_fmri_runs", None) is not None:
        contrast_cfg["runs"] = [int(r.strip()) for r in args.source_fmri_runs.split(",") if r.strip()]
    if getattr(args, "source_fmri_hrf_model", None) is not None:
        contrast_cfg["hrf_model"] = args.source_fmri_hrf_model
    if getattr(args, "source_fmri_drift_model", None) is not None:
        contrast_cfg["drift_model"] = args.source_fmri_drift_model
    if getattr(args, "source_fmri_high_pass", None) is not None:
        contrast_cfg["high_pass_hz"] = args.source_fmri_high_pass
    if getattr(args, "source_fmri_low_pass", None) is not None:
        contrast_cfg["low_pass_hz"] = args.source_fmri_low_pass
    if getattr(args, "source_fmri_events_to_model", None) is not None:
        contrast_cfg["events_to_model"] = args.source_fmri_events_to_model
    if getattr(args, "source_fmri_events_to_model_column", None) is not None:
        contrast_cfg["events_to_model_column"] = args.source_fmri_events_to_model_column
    if getattr(args, "source_fmri_stim_phases_to_model", None) is not None:
        contrast_cfg["stim_phases_to_model"] = args.source_fmri_stim_phases_to_model
    if getattr(args, "source_fmri_phase_column", None) is not None:
        contrast_cfg["phase_column"] = args.source_fmri_phase_column
    if getattr(args, "source_fmri_phase_scope_column", None) is not None:
        contrast_cfg["phase_scope_column"] = args.source_fmri_phase_scope_column
    if getattr(args, "source_fmri_phase_scope_value", None) is not None:
        contrast_cfg["phase_scope_value"] = args.source_fmri_phase_scope_value
    if getattr(args, "source_fmri_condition_scope_trial_types", None) is not None:
        contrast_cfg["condition_scope_trial_types"] = args.source_fmri_condition_scope_trial_types
    if getattr(args, "source_fmri_condition_scope_column", None) is not None:
        contrast_cfg["condition_scope_column"] = args.source_fmri_condition_scope_column
    if getattr(args, "source_fmri_output_type", None) is not None:
        contrast_cfg["output_type"] = args.source_fmri_output_type
    if getattr(args, "source_fmri_resample_to_fs", None) is not None:
        contrast_cfg["resample_to_freesurfer"] = args.source_fmri_resample_to_fs
    if getattr(args, "source_fmri_input_source", None) is not None:
        contrast_cfg["input_source"] = args.source_fmri_input_source
    if getattr(args, "source_fmri_require_fmriprep", None) is not None:
        contrast_cfg["require_fmriprep"] = args.source_fmri_require_fmriprep


def _apply_pac_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply PAC/CFC-related config overrides."""
    if getattr(args, "pac_phase_range", None) is not None:
        config["feature_engineering.pac.phase_range"] = list(args.pac_phase_range)
    if getattr(args, "pac_amp_range", None) is not None:
        config["feature_engineering.pac.amp_range"] = list(args.pac_amp_range)
    if getattr(args, "pac_method", None) is not None:
        config["feature_engineering.pac.method"] = args.pac_method
    if getattr(args, "pac_min_epochs", None) is not None:
        config["feature_engineering.pac.min_epochs"] = args.pac_min_epochs
    if getattr(args, "pac_pairs", None) is not None:
        config["feature_engineering.pac.pairs"] = _parse_pair_tokens(args.pac_pairs, label="PAC")
    
    pac_cfg = config.setdefault("feature_engineering", {}).setdefault("pac", {})
    if getattr(args, "pac_source", None) is not None:
        pac_cfg["source"] = args.pac_source
    if getattr(args, "pac_normalize", None) is not None:
        pac_cfg["normalize"] = args.pac_normalize
    if getattr(args, "pac_n_surrogates", None) is not None:
        pac_cfg["n_surrogates"] = args.pac_n_surrogates
    if getattr(args, "pac_allow_harmonic_overlap", None) is not None:
        pac_cfg["allow_harmonic_overlap"] = args.pac_allow_harmonic_overlap
    if getattr(args, "pac_max_harmonic", None) is not None:
        pac_cfg["max_harmonic"] = args.pac_max_harmonic
    if getattr(args, "pac_harmonic_tolerance_hz", None) is not None:
        pac_cfg["harmonic_tolerance_hz"] = args.pac_harmonic_tolerance_hz
    if getattr(args, "pac_compute_waveform_qc", None) is not None:
        pac_cfg["compute_waveform_qc"] = args.pac_compute_waveform_qc
    if getattr(args, "pac_waveform_offset_ms", None) is not None:
        pac_cfg["waveform_offset_ms"] = args.pac_waveform_offset_ms
    if getattr(args, "pac_random_seed", None) is not None:
        pac_cfg["random_seed"] = args.pac_random_seed
    if getattr(args, "pac_min_segment_sec", None) is not None:
        pac_cfg["min_segment_sec"] = args.pac_min_segment_sec
    if getattr(args, "pac_min_cycles_at_fmin", None) is not None:
        pac_cfg["min_cycles_at_fmin"] = args.pac_min_cycles_at_fmin
    if getattr(args, "pac_surrogate_method", None) is not None:
        pac_cfg["surrogate_method"] = args.pac_surrogate_method


def _apply_aperiodic_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply aperiodic-related config overrides."""
    if getattr(args, "aperiodic_range", None) is not None:
        config["feature_engineering.aperiodic.fmin"] = args.aperiodic_range[0]
        config["feature_engineering.aperiodic.fmax"] = args.aperiodic_range[1]
    if getattr(args, "aperiodic_peak_z", None) is not None:
        config["feature_engineering.aperiodic.peak_rejection_z"] = args.aperiodic_peak_z
    if getattr(args, "aperiodic_min_r2", None) is not None:
        config["feature_engineering.aperiodic.min_r2"] = args.aperiodic_min_r2
    if getattr(args, "aperiodic_min_points", None) is not None:
        config["feature_engineering.aperiodic.min_fit_points"] = args.aperiodic_min_points
    if getattr(args, "aperiodic_min_segment_sec", None) is not None:
        config["feature_engineering.aperiodic.min_segment_sec"] = args.aperiodic_min_segment_sec
    if getattr(args, "aperiodic_psd_bandwidth", None) is not None:
        config["feature_engineering.aperiodic.psd_bandwidth"] = args.aperiodic_psd_bandwidth
    if getattr(args, "aperiodic_max_rms", None) is not None:
        config["feature_engineering.aperiodic.max_rms"] = args.aperiodic_max_rms
    
    # Scientific validity: induced spectra option
    if getattr(args, "aperiodic_subtract_evoked", None) is not None:
        config["feature_engineering.aperiodic.subtract_evoked"] = args.aperiodic_subtract_evoked
    
    aperiodic_cfg = config.setdefault("feature_engineering", {}).setdefault("aperiodic", {})
    if getattr(args, "aperiodic_model", None) is not None:
        aperiodic_cfg["model"] = args.aperiodic_model
    if getattr(args, "aperiodic_psd_method", None) is not None:
        aperiodic_cfg["psd_method"] = args.aperiodic_psd_method
    if getattr(args, "aperiodic_exclude_line_noise", None) is not None:
        aperiodic_cfg["exclude_line_noise"] = args.aperiodic_exclude_line_noise
    if getattr(args, "aperiodic_line_noise_freq", None) is not None:
        aperiodic_cfg["line_noise_freqs"] = [args.aperiodic_line_noise_freq]
    if getattr(args, "aperiodic_line_noise_width_hz", None) is not None:
        aperiodic_cfg["line_noise_width_hz"] = args.aperiodic_line_noise_width_hz
    if getattr(args, "aperiodic_line_noise_harmonics", None) is not None:
        aperiodic_cfg["line_noise_harmonics"] = args.aperiodic_line_noise_harmonics
    if getattr(args, "aperiodic_max_freq_resolution_hz", None) is not None:
        aperiodic_cfg["max_freq_resolution_hz"] = args.aperiodic_max_freq_resolution_hz
    if getattr(args, "aperiodic_multitaper_adaptive", None) is not None:
        aperiodic_cfg["multitaper_adaptive"] = bool(args.aperiodic_multitaper_adaptive)


def _apply_complexity_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply complexity-related config overrides."""
    if getattr(args, "pe_order", None) is not None:
        config["feature_engineering.complexity.pe_order"] = args.pe_order
    if getattr(args, "pe_delay", None) is not None:
        config["feature_engineering.complexity.pe_delay"] = args.pe_delay

    complexity_cfg = config.setdefault("feature_engineering", {}).setdefault("complexity", {})
    if getattr(args, "pe_order", None) is not None:
        complexity_cfg["pe_order"] = args.pe_order
    if getattr(args, "pe_delay", None) is not None:
        complexity_cfg["pe_delay"] = args.pe_delay
    if getattr(args, "complexity_signal_basis", None) is not None:
        complexity_cfg["signal_basis"] = args.complexity_signal_basis
    if getattr(args, "complexity_min_segment_sec", None) is not None:
        complexity_cfg["min_segment_sec"] = args.complexity_min_segment_sec
    if getattr(args, "complexity_min_samples", None) is not None:
        complexity_cfg["min_samples"] = args.complexity_min_samples
    if getattr(args, "complexity_zscore", None) is not None:
        complexity_cfg["zscore"] = args.complexity_zscore
    if getattr(args, "complexity_sampen_order", None) is not None:
        complexity_cfg["sampen_order"] = args.complexity_sampen_order
    if getattr(args, "complexity_sampen_r", None) is not None:
        complexity_cfg["sampen_r"] = args.complexity_sampen_r
    if getattr(args, "complexity_mse_scale_min", None) is not None:
        complexity_cfg["mse_scale_min"] = args.complexity_mse_scale_min
    if getattr(args, "complexity_mse_scale_max", None) is not None:
        complexity_cfg["mse_scale_max"] = args.complexity_mse_scale_max


def _apply_erp_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply ERP-related config overrides."""
    if getattr(args, "erp_baseline", None) is not None:
        config["feature_engineering.erp.baseline_correction"] = args.erp_baseline
    if getattr(args, "erp_allow_no_baseline", None) is not None:
        config["feature_engineering.erp.allow_no_baseline"] = args.erp_allow_no_baseline
    if getattr(args, "erp_components", None) is not None:
        config["feature_engineering.erp.components"] = _parse_erp_components(args.erp_components)
    if getattr(args, "erp_lowpass_hz", None) is not None:
        config["feature_engineering.erp.lowpass_hz"] = args.erp_lowpass_hz
    if getattr(args, "erp_smooth_ms", None) is not None:
        config["feature_engineering.erp.smooth_ms"] = args.erp_smooth_ms
    if getattr(args, "erp_peak_prominence_uv", None) is not None:
        config["feature_engineering.erp.peak_prominence_uv"] = args.erp_peak_prominence_uv


def _apply_burst_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply burst-related config overrides."""
    if getattr(args, "burst_threshold", None) is not None:
        config["feature_engineering.bursts.threshold_z"] = args.burst_threshold
    if getattr(args, "burst_threshold_method", None) is not None:
        config["feature_engineering.bursts.threshold_method"] = args.burst_threshold_method
    if getattr(args, "burst_threshold_percentile", None) is not None:
        config["feature_engineering.bursts.threshold_percentile"] = args.burst_threshold_percentile
    if getattr(args, "burst_threshold_reference", None) is not None:
        config["feature_engineering.bursts.threshold_reference"] = args.burst_threshold_reference
    if getattr(args, "burst_min_trials_per_condition", None) is not None:
        config["feature_engineering.bursts.min_trials_per_condition"] = int(
            args.burst_min_trials_per_condition
        )
    if getattr(args, "burst_min_segment_sec", None) is not None:
        config["feature_engineering.bursts.min_segment_sec"] = args.burst_min_segment_sec
    if getattr(args, "burst_skip_invalid_segments", None) is not None:
        config["feature_engineering.bursts.skip_invalid_segments"] = args.burst_skip_invalid_segments
    if getattr(args, "burst_bands", None) is not None:
        config["feature_engineering.bursts.bands"] = list(_split_list_tokens(args.burst_bands))
    if getattr(args, "burst_min_duration", None) is not None:
        config["feature_engineering.bursts.min_duration_ms"] = args.burst_min_duration
    if getattr(args, "burst_min_cycles", None) is not None:
        config["feature_engineering.bursts.min_cycles"] = args.burst_min_cycles


def _apply_power_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply power-related config overrides."""
    if getattr(args, "power_baseline_mode", None) is not None:
        config["time_frequency_analysis.baseline_mode"] = args.power_baseline_mode
    if getattr(args, "power_require_baseline", None) is not None:
        config["feature_engineering.power.require_baseline"] = args.power_require_baseline
    if getattr(args, "power_subtract_evoked", None) is not None:
        config["feature_engineering.power.subtract_evoked"] = args.power_subtract_evoked
    if getattr(args, "power_min_trials_per_condition", None) is not None:
        config["feature_engineering.power.min_trials_per_condition"] = args.power_min_trials_per_condition
    if getattr(args, "power_exclude_line_noise", None) is not None:
        config["feature_engineering.power.exclude_line_noise"] = args.power_exclude_line_noise
    if getattr(args, "power_line_noise_freq", None) is not None:
        config["feature_engineering.power.line_noise_freqs"] = [args.power_line_noise_freq]
    if getattr(args, "power_line_noise_width_hz", None) is not None:
        config["feature_engineering.power.line_noise_width_hz"] = args.power_line_noise_width_hz
    if getattr(args, "power_line_noise_harmonics", None) is not None:
        config["feature_engineering.power.line_noise_harmonics"] = args.power_line_noise_harmonics
    if getattr(args, "power_emit_db", None) is not None:
        config["feature_engineering.power.emit_db"] = args.power_emit_db


def _apply_spectral_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply spectral-related config overrides."""
    if getattr(args, "ratio_pairs", None) is not None:
        config["feature_engineering.spectral.ratio_pairs"] = _parse_pair_tokens(args.ratio_pairs, label="ratio")
    
    spectral_cfg = config.setdefault("feature_engineering", {}).setdefault("spectral", {})
    if getattr(args, "spectral_include_log_ratios", None) is not None:
        spectral_cfg["include_log_ratios"] = args.spectral_include_log_ratios
    if getattr(args, "spectral_psd_method", None) is not None:
        spectral_cfg["psd_method"] = args.spectral_psd_method
    if getattr(args, "spectral_psd_adaptive", None) is not None:
        spectral_cfg["psd_adaptive"] = args.spectral_psd_adaptive
    if getattr(args, "spectral_multitaper_adaptive", None) is not None:
        spectral_cfg["multitaper_adaptive"] = args.spectral_multitaper_adaptive
    if getattr(args, "spectral_fmin", None) is not None:
        spectral_cfg["fmin"] = args.spectral_fmin
    if getattr(args, "spectral_fmax", None) is not None:
        spectral_cfg["fmax"] = args.spectral_fmax
    if getattr(args, "spectral_exclude_line_noise", None) is not None:
        spectral_cfg["exclude_line_noise"] = args.spectral_exclude_line_noise
    if getattr(args, "spectral_line_noise_freq", None) is not None:
        spectral_cfg["line_noise_freqs"] = [args.spectral_line_noise_freq]
    if getattr(args, "spectral_line_noise_width_hz", None) is not None:
        spectral_cfg["line_noise_width_hz"] = args.spectral_line_noise_width_hz
    if getattr(args, "spectral_line_noise_harmonics", None) is not None:
        spectral_cfg["line_noise_harmonics"] = args.spectral_line_noise_harmonics
    if getattr(args, "spectral_segments", None) is not None:
        spectral_cfg["segments"] = args.spectral_segments
    if getattr(args, "spectral_min_segment_sec", None) is not None:
        spectral_cfg["min_segment_sec"] = args.spectral_min_segment_sec
    if getattr(args, "spectral_min_cycles_at_fmin", None) is not None:
        spectral_cfg["min_cycles_at_fmin"] = args.spectral_min_cycles_at_fmin


def _apply_asymmetry_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply asymmetry-related config overrides."""
    if getattr(args, "asymmetry_channel_pairs", None) is not None:
        config["feature_engineering.asymmetry.channel_pairs"] = _parse_pair_tokens(args.asymmetry_channel_pairs, label="asymmetry")
    if getattr(args, "asymmetry_activation_bands", None) is not None:
        config["feature_engineering.asymmetry.activation_bands"] = list(_split_list_tokens(args.asymmetry_activation_bands))
    if getattr(args, "asymmetry_emit_activation_convention", None) is not None:
        config["feature_engineering.asymmetry.emit_activation_convention"] = args.asymmetry_emit_activation_convention
    
    asym_cfg = config.setdefault("feature_engineering", {}).setdefault("asymmetry", {})
    if getattr(args, "asymmetry_min_segment_sec", None) is not None:
        asym_cfg["min_segment_sec"] = args.asymmetry_min_segment_sec
    if getattr(args, "asymmetry_min_cycles_at_fmin", None) is not None:
        asym_cfg["min_cycles_at_fmin"] = args.asymmetry_min_cycles_at_fmin
    if getattr(args, "asymmetry_skip_invalid_segments", None) is not None:
        asym_cfg["skip_invalid_segments"] = args.asymmetry_skip_invalid_segments


def _apply_ratios_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply band-ratio validity config overrides."""
    ratios_cfg = config.setdefault("feature_engineering", {}).setdefault("ratios", {})
    if getattr(args, "ratios_min_segment_sec", None) is not None:
        ratios_cfg["min_segment_sec"] = args.ratios_min_segment_sec
    if getattr(args, "ratios_min_cycles_at_fmin", None) is not None:
        ratios_cfg["min_cycles_at_fmin"] = args.ratios_min_cycles_at_fmin
    if getattr(args, "ratios_skip_invalid_segments", None) is not None:
        ratios_cfg["skip_invalid_segments"] = args.ratios_skip_invalid_segments


def _apply_tfr_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply TFR-related config overrides."""
    tfr_config = config.setdefault("time_frequency_analysis", {}).setdefault("tfr", {})
    if getattr(args, "tfr_freq_min", None) is not None:
        tfr_config["freq_min"] = args.tfr_freq_min
    if getattr(args, "tfr_freq_max", None) is not None:
        tfr_config["freq_max"] = args.tfr_freq_max
    if getattr(args, "tfr_n_freqs", None) is not None:
        tfr_config["n_freqs"] = args.tfr_n_freqs
    if getattr(args, "tfr_min_cycles", None) is not None:
        tfr_config["min_cycles"] = args.tfr_min_cycles
    if getattr(args, "tfr_n_cycles_factor", None) is not None:
        tfr_config["n_cycles_factor"] = args.tfr_n_cycles_factor
    if getattr(args, "tfr_workers", None) is not None:
        tfr_config["workers"] = args.tfr_workers
    if getattr(args, "tfr_max_cycles", None) is not None:
        tfr_config["max_cycles"] = args.tfr_max_cycles
    if getattr(args, "tfr_decim_power", None) is not None:
        tfr_config["decim_power"] = args.tfr_decim_power
    if getattr(args, "tfr_decim_phase", None) is not None:
        tfr_config["decim_phase"] = args.tfr_decim_phase


def _apply_itpc_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply ITPC-related config overrides."""
    if getattr(args, "itpc_method", None) is not None:
        config["feature_engineering.itpc.method"] = args.itpc_method
    if getattr(args, "itpc_allow_unsafe_loo", None) is not None:
        config["feature_engineering.itpc.allow_unsafe_loo"] = args.itpc_allow_unsafe_loo
    if getattr(args, "itpc_baseline_correction", None) is not None:
        config["feature_engineering.itpc.baseline_correction"] = args.itpc_baseline_correction
    if getattr(args, "itpc_condition_column", None) is not None:
        config["feature_engineering.itpc.condition_column"] = args.itpc_condition_column
    if getattr(args, "itpc_condition_values", None) is not None:
        config["feature_engineering.itpc.condition_values"] = args.itpc_condition_values
    if getattr(args, "itpc_min_trials_per_condition", None) is not None:
        config["feature_engineering.itpc.min_trials_per_condition"] = args.itpc_min_trials_per_condition
    if getattr(args, "itpc_n_jobs", None) is not None:
        parallel_cfg = config.setdefault("feature_engineering", {}).setdefault("parallel", {})
        parallel_cfg["n_jobs_itpc"] = args.itpc_n_jobs
    itpc_cfg = config.setdefault("feature_engineering", {}).setdefault("itpc", {})
    if getattr(args, "itpc_min_segment_sec", None) is not None:
        itpc_cfg["min_segment_sec"] = args.itpc_min_segment_sec
    if getattr(args, "itpc_min_cycles_at_fmin", None) is not None:
        itpc_cfg["min_cycles_at_fmin"] = args.itpc_min_cycles_at_fmin


def _apply_band_envelope_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply band envelope-related config overrides."""
    band_env_cfg = config.setdefault("feature_engineering", {}).setdefault("band_envelope", {})
    if getattr(args, "band_envelope_pad_sec", None) is not None:
        band_env_cfg["pad_sec"] = args.band_envelope_pad_sec
    if getattr(args, "band_envelope_pad_cycles", None) is not None:
        band_env_cfg["pad_cycles"] = args.band_envelope_pad_cycles


def _apply_iaf_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply IAF-related config overrides."""
    bands_cfg = config.setdefault("feature_engineering", {}).setdefault("bands", {})
    if getattr(args, "iaf_enabled", None) is not None:
        bands_cfg["use_iaf"] = args.iaf_enabled
    if getattr(args, "iaf_alpha_width_hz", None) is not None:
        bands_cfg["alpha_width_hz"] = args.iaf_alpha_width_hz
    if getattr(args, "iaf_search_range", None) is not None:
        bands_cfg["iaf_search_range_hz"] = list(args.iaf_search_range)
    if getattr(args, "iaf_min_prominence", None) is not None:
        bands_cfg["iaf_min_prominence"] = args.iaf_min_prominence
    if getattr(args, "iaf_rois", None) is not None:
        bands_cfg["iaf_rois"] = args.iaf_rois
    if getattr(args, "iaf_min_cycles_at_fmin", None) is not None:
        bands_cfg["iaf_min_cycles_at_fmin"] = args.iaf_min_cycles_at_fmin
    if getattr(args, "iaf_min_baseline_sec", None) is not None:
        bands_cfg["iaf_min_baseline_sec"] = args.iaf_min_baseline_sec
    if getattr(args, "iaf_allow_full_fallback", None) is not None:
        bands_cfg["allow_full_fallback"] = args.iaf_allow_full_fallback
    if getattr(args, "iaf_allow_all_channels_fallback", None) is not None:
        bands_cfg["allow_all_channels_fallback"] = args.iaf_allow_all_channels_fallback


def _apply_quality_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply quality-related config overrides."""
    quality_cfg = config.setdefault("feature_engineering", {}).setdefault("quality", {})
    if getattr(args, "quality_psd_method", None) is not None:
        quality_cfg["psd_method"] = args.quality_psd_method
    if getattr(args, "quality_fmin", None) is not None:
        quality_cfg["fmin"] = args.quality_fmin
    if getattr(args, "quality_fmax", None) is not None:
        quality_cfg["fmax"] = args.quality_fmax
    if getattr(args, "quality_n_fft", None) is not None:
        quality_cfg["n_fft"] = args.quality_n_fft
    if getattr(args, "quality_exclude_line_noise", None) is not None:
        quality_cfg["exclude_line_noise"] = args.quality_exclude_line_noise
    if getattr(args, "quality_line_noise_freq", None) is not None:
        quality_cfg["line_noise_freqs"] = [args.quality_line_noise_freq]
    if getattr(args, "quality_line_noise_width_hz", None) is not None:
        quality_cfg["line_noise_width_hz"] = args.quality_line_noise_width_hz
    if getattr(args, "quality_line_noise_harmonics", None) is not None:
        quality_cfg["line_noise_harmonics"] = args.quality_line_noise_harmonics
    if getattr(args, "quality_snr_signal_band", None) is not None:
        quality_cfg["snr_signal_band"] = list(args.quality_snr_signal_band)
    if getattr(args, "quality_snr_noise_band", None) is not None:
        quality_cfg["snr_noise_band"] = list(args.quality_snr_noise_band)
    if getattr(args, "quality_muscle_band", None) is not None:
        quality_cfg["muscle_band"] = list(args.quality_muscle_band)


def _apply_microstates_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply microstate-related config overrides."""
    micro_cfg = config.setdefault("feature_engineering", {}).setdefault("microstates", {})
    if getattr(args, "microstates_n_states", None) is not None:
        micro_cfg["n_states"] = int(args.microstates_n_states)
    if getattr(args, "microstates_min_peak_distance_ms", None) is not None:
        micro_cfg["min_peak_distance_ms"] = float(args.microstates_min_peak_distance_ms)
    if getattr(args, "microstates_max_gfp_peaks_per_epoch", None) is not None:
        micro_cfg["max_gfp_peaks_per_epoch"] = int(args.microstates_max_gfp_peaks_per_epoch)
    if getattr(args, "microstates_min_duration_ms", None) is not None:
        micro_cfg["min_duration_ms"] = float(args.microstates_min_duration_ms)
    if getattr(args, "microstates_gfp_peak_prominence", None) is not None:
        micro_cfg["gfp_peak_prominence"] = float(args.microstates_gfp_peak_prominence)
    if getattr(args, "microstates_random_state", None) is not None:
        micro_cfg["random_state"] = int(args.microstates_random_state)
    if getattr(args, "microstates_assign_from_gfp_peaks", None) is not None:
        micro_cfg["assign_from_gfp_peaks"] = bool(args.microstates_assign_from_gfp_peaks)


def _apply_erds_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply ERDS-related config overrides."""
    erds_cfg = config.setdefault("feature_engineering", {}).setdefault("erds", {})
    if getattr(args, "erds_use_log_ratio", None) is not None:
        erds_cfg["use_log_ratio"] = args.erds_use_log_ratio
    if getattr(args, "erds_min_baseline_power", None) is not None:
        erds_cfg["min_baseline_power"] = args.erds_min_baseline_power
    if getattr(args, "erds_min_active_power", None) is not None:
        erds_cfg["min_active_power"] = args.erds_min_active_power
    if getattr(args, "erds_min_segment_sec", None) is not None:
        erds_cfg["min_segment_sec"] = args.erds_min_segment_sec
    if getattr(args, "erds_bands", None) is not None:
        erds_cfg["bands"] = list(args.erds_bands)
    if getattr(args, "erds_onset_threshold_sigma", None) is not None:
        erds_cfg["onset_threshold_sigma"] = args.erds_onset_threshold_sigma
    if getattr(args, "erds_onset_min_duration_ms", None) is not None:
        erds_cfg["onset_min_duration_ms"] = args.erds_onset_min_duration_ms
    if getattr(args, "erds_rebound_min_latency_ms", None) is not None:
        erds_cfg["rebound_min_latency_ms"] = args.erds_rebound_min_latency_ms
    if getattr(args, "erds_infer_contralateral", None) is not None:
        erds_cfg["infer_contralateral_when_missing"] = args.erds_infer_contralateral
    marker_options_set = False
    if getattr(args, "erds_condition_marker_bands", None) is not None:
        erds_cfg["laterality_marker_bands"] = list(args.erds_condition_marker_bands)
        marker_options_set = True
    if getattr(args, "erds_laterality_columns", None) is not None:
        erds_cfg["laterality_columns"] = list(args.erds_laterality_columns)
        marker_options_set = True
    if getattr(args, "erds_somatosensory_left_channels", None) is not None:
        erds_cfg["somatosensory_left_channels"] = list(args.erds_somatosensory_left_channels)
        marker_options_set = True
    if getattr(args, "erds_somatosensory_right_channels", None) is not None:
        erds_cfg["somatosensory_right_channels"] = list(args.erds_somatosensory_right_channels)
        marker_options_set = True
    if marker_options_set:
        # CLI exposes marker tuning options but no explicit enable flag.
        # If any marker-specific override is provided, enable marker extraction.
        erds_cfg["enable_laterality_markers"] = True
    if getattr(args, "erds_onset_min_threshold_percent", None) is not None:
        erds_cfg["onset_min_threshold_percent"] = float(args.erds_onset_min_threshold_percent)
    if getattr(args, "erds_rebound_threshold_sigma", None) is not None:
        erds_cfg["rebound_threshold_sigma"] = float(args.erds_rebound_threshold_sigma)
    if getattr(args, "erds_rebound_min_threshold_percent", None) is not None:
        erds_cfg["rebound_min_threshold_percent"] = float(args.erds_rebound_min_threshold_percent)


def _apply_validation_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply validation-related config overrides."""
    if getattr(args, "min_epochs", None) is not None:
        config["feature_engineering.constants.min_epochs_for_features"] = args.min_epochs


def _apply_output_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply output-related config overrides."""
    if getattr(args, "also_save_csv", None) is not None:
        output_cfg = config.setdefault("feature_engineering", {}).setdefault("output", {})
        output_cfg["also_save_csv"] = bool(args.also_save_csv)


def _apply_rest_mode_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply explicit resting-state overrides for the features pipeline."""
    if getattr(args, "task_is_rest", None) is None:
        return

    task_is_rest = bool(args.task_is_rest)
    config.setdefault("feature_engineering", {})["task_is_rest"] = task_is_rest
    if not task_is_rest:
        return

    if getattr(args, "analysis_mode", None) == "trial_ml_safe":
        raise ValueError(
            "--analysis-mode trial_ml_safe is incompatible with --task-is-rest."
        )

    if getattr(args, "source_contrast_enabled", None):
        raise ValueError(
            "--source-contrast is incompatible with --task-is-rest."
        )

    power_cfg = config.setdefault("feature_engineering", {}).setdefault("power", {})
    if getattr(args, "power_require_baseline", None) is None:
        power_cfg["require_baseline"] = False
    if getattr(args, "power_subtract_evoked", None) is None:
        power_cfg["subtract_evoked"] = False

    spectral_cfg = config.setdefault("feature_engineering", {}).setdefault("spectral", {})
    if getattr(args, "spectral_segments", None) is None:
        spectral_cfg["segments"] = []

    bands_cfg = config.setdefault("feature_engineering", {}).setdefault("bands", {})
    if getattr(args, "iaf_allow_full_fallback", None) is None:
        bands_cfg["allow_full_fallback"] = True

    aperiodic_subtract_evoked = getattr(args, "aperiodic_subtract_evoked", None)
    if aperiodic_subtract_evoked:
        raise ValueError(
            "--aperiodic-subtract-evoked is incompatible with --task-is-rest."
        )
    if aperiodic_subtract_evoked is None:
        config.setdefault("feature_engineering", {}).setdefault("aperiodic", {})[
            "subtract_evoked"
        ] = False


def _apply_spatial_transform_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply spatial transform-related config overrides."""
    if getattr(args, "spatial_transform", None) is not None:
        config["feature_engineering.spatial_transform"] = args.spatial_transform
    if getattr(args, "spatial_transform_lambda2", None) is not None:
        config.setdefault("feature_engineering", {}).setdefault("spatial_transform_params", {})["lambda2"] = args.spatial_transform_lambda2
    if getattr(args, "spatial_transform_stiffness", None) is not None:
        config.setdefault("feature_engineering", {}).setdefault("spatial_transform_params", {})["stiffness"] = args.spatial_transform_stiffness
    per_family_cfg = config.setdefault("feature_engineering", {}).setdefault("spatial_transform_per_family", {})
    for family in (
        "connectivity",
        "itpc",
        "pac",
        "power",
        "aperiodic",
        "bursts",
        "erds",
        "complexity",
        "ratios",
        "asymmetry",
        "spectral",
        "erp",
        "quality",
        "microstates",
    ):
        arg_name = f"spatial_transform_{family}"
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        if value == "inherit":
            per_family_cfg.pop(family, None)
        else:
            per_family_cfg[family] = value


def _apply_frequency_bands_override(args: argparse.Namespace, config: Any) -> None:
    """Apply custom frequency band definitions to config."""
    if getattr(args, "frequency_bands", None) is not None:
        custom_bands = parse_frequency_band_definitions(args.frequency_bands)
        config["frequency_bands"] = custom_bands
        config.setdefault("time_frequency_analysis", {})["bands"] = custom_bands


def _apply_rois_override(args: argparse.Namespace, config: Any) -> None:
    """Apply custom ROI definitions to config.
    
    Sets ROIs in both locations used by different subsystems:
    - Top-level 'rois': Used by get_roi_definitions() in feature extraction
    - 'time_frequency_analysis.rois': Used by get_rois() in TFR analysis
    """
    if getattr(args, "rois", None) is not None:
        custom_rois = parse_roi_definitions(args.rois)
        # Apply to top-level rois (used by get_roi_definitions in feature extraction)
        config["rois"] = custom_rois
        # Apply to TFR-specific rois (used by get_rois in TFR extraction)
        config.setdefault("time_frequency_analysis", {})["rois"] = custom_rois


def _apply_feature_config_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply all feature-specific config overrides from CLI arguments."""
    _apply_frequency_bands_override(args, config)
    _apply_rois_override(args, config)
    _apply_rest_mode_overrides(args, config)
    _apply_connectivity_overrides(args, config)
    _apply_directedconnectivity_overrides(args, config)
    _apply_sourcelocalization_overrides(args, config)
    _apply_pac_overrides(args, config)
    _apply_aperiodic_overrides(args, config)
    _apply_complexity_overrides(args, config)
    _apply_erp_overrides(args, config)
    _apply_burst_overrides(args, config)
    _apply_power_overrides(args, config)
    _apply_spectral_overrides(args, config)
    _apply_ratios_overrides(args, config)
    _apply_asymmetry_overrides(args, config)
    _apply_tfr_overrides(args, config)
    _apply_itpc_overrides(args, config)
    _apply_band_envelope_overrides(args, config)
    _apply_iaf_overrides(args, config)
    _apply_quality_overrides(args, config)
    _apply_microstates_overrides(args, config)
    _apply_erds_overrides(args, config)
    _apply_execution_overrides(args, config)
    _apply_validation_overrides(args, config)
    _apply_output_overrides(args, config)
    _apply_spatial_transform_overrides(args, config)


def _apply_execution_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply execution/runtime overrides for the features pipeline."""
    if getattr(args, "analysis_mode", None) is not None:
        config["feature_engineering.analysis_mode"] = args.analysis_mode
    if getattr(args, "compute_change_scores", None) is not None:
        config["feature_engineering.compute_change_scores"] = args.compute_change_scores
    if getattr(args, "change_scores_transform", None) is not None:
        transform = str(args.change_scores_transform).strip().lower()
        if transform == "ratio":
            transform = "percent"
        config["feature_engineering.change_scores.transform"] = transform
    if getattr(args, "change_scores_window_pairs", None):
        config["feature_engineering.change_scores.window_pairs"] = _parse_pair_tokens(
            args.change_scores_window_pairs, label="change_scores"
        )
    if getattr(args, "save_tfr_with_sidecar", None) is not None:
        config["feature_engineering.save_tfr_with_sidecar"] = args.save_tfr_with_sidecar

    parallel_cfg = config.setdefault("feature_engineering", {}).setdefault("parallel", {})
    if getattr(args, "n_jobs_bands", None) is not None:
        parallel_cfg["n_jobs_bands"] = args.n_jobs_bands
    if getattr(args, "n_jobs_connectivity", None) is not None:
        parallel_cfg["n_jobs_connectivity"] = args.n_jobs_connectivity
    if getattr(args, "n_jobs_aperiodic", None) is not None:
        parallel_cfg["n_jobs_aperiodic"] = args.n_jobs_aperiodic
    if getattr(args, "n_jobs_complexity", None) is not None:
        parallel_cfg["n_jobs_complexity"] = args.n_jobs_complexity
