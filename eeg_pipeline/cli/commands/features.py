"""Features extraction CLI command."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
    create_progress_reporter,
    resolve_task,
)
from eeg_pipeline.pipelines.constants import (
    FEATURE_CATEGORIES,
    FREQUENCY_BANDS,
)
from eeg_pipeline.domain.features.constants import SPATIAL_MODES
from eeg_pipeline.cli.commands.base import FEATURE_VISUALIZE_CATEGORIES

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
    if getattr(args, "conn_granularity", None) is not None:
        conn_cfg["granularity"] = args.conn_granularity
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
    if getattr(args, "source_fmri_max_clusters", None) is not None:
        fmri_cfg["max_clusters"] = args.source_fmri_max_clusters
    if getattr(args, "source_fmri_max_voxels_per_cluster", None) is not None:
        fmri_cfg["max_voxels_per_cluster"] = args.source_fmri_max_voxels_per_cluster
    if getattr(args, "source_fmri_max_total_voxels", None) is not None:
        fmri_cfg["max_total_voxels"] = args.source_fmri_max_total_voxels
    if getattr(args, "source_fmri_random_seed", None) is not None:
        fmri_cfg["random_seed"] = args.source_fmri_random_seed
    
    time_windows_cfg = fmri_cfg.setdefault("time_windows", {})
    window_a_cfg = time_windows_cfg.setdefault("window_a", {})
    window_b_cfg = time_windows_cfg.setdefault("window_b", {})
    if getattr(args, "source_fmri_window_a_name", None) is not None:
        window_a_cfg["name"] = args.source_fmri_window_a_name
    if getattr(args, "source_fmri_window_a_tmin", None) is not None:
        window_a_cfg["tmin"] = args.source_fmri_window_a_tmin
    if getattr(args, "source_fmri_window_a_tmax", None) is not None:
        window_a_cfg["tmax"] = args.source_fmri_window_a_tmax
    if getattr(args, "source_fmri_window_b_name", None) is not None:
        window_b_cfg["name"] = args.source_fmri_window_b_name
    if getattr(args, "source_fmri_window_b_tmin", None) is not None:
        window_b_cfg["tmin"] = args.source_fmri_window_b_tmin
    if getattr(args, "source_fmri_window_b_tmax", None) is not None:
        window_b_cfg["tmax"] = args.source_fmri_window_b_tmax

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
    if getattr(args, "source_fmri_cluster_correction", None) is not None:
        contrast_cfg["cluster_correction"] = args.source_fmri_cluster_correction
    if getattr(args, "source_fmri_cluster_p_threshold", None) is not None:
        contrast_cfg["cluster_p_threshold"] = args.source_fmri_cluster_p_threshold
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


def _apply_complexity_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply complexity-related config overrides."""
    if getattr(args, "pe_order", None) is not None:
        config["feature_engineering.complexity.pe_order"] = args.pe_order
    if getattr(args, "pe_delay", None) is not None:
        config["feature_engineering.complexity.pe_delay"] = args.pe_delay
    
    complexity_cfg = config.setdefault("feature_engineering", {}).setdefault("complexity", {})
    if getattr(args, "complexity_signal_basis", None) is not None:
        complexity_cfg["signal_basis"] = args.complexity_signal_basis
    if getattr(args, "complexity_min_segment_sec", None) is not None:
        complexity_cfg["min_segment_sec"] = args.complexity_min_segment_sec
    if getattr(args, "complexity_min_samples", None) is not None:
        complexity_cfg["min_samples"] = args.complexity_min_samples
    if getattr(args, "complexity_zscore", None) is not None:
        complexity_cfg["zscore"] = args.complexity_zscore


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


def _apply_burst_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply burst-related config overrides."""
    if getattr(args, "burst_threshold", None) is not None:
        config["feature_engineering.bursts.threshold_z"] = args.burst_threshold
    if getattr(args, "burst_threshold_method", None) is not None:
        config["feature_engineering.bursts.threshold_method"] = args.burst_threshold_method
    if getattr(args, "burst_threshold_percentile", None) is not None:
        config["feature_engineering.bursts.threshold_percentile"] = args.burst_threshold_percentile
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


def _apply_spectral_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply spectral-related config overrides."""
    if getattr(args, "spectral_edge_percentile", None) is not None:
        config["feature_engineering.spectral.edge_percentile"] = args.spectral_edge_percentile
    if getattr(args, "ratio_pairs", None) is not None:
        config["feature_engineering.spectral.ratio_pairs"] = _parse_pair_tokens(args.ratio_pairs, label="ratio")
    if getattr(args, "ratio_source", None) is not None:
        config["feature_engineering.spectral.ratio_source"] = args.ratio_source
    
    spectral_cfg = config.setdefault("feature_engineering", {}).setdefault("spectral", {})
    if getattr(args, "spectral_include_log_ratios", None) is not None:
        spectral_cfg["include_log_ratios"] = args.spectral_include_log_ratios
    if getattr(args, "spectral_psd_method", None) is not None:
        spectral_cfg["psd_method"] = args.spectral_psd_method
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
    if getattr(args, "tfr_decim", None) is not None:
        tfr_config["decim"] = args.tfr_decim
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
    if getattr(args, "quality_snr_signal_band", None) is not None:
        quality_cfg["snr_signal_band"] = list(args.quality_snr_signal_band)
    if getattr(args, "quality_snr_noise_band", None) is not None:
        quality_cfg["snr_noise_band"] = list(args.quality_snr_noise_band)
    if getattr(args, "quality_muscle_band", None) is not None:
        quality_cfg["muscle_band"] = list(args.quality_muscle_band)


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
        erds_cfg["bands"] = args.erds_bands


def _apply_validation_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply validation-related config overrides."""
    if getattr(args, "min_epochs", None) is not None:
        config["feature_engineering.constants.min_epochs_for_features"] = args.min_epochs


def _apply_output_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply output-related config overrides."""
    if getattr(args, "save_subject_level_features", None) is not None:
        config["feature_engineering.output.save_subject_level_features"] = args.save_subject_level_features
    if getattr(args, "also_save_csv", None) is not None:
        output_cfg = config.setdefault("feature_engineering", {}).setdefault("output", {})
        output_cfg["also_save_csv"] = bool(args.also_save_csv)


def _apply_spatial_transform_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply spatial transform-related config overrides."""
    if getattr(args, "spatial_transform", None) is not None:
        config["feature_engineering.spatial_transform"] = args.spatial_transform
    if getattr(args, "spatial_transform_lambda2", None) is not None:
        config.setdefault("feature_engineering", {}).setdefault("spatial_transform_params", {})["lambda2"] = args.spatial_transform_lambda2
    if getattr(args, "spatial_transform_stiffness", None) is not None:
        config.setdefault("feature_engineering", {}).setdefault("spatial_transform_params", {})["stiffness"] = args.spatial_transform_stiffness


def _parse_frequency_band_definitions(band_defs: List[str]) -> dict:
    """Parse frequency band definitions from CLI format 'name:low:high'."""
    bands = {}
    for band_def in band_defs:
        parts = band_def.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid frequency band definition '{band_def}'; expected 'name:low:high'")
        name = parts[0].strip().lower()
        try:
            low = float(parts[1].strip())
            high = float(parts[2].strip())
        except ValueError:
            raise ValueError(f"Invalid frequency values in '{band_def}'; expected numeric low:high")
        if low >= high:
            raise ValueError(f"Invalid frequency range in '{band_def}'; low must be < high")
        bands[name] = [low, high]
    return bands


def _parse_roi_definitions(roi_defs: List[str]) -> dict:
    """Parse ROI definitions from CLI format 'name:ch1,ch2,...'."""
    rois = {}
    for roi_def in roi_defs:
        if ":" not in roi_def:
            raise ValueError(f"Invalid ROI definition '{roi_def}'; expected 'name:ch1,ch2,...'")
        name, channels_str = roi_def.split(":", 1)
        name = name.strip()
        channels = [ch.strip() for ch in channels_str.split(",") if ch.strip()]
        if not channels:
            raise ValueError(f"Invalid ROI definition '{roi_def}'; no channels specified")
        rois[name] = [f"^({'|'.join(channels)})$"]
    return rois


def _apply_frequency_bands_override(args: argparse.Namespace, config: Any) -> None:
    """Apply custom frequency band definitions to config."""
    if getattr(args, "frequency_bands", None) is not None:
        custom_bands = _parse_frequency_band_definitions(args.frequency_bands)
        config["frequency_bands"] = custom_bands
        config.setdefault("time_frequency_analysis", {})["bands"] = custom_bands


def _apply_rois_override(args: argparse.Namespace, config: Any) -> None:
    """Apply custom ROI definitions to config.
    
    Sets ROIs in both locations used by different subsystems:
    - Top-level 'rois': Used by get_roi_definitions() in feature extraction
    - 'time_frequency_analysis.rois': Used by get_rois() in TFR analysis
    """
    if getattr(args, "rois", None) is not None:
        custom_rois = _parse_roi_definitions(args.rois)
        # Apply to top-level rois (used by get_roi_definitions in feature extraction)
        config["rois"] = custom_rois
        # Apply to TFR-specific rois (used by get_rois in TFR extraction)
        config.setdefault("time_frequency_analysis", {})["rois"] = custom_rois


def _apply_feature_config_overrides(args: argparse.Namespace, config: Any) -> None:
    """Apply all feature-specific config overrides from CLI arguments."""
    _apply_frequency_bands_override(args, config)
    _apply_rois_override(args, config)
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
    _apply_asymmetry_overrides(args, config)
    _apply_tfr_overrides(args, config)
    _apply_itpc_overrides(args, config)
    _apply_band_envelope_overrides(args, config)
    _apply_iaf_overrides(args, config)
    _apply_quality_overrides(args, config)
    _apply_erds_overrides(args, config)
    _apply_validation_overrides(args, config)
    _apply_output_overrides(args, config)
    _apply_spatial_transform_overrides(args, config)


def setup_features(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the features command parser."""
    parser = subparsers.add_parser(
        "features",
        help="Features analysis: extract or visualize",
        description="Features pipeline: extract features or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)

    # Core arguments
    parser.add_argument("--categories", nargs="+", choices=FEATURE_CATEGORY_CHOICES, default=None, metavar="CATEGORY", help="Feature categories to process (some are compute-only or visualize-only)")
    parser.add_argument("--bands", nargs="+", default=None, help="Frequency bands to compute (default: all)")
    parser.add_argument("--frequency-bands", nargs="+", default=None, metavar="BAND_DEF", help="Custom frequency band definitions in format 'name:low:high' (e.g., delta:1.0:3.9 theta:4.0:7.9)")
    parser.add_argument("--rois", nargs="+", default=None, metavar="ROI_DEF", help="Custom ROI definitions in format 'name:ch1,ch2,...' (e.g., 'Frontal:Fp1,Fp2,F3,F4')")
    parser.add_argument("--spatial", nargs="+", choices=SPATIAL_MODES, default=None, metavar="MODE", help="Spatial aggregation modes: roi, channels, global (default: roi, global)")
    parser.add_argument("--spatial-transform", choices=["none", "csd", "laplacian"], default=None, help="Spatial transform to reduce volume conduction: none, csd, laplacian")
    parser.add_argument("--spatial-transform-lambda2", type=float, default=None, help="Lambda2 regularization for CSD/Laplacian (default: 1e-5)")
    parser.add_argument("--spatial-transform-stiffness", type=float, default=None, help="Stiffness for CSD/Laplacian (default: 4.0)")
    parser.add_argument("--tmin", type=float, default=None, help="Start time in seconds for feature extraction window")
    parser.add_argument("--tmax", type=float, default=None, help="End time in seconds for feature extraction window")
    parser.add_argument("--time-range", nargs=3, action="append", metavar=("NAME", "TMIN", "TMAX"), help="Define a named time range (e.g. baseline 0 1). Can be specified multiple times.")
    parser.add_argument("--aggregation-method", choices=["mean", "median"], default="mean", help="Aggregation method for spatial modes (default: mean)")

    # Connectivity
    parser.add_argument("--connectivity-measures", nargs="+", choices=["wpli", "aec", "plv", "pli"], default=None, help="Connectivity measures to compute")
    parser.add_argument("--conn-output-level", choices=["full", "global_only"], default=None, help="Connectivity output level")
    parser.add_argument("--conn-graph-metrics", action="store_true", default=None, help="Enable graph metrics for connectivity")
    parser.add_argument("--no-conn-graph-metrics", action="store_false", dest="conn_graph_metrics", help="Disable graph metrics for connectivity")
    parser.add_argument("--conn-aec-mode", choices=["orth", "sym", "none"], default=None, help="AEC orthogonalization mode")
    parser.add_argument("--conn-graph-prop", type=float, default=None, help="Proportion of top edges to keep for graph metrics")
    parser.add_argument("--aec-output", nargs="+", choices=["r", "z"], default=None, help="AEC output format: r (raw), z (Fisher-z transform), or both")
    parser.add_argument("--conn-force-within-epoch-for-ml", action="store_true", default=None, help="Force within_epoch phase estimator when train_mask detected (CV-safe)")
    parser.add_argument("--no-conn-force-within-epoch-for-ml", action="store_false", dest="conn_force_within_epoch_for_ml", help="Allow across_epochs phase estimator even in CV/machine learning mode")
    parser.add_argument("--conn-granularity", choices=["trial", "condition", "subject"], default=None, help="Connectivity granularity")
    parser.add_argument("--conn-min-epochs-per-group", type=int, default=None, help="Min epochs per group for connectivity")
    parser.add_argument("--conn-min-cycles-per-band", type=float, default=None, help="Min cycles per band for connectivity")
    parser.add_argument("--conn-warn-no-spatial-transform", action="store_true", default=None, help="Warn if no spatial transform for phase connectivity")
    parser.add_argument("--no-conn-warn-no-spatial-transform", action="store_false", dest="conn_warn_no_spatial_transform")
    parser.add_argument("--conn-phase-estimator", choices=["within_epoch", "across_epochs"], default=None, help="Phase estimator mode")
    parser.add_argument("--conn-min-segment-sec", type=float, default=None, help="Min segment duration for connectivity")

    # Directed connectivity
    parser.add_argument("--directed-connectivity-measures", nargs="+", choices=["psi", "dtf", "pdc"], default=None, help="Directed connectivity measures: psi (Phase Slope Index), dtf (Directed Transfer Function), pdc (Partial Directed Coherence)")
    parser.add_argument("--directed-conn-output-level", choices=["full", "global_only"], default=None, help="Directed connectivity output level: full (all channel pairs) or global_only (mean only)")
    parser.add_argument("--directed-conn-mvar-order", type=int, default=None, help="MVAR model order for DTF/PDC computation (default: 10)")
    parser.add_argument("--directed-conn-n-freqs", type=int, default=None, help="Number of frequency bins for directed connectivity (default: 16)")
    parser.add_argument("--directed-conn-min-segment-samples", type=int, default=None, help="Minimum segment samples for directed connectivity (default: 100)")

    # Source localization
    parser.add_argument("--source-method", choices=["lcmv", "eloreta"], default=None, help="Source localization method: lcmv (beamformer) or eloreta (inverse)")
    parser.add_argument("--source-spacing", choices=["oct5", "oct6", "ico4", "ico5"], default=None, help="Source space spacing (default: oct6)")
    parser.add_argument("--source-reg", type=float, default=None, help="LCMV regularization parameter (default: 0.05)")
    parser.add_argument("--source-snr", type=float, default=None, help="eLORETA assumed SNR for regularization (default: 3.0)")
    parser.add_argument("--source-loose", type=float, default=None, help="eLORETA loose orientation constraint 0-1 (default: 0.2)")
    parser.add_argument("--source-depth", type=float, default=None, help="eLORETA depth weighting 0-1 (default: 0.8)")
    parser.add_argument("--source-parc", choices=["aparc", "aparc.a2009s", "HCPMMP1"], default=None, help="Brain parcellation for ROI extraction (default: aparc)")
    parser.add_argument("--source-connectivity-method", choices=["aec", "wpli", "plv"], default=None, help="Connectivity method for source-space analysis (default: aec)")
    parser.add_argument("--source-subject", default=None, help="FreeSurfer subject name to use for source localization (e.g., sub-0001). If unset, defaults to sub-{subject}.")
    parser.add_argument("--source-subjects-dir", default=None, help="FreeSurfer SUBJECTS_DIR path for subject-specific source localization.")
    parser.add_argument("--source-trans", default=None, help="EEG↔MRI coregistration transform .fif (required for subject-specific/fMRI-constrained source localization).")
    parser.add_argument("--source-bem", default=None, help="BEM solution .fif (e.g., *-bem-sol.fif) (required for subject-specific/fMRI-constrained source localization).")
    parser.add_argument("--source-mindist-mm", type=float, default=None, help="Minimum distance from sources to inner skull (mm) (default: 5.0).")
    parser.add_argument("--source-create-trans", action="store_true", default=None, dest="source_create_trans", help="Auto-create coregistration transform via Docker (requires Docker; FS license from global config).")
    parser.add_argument("--source-create-bem-model", action="store_true", default=None, dest="source_create_bem_model", help="Auto-create BEM model via Docker (requires Docker; FS license from global config).")
    parser.add_argument("--source-create-bem-solution", action="store_true", default=None, dest="source_create_bem_solution", help="Auto-create BEM solution via Docker (requires Docker; FS license from global config).")
    parser.add_argument("--source-allow-identity-trans", action="store_true", default=None, dest="source_allow_identity_trans", help="Allow creating identity transform (DEBUG ONLY - scientifically invalid for production; use only when proper coregistration is unavailable).")

    # fMRI-informed source localization
    parser.add_argument("--source-fmri", action="store_true", default=None, dest="source_fmri_enabled", help="Enable fMRI-informed source localization (requires --source-subjects-dir/--source-trans/--source-bem and a stats map).")
    parser.add_argument("--no-source-fmri", action="store_false", dest="source_fmri_enabled", help="Disable fMRI-informed source localization (overrides config).")
    parser.add_argument("--source-fmri-stats-map", default=None, help="Path to an fMRI statistical map NIfTI in the same MRI space as the FreeSurfer subject (typically resampled to orig.mgz space).")
    parser.add_argument("--source-fmri-provenance", choices=["independent", "same_dataset"], default=None, help="Provenance of the fMRI constraint relative to EEG labels: independent (recommended) or same_dataset (circularity risk).")
    parser.add_argument("--source-fmri-require-provenance", action="store_true", default=None, dest="source_fmri_require_provenance", help="Require explicit fMRI provenance when using fMRI constraints.")
    parser.add_argument("--no-source-fmri-require-provenance", action="store_false", dest="source_fmri_require_provenance", help="Allow unknown fMRI provenance (not recommended).")
    parser.add_argument("--source-fmri-threshold", type=float, default=None, help="Threshold applied to fMRI stats map (default: 3.1).")
    parser.add_argument("--source-fmri-tail", choices=["pos", "abs"], default=None, help="Threshold tail: pos (positive only) or abs (absolute value) (default: pos).")
    parser.add_argument("--source-fmri-cluster-min-voxels", type=int, default=None, help="Minimum cluster size in voxels after thresholding (default: 50).")
    parser.add_argument("--source-fmri-max-clusters", type=int, default=None, help="Maximum number of clusters kept from fMRI map (default: 20).")
    parser.add_argument("--source-fmri-max-voxels-per-cluster", type=int, default=None, help="Maximum voxels sampled per cluster (default: 2000; set 0 for no limit).")
    parser.add_argument("--source-fmri-max-total-voxels", type=int, default=None, help="Maximum total voxels across all clusters (default: 20000; set 0 for no limit).")
    parser.add_argument("--source-fmri-random-seed", type=int, default=None, help="Random seed for voxel subsampling (default: 0 -> nondeterministic).")
    parser.add_argument("--source-fmri-window-a-name", default=None, help="Name for window A (e.g., 'plateau').")
    parser.add_argument("--source-fmri-window-a-tmin", type=float, default=None, help="Start time for window A in seconds.")
    parser.add_argument("--source-fmri-window-a-tmax", type=float, default=None, help="End time for window A in seconds.")
    parser.add_argument("--source-fmri-window-b-name", default=None, help="Name for window B (e.g., 'baseline').")
    parser.add_argument("--source-fmri-window-b-tmin", type=float, default=None, help="Start time for window B in seconds.")
    parser.add_argument("--source-fmri-window-b-tmax", type=float, default=None, help="End time for window B in seconds.")
    parser.add_argument("--source-fmri-contrast-enabled", action="store_true", default=None, dest="source_fmri_contrast_enabled", help="Enable building fMRI contrast from BOLD data (vs. loading pre-computed stats map).")
    parser.add_argument("--source-fmri-cond-a-column", default=None, help="Column for condition A in events.tsv (e.g., 'trial_type', 'pain_binary').")
    parser.add_argument("--source-fmri-cond-a-value", default=None, help="Value for condition A (e.g., 'temp49p3', '1').")
    parser.add_argument("--source-fmri-cond-b-column", default=None, help="Column for condition B in events.tsv.")
    parser.add_argument("--source-fmri-cond-b-value", default=None, help="Value for condition B.")
    parser.add_argument("--source-fmri-contrast-type", choices=["t-test", "paired-t-test", "f-test", "custom"], default=None, help="Type of statistical contrast to compute.")
    parser.add_argument("--source-fmri-contrast-formula", default=None, help="Custom contrast formula (e.g., 'pain_high - pain_low').")
    parser.add_argument("--source-fmri-contrast-name", default=None, help="Name for the contrast output (default: 'pain_vs_baseline').")
    parser.add_argument("--source-fmri-runs", default=None, help="Comma-separated run numbers to include (e.g., '1,2,3').")
    parser.add_argument("--source-fmri-hrf-model", choices=["spm", "flobs", "fir"], default=None, help="HRF model for GLM (default: spm).")
    parser.add_argument("--source-fmri-drift-model", choices=["none", "cosine", "polynomial"], default=None, help="Drift model for GLM (default: cosine).")
    parser.add_argument("--source-fmri-high-pass", type=float, default=None, help="High-pass filter cutoff in Hz (default: 0.008).")
    parser.add_argument("--source-fmri-low-pass", type=float, default=None, help="Low-pass filter cutoff in Hz (default: 0.1).")
    parser.add_argument("--source-fmri-cluster-correction", action="store_true", default=None, dest="source_fmri_cluster_correction", help="Enable cluster-level FWE correction.")
    parser.add_argument("--source-fmri-cluster-p-threshold", type=float, default=None, help="Cluster-forming p-threshold (default: 0.001).")
    parser.add_argument("--source-fmri-output-type", choices=["z-score", "t-stat", "cope", "beta"], default=None, help="Output statistical map type (default: z-score).")
    parser.add_argument("--source-fmri-resample-to-fs", action="store_true", default=None, dest="source_fmri_resample_to_fs", help="Auto-resample stats map to FreeSurfer subject space.")
    parser.add_argument("--no-source-fmri-resample-to-fs", action="store_false", dest="source_fmri_resample_to_fs", help="Do not auto-resample stats map to FreeSurfer subject space.")
    parser.add_argument("--source-fmri-input-source", choices=["fmriprep", "bids_raw"], default=None, help="Input source for contrast builder: 'fmriprep' (default) or 'bids_raw' (uses files in func/).")
    parser.add_argument("--source-fmri-require-fmriprep", action="store_true", default=None, dest="source_fmri_require_fmriprep", help="Require fMRIPrep outputs for contrast building (default: true).")
    parser.add_argument("--no-source-fmri-require-fmriprep", action="store_false", dest="source_fmri_require_fmriprep", help="Allow using raw BIDS files if fMRIPrep outputs are missing.")

    # PAC
    parser.add_argument("--pac-phase-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Phase frequency range for PAC/CFC (Hz)")
    parser.add_argument("--pac-amp-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Amplitude frequency range for PAC/CFC (Hz)")
    parser.add_argument("--pac-method", choices=["mvl", "kl", "tort", "ozkurt"], default=None, help="PAC estimation method")
    parser.add_argument("--pac-min-epochs", type=int, default=None, help="Minimum epochs for PAC computation")
    parser.add_argument("--pac-pairs", nargs="+", default=None, metavar="PAIR", help="PAC band pairs, e.g. theta:gamma alpha:gamma (uses time_frequency_analysis.bands)")
    parser.add_argument("--pac-source", choices=["precomputed", "tfr"], default=None, help="PAC source: precomputed (Hilbert) or tfr (wavelet)")
    parser.add_argument("--pac-normalize", action="store_true", default=None, help="Normalize PAC values")
    parser.add_argument("--no-pac-normalize", action="store_false", dest="pac_normalize")
    parser.add_argument("--pac-n-surrogates", type=int, default=None, help="Number of surrogates for PAC (0=none)")
    parser.add_argument("--pac-allow-harmonic-overlap", action="store_true", default=None, help="Allow harmonic overlap in PAC")
    parser.add_argument("--no-pac-allow-harmonic-overlap", action="store_false", dest="pac_allow_harmonic_overlap")
    parser.add_argument("--pac-max-harmonic", type=int, default=None, help="Max harmonic to check for overlap")
    parser.add_argument("--pac-harmonic-tolerance-hz", type=float, default=None, help="Harmonic tolerance in Hz")
    parser.add_argument("--pac-compute-waveform-qc", action="store_true", default=None, help="Compute waveform QC for PAC")
    parser.add_argument("--no-pac-compute-waveform-qc", action="store_false", dest="pac_compute_waveform_qc")
    parser.add_argument("--pac-waveform-offset-ms", type=float, default=None, help="Waveform offset in ms for PAC QC")
    parser.add_argument("--pac-random-seed", type=int, default=None, help="Random seed for PAC surrogate testing")

    # Aperiodic
    parser.add_argument("--aperiodic-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Frequency range for aperiodic fit (Hz)")
    parser.add_argument("--aperiodic-peak-z", type=float, default=None, help="Peak rejection Z-threshold for aperiodic fit")
    parser.add_argument("--aperiodic-min-r2", type=float, default=None, help="Minimum R2 for aperiodic fit")
    parser.add_argument("--aperiodic-min-points", type=int, default=None, help="Minimum fit points for aperiodic")
    parser.add_argument("--aperiodic-subtract-evoked", action="store_true", default=None, help="Subtract evoked response for induced spectra (recommended for pain paradigms)")
    parser.add_argument("--aperiodic-min-segment-sec", type=float, default=None, help="Minimum segment duration (seconds) for stable aperiodic fits (default: 2.0)")
    parser.add_argument("--aperiodic-model", choices=["fixed", "knee"], default=None, help="Aperiodic model type")
    parser.add_argument("--aperiodic-psd-method", choices=["multitaper", "welch"], default=None, help="PSD method for aperiodic")
    parser.add_argument("--aperiodic-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from aperiodic fit")
    parser.add_argument("--no-aperiodic-exclude-line-noise", action="store_false", dest="aperiodic_exclude_line_noise")
    parser.add_argument("--aperiodic-line-noise-freq", type=float, default=None, help="Line noise frequency for aperiodic")
    parser.add_argument("--aperiodic-line-noise-width-hz", type=float, default=None, help="Line noise frequency band width to exclude from aperiodic fit")
    parser.add_argument("--aperiodic-line-noise-harmonics", type=int, default=None, help="Number of line noise harmonics to exclude from aperiodic fit")

    # ERP
    parser.add_argument("--erp-baseline", action="store_true", default=None, help="Enable baseline correction for ERP")
    parser.add_argument("--no-erp-baseline", action="store_false", dest="erp_baseline", help="Disable baseline correction for ERP")
    parser.add_argument("--erp-allow-no-baseline", action="store_true", default=None, help="Allow ERP extraction when baseline window is missing")
    parser.add_argument("--no-erp-allow-no-baseline", action="store_false", dest="erp_allow_no_baseline", help="Require baseline window when ERP baseline correction is enabled")
    parser.add_argument("--erp-components", nargs="+", default=None, metavar="COMP", help="ERP component windows, e.g. n1=0.10-0.20 n2=0.20-0.35 p2=0.35-0.50")
    parser.add_argument("--erp-lowpass-hz", type=float, default=None, help="Low-pass filter frequency (Hz) for ERP peak detection (default: 30.0)")

    # Burst
    parser.add_argument("--burst-threshold", type=float, default=None, help="Z-score threshold for burst detection (used with zscore/mad methods)")
    parser.add_argument("--burst-threshold-method", choices=["percentile", "zscore", "mad"], default=None, help="Burst threshold method: percentile, zscore, or mad (default: percentile)")
    parser.add_argument("--burst-threshold-percentile", type=float, default=None, help="Percentile threshold for burst detection (0-100, default: 95.0)")
    parser.add_argument("--burst-bands", nargs="+", default=None, metavar="BAND", help="Burst bands to compute, e.g. beta gamma")
    parser.add_argument("--burst-min-duration", type=int, default=None, help="Minimum burst duration (ms)")
    parser.add_argument("--burst-min-cycles", type=float, default=None, help="Minimum oscillatory cycles for burst detection")

    # Power
    parser.add_argument("--power-baseline-mode", choices=["logratio", "mean", "ratio", "zscore", "zlogratio"], default=None, help="Baseline normalization mode for power")
    parser.add_argument("--power-require-baseline", action="store_true", default=None, help="Require baseline for power normalization")
    parser.add_argument("--no-power-require-baseline", action="store_false", dest="power_require_baseline", help="Allow raw log power without baseline")

    # Spectral
    parser.add_argument("--spectral-edge-percentile", type=float, default=None, help="Percentile for spectral edge frequency (0-1)")
    parser.add_argument("--ratio-pairs", nargs="+", default=None, metavar="PAIR", help="Band power ratio pairs, e.g. theta:beta theta:alpha alpha:beta")
    parser.add_argument("--ratio-source", choices=["raw", "powcorr"], default=None, help="Power source for band ratios: raw (absolute) or powcorr (aperiodic-adjusted)")
    parser.add_argument("--spectral-include-log-ratios", action="store_true", default=None, help="Include log ratios in spectral features")
    parser.add_argument("--no-spectral-include-log-ratios", action="store_false", dest="spectral_include_log_ratios")
    parser.add_argument("--spectral-psd-method", choices=["multitaper", "welch"], default=None, help="PSD method for spectral features")
    parser.add_argument("--spectral-fmin", type=float, default=None, help="Min frequency for spectral features")
    parser.add_argument("--spectral-fmax", type=float, default=None, help="Max frequency for spectral features")
    parser.add_argument("--spectral-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from spectral computation")
    parser.add_argument("--no-spectral-exclude-line-noise", action="store_false", dest="spectral_exclude_line_noise")
    parser.add_argument("--spectral-line-noise-freq", type=float, default=None, help="Line noise frequency (50 or 60 Hz)")
    parser.add_argument("--spectral-line-noise-width-hz", type=float, default=None, help="Line noise frequency band width to exclude")
    parser.add_argument("--spectral-line-noise-harmonics", type=int, default=None, help="Number of line noise harmonics to exclude")
    parser.add_argument("--spectral-segments", nargs="+", default=None, help="Segments for spectral features (e.g., baseline active)")
    parser.add_argument("--spectral-min-segment-sec", type=float, default=None, help="Minimum segment duration for spectral")
    parser.add_argument("--spectral-min-cycles-at-fmin", type=float, default=None, help="Minimum cycles at lowest frequency")

    # Asymmetry
    parser.add_argument("--asymmetry-channel-pairs", nargs="+", default=None, metavar="PAIR", help="Channel pairs for asymmetry, e.g. F3:F4 C3:C4")

    # TFR
    parser.add_argument("--tfr-freq-min", type=float, default=None, help="Minimum frequency for TFR (Hz)")
    parser.add_argument("--tfr-freq-max", type=float, default=None, help="Maximum frequency for TFR (Hz)")
    parser.add_argument("--tfr-n-freqs", type=int, default=None, help="Number of frequencies for TFR")
    parser.add_argument("--tfr-min-cycles", type=float, default=None, help="Minimum number of cycles for Morlet wavelets")
    parser.add_argument("--tfr-n-cycles-factor", type=float, default=None, help="Cycles factor (freq/factor) for Morlet wavelets")
    parser.add_argument("--tfr-decim", type=int, default=None, help="Decimation factor for TFR")
    parser.add_argument("--tfr-workers", type=int, default=None, help="Number of parallel workers for TFR computation")
    parser.add_argument("--tfr-max-cycles", type=float, default=None, help="Maximum cycles for Morlet wavelets")
    parser.add_argument("--tfr-decim-power", type=int, default=None, help="Decimation factor for power TFR")
    parser.add_argument("--tfr-decim-phase", type=int, default=None, help="Decimation factor for phase TFR")

    # ITPC
    parser.add_argument("--itpc-method", choices=["global", "fold_global", "loo", "condition"], default=None, help="ITPC computation method: global (all trials), fold_global (training only, CV-safe), loo (leave-one-out), condition (per condition group, avoids pseudo-replication)")
    parser.add_argument("--itpc-allow-unsafe-loo", action="store_true", default=None, help="Allow unsafe LOO ITPC computation")
    parser.add_argument("--no-itpc-allow-unsafe-loo", action="store_false", dest="itpc_allow_unsafe_loo")
    parser.add_argument("--itpc-baseline-correction", choices=["none", "subtract"], default=None, help="ITPC baseline correction mode")
    parser.add_argument("--itpc-condition-column", default=None, help="Column for condition-based ITPC (avoids pseudo-replication)")
    parser.add_argument("--itpc-condition-values", nargs="+", default=None, help="Specific condition values to compute ITPC for (space-separated)")
    parser.add_argument("--itpc-min-trials-per-condition", type=int, default=None, help="Minimum trials per condition for reliable ITPC (default: 10)")

    # Band envelope
    parser.add_argument("--band-envelope-pad-sec", type=float, default=None, help="Padding in seconds for band envelope")
    parser.add_argument("--band-envelope-pad-cycles", type=float, default=None, help="Padding in cycles for band envelope")

    # IAF
    parser.add_argument("--iaf-enabled", action="store_true", default=None, help="Enable individualized alpha frequency")
    parser.add_argument("--no-iaf-enabled", action="store_false", dest="iaf_enabled")
    parser.add_argument("--iaf-alpha-width-hz", type=float, default=None, help="IAF alpha band width in Hz")
    parser.add_argument("--iaf-search-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="IAF search range in Hz")
    parser.add_argument("--iaf-min-prominence", type=float, default=None, help="IAF minimum peak prominence")
    parser.add_argument("--iaf-rois", nargs="+", default=None, help="ROIs for IAF detection")

    # Complexity
    parser.add_argument("--pe-order", type=int, default=None, help="Permutation entropy order (3-7, default: from config)")
    parser.add_argument("--pe-delay", type=int, default=None, help="Permutation entropy delay")
    parser.add_argument("--complexity-signal-basis", choices=["filtered", "envelope"], default=None, help="Complexity signal basis")
    parser.add_argument("--complexity-min-segment-sec", type=float, default=None, help="Minimum segment duration for complexity (sec)")
    parser.add_argument("--complexity-min-samples", type=int, default=None, help="Minimum samples for complexity")
    parser.add_argument("--complexity-zscore", action="store_true", default=None, help="Apply z-score normalization for complexity")
    parser.add_argument("--no-complexity-zscore", action="store_false", dest="complexity_zscore")

    # Quality
    parser.add_argument("--quality-psd-method", choices=["welch", "multitaper"], default=None, help="PSD method for quality metrics")
    parser.add_argument("--quality-fmin", type=float, default=None, help="Min frequency for quality metrics")
    parser.add_argument("--quality-fmax", type=float, default=None, help="Max frequency for quality metrics")
    parser.add_argument("--quality-n-fft", type=int, default=None, help="FFT size for quality metrics")
    parser.add_argument("--quality-exclude-line-noise", action="store_true", default=None, help="Exclude line noise from quality metrics")
    parser.add_argument("--no-quality-exclude-line-noise", action="store_false", dest="quality_exclude_line_noise")
    parser.add_argument("--quality-snr-signal-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Signal band for SNR computation")
    parser.add_argument("--quality-snr-noise-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Noise band for SNR computation")
    parser.add_argument("--quality-muscle-band", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="Muscle band for artifact detection")

    # ERDS
    parser.add_argument("--erds-use-log-ratio", action="store_true", default=None, help="Use dB (log ratio) instead of percent for ERDS")
    parser.add_argument("--no-erds-use-log-ratio", action="store_false", dest="erds_use_log_ratio")
    parser.add_argument("--erds-min-baseline-power", type=float, default=None, help="Min baseline power for ERDS")
    parser.add_argument("--erds-min-active-power", type=float, default=None, help="Min active power for ERDS")
    parser.add_argument("--erds-min-segment-sec", type=float, default=None, help="Min segment duration for ERDS")
    parser.add_argument("--erds-bands", nargs="+", default=None, help="Bands for ERDS computation (e.g., alpha beta)")

    # Validation and output
    parser.add_argument("--min-epochs", type=int, default=None, help="Minimum epochs required for features")
    parser.add_argument("--save-subject-level-features", action="store_true", default=None, help="Save subject-level features for constant values")
    parser.add_argument("--no-save-subject-level-features", action="store_false", dest="save_subject_level_features", help="Do not save subject-level features")
    
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--also-save-csv",
        action="store_true",
        default=None,
        dest="also_save_csv",
        help="Also save feature tables as CSV files (in addition to parquet)",
    )
    output_group.add_argument(
        "--no-also-save-csv",
        action="store_false",
        dest="also_save_csv",
    )

    add_path_args(parser)

    return parser


def run_features(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the features command."""
    import sys
    from eeg_pipeline.pipelines.features import FeaturePipeline
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    
    # Capture CLI command for reproducibility (stored in extraction config)
    cli_command = " ".join(sys.argv)
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    if getattr(args, "freesurfer_dir", None):
        config.setdefault("paths", {})["freesurfer_dir"] = args.freesurfer_dir
    
    if args.mode == "compute":
        _apply_feature_config_overrides(args, config)

        time_ranges = []
        if getattr(args, "time_range", None):
            for name, tmin, tmax in args.time_range:
                time_ranges.append({
                    "name": name,
                    "tmin": float(tmin) if tmin.lower() != "none" and tmin != "" else None,
                    "tmax": float(tmax) if tmax.lower() != "none" and tmax != "" else None,
                })
        
        pipeline = FeaturePipeline(config=config)
        pipeline.run_batch(
            subjects=subjects,
            task=task,
            feature_categories=categories,
            bands=getattr(args, "bands", None),
            spatial_modes=getattr(args, "spatial", None),
            tmin=getattr(args, "tmin", None),
            tmax=getattr(args, "tmax", None),
            time_ranges=time_ranges or None,
            aggregation_method=getattr(args, "aggregation_method", "mean"),
            progress=progress,
            cli_command=cli_command,
        )
    elif args.mode == "visualize":
        visualize_features_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=categories,
        )
