"""Filesystem layout for Study 2 source-space analysis artifacts.

Stages hand off through these paths: each stage reads upstream artifacts and
writes its own, so the layout is the single contract between stages and the
runner's fail-fast input checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root


def study2_output_root(config: Any) -> Path:
    root_name = str(get_config_value(config, "study2.outputs.root_name", "study2")).strip()
    if not root_name:
        raise ValueError("study2.outputs.root_name must be a non-empty string.")
    return resolve_eeg_deriv_root(config) / "group" / "multimodal" / root_name


def study1_report_path(config: Any) -> Path:
    root_name = str(get_config_value(config, "study2.inputs.study1_root_name", "study1")).strip()
    if not root_name:
        raise ValueError("study2.inputs.study1_root_name must be a non-empty string.")
    return (
        resolve_eeg_deriv_root(config)
        / "group"
        / "multimodal"
        / root_name
        / "reports"
        / "study1_report.tsv"
    )


def gate_dir(config: Any) -> Path:
    return study2_output_root(config) / "gate"


def gate_qc_path(config: Any) -> Path:
    return gate_dir(config) / "gate_qc.json"


def source_stage_dir(config: Any) -> Path:
    return study2_output_root(config) / "source_stage"


def source_stage_frame_path(config: Any) -> Path:
    return source_stage_dir(config) / "source_stage_input.tsv"


def subject_source_power_path(config: Any, *, subject_id: str, band: str) -> Path:
    return study2_output_root(config) / subject_id / "eeg" / "source" / f"source_power_{band}.npy"


def source_stage_fisher_z_path(config: Any, *, band: str) -> Path:
    return source_stage_dir(config) / f"fisher_z_{band}.npy"


def inference_dir(config: Any) -> Path:
    return study2_output_root(config) / "inference"


def source_adjacency_path(config: Any) -> Path:
    return inference_dir(config) / "adjacency.npy"


def null_source_maps_path(config: Any, *, band: str) -> Path:
    return inference_dir(config) / f"null_{band}.npy"


def source_family_summary_path(config: Any) -> Path:
    return inference_dir(config) / "source_family_summary.tsv"


def sensor_dir(config: Any) -> Path:
    return study2_output_root(config) / "sensor"


def haufe_input_path(config: Any) -> Path:
    return sensor_dir(config) / "haufe_input.npz"


def haufe_pattern_path(config: Any) -> Path:
    return sensor_dir(config) / "haufe_pattern.npy"


def haufe_covariance_path(config: Any) -> Path:
    return sensor_dir(config) / "haufe_feature_covariance.npy"


def haufe_summary_path(config: Any) -> Path:
    return sensor_dir(config) / "haufe_summary.tsv"


def source_model_dir(config: Any) -> Path:
    return study2_output_root(config) / "source_model"


def source_model_metrics_path(config: Any) -> Path:
    return source_model_dir(config) / "source_model_metrics.tsv"


def source_model_qc_path(config: Any) -> Path:
    return source_model_dir(config) / "source_model_qc.tsv"


def point_spread_resolution_matrix_path(config: Any) -> Path:
    return source_model_dir(config) / "point_spread_resolution_matrix.npy"


def point_spread_distances_path(config: Any) -> Path:
    return source_model_dir(config) / "point_spread_distances_mm.npy"


def point_spread_vertex_fwhm_path(config: Any) -> Path:
    return source_model_dir(config) / "point_spread_vertex_fwhm_mm.npy"


def point_spread_summary_path(config: Any) -> Path:
    return source_model_dir(config) / "point_spread_summary.tsv"


def diagnostics_dir(config: Any) -> Path:
    return study2_output_root(config) / "diagnostics"


def directional_prediction_map_path(config: Any, *, band: str) -> Path:
    return diagnostics_dir(config) / f"directional_prediction_{band}.npy"


def directional_target_map_path(config: Any, *, band: str) -> Path:
    return diagnostics_dir(config) / f"directional_target_{band}.npy"


def directional_cluster_mask_path(config: Any, *, band: str) -> Path:
    return diagnostics_dir(config) / f"directional_cluster_mask_{band}.npy"


def directional_consistency_summary_path(config: Any) -> Path:
    return diagnostics_dir(config) / "directional_consistency.tsv"


def artifact_metrics_path(config: Any) -> Path:
    return diagnostics_dir(config) / "artifact_metrics.tsv"


def artifact_controls_summary_path(config: Any) -> Path:
    return diagnostics_dir(config) / "artifact_controls.tsv"


def robustness_metrics_path(config: Any) -> Path:
    return diagnostics_dir(config) / "robustness_metrics.tsv"


def robustness_summary_path(config: Any) -> Path:
    return diagnostics_dir(config) / "robustness_summary.tsv"


def spatial_dir(config: Any) -> Path:
    return study2_output_root(config) / "spatial"


def spatial_eeg_map_path(config: Any, *, band: str) -> Path:
    return spatial_dir(config) / f"eeg_map_{band}.npy"


def spatial_fmri_map_path(config: Any, *, band: str) -> Path:
    return spatial_dir(config) / f"fmri_map_{band}.npy"


def spatial_surrogate_maps_path(config: Any, *, band: str) -> Path:
    return spatial_dir(config) / f"surrogate_maps_{band}.npy"


def spatial_mask_path(config: Any) -> Path:
    return spatial_dir(config) / "analysis_mask.npy"


def spatial_correspondence_summary_path(config: Any) -> Path:
    return spatial_dir(config) / "spatial_correspondence.tsv"


def behavioral_dir(config: Any) -> Path:
    return study2_output_root(config) / "behavioral"


def behavioral_convergence_input_path(config: Any) -> Path:
    return behavioral_dir(config) / "behavioral_convergence_input.tsv"


def behavioral_convergence_summary_path(config: Any) -> Path:
    return behavioral_dir(config) / "behavioral_convergence_summary.tsv"


def band_unique_dir(config: Any) -> Path:
    return study2_output_root(config) / "band_unique"


def band_unique_fisher_z_path(config: Any, *, band: str) -> Path:
    return band_unique_dir(config) / f"fisher_z_{band}.npy"


def band_unique_partial_r_path(config: Any, *, band: str) -> Path:
    return band_unique_dir(config) / f"partial_r_{band}.npy"


def band_unique_qc_path(config: Any, *, band: str) -> Path:
    return band_unique_dir(config) / f"qc_{band}.tsv"


def band_unique_null_maps_path(config: Any, *, band: str) -> Path:
    return band_unique_dir(config) / f"null_{band}.npy"


def band_unique_family_summary_path(config: Any) -> Path:
    return band_unique_dir(config) / "source_family_summary.tsv"


__all__ = [
    "gate_dir",
    "gate_qc_path",
    "artifact_controls_summary_path",
    "artifact_metrics_path",
    "band_unique_dir",
    "band_unique_family_summary_path",
    "band_unique_fisher_z_path",
    "band_unique_null_maps_path",
    "band_unique_partial_r_path",
    "band_unique_qc_path",
    "behavioral_convergence_input_path",
    "behavioral_convergence_summary_path",
    "behavioral_dir",
    "diagnostics_dir",
    "directional_cluster_mask_path",
    "directional_consistency_summary_path",
    "directional_prediction_map_path",
    "directional_target_map_path",
    "haufe_covariance_path",
    "haufe_input_path",
    "haufe_pattern_path",
    "haufe_summary_path",
    "inference_dir",
    "null_source_maps_path",
    "point_spread_distances_path",
    "point_spread_resolution_matrix_path",
    "point_spread_summary_path",
    "point_spread_vertex_fwhm_path",
    "robustness_metrics_path",
    "robustness_summary_path",
    "sensor_dir",
    "source_adjacency_path",
    "source_family_summary_path",
    "source_model_dir",
    "source_model_metrics_path",
    "source_model_qc_path",
    "spatial_correspondence_summary_path",
    "spatial_dir",
    "spatial_eeg_map_path",
    "spatial_fmri_map_path",
    "spatial_mask_path",
    "spatial_surrogate_maps_path",
    "source_stage_dir",
    "source_stage_fisher_z_path",
    "source_stage_frame_path",
    "study1_report_path",
    "study2_output_root",
    "subject_source_power_path",
]
