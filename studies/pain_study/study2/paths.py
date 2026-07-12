"""Filesystem layout for Study 2 source-space analysis artifacts.

Stages hand off through these paths: each stage reads upstream artifacts and
writes its own, so the layout is the single contract between stages and the
runner's fail-fast input checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from studies.pain_study.study2.validation import require_config_string


def study2_output_root(config: Any) -> Path:
    root_name = require_config_string(config, "study2.outputs.root_name")
    return resolve_eeg_deriv_root(config) / "group" / "multimodal" / root_name


def study1_report_path(config: Any) -> Path:
    root_name = require_config_string(config, "study2.inputs.study1_root_name")
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


def subject_source_power_metadata_path(config: Any, *, subject_id: str, band: str) -> Path:
    return study2_output_root(config) / subject_id / "eeg" / "source" / f"source_power_{band}.json"


def source_vertex_manifest_path(config: Any) -> Path:
    return source_model_dir(config) / "common_source_vertices.npz"


def source_vertex_metadata_path(config: Any) -> Path:
    return source_model_dir(config) / "common_source_vertices.json"


def source_stage_fisher_z_path(config: Any, *, band: str) -> Path:
    return source_stage_dir(config) / f"fisher_z_{band}.npy"


def source_stage_partial_r_path(config: Any, *, band: str) -> Path:
    return source_stage_dir(config) / f"partial_r_{band}.npy"


def source_stage_qc_path(config: Any, *, band: str) -> Path:
    return source_stage_dir(config) / f"qc_{band}.tsv"


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


def haufe_fold_patterns_path(config: Any) -> Path:
    return sensor_dir(config) / "haufe_forward_patterns_by_fold.tsv"


def haufe_aggregate_patterns_path(config: Any) -> Path:
    return sensor_dir(config) / "haufe_forward_patterns_summary.tsv"


def haufe_stability_path(config: Any) -> Path:
    return sensor_dir(config) / "haufe_forward_patterns_stability.tsv"


def figures_dir(config: Any) -> Path:
    return study2_output_root(config) / "figures"


def haufe_figure_path(config: Any) -> Path:
    return figures_dir(config) / "haufe_forward_patterns.svg"


def primary_source_figure_path(config: Any) -> Path:
    return figures_dir(config) / "primary_source_associations.svg"


def primary_source_png_path(config: Any) -> Path:
    return figures_dir(config) / "primary_source_associations.png"


def primary_source_vertices_path(config: Any) -> Path:
    return figures_dir(config) / "primary_source_associations_vertices.tsv"


def primary_source_clusters_path(config: Any) -> Path:
    return figures_dir(config) / "primary_source_associations_clusters.tsv"


def primary_source_summary_path(config: Any) -> Path:
    return figures_dir(config) / "primary_source_associations_summary.tsv"


def primary_source_caption_path(config: Any) -> Path:
    return figures_dir(config) / "primary_source_associations_caption.txt"


def primary_source_manifest_path(config: Any) -> Path:
    return figures_dir(config) / "primary_source_associations_manifest.json"


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


def spatial_distance_matrix_path(config: Any) -> Path:
    return spatial_dir(config) / "surface_distance_matrix.npy"


def spatial_surrogate_metadata_path(config: Any) -> Path:
    return spatial_dir(config) / "surrogate_metadata.json"


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
    "figures_dir",
    "haufe_aggregate_patterns_path",
    "haufe_figure_path",
    "haufe_fold_patterns_path",
    "haufe_stability_path",
    "inference_dir",
    "null_source_maps_path",
    "point_spread_distances_path",
    "point_spread_resolution_matrix_path",
    "point_spread_summary_path",
    "point_spread_vertex_fwhm_path",
    "primary_source_caption_path",
    "primary_source_clusters_path",
    "primary_source_figure_path",
    "primary_source_manifest_path",
    "primary_source_png_path",
    "primary_source_summary_path",
    "primary_source_vertices_path",
    "robustness_metrics_path",
    "robustness_summary_path",
    "sensor_dir",
    "source_adjacency_path",
    "source_family_summary_path",
    "source_model_dir",
    "source_model_metrics_path",
    "source_model_qc_path",
    "spatial_correspondence_summary_path",
    "spatial_distance_matrix_path",
    "spatial_dir",
    "spatial_eeg_map_path",
    "spatial_fmri_map_path",
    "spatial_mask_path",
    "spatial_surrogate_maps_path",
    "spatial_surrogate_metadata_path",
    "source_stage_dir",
    "source_stage_fisher_z_path",
    "source_stage_frame_path",
    "source_stage_partial_r_path",
    "source_stage_qc_path",
    "source_vertex_manifest_path",
    "source_vertex_metadata_path",
    "study1_report_path",
    "study2_output_root",
    "subject_source_power_path",
    "subject_source_power_metadata_path",
]
