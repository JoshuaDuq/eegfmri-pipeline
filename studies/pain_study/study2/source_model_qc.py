"""Study 2 source-model fixed exclusion rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from studies.pain_study.study2.validation import (
    require_bool_metric,
    require_config_float,
    require_finite_metric,
)


@dataclass(frozen=True)
class SourceModelQC:
    subject_id: str
    eligible: bool
    valid_eeg_channel_location_fraction: float
    mean_coregistration_error_mm: float
    max_coregistration_error_mm: float
    failed_gates: tuple[str, ...]
    reason: str


def evaluate_source_model_qc(
    *,
    subject_id: str,
    metrics: Mapping[str, object],
    config: Any,
) -> SourceModelQC:
    """Evaluate README Section 7 source-model QC exclusions for one subject."""
    subject_label = str(subject_id).strip()
    if not subject_label:
        raise ValueError("Study 2 source-model QC requires a non-empty subject_id.")
    if not isinstance(metrics, Mapping):
        raise TypeError("Study 2 source-model QC metrics must be a mapping.")

    min_location_fraction = require_config_float(
        config,
        "study2.source_model_qc.min_valid_eeg_channel_location_fraction",
    )
    max_mean_error = require_config_float(
        config,
        "study2.source_model_qc.max_mean_coregistration_error_mm",
    )
    max_error = require_config_float(
        config,
        "study2.source_model_qc.max_coregistration_error_mm",
    )
    _validate_source_model_thresholds(
        min_location_fraction=min_location_fraction,
        max_mean_error=max_mean_error,
        max_error=max_error,
    )

    freesurfer_passed = require_bool_metric(metrics, "freesurfer_visual_qc_passed")
    bem_succeeded = require_bool_metric(metrics, "bem_succeeded")
    has_measured_electrodes = require_bool_metric(
        metrics,
        "has_subject_specific_electrode_positions",
    )
    uses_template_electrodes = require_bool_metric(
        metrics,
        "uses_template_electrode_coordinates",
    )
    valid_location_fraction = require_finite_metric(
        metrics,
        "valid_eeg_channel_location_fraction",
    )
    mean_coreg_error = require_finite_metric(metrics, "mean_coregistration_error_mm")
    max_coreg_error = require_finite_metric(metrics, "max_coregistration_error_mm")
    forward_solution_valid = require_bool_metric(metrics, "forward_solution_valid")
    forward_rank_deficient = require_bool_metric(
        metrics,
        "forward_solution_rank_deficient_channels",
    )
    morph_succeeded = require_bool_metric(metrics, "morph_to_fsaverage_succeeded")
    _validate_source_model_metrics(
        valid_location_fraction=valid_location_fraction,
        mean_coreg_error=mean_coreg_error,
        max_coreg_error=max_coreg_error,
    )

    failed_gates: list[str] = []
    if not freesurfer_passed:
        failed_gates.append("freesurfer_visual_qc")
    if not bem_succeeded:
        failed_gates.append("boundary_element_model")
    if not has_measured_electrodes:
        failed_gates.append("subject_specific_electrode_positions")
    if uses_template_electrodes:
        failed_gates.append("template_electrode_coordinates")
    if valid_location_fraction < min_location_fraction:
        failed_gates.append("valid_eeg_channel_locations")
    if mean_coreg_error > max_mean_error:
        failed_gates.append("mean_coregistration_error")
    if max_coreg_error > max_error:
        failed_gates.append("max_coregistration_error")
    if not forward_solution_valid:
        failed_gates.append("forward_solution")
    if forward_rank_deficient:
        failed_gates.append("forward_solution_rank")
    if not morph_succeeded:
        failed_gates.append("morph_to_fsaverage")

    failed = tuple(failed_gates)
    return SourceModelQC(
        subject_id=subject_label,
        eligible=not failed,
        valid_eeg_channel_location_fraction=valid_location_fraction,
        mean_coregistration_error_mm=mean_coreg_error,
        max_coregistration_error_mm=max_coreg_error,
        failed_gates=failed,
        reason="" if not failed else f"failed source-model QC gates: {', '.join(failed)}",
    )


def _validate_source_model_thresholds(
    *,
    min_location_fraction: float,
    max_mean_error: float,
    max_error: float,
) -> None:
    if min_location_fraction <= 0.0 or min_location_fraction > 1.0:
        raise ValueError("Study 2 source-model QC location threshold must be in (0, 1].")
    if max_mean_error < 0.0:
        raise ValueError("Study 2 mean coregistration threshold must be non-negative.")
    if max_error < 0.0:
        raise ValueError("Study 2 maximum coregistration threshold must be non-negative.")


def _validate_source_model_metrics(
    *,
    valid_location_fraction: float,
    mean_coreg_error: float,
    max_coreg_error: float,
) -> None:
    if valid_location_fraction < 0.0 or valid_location_fraction > 1.0:
        raise ValueError("Study 2 valid_eeg_channel_location_fraction must be in [0, 1].")
    if mean_coreg_error < 0.0:
        raise ValueError("Study 2 mean_coregistration_error_mm must be non-negative.")
    if max_coreg_error < 0.0:
        raise ValueError("Study 2 max_coregistration_error_mm must be non-negative.")


__all__ = [
    "SourceModelQC",
    "evaluate_source_model_qc",
]
