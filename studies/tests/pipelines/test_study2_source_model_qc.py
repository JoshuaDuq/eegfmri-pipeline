from __future__ import annotations

import pytest

from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.source_model_qc import evaluate_source_model_qc


def _passing_source_model_metrics() -> dict[str, object]:
    return {
        "freesurfer_visual_qc_passed": True,
        "bem_succeeded": True,
        "has_subject_specific_electrode_positions": True,
        "uses_template_electrode_coordinates": False,
        "valid_eeg_channel_location_fraction": 0.95,
        "mean_coregistration_error_mm": 4.5,
        "max_coregistration_error_mm": 9.5,
        "forward_solution_valid": True,
        "forward_solution_rank_deficient_channels": False,
        "morph_to_fsaverage_succeeded": True,
    }


def test_source_model_qc_accepts_readme_aligned_subject() -> None:
    config = load_study2_config()

    qc = evaluate_source_model_qc(
        subject_id="sub-01",
        metrics=_passing_source_model_metrics(),
        config=config,
    )

    assert qc.subject_id == "sub-01"
    assert qc.source_model_criteria_met is True
    assert qc.unmet_criteria == ()


def test_source_model_qc_reports_fixed_readme_exclusion_rules() -> None:
    config = load_study2_config()
    metrics = _passing_source_model_metrics()
    metrics.update(
        {
            "freesurfer_visual_qc_passed": False,
            "uses_template_electrode_coordinates": True,
            "valid_eeg_channel_location_fraction": 0.89,
            "mean_coregistration_error_mm": 5.1,
            "max_coregistration_error_mm": 10.1,
            "forward_solution_rank_deficient_channels": True,
            "morph_to_fsaverage_succeeded": False,
        }
    )

    qc = evaluate_source_model_qc(
        subject_id="sub-01",
        metrics=metrics,
        config=config,
    )

    assert qc.source_model_criteria_met is False
    assert qc.unmet_criteria == (
        "freesurfer_visual_qc",
        "template_electrode_coordinates",
        "valid_eeg_channel_locations",
        "mean_coregistration_error",
        "max_coregistration_error",
        "forward_solution_rank",
        "morph_to_fsaverage",
    )


def test_source_model_qc_requires_all_readme_metrics() -> None:
    config = load_study2_config()
    metrics = _passing_source_model_metrics()
    del metrics["bem_succeeded"]

    with pytest.raises(ValueError, match="bem_succeeded"):
        evaluate_source_model_qc(subject_id="sub-01", metrics=metrics, config=config)


def test_source_model_qc_rejects_numeric_boolean_values() -> None:
    config = load_study2_config()
    metrics = _passing_source_model_metrics()
    metrics["valid_eeg_channel_location_fraction"] = True

    with pytest.raises(TypeError, match="valid_eeg_channel_location_fraction"):
        evaluate_source_model_qc(subject_id="sub-01", metrics=metrics, config=config)
