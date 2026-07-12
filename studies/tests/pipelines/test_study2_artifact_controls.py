from __future__ import annotations

import pytest

from studies.pain_study.study2.config import load_study2_config


def test_artifact_controls_report_unmet_criteria() -> None:
    from studies.pain_study.study2.artifact_controls import evaluate_artifact_controls

    qc = evaluate_artifact_controls(
        band="gamma",
        sensor_template_abs_r={"ocular": 0.10},
        source_artifact_map_abs_r={"scanner": 0.60},
        expression_p_values={"fd": 0.20, "dvars": 0.01},
        config=load_study2_config(),
    )

    assert qc.artifact_control_criteria_met is False
    assert qc.expression_adjusted_p_values["dvars"] == pytest.approx(0.02)
    assert qc.unmet_criteria == ("source_artifact_template", "artifact_expression")


def test_robustness_summary_uses_configured_thresholds() -> None:
    from studies.pain_study.study2.artifact_controls import evaluate_robustness_summary

    passing = evaluate_robustness_summary(
        significance_retained=True,
        sign_retained=True,
        unthresholded_spatial_r=0.51,
        cluster_dice=0.41,
        centroid_displacement_mm=14.0,
        config=load_study2_config(),
    )
    failing = evaluate_robustness_summary(
        significance_retained=True,
        sign_retained=True,
        unthresholded_spatial_r=0.49,
        cluster_dice=0.39,
        centroid_displacement_mm=16.0,
        config=load_study2_config(),
    )

    assert passing.robustness_criteria_met is True
    assert passing.unmet_criteria == ()
    assert failing.robustness_criteria_met is False
    assert failing.unmet_criteria == (
        "unthresholded_spatial_r",
        "cluster_dice",
        "centroid_displacement",
    )
