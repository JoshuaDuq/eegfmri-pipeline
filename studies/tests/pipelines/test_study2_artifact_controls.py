from __future__ import annotations

import pytest

from studies.pain_study.study2.config import load_study2_config


def test_artifact_controls_relabel_contaminated_gamma_as_exploratory() -> None:
    from studies.pain_study.study2.artifact_controls import evaluate_artifact_controls

    qc = evaluate_artifact_controls(
        band="gamma",
        sensor_template_abs_r={"ocular": 0.10},
        source_artifact_map_abs_r={"scanner": 0.60},
        expression_p_values={"fd": 0.20, "dvars": 0.01},
        config=load_study2_config(),
    )

    assert qc.contaminated is True
    assert qc.interpretation == "exploratory_artifact_contaminated"
    assert qc.expression_q_values["dvars"] == pytest.approx(0.02)
    assert qc.failed_gates == ("source_artifact_template", "artifact_expression")


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

    assert passing.passed is True
    assert passing.failed_gates == ()
    assert failing.passed is False
    assert failing.failed_gates == (
        "unthresholded_spatial_r",
        "cluster_dice",
        "centroid_displacement",
    )
