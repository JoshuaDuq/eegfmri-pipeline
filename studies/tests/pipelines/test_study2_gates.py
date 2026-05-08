from __future__ import annotations

import pytest

from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.gates import evaluate_study1_confirmatory_gates


def _passing_study1_metrics() -> dict[str, object]:
    return {
        "mean_delta_r2": 0.031,
        "p_value_delta_r2_holm": 0.018,
        "delta_r2_lower_ci": 0.006,
        "level2_mean_delta_r2": 0.006,
        "within_subject_centered_delta_r2": 0.004,
        "temporal_negative_controls_passed": True,
        "artifact_censoring_robustness_passed": True,
    }


def test_study2_confirmatory_gates_accept_readme_aligned_metrics() -> None:
    config = load_study2_config()

    qc = evaluate_study1_confirmatory_gates(_passing_study1_metrics(), config)

    assert qc.confirmatory_eligible is True
    assert qc.failed_gates == ()
    assert qc.reason == "all Study 1 gates passed"


def test_study2_confirmatory_gates_report_failed_readme_gates() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    metrics.update(
        {
            "p_value_delta_r2_holm": 0.051,
            "delta_r2_lower_ci": 0.005,
            "within_subject_centered_delta_r2": -0.001,
            "artifact_censoring_robustness_passed": False,
        }
    )

    qc = evaluate_study1_confirmatory_gates(metrics, config)

    assert qc.confirmatory_eligible is False
    assert qc.failed_gates == (
        "significant_positive_delta_r2",
        "practical_effect_lower_ci",
        "positive_within_subject_delta_r2",
        "artifact_censoring_robustness",
    )
    assert qc.reason == (
        "failed Study 1 gates: significant_positive_delta_r2, practical_effect_lower_ci, "
        "positive_within_subject_delta_r2, artifact_censoring_robustness"
    )


def test_study2_confirmatory_gates_require_all_readme_metrics() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    del metrics["level2_mean_delta_r2"]

    with pytest.raises(ValueError, match="level2_mean_delta_r2"):
        evaluate_study1_confirmatory_gates(metrics, config)


def test_study2_confirmatory_gates_require_boolean_control_flags() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    metrics["temporal_negative_controls_passed"] = 1

    with pytest.raises(TypeError, match="temporal_negative_controls_passed"):
        evaluate_study1_confirmatory_gates(metrics, config)
