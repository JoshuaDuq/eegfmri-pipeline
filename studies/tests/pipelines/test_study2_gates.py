from __future__ import annotations

import pytest

from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.gates import evaluate_study1_confirmatory_gates


def _passing_study1_metrics() -> dict[str, object]:
    return {
        "mean_delta_r2": 0.031,
        "p_value_delta_r2_holm": 0.018,
        "ci_low_delta_r2": 0.006,
        "level2_mean_delta_r2": 0.006,
        "target_split_half_reliability": 0.51,
        "within_subject_centered_delta_r2": 0.004,
        "temporal_negative_controls_passed": True,
        "artifact_censoring_robustness_passed": True,
    }


def test_study2_confirmatory_gates_accept_study1_report_metrics() -> None:
    config = load_study2_config()

    qc = evaluate_study1_confirmatory_gates(_passing_study1_metrics(), config)

    assert qc.confirmatory_eligible is True
    assert qc.failed_gates == ()
    assert qc.reason == "all Study 1 gates passed"


def test_study2_confirmatory_gates_report_failed_required_prediction_gate() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    metrics.update(
        {
            "p_value_delta_r2_holm": 0.051,
        }
    )

    qc = evaluate_study1_confirmatory_gates(metrics, config)

    assert qc.confirmatory_eligible is False
    assert qc.failed_gates == ("significant_positive_delta_r2",)
    assert qc.reason == "failed Study 1 gates: significant_positive_delta_r2"


def test_study2_confirmatory_gates_require_configured_level2_metric() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    del metrics["level2_mean_delta_r2"]

    with pytest.raises(ValueError, match="level2_mean_delta_r2"):
        evaluate_study1_confirmatory_gates(metrics, config)


def test_study2_confirmatory_gates_require_configured_boolean_control_flags() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    metrics["temporal_negative_controls_passed"] = 1

    with pytest.raises(TypeError, match="temporal_negative_controls_passed"):
        evaluate_study1_confirmatory_gates(metrics, config)


def test_study2_confirmatory_gates_require_configured_lower_ci_metric() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    del metrics["ci_low_delta_r2"]

    with pytest.raises(ValueError, match="ci_low_delta_r2"):
        evaluate_study1_confirmatory_gates(metrics, config)


def test_study2_confirmatory_gates_report_target_reliability_failure() -> None:
    config = load_study2_config()
    metrics = _passing_study1_metrics()
    metrics["target_split_half_reliability"] = 0.39

    qc = evaluate_study1_confirmatory_gates(metrics, config)

    assert qc.confirmatory_eligible is False
    assert qc.failed_gates == ("target_split_half_reliability",)
