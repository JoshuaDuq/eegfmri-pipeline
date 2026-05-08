"""Study 2 confirmatory eligibility gates inherited from Study 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from studies.pain_study.study2.validation import (
    require_bool_metric,
    require_config_bool,
    require_config_float,
    require_config_probability,
    require_finite_metric,
    require_probability_metric,
)


@dataclass(frozen=True)
class Study1ConfirmatoryGateQC:
    confirmatory_eligible: bool
    failed_gates: tuple[str, ...]
    reason: str


def evaluate_study1_confirmatory_gates(
    metrics: Mapping[str, object],
    config: Any,
) -> Study1ConfirmatoryGateQC:
    """Evaluate the README-defined Study 1 gates for confirmatory Study 2 entry."""
    if not isinstance(metrics, Mapping):
        raise TypeError("Study 2 Study 1 gate metrics must be a mapping.")

    max_holm_p = require_config_probability(
        config,
        "study2.confirmatory.study1_gates.max_holm_p",
    )
    min_delta_r2 = require_config_float(
        config,
        "study2.confirmatory.study1_gates.min_delta_r2",
    )
    min_delta_r2_lower_ci = require_config_float(
        config,
        "study2.confirmatory.study1_gates.min_delta_r2_lower_ci",
    )
    min_level2_delta_r2 = require_config_float(
        config,
        "study2.confirmatory.study1_gates.min_level2_delta_r2",
    )

    mean_delta_r2 = require_finite_metric(metrics, "mean_delta_r2")
    p_value_delta_r2_holm = require_probability_metric(
        metrics,
        "p_value_delta_r2_holm",
    )
    delta_r2_lower_ci = require_finite_metric(metrics, "delta_r2_lower_ci")
    level2_mean_delta_r2 = require_finite_metric(metrics, "level2_mean_delta_r2")
    within_subject_delta_r2 = require_finite_metric(
        metrics,
        "within_subject_centered_delta_r2",
    )
    temporal_controls_passed = require_bool_metric(
        metrics,
        "temporal_negative_controls_passed",
    )
    artifact_robustness_passed = require_bool_metric(
        metrics,
        "artifact_censoring_robustness_passed",
    )

    require_positive_within_subject = require_config_bool(
        config,
        "study2.confirmatory.study1_gates.require_positive_within_subject_delta_r2",
    )
    require_temporal_controls = require_config_bool(
        config,
        "study2.confirmatory.study1_gates.require_temporal_negative_controls",
    )
    require_artifact_robustness = require_config_bool(
        config,
        "study2.confirmatory.study1_gates.require_artifact_censoring_robustness",
    )

    failed_gates: list[str] = []
    if mean_delta_r2 <= 0.0 or p_value_delta_r2_holm > max_holm_p:
        failed_gates.append("significant_positive_delta_r2")
    if mean_delta_r2 < min_delta_r2:
        failed_gates.append("practical_effect_delta_r2")
    if delta_r2_lower_ci <= min_delta_r2_lower_ci:
        failed_gates.append("practical_effect_lower_ci")
    if level2_mean_delta_r2 < min_level2_delta_r2:
        failed_gates.append("level2_delta_r2")
    if require_positive_within_subject and within_subject_delta_r2 <= 0.0:
        failed_gates.append("positive_within_subject_delta_r2")
    if require_temporal_controls and not temporal_controls_passed:
        failed_gates.append("temporal_negative_controls")
    if require_artifact_robustness and not artifact_robustness_passed:
        failed_gates.append("artifact_censoring_robustness")

    failed = tuple(failed_gates)
    if not failed:
        return Study1ConfirmatoryGateQC(
            confirmatory_eligible=True,
            failed_gates=(),
            reason="all Study 1 gates passed",
        )

    return Study1ConfirmatoryGateQC(
        confirmatory_eligible=False,
        failed_gates=failed,
        reason=f"failed Study 1 gates: {', '.join(failed)}",
    )


__all__ = [
    "Study1ConfirmatoryGateQC",
    "evaluate_study1_confirmatory_gates",
]
