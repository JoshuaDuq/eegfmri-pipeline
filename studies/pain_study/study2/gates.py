"""Study 2 confirmatory eligibility gates inherited from Study 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value
from studies.pain_study.study2.validation import (
    finite_number,
    require_bool_metric,
    require_config_bool,
    require_config_probability,
    require_finite_metric,
    require_probability_metric,
)


@dataclass(frozen=True)
class Study1ConfirmatoryGateQC:
    confirmatory_eligible: bool
    failed_gates: tuple[str, ...]
    reason: str


STUDY1_REPORT_REQUIRED_COLUMNS = (
    "analysis_partition",
    "target",
    "feature_spec",
    "model",
    "mean_delta_r2",
    "ci_low_delta_r2",
    "p_value_delta_r2_holm",
    "temporal_negative_controls_passed",
)


def load_study1_confirmatory_row(
    report_path: Path,
    *,
    config: Any,
) -> dict[str, object]:
    """Select the single Study 1 primary confirmatory-cell row from a report."""
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing Study 1 report: {report_path}")
    report = pd.read_csv(report_path, sep="\t")
    missing = [column for column in STUDY1_REPORT_REQUIRED_COLUMNS if column not in report.columns]
    if missing:
        raise ValueError(f"Study 1 report is missing columns: {missing}.")

    cell = get_config_value(config, "study2.confirmatory.study1_cell", None)
    if not isinstance(cell, Mapping):
        raise ValueError("Study 2 config is missing study1_cell mapping.")

    mask = (
        (report["analysis_partition"].astype(str) == "primary")
        & (report["target"].astype(str) == str(cell["target"]))
        & (report["feature_spec"].astype(str) == str(cell["frequency_preset"]))
        & (report["model"].astype(str) == str(cell["model"]))
    )
    matches = report.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            "Study 2 expected exactly one Study 1 confirmatory row, "
            f"found {len(matches)}."
        )

    metrics = matches.iloc[0].to_dict()
    for key, value in list(metrics.items()):
        if isinstance(value, np.bool_):
            metrics[key] = bool(value)
    return metrics


def _optional_config_float(config: Any, key: str) -> float | None:
    value = get_config_value(config, key, None)
    if value is None:
        return None
    return finite_number(value, key)


def evaluate_study1_confirmatory_gates(
    metrics: Mapping[str, object],
    config: Any,
) -> Study1ConfirmatoryGateQC:
    """Evaluate configured Study 1 gates for confirmatory Study 2 entry."""
    if not isinstance(metrics, Mapping):
        raise TypeError("Study 2 Study 1 gate metrics must be a mapping.")

    max_holm_p = require_config_probability(
        config,
        "study2.confirmatory.study1_gates.max_holm_p",
    )
    min_delta_r2 = _optional_config_float(
        config,
        "study2.confirmatory.study1_gates.min_delta_r2",
    )
    min_delta_r2_lower_ci = _optional_config_float(
        config,
        "study2.confirmatory.study1_gates.min_delta_r2_lower_ci",
    )
    min_level2_delta_r2 = _optional_config_float(
        config,
        "study2.confirmatory.study1_gates.min_level2_delta_r2",
    )
    min_target_split_half_reliability = _optional_config_float(
        config,
        "study2.confirmatory.study1_gates.min_target_split_half_reliability",
    )
    min_target_reliability_n_trials = _optional_config_float(
        config,
        "study2.confirmatory.study1_gates.min_target_reliability_n_trials",
    )

    mean_delta_r2 = require_finite_metric(metrics, "mean_delta_r2")
    p_value_delta_r2_holm = require_probability_metric(
        metrics,
        "p_value_delta_r2_holm",
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
    if min_delta_r2 is not None and mean_delta_r2 < min_delta_r2:
        failed_gates.append("practical_effect_delta_r2")
    if min_delta_r2_lower_ci is not None:
        delta_r2_lower_ci = require_finite_metric(metrics, "ci_low_delta_r2")
        if delta_r2_lower_ci <= min_delta_r2_lower_ci:
            failed_gates.append("practical_effect_lower_ci")
    if min_level2_delta_r2 is not None:
        level2_mean_delta_r2 = require_finite_metric(metrics, "level2_mean_delta_r2")
        if level2_mean_delta_r2 < min_level2_delta_r2:
            failed_gates.append("level2_delta_r2")
    if min_target_split_half_reliability is not None:
        target_split_half_reliability = require_finite_metric(
            metrics,
            "target_split_half_reliability",
        )
        if target_split_half_reliability < min_target_split_half_reliability:
            failed_gates.append("target_split_half_reliability")
    if min_target_reliability_n_trials is not None:
        target_reliability_n_trials = require_finite_metric(
            metrics,
            "target_reliability_n_trials",
        )
        if target_reliability_n_trials < min_target_reliability_n_trials:
            failed_gates.append("target_reliability_n_trials")
    if require_positive_within_subject:
        within_subject_delta_r2 = require_finite_metric(
            metrics,
            "within_subject_centered_delta_r2",
        )
        if within_subject_delta_r2 <= 0.0:
            failed_gates.append("positive_within_subject_delta_r2")
    if require_temporal_controls:
        temporal_controls_passed = require_bool_metric(
            metrics,
            "temporal_negative_controls_passed",
        )
        if not temporal_controls_passed:
            failed_gates.append("temporal_negative_controls")
    if require_artifact_robustness:
        artifact_robustness_passed = require_bool_metric(
            metrics,
            "artifact_censoring_robustness_passed",
        )
        if not artifact_robustness_passed:
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
    "load_study1_confirmatory_row",
]
