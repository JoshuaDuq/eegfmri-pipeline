"""Artifact and robustness criteria for Study 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from studies.pain_study.study2.statistics import holm_q_values
from studies.pain_study.study2.validation import (
    finite_number,
    require_config_float,
    require_config_value,
)


@dataclass(frozen=True)
class ArtifactControlQC:
    band: str
    artifact_control_criteria_met: bool
    unmet_criteria: tuple[str, ...]
    expression_q_values: dict[str, float]
    missing_controls: tuple[str, ...]


@dataclass(frozen=True)
class RobustnessQC:
    robustness_criteria_met: bool
    unmet_criteria: tuple[str, ...]


def evaluate_artifact_controls(
    *,
    band: str,
    sensor_template_abs_r: Mapping[str, object],
    source_artifact_map_abs_r: Mapping[str, object],
    expression_p_values: Mapping[str, object],
    config: Any,
) -> ArtifactControlQC:
    band_name = str(band).strip().lower()
    if not band_name:
        raise ValueError("Study 2 artifact-controls band must be non-empty.")

    sensor_threshold = require_config_float(
        config,
        "study2.artifact_controls.sensor_template_abs_r_threshold",
    )
    source_threshold = require_config_float(
        config,
        "study2.artifact_controls.source_artifact_map_abs_r_threshold",
    )
    alpha = require_config_float(config, "study2.artifact_controls.holm_alpha")
    sensor_values = _finite_mapping(sensor_template_abs_r, name="sensor artifact templates")
    source_values = _finite_mapping(source_artifact_map_abs_r, name="source artifact maps")
    expression_q_values = holm_q_values(expression_p_values)
    required_metrics = _required_metric_names(config)
    available_metrics = set(sensor_values) | set(source_values) | set(expression_q_values)
    missing_controls = tuple(name for name in required_metrics if name not in available_metrics)

    unmet_criteria: list[str] = []
    if sensor_values and max(sensor_values.values()) > sensor_threshold:
        unmet_criteria.append("sensor_artifact_template")
    if source_values and max(source_values.values()) > source_threshold:
        unmet_criteria.append("source_artifact_template")
    if expression_q_values and min(expression_q_values.values()) <= alpha:
        unmet_criteria.append("artifact_expression")
    unmet_criteria.extend(f"missing_control:{name}" for name in missing_controls)

    return ArtifactControlQC(
        band=band_name,
        artifact_control_criteria_met=not unmet_criteria,
        unmet_criteria=tuple(unmet_criteria),
        expression_q_values=expression_q_values,
        missing_controls=missing_controls,
    )


def _required_metric_names(config: Any) -> tuple[str, ...]:
    values = require_config_value(config, "study2.artifact_controls.required_metrics")
    if not isinstance(values, list) or not values:
        raise ValueError("Study 2 artifact control required_metrics must be a non-empty list.")
    names = tuple(str(value).strip() for value in values)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Study 2 artifact control required_metrics must be unique names.")
    return names


def evaluate_robustness_summary(
    *,
    significance_retained: bool,
    sign_retained: bool,
    unthresholded_spatial_r: float,
    cluster_dice: float,
    centroid_displacement_mm: float,
    config: Any,
) -> RobustnessQC:
    if not isinstance(significance_retained, bool):
        raise TypeError("Study 2 robustness significance_retained must be boolean.")
    if not isinstance(sign_retained, bool):
        raise TypeError("Study 2 robustness sign_retained must be boolean.")

    min_spatial_r = require_config_float(config, "study2.robustness.min_unthresholded_spatial_r")
    min_dice = require_config_float(config, "study2.robustness.min_cluster_dice")
    max_displacement = require_config_float(config, "study2.robustness.max_centroid_displacement_mm")
    spatial_r = _finite_float(unthresholded_spatial_r, "unthresholded_spatial_r")
    dice = _finite_float(cluster_dice, "cluster_dice")
    displacement = _finite_float(centroid_displacement_mm, "centroid_displacement_mm")

    unmet_criteria: list[str] = []
    if not significance_retained:
        unmet_criteria.append("significance_retained")
    if not sign_retained:
        unmet_criteria.append("sign_retained")
    if spatial_r < min_spatial_r:
        unmet_criteria.append("unthresholded_spatial_r")
    if dice < min_dice:
        unmet_criteria.append("cluster_dice")
    if displacement > max_displacement:
        unmet_criteria.append("centroid_displacement")
    return RobustnessQC(
        robustness_criteria_met=not unmet_criteria,
        unmet_criteria=tuple(unmet_criteria),
    )


def _finite_mapping(values: Mapping[str, object], *, name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"Study 2 {name} must be a mapping.")
    parsed: dict[str, float] = {}
    for key, value in values.items():
        number = _finite_float(value, f"{name}.{key}")
        if number < 0.0:
            raise ValueError(f"Study 2 {name} values must be non-negative.")
        parsed[str(key)] = number
    return parsed


def _finite_float(value: object, name: str) -> float:
    return finite_number(value, name)


__all__ = [
    "ArtifactControlQC",
    "RobustnessQC",
    "evaluate_artifact_controls",
    "evaluate_robustness_summary",
]
