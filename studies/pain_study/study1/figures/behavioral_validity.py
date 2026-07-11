"""Participant-level behavioral construct validity for Study 1 signatures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.validity_data import (
    WITHIN_SCALE_INTENSITY_COLUMN,
)

PAIN_REPORT_COLUMN = "pain_binary_coded"
TERM_COLUMNS = (
    ("painful_report", "painful_report_beta"),
    (WITHIN_SCALE_INTENSITY_COLUMN, "within_scale_intensity_beta"),
)
DESIGN_RANK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class BehavioralValiditySpecification:
    """Target and adjustment contract for one validity model."""

    target: str
    color_config_key: str
    adjustment_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class BehavioralValiditySummary:
    """Participant models and equally weighted cohort estimates."""

    target: str
    participant_models: pd.DataFrame
    cohort_estimates: pd.DataFrame


NPS_SPECIFICATION = BehavioralValiditySpecification(
    target="NPS",
    color_config_key="nps",
)
SIIPS1_SPECIFICATION = BehavioralValiditySpecification(
    target="SIIPS1",
    color_config_key="siips1",
    adjustment_columns=("NPS",),
)


def require_behavioral_validity_target(
    summary: BehavioralValiditySummary,
    expected_target: str,
) -> None:
    observed_targets = {str(summary.target)}
    for table_name, frame in (
        ("participant models", summary.participant_models),
        ("cohort estimates", summary.cohort_estimates),
    ):
        if frame.empty or "target" not in frame.columns:
            raise ValueError(
                f"Behavioral validity {table_name} must contain target-labelled rows."
            )
        observed_targets.update(frame["target"].astype(str).unique().tolist())
    if observed_targets != {expected_target}:
        raise ValueError(
            f"Behavioral validity writer requires a {expected_target} summary; "
            f"observed {sorted(observed_targets)}."
        )


def build_behavioral_validity_summary(
    trials: pd.DataFrame,
    *,
    specification: BehavioralValiditySpecification,
    config: Any,
) -> BehavioralValiditySummary:
    temperatures = _configured_temperatures(config)
    validated = _validated_trials(trials, specification, temperatures)
    participant_models = _fit_participant_models(
        validated,
        specification,
        temperatures,
    )
    estimable = participant_models.loc[participant_models["estimable"]].copy()
    minimum = int(require_config_value(config, "study1.cohort.min_subjects"))
    if minimum < 2:
        raise ValueError("study1.cohort.min_subjects must be at least 2.")
    if len(estimable) < minimum:
        raise ValueError(
            f"{specification.target} behavioral validity requires at least {minimum} "
            f"estimable participants; observed {len(estimable)}."
        )
    cohort_estimates = _participant_bootstrap(
        estimable,
        target=specification.target,
        config=config,
    )
    return BehavioralValiditySummary(
        target=specification.target,
        participant_models=participant_models,
        cohort_estimates=cohort_estimates,
    )


def _validated_trials(
    trials: pd.DataFrame,
    specification: BehavioralValiditySpecification,
    temperatures: tuple[float, ...],
) -> pd.DataFrame:
    required = (
        "subject_id",
        "stimulus_temp",
        PAIN_REPORT_COLUMN,
        WITHIN_SCALE_INTENSITY_COLUMN,
        specification.target,
        *specification.adjustment_columns,
    )
    missing = [column for column in required if column not in trials.columns]
    if missing:
        raise ValueError(f"Behavioral validity trials are missing required columns: {missing}.")

    validated = trials.loc[:, list(dict.fromkeys(required))].copy()
    validated["subject_id"] = validated["subject_id"].astype(str)
    if (validated["subject_id"].str.strip() == "").any():
        raise ValueError("Behavioral validity subject identifiers must be non-empty.")
    for column in required[1:]:
        validated[column] = _finite_numeric_series(validated, column)

    pain_report = validated[PAIN_REPORT_COLUMN]
    if not np.allclose(pain_report, np.round(pain_report)) or not pain_report.isin((0, 1)).all():
        raise ValueError("Behavioral validity pain reports must contain only 0 and 1.")
    validated[PAIN_REPORT_COLUMN] = pain_report.round().astype(int)

    observed = tuple(sorted(validated["stimulus_temp"].unique().tolist()))
    if observed != temperatures:
        raise ValueError(
            "Observed behavioral validity temperatures do not match configuration: "
            f"observed={observed}, configured={temperatures}."
        )
    return validated.sort_values(
        ["subject_id", "stimulus_temp"],
        kind="stable",
    ).reset_index(drop=True)


def _fit_participant_models(
    trials: pd.DataFrame,
    specification: BehavioralValiditySpecification,
    temperatures: tuple[float, ...],
) -> pd.DataFrame:
    records = [
        _fit_participant(rows, specification, temperatures)
        for _, rows in trials.groupby("subject_id", sort=True)
    ]
    return pd.DataFrame.from_records(records)


def _fit_participant(
    rows: pd.DataFrame,
    specification: BehavioralValiditySpecification,
    temperatures: tuple[float, ...],
) -> dict[str, Any]:
    subject_id = str(rows["subject_id"].iloc[0])
    record: dict[str, Any] = {
        "subject_id": subject_id,
        "target": specification.target,
        "estimable": False,
        "non_estimability_reason": "none",
        "painful_report_beta": float("nan"),
        "within_scale_intensity_beta": float("nan"),
        "n_trials": int(len(rows)),
        "residual_degrees_of_freedom": pd.NA,
    }
    reason = _basic_non_estimability_reason(rows, specification, temperatures)
    if reason is not None:
        record["non_estimability_reason"] = reason
        return record

    design = _design_matrix(rows, specification, temperatures)
    residual_degrees_of_freedom = len(rows) - design.shape[1]
    record["residual_degrees_of_freedom"] = residual_degrees_of_freedom
    if residual_degrees_of_freedom <= 0:
        record["non_estimability_reason"] = "nonpositive_residual_degrees_of_freedom"
        return record
    if not _is_full_rank(design):
        record["non_estimability_reason"] = "rank_deficient_design"
        return record

    target = _standardize(rows[specification.target]).to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    focal = coefficients[-2:]
    if not np.isfinite(focal).all():
        raise ValueError(
            f"{specification.target} produced non-finite coefficients for {subject_id}."
        )
    record.update(
        estimable=True,
        painful_report_beta=float(focal[0]),
        within_scale_intensity_beta=float(focal[1]),
    )
    return record


def _basic_non_estimability_reason(
    rows: pd.DataFrame,
    specification: BehavioralValiditySpecification,
    temperatures: tuple[float, ...],
) -> str | None:
    observed_temperatures = tuple(sorted(rows["stimulus_temp"].unique().tolist()))
    if observed_temperatures != temperatures:
        return "missing_temperature_levels"
    variance_columns = (
        specification.target,
        PAIN_REPORT_COLUMN,
        WITHIN_SCALE_INTENSITY_COLUMN,
        *specification.adjustment_columns,
    )
    for column in variance_columns:
        if rows[column].nunique() < 2 or rows[column].std(ddof=1) <= 0.0:
            return f"constant_{column}"
    return None


def _design_matrix(
    rows: pd.DataFrame,
    specification: BehavioralValiditySpecification,
    temperatures: tuple[float, ...],
) -> np.ndarray:
    temperature = pd.Categorical(
        rows["stimulus_temp"],
        categories=temperatures,
        ordered=True,
    )
    temperature_dummies = pd.get_dummies(temperature, drop_first=True, dtype=float)
    columns: list[np.ndarray] = [
        np.ones(len(rows), dtype=float),
        *temperature_dummies.to_numpy(dtype=float).T,
    ]
    columns.extend(
        _standardize(rows[column]).to_numpy(dtype=float)
        for column in specification.adjustment_columns
    )
    columns.extend(
        (
            _standardize(rows[PAIN_REPORT_COLUMN]).to_numpy(dtype=float),
            _standardize(rows[WITHIN_SCALE_INTENSITY_COLUMN]).to_numpy(dtype=float),
        )
    )
    return np.column_stack(columns)


def _standardize(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std(ddof=1)


def _is_full_rank(design: np.ndarray) -> bool:
    singular_values = np.linalg.svd(design, compute_uv=False)
    return bool(
        singular_values.size == design.shape[1]
        and singular_values[-1] / singular_values[0] >= DESIGN_RANK_TOLERANCE
    )


def _participant_bootstrap(
    participant_models: pd.DataFrame,
    *,
    target: str,
    config: Any,
) -> pd.DataFrame:
    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    iterations = int(bootstrap["iterations"])
    confidence_level = float(bootstrap["confidence_level"])
    seed = int(bootstrap["seed"])
    max_invalid_fraction = float(bootstrap["max_invalid_fraction"])
    if iterations < 1:
        raise ValueError("Study 1 validity bootstrap iterations must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Study 1 validity bootstrap confidence level must be in (0, 1).")
    if not 0.0 <= max_invalid_fraction < 1.0:
        raise ValueError("Study 1 validity bootstrap invalid fraction must be in [0, 1).")

    coefficient_columns = [column for _, column in TERM_COLUMNS]
    matrix = participant_models[coefficient_columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Estimable behavioral validity coefficients must be finite.")
    n_subjects = len(matrix)
    max_attempts = math.ceil(iterations / (1.0 - max_invalid_fraction))
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    attempted = 0
    while len(accepted) < iterations and attempted < max_attempts:
        sampled = matrix[rng.integers(0, n_subjects, size=n_subjects)]
        attempted += 1
        estimate = sampled.mean(axis=0)
        if np.isfinite(estimate).all():
            accepted.append(estimate)
    if len(accepted) != iterations:
        raise ValueError(
            "Study 1 behavioral validity bootstrap exceeded its invalid-draw budget: "
            f"valid={len(accepted)}, attempted={attempted}."
        )

    bootstrap_means = np.vstack(accepted)
    tail = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, [tail, 1.0 - tail], axis=0)
    means = matrix.mean(axis=0)
    return pd.DataFrame(
        {
            "target": target,
            "term": [term for term, _ in TERM_COLUMNS],
            "mean": means,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_subjects": n_subjects,
        }
    )


def _configured_temperatures(config: Any) -> tuple[float, ...]:
    raw = require_config_value(config, "study1.figures.validity.temperatures")
    temperatures = tuple(float(value) for value in raw)
    if len(temperatures) < 2 or len(set(temperatures)) != len(temperatures):
        raise ValueError("Study 1 validity temperatures must contain unique levels.")
    if temperatures != tuple(sorted(temperatures)):
        raise ValueError("Study 1 validity temperatures must be ordered ascending.")
    return temperatures


def _finite_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Behavioral validity column '{column}' must contain finite numbers.")
    return values.astype(float)


__all__ = [
    "BehavioralValiditySpecification",
    "BehavioralValiditySummary",
    "NPS_SPECIFICATION",
    "SIIPS1_SPECIFICATION",
    "build_behavioral_validity_summary",
    "require_behavioral_validity_target",
]
