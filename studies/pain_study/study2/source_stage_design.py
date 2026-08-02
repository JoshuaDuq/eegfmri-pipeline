"""Design-matrix helpers for Study 2 source-stage analyses."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from studies.pain_study.study1.targets import _categorical_level_column
from studies.pain_study.study2.validation import require_config_string, require_config_value

RAW_LEVEL2_ARTIFACT_COLUMNS = {
    "framewise_displacement": "hrf_weighted_framewise_displacement",
    "std_dvars": "hrf_weighted_std_dvars",
    "fp1_fp2_high_frequency_power": "hrf_weighted_fp1_fp2_high_frequency_power",
}
BAND_STANDARDIZED_COLUMN_KEYS = {
    "alpha": ("study2.contributions.alpha_standardized_column", "eta_alpha_z"),
    "beta": ("study2.contributions.beta_standardized_column", "eta_beta_z"),
    "gamma": ("study2.contributions.gamma_standardized_column", "eta_gamma_z"),
}


def source_stage_required_columns(
    config: Any,
    *,
    target_column: str,
    adjustment_columns: tuple[str, ...],
) -> tuple[str, ...]:
    columns = (
        "run",
        "trial_index",
        *source_stage_continuous_columns(config),
        *source_stage_categorical_columns(config),
        target_column,
        *adjustment_columns,
    )
    return tuple(dict.fromkeys(columns))


def combined_score_column(config: Any) -> str:
    return require_config_string(config, "study2.contributions.combined_standardized_column")


def band_standardized_column(config: Any, band: str) -> str:
    entry = BAND_STANDARDIZED_COLUMN_KEYS.get(band)
    if entry is None:
        raise ValueError("Study 2 source-stage band must be 'alpha', 'beta', or 'gamma'.")
    config_key, _default_column = entry
    return require_config_string(config, config_key)


def contribution_bands(config: Any) -> tuple[str, ...]:
    raw_bands = require_config_value(
        config,
        "study2.confirmatory.study1_cell.contribution_bands",
    )
    bands = required_string_tuple(
        raw_bands,
        field_name="study2.confirmatory.study1_cell.contribution_bands",
    )
    if not bands:
        raise ValueError("study2.confirmatory.study1_cell.contribution_bands must be non-empty.")
    return tuple(band.lower() for band in bands)


def band_adjustment_columns(config: Any, band: str) -> tuple[str, ...]:
    bands = contribution_bands(config)
    if band not in bands:
        raise ValueError(
            f"Study 2 source-stage band '{band}' is not in the configured "
            f"contribution bands {bands}."
        )
    return tuple(band_standardized_column(config, other) for other in bands if other != band)


def build_source_stage_design(
    frame: pd.DataFrame,
    *,
    config: Any,
    adjustment_columns: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    design_parts: list[pd.DataFrame] = []
    continuous_columns = source_stage_continuous_columns(config)
    categorical_columns = source_stage_categorical_columns(config)
    for column in continuous_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Source-stage continuous column '{column}' contains non-finite values."
            )
        design_parts.append(pd.DataFrame({column: values.to_numpy(dtype=float)}))

    for column in categorical_columns:
        design_parts.append(
            build_fixed_categorical_design(
                frame=frame,
                config=config,
                column=column,
            )
        )

    for column in adjustment_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Source-stage adjacent-band contribution '{column}' contains non-finite values."
            )
        design_parts.append(pd.DataFrame({column: values.to_numpy(dtype=float)}))

    design_frame = pd.concat(design_parts, axis=1)
    return design_frame.to_numpy(dtype=float), list(design_frame.columns)


def source_stage_contribution_values(frame: pd.DataFrame, *, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Source-stage contribution column '{column}' contains non-finite values.")
    return values.to_numpy(dtype=float)


def contribution_stability_condition_number(
    *,
    design: np.ndarray,
    target_values: np.ndarray,
) -> tuple[float, str]:
    augmented_design = np.column_stack([design, target_values])
    _rank, condition_number, rank_criterion = rank_and_condition_number(augmented_design)
    if not rank_criterion:
        return condition_number, ""
    return condition_number, "contribution_design_rank"


def source_stage_continuous_columns(config: Any) -> tuple[str, ...]:
    raw_columns = require_config_value(config, "study2.source_stage.continuous_columns")
    columns = required_string_tuple(
        raw_columns,
        field_name="study2.source_stage.continuous_columns",
    )
    reject_raw_level2_artifact_columns(columns)
    return columns


def source_stage_categorical_columns(config: Any) -> tuple[str, ...]:
    raw_columns = require_config_value(config, "study2.source_stage.categorical_columns")
    return required_string_tuple(
        raw_columns,
        field_name="study2.source_stage.categorical_columns",
    )


def required_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of column names.")
    columns = tuple(str(column).strip() for column in value)
    if any(not column for column in columns):
        raise ValueError(f"{field_name} must not contain empty column names.")
    if not columns:
        raise ValueError(f"{field_name} must contain at least one column name.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{field_name} must not contain duplicate column names.")
    return columns


def reject_raw_level2_artifact_columns(columns: tuple[str, ...]) -> None:
    raw_columns = [column for column in columns if column in RAW_LEVEL2_ARTIFACT_COLUMNS]
    if not raw_columns:
        return

    replacements = {column: RAW_LEVEL2_ARTIFACT_COLUMNS[column] for column in raw_columns}
    raise ValueError(
        "Study 2 source-stage design requires HRF-weighted Study 1 Level 2 "
        "artifact covariates, got raw columns: "
        f"{json.dumps(replacements, sort_keys=True)}."
    )


def build_fixed_categorical_design(
    *,
    frame: pd.DataFrame,
    config: Any,
    column: str,
) -> pd.DataFrame:
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"Source-stage categorical column '{column}' contains missing values.")

    levels = fixed_categorical_levels(config, column=column)
    unknown_levels = unknown_categorical_levels(values, allowed_levels=levels)
    if unknown_levels:
        raise ValueError(
            f"Source-stage categorical column '{column}' contains levels outside "
            f"the fixed Study 1 coding: {unknown_levels}."
        )

    dummies = pd.DataFrame(index=frame.index)
    for level in levels[1:]:
        dummy_column = _categorical_level_column(column, level)
        membership = level_membership(values, level)
        if not np.any(membership):
            continue
        dummies[dummy_column] = membership.astype(float)
    return dummies.reset_index(drop=True)


def fixed_categorical_levels(config: Any, *, column: str) -> tuple[Any, ...]:
    raw_levels = require_config_value(
        config,
        f"study2.source_stage.fixed_categorical_levels.{column}",
    )
    if not isinstance(raw_levels, (list, tuple)):
        raise ValueError(
            "Study 2 source-stage fixed categorical levels must be configured for " f"'{column}'."
        )
    levels = tuple(raw_levels)
    if len(levels) < 2:
        raise ValueError(
            "Study 2 source-stage fixed categorical levels must include at least "
            f"two levels for '{column}'."
        )

    dummy_columns = [_categorical_level_column(column, level) for level in levels]
    if len(dummy_columns) != len(set(dummy_columns)):
        raise ValueError(
            "Study 2 source-stage fixed categorical levels must be unique after "
            f"Study 1 dummy-name normalization for '{column}'."
        )
    return levels


def unknown_categorical_levels(
    values: pd.Series,
    *,
    allowed_levels: tuple[Any, ...],
) -> tuple[Any, ...]:
    unknown: list[Any] = []
    for value in pd.unique(values):
        if not any(values_match_level(value, level) for level in allowed_levels):
            unknown.append(value)
    return tuple(unknown)


def level_membership(values: pd.Series, level: Any) -> np.ndarray:
    numeric_level = numeric_value(level)
    if numeric_level is not None:
        numeric_values = pd.to_numeric(values, errors="coerce")
        return np.isclose(numeric_values.to_numpy(dtype=float), numeric_level)
    return (values.astype(str).str.strip() == str(level).strip()).to_numpy(dtype=bool)


def values_match_level(value: Any, level: Any) -> bool:
    numeric = numeric_value(value)
    numeric_level = numeric_value(level)
    if numeric is not None and numeric_level is not None:
        return bool(np.isclose(numeric, numeric_level))
    return str(value).strip() == str(level).strip()


def numeric_value(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def rank_and_condition_number(
    design: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> tuple[int, float, str]:
    centered = design - np.mean(design, axis=0, keepdims=True)
    column_norms = np.linalg.norm(centered, axis=0)
    if np.any(column_norms <= 0.0):
        rank = int(np.sum(column_norms > 0.0)) + 1
        return rank, float("inf"), "source_stage_design_rank"

    scaled = centered / column_norms
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    if singular_values.size == 0:
        return 1, 1.0, ""
    max_singular = float(singular_values[0])
    if max_singular <= 0.0:
        return 1, float("inf"), "source_stage_design_rank"
    singular_ratios = singular_values / max_singular
    non_intercept_rank = int(np.sum(singular_ratios >= float(tolerance)))
    rank = non_intercept_rank + 1
    if non_intercept_rank < design.shape[1]:
        return rank, float("inf"), "source_stage_design_rank"
    condition_number = float(max_singular / singular_values[-1])
    return rank, condition_number, ""


def max_adjacent_band_vif(
    design: np.ndarray,
    *,
    design_columns: list[str],
    adjustment_columns: tuple[str, ...],
) -> float:
    return max(
        single_column_vif(design, design_columns=design_columns, column=column)
        for column in adjustment_columns
    )


def single_column_vif(
    design: np.ndarray,
    *,
    design_columns: list[str],
    column: str,
) -> float:
    column_index = design_columns.index(column)
    target = design[:, column_index]
    other_columns = np.delete(design, column_index, axis=1)
    predictors = np.column_stack([np.ones(len(target), dtype=float), other_columns])
    coefficients, *_ = np.linalg.lstsq(predictors, target, rcond=None)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        prediction = predictors @ coefficients
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot <= 0.0:
        return float("inf")
    ss_res = float(np.sum((target - prediction) ** 2))
    r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    residual_fraction = 1.0 - r_squared
    if residual_fraction <= 1e-12:
        return float("inf")
    return float(1.0 / residual_fraction)


__all__ = [
    "band_adjustment_columns",
    "band_standardized_column",
    "build_source_stage_design",
    "combined_score_column",
    "contribution_bands",
    "contribution_stability_condition_number",
    "max_adjacent_band_vif",
    "rank_and_condition_number",
    "source_stage_contribution_values",
    "source_stage_required_columns",
]
