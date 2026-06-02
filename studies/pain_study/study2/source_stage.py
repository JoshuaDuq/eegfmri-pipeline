"""Study 2 source-stage subject eligibility checks.

The primary path associates per-band source power with the single combined
NPS-predictive score, adjusting only for the Study 1 Level 2 nuisance design.
The secondary band-unique path mutually adjusts each band's contribution score
against the other bands and gates on the resulting collinearity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.machine_learning.circular_shift import admissible_circular_shifts
from eeg_pipeline.utils.config.loader import get_config_value
from studies.pain_study.study1.targets import _categorical_level_column

RAW_LEVEL2_ARTIFACT_COLUMNS = {
    "framewise_displacement": "hrf_weighted_framewise_displacement",
    "std_dvars": "hrf_weighted_std_dvars",
    "fp1_fp2_high_frequency_power": "hrf_weighted_fp1_fp2_high_frequency_power",
}
PERMUTATION_STRUCTURE_COLUMNS = ("block", "trial_index")
BAND_STANDARDIZED_COLUMN_KEYS = {
    "alpha": ("study2.contributions.alpha_standardized_column", "eta_alpha_z"),
    "beta": ("study2.contributions.beta_standardized_column", "eta_beta_z"),
    "gamma": ("study2.contributions.gamma_standardized_column", "eta_gamma_z"),
}


@dataclass(frozen=True)
class SourceStageSubjectQC:
    subject_id: str
    band: str
    eligible: bool
    retained_trials: int
    valid_blocks: int
    design_rank: int
    residual_degrees_of_freedom: int
    condition_number: float
    max_adjacent_band_vif: float
    reason: str


@dataclass(frozen=True)
class SourceStageCohortQC:
    band: str
    confirmatory_eligible: bool
    feasibility_eligible: bool
    n_subjects: int
    n_source_valid_subjects: int
    min_source_valid_subjects: int
    min_feasibility_subjects: int
    collinearity_failure_fraction: float
    max_collinearity_failure_fraction: float
    reason: str


@dataclass(frozen=True)
class SourceStageAssociationInputs:
    qc: SourceStageSubjectQC
    retained_row_indices: np.ndarray
    score: np.ndarray
    design: np.ndarray
    design_columns: tuple[str, ...]


def evaluate_source_stage_subject(
    frame: pd.DataFrame,
    *,
    config: Any,
) -> SourceStageSubjectQC:
    """Evaluate primary source-stage eligibility against the combined score."""
    return _evaluate_source_stage_subject(
        frame,
        config=config,
        band_label="combined",
        target_column=_combined_score_column(config),
        adjustment_columns=(),
    )


def evaluate_band_unique_source_stage_subject(
    frame: pd.DataFrame,
    *,
    band: str,
    config: Any,
) -> SourceStageSubjectQC:
    """Evaluate secondary band-unique eligibility with mutual band adjustment."""
    band_name = str(band).strip().lower()
    return _evaluate_source_stage_subject(
        frame,
        config=config,
        band_label=band_name,
        target_column=_band_standardized_column(config, band_name),
        adjustment_columns=_band_adjustment_columns(config, band_name),
    )


def evaluate_source_stage_cohort(
    frame: pd.DataFrame,
    *,
    config: Any,
) -> tuple[pd.DataFrame, SourceStageCohortQC]:
    """Aggregate primary source-stage eligibility across the cohort."""
    return _evaluate_source_stage_cohort(
        frame,
        config=config,
        band_label="combined",
        subject_evaluator=lambda subject_frame: evaluate_source_stage_subject(
            subject_frame,
            config=config,
        ),
    )


def evaluate_band_unique_source_stage_cohort(
    frame: pd.DataFrame,
    *,
    band: str,
    config: Any,
) -> tuple[pd.DataFrame, SourceStageCohortQC]:
    """Aggregate secondary band-unique eligibility across the cohort."""
    band_name = str(band).strip().lower()
    return _evaluate_source_stage_cohort(
        frame,
        config=config,
        band_label=band_name,
        subject_evaluator=lambda subject_frame: evaluate_band_unique_source_stage_subject(
            subject_frame,
            band=band_name,
            config=config,
        ),
    )


def prepare_source_stage_association_inputs(
    frame: pd.DataFrame,
    *,
    config: Any,
) -> SourceStageAssociationInputs:
    """Prepare retained trials, score, and nuisance design for the primary map."""
    return _prepare_source_stage_association_inputs(
        frame,
        config=config,
        qc=evaluate_source_stage_subject(frame, config=config),
        target_column=_combined_score_column(config),
        adjustment_columns=(),
    )


def prepare_band_unique_source_stage_association_inputs(
    frame: pd.DataFrame,
    *,
    band: str,
    config: Any,
) -> SourceStageAssociationInputs:
    """Prepare retained trials and design for a band-unique source map."""
    band_name = str(band).strip().lower()
    return _prepare_source_stage_association_inputs(
        frame,
        config=config,
        qc=evaluate_band_unique_source_stage_subject(
            frame,
            band=band_name,
            config=config,
        ),
        target_column=_band_standardized_column(config, band_name),
        adjustment_columns=_band_adjustment_columns(config, band_name),
    )


def _evaluate_source_stage_cohort(
    frame: pd.DataFrame,
    *,
    config: Any,
    band_label: str,
    subject_evaluator: Callable[[pd.DataFrame], SourceStageSubjectQC],
) -> tuple[pd.DataFrame, SourceStageCohortQC]:
    subject_column = str(
        get_config_value(config, "study2.contributions.subject_column", "subject_id")
    )
    _require_columns(frame, (subject_column,))
    qc_records: list[dict[str, Any]] = []
    for subject_id, subject_frame in frame.groupby(subject_column, sort=True):
        qc = subject_evaluator(subject_frame.reset_index(drop=True))
        qc_records.append(
            {
                "subject_id": str(subject_id),
                "band": qc.band,
                "eligible": qc.eligible,
                "retained_trials": qc.retained_trials,
                "valid_blocks": qc.valid_blocks,
                "design_rank": qc.design_rank,
                "residual_degrees_of_freedom": qc.residual_degrees_of_freedom,
                "condition_number": qc.condition_number,
                "max_adjacent_band_vif": qc.max_adjacent_band_vif,
                "reason": qc.reason,
            }
        )

    qc_frame = pd.DataFrame(qc_records)
    n_subjects = int(len(qc_frame))
    n_source_valid_subjects = int(qc_frame["eligible"].sum()) if n_subjects else 0
    min_source_valid_subjects = int(
        get_config_value(config, "study2.source_stage.min_source_valid_subjects", 30)
    )
    min_feasibility_subjects = int(
        get_config_value(config, "study2.source_stage.min_feasibility_subjects", 20)
    )
    max_collinearity_failure_fraction = float(
        get_config_value(
            config,
            "study2.source_stage.max_collinearity_failure_fraction",
            0.20,
        )
    )
    collinearity_failure_fraction = _collinearity_failure_fraction(qc_frame)

    reason = ""
    feasibility_eligible = n_source_valid_subjects >= min_feasibility_subjects
    confirmatory_eligible = feasibility_eligible
    if not feasibility_eligible:
        confirmatory_eligible = False
        reason = (
            f"Study 2 source-stage cohort has fewer than {min_feasibility_subjects} "
            f"feasibility source-valid subjects: {n_source_valid_subjects}."
        )
    elif n_source_valid_subjects < min_source_valid_subjects:
        confirmatory_eligible = False
        reason = (
            f"Study 2 source-stage cohort has fewer than {min_source_valid_subjects} "
            f"confirmatory source-valid subjects: {n_source_valid_subjects}."
        )
    elif collinearity_failure_fraction > max_collinearity_failure_fraction:
        confirmatory_eligible = False
        reason = (
            "Study 2 source-stage collinearity failure fraction exceeds threshold: "
            f"{collinearity_failure_fraction:.6g}."
        )

    status = SourceStageCohortQC(
        band=str(band_label).strip().lower(),
        confirmatory_eligible=confirmatory_eligible,
        feasibility_eligible=feasibility_eligible,
        n_subjects=n_subjects,
        n_source_valid_subjects=n_source_valid_subjects,
        min_source_valid_subjects=min_source_valid_subjects,
        min_feasibility_subjects=min_feasibility_subjects,
        collinearity_failure_fraction=collinearity_failure_fraction,
        max_collinearity_failure_fraction=max_collinearity_failure_fraction,
        reason=reason,
    )
    return qc_frame, status


def _collinearity_failure_fraction(qc_frame: pd.DataFrame) -> float:
    if qc_frame.empty:
        return 0.0

    collinearity_failures = _collinearity_failure_mask(qc_frame)
    otherwise_valid = qc_frame["eligible"].astype(bool) | collinearity_failures
    denominator = int(otherwise_valid.sum())
    if denominator == 0:
        return 0.0
    return float(collinearity_failures.sum() / denominator)


def _collinearity_failure_mask(qc_frame: pd.DataFrame) -> pd.Series:
    return (
        qc_frame["reason"]
        .astype(str)
        .str.contains(
            "condition number|contribution design|adjacent-band VIF",
            regex=True,
        )
    )


def _evaluate_source_stage_subject(
    frame: pd.DataFrame,
    *,
    config: Any,
    band_label: str,
    target_column: str,
    adjustment_columns: tuple[str, ...],
) -> SourceStageSubjectQC:
    subject_column = str(
        get_config_value(config, "study2.contributions.subject_column", "subject_id")
    )
    _require_columns(frame, (subject_column,))
    subject_ids = sorted({str(value) for value in frame[subject_column].astype(str)})
    if len(subject_ids) != 1:
        raise ValueError(
            "Study 2 source-stage subject QC expects exactly one subject, " f"got {subject_ids}."
        )
    subject_id = subject_ids[0]

    required_columns = _source_stage_required_columns(
        config,
        target_column=target_column,
        adjustment_columns=adjustment_columns,
    )
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        return _ineligible_qc(
            subject_id=subject_id,
            band=band_label,
            reason=f"Missing source-stage columns: {missing}.",
        )

    valid_frame = _permutation_valid_source_blocks(frame)
    retained_trials = int(len(valid_frame))
    valid_blocks = int(valid_frame["block"].nunique()) if not valid_frame.empty else 0
    min_blocks = int(
        get_config_value(config, "study2.source_stage.min_valid_blocks_per_subject", 3)
    )
    min_trials = int(
        get_config_value(config, "study2.source_stage.min_retained_trials_per_subject", 25)
    )
    if valid_blocks < min_blocks or retained_trials < min_trials:
        return _ineligible_qc(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            reason=(
                "Source-stage subject failed permutation-valid block/trial gates: "
                f"valid_blocks={valid_blocks}, retained_trials={retained_trials}."
            ),
        )

    design, design_columns = _build_source_stage_design(
        valid_frame,
        config=config,
        adjustment_columns=adjustment_columns,
    )
    target_values = _source_stage_contribution_values(valid_frame, column=target_column)
    rank, condition_number, rank_reason = _rank_and_condition_number(design)
    residual_degrees_of_freedom = retained_trials - rank
    if rank_reason:
        return _ineligible_qc(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=condition_number,
            reason=rank_reason,
        )

    contribution_condition_number, contribution_rank_reason = (
        _contribution_stability_condition_number(
            design=design,
            target_values=target_values,
        )
    )
    if contribution_rank_reason:
        return _ineligible_qc(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=contribution_condition_number,
            reason=contribution_rank_reason,
        )

    min_residual_df = int(
        get_config_value(config, "study2.source_stage.min_residual_degrees_of_freedom", 15)
    )
    if residual_degrees_of_freedom < min_residual_df:
        return _ineligible_qc(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=condition_number,
            reason=(
                "Source-stage subject failed residual degrees-of-freedom gate: "
                f"residual_df={residual_degrees_of_freedom}."
            ),
        )

    max_condition_number = float(
        get_config_value(config, "study2.source_stage.max_condition_number", 100)
    )
    if contribution_condition_number > max_condition_number:
        return _ineligible_qc(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=contribution_condition_number,
            reason=(
                "Source-stage contribution design condition number exceeds threshold: "
                f"condition_number={contribution_condition_number:.6g}."
            ),
        )

    max_adjacent_band_vif = float("nan")
    if adjustment_columns:
        max_adjacent_band_vif = _max_adjacent_band_vif(
            design,
            design_columns=design_columns,
            adjustment_columns=adjustment_columns,
        )
        max_vif = float(get_config_value(config, "study2.source_stage.max_opposite_band_vif", 5))
        if max_adjacent_band_vif > max_vif:
            return _ineligible_qc(
                subject_id=subject_id,
                band=band_label,
                retained_trials=retained_trials,
                valid_blocks=valid_blocks,
                design_rank=rank,
                residual_degrees_of_freedom=residual_degrees_of_freedom,
                condition_number=contribution_condition_number,
                max_adjacent_band_vif=max_adjacent_band_vif,
                reason=(
                    "Source-stage adjacent-band VIF exceeds threshold: "
                    f"max_adjacent_band_vif={max_adjacent_band_vif:.6g}."
                ),
            )

    return SourceStageSubjectQC(
        subject_id=subject_id,
        band=band_label,
        eligible=True,
        retained_trials=retained_trials,
        valid_blocks=valid_blocks,
        design_rank=rank,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
        condition_number=contribution_condition_number,
        max_adjacent_band_vif=max_adjacent_band_vif,
        reason="",
    )


def _prepare_source_stage_association_inputs(
    frame: pd.DataFrame,
    *,
    config: Any,
    qc: SourceStageSubjectQC,
    target_column: str,
    adjustment_columns: tuple[str, ...],
) -> SourceStageAssociationInputs:
    if not qc.eligible:
        return _empty_source_stage_association_inputs(qc)

    valid_frame, retained_row_indices = _permutation_valid_source_blocks_with_rows(frame)
    design, design_columns = _build_source_stage_design(
        valid_frame,
        config=config,
        adjustment_columns=adjustment_columns,
    )
    score = _source_stage_contribution_values(valid_frame, column=target_column)
    return SourceStageAssociationInputs(
        qc=qc,
        retained_row_indices=retained_row_indices,
        score=score,
        design=design,
        design_columns=tuple(design_columns),
    )


def _empty_source_stage_association_inputs(
    qc: SourceStageSubjectQC,
) -> SourceStageAssociationInputs:
    return SourceStageAssociationInputs(
        qc=qc,
        retained_row_indices=np.empty(0, dtype=int),
        score=np.empty(0, dtype=float),
        design=np.empty((0, 0), dtype=float),
        design_columns=(),
    )


def _source_stage_required_columns(
    config: Any,
    *,
    target_column: str,
    adjustment_columns: tuple[str, ...],
) -> tuple[str, ...]:
    columns = (
        *PERMUTATION_STRUCTURE_COLUMNS,
        *_source_stage_continuous_columns(config),
        *_source_stage_categorical_columns(config),
        target_column,
        *adjustment_columns,
    )
    return tuple(dict.fromkeys(columns))


def _combined_score_column(config: Any) -> str:
    return str(
        get_config_value(
            config,
            "study2.contributions.combined_standardized_column",
            "eta_combined_z",
        )
    )


def _band_standardized_column(config: Any, band: str) -> str:
    entry = BAND_STANDARDIZED_COLUMN_KEYS.get(band)
    if entry is None:
        raise ValueError("Study 2 source-stage band must be 'alpha', 'beta', or 'gamma'.")
    config_key, default_column = entry
    return str(get_config_value(config, config_key, default_column))


def _contribution_bands(config: Any) -> tuple[str, ...]:
    raw_bands = get_config_value(
        config,
        "study2.confirmatory.study1_cell.contribution_bands",
        [],
    )
    bands = _required_string_tuple(
        raw_bands,
        field_name="study2.confirmatory.study1_cell.contribution_bands",
    )
    return tuple(band.lower() for band in bands)


def _band_adjustment_columns(config: Any, band: str) -> tuple[str, ...]:
    bands = _contribution_bands(config)
    if band not in bands:
        raise ValueError(
            f"Study 2 source-stage band '{band}' is not in the configured "
            f"contribution bands {bands}."
        )
    return tuple(_band_standardized_column(config, other) for other in bands if other != band)


def _permutation_valid_source_blocks_with_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    row_index_column = "__source_stage_row_index"
    if row_index_column in frame.columns:
        raise ValueError(f"Study 2 source-stage table uses reserved column '{row_index_column}'.")

    working = frame.reset_index(drop=True).copy()
    working[row_index_column] = np.arange(len(working), dtype=int)
    valid_frame = _permutation_valid_source_blocks(working)
    retained_row_indices = valid_frame.pop(row_index_column).to_numpy(dtype=int)
    return valid_frame, retained_row_indices


def _permutation_valid_source_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    blocks = pd.to_numeric(frame["block"], errors="coerce")
    trial_indices = pd.to_numeric(frame["trial_index"], errors="coerce")
    if blocks.isna().any() or trial_indices.isna().any():
        raise ValueError("Source-stage block and trial_index columns must be finite.")

    valid_indices: list[int] = []
    working = frame.copy()
    working["block"] = blocks.to_numpy(dtype=int)
    working["trial_index"] = trial_indices.to_numpy(dtype=int)
    for _block_id, block_frame in working.groupby("block", sort=True):
        ordered = block_frame.sort_values("trial_index")
        admissible = admissible_circular_shifts(
            ordered["trial_index"].to_numpy(dtype=int),
        )
        if admissible:
            valid_indices.extend(ordered.index.tolist())

    if not valid_indices:
        return working.iloc[0:0].copy()
    return working.loc[sorted(valid_indices)].reset_index(drop=True)


def _build_source_stage_design(
    frame: pd.DataFrame,
    *,
    config: Any,
    adjustment_columns: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    design_parts: list[pd.DataFrame] = []
    continuous_columns = _source_stage_continuous_columns(config)
    categorical_columns = _source_stage_categorical_columns(config)
    for column in continuous_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Source-stage continuous column '{column}' contains non-finite values."
            )
        design_parts.append(pd.DataFrame({column: values.to_numpy(dtype=float)}))

    for column in categorical_columns:
        design_parts.append(
            _build_fixed_categorical_design(
                frame=frame,
                config=config,
                column=column,
            )
        )

    for column in adjustment_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Source-stage adjacent-band contribution '{column}' " "contains non-finite values."
            )
        design_parts.append(pd.DataFrame({column: values.to_numpy(dtype=float)}))

    design_frame = pd.concat(design_parts, axis=1)
    return design_frame.to_numpy(dtype=float), list(design_frame.columns)


def _source_stage_contribution_values(frame: pd.DataFrame, *, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Source-stage contribution column '{column}' contains non-finite values.")
    return values.to_numpy(dtype=float)


def _contribution_stability_condition_number(
    *,
    design: np.ndarray,
    target_values: np.ndarray,
) -> tuple[float, str]:
    augmented_design = np.column_stack([design, target_values])
    _rank, condition_number, rank_reason = _rank_and_condition_number(augmented_design)
    if not rank_reason:
        return condition_number, ""
    return (
        condition_number,
        rank_reason.replace(
            "Source-stage design",
            "Source-stage contribution design",
        ),
    )


def _source_stage_continuous_columns(config: Any) -> tuple[str, ...]:
    raw_columns = get_config_value(config, "study2.source_stage.continuous_columns", [])
    columns = _required_string_tuple(
        raw_columns,
        field_name="study2.source_stage.continuous_columns",
    )
    _reject_raw_level2_artifact_columns(columns)
    return columns


def _source_stage_categorical_columns(config: Any) -> tuple[str, ...]:
    raw_columns = get_config_value(config, "study2.source_stage.categorical_columns", [])
    return _required_string_tuple(
        raw_columns,
        field_name="study2.source_stage.categorical_columns",
    )


def _required_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of column names.")
    columns = tuple(str(column).strip() for column in value if str(column).strip())
    if len(columns) != len(set(columns)):
        raise ValueError(f"{field_name} must not contain duplicate column names.")
    return columns


def _reject_raw_level2_artifact_columns(columns: tuple[str, ...]) -> None:
    raw_columns = [column for column in columns if column in RAW_LEVEL2_ARTIFACT_COLUMNS]
    if not raw_columns:
        return

    replacements = {column: RAW_LEVEL2_ARTIFACT_COLUMNS[column] for column in raw_columns}
    raise ValueError(
        "Study 2 source-stage design requires HRF-weighted Study 1 Level 2 "
        "artifact covariates, got raw columns: "
        f"{json.dumps(replacements, sort_keys=True)}."
    )


def _build_fixed_categorical_design(
    *,
    frame: pd.DataFrame,
    config: Any,
    column: str,
) -> pd.DataFrame:
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"Source-stage categorical column '{column}' contains missing values.")

    levels = _fixed_categorical_levels(config, column=column)
    unknown_levels = _unknown_categorical_levels(values, allowed_levels=levels)
    if unknown_levels:
        raise ValueError(
            f"Source-stage categorical column '{column}' contains levels outside "
            f"the fixed Study 1 coding: {unknown_levels}."
        )

    dummies = pd.DataFrame(index=frame.index)
    for level in levels[1:]:
        dummy_column = _categorical_level_column(column, level)
        dummies[dummy_column] = _level_membership(values, level).astype(float)
    return dummies.reset_index(drop=True)


def _fixed_categorical_levels(config: Any, *, column: str) -> tuple[Any, ...]:
    raw_levels = get_config_value(
        config,
        f"study2.source_stage.fixed_categorical_levels.{column}",
        None,
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


def _unknown_categorical_levels(
    values: pd.Series,
    *,
    allowed_levels: tuple[Any, ...],
) -> tuple[Any, ...]:
    unknown: list[Any] = []
    for value in pd.unique(values):
        if not any(_values_match_level(value, level) for level in allowed_levels):
            unknown.append(value)
    return tuple(unknown)


def _level_membership(values: pd.Series, level: Any) -> np.ndarray:
    numeric_level = _numeric_value(level)
    if numeric_level is not None:
        numeric_values = pd.to_numeric(values, errors="coerce")
        return np.isclose(numeric_values.to_numpy(dtype=float), numeric_level)
    return (values.astype(str).str.strip() == str(level).strip()).to_numpy(dtype=bool)


def _values_match_level(value: Any, level: Any) -> bool:
    numeric_value = _numeric_value(value)
    numeric_level = _numeric_value(level)
    if numeric_value is not None and numeric_level is not None:
        return bool(np.isclose(numeric_value, numeric_level))
    return str(value).strip() == str(level).strip()


def _numeric_value(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _rank_and_condition_number(
    design: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> tuple[int, float, str]:
    centered = design - np.mean(design, axis=0, keepdims=True)
    column_norms = np.linalg.norm(centered, axis=0)
    if np.any(column_norms <= 0.0):
        rank = int(np.sum(column_norms > 0.0)) + 1
        return rank, float("inf"), "Source-stage design is rank deficient."

    scaled = centered / column_norms
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    if singular_values.size == 0:
        return 1, 1.0, ""
    max_singular = float(singular_values[0])
    if max_singular <= 0.0:
        return 1, float("inf"), "Source-stage design is rank deficient."
    singular_ratios = singular_values / max_singular
    non_intercept_rank = int(np.sum(singular_ratios >= float(tolerance)))
    rank = non_intercept_rank + 1
    if non_intercept_rank < design.shape[1]:
        return rank, float("inf"), "Source-stage design is rank deficient."
    condition_number = float(max_singular / singular_values[-1])
    return rank, condition_number, ""


def _max_adjacent_band_vif(
    design: np.ndarray,
    *,
    design_columns: list[str],
    adjustment_columns: tuple[str, ...],
) -> float:
    return max(
        _single_column_vif(design, design_columns=design_columns, column=column)
        for column in adjustment_columns
    )


def _single_column_vif(
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
    prediction = predictors @ coefficients
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot <= 0.0:
        return float("inf")
    ss_res = float(np.sum((target - prediction) ** 2))
    r_squared = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    residual_fraction = 1.0 - r_squared
    if residual_fraction <= 1e-12:
        return float("inf")
    return float(1.0 / residual_fraction)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 2 source-stage table is missing columns: {missing}.")


def _ineligible_qc(
    *,
    subject_id: str,
    band: str,
    reason: str,
    retained_trials: int = 0,
    valid_blocks: int = 0,
    design_rank: int = 0,
    residual_degrees_of_freedom: int = 0,
    condition_number: float = float("nan"),
    max_adjacent_band_vif: float = float("nan"),
) -> SourceStageSubjectQC:
    return SourceStageSubjectQC(
        subject_id=subject_id,
        band=band,
        eligible=False,
        retained_trials=retained_trials,
        valid_blocks=valid_blocks,
        design_rank=design_rank,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
        condition_number=condition_number,
        max_adjacent_band_vif=max_adjacent_band_vif,
        reason=reason,
    )


__all__ = [
    "SourceStageAssociationInputs",
    "SourceStageCohortQC",
    "SourceStageSubjectQC",
    "evaluate_band_unique_source_stage_cohort",
    "evaluate_band_unique_source_stage_subject",
    "evaluate_source_stage_cohort",
    "evaluate_source_stage_subject",
    "prepare_band_unique_source_stage_association_inputs",
    "prepare_source_stage_association_inputs",
]
