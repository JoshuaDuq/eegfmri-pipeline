"""Study 2 source-stage subject inclusion checks.

The primary path associates per-band source power with the single combined
NPS-predictive score, adjusting only for the Study 1 Level 2 nuisance design.
The secondary band-unique path mutually adjusts each band's contribution score
against the other bands and reports the resulting collinearity criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from studies.pain_study.study2.source_stage_design import (
    band_adjustment_columns,
    band_standardized_column,
    build_source_stage_design,
    combined_score_column,
    contribution_stability_condition_number,
    max_adjacent_band_vif as compute_max_adjacent_band_vif,
    rank_and_condition_number,
    source_stage_contribution_values,
    source_stage_required_columns,
)
from studies.pain_study.study2.validation import (
    require_config_float,
    require_config_int,
    require_config_string,
)
from studies.pain_study.study2.source_stage_trials import (
    permutation_valid_source_blocks,
    permutation_valid_source_blocks_with_rows,
)


@dataclass(frozen=True)
class SourceStageSubjectQC:
    subject_id: str
    band: str
    source_stage_criteria_met: bool
    retained_trials: int
    valid_blocks: int
    design_rank: int
    residual_degrees_of_freedom: int
    condition_number: float
    max_adjacent_band_vif: float
    unmet_criteria: tuple[str, ...]


@dataclass(frozen=True)
class SourceStageCohortQC:
    band: str
    confirmatory_cohort_criteria_met: bool
    feasibility_cohort_criteria_met: bool
    n_subjects: int
    n_source_valid_subjects: int
    min_source_valid_subjects: int
    min_feasibility_subjects: int
    collinearity_failure_fraction: float
    max_collinearity_failure_fraction: float
    unmet_criteria: tuple[str, ...]


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
        target_column=combined_score_column(config),
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
        target_column=band_standardized_column(config, band_name),
        adjustment_columns=band_adjustment_columns(config, band_name),
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
        target_column=combined_score_column(config),
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
        target_column=band_standardized_column(config, band_name),
        adjustment_columns=band_adjustment_columns(config, band_name),
    )


def _evaluate_source_stage_cohort(
    frame: pd.DataFrame,
    *,
    config: Any,
    band_label: str,
    subject_evaluator: Callable[[pd.DataFrame], SourceStageSubjectQC],
) -> tuple[pd.DataFrame, SourceStageCohortQC]:
    subject_column = require_config_string(config, "study2.contributions.subject_column")
    _require_columns(frame, (subject_column,))
    qc_records: list[dict[str, Any]] = []
    for subject_id, subject_frame in frame.groupby(subject_column, sort=True):
        qc = subject_evaluator(subject_frame.reset_index(drop=True))
        qc_records.append(
            {
                "subject_id": str(subject_id),
                "band": qc.band,
                "source_stage_criteria_met": qc.source_stage_criteria_met,
                "retained_trials": qc.retained_trials,
                "valid_blocks": qc.valid_blocks,
                "design_rank": qc.design_rank,
                "residual_degrees_of_freedom": qc.residual_degrees_of_freedom,
                "condition_number": qc.condition_number,
                "max_adjacent_band_vif": qc.max_adjacent_band_vif,
                "unmet_criteria": ";".join(qc.unmet_criteria),
            }
        )

    qc_frame = pd.DataFrame(qc_records)
    n_subjects = int(len(qc_frame))
    n_source_valid_subjects = (
        int(qc_frame["source_stage_criteria_met"].sum()) if n_subjects else 0
    )
    min_source_valid_subjects = require_config_int(
        config,
        "study2.source_stage.min_source_valid_subjects",
    )
    min_feasibility_subjects = require_config_int(
        config,
        "study2.source_stage.min_feasibility_subjects",
    )
    max_collinearity_failure_fraction = require_config_float(
        config,
        "study2.source_stage.max_collinearity_failure_fraction",
    )
    collinearity_failure_fraction = _collinearity_failure_fraction(qc_frame)

    unmet_criteria: list[str] = []
    feasibility_criteria_met = n_source_valid_subjects >= min_feasibility_subjects
    confirmatory_criteria_met = feasibility_criteria_met
    if not feasibility_criteria_met:
        confirmatory_criteria_met = False
        unmet_criteria.append("min_feasibility_subjects")
    elif n_source_valid_subjects < min_source_valid_subjects:
        confirmatory_criteria_met = False
        unmet_criteria.append("min_source_valid_subjects")
    elif collinearity_failure_fraction > max_collinearity_failure_fraction:
        confirmatory_criteria_met = False
        unmet_criteria.append("max_collinearity_failure_fraction")

    status = SourceStageCohortQC(
        band=str(band_label).strip().lower(),
        confirmatory_cohort_criteria_met=confirmatory_criteria_met,
        feasibility_cohort_criteria_met=feasibility_criteria_met,
        n_subjects=n_subjects,
        n_source_valid_subjects=n_source_valid_subjects,
        min_source_valid_subjects=min_source_valid_subjects,
        min_feasibility_subjects=min_feasibility_subjects,
        collinearity_failure_fraction=collinearity_failure_fraction,
        max_collinearity_failure_fraction=max_collinearity_failure_fraction,
        unmet_criteria=tuple(unmet_criteria),
    )
    return qc_frame, status


def _collinearity_failure_fraction(qc_frame: pd.DataFrame) -> float:
    if qc_frame.empty:
        return 0.0

    collinearity_failures = _collinearity_failure_mask(qc_frame)
    otherwise_valid = qc_frame["source_stage_criteria_met"].astype(bool) | collinearity_failures
    denominator = int(otherwise_valid.sum())
    if denominator == 0:
        return 0.0
    return float(collinearity_failures.sum() / denominator)


def _collinearity_failure_mask(qc_frame: pd.DataFrame) -> pd.Series:
    return (
        qc_frame["unmet_criteria"]
        .astype(str)
        .str.contains(
            (
                "source_stage_design_rank|contribution_design_rank|"
                "max_condition_number|max_adjacent_band_vif"
            ),
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
    subject_column = require_config_string(config, "study2.contributions.subject_column")
    _require_columns(frame, (subject_column,))
    subject_ids = sorted({str(value) for value in frame[subject_column].astype(str)})
    if len(subject_ids) != 1:
        raise ValueError(
            "Study 2 source-stage subject QC expects exactly one subject, " f"got {subject_ids}."
        )
    subject_id = subject_ids[0]

    required_columns = source_stage_required_columns(
        config,
        target_column=target_column,
        adjustment_columns=adjustment_columns,
    )
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 2 source-stage table is missing columns: {missing}.")

    valid_frame = permutation_valid_source_blocks(frame)
    retained_trials = int(len(valid_frame))
    valid_blocks = int(valid_frame["block"].nunique()) if not valid_frame.empty else 0
    min_blocks = require_config_int(
        config,
        "study2.source_stage.min_valid_blocks_per_subject",
    )
    min_trials = require_config_int(
        config,
        "study2.source_stage.min_retained_trials_per_subject",
    )
    if valid_blocks < min_blocks or retained_trials < min_trials:
        unmet_count_criteria: list[str] = []
        if valid_blocks < min_blocks:
            unmet_count_criteria.append("min_valid_blocks_per_subject")
        if retained_trials < min_trials:
            unmet_count_criteria.append("min_retained_trials_per_subject")
        return _source_stage_qc_with_unmet_criteria(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            unmet_criteria=tuple(unmet_count_criteria),
        )

    design, design_columns = build_source_stage_design(
        valid_frame,
        config=config,
        adjustment_columns=adjustment_columns,
    )
    target_values = source_stage_contribution_values(valid_frame, column=target_column)
    rank, condition_number, rank_criterion = rank_and_condition_number(design)
    residual_degrees_of_freedom = retained_trials - rank
    if rank_criterion:
        return _source_stage_qc_with_unmet_criteria(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=condition_number,
            unmet_criteria=(rank_criterion,),
        )

    contribution_condition_number, contribution_rank_criterion = (
        contribution_stability_condition_number(
            design=design,
            target_values=target_values,
        )
    )
    if contribution_rank_criterion:
        return _source_stage_qc_with_unmet_criteria(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=contribution_condition_number,
            unmet_criteria=(contribution_rank_criterion,),
        )

    min_residual_df = require_config_int(
        config,
        "study2.source_stage.min_residual_degrees_of_freedom",
    )
    if residual_degrees_of_freedom < min_residual_df:
        return _source_stage_qc_with_unmet_criteria(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=condition_number,
            unmet_criteria=("min_residual_degrees_of_freedom",),
        )

    max_condition_number = require_config_float(
        config,
        "study2.source_stage.max_condition_number",
    )
    if contribution_condition_number > max_condition_number:
        return _source_stage_qc_with_unmet_criteria(
            subject_id=subject_id,
            band=band_label,
            retained_trials=retained_trials,
            valid_blocks=valid_blocks,
            design_rank=rank,
            residual_degrees_of_freedom=residual_degrees_of_freedom,
            condition_number=contribution_condition_number,
            unmet_criteria=("max_condition_number",),
        )

    max_adjacent_band_vif = float("nan")
    if adjustment_columns:
        max_adjacent_band_vif = compute_max_adjacent_band_vif(
            design,
            design_columns=design_columns,
            adjustment_columns=adjustment_columns,
        )
        max_vif = require_config_float(
            config,
            "study2.source_stage.max_opposite_band_vif",
        )
        if max_adjacent_band_vif > max_vif:
            return _source_stage_qc_with_unmet_criteria(
                subject_id=subject_id,
                band=band_label,
                retained_trials=retained_trials,
                valid_blocks=valid_blocks,
                design_rank=rank,
                residual_degrees_of_freedom=residual_degrees_of_freedom,
                condition_number=contribution_condition_number,
                max_adjacent_band_vif=max_adjacent_band_vif,
                unmet_criteria=("max_adjacent_band_vif",),
            )

    return SourceStageSubjectQC(
        subject_id=subject_id,
        band=band_label,
        source_stage_criteria_met=True,
        retained_trials=retained_trials,
        valid_blocks=valid_blocks,
        design_rank=rank,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
        condition_number=contribution_condition_number,
        max_adjacent_band_vif=max_adjacent_band_vif,
        unmet_criteria=(),
    )


def _prepare_source_stage_association_inputs(
    frame: pd.DataFrame,
    *,
    config: Any,
    qc: SourceStageSubjectQC,
    target_column: str,
    adjustment_columns: tuple[str, ...],
) -> SourceStageAssociationInputs:
    if not qc.source_stage_criteria_met:
        return _empty_source_stage_association_inputs(qc)

    valid_frame, retained_row_indices = permutation_valid_source_blocks_with_rows(frame)
    design, design_columns = build_source_stage_design(
        valid_frame,
        config=config,
        adjustment_columns=adjustment_columns,
    )
    score = source_stage_contribution_values(valid_frame, column=target_column)
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


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 2 source-stage table is missing columns: {missing}.")


def _source_stage_qc_with_unmet_criteria(
    *,
    subject_id: str,
    band: str,
    unmet_criteria: tuple[str, ...],
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
        source_stage_criteria_met=False,
        retained_trials=retained_trials,
        valid_blocks=valid_blocks,
        design_rank=design_rank,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
        condition_number=condition_number,
        max_adjacent_band_vif=max_adjacent_band_vif,
        unmet_criteria=unmet_criteria,
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
