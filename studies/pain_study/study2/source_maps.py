"""Study 2 subject-level source-map orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from studies.pain_study.study2.association import (
    SourcePowerAssociationMap,
    compute_source_power_association_map,
)
from studies.pain_study.study2.source_stage import (
    SourceStageAssociationInputs,
    SourceStageSubjectQC,
    prepare_band_unique_source_stage_association_inputs,
    prepare_source_stage_association_inputs,
)
from studies.pain_study.study2.validation import require_config_string


@dataclass(frozen=True)
class SubjectSourceAssociationResult:
    subject_id: str
    source_band: str
    qc: SourceStageSubjectQC
    association: SourcePowerAssociationMap | None


@dataclass(frozen=True)
class CohortSourceAssociationResult:
    source_band: str
    qc: pd.DataFrame
    subject_ids: tuple[str, ...]
    partial_r_maps: np.ndarray
    fisher_z_maps: np.ndarray


class SubjectMapFunction(Protocol):
    def __call__(
        self,
        frame: pd.DataFrame,
        source_power: np.ndarray,
        *,
        band: str,
        config: Any,
    ) -> SubjectSourceAssociationResult: ...


def compute_subject_source_association_map(
    frame: pd.DataFrame,
    source_power: np.ndarray,
    *,
    band: str,
    config: Any,
) -> SubjectSourceAssociationResult:
    """Compute a subject source map for one source-power band and combined score."""
    source_band = _normalized_band(band)
    source_arr = _validate_source_power(source_power, n_trials=len(frame))
    inputs = prepare_source_stage_association_inputs(frame, config=config)
    return _compute_prepared_source_association(
        source_power=source_arr,
        inputs=inputs,
        source_band=source_band,
    )


def compute_band_unique_subject_source_association_map(
    frame: pd.DataFrame,
    source_power: np.ndarray,
    *,
    band: str,
    config: Any,
) -> SubjectSourceAssociationResult:
    """Compute a subject source map for a band's unique contribution score."""
    source_band = _normalized_band(band)
    source_arr = _validate_source_power(source_power, n_trials=len(frame))
    inputs = prepare_band_unique_source_stage_association_inputs(
        frame,
        band=source_band,
        config=config,
    )
    return _compute_prepared_source_association(
        source_power=source_arr,
        inputs=inputs,
        source_band=source_band,
    )


def compute_cohort_source_association_maps(
    frame: pd.DataFrame,
    source_power_by_subject: Mapping[str, np.ndarray],
    *,
    band: str,
    config: Any,
) -> CohortSourceAssociationResult:
    """Compute primary source association maps for each source-valid cohort subject."""
    source_band = _normalized_band(band)
    return _compute_cohort_source_association_maps(
        frame,
        source_power_by_subject,
        source_band=source_band,
        config=config,
        subject_map_function=compute_subject_source_association_map,
    )


def compute_band_unique_cohort_source_association_maps(
    frame: pd.DataFrame,
    source_power_by_subject: Mapping[str, np.ndarray],
    *,
    band: str,
    config: Any,
) -> CohortSourceAssociationResult:
    """Compute band-unique source association maps for each source-valid subject."""
    source_band = _normalized_band(band)
    return _compute_cohort_source_association_maps(
        frame,
        source_power_by_subject,
        source_band=source_band,
        config=config,
        subject_map_function=compute_band_unique_subject_source_association_map,
    )


def _compute_prepared_source_association(
    *,
    source_power: np.ndarray,
    inputs: SourceStageAssociationInputs,
    source_band: str,
) -> SubjectSourceAssociationResult:
    if not inputs.qc.source_stage_criteria_met:
        return SubjectSourceAssociationResult(
            subject_id=inputs.qc.subject_id,
            source_band=source_band,
            qc=inputs.qc,
            association=None,
        )

    retained_source_power = source_power[inputs.retained_row_indices, :]
    association = compute_source_power_association_map(
        source_power=retained_source_power,
        score=inputs.score,
        design=inputs.design,
    )
    return SubjectSourceAssociationResult(
        subject_id=inputs.qc.subject_id,
        source_band=source_band,
        qc=inputs.qc,
        association=association,
    )


def _compute_cohort_source_association_maps(
    frame: pd.DataFrame,
    source_power_by_subject: Mapping[str, np.ndarray],
    *,
    source_band: str,
    config: Any,
    subject_map_function: SubjectMapFunction,
) -> CohortSourceAssociationResult:
    subject_column = _subject_column(config)
    subject_frames = _subject_frames(frame, subject_column=subject_column)
    source_maps = _source_power_mapping(
        source_power_by_subject,
        expected_subject_ids=tuple(subject_frames),
    )

    subject_ids: list[str] = []
    partial_r_maps: list[np.ndarray] = []
    fisher_z_maps: list[np.ndarray] = []
    qc_records: list[dict[str, Any]] = []
    n_vertices = _n_vertices(source_maps)
    for subject_id, subject_frame in subject_frames.items():
        aligned_source_power = _align_source_power_to_frame(
            source_maps[subject_id],
            subject_frame,
            config=config,
        )
        result = subject_map_function(
            subject_frame,
            aligned_source_power,
            band=source_band,
            config=config,
        )
        qc_records.append(_qc_record(result.qc))
        if result.association is None:
            continue

        _validate_vertex_count(result.association.partial_r, n_vertices=n_vertices)
        subject_ids.append(subject_id)
        partial_r_maps.append(result.association.partial_r)
        fisher_z_maps.append(result.association.fisher_z)

    return CohortSourceAssociationResult(
        source_band=source_band,
        qc=pd.DataFrame(qc_records),
        subject_ids=tuple(subject_ids),
        partial_r_maps=_stack_maps(partial_r_maps, n_vertices=n_vertices),
        fisher_z_maps=_stack_maps(fisher_z_maps, n_vertices=n_vertices),
    )


def _validate_source_power(source_power: np.ndarray, *, n_trials: int) -> np.ndarray:
    source_arr = np.asarray(source_power, dtype=float)
    if source_arr.ndim != 2:
        raise ValueError(f"Study 2 source_power must be 2D, got shape {source_arr.shape}.")
    if source_arr.shape[0] != n_trials:
        raise ValueError(
            "Study 2 source_power and source-stage frame must have the same number "
            f"of rows, got {source_arr.shape[0]} and {n_trials}."
        )
    if not np.all(np.isfinite(source_arr)):
        raise ValueError("Study 2 source_power contains non-finite values.")
    return source_arr


def _align_source_power_to_frame(
    source_power: np.ndarray,
    frame: pd.DataFrame,
    *,
    config: Any,
) -> np.ndarray:
    source_arr = np.asarray(source_power, dtype=float)
    if source_arr.ndim != 2:
        raise ValueError(f"Study 2 source_power must be 2D, got shape {source_arr.shape}.")

    trial_column = require_config_string(config, "study2.contributions.trial_column")
    if trial_column not in frame.columns:
        raise ValueError(
            f"Study 2 source-stage frame is missing required trial column '{trial_column}'."
        )

    trial_ids = _source_power_trial_indices(frame[trial_column], n_source_rows=source_arr.shape[0])
    return _validate_source_power(source_arr[trial_ids, :], n_trials=len(frame))


def _source_power_trial_indices(trial_ids: pd.Series, *, n_source_rows: int) -> np.ndarray:
    values = pd.to_numeric(trial_ids, errors="raise").to_numpy(dtype=float)
    if values.ndim != 1:
        raise ValueError("Study 2 source-stage trial IDs must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Study 2 source-stage trial IDs must be finite.")

    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        raise ValueError("Study 2 source-stage trial IDs must be integers.")
    indices = rounded.astype(int) - 1
    if np.any(indices < 0) or np.any(indices >= n_source_rows):
        raise ValueError(
            "Study 2 source-stage trial IDs exceed source_power row bounds: "
            f"min={int(indices.min()) + 1}, max={int(indices.max()) + 1}, "
            f"n_source_rows={n_source_rows}."
        )
    return indices


def _subject_frames(
    frame: pd.DataFrame,
    *,
    subject_column: str,
) -> dict[str, pd.DataFrame]:
    if subject_column not in frame.columns:
        raise ValueError(f"Study 2 source-stage frame is missing '{subject_column}'.")

    subject_frames = {
        str(subject_id): subject_frame.reset_index(drop=True)
        for subject_id, subject_frame in frame.groupby(subject_column, sort=True)
    }
    if not subject_frames:
        raise ValueError("Study 2 cohort source mapping requires at least one subject.")
    return subject_frames


def _source_power_mapping(
    source_power_by_subject: Mapping[str, np.ndarray],
    *,
    expected_subject_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    source_maps = {
        str(subject_id): source_power
        for subject_id, source_power in source_power_by_subject.items()
    }
    if len(source_maps) != len(source_power_by_subject):
        raise ValueError("Study 2 source_power subject IDs must be unique as strings.")

    expected = set(expected_subject_ids)
    observed = set(source_maps)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            "Study 2 source mapping requires one source_power array per subject; "
            f"missing source_power={missing}, extra source_power={extra}."
        )
    return source_maps


def _n_vertices(source_maps: Mapping[str, np.ndarray]) -> int:
    n_vertices: int | None = None
    for subject_id, source_power in source_maps.items():
        source_arr = np.asarray(source_power, dtype=float)
        if source_arr.ndim != 2:
            raise ValueError(
                "Study 2 source_power arrays must be 2D, "
                f"got shape {source_arr.shape} for {subject_id}."
            )
        if n_vertices is None:
            n_vertices = int(source_arr.shape[1])
            continue
        if source_arr.shape[1] != n_vertices:
            raise ValueError("Study 2 source_power arrays must share one vertex count.")

    if n_vertices is None:
        raise ValueError("Study 2 source_power mapping must not be empty.")
    return n_vertices


def _validate_vertex_count(values: np.ndarray, *, n_vertices: int) -> None:
    if values.shape != (n_vertices,):
        raise ValueError("Study 2 source association maps must have the configured vertex count.")


def _stack_maps(maps: list[np.ndarray], *, n_vertices: int) -> np.ndarray:
    if not maps:
        return np.empty((0, n_vertices), dtype=float)
    return np.vstack(maps)


def _qc_record(qc: SourceStageSubjectQC) -> dict[str, Any]:
    return {
        "subject_id": qc.subject_id,
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


def _subject_column(config: Any) -> str:
    return require_config_string(config, "study2.contributions.subject_column")


def _normalized_band(band: str) -> str:
    band_name = str(band).strip().lower()
    if not band_name:
        raise ValueError("Study 2 source band must be a non-empty string.")
    return band_name


__all__ = [
    "CohortSourceAssociationResult",
    "SubjectSourceAssociationResult",
    "compute_band_unique_cohort_source_association_maps",
    "compute_band_unique_subject_source_association_map",
    "compute_cohort_source_association_maps",
    "compute_subject_source_association_map",
]
