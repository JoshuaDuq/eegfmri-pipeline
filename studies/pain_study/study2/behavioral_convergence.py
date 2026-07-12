"""Behavioral convergence tests for Study 2 source-pattern expression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from studies.pain_study.study2.statistics import plus_one_p_value, standardized_residual
from studies.pain_study.study2.validation import require_config_int


@dataclass(frozen=True)
class BehavioralConvergenceResult:
    mean_beta: float
    p_value: float
    null_distribution: np.ndarray
    subject_results: pd.DataFrame
    n_subjects: int


def compute_behavioral_convergence(
    frame: pd.DataFrame,
    *,
    expression_column: str,
    rating_column: str,
    design_columns: tuple[str, ...],
    config: Any,
    n_permutations: int | None = None,
    random_state: int = 0,
    subject_column: str = "subject_id",
    run_column: str = "run",
    trial_column: str = "trial_id",
) -> BehavioralConvergenceResult:
    _require_columns(
        frame,
        (
            subject_column,
            run_column,
            trial_column,
            expression_column,
            rating_column,
            *design_columns,
        ),
    )
    permutation_count = _permutation_count(config, n_permutations)
    min_trials = require_config_int(
        config,
        "study2.behavioral_convergence.min_rated_trials",
    )
    min_runs = require_config_int(
        config,
        "study2.behavioral_convergence.min_permutation_valid_runs",
    )

    subject_records: list[dict[str, object]] = []
    criteria_met_frames: list[pd.DataFrame] = []
    observed_betas: list[float] = []
    for subject_id, subject_frame in frame.groupby(subject_column, sort=True):
        if subject_frame.duplicated([run_column, trial_column]).any():
            raise ValueError("Study 2 behavioral convergence contains duplicate run/trial rows.")
        subject_copy = (
            subject_frame.sort_values([run_column, trial_column], kind="mergesort")
            .reset_index(drop=True)
            .copy()
        )
        unmet_criteria = _unmet_subject_criteria(
            subject_copy,
            run_column=run_column,
            expression_column=expression_column,
            rating_column=rating_column,
            design_columns=design_columns,
            min_trials=min_trials,
            min_runs=min_runs,
        )
        if unmet_criteria:
            subject_records.append(
                {
                    "subject_id": str(subject_id),
                    "behavioral_convergence_criteria_met": False,
                    "beta": np.nan,
                    "unmet_criteria": ";".join(unmet_criteria),
                }
            )
            continue

        beta = _subject_beta(
            subject_copy,
            expression_column=expression_column,
            rating_column=rating_column,
            design_columns=design_columns,
        )
        observed_betas.append(beta)
        criteria_met_frames.append(subject_copy)
        subject_records.append(
            {
                "subject_id": str(subject_id),
                "behavioral_convergence_criteria_met": True,
                "beta": beta,
                "unmet_criteria": "",
            }
        )

    if not observed_betas:
        raise ValueError(
            "Study 2 behavioral convergence has no subjects meeting the configured criteria."
        )

    observed = float(np.mean(observed_betas))
    rng = np.random.default_rng(random_state)
    null_distribution = np.asarray(
        [
            np.mean(
                [
                    _subject_beta(
                        _circular_shift_expression(
                            subject_frame,
                            expression_column=expression_column,
                            run_column=run_column,
                            rng=rng,
                        ),
                        expression_column=expression_column,
                        rating_column=rating_column,
                        design_columns=design_columns,
                    )
                    for subject_frame in criteria_met_frames
                ]
            )
            for _ in range(permutation_count)
        ],
        dtype=float,
    )
    p_value = plus_one_p_value(abs(observed), np.abs(null_distribution))
    return BehavioralConvergenceResult(
        mean_beta=observed,
        p_value=p_value,
        null_distribution=null_distribution,
        subject_results=pd.DataFrame(
            subject_records,
            columns=[
                "subject_id",
                "behavioral_convergence_criteria_met",
                "beta",
                "unmet_criteria",
            ],
        ),
        n_subjects=len(observed_betas),
    )


def _permutation_count(config: Any, n_permutations: int | None) -> int:
    value = (
        n_permutations
        if n_permutations is not None
        else require_config_int(config, "study2.behavioral_convergence.permutations")
    )
    if value is None:
        raise ValueError("Study 2 behavioral convergence permutation count is missing.")
    if isinstance(value, bool):
        raise TypeError("Study 2 behavioral convergence permutations must be an integer.")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError("Study 2 behavioral convergence permutations must be an integer.")
    count = int(numeric_value)
    if count <= 0:
        raise ValueError("Study 2 behavioral convergence permutations must be positive.")
    return count


def _unmet_subject_criteria(
    frame: pd.DataFrame,
    *,
    run_column: str,
    expression_column: str,
    rating_column: str,
    design_columns: tuple[str, ...],
    min_trials: int,
    min_runs: int,
) -> tuple[str, ...]:
    numeric_columns = (expression_column, rating_column, *design_columns)
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Study 2 behavioral convergence contains non-finite numeric values.")
    if len(frame) < min_trials:
        return ("min_rated_trials",)
    shiftable_runs = int((frame.groupby(run_column, sort=False).size() > 1).sum())
    if shiftable_runs < min_runs:
        return ("min_valid_runs",)
    if float(np.std(numeric[rating_column].to_numpy(dtype=float), ddof=0)) <= 0.0:
        return ("zero_variance_rating",)
    if float(np.std(numeric[expression_column].to_numpy(dtype=float), ddof=0)) <= 0.0:
        return ("zero_variance_expression",)
    design = numeric.loc[:, design_columns].to_numpy(dtype=float)
    if _residual_std(numeric[rating_column].to_numpy(dtype=float), design) <= 1.0e-12:
        return ("zero_residual_variance_rating",)
    if _residual_std(numeric[expression_column].to_numpy(dtype=float), design) <= 1.0e-12:
        return ("zero_residual_variance_expression",)
    return ()


def _subject_beta(
    frame: pd.DataFrame,
    *,
    expression_column: str,
    rating_column: str,
    design_columns: tuple[str, ...],
) -> float:
    design = frame.loc[:, design_columns].to_numpy(dtype=float)
    expression = standardized_residual(
        frame[expression_column].to_numpy(dtype=float),
        design,
        name="behavioral expression",
    )
    rating = standardized_residual(
        frame[rating_column].to_numpy(dtype=float),
        design,
        name="behavioral rating",
    )
    return float((expression @ rating) / len(expression))


def _residual_std(values: np.ndarray, design: np.ndarray) -> float:
    predictors = np.column_stack([np.ones(len(values), dtype=float), design])
    if np.linalg.matrix_rank(predictors) < predictors.shape[1]:
        raise ValueError("Study 2 behavioral convergence design must have full column rank.")
    coefficients, *_ = np.linalg.lstsq(predictors, values, rcond=None)
    residual = values - predictors @ coefficients
    return float(np.std(residual, ddof=0))


def _circular_shift_expression(
    frame: pd.DataFrame,
    *,
    expression_column: str,
    run_column: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    shifted = frame.copy()
    for _run, run_index in shifted.groupby(run_column, sort=False).groups.items():
        indices = np.asarray(list(run_index), dtype=int)
        if indices.size <= 1:
            continue
        shift = int(rng.integers(1, indices.size))
        shifted.loc[indices, expression_column] = np.roll(
            shifted.loc[indices, expression_column].to_numpy(dtype=float),
            shift,
        )
    return shifted


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Study 2 behavioral convergence table is missing columns: {missing}.")


__all__ = ["BehavioralConvergenceResult", "compute_behavioral_convergence"]
