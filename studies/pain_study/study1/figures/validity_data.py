"""Validated data and estimators for Study 1 validity figures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import find_clean_events_path
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.cohort import load_primary_target_table

TARGET_REQUIRED_COLUMNS = (
    "subject_id",
    "task",
    "run",
    "trial_index",
    "within_run_trial",
    "NPS",
    "SIIPS1",
    "stimulus_temp",
    "selected_surface",
)
EVENT_REQUIRED_COLUMNS = (
    "run",
    "trial_number",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
    "residual_ecg_coupling",
    "fp1_fp2_high_frequency_power",
)
TRIAL_KEY_COLUMNS = ("subject_id", "_run_key", "_within_run_trial_key")


@dataclass(frozen=True)
class ValidityTrialData:
    """Retained targets, clean events, and their strict one-to-one join."""

    targets: pd.DataFrame
    clean_events: pd.DataFrame
    enriched_targets: pd.DataFrame


@dataclass(frozen=True)
class DoseResponseSummary:
    """Participant and cohort summaries for one dose-response outcome."""

    outcome: str
    temperatures: tuple[float, ...]
    participant_means: pd.DataFrame
    participant_matrix: pd.DataFrame
    cohort_estimates: pd.DataFrame


def load_validity_trial_data(*, task: str, config: Any) -> ValidityTrialData:
    targets = load_primary_target_table(config).copy()
    _require_columns(targets, TARGET_REQUIRED_COLUMNS, table_name="Study 1 target table")
    _require_single_task(targets, task)
    for column in ("NPS", "SIIPS1", "stimulus_temp"):
        targets[column] = _finite_numeric_series(targets, column, table_name="target table")

    subjects = sorted(targets["subject_id"].astype(str).unique())
    clean_events = _load_clean_events(subjects=subjects, task=task, config=config)
    enriched_targets = _merge_targets_with_clean_events(targets, clean_events)
    return ValidityTrialData(
        targets=targets,
        clean_events=clean_events,
        enriched_targets=enriched_targets,
    )


def build_dose_response_summary(
    trials: pd.DataFrame,
    outcome: str,
    config: Any,
) -> DoseResponseSummary:
    temperatures = _configured_temperatures(config)
    values = _validated_outcome_frame(trials, outcome, temperatures)
    participant_means = (
        values.groupby(["subject_id", "stimulus_temp"], sort=True, as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "value"})
    )
    participant_matrix = participant_means.pivot(
        index="subject_id",
        columns="stimulus_temp",
        values="value",
    ).reindex(columns=temperatures)
    cohort_estimates = _participant_bootstrap_estimates(participant_matrix, config)
    return DoseResponseSummary(
        outcome=outcome,
        temperatures=temperatures,
        participant_means=participant_means,
        participant_matrix=participant_matrix,
        cohort_estimates=cohort_estimates,
    )


def _load_clean_events(*, subjects: list[str], task: str, config: Any) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for subject_id in subjects:
        event_path = find_clean_events_path(subject_id, task, config=config)
        if event_path is None or not Path(event_path).exists():
            raise FileNotFoundError(
                "Study 1 validity figures require clean events for every retained "
                f"subject. Missing: {subject_id}, task-{task}."
            )
        events = pd.read_csv(event_path, sep="\t")
        _require_columns(
            events,
            EVENT_REQUIRED_COLUMNS,
            table_name=f"clean events for {subject_id}",
        )
        events = events.copy()
        events["subject_id"] = subject_id
        events["_run_key"] = _integer_series(events, "run", table_name="clean events")
        events["_within_run_trial_key"] = _integer_series(
            events,
            "trial_number",
            table_name="clean events",
        )
        for column in (
            "stimulus_temp",
            "pain_binary_coded",
            "vas_final_coded_rating",
            "residual_ecg_coupling",
            "fp1_fp2_high_frequency_power",
        ):
            events[column] = _finite_numeric_series(events, column, table_name="clean events")
        frames.append(events)
    if not frames:
        raise ValueError("Study 1 validity figures require at least one retained subject.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _merge_targets_with_clean_events(
    targets: pd.DataFrame,
    clean_events: pd.DataFrame,
) -> pd.DataFrame:
    target_rows = targets.copy()
    target_rows["subject_id"] = target_rows["subject_id"].astype(str)
    target_rows["_run_key"] = _integer_series(target_rows, "run", table_name="target table")
    target_rows["_within_run_trial_key"] = _integer_series(
        target_rows,
        "within_run_trial",
        table_name="target table",
    )
    if target_rows.duplicated(list(TRIAL_KEY_COLUMNS)).any():
        raise ValueError("Study 1 target table contains duplicate target trial keys.")
    if clean_events.duplicated(list(TRIAL_KEY_COLUMNS)).any():
        raise ValueError("Study 1 clean events contain duplicate clean-event trial keys.")

    event_columns = [*TRIAL_KEY_COLUMNS, *EVENT_REQUIRED_COLUMNS[2:]]
    renamed_events = clean_events[event_columns].rename(
        columns={
            "stimulus_temp": "event_stimulus_temp",
            "selected_surface": "event_selected_surface",
            "residual_ecg_coupling": "event_residual_ecg_coupling",
        }
    )
    merged = target_rows.merge(
        renamed_events,
        how="left",
        on=list(TRIAL_KEY_COLUMNS),
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        missing = merged.loc[unmatched, ["subject_id", "run", "within_run_trial"]]
        raise ValueError(
            "Study 1 target rows were found without matching clean events:\n"
            f"{missing.to_string(index=False)}"
        )
    target_temperature = merged["stimulus_temp"].to_numpy(dtype=float)
    event_temperature = merged["event_stimulus_temp"].to_numpy(dtype=float)
    if not np.allclose(target_temperature, event_temperature, rtol=0.0, atol=1e-9):
        raise ValueError("Study 1 target and clean-event stimulus temperature disagrees.")
    return merged.drop(columns=["_run_key", "_within_run_trial_key", "_merge"])


def _validated_outcome_frame(
    trials: pd.DataFrame,
    outcome: str,
    temperatures: tuple[float, ...],
) -> pd.DataFrame:
    _require_columns(
        trials,
        ("subject_id", "stimulus_temp", outcome),
        table_name="dose-response trials",
    )
    values = trials[["subject_id", "stimulus_temp", outcome]].copy()
    values["subject_id"] = values["subject_id"].astype(str)
    if (values["subject_id"].str.strip() == "").any():
        raise ValueError("Dose-response subject identifiers must be non-empty.")
    values["stimulus_temp"] = _finite_numeric_series(
        values,
        "stimulus_temp",
        table_name="dose-response trials",
    )
    values[outcome] = _finite_numeric_series(
        values,
        outcome,
        table_name="dose-response trials",
    )
    observed = tuple(sorted(values["stimulus_temp"].unique().tolist()))
    if observed != temperatures:
        raise ValueError(
            "Observed stimulus temperatures do not match configured validity temperatures: "
            f"observed={observed}, configured={temperatures}."
        )
    represented = values.groupby("stimulus_temp")["subject_id"].nunique()
    inadequate = represented[represented < 2]
    if not inadequate.empty:
        raise ValueError(
            "Every validity temperature requires at least two represented participants: "
            f"{inadequate.to_dict()}."
        )
    return values


def _participant_bootstrap_estimates(
    participant_matrix: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    iterations = int(bootstrap["iterations"])
    confidence_level = float(bootstrap["confidence_level"])
    seed = int(bootstrap["seed"])
    max_invalid_fraction = float(bootstrap["max_invalid_fraction"])
    max_attempts = math.ceil(iterations / (1.0 - max_invalid_fraction))

    matrix = participant_matrix.to_numpy(dtype=float)
    n_subjects = matrix.shape[0]
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    attempted = 0
    invalid = 0
    while len(accepted) < iterations and attempted < max_attempts:
        sampled = matrix[rng.integers(0, n_subjects, size=n_subjects)]
        attempted += 1
        if not np.isfinite(sampled).any(axis=0).all():
            invalid += 1
            continue
        estimate = np.nanmean(sampled, axis=0)
        if not np.isfinite(estimate).all():
            raise ValueError("Study 1 validity bootstrap produced a non-finite accepted estimate.")
        accepted.append(estimate)
    if len(accepted) != iterations:
        raise ValueError(
            "Study 1 validity bootstrap exceeded its invalid-draw budget: "
            f"valid={len(accepted)}, attempted={attempted}, invalid={invalid}."
        )

    bootstrap_means = np.vstack(accepted)
    tail = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, [tail, 1.0 - tail], axis=0)
    means = np.nanmean(matrix, axis=0)
    if not np.isfinite(np.concatenate((means, ci_low, ci_high))).all():
        raise ValueError("Study 1 validity summary contains non-finite cohort estimates.")
    return pd.DataFrame(
        {
            "stimulus_temp": participant_matrix.columns.to_numpy(dtype=float),
            "mean": means,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_subjects": np.isfinite(matrix).sum(axis=0).astype(int),
        }
    )


def _configured_temperatures(config: Any) -> tuple[float, ...]:
    configured = require_config_value(config, "study1.figures.validity.temperatures")
    return tuple(float(value) for value in configured)


def _require_single_task(frame: pd.DataFrame, task: str) -> None:
    observed = sorted(frame["task"].astype(str).unique().tolist())
    if observed != [str(task)]:
        raise ValueError(
            "Study 1 target table must contain only the requested task: "
            f"requested={task!r}, observed={observed}."
        )


def _integer_series(frame: pd.DataFrame, column: str, *, table_name: str) -> pd.Series:
    values = _finite_numeric_series(frame, column, table_name=table_name)
    if not np.allclose(values, np.round(values)):
        raise ValueError(f"{table_name} column '{column}' must contain integer-valued labels.")
    return values.round().astype(int)


def _finite_numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    table_name: str,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{table_name} column '{column}' must contain finite numeric values.")
    return values.astype(float)


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}.")


__all__ = [
    "DoseResponseSummary",
    "ValidityTrialData",
    "build_dose_response_summary",
    "load_validity_trial_data",
]
