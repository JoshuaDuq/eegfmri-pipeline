"""Participant estimands for Study 1 EEG power construct validity."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.validity_data import WITHIN_SCALE_INTENSITY_COLUMN

FIGURE_CONFIG_KEY = "study1.figures.power_construct_validity"
CONTINUOUS_NUISANCE_COLUMNS = (
    "within_run_trial",
    "residual_ecg_coupling",
    "fp1_fp2_high_frequency_power",
)
CATEGORICAL_NUISANCE_COLUMNS = ("stimulus_temp", "run", "selected_surface")


def build_temperature_association(
    trials: pd.DataFrame,
    *,
    bands: Sequence[str],
    temperatures: Sequence[float],
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return participant-centered trajectories and simultaneous cohort intervals."""

    _require_columns(
        trials,
        {"subject_id", "band", "stimulus_temp", "global_power_db"},
        "Power association trials",
    )
    values = trials[["subject_id", "band", "stimulus_temp", "global_power_db"]].copy()
    _require_finite_columns(values, ("stimulus_temp", "global_power_db"))
    _require_exact_cells(values, bands=bands, temperatures=temperatures)
    participant = (
        values.groupby(["subject_id", "band", "stimulus_temp"], sort=True, as_index=False)[
            "global_power_db"
        ]
        .mean()
        .rename(columns={"global_power_db": "mean_power_db"})
    )
    participant["centered_power_db"] = participant["mean_power_db"] - participant.groupby(
        ["subject_id", "band"], sort=False
    )["mean_power_db"].transform("mean")
    participant["temperature_slope_db_per_c"] = participant.groupby(
        ["subject_id", "band"], sort=False, group_keys=False
    ).apply(_temperature_slope, include_groups=False)

    matrix = participant.pivot(
        index="subject_id",
        columns=["band", "stimulus_temp"],
        values="centered_power_db",
    ).reindex(columns=pd.MultiIndex.from_product((bands, temperatures)))
    if matrix.isna().any().any():
        raise ValueError(
            "Every participant must contribute every configured band-temperature cell."
        )
    means, lows, highs = _simultaneous_bootstrap(matrix.to_numpy(dtype=float), config)
    cohort = pd.DataFrame(
        {
            "band": [str(band) for band, _temperature in matrix.columns],
            "stimulus_temp": [float(temperature) for _band, temperature in matrix.columns],
            "mean": means,
            "ci_low": lows,
            "ci_high": highs,
            "n_subjects": len(matrix),
            "interval_type": "simultaneous_max_studentized_participant_bootstrap",
        }
    )
    return participant, cohort


def build_rating_association(
    trials: pd.DataFrame,
    *,
    bands: Sequence[str],
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute participant-level partial rating-power correlations by band."""

    required = {
        "subject_id",
        "band",
        "global_power_db",
        WITHIN_SCALE_INTENSITY_COLUMN,
        *CATEGORICAL_NUISANCE_COLUMNS,
        *CONTINUOUS_NUISANCE_COLUMNS,
    }
    _require_columns(trials, required, "Power association trials")
    records = [
        _fit_rating_cell(rows, config)
        for (_subject, _band), rows in trials.groupby(["subject_id", "band"], sort=True)
    ]
    participants = pd.DataFrame.from_records(records)
    if set(participants["band"].astype(str)) != set(bands):
        raise ValueError("Rating association is missing configured spectral bands.")
    estimable = participants.loc[participants["estimable"]].copy()
    return participants, _rating_bootstrap(estimable, bands=bands, config=config)


def _fit_rating_cell(rows: pd.DataFrame, config: Any) -> dict[str, object]:
    subject_id = str(rows["subject_id"].iloc[0])
    band = str(rows["band"].iloc[0])
    record: dict[str, object] = {
        "subject_id": subject_id,
        "band": band,
        "estimable": False,
        "non_estimability_reason": "none",
        "partial_r": float("nan"),
        "n_trials": int(len(rows)),
        "n_runs": int(rows["run"].nunique()),
        "design_rank": pd.NA,
        "design_columns": pd.NA,
        "residual_degrees_of_freedom": pd.NA,
        "condition_number": float("nan"),
    }
    rating_config = _figure_config(config)["rating_model"]
    if len(rows) < int(rating_config["minimum_trials"]):
        record["non_estimability_reason"] = "too_few_trials"
        return record
    if rows["run"].nunique() < int(rating_config["minimum_runs"]):
        record["non_estimability_reason"] = "too_few_runs"
        return record
    numeric_columns = (
        "global_power_db",
        WITHIN_SCALE_INTENSITY_COLUMN,
        *CATEGORICAL_NUISANCE_COLUMNS,
        *CONTINUOUS_NUISANCE_COLUMNS,
    )
    try:
        _require_finite_columns(rows, numeric_columns)
    except ValueError:
        record["non_estimability_reason"] = "non_finite_values"
        return record
    try:
        design = _rating_design(rows, config)
    except ValueError as error:
        record["non_estimability_reason"] = str(error)
        return record
    rank = int(np.linalg.matrix_rank(design))
    condition_number = float(np.linalg.cond(design))
    residual_df = len(rows) - design.shape[1]
    record.update(
        design_rank=rank,
        design_columns=int(design.shape[1]),
        residual_degrees_of_freedom=int(residual_df),
        condition_number=condition_number,
    )
    if rank != design.shape[1]:
        record["non_estimability_reason"] = "rank_deficient_design"
        return record
    if condition_number > float(rating_config["max_condition_number"]):
        record["non_estimability_reason"] = "excessive_condition_number"
        return record
    if residual_df <= 0:
        record["non_estimability_reason"] = "nonpositive_residual_degrees_of_freedom"
        return record
    power_residual = _residualize(rows["global_power_db"].to_numpy(dtype=float), design)
    rating_residual = _residualize(
        rows[WITHIN_SCALE_INTENSITY_COLUMN].to_numpy(dtype=float), design
    )
    if np.std(power_residual, ddof=1) <= 1.0e-12:
        record["non_estimability_reason"] = "zero_residual_variance_power"
        return record
    if np.std(rating_residual, ddof=1) <= 1.0e-12:
        record["non_estimability_reason"] = "zero_residual_variance_rating"
        return record
    partial_r = float(np.corrcoef(power_residual, rating_residual)[0, 1])
    if not np.isfinite(partial_r):
        raise ValueError(f"Rating-power partial correlation is non-finite for {subject_id}/{band}.")
    record.update(estimable=True, partial_r=partial_r)
    return record


def _rating_design(rows: pd.DataFrame, config: Any) -> np.ndarray:
    temperatures = tuple(
        float(value)
        for value in require_config_value(config, "study1.figures.validity.temperatures")
    )
    columns: list[np.ndarray] = [np.ones(len(rows), dtype=float)]
    category_levels: Mapping[str, Sequence[object]] = {
        "stimulus_temp": temperatures,
        "run": tuple(sorted(rows["run"].unique().tolist())),
        "selected_surface": tuple(sorted(rows["selected_surface"].unique().tolist())),
    }
    for column in CATEGORICAL_NUISANCE_COLUMNS:
        categorical = pd.Categorical(rows[column], categories=category_levels[column], ordered=True)
        if categorical.isna().any():
            raise ValueError(f"unknown_{column}_levels")
        dummies = pd.get_dummies(categorical, drop_first=True, dtype=float)
        columns.extend(dummies.to_numpy(dtype=float).T)
    for column in CONTINUOUS_NUISANCE_COLUMNS:
        values = rows[column].to_numpy(dtype=float)
        standard_deviation = float(np.std(values, ddof=1))
        if standard_deviation <= 1.0e-12:
            raise ValueError(f"constant_{column}")
        columns.append((values - np.mean(values)) / standard_deviation)
    return np.column_stack(columns)


def _rating_bootstrap(
    estimable: pd.DataFrame,
    *,
    bands: Sequence[str],
    config: Any,
) -> pd.DataFrame:
    columns = ("band", "mean_partial_r", "ci_low", "ci_high", "n_subjects")
    if estimable.empty:
        return pd.DataFrame(columns=columns)
    matrix = estimable.pivot(index="subject_id", columns="band", values="partial_r").reindex(
        columns=bands
    )
    complete = matrix.dropna(axis=0, how="any")
    if len(complete) < 2:
        raise ValueError("Rating-power cohort estimation requires two complete participants.")
    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    iterations = int(bootstrap["iterations"])
    confidence_level = float(bootstrap["confidence_level"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    values = complete.to_numpy(dtype=float)
    sampled = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(sampled, [tail, 1.0 - tail], axis=0)
    return pd.DataFrame(
        {
            "band": list(bands),
            "mean_partial_r": values.mean(axis=0),
            "ci_low": low,
            "ci_high": high,
            "n_subjects": len(values),
        }
    )


def _require_exact_cells(
    values: pd.DataFrame,
    *,
    bands: Sequence[str],
    temperatures: Sequence[float],
) -> None:
    if set(values["band"].astype(str)) != set(bands):
        raise ValueError("Power association trials do not contain every configured band.")
    expected_temperatures = tuple(float(value) for value in temperatures)
    for (subject_id, band), rows in values.groupby(["subject_id", "band"], sort=False):
        observed = tuple(sorted(rows["stimulus_temp"].unique().tolist()))
        if observed != expected_temperatures:
            raise ValueError(
                "Every participant-band cell requires all configured temperatures: "
                f"{subject_id}/{band}, observed={observed}."
            )


def _temperature_slope(rows: pd.DataFrame) -> pd.Series:
    slope = np.polyfit(
        rows["stimulus_temp"].to_numpy(dtype=float),
        rows["mean_power_db"].to_numpy(dtype=float),
        deg=1,
    )[0]
    return pd.Series(float(slope), index=rows.index)


def _simultaneous_bootstrap(
    matrix: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bootstrap = require_config_value(config, "study1.figures.validity.bootstrap")
    iterations = int(bootstrap["iterations"])
    confidence_level = float(bootstrap["confidence_level"])
    seed = int(bootstrap["seed"])
    if iterations < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Power validity bootstrap settings are invalid.")
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Power validity bootstrap requires a finite participant matrix.")
    means = matrix.mean(axis=0)
    standard_errors = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    if np.any(standard_errors <= np.finfo(float).eps):
        raise ValueError("Simultaneous power bootstrap requires non-zero cell variability.")
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, matrix.shape[0], size=(iterations, matrix.shape[0]))
    bootstrap_means = matrix[sampled_indices].mean(axis=1)
    maxima = np.max(np.abs((bootstrap_means - means) / standard_errors), axis=1)
    critical = float(np.quantile(maxima, confidence_level))
    return means, means - critical * standard_errors, means + critical * standard_errors


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _figure_config(config: Any) -> Mapping[str, Any]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


def _require_finite_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Column {column!r} must contain finite numeric values.")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


__all__ = ["build_rating_association", "build_temperature_association"]
