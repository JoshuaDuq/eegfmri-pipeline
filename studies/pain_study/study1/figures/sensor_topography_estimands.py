"""Participant-level estimands for Study 1 sensor topographies."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.figures.power_construct_models import (
    CATEGORICAL_NUISANCE_COLUMNS,
    CONTINUOUS_NUISANCE_COLUMNS,
)
from studies.pain_study.study1.figures.sensor_topography_data import SensorPowerData
from studies.pain_study.study1.figures.validity_data import WITHIN_SCALE_INTENSITY_COLUMN
from studies.pain_study.study1.targets import resolve_target_residualization_columns

CONSTRUCT_ESTIMANDS = ("temperature", "intensity")
SIGNATURE_ESTIMANDS = ("NPS", "SIIPS1")
FRONTAL_CHANNELS = frozenset({"Fp1", "Fp2"})
INFERENCE_VALUE_COLUMN = "inference_value"
EXCLUSION_COLUMNS = ("subject_id", "reason", "estimand", "band", "channel")


@dataclass(frozen=True)
class ParticipantEffects:
    """Complete participant maps and their deterministic analysis contract."""

    effects: pd.DataFrame
    summary: pd.DataFrame
    exclusions: pd.DataFrame
    sensitivity_effects: pd.DataFrame | None
    sensitivity_summary: pd.DataFrame | None
    map_order: tuple[tuple[str, str], ...]
    sensor_order: tuple[str, ...]
    participant_order: tuple[str, ...]
    inference_value_column: str = INFERENCE_VALUE_COLUMN

    def inference_tensor(self) -> np.ndarray:
        """Return participant-by-map-by-sensor values in the declared order."""

        names = ("subject_id", "estimand", "band", "channel")
        requested = [
            (subject, estimand, band, channel)
            for subject in self.participant_order
            for estimand, band in self.map_order
            for channel in self.sensor_order
        ]
        expected = pd.MultiIndex.from_tuples(requested, names=names)
        values = self.effects.set_index(list(names))[self.inference_value_column]
        if values.index.has_duplicates:
            raise ValueError("Participant effects contain duplicate map-sensor cells.")
        ordered = values.reindex(expected)
        if ordered.isna().any():
            raise ValueError("Participant effects do not form the declared complete tensor.")
        return ordered.to_numpy(dtype=float).reshape(
            len(self.participant_order),
            len(self.map_order),
            len(self.sensor_order),
        )


class _NonEstimable(ValueError):
    def __init__(self, reason: str, estimand: str, band: str, channel: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.estimand = estimand
        self.band = band
        self.channel = channel

    def record(self, subject_id: str) -> dict[str, str]:
        return {
            "subject_id": subject_id,
            "reason": self.reason,
            "estimand": self.estimand,
            "band": self.band,
            "channel": self.channel,
        }


def build_construct_effects(data: SensorPowerData, config: Any) -> ParticipantEffects:
    """Estimate complete temperature and intensity sensor maps."""

    required = (
        "stimulus_temp",
        "run",
        "selected_surface",
        *CONTINUOUS_NUISANCE_COLUMNS,
        WITHIN_SCALE_INTENSITY_COLUMN,
    )
    _validate_sensor_power_data(data, required_numeric=required)
    temperatures = _configured_temperatures(config)
    primary_include, compute_sensitivity = _construct_scope_config(config)
    primary_channels = _construct_channels(data.channels, primary_include)
    sensitivity_channels = (
        _construct_channels(data.channels, not primary_include) if compute_sensitivity else None
    )
    map_order = _map_order(CONSTRUCT_ESTIMANDS, data.bands)

    primary_records: list[dict[str, object]] = []
    sensitivity_records: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    included: list[str] = []
    for subject_id in sorted(data.subjects):
        subject_rows = data.trials.loc[data.trials["subject_id"].eq(subject_id)]
        try:
            primary = _estimate_construct_subject(
                subject_rows,
                map_order=map_order,
                channels=primary_channels,
                channel_scope="primary",
                include_fp1_fp2=primary_include,
                temperatures=temperatures,
                config=config,
            )
            sensitivity = []
            if sensitivity_channels is not None:
                sensitivity = _estimate_construct_subject(
                    subject_rows,
                    map_order=map_order,
                    channels=sensitivity_channels,
                    channel_scope="complementary",
                    include_fp1_fp2=not primary_include,
                    temperatures=temperatures,
                    config=config,
                )
        except _NonEstimable as error:
            exclusions.append(error.record(subject_id))
            continue
        primary_records.extend(primary)
        sensitivity_records.extend(sensitivity)
        included.append(subject_id)

    effects = _complete_effect_frame(primary_records, included, map_order, primary_channels)
    sensitivity_effects = None
    sensitivity_summary = None
    if sensitivity_channels is not None:
        sensitivity_effects = _complete_effect_frame(
            sensitivity_records,
            included,
            map_order,
            sensitivity_channels,
        )
        sensitivity_summary = _summarize_effects(
            sensitivity_effects,
            map_order=map_order,
            channels=sensitivity_channels,
        )
    return ParticipantEffects(
        effects=effects,
        summary=_summarize_effects(effects, map_order=map_order, channels=primary_channels),
        exclusions=_exclusion_frame(exclusions),
        sensitivity_effects=sensitivity_effects,
        sensitivity_summary=sensitivity_summary,
        map_order=map_order,
        sensor_order=primary_channels,
        participant_order=tuple(included),
    )


def build_signature_effects(data: SensorPowerData, config: Any) -> ParticipantEffects:
    """Estimate complete NPS and SIIPS1 partial-correlation sensor maps."""

    _validate_sensor_power_data(data, required_numeric=("run", *SIGNATURE_ESTIMANDS))
    nuisance_columns = {
        target: resolve_target_residualization_columns(
            frame=data.trials,
            config=config,
            target_name=target,
        )
        for target in SIGNATURE_ESTIMANDS
    }
    if "NPS" not in nuisance_columns["SIIPS1"]:
        raise ValueError("SIIPS1 sensor effects require NPS as an ordered nuisance column.")
    _require_columns(
        data.trials,
        tuple(dict.fromkeys(column for values in nuisance_columns.values() for column in values)),
        "Sensor-power trials",
    )
    _require_finite_numeric(
        data.trials,
        tuple(dict.fromkeys(column for values in nuisance_columns.values() for column in values)),
    )
    _require_trial_covariate_consistency(
        data.trials,
        tuple(dict.fromkeys(column for values in nuisance_columns.values() for column in values)),
    )
    channels = _signature_channels(data.channels, config)
    model = _signature_model_config(config)
    minimum_trials, minimum_runs = _signature_sample_requirements(config)
    map_order = _map_order(SIGNATURE_ESTIMANDS, data.bands)

    records: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    included: list[str] = []
    for subject_id in sorted(data.subjects):
        subject_rows = data.trials.loc[data.trials["subject_id"].eq(subject_id)]
        try:
            subject_records = _estimate_signature_subject(
                subject_rows,
                map_order=map_order,
                channels=channels,
                nuisance_columns=nuisance_columns,
                model=model,
                minimum_trials=minimum_trials,
                minimum_runs=minimum_runs,
            )
        except _NonEstimable as error:
            exclusions.append(error.record(subject_id))
            continue
        records.extend(subject_records)
        included.append(subject_id)

    effects = _complete_effect_frame(records, included, map_order, channels)
    return ParticipantEffects(
        effects=effects,
        summary=_summarize_effects(effects, map_order=map_order, channels=channels),
        exclusions=_exclusion_frame(exclusions),
        sensitivity_effects=None,
        sensitivity_summary=None,
        map_order=map_order,
        sensor_order=channels,
        participant_order=tuple(included),
    )


def _estimate_construct_subject(
    rows: pd.DataFrame,
    *,
    map_order: tuple[tuple[str, str], ...],
    channels: tuple[str, ...],
    channel_scope: str,
    include_fp1_fp2: bool,
    temperatures: tuple[float, ...],
    config: Any,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for estimand, band in map_order:
        for channel in channels:
            cell = rows.loc[rows["band"].eq(band) & rows["channel"].eq(channel)]
            if cell.empty:
                raise _NonEstimable("incomplete_effect_grid", estimand, band, channel)
            if estimand == "temperature":
                record = _fit_temperature_cell(
                    cell,
                    band=band,
                    channel=channel,
                    temperatures=temperatures,
                )
            else:
                record = _fit_intensity_cell(
                    cell,
                    band=band,
                    channel=channel,
                    config=config,
                )
            record.update(
                subject_id=str(rows["subject_id"].iloc[0]),
                estimand=estimand,
                band=band,
                channel=channel,
                channel_scope=channel_scope,
                include_fp1_fp2=include_fp1_fp2,
            )
            records.append(record)
    return records


def _fit_temperature_cell(
    rows: pd.DataFrame,
    *,
    band: str,
    channel: str,
    temperatures: tuple[float, ...],
) -> dict[str, object]:
    observed = tuple(sorted(rows["stimulus_temp"].unique().tolist()))
    if observed != temperatures:
        raise _NonEstimable("missing_temperature_cell", "temperature", band, channel)
    means = rows.groupby("stimulus_temp", sort=True)["power_db"].mean().reindex(temperatures)
    if means.isna().any():
        raise _NonEstimable("missing_temperature_cell", "temperature", band, channel)
    design = np.column_stack((np.ones(len(temperatures)), temperatures))
    slope = float(np.linalg.lstsq(design, means.to_numpy(dtype=float), rcond=None)[0][1])
    return {
        "effect_value": slope,
        "partial_r": float("nan"),
        "fisher_z": float("nan"),
        INFERENCE_VALUE_COLUMN: slope,
        "n_trials": len(rows),
        "n_runs": int(rows["run"].nunique()),
    }


def _fit_intensity_cell(
    rows: pd.DataFrame,
    *,
    band: str,
    channel: str,
    config: Any,
) -> dict[str, object]:
    rating = _required_mapping(
        require_config_value(config, "study1.figures.power_construct_validity.rating_model"),
        "study1.figures.power_construct_validity.rating_model",
    )
    minimum_trials = _positive_integer(rating["minimum_trials"], "Rating minimum trials")
    minimum_runs = _positive_integer(rating["minimum_runs"], "Rating minimum runs")
    if len(rows) < minimum_trials:
        raise _NonEstimable("too_few_trials", "intensity", band, channel)
    if rows["run"].nunique() < minimum_runs:
        raise _NonEstimable("too_few_runs", "intensity", band, channel)
    try:
        design = _construct_design(rows, config)
    except ValueError as error:
        raise _NonEstimable(str(error), "intensity", band, channel) from error
    rank = int(np.linalg.matrix_rank(design))
    condition_number = float(np.linalg.cond(design))
    residual_df = len(rows) - design.shape[1]
    if rank != design.shape[1]:
        raise _NonEstimable("rank_deficient_design", "intensity", band, channel)
    if condition_number > float(rating["max_condition_number"]):
        raise _NonEstimable("excessive_condition_number", "intensity", band, channel)
    if residual_df <= 0:
        raise _NonEstimable("nonpositive_residual_degrees_of_freedom", "intensity", band, channel)
    power_residual = _residualize(rows["power_db"].to_numpy(dtype=float), design)
    target_residual = _residualize(
        rows[WITHIN_SCALE_INTENSITY_COLUMN].to_numpy(dtype=float),
        design,
    )
    _require_residual_variation(
        power_residual,
        target_residual,
        minimum=1.0e-12,
        estimand="intensity",
        band=band,
        channel=channel,
        target_label="rating",
    )
    partial_r, fisher_z = _correlation_effect(
        power_residual,
        target_residual,
        estimand="intensity",
        band=band,
        channel=channel,
    )
    return {
        "effect_value": partial_r,
        "partial_r": partial_r,
        "fisher_z": fisher_z,
        INFERENCE_VALUE_COLUMN: fisher_z,
        "n_trials": len(rows),
        "n_runs": int(rows["run"].nunique()),
        "design_rank": rank,
        "design_parameters": int(design.shape[1]),
        "residual_degrees_of_freedom": residual_df,
        "condition_number": condition_number,
    }


def _construct_design(rows: pd.DataFrame, config: Any) -> np.ndarray:
    temperatures = _configured_temperatures(config)
    columns: list[np.ndarray] = [np.ones(len(rows), dtype=float)]
    levels: Mapping[str, Sequence[object]] = {
        "stimulus_temp": temperatures,
        "run": tuple(sorted(rows["run"].unique())),
        "selected_surface": tuple(sorted(rows["selected_surface"].unique())),
    }
    for column in CATEGORICAL_NUISANCE_COLUMNS:
        categorical = pd.Categorical(rows[column], categories=levels[column], ordered=True)
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


def _estimate_signature_subject(
    rows: pd.DataFrame,
    *,
    map_order: tuple[tuple[str, str], ...],
    channels: tuple[str, ...],
    nuisance_columns: Mapping[str, tuple[str, ...]],
    model: Mapping[str, float],
    minimum_trials: int,
    minimum_runs: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    subject_id = str(rows["subject_id"].iloc[0])
    for target, band in map_order:
        for channel in channels:
            cell = rows.loc[rows["band"].eq(band) & rows["channel"].eq(channel)]
            if cell.empty:
                raise _NonEstimable("incomplete_effect_grid", target, band, channel)
            record = _fit_signature_cell(
                cell,
                target=target,
                band=band,
                channel=channel,
                nuisance_columns=nuisance_columns[target],
                model=model,
                minimum_trials=minimum_trials,
                minimum_runs=minimum_runs,
            )
            record.update(
                subject_id=subject_id,
                estimand=target,
                band=band,
                channel=channel,
                channel_scope="feature_benchmark",
                include_fp1_fp2=False,
            )
            records.append(record)
    return records


def _fit_signature_cell(
    rows: pd.DataFrame,
    *,
    target: str,
    band: str,
    channel: str,
    nuisance_columns: tuple[str, ...],
    model: Mapping[str, float],
    minimum_trials: int,
    minimum_runs: int,
) -> dict[str, object]:
    if len(rows) < minimum_trials:
        raise _NonEstimable("too_few_trials", target, band, channel)
    if rows["run"].nunique() < minimum_runs:
        raise _NonEstimable("too_few_runs", target, band, channel)
    design, audit = _signature_design(
        rows,
        nuisance_columns=nuisance_columns,
        target=target,
        band=band,
        channel=channel,
        model=model,
    )
    power_residual = _residualize(rows["power_db"].to_numpy(dtype=float), design)
    target_residual = _residualize(rows[target].to_numpy(dtype=float), design)
    _require_residual_variation(
        power_residual,
        target_residual,
        minimum=float(model["minimum_residual_standard_deviation"]),
        estimand=target,
        band=band,
        channel=channel,
        target_label="target",
    )
    partial_r, fisher_z = _correlation_effect(
        power_residual,
        target_residual,
        estimand=target,
        band=band,
        channel=channel,
    )
    return {
        "effect_value": partial_r,
        "partial_r": partial_r,
        "fisher_z": fisher_z,
        INFERENCE_VALUE_COLUMN: fisher_z,
        "n_trials": len(rows),
        "n_runs": int(rows["run"].nunique()),
        **audit,
    }


def _signature_design(
    rows: pd.DataFrame,
    *,
    nuisance_columns: tuple[str, ...],
    target: str,
    band: str,
    channel: str,
    model: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, object]]:
    omitted: list[str] = []
    retained: list[str] = []
    standardized: list[np.ndarray] = []
    for column in nuisance_columns:
        values = rows[column].to_numpy(dtype=float)
        if np.max(values) == np.min(values):
            omitted.append(column)
            continue
        centered = values - np.mean(values)
        standardized.append(centered / np.linalg.norm(centered))
        retained.append(column)

    design = np.column_stack((np.ones(len(rows)), *standardized))
    residual_df = len(rows) - design.shape[1]
    if residual_df <= 0:
        raise _NonEstimable("nonpositive_residual_degrees_of_freedom", target, band, channel)
    singular_ratio = 1.0
    if standardized:
        singular_values = np.linalg.svd(np.column_stack(standardized), compute_uv=False)
        singular_ratio = float(singular_values[-1] / singular_values[0])
        if singular_ratio < float(model["rank_tolerance"]):
            raise _NonEstimable("rank_deficient_design", target, band, channel)
    condition_number = float(np.linalg.cond(design))
    if condition_number > float(model["max_condition_number"]):
        raise _NonEstimable("excessive_condition_number", target, band, channel)
    norms = [float(np.linalg.norm(column)) for column in standardized]
    return design, {
        "resolved_nuisance_columns": ",".join(nuisance_columns),
        "omitted_constant_nuisance_columns": ",".join(omitted),
        "retained_nuisance_columns": ",".join(retained),
        "design_parameters": int(design.shape[1]),
        "residual_degrees_of_freedom": residual_df,
        "minimum_singular_value_ratio": singular_ratio,
        "condition_number": condition_number,
        "minimum_standardized_nuisance_norm": min(norms, default=float("nan")),
        "maximum_standardized_nuisance_norm": max(norms, default=float("nan")),
    }


def _require_residual_variation(
    power: np.ndarray,
    target: np.ndarray,
    *,
    minimum: float,
    estimand: str,
    band: str,
    channel: str,
    target_label: str,
) -> None:
    if np.std(power, ddof=1) <= minimum:
        raise _NonEstimable("zero_residual_variance_power", estimand, band, channel)
    if np.std(target, ddof=1) <= minimum:
        raise _NonEstimable(f"zero_residual_variance_{target_label}", estimand, band, channel)


def _correlation_effect(
    left: np.ndarray,
    right: np.ndarray,
    *,
    estimand: str,
    band: str,
    channel: str,
) -> tuple[float, float]:
    partial_r = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(partial_r) or abs(partial_r) >= 1.0:
        raise _NonEstimable("non_finite_correlation", estimand, band, channel)
    return partial_r, float(np.arctanh(partial_r))


def _complete_effect_frame(
    records: list[dict[str, object]],
    participants: Sequence[str],
    map_order: tuple[tuple[str, str], ...],
    channels: tuple[str, ...],
) -> pd.DataFrame:
    if not participants:
        raise ValueError("No participants form a complete sensor-topography effect family.")
    effects = pd.DataFrame.from_records(records)
    expected = len(participants) * len(map_order) * len(channels)
    if len(effects) != expected:
        raise ValueError(
            "Participant effects do not form the required complete family: "
            f"expected={expected}, observed={len(effects)}."
        )
    key_columns = ["subject_id", "estimand", "band", "channel"]
    expected_keys = [
        (subject_id, estimand, band, channel)
        for subject_id in participants
        for estimand, band in map_order
        for channel in channels
    ]
    observed_keys = list(effects[key_columns].itertuples(index=False, name=None))
    if observed_keys != expected_keys:
        raise ValueError("Participant effects require exact unique subject-map-sensor keys.")
    if not np.isfinite(effects[INFERENCE_VALUE_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Complete participant inference values must be finite.")
    return effects.reset_index(drop=True)


def _summarize_effects(
    effects: pd.DataFrame,
    *,
    map_order: tuple[tuple[str, str], ...],
    channels: tuple[str, ...],
) -> pd.DataFrame:
    records = []
    for estimand, band in map_order:
        for channel in channels:
            rows = effects.loc[
                effects["estimand"].eq(estimand)
                & effects["band"].eq(band)
                & effects["channel"].eq(channel)
            ]
            mean_inference = float(rows[INFERENCE_VALUE_COLUMN].mean())
            display_value = (
                mean_inference if estimand == "temperature" else float(np.tanh(mean_inference))
            )
            records.append(
                {
                    "estimand": estimand,
                    "band": band,
                    "channel": channel,
                    "mean_inference_value": mean_inference,
                    "display_value": display_value,
                    "n_participants": len(rows),
                }
            )
    return pd.DataFrame.from_records(records)


def _validate_sensor_power_data(
    data: SensorPowerData,
    *,
    required_numeric: Sequence[str],
) -> None:
    if not isinstance(data, SensorPowerData):
        raise TypeError("Sensor topography estimands require SensorPowerData.")
    for values, label in (
        (data.subjects, "subjects"),
        (data.bands, "bands"),
        (data.channels, "channels"),
    ):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"SensorPowerData {label} must be non-empty and unique.")
    required = ("subject_id", "trial_id", "band", "channel", "power_db", *required_numeric)
    _require_columns(data.trials, required, "Sensor-power trials")
    if data.trials.duplicated(["subject_id", "trial_id", "band", "channel"]).any():
        raise ValueError("Sensor-power trials contain duplicate trial-band-channel cells.")
    expected = {
        "subject_id": set(data.subjects),
        "band": set(data.bands),
        "channel": set(data.channels),
    }
    for column, values in expected.items():
        observed = set(data.trials[column].astype(str))
        if observed != values:
            raise ValueError(
                f"SensorPowerData {column} values disagree with its declared order: "
                f"declared={list(values)}, observed={sorted(observed)}."
            )
    _require_finite_numeric(data.trials, ("trial_id", "power_db", *required_numeric))

    covariates = tuple(column for column in required_numeric if column != "power_db")
    _require_trial_covariate_consistency(data.trials, covariates)


def _require_trial_covariate_consistency(
    trials: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    if not columns:
        return
    grouped = trials.groupby(["subject_id", "trial_id"], sort=False)[list(columns)]
    if grouped.nunique(dropna=False).gt(1).any().any():
        raise ValueError("Sensor-power trial covariates disagree across bands or channels.")


def _configured_temperatures(config: Any) -> tuple[float, ...]:
    values = require_config_value(config, "study1.figures.validity.temperatures")
    temperatures = tuple(float(value) for value in values)
    if not np.isfinite(temperatures).all():
        raise ValueError("Configured validity temperatures must be finite.")
    if len(temperatures) < 2 or any(
        right <= left for left, right in zip(temperatures, temperatures[1:])
    ):
        raise ValueError("Configured validity temperatures must be strictly increasing.")
    return temperatures


def _construct_scope_config(config: Any) -> tuple[bool, bool]:
    channels = _required_mapping(
        require_config_value(config, "study1.figures.power_construct_validity.channels"),
        "study1.figures.power_construct_validity.channels",
    )
    primary = channels.get("include_fp1_fp2")
    sensitivity = channels.get("compute_complementary_sensitivity")
    if not isinstance(primary, bool) or not isinstance(sensitivity, bool):
        raise ValueError("Construct channel-scope settings must be Boolean.")
    return primary, sensitivity


def _construct_channels(channels: tuple[str, ...], include_fp1_fp2: bool) -> tuple[str, ...]:
    selected = tuple(
        channel for channel in channels if include_fp1_fp2 or channel not in FRONTAL_CHANNELS
    )
    if not selected:
        raise ValueError("Construct channel selection removed every sensor.")
    return selected


def _signature_channels(channels: tuple[str, ...], config: Any) -> tuple[str, ...]:
    raw = require_config_value(config, "study1.feature_benchmark.excluded_channels")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("Feature-benchmark excluded channels must be a non-empty sequence.")
    excluded = tuple(str(channel).strip() for channel in raw)
    if not excluded or any(not channel for channel in excluded):
        raise ValueError("Feature-benchmark excluded channels must be non-empty names.")
    if len(excluded) != len(set(excluded)):
        raise ValueError("Feature-benchmark excluded channels must be unique.")
    missing = [channel for channel in excluded if channel not in channels]
    if missing:
        raise ValueError(f"Feature-benchmark excluded channels are absent from sensors: {missing}.")
    selected = tuple(channel for channel in channels if channel not in set(excluded))
    if not selected:
        raise ValueError("Feature-benchmark channel exclusion removed every sensor.")
    return selected


def _signature_model_config(config: Any) -> dict[str, float]:
    raw = _required_mapping(
        require_config_value(config, "study1.figures.sensor_topographies.signature_model"),
        "study1.figures.sensor_topographies.signature_model",
    )
    values = {
        key: float(raw[key])
        for key in (
            "rank_tolerance",
            "max_condition_number",
            "minimum_residual_standard_deviation",
        )
    }
    if not 0.0 < values["rank_tolerance"] <= 1.0:
        raise ValueError("Signature rank tolerance must be in (0, 1].")
    if values["max_condition_number"] < 1.0:
        raise ValueError("Signature maximum condition number must be at least one.")
    if values["minimum_residual_standard_deviation"] <= 0.0:
        raise ValueError("Signature minimum residual standard deviation must be positive.")
    if not np.isfinite(tuple(values.values())).all():
        raise ValueError("Signature model settings must be finite.")
    return values


def _signature_sample_requirements(config: Any) -> tuple[int, int]:
    minimum_trials = _positive_integer(
        require_config_value(
            config,
            "study1.feature_benchmark.circular_shift.min_retained_trials_per_subject",
        ),
        "Signature minimum retained trials",
    )
    minimum_runs = _positive_integer(
        require_config_value(
            config,
            "study1.feature_benchmark.circular_shift.min_valid_runs_per_subject",
        ),
        "Signature minimum valid runs",
    )
    return minimum_trials, minimum_runs


def _map_order(
    estimands: Sequence[str],
    bands: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    return tuple((estimand, band) for estimand in estimands for band in bands)


def _exclusion_frame(records: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records, columns=EXCLUSION_COLUMNS)


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


def _require_finite_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Sensor-power column {column!r} must contain finite numeric values.")


__all__ = ["ParticipantEffects", "build_construct_effects", "build_signature_effects"]
