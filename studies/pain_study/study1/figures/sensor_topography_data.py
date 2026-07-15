"""Strict trial-level sensor-power reconstruction for Study 1 topographies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS
from studies.pain_study.study1.figures.power_construct_data import (
    load_power_feature_tables,
    retained_event_trials,
)
from studies.pain_study.study1.figures.validity_data import (
    WITHIN_SCALE_INTENSITY_COLUMN,
    ValidityTrialData,
    load_validity_trial_data,
)
from studies.pain_study.study1.targets import resolve_target_residualization_columns

EXPECTED_BANDS = tuple(PRIMARY_BAND_PRESETS["alpha_beta_gamma"])
POWER_SEGMENT_STATISTICS = {
    ("baseline", "mean"): "baseline",
    ("active", "logratio"): "logratio",
    ("active", "db"): "db",
}
EVENT_COLUMNS = (
    "stimulus_temp",
    "run",
    "selected_surface",
    "within_run_trial",
    "residual_ecg_coupling",
    "fp1_fp2_high_frequency_power",
    WITHIN_SCALE_INTENSITY_COLUMN,
)
TARGET_COLUMNS = ("NPS", "SIIPS1")
TARGET_KEY_COLUMNS = ("subject_id", "run", "within_run_trial")


@dataclass(frozen=True)
class SensorPowerData:
    """Validated tidy sensor-power trials and their deterministic dimensions."""

    trials: pd.DataFrame
    subjects: tuple[str, ...]
    bands: tuple[str, ...]
    channels: tuple[str, ...]


@dataclass(frozen=True)
class SensorMontage:
    """Analyzed EEG channels and their exact configured montage coordinates."""

    name: str
    channels: tuple[str, ...]
    positions_3d: tuple[tuple[float, float, float], ...]
    positions_xy: tuple[tuple[float, float], ...]


def load_sensor_power_data(*, task: str, config: Any) -> SensorPowerData:
    """Load retained Study 1 inputs and reconstruct their channel-level power."""

    validity = load_validity_trial_data(task=task, config=config)
    return build_sensor_power_data(
        validity=validity,
        feature_tables=load_power_feature_tables(validity, config),
        config=config,
    )


def build_sensor_power_data(
    *,
    validity: ValidityTrialData,
    feature_tables: Mapping[str, pd.DataFrame],
    config: Any,
) -> SensorPowerData:
    """Build the immutable boundary from validated in-memory Study 1 inputs."""

    bands = _configured_bands(config)
    event_targets = _retained_event_targets(validity, config)
    subjects = tuple(sorted(event_targets["subject_id"].astype(str).unique()))
    normalized_tables = _normalized_feature_tables(feature_tables)
    if set(normalized_tables) != set(subjects):
        raise ValueError(
            "Power feature-table subjects do not exactly match retained Study 1 subjects: "
            f"retained={list(subjects)}, features={sorted(normalized_tables)}."
        )

    power_frames: list[pd.DataFrame] = []
    expected_channels: tuple[str, ...] | None = None
    for subject_id in subjects:
        subject_power = reconstruct_channel_power(
            normalized_tables[subject_id],
            subject_id=subject_id,
            bands=bands,
        )
        _require_exact_trial_alignment(subject_power, event_targets, subject_id=subject_id)
        channels = tuple(subject_power["channel"].drop_duplicates())
        if expected_channels is None:
            expected_channels = channels
        elif channels != expected_channels:
            raise ValueError(
                "Power feature tables must contain identical channel sets across subjects: "
                f"expected={list(expected_channels)}, {subject_id}={list(channels)}."
            )
        power_frames.append(subject_power)

    if expected_channels is None:
        raise ValueError("Sensor-power reconstruction requires at least one retained subject.")
    power = pd.concat(power_frames, ignore_index=True)
    trials = power.merge(
        event_targets,
        on=["subject_id", "trial_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not trials["_merge"].eq("both").all():
        raise ValueError("Sensor power does not align exactly with retained event and target rows.")
    trials = trials.drop(columns="_merge")
    expected_rows = len(event_targets) * len(bands) * len(expected_channels)
    if len(trials) != expected_rows:
        raise ValueError(
            "Sensor-power row count does not match the retained trial grid: "
            f"expected={expected_rows}, observed={len(trials)}."
        )
    trials = _sort_sensor_trials(trials, bands=bands, channels=expected_channels)
    return SensorPowerData(
        trials=trials,
        subjects=subjects,
        bands=bands,
        channels=expected_channels,
    )


def reconstruct_channel_power(
    feature_table: pd.DataFrame,
    *,
    subject_id: str,
    bands: Sequence[str],
) -> pd.DataFrame:
    """Reconstruct active-versus-baseline dB power for every trial and channel."""

    configured_bands = _require_exact_bands(bands)
    subject = str(subject_id).strip()
    if not subject:
        raise ValueError("Channel-power reconstruction requires a non-empty subject identifier.")
    _require_unique_columns(feature_table)
    _require_columns(feature_table, ("trial_id",), table_name="Power feature table")
    trial_ids = _integer_values(feature_table["trial_id"], "Power feature trial_id")
    if pd.Series(trial_ids).duplicated().any():
        raise ValueError(f"Power feature table contains duplicate trial IDs for {subject}.")

    columns = _discover_channel_power_columns(feature_table, configured_bands)
    channels = _require_identical_band_channels(columns)
    frames = [
        _reconstruct_band_channels(
            feature_table,
            subject_id=subject,
            trial_ids=trial_ids,
            band=band,
            channels=channels,
            columns=columns[band],
        )
        for band in configured_bands
    ]
    reconstructed = pd.concat(frames, ignore_index=True)
    return _sort_sensor_trials(reconstructed, bands=configured_bands, channels=channels)


def load_sensor_montage(channels: Sequence[str], config: Any) -> SensorMontage:
    """Resolve exact configured montage positions for analyzed EEG channels only."""

    analyzed_channels = _validated_channels(channels)
    montage_name = str(require_config_value(config, "preprocessing.montage")).strip()
    montage = mne.channels.make_standard_montage(montage_name)
    channel_positions = montage.get_positions()["ch_pos"]
    missing = [channel for channel in analyzed_channels if channel not in channel_positions]
    if missing:
        raise ValueError(
            f"Analyzed EEG channels are absent from montage {montage_name!r}: {missing}."
        )

    positions_3d = np.asarray(
        [channel_positions[channel] for channel in analyzed_channels],
        dtype=float,
    )
    if positions_3d.shape != (len(analyzed_channels), 3):
        raise ValueError("Analyzed montage channels must have one 3D position each.")
    if not np.isfinite(positions_3d).all() or len(np.unique(positions_3d, axis=0)) != len(
        analyzed_channels
    ):
        raise ValueError("Analyzed montage channels require finite unique 3D positions.")

    positions_xy = positions_3d[:, :2]
    if len(np.unique(positions_xy, axis=0)) != len(analyzed_channels):
        raise ValueError("Analyzed montage channels require unique top-view x-y positions.")
    return SensorMontage(
        name=montage_name,
        channels=analyzed_channels,
        positions_3d=tuple(tuple(float(value) for value in row) for row in positions_3d),
        positions_xy=tuple(tuple(float(value) for value in row) for row in positions_xy),
    )


def _configured_bands(config: Any) -> tuple[str, ...]:
    specifications = require_config_value(config, "study1.figures.sensor_topographies.bands")
    if not isinstance(specifications, (list, tuple)):
        raise ValueError("Sensor-topography bands must be an ordered list of mappings.")
    try:
        bands = tuple(str(specification["name"]).strip() for specification in specifications)
    except (KeyError, TypeError) as error:
        raise ValueError("Every sensor-topography band must define a name.") from error
    return _require_exact_bands(bands)


def _require_exact_bands(bands: Sequence[str]) -> tuple[str, ...]:
    if isinstance(bands, (str, bytes)):
        raise ValueError("Sensor-topography bands must exactly match alpha_beta_gamma in order.")
    normalized = tuple(str(band).strip() for band in bands)
    if normalized != EXPECTED_BANDS:
        raise ValueError(
            "Sensor-topography bands must exactly match alpha_beta_gamma in order: "
            f"expected={list(EXPECTED_BANDS)}, observed={list(normalized)}."
        )
    return normalized


def _discover_channel_power_columns(
    feature_table: pd.DataFrame,
    bands: tuple[str, ...],
) -> dict[str, dict[str, dict[str, str]]]:
    discovered = {band: {"baseline": {}, "logratio": {}, "db": {}} for band in bands}
    for raw_column in feature_table.columns:
        column = str(raw_column)
        parsed = NamingSchema.parse(column)
        if not parsed.get("valid"):
            if _is_power_column_name(column):
                raise ValueError(f"Malformed power feature column: {column!r}.")
            continue
        if parsed.get("group") != "power" or parsed.get("scope") != "ch":
            continue

        band = str(parsed["band"])
        if band not in discovered:
            raise ValueError(f"Channel-power column uses unexpected band {band!r}: {column!r}.")
        segment_statistic = (str(parsed["segment"]), str(parsed["stat"]))
        kind = POWER_SEGMENT_STATISTICS.get(segment_statistic)
        if kind is None:
            raise ValueError(f"Unsupported channel-power column: {column!r}.")
        channel = str(parsed["identifier"])
        if not channel.strip() or channel != channel.strip():
            raise ValueError(f"Invalid channel identifier in power column: {column!r}.")
        if channel in discovered[band][kind]:
            raise ValueError(
                f"Duplicate {kind} channel-power columns for band {band!r}, channel {channel!r}."
            )
        discovered[band][kind][channel] = column

    for band, band_columns in discovered.items():
        baseline_channels = set(band_columns["baseline"])
        logratio_channels = set(band_columns["logratio"])
        if not baseline_channels or not logratio_channels:
            raise ValueError(f"Power feature table is missing configured band {band!r}.")
        if baseline_channels != logratio_channels:
            raise ValueError(
                f"Baseline and active log-ratio channel sets differ for band {band!r}."
            )
        db_channels = set(band_columns["db"])
        if db_channels and db_channels != baseline_channels:
            raise ValueError(f"Emitted dB and reconstructed channel sets differ for band {band!r}.")
    db_presence = {bool(band_columns["db"]) for band_columns in discovered.values()}
    if len(db_presence) != 1:
        raise ValueError("Emitted channel dB columns must be present for every configured band.")
    return discovered


def _require_identical_band_channels(
    columns: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> tuple[str, ...]:
    channel_sets = {tuple(sorted(band_columns["baseline"])) for band_columns in columns.values()}
    if len(channel_sets) != 1:
        raise ValueError("Power feature bands must contain identical channel sets.")
    return next(iter(channel_sets))


def _reconstruct_band_channels(
    feature_table: pd.DataFrame,
    *,
    subject_id: str,
    trial_ids: np.ndarray,
    band: str,
    channels: tuple[str, ...],
    columns: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    baseline_names = [columns["baseline"][channel] for channel in channels]
    logratio_names = [columns["logratio"][channel] for channel in channels]
    baseline = _numeric_matrix(feature_table, baseline_names)
    logratio = _numeric_matrix(feature_table, logratio_names)
    if not np.isfinite(baseline).all() or np.any(baseline <= 0.0):
        raise ValueError(
            f"Baseline channel power must be finite and positive for {subject_id}/{band}."
        )
    if not np.isfinite(logratio).all():
        raise ValueError(f"Channel log-ratio power must be finite for {subject_id}/{band}.")
    if columns["db"]:
        emitted_db_names = [columns["db"][channel] for channel in channels]
        emitted_db = _numeric_matrix(feature_table, emitted_db_names)
        if not np.isfinite(emitted_db).all():
            raise ValueError(f"Emitted channel dB power must be finite for {subject_id}/{band}.")
        expected_db = 10.0 * logratio
        if not np.allclose(emitted_db, expected_db, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Emitted channel dB power does not equal 10 * logratio for {subject_id}/{band}."
            )
    with np.errstate(over="ignore", invalid="ignore"):
        active = baseline * np.power(10.0, logratio)
    if not np.isfinite(active).all() or np.any(active <= 0.0):
        raise ValueError(
            f"Reconstructed active power must be finite and positive for {subject_id}/{band}."
        )
    power_db = 10.0 * np.log10(active / baseline)
    if not np.isfinite(power_db).all():
        raise ValueError(f"Channel dB power must be finite for {subject_id}/{band}.")

    trial_grid = np.repeat(trial_ids, len(channels))
    channel_grid = np.tile(channels, len(trial_ids))
    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "trial_id": trial_grid,
            "band": band,
            "channel": channel_grid,
            "power_db": power_db.reshape(-1),
        }
    )


def _retained_event_targets(validity: ValidityTrialData, config: Any) -> pd.DataFrame:
    events = retained_event_trials(validity)
    targets = validity.enriched_targets.copy()
    _require_columns(targets, (*TARGET_KEY_COLUMNS, *TARGET_COLUMNS), table_name="Target table")
    targets["subject_id"] = targets["subject_id"].astype(str)
    for column in ("run", "within_run_trial"):
        targets[column] = _integer_values(targets[column], f"Target {column}")
    if targets.duplicated(list(TARGET_KEY_COLUMNS)).any():
        raise ValueError("Retained Study 1 target rows contain duplicate trial keys.")

    nuisance_columns = _target_nuisance_columns(targets, config)
    appended_columns = tuple(
        dict.fromkeys(
            column
            for column in (*TARGET_COLUMNS, *nuisance_columns)
            if column not in TARGET_KEY_COLUMNS and column not in events.columns
        )
    )
    target_values = targets[[*TARGET_KEY_COLUMNS, *appended_columns]]
    merged = events.merge(
        target_values,
        on=list(TARGET_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Retained event trials do not align exactly with prepared target rows.")
    merged = merged.drop(columns="_merge")
    _require_finite_columns(merged, (*EVENT_COLUMNS, *TARGET_COLUMNS, *nuisance_columns))
    return merged


def _target_nuisance_columns(targets: pd.DataFrame, config: Any) -> tuple[str, ...]:
    columns: list[str] = []
    for target in TARGET_COLUMNS:
        resolved = resolve_target_residualization_columns(
            frame=targets,
            config=config,
            target_name=target,
        )
        for column in resolved:
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def _require_exact_trial_alignment(
    power: pd.DataFrame,
    event_targets: pd.DataFrame,
    *,
    subject_id: str,
) -> None:
    power_ids = tuple(sorted(power["trial_id"].unique()))
    retained_ids = tuple(
        sorted(event_targets.loc[event_targets["subject_id"].eq(subject_id), "trial_id"].unique())
    )
    if power_ids != retained_ids:
        raise ValueError(
            f"Power feature trial IDs do not exactly match retained trials for {subject_id}: "
            f"retained={list(retained_ids)}, features={list(power_ids)}."
        )


def _sort_sensor_trials(
    trials: pd.DataFrame,
    *,
    bands: tuple[str, ...],
    channels: tuple[str, ...],
) -> pd.DataFrame:
    ordered = trials.copy()
    ordered["band"] = pd.Categorical(ordered["band"], categories=bands, ordered=True)
    ordered["channel"] = pd.Categorical(ordered["channel"], categories=channels, ordered=True)
    ordered = ordered.sort_values(
        ["subject_id", "trial_id", "band", "channel"],
        kind="stable",
    ).reset_index(drop=True)
    ordered["band"] = ordered["band"].astype(str)
    ordered["channel"] = ordered["channel"].astype(str)
    return ordered


def _normalized_feature_tables(
    feature_tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    normalized: dict[str, pd.DataFrame] = {}
    for raw_subject, table in feature_tables.items():
        subject = str(raw_subject).strip()
        if not subject:
            raise ValueError("Power feature-table subject identifiers must be non-empty.")
        if subject in normalized:
            raise ValueError(f"Duplicate normalized power feature-table subject: {subject}.")
        normalized[subject] = table
    return normalized


def _validated_channels(channels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(channels, (str, bytes)) or not channels:
        raise ValueError("Analyzed EEG channels must be a non-empty ordered sequence.")
    if any(not isinstance(channel, str) or not channel.strip() for channel in channels):
        raise ValueError("Analyzed EEG channels must contain non-empty string names.")
    normalized = tuple(channels)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Analyzed EEG channels must be unique.")
    return normalized


def _numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _integer_values(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.round(numeric)):
        raise ValueError(f"{label} must contain finite integer values.")
    return np.round(numeric).astype(int)


def _require_finite_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in dict.fromkeys(columns):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Retained Study 1 column {column!r} must contain finite values.")


def _require_unique_columns(frame: pd.DataFrame) -> None:
    duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicates:
        raise ValueError(f"Power feature table contains duplicate columns: {duplicates}.")


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    table_name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}.")


def _is_power_column_name(column: str) -> bool:
    return column == "power" or column.startswith("power_")


__all__ = [
    "SensorMontage",
    "SensorPowerData",
    "build_sensor_power_data",
    "load_sensor_montage",
    "load_sensor_power_data",
    "reconstruct_channel_power",
]
