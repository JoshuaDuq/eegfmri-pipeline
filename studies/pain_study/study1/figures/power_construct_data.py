"""Strict feature loading and global-power reconstruction for Study 1."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.tsv import read_parquet
from studies.pain_study.study1.cohort import study1_feature_table_path
from studies.pain_study.study1.figures.validity_data import (
    WITHIN_SCALE_INTENSITY_COLUMN,
    ValidityTrialData,
)

FRONTAL_CHANNELS = frozenset({"Fp1", "Fp2"})


def load_power_feature_tables(
    validity: ValidityTrialData,
    config: object,
) -> dict[str, pd.DataFrame]:
    """Read one trial-safe power table for every retained participant."""

    subjects = sorted(validity.enriched_targets["subject_id"].astype(str).unique())
    tables = {}
    for subject_id in subjects:
        feature_path = study1_feature_table_path(config, subject_id, "power")
        if not feature_path.exists():
            raise FileNotFoundError(f"Study 1 power feature table not found: {feature_path}")
        tables[subject_id] = read_parquet(feature_path)
    return tables


def reconstruct_all_subject_power(
    *,
    event_trials: pd.DataFrame,
    feature_tables: Mapping[str, pd.DataFrame],
    bands: Sequence[str],
    scopes: Sequence[tuple[str, bool]],
) -> pd.DataFrame:
    """Reconstruct global power for every retained subject and channel scope."""

    frames = []
    for subject_id in sorted(event_trials["subject_id"].astype(str).unique()):
        if subject_id not in feature_tables:
            raise ValueError(f"Power feature tables are missing retained subject {subject_id}.")
        for scope, include_fp1_fp2 in scopes:
            frames.append(
                reconstruct_global_power(
                    feature_tables[subject_id],
                    subject_id=subject_id,
                    bands=bands,
                    include_fp1_fp2=include_fp1_fp2,
                    channel_scope=scope,
                )
            )
    return pd.concat(frames, ignore_index=True)


def reconstruct_global_power(
    feature_table: pd.DataFrame,
    *,
    subject_id: str,
    bands: Sequence[str],
    include_fp1_fp2: bool,
    channel_scope: str,
) -> pd.DataFrame:
    """Reconstruct global dB power by spatially averaging linear power first."""

    _require_columns(feature_table, {"trial_id"}, "Power feature table")
    trial_ids = _integer_values(feature_table["trial_id"], "Power feature trial_id")
    if pd.Series(trial_ids).duplicated().any():
        raise ValueError(f"Power feature table contains duplicate trial IDs for {subject_id}.")
    if not str(subject_id).strip():
        raise ValueError("Power reconstruction requires a non-empty subject identifier.")
    if not str(channel_scope).strip():
        raise ValueError("Power reconstruction requires a non-empty channel scope.")

    band_columns = _discover_channel_columns(feature_table, bands)
    all_channel_sets = {tuple(sorted(columns["baseline"])) for columns in band_columns.values()}
    if len(all_channel_sets) != 1:
        raise ValueError("Power feature bands must contain identical channel sets.")
    available_channels = next(iter(all_channel_sets))
    included_channels = tuple(
        channel
        for channel in available_channels
        if include_fp1_fp2 or channel not in FRONTAL_CHANNELS
    )
    if not included_channels:
        raise ValueError("Power reconstruction channel selection removed every channel.")

    frames = [
        _reconstruct_band(
            feature_table,
            subject_id=str(subject_id),
            trial_ids=trial_ids,
            band=str(band),
            columns=band_columns[str(band)],
            included_channels=included_channels,
            include_fp1_fp2=include_fp1_fp2,
            channel_scope=str(channel_scope),
        )
        for band in bands
    ]
    return pd.concat(frames, ignore_index=True)


def retained_event_trials(validity: ValidityTrialData) -> pd.DataFrame:
    """Attach feature trial IDs and behavioral covariates to retained target trials."""

    events = validity.clean_events.copy()
    _require_columns(
        events,
        {
            "subject_id",
            "trial_id",
            "trial_number",
            "run",
            "stimulus_temp",
            "selected_surface",
            "residual_ecg_coupling",
            "fp1_fp2_high_frequency_power",
        },
        "Clean events",
    )
    retained = validity.enriched_targets[
        [
            "subject_id",
            "run",
            "within_run_trial",
            "stimulus_temp",
            "selected_surface",
            WITHIN_SCALE_INTENSITY_COLUMN,
        ]
    ].copy()
    retained["subject_id"] = retained["subject_id"].astype(str)
    event_rows = events[
        [
            "subject_id",
            "trial_id",
            "run",
            "trial_number",
            "stimulus_temp",
            "selected_surface",
            "residual_ecg_coupling",
            "fp1_fp2_high_frequency_power",
        ]
    ].rename(columns={"trial_number": "within_run_trial"})
    merged = retained.merge(
        event_rows,
        on=["subject_id", "run", "within_run_trial"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_event"),
    )
    if merged["trial_id"].isna().any():
        raise ValueError("Retained Study 1 targets do not all have matching feature trial IDs.")
    if not np.allclose(merged["stimulus_temp"], merged["stimulus_temp_event"]):
        raise ValueError("Retained target and clean-event temperatures disagree.")
    if not np.allclose(merged["selected_surface"], merged["selected_surface_event"]):
        raise ValueError("Retained target and clean-event thermode surfaces disagree.")
    output = merged[
        [
            "subject_id",
            "trial_id",
            "stimulus_temp",
            "run",
            "selected_surface",
            "within_run_trial",
            "residual_ecg_coupling",
            "fp1_fp2_high_frequency_power",
            WITHIN_SCALE_INTENSITY_COLUMN,
        ]
    ].copy()
    output["trial_id"] = _integer_values(output["trial_id"], "Retained trial_id")
    if output.duplicated(["subject_id", "trial_id"]).any():
        raise ValueError("Retained Study 1 trial IDs are not unique within participant.")
    return output


def _reconstruct_band(
    feature_table: pd.DataFrame,
    *,
    subject_id: str,
    trial_ids: np.ndarray,
    band: str,
    columns: Mapping[str, Mapping[str, str]],
    included_channels: tuple[str, ...],
    include_fp1_fp2: bool,
    channel_scope: str,
) -> pd.DataFrame:
    baseline_names = [columns["baseline"][channel] for channel in included_channels]
    logratio_names = [columns["logratio"][channel] for channel in included_channels]
    baseline = feature_table[baseline_names].to_numpy(dtype=float)
    logratio = feature_table[logratio_names].to_numpy(dtype=float)
    if not np.isfinite(baseline).all() or np.any(baseline <= 0.0):
        raise ValueError(
            f"Baseline channel power must be finite and positive for {subject_id}/{band}."
        )
    if not np.isfinite(logratio).all():
        raise ValueError(f"Channel log-ratio power must be finite for {subject_id}/{band}.")
    active = baseline * np.power(10.0, logratio)
    if not np.isfinite(active).all() or np.any(active <= 0.0):
        raise ValueError(
            f"Reconstructed active power must be finite and positive for {subject_id}/{band}."
        )
    global_power_db = 10.0 * np.log10(active.mean(axis=1) / baseline.mean(axis=1))
    if not np.isfinite(global_power_db).all():
        raise ValueError(f"Global dB power is non-finite for {subject_id}/{band}.")
    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "trial_id": trial_ids,
            "band": band,
            "channel_scope": channel_scope,
            "include_fp1_fp2": include_fp1_fp2,
            "n_channels": len(included_channels),
            "included_channels": ",".join(included_channels),
            "global_power_db": global_power_db,
        }
    )


def _discover_channel_columns(
    feature_table: pd.DataFrame,
    bands: Sequence[str],
) -> dict[str, dict[str, dict[str, str]]]:
    discovered = {str(band): {"baseline": {}, "logratio": {}} for band in bands}
    for column in feature_table.columns:
        parsed = NamingSchema.parse(str(column))
        if not parsed.get("valid") or parsed.get("group") != "power":
            continue
        band = str(parsed.get("band"))
        if band not in discovered or parsed.get("scope") != "ch":
            continue
        channel = str(parsed.get("identifier"))
        segment = str(parsed.get("segment"))
        statistic = str(parsed.get("stat"))
        if segment == "baseline" and statistic == "mean":
            discovered[band]["baseline"][channel] = str(column)
        elif segment == "active" and statistic == "logratio":
            discovered[band]["logratio"][channel] = str(column)
    for band, columns in discovered.items():
        baseline_channels = set(columns["baseline"])
        logratio_channels = set(columns["logratio"])
        if not baseline_channels or not logratio_channels:
            raise ValueError(f"Power feature table is missing configured band {band!r}.")
        if baseline_channels != logratio_channels:
            raise ValueError(
                f"Baseline and active log-ratio channel sets differ for band {band!r}."
            )
    return discovered


def _integer_values(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.allclose(numeric, np.round(numeric)):
        raise ValueError(f"{label} must contain finite integer values.")
    return np.round(numeric).astype(int)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}.")


__all__ = [
    "load_power_feature_tables",
    "reconstruct_all_subject_power",
    "reconstruct_global_power",
    "retained_event_trials",
]
