"""Orchestration for the Study 1 EEG power construct-validity figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS
from studies.pain_study.study1.figures.power_construct_data import (
    load_power_feature_tables,
    reconstruct_all_subject_power,
    retained_event_trials,
)
from studies.pain_study.study1.figures.power_construct_models import (
    build_rating_association,
    build_temperature_association,
)
from studies.pain_study.study1.figures.validity_data import (
    ValidityTrialData,
    load_validity_trial_data,
)

FIGURE_CONFIG_KEY = "study1.figures.power_construct_validity"


@dataclass(frozen=True)
class PowerConstructValiditySummary:
    """Plot-ready primary estimates and complete reproducibility audits."""

    trials: pd.DataFrame
    temperature_by_subject: pd.DataFrame
    temperature_summary: pd.DataFrame
    rating_by_subject: pd.DataFrame
    rating_summary: pd.DataFrame
    sensitivity_by_subject: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    bands: tuple[str, ...]
    temperatures: tuple[float, ...]
    primary_include_fp1_fp2: bool
    n_subjects: int
    article_ready: bool


def load_power_construct_validity_summary(
    *,
    task: str,
    config: Any,
) -> PowerConstructValiditySummary:
    """Load retained trials and feature tables, then build every estimand."""

    validity = load_validity_trial_data(task=task, config=config)
    return build_power_construct_validity_summary(
        validity=validity,
        feature_tables=load_power_feature_tables(validity, config),
        config=config,
    )


def build_power_construct_validity_summary(
    *,
    validity: ValidityTrialData,
    feature_tables: Mapping[str, pd.DataFrame],
    config: Any,
) -> PowerConstructValiditySummary:
    """Build primary and complementary estimands from validated in-memory inputs."""

    figure_config = _figure_config(config)
    bands = tuple(str(spec["name"]) for spec in figure_config["bands"])
    if list(bands) != PRIMARY_BAND_PRESETS["alpha_beta_gamma"]:
        raise ValueError("Power construct-validity bands must match alpha_beta_gamma exactly.")
    temperatures = tuple(
        float(value)
        for value in require_config_value(config, "study1.figures.validity.temperatures")
    )
    primary_include = bool(figure_config["channels"]["include_fp1_fp2"])
    scopes = [("primary", primary_include)]
    if bool(figure_config["channels"]["compute_complementary_sensitivity"]):
        scopes.append(("complementary", not primary_include))

    events = retained_event_trials(validity)
    power = reconstruct_all_subject_power(
        event_trials=events,
        feature_tables=feature_tables,
        bands=bands,
        scopes=scopes,
    )
    trials = power.merge(
        events,
        on=["subject_id", "trial_id"],
        how="inner",
        validate="many_to_one",
    )
    expected_rows = len(events) * len(bands) * len(scopes)
    if len(trials) != expected_rows:
        raise ValueError(
            "Power features do not align one-to-one with retained Study 1 trials: "
            f"expected={expected_rows}, observed={len(trials)}."
        )

    primary_trials = trials.loc[trials["channel_scope"].eq("primary")].copy()
    temperature_by_subject, temperature_summary = build_temperature_association(
        primary_trials,
        bands=bands,
        temperatures=temperatures,
        config=config,
    )
    rating_by_subject, rating_summary = build_rating_association(
        primary_trials,
        bands=bands,
        config=config,
    )
    sensitivity_by_subject, sensitivity_summary = _build_sensitivity(
        trials,
        bands=bands,
        temperatures=temperatures,
        config=config,
    )
    n_subjects = int(primary_trials["subject_id"].nunique())
    minimum_article_subjects = int(figure_config["minimum_article_subjects"])
    if minimum_article_subjects < 2:
        raise ValueError("minimum_article_subjects must be at least two.")
    return PowerConstructValiditySummary(
        trials=trials,
        temperature_by_subject=temperature_by_subject,
        temperature_summary=temperature_summary,
        rating_by_subject=rating_by_subject,
        rating_summary=rating_summary,
        sensitivity_by_subject=sensitivity_by_subject,
        sensitivity_summary=sensitivity_summary,
        bands=bands,
        temperatures=temperatures,
        primary_include_fp1_fp2=primary_include,
        n_subjects=n_subjects,
        article_ready=n_subjects >= minimum_article_subjects,
    )


def _build_sensitivity(
    trials: pd.DataFrame,
    *,
    bands: Sequence[str],
    temperatures: Sequence[float],
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = tuple(sorted(trials["channel_scope"].astype(str).unique()))
    if scopes == ("primary",):
        return pd.DataFrame(), pd.DataFrame()
    if set(scopes) != {"complementary", "primary"}:
        raise ValueError(f"Unexpected Fp1/Fp2 sensitivity scopes: {scopes}.")

    participant_frames = []
    summary_frames = []
    for scope in scopes:
        scoped = trials.loc[trials["channel_scope"].eq(scope)].copy()
        temperature, _cohort = build_temperature_association(
            scoped,
            bands=bands,
            temperatures=temperatures,
            config=config,
        )
        rating, _cohort = build_rating_association(scoped, bands=bands, config=config)
        slopes = temperature[["subject_id", "band", "temperature_slope_db_per_c"]].drop_duplicates()
        combined = slopes.merge(
            rating[["subject_id", "band", "estimable", "partial_r"]],
            on=["subject_id", "band"],
            validate="one_to_one",
        )
        combined["channel_scope"] = scope
        combined["include_fp1_fp2"] = bool(scoped["include_fp1_fp2"].iloc[0])
        participant_frames.append(combined)
        summary_frames.append(
            combined.groupby("band", sort=False, as_index=False)
            .agg(
                mean_temperature_slope_db_per_c=("temperature_slope_db_per_c", "mean"),
                mean_partial_r=("partial_r", "mean"),
                n_subjects=("subject_id", "nunique"),
                n_rating_estimable=("estimable", "sum"),
            )
            .assign(channel_scope=scope)
        )
    return (
        _wide_sensitivity_participants(pd.concat(participant_frames, ignore_index=True)),
        _wide_sensitivity_summary(pd.concat(summary_frames, ignore_index=True)),
    )


def _wide_sensitivity_participants(participant: pd.DataFrame) -> pd.DataFrame:
    wide = participant.pivot(
        index=["subject_id", "band"],
        columns="channel_scope",
        values=["temperature_slope_db_per_c", "partial_r"],
    )
    wide[("temperature_slope_difference", "value")] = (
        wide[("temperature_slope_db_per_c", "primary")]
        - wide[("temperature_slope_db_per_c", "complementary")]
    )
    wide[("partial_r_difference", "value")] = (
        wide[("partial_r", "primary")] - wide[("partial_r", "complementary")]
    )
    return _flatten_columns(wide.reset_index())


def _wide_sensitivity_summary(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(index="band", columns="channel_scope")
    wide[("mean_temperature_slope_difference", "value")] = (
        wide[("mean_temperature_slope_db_per_c", "primary")]
        - wide[("mean_temperature_slope_db_per_c", "complementary")]
    )
    wide[("mean_partial_r_difference", "value")] = (
        wide[("mean_partial_r", "primary")] - wide[("mean_partial_r", "complementary")]
    )
    return _flatten_columns(wide.reset_index())


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [
        (
            "_".join(str(part) for part in column if str(part) not in {"", "value"})
            if isinstance(column, tuple)
            else str(column)
        )
        for column in output.columns
    ]
    return output


def _figure_config(config: Any) -> Mapping[str, Any]:
    value = require_config_value(config, FIGURE_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIGURE_CONFIG_KEY} must be a mapping.")
    return value


__all__ = [
    "PowerConstructValiditySummary",
    "build_power_construct_validity_summary",
    "load_power_construct_validity_summary",
]
