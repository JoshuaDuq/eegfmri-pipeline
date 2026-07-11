"""Validated inputs and first-level event designs for fMRI construct validity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import require_config_value
from eeg_pipeline.utils.config.roots import (
    resolve_fmri_bids_root,
    resolve_fmri_deriv_root,
)
from fmri_pipeline.utils.bold_discovery import (
    discover_brain_mask_for_bold,
    discover_fmriprep_preproc_bold,
)
from studies.pain_study.study1.figures.validity_data import (
    WITHIN_SCALE_INTENSITY_COLUMN,
    load_validity_trial_data,
)

MODEL_EVENT_COLUMNS = ("onset", "duration", "trial_type", "modulation")
RAW_REQUIRED_COLUMNS = (
    "run_id",
    "trial_number",
    "onset",
    "duration",
    "trial_type",
    "stim_phase",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
)
RETAINED_REQUIRED_COLUMNS = (
    "subject_id",
    "run",
    "within_run_trial",
    "stimulus_temp",
    "selected_surface",
    "pain_binary_coded",
    "vas_final_coded_rating",
    WITHIN_SCALE_INTENSITY_COLUMN,
)
TRIAL_KEYS = ("run", "within_run_trial")


@dataclass(frozen=True)
class FmriRunInput:
    """One retained participant run and its required fMRI inputs."""

    subject_id: str
    run: int
    bold_path: Path
    mask_path: Path
    confounds_path: Path
    raw_events_path: Path
    raw_events: pd.DataFrame
    retained_trials: pd.DataFrame


@dataclass(frozen=True)
class FirstLevelRunDesign:
    """One explicit run-level event design for one estimand."""

    subject_id: str
    run: int
    estimand: str
    target_column: str
    events: pd.DataFrame
    audit: Mapping[str, object]


def load_fmri_run_inputs(*, task: str, config: Any) -> tuple[FmriRunInput, ...]:
    """Resolve the retained cohort to complete MNI fMRIPrep run inputs."""

    validity = load_validity_trial_data(task=task, config=config)
    retained = validity.enriched_targets.copy()
    _require_columns(retained, RETAINED_REQUIRED_COLUMNS, "retained Study 1 trials")
    retained["subject_id"] = retained["subject_id"].astype(str)
    retained["run"] = _integer_series(retained, "run", "retained Study 1 trials")
    retained["within_run_trial"] = _integer_series(
        retained,
        "within_run_trial",
        "retained Study 1 trials",
    )
    _validate_configured_temperatures(retained, config)

    bids_root = resolve_fmri_bids_root(config)
    deriv_root = resolve_fmri_deriv_root(config)
    space = str(require_config_value(config, "study1.targets.fmriprep_space"))
    inputs: list[FmriRunInput] = []
    for (subject_id, run), trials in retained.groupby(
        ["subject_id", "run"],
        sort=True,
    ):
        raw_events_path = _raw_events_path(
            bids_root=bids_root,
            subject_id=str(subject_id),
            task=task,
            run=int(run),
        )
        bold_path = discover_fmriprep_preproc_bold(
            deriv_root,
            str(subject_id),
            task,
            int(run),
            space=space,
        )
        if bold_path is None:
            raise FileNotFoundError(
                "Study 1 fMRI construct validity requires an MNI fMRIPrep BOLD image for "
                f"{subject_id}, run {int(run):02d}."
            )
        mask_path = discover_brain_mask_for_bold(bold_path)
        if mask_path is None:
            raise FileNotFoundError(f"Missing fMRIPrep brain mask for {bold_path}.")
        confounds_path = _confounds_path_for_bold(bold_path)
        inputs.append(
            FmriRunInput(
                subject_id=str(subject_id),
                run=int(run),
                bold_path=bold_path,
                mask_path=mask_path,
                confounds_path=confounds_path,
                raw_events_path=raw_events_path,
                raw_events=pd.read_csv(raw_events_path, sep="\t"),
                retained_trials=trials.reset_index(drop=True),
            )
        )
    if not inputs:
        raise ValueError("Study 1 fMRI construct validity resolved no retained runs.")
    return tuple(inputs)


def build_subject_designs(
    runs: Sequence[FmriRunInput],
) -> tuple[FirstLevelRunDesign, ...]:
    """Build temperature and rating designs across all runs for one participant."""

    if not runs:
        raise ValueError("First-level fMRI design construction requires at least one run.")
    subjects = {run.subject_id for run in runs}
    if len(subjects) != 1:
        raise ValueError("First-level fMRI designs must contain exactly one participant.")
    run_numbers = [run.run for run in runs]
    if len(set(run_numbers)) != len(run_numbers):
        raise ValueError("First-level fMRI designs contain duplicate runs.")

    raw = pd.concat([run.raw_events for run in runs], ignore_index=True)
    retained = pd.concat([run.retained_trials for run in runs], ignore_index=True)
    temperature = build_temperature_events(raw, retained)
    rating = build_rating_events(raw, retained)
    subject_id = next(iter(subjects))
    designs: list[FirstLevelRunDesign] = []
    for estimand, target, events in (
        ("temperature", "temperature_linear", temperature),
        ("rating", "rating_within_temperature", rating),
    ):
        for run in sorted(runs, key=lambda item: item.run):
            run_events = events.loc[events["run"].eq(run.run)].copy()
            if run_events.empty:
                raise ValueError(f"The {estimand} design is empty for run {run.run}.")
            model_events = run_events.loc[:, MODEL_EVENT_COLUMNS].reset_index(drop=True)
            designs.append(
                FirstLevelRunDesign(
                    subject_id=subject_id,
                    run=run.run,
                    estimand=estimand,
                    target_column=target,
                    events=model_events,
                    audit={
                        "n_events": len(model_events),
                        "n_retained_trials": len(run.retained_trials),
                        "source_events_path": str(run.raw_events_path),
                    },
                )
            )
    return tuple(designs)


def build_temperature_events(
    raw_events: pd.DataFrame,
    retained_trials: pd.DataFrame,
) -> pd.DataFrame:
    """Build the total delivered-temperature first-level event design."""

    raw, retained, plateau = _validated_event_inputs(raw_events, retained_trials)
    plateau = plateau.copy()
    plateau["temperature_modulation"] = plateau.groupby("run")["stimulus_temp"].transform(
        lambda values: values - values.mean()
    )
    plateau["order_modulation"] = plateau.groupby("run")["within_run_trial"].transform(
        lambda values: values - values.mean()
    )
    modeled = [
        _event_rows(plateau, trial_type="plateau_mean", modulation=1.0),
        _event_rows(
            plateau,
            trial_type="temperature_linear",
            modulation=plateau["temperature_modulation"],
        ),
        _event_rows(
            plateau,
            trial_type="within_run_trial_order",
            modulation=plateau["order_modulation"],
        ),
        _nuisance_events(raw, retained),
    ]
    return _finalize_events(pd.concat(modeled, ignore_index=True))


def build_rating_events(
    raw_events: pd.DataFrame,
    retained_trials: pd.DataFrame,
) -> pd.DataFrame:
    """Build the within-temperature subjective-intensity event design."""

    raw, retained, plateau = _validated_event_inputs(raw_events, retained_trials)
    plateau = plateau.copy()
    centered_rating = plateau[WITHIN_SCALE_INTENSITY_COLUMN] - plateau.groupby("stimulus_temp")[
        WITHIN_SCALE_INTENSITY_COLUMN
    ].transform("mean")
    if not np.any(np.abs(centered_rating.to_numpy(dtype=float)) > 1e-12):
        raise ValueError("The rating design requires non-zero within-temperature rating variation.")
    plateau["rating_modulation"] = centered_rating / 10.0
    plateau["order_modulation"] = plateau.groupby("run")["within_run_trial"].transform(
        lambda values: values - values.mean()
    )
    modeled = [
        _event_rows(
            plateau,
            trial_type="rating_within_temperature",
            modulation=plateau["rating_modulation"],
        ),
        _event_rows(
            plateau,
            trial_type="within_run_trial_order",
            modulation=plateau["order_modulation"],
        ),
        _nuisance_events(raw, retained),
    ]
    for temperature, rows in plateau.groupby("stimulus_temp", sort=True):
        modeled.append(
            _event_rows(rows, trial_type=f"temperature_{float(temperature):g}", modulation=1.0)
        )
    return _finalize_events(pd.concat(modeled, ignore_index=True))


def _validated_event_inputs(
    raw_events: pd.DataFrame,
    retained_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(raw_events, RAW_REQUIRED_COLUMNS, "raw fMRI events")
    _require_columns(retained_trials, RETAINED_REQUIRED_COLUMNS, "retained Study 1 trials")
    raw = raw_events.copy()
    retained = retained_trials.copy()
    raw["run"] = _integer_series(raw, "run_id", "raw fMRI events")
    raw["within_run_trial"] = pd.to_numeric(raw["trial_number"], errors="raise")
    retained["run"] = _integer_series(retained, "run", "retained Study 1 trials")
    retained["within_run_trial"] = _integer_series(
        retained,
        "within_run_trial",
        "retained Study 1 trials",
    )
    if retained.duplicated(list(TRIAL_KEYS)).any():
        raise ValueError("Study 1 fMRI design contains duplicate retained trial keys.")

    plateau_mask = raw["trial_type"].astype(str).eq("stimulation") & raw["stim_phase"].astype(
        str
    ).eq("plateau")
    raw_plateau = raw.loc[plateau_mask].copy()
    raw_plateau["within_run_trial"] = _integer_series(
        raw_plateau,
        "within_run_trial",
        "raw fMRI plateau events",
    )
    if raw_plateau.duplicated(list(TRIAL_KEYS)).any():
        raise ValueError("Raw fMRI events contain duplicate plateau trial keys.")
    surfaces = raw_plateau.groupby("run")["selected_surface"].nunique(dropna=False)
    if (surfaces != 1).any():
        raise ValueError("Study 1 fMRI designs require one thermode surface per run.")

    retained_columns = [
        *TRIAL_KEYS,
        "stimulus_temp",
        "selected_surface",
        "pain_binary_coded",
        "vas_final_coded_rating",
        WITHIN_SCALE_INTENSITY_COLUMN,
    ]
    plateau = retained[retained_columns].merge(
        raw_plateau[
            [
                *TRIAL_KEYS,
                "onset",
                "duration",
                "stimulus_temp",
                "selected_surface",
                "pain_binary_coded",
                "vas_final_coded_rating",
            ]
        ],
        on=list(TRIAL_KEYS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_raw"),
        indicator=True,
    )
    unmatched = plateau["_merge"].ne("both")
    if unmatched.any():
        keys = plateau.loc[unmatched, list(TRIAL_KEYS)].to_dict("records")
        raise ValueError(
            "Retained Study 1 trials were found without matching raw fMRI plateau events: "
            f"{keys}."
        )
    for column in (
        "stimulus_temp",
        "selected_surface",
        "pain_binary_coded",
        "vas_final_coded_rating",
    ):
        retained_values = pd.to_numeric(plateau[column], errors="raise").to_numpy(dtype=float)
        raw_values = pd.to_numeric(plateau[f"{column}_raw"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(retained_values).all() or not np.isfinite(raw_values).all():
            raise ValueError(f"Study 1 fMRI trial column '{column}' must be finite.")
        if not np.allclose(retained_values, raw_values, rtol=0.0, atol=1e-9):
            raise ValueError(f"Retained and raw fMRI trial values disagree for column '{column}'.")
    within_scale = pd.to_numeric(
        plateau[WITHIN_SCALE_INTENSITY_COLUMN],
        errors="raise",
    ).to_numpy(dtype=float)
    if not np.isfinite(within_scale).all() or ((within_scale < 0) | (within_scale > 100)).any():
        raise ValueError("Within-scale intensity must contain finite values in [0, 100].")
    return raw, retained, plateau.drop(columns="_merge")


def _event_rows(
    trials: pd.DataFrame,
    *,
    trial_type: str,
    modulation: float | pd.Series,
) -> pd.DataFrame:
    modulation_values = (
        np.full(len(trials), float(modulation), dtype=float)
        if np.isscalar(modulation)
        else np.asarray(modulation, dtype=float)
    )
    if modulation_values.shape != (len(trials),) or not np.isfinite(modulation_values).all():
        raise ValueError(f"Event modulation for '{trial_type}' must be finite and row-aligned.")
    return pd.DataFrame(
        {
            "run": trials["run"].to_numpy(dtype=int),
            "onset": trials["onset"].to_numpy(dtype=float),
            "duration": trials["duration"].to_numpy(dtype=float),
            "trial_type": trial_type,
            "modulation": modulation_values,
            "stimulus_temp": trials["stimulus_temp"].to_numpy(dtype=float),
        }
    )


def _nuisance_events(raw: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame:
    retained_keys = pd.MultiIndex.from_frame(retained[list(TRIAL_KEYS)])
    raw_keys = pd.MultiIndex.from_frame(raw[list(TRIAL_KEYS)])
    is_plateau = raw["trial_type"].astype(str).eq("stimulation") & raw["stim_phase"].astype(str).eq(
        "plateau"
    )
    is_retained_plateau = is_plateau & raw_keys.isin(retained_keys)
    nuisance = raw.loc[~is_retained_plateau].copy()
    phase = nuisance["stim_phase"].where(nuisance["stim_phase"].notna(), "")
    labels = nuisance["trial_type"].astype(str) + "_" + phase.astype(str)
    nuisance["nuisance_label"] = labels.map(lambda value: f"nuisance_{_slug(value.strip('_'))}")
    return pd.DataFrame(
        {
            "run": nuisance["run"].to_numpy(dtype=int),
            "onset": pd.to_numeric(nuisance["onset"], errors="raise"),
            "duration": pd.to_numeric(nuisance["duration"], errors="raise"),
            "trial_type": nuisance["nuisance_label"].to_numpy(dtype=str),
            "modulation": np.ones(len(nuisance), dtype=float),
            "stimulus_temp": pd.to_numeric(nuisance["stimulus_temp"], errors="raise"),
        }
    )


def _finalize_events(events: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ("run", "onset", "duration", "modulation")
    for column in numeric_columns:
        events[column] = pd.to_numeric(events[column], errors="raise")
        if not np.isfinite(events[column].to_numpy(dtype=float)).all():
            raise ValueError(f"First-level event column '{column}' must be finite.")
    if (events["duration"] <= 0).any():
        raise ValueError("First-level event durations must be positive.")
    return events.sort_values(["run", "onset", "trial_type"], kind="stable").reset_index(drop=True)


def _validate_configured_temperatures(retained: pd.DataFrame, config: Any) -> None:
    expected = tuple(
        float(value)
        for value in require_config_value(config, "study1.figures.validity.temperatures")
    )
    for subject_id, rows in retained.groupby("subject_id", sort=True):
        observed = tuple(sorted(pd.to_numeric(rows["stimulus_temp"], errors="raise").unique()))
        if observed != expected:
            raise ValueError(
                "Every retained participant requires all configured temperatures: "
                f"subject={subject_id}, observed={observed}, expected={expected}."
            )


def _raw_events_path(*, bids_root: Path, subject_id: str, task: str, run: int) -> Path:
    subject_label = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    func_dir = bids_root / subject_label / "func"
    candidates = [
        func_dir / f"{subject_label}_task-{task}_run-{run:02d}_events.tsv",
        func_dir / f"{subject_label}_task-{task}_run-{run}_events.tsv",
    ]
    existing = tuple(dict.fromkeys(path for path in candidates if path.exists()))
    if len(existing) != 1:
        raise FileNotFoundError(
            "Study 1 fMRI construct validity requires exactly one raw fMRI events file for "
            f"{subject_label}, run {run:02d}; found {[str(path) for path in existing]}."
        )
    return existing[0]


def _confounds_path_for_bold(bold_path: Path) -> Path:
    prefix = bold_path.name.split("_space-", maxsplit=1)[0]
    if prefix == bold_path.name:
        suffix = "_desc-preproc_bold.nii.gz"
        if not bold_path.name.endswith(suffix):
            raise ValueError(f"Unsupported fMRIPrep BOLD filename: {bold_path.name}.")
        prefix = bold_path.name.removesuffix(suffix)
    candidate = bold_path.with_name(f"{prefix}_desc-confounds_timeseries.tsv")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing fMRIPrep confounds for {bold_path}: {candidate}.")
    return candidate


def _integer_series(frame: pd.DataFrame, column: str, context: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all() or not np.allclose(
        values,
        np.round(values),
    ):
        raise ValueError(f"{context} column '{column}' must contain finite integers.")
    return values.round().astype(int)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}.")


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Raw fMRI nuisance events require non-empty labels.")
    return slug


__all__ = [
    "FirstLevelRunDesign",
    "FmriRunInput",
    "build_rating_events",
    "build_subject_designs",
    "build_temperature_events",
    "load_fmri_run_inputs",
]
