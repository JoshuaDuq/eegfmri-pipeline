#!/usr/bin/env python
"""Audit Study 1 target, clean-event, and temporal-feature trial alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import load_events_df
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.utils.config.loader import ConfigDict
from eeg_pipeline.utils.config.loader import require_config_value
from eeg_pipeline.utils.config.roots import resolve_fmri_bids_root
from eeg_pipeline.utils.data.fmri_signature_targets import parse_run_label_to_int
from studies.pain_study.study1.config import load_study1_config
from studies.pain_study.study1.temporal_controls import resolve_temporal_control_windows

TRIAL_INDEX_COLUMNS = ("trial_number", "trial_index", "epoch")
TARGET_COLUMNS = (
    "subject_id",
    "task",
    "run",
    "trial_index",
    "within_run_trial",
    "onset",
    "duration",
)
EVENT_COLUMNS = ("trial_id", "onset", "duration")
FMRI_EVENT_COLUMNS = ("onset", "duration", "trial_type", "stim_phase")
LSS_TRIAL_COLUMNS = ("run", "onset", "duration", "events_stim_phase")
PLATEAU_START_S = 3.0
PLATEAU_DURATION_S = 7.5
TIMING_TOLERANCE_S = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study1-root", required=True, type=Path)
    parser.add_argument("--study1-config", required=True, type=Path)
    parser.add_argument("--deriv-root", required=True, type=Path)
    parser.add_argument("--bids-fmri-root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigDict(load_study1_config(args.study1_config))
    config["paths.deriv_root"] = str(args.deriv_root)
    config["paths.bids_fmri_root"] = str(args.bids_fmri_root)

    targets = read_targets(args.study1_root)
    subjects = (
        normalize_subjects(args.subjects)
        if args.subjects
        else sorted(targets["subject_id"].unique())
    )
    events_by_subject = {
        subject_id: read_clean_events(subject_id, task=args.task, config=config)
        for subject_id in subjects
    }
    fmri_events_by_subject = {
        subject_id: read_fmri_events(subject_id, task=args.task, config=config)
        for subject_id in subjects
    }
    lss_trials_by_subject = {
        subject_id: read_lss_trials(subject_id, task=args.task, config=config)
        for subject_id in subjects
    }
    temporal_features_by_subject = {
        subject_id: read_temporal_features(args.study1_root, subject_id) for subject_id in subjects
    }

    summary, trials = build_timing_audit(
        targets=targets.loc[targets["subject_id"].isin(subjects)].copy(),
        events_by_subject=events_by_subject,
        fmri_events_by_subject=fmri_events_by_subject,
        lss_trials_by_subject=lss_trials_by_subject,
        temporal_features_by_subject=temporal_features_by_subject,
        config=config,
    )
    write_outputs(summary, trials, args.output_dir)
    print(f"Wrote Study 1 timing audit to {args.output_dir}")
    print(summary.to_string(index=False))


def read_targets(study1_root: Path) -> pd.DataFrame:
    path = study1_root / "targets" / "primary_targets.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Study 1 primary target table not found: {path}")
    frame = pd.read_parquet(path)
    require_columns(frame, TARGET_COLUMNS, table_name=str(path))
    return frame


def read_clean_events(subject_id: str, *, task: str, config: Any) -> pd.DataFrame:
    subject_raw = subject_id.removeprefix("sub-")
    frame = load_events_df(subject_raw, task, config=config, prefer_clean=True)
    if frame is None or frame.empty:
        raise FileNotFoundError(f"Clean events not found for {subject_id}, task-{task}.")
    frame = frame.reset_index(drop=True)
    require_columns(frame, EVENT_COLUMNS, table_name=f"{subject_id} clean events")
    if "run" not in frame.columns:
        raise ValueError(f"{subject_id} clean events are missing required run column.")
    if find_first_column(frame, TRIAL_INDEX_COLUMNS) is None:
        raise ValueError(f"{subject_id} clean events are missing a trial-number column.")
    return frame


def read_fmri_events(subject_id: str, *, task: str, config: Any) -> pd.DataFrame:
    bids_root = resolve_fmri_bids_root(config, task_is_rest=False)
    subject_dir = bids_root / subject_id / "func"
    pattern = f"{subject_id}_task-{task}_run-*_events.tsv"
    paths = sorted(subject_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"fMRI BIDS event files not found: {subject_dir / pattern}")

    frames = []
    for path in paths:
        frame = read_tsv(path)
        if frame is None or frame.empty:
            raise ValueError(f"Empty fMRI BIDS event file: {path}")
        require_columns(frame, FMRI_EVENT_COLUMNS, table_name=str(path))
        if "run" not in frame.columns:
            raise ValueError(f"{path} is missing required run column.")
        if find_first_column(frame, TRIAL_INDEX_COLUMNS) is None:
            raise ValueError(f"{path} is missing a trial-number column.")
        frames.append(frame)

    return pd.concat(frames, axis=0, ignore_index=True)


def read_lss_trials(subject_id: str, *, task: str, config: Any) -> pd.DataFrame:
    deriv_root = Path(str(require_config_value(config, "paths.deriv_root")))
    contrast = str(require_config_value(config, "study1.targets.contrast_name")).strip()
    path = (
        deriv_root
        / subject_id
        / "fmri"
        / "lss"
        / f"task-{task}"
        / f"contrast-{contrast}"
        / "trials.tsv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Study 1 LSS trials table not found: {path}")
    frame = read_tsv(path)
    if frame is None or frame.empty:
        raise ValueError(f"Empty Study 1 LSS trials table: {path}")
    require_columns(frame, LSS_TRIAL_COLUMNS, table_name=str(path))
    return frame


def read_temporal_features(study1_root: Path, subject_id: str) -> pd.DataFrame:
    path = (
        study1_root
        / "features_temporal_controls"
        / subject_id
        / "eeg"
        / "features"
        / "power"
        / "features_power.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"Study 1 temporal feature table not found: {path}")
    frame = pd.read_parquet(path)
    require_columns(frame, ("trial_id",), table_name=str(path))
    return frame


def build_timing_audit(
    *,
    targets: pd.DataFrame,
    events_by_subject: dict[str, pd.DataFrame],
    fmri_events_by_subject: dict[str, pd.DataFrame],
    lss_trials_by_subject: dict[str, pd.DataFrame],
    temporal_features_by_subject: dict[str, pd.DataFrame],
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(targets, TARGET_COLUMNS, table_name="Study 1 targets")
    temporal_windows = tuple(window.name for window in resolve_temporal_control_windows(config))
    summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for subject_id, target_rows in targets.groupby("subject_id", sort=True):
        events = events_by_subject.get(str(subject_id))
        if events is None:
            raise ValueError(f"Missing clean events for {subject_id}.")
        fmri_events = fmri_events_by_subject.get(str(subject_id))
        if fmri_events is None:
            raise ValueError(f"Missing fMRI events for {subject_id}.")
        lss_trials = lss_trials_by_subject.get(str(subject_id))
        if lss_trials is None:
            raise ValueError(f"Missing LSS trials for {subject_id}.")
        temporal_features = temporal_features_by_subject.get(str(subject_id))
        if temporal_features is None:
            raise ValueError(f"Missing temporal features for {subject_id}.")

        subject_summary, subject_trials = audit_subject(
            subject_id=str(subject_id),
            target_rows=target_rows.copy(),
            events=events,
            fmri_events=fmri_events,
            lss_trials=lss_trials,
            temporal_features=temporal_features,
            temporal_windows=temporal_windows,
        )
        summary_rows.append(subject_summary)
        trial_rows.extend(subject_trials)

    return pd.DataFrame(summary_rows), pd.DataFrame(trial_rows)


def audit_subject(
    *,
    subject_id: str,
    target_rows: pd.DataFrame,
    events: pd.DataFrame,
    fmri_events: pd.DataFrame,
    lss_trials: pd.DataFrame,
    temporal_features: pd.DataFrame,
    temporal_windows: tuple[str, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    event_trial_ids = integer_series(events["trial_id"], f"{subject_id} clean event trial_id")
    feature_trial_ids = integer_series(
        temporal_features["trial_id"],
        f"{subject_id} temporal feature trial_id",
    )
    require_unique(feature_trial_ids, f"{subject_id} temporal feature trial_id values")
    feature_trial_id_set = set(feature_trial_ids.tolist())
    feature_sequence_matches = event_trial_ids.tolist() == feature_trial_ids.tolist()

    event_key_frame = event_alignment_keys(events, subject_id=subject_id)
    target_key_frame = target_alignment_keys(target_rows, subject_id=subject_id)
    fmri_plateau_frame = fmri_plateau_alignment_keys(fmri_events, subject_id=subject_id)
    lss_plateau_frame = lss_plateau_alignment_keys(lss_trials, subject_id=subject_id)
    indexed_events = event_key_frame.set_index("alignment_key", verify_integrity=True)
    indexed_fmri_plateaus = fmri_plateau_frame.set_index(
        "alignment_key",
        verify_integrity=True,
    )
    indexed_lss_plateaus = lss_plateau_frame.set_index("alignment_key", verify_integrity=True)

    missing_temporal_windows = missing_windows(temporal_features, temporal_windows)
    subject_trials: list[dict[str, object]] = []
    n_unmatched = 0
    n_missing_fmri_plateaus = 0
    n_invalid_fmri_plateaus = 0
    n_missing_lss_plateaus = 0
    n_invalid_lss_plateaus = 0
    n_missing_temporal_features = 0

    for row in target_key_frame.itertuples(index=False):
        alignment_key = str(row.alignment_key)
        if alignment_key not in indexed_events.index:
            n_unmatched += 1
            subject_trials.append(unmatched_trial_row(subject_id, row))
            continue

        event_row = indexed_events.loc[alignment_key]
        has_fmri_plateau = alignment_key in indexed_fmri_plateaus.index
        if not has_fmri_plateau:
            n_missing_fmri_plateaus += 1
            n_invalid_fmri_plateaus += 1
            fmri_plateau_onset = np.nan
            fmri_plateau_duration = np.nan
            fmri_plateau_start_delta = np.nan
            fmri_plateau_duration_delta = np.nan
        else:
            fmri_plateau_row = indexed_fmri_plateaus.loc[alignment_key]
            fmri_plateau_onset = float(fmri_plateau_row["onset"])
            fmri_plateau_duration = float(fmri_plateau_row["duration"])
            fmri_plateau_start_delta = round(
                fmri_plateau_onset - float(row.onset) - PLATEAU_START_S, 6
            )
            fmri_plateau_duration_delta = round(fmri_plateau_duration - PLATEAU_DURATION_S, 6)
            if (
                abs(fmri_plateau_start_delta) > TIMING_TOLERANCE_S
                or abs(fmri_plateau_duration_delta) > TIMING_TOLERANCE_S
            ):
                n_invalid_fmri_plateaus += 1
        has_lss_plateau = alignment_key in indexed_lss_plateaus.index
        if not has_lss_plateau:
            n_missing_lss_plateaus += 1
            n_invalid_lss_plateaus += 1
            lss_plateau_onset = np.nan
            lss_plateau_duration = np.nan
            lss_plateau_start_delta = np.nan
            lss_plateau_duration_delta = np.nan
        else:
            lss_plateau_row = indexed_lss_plateaus.loc[alignment_key]
            lss_plateau_onset = float(lss_plateau_row["onset"])
            lss_plateau_duration = float(lss_plateau_row["duration"])
            lss_plateau_start_delta = round(
                lss_plateau_onset - float(row.onset) - PLATEAU_START_S, 6
            )
            lss_plateau_duration_delta = round(lss_plateau_duration - PLATEAU_DURATION_S, 6)
            if (
                abs(lss_plateau_start_delta) > TIMING_TOLERANCE_S
                or abs(lss_plateau_duration_delta) > TIMING_TOLERANCE_S
            ):
                n_invalid_lss_plateaus += 1
        event_trial_id = int(event_row["trial_id"])
        temporal_feature_present = event_trial_id in feature_trial_id_set
        if not temporal_feature_present:
            n_missing_temporal_features += 1
        subject_trials.append(
            {
                "subject_id": subject_id,
                "alignment_key": alignment_key,
                "run": int(row.run),
                "trial_index": int(row.trial_index),
                "within_run_trial": int(row.within_run_trial),
                "target_onset": float(row.onset),
                "target_duration": float(row.duration),
                "event_onset": float(event_row["onset"]),
                "event_duration": float(event_row["duration"]),
                "event_trial_id": event_trial_id,
                "fmri_plateau_onset": fmri_plateau_onset,
                "fmri_plateau_duration": fmri_plateau_duration,
                "fmri_plateau_start_delta_s": fmri_plateau_start_delta,
                "fmri_plateau_duration_delta_s": fmri_plateau_duration_delta,
                "lss_plateau_onset": lss_plateau_onset,
                "lss_plateau_duration": lss_plateau_duration,
                "lss_plateau_start_delta_s": lss_plateau_start_delta,
                "lss_plateau_duration_delta_s": lss_plateau_duration_delta,
                "temporal_feature_row_present": temporal_feature_present,
                "timing_delta_s": round(float(row.onset) - float(event_row["onset"]), 6),
            }
        )

    summary = {
        "subject_id": subject_id,
        "n_target_trials": int(len(target_rows)),
        "n_clean_events": int(len(events)),
        "n_fmri_plateau_events": int(len(fmri_plateau_frame)),
        "n_lss_plateau_trials": int(len(lss_plateau_frame)),
        "n_temporal_feature_rows": int(len(temporal_features)),
        "n_unmatched_target_trials": int(n_unmatched),
        "n_missing_fmri_plateau_events": int(n_missing_fmri_plateaus),
        "n_invalid_fmri_plateau_events": int(n_invalid_fmri_plateaus),
        "n_missing_lss_plateau_trials": int(n_missing_lss_plateaus),
        "n_invalid_lss_plateau_trials": int(n_invalid_lss_plateaus),
        "n_missing_temporal_feature_rows": int(n_missing_temporal_features),
        "temporal_feature_sequence_matches_clean_events": bool(feature_sequence_matches),
        "missing_temporal_windows": ",".join(missing_temporal_windows),
        "max_abs_target_event_onset_delta_s": max_abs_delta(subject_trials),
        "max_abs_fmri_plateau_start_delta_s": max_abs_trial_value(
            subject_trials,
            "fmri_plateau_start_delta_s",
        ),
        "max_abs_fmri_plateau_duration_delta_s": max_abs_trial_value(
            subject_trials,
            "fmri_plateau_duration_delta_s",
        ),
        "max_abs_lss_plateau_start_delta_s": max_abs_trial_value(
            subject_trials,
            "lss_plateau_start_delta_s",
        ),
        "max_abs_lss_plateau_duration_delta_s": max_abs_trial_value(
            subject_trials,
            "lss_plateau_duration_delta_s",
        ),
    }
    return summary, subject_trials


def event_alignment_keys(events: pd.DataFrame, *, subject_id: str) -> pd.DataFrame:
    trial_column = find_first_column(events, TRIAL_INDEX_COLUMNS)
    if "run" not in events.columns or trial_column is None:
        raise ValueError(f"{subject_id} clean events are missing alignment columns.")

    frame = pd.DataFrame(
        {
            "alignment_key": alignment_keys(events["run"], events[trial_column]),
            "trial_id": integer_series(events["trial_id"], f"{subject_id} clean event trial_id"),
            "onset": numeric_series(events["onset"], f"{subject_id} clean event onset"),
            "duration": numeric_series(events["duration"], f"{subject_id} clean event duration"),
        }
    )
    require_unique(frame["alignment_key"], f"{subject_id} clean event alignment keys")
    return frame


def fmri_plateau_alignment_keys(fmri_events: pd.DataFrame, *, subject_id: str) -> pd.DataFrame:
    require_columns(fmri_events, FMRI_EVENT_COLUMNS, table_name=f"{subject_id} fMRI events")
    trial_column = find_first_column(fmri_events, TRIAL_INDEX_COLUMNS)
    if "run" not in fmri_events.columns or trial_column is None:
        raise ValueError(f"{subject_id} fMRI events are missing alignment columns.")

    trial_type = fmri_events["trial_type"].astype(str).str.strip()
    stim_phase = fmri_events["stim_phase"].astype(str).str.strip()
    plateau_rows = fmri_events.loc[trial_type.eq("stimulation") & stim_phase.eq("plateau")].copy()
    if plateau_rows.empty:
        raise ValueError(f"{subject_id} fMRI events contain no stimulation plateau rows.")

    frame = pd.DataFrame(
        {
            "alignment_key": alignment_keys(plateau_rows["run"], plateau_rows[trial_column]),
            "onset": numeric_series(plateau_rows["onset"], f"{subject_id} fMRI plateau onset"),
            "duration": numeric_series(
                plateau_rows["duration"],
                f"{subject_id} fMRI plateau duration",
            ),
        }
    )
    require_unique(frame["alignment_key"], f"{subject_id} fMRI plateau alignment keys")
    return frame


def lss_plateau_alignment_keys(lss_trials: pd.DataFrame, *, subject_id: str) -> pd.DataFrame:
    require_columns(lss_trials, LSS_TRIAL_COLUMNS, table_name=f"{subject_id} LSS trials")
    stim_phase = lss_trials["events_stim_phase"].astype(str).str.strip()
    plateau_rows = lss_trials.loc[stim_phase.eq("plateau")].copy()
    if plateau_rows.empty:
        raise ValueError(f"{subject_id} LSS trials contain no plateau rows.")

    trial_column = "events_trial_number" if "events_trial_number" in plateau_rows else "trial_index"
    run_values = pd.Series(
        [parse_run_label_to_int(value) for value in plateau_rows["run"]],
        index=plateau_rows.index,
    )
    frame = pd.DataFrame(
        {
            "alignment_key": alignment_keys(run_values, plateau_rows[trial_column]),
            "onset": numeric_series(plateau_rows["onset"], f"{subject_id} LSS plateau onset"),
            "duration": numeric_series(
                plateau_rows["duration"],
                f"{subject_id} LSS plateau duration",
            ),
        }
    )
    require_unique(frame["alignment_key"], f"{subject_id} LSS plateau alignment keys")
    return frame


def target_alignment_keys(targets: pd.DataFrame, *, subject_id: str) -> pd.DataFrame:
    frame = targets.copy()
    frame["alignment_key"] = alignment_keys(frame["run"], frame["trial_index"])
    require_unique(frame["alignment_key"], f"{subject_id} target alignment keys")
    return frame


def alignment_keys(runs: pd.Series, trials: pd.Series) -> list[str]:
    run_values = integer_series(runs, "alignment run")
    trial_values = integer_series(trials, "alignment trial")
    return [
        f"{int(run)}|{int(trial)}"
        for run, trial in zip(run_values.to_numpy(), trial_values.to_numpy())
    ]


def missing_windows(features: pd.DataFrame, temporal_windows: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    columns = [str(column) for column in features.columns]
    for window in temporal_windows:
        prefix = f"power_{window}_"
        if not any(column.startswith(prefix) for column in columns):
            missing.append(window)
    return missing


def unmatched_trial_row(subject_id: str, row: object) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "alignment_key": str(getattr(row, "alignment_key")),
        "run": int(getattr(row, "run")),
        "trial_index": int(getattr(row, "trial_index")),
        "within_run_trial": int(getattr(row, "within_run_trial")),
        "target_onset": float(getattr(row, "onset")),
        "target_duration": float(getattr(row, "duration")),
        "event_onset": np.nan,
        "event_duration": np.nan,
        "event_trial_id": "",
        "fmri_plateau_onset": np.nan,
        "fmri_plateau_duration": np.nan,
        "fmri_plateau_start_delta_s": np.nan,
        "fmri_plateau_duration_delta_s": np.nan,
        "lss_plateau_onset": np.nan,
        "lss_plateau_duration": np.nan,
        "lss_plateau_start_delta_s": np.nan,
        "lss_plateau_duration_delta_s": np.nan,
        "temporal_feature_row_present": False,
        "timing_delta_s": np.nan,
    }


def max_abs_delta(trial_rows: list[dict[str, object]]) -> float:
    return max_abs_trial_value(trial_rows, "timing_delta_s")


def max_abs_trial_value(trial_rows: list[dict[str, object]], column: str) -> float:
    values = pd.to_numeric(
        pd.Series([row[column] for row in trial_rows]),
        errors="coerce",
    )
    if not values.notna().any():
        return float("nan")
    return round(float(values.abs().max()), 6)


def find_first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def integer_series(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().all():
        raise ValueError(f"{label} must contain finite numeric values.")
    rounded = np.rint(numeric.to_numpy(dtype=float))
    if not np.allclose(numeric.to_numpy(dtype=float), rounded):
        raise ValueError(f"{label} must contain integer values.")
    return pd.Series(rounded.astype(int), index=values.index)


def numeric_series(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().all():
        raise ValueError(f"{label} must contain finite numeric values.")
    return numeric.astype(float)


def require_columns(frame: pd.DataFrame, required: tuple[str, ...], *, table_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required column(s): {missing}")


def require_unique(values: pd.Series, label: str) -> None:
    duplicates = values[values.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"{label} contain duplicate value(s): {duplicates}")


def normalize_subjects(subjects: list[str]) -> list[str]:
    normalized = []
    for subject in subjects:
        subject_id = str(subject).strip()
        if not subject_id:
            raise ValueError("Subject identifiers must be non-empty.")
        normalized.append(subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}")
    return normalized


def write_outputs(summary: pd.DataFrame, trials: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "study1_timing_audit_summary.tsv", sep="\t", index=False)
    trials.to_csv(output_dir / "study1_timing_audit_trials.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
