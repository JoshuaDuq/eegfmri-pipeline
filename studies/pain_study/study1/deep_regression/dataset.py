"""Study 1 deep-regression dataset assembly."""

from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.fmri_signature_targets import find_run_column
from studies.pain_study.study1.cohort import (
    resolve_primary_subjects,
    resolve_primary_target_name,
    subject_target_rows,
)
from studies.pain_study.study1.deep_regression.bands import build_band_tensor


def _subject_eeg_channels(epochs: mne.Epochs) -> list[str]:
    eeg_picks = mne.pick_types(
        epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
    )
    return [str(epochs.ch_names[pick]) for pick in eeg_picks]


def _trial_key(run_num: float, trial_index: float) -> str | None:
    if not np.isfinite(run_num) or not np.isfinite(trial_index):
        return None
    if run_num != int(run_num) or trial_index != int(trial_index):
        raise ValueError("Deep regression run and trial identifiers must be integer-valued.")
    return f"{int(run_num)}|{int(trial_index)}"


def _event_trial_indices(events: pd.DataFrame) -> pd.Series:
    for column in ("trial_number", "trial_index", "epoch"):
        if column in events.columns:
            return pd.to_numeric(events[column], errors="coerce")
    raise ValueError("Deep regression alignment requires trial identifiers in clean EEG events.")


def _time_key(run_num: float, onset: float, duration: float) -> str | None:
    if not np.isfinite(run_num) or not np.isfinite(onset) or not np.isfinite(duration):
        return None
    return f"{int(run_num)}|{round(float(onset), 3):.3f}|{round(float(duration), 3):.3f}"


def _raise_on_duplicate_keys(
    frame: pd.DataFrame,
    *,
    key_column: str,
    label: str,
) -> None:
    keyed = frame.loc[frame[key_column].notna(), key_column]
    if keyed.empty:
        return
    counts = keyed.value_counts()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        examples = ", ".join(str(key) for key in duplicates.index[:5])
        raise ValueError(f"Deep regression alignment has duplicate {label} keys: {examples}.")


def _raise_on_missing_or_duplicate_trial_keys(keys: list[str | None], *, label: str) -> None:
    missing = [idx for idx, key in enumerate(keys) if key is None]
    if missing:
        raise ValueError(
            f"Deep regression alignment requires finite trial identifiers for every {label} row; "
            f"missing at rows {missing[:5]}."
        )
    _raise_on_duplicate_keys(
        pd.DataFrame({"key": keys}),
        key_column="key",
        label=f"{label} trial",
    )


def _unique_values_by_key(
    frame: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
    label: str,
) -> pd.Series:
    _raise_on_duplicate_keys(frame, key_column=key_column, label=label)
    keyed = frame.loc[frame[key_column].notna(), [key_column, value_column]].copy()
    if keyed.empty:
        return pd.Series(dtype=float)
    return keyed.set_index(key_column)[value_column]


def _alignment_key_data(
    *,
    aligned_events: pd.DataFrame,
    target_rows: pd.DataFrame,
    value_column: str,
) -> tuple[list[str | None], pd.Series]:
    event_runs = find_run_column(aligned_events)
    if event_runs is None:
        raise ValueError("Clean EEG events must contain a usable 'run' column for deep regression.")
    event_runs = pd.to_numeric(event_runs, errors="coerce")

    event_trial = _event_trial_indices(aligned_events)

    target_runs = pd.to_numeric(target_rows["run"], errors="coerce")
    target_trials = pd.to_numeric(target_rows["trial_index"], errors="coerce")
    target_values = pd.to_numeric(target_rows[value_column], errors="coerce")

    target_trial_keys = [
        _trial_key(run_num, trial_index)
        for run_num, trial_index in zip(
            target_runs.to_numpy(dtype=float),
            target_trials.to_numpy(dtype=float),
        )
    ]
    _raise_on_missing_or_duplicate_trial_keys(target_trial_keys, label="target")
    trial_frame = pd.DataFrame({"key": target_trial_keys, "value": target_values})
    trial_lookup = _unique_values_by_key(
        trial_frame,
        key_column="key",
        value_column="value",
        label="target trial",
    )
    event_trial_keys = [
        _trial_key(run_num, trial_index)
        for run_num, trial_index in zip(
            event_runs.to_numpy(dtype=float),
            event_trial.to_numpy(dtype=float),
        )
    ]
    _raise_on_missing_or_duplicate_trial_keys(event_trial_keys, label="EEG event")
    trial_matches = sum(
        1 for key in event_trial_keys if key is not None and key in trial_lookup.index
    )

    event_time_keys = [
        _time_key(run_num, onset, duration)
        for run_num, onset, duration in zip(
            event_runs.to_numpy(dtype=float),
            pd.to_numeric(aligned_events["onset"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(aligned_events["duration"], errors="coerce").to_numpy(dtype=float),
        )
    ]
    target_time_keys = [
        _time_key(run_num, onset, duration)
        for run_num, onset, duration in zip(
            target_runs.to_numpy(dtype=float),
            pd.to_numeric(target_rows["onset"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(target_rows["duration"], errors="coerce").to_numpy(dtype=float),
        )
    ]
    if trial_matches == 0:
        raise ValueError(
            f"Deep regression alignment failed for target '{value_column}': no trial-id matches."
        )

    target_times_by_trial = dict(zip(target_trial_keys, target_time_keys, strict=True))
    for trial_key, event_time in zip(event_trial_keys, event_time_keys, strict=True):
        if trial_key not in target_times_by_trial:
            continue
        if event_time is None or event_time != target_times_by_trial[trial_key]:
            raise ValueError(
                "Deep regression temporal audit failed: onset/duration must match "
                f"the prepared target row for trial {trial_key}."
            )

    return event_trial_keys, trial_lookup


def _align_subject_targets(
    *,
    aligned_events: pd.DataFrame,
    target_rows: pd.DataFrame,
    target_name: str,
) -> np.ndarray:
    active_keys, active_lookup = _alignment_key_data(
        aligned_events=aligned_events,
        target_rows=target_rows,
        value_column=target_name,
    )
    y = np.asarray(
        [
            float(active_lookup[key]) if key is not None and key in active_lookup.index else np.nan
            for key in active_keys
        ],
        dtype=float,
    )
    return y


def _append_target_table_metadata(
    *,
    meta: pd.DataFrame,
    aligned_events: pd.DataFrame,
    target_rows: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    excluded = {
        "subject_id",
        "task",
        "run",
        "trial_index",
        "onset",
        "duration",
        "SIIPS1",
        target_name,
    }
    out = meta.copy()
    for column in target_rows.columns:
        if column in excluded or column in out.columns:
            continue
        values = pd.to_numeric(target_rows[column], errors="coerce")
        if not np.any(np.isfinite(values.to_numpy(dtype=float))):
            continue
        active_keys, active_lookup = _alignment_key_data(
            aligned_events=aligned_events,
            target_rows=target_rows.assign(**{column: values}),
            value_column=column,
        )
        out[column] = [
            float(active_lookup[key]) if key is not None and key in active_lookup.index else np.nan
            for key in active_keys
        ]
    return out


def load_band_tensor_matrix(
    *,
    subjects: list[str],
    task: str,
    config: Any,
    target_name: str,
    bands: list[str],
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    resolved_target = resolve_primary_target_name(config, target_name)
    resolved_subjects = resolve_primary_subjects(
        requested_subjects=subjects,
        task=task,
        config=config,
    )
    deriv_root = resolve_eeg_deriv_root(config)

    payloads: list[tuple[str, mne.Epochs, pd.DataFrame, pd.DataFrame]] = []
    channel_sets: list[set[str]] = []
    for subject_id in resolved_subjects:
        subject_raw = str(subject_id).replace("sub-", "", 1)
        epochs, aligned_events = load_epochs_for_analysis(
            subject_raw,
            task,
            align="strict",
            preload=True,
            deriv_root=deriv_root,
            config=config,
            logger=logger,
        )
        if epochs is None or aligned_events is None:
            raise FileNotFoundError(
                f"Clean epochs/events are required for Study 1 deep regression: {subject_id}, task-{task}."
            )
        target_rows = subject_target_rows(subject_id=subject_id, task=task, config=config)
        payloads.append((subject_id, epochs, aligned_events.reset_index(drop=True), target_rows))
        channel_sets.append(set(_subject_eeg_channels(epochs)))

    common_channels = sorted(set.intersection(*channel_sets)) if channel_sets else []
    if not common_channels:
        raise ValueError("Study 1 deep regression found no common EEG channels across subjects.")

    tensor_runs: list[np.ndarray] = []
    target_runs: list[np.ndarray] = []
    groups: list[str] = []
    meta_runs: list[pd.DataFrame] = []
    reference_times: np.ndarray | None = None
    for subject_id, epochs, aligned_events, target_rows in payloads:
        y = _align_subject_targets(
            aligned_events=aligned_events,
            target_rows=target_rows,
            target_name=resolved_target,
        )
        tensors, times = build_band_tensor(
            epochs=epochs,
            config=config,
            bands=bands,
            channels=common_channels,
            logger=logger,
        )
        if reference_times is None:
            reference_times = times
        elif times.shape != reference_times.shape or not np.allclose(
            times, reference_times, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "Study 1 deep regression requires a common sampled time axis across "
                f"participants; {subject_id} differs from {payloads[0][0]}."
            )
        if not np.all(np.isfinite(y)):
            n_invalid = int((~np.isfinite(y)).sum())
            raise ValueError(
                f"Study 1 deep regression requires finite aligned targets for every trial; "
                f"found {n_invalid} non-finite '{resolved_target}' target(s) for {subject_id}. "
                "Non-finite targets indicate an EEG/fMRI alignment or preparation error and are "
                "not dropped."
            )
        if tensors.shape[0] != len(y):
            raise ValueError(
                f"Band tensor/target length mismatch for {subject_id}: tensors={tensors.shape[0]}, targets={len(y)}."
            )

        run = find_run_column(aligned_events)
        if run is None:
            raise ValueError(
                "Clean EEG events must contain a usable 'run' column for deep regression."
            )
        trial_index = _event_trial_indices(aligned_events)

        meta = pd.DataFrame(
            {
                "subject_id": [subject_id] * len(y),
                "task": [task] * len(y),
                "run": pd.to_numeric(run, errors="coerce"),
                "trial_index": trial_index,
                "onset": pd.to_numeric(aligned_events["onset"], errors="coerce"),
                "duration": pd.to_numeric(aligned_events["duration"], errors="coerce"),
                "target_name": [resolved_target] * len(y),
                "target_value": y,
            }
        )
        meta = _append_target_table_metadata(
            meta=meta,
            aligned_events=aligned_events,
            target_rows=target_rows,
            target_name=resolved_target,
        )
        tensor_runs.append(tensors)
        target_runs.append(y)
        groups.extend([subject_id] * len(y))
        meta_runs.append(meta)

    X = np.concatenate(tensor_runs, axis=0)
    y_all = np.concatenate(target_runs, axis=0)
    groups_arr = np.asarray(groups, dtype=object)
    meta = pd.concat(meta_runs, axis=0, ignore_index=True)
    meta["trial_id"] = np.arange(len(meta), dtype=int)

    if X.ndim != 4:
        raise ValueError(f"Deep regression expects 4D tensors, got shape {X.shape}.")
    if len(X) != len(y_all) or len(X) != len(groups_arr):
        raise ValueError("Deep regression tensor assembly produced inconsistent lengths.")
    if not np.all(np.isfinite(y_all)):
        raise ValueError("Deep regression targets must be finite after alignment.")
    return X, y_all.astype(float), groups_arr, common_channels, meta


__all__ = ["load_band_tensor_matrix"]
