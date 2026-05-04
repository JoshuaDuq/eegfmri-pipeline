"""Study 1 deep-regression dataset assembly."""

from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from eeg_pipeline.utils.data.fmri_signature_targets import find_block_column
from studies.pain_study.study1.cohort import (
    resolve_primary_subjects,
    resolve_primary_target_name,
    subject_target_rows,
)
from studies.pain_study.study1.deep_regression.bands import build_band_tensor


def _subject_eeg_channels(epochs: mne.Epochs) -> list[str]:
    eeg_picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    return [str(epochs.ch_names[pick]) for pick in eeg_picks]


def _trial_key(run_num: float, trial_index: float) -> str | None:
    if not np.isfinite(run_num) or not np.isfinite(trial_index):
        return None
    return f"{int(run_num)}|{int(round(trial_index))}"


def _time_key(run_num: float, onset: float, duration: float) -> str | None:
    if not np.isfinite(run_num) or not np.isfinite(onset) or not np.isfinite(duration):
        return None
    return f"{int(run_num)}|{round(float(onset), 3):.3f}|{round(float(duration), 3):.3f}"


def _align_subject_targets(
    *,
    aligned_events: pd.DataFrame,
    target_rows: pd.DataFrame,
    target_name: str,
) -> np.ndarray:
    event_runs = find_block_column(aligned_events)
    if event_runs is None:
        raise ValueError("Clean EEG events must contain a usable run/block column for deep regression.")
    event_runs = pd.to_numeric(event_runs, errors="coerce")

    event_trial = None
    if "trial_number" in aligned_events.columns:
        event_trial = pd.to_numeric(aligned_events["trial_number"], errors="coerce")
    elif "trial_index" in aligned_events.columns:
        event_trial = pd.to_numeric(aligned_events["trial_index"], errors="coerce")
    elif "epoch" in aligned_events.columns:
        event_trial = pd.to_numeric(aligned_events["epoch"], errors="coerce")

    target_runs = pd.to_numeric(target_rows["block"], errors="coerce")
    target_trials = pd.to_numeric(target_rows["trial_index"], errors="coerce")
    target_values = pd.to_numeric(target_rows[target_name], errors="coerce")

    trial_lookup = pd.Series(dtype=float)
    if event_trial is not None:
        target_trial_keys = [
            _trial_key(run_num, trial_index)
            for run_num, trial_index in zip(
                target_runs.to_numpy(dtype=float),
                target_trials.to_numpy(dtype=float),
            )
        ]
        trial_frame = pd.DataFrame({"key": target_trial_keys, "value": target_values})
        trial_lookup = trial_frame.loc[trial_frame["key"].notna()].groupby("key")["value"].mean()
        event_trial_keys = [
            _trial_key(run_num, trial_index)
            for run_num, trial_index in zip(
                event_runs.to_numpy(dtype=float),
                event_trial.to_numpy(dtype=float),
            )
        ]
        trial_matches = sum(
            1 for key in event_trial_keys if key is not None and key in trial_lookup.index
        )
    else:
        event_trial_keys = [None] * len(aligned_events)
        trial_matches = 0

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
    time_frame = pd.DataFrame({"key": target_time_keys, "value": target_values})
    time_lookup = time_frame.loc[time_frame["key"].notna()].groupby("key")["value"].mean()
    time_matches = sum(1 for key in event_time_keys if key is not None and key in time_lookup.index)

    if trial_matches == 0 and time_matches == 0:
        raise ValueError(
            f"Deep regression alignment failed for target '{target_name}': no trial or onset matches."
        )

    use_trial = trial_matches >= time_matches and trial_matches > 0
    active_keys = event_trial_keys if use_trial else event_time_keys
    active_lookup = trial_lookup if use_trial else time_lookup
    y = np.asarray(
        [
            float(active_lookup[key]) if key is not None and key in active_lookup.index else np.nan
            for key in active_keys
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(y)):
        raise ValueError(
            f"Deep regression requires finite aligned targets for every retained trial ({target_name})."
        )
    return y


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

    tensor_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    groups: list[str] = []
    meta_blocks: list[pd.DataFrame] = []
    for subject_id, epochs, aligned_events, target_rows in payloads:
        y = _align_subject_targets(
            aligned_events=aligned_events,
            target_rows=target_rows,
            target_name=resolved_target,
        )
        tensors = build_band_tensor(
            epochs=epochs,
            config=config,
            bands=bands,
            channels=common_channels,
            logger=logger,
        )
        if tensors.shape[0] != len(y):
            raise ValueError(
                f"Band tensor/target length mismatch for {subject_id}: tensors={tensors.shape[0]}, targets={len(y)}."
            )

        block = find_block_column(aligned_events)
        if "trial_number" in aligned_events.columns:
            trial_index = pd.to_numeric(aligned_events["trial_number"], errors="coerce")
        elif "trial_index" in aligned_events.columns:
            trial_index = pd.to_numeric(aligned_events["trial_index"], errors="coerce")
        else:
            trial_index = pd.Series(np.arange(1, len(y) + 1), dtype=float)

        meta = pd.DataFrame(
            {
                "subject_id": [subject_id] * len(y),
                "task": [task] * len(y),
                "block": pd.to_numeric(block, errors="coerce") if block is not None else np.nan,
                "trial_index": trial_index,
                "onset": pd.to_numeric(aligned_events["onset"], errors="coerce"),
                "duration": pd.to_numeric(aligned_events["duration"], errors="coerce"),
                "target_name": [resolved_target] * len(y),
                "target_value": y,
            }
        )
        tensor_blocks.append(tensors)
        target_blocks.append(y)
        groups.extend([subject_id] * len(y))
        meta_blocks.append(meta)

    X = np.concatenate(tensor_blocks, axis=0)
    y_all = np.concatenate(target_blocks, axis=0)
    groups_arr = np.asarray(groups, dtype=object)
    meta = pd.concat(meta_blocks, axis=0, ignore_index=True)
    meta["trial_id"] = np.arange(len(meta), dtype=int)

    if X.ndim != 4:
        raise ValueError(f"Deep regression expects 4D tensors, got shape {X.shape}.")
    if len(X) != len(y_all) or len(X) != len(groups_arr):
        raise ValueError("Deep regression tensor assembly produced inconsistent lengths.")
    if not np.all(np.isfinite(y_all)):
        raise ValueError("Deep regression targets must be finite after alignment.")
    return X, y_all.astype(float), groups_arr, common_channels, meta


__all__ = ["load_band_tensor_matrix"]
