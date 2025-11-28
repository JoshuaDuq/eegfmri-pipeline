"""
Data loading utilities for decoding.

Provides functions to load and prepare feature matrices for ML decoding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_plateau_matrix(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    log: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load plateau features and targets for multiple subjects.

    Parameters
    ----------
    subjects : List[str]
        Subject IDs to load
    task : str
        Task name
    deriv_root : Path
        Derivatives directory
    config : Any
        Configuration object
    log : logging.Logger, optional
        Logger instance

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    groups : np.ndarray
        Subject IDs for each sample
    feature_cols : List[str]
        Feature column names
    meta : pd.DataFrame
        Metadata with subject_id and trial_id
    """
    from eeg_pipeline.utils.data.loading import _load_features_and_targets

    if log is None:
        log = logger

    X_blocks = []
    y_blocks = []
    groups = []
    trial_ids = []
    feature_cols: Optional[List[str]] = None

    for sub in subjects:
        _, plateau_df, _, y, _ = _load_features_and_targets(sub, task, deriv_root, config)

        if plateau_df is None or plateau_df.empty:
            log.warning(f"No plateau features for sub-{sub}; skipping")
            continue

        # Validate feature columns match
        if feature_cols is None:
            feature_cols = plateau_df.columns.tolist()
        else:
            missing = set(feature_cols) - set(plateau_df.columns)
            if missing:
                raise ValueError(f"Feature mismatch for sub-{sub}; missing: {sorted(missing)}")

        X_block = plateau_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y_block = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

        X_blocks.append(X_block)
        y_blocks.append(y_block)
        groups.extend([sub] * len(y_block))
        trial_ids.extend(list(range(len(y_block))))

    if not X_blocks:
        raise RuntimeError("No subjects with usable plateau features")

    X = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.DataFrame({"subject_id": groups_arr, "trial_id": trial_ids})

    # Filter non-finite targets
    finite_mask = np.isfinite(y_all)
    if not np.all(finite_mask):
        n_dropped = np.sum(~finite_mask)
        log.info(f"Dropping {n_dropped} non-finite targets out of {len(y_all)}")
        X = X[finite_mask]
        y_all = y_all[finite_mask]
        groups_arr = groups_arr[finite_mask]
        meta = meta.loc[finite_mask].reset_index(drop=True)

    return X, y_all, groups_arr, feature_cols or [], meta


def load_epoch_windows(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    window_size: float = 0.5,
    window_step: float = 0.25,
    log: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load epoch data windowed for time-generalization analysis.

    Parameters
    ----------
    subjects : List[str]
        Subject IDs
    task : str
        Task name
    deriv_root : Path
        Derivatives directory
    config : Any
        Configuration object
    window_size : float
        Window size in seconds
    window_step : float
        Step between windows in seconds
    log : logging.Logger, optional
        Logger instance

    Returns
    -------
    X_windows : np.ndarray
        Windowed features (n_samples, n_windows, n_features)
    y : np.ndarray
        Target vector
    groups : np.ndarray
        Subject IDs
    window_centers : np.ndarray
        Center times of each window
    meta : pd.DataFrame
        Metadata
    """
    from eeg_pipeline.utils.data.loading import load_epochs_for_analysis

    if log is None:
        log = logger

    X_blocks = []
    y_blocks = []
    groups = []
    window_centers = None

    for sub in subjects:
        epochs, aligned_events = load_epochs_for_analysis(
            sub, task,
            align="strict",
            preload=True,
            deriv_root=deriv_root,
            config=config,
            logger=log,
        )

        if epochs is None or aligned_events is None:
            log.warning(f"No epochs for sub-{sub}; skipping")
            continue

        # Get target
        target_cols = config.get("event_columns.rating", [])
        target_col = None
        for col in target_cols:
            if col in aligned_events.columns:
                target_col = col
                break

        if target_col is None:
            log.warning(f"No target column for sub-{sub}; skipping")
            continue

        y_sub = pd.to_numeric(aligned_events[target_col], errors="coerce").to_numpy()

        # Window the data
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        times = epochs.times
        sfreq = epochs.info["sfreq"]

        window_samples = int(window_size * sfreq)
        step_samples = int(window_step * sfreq)

        # Create windows
        n_windows = (len(times) - window_samples) // step_samples + 1
        if n_windows < 1:
            log.warning(f"Epochs too short for windowing in sub-{sub}")
            continue

        if window_centers is None:
            window_centers = np.array([
                times[i * step_samples + window_samples // 2]
                for i in range(n_windows)
            ])

        # Extract windowed features (mean power per channel per window)
        X_sub = np.zeros((len(data), n_windows, data.shape[1]))
        for w in range(n_windows):
            start = w * step_samples
            end = start + window_samples
            X_sub[:, w, :] = np.mean(data[:, :, start:end] ** 2, axis=2)

        X_blocks.append(X_sub)
        y_blocks.append(y_sub)
        groups.extend([sub] * len(y_sub))

    if not X_blocks:
        raise RuntimeError("No subjects with usable epoch data")

    X_windows = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.DataFrame({"subject_id": groups_arr})

    # Filter non-finite
    finite_mask = np.isfinite(y_all)
    if not np.all(finite_mask):
        X_windows = X_windows[finite_mask]
        y_all = y_all[finite_mask]
        groups_arr = groups_arr[finite_mask]
        meta = meta.loc[finite_mask].reset_index(drop=True)

    return X_windows, y_all, groups_arr, window_centers, meta
