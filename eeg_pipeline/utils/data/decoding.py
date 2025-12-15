from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from eeg_pipeline.io.columns import pick_target_column
from ..config.loader import ConfigDict



EEGConfig = ConfigDict


def _get_trial_alignment_manifest_path(deriv_root: Path, subject: str) -> Path:
    sub = f"sub-{subject}" if not subject.startswith("sub-") else subject
    return deriv_root / sub / "eeg" / "features" / "trial_alignment.tsv"


def _load_trial_alignment_manifest(
    manifest_path: Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if logger is None:
        logger = logging.getLogger(__name__)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Trial alignment manifest missing: {manifest_path}\n"
            f"This file is required and must be created by 03_feature_extraction.py. "
            f"Run 03_feature_extraction.py first to generate features with proper alignment."
        )

    manifest = pd.read_csv(manifest_path, sep="\t")
    if "trial_index" not in manifest.columns:
        raise ValueError(
            f"Invalid trial alignment manifest: missing 'trial_index' column in {manifest_path}"
        )

    if len(manifest) == 0:
        raise ValueError(f"Trial alignment manifest is empty: {manifest_path}")

    logger.debug(f"Loaded trial alignment manifest: {len(manifest)} trials from {manifest_path}")
    return manifest


def _resolve_columns(
    df: pd.DataFrame,
    config: EEGConfig,
    deriv_root: Path,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    rating_cols = config.get("event_columns.rating", [])
    temp_cols = config.get("event_columns.temperature", [])
    pain_cols = config.get("event_columns.pain_binary", [])

    def _pick_first_existing(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in df.columns:
                return cand
        return None

    pain_col = _pick_first_existing(list(pain_cols) if pain_cols else [])
    temp_col = _pick_first_existing(list(temp_cols) if temp_cols else [])
    rating_col = _pick_first_existing(list(rating_cols) if rating_cols else [])
    return pain_col, temp_col, rating_col


def load_decoding_data(
    subject: str,
    deriv_root: Path,
    config: EEGConfig,
    bids_root: Optional[Path] = None,
    task: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if task is None:
        task = config.get("project.task")

    if bids_root is None:
        bids_root = config.bids_root

    sub = f"sub-{subject}"
    feat_dir = deriv_root / sub / "eeg" / "features"
    X_path = feat_dir / "features_eeg_direct.tsv"
    y_path = feat_dir / "target_vas_ratings.tsv"
    manifest_path = _get_trial_alignment_manifest_path(deriv_root, subject)

    if not (X_path.exists() and y_path.exists()):
        raise FileNotFoundError(
            f"Missing features/targets for {sub}: {X_path} or {y_path} not found"
        )

    manifest = _load_trial_alignment_manifest(manifest_path, logger)
    expected_n_trials = len(manifest)

    X = pd.read_csv(X_path, sep="\t")
    y_df = pd.read_csv(y_path, sep="\t")

    if len(X) != expected_n_trials:
        raise ValueError(
            f"Feature count mismatch for subject {sub}, task {task}: "
            f"features have {len(X)} rows but trial_alignment.tsv specifies {expected_n_trials} trials. "
            f"This indicates a misalignment between feature extraction and trial manifest. "
            f"Re-run 03_feature_extraction.py to regenerate features with proper alignment."
        )

    if len(y_df) != expected_n_trials:
        raise ValueError(
            f"Target count mismatch for subject {sub}, task {task}: "
            f"targets have {len(y_df)} rows but trial_alignment.tsv specifies {expected_n_trials} trials. "
            f"This indicates a misalignment between target extraction and trial manifest. "
            f"Re-run 03_feature_extraction.py to regenerate features with proper alignment."
        )

    constants = {"TARGET_COLUMNS": config.get("event_columns.rating", [])}
    tgt_col = pick_target_column(y_df, constants=constants)
    if tgt_col is None:
        _, _, rating_col = _resolve_columns(y_df, config=config, deriv_root=deriv_root)
        if rating_col is None:
            raise ValueError(
                f"No suitable target column found in {y_path} for subject {sub}, task {task}. "
                f"Available columns: {list(y_df.columns)}"
            )
        tgt_col = rating_col

    y = pd.to_numeric(y_df[tgt_col], errors="coerce")

    if len(X) != len(y) or len(X) == 0:
        raise ValueError(
            f"Length mismatch or empty data for subject {sub}, task {task}: X={len(X)}, y={len(y)}"
        )

    mask_valid = ~y.isna()
    n_dropped = int((~mask_valid).sum())
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} trials with NaN targets for {sub}")

    X = X.loc[mask_valid].reset_index(drop=True)
    y = y.loc[mask_valid].reset_index(drop=True)

    groups = np.array([sub] * len(X))

    meta = pd.DataFrame(
        {
            "subject_id": [sub] * len(X),
            "trial_id": list(range(len(X))),
        }
    )

    logger.info(f"Loaded data for {sub}: n_trials={len(X)}, n_features={X.shape[1]}")

    return X, y, groups, meta


def load_multiple_subjects_decoding_data(
    deriv_root: Path,
    config: EEGConfig,
    subjects: Optional[List[str]] = None,
    subject_discovery_policy: str = "intersection",
    bids_root: Optional[Path] = None,
    task: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if task is None:
        task = config.get("project.task")

    if bids_root is None:
        bids_root = config.bids_root

    from .discovery import get_available_subjects

    if subjects is None or subjects == ["all"]:
        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            bids_root=bids_root,
            task=task,
            discovery_sources=["features"],
            subject_discovery_policy=subject_discovery_policy,
            logger=logger,
        )
    else:
        logger.info(f"Using explicitly provided subjects: {subjects}")

    X_list: List[pd.DataFrame] = []
    y_list: List[pd.Series] = []
    g_list: List[str] = []
    trial_ids: List[int] = []
    subj_ids: List[str] = []

    col_template = None
    n_found = 0

    for s in subjects:
        try:
            X, y, groups, meta = load_decoding_data(
                subject=s,
                deriv_root=deriv_root,
                config=config,
                bids_root=bids_root,
                task=task,
                logger=logger,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(f"Skipping {s}: {exc}")
            continue

        if col_template is None:
            col_template = list(X.columns)
        elif list(X.columns) != col_template:
            common = [c for c in col_template if c in X.columns]
            if not common:
                raise RuntimeError(f"No overlapping features for {s}")
            if len(common) < len(col_template):
                logger.warning(f"Using {len(common)} common features for {s}")
            X = X.loc[:, common]
            col_template = common

        n = len(X)
        X_list.append(X)
        y_list.append(y)
        g_list.extend([meta["subject_id"].iloc[0]] * n)
        trial_ids.extend(list(range(n)))
        subj_ids.extend([meta["subject_id"].iloc[0]] * n)
        n_found += 1

    if n_found == 0:
        raise RuntimeError(
            "No subjects with both features and targets were found. "
            "Run 03_feature_extraction.py first."
        )

    if col_template is None:
        raise RuntimeError("No feature columns detected.")

    X_all = pd.concat([Xi.loc[:, col_template].copy() for Xi in X_list], axis=0, ignore_index=True)
    y_all = pd.concat(y_list, axis=0, ignore_index=True)
    groups = np.array(g_list)

    meta = pd.DataFrame({"subject_id": subj_ids, "trial_id": trial_ids})

    logger.info(
        f"Aggregated features: n_trials={len(X_all)}, n_features={X_all.shape[1]}, n_subjects={n_found}"
    )

    return X_all, y_all, groups, meta


def load_plateau_matrix(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    log: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    if log is None:
        log = logging.getLogger(__name__)

    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    groups: List[str] = []
    meta_blocks: List[pd.DataFrame] = []
    feature_cols: Optional[List[str]] = None

    from .loading import load_epochs_for_analysis

    for sub in subjects:
        epochs, aligned_events = load_epochs_for_analysis(
            sub,
            task,
            align="strict",
            preload=True,
            deriv_root=deriv_root,
            config=config,
            logger=log,
        )
        if epochs is None or aligned_events is None:
            log.warning(f"No epochs for sub-{sub}; skipping")
            continue

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
        df = pd.DataFrame(epochs.get_data().mean(axis=2))
        if feature_cols is None:
            feature_cols = [f"ch{i}" for i in range(df.shape[1])]
        X_sub = df.to_numpy(dtype=float)
        if X_sub.shape[0] != len(y_sub):
            log.warning(f"Mismatch X/y for sub-{sub}; skipping")
            continue

        X_blocks.append(X_sub)
        y_blocks.append(y_sub)
        groups.extend([sub] * len(y_sub))
        meta_blocks.append(pd.DataFrame({"subject_id": [sub] * len(y_sub)}))

    if not X_blocks:
        raise RuntimeError("No subjects with usable epoch data")

    X = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.concat(meta_blocks, axis=0, ignore_index=True)

    finite_mask = np.isfinite(y_all)
    if not np.all(finite_mask):
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
    if log is None:
        log = logging.getLogger(__name__)

    from .loading import load_epochs_for_analysis

    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    groups: List[str] = []
    window_centers: Optional[np.ndarray] = None

    for sub in subjects:
        epochs, aligned_events = load_epochs_for_analysis(
            sub,
            task,
            align="strict",
            preload=True,
            deriv_root=deriv_root,
            config=config,
            logger=log,
        )

        if epochs is None or aligned_events is None:
            log.warning(f"No epochs for sub-{sub}; skipping")
            continue

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

        data = epochs.get_data()
        times = epochs.times
        sfreq = epochs.info["sfreq"]

        window_samples = int(window_size * sfreq)
        step_samples = int(window_step * sfreq)

        n_windows = (len(times) - window_samples) // step_samples + 1
        if n_windows < 1:
            log.warning(f"Epochs too short for windowing in sub-{sub}")
            continue

        if window_centers is None:
            window_centers = np.array(
                [
                    times[i * step_samples + window_samples // 2]
                    for i in range(n_windows)
                ]
            )

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

    finite_mask = np.isfinite(y_all)
    if not np.all(finite_mask):
        X_windows = X_windows[finite_mask]
        y_all = y_all[finite_mask]
        groups_arr = groups_arr[finite_mask]
        meta = meta.loc[finite_mask].reset_index(drop=True)

    if window_centers is None:
        window_centers = np.array([])

    return X_windows, y_all, groups_arr, window_centers, meta
