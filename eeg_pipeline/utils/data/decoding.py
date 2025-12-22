from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.infra.tsv import read_tsv
from ..config.loader import ConfigDict



EEGConfig = ConfigDict


def _get_trial_alignment_manifest_path(deriv_root: Path, subject: str) -> Path:
    sub = f"sub-{subject}" if not subject.startswith("sub-") else subject
    base = deriv_root / sub / "eeg" / "features"
    json_path = base / "trial_alignment.json"
    if json_path.exists():
        return json_path
    return base / "trial_alignment.tsv"


def _load_trial_alignment_manifest(
    manifest_path: Path,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    import json as json_module
    
    if logger is None:
        logger = logging.getLogger(__name__)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Trial alignment manifest missing: {manifest_path}\n"
            f"This file is required and must be created by 03_feature_extraction.py. "
            f"Run 03_feature_extraction.py first to generate features with proper alignment."
        )

    if manifest_path.suffix == ".json" or manifest_path.name.endswith(".tsv"):
        try:
            with open(manifest_path, "r") as f:
                content = f.read().strip()
            if content.startswith("{"):
                data = json_module.loads(content)
                n_epochs = data.get("n_epochs", 0)
                manifest = pd.DataFrame({"trial_index": list(range(n_epochs))})
                logger.debug(f"Loaded JSON trial alignment manifest: {n_epochs} trials from {manifest_path}")
                return manifest
        except (json_module.JSONDecodeError, KeyError):
            pass
    
    manifest = read_tsv(manifest_path)
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
    X_path = feat_dir / "features_power.tsv"
    y_path = feat_dir / "target_vas_ratings.tsv"
    manifest_path = _get_trial_alignment_manifest_path(deriv_root, subject)

    if not (X_path.exists() and y_path.exists()):
        raise FileNotFoundError(
            f"Missing features/targets for {sub}: {X_path} or {y_path} not found"
        )

    manifest = _load_trial_alignment_manifest(manifest_path, logger)
    expected_n_trials = len(manifest)

    X = read_tsv(X_path)
    y_df = read_tsv(y_path)

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

    from .subjects import get_available_subjects

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


def load_epochs_with_targets(
    deriv_root: Path,
    config: EEGConfig,
    subjects: Optional[List[str]] = None,
    subject_discovery_policy: Literal["intersection", "union", "config_only"] = "intersection",
    task: str = "",
    bids_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[str, mne.Epochs, pd.Series]], List[str]]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if task == "":
        task = config.get("project.task")
    if bids_root is None:
        bids_root = config.bids_root

    if subjects is None or subjects == ["all"]:
        from .subjects import get_available_subjects

        subjects = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            bids_root=bids_root,
            task=task,
            discovery_sources=["features"],
            subject_discovery_policy=subject_discovery_policy,
            logger=logger,
        )

    from .epochs import load_epochs_for_analysis

    out: List[Tuple[str, mne.Epochs, pd.Series]] = []
    ch_sets: List[set] = []

    for s in subjects:
        sub = f"sub-{s}" if not str(s).startswith("sub-") else str(s)

        epochs_path = None
        try:
            from eeg_pipeline.infra.paths import find_clean_epochs_path

            epochs_path = find_clean_epochs_path(s, task, deriv_root=deriv_root, config=config)
        except Exception:
            epochs_path = None

        if epochs_path is None or not Path(epochs_path).exists():
            logger.warning(f"Clean epochs not found for {sub}; skipping.")
            continue

        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        epochs.set_montage(mne.channels.make_standard_montage("standard_1005"))
        if len(epochs.info.get("bads", [])) > 0:
            epochs.interpolate_bads(reset_bads=True)

        manifest_path = _get_trial_alignment_manifest_path(deriv_root, str(s))
        manifest = _load_trial_alignment_manifest(manifest_path, logger)
        if len(epochs) != len(manifest):
            raise ValueError(
                f"Epoch count mismatch for subject {sub}, task {task}: "
                f"epochs have {len(epochs)} trials but trial_alignment.tsv specifies {len(manifest)} trials. "
                "Re-run feature extraction to regenerate features with the current epochs."
            )

        _, aligned = load_epochs_for_analysis(
            str(s),
            task,
            align="strict",
            preload=False,
            deriv_root=deriv_root,
            bids_root=bids_root,
            config=config,
            logger=logger,
        )
        if aligned is None or len(aligned) == 0:
            logger.warning(f"No aligned events/targets for {sub}; skipping.")
            continue

        target_columns = list(config.get("event_columns.rating", []) or [])
        tgt_col = pick_target_column(aligned, target_columns=target_columns)
        if tgt_col is None:
            logger.warning(f"No suitable target column for {sub}; skipping.")
            continue

        y = pd.to_numeric(aligned[tgt_col], errors="coerce")
        if len(epochs) != len(y):
            logger.error(
                f"Epochs-target length mismatch for subject {sub}, task {task}: "
                f"epochs={len(epochs)}, y={len(y)}. Skipping subject."
            )
            continue

        if len(epochs) == 0:
            logger.warning(f"No trials for {sub}; skipping.")
            continue

        out.append((sub, epochs, y))
        ch_sets.append(
            set(
                [
                    ch
                    for ch in epochs.info["ch_names"]
                    if epochs.get_channel_types(picks=[ch])[0] == "eeg"
                ]
            )
        )

    if not out:
        raise RuntimeError("No epochs + targets could be loaded for any subject.")

    if not ch_sets:
        return out, []

    common_channels = (
        sorted(set.intersection(*ch_sets)) if len(ch_sets) > 1 else sorted(ch_sets[0])
    )
    return out, common_channels


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

    from .epochs import load_epochs_for_analysis

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

    from .epochs import load_epochs_for_analysis

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


def filter_finite_targets(
    indices: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    target_values = targets[indices]
    finite_mask = np.isfinite(target_values)
    filtered_indices = indices[finite_mask]
    filtered_targets = target_values[finite_mask]
    return filtered_indices, filtered_targets


def extract_epoch_data_block(
    indices: np.ndarray,
    trial_records: List[Tuple[str, int]],
    aligned_epochs: Dict[str, mne.Epochs],
) -> np.ndarray:
    X_list = []
    for i in indices:
        sub_i, ti = trial_records[int(i)]
        try:
            X_i = aligned_epochs[sub_i].get_data(picks="eeg", reject_by_annotation=None)[ti]
        except TypeError:
            X_i = aligned_epochs[sub_i].get_data(picks="eeg")[ti]
        X_list.append(X_i)
    return np.stack(X_list, axis=0)


def prepare_trial_records_from_epochs(
    tuples: List[Tuple[str, mne.Epochs, pd.Series]],
) -> Tuple[
    List[Tuple[str, int]],
    np.ndarray,
    np.ndarray,
    Dict[str, mne.Epochs],
    Dict[str, pd.Series],
]:
    trial_records: List[Tuple[str, int]] = []
    y_all_list: List[float] = []
    groups_list: List[str] = []
    subj_to_epochs: Dict[str, mne.Epochs] = {}
    subj_to_y: Dict[str, pd.Series] = {}

    for sub, epochs, y in tuples:
        n = min(len(epochs), len(y))
        if n == 0:
            continue
        subj_to_epochs[sub] = epochs
        subj_to_y[sub] = pd.to_numeric(y.iloc[:n], errors="coerce")
        for ti in range(n):
            trial_records.append((sub, ti))
            y_all_list.append(float(subj_to_y[sub].iloc[ti]))
            groups_list.append(sub)

    if len(trial_records) == 0:
        raise RuntimeError("No trial data available.")

    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    return trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y


def load_kept_indices(
    subject_label: str,
    deriv_root: Path,
    n_events: int,
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    if logger is None:
        logger = logging.getLogger(__name__)

    dropped_path = deriv_root / subject_label / "eeg" / "features" / "dropped_trials.tsv"
    if not dropped_path.exists():
        return None

    dropped_df = read_tsv(dropped_path)
    if "original_index" not in dropped_df.columns:
        return None

    dropped_indices_raw = pd.to_numeric(dropped_df["original_index"], errors="coerce").dropna()
    if len(dropped_indices_raw) == 0:
        return None

    dropped_indices = set(dropped_indices_raw.astype(int).tolist())
    kept_indices = np.array([i for i in range(n_events) if i not in dropped_indices])
    logger.info(f"{subject_label}: {len(dropped_indices)} trials dropped, {len(kept_indices)} kept")
    return kept_indices


def extract_channel_importance_from_coefficients(
    coef_matrix: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    from eeg_pipeline.domain.features.naming import NamingSchema

    channel_band_to_indices: Dict[Tuple[str, str], List[int]] = {}
    for idx, feat in enumerate(feature_names):
        parsed = NamingSchema.parse(str(feat))
        if not (parsed.get("valid") and parsed.get("group") == "power"):
            continue
        if parsed.get("scope") != "ch":
            continue
        band = parsed.get("band")
        channel = parsed.get("identifier")
        if band and channel:
            channel_band_to_indices.setdefault((str(channel), str(band)), []).append(idx)

    channel_to_all_indices: Dict[str, List[int]] = {}
    for (channel, _band), indices in channel_band_to_indices.items():
        channel_to_all_indices.setdefault(channel, []).extend(indices)

    n_folds = coef_matrix.shape[0]
    channel_importance_data: List[Dict[str, float]] = []

    for channel, indices in channel_to_all_indices.items():
        channel_coefs = coef_matrix[:, indices]
        channel_mean_abs = np.nanmean(np.abs(channel_coefs), axis=1)

        for fold_idx in range(n_folds):
            if np.isfinite(channel_mean_abs[fold_idx]):
                channel_importance_data.append(
                    {
                        "channel": str(channel),
                        "importance": float(channel_mean_abs[fold_idx]),
                        "fold": float(fold_idx),
                    }
                )

    return pd.DataFrame(channel_importance_data)


def extract_importance_column(
    importance_df: pd.DataFrame,
    top_n: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if "mean_abs_shap" in importance_df.columns:
        df_sorted = importance_df.sort_values("mean_abs_shap", ascending=False).head(top_n)
        return df_sorted["mean_abs_shap"].values, "Mean |SHAP value|"

    if "importance" in importance_df.columns:
        df_sorted = importance_df.sort_values("importance", ascending=False).head(top_n)
        return df_sorted["importance"].values, "Importance (ΔR²)"

    return None, None
