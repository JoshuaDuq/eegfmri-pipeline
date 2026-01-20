from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.infra.paths import _find_clean_events_path
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis
from ..config.loader import ConfigDict

EEGConfig = ConfigDict


def _filter_finite_targets(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Filter out samples with non-finite target values."""
    finite_mask = np.isfinite(y)
    if np.all(finite_mask):
        return X, y, groups, meta

    X_filtered = X[finite_mask]
    y_filtered = y[finite_mask]
    groups_filtered = groups[finite_mask]
    meta_filtered = meta.loc[finite_mask].reset_index(drop=True)
    return X_filtered, y_filtered, groups_filtered, meta_filtered


def _find_block_column(aligned_events: pd.DataFrame) -> Optional[pd.Series]:
    """Find block/run identifier column from aligned events."""
    for candidate in ("block", "run_id", "run", "session"):
        if candidate in aligned_events.columns:
            return pd.to_numeric(aligned_events[candidate], errors="coerce")
    return None




def load_ml_data(
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
    
    X_path = feat_dir / "power" / "features_power.parquet"

    if not X_path.exists():
        raise FileNotFoundError(
            f"Missing features for {sub}: {X_path} not found"
        )

    clean_events_path = _find_clean_events_path(
        subject=subject,
        task=task,
        deriv_root=deriv_root,
        config=config,
        constants=None,
    )
    
    if clean_events_path is None or not clean_events_path.exists():
        raise FileNotFoundError(
            f"Clean events.tsv not found for {sub}, task-{task}. "
            f"Required to load target ratings."
        )
    
    from eeg_pipeline.infra.tsv import read_table
    X = read_table(X_path)
    events_df = read_table(clean_events_path)
    
    if len(X) != len(events_df):
        raise ValueError(
            f"Feature/events count mismatch for subject {sub}, task {task}: "
            f"features have {len(X)} rows but clean events.tsv has {len(events_df)} rows. "
            f"This indicates a misalignment. Re-run feature extraction to regenerate features."
        )

    rating_columns = config.get("event_columns.rating", [])
    tgt_col = pick_target_column(events_df, target_columns=list(rating_columns) if rating_columns else [])
    if tgt_col is None:
        raise ValueError(
            f"No suitable target column found in clean events.tsv for subject {sub}, task {task}. "
            f"Available columns: {list(events_df.columns)}"
        )

    y = pd.to_numeric(events_df[tgt_col], errors="coerce")

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

    if "trial_index" in events_df.columns:
        trial_index = pd.to_numeric(events_df["trial_index"], errors="coerce")
        trial_index_filtered = trial_index.loc[mask_valid].to_numpy(dtype=float)
    else:
        trial_index_filtered = np.arange(len(X), dtype=float)

    meta = pd.DataFrame(
        {
            "subject_id": [sub] * len(X),
            "trial_id": list(range(len(X))),
            "trial_index": trial_index_filtered,
        }
    )

    logger.info(f"Loaded data for {sub}: n_trials={len(X)}, n_features={X.shape[1]}")

    return X, y, groups, meta


def load_multiple_subjects_ml_data(
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

    first_subject_columns: Optional[List[str]] = None
    column_sets: List[set] = []
    n_found = 0

    for s in subjects:
        try:
            X, y, groups, meta = load_ml_data(
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

        if first_subject_columns is None:
            first_subject_columns = list(X.columns)
        column_sets.append(set(X.columns))

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

    if first_subject_columns is None:
        raise RuntimeError("No feature columns detected.")

    common_columns = (
        set.intersection(*column_sets) if column_sets else set(first_subject_columns)
    )
    harmonized_columns = [c for c in first_subject_columns if c in common_columns]
    if not harmonized_columns:
        raise RuntimeError("No overlapping features across subjects after harmonization.")
    X_all = pd.concat(
        [Xi.loc[:, harmonized_columns].copy() for Xi in X_list], axis=0, ignore_index=True
    )
    y_all = pd.concat(y_list, axis=0, ignore_index=True)
    groups = np.array(g_list)

    meta = pd.DataFrame({"subject_id": subj_ids, "trial_id": trial_ids})

    logger.info(
        f"Aggregated features: n_trials={len(X_all)}, n_features={X_all.shape[1]}, n_subjects={n_found}"
    )

    return X_all, y_all, groups, meta


def load_epochs_with_targets(
    deriv_root: Path,
    config: Optional[EEGConfig] = None,
    subjects: Optional[List[str]] = None,
    subject_discovery_policy: Literal["intersection", "union", "config_only"] = "intersection",
    task: str = "",
    bids_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[str, mne.Epochs, pd.Series]], List[str]]:
    from eeg_pipeline.utils.config.loader import load_config
    
    if config is None:
        config = load_config()
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

        try:
            from eeg_pipeline.infra.paths import find_clean_epochs_path

            epochs_path = find_clean_epochs_path(s, task, deriv_root=deriv_root, config=config)
        except (FileNotFoundError, ValueError, KeyError):
            epochs_path = None

        if epochs_path is None or not Path(epochs_path).exists():
            logger.warning(f"Clean epochs not found for {sub}; skipping.")
            continue

        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
        epochs.set_montage(mne.channels.make_standard_montage("standard_1005"))
        bad_channels = epochs.info.get("bads", [])
        if bad_channels:
            epochs.interpolate_bads(reset_bads=True)

        clean_events_path = _find_clean_events_path(
            subject=str(s),
            task=task,
            deriv_root=deriv_root,
            config=config,
            constants=None,
        )
        
        if clean_events_path is None or not clean_events_path.exists():
            logger.warning(f"Clean events.tsv not found for {sub}; skipping.")
            continue
        
        aligned = read_tsv(clean_events_path)
        
        if len(epochs) != len(aligned):
            raise ValueError(
                f"Epoch/events count mismatch for subject {sub}, task {task}: "
                f"epochs have {len(epochs)} trials but clean events.tsv has {len(aligned)} rows. "
                "Re-run preprocessing to regenerate clean events with the current epochs."
            )
        
        if len(aligned) == 0:
            logger.warning(f"Clean events.tsv is empty for {sub}; skipping.")
            continue

        rating_columns = config.get("event_columns.rating", [])
        tgt_col = pick_target_column(
            aligned, target_columns=list(rating_columns) if rating_columns else []
        )
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
        eeg_channels = [
            ch
            for ch in epochs.info["ch_names"]
            if epochs.get_channel_types(picks=[ch])[0] == "eeg"
        ]
        ch_sets.append(set(eeg_channels))

    if not out:
        raise RuntimeError("No epochs + targets could be loaded for any subject.")

    if not ch_sets:
        return out, []

    if len(ch_sets) > 1:
        common_channels = sorted(set.intersection(*ch_sets))
    else:
        common_channels = sorted(ch_sets[0]) if ch_sets else []
    return out, common_channels


def load_active_matrix(
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

        rating_columns = config.get("event_columns.rating", [])
        target_col = pick_target_column(
            aligned_events, target_columns=list(rating_columns) if rating_columns else []
        )
        if target_col is None:
            log.warning(f"No target column for sub-{sub}; skipping")
            continue

        y_sub = pd.to_numeric(aligned_events[target_col], errors="coerce").to_numpy()

        picks = mne.pick_types(
            epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
        )
        if len(picks) == 0:
            log.warning(f"No EEG channels for sub-{sub}; skipping")
            continue

        data = epochs.get_data(picks=picks)
        times = np.asarray(epochs.times, dtype=float)

        baseline_window = get_config_value(
            config, "time_frequency_analysis.baseline_window", [-3.0, -0.5]
        )
        active_window = get_config_value(
            config, "time_frequency_analysis.active_window", [3.0, 10.5]
        )
        try:
            b0, b1 = float(baseline_window[0]), float(baseline_window[1])
            a0, a1 = float(active_window[0]), float(active_window[1])
        except (ValueError, TypeError, IndexError):
            b0, b1 = -3.0, -0.5
            a0, a1 = 3.0, 10.5

        bmask = (times >= b0) & (times < b1)
        amask = (times >= a0) & (times < a1)
        if not np.any(amask):
            log.warning(f"Active window empty for sub-{sub}; using full epoch mean.")
            amask = np.ones_like(times, dtype=bool)
        if not np.any(bmask):
            log.warning(f"Baseline window empty for sub-{sub}; using zero baseline.")

        active_mean = np.nanmean(data[..., amask], axis=2)
        epochs_baseline = getattr(epochs, "baseline", None)
        epochs_already_baselined = epochs_baseline not in (None, (None, None))

        if epochs_already_baselined:
            log.debug(
                "sub-%s: epochs already baseline-corrected (epochs.baseline=%s); skipping baseline subtraction in ML amplitude features.",
                sub,
                epochs_baseline,
            )
            baseline_mean = np.zeros_like(active_mean)
        elif np.any(bmask):
            baseline_mean = np.nanmean(data[..., bmask], axis=2)
        else:
            baseline_mean = np.zeros_like(active_mean)

        # Baseline-corrected mean amplitude in the active window (scientifically interpretable).
        X_sub = (active_mean - baseline_mean).astype(float)

        if feature_cols is None:
            feature_cols = list(np.asarray(epochs.ch_names)[picks])
        if X_sub.shape[0] != len(y_sub):
            log.warning(f"Mismatch X/y for sub-{sub}; skipping")
            continue

        X_blocks.append(X_sub)
        y_blocks.append(y_sub)
        groups.extend([sub] * len(y_sub))
        meta_rec = {"subject_id": [sub] * len(y_sub)}
        block_series = _find_block_column(aligned_events)
        if block_series is not None:
            meta_rec["block"] = block_series.to_numpy()
        meta_blocks.append(pd.DataFrame(meta_rec))

    if not X_blocks:
        raise RuntimeError("No subjects with usable epoch data")

    X = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.concat(meta_blocks, axis=0, ignore_index=True)

    X, y_all, groups_arr, meta = _filter_finite_targets(X, y_all, groups_arr, meta)

    meta = meta.reset_index(drop=True)
    meta["trial_id"] = np.arange(len(meta), dtype=int)

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

        rating_columns = config.get("event_columns.rating", [])
        target_col = pick_target_column(
            aligned_events, target_columns=list(rating_columns) if rating_columns else []
        )
        if target_col is None:
            log.warning(f"No target column for sub-{sub}; skipping")
            continue

        y_sub = pd.to_numeric(aligned_events[target_col], errors="coerce").to_numpy()

        data = epochs.get_data()
        times = epochs.times

        window_samples = int(window_size * epochs.info["sfreq"])
        step_samples = int(window_step * epochs.info["sfreq"])

        n_windows = (len(times) - window_samples) // step_samples + 1
        if n_windows < 1:
            log.warning(f"Epochs too short for windowing in sub-{sub}")
            continue

        if window_centers is None:
            window_centers = np.array([
                times[i * step_samples + window_samples // 2] for i in range(n_windows)
            ])

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

    X_windows, y_all, groups_arr, meta = _filter_finite_targets(
        X_windows, y_all, groups_arr, meta
    )

    if window_centers is None:
        window_centers = np.array([], dtype=float)

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
    """Extract epoch data for specified trial indices."""
    X_list = []
    for idx in indices:
        subject_id, trial_idx = trial_records[int(idx)]
        epochs = aligned_epochs[subject_id]
        X_i = epochs.get_data(picks="eeg", reject_by_annotation=None)[trial_idx]
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

    for subject_id, epochs, y in tuples:
        n_trials = min(len(epochs), len(y))
        if n_trials == 0:
            continue
        subj_to_epochs[subject_id] = epochs
        y_numeric = pd.to_numeric(y.iloc[:n_trials], errors="coerce")
        subj_to_y[subject_id] = y_numeric
        for trial_idx in range(n_trials):
            trial_records.append((subject_id, trial_idx))
            y_all_list.append(float(y_numeric.iloc[trial_idx]))
            groups_list.append(subject_id)

    if len(trial_records) == 0:
        raise RuntimeError("No trial data available.")

    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    return trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y


def extract_channel_importance_from_coefficients(
    coef_matrix: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract channel importance from coefficient matrix using feature naming schema."""
    from eeg_pipeline.domain.features.naming import NamingSchema

    channel_band_to_indices: Dict[Tuple[str, str], List[int]] = {}
    for idx, feature_name in enumerate(feature_names):
        parsed = NamingSchema.parse(str(feature_name))
        is_valid_power = parsed.get("valid") and parsed.get("group") == "power"
        is_channel_scope = parsed.get("scope") == "ch"
        if not (is_valid_power and is_channel_scope):
            continue

        band = parsed.get("band")
        channel = parsed.get("identifier")
        if band and channel:
            key = (str(channel), str(band))
            channel_band_to_indices.setdefault(key, []).append(idx)

    channel_to_all_indices: Dict[str, List[int]] = {}
    for (channel, _band), indices in channel_band_to_indices.items():
        channel_to_all_indices.setdefault(channel, []).extend(indices)

    n_folds = coef_matrix.shape[0]
    channel_importance_data: List[Dict[str, float]] = []

    for channel, indices in channel_to_all_indices.items():
        channel_coefs = coef_matrix[:, indices]
        channel_mean_abs = np.nanmean(np.abs(channel_coefs), axis=1)

        for fold_idx in range(n_folds):
            importance_value = channel_mean_abs[fold_idx]
            if np.isfinite(importance_value):
                channel_importance_data.append(
                    {
                        "channel": str(channel),
                        "importance": float(importance_value),
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


###################################################################
# Unified Feature Matrix Builder
###################################################################


def ml_build_feature_matrix(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    feature_set: Literal["channels_mean", "power", "connectivity", "itpc", "aperiodic", "combined"] = "channels_mean",
    log: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Unified feature matrix builder for ML pipelines.
    
    Provides a single entry point for building feature matrices with different
    feature-set options, reusing the same feature registry conventions as behavior.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs
    task : str
        Task name
    deriv_root : Path
        Derivatives root directory
    config : Any
        Configuration object
    feature_set : str
        Feature set to use:
        - 'channels_mean': Baseline-corrected mean amplitude by channel (default, fast)
        - 'power': Band power features from feature extraction
        - 'connectivity': Connectivity features (wPLI, AEC)
        - 'itpc': Inter-trial phase coherence features
        - 'aperiodic': Aperiodic (1/f) features
        - 'combined': All available feature types concatenated
    log : Logger, optional
        Logger instance
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target values
    groups : np.ndarray
        Subject group labels
    feature_names : List[str]
        Feature column names
    meta : pd.DataFrame
        Trial metadata with subject_id, block, trial_id
    """
    if log is None:
        log = logging.getLogger(__name__)
    
    if feature_set == "channels_mean":
        return load_active_matrix(subjects, task, deriv_root, config, log)
    
    from eeg_pipeline.utils.data.feature_io import load_feature_bundle
    from eeg_pipeline.infra.paths import deriv_features_path
    
    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    groups: List[str] = []
    meta_blocks: List[pd.DataFrame] = []
    feature_cols: Optional[List[str]] = None
    
    for sub in subjects:
        bundle = load_feature_bundle(sub, deriv_root, log, include_targets=True, config=config)
        
        if bundle.empty or bundle.targets is None:
            log.warning(f"No features or targets for sub-{sub}; skipping")
            continue
        
        y_sub = bundle.targets.to_numpy()

        feature_dfs = []
        include_all = feature_set == "combined"

        if feature_set == "power" or include_all:
            if bundle.power_df is not None and not bundle.power_df.empty:
                numeric_cols = bundle.power_df.select_dtypes(include=[np.number]).columns
                feature_dfs.append(bundle.power_df[numeric_cols])

        if feature_set == "connectivity" or include_all:
            if bundle.connectivity_df is not None and not bundle.connectivity_df.empty:
                numeric_cols = bundle.connectivity_df.select_dtypes(include=[np.number]).columns
                feature_dfs.append(bundle.connectivity_df[numeric_cols])

        if feature_set == "itpc" or include_all:
            if bundle.itpc_df is not None and not bundle.itpc_df.empty:
                numeric_cols = bundle.itpc_df.select_dtypes(include=[np.number]).columns
                feature_dfs.append(bundle.itpc_df[numeric_cols])

        if feature_set == "aperiodic" or include_all:
            if bundle.aperiodic_df is not None and not bundle.aperiodic_df.empty:
                numeric_cols = bundle.aperiodic_df.select_dtypes(include=[np.number]).columns
                feature_dfs.append(bundle.aperiodic_df[numeric_cols])

        if not feature_dfs:
            log.warning(f"No {feature_set} features found for sub-{sub}; skipping")
            continue

        features_df = feature_dfs[0] if len(feature_dfs) == 1 else pd.concat(feature_dfs, axis=1)
        
        if len(features_df) != len(y_sub):
            log.warning(
                f"Feature/target length mismatch for sub-{sub}: "
                f"{len(features_df)} != {len(y_sub)}; skipping"
            )
            continue

        if feature_cols is None:
            feature_cols = list(features_df.columns)
        elif list(features_df.columns) != feature_cols:
            common_cols = [c for c in feature_cols if c in features_df.columns]
            if not common_cols:
                log.warning(f"No common features for sub-{sub}; skipping")
                continue
            features_df = features_df[common_cols]
            feature_cols = common_cols

        X_sub = features_df.to_numpy().astype(float)
        
        X_blocks.append(X_sub)
        y_blocks.append(y_sub)
        groups.extend([sub] * len(y_sub))
        
        meta_rec = {"subject_id": [sub] * len(y_sub)}
        meta_blocks.append(pd.DataFrame(meta_rec))
    
    if not X_blocks:
        raise RuntimeError(f"No subjects with usable {feature_set} features")
    
    X = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.concat(meta_blocks, axis=0, ignore_index=True)

    X, y_all, groups_arr, meta = _filter_finite_targets(X, y_all, groups_arr, meta)

    meta = meta.reset_index(drop=True)
    meta["trial_id"] = np.arange(len(meta), dtype=int)

    log.info(
        f"Built ML feature matrix: {feature_set}, shape={X.shape}, "
        f"n_subjects={len(np.unique(groups_arr))}"
    )

    return X, y_all, groups_arr, feature_cols or [], meta
