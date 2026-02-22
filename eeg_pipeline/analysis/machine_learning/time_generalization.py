from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import warnings
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests

from eeg_pipeline.utils.analysis.tfr import (
    find_common_channels_train_test,
)
from eeg_pipeline.utils.analysis.windowing import build_time_windows
from eeg_pipeline.utils.data.machine_learning import (
    filter_finite_targets,
    extract_epoch_data_block,
    prepare_trial_records_from_epochs,
)
from eeg_pipeline.utils.data.machine_learning import load_epochs_with_targets
from eeg_pipeline.analysis.machine_learning.cv import (
    get_min_channels_required,
    is_effective_permutation,
    safe_pearsonr,
)
from eeg_pipeline.utils.config.loader import load_config, get_fisher_z_clip_values, get_config_value
from eeg_pipeline.infra.logging import get_logger

logger = get_logger(__name__)


def _extract_trial_blocks_from_records(
    trial_records: List[Tuple[str, int]],
    subj_to_epochs: Dict[str, Any],
) -> np.ndarray:
    """Extract per-trial block/run labels aligned to trial_records."""
    candidate_cols = ("block", "run_id", "run", "session", "run_num")
    block_values = np.full(len(trial_records), np.nan, dtype=object)
    block_series_by_subject: Dict[str, Optional[pd.Series]] = {}

    for subject_id, epochs in subj_to_epochs.items():
        metadata = getattr(epochs, "metadata", None)
        if not isinstance(metadata, pd.DataFrame) or metadata.empty:
            block_series_by_subject[str(subject_id)] = None
            continue
        col = next((c for c in candidate_cols if c in metadata.columns), None)
        if col is None:
            block_series_by_subject[str(subject_id)] = None
            continue
        block_series_by_subject[str(subject_id)] = metadata[col].reset_index(drop=True)

    for i, (subject_id, trial_idx) in enumerate(trial_records):
        series = block_series_by_subject.get(str(subject_id))
        if series is None:
            continue
        idx = int(trial_idx)
        if idx < 0 or idx >= len(series):
            continue
        val = series.iloc[idx]
        block_values[i] = np.nan if pd.isna(val) else val

    return block_values


def _permute_labels_within_subject_structure(
    y: np.ndarray,
    groups: np.ndarray,
    blocks: Optional[np.ndarray],
    *,
    rng: np.random.Generator,
    scheme: str,
) -> np.ndarray:
    """Permute labels within subject or within subject×block."""
    y_perm = np.asarray(y, dtype=float).copy()
    groups_arr = np.asarray(groups, dtype=object)
    blocks_arr = np.asarray(blocks, dtype=object) if blocks is not None else None
    perm_scheme = str(scheme).strip().lower()
    if perm_scheme not in {"within_subject", "within_subject_within_block"}:
        perm_scheme = "within_subject_within_block"
    if perm_scheme == "within_subject_within_block" and blocks_arr is None:
        perm_scheme = "within_subject"

    for subject_id in np.unique(groups_arr):
        subject_mask = groups_arr == subject_id
        if perm_scheme == "within_subject_within_block" and blocks_arr is not None:
            for block_label in pd.unique(blocks_arr[subject_mask]):
                if pd.isna(block_label):
                    block_mask = subject_mask & pd.isna(blocks_arr)
                else:
                    block_mask = subject_mask & (blocks_arr == block_label)
                if np.sum(block_mask) >= 2:
                    y_perm[block_mask] = rng.permutation(y_perm[block_mask])
        else:
            y_perm[subject_mask] = rng.permutation(y_perm[subject_mask])
    return y_perm


def _aggregate_time_generalization_matrices(
    stacked_r: np.ndarray,
    stacked_r2: np.ndarray,
    stacked_counts: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate fold-level TG matrices using subject-first statistics."""
    n_windows = stacked_r.shape[1]
    tg_r = np.full((n_windows, n_windows), np.nan, dtype=float)
    tg_r2 = np.full_like(tg_r, np.nan)
    coverage_map = np.zeros_like(tg_r, dtype=int)
    subject_coverage_map = np.zeros_like(tg_r, dtype=int)

    min_subjects_per_cell = int(
        get_config_value(config, "machine_learning.analysis.time_generalization.min_subjects_per_cell", 2)
    )
    min_count_per_cell = int(
        get_config_value(config, "machine_learning.analysis.time_generalization.min_count_per_cell", 15)
    )
    clip_min, clip_max = get_fisher_z_clip_values(config)

    for i in range(n_windows):
        for j in range(n_windows):
            cell_r = stacked_r[:, i, j]
            cell_r2 = stacked_r2[:, i, j]
            cell_counts = stacked_counts[:, i, j]

            finite_mask = np.isfinite(cell_r) & np.isfinite(cell_r2) & (cell_counts > 0)
            if not np.any(finite_mask):
                continue

            valid_r = cell_r[finite_mask]
            valid_r2 = cell_r2[finite_mask]
            valid_counts = cell_counts[finite_mask]
            coverage_map[i, j] = int(valid_counts.sum())
            subject_coverage_map[i, j] = int(np.sum(finite_mask))

            # Subject-level reliability is primary: require minimum number of subjects first.
            if subject_coverage_map[i, j] < min_subjects_per_cell:
                continue
            if coverage_map[i, j] < min_count_per_cell:
                continue

            r_clipped = np.clip(valid_r, clip_min, clip_max)
            tg_r[i, j] = float(np.tanh(np.mean(np.arctanh(r_clipped))))
            # Predeclared rule for R² aggregation: equal-subject weighting.
            tg_r2[i, j] = float(np.mean(valid_r2))

    tested_mask = np.isfinite(tg_r)
    return tg_r, tg_r2, coverage_map, subject_coverage_map, tested_mask


def _compute_time_generalization_significance(
    *,
    tg_r: np.ndarray,
    null_r: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TG significance maps using only empirically tested cells."""
    tested_mask = np.isfinite(tg_r)
    p_matrix = np.full_like(tg_r, np.nan, dtype=float)
    sig_fdr = np.zeros_like(tg_r, dtype=bool)
    sig_maxstat = np.zeros_like(tg_r, dtype=bool)
    sig_cluster = np.zeros_like(tg_r, dtype=bool)

    if not isinstance(null_r, np.ndarray) or null_r.size == 0 or tg_r.size == 0 or not np.any(tested_mask):
        return p_matrix, sig_fdr, sig_maxstat, sig_cluster, tested_mask

    n_perm_valid = int(null_r.shape[0])

    for i in range(tg_r.shape[0]):
        for j in range(tg_r.shape[1]):
            if not tested_mask[i, j]:
                continue
            null_vals = null_r[:, i, j]
            finite_null = null_vals[np.isfinite(null_vals)]
            if finite_null.size > 0:
                p_matrix[i, j] = float(
                    (np.sum(np.abs(finite_null) >= np.abs(tg_r[i, j])) + 1) / (finite_null.size + 1)
                )

    tested_p = p_matrix[tested_mask]
    finite_p = np.isfinite(tested_p)
    if np.any(finite_p):
        reject = np.zeros_like(tested_p, dtype=bool)
        _, p_fdr_valid, _, _ = multipletests(tested_p[finite_p], alpha=0.05, method="fdr_bh")
        reject[finite_p] = p_fdr_valid < 0.05
        sig_fdr[tested_mask] = reject

    null_abs_masked = np.where(tested_mask[None, :, :], np.abs(null_r), np.nan)
    null_max_abs = np.nanmax(null_abs_masked, axis=(1, 2))
    finite_null_max = null_max_abs[np.isfinite(null_max_abs)]
    max_stat_threshold = float(np.percentile(finite_null_max, 95)) if finite_null_max.size > 0 else np.inf
    sig_maxstat = tested_mask & (np.abs(tg_r) >= max_stat_threshold)

    cluster_alpha = float(
        get_config_value(config, "machine_learning.analysis.time_generalization.cluster_threshold", 0.05)
    )
    null_tested_vals = null_r[:, tested_mask]
    finite_null_abs = np.abs(null_tested_vals[np.isfinite(null_tested_vals)])
    if finite_null_abs.size == 0:
        cluster_stat_threshold = np.inf
    else:
        cluster_stat_threshold = float(np.percentile(finite_null_abs, 100.0 * (1.0 - cluster_alpha)))

    uncorrected_sig = tested_mask & (np.abs(tg_r) >= cluster_stat_threshold)
    labeled_clusters, n_clusters = ndimage.label(uncorrected_sig)
    if n_clusters > 0:
        observed_cluster_sizes = ndimage.sum(uncorrected_sig, labeled_clusters, range(1, n_clusters + 1))
        null_max_clusters = np.zeros(n_perm_valid, dtype=float)
        for perm_idx in range(n_perm_valid):
            null_map = null_r[perm_idx]
            null_sig = tested_mask & np.isfinite(null_map) & (np.abs(null_map) >= cluster_stat_threshold)
            labeled_null, n_null_clusters = ndimage.label(null_sig)
            if n_null_clusters > 0:
                null_cluster_sizes = ndimage.sum(null_sig, labeled_null, range(1, n_null_clusters + 1))
                null_max_clusters[perm_idx] = float(np.max(null_cluster_sizes))

        cluster_size_threshold = float(np.percentile(null_max_clusters, 95)) if n_perm_valid > 0 else np.inf
        for cluster_id in range(1, n_clusters + 1):
            if observed_cluster_sizes[cluster_id - 1] >= cluster_size_threshold:
                sig_cluster |= labeled_clusters == cluster_id

    return p_matrix, sig_fdr, sig_maxstat, sig_cluster, tested_mask


def _extract_window_features(
    data: np.ndarray,
    trial_records: List[Tuple[str, int]],
    aligned_epochs: Dict[str, Any],
    windows: List[Tuple[float, float]],
) -> np.ndarray:
    """Compute mean channel activity within specified time windows for each trial.

    Parameters
    ----------
    data
        Array of shape ``(n_trials, n_channels, n_timepoints)`` containing the
        time-domain data for each trial.
    trial_records
        List of ``(subject, epoch_index)`` tuples that index into
        ``aligned_epochs`` for each trial in ``data``.
    aligned_epochs
        Mapping from subject ID to MNE ``Epochs`` objects that have already been
        restricted to a common channel set.
    windows
        List of ``(window_start, window_end)`` time boundaries (in seconds) in
        the *epochs* time coordinate system.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_trials, n_windows, n_channels)`` where each entry
        contains the mean activity of a channel within a given time window for a
        given trial. Entries are ``np.nan`` when the requested window lies
        outside the available time range or when it contains no samples.
    """
    n_trials, n_channels, n_timepoints = data.shape
    n_windows = len(windows)
    
    features = np.full((n_trials, n_windows, n_channels), np.nan, dtype=float)
    
    for trial_idx in range(n_trials):
        sub, epoch_idx = trial_records[trial_idx]
        epochs = aligned_epochs[sub]
        times = epochs.times
        
        trial_data = data[trial_idx]
        
        for window_idx, (window_start, window_end) in enumerate(windows):
            if window_start < times[0] or window_end > times[-1]:
                continue
            
            time_mask = (times >= window_start) & (times <= window_end)
            if not np.any(time_mask):
                continue
            
            window_data = trial_data[:, time_mask]
            features[trial_idx, window_idx, :] = np.mean(window_data, axis=1)
    
    return features


def time_generalization_regression(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    results_dir: Optional[Path] = None,
    config_dict: Optional[dict] = None,
    n_perm: int = 0,
    seed: int = 42,
    target: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run time-generalization regression with LOSO cross-validation.

    This function trains ridge regression models in one time window and tests
    them in all other windows, yielding time-by-time generalization matrices of
    prediction performance. Group labels are taken as subject identifiers and a
    leave-one-subject-out (LOSO) scheme is used.

    Parameters
    ----------
    deriv_root
        Root directory containing derivative EEG data and behavioral targets.
    subjects
        Optional list of subject identifiers to include. If ``None``, all
        available subjects are used.
    task
        Task identifier string used when loading epochs and targets.
    results_dir
        If provided, results are saved as a compressed ``.npz`` archive in this
        directory. When ``None``, no files are written.
    config_dict
        Optional configuration dictionary. When ``None``, configuration is
        loaded via :func:`load_config`.
    n_perm
        Number of label-permutation runs used to build a null distribution for
        inference. When ``0``, no permutations are performed and no
        inferential statistics are computed.
    seed
        Seed for the permutation random number generator.
    target
        Optional target selector from events.tsv (e.g., ``"rating"``,
        ``"temperature"``, or explicit column name). Defaults to rating when
        omitted.

    Returns
    -------
    tg_r
        Time-generalization matrix of Fisher-z–averaged Pearson correlations
        (shape ``(n_windows, n_windows)``). May be empty when no valid folds
        are available.
    tg_r2
        Corresponding matrix of averaged coefficient of determination
        (R²) scores with the same shape as ``tg_r``.
    window_centers_out
        One-dimensional array of window center times (in seconds) used for both
        axes of the time-generalization matrices. Empty if no valid windows are
        available.
    """
    tuples, _ = load_epochs_with_targets(
        deriv_root,
        subjects=subjects,
        task=task,
        target=target,
        target_kind="continuous",
    )
    trial_records, y_all_arr, groups_arr, subj_to_epochs, _ = prepare_trial_records_from_epochs(tuples)
    trial_blocks_arr = _extract_trial_blocks_from_records(trial_records, subj_to_epochs)

    config = config_dict or load_config()
    min_subjects_for_loso = config.get("analysis.min_subjects_for_group", 2)
    if len(np.unique(groups_arr)) < min_subjects_for_loso:
        raise RuntimeError(f"Need at least {min_subjects_for_loso} subjects for LOSO.")

    active_window = tuple(config.get("machine_learning.analysis.time_generalization.active_window", [3.0, 10.5]))
    window_len = config.get("machine_learning.analysis.time_generalization.window_len", 0.75)
    step = config.get("machine_learning.analysis.time_generalization.step", 0.25)

    tmin_pl, tmax_pl = active_window
    windows = build_time_windows(window_len, step, tmin_pl, tmax_pl)

    if not windows:
        logger.warning("No valid time windows found.")
        return np.array([]), np.array([]), np.array([])

    logo = LeaveOneGroupOut()
    n_folds_total = len(list(logo.split(np.arange(len(trial_records)), groups=groups_arr)))
    
    def _run_time_gen(y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fold_mats_r = []
        fold_mats_r2 = []
        fold_counts = []
        window_centers_out = None
        n_folds_successful = 0
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(np.arange(len(trial_records)), groups=groups_arr), start=1):
            train_subjects = list({trial_records[i][0] for i in train_idx if trial_records[i][0] is not None})
            test_subject = trial_records[int(test_idx[0])][0] if len(test_idx) else None
            
            common_chs = find_common_channels_train_test(train_subjects, test_subject, subj_to_epochs)
            if not common_chs:
                continue
            
            min_channels_required = get_min_channels_required(config)
            if len(common_chs) < min_channels_required:
                logger.warning(
                    f"Fold {fold}: Common channels ({len(common_chs)}) < minimum required "
                    f"({min_channels_required}). Skipping fold."
                )
                continue
            
            subjects_in_fold = list({trial_records[i][0] for i in np.concatenate([train_idx, test_idx])})
            aligned_epochs = {s: subj_to_epochs[s].copy().pick(common_chs) for s in subjects_in_fold}
            
            train_idx_f, y_train = filter_finite_targets(train_idx, y_values)
            test_idx_f, y_test = filter_finite_targets(test_idx, y_values)
            if len(train_idx_f) == 0 or len(test_idx_f) == 0:
                continue
            
            X_train = extract_epoch_data_block(train_idx_f, trial_records, aligned_epochs)
            X_test = extract_epoch_data_block(test_idx_f, trial_records, aligned_epochs)
            
            train_trial_records = [trial_records[int(i)] for i in train_idx_f]
            test_trial_records = [trial_records[int(i)] for i in test_idx_f]
            
            train_feats = _extract_window_features(X_train, train_trial_records, aligned_epochs, windows)
            test_feats = _extract_window_features(X_test, test_trial_records, aligned_epochs, windows)
            
            n_windows = len(windows)
            r_mat = np.full((n_windows, n_windows), np.nan, dtype=float)
            r2_mat = np.full_like(r_mat, np.nan)
            count_mat = np.zeros_like(r_mat, dtype=int)
            
            if window_centers_out is None:
                aligned_epochs[subjects_in_fold[0]]
                window_centers_out = np.array([(w_start + w_end) / 2 for w_start, w_end in windows])
            
            min_samples_per_window = config.get(
                "machine_learning.analysis.time_generalization.min_samples_per_window", 15
            )
            
            for i in range(n_windows):
                train_feat_i = train_feats[:, i, :]
                finite_mask_train = np.isfinite(train_feat_i).any(axis=1)
                if np.sum(finite_mask_train) < min_samples_per_window:
                    continue
                
                train_feat_i_clean = train_feat_i[finite_mask_train].copy()
                col_mask = np.isfinite(train_feat_i_clean).sum(axis=0) > 0
                if not col_mask.any():
                    continue
                
                train_feat_i_clean = train_feat_i_clean[:, col_mask]
                groups_train_i = groups_arr[train_idx_f][finite_mask_train]
                y_train_i = y_train[finite_mask_train]

                use_ridgecv = config.get("machine_learning.analysis.time_generalization.use_ridgecv", False)
                alpha_grid = config.get("machine_learning.analysis.time_generalization.alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0])

                model = None
                if use_ridgecv and len(y_train_i) >= 5:
                    n_unique_groups = len(np.unique(groups_train_i))
                    if n_unique_groups >= 2:
                        try:
                            n_splits_inner = min(5, n_unique_groups)
                            inner_cv = GroupKFold(n_splits=n_splits_inner)
                            grid = GridSearchCV(
                                estimator=TransformedTargetRegressor(
                                    regressor=Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler()), ("ridge", Ridge())]),
                                    transformer=PowerTransformer(
                                        method=str(config.get("machine_learning.preprocessing.power_transformer_method", "yeo-johnson")),
                                        standardize=bool(config.get("machine_learning.preprocessing.power_transformer_standardize", True))
                                    )
                                ),
                                param_grid={"regressor__ridge__alpha": alpha_grid},
                                scoring="r2",
                                cv=inner_cv,
                                n_jobs=1,
                                refit=True,
                                error_score="raise",
                            )
                            grid.fit(train_feat_i_clean, y_train_i, groups=groups_train_i)
                            model = grid.best_estimator_
                        except Exception:
                            model = None

                if model is None:
                    default_alpha = config.get("machine_learning.analysis.time_generalization.default_alpha", 1.0)
                    model = TransformedTargetRegressor(
                        regressor=Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=default_alpha))]),
                        transformer=PowerTransformer(
                            method=str(config.get("machine_learning.preprocessing.power_transformer_method", "yeo-johnson")),
                            standardize=bool(config.get("machine_learning.preprocessing.power_transformer_standardize", True))
                        )
                    )
                    try:
                        model.fit(train_feat_i_clean, y_train_i)
                    except Exception:
                        continue

                for j in range(n_windows):
                    test_feat_j = test_feats[:, j, :]
                    finite_mask_test = np.isfinite(test_feat_j).any(axis=1)
                    if np.sum(finite_mask_test) < min_samples_per_window:
                        continue
                    
                    test_feat_j_clean = test_feat_j[finite_mask_test].copy()
                    test_feat_j_clean = test_feat_j_clean[:, col_mask]
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UserWarning)
                            y_pred = model.predict(test_feat_j_clean)
                        
                        finite_mask_pred = np.isfinite(y_pred) & np.isfinite(y_test[finite_mask_test])
                        if np.sum(finite_mask_pred) < min_samples_per_window:
                            continue
                        y_test_finite = y_test[finite_mask_test][finite_mask_pred]
                        y_pred_finite = y_pred[finite_mask_pred]
                        n_valid = len(y_test_finite)
                        
                        min_samples_for_corr = config.get(
                            "machine_learning.analysis.time_generalization.min_samples_for_corr", 10
                        )
                        if n_valid >= min_samples_for_corr:
                            r_val, _ = safe_pearsonr(y_test_finite, y_pred_finite)
                            r2_val = r2_score(y_test_finite, y_pred_finite) if n_valid > 1 else np.nan
                            if np.isfinite(r_val) and np.isfinite(r2_val):
                                r_mat[i, j] = r_val
                                r2_mat[i, j] = r2_val
                                count_mat[i, j] = n_valid
                    except Exception:
                        continue
            
            if not np.isfinite(r_mat).any() or not np.any(count_mat > 0):
                logger.warning(
                    "Fold %d: no evaluable time-generalization cells after quality filters; skipping fold.",
                    int(fold),
                )
                continue

            fold_mats_r.append(r_mat)
            fold_mats_r2.append(r2_mat)
            fold_counts.append(count_mat)
            n_folds_successful += 1
        
        if not fold_mats_r:
            logger.warning(
                f"No successful folds out of {n_folds_total} total folds. "
                "Cannot compute time-generalization matrices."
            )
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        if n_folds_successful < n_folds_total:
            coverage_pct = 100 * n_folds_successful / n_folds_total
            logger.warning(
                f"Only {n_folds_successful}/{n_folds_total} folds ({coverage_pct:.1f}%) succeeded. "
                f"Time-generalization matrices are based on a subset of folds."
            )
            min_valid_fold_fraction = float(
                get_config_value(
                    config,
                    "machine_learning.analysis.time_generalization.min_valid_fold_fraction",
                    0.8,
                )
            )
            completion_rate = float(n_folds_successful / max(n_folds_total, 1))
            if completion_rate < min_valid_fold_fraction:
                raise RuntimeError(
                    "Insufficient valid time-generalization fold coverage: "
                    f"completed={n_folds_successful}/{n_folds_total} "
                    f"(rate={completion_rate:.3f} < required {min_valid_fold_fraction:.3f})."
                )
            if n_folds_successful < min_subjects_for_loso:
                raise RuntimeError(
                    f"Insufficient fold coverage ({n_folds_successful}/{n_folds_total}) "
                    f"for reliable time-generalization analysis. "
                    f"Minimum {min_subjects_for_loso} successful folds required."
                )
        
        stacked_r = np.stack(fold_mats_r, axis=0)
        stacked_r2 = np.stack(fold_mats_r2, axis=0)
        stacked_counts = np.stack(fold_counts, axis=0)
        tg_r, tg_r2, coverage_map, subject_coverage_map, _tested_mask = _aggregate_time_generalization_matrices(
            stacked_r,
            stacked_r2,
            stacked_counts,
            config,
        )

        return (
            tg_r,
            tg_r2,
            window_centers_out if window_centers_out is not None else np.array([]),
            coverage_map,
            subject_coverage_map,
        )
    
    tg_r, tg_r2, window_centers_out, coverage_map, subject_coverage_map = _run_time_gen(y_all_arr)

    null_r_list: List[np.ndarray] = []
    null_r2_list: List[np.ndarray] = []
    if n_perm > 0 and len(tg_r) > 0:
        rng = np.random.default_rng(seed)
        expected_shape = tg_r.shape
        n_effective = 0
        min_shuffle_fraction = float(
            get_config_value(config, "machine_learning.cv.min_label_shuffle_fraction", 0.01)
        )
        perm_scheme = str(
            get_config_value(config, "machine_learning.cv.permutation_scheme", "within_subject_within_block")
        ).strip().lower()
        if perm_scheme not in {"within_subject", "within_subject_within_block"}:
            perm_scheme = "within_subject_within_block"

        if perm_scheme == "within_subject_within_block" and np.all(pd.isna(trial_blocks_arr)):
            logger.warning(
                "Time-generalization permutation requested subject×block shuffling but no block labels were found; "
                "falling back to within-subject permutation."
            )
            perm_scheme = "within_subject"

        for perm_idx in range(n_perm):
            y_perm = _permute_labels_within_subject_structure(
                y_all_arr,
                groups_arr,
                trial_blocks_arr if perm_scheme == "within_subject_within_block" else None,
                rng=rng,
                scheme=perm_scheme,
            )
            effective, _changed_fraction = is_effective_permutation(
                y_all_arr,
                y_perm,
                min_changed_fraction=min_shuffle_fraction,
            )
            if not effective:
                continue
            n_effective += 1
            tg_r_perm, tg_r2_perm, _, _, _ = _run_time_gen(y_perm)
            if tg_r_perm.size > 0 and tg_r_perm.shape == expected_shape:
                null_r_list.append(tg_r_perm)
                null_r2_list.append(tg_r2_perm)
            else:
                logger.warning(
                    f"Permutation {perm_idx + 1}/{n_perm}: Empty or shape-mismatched output "
                    f"(shape={tg_r_perm.shape}, expected={expected_shape}). Skipping."
                )
        if n_effective == 0 and int(n_perm) > 0:
            raise RuntimeError(
                "No effective time-generalization permutations could be generated. "
                "Try machine_learning.cv.permutation_scheme='within_subject' and/or lower "
                "machine_learning.cv.min_label_shuffle_fraction."
            )
        n_perm_valid = len(null_r_list)
        completion_rate = (n_perm_valid / int(n_perm)) if int(n_perm) > 0 else 0.0
        min_completion = float(
            get_config_value(config, "machine_learning.analysis.time_generalization.min_valid_permutation_fraction", 0.8)
        )
        if int(n_perm) > 0 and completion_rate < min_completion:
            raise RuntimeError(
                f"Insufficient valid time-generalization permutations ({n_perm_valid}/{int(n_perm)}, "
                f"rate={completion_rate:.3f} < required {min_completion:.3f})"
            )
    null_r = np.stack(null_r_list, axis=0) if null_r_list else np.array([])
    null_r2 = np.stack(null_r2_list, axis=0) if null_r2_list else np.array([])

    p_matrix, sig_fdr, sig_maxstat, sig_cluster, tested_mask = _compute_time_generalization_significance(
        tg_r=tg_r,
        null_r=null_r,
        config=config,
    )
    n_tested = int(np.sum(tested_mask))
    if n_tested > 0:
        logger.info("Time-generalization FDR: %d/%d significant cells", int(np.sum(sig_fdr)), n_tested)
        logger.info("Time-generalization max-stat (FWER): %d/%d significant cells", int(np.sum(sig_maxstat)), n_tested)
        logger.info("Time-generalization cluster (FWER): %d/%d significant cells", int(np.sum(sig_cluster)), n_tested)

    if results_dir is not None:
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                results_dir / "time_generalization_regression.npz",
                r_matrix=tg_r,
                r2_matrix=tg_r2,
                window_centers=window_centers_out,
                coverage_map=coverage_map,
                subject_coverage_map=subject_coverage_map,
                tested_mask=tested_mask,
                null_r=null_r,
                null_r2=null_r2,
                p_matrix=p_matrix,
                sig_fdr=sig_fdr,
                sig_maxstat=sig_maxstat,
                sig_cluster=sig_cluster,
            )
        except (OSError, PermissionError, ValueError) as exc:
            logger.warning("Failed to save time-generalization outputs: %s", exc)

    return tg_r, tg_r2, window_centers_out
