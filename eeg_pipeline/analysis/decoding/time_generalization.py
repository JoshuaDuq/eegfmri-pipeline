from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from eeg_pipeline.utils.analysis.tfr import (
    find_common_channels_train_test,
)
from eeg_pipeline.utils.data.loading import (
    filter_finite_targets,
    extract_epoch_data_block,
    prepare_trial_records_from_epochs,
    load_epochs_with_targets,
)
from eeg_pipeline.analysis.decoding.cv import (
    get_min_channels_required,
    safe_pearsonr,
)
from eeg_pipeline.utils.config.loader import load_settings, get_fisher_z_clip_values
from eeg_pipeline.plotting.decoding import (
    plot_time_generalization_with_null,
)
from eeg_pipeline.utils.io.logging import get_logger

logger = get_logger(__name__)


###################################################################
# Time Generalization Decoding
###################################################################


def _build_time_windows(
    window_len: float,
    step: float,
    tmin: float,
    tmax: float,
) -> List[Tuple[float, float]]:
    windows = []
    t = tmin
    while t + window_len <= tmax + 1e-6:
        windows.append((t, t + window_len))
        t += step
    return windows


def _extract_window_features(
    data: np.ndarray,
    trial_records: List[Tuple[str, int]],
    aligned_epochs: Dict[str, Any],
    windows: List[Tuple[float, float]],
) -> np.ndarray:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)
    trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y = prepare_trial_records_from_epochs(tuples)

    config_local = config_dict or load_settings()
    min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
    if len(np.unique(groups_arr)) < min_subjects_for_loso:
        raise RuntimeError(f"Need at least {min_subjects_for_loso} subjects for LOSO.")

    plateau_window = tuple(config_local.get("decoding.analysis.time_generalization.plateau_window", [3.0, 10.5]))
    window_len = config_local.get("decoding.analysis.time_generalization.window_len", 0.75)
    step = config_local.get("decoding.analysis.time_generalization.step", 0.25)

    tmin_pl, tmax_pl = plateau_window
    windows = _build_time_windows(window_len, step, tmin_pl, tmax_pl)

    if not windows:
        logger.warning("No valid time windows found.")
        return np.array([]), np.array([]), np.array([])

    logo = LeaveOneGroupOut()
    n_folds_total = len(list(logo.split(np.arange(len(trial_records)), groups=groups_arr)))
    
    def _run_time_gen(y_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = config_local
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
            
            min_channels_required = get_min_channels_required(cfg)
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
                first_epochs = aligned_epochs[subjects_in_fold[0]]
                times = first_epochs.times
                window_centers_out = np.array([(w_start + w_end) / 2 for w_start, w_end in windows])
            
            # Minimum samples per window for reliable regression and correlation
            min_samples_per_window = cfg.get(
                "decoding.analysis.time_generalization.min_samples_per_window", 15
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
                imputer = SimpleImputer(strategy='mean')
                train_feat_i_clean = imputer.fit_transform(train_feat_i_clean)
                
                model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
                try:
                    model.fit(train_feat_i_clean, y_train[finite_mask_train])
                except Exception:
                    continue
                
                for j in range(n_windows):
                    test_feat_j = test_feats[:, j, :]
                    finite_mask_test = np.isfinite(test_feat_j).any(axis=1)
                    if np.sum(finite_mask_test) < min_samples_per_window:
                        continue
                    
                    test_feat_j_clean = test_feat_j[finite_mask_test].copy()
                    test_feat_j_clean = test_feat_j_clean[:, col_mask]
                    test_feat_j_clean = imputer.transform(test_feat_j_clean)
                    
                    try:
                        y_pred = model.predict(test_feat_j_clean)
                        finite_mask_pred = np.isfinite(y_pred) & np.isfinite(y_test[finite_mask_test])
                        if np.sum(finite_mask_pred) < min_samples_per_window:
                            continue
                        y_test_finite = y_test[finite_mask_test][finite_mask_pred]
                        y_pred_finite = y_pred[finite_mask_pred]
                        n_valid = len(y_test_finite)
                        
                        # Configurable minimum samples for reliable correlation estimation
                        min_samples_for_corr = cfg.get(
                            "decoding.analysis.time_generalization.min_samples_for_corr", 10
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
            
            fold_mats_r.append(r_mat)
            fold_mats_r2.append(r2_mat)
            fold_counts.append(count_mat)
            n_folds_successful += 1
        
        if not fold_mats_r:
            logger.warning(
                f"No successful folds out of {n_folds_total} total folds. "
                "Cannot compute time-generalization matrices."
            )
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        if n_folds_successful < n_folds_total:
            coverage_pct = 100 * n_folds_successful / n_folds_total
            logger.warning(
                f"Only {n_folds_successful}/{n_folds_total} folds ({coverage_pct:.1f}%) succeeded. "
                f"Time-generalization matrices are based on a subset of folds."
            )
            min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
            if n_folds_successful < min_subjects_for_loso:
                raise RuntimeError(
                    f"Insufficient fold coverage ({n_folds_successful}/{n_folds_total}) "
                    f"for reliable time-generalization analysis. "
                    f"Minimum {min_subjects_for_loso} successful folds required."
                )
        
        stacked_r = np.stack(fold_mats_r, axis=0)
        stacked_r2 = np.stack(fold_mats_r2, axis=0)
        stacked_counts = np.stack(fold_counts, axis=0)
        
        n_windows = stacked_r.shape[1]
        tg_r = np.full((n_windows, n_windows), np.nan, dtype=float)
        tg_r2 = np.full_like(tg_r, np.nan)
        coverage_map = np.zeros_like(tg_r, dtype=int)
        
        # Minimum total samples across folds for reliable Fisher z-averaging
        min_count_per_cell = cfg.get(
            "decoding.analysis.time_generalization.min_count_per_cell", 15
        )
        
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
                total_count = valid_counts.sum()
                
                coverage_map[i, j] = total_count
                
                if total_count < min_count_per_cell:
                    continue
                
                clip_min, clip_max = get_fisher_z_clip_values(config_local)
                r_clipped = np.clip(valid_r, clip_min, clip_max)
                r_z = np.arctanh(r_clipped)
                
                # Use n-3 weighting for proper Fisher z variance (variance of z ~ 1/(n-3))
                fisher_weights = np.maximum(valid_counts - 3.0, 1.0)
                fisher_weights = fisher_weights / fisher_weights.sum()
                weighted_z_mean = np.average(r_z, weights=fisher_weights)
                tg_r[i, j] = np.tanh(weighted_z_mean)
                
                tg_r2[i, j] = np.average(valid_r2, weights=fisher_weights)
        
        return tg_r, tg_r2, window_centers_out if window_centers_out is not None else np.array([]), coverage_map
    
    tg_r, tg_r2, window_centers_out, coverage_map = _run_time_gen(y_all_arr)

    null_r = []
    null_r2 = []
    if n_perm > 0 and len(tg_r) > 0:
        rng = np.random.default_rng(seed)
        expected_shape = tg_r.shape
        for perm_idx in range(n_perm):
            y_perm = y_all_arr.copy()
            for subject_id in np.unique(groups_arr):
                subject_mask = groups_arr == subject_id
                subject_indices = np.where(subject_mask)[0]
                y_perm[subject_indices] = rng.permutation(y_perm[subject_indices])
            tg_r_perm, tg_r2_perm, _, _ = _run_time_gen(y_perm)
            if tg_r_perm.size > 0 and tg_r_perm.shape == expected_shape:
                null_r.append(tg_r_perm)
                null_r2.append(tg_r2_perm)
            else:
                logger.warning(
                    f"Permutation {perm_idx + 1}/{n_perm}: Empty or shape-mismatched output "
                    f"(shape={tg_r_perm.shape}, expected={expected_shape}). Skipping."
                )
        
        if len(null_r) == 0:
            logger.error(
                f"All {n_perm} permutations produced empty or invalid outputs. "
                "Cannot compute null distribution for time-generalization."
            )
            null_r = np.array([])
            null_r2 = np.array([])
        elif len(null_r) < n_perm:
            logger.warning(
                f"Only {len(null_r)}/{n_perm} permutations produced valid outputs. "
                "Null distribution may be unreliable."
            )
            null_r = np.stack(null_r, axis=0)
            null_r2 = np.stack(null_r2, axis=0)
        else:
            null_r = np.stack(null_r, axis=0)
            null_r2 = np.stack(null_r2, axis=0)

    if results_dir is not None:
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                results_dir / "time_generalization_regression.npz",
                r_matrix=tg_r,
                r2_matrix=tg_r2,
                window_centers=window_centers_out,
                coverage_map=coverage_map,
                null_r=null_r,
                null_r2=null_r2,
            )
            plot_time_generalization_with_null(
                tg_matrix=tg_r,
                null_matrix=null_r,
                window_centers=window_centers_out,
                save_path=results_dir / "time_generalization_r.png",
                metric="r",
                config=config_local,
            )
            plot_time_generalization_with_null(
                tg_matrix=tg_r2,
                null_matrix=null_r2,
                window_centers=window_centers_out,
                save_path=results_dir / "time_generalization_r2.png",
                metric="r2",
                config=config_local,
            )
        except (OSError, PermissionError, ValueError) as exc:
            logger.warning("Failed to save time-generalization outputs: %s", exc)

    return tg_r, tg_r2, window_centers_out
