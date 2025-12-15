from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from eeg_pipeline.analysis.decoding.cv import (
    create_inner_cv,
    create_block_aware_inner_cv,
    create_scoring_dict,
    safe_pearsonr,
)
from eeg_pipeline.utils.data.loading import (
    load_kept_indices,
)
from eeg_pipeline.io.tsv import read_tsv
from eeg_pipeline.io.logging import get_logger
from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values, load_settings, get_config_value

logger = get_logger(__name__)


###################################################################
# Permutation Importance Analysis
###################################################################


def _prepare_block_metadata_for_permutation(
    deriv_root: Path,
    subjects: List[str],
    task: str,
    trial_records: List[Tuple],
    groups_arr: np.ndarray,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    blocks = np.full(len(trial_records), np.nan)
    bids_root = Path(deriv_root).parent
    
    for subject_id in subjects:
        subject_label = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
        events_path = bids_root / subject_label / "eeg" / f"{subject_label}_task-{task}_events.tsv"
        
        if not events_path.exists():
            logger.warning(f"Events TSV not found for {subject_label}: {events_path}")
            return None
        
        events = read_tsv(events_path)
        subject_indices = np.where(groups_arr == subject_id)[0]
        
        kept_indices = load_kept_indices(subject_label, deriv_root, len(events), logger)
        if kept_indices is None or len(kept_indices) == 0:
            logger.error(
                f"Permutation importance requires kept_indices manifest for {subject_label}. "
                "Cannot align block labels without knowing which epochs were retained. "
                "Aborting permutation importance."
            )
            return None
        
        if len(kept_indices) != len(subject_indices):
            logger.error(
                f"kept_indices length ({len(kept_indices)}) does not match "
                f"subject trial count ({len(subject_indices)}) for {subject_label}. "
                "Cannot align block labels. Aborting permutation importance."
            )
            return None
        
        if "block" not in events.columns:
            logger.warning(f"No 'block' column in events file for {subject_label}")
            return None
        
        kept_indices_arr = np.asarray(kept_indices)
        if np.any(kept_indices_arr < 0) or np.any(kept_indices_arr >= len(events)):
            logger.error(
                f"kept_indices out of bounds for events TSV for {subject_label}. "
                "Aborting permutation importance."
            )
            return None
        
        blocks_subj = events.iloc[kept_indices_arr]["block"].to_numpy() if len(kept_indices_arr) <= len(events) else None
        if blocks_subj is None:
            return None
        
        epoch_to_block = dict(enumerate(blocks_subj))
        for idx in subject_indices:
            epoch_idx = trial_records[idx][1]
            block_val = epoch_to_block.get(epoch_idx, np.nan)
            blocks[idx] = block_val
    
    return blocks


def _is_feature_constant_within_blocks(
    X: np.ndarray,
    blocks: np.ndarray,
    threshold: Optional[float] = None,
    config: Optional[Any] = None,
) -> np.ndarray:
    if threshold is None:
        from eeg_pipeline.utils.config.loader import ensure_config
        config = ensure_config(config)
        threshold = float(get_config_value(config, "decoding.constants.min_variance_threshold", 1e-10))
    
    if blocks is None or len(np.unique(blocks)) < 2:
        return np.zeros(X.shape[1], dtype=bool)
    
    constant_mask = np.zeros(X.shape[1], dtype=bool)
    for feat_idx in range(X.shape[1]):
        feat_values = X[:, feat_idx]
        block_means = []
        for block_id in np.unique(blocks):
            block_mask = blocks == block_id
            if np.sum(block_mask) > 0:
                block_means.append(np.nanmean(feat_values[block_mask]))
        if len(block_means) > 1:
            constant_mask[feat_idx] = np.std(block_means) < threshold
    
    return constant_mask


def _build_per_subject_indices(
    trial_records: List[Tuple],
    groups_arr: np.ndarray,
) -> Dict[str, np.ndarray]:
    per_subject_indices = {}
    for subject_id in np.unique(groups_arr):
        subject_mask = groups_arr == subject_id
        per_subject_indices[subject_id] = np.where(subject_mask)[0]
    return per_subject_indices


def _compute_permutation_importance_for_feature(
    feature_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    blocks: Optional[np.ndarray],
    pipe: Pipeline,
    n_repeats: int,
    seed: int,
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> float:
    # Drop samples with non-finite targets to avoid failures in CV/regression
    valid_y_mask = np.isfinite(y)
    if not np.any(valid_y_mask):
        logger.warning("All targets are non-finite; permutation importance undefined.")
        return np.nan
    if np.sum(valid_y_mask) < 3:
        logger.warning("Fewer than 3 finite targets after filtering; permutation importance undefined.")
        return np.nan

    X = X[valid_y_mask]
    y = y[valid_y_mask]
    groups = groups[valid_y_mask]
    if blocks is not None:
        blocks = blocks[valid_y_mask]

    # Need at least two groups for Leave-One-Group-Out
    if len(np.unique(groups)) < 2:
        logger.warning("Permutation importance requires at least two groups; only one group present.")
        return np.nan

    if config is None:
        config = load_settings()
    
    inner_splits_config = get_config_value(config, "decoding.cv.inner_splits", 5)
    
    rng = np.random.default_rng(seed)
    # Skip features with no variation to avoid undefined correlations
    feat_values_all = X[:, feature_idx]
    if not np.any(np.isfinite(feat_values_all)) or np.nanstd(feat_values_all) < 1e-12:
        logger.warning("Feature %d has near-zero variance across all samples; skipping permutation importance.", feature_idx)
        return np.nan
    if blocks is not None:
        const_mask = _is_feature_constant_within_blocks(X[:, [feature_idx]], blocks, threshold=None, config=config)
        if const_mask[0]:
            logger.warning(
                "Feature %d is effectively constant within blocks; permutation importance is undefined. Returning NaN.",
                feature_idx,
            )
            return np.nan

    orig_z_vals = []
    orig_weights = []
    perm_z_vals = []
    perm_weights = []
    
    logo = LeaveOneGroupOut()
    folds = list(logo.split(np.arange(len(y)), groups=groups))
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        if len(y_test) < 3:
            continue
        
        pipe_fitted = clone(pipe)
        regressor_step = pipe_fitted.named_steps.get("regressor")
        if regressor_step is not None and hasattr(regressor_step, "regressor") and hasattr(regressor_step.regressor, "random_state"):
            regressor_step.regressor.random_state = seed + fold_idx
        
        if blocks is not None:
            blocks_train = blocks[train_idx]
            n_unique_blocks = len(np.unique(blocks_train))
            if n_unique_blocks >= 2:
                n_splits_inner = min(inner_splits_config, n_unique_blocks)
                inner_cv_splits = create_block_aware_inner_cv(blocks_train, n_splits_inner, seed, fold_idx, "permutation")
                if inner_cv_splits is not None and len(inner_cv_splits) >= 2:
                    param_grid = getattr(pipe.named_steps.get("regressor"), "param_grid", {})
                    if param_grid:
                        scoring = create_scoring_dict()
                        gs = GridSearchCV(
                            estimator=pipe_fitted,
                            param_grid=param_grid,
                            scoring=scoring,
                            cv=inner_cv_splits,
                            n_jobs=1,
                            refit='r',
                        )
                        gs.fit(X_train, y_train, groups=blocks_train)
                        pipe_fitted = gs.best_estimator_
                    else:
                        pipe_fitted.fit(X_train, y_train)
                else:
                    pipe_fitted.fit(X_train, y_train)
            else:
                pipe_fitted.fit(X_train, y_train)
        else:
            n_unique_groups = len(np.unique(groups_train))
            if n_unique_groups >= 2:
                n_splits_inner = min(inner_splits_config, n_unique_groups)
                inner_cv = create_inner_cv(groups_train, n_splits_inner)
                param_grid = getattr(pipe.named_steps.get("regressor"), "param_grid", {})
                if param_grid:
                    scoring = create_scoring_dict()
                    gs = GridSearchCV(
                        estimator=pipe_fitted,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=inner_cv,
                        n_jobs=1,
                        refit='r',
                    )
                    gs.fit(X_train, y_train, groups=groups_train)
                    pipe_fitted = gs.best_estimator_
                else:
                    pipe_fitted.fit(X_train, y_train)
            else:
                pipe_fitted.fit(X_train, y_train)
        
        y_pred_orig = pipe_fitted.predict(X_test)
        valid_mask_orig = np.isfinite(y_test) & np.isfinite(y_pred_orig)
        n_valid_orig = int(valid_mask_orig.sum())
        if n_valid_orig >= 3:
            r_orig, _ = safe_pearsonr(y_test[valid_mask_orig], y_pred_orig[valid_mask_orig])
            if np.isfinite(r_orig):
                weight_orig = max(n_valid_orig - 3.0, 1.0)
                clip_min, clip_max = get_fisher_z_clip_values(config)
                orig_z_vals.append(np.arctanh(np.clip(r_orig, clip_min, clip_max)))
                orig_weights.append(weight_orig)
    
        groups_test = groups[test_idx] if groups is not None else None
        blocks_test = blocks[test_idx] if blocks is not None else None
        
        for _ in range(n_repeats):
            X_test_perm = X_test.copy()
            feature_values = X_test_perm[:, feature_idx].copy()
            
            if blocks_test is not None:
                valid_mask = np.isfinite(blocks_test)
                if np.any(valid_mask):
                    for block_id in np.unique(blocks_test[valid_mask]):
                        block_mask = blocks_test == block_id
                        block_indices = np.where(block_mask)[0]
                        if len(block_indices) > 0:
                            feature_values[block_indices] = rng.permutation(feature_values[block_indices])
                if np.any(~valid_mask):
                    nan_indices = np.where(~valid_mask)[0]
                    if len(nan_indices) > 0:
                        feature_values[nan_indices] = rng.permutation(feature_values[nan_indices])
            elif groups_test is not None:
                for group_id in np.unique(groups_test):
                    group_mask = groups_test == group_id
                    group_indices = np.where(group_mask)[0]
                    feature_values[group_indices] = rng.permutation(feature_values[group_indices])
            else:
                feature_values = rng.permutation(feature_values)
            
            X_test_perm[:, feature_idx] = feature_values
            y_pred_perm = pipe_fitted.predict(X_test_perm)
            valid_mask_perm = np.isfinite(y_test) & np.isfinite(y_pred_perm)
            n_valid_perm = int(valid_mask_perm.sum())
            if n_valid_perm >= 3:
                r_perm, _ = safe_pearsonr(y_test[valid_mask_perm], y_pred_perm[valid_mask_perm])
                if np.isfinite(r_perm):
                    weight_perm = max(n_valid_perm - 3.0, 1.0)
                    clip_min, clip_max = get_fisher_z_clip_values(config)
                    perm_z_vals.append(np.arctanh(np.clip(r_perm, clip_min, clip_max)))
                    perm_weights.append(weight_perm)
    
    if not orig_z_vals or not perm_z_vals:
        return np.nan
    
    def _weighted_mean_z(z_list, w_list):
        w = np.asarray(w_list, dtype=float)
        z = np.asarray(z_list, dtype=float)
        w = np.where(np.isfinite(w), w, 0.0)
        z = np.where(np.isfinite(z), z, 0.0)
        if np.sum(w) <= 0:
            return np.nan
        return float(np.sum(z * w) / np.sum(w))
    
    z_orig_mean = _weighted_mean_z(orig_z_vals, orig_weights)
    z_perm_mean = _weighted_mean_z(perm_z_vals, perm_weights)
    
    if not np.isfinite(z_orig_mean) or not np.isfinite(z_perm_mean):
        return np.nan
    
    delta = np.tanh(z_orig_mean) - np.tanh(z_perm_mean)
    return float(delta)
