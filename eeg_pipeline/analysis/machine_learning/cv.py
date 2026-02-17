"""
Cross-validation utilities for machine learning.

This module provides CV helpers, fold creation, and result aggregation.
"""

from __future__ import annotations

import json
import logging
import random as pyrandom
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values, get_config_value

if TYPE_CHECKING:
    from eeg_pipeline.analysis.features.cv_hygiene import FoldSpecificParams

logger = get_logger(__name__)


###################################################################
# CV Hygiene Integration
###################################################################


def apply_fold_specific_hygiene(
    fold_idx: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: Optional[Any] = None,
    config: Optional[Any] = None,
    log: Optional[logging.Logger] = None,
) -> Optional[FoldSpecificParams]:
    """
    Apply CV hygiene for a fold: compute fold-specific parameters on training data only.
    
    This prevents leakage from test trials into:
    - IAF (Individual Alpha Frequency) band definitions
    - Global/broadcast features (e.g., ITPC)
    - Feature scaling parameters
    
    Parameters
    ----------
    fold_idx : int
        Fold index
    train_indices : np.ndarray
        Indices of training trials
    test_indices : np.ndarray
        Indices of test trials
    epochs : mne.Epochs, optional
        Epochs object (required for IAF computation)
    config : Any, optional
        Configuration object
    log : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    FoldSpecificParams or None
        Fold-specific parameters, or None if hygiene is disabled
    """
    try:
        from eeg_pipeline.analysis.features.cv_hygiene import (
            FoldSpecificParams,
            create_fold_specific_context,
        )
    except ImportError:
        if log:
            log.debug("CV hygiene module not available")
        return None
    
    if config is None:
        return None
    
    cv_hygiene_enabled = bool(get_config_value(config, "machine_learning.cv.hygiene_enabled", True))
    if not cv_hygiene_enabled:
        return None
    
    if epochs is None:
        if log:
            log.debug("CV hygiene: epochs not provided, skipping fold-specific IAF")
        return FoldSpecificParams(
            fold_idx=fold_idx,
            train_indices=train_indices,
            test_indices=test_indices,
        )
    
    try:
        params = create_fold_specific_context(
            epochs=epochs,
            train_indices=train_indices,
            test_indices=test_indices,
            fold_idx=fold_idx,
            config=config,
            logger=log,
        )
        return params
    except Exception as exc:
        if log:
            log.warning("CV hygiene: failed to create fold context (%s)", exc)
        return None


###################################################################
# Seed Management
###################################################################


def set_random_seeds(seed: int, fold: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed + fold)
    pyrandom.seed(seed + fold)


###################################################################
# Fold Creation
###################################################################


def create_loso_folds(X: np.ndarray, groups: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Create leave-one-subject-out folds."""
    logo = LeaveOneGroupOut()
    return [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, groups=groups), start=1)
    ]


def compute_train_group_intersection_mask(
    X_train: np.ndarray,
    groups_train: np.ndarray,
) -> np.ndarray:
    """Return per-column mask where each train subject has at least one finite value."""
    X_arr = np.asarray(X_train, dtype=float)
    groups_arr = np.asarray(groups_train)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X_train, got shape={X_arr.shape}")
    if X_arr.shape[0] != len(groups_arr):
        raise ValueError(
            f"Length mismatch for X_train/groups_train: {X_arr.shape[0]} vs {len(groups_arr)}"
        )
    n_features = X_arr.shape[1]
    if n_features == 0:
        return np.zeros(0, dtype=bool)

    keep_mask = np.ones(n_features, dtype=bool)
    unique_groups = np.unique(groups_arr)
    for grp in unique_groups:
        grp_mask = groups_arr == grp
        if not np.any(grp_mask):
            continue
        # Feature must be present for this group (at least one finite row).
        grp_has = np.any(np.isfinite(X_arr[grp_mask]), axis=0)
        keep_mask &= grp_has

    # If strict per-group intersection is empty, fall back to any-finite in train.
    if not np.any(keep_mask):
        keep_mask = np.any(np.isfinite(X_arr), axis=0)
    return keep_mask


def apply_fold_feature_harmonization(
    X_train: np.ndarray,
    X_test: np.ndarray,
    groups_train: np.ndarray,
    harmonization_mode: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply fold-specific feature harmonization (intersection/union) safely."""
    mode = (harmonization_mode or "union_impute").strip().lower()
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    if Xtr.shape[1] != Xte.shape[1]:
        raise ValueError(f"X_train/X_test feature mismatch: {Xtr.shape[1]} vs {Xte.shape[1]}")

    if mode != "intersection":
        keep = np.ones(Xtr.shape[1], dtype=bool)
        return Xtr, Xte, keep

    keep = compute_train_group_intersection_mask(Xtr, np.asarray(groups_train))
    if keep.size == 0 or not np.any(keep):
        raise ValueError("Fold-specific intersection harmonization removed all features.")
    return Xtr[:, keep], Xte[:, keep], keep


def create_inner_cv(train_groups: np.ndarray, inner_cv_splits: int) -> GroupKFold:
    """Create inner CV splitter for hyperparameter tuning."""
    n_unique = len(np.unique(train_groups))
    if n_unique < 2:
        raise ValueError(
            "Inner CV requires at least 2 unique groups in the training split."
        )
    n_splits = max(2, min(inner_cv_splits, n_unique))
    return GroupKFold(n_splits=n_splits)


def create_block_aware_cv(
    blocks: np.ndarray,
    n_splits: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[Optional[GroupKFold], int]:
    """Create block-aware GroupKFold CV."""
    if n_splits is None:
        n_splits = int(get_config_value(config, "machine_learning.cv.default_n_splits", 5))
    unique_blocks = np.unique(blocks[~pd.isna(blocks)])
    n_unique = len(unique_blocks)

    if n_unique < 2:
        logger.warning(f"Insufficient blocks ({n_unique}) for GroupKFold")
        return None, 0

    effective_splits = min(n_splits, n_unique)
    if effective_splits < n_splits:
        logger.info(f"Reduced splits from {n_splits} to {effective_splits}")

    return GroupKFold(n_splits=effective_splits), effective_splits


def permutation_changed_fraction(
    y_original: np.ndarray,
    y_permuted: np.ndarray,
) -> float:
    """Return the fraction of labels that changed under permutation."""
    original = np.asarray(y_original)
    permuted = np.asarray(y_permuted)
    if original.shape != permuted.shape:
        raise ValueError(f"Permutation shape mismatch: {original.shape} vs {permuted.shape}")
    if original.size == 0:
        return 0.0

    if np.issubdtype(original.dtype, np.number) and np.issubdtype(permuted.dtype, np.number):
        orig = np.asarray(original, dtype=float)
        perm = np.asarray(permuted, dtype=float)
        finite = np.isfinite(orig) & np.isfinite(perm)
        if not np.any(finite):
            return 0.0
        changed = int(np.sum(orig[finite] != perm[finite]))
        return float(changed / int(np.sum(finite)))

    changed = int(np.sum(original != permuted))
    return float(changed / int(original.size))


def is_effective_permutation(
    y_original: np.ndarray,
    y_permuted: np.ndarray,
    *,
    min_changed_fraction: float,
) -> Tuple[bool, float]:
    """Check whether a permutation changed enough labels to be inferentially useful."""
    changed_fraction = permutation_changed_fraction(y_original, y_permuted)
    return bool(changed_fraction >= float(max(0.0, min_changed_fraction))), float(changed_fraction)


###################################################################
# Fold Execution
###################################################################


def determine_inner_n_jobs(outer_n_jobs: int, n_jobs: int) -> int:
    """Determine inner CV parallelism based on outer parallelism."""
    return 1 if (outer_n_jobs and outer_n_jobs != 1) else n_jobs


def should_parallelize_folds(outer_n_jobs: int, n_folds: int) -> bool:
    """Check if folds should be parallelized."""
    return outer_n_jobs and outer_n_jobs != 1 and n_folds > 1


def execute_folds_parallel(
    folds: List[Tuple],
    fold_func: Callable,
    outer_n_jobs: int,
) -> List[dict]:
    """Execute folds with optional parallelization."""
    from joblib import Parallel, delayed

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"sklearn\.utils\.parallel",
        )
        if should_parallelize_folds(outer_n_jobs, len(folds)):
            return Parallel(n_jobs=outer_n_jobs, prefer="threads")(
                delayed(fold_func)(fold, train_idx, test_idx)
                for (fold, train_idx, test_idx) in folds
            )
        return [fold_func(fold, train_idx, test_idx) for (fold, train_idx, test_idx) in folds]


###################################################################
# Warning Handling
###################################################################


def fit_with_warning_logging(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    fold_info: str = "",
    log: Optional[logging.Logger] = None,
):
    """Fit estimator with warning logging."""
    log = log or logger
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimator.fit(X, y)
        for warning in w:
            if isinstance(warning.message, (ConvergenceWarning, ConstantInputWarning)):
                warning_type = type(warning.message).__name__
                log.warning(f"Fold {fold_info}: {warning_type} - {warning.message}")
    return estimator


def grid_search_with_warning_logging(
    grid: GridSearchCV,
    X: np.ndarray,
    y: np.ndarray,
    fold_info: str = "",
    log: Optional[logging.Logger] = None,
    **fit_params,
) -> GridSearchCV:
    """Fit GridSearchCV with warning logging."""
    log = log or logger
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid.fit(X, y, **fit_params)
        for warning in w:
            if isinstance(warning.message, (ConvergenceWarning, ConstantInputWarning)):
                warning_type = type(warning.message).__name__
                log.warning(f"Fold {fold_info}: {warning_type} during GridSearchCV - {warning.message}")
    return grid


###################################################################
# Scoring
###################################################################


def safe_pearsonr(
    x: np.ndarray,
    y: np.ndarray,
    min_variance: float = 1e-10,
) -> Tuple[float, float]:
    """Compute Pearson correlation with safety checks."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        return np.nan, np.nan

    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return np.nan, np.nan

    x_valid, y_valid = x_arr[valid], y_arr[valid]

    if np.var(x_valid, ddof=1) < min_variance or np.var(y_valid, ddof=1) < min_variance:
        return np.nan, np.nan

    r, p = pearsonr(x_valid, y_valid)
    if not (np.isfinite(r) and np.isfinite(p)):
        return np.nan, np.nan

    return float(np.clip(r, -1.0, 1.0)), float(p)


def make_pearsonr_scorer():
    """Create sklearn scorer for Pearson r."""
    from sklearn.metrics import make_scorer

    return make_scorer(
        lambda yt, yp: safe_pearsonr(np.asarray(yt), np.asarray(yp))[0],
        greater_is_better=True,
    )


def create_scoring_dict() -> dict:
    """Create scoring dictionary for GridSearchCV."""
    return {"r": make_pearsonr_scorer(), "neg_mse": "neg_mean_squared_error"}


###################################################################
# Result Aggregation
###################################################################


def aggregate_fold_results(
    results: List[dict],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """Aggregate results from all folds."""
    sorted_results = sorted(results, key=lambda r: r["fold"])

    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    groups_ordered: List[str] = []
    test_indices: List[int] = []
    fold_ids: List[int] = []

    for record in sorted_results:
        n_test = len(record["test_idx"])
        y_true_all.extend(record["y_true"])
        y_pred_all.extend(record["y_pred"])
        groups_ordered.extend(record["groups"])
        test_indices.extend(record["test_idx"])
        fold_ids.extend([record["fold"]] * n_test)

    return (
        np.asarray(y_true_all),
        np.asarray(y_pred_all),
        groups_ordered,
        test_indices,
        fold_ids,
    )


def compute_subject_level_r(
    pred_df: pd.DataFrame,
    config: Optional[Any] = None,
    ci_method: str = "fixed_effects",
) -> Tuple[float, List[Tuple[str, float]], float, float]:
    """Compute subject-level Pearson r with Fisher z aggregation.
    
    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions with columns: subject_id, y_true, y_pred
    config : Any, optional
        Configuration object
    ci_method : str
        CI method: 'fixed_effects' (variance = 1/sum(n_i-3)) or 
        'bootstrap' (subject-level bootstrap, often preferable for EEG)
    
    Returns
    -------
    agg_r : float
        Aggregated correlation (Fisher z weighted average back-transformed)
    per_subject : list
        List of (subject_id, r) tuples
    ci_low, ci_high : float
        95% confidence interval bounds
    """
    per_subject: List[Tuple[str, float, int]] = []

    for subj, df_sub in pred_df.groupby("subject_id"):
        yt = pd.to_numeric(df_sub["y_true"], errors="coerce").to_numpy()
        yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").to_numpy()
        finite = np.isfinite(yt) & np.isfinite(yp)
        n_trials = finite.sum()

        if n_trials < 2:
            continue
        if np.std(yt[finite]) <= 0 or np.std(yp[finite]) <= 0:
            continue

        try:
            r_subj, _ = pearsonr(yt[finite], yp[finite])
        except Exception:
            continue

        if np.isfinite(r_subj):
            per_subject.append((str(subj), float(r_subj), int(n_trials)))

    if not per_subject:
        return np.nan, [], np.nan, np.nan

    clip_min, clip_max = get_fisher_z_clip_values(config)
    r_vals = np.clip([r for _, r, _ in per_subject], clip_min, clip_max)
    n_trials_vals = np.asarray([n for _, _, n in per_subject], dtype=int)

    weighting_mode = str(
        get_config_value(config, "machine_learning.evaluation.subject_weighting", "equal")
    ).strip().lower()
    if weighting_mode not in {"equal", "trial_count"}:
        weighting_mode = "equal"

    # Fisher z transform and weighted average
    z_vals = np.arctanh(r_vals)
    if weighting_mode == "trial_count":
        weights = np.maximum(n_trials_vals - 3, 1.0)
    else:
        weights = np.ones_like(n_trials_vals, dtype=float)
    sum_weights = weights.sum()
    norm_weights = weights / sum_weights
    mean_z = float(np.average(z_vals, weights=norm_weights))
    agg_r = float(np.tanh(mean_z))

    # Confidence interval
    ci_low, ci_high = np.nan, np.nan
    
    if ci_method == "bootstrap" and len(z_vals) >= 3:
        # Subject-level bootstrap CI (often preferable for EEG)
        boot_seed = int(get_config_value(config, "project.random_state", 42))
        n_boot = int(get_config_value(config, "machine_learning.evaluation.bootstrap_iterations", 1000))
        rng = np.random.default_rng(boot_seed)
        boot_means = []
        n_subjects = len(z_vals)
        for _ in range(n_boot):
            boot_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
            boot_z = z_vals[boot_idx]
            if weighting_mode == "trial_count":
                boot_w = weights[boot_idx]
                boot_w = boot_w / boot_w.sum()
                boot_means.append(np.average(boot_z, weights=boot_w))
            else:
                boot_means.append(float(np.mean(boot_z)))
        boot_means = np.array(boot_means)
        ci_low = float(np.tanh(np.percentile(boot_means, 2.5)))
        ci_high = float(np.tanh(np.percentile(boot_means, 97.5)))
    elif len(z_vals) > 1:
        if weighting_mode == "trial_count":
            # Fixed-effects CI: variance = 1 / sum(n_i - 3)
            # This is the standard meta-analytic fixed-effects variance.
            se = float(np.sqrt(1.0 / sum_weights))
        else:
            # Equal-subject CI from between-subject Fisher-z variability.
            se = float(np.std(z_vals, ddof=1) / np.sqrt(len(z_vals)))
        if np.isfinite(se) and se > 0:
            delta = 1.96 * se
            ci_low = float(np.tanh(mean_z - delta))
            ci_high = float(np.tanh(mean_z + delta))

    return agg_r, [(s, r) for s, r, _ in per_subject], ci_low, ci_high


def compute_subject_level_errors(
    pred_df: pd.DataFrame,
    config: Optional[Any] = None,
    ci_method: str = "bootstrap",
) -> Dict[str, Any]:
    """Compute subject-level MAE/RMSE (statistical unit = subject).

    Returns unweighted mean across subjects plus optional bootstrap CI.
    """
    per_subject: List[Dict[str, Any]] = []

    for subj, df_sub in pred_df.groupby("subject_id"):
        yt = pd.to_numeric(df_sub["y_true"], errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(df_sub["y_pred"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(yt) & np.isfinite(yp)
        n_trials = int(finite.sum())
        if n_trials < 1:
            continue
        err = yp[finite] - yt[finite]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        per_subject.append(
            {
                "subject_id": str(subj),
                "n_trials": n_trials,
                "mae": mae,
                "rmse": rmse,
            }
        )

    if not per_subject:
        return {
            "mean_mae": np.nan,
            "mean_rmse": np.nan,
            "ci_low_mae": np.nan,
            "ci_high_mae": np.nan,
            "ci_low_rmse": np.nan,
            "ci_high_rmse": np.nan,
            "per_subject": [],
        }

    maes = np.asarray([r["mae"] for r in per_subject], dtype=float)
    rmses = np.asarray([r["rmse"] for r in per_subject], dtype=float)
    mean_mae = float(np.mean(maes))
    mean_rmse = float(np.mean(rmses))

    ci_low_mae = ci_high_mae = np.nan
    ci_low_rmse = ci_high_rmse = np.nan

    if ci_method == "bootstrap" and len(per_subject) >= 3:
        boot_seed = int(get_config_value(config, "project.random_state", 42))
        n_boot = int(get_config_value(config, "machine_learning.evaluation.bootstrap_iterations", 1000))
        rng = np.random.default_rng(boot_seed + 11)
        n_sub = len(per_subject)

        boot_mae = np.empty(n_boot, dtype=float)
        boot_rmse = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.choice(n_sub, size=n_sub, replace=True)
            boot_mae[i] = float(np.mean(maes[idx]))
            boot_rmse[i] = float(np.mean(rmses[idx]))

        ci_low_mae = float(np.percentile(boot_mae, 2.5))
        ci_high_mae = float(np.percentile(boot_mae, 97.5))
        ci_low_rmse = float(np.percentile(boot_rmse, 2.5))
        ci_high_rmse = float(np.percentile(boot_rmse, 97.5))

    return {
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "ci_low_mae": ci_low_mae,
        "ci_high_mae": ci_high_mae,
        "ci_low_rmse": ci_low_rmse,
        "ci_high_rmse": ci_high_rmse,
        "per_subject": per_subject,
    }


###################################################################
# Best Params Handling
###################################################################


def create_best_params_record(
    model_name: str,
    fold: int,
    cv_results: pd.DataFrame,
    grid_search: GridSearchCV,
    subject: Optional[str] = None,
    heldout_subjects: Optional[List[str]] = None,
) -> dict:
    """Create record of best hyperparameters from CV."""
    # Get best by each metric
    idx_r = cv_results["rank_test_r"].idxmin()
    idx_mse = cv_results["rank_test_neg_mse"].idxmin()

    record = {
        "model": model_name,
        "fold": int(fold),
        "best_params_by_r": cv_results.loc[idx_r, "params"],
        "best_params_by_neg_mse": cv_results.loc[idx_mse, "params"],
        "best_score_r": float(cv_results.loc[idx_r, "mean_test_r"]),
        "best_score_neg_mse": float(cv_results.loc[idx_mse, "mean_test_neg_mse"]),
        "best_params": grid_search.best_params_,
    }

    if subject:
        record["subject"] = subject
    if heldout_subjects:
        record["heldout_subjects"] = heldout_subjects

    return record


def _save_best_params(records: List[dict], path: Path) -> None:
    """Save best params records to JSONL file."""
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


###################################################################
# Within-Subject CV
###################################################################


def create_within_subject_folds(
    groups: np.ndarray,
    blocks_all: Optional[np.ndarray],
    inner_cv_splits: int,
    seed: int,
    outer_cv_splits: Optional[int] = None,
    config: Optional[Any] = None,
    epochs: Optional[Any] = None,
    apply_hygiene: bool = True,
) -> List[Tuple[int, np.ndarray, np.ndarray, str, Optional[Any]]]:
    """
    Create within-subject CV folds with optional CV hygiene.
    
    Returns list of (fold_idx, train_idx, test_idx, subject, fold_params)
    where fold_params contains fold-specific parameters computed on training data only.
    """
    folds: List[Tuple[int, np.ndarray, np.ndarray, str, Optional[Any]]] = []
    fold_counter = 0
    unique_subs = [str(s) for s in np.unique(groups)]

    for subject in unique_subs:
        subject_indices = np.where(groups == subject)[0]
        n_samples = len(subject_indices)

        requested_outer_splits = int(outer_cv_splits) if outer_cv_splits is not None else int(inner_cv_splits)
        n_splits = min(max(2, requested_outer_splits), n_samples)

        if blocks_all is None:
            logger.warning(f"Subject {subject}: missing run_id, skipping")
            continue

        subject_blocks = blocks_all[subject_indices]
        ordered_blocks = bool(
            get_config_value(config, "machine_learning.cv.within_subject_ordered_blocks", False)
        )
        if ordered_blocks:
            subject_blocks_num = pd.to_numeric(subject_blocks, errors="coerce").to_numpy(dtype=float)
            ordered_unique = sorted(np.unique(subject_blocks_num[np.isfinite(subject_blocks_num)]))
            ordered_splits: List[Tuple[np.ndarray, np.ndarray]] = []
            for idx_block in range(1, len(ordered_unique)):
                train_block_vals = ordered_unique[:idx_block]
                test_block_val = ordered_unique[idx_block]
                train_local_mask = np.isin(subject_blocks_num, train_block_vals)
                test_local_mask = subject_blocks_num == test_block_val
                if np.any(train_local_mask) and np.any(test_local_mask):
                    ordered_splits.append(
                        (
                            np.flatnonzero(train_local_mask).astype(int),
                            np.flatnonzero(test_local_mask).astype(int),
                        )
                    )

            if ordered_splits:
                if len(ordered_splits) > n_splits:
                    ordered_splits = ordered_splits[-n_splits:]
                for train_local, test_local in ordered_splits:
                    fold_counter += 1
                    train_idx = subject_indices[train_local]
                    test_idx = subject_indices[test_local]

                    fold_params = None
                    if apply_hygiene:
                        fold_params = apply_fold_specific_hygiene(
                            fold_idx=fold_counter,
                            train_indices=train_idx,
                            test_indices=test_idx,
                            epochs=epochs,
                            config=config,
                            log=logger,
                        )

                    folds.append((fold_counter, train_idx, test_idx, subject, fold_params))
                continue
            logger.warning(
                "Subject %s: ordered within-subject CV requested but no valid ordered block folds were found; falling back to GroupKFold.",
                subject,
            )

        block_cv, _ = create_block_aware_cv(subject_blocks, n_splits)

        if block_cv is None:
            logger.warning(f"Subject {subject}: insufficient blocks, skipping")
            continue

        for train_local, test_local in block_cv.split(subject_indices, groups=subject_blocks):
            fold_counter += 1
            train_idx = subject_indices[train_local]
            test_idx = subject_indices[test_local]
            
            # Apply CV hygiene to compute fold-specific parameters
            fold_params = None
            if apply_hygiene:
                fold_params = apply_fold_specific_hygiene(
                    fold_idx=fold_counter,
                    train_indices=train_idx,
                    test_indices=test_idx,
                    epochs=epochs,
                    config=config,
                    log=logger,
                )
            
            folds.append((fold_counter, train_idx, test_idx, subject, fold_params))

    return folds


def create_block_aware_inner_cv(
    blocks_train: np.ndarray,
    n_splits_inner: int,
    seed: int,
    fold: int,
    subject: str,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """Create block-aware inner CV splits."""
    block_cv, effective_splits = create_block_aware_cv(blocks_train, n_splits_inner)

    if block_cv is None:
        return None

    cv_splits = list(block_cv.split(np.arange(len(blocks_train)), groups=blocks_train))
    logger.info(f"Within fold {fold} ({subject}): block-aware inner CV ({effective_splits} splits)")
    return cv_splits


def get_inner_cv_splits(n_unique_groups: int, default: Optional[int] = None, config: Optional[Any] = None) -> int:
    """Get number of inner CV splits, capped by available groups."""
    if default is None:
        default = int(get_config_value(config, "machine_learning.cv.default_n_splits", 5))
    return int(np.clip(default, 2, n_unique_groups))


def get_min_channels_required(config: Any = None, min_absolute: int = 4) -> int:
    """Get minimum channels required for Riemann analysis."""
    if config is None:
        return min_absolute

    riemann_min = config.get("machine_learning.analysis.riemann.min_channels_for_fold", min_absolute)
    return max(min_absolute, int(riemann_min))


###################################################################
# Fitting Helpers
###################################################################


def _fit_default_pipeline(
    pipe,
    X_train: np.ndarray,
    y_train: np.ndarray,
    fold: int,
    random_state: Optional[int] = None,
) -> Any:
    """Fit pipeline without hyperparameter tuning."""
    pipe_clone = clone(pipe)
    if random_state is not None:
        if hasattr(pipe_clone, "set_params"):
            try:
                pipe_clone.set_params(regressor__regressor__random_state=random_state)
            except ValueError:
                pass
    pipe_clone.fit(X_train, y_train)
    return pipe_clone


###################################################################
# Metrics
###################################################################


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Compute regression metrics, optionally per-subject.
    
    Returns (pooled_metrics, per_subject_metrics).
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        empty = {"pearson_r": np.nan, "r2": np.nan, "explained_variance": np.nan, "avg_subject_r_fisher_z": np.nan, "n": 0}
        return empty, []

    yt, yp = y_true[mask], y_pred[mask]
    r, _ = safe_pearsonr(yt, yp)

    pooled = {
        "pearson_r": r,
        "r2": float(r2_score(yt, yp)),
        "explained_variance": float(explained_variance_score(yt, yp)),
        "n": int(mask.sum()),
    }

    per_subject: List[Dict] = []

    if groups is not None:
        gm = groups[mask] if len(groups) == len(y_true) else groups
        for subj in np.unique(gm):
            subj_mask = gm == subj
            if subj_mask.sum() < 2:
                continue
            ys, ps = yt[subj_mask], yp[subj_mask]
            rs, _ = safe_pearsonr(ys, ps)
            per_subject.append({
                "subject": str(subj),
                "r": rs,
                "n": int(subj_mask.sum()),
            })

        # Compute average Fisher-z transformed r
        valid_entries = [
            (p["r"], p["n"])
            for p in per_subject
            if np.isfinite(p["r"]) and p.get("n", 0) >= 2
        ]
        if valid_entries:
            r_vals, n_vals = zip(*valid_entries)
            clip_min, clip_max = get_fisher_z_clip_values(config)
            fisher_z = np.arctanh(
                np.clip(np.asarray(r_vals, dtype=float), clip_min, clip_max)
            )
            weights = np.maximum(np.asarray(n_vals, dtype=float) - 3.0, 1.0)
            weight_sum = np.sum(weights)
            pooled["avg_subject_r_fisher_z"] = (
                float(np.sum(fisher_z * weights) / weight_sum) if weight_sum > 0 else np.nan
            )
        else:
            pooled["avg_subject_r_fisher_z"] = np.nan
    else:
        pooled["avg_subject_r_fisher_z"] = np.nan

    return pooled, per_subject


###################################################################
# Feature-Matrix LOSO (Canonical)
###################################################################


def nested_loso_predictions_matrix(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    blocks: Optional[np.ndarray] = None,
    n_jobs: int = -1,
    seed: int = 42,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "elasticnet",
    outer_n_jobs: int = 1,
    null_n_perm: int = 0,
    null_output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    harmonization_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """
    Nested leave-one-subject-out cross-validation with hyperparameter tuning.
    
    This is the canonical feature-matrix-based LOSO implementation.
    Epoch-based CV helpers were removed; keep ML synchronized with the feature-table pipeline.

    Returns
    -------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    groups_ordered : List[str]
        Subject groups for each prediction
    test_indices : List[int]
        Original indices of test samples
    fold_ids : List[int]
        Fold ID for each prediction
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if blocks is not None:
        blocks = np.asarray(blocks)

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}")
    if blocks is not None and len(blocks) != len(y):
        raise ValueError(f"Length mismatch: blocks={len(blocks)} vs y={len(y)}")

    folds = create_loso_folds(X, groups)
    inner_n_jobs = determine_inner_n_jobs(outer_n_jobs, n_jobs)

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
        set_random_seeds(seed, fold)

        train_mask = np.isfinite(y[train_idx])
        test_mask = np.isfinite(y[test_idx])
        train_idx_f = train_idx[train_mask]
        test_idx_f = test_idx[test_mask]

        if len(train_idx_f) == 0 or len(test_idx_f) == 0:
            logger.warning(f"Fold {fold}: Empty after filtering; skipping")
            return {
                "fold": fold,
                "y_true": y[test_idx].tolist() if len(test_idx) > 0 else [],
                "y_pred": np.full(len(test_idx), np.nan).tolist() if len(test_idx) > 0 else [],
                "groups": groups[test_idx].tolist() if len(test_idx) > 0 else [],
                "test_idx": test_idx.tolist(),
                "best_params_rec": None,
            }

        X_train, X_test = X[train_idx_f], X[test_idx_f]
        y_train, y_test = y[train_idx_f], y[test_idx_f]
        groups_train = groups[train_idx_f]
        X_train, X_test, _ = apply_fold_feature_harmonization(
            X_train,
            X_test,
            groups_train,
            harmonization_mode,
        )

        n_train_subjects = len(np.unique(groups_train))

        if n_train_subjects < 2:
            logger.warning(f"Fold {fold}: <2 subjects for inner CV")
            estimator = clone(pipe)
            best_estimator = fit_with_warning_logging(estimator, X_train, y_train, f"fold {fold}", logger)
            best_params_rec = None
        else:
            inner_cv = create_inner_cv(groups_train, inner_cv_splits)
            scoring = create_scoring_dict()

            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=inner_n_jobs,
                refit="r",
                error_score="raise",
            )
            gs = grid_search_with_warning_logging(gs, X_train, y_train, f"fold {fold}", logger, groups=groups_train)
            best_estimator = gs.best_estimator_

            if best_params_log_path and hasattr(gs, "cv_results_"):
                cv_df = pd.DataFrame(gs.cv_results_)
                heldout = [str(groups[test_idx[0]])] if len(test_idx) > 0 else None
                best_params_rec = create_best_params_record(model_name, fold, cv_df, gs, heldout_subjects=heldout)
            else:
                best_params_rec = None

        y_pred = best_estimator.predict(X_test)

        return {
            "fold": fold,
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "groups": groups[test_idx_f].tolist(),
            "test_idx": test_idx_f.tolist(),
            "best_params_rec": best_params_rec,
        }

    results = execute_folds_parallel(folds, _run_fold, outer_n_jobs)

    if best_params_log_path:
        records = [r["best_params_rec"] for r in results if r.get("best_params_rec")]
        if records:
            _save_best_params(records, best_params_log_path)

    y_true, y_pred, groups_ordered, test_indices, fold_ids = aggregate_fold_results(results)

    if null_n_perm > 0 and null_output_path:
        run_permutation_test(
            X, y, groups, blocks, pipe, param_grid, inner_cv_splits, inner_n_jobs,
            seed, model_name, null_n_perm, null_output_path, config,
            harmonization_mode=harmonization_mode,
        )

    return y_true, y_pred, groups_ordered, test_indices, fold_ids


def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    blocks: Optional[np.ndarray],
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    inner_n_jobs: int,
    seed: int,
    model_name: str,
    null_n_perm: int,
    null_output_path: Path,
    config: Optional[Dict[str, Any]],
    harmonization_mode: Optional[str] = None,
) -> None:
    """Run permutation test for null distribution."""
    logger.info(f"Computing {null_n_perm} permutation null distributions...")
    rng = np.random.default_rng(seed)
    null_r = []
    null_r2 = []
    n_completed = 0
    n_effective = 0
    min_shuffle_fraction = float(
        get_config_value(config, "machine_learning.cv.min_label_shuffle_fraction", 0.01)
    )

    perm_scheme = "within_subject_within_block"
    if config is not None:
        try:
            perm_scheme = str(get_config_value(config, "machine_learning.cv.permutation_scheme", perm_scheme)).strip().lower()
        except Exception:
            perm_scheme = "within_subject_within_block"
    if perm_scheme not in {"within_subject", "within_subject_within_block"}:
        perm_scheme = "within_subject_within_block"

    blocks_arr = None
    if perm_scheme == "within_subject_within_block":
        if blocks is None:
            logger.warning("Permutation scheme 'within_subject_within_block' requested but blocks are missing; falling back to within_subject.")
            perm_scheme = "within_subject"
        else:
            blocks_arr = np.asarray(blocks)
            if len(blocks_arr) != len(y):
                logger.warning("Permutation blocks length mismatch; falling back to within_subject.")
                perm_scheme = "within_subject"
                blocks_arr = None

    for perm in range(null_n_perm):
        y_perm = y.copy()
        for subj in np.unique(groups):
            subj_mask = groups == subj
            if perm_scheme == "within_subject_within_block" and blocks_arr is not None:
                subj_blocks = blocks_arr[subj_mask]
                for b in np.unique(subj_blocks):
                    if np.isfinite(b):
                        bm = subj_mask & (blocks_arr == b)
                    else:
                        bm = subj_mask & (~np.isfinite(blocks_arr))
                    if np.sum(bm) >= 2:
                        y_perm[bm] = rng.permutation(y_perm[bm])
            else:
                y_perm[subj_mask] = rng.permutation(y_perm[subj_mask])

        effective, changed_fraction = is_effective_permutation(
            y,
            y_perm,
            min_changed_fraction=min_shuffle_fraction,
        )
        if not effective:
            logger.debug(
                "Perm %d skipped: ineffective shuffle (changed_fraction=%.4f, required>=%.4f)",
                int(perm + 1),
                float(changed_fraction),
                float(min_shuffle_fraction),
            )
            continue
        n_effective += 1

        y_true_p, y_pred_p, groups_p, _, _ = nested_loso_predictions_matrix(
            X=X, y=y_perm, groups=groups, blocks=blocks_arr, pipe=pipe, param_grid=param_grid,
            inner_cv_splits=inner_cv_splits, n_jobs=inner_n_jobs, seed=seed + perm,
            model_name=model_name, outer_n_jobs=1, null_n_perm=0, config=config,
            harmonization_mode=harmonization_mode,
        )

        pred_df = pd.DataFrame({"y_true": y_true_p, "y_pred": y_pred_p, "subject_id": groups_p})
        r_subj, _, _, _ = compute_subject_level_r(pred_df, config)
        r2 = np.nan
        if len(y_true_p) > 1:
            y_true_arr = np.asarray(y_true_p, dtype=float)
            y_pred_arr = np.asarray(y_pred_p, dtype=float)
            if np.all(np.isfinite(y_true_arr)) and np.all(np.isfinite(y_pred_arr)):
                try:
                    r2 = float(r2_score(y_true_arr, y_pred_arr))
                except Exception:
                    r2 = np.nan

        if np.isfinite(r_subj) and np.isfinite(r2):
            null_r.append(float(r_subj))
            null_r2.append(float(r2))
            n_completed += 1
        else:
            logger.warning(f"Perm {perm + 1}: non-finite (r={r_subj:.3f}, r2={r2:.3f})")

    if n_effective == 0:
        raise RuntimeError(
            "No effective permutations could be generated under the current permutation scheme. "
            "Try machine_learning.cv.permutation_scheme='within_subject' and/or lower "
            "machine_learning.cv.min_label_shuffle_fraction."
        )

    completion_rate = n_completed / null_n_perm if null_n_perm > 0 else 0.0
    min_completion = float(get_config_value(config, "machine_learning.cv.min_valid_permutation_fraction", 0.5))
    if completion_rate < min_completion:
        raise RuntimeError(
            f"Insufficient valid permutations ({n_completed}/{null_n_perm}, "
            f"rate={completion_rate:.3f} < required {min_completion:.3f})"
        )

    np.savez(
        null_output_path,
        null_r=np.asarray(null_r),
        null_r2=np.asarray(null_r2),
        n_requested=null_n_perm,
        n_completed=n_completed,
        n_effective=n_effective,
    )
    logger.info(f"Saved null distributions to {null_output_path}")
