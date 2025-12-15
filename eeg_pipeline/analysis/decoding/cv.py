"""
Cross-validation utilities for decoding.

This module provides CV helpers, fold creation, and result aggregation.
"""

from __future__ import annotations

import json
import logging
import random as pyrandom
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning, norm
import scipy.stats
from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from eeg_pipeline.io.logging import get_logger
from eeg_pipeline.utils.config.loader import get_fisher_z_clip_values, get_config_value

logger = get_logger(__name__)


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


def create_inner_cv(train_groups: np.ndarray, inner_cv_splits: int) -> GroupKFold:
    """Create inner CV splitter for hyperparameter tuning."""
    n_unique = len(np.unique(train_groups))
    n_splits = min(inner_cv_splits, n_unique)
    return GroupKFold(n_splits=n_splits)


def create_stratified_cv_by_binned_targets(
    y: np.ndarray,
    n_splits: Optional[int] = None,
    n_bins: Optional[int] = None,
    random_state: int = 42,
    config: Optional[Any] = None,
) -> Tuple[StratifiedKFold, np.ndarray]:
    """Create stratified CV by binning continuous targets."""
    if n_splits is None:
        n_splits = int(get_config_value(config, "decoding.cv.default_n_splits", 5))
    if n_bins is None:
        n_bins = int(get_config_value(config, "decoding.cv.default_n_bins", 5))
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    if pd.isna(y_binned).all():
        y_binned = np.zeros_like(y, dtype=int)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_binned


def create_block_aware_cv(
    blocks: np.ndarray,
    n_splits: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[Optional[GroupKFold], int]:
    """Create block-aware GroupKFold CV."""
    if n_splits is None:
        n_splits = int(get_config_value(config, "decoding.cv.default_n_splits", 5))
    unique_blocks = np.unique(blocks[~pd.isna(blocks)])
    n_unique = len(unique_blocks)

    if n_unique < 2:
        logger.warning(f"Insufficient blocks ({n_unique}) for GroupKFold")
        return None, 0

    effective_splits = min(n_splits, n_unique)
    if effective_splits < n_splits:
        logger.info(f"Reduced splits from {n_splits} to {effective_splits}")

    return GroupKFold(n_splits=effective_splits), effective_splits


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

    if should_parallelize_folds(outer_n_jobs, len(folds)):
        return Parallel(n_jobs=outer_n_jobs, prefer="threads")(
            delayed(fold_func)(fold, train_idx, test_idx)
            for (fold, train_idx, test_idx) in folds
        )
    return [fold_func(fold, train_idx, test_idx) for (fold, train_idx, test_idx) in folds]


###################################################################
# Warning Handling
###################################################################


def _log_warning(warning, fold_info: str, context: str) -> None:
    """Log sklearn warnings with context."""
    warning_types = {
        ConvergenceWarning: "ConvergenceWarning",
        ConstantInputWarning: "ConstantInputWarning",
    }
    warning_type = warning_types.get(type(warning.message), "Warning")
    logger.warning(f"Fold {fold_info}: {warning_type} {context} - {warning.message}")


def fit_with_warning_logging(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    fold_info: str = "",
    log: Optional[logging.Logger] = None,
):
    """Fit estimator with warning logging."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimator.fit(X, y)
        for warning in w:
            if isinstance(warning.message, (ConvergenceWarning, ConstantInputWarning)):
                _log_warning(warning, fold_info, "")
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid.fit(X, y, **fit_params)
        for warning in w:
            if isinstance(warning.message, (ConvergenceWarning, ConstantInputWarning)):
                _log_warning(warning, fold_info, "during GridSearchCV")
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
) -> Tuple[float, List[Tuple[str, float]], float, float]:
    """Compute subject-level Pearson r with Fisher z aggregation."""
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
            r_subj, _ = scipy.stats.pearsonr(yt[finite], yp[finite])
        except Exception:
            continue

        if np.isfinite(r_subj):
            per_subject.append((str(subj), float(r_subj), int(n_trials)))

    if not per_subject:
        return np.nan, [], np.nan, np.nan

    clip_min, clip_max = get_fisher_z_clip_values(config)
    r_vals = np.clip([r for _, r, _ in per_subject], clip_min, clip_max)
    n_trials_vals = np.asarray([n for _, _, n in per_subject], dtype=int)

    # Fisher z transform and weighted average
    z_vals = np.arctanh(r_vals)
    weights = np.maximum(n_trials_vals - 3, 1.0)
    weights = weights / weights.sum()
    mean_z = float(np.average(z_vals, weights=weights))
    agg_r = float(np.tanh(mean_z))

    # Confidence interval
    ci_low, ci_high = np.nan, np.nan
    if len(z_vals) > 1:
        weighted_var = np.average((z_vals - mean_z) ** 2, weights=weights)
        se = float(np.sqrt(weighted_var / len(z_vals)))
        if np.isfinite(se) and se > 0:
            delta = 1.96 * se
            ci_low = float(np.tanh(mean_z - delta))
            ci_high = float(np.tanh(mean_z + delta))

    return agg_r, [(s, r) for s, r, _ in per_subject], ci_low, ci_high


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
    min_trials_for_within_subject: Optional[int] = None,
    config: Optional[Any] = None,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    """Create within-subject CV folds."""
    if min_trials_for_within_subject is None:
        min_trials_for_within_subject = int(get_config_value(config, "decoding.cv.min_trials_for_within_subject", 2))
    
    folds: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
    fold_counter = 0
    unique_subs = [str(s) for s in np.unique(groups)]

    for subject in unique_subs:
        subject_indices = np.where(groups == subject)[0]
        n_samples = len(subject_indices)

        if n_samples < min_trials_for_within_subject:
            logger.warning(f"Subject {subject}: <{min_trials_for_within_subject} trials, skipping")
            continue

        n_splits = min(inner_cv_splits, n_samples)

        if blocks_all is None:
            logger.warning(f"Subject {subject}: missing run_id, skipping")
            continue

        subject_blocks = blocks_all[subject_indices]
        block_cv, effective_splits = create_block_aware_cv(subject_blocks, n_splits)

        if block_cv is None:
            logger.warning(f"Subject {subject}: insufficient blocks, skipping")
            continue

        for train_local, test_local in block_cv.split(subject_indices, groups=subject_blocks):
            fold_counter += 1
            train_idx = subject_indices[train_local]
            test_idx = subject_indices[test_local]
            folds.append((fold_counter, train_idx, test_idx, subject))

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
        default = int(get_config_value(config, "decoding.cv.default_n_splits", 5))
    return int(np.clip(default, 2, n_unique_groups))


def get_min_channels_required(config: Any = None, min_absolute: int = 4) -> int:
    """Get minimum channels required for Riemann analysis."""
    if config is None:
        return min_absolute

    riemann_min = config.get("decoding.analysis.riemann.min_channels_for_fold", min_absolute)
    return max(min_absolute, int(riemann_min))


###################################################################
# Fitting Helpers
###################################################################


def _fit_with_inner_cv(
    pipe,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    config_dict: Optional[Dict],
    fold: int,
    inner_n_jobs: int,
    log: logging.Logger,
    seed: int,
    inner_splits: int,
) -> Tuple[Any, Optional[pd.DataFrame], Optional[GridSearchCV]]:
    """Fit pipeline with inner CV for hyperparameter tuning.
    
    Returns:
        Tuple of (best_estimator, cv_results_df, grid_search_cv)
        If n_splits < 2, returns (fitted_pipe, None, None)
    """
    from eeg_pipeline.analysis.decoding.pipelines import build_elasticnet_param_grid

    n_unique = len(np.unique(train_groups))
    n_splits = get_inner_cv_splits(n_unique, inner_splits)

    if n_splits < 2:
        log.info(f"Fold {fold}: <2 groups, fitting without inner CV")
        pipe.fit(X_train, y_train)
        return pipe, None, None

    inner_cv = create_inner_cv(train_groups, n_splits)
    param_grid = build_elasticnet_param_grid(config_dict)
    scoring = create_scoring_dict()

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=inner_cv,
        n_jobs=inner_n_jobs,
        refit="r",
    )
    gs = grid_search_with_warning_logging(gs, X_train, y_train, f"fold {fold}", log, groups=train_groups)

    cv_results = pd.DataFrame(gs.cv_results_)
    return gs.best_estimator_, cv_results, gs


def _fit_default_pipeline(
    pipe,
    X_train: np.ndarray,
    y_train: np.ndarray,
    fold: int,
    random_state: Optional[int] = None,
) -> Any:
    """Fit pipeline without hyperparameter tuning."""
    from sklearn.base import clone
    pipe_clone = clone(pipe)
    if random_state is not None:
        if hasattr(pipe_clone, "set_params"):
            try:
                pipe_clone.set_params(regressor__regressor__random_state=random_state)
            except ValueError:
                pass
    pipe_clone.fit(X_train, y_train)
    return pipe_clone


def _predict_and_log(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    log: logging.Logger,
) -> np.ndarray:
    """Make predictions and log fold performance."""
    y_pred = estimator.predict(X_test)
    r, _ = safe_pearsonr(y_test, y_pred)
    log.info(f"Fold {fold}: r={r:.3f}, n_test={len(y_test)}")
    return y_pred


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
    from sklearn.metrics import r2_score, explained_variance_score

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
    n_jobs: int = -1,
    seed: int = 42,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "elasticnet",
    outer_n_jobs: int = 1,
    null_n_perm: int = 0,
    null_output_path: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """
    Nested leave-one-subject-out cross-validation with hyperparameter tuning.
    
    This is the canonical feature-matrix-based LOSO implementation.
    For epoch-based LOSO with channel harmonization, use orchestration.nested_loso_predictions.

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

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}")

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
            X, y, groups, pipe, param_grid, inner_cv_splits, inner_n_jobs,
            seed, model_name, null_n_perm, null_output_path, config
        )

    return y_true, y_pred, groups_ordered, test_indices, fold_ids


def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    inner_n_jobs: int,
    seed: int,
    model_name: str,
    null_n_perm: int,
    null_output_path: Path,
    config: Optional[Dict[str, Any]],
) -> None:
    """Run permutation test for null distribution."""
    logger.info(f"Computing {null_n_perm} permutation null distributions...")
    rng = np.random.default_rng(seed)
    null_r = []
    null_r2 = []
    n_completed = 0

    for perm in range(null_n_perm):
        y_perm = y.copy()
        for subj in np.unique(groups):
            mask = groups == subj
            y_perm[mask] = rng.permutation(y_perm[mask])

        y_true_p, y_pred_p, groups_p, _, _ = nested_loso_predictions_matrix(
            X=X, y=y_perm, groups=groups, pipe=pipe, param_grid=param_grid,
            inner_cv_splits=inner_cv_splits, n_jobs=inner_n_jobs, seed=seed + perm,
            model_name=model_name, outer_n_jobs=1, null_n_perm=0, config=config,
        )

        pred_df = pd.DataFrame({"y_true": y_true_p, "y_pred": y_pred_p, "subject_id": groups_p})
        r_subj, _, _, _ = compute_subject_level_r(pred_df, config)
        r2 = r2_score(y_true_p, y_pred_p) if len(y_true_p) > 1 else np.nan

        if np.isfinite(r_subj) and np.isfinite(r2):
            null_r.append(float(r_subj))
            null_r2.append(float(r2))
            n_completed += 1
        else:
            logger.warning(f"Perm {perm + 1}: non-finite (r={r_subj:.3f}, r2={r2:.3f})")

    completion_rate = n_completed / null_n_perm if null_n_perm > 0 else 0.0
    if completion_rate < 0.5:
        raise RuntimeError(f"Insufficient valid permutations ({n_completed}/{null_n_perm})")

    np.savez(
        null_output_path,
        null_r=np.asarray(null_r),
        null_r2=np.asarray(null_r2),
        n_requested=null_n_perm,
        n_completed=n_completed,
    )
    logger.info(f"Saved null distributions to {null_output_path}")
