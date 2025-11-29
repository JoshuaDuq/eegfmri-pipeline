"""
Decoding Pipeline.

Machine learning decoding for EEG-based prediction of behavioral outcomes.

Features:
- Nested leave-one-subject-out cross-validation
- Hyperparameter tuning with inner CV
- Permutation testing for statistical inference
- Time-generalization analysis

Usage:
    run_regression_decoding(subjects, task, deriv_root, config, ...)
    run_time_generalization(subjects, task, deriv_root, config, ...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from eeg_pipeline.utils.io.general import get_logger

logger = get_logger(__name__)


###################################################################
# Nested LOSO Cross-Validation
###################################################################


def nested_loso_predictions(
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
    from eeg_pipeline.analysis.decoding.cv import (
        create_loso_folds,
        determine_inner_n_jobs,
        set_random_seeds,
        fit_with_warning_logging,
        create_inner_cv,
        create_scoring_dict,
        grid_search_with_warning_logging,
        create_best_params_record,
        execute_folds_parallel,
        aggregate_fold_results,
        compute_subject_level_r,
        _save_best_params,
    )

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}")

    folds = create_loso_folds(X, groups)
    inner_n_jobs = determine_inner_n_jobs(outer_n_jobs, n_jobs)

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
        set_random_seeds(seed, fold)

        # Filter finite targets
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

    # Save best params
    if best_params_log_path:
        records = [r["best_params_rec"] for r in results if r.get("best_params_rec")]
        if records:
            _save_best_params(records, best_params_log_path)

    y_true, y_pred, groups_ordered, test_indices, fold_ids = aggregate_fold_results(results)

    # Permutation testing
    if null_n_perm > 0 and null_output_path:
        _run_permutation_test(
            X, y, groups, pipe, param_grid, inner_cv_splits, inner_n_jobs,
            seed, model_name, null_n_perm, null_output_path, config
        )

    return y_true, y_pred, groups_ordered, test_indices, fold_ids


def _run_permutation_test(
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
    from eeg_pipeline.analysis.decoding.cv import compute_subject_level_r

    logger.info(f"Computing {null_n_perm} permutation null distributions...")
    rng = np.random.default_rng(seed)
    null_r = []
    null_r2 = []
    n_completed = 0

    for perm in range(null_n_perm):
        # Shuffle within subjects
        y_perm = y.copy()
        for subj in np.unique(groups):
            mask = groups == subj
            y_perm[mask] = rng.permutation(y_perm[mask])

        y_true_p, y_pred_p, groups_p, _, _ = nested_loso_predictions(
            X=X, y=y_perm, groups=groups, pipe=pipe, param_grid=param_grid,
            inner_cv_splits=inner_cv_splits, n_jobs=inner_n_jobs, seed=seed + perm,
            model_name=model_name, outer_n_jobs=1, null_n_perm=0, config=config,
        )

        pred_df = pd.DataFrame({"y_true": y_true_p, "y_pred": y_pred_p, "subject_id": groups_p})
        r_subj, _, _, _ = compute_subject_level_r(pred_df)
        r2 = r2_score(y_true_p, y_pred_p) if len(y_true_p) > 1 else np.nan

        if np.isfinite(r_subj) and np.isfinite(r2):
            null_r.append(float(r_subj))
            null_r2.append(float(r2))
            n_completed += 1
        else:
            logger.warning(f"Perm {perm + 1}: non-finite (r={r_subj:.3f}, r2={r2:.3f})")

    # Validate completion rate
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


###################################################################
# Regression Decoding
###################################################################


def run_regression_decoding(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
) -> Path:
    """Run LOSO regression decoding on plateau features."""
    from eeg_pipeline.utils.io.general import ensure_dir
    from eeg_pipeline.utils.io.decoding import export_predictions, export_indices, prepare_best_params_path
    from eeg_pipeline.plotting.decoding import (
        plot_prediction_scatter,
        plot_per_subject_performance,
        plot_residual_diagnostics,
        plot_permutation_null,
    )
    from eeg_pipeline.analysis.decoding.cv import compute_subject_level_r
    from eeg_pipeline.analysis.decoding.pipelines import (
        create_elasticnet_pipeline,
        build_elasticnet_param_grid,
    )
    from eeg_pipeline.analysis.decoding.data import load_plateau_matrix

    X, y, groups, feature_names, meta = load_plateau_matrix(subjects, task, deriv_root, config, logger)

    results_dir = results_root / "regression"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    pipe = create_elasticnet_pipeline(seed=rng_seed, config=config)
    param_grid = build_elasticnet_param_grid(config)
    best_params_path = prepare_best_params_path(results_dir / "best_params_elasticnet.jsonl", mode="truncate")
    null_path = results_dir / "loso_null_elasticnet.npz" if n_perm > 0 else None

    y_true, y_pred, groups_ordered, test_indices, fold_ids = nested_loso_predictions(
        X=X, y=y, groups=groups, pipe=pipe, param_grid=param_grid,
        inner_cv_splits=inner_splits, n_jobs=-1, seed=rng_seed,
        best_params_log_path=best_params_path, model_name="elasticnet",
        outer_n_jobs=outer_jobs, null_n_perm=n_perm, null_output_path=null_path,
    )

    # Export predictions
    pred_path = results_dir / "loso_predictions.tsv"
    pred_df = export_predictions(
        y_true, y_pred, groups_ordered, test_indices, fold_ids,
        "elasticnet", meta.reset_index(drop=True), pred_path,
    )
    export_indices(groups_ordered, test_indices, fold_ids, meta.reset_index(drop=True),
                   results_dir / "loso_indices.tsv", add_heldout_subject_id=True)

    # Compute metrics
    r_subj, per_subj_r, ci_low, ci_high = compute_subject_level_r(pred_df)
    p_val = np.nan

    if null_path and null_path.exists():
        data = np.load(null_path)
        null_rs = data.get("null_r")
        if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subj):
            finite = null_rs[np.isfinite(null_rs)]
            if finite.size > 0:
                p_val = float(((np.abs(finite) >= abs(r_subj)).sum() + 1) / (finite.size + 1))

    try:
        r2_val = float(r2_score(y_true, y_pred))
    except Exception:
        r2_val = np.nan

    metrics = {
        "pearson_r": r_subj,
        "r_subject_ci_low": ci_low,
        "r_subject_ci_high": ci_high,
        "r2": r2_val,
        "p_value": p_val,
    }

    # Plots
    plot_prediction_scatter(pred_df, "elasticnet", metrics, plots_dir / "loso_prediction_scatter.png", config=config)
    plot_per_subject_performance(pred_df, "elasticnet", plots_dir / "loso_per_subject_performance.png", config=config)
    plot_residual_diagnostics(pred_df, "elasticnet", plots_dir / "loso_residual_diagnostics.png", config=config)

    if null_path and null_path.exists():
        data = np.load(null_path)
        null_rs = data.get("null_r")
        if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subj):
            plot_permutation_null(null_rs, r_subj, p_val, plots_dir / "loso_null_distribution.png", config=config)

    logger.info(f"Saved decoding results to {results_dir}")
    return results_dir


###################################################################
# Time Generalization
###################################################################


def run_time_generalization(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
) -> None:
    """Run time-generalization decoding analysis."""
    from eeg_pipeline.utils.io.general import ensure_dir
    from eeg_pipeline.plotting.decoding import plot_time_generalization_with_null
    from eeg_pipeline.analysis.decoding.time_generalization import time_generalization_regression

    results_dir = results_root / "time_generalization"
    ensure_dir(results_dir)

    try:
        tg_r, tg_r2, window_centers = time_generalization_regression(
            deriv_root=deriv_root,
            subjects=subjects,
            task=task,
            results_dir=results_dir,
            config_dict=config,
            n_perm=n_perm,
            seed=rng_seed,
        )
    except Exception as exc:
        logger.warning(f"Time-generalization failed: {exc}")
        return

    tg_path = results_dir / "time_generalization_regression.npz"
    tg_npz = np.load(tg_path) if tg_path.exists() else None
    null_mat = tg_npz["null_r"] if tg_npz and "null_r" in tg_npz else None

    plot_time_generalization_with_null(
        tg_matrix=tg_r, null_matrix=null_mat, window_centers=window_centers,
        save_path=results_dir / "time_generalization_r.png", metric="r", config=config,
    )
    plot_time_generalization_with_null(
        tg_matrix=tg_r2, null_matrix=null_mat, window_centers=window_centers,
        save_path=results_dir / "time_generalization_r2.png", metric="r2", config=config,
    )

    logger.info(f"Saved time-generalization outputs to {results_dir}")
