from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from eeg_pipeline.utils.io.general import ensure_dir
from eeg_pipeline.utils.io.decoding import (
    export_predictions,
    export_indices,
    prepare_best_params_path,
)
from eeg_pipeline.plotting.decoding import (
    plot_prediction_scatter,
    plot_per_subject_performance,
    plot_residual_diagnostics,
    plot_permutation_null,
    plot_time_generalization_with_null,
)
from eeg_pipeline.analysis.decoding.time_generalization import time_generalization_regression

from ..analysis.decoding import (
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
    load_plateau_matrix,
    create_elasticnet_pipeline,
    build_elasticnet_param_grid,
)

logger = logging.getLogger(__name__)


###################################################################
# Decoding Pipeline Functions
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
    """Perform nested leave-one-subject-out cross-validation with hyperparameter tuning.
    
    Returns:
        y_true: True target values
        y_pred: Predicted target values
        groups_ordered: Subject groups for each prediction
        test_indices: Original indices of test samples
        fold_ids: Fold ID for each prediction
    """
    from eeg_pipeline.utils.analysis.decoding import _save_best_params
    
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(f"X ({len(X)}), y ({len(y)}), and groups ({len(groups)}) must have the same length")
    
    folds = create_loso_folds(X, groups)
    inner_n_jobs = determine_inner_n_jobs(outer_n_jobs, n_jobs)
    
    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
        set_random_seeds(seed, fold)
        
        train_finite_mask = np.isfinite(y[train_idx])
        test_finite_mask = np.isfinite(y[test_idx])
        
        train_idx_filtered = train_idx[train_finite_mask]
        test_idx_filtered = test_idx[test_finite_mask]
        
        if len(train_idx_filtered) == 0 or len(test_idx_filtered) == 0:
            logger.warning(
                f"Fold {fold}: Empty train ({len(train_idx_filtered)}) or test ({len(test_idx_filtered)}) "
                f"set after filtering finite targets. Skipping fold."
            )
            return {
                "fold": fold,
                "y_true": y[test_idx].tolist() if len(test_idx) > 0 else [],
                "y_pred": np.full(len(test_idx), np.nan, dtype=float).tolist() if len(test_idx) > 0 else [],
                "groups": groups[test_idx].tolist() if len(test_idx) > 0 else [],
                "test_idx": test_idx.tolist(),
                "best_params_rec": None,
            }
        
        X_train, X_test = X[train_idx_filtered], X[test_idx_filtered]
        y_train, y_test = y[train_idx_filtered], y[test_idx_filtered]
        groups_train = groups[train_idx_filtered]
        
        train_groups_unique = np.unique(groups_train)
        n_train_subjects = len(train_groups_unique)
        
        if n_train_subjects < 2:
            logger.warning(f"Fold {fold}: insufficient subjects ({n_train_subjects}) for inner CV")
            estimator = clone(pipe)
            best_estimator = fit_with_warning_logging(estimator, X_train, y_train, fold_info=f"fold {fold}", logger=logger)
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
            
            gs = grid_search_with_warning_logging(
                gs, X_train, y_train, fold_info=f"fold {fold}", logger=logger, groups=groups_train
            )
            
            best_estimator = gs.best_estimator_
            
            if best_params_log_path is not None and hasattr(gs, "cv_results_"):
                cv_results_df = pd.DataFrame(gs.cv_results_)
                best_params_rec = create_best_params_record(
                    model_name, fold, cv_results_df, gs,
                    heldout_subjects=[str(groups[test_idx[0]])] if len(test_idx) > 0 else None
                )
            else:
                best_params_rec = None
        
        y_pred = best_estimator.predict(X_test)
        
        return {
            "fold": fold,
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "groups": groups[test_idx_filtered].tolist(),
            "test_idx": test_idx_filtered.tolist(),
            "best_params_rec": best_params_rec,
        }
    
    results = execute_folds_parallel(folds, _run_fold, outer_n_jobs)
    
    if best_params_log_path is not None:
        best_param_records = [r["best_params_rec"] for r in results if r.get("best_params_rec") is not None]
        if best_param_records:
            _save_best_params(best_param_records, best_params_log_path)
    
    y_true, y_pred, groups_ordered, test_indices, fold_ids = aggregate_fold_results(results)
    
    if null_n_perm > 0 and null_output_path is not None:
        logger.info(f"Computing {null_n_perm} permutation null distributions...")
        rng = np.random.default_rng(seed)
        null_r = []
        null_r2 = []
        n_completed = 0
        n_requested = null_n_perm
        min_completion_threshold = 0.5
        
        for perm in range(null_n_perm):
            y_perm = y.copy()
            for subject_id in np.unique(groups):
                subject_mask = groups == subject_id
                subject_indices = np.where(subject_mask)[0]
                y_perm[subject_indices] = rng.permutation(y_perm[subject_indices])
            
            y_true_perm, y_pred_perm, groups_perm, _, _ = nested_loso_predictions(
                X=X, y=y_perm, groups=groups, pipe=pipe, param_grid=param_grid,
                inner_cv_splits=inner_cv_splits, n_jobs=inner_n_jobs, seed=seed + perm,
                best_params_log_path=None, model_name=model_name, outer_n_jobs=1,
                null_n_perm=0, null_output_path=None, config=config
            )
            
            pred_df_perm = pd.DataFrame({
                "y_true": y_true_perm,
                "y_pred": y_pred_perm,
                "subject_id": groups_perm,
            })
            r_subject_perm, _, _, _ = compute_subject_level_r(pred_df_perm)
            r2_perm = r2_score(y_true_perm, y_pred_perm) if len(y_true_perm) > 1 else np.nan
            
            if np.isfinite(r_subject_perm) and np.isfinite(r2_perm):
                null_r.append(float(r_subject_perm))
                null_r2.append(float(r2_perm))
                n_completed += 1
            else:
                logger.warning(
                    f"Permutation {perm + 1}/{null_n_perm}: Non-finite correlation "
                    f"(r={r_subject_perm:.3f}, r2={r2_perm:.3f}). Skipping this permutation."
                )
        
        completion_rate = n_completed / n_requested if n_requested > 0 else 0.0
        
        if completion_rate < min_completion_threshold:
            logger.error(
                f"Permutation null completion rate ({n_completed}/{n_requested} = {completion_rate:.1%}) "
                f"below threshold ({min_completion_threshold:.1%}). "
                f"P-values may be unreliable. Aborting."
            )
            raise RuntimeError(
                f"Insufficient valid permutations ({n_completed}/{n_requested}). "
                f"Minimum completion rate {min_completion_threshold:.1%} required."
            )
        elif completion_rate < 1.0:
            logger.warning(
                f"Permutation null completion rate ({n_completed}/{n_requested} = {completion_rate:.1%}). "
                f"P-values computed from {n_completed} valid permutations."
            )
        
        np.savez(
            null_output_path,
            null_r=np.asarray(null_r),
            null_r2=np.asarray(null_r2),
            n_requested=n_requested,
            n_completed=n_completed,
        )
        logger.info(f"Saved null distributions to {null_output_path} ({n_completed}/{n_requested} valid permutations)")
    
    return y_true, y_pred, groups_ordered, test_indices, fold_ids


def run_regression_decoding(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config,
    n_perm: int,
    inner_splits: int,
    outer_jobs: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
) -> Path:
    """Run LOSO regression decoding on plateau features and save outputs."""
    from eeg_pipeline.utils.io.general import ensure_dir
    from eeg_pipeline.utils.io.decoding import (
        export_predictions,
        export_indices,
        prepare_best_params_path,
    )
    from eeg_pipeline.plotting.decoding import (
        plot_prediction_scatter,
        plot_per_subject_performance,
        plot_residual_diagnostics,
        plot_permutation_null,
    )

    X, y, groups, feature_names, meta = load_plateau_matrix(subjects, task, deriv_root, config, logger)

    results_dir = results_root / "regression"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    pipe = create_elasticnet_pipeline(seed=rng_seed, config=config)
    param_grid = build_elasticnet_param_grid(config)

    best_params_log_path = prepare_best_params_path(results_dir / "best_params_elasticnet.jsonl", mode="truncate", run_id=None)
    null_output_path = results_dir / "loso_null_elasticnet.npz" if n_perm > 0 else None

    y_true, y_pred, groups_ordered, test_indices, fold_ids = nested_loso_predictions(
        X=X,
        y=y,
        groups=groups,
        pipe=pipe,
        param_grid=param_grid,
        inner_cv_splits=inner_splits,
        n_jobs=-1,
        seed=rng_seed,
        best_params_log_path=best_params_log_path,
        model_name="elasticnet",
        outer_n_jobs=outer_jobs,
        null_n_perm=n_perm,
        null_output_path=null_output_path,
    )

    pred_path = results_dir / "loso_predictions.tsv"
    meta_full = meta.reset_index(drop=True)
    pred_df = export_predictions(
        y_true=y_true,
        y_pred=y_pred,
        groups_ordered=groups_ordered,
        test_indices=test_indices,
        fold_ids=fold_ids,
        model_name="elasticnet",
        meta=meta_full,
        save_path=pred_path,
    )
    idx_path = results_dir / "loso_indices.tsv"
    export_indices(groups_ordered, test_indices, fold_ids, meta_full, idx_path, add_heldout_subject_id=True)

    r_subject, per_subject_r, ci_low, ci_high = compute_subject_level_r(pred_df)
    p_val = np.nan
    
    if null_output_path is not None and null_output_path.exists():
        data = np.load(null_output_path)
        null_rs = data.get("null_r")
        n_requested = data.get("n_requested", None)
        n_completed = data.get("n_completed", None)
        
        if n_requested is not None and n_completed is not None:
            completion_rate = n_completed / n_requested if n_requested > 0 else 0.0
            if completion_rate < 1.0:
                logger.warning(
                    f"Loaded null distribution with completion rate {n_completed}/{n_requested} = {completion_rate:.1%}. "
                    f"P-values computed from {n_completed} valid permutations."
                )
        
        if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subject):
            finite = null_rs[np.isfinite(null_rs)]
            if finite.size > 0:
                p_val = float(((np.abs(finite) >= abs(r_subject)).sum() + 1) / (finite.size + 1))
    
    try:
        r2_val = float(r2_score(y_true, y_pred))
    except Exception:
        r2_val = np.nan

    pooled_metrics = {
        "pearson_r": r_subject,
        "r_subject_ci_low": ci_low,
        "r_subject_ci_high": ci_high,
        "r2": r2_val,
        "p_value": p_val,
    }
    scatter_path = plots_dir / "loso_prediction_scatter.png"
    plot_prediction_scatter(pred_df, "elasticnet", pooled_metrics, scatter_path, config=config)

    perf_path = plots_dir / "loso_per_subject_performance.png"
    plot_per_subject_performance(pred_df, "elasticnet", perf_path, config=config)

    resid_path = plots_dir / "loso_residual_diagnostics.png"
    plot_residual_diagnostics(pred_df, "elasticnet", resid_path, config=config)

    if null_output_path is not None and null_output_path.exists():
        data = np.load(null_output_path)
        null_rs = data.get("null_r")
        if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subject):
            perm_path = plots_dir / "loso_null_distribution.png"
            plot_permutation_null(null_rs, r_subject, p_val, perm_path, config=config)

    logger.info("Saved decoding results to %s", results_dir)
    return results_dir


def run_time_generalization(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config,
    n_perm: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
) -> None:
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
        logger.warning("Time-generalization decoding failed: %s", exc)
        return

    tg_path = results_dir / "time_generalization_regression.npz"
    tg_npz = np.load(tg_path) if tg_path.exists() else None
    null_mat = None
    if tg_npz is not None and "null_r" in tg_npz:
        null_mat = tg_npz["null_r"]

    plot_time_generalization_with_null(
        tg_matrix=tg_r,
        null_matrix=null_mat,
        window_centers=window_centers,
        save_path=results_dir / "time_generalization_r.png",
        metric="r",
        config=config,
    )
    plot_time_generalization_with_null(
        tg_matrix=tg_r2,
        null_matrix=null_mat,
        window_centers=window_centers,
        save_path=results_dir / "time_generalization_r2.png",
        metric="r2",
        config=config,
    )
    logger.info("Saved time-generalization outputs to %s", results_dir)

