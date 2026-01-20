"""
Machine Learning Orchestration (Canonical)
=============================================

Single source of truth for LOSO ML orchestration.
This module handles epoch-based and matrix-based cross-validation strategies:

- nested_loso_predictions: Epoch-based nested LOSO with channel harmonization
- within_subject_kfold_predictions: Within-subject k-fold CV
- loso_baseline_predictions: Baseline (mean) predictions
- nested_loso_predictions_from_matrix: Matrix-based nested LOSO
"""

from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from eeg_pipeline.analysis.machine_learning.cv import (
    compute_metrics,
    create_loso_folds,
    create_within_subject_folds,
    create_block_aware_inner_cv,
    create_scoring_dict,
    get_inner_cv_splits,
    get_min_channels_required,
    create_best_params_record,
    determine_inner_n_jobs,
    execute_folds_parallel,
    grid_search_with_warning_logging,
    _save_best_params,
    create_inner_cv,
    _fit_with_inner_cv,
    _fit_default_pipeline,
    _predict_and_log,
    nested_loso_predictions_matrix,
    compute_subject_level_r,
    safe_pearsonr,
)
from eeg_pipeline.analysis.machine_learning.pipelines import (
    create_elasticnet_pipeline,
    create_ridge_pipeline,
    create_rf_pipeline,
    build_elasticnet_param_grid,
    build_ridge_param_grid,
    build_rf_param_grid,
)
from eeg_pipeline.utils.analysis.tfr import (
    find_common_channels_train_test,
)
from eeg_pipeline.utils.data.machine_learning import (
    filter_finite_targets,
    extract_epoch_data_block,
    prepare_trial_records_from_epochs,
)
from eeg_pipeline.utils.data.machine_learning import load_epochs_with_targets
from eeg_pipeline.utils.data.machine_learning import load_active_matrix
from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.infra.tsv import read_tsv, write_tsv, write_parquet, write_stats_table
from eeg_pipeline.infra.paths import ensure_dir, load_events_df
from eeg_pipeline.infra.machine_learning import (
    export_predictions,
    export_indices,
    prepare_best_params_path,
)
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.analysis.machine_learning.time_generalization import time_generalization_regression

logger = get_logger(__name__)


###################################################################
# Cross-Validation Strategies
###################################################################


def nested_loso_predictions(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    results_dir: Optional[Path] = None,
    n_perm: int = 0,
    inner_splits: Optional[int] = None,
    n_jobs: int = -1,
    outer_jobs: int = 1,
    seed: Optional[int] = None,
    config_dict: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, pd.DataFrame]:
    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)
    trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y = prepare_trial_records_from_epochs(tuples)

    config_local = load_config()
    if inner_splits is None:
        inner_splits = config_local.get("machine_learning.cv.inner_splits", 5)
    if seed is None:
        seed = config_local.get("project.random_state", 42)
    if config_dict is None:
        config_dict = config_local
    min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
    if len(np.unique(groups_arr)) < min_subjects_for_loso:
        raise RuntimeError(f"Need at least {min_subjects_for_loso} subjects for LOSO.")

    min_channels_required = get_min_channels_required(config_local)

    pipe = create_elasticnet_pipeline(seed=seed)
    param_grid = build_elasticnet_param_grid(config_local)
    pipe.named_steps["regressor"].param_grid = param_grid

    folds = create_loso_folds(trial_records, groups_arr)

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray, y_target: np.ndarray):

        train_subjects = list({trial_records[i][0] for i in train_idx if trial_records[i][0] is not None})
        test_subject = trial_records[int(test_idx[0])][0]
        
        common_chs_fold = find_common_channels_train_test(train_subjects, test_subject, subj_to_epochs)
        
        if not common_chs_fold:
            raise RuntimeError(
                f"Fold {fold}: No common EEG channels across training subjects. "
                "Channel harmonization must be resolved before running LOSO."
            )

        if len(common_chs_fold) < min_channels_required:
            raise RuntimeError(
                f"Fold {fold}: Common channels with test subject too few "
                f"(n={len(common_chs_fold)} < {min_channels_required}). "
                "Channel harmonization must be resolved before running LOSO."
            )

        logger.info(f"Fold {fold}: Using {len(common_chs_fold)} EEG channels (train∩test) across {len(train_subjects)} training subjects.")

        subjects_in_fold = list({trial_records[i][0] for i in np.concatenate([train_idx, test_idx])})
        aligned_epochs = {s: subj_to_epochs[s].copy().pick(common_chs_fold) for s in subjects_in_fold}

        train_idx_f, y_train = filter_finite_targets(train_idx, y_target)
        test_idx_f, y_test = filter_finite_targets(test_idx, y_target)
        
        if len(train_idx_f) == 0 or len(test_idx_f) == 0:
            logger.error(
                f"Fold {fold}: Empty train ({len(train_idx_f)}) or test ({len(test_idx_f)}) set "
                f"after filtering finite targets. Skipping fold."
            )
            return {
                "fold": fold,
                "y_true": y_test.tolist() if len(test_idx_f) > 0 else [],
                "y_pred": np.full(len(test_idx_f), np.nan, dtype=float).tolist() if len(test_idx_f) > 0 else [],
                "groups": groups_arr[test_idx_f].tolist() if len(test_idx_f) > 0 else [],
                "test_idx": test_idx_f.tolist(),
                "best_params_rec": None,
            }

        X_train = extract_epoch_data_block(train_idx_f, trial_records, aligned_epochs)
        X_test = extract_epoch_data_block(test_idx_f, trial_records, aligned_epochs)
        
        if X_train.ndim == 3:
            X_train = X_train.reshape(len(X_train), -1)
        if X_test.ndim == 3:
            X_test = X_test.reshape(len(X_test), -1)
        
        train_groups = groups_arr[train_idx_f]

        inner_n_jobs = determine_inner_n_jobs(outer_jobs, n_jobs)
        best_estimator, cv_results, gs = _fit_with_inner_cv(
            pipe, X_train, y_train, train_groups, config_dict, fold, inner_n_jobs, logger, seed + fold, inner_splits
        )

        y_pred = _predict_and_log(best_estimator, X_test, y_test, fold, logger)

        best_params_rec = create_best_params_record("ElasticNet", fold, cv_results, gs) if cv_results is not None and gs is not None else None

        return {
            "fold": fold,
            "y_true": y_test.tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
            "groups": groups_arr[test_idx_f].tolist(),
            "test_idx": test_idx_f.tolist(),
            "best_params_rec": best_params_rec,
        }

    results = execute_folds_parallel(folds, lambda f, tr, te: _run_fold(f, tr, te, y_all_arr), outer_jobs)

    results = sorted(results, key=lambda r: r["fold"])
    y_true, y_pred, groups_ordered, test_indices_order, fold_ids = [], [], [], [], []
    best_param_records = []
    for rec in results:
        y_true.extend(rec["y_true"])
        y_pred.extend(rec["y_pred"])
        groups_ordered.extend(rec["groups"])
        test_indices_order.extend(rec["test_idx"])
        fold_ids.extend([rec["fold"]] * len(rec["test_idx"]))
        if rec.get("best_params_rec") is not None:
            best_param_records.append(rec["best_params_rec"])

    if results_dir is not None and len(best_param_records) > 0:
        results_dir.mkdir(parents=True, exist_ok=True)
        best_params_path = config_dict.get("paths", {}).get("best_params", {}).get("elasticnet_loso", "machine_learning/best_params/elasticnet_loso.jsonl")
        path = results_dir / best_params_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_best_params(best_param_records, path)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pooled, per_subj = compute_metrics(y_true, y_pred, np.asarray(groups_ordered), config_dict)
    logger.info(f"Model 1 (ElasticNet) pooled: r={pooled['pearson_r']:.3f}, R2={pooled['r2']:.3f}, EVS={pooled['explained_variance']:.3f}, avg_r_Fz={pooled['avg_subject_r_fisher_z']:.3f}")

    null_r = []
    null_r2 = []
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        for perm_idx in range(n_perm):
            y_perm = y_all_arr.copy()
            for subject_id in np.unique(groups_arr):
                subject_mask = groups_arr == subject_id
                subject_indices = np.where(subject_mask)[0]
                y_perm[subject_indices] = rng.permutation(y_perm[subject_indices])
            
            folds_perm = create_loso_folds(trial_records, groups_arr)
            results_perm = execute_folds_parallel(folds_perm, lambda f, tr, te: _run_fold(f, tr, te, y_perm), outer_jobs)
            
            y_true_perm, y_pred_perm, groups_perm = [], [], []
            for rec in sorted(results_perm, key=lambda r: r["fold"]):
                y_true_perm.extend(rec["y_true"])
                y_pred_perm.extend(rec["y_pred"])
                groups_perm.extend(rec["groups"])
            
            if y_true_perm and y_pred_perm:
                pooled_perm, _ = compute_metrics(
                    np.asarray(y_true_perm),
                    np.asarray(y_pred_perm),
                    np.asarray(groups_perm),
                    config_dict,
                )
                r_perm = pooled_perm.get("pearson_r", np.nan)
                r2_perm = pooled_perm.get("r2", np.nan)
                if np.isfinite(r_perm):
                    null_r.append(r_perm)
                if np.isfinite(r2_perm):
                    null_r2.append(r2_perm)
        null_r = np.asarray(null_r) if null_r else np.array([])
        null_r2 = np.asarray(null_r2) if null_r2 else np.array([])

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        paths = config_dict.get("paths", {})
        predictions_path = paths.get("predictions", {}).get("elasticnet_loso", "machine_learning/predictions/elasticnet_loso.parquet")
        metrics_path = paths.get("per_subject_metrics", {}).get("elasticnet_loso", "machine_learning/per_subject_metrics/elasticnet_loso.parquet")
        indices_path = paths.get("indices", {}).get("elasticnet_loso", "machine_learning/indices/elasticnet_loso.parquet")
        
        write_parquet(pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "group": np.asarray(groups_ordered),
            "fold": fold_ids,
            "trial_index": test_indices_order,
        }), results_dir / predictions_path)

        write_parquet(pd.DataFrame(per_subj), results_dir / metrics_path)

        idx_df = pd.DataFrame({
            "subject_id": np.asarray(groups_ordered),
            "heldout_subject_id": np.asarray(groups_ordered),
            "fold": fold_ids,
            "trial_index": test_indices_order,
        })
        write_parquet(idx_df, results_dir / indices_path)

        if n_perm > 0 and len(null_r) > 0:
            null_path = results_dir / "machine_learning" / "null_distribution.npz"
            null_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(null_path, null_r=np.asarray(null_r), null_r2=np.asarray(null_r2))

    return y_true, y_pred, pooled, per_subj


def _fit_within_subject_fold(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    blocks_train: Optional[np.ndarray],
    config_dict: Optional[dict],
    fold: int,
    subject_id: str,
    random_state: int,
    n_jobs: int,
    logger: logging.Logger,
) -> Pipeline:
    config_local = load_config()
    
    if blocks_train is not None:
        n_unique_blocks = len(np.unique(blocks_train))
        default_splits = config_local.get("machine_learning.cv.default_n_splits", 5) if config_local else 5
        n_splits_inner = get_inner_cv_splits(n_unique_blocks, default_splits)
        n_splits_inner = max(2, min(n_unique_blocks, n_splits_inner))
        
        if n_splits_inner < 2:
            return _fit_default_pipeline(pipe, X_train, y_train, fold)
        
        inner_cv_splits = create_block_aware_inner_cv(blocks_train, n_splits_inner, random_state, fold, subject_id)
        
        if inner_cv_splits is not None and len(inner_cv_splits) >= 2:
            scoring = create_scoring_dict()
            refit_metric = 'r'
            
            param_grid = getattr(pipe.named_steps.get("regressor"), "param_grid", None) or {}
            pipe_seeded = clone(pipe)
            regressor_step = pipe_seeded.named_steps.get("regressor")
            if regressor_step is not None and hasattr(regressor_step, "regressor") and hasattr(regressor_step.regressor, "random_state"):
                regressor_step.regressor.random_state = random_state
            gs = GridSearchCV(
                estimator=pipe_seeded,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv_splits,
                n_jobs=n_jobs,
                refit=refit_metric,
            )
            gs = grid_search_with_warning_logging(
                gs, X_train, y_train,
                fold_info=f"within-subject fold {fold} (subject {subject_id})",
                log=logger,
                groups=blocks_train
            )
            return gs.best_estimator_
    
    return _fit_default_pipeline(pipe, X_train, y_train, fold, random_state)


def within_subject_kfold_predictions(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    results_dir: Optional[Path] = None,
    n_splits: Optional[int] = None,
    n_jobs: int = -1,
    seed: Optional[int] = None,
    config_dict: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, pd.DataFrame]:
    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)
    trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y = prepare_trial_records_from_epochs(tuples)

    config_local = load_config()
    if n_splits is None:
        n_splits = config_local.get("machine_learning.cv.default_n_splits", 5)
    if seed is None:
        seed = config_local.get("project.random_state", 42)
    if config_dict is None:
        config_dict = config_local
    min_channels_required = get_min_channels_required(config_local)

    pipe = create_elasticnet_pipeline(seed=seed)
    param_grid = build_elasticnet_param_grid(config_local)
    pipe.named_steps["regressor"].param_grid = param_grid

    all_results = []
    for subject_id in np.unique(groups_arr):
        subject_indices = np.where(groups_arr == subject_id)[0]
        if len(subject_indices) < n_splits:
            logger.warning(f"Subject {subject_id}: Insufficient trials ({len(subject_indices)}) for {n_splits}-fold CV. Skipping.")
            continue

        epochs_subj = subj_to_epochs[subject_id]
        eeg_chs = [ch for ch in epochs_subj.info["ch_names"] if epochs_subj.get_channel_types(picks=[ch])[0] == "eeg"]
        if len(eeg_chs) < min_channels_required:
            logger.warning(f"Subject {subject_id}: Insufficient channels ({len(eeg_chs)} < {min_channels_required}). Skipping.")
            continue

        epochs_subj = epochs_subj.copy().pick(eeg_chs)
        subject_indices_original = subject_indices.copy()
        X_subj = extract_epoch_data_block(subject_indices, trial_records, {subject_id: epochs_subj})
        y_subj = y_all_arr[subject_indices]
        
        finite_mask = np.isfinite(y_subj)
        if not np.all(finite_mask):
            logger.warning(
                f"Subject {subject_id}: Removing {np.sum(~finite_mask)} non-finite targets "
                f"(out of {len(y_subj)} total trials)."
            )
            subject_indices = subject_indices[finite_mask]
            X_subj = X_subj[finite_mask]
            y_subj = y_subj[finite_mask]
        
        if len(y_subj) < n_splits:
            logger.warning(
                f"Subject {subject_id}: Insufficient trials ({len(y_subj)}) after filtering "
                f"non-finite targets for {n_splits}-fold CV. Skipping."
            )
            continue
        
        blocks_subj = None
        events = load_events_df(subject_id, task, config=config_local)
        if events is not None and "block" in events.columns:
            blocks_all_epochs = pd.to_numeric(events["block"], errors="coerce").to_numpy()
            epoch_indices = np.array([trial_records[idx][1] for idx in subject_indices_original], dtype=int)
            if np.any(epoch_indices < 0) or np.any(epoch_indices >= len(blocks_all_epochs)):
                logger.warning(
                    "Subject %s: epoch_indices out of bounds for events.tsv (n=%d); cannot use blocks.",
                    subject_id,
                    len(blocks_all_epochs),
                )
            else:
                blocks_subj = blocks_all_epochs[epoch_indices]
                if blocks_subj is not None:
                    blocks_subj = blocks_subj[finite_mask]
                
                if blocks_subj is not None:
                    if np.all(np.isnan(blocks_subj)):
                        blocks_subj = None
                    elif len(blocks_subj) != len(y_subj):
                        logger.warning(f"Subject {subject_id}: Block array length ({len(blocks_subj)}) != y_subj length ({len(y_subj)}). Setting blocks_subj=None.")
                        blocks_subj = None
                    elif not np.all(np.isfinite(blocks_subj)):
                        nan_count = np.sum(~np.isfinite(blocks_subj))
                        logger.error(
                            f"Subject {subject_id}: {nan_count} trials have missing block labels. "
                            "Cannot use block-aware CV with incomplete block information. "
                            "Dropping trials with missing blocks."
                        )
                        finite_blocks_mask = np.isfinite(blocks_subj)
                        subject_indices = subject_indices[finite_blocks_mask]
                        X_subj = X_subj[finite_blocks_mask]
                        y_subj = y_subj[finite_blocks_mask]
                        blocks_subj = blocks_subj[finite_blocks_mask]

                        if len(y_subj) < n_splits:
                            logger.warning(
                                f"Subject {subject_id}: Insufficient trials ({len(y_subj)}) after "
                                f"removing missing blocks for {n_splits}-fold CV. Skipping."
                            )
                            continue

        if blocks_subj is None:
            logger.error(
                f"Subject {subject_id}: Block/run labels unavailable. "
                f"Within-subject CV requires block/run identifiers to prevent temporal data leakage. "
                "Ensure a clean events.tsv exists with one of: block, run_id, run, session."
            )
            raise ValueError(f"Block/run labels required for within-subject CV (subject {subject_id})")

        groups_subj = groups_arr[subject_indices]
        # NOTE: These folds are used for epoch-based machine learning; feature-extraction CV hygiene
        # (e.g., IAF re-estimation) is not applied here because we do not recompute features in-fold.
        folds = create_within_subject_folds(
            groups_subj,
            blocks_subj,
            n_splits,
            seed,
            config=config_local,
            epochs=None,
            apply_hygiene=False,
        )

        for fold_counter, train_idx_rel, test_idx_rel, subject_name, _fold_params in folds:
            X_train = X_subj[train_idx_rel]
            X_test = X_subj[test_idx_rel]
            
            if X_train.ndim == 3:
                X_train = X_train.reshape(len(X_train), -1)
            if X_test.ndim == 3:
                X_test = X_test.reshape(len(X_test), -1)
            
            y_train = y_subj[train_idx_rel]
            y_test = y_subj[test_idx_rel]
            blocks_train = blocks_subj[train_idx_rel] if blocks_subj is not None else None

            best_estimator = _fit_within_subject_fold(
                pipe=pipe,
                X_train=X_train,
                y_train=y_train,
                blocks_train=blocks_train,
                config_dict=config_dict,
                fold=fold_counter,
                subject_id=subject_id,
                random_state=seed + fold_counter,
                n_jobs=n_jobs,
                logger=logger,
            )

            y_pred = best_estimator.predict(X_test)
            r = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else np.nan
            logger.info(f"Subject {subject_id}, fold {fold_counter}: test r={r:.3f}, n={len(y_test)}")

            all_results.append({
                "subject_id": subject_id,
                "fold": fold_counter,
                "y_true": y_test.tolist(),
                "y_pred": np.asarray(y_pred).tolist(),
            })

    if not all_results:
        raise RuntimeError("No valid predictions generated.")

    y_true_all = []
    y_pred_all = []
    groups_all = []
    for res in all_results:
        y_true_all.extend(res["y_true"])
        y_pred_all.extend(res["y_pred"])
        groups_all.extend([res["subject_id"]] * len(res["y_true"]))

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    pooled, per_subj = compute_metrics(y_true_all, y_pred_all, np.asarray(groups_all), config_dict)
    logger.info(f"Within-subject KFold pooled: r={pooled['pearson_r']:.3f}, R2={pooled['r2']:.3f}, EVS={pooled['explained_variance']:.3f}")

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        paths = config_dict.get("paths", {})
        predictions_path = paths.get("predictions", {}).get("within_subject_kfold", "machine_learning/predictions/within_subject_kfold.parquet")
        metrics_path = paths.get("per_subject_metrics", {}).get("within_subject_kfold", "machine_learning/per_subject_metrics/within_subject_kfold.parquet")
        
        write_parquet(pd.DataFrame({
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "group": groups_all,
        }), results_dir / predictions_path)

        write_parquet(pd.DataFrame(per_subj), results_dir / metrics_path)

    return y_true_all, y_pred_all, pooled, per_subj


def loso_baseline_predictions(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    results_dir: Optional[Path] = None,
    config_dict: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, pd.DataFrame]:
    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)
    trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y = prepare_trial_records_from_epochs(tuples)

    config_local = load_config()
    min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
    if len(np.unique(groups_arr)) < min_subjects_for_loso:
        raise RuntimeError(f"Need at least {min_subjects_for_loso} subjects for LOSO.")

    folds = create_loso_folds(trial_records, groups_arr)

    y_true_all = []
    y_pred_all = []
    groups_all = []
    test_indices_all = []
    fold_ids_all = []

    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        y_train_mean = np.nanmean(y_all_arr[train_idx])
        y_test = y_all_arr[test_idx]
        y_pred = np.full_like(y_test, y_train_mean)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        groups_all.extend(groups_arr[test_idx].tolist())
        test_indices_all.extend(test_idx.tolist())
        fold_ids_all.extend([fold] * len(test_idx))

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    pooled, per_subj = compute_metrics(y_true_all, y_pred_all, np.asarray(groups_all), config_dict)
    logger.info(f"Baseline (mean) pooled: r={pooled['pearson_r']:.3f}, R2={pooled['r2']:.3f}, EVS={pooled['explained_variance']:.3f}")

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        paths = config_dict.get("paths", {})
        predictions_path = paths.get("predictions", {}).get("baseline_loso", "machine_learning/predictions/baseline_loso.parquet")
        metrics_path = paths.get("per_subject_metrics", {}).get("baseline_loso", "machine_learning/per_subject_metrics/baseline_loso.parquet")
        indices_path = paths.get("indices", {}).get("baseline_loso", "machine_learning/indices/baseline_loso.parquet")
        
        write_parquet(pd.DataFrame({
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "group": groups_all,
            "fold": fold_ids_all,
            "trial_index": test_indices_all,
        }), results_dir / predictions_path)

        write_parquet(pd.DataFrame(per_subj), results_dir / metrics_path)

        idx_df = pd.DataFrame({
            "subject_id": groups_all,
            "heldout_subject_id": groups_all,
            "fold": fold_ids_all,
            "trial_index": test_indices_all,
        })
        write_parquet(idx_df, results_dir / indices_path)

    return y_true_all, y_pred_all, pooled, per_subj


###################################################################
# Matrix-Based LOSO (Feature-Matrix Interface)
###################################################################


def nested_loso_predictions_from_matrix(
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
    config: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """
    Nested leave-one-subject-out cross-validation with hyperparameter tuning.
    
    This is the matrix-based interface that operates on pre-loaded feature matrices.

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
    return nested_loso_predictions_matrix(
        X=X,
        y=y,
        groups=groups,
        blocks=blocks,
        pipe=pipe,
        param_grid=param_grid,
        inner_cv_splits=inner_cv_splits,
        n_jobs=n_jobs,
        seed=seed,
        best_params_log_path=best_params_log_path,
        model_name=model_name,
        outer_n_jobs=outer_n_jobs,
        null_n_perm=null_n_perm,
        null_output_path=null_output_path,
        config=config,
    )


###################################################################
# Pipeline-Level Runners
###################################################################


def run_regression_ml(
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
    model: str = "elasticnet",
) -> Path:
    """Run LOSO regression machine learning on active features.
    
    Parameters
    ----------
    model : str
        Model family: 'elasticnet' (default), 'ridge', or 'rf' (RandomForest).
    """
    X, y, groups, _feature_names, meta = load_active_matrix(subjects, task, deriv_root, config, logger)
    blocks = None
    if meta is not None and hasattr(meta, "columns") and "block" in meta.columns:
        blocks = pd.to_numeric(meta["block"], errors="coerce").to_numpy()

    results_dir = results_root / "regression"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    if model == "ridge":
        pipe = create_ridge_pipeline(seed=rng_seed, config=config)
        param_grid = build_ridge_param_grid(config)
        model_name = "ridge"
    elif model == "rf":
        pipe = create_rf_pipeline(seed=rng_seed, config=config)
        param_grid = build_rf_param_grid(config)
        model_name = "rf"
    else:
        pipe = create_elasticnet_pipeline(seed=rng_seed, config=config)
        param_grid = build_elasticnet_param_grid(config)
        model_name = "elasticnet"
    
    best_params_path = prepare_best_params_path(results_dir / f"best_params_{model_name}.jsonl", mode="truncate")
    null_path = results_dir / f"loso_null_{model_name}.npz" if n_perm > 0 else None

    logger.info(f"Running regression with model={model_name}")

    y_true, y_pred, groups_ordered, test_indices, fold_ids = nested_loso_predictions_matrix(
        X=X,
        y=y,
        groups=groups,
        blocks=blocks,
        pipe=pipe,
        param_grid=param_grid,
        inner_cv_splits=inner_splits,
        n_jobs=-1,
        seed=rng_seed,
        best_params_log_path=best_params_path,
        model_name=model_name,
        outer_n_jobs=outer_jobs,
        null_n_perm=n_perm,
        null_output_path=null_path,
        config=config,
    )

    pred_path = results_dir / "loso_predictions.tsv"
    pred_df = export_predictions(
        y_true,
        y_pred,
        groups_ordered,
        test_indices,
        fold_ids,
        model_name,
        meta.reset_index(drop=True),
        pred_path,
    )
    export_indices(
        groups_ordered,
        test_indices,
        fold_ids,
        meta.reset_index(drop=True),
        results_dir / "loso_indices.tsv",
        add_heldout_subject_id=True,
    )

    r_subj, _per_subj_r, ci_low, ci_high = compute_subject_level_r(pred_df, config)
    p_val = np.nan

    if null_path and null_path.exists():
        data = np.load(null_path)
        null_rs = data.get("null_r")
        if null_rs is not None and null_rs.size > 0 and np.isfinite(r_subj):
            finite = null_rs[np.isfinite(null_rs)]
            if finite.size > 0:
                p_val = float(((np.abs(finite) >= abs(r_subj)).sum() + 1) / (finite.size + 1))

    try:
        from sklearn.metrics import r2_score

        r2_val = float(r2_score(y_true, y_pred))
    except Exception:
        r2_val = np.nan

    # Compute and export baseline predictions for sanity check
    baseline_metrics = export_baseline_predictions(y_true, groups_ordered, results_dir, task="regression")

    # Compute pooled (trial-level) r for secondary reporting
    pooled_r, _ = safe_pearsonr(y_true, y_pred)

    # Structure metrics with subject-level as PRIMARY (statistical unit for LOSO)
    metrics = {
        "model": model_name,
        "n_subjects": len(np.unique(groups_ordered)),
        "n_trials": len(y_true),
        "subject_level": {
            "r": r_subj,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_val,
            "n_subjects": len(_per_subj_r),
        },
        "pooled_trials": {
            "r": float(pooled_r) if np.isfinite(pooled_r) else None,
            "r2": r2_val,
        },
        **baseline_metrics,
    }

    # Write reproducibility info
    write_reproducibility_info(results_dir, subjects, config, rng_seed)

    metrics_path = results_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Saved machine learning results to {results_dir}")
    return results_dir


def run_within_subject_regression_ml(
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
    model: str = "elasticnet",
) -> Path:
    """Run within-subject (block-aware) regression machine learning on active features.
    
    Parameters
    ----------
    model : str
        Model family: 'elasticnet' (default), 'ridge', or 'rf' (RandomForest).
    """
    X, y, groups, _feature_names, meta = load_active_matrix(subjects, task, deriv_root, config, logger)

    if meta is None or not hasattr(meta, "columns"):
        raise ValueError("Within-subject machine learning requires trial metadata with block/run labels.")

    if "block" not in meta.columns:
        raise ValueError(
            "Within-subject machine learning requires block/run labels to prevent temporal data leakage. "
            "Ensure your events.tsv contains one of: block, run_id, run, session."
        )

    blocks_all = pd.to_numeric(meta["block"], errors="coerce").to_numpy()

    finite_mask = np.isfinite(y)
    if not np.all(finite_mask):
        X = X[finite_mask]
        y = y[finite_mask]
        groups = groups[finite_mask]
        meta = meta.loc[finite_mask].reset_index(drop=True)
        blocks_all = blocks_all[finite_mask]

    finite_blocks = np.isfinite(blocks_all)
    if not np.all(finite_blocks):
        dropped = int((~finite_blocks).sum())
        logger.warning(f"Dropping {dropped} trials with missing block labels for within-subject machine learning.")
        X = X[finite_blocks]
        y = y[finite_blocks]
        groups = groups[finite_blocks]
        meta = meta.loc[finite_blocks].reset_index(drop=True)
        blocks_all = blocks_all[finite_blocks]

    results_dir = results_root / "within_subject_regression"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)
    
    model_name = model if model else "elasticnet"

    folds = create_within_subject_folds(
        groups=groups,
        blocks_all=blocks_all,
        inner_cv_splits=inner_splits,
        seed=rng_seed,
        config=config,
        epochs=None,
        apply_hygiene=False,
    )
    if not folds:
        raise ValueError(
            "No within-subject folds could be created. "
            "Ensure each selected subject has at least 2 unique block/run labels."
        )

    fold_records = []
    for fold_counter, train_idx, test_idx, subject_id, _fold_params in folds:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        blocks_train = blocks_all[train_idx] if blocks_all is not None else None

        if model == "ridge":
            pipe = create_ridge_pipeline(seed=rng_seed + int(fold_counter), config=config)
            param_grid = build_ridge_param_grid(config)
        elif model == "rf":
            pipe = create_rf_pipeline(seed=rng_seed + int(fold_counter), config=config)
            param_grid = build_rf_param_grid(config)
        else:
            pipe = create_elasticnet_pipeline(seed=rng_seed + int(fold_counter), config=config)
            param_grid = build_elasticnet_param_grid(config)
        
        if "regressor" in pipe.named_steps:
            pipe.named_steps["regressor"].param_grid = param_grid

        best_estimator = _fit_within_subject_fold(
            pipe=pipe,
            X_train=X_train,
            y_train=y_train,
            blocks_train=blocks_train,
            config_dict=config,
            fold=int(fold_counter),
            subject_id=str(subject_id),
            random_state=rng_seed + int(fold_counter),
            n_jobs=-1,
            logger=logger,
        )
        y_pred = best_estimator.predict(X_test)

        fold_records.append(
            {
                "fold": int(fold_counter),
                "y_true": y_test,
                "y_pred": y_pred,
                "subject_id": [str(subject_id)] * len(y_test),
                "test_idx": np.asarray(test_idx, dtype=int),
            }
        )

    fold_records = sorted(fold_records, key=lambda r: r["fold"])
    y_true_all = np.concatenate([np.asarray(r["y_true"]) for r in fold_records])
    y_pred_all = np.concatenate([np.asarray(r["y_pred"]) for r in fold_records])
    groups_ordered: List[str] = []
    test_indices: List[int] = []
    fold_ids: List[int] = []
    for rec in fold_records:
        n = len(rec["y_true"])
        groups_ordered.extend(rec["subject_id"])
        test_indices.extend(rec["test_idx"].tolist())
        fold_ids.extend([rec["fold"]] * n)

    pred_path = results_dir / "cv_predictions.tsv"
    pred_df = export_predictions(
        y_true_all,
        y_pred_all,
        groups_ordered,
        test_indices,
        fold_ids,
        model_name,
        meta.reset_index(drop=True),
        pred_path,
    )
    export_indices(
        groups_ordered,
        test_indices,
        fold_ids,
        meta.reset_index(drop=True),
        results_dir / "cv_indices.tsv",
    )

    r_subj, _per_subj_r, ci_low, ci_high = compute_subject_level_r(pred_df, config)
    try:
        from sklearn.metrics import r2_score

        r2_val = float(r2_score(y_true_all, y_pred_all))
    except Exception:
        r2_val = np.nan

    p_value = np.nan
    null_rs = []
    
    if n_perm > 0:
        logger.info(f"Running {n_perm} block-aware permutations for within-subject inference...")
        rng = np.random.default_rng(rng_seed)
        
        for perm_idx in range(n_perm):
            y_perm = y.copy()
            for subj in np.unique(groups):
                subj_mask = groups == subj
                subj_blocks = blocks_all[subj_mask]
                subj_y = y_perm[subj_mask]
                
                for block_id in np.unique(subj_blocks):
                    block_mask = subj_blocks == block_id
                    block_indices = np.where(block_mask)[0]
                    subj_y[block_indices] = rng.permutation(subj_y[block_indices])
                
                y_perm[subj_mask] = subj_y
            
            perm_folds = create_within_subject_folds(
                groups=groups,
                blocks_all=blocks_all,
                inner_cv_splits=inner_splits,
                seed=rng_seed + perm_idx + 1,
                config=config,
                epochs=None,
                apply_hygiene=False,
            )
            
            perm_y_true = []
            perm_y_pred = []
            
            for fold_counter, train_idx, test_idx, subject_id, _ in perm_folds:
                X_train_p = X[train_idx]
                X_test_p = X[test_idx]
                y_train_p = y_perm[train_idx]
                y_test_p = y_perm[test_idx]
                
                if model == "ridge":
                    pipe_p = create_ridge_pipeline(seed=rng_seed + perm_idx + fold_counter, config=config)
                elif model == "rf":
                    pipe_p = create_rf_pipeline(seed=rng_seed + perm_idx + fold_counter, config=config)
                else:
                    pipe_p = create_elasticnet_pipeline(seed=rng_seed + perm_idx + fold_counter, config=config)
                
                try:
                    pipe_p.fit(X_train_p, y_train_p)
                    y_pred_p = pipe_p.predict(X_test_p)
                    perm_y_true.extend(y_test_p)
                    perm_y_pred.extend(y_pred_p)
                except Exception:
                    continue
            
            if len(perm_y_true) >= 3:
                r_perm, _ = safe_pearsonr(np.asarray(perm_y_true), np.asarray(perm_y_pred))
                if np.isfinite(r_perm):
                    null_rs.append(r_perm)
            
            if (perm_idx + 1) % 10 == 0:
                logger.info(f"Permutation {perm_idx + 1}/{n_perm}")
        
        if len(null_rs) > 0 and np.isfinite(r_subj):
            null_rs = np.asarray(null_rs)
            p_value = float(((np.abs(null_rs) >= np.abs(r_subj)).sum() + 1) / (len(null_rs) + 1))
            np.savez(results_dir / "within_subject_null.npz", null_r=null_rs, empirical_r=r_subj)
            logger.info(f"Within-subject permutation p-value: {p_value:.4f} (n_perm={len(null_rs)})")

    # Compute pooled r for secondary reporting
    pooled_r, _ = safe_pearsonr(y_true_all, y_pred_all)
    
    # Structure metrics with subject-level as PRIMARY
    metrics = {
        "model": model_name,
        "cv_scope": "subject",
        "n_subjects": len(np.unique(groups)),
        "n_trials": len(y_true_all),
        "subject_level": {
            "r": r_subj,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
        },
        "pooled_trials": {
            "r": float(pooled_r) if np.isfinite(pooled_r) else None,
            "r2": r2_val,
        },
        "n_perm": n_perm if n_perm > 0 else 0,
    }
    metrics_path = results_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Saved within-subject machine learning results to {results_dir}")
    return results_dir


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
    """Run time-generalization machine learning analysis."""
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

    logger.info(f"Saved time-generalization outputs to {results_dir}")


###################################################################
# Classification Pipeline Runner
###################################################################


def run_classification_ml(
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
    """Run LOSO classification machine learning for pain vs no-pain.
    
    Uses nested CV with hyperparameter tuning in inner loop.
    Reports AUC, balanced accuracy, and calibrated metrics.
    """
    from eeg_pipeline.analysis.machine_learning.classification import (
        nested_loso_classification,
        ClassificationResult,
    )
    
    X, y, groups, feature_names, meta = load_active_matrix(subjects, task, deriv_root, config, logger)
    
    # Create binary labels from pain column if available
    if meta is not None and "pain" in meta.columns:
        y_binary = (pd.to_numeric(meta["pain"], errors="coerce") > 0).astype(int).to_numpy()
    else:
        # Fall back to median split of continuous target
        y_binary = (y > np.nanmedian(y)).astype(int)
        logger.warning("No 'pain' column found; using median split of target for classification.")

    results_dir = results_root / "classification"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    model_type = config.get("machine_learning.classification.model", "svm")
    
    result, best_params_df = nested_loso_classification(
        X=X,
        y=y_binary,
        groups=groups,
        model=model_type,
        inner_splits=inner_splits,
        seed=rng_seed,
        config=config,
        logger=logger,
    )

    # Export predictions
    pred_df = pd.DataFrame({
        "subject_id": groups,
        "y_true": result.y_true,
        "y_pred": result.y_pred,
        "y_prob": result.y_prob,
    })
    pred_path = results_dir / "loso_predictions.tsv"
    pred_df.to_csv(pred_path, sep="\t", index=False)

    # Export best params
    if not best_params_df.empty:
        best_params_df.to_csv(results_dir / f"best_params_{model_type}.tsv", sep="\t", index=False)

    # Compute calibration metrics (Brier score, reliability)
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    
    brier = float(brier_score_loss(result.y_true, result.y_prob))
    
    # Calibration curve (reliability diagram data)
    try:
        prob_true, prob_pred = calibration_curve(result.y_true, result.y_prob, n_bins=10, strategy="uniform")
        calibration_data = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "n_bins": 10,
        }
        # Expected calibration error (ECE)
        bin_counts = np.histogram(result.y_prob, bins=10, range=(0, 1))[0]
        bin_weights = bin_counts / len(result.y_prob)
        ece = float(np.sum(bin_weights[:len(prob_true)] * np.abs(prob_true - prob_pred)))
    except Exception:
        calibration_data = {}
        ece = np.nan
    
    # Compute and save metrics
    metrics = {
        "auc": result.auc,
        "balanced_accuracy": result.balanced_accuracy,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "model": model_type,
        "n_subjects": len(np.unique(groups)),
        "n_samples": len(y_binary),
        "class_balance": float(y_binary.mean()),
    }
    
    # Save calibration data separately
    if calibration_data:
        with open(results_dir / "calibration_data.json", "w") as f:
            json.dump(calibration_data, f, indent=2)

    # Add permutation p-value if available
    if n_perm > 0:
        logger.info(f"Running {n_perm} permutations for classification...")
        null_aucs = _run_classification_permutations(
            X, y_binary, groups, model_type, inner_splits, rng_seed, n_perm, config, logger
        )
        if null_aucs is not None and len(null_aucs) > 0:
            p_val = float(((null_aucs >= result.auc).sum() + 1) / (len(null_aucs) + 1))
            metrics["p_value"] = p_val
            np.savez(results_dir / f"loso_null_{model_type}.npz", null_auc=null_aucs)

    # Write reproducibility info
    write_reproducibility_info(results_dir, subjects, config, rng_seed)

    metrics_path = results_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Classification: AUC={result.auc:.3f}, Balanced Acc={result.balanced_accuracy:.3f}")
    logger.info(f"Saved classification results to {results_dir}")
    return results_dir


def _run_classification_permutations(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    inner_splits: int,
    seed: int,
    n_perm: int,
    config: Any,
    logger: logging.Logger,
) -> Optional[np.ndarray]:
    """Run permutation test for classification."""
    from eeg_pipeline.analysis.machine_learning.classification import nested_loso_classification
    
    rng = np.random.default_rng(seed)
    null_aucs = []
    
    for i in range(n_perm):
        y_perm = y.copy()
        # Permute within each subject to respect block structure
        for subj in np.unique(groups):
            mask = groups == subj
            y_perm[mask] = rng.permutation(y_perm[mask])
        
        try:
            result, _ = nested_loso_classification(
                X=X,
                y=y_perm,
                groups=groups,
                model=model,
                inner_splits=inner_splits,
                seed=seed + i + 1,
                config=config,
                logger=None,
            )
            null_aucs.append(result.auc)
        except Exception:
            continue
        
        if (i + 1) % 10 == 0:
            logger.info(f"Permutation {i + 1}/{n_perm}")
    
    return np.array(null_aucs) if null_aucs else None


###################################################################
# Reproducibility & Baseline Outputs
###################################################################


def write_reproducibility_info(
    results_dir: Path,
    subjects: List[str],
    config: Any,
    rng_seed: int,
) -> Path:
    """Write reproducibility information for ML results.
    
    Includes: config snapshot, data signature, sklearn version, RNG seed.
    """
    import sklearn
    import hashlib
    
    # Data signature: subjects list + hash
    subjects_str = ",".join(sorted(subjects))
    data_hash = hashlib.sha256(subjects_str.encode()).hexdigest()[:16]
    
    repro_info = {
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rng_seed": rng_seed,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "data_signature": data_hash,
        "config_snapshot": {
            "machine_learning": config.get("machine_learning", {}) if hasattr(config, "get") else {},
        },
    }
    
    repro_path = results_dir / "reproducibility_info.json"
    with open(repro_path, "w") as f:
        json.dump(repro_info, f, indent=2, default=str)
    
    return repro_path


def compute_baseline_predictions(
    y: np.ndarray,
    groups: np.ndarray,
    task: str = "regression",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute baseline (null) model predictions for sanity checks.
    
    Regression: mean predictor (leave-one-subject-out mean)
    Classification: majority class predictor
    
    Returns:
        y_pred_baseline: Baseline predictions
        baseline_metrics: Baseline model metrics
    """
    from sklearn.model_selection import LeaveOneGroupOut
    
    y_pred = np.zeros_like(y, dtype=float)
    logo = LeaveOneGroupOut()
    
    for train_idx, test_idx in logo.split(y, y, groups):
        if task == "regression":
            y_pred[test_idx] = np.nanmean(y[train_idx])
        else:
            # Majority class
            classes, counts = np.unique(y[train_idx], return_counts=True)
            y_pred[test_idx] = classes[np.argmax(counts)]
    
    if task == "regression":
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        metrics = {"baseline_r2": r2, "baseline_mae": mae}
    else:
        from sklearn.metrics import balanced_accuracy_score, accuracy_score
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        metrics = {"baseline_accuracy": acc, "baseline_balanced_accuracy": bal_acc}
    
    return y_pred, metrics


def export_baseline_predictions(
    y_true: np.ndarray,
    groups: np.ndarray,
    results_dir: Path,
    task: str = "regression",
) -> Dict[str, float]:
    """Compute and export baseline model predictions."""
    y_pred_baseline, baseline_metrics = compute_baseline_predictions(y_true, groups, task)
    
    baseline_df = pd.DataFrame({
        "subject_id": groups,
        "y_true": y_true,
        "y_pred_baseline": y_pred_baseline,
    })
    baseline_df.to_csv(results_dir / "baseline_predictions.tsv", sep="\t", index=False)
    
    return baseline_metrics


###################################################################
# Model Comparison Compute Stage
###################################################################


def run_model_comparison_ml(
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
    """Compare multiple model families with identical outer folds.
    
    Compares ElasticNet vs Random Forest vs Ridge/SVR using nested CV.
    All models share the same outer folds for valid comparison.
    
    Outputs:
        model_comparison.tsv: Per-fold metrics for each model
        model_comparison_summary.json: Aggregated comparison statistics
    """
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    
    X, y, groups, feature_names, meta = load_active_matrix(subjects, task, deriv_root, config, logger)
    
    results_dir = results_root / "model_comparison"
    ensure_dir(results_dir)
    
    # Define model pipelines
    models = {
        "elasticnet": {
            "pipe": create_elasticnet_pipeline(seed=rng_seed, config=config),
            "param_grid": build_elasticnet_param_grid(config),
        },
        "ridge": {
            "pipe": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", Ridge(random_state=rng_seed)),
            ]),
            "param_grid": {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "rf": {
            "pipe": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(n_estimators=100, random_state=rng_seed, n_jobs=-1)),
            ]),
            "param_grid": {"rf__max_depth": [5, 10, 20, None]},
        },
    }
    
    # Shared outer CV folds
    from sklearn.model_selection import LeaveOneGroupOut
    outer_cv = LeaveOneGroupOut()
    outer_folds = list(outer_cv.split(X, y, groups))
    
    comparison_records = []
    
    for model_name, model_spec in models.items():
        logger.info(f"Running {model_name}...")
        pipe = model_spec["pipe"]
        param_grid = model_spec["param_grid"]
        
        y_pred = np.zeros(len(y))
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            
            # Inner CV for hyperparameter tuning (group-aware)
            inner_cv = create_inner_cv(groups_train, inner_splits)
            grid = GridSearchCV(clone(pipe), param_grid, cv=inner_cv, scoring="r2", n_jobs=-1)
            grid.fit(X_train, y_train, groups=groups_train)
            
            y_pred[test_idx] = grid.predict(X_test)
            
            # Record fold metrics
            from sklearn.metrics import r2_score, mean_absolute_error
            fold_r2 = r2_score(y_test, y_pred[test_idx])
            fold_mae = mean_absolute_error(y_test, y_pred[test_idx])
            
            comparison_records.append({
                "model": model_name,
                "fold": fold_idx,
                "test_subject": groups[test_idx[0]],
                "r2": fold_r2,
                "mae": fold_mae,
                "best_params": str(grid.best_params_),
            })
        
        # Overall metrics
        from sklearn.metrics import r2_score
        overall_r2 = r2_score(y, y_pred)
        logger.info(f"  {model_name}: R² = {overall_r2:.4f}")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(results_dir / "model_comparison.tsv", sep="\t", index=False)
    
    # Summary statistics
    summary = {}
    for model_name in models.keys():
        model_rows = comparison_df[comparison_df["model"] == model_name]
        summary[model_name] = {
            "mean_r2": float(model_rows["r2"].mean()),
            "std_r2": float(model_rows["r2"].std()),
            "mean_mae": float(model_rows["mae"].mean()),
            "std_mae": float(model_rows["mae"].std()),
        }
    
    with open(results_dir / "model_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    write_reproducibility_info(results_dir, subjects, config, rng_seed)
    logger.info(f"Model comparison saved to {results_dir}")
    
    return results_dir


###################################################################
# Incremental Validity Compute Stage
###################################################################


def run_incremental_validity_ml(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config: Any,
    n_perm: int,
    inner_splits: int,
    rng_seed: int,
    results_root: Path,
    logger: logging.Logger,
) -> Path:
    """Quantify Δperformance when adding EEG features over baseline predictors.
    
    Compares:
    - Baseline: temperature-only predictor
    - Full: temperature + EEG features
    
    All evaluations strictly out-of-fold to avoid leakage.
    
    Outputs:
        incremental_validity.tsv: Per-fold performance for baseline vs full
        incremental_validity_summary.json: Aggregated Δ statistics
    """
    X, y, groups, feature_names, meta = load_active_matrix(subjects, task, deriv_root, config, logger)
    
    results_dir = results_root / "incremental_validity"
    ensure_dir(results_dir)
    
    # Extract temperature as baseline predictor
    if meta is not None and "temperature" in meta.columns:
        X_baseline = meta[["temperature"]].to_numpy()
    else:
        logger.warning("No temperature column found; using intercept-only baseline")
        X_baseline = np.ones((len(y), 1))
    
    # Full model includes EEG features
    X_full = X
    
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error
    
    outer_cv = LeaveOneGroupOut()
    
    records = []
    y_pred_baseline = np.zeros(len(y))
    y_pred_full = np.zeros(len(y))
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        test_subj = groups[test_idx[0]]
        groups_train = groups[train_idx]
        
        # Baseline model (temperature only)
        model_base = Ridge(alpha=1.0)
        model_base.fit(X_baseline[train_idx], y[train_idx])
        y_pred_baseline[test_idx] = model_base.predict(X_baseline[test_idx])
        r2_base = r2_score(y[test_idx], y_pred_baseline[test_idx])
        
        # Full model (temperature + EEG)
        pipe_full = create_elasticnet_pipeline(seed=rng_seed, config=config)
        param_grid = build_elasticnet_param_grid(config)
        inner_cv = create_inner_cv(groups_train, inner_splits)
        grid = GridSearchCV(pipe_full, param_grid, cv=inner_cv, scoring="r2", n_jobs=-1)
        grid.fit(X_full[train_idx], y[train_idx], groups=groups_train)
        y_pred_full[test_idx] = grid.predict(X_full[test_idx])
        r2_full = r2_score(y[test_idx], y_pred_full[test_idx])
        
        records.append({
            "fold": fold_idx,
            "test_subject": test_subj,
            "r2_baseline": r2_base,
            "r2_full": r2_full,
            "delta_r2": r2_full - r2_base,
            "mae_baseline": mean_absolute_error(y[test_idx], y_pred_baseline[test_idx]),
            "mae_full": mean_absolute_error(y[test_idx], y_pred_full[test_idx]),
        })
    
    records_df = pd.DataFrame(records)
    records_df.to_csv(results_dir / "incremental_validity.tsv", sep="\t", index=False)
    
    # Overall summary
    r2_baseline_overall = r2_score(y, y_pred_baseline)
    r2_full_overall = r2_score(y, y_pred_full)
    
    summary = {
        "r2_baseline": r2_baseline_overall,
        "r2_full": r2_full_overall,
        "delta_r2": r2_full_overall - r2_baseline_overall,
        "mean_fold_delta_r2": float(records_df["delta_r2"].mean()),
        "std_fold_delta_r2": float(records_df["delta_r2"].std()),
        "n_folds_positive_delta": int((records_df["delta_r2"] > 0).sum()),
        "n_folds_total": len(records_df),
    }
    
    with open(results_dir / "incremental_validity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    write_reproducibility_info(results_dir, subjects, config, rng_seed)
    logger.info(f"Incremental validity: Δ R² = {summary['delta_r2']:.4f}")
    logger.info(f"Saved to {results_dir}")
    
    return results_dir




def _run_permutation_importance_stage(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    config: Any,
    seed: int,
    n_repeats: int,
    results_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """Run per-fold permutation importance and aggregate."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.inspection import permutation_importance
    
    pipe = create_elasticnet_pipeline(seed=seed, config=config)
    logo = LeaveOneGroupOut()
    
    all_importances = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipe_fold = clone(pipe)
        try:
            pipe_fold.fit(X_train, y_train)
            result = permutation_importance(
                pipe_fold, X_test, y_test,
                n_repeats=n_repeats,
                random_state=seed + fold_idx,
                scoring="r2",
                n_jobs=-1,
            )
            all_importances.append(result.importances_mean)
        except Exception as e:
            logger.warning(f"Fold {fold_idx} permutation importance failed: {e}")
            continue
    
    if not all_importances:
        logger.warning("No permutation importance results")
        return None
    
    mean_importance = np.mean(np.stack(all_importances), axis=0)
    std_importance = np.std(np.stack(all_importances), axis=0)
    
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(mean_importance)],
        "importance_mean": mean_importance,
        "importance_std": std_importance,
        "n_folds": len(all_importances),
    }).sort_values("importance_mean", ascending=False)
    
    output_path = results_dir / "permutation_importance.tsv"
    importance_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved permutation importance to {output_path}")
    
    return output_path


def _run_shap_importance_stage(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    config: Any,
    seed: int,
    results_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """Run per-fold SHAP importance and aggregate."""
    try:
        from eeg_pipeline.analysis.machine_learning.shap_importance import compute_shap_for_cv_folds
    except ImportError:
        logger.warning("SHAP not available; skipping SHAP importance")
        return None
    
    from sklearn.model_selection import LeaveOneGroupOut
    
    logo = LeaveOneGroupOut()
    cv_splits = list(logo.split(X, y, groups))
    
    def model_factory():
        return create_elasticnet_pipeline(seed=seed, config=config)
    
    try:
        importance_df = compute_shap_for_cv_folds(
            model_factory, X, y, cv_splits, feature_names, seed
        )
        
        if importance_df.empty:
            logger.warning("SHAP importance computation returned empty results")
            return None
        
        output_path = results_dir / "shap_importance.tsv"
        importance_df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Saved SHAP importance to {output_path}")
        
        return output_path
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


def _run_uncertainty_stage(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Any,
    seed: int,
    alpha: float,
    results_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """Run conformal prediction for uncertainty quantification."""
    from eeg_pipeline.analysis.machine_learning.uncertainty import (
        compute_prediction_intervals,
        save_prediction_intervals,
    )
    from sklearn.model_selection import LeaveOneGroupOut
    
    pipe = create_elasticnet_pipeline(seed=seed, config=config)
    logo = LeaveOneGroupOut()
    
    all_intervals = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        try:
            result = compute_prediction_intervals(
                model=clone(pipe),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                alpha=alpha,
                method="cv_plus",
                groups=groups_train,
                seed=seed + fold_idx,
            )
            result.compute_coverage(y_test)
            
            all_intervals.append({
                "fold": fold_idx,
                "y_test": y_test,
                "y_pred": result.y_pred,
                "lower": result.lower,
                "upper": result.upper,
                "coverage": result.coverage,
                "mean_width": result.mean_width,
            })
        except Exception as e:
            logger.warning(f"Fold {fold_idx} uncertainty failed: {e}")
            continue
    
    if not all_intervals:
        logger.warning("No uncertainty results")
        return None
    
    y_pred_all = np.concatenate([r["y_pred"] for r in all_intervals])
    lower_all = np.concatenate([r["lower"] for r in all_intervals])
    upper_all = np.concatenate([r["upper"] for r in all_intervals])
    y_test_all = np.concatenate([r["y_test"] for r in all_intervals])
    
    coverage = np.mean((y_test_all >= lower_all) & (y_test_all <= upper_all))
    mean_width = np.mean(upper_all - lower_all)
    
    intervals_df = pd.DataFrame({
        "y_pred": y_pred_all,
        "lower": lower_all,
        "upper": upper_all,
        "y_true": y_test_all,
        "in_interval": (y_test_all >= lower_all) & (y_test_all <= upper_all),
    })
    
    output_path = results_dir / "prediction_intervals.tsv"
    intervals_df.to_csv(output_path, sep="\t", index=False)
    
    metrics = {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "mean_width": mean_width,
        "n_folds": len(all_intervals),
    }
    with open(results_dir / "uncertainty_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Uncertainty: coverage={coverage:.1%}, mean_width={mean_width:.3f}")
    logger.info(f"Saved uncertainty to {output_path}")
    
    return output_path
