"""
Decoding Orchestration (Canonical)
==================================

Single source of truth for LOSO decoding orchestration.
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
from typing import Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from eeg_pipeline.analysis.decoding.cv import (
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
    fit_with_warning_logging,
    _fit_with_inner_cv,
    _fit_default_pipeline,
    _predict_and_log,
    nested_loso_predictions_matrix,
    compute_subject_level_r,
)
from eeg_pipeline.analysis.decoding.pipelines import create_elasticnet_pipeline, build_elasticnet_param_grid
from eeg_pipeline.utils.analysis.tfr import (
    find_common_channels_train_test,
)
from eeg_pipeline.utils.data.decoding import (
    filter_finite_targets,
    extract_epoch_data_block,
    prepare_trial_records_from_epochs,
    load_kept_indices,
)
from eeg_pipeline.utils.data.decoding import load_epochs_with_targets
from eeg_pipeline.utils.data.decoding import load_plateau_matrix
from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.infra.tsv import read_tsv, write_tsv
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.io.decoding import (
    export_predictions,
    export_indices,
    prepare_best_params_path,
)
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.analysis.decoding.time_generalization import time_generalization_regression

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

    config_local = load_settings()
    if inner_splits is None:
        inner_splits = config_local.get("decoding.cv.inner_splits", 5)
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
        if config_dict is None:
            config_dict = {}
        best_params_path = config_dict.get("paths", {}).get("best_params", {}).get("elasticnet_loso", "decoding/best_params/elasticnet_loso.jsonl")
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
        if config_dict is None:
            config_dict = {}
        paths = config_dict.get("paths", {})
        predictions_path = paths.get("predictions", {}).get("elasticnet_loso", "decoding/predictions/elasticnet_loso.tsv")
        metrics_path = paths.get("per_subject_metrics", {}).get("elasticnet_loso", "decoding/per_subject_metrics/elasticnet_loso.tsv")
        indices_path = paths.get("indices", {}).get("elasticnet_loso", "decoding/indices/elasticnet_loso.tsv")
        
        write_tsv(pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "group": np.asarray(groups_ordered),
            "fold": fold_ids,
            "trial_index": test_indices_order,
        }), results_dir / predictions_path)

        write_tsv(pd.DataFrame(per_subj), results_dir / metrics_path)

        idx_df = pd.DataFrame({
            "subject_id": np.asarray(groups_ordered),
            "heldout_subject_id": np.asarray(groups_ordered),
            "fold": fold_ids,
            "trial_index": test_indices_order,
        })
        write_tsv(idx_df, results_dir / indices_path)

        if n_perm > 0 and len(null_r) > 0:
            plot_decoding_null_hist(
                null_r=null_r,
                empirical_r=pooled["pearson_r"],
                save_path=results_dir / "decoding" / "plots" / "elasticnet_loso_null_r.png",
                config=config_local,
                title="LOSO null (r)",
            )
            pred_df_for_plot = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            plot_residual_diagnostics(pred_df_for_plot, "elasticnet_loso", results_dir / "decoding" / "plots" / "elasticnet_loso_residuals.png", config=config_local)

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
    config_local = load_settings()
    
    if blocks_train is not None:
        n_unique_blocks = len(np.unique(blocks_train))
        default_splits = config_local.get("decoding.cv.default_n_splits", 5) if config_local else 5
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

    config_local = load_settings()
    if n_splits is None:
        n_splits = config_local.get("decoding.cv.default_n_splits", 5)
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
        bids_root = Path(deriv_root).parent
        subject_label = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
        events_path = bids_root / subject_label / "eeg" / f"{subject_label}_task-{task}_events.tsv"
        
        if events_path.exists():
            events = read_tsv(events_path)
            if "block" in events.columns:
                epochs_subj_obj = subj_to_epochs[subject_id]
                if hasattr(epochs_subj_obj, 'metadata') and epochs_subj_obj.metadata is not None and 'block' in epochs_subj_obj.metadata.columns:
                    epoch_indices = np.array([trial_records[idx][1] for idx in subject_indices_original])
                    blocks_subj = epochs_subj_obj.metadata.loc[epoch_indices, 'block'].to_numpy() if len(epochs_subj_obj.metadata) > 0 else None
                    if blocks_subj is not None:
                        blocks_subj = blocks_subj[finite_mask]
                else:
                    kept_indices = load_kept_indices(subject_label, deriv_root, len(events), logger)
                    if kept_indices is None or len(kept_indices) == 0:
                        logger.error(
                            f"Subject {subject_id}: No kept_indices file found and epochs.metadata['block'] unavailable. "
                            f"Cannot safely map block labels. Skipping subject."
                        )
                        blocks_subj = None
                    elif len(kept_indices) <= len(events):
                        kept_indices_arr = np.asarray(kept_indices)
                        if np.any(kept_indices_arr < 0) or np.any(kept_indices_arr >= len(events)):
                            logger.error(
                                f"Subject {subject_id}: kept_indices out of bounds for events TSV. "
                                f"Skipping subject."
                            )
                            blocks_subj = None
                        else:
                            blocks_from_kept = events.iloc[kept_indices_arr]["block"].to_numpy()
                            epoch_to_block = dict(enumerate(blocks_from_kept))
                        blocks_subj = np.array([epoch_to_block.get(trial_records[idx][1], np.nan) for idx in subject_indices])
                    else:
                        blocks_subj = None
                
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
            else:
                blocks_subj = None

        if blocks_subj is None:
            logger.error(
                f"Subject {subject_id}: Block/run labels unavailable. "
                f"Within-subject CV requires block/run identifiers to prevent temporal data leakage. "
                f"Ensure 'block' or 'run_id' column exists in {events_path} or skip within-subject analysis."
            )
            raise ValueError(f"Block/run labels required for within-subject CV (subject {subject_id})")

        groups_subj = groups_arr[subject_indices]
        folds = create_within_subject_folds(groups_subj, blocks_subj, n_splits, seed, config=config_local)

        for fold_counter, train_idx_rel, test_idx_rel, subject_name in folds:
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
        if config_dict is None:
            config_dict = {}
        paths = config_dict.get("paths", {})
        predictions_path = paths.get("predictions", {}).get("within_subject_kfold", "decoding/predictions/within_subject_kfold.tsv")
        metrics_path = paths.get("per_subject_metrics", {}).get("within_subject_kfold", "decoding/per_subject_metrics/within_subject_kfold.tsv")
        
        write_tsv(pd.DataFrame({
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "group": groups_all,
        }), results_dir / predictions_path)

        write_tsv(pd.DataFrame(per_subj), results_dir / metrics_path)

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

    config_local = load_settings()
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
        if config_dict is None:
            config_dict = {}
        paths = config_dict.get("paths", {})
        predictions_path = paths.get("predictions", {}).get("baseline_loso", "decoding/predictions/baseline_loso.tsv")
        metrics_path = paths.get("per_subject_metrics", {}).get("baseline_loso", "decoding/per_subject_metrics/baseline_loso.tsv")
        indices_path = paths.get("indices", {}).get("baseline_loso", "decoding/indices/baseline_loso.tsv")
        
        write_tsv(pd.DataFrame({
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "group": groups_all,
            "fold": fold_ids_all,
            "trial_index": test_indices_all,
        }), results_dir / predictions_path)

        write_tsv(pd.DataFrame(per_subj), results_dir / metrics_path)

        idx_df = pd.DataFrame({
            "subject_id": groups_all,
            "heldout_subject_id": groups_all,
            "fold": fold_ids_all,
            "trial_index": test_indices_all,
        })
        write_tsv(idx_df, results_dir / indices_path)

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
    X, y, groups, _feature_names, meta = load_plateau_matrix(subjects, task, deriv_root, config, logger)

    results_dir = results_root / "regression"
    plots_dir = results_dir / "plots"
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    pipe = create_elasticnet_pipeline(seed=rng_seed, config=config)
    param_grid = build_elasticnet_param_grid(config)
    best_params_path = prepare_best_params_path(results_dir / "best_params_elasticnet.jsonl", mode="truncate")
    null_path = results_dir / "loso_null_elasticnet.npz" if n_perm > 0 else None

    y_true, y_pred, groups_ordered, test_indices, fold_ids = nested_loso_predictions_matrix(
        X=X,
        y=y,
        groups=groups,
        pipe=pipe,
        param_grid=param_grid,
        inner_cv_splits=inner_splits,
        n_jobs=-1,
        seed=rng_seed,
        best_params_log_path=best_params_path,
        model_name="elasticnet",
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
        "elasticnet",
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

    metrics = {
        "pearson_r": r_subj,
        "r_subject_ci_low": ci_low,
        "r_subject_ci_high": ci_high,
        "r2": r2_val,
        "p_value": p_val,
    }

    metrics_path = results_dir / "pooled_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Saved decoding results to {results_dir}")
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
    """Run time-generalization decoding analysis."""
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


