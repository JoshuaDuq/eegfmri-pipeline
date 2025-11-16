from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed

from eeg_pipeline.utils.decoding_utils import (
    create_base_preprocessing_pipeline,
    create_elasticnet_pipeline,
    make_pearsonr_scorer,
    grid_search_with_warning_logging,
    blocks_constant_per_subject,
    compute_metrics,
    create_scoring_dict,
    set_random_seeds,
    create_loso_folds,
    create_within_subject_folds,
    create_inner_cv,
    create_block_aware_inner_cv,
    validate_blocks_for_within_subject,
    validate_block_aware_cv_required,
    should_parallelize_folds,
    execute_folds_parallel,
    determine_inner_n_jobs,
    get_inner_cv_splits,
    get_min_channels_required,
    aggregate_fold_results,
    extract_best_params_from_cv_results,
    create_best_params_record,
    get_best_params_for_fold,
    create_empty_fold_result,
)
from eeg_pipeline.utils.tfr_utils import (
    find_common_channels_train_test,
)
from eeg_pipeline.utils.data_loading import (
    filter_finite_targets,
    extract_epoch_data_block,
    prepare_trial_records_from_epochs,
    load_kept_indices,
    load_epochs_with_targets,
    process_subject_metadata,
)
from eeg_pipeline.utils.stats_utils import check_pyriemann
from eeg_pipeline.utils.config_loader import load_settings
from eeg_pipeline.utils.io_utils import read_tsv, write_tsv

logger = logging.getLogger(__name__)


###################################################################
# Helper Functions
###################################################################

def _save_best_params(best_param_records: List[dict], best_params_log_path: Optional[Path]) -> None:
    if best_params_log_path is None:
        return
    if len(best_param_records) == 0:
        return
    
    best_params_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_log_path, "a", encoding="utf-8") as f:
        for record in best_param_records:
            f.write(json.dumps(record) + "\n")


def _fit_with_inner_cv(
    pipe: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    inner_cv_splits: int,
    n_jobs: int,
    scoring: dict,
    refit_metric: str,
    fold: int,
    test_subs: List[str],
    model_name: str,
) -> Tuple[Pipeline, dict]:
    inner_cv = create_inner_cv(train_groups, inner_cv_splits)
    
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit_metric,
        error_score=np.nan,
        pre_dispatch="2*n_jobs",
    )
    
    start_time = time.time()
    grid = grid_search_with_warning_logging(
        grid, X_train, y_train,
        fold_info=f"LOSO fold {fold}",
        logger=logger,
        groups=train_groups
    )
    elapsed_time = time.time() - start_time
    best_estimator = grid.best_estimator_

    cv_results = pd.DataFrame(grid.cv_results_)
    best_params_record = create_best_params_record(
        model_name, fold, cv_results, grid, heldout_subjects=test_subs
    )

    logger.info(f"Fold {fold}: best params by r = {best_params_record['best_params_by_r']}")
    logger.info(f"Fold {fold}: best params by neg_mse = {best_params_record['best_params_by_neg_mse']}")
    logger.info(f"Fold {fold}: inner CV (n_splits={inner_cv.n_splits}) grid-search took {elapsed_time:.1f}s")
    
    return best_estimator, best_params_record

def _fit_default_pipeline(pipe: Pipeline, X_train: np.ndarray, y_train: np.ndarray, fold: int) -> Pipeline:
    estimator = clone(pipe)
    start_time = time.time()
    estimator.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    logger.info(f"Fold {fold}: default fit took {elapsed_time:.1f}s")
    return estimator

def _predict_and_log(
    estimator: Pipeline,
    X_test: np.ndarray,
    n_test_samples: int,
    fold: int,
    fold_start_time: float,
) -> np.ndarray:
    predict_start = time.time()
    y_pred = estimator.predict(X_test)
    predict_time = time.time() - predict_start
    total_fold_time = time.time() - fold_start_time
    logger.info(
        f"Fold {fold}: predict on {n_test_samples} trials took {predict_time:.1f}s; "
        f"total fold {total_fold_time:.1f}s"
    )
    return y_pred


def _create_riemann_pipeline() -> Pipeline:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    return Pipeline(steps=[
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("ridge", Ridge()),
    ])

def _get_riemann_param_grid() -> dict:
    return {
        "cov__estimator": ["oas", "lwf"],
        "ridge__alpha": [1e-3, 1e-2, 1e-1, 1, 10],
    }



###################################################################
# Cross-Validation Predictions
###################################################################

def nested_loso_predictions(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    n_jobs: int,
    seed: int,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "",
    outer_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    X = np.asarray(X)
    y = np.asarray(y)
    scoring = create_scoring_dict()
    refit_metric = 'r'

    folds = create_loso_folds(X, groups)
    inner_n_jobs = determine_inner_n_jobs(outer_n_jobs, n_jobs)
    logger.info(
        f"Executing {len(folds)} LOSO folds with outer_n_jobs={outer_n_jobs}; "
        f"inner GridSearchCV n_jobs={inner_n_jobs}"
    )

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        set_random_seeds(seed, fold)

        test_subjects = np.unique(groups[test_idx]).tolist()
        logger.info(f"LOSO fold {fold}: held-out {test_subjects}")
        fold_start_time = time.time()

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = groups[train_idx]

        n_unique_groups = len(np.unique(train_groups))
        
        config_local = load_settings()
        min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
        if n_unique_groups >= min_subjects_for_loso:
            best_estimator, best_params_rec = _fit_with_inner_cv(
                pipe, param_grid, X_train, y_train, train_groups,
                inner_cv_splits, inner_n_jobs, scoring, refit_metric,
                fold, test_subjects, model_name
            )
        else:
            logger.info(
                f"Only one training group available in fold {fold}; "
                f"skipping inner CV and fitting default pipeline params."
            )
            best_estimator = _fit_default_pipeline(pipe, X_train, y_train, fold)
            best_params_rec = None

        y_pred = _predict_and_log(best_estimator, X_test, len(test_idx), fold, fold_start_time)
        
        return {
            "fold": fold,
            "y_true": y_test.tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
            "groups": groups[test_idx].tolist(),
            "test_idx": test_idx.tolist(),
            "best_params_rec": best_params_rec,
        }

    results = execute_folds_parallel(folds, _run_fold, outer_n_jobs)
    best_param_records = [r["best_params_rec"] for r in results if r["best_params_rec"] is not None]
    _save_best_params(best_param_records, best_params_log_path)
    
    return aggregate_fold_results(results)

def loso_predictions_with_fixed_params(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    pipe_template: Pipeline,
    best_params_map: dict,
    seed: int = 42,
    outer_n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    logo = LeaveOneGroupOut()
    folds = [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1)
    ]

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        set_random_seeds(seed, fold)
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        pipe = clone(pipe_template)
        heldout_subject = str(np.asarray(groups)[test_idx][0])
        params = get_best_params_for_fold(best_params_map, heldout_subject, fold)
        
        if params:
            pipe.set_params(**params)
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        return {
            "fold": fold,
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "groups": groups[test_idx].tolist(),
            "test_idx": test_idx.tolist(),
        }

    results = execute_folds_parallel(folds, _run_fold, outer_n_jobs)
    return aggregate_fold_results(results)


def _fit_within_subject_fold(
    pipe: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    blocks_all: Optional[np.ndarray],
    train_idx: np.ndarray,
    inner_cv_splits: int,
    n_jobs: int,
    outer_n_jobs: int,
    scoring: dict,
    refit_metric: str,
    seed: int,
    fold: int,
    subject: str,
    model_name: str,
) -> Tuple[Pipeline, Optional[dict]]:
    from eeg_pipeline.utils.decoding_utils import log_cv_adjacency_info
    
    n_train_samples = len(train_idx)
    
    config_local = load_settings()
    min_trials_for_inner_cv = config_local.get("decoding.cv.min_trials_for_inner_cv", 3)
    if n_train_samples < min_trials_for_inner_cv:
        logger.info(
            f"Within fold {fold} ({subject}): insufficient train trials for inner CV; "
            f"fitting default pipeline params."
        )
        return _fit_default_pipeline(pipe, X_train.to_numpy(), y_train.to_numpy(), fold), None
    
    min_trials_for_within_subject = config_local.get("decoding.cv.min_trials_for_within_subject", 2)
    n_splits_inner = int(np.clip(inner_cv_splits, min_trials_for_within_subject, n_train_samples))
    
    if blocks_all is None:
        raise ValueError(
            f"Within-subject CV for subject {subject} fold {fold}: block information unavailable. "
            f"Cannot construct valid CV splits without block/run identifiers."
        )
    
    blocks_train = blocks_all[train_idx]
    cv_splits = create_block_aware_inner_cv(blocks_train, n_splits_inner, seed, fold, subject)
    validate_block_aware_cv_required(cv_splits, subject, fold)
    
    for i, (train_inner_idx, test_inner_idx) in enumerate(cv_splits[:2]):
        log_cv_adjacency_info(
            test_inner_idx,
            f"Within fold {fold} ({subject}) inner split {i+1} test (block-aware)"
        )
    
    inner_n_jobs = determine_inner_n_jobs(outer_n_jobs, n_jobs)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv_splits,
        scoring=scoring,
        n_jobs=inner_n_jobs,
        refit=refit_metric,
        error_score=np.nan,
        pre_dispatch="2*n_jobs",
    )
    
    start_time = time.time()
    grid = grid_search_with_warning_logging(
        grid, X_train, y_train,
        fold_info=f"Within fold {fold} ({subject})",
        logger=logger
    )
    elapsed_time = time.time() - start_time
    best_estimator = grid.best_estimator_
    
    cv_results = pd.DataFrame(grid.cv_results_)
    best_params_record = create_best_params_record(
        model_name, fold, cv_results, grid, subject=subject
    )
    
    logger.info(
        f"Within fold {fold} ({subject}): best params by r = {best_params_record['best_params_by_r']}"
    )
    logger.info(
        f"Within fold {fold} ({subject}): best params by neg_mse = "
        f"{best_params_record['best_params_by_neg_mse']}"
    )
    logger.info(
        f"Within fold {fold} ({subject}): inner CV (n_splits={n_splits_inner}) took {elapsed_time:.1f}s"
    )
    
    return best_estimator, best_params_record

def within_subject_kfold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    pipe: Pipeline,
    param_grid: dict,
    inner_cv_splits: int,
    n_jobs: int,
    seed: int,
    best_params_log_path: Optional[Path] = None,
    model_name: str = "",
    outer_n_jobs: int = 1,
    deriv_root: Optional[Path] = None,
    task: str = "",
    blocks_all: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    scoring = create_scoring_dict()
    refit_metric = 'r'

    blocks_all = validate_blocks_for_within_subject(blocks_all, groups)
    config_local = load_settings()
    folds = create_within_subject_folds(groups, blocks_all, inner_cv_splits, seed, config=config_local)
    
    unique_subjects = len(np.unique(groups))
    inner_n_jobs = determine_inner_n_jobs(outer_n_jobs, n_jobs)
    logger.info(
        f"Executing {len(folds)} within-subject KFold folds over {unique_subjects} subjects; "
        f"outer_n_jobs={outer_n_jobs}; inner GridSearchCV n_jobs={inner_n_jobs}"
    )

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray, subject: str):
        set_random_seeds(seed, fold)
        fold_start_time = time.time()
        
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_estimator, best_params_rec = _fit_within_subject_fold(
            pipe, param_grid, X_train, y_train, blocks_all, train_idx,
            inner_cv_splits, n_jobs, outer_n_jobs, scoring, refit_metric, seed, fold, subject, model_name
        )

        y_pred = _predict_and_log(best_estimator, X_test.to_numpy(), len(test_idx), fold, fold_start_time)
        
        return {
            "fold": fold,
            "subject": subject,
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "groups": groups[test_idx].tolist(),
            "test_idx": test_idx.tolist(),
            "best_params_rec": best_params_rec,
        }

    results = execute_folds_parallel(folds, _run_fold, outer_n_jobs)
    best_param_records = [r["best_params_rec"] for r in results if r["best_params_rec"] is not None]
    _save_best_params(best_param_records, best_params_log_path)
    
    return aggregate_fold_results(results)

def loso_baseline_predictions(
    y: pd.Series,
    groups: np.ndarray,
    mode: str = "global",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    if mode != "global":
        raise ValueError(
            f"Invalid baseline mode '{mode}'. Only 'global' is supported to prevent data leakage."
        )
    
    logo = LeaveOneGroupOut()
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    groups_ordered: List[str] = []
    test_indices_order: List[int] = []
    fold_ids: List[int] = []

    dummy_X = np.empty(len(y))
    for fold, (train_idx, test_idx) in enumerate(logo.split(dummy_X, groups=groups), start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        test_groups = groups[test_idx]
        
        mean_train = float(y_train.mean())
        y_pred = np.full_like(y_test.values, fill_value=mean_train, dtype=float)

        n_test = len(test_idx)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        groups_ordered.extend(test_groups.tolist())
        test_indices_order.extend(test_idx.tolist())
        fold_ids.extend([fold] * n_test)

    return (
        np.asarray(y_true_all),
        np.asarray(y_pred_all),
        groups_ordered,
        test_indices_order,
        fold_ids
    )


###################################################################
# Feature Importance and Interpretability
###################################################################

def _prepare_block_metadata_for_permutation(
    fold: int,
    heldout_subject: Optional[str],
    blocks_all: Optional[np.ndarray],
    groups: np.ndarray,
    test_idx: np.ndarray,
    blocks_origin: str,
) -> Tuple[Optional[np.ndarray], dict]:
    fold_meta = {
        "fold": int(fold),
        "heldout_subject": heldout_subject,
        "block_strategy": "none",
        "notes": "",
        "n_unique_blocks": 0,
    }
    
    notes: List[str] = []
    b_test = None
    
    if blocks_all is None:
        return None, fold_meta
    
    if len(blocks_all) != len(groups):
        logger.warning(
            "RF permutation fold %d: provided run_id array has length %d but expected %d; ignoring run-aware permutations.",
            fold, len(blocks_all), len(groups),
        )
        return None, fold_meta
    
    g_test = np.asarray(groups)[test_idx]
    candidate = np.asarray(blocks_all)[test_idx]
    
    if blocks_constant_per_subject(candidate, g_test):
        notes.append("constant_blocks_from_source")
        logger.info(
            "RF permutation fold %d: run identifiers constant within subject; permutations will ignore run structure.",
            fold,
        )
        fold_meta["block_strategy"] = "constant"
        b_test = None
    else:
        fold_meta["block_strategy"] = blocks_origin if blocks_origin else "provided"
        b_test = candidate
    
    if b_test is None and not fold_meta["block_strategy"]:
        fold_meta["block_strategy"] = "none"
    
    if b_test is not None:
        uniq_blocks = pd.unique(b_test[~pd.isna(b_test)])
        fold_meta["n_unique_blocks"] = int(len(uniq_blocks))
    
    if not notes and b_test is None and fold_meta["block_strategy"] == "constant":
        notes.append("constant_blocks")
    
    if notes:
        existing = fold_meta["notes"]
        fold_meta["notes"] = ";".join([n for n in ([existing] if existing else []) + notes])
    
    return b_test, fold_meta

def _is_feature_constant_within_blocks(
    feature_idx: int,
    X_test: pd.DataFrame,
    test_groups: np.ndarray,
    blocks_test: Optional[np.ndarray],
) -> bool:
    for subject in np.unique(test_groups):
        subject_indices = np.where(test_groups == subject)[0]
        if len(subject_indices) <= 1:
            continue
        
        if blocks_test is None:
            index_sets = [subject_indices]
        else:
            subject_blocks = blocks_test[subject_indices]
            unique_blocks = np.unique(subject_blocks)
            index_sets = [
                subject_indices[subject_blocks == block_id]
                for block_id in unique_blocks
            ]
        
        for indices in index_sets:
            if len(indices) <= 1:
                continue
            
            feature_values = X_test.iloc[indices, feature_idx].to_numpy()
            try:
                n_unique_values = pd.Series(feature_values).nunique(dropna=False)
            except (ValueError, MemoryError):
                n_unique_values = 2
            
            if int(n_unique_values) > 1:
                return False
    
    return True

def _build_per_subject_indices(
    test_groups: np.ndarray,
    blocks_test: Optional[np.ndarray],
) -> Dict[str, List[np.ndarray]]:
    per_subject_indices = {}
    for subject in np.unique(test_groups):
        subject_indices = np.where(test_groups == subject)[0]
        if blocks_test is None:
            per_subject_indices[subject] = [subject_indices]
        else:
            subject_blocks = blocks_test[subject_indices]
            unique_blocks = np.unique(subject_blocks)
            per_subject_indices[subject] = [
                subject_indices[subject_blocks == block_id]
                for block_id in unique_blocks
            ]
    return per_subject_indices

def _compute_permutation_importance_for_feature(
    feature_idx: int,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_groups: np.ndarray,
    blocks_test: Optional[np.ndarray],
    pipe: Pipeline,
    base_r2: float,
    n_repeats: int,
    rng: np.random.Generator,
) -> float:
    deltas = np.zeros(n_repeats, dtype=float)
    X_test_array = X_test.to_numpy()
    
    per_subject_indices = _build_per_subject_indices(test_groups, blocks_test)
    
    for repeat in range(n_repeats):
        X_test_permuted = X_test_array.copy()
        for subject, index_sets in per_subject_indices.items():
            for indices in index_sets:
                if len(indices) > 1:
                    X_test_permuted[indices, feature_idx] = rng.permutation(
                        X_test_permuted[indices, feature_idx]
                    )
        
        X_permuted_df = pd.DataFrame(
            X_test_permuted, columns=X_test.columns, index=X_test.index
        )
        try:
            y_pred_permuted = pipe.predict(X_permuted_df)
            r2_permuted = r2_score(y_test, np.asarray(y_pred_permuted))
            delta = base_r2 - r2_permuted
            deltas[repeat] = delta if np.isfinite(delta) else 0.0
        except Exception:
            deltas[repeat] = 0.0
        finally:
            del X_permuted_df
    
    return float(np.mean(deltas))

def compute_enet_coefs_per_fold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    best_params_map: dict,
    seed: int,
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    coefficients = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        heldout_subject = str(np.asarray(groups)[test_idx][0])
        params = get_best_params_for_fold(best_params_map, heldout_subject, fold)
        
        pipeline = create_elasticnet_pipeline(seed=seed)
        if params:
            pipeline.set_params(**params)
        
        pipeline.fit(X.iloc[train_idx, :], y.iloc[train_idx])
        elasticnet_regressor = pipeline.named_steps["regressor"].regressor_
        coefficients_selected = np.asarray(elasticnet_regressor.coef_, dtype=float)
        
        variance_mask = pipeline.named_steps["var"].get_support(indices=False)
        n_features_kept = variance_mask.sum()
        logger.info(f"Fold {fold}: Var-kept={n_features_kept}")
        
        full_coefficients = np.zeros(X.shape[1], dtype=float)
        if coefficients_selected.shape[0] == n_features_kept:
            full_coefficients[variance_mask] = coefficients_selected
        else:
            logger.warning(f"Fold {fold}: coef length mismatch; setting to NaN")
            full_coefficients[:] = np.nan
        
        coefficients.append(full_coefficients)
    
    return np.asarray(coefficients)

def compute_rf_block_permutation_importance_per_fold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    best_params_map: dict,
    seed: int,
    n_repeats: Optional[int] = None,
    blocks_all: Optional[np.ndarray] = None,
    blocks_origin: str = "unknown",
    config_dict: Optional[dict] = None,
) -> Tuple[np.ndarray, List[dict]]:
    config_local = load_settings()
    default_rf_perm_importance_repeats = config_local.get("decoding.analysis.rf_perm_importance_repeats", 20)
    default_rf_n_estimators = config_local.get("decoding.models.random_forest.n_estimators", 500)
    
    if n_repeats is None:
        if config_dict is not None:
            n_repeats = int(config_dict.get("analysis", {}).get("rf_perm_importance_repeats", default_rf_perm_importance_repeats))
        else:
            n_repeats = default_rf_perm_importance_repeats
    
    logo = LeaveOneGroupOut()
    imps: List[np.ndarray] = []
    block_meta: List[dict] = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        try:
            heldout_subject = str(np.asarray(groups)[test_idx][0])
        except (IndexError, ValueError, TypeError):
            heldout_subject = None
        
        b_test, fold_meta = _prepare_block_metadata_for_permutation(
            fold, heldout_subject, blocks_all, groups, test_idx, blocks_origin
        )
        
        rf_params = get_best_params_for_fold(best_params_map, heldout_subject, fold)
        rf = RandomForestRegressor(
            n_estimators=rf_params.get("rf__n_estimators", default_rf_n_estimators),
            max_depth=rf_params.get("rf__max_depth", None),
            max_features=rf_params.get("rf__max_features", "sqrt"),
            min_samples_leaf=rf_params.get("rf__min_samples_leaf", 1),
            random_state=seed,
            n_jobs=1,
            bootstrap=True,
        )
        pipe = create_base_preprocessing_pipeline(include_scaling=False)
        pipe.steps.append(("rf", rf))
        
        try:
            X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            g_test = np.asarray(groups)[test_idx]
            
            pipe.fit(X_train, y_train)
            y_pred_base = pipe.predict(X_test)
            base_r2 = r2_score(y_test.to_numpy(), np.asarray(y_pred_base))
            
            rng = np.random.default_rng(seed + fold)
            n_features = X.shape[1]
            deltas_mean = np.zeros(n_features, dtype=float)
            skipped_feats: List[str] = []
            
            for feature_idx in range(n_features):
                if _is_feature_constant_within_blocks(feature_idx, X_test, g_test, b_test):
                    deltas_mean[feature_idx] = 0.0
                    skipped_feats.append(str(X.columns[feature_idx]))
                    continue
                
                deltas_mean[feature_idx] = _compute_permutation_importance_for_feature(
                    feature_idx, X_test, y_test.to_numpy(), g_test, b_test,
                    pipe, base_r2, n_repeats, rng
                )
            
            if skipped_feats:
                prev = skipped_feats[:10]
                logger.info(
                    f"RF block perm fold {fold}: skipped {len(skipped_feats)}/{n_features} "
                    f"constant-within-block features (showing up to 10): {prev}"
                )
            
            imps.append(deltas_mean)
            block_meta.append(fold_meta)
            
        except (ValueError, MemoryError, RuntimeError) as e:
            logger.warning(f"RF block permutation importance failed on fold {fold}: {e}")
            imps.append(np.full(X.shape[1], np.nan, dtype=float))
            fold_meta["block_strategy"] = "error"
            fold_meta["notes"] = f"exception:{type(e).__name__}"
            block_meta.append(fold_meta)
    
    return np.asarray(imps), block_meta

def run_shap_rf_loso(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    feature_names: List[str],
    best_params_map: dict,
    export_dir: Path,
    seed: int,
) -> None:
    try:
        import shap
    except ImportError:
        logger.info("SHAP not installed; skipping fold-wise SHAP analysis.")
        return

    logo = LeaveOneGroupOut()

    shap_segments: List[np.ndarray] = []
    feat_segments: List[np.ndarray] = []
    y_segments: List[np.ndarray] = []
    idx_segments: List[np.ndarray] = []
    fold_id_segments: List[np.ndarray] = []
    heldout_subjects: List[List[str]] = []

    n_features = X.shape[1]

    config_local = load_settings()
    default_rf_n_estimators = config_local.get("decoding.models.random_forest.n_estimators", 500)

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        if not isinstance(best_params_map, dict):
            params = {}
        else:
            try:
                heldout_subject = str(np.asarray(groups)[test_idx][0])
            except (KeyError, IndexError, ValueError):
                heldout_subject = None
            params = get_best_params_for_fold(best_params_map, heldout_subject, fold)
        pre = create_base_preprocessing_pipeline(include_scaling=False)
        try:
            X_train = X.iloc[train_idx, :]
            X_test = X.iloc[test_idx, :]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"SHAP fold {fold}: indexing failed: {e}")
            continue

        try:
            pre.fit(X_train, y_train)
            X_train_t = pre.transform(X_train)
            X_test_t = pre.transform(X_test)
            var_mask = pre.named_steps["var"].get_support(indices=False)
        except (ValueError, RuntimeError, MemoryError) as e:
            logger.warning(f"SHAP fold {fold}: preprocessing failed: {e}")
            continue

        try:
            rf = RandomForestRegressor(
                n_estimators=params.get("rf__n_estimators", default_rf_n_estimators),
                max_depth=params.get("rf__max_depth", None),
                max_features=params.get("rf__max_features", "sqrt"),
                min_samples_leaf=params.get("rf__min_samples_leaf", 1),
                random_state=seed + fold,
                n_jobs=1,
                bootstrap=True,
            )
            rf.fit(X_train_t, y_train.values)
        except (ValueError, MemoryError) as e:
            logger.warning(f"SHAP fold {fold}: RF fit failed: {e}")
            continue

        try:
            explainer = shap.TreeExplainer(rf)
            shap_vals_fold = explainer.shap_values(X_test_t)
            if isinstance(shap_vals_fold, list):
                shap_vals_fold = np.asarray(shap_vals_fold[0]) if len(shap_vals_fold) > 0 else np.empty_like(X_test_t)
            shap_vals_fold = np.asarray(shap_vals_fold)
            shap_full = np.zeros((X_test.shape[0], n_features), dtype=float)
            try:
                shap_full[:, var_mask] = shap_vals_fold
            except (ValueError, IndexError, TypeError):
                if shap_vals_fold.shape[1] == int(np.sum(var_mask)):
                    shap_full[:, var_mask] = shap_vals_fold
                else:
                    logger.warning(f"SHAP fold {fold}: var-mask alignment failed; padding zeros for unmatched features")
            shap_segments.append(shap_full)
            feat_segments.append(X_test.to_numpy())
            y_segments.append(y_test.to_numpy())
            idx_segments.append(np.asarray(test_idx))
            fold_id_segments.append(np.full(X_test.shape[0], fold, dtype=int))
            heldout_subjects.append([str(s) for s in np.asarray(groups)[test_idx]])
        except (ValueError, RuntimeError, MemoryError) as e:
            logger.warning(f"SHAP fold {fold}: SHAP computation failed: {e}")
            continue

    if len(shap_segments) == 0:
        logger.warning("No SHAP segments computed; skipping SHAP aggregation outputs.")
        return

    SHAP = np.vstack(shap_segments)
    X_explain = np.vstack(feat_segments)
    y_explain = np.concatenate(y_segments)
    sample_indices = np.concatenate(idx_segments)
    fold_ids = np.concatenate(fold_id_segments)
    subj_ids = np.concatenate([np.asarray(h) for h in heldout_subjects])

    try:
        shap_abs = np.abs(SHAP)
        mean_abs = np.nanmean(shap_abs, axis=0)
        std_abs = np.nanstd(shap_abs, axis=0)
        rank_order = np.argsort(mean_abs)[::-1]
        feat_imp_df = pd.DataFrame({
            "feature_name": feature_names,
            "mean_abs_shap": mean_abs,
            "std_abs_shap": std_abs,
        })
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(1, len(rank_order) + 1)
        feat_imp_df["rank"] = ranks
        write_tsv(feat_imp_df.sort_values("rank"), export_dir / "rf_shap_loso_feature_importance.tsv")

        by_fold_records = []
        uniq_folds = np.unique(fold_ids)
        for f in uniq_folds:
            m = (fold_ids == f)
            if not np.any(m):
                continue
            mean_abs_f = np.nanmean(shap_abs[m, :], axis=0)
            for j, nm in enumerate(feature_names):
                by_fold_records.append({
                    "fold": int(f),
                    "feature_name": nm,
                    "mean_abs_shap": float(mean_abs_f[j]),
                    "n_samples": int(np.sum(m)),
                })
        if by_fold_records:
            write_tsv(pd.DataFrame(by_fold_records), export_dir / "rf_shap_loso_feature_importance_by_fold.tsv")
    except (OSError, PermissionError, ValueError) as e:
        logger.warning(f"Failed to export SHAP feature importance TSVs: {e}")

    try:
        np.savez_compressed(
            export_dir / "rf_shap_loso_values.npz",
            shap_values=SHAP,
            X=X_explain,
            y=y_explain,
            feature_names=np.asarray(feature_names),
            sample_indices=sample_indices,
            fold_ids=fold_ids,
            subject_ids=subj_ids,
        )
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to save LOSO SHAP arrays: {e}")

    try:
        uniq_folds, fold_counts = np.unique(fold_ids, return_counts=True)
        config_local = load_settings()
        top_features_display_limit = config_local.get("decoding.visualization.top_n_features", 20)
        top_n = min(top_features_display_limit, len(feature_names))
        mean_abs = np.nanmean(np.abs(SHAP), axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        top_feats = [feature_names[i] for i in order]
        meta = {
            "n_samples": int(SHAP.shape[0]),
            "n_features": int(SHAP.shape[1]),
            "n_subjects": int(len(np.unique(subj_ids))),
            "n_folds": int(len(np.unique(fold_ids))),
            "fold_counts": {int(f): int(c) for f, c in zip(uniq_folds.tolist(), fold_counts.tolist())},
            "top_features_by_mean_abs_shap": top_feats,
        }
        with open(export_dir / "rf_shap_loso_summary.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"Failed to save SHAP LOSO summary JSON: {e}")

def subject_id_decodability_auc(
    X: pd.DataFrame,
    groups: np.ndarray,
    results_dir: Optional[Path] = None,
    seed: int = 42,
) -> Optional[float]:
    subjects = pd.Series(groups)
    classes = subjects.unique()
    counts = subjects.value_counts()
    min_n = int(counts.min()) if len(counts) > 0 else 0
    
    config_local = load_settings()
    min_trials_for_within_subject = config_local.get("decoding.cv.min_trials_for_within_subject", 2)
    min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
    if min_n < min_trials_for_within_subject or len(classes) < min_subjects_for_loso:
        logger.warning(
            f"Subject-ID decodability check skipped: need >={min_subjects_for_loso} classes and "
            f">={min_trials_for_within_subject} trials per subject."
        )
        return None
    
    n_splits = min(5, max(min_trials_for_within_subject, min_n))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    from sklearn.preprocessing import StandardScaler
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(random_state=seed, max_iter=1000, multi_class="ovr"))
    ])
    proba_list, y_list, pred_list = [], [], []
    all_predictions = []
    class_labels = None
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = groups[train_idx], groups[test_idx]
        try:
            lr.fit(X_train, y_train)
            proba = lr.predict_proba(X_test)
            y_pred = lr.predict(X_test)
            if class_labels is None:
                try:
                    class_labels = lr.named_steps["logreg"].classes_
                except (KeyError, AttributeError):
                    class_labels = np.unique(groups)
            
            proba_list.append(proba)
            y_list.append(y_test)
            pred_list.append(y_pred)
            
            for i, (true_subj, pred_subj, prob_vec) in enumerate(zip(y_test, y_pred, proba)):
                cls_list = list(class_labels) if class_labels is not None else []
                proba_map = {f"prob_{cls}": float(prob_vec[j]) for j, cls in enumerate(cls_list)}
                all_predictions.append({
                    "fold": fold_idx,
                    "true_subject": str(true_subj),
                    "pred_subject": str(pred_subj),
                    "correct": bool(true_subj == pred_subj),
                    **proba_map,
                })
                
        except (ValueError, RuntimeError, MemoryError) as e:
            logger.warning(f"Subject-ID CV fold {fold_idx} failed: {e}")

    if len(proba_list) == 0:
        logger.warning("Subject-ID decodability: no predictions produced.")
        return None
    Y = np.concatenate(y_list, axis=0)
    P = np.concatenate(proba_list, axis=0)
    Y_pred = np.concatenate(pred_list, axis=0)
    
    try:
        auc_macro = roc_auc_score(Y, P, multi_class="ovr", average="macro")
    except ValueError:
        logger.warning("AUC computation failed for subject-ID decodability.")
        return None

    if results_dir is not None:
        try:
            per_subject_aucs = []
            unique_subjects = np.unique(Y)
            
            for subj in unique_subjects:
                y_binary = (Y == subj).astype(int)
                if len(np.unique(y_binary)) == 2:
                    subj_idx = np.where(np.asarray(class_labels) == subj)[0] if class_labels is not None else []
                    if len(subj_idx) > 0:
                        p_subj = P[:, subj_idx[0]]
                        try:
                            auc_subj = roc_auc_score(y_binary, p_subj)
                            per_subject_aucs.append({
                                "subject_id": str(subj),
                                "auc": float(auc_subj),
                                "n_trials": int(np.sum(Y == subj))
                            })
                        except ValueError as e:
                            logger.warning(f"Failed to compute AUC for subject {subj}: {e}")
            
            if per_subject_aucs:
                per_subj_df = pd.DataFrame(per_subject_aucs)
                results_dir.mkdir(parents=True, exist_ok=True)
                write_tsv(per_subj_df, results_dir / "subject_id_per_subject_aucs.tsv")
                logger.info(f"Saved per-subject ID decodability AUCs: {len(per_subject_aucs)} subjects")
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(Y, Y_pred, labels=unique_subjects)
            cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in unique_subjects], 
                               columns=[f"pred_{s}" for s in unique_subjects])
            write_tsv(cm_df, results_dir / "subject_id_confusion_matrix.tsv", index=True)
            
            auc_table = {
                "macro_auc": float(auc_macro),
                "n_subjects": len(unique_subjects),
                "n_trials_total": len(Y),
                "n_cv_splits": n_splits,
                "accuracy": float(np.mean(Y == Y_pred)),
                "per_subject_auc_mean": float(np.mean([s["auc"] for s in per_subject_aucs])) if per_subject_aucs else np.nan,
                "per_subject_auc_std": float(np.std([s["auc"] for s in per_subject_aucs])) if per_subject_aucs else np.nan
            }
            
            with open(results_dir / "subject_id_auc_table.json", "w", encoding="utf-8") as f:
                json.dump(auc_table, f, indent=2)
            
            if all_predictions:
                pred_df = pd.DataFrame(all_predictions)
                write_tsv(pred_df, results_dir / "subject_id_detailed_predictions.tsv")
                
            logger.info(f"Subject-ID decodability: macro AUC={auc_macro:.3f}, accuracy={auc_table['accuracy']:.3f}")
            
        except (OSError, PermissionError, ValueError) as e:
            logger.warning(f"Failed to save detailed subject-ID decodability results: {e}")

    return float(auc_macro)


###################################################################
# Metadata Aggregation
###################################################################


def aggregate_temperature_trial_and_block(meta: pd.DataFrame, deriv_root: Path, task: str, config=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    bids_root = Path(deriv_root).parent
    temps_out = np.full(len(meta), np.nan, dtype=float)
    trials_out = np.full(len(meta), np.nan, dtype=float)
    blocks_out = np.full(len(meta), np.nan, dtype=float)
    
    subjects_with_blocks = 0
    subjects_total = 0
    total_trials_with_blocks = 0
    total_trials = len(meta)
    
    if config is None:
        config = load_settings()

    for subject_id in sorted(meta["subject_id"].unique()):
        subjects_total += 1
        subject_str = str(subject_id)
        subject_label = subject_str if subject_str.startswith("sub-") else f"sub-{subject_str}"
        
        events_path = bids_root / subject_label / "eeg" / f"{subject_label}_task-{task}_events.tsv"
        meta_indices = meta.index[meta["subject_id"] == subject_label].to_numpy()
        n_subject_trials = len(meta_indices)
        
        if not events_path.exists():
            logger.warning(f"Events TSV not found for {subject_label}: {events_path}")
            meta_trial_ids = meta.loc[meta_indices, "trial_id"].to_numpy()
            trials_out[meta_indices] = meta_trial_ids.astype(float) + 1.0
            continue
        
        events = read_tsv(events_path)
        meta_trial_ids = meta.loc[meta_indices, "trial_id"].to_numpy()
        
        kept_indices = load_kept_indices(subject_label, deriv_root, len(events), logger)
        if kept_indices is None or len(kept_indices) == 0:
            kept_indices = np.arange(min(n_subject_trials, len(events)))
            logger.debug(f"{subject_label}: No dropped trials, using all {len(kept_indices)} trials")
        
        subj_with_blocks, trials_with_blocks = process_subject_metadata(
            subject_label,
            meta_indices,
            events,
            kept_indices,
            meta_trial_ids,
            config,
            temps_out,
            trials_out,
            blocks_out,
            logger,
        )
        subjects_with_blocks += subj_with_blocks
        total_trials_with_blocks += trials_with_blocks

    blocks_coverage = total_trials_with_blocks / max(1, total_trials)
    subjects_coverage = subjects_with_blocks / max(1, subjects_total)
    
    if subjects_coverage >= 1.0 and blocks_coverage >= 1.0:
        provenance = "events"
        logger.info(
            f"Block metadata: Complete coverage ({subjects_with_blocks}/{subjects_total} subjects, "
            f"{total_trials_with_blocks}/{total_trials} trials)"
        )
    elif subjects_coverage > 0:
        provenance = "events"
        logger.warning(
            f"Block metadata: Partial coverage ({subjects_with_blocks}/{subjects_total} subjects, "
            f"{total_trials_with_blocks}/{total_trials} trials, {blocks_coverage:.1%})"
        )
    else:
        provenance = "none"
        logger.info("Block metadata: Not found in events files")
    
    return temps_out, trials_out, blocks_out, provenance


###################################################################
# Riemann Decoding
###################################################################


def loso_riemann_regression(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    results_dir: Path = None,
    n_jobs: int = -1,
    seed: int = 42,
    outer_n_jobs: int = 1,
    config_dict: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, pd.DataFrame]:
    if not check_pyriemann():
        raise ImportError("pyriemann is not installed. Install with `pip install pyriemann` to run Model 2.")

    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)
    trial_records, y_all_arr, groups_arr, subj_to_epochs, subj_to_y = prepare_trial_records_from_epochs(tuples)

    config_local = load_settings()
    min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
    if len(np.unique(groups_arr)) < min_subjects_for_loso:
        raise RuntimeError(f"Need at least {min_subjects_for_loso} subjects for LOSO.")

    min_channels_required = get_min_channels_required(config_dict, config=config_local)

    pipe = _create_riemann_pipeline()
    param_grid = _get_riemann_param_grid()

    logo = LeaveOneGroupOut()
    folds = list(enumerate(logo.split(np.arange(len(trial_records)), groups=groups_arr), start=1))

    def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
        set_random_seeds(seed, fold)

        train_subjects = list({trial_records[i][0] for i in train_idx if trial_records[i][0] is not None})
        test_subject = trial_records[int(test_idx[0])][0]
        
        common_chs_fold = find_common_channels_train_test(train_subjects, test_subject, subj_to_epochs)
        
        if not common_chs_fold:
            logger.warning(f"Fold {fold}: No common EEG channels across training subjects. Skipping fold.")
            return create_empty_fold_result(fold, test_idx, y_all_arr, groups_arr)

        if len(common_chs_fold) < min_channels_required:
            logger.warning(f"Fold {fold}: common channels with test subject too few (n={len(common_chs_fold)} < {min_channels_required}). Skipping fold.")
            return create_empty_fold_result(fold, test_idx, y_all_arr, groups_arr)

        logger.info(f"Fold {fold}: Using {len(common_chs_fold)} EEG channels (train∩test) across {len(train_subjects)} training subjects.")

        subjects_in_fold = list({trial_records[i][0] for i in np.concatenate([train_idx, test_idx])})
        aligned_epochs = {s: subj_to_epochs[s].copy().pick(common_chs_fold) for s in subjects_in_fold}

        train_idx_f, y_train = filter_finite_targets(train_idx, y_all_arr)
        test_idx_f, y_test = filter_finite_targets(test_idx, y_all_arr)

        X_train = extract_epoch_data_block(train_idx_f, trial_records, aligned_epochs)
        X_test = extract_epoch_data_block(test_idx_f, trial_records, aligned_epochs)
        train_groups = groups_arr[train_idx_f]

        n_unique = len(np.unique(train_groups))
        scoring = create_scoring_dict()
        refit_metric = 'r'
        if n_unique >= min_subjects_for_loso:
            n_splits_inner = get_inner_cv_splits(config_dict, n_unique, config=config_local)
            inner_cv = GroupKFold(n_splits=n_splits_inner)
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=n_jobs,
                refit=refit_metric,
            )
            gs = grid_search_with_warning_logging(
                gs, X_train, y_train,
                fold_info=f"Riemann fold {fold}",
                logger=logger,
                groups=train_groups
            )
            best_estimator = gs.best_estimator_
            
            cv_results = pd.DataFrame(gs.cv_results_)
            best_params_rec = create_best_params_record("Riemann", fold, cv_results, gs)
            logger.info(f"Fold {fold}: best params by r = {best_params_rec['best_params_by_r']}")
            logger.info(f"Fold {fold}: best params by neg_mse = {best_params_rec['best_params_by_neg_mse']}")
        else:
            best_estimator = clone(pipe)
            best_estimator.fit(X_train, y_train)
            best_params_rec = None

        y_pred = best_estimator.predict(X_test)

        return {
            "fold": fold,
            "y_true": y_test.tolist(),
            "y_pred": np.asarray(y_pred).tolist(),
            "groups": groups_arr[test_idx_f].tolist(),
            "test_idx": test_idx_f.tolist(),
            "best_params_rec": best_params_rec,
        }

    if should_parallelize_folds(outer_n_jobs, len(folds)):
        results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(fold, tr, te) for (fold, (tr, te)) in folds)
    else:
        results = [_run_fold(fold, tr, te) for (fold, (tr, te)) in folds]

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
        best_params_path = config_dict.get("paths", {}).get("best_params", {}).get("riemann_loso", "decoding/best_params/riemann_loso.jsonl")
        path = results_dir / best_params_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for r in best_param_records:
                f.write(json.dumps(r) + "\n")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pooled, per_subj = compute_metrics(y_true, y_pred, np.asarray(groups_ordered))
    logger.info(f"Model 2 (Riemann) pooled: r={pooled['pearson_r']:.3f}, R2={pooled['r2']:.3f}, EVS={pooled['explained_variance']:.3f}, avg_r_Fz={pooled['avg_subject_r_fisher_z']:.3f}")

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        if config_dict is None:
            config_dict = {}
        paths = config_dict.get("paths", {})
        predictions_path = paths.get("predictions", {}).get("riemann_loso", "decoding/predictions/riemann_loso.tsv")
        metrics_path = paths.get("per_subject_metrics", {}).get("riemann_loso", "decoding/per_subject_metrics/riemann_loso.tsv")
        indices_path = paths.get("indices", {}).get("riemann_loso", "decoding/indices/riemann_loso.tsv")
        
        write_tsv(pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "group": np.asarray(groups_ordered),
            "fold": fold_ids,
            "trial_index": test_indices_order,
        }), results_dir / predictions_path)

        write_tsv(per_subj, results_dir / metrics_path)

        idx_df = pd.DataFrame({
            "subject_id": np.asarray(groups_ordered),
            "heldout_subject_id": np.asarray(groups_ordered),
            "fold": fold_ids,
            "trial_index": test_indices_order,
        })
        write_tsv(idx_df, results_dir / indices_path)

    return y_true, y_pred, pooled, per_subj


def riemann_export_cov_bins(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    export_dir: Path = None,
    plateau_window: Optional[Tuple[float, float]] = None,
) -> None:
    if export_dir is None:
        return
    if not check_pyriemann():
        logger.warning("pyriemann not installed; skipping Riemann exports.")
        return
    
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann

    config_local = load_settings()
    if plateau_window is None:
        plateau_window = tuple(config_local.get("decoding.analysis.riemann.plateau_window", [3.0, 10.5]))

    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)

    eeg_sets = []
    for _sub, _epochs, _y in tuples:
        eeg_chs = [ch for ch in _epochs.info["ch_names"] if _epochs.get_channel_types(picks=[ch])[0] == "eeg"]
        eeg_sets.append(set(eeg_chs))
    
    if len(eeg_sets) == 0:
        logger.warning("No subjects found for Riemann export.")
        return
    
    local_common_chs = sorted(list(set.intersection(*eeg_sets))) if len(eeg_sets) > 1 else sorted(list(eeg_sets[0]))
    if len(local_common_chs) == 0:
        logger.warning("No common EEG channels across subjects; skipping export.")
        return
    
    first_ep = tuples[0][1]
    first_eeg_order = [ch for ch in first_ep.info["ch_names"] if first_ep.get_channel_types(picks=[ch])[0] == "eeg"]
    canonical_chs = [ch for ch in first_eeg_order if ch in set(local_common_chs)]

    subj_data = []
    for sub, epochs, y in tuples:
        epochs_use = epochs.copy().pick(canonical_chs)
        epochs_use.reorder_channels(canonical_chs)
        
        if len(epochs_use.info["ch_names"]) != len(canonical_chs):
            logger.warning(f"Skipping {sub}: channel count {len(epochs_use.info['ch_names'])} != canonical {len(canonical_chs)}")
            continue
            
        if plateau_window is not None:
            tmin, tmax = plateau_window
            epochs_use.crop(tmin=tmin, tmax=tmax)
            
        X = epochs_use.get_data(picks="eeg")
        if len(X) != len(y):
            logger.error(f"X-y length mismatch: X={len(X)}, y={len(y)}. Skipping fold.")
            continue
        if len(X) < 2:
            continue
            
        yv = pd.to_numeric(y, errors="coerce").to_numpy()
        n = min(len(X), len(yv))
        subj_data.append((sub, X[:n], yv[:n], epochs_use.info))

    if not subj_data:
        logger.warning("No subject data available for Riemann export.")
        return

    y_all = np.concatenate([v[2] for v in subj_data])
    y_all = y_all[np.isfinite(y_all)]
    config_local = load_settings()
    min_trials_for_binning = config_local.get("decoding.constants.min_trials_for_binning", 4)
    if len(y_all) < min_trials_for_binning:
        logger.warning("Too few valid ratings for binning; skipping export.")
        return
    
    percentile_low = config_local.get("decoding.constants.percentile_low", 33.3)
    percentile_high = config_local.get("decoding.constants.percentile_high", 66.7)
    min_trials_per_bin = config_local.get("decoding.constants.min_trials_per_bin", 2)
    q_low = float(np.percentile(y_all, percentile_low))
    q_high = float(np.percentile(y_all, percentile_high))

    cov_means_low = []
    cov_means_high = []
    for sub, X, yv, _info in subj_data:
        mask_low = np.isfinite(yv) & (yv <= q_low)
        mask_high = np.isfinite(yv) & (yv >= q_high)
        if mask_low.sum() >= min_trials_per_bin:
            cov_low = Covariances(estimator="oas").transform(X[mask_low])
            C_low = mean_riemann(cov_low)
            cov_means_low.append(C_low)
        if mask_high.sum() >= min_trials_per_bin:
            cov_high = Covariances(estimator="oas").transform(X[mask_high])
            C_high = mean_riemann(cov_high)
            cov_means_high.append(C_high)

    if len(cov_means_low) == 0 or len(cov_means_high) == 0:
        logger.warning("Insufficient trials in bins for export.")
        return

    M_low = mean_riemann(np.stack(cov_means_low))
    M_high = mean_riemann(np.stack(cov_means_high))
    D = M_high - M_low

    ch_names = canonical_chs
    if D.shape[0] != len(ch_names):
        ch_preview = ch_names[:5] if ch_names else []
        logger.warning(f"Skipping Riemann aggregation: data channels (D:{D.shape[0]}) != ch_names ({len(ch_names)}). First 5 ch_names: {ch_preview}")
        return

    dfD = pd.DataFrame(D, index=ch_names, columns=ch_names)
    write_tsv(dfD, export_dir / "riemann_cov_diff_matrix_global.tsv", index=True)

    A = D.copy()
    np.fill_diagonal(A, 0.0)
    node_strength = np.sum(np.abs(A), axis=1)

    ns_df = pd.DataFrame({"channel": ch_names, "node_strength": node_strength})
    write_tsv(ns_df, export_dir / "riemann_node_strength_global.tsv")


def riemann_export_cov_bins_per_fold(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    export_dir: Path = None,
    plateau_window: Optional[Tuple[float, float]] = None,
) -> None:
    if export_dir is None:
        return
    if not check_pyriemann():
        logger.warning("pyriemann not installed; skipping per-fold Riemann exports.")
        return
    
    from pyriemann.estimation import Covariances
    from pyriemann.utils.mean import mean_riemann

    config_local = load_settings()
    if plateau_window is None:
        plateau_window = tuple(config_local.get("decoding.analysis.riemann.plateau_window", [3.0, 10.5]))

    min_trials_for_binning = config_local.get("decoding.constants.min_trials_for_binning", 4)
    percentile_low = config_local.get("decoding.constants.percentile_low", 33.3)
    percentile_high = config_local.get("decoding.constants.percentile_high", 66.7)
    min_trials_per_bin = config_local.get("decoding.constants.min_trials_per_bin", 2)

    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)
    if not tuples:
        logger.warning("No subjects found for per-fold Riemann export.")
        return

    sub_ids = [sub for sub, _, _ in tuples]
    subj_map = {sub: (epochs, y) for sub, epochs, y in tuples}

    agg_ns_sum = {}
    agg_ns_count = {}

    for i_test, test_sub in enumerate(sub_ids, start=1):
        train_subs = [s for s in sub_ids if s != test_sub]
        if len(train_subs) == 0:
            continue

        train_sets = []
        for s in train_subs:
            ep, _y = subj_map[s]
            train_sets.append({ch for ch in ep.info["ch_names"] if ep.get_channel_types(picks=[ch])[0] == "eeg"})
        common_train = sorted(list(set.intersection(*train_sets))) if len(train_sets) > 1 else sorted(list(train_sets[0]))
        
        if len(common_train) == 0:
            logger.warning(f"Per-fold export: no common training channels for fold {i_test} (test={test_sub}); skipping.")
            continue

        first_ep = subj_map[train_subs[0]][0]
        first_eeg_order = [ch for ch in first_ep.info["ch_names"] if first_ep.get_channel_types(picks=[ch])[0] == "eeg"]
        canonical = [ch for ch in first_eeg_order if ch in set(common_train)]
        if len(canonical) == 0:
            continue

        subj_data = []
        for s in train_subs:
            epochs, y = subj_map[s]
            ep_use = epochs.copy().pick(canonical)
            ep_use.reorder_channels(canonical)
            if plateau_window is not None:
                tmin, tmax = plateau_window
                ep_use.crop(tmin=tmin, tmax=tmax)
            X = ep_use.get_data(picks="eeg")
            if len(X) != len(y):
                logger.warning(f"Per-fold export: X-y mismatch for {s}; skipping.")
                continue
            yv = pd.to_numeric(y, errors="coerce").to_numpy()
            subj_data.append((s, X, yv))
        if not subj_data:
            continue

        y_all = np.concatenate([v[2] for v in subj_data])
        y_all = y_all[np.isfinite(y_all)]
        if len(y_all) < min_trials_for_binning:
            continue
        q_low = float(np.percentile(y_all, percentile_low))
        q_high = float(np.percentile(y_all, percentile_high))

        cov_means_low = []
        cov_means_high = []
        for s, X, yv in subj_data:
            mask_low = np.isfinite(yv) & (yv <= q_low)
            mask_high = np.isfinite(yv) & (yv >= q_high)
            if mask_low.sum() >= min_trials_per_bin:
                cov_low = Covariances(estimator="oas").transform(X[mask_low])
                cov_means_low.append(mean_riemann(cov_low))
            if mask_high.sum() >= min_trials_per_bin:
                cov_high = Covariances(estimator="oas").transform(X[mask_high])
                cov_means_high.append(mean_riemann(cov_high))
        if len(cov_means_low) == 0 or len(cov_means_high) == 0:
            continue

        M_low = mean_riemann(np.stack(cov_means_low))
        M_high = mean_riemann(np.stack(cov_means_high))
        D = M_high - M_low

        A = D.copy()
        np.fill_diagonal(A, 0.0)
        node_strength = np.sum(np.abs(A), axis=1)

        for ch, val in zip(canonical, node_strength):
            agg_ns_sum[ch] = agg_ns_sum.get(ch, 0.0) + float(val)
            agg_ns_count[ch] = agg_ns_count.get(ch, 0) + 1

        write_tsv(pd.DataFrame({"channel": canonical, "node_strength": node_strength}),
                  export_dir / f"riemann_node_strength_fold-{i_test:02d}.tsv")

    if len(agg_ns_sum) > 0:
        ch_list = sorted(agg_ns_sum.keys())
        vals = np.array([agg_ns_sum[ch] / max(1, agg_ns_count.get(ch, 1)) for ch in ch_list], dtype=float)

        write_tsv(pd.DataFrame({
            "channel": ch_list,
            "mean_node_strength": vals,
            "n_folds": [int(agg_ns_count[ch]) for ch in ch_list],
        }), export_dir / "riemann_node_strength_train_only_mean_across_folds.tsv")


def run_riemann_band_limited_decoding(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    results_dir: Path = None,
    export_dir: Path = None,
    bands: Optional[List[Tuple[float, float]]] = None,
    n_jobs: int = -1,
    seed: int = 42,
    outer_n_jobs: int = 1,
    config_dict: Optional[dict] = None,
) -> Optional[dict]:
    if not check_pyriemann():
        logger.warning("pyriemann not installed; skipping band-limited Riemann decoding.")
        return None
    
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    if bands is None:
        bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]

    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)

    summary = {}
    r_vals = []
    labels = []

    trial_records = []
    y_all_list = []
    groups_list = []
    subj_to_epochs = {}
    subj_to_y = {}
    
    for sub, epochs, y in tuples:
        n = min(len(epochs), len(y))
        if n == 0:
            continue
        subj_to_epochs[sub] = epochs
        subj_to_y[sub] = pd.to_numeric(y.iloc[:n], errors="coerce")
        for ti in range(n):
            trial_records.append((sub, ti))
            y_all_list.append(float(subj_to_y[sub].iloc[ti]))
            groups_list.append(sub)
    
    if len(trial_records) == 0:
        logger.warning("No trials available for band-limited decoding.")
        return None

    y_all_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    logo = LeaveOneGroupOut()
    folds = list(enumerate(logo.split(np.arange(len(trial_records)), groups=groups_arr), start=1))

    config_local = load_settings()

    for (l_freq, h_freq) in bands:
        label = f"{int(l_freq)}-{int(h_freq)}Hz"
        labels.append(label)

        pipe = _create_riemann_pipeline()
        param_grid = _get_riemann_param_grid()

        def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
            set_random_seeds(seed, fold)
            
            train_subs_seq = [trial_records[i][0] for i in train_idx]
            train_subjects = list({s for s in train_subs_seq if s is not None})
            
            train_subject_eeg_chs = {}
            for s in train_subjects:
                train_subject_eeg_chs[s] = [
                    ch for ch in subj_to_epochs[s].info["ch_names"]
                    if subj_to_epochs[s].get_channel_types(picks=[ch])[0] == "eeg"
                ]
            
            common_chs = train_subject_eeg_chs[train_subjects[0]] if len(train_subjects) == 1 else list(set.intersection(*[set(train_subject_eeg_chs[s]) for s in train_subjects]))
            
            y_train = y_all_arr[train_idx]
            y_test = y_all_arr[test_idx]
            train_sel = np.isfinite(y_train)
            test_sel = np.isfinite(y_test)
            train_idx_f = train_idx[train_sel]
            test_idx_f = test_idx[test_sel]
            y_train = y_train[train_sel]
            y_test = y_test[test_sel]
            
            if not common_chs:
                logger.warning(f"Band {label} fold {fold}: No common EEG channels across training subjects. Skipping fold.")
                return {
                    "fold": fold,
                    "y_true": y_test.tolist(),
                    "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx_f].tolist(),
                    "test_idx": test_idx_f.tolist(),
                    "best_params_rec": None,
                }
            
            common_chs = sorted(common_chs)
            logger.info(f"Band {label} fold {fold}: Using {len(common_chs)} common EEG channels across {len(train_subjects)} training subjects.")
            
            test_subjects = list({trial_records[i][0] for i in test_idx})
            all_subjects = list(set(train_subjects + test_subjects))
            
            train_data_cache = {}
            test_data_cache = {}
            
            for s in all_subjects:
                subject_chs = subj_to_epochs[s].info["ch_names"]
                missing_chs = [ch for ch in common_chs if ch not in subject_chs]
                
                if missing_chs:
                    logger.error(f"Band {label} fold {fold}: Subject {s} missing {len(missing_chs)}/{len(common_chs)} required channels: {missing_chs[:5]}{'...' if len(missing_chs) > 5 else ''}")
                    return {
                        "fold": fold,
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                        "test_idx": test_idx_f.tolist(),
                        "best_params_rec": None,
                    }
                
                ep = subj_to_epochs[s].copy().pick(common_chs)
                logger.info(f"Band {label} fold {fold}: applying {l_freq}-{h_freq}Hz filter to subject {s} (sfreq={ep.info['sfreq']:.1f}Hz)")
                ep = ep.copy()
                ep.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
                
                try:
                    data = ep.get_data(picks="eeg", reject_by_annotation=None)
                except TypeError:
                    data = ep.get_data(picks="eeg")
                
                data = data.astype(np.float32)
                
                if s in train_subjects:
                    train_data_cache[s] = data
                if s in test_subjects:
                    test_data_cache[s] = data

            def _extract_block(indices: np.ndarray) -> np.ndarray:
                X_list = []
                for i in indices:
                    sub_i, ti = trial_records[int(i)]
                    if i in train_idx_f:
                        X_i = train_data_cache[sub_i][ti]
                    else:
                        X_i = test_data_cache[sub_i][ti]
                    X_list.append(X_i)
                return np.stack(X_list, axis=0)

            X_train = _extract_block(train_idx_f)
            X_test = _extract_block(test_idx_f)
            train_groups = groups_arr[train_idx_f]

            n_unique = len(np.unique(train_groups))
            best = None
            best_params_rec = None
            scoring = create_scoring_dict()
            refit_metric = 'r'
            
            if n_unique >= 2:
                n_splits_inner = get_inner_cv_splits(config_dict, n_unique, config=config_local)
                inner_cv = GroupKFold(n_splits=n_splits_inner)
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    refit=refit_metric,
                )
                gs = grid_search_with_warning_logging(
                    gs, X_train, y_train,
                    fold_info=f"Riemann fold {fold}",
                    logger=logger,
                    groups=train_groups
                )
                best = gs.best_estimator_
                
                cv_results = pd.DataFrame(gs.cv_results_)
                best_params_rec = create_best_params_record(f"Riemann_{label}", fold, cv_results, gs)
            else:
                best = clone(pipe)
                best.fit(X_train, y_train)

            y_pred = best.predict(X_test)

            return {
                "fold": fold,
                "y_true": y_test.tolist(),
                "y_pred": np.asarray(y_pred).tolist(),
                "groups": groups_arr[test_idx_f].tolist(),
                "test_idx": test_idx_f.tolist(),
                "best_params_rec": best_params_rec,
            }

        if outer_n_jobs and outer_n_jobs != 1 and len(folds) > 1:
            results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(fold, tr, te) for (fold, (tr, te)) in folds)
        else:
            results = [_run_fold(fold, tr, te) for (fold, (tr, te)) in folds]

        results = sorted(results, key=lambda r: r["fold"])
        y_true_b, y_pred_b, groups_ordered, test_indices_order, fold_ids = [], [], [], [], []
        best_param_records = []
        for rec in results:
            y_true_b.extend(rec["y_true"])
            y_pred_b.extend(rec["y_pred"])
            groups_ordered.extend(rec["groups"])
            test_indices_order.extend(rec["test_idx"])
            fold_ids.extend([rec["fold"]] * len(rec["test_idx"]))
            if rec.get("best_params_rec") is not None:
                best_param_records.append(rec["best_params_rec"])

        y_true_b = np.asarray(y_true_b)
        y_pred_b = np.asarray(y_pred_b)
        pooled, per_subj = compute_metrics(y_true_b, y_pred_b, np.asarray(groups_ordered))
        summary[label] = pooled
        r_vals.append(pooled.get("pearson_r", np.nan))

        if results_dir is not None:
            if config_dict is None:
                config_dict = {}
            paths = config_dict.get("paths", {})
            predictions_template = paths.get("predictions", {}).get("riemann_band_template", "decoding/predictions/riemann_{label}_loso.tsv")
            indices_template = paths.get("indices", {}).get("riemann_band_template", "decoding/indices/riemann_{label}_loso.tsv")
            best_params_template = paths.get("best_params", {}).get("riemann_band_template", "decoding/best_params/riemann_{label}_loso.jsonl")
            
            predictions_path = predictions_template.replace("{label}", label)
            indices_path = indices_template.replace("{label}", label)
            best_params_path = best_params_template.replace("{label}", label)
            
            dfp = pd.DataFrame({
                "y_true": y_true_b,
                "y_pred": y_pred_b,
                "group": np.asarray(groups_ordered),
                "fold": fold_ids,
                "trial_index": test_indices_order,
                "band": label,
            })
            write_tsv(dfp, results_dir / predictions_path)
            
            write_tsv(pd.DataFrame({
                "group": np.asarray(groups_ordered),
                "fold": fold_ids,
                "trial_index": test_indices_order,
            }), results_dir / indices_path)
            
            if len(best_param_records) > 0:
                path = results_dir / best_params_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    for r in best_param_records:
                        f.write(json.dumps(r) + "\n")

    if results_dir is not None:
        if config_dict is None:
            config_dict = {}
        summaries_path = config_dict.get("paths", {}).get("summaries", {}).get("riemann_bands", "decoding/summaries/riemann_bands.json")
        with open(results_dir / summaries_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def run_riemann_sliding_window(
    deriv_root: Path,
    subjects: Optional[List[str]] = None,
    task: str = "",
    results_dir: Path = None,
    plateau_window: Tuple[float, float] = None,
    window_len: float = None,
    step: float = None,
    n_jobs: int = -1,
    seed: int = 42,
    outer_n_jobs: int = 1,
    config_dict: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    if not check_pyriemann():
        logger.warning("pyriemann not installed; skipping sliding-window Riemann analysis.")
        return None
    
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    config_local = load_settings()
    if plateau_window is None:
        plateau_window = tuple(config_local.get("decoding.analysis.riemann.plateau_window", [3.0, 10.5]))
    if window_len is None:
        window_len = config_local.get("decoding.analysis.riemann.sliding_window.window_len", 0.75)
    if step is None:
        step = config_local.get("decoding.analysis.riemann.sliding_window.step", 0.25)

    tuples, _ = load_epochs_with_targets(deriv_root, subjects=subjects, task=task)

    tmin_pl, tmax_pl = plateau_window
    starts = []
    t = tmin_pl
    while t + window_len <= tmax_pl + 1e-6:
        starts.append(t)
        t += step

    records = []

    trial_records = []
    y_all_list = []
    groups_list = []
    subj_to_epochs = {}
    subj_to_y = {}
    
    for sub, epochs, y in tuples:
        n = min(len(epochs), len(y))
        if n == 0:
            continue
        subj_to_epochs[sub] = epochs
        subj_to_y[sub] = pd.to_numeric(y.iloc[:n], errors="coerce")
        for ti in range(n):
            trial_records.append((sub, ti))
            y_all_list.append(float(subj_to_y[sub].iloc[ti]))
            groups_list.append(sub)
    
    if not trial_records:
        return None

    y_arr = np.asarray(y_all_list)
    groups_arr = np.asarray(groups_list)
    logo = LeaveOneGroupOut()
    folds = list(enumerate(logo.split(np.arange(len(trial_records)), groups=groups_arr), start=1))

    for t0 in starts:
        pipe = _create_riemann_pipeline()
        param_grid = {
            "cov__estimator": ["oas", "lwf"],
            "ridge__alpha": [1e-2, 1e-1, 1, 10],
        }

        def _run_fold(fold: int, train_idx: np.ndarray, test_idx: np.ndarray):
            set_random_seeds(seed, fold)
            
            train_subs = list({trial_records[i][0] for i in train_idx if trial_records[i][0] is not None})
            
            train_eeg_chs = {}
            for s in train_subs:
                train_eeg_chs[s] = [
                    ch for ch in subj_to_epochs[s].info["ch_names"]
                    if subj_to_epochs[s].get_channel_types(picks=[ch])[0] == "eeg"
                ]
            
            common_chs = train_eeg_chs[train_subs[0]] if len(train_subs) == 1 else list(set.intersection(*[set(train_eeg_chs[s]) for s in train_subs]))
            
            y_train = y_arr[train_idx]
            y_test = y_arr[test_idx]
            train_sel = np.isfinite(y_train)
            test_sel = np.isfinite(y_test)
            train_idx_f = train_idx[train_sel]
            test_idx_f = test_idx[test_sel]
            y_train = y_train[train_sel]
            y_test = y_test[test_sel]
            
            if not common_chs:
                logger.warning(f"Sliding t0={t0:.2f}s fold {fold}: No common EEG channels across training subjects. Skipping fold.")
                return {
                    "y_true": y_test.tolist(),
                    "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                    "groups": groups_arr[test_idx_f].tolist(),
                }
            
            common_chs = sorted(common_chs)
            logger.info(f"Sliding t0={t0:.2f}s fold {fold}: Using {len(common_chs)} common EEG channels across {len(train_subs)} training subjects.")
            
            test_subs = list({trial_records[i][0] for i in test_idx})
            all_subs = list(set(train_subs + test_subs))
            
            train_cache: Dict[str, np.ndarray] = {}
            test_cache: Dict[str, np.ndarray] = {}
            
            for s in all_subs:
                subject_chs = subj_to_epochs[s].info["ch_names"]
                missing_chs = [ch for ch in common_chs if ch not in subject_chs]
                
                if missing_chs:
                    logger.error(f"Sliding t0={t0:.2f}s fold {fold}: Subject {s} missing {len(missing_chs)}/{len(common_chs)} required channels: {missing_chs[:5]}{'...' if len(missing_chs) > 5 else ''}")
                    logger.error(f"Sliding t0={t0:.2f}s fold {fold}: Cannot maintain tangent-space dimensionality. Skipping fold.")
                    return {
                        "y_true": y_test.tolist(),
                        "y_pred": np.full(len(y_test), np.nan, dtype=float).tolist(),
                        "groups": groups_arr[test_idx_f].tolist(),
                    }
                
                ep = subj_to_epochs[s].copy().pick(common_chs)
                logger.info(f"Sliding t0={t0:.2f}s fold {fold}: cropping subject {s} to [{t0:.2f}, {t0 + window_len:.2f}]s (sfreq={ep.info['sfreq']:.1f}Hz)")
                ep = ep.copy().crop(tmin=t0, tmax=t0 + window_len)
                logger.info(f"Sliding t0={t0:.2f}s fold {fold}: subject {s} cropped successfully")
                    
                try:
                    data = ep.get_data(picks="eeg", reject_by_annotation=None)
                except TypeError:
                    data = ep.get_data(picks="eeg")
                
                data = data.astype(np.float32)
                
                if s in train_subs:
                    train_cache[s] = data
                if s in test_subs:
                    test_cache[s] = data

            def _extract_block(indices: np.ndarray) -> np.ndarray:
                X_list: List[np.ndarray] = []
                for i in indices:
                    sub_i, ti = trial_records[int(i)]
                    cache = train_cache if i in train_idx_f else test_cache
                    X_list.append(cache[sub_i][ti])
                return np.stack(X_list, axis=0)

            X_train = _extract_block(train_idx_f)
            X_test = _extract_block(test_idx_f)
            train_groups = groups_arr[train_idx_f]

            n_unique = len(np.unique(train_groups))
            best = None
            scoring = create_scoring_dict()
            refit_metric = 'r'
            
            if n_unique >= 2:
                n_splits_inner = get_inner_cv_splits(config_dict, n_unique, config=config_local)
                inner_cv = GroupKFold(n_splits=n_splits_inner)
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=inner_cv,
                    n_jobs=n_jobs,
                    refit=refit_metric,
                )
                gs = grid_search_with_warning_logging(
                    gs, X_train, y_train,
                    fold_info=f"Riemann fold {fold}",
                    logger=logger,
                    groups=train_groups
                )
                best = gs.best_estimator_
                
                cv_results = pd.DataFrame(gs.cv_results_)
                best_params_rec = create_best_params_record("Riemann_sliding", fold, cv_results, gs)
                logger.debug(f"Sliding t0={t0} fold {fold}: best params by r = {best_params_rec['best_params_by_r']}")
                logger.debug(f"Sliding t0={t0} fold {fold}: best params by neg_mse = {best_params_rec['best_params_by_neg_mse']}")
            else:
                best = clone(pipe)
                best.fit(X_train, y_train)

            y_pred = best.predict(X_test)

            return {"y_true": y_test.tolist(), "y_pred": np.asarray(y_pred).tolist(), "groups": groups_arr[test_idx_f].tolist()}

        if outer_n_jobs and outer_n_jobs != 1 and len(folds) > 1:
            results = Parallel(n_jobs=outer_n_jobs, prefer="threads")(delayed(_run_fold)(fold, tr, te) for (fold, (tr, te)) in folds)
        else:
            results = [_run_fold(fold, tr, te) for (fold, (tr, te)) in folds]

        y_true_sw, y_pred_sw, groups_ordered = [], [], []
        for rec in results:
            y_true_sw.extend(rec["y_true"])
            y_pred_sw.extend(rec["y_pred"])
            groups_ordered.extend(rec["groups"])

        y_true_sw = np.asarray(y_true_sw)
        y_pred_sw = np.asarray(y_pred_sw)
        pooled, _per_subj = compute_metrics(y_true_sw, y_pred_sw, np.asarray(groups_ordered))
        records.append({
            "t_center": float(t0 + window_len / 2.0),
            "pearson_r": pooled.get("pearson_r", np.nan),
            "r2": pooled.get("r2", np.nan),
            "explained_variance": pooled.get("explained_variance", np.nan),
        })

    if len(records) == 0:
        return None

    df = pd.DataFrame.from_records(records).sort_values("t_center").reset_index(drop=True)
    if results_dir is not None:
        if config_dict is None:
            config_dict = {}
        summaries_path = config_dict.get("paths", {}).get("summaries", {}).get("riemann_sliding_window", "decoding/summaries/riemann_sliding_window.json")
        with open(results_dir / summaries_path, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="list"), f, indent=2)

    return df

