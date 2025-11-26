from __future__ import annotations

import json
import logging
import random as pyrandom
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning, norm
import scipy.stats
from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, make_scorer, explained_variance_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import clone
from joblib import Parallel, delayed

from ..config.loader import load_settings, get_nested_value, get_config_value, ensure_config
from ..data.loading import _load_features_and_targets

logger = logging.getLogger(__name__)


###################################################################
# Constants Loading
###################################################################

def _load_decoding_constants(config=None):
    config = ensure_config(config)
    
    if config is None:
        raise ValueError("Config is required. Cannot load decoding constants without config.")
    
    cv_config = config.get("decoding.cv")
    constants_config = config.get("decoding.constants")
    preprocessing_config = config.get("decoding.preprocessing")
    models_config = config.get("decoding.models")
    analysis_config = config.get("decoding.analysis")
    
    if not all([cv_config, constants_config, preprocessing_config, models_config, analysis_config]):
        missing = [name for name, val in [
            ("decoding.cv", cv_config),
            ("decoding.constants", constants_config),
            ("decoding.preprocessing", preprocessing_config),
            ("decoding.models", models_config),
            ("decoding.analysis", analysis_config)
        ] if val is None]
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    
    elasticnet_config = models_config.get("elasticnet")
    rf_config = models_config.get("random_forest")
    
    if not elasticnet_config:
        raise ValueError("decoding.models.elasticnet not found in config.")
    if not rf_config:
        raise ValueError("decoding.models.random_forest not found in config.")
    
    return {
        "min_trials_for_within_subject": cv_config["min_trials_for_within_subject"],
        "default_n_splits": cv_config["default_n_splits"],
        "default_n_bins": cv_config["default_n_bins"],
        "variance_threshold": constants_config["variance_threshold"],
        "min_variance_for_correlation": constants_config["min_variance_for_correlation"],
        "bca_ci_min_samples": constants_config["bca_ci_min_samples"],
        "bca_ci_min_jackknife": constants_config["bca_ci_min_jackknife"],
        "bootstrap_ci_alpha_low": constants_config["bootstrap_ci_alpha_low"],
        "bootstrap_ci_alpha_high": constants_config["bootstrap_ci_alpha_high"],
        "bootstrap_n": analysis_config["bootstrap_n"],
        "imputer_strategy": preprocessing_config["imputer_strategy"],
        "power_transformer_method": preprocessing_config["power_transformer_method"],
        "power_transformer_standardize": preprocessing_config["power_transformer_standardize"],
        "elasticnet_max_iter": elasticnet_config["max_iter"],
        "elasticnet_tol": elasticnet_config["tol"],
        "elasticnet_selection": elasticnet_config["selection"],
        "rf_n_estimators": rf_config["n_estimators"],
        "rf_bootstrap": rf_config["bootstrap"],
        "min_channels_for_riemann_absolute": constants_config["min_channels_for_riemann_absolute"],
    }


###################################################################
# Pipeline Factory Functions
###################################################################

def create_base_preprocessing_pipeline(include_scaling: bool = True, config: Optional[Dict[str, Any]] = None) -> Pipeline:
    constants = _load_decoding_constants(config)
    steps = [
        ("impute", SimpleImputer(strategy=constants["imputer_strategy"])),
        ("var", VarianceThreshold(threshold=constants["variance_threshold"]))
    ]
    if include_scaling:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)

def create_elasticnet_pipeline(seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> Pipeline:
    if seed is None:
        config = ensure_config(config)
        seed = int(get_config_value(config, "project.random_state", 42))
    
    constants = _load_decoding_constants(config)
    base_steps = create_base_preprocessing_pipeline(include_scaling=True, config=config).steps
    base_steps.append((
        "regressor", 
        TransformedTargetRegressor(
            regressor=ElasticNet(
                random_state=seed,
                max_iter=constants["elasticnet_max_iter"],
                tol=constants["elasticnet_tol"],
                selection=constants["elasticnet_selection"]
            ),
            transformer=PowerTransformer(
                method=constants["power_transformer_method"],
                standardize=constants["power_transformer_standardize"]
            )
        )
    ))
    return Pipeline(base_steps)

def create_rf_pipeline(n_estimators: Optional[int] = None, n_jobs: int = 1, seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None) -> Pipeline:
    if seed is None:
        config = ensure_config(config)
        seed = int(get_config_value(config, "project.random_state", 42))
    
    constants = _load_decoding_constants(config)
    if n_estimators is None:
        n_estimators = constants["rf_n_estimators"]
    steps = [
        ("impute", SimpleImputer(strategy=constants["imputer_strategy"])),
        ("var", VarianceThreshold(threshold=constants["variance_threshold"])),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=seed,
            bootstrap=constants["rf_bootstrap"]
        ))
    ]
    return Pipeline(steps)


###################################################################
# Warning Logging Helpers
###################################################################

def _log_warning(warning, fold_info: str, context: str, logger=None):
    warning_types = {
        ConvergenceWarning: "ConvergenceWarning",
        ConstantInputWarning: "ConstantInputWarning"
    }
    warning_type_name = warning_types.get(type(warning.message), "Warning")
    msg = f"Fold {fold_info}: {warning_type_name} {context} - {warning.message}"
    if logger:
        logger.warning(msg)
    else:
        print(f"[WARNING] {msg}")

def fit_with_warning_logging(estimator, X, y, fold_info: str = "", logger=None):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimator.fit(X, y)
        for warning in w:
            if isinstance(warning.message, (ConvergenceWarning, ConstantInputWarning)):
                _log_warning(warning, fold_info, "", logger)
    return estimator

def grid_search_with_warning_logging(grid, X, y, fold_info: str = "", logger=None, **fit_params):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid.fit(X, y, **fit_params)
        for warning in w:
            if isinstance(warning.message, (ConvergenceWarning, ConstantInputWarning)):
                _log_warning(warning, fold_info, "during GridSearchCV", logger)
    return grid


###################################################################
# Cross-Validation Helpers
###################################################################

def make_pearsonr_scorer():
    return make_scorer(lambda yt, yp: safe_pearsonr(np.asarray(yt), np.asarray(yp))[0], greater_is_better=True)

def create_scoring_dict() -> dict:
    return {'r': make_pearsonr_scorer(), 'neg_mse': 'neg_mean_squared_error'}


###################################################################
# Decoding Data Utilities
###################################################################


def compute_subject_level_r(pred_df: pd.DataFrame) -> tuple[float, list[tuple[str, float]], float, float]:
    """Compute subject-level Pearson r, aggregate via Fisher z weighted by trial counts, and CI."""
    per_subject: list[tuple[str, float, int]] = []
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

    r_vals = np.asarray([r for _, r, _ in per_subject], dtype=float)
    n_trials_vals = np.asarray([n for _, _, n in per_subject], dtype=int)
    r_vals = np.clip(r_vals, -0.999999, 0.999999)
    z_vals = np.arctanh(r_vals)
    
    weights = np.maximum(n_trials_vals - 3, 1.0)
    weights = weights / weights.sum()
    mean_z = float(np.average(z_vals, weights=weights))
    agg_r = float(np.tanh(mean_z))

    per_subject_list = [(subj, r) for subj, r, _ in per_subject]
    
    ci_low = np.nan
    ci_high = np.nan
    if z_vals.size > 1:
        weighted_var = np.average((z_vals - mean_z) ** 2, weights=weights)
        se = float(np.sqrt(weighted_var / z_vals.size))
        if np.isfinite(se) and se > 0:
            delta = 1.96 * se
            ci_low = float(np.tanh(mean_z - delta))
            ci_high = float(np.tanh(mean_z + delta))
    
    return agg_r, per_subject_list, ci_low, ci_high


def load_plateau_matrix(
    subjects: List[str],
    task: str,
    deriv_root: Path,
    config,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Load plateau features and targets for the requested subjects."""
    X_blocks = []
    y_blocks = []
    groups = []
    trial_ids = []
    feature_cols: Optional[List[str]] = None

    for sub in subjects:
        _, plateau_df, _, y, _ = _load_features_and_targets(sub, task, deriv_root, config)
        if plateau_df is None or plateau_df.empty:
            logger.warning(f"No plateau features for sub-{sub}; skipping.")
            continue
        if feature_cols is None:
            feature_cols = plateau_df.columns.tolist()
        else:
            missing = set(feature_cols) - set(plateau_df.columns)
            if missing:
                raise ValueError(f"Feature mismatch for sub-{sub}; missing columns: {sorted(missing)}")

        X_block = plateau_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y_block = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

        X_blocks.append(X_block)
        y_blocks.append(y_block)
        groups.extend([sub] * len(y_block))
        trial_ids.extend(list(range(len(y_block))))

    if not X_blocks:
        raise RuntimeError("No subjects with usable plateau features were loaded.")

    X = np.vstack(X_blocks)
    y_all = np.concatenate(y_blocks)
    groups_arr = np.asarray(groups)
    meta = pd.DataFrame({"subject_id": groups_arr, "trial_id": trial_ids})
    
    finite_mask = np.isfinite(y_all)
    if not np.all(finite_mask):
        n_dropped = np.sum(~finite_mask)
        logger.info(f"Dropping {n_dropped} non-finite targets out of {len(y_all)} total trials")
        X = X[finite_mask]
        y_all = y_all[finite_mask]
        groups_arr = groups_arr[finite_mask]
        meta = meta.loc[finite_mask].reset_index(drop=True)
    
    return X, y_all, groups_arr, feature_cols or [], meta


def build_elasticnet_param_grid(config) -> dict:
    en_cfg = config.get("decoding", {}).get("models", {}).get("elasticnet", {})
    alphas = en_cfg.get("alpha_grid", [0.01, 0.1, 1.0, 10.0])
    l1_ratios = en_cfg.get("l1_ratio_grid", [0.1, 0.5, 0.9])
    return {
        "regressor__regressor__alpha": alphas,
        "regressor__regressor__l1_ratio": l1_ratios,
    }


def set_random_seeds(seed: int, fold: int) -> None:
    np.random.seed(seed + fold)
    pyrandom.seed(seed + fold)

def create_loso_folds(X: np.ndarray, groups: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    logo = LeaveOneGroupOut()
    return [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(
            logo.split(X, groups=groups), start=1
        )
    ]

def create_within_subject_folds(
    groups: np.ndarray,
    blocks_all: Optional[np.ndarray],
    inner_cv_splits: int,
    seed: int,
    min_trials_for_within_subject: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
    if min_trials_for_within_subject is None:
        constants = _load_decoding_constants(config)
        min_trials_for_within_subject = constants["min_trials_for_within_subject"]
    folds: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
    fold_counter = 0
    unique_subs = [str(s) for s in np.unique(groups)]
    
    for subject in unique_subs:
        subject_indices = np.where(groups == subject)[0]
        n_samples = len(subject_indices)
        
        if n_samples < min_trials_for_within_subject:
            logger.warning(f"Subject {subject}: <{min_trials_for_within_subject} trials, skipping within-subject KFold")
            continue
        
        n_splits = min(inner_cv_splits, n_samples)
        if n_splits < min_trials_for_within_subject:
            n_splits = min_trials_for_within_subject
        
        if blocks_all is None:
            logger.warning(
                f"Subject {subject}: missing run_id for block-aware CV. "
                f"Skipping subject (cannot construct valid CV splits without run identifiers)."
            )
            continue
        
        subject_blocks = blocks_all[subject_indices]
        block_cv, effective_splits = create_block_aware_cv_for_within_subject(
            subject_blocks, n_splits=n_splits, random_state=seed
        )
        
        if block_cv is None:
            logger.warning(
                f"Subject {subject}: insufficient unique blocks for GroupKFold. "
                f"Skipping subject (cannot construct valid CV splits)."
            )
            continue
        
        logger.info(f"Subject {subject}: using block-aware GroupKFold ({effective_splits} splits)")
        for train_local, test_local in block_cv.split(subject_indices, groups=subject_blocks):
            fold_counter += 1
            train_idx = subject_indices[train_local]
            test_idx = subject_indices[test_local]
            folds.append((fold_counter, train_idx, test_idx, subject))
    
    return folds

def create_inner_cv(train_groups: np.ndarray, inner_cv_splits: int) -> GroupKFold:
    n_unique_groups = len(np.unique(train_groups))
    n_splits = min(inner_cv_splits, n_unique_groups)
    return GroupKFold(n_splits=n_splits)

def create_block_aware_inner_cv(
    blocks_train: np.ndarray,
    n_splits_inner: int,
    seed: int,
    fold: int,
    subject: str,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    block_inner_cv, effective_inner_splits = create_block_aware_cv_for_within_subject(
        blocks_train, n_splits=n_splits_inner, random_state=seed + fold
    )
    
    if block_inner_cv is None:
        return None
    
    cv_splits = list(block_inner_cv.split(np.arange(len(blocks_train)), groups=blocks_train))
    logger.info(f"Within fold {fold} ({subject}): using block-aware inner CV ({effective_inner_splits} splits)")
    return cv_splits

def validate_blocks_for_within_subject(blocks_all: Optional[np.ndarray], groups: np.ndarray) -> Optional[np.ndarray]:
    if blocks_all is None:
        return None
    
    if len(blocks_all) != len(groups):
        logger.warning(
            "Within-subject CV: provided run_id array length %d does not match sample count %d; "
            "ignoring run-aware blocks.",
            len(blocks_all),
            len(groups),
        )
        return None
    
    logger.info("Within-subject CV: using provided run identifiers for block-aware splits.")
    return blocks_all

def validate_block_aware_cv_required(
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    subject: str,
    fold: int,
) -> None:
    if cv_splits is not None:
        return
    
    error_msg = (
        f"Within-subject CV for subject {subject} fold {fold}: block information unavailable. "
        f"Cannot construct valid CV splits without block/run identifiers. "
        f"Contiguous splits would create temporal data leakage and invalidate results. "
        f"Ensure run_id/block information is available in events data or skip within-subject analysis."
    )
    logger.error(error_msg)
    raise ValueError(error_msg)

def should_parallelize_folds(outer_n_jobs: int, n_folds: int) -> bool:
    return outer_n_jobs and outer_n_jobs != 1 and n_folds > 1

def execute_folds_parallel(folds: List[Tuple], fold_func, outer_n_jobs: int) -> List[dict]:
    if should_parallelize_folds(outer_n_jobs, len(folds)):
        return Parallel(n_jobs=outer_n_jobs, prefer="threads")(
            delayed(fold_func)(fold, train_idx, test_idx) for (fold, train_idx, test_idx) in folds
        )
    return [fold_func(fold, train_idx, test_idx) for (fold, train_idx, test_idx) in folds]

def determine_inner_n_jobs(outer_n_jobs: int, n_jobs: int) -> int:
    return 1 if (outer_n_jobs and outer_n_jobs != 1) else n_jobs

def get_inner_cv_splits(config_dict: Optional[dict], n_unique_groups: int, config: Optional[Dict[str, Any]] = None) -> int:
    if config_dict is None:
        constants = _load_decoding_constants(config)
        inner_splits_default = constants["default_n_splits"]
    else:
        cv_config = config_dict.get("cv")
        if cv_config is None:
            raise ValueError("cv not found in config_dict")
        inner_splits_default = cv_config.get("inner_splits")
        if inner_splits_default is None:
            constants = _load_decoding_constants(config)
            inner_splits_default = constants["default_n_splits"]
    return int(np.clip(int(inner_splits_default), 2, n_unique_groups))

def get_min_channels_required(config_dict: Optional[dict], config: Optional[Dict[str, Any]] = None) -> int:
    constants = _load_decoding_constants(config)
    min_channels_absolute = constants["min_channels_for_riemann_absolute"]
    
    if config_dict is None:
        raise ValueError("config_dict is required for get_min_channels_required")
    analysis_config = config_dict.get("analysis")
    if analysis_config is None:
        raise ValueError("analysis not found in config_dict")
    riemann_config = analysis_config.get("riemann")
    if riemann_config is None:
        raise ValueError("analysis.riemann not found in config_dict")
    min_ch = riemann_config.get("min_channels_for_fold")
    if min_ch is None:
        raise ValueError("analysis.riemann.min_channels_for_fold not found in config_dict")
    return max(min_channels_absolute, int(min_ch))

def aggregate_fold_results(results: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    sorted_results = sorted(results, key=lambda r: r["fold"])
    
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    groups_ordered: List[str] = []
    test_indices_order: List[int] = []
    fold_ids: List[int] = []
    
    for record in sorted_results:
        n_test_samples = len(record["test_idx"])
        y_true_all.extend(record["y_true"])
        y_pred_all.extend(record["y_pred"])
        groups_ordered.extend(record["groups"])
        test_indices_order.extend(record["test_idx"])
        fold_ids.extend([record["fold"]] * n_test_samples)
    
    return (
        np.asarray(y_true_all),
        np.asarray(y_pred_all),
        groups_ordered,
        test_indices_order,
        fold_ids
    )

def extract_best_params_from_cv_results(cv_results: pd.DataFrame) -> Tuple[dict, dict, float, float]:
    def _get_best_by_rank(rank_col: str, params_col: str, score_col: str):
        idx = cv_results[rank_col].idxmin()
        return cv_results.loc[idx, params_col], float(cv_results.loc[idx, score_col])
    
    best_params_r, best_score_r = _get_best_by_rank('rank_test_r', 'params', 'mean_test_r')
    best_params_mse, best_score_neg_mse = _get_best_by_rank('rank_test_neg_mse', 'params', 'mean_test_neg_mse')
    
    return best_params_r, best_params_mse, best_score_r, best_score_neg_mse

def create_best_params_record(
    model_name: str,
    fold: int,
    cv_results: pd.DataFrame,
    grid_search: GridSearchCV,
    subject: Optional[str] = None,
    heldout_subjects: Optional[List[str]] = None,
) -> dict:
    best_params_r, best_params_mse, best_score_r, best_score_neg_mse = extract_best_params_from_cv_results(cv_results)
    
    record = {
        "model": model_name or None,
        "fold": int(fold),
        "best_params_by_r": best_params_r,
        "best_params_by_neg_mse": best_params_mse,
        "best_score_r": best_score_r,
        "best_score_neg_mse": best_score_neg_mse,
        "best_params": grid_search.best_params_,
    }
    
    if subject is not None:
        record["subject"] = subject
    if heldout_subjects is not None:
        record["heldout_subjects"] = heldout_subjects
    
    return record

def get_best_params_for_fold(
    best_params_map: dict,
    heldout_subject: Optional[str],
    fold: int
) -> dict:
    if heldout_subject and heldout_subject in best_params_map:
        return best_params_map[heldout_subject]
    return best_params_map.get(fold, {})

def create_empty_fold_result(
    fold: int,
    test_idx: np.ndarray,
    y_all_arr: np.ndarray,
    groups_arr: np.ndarray
) -> dict:
    return {
        "fold": fold,
        "y_true": y_all_arr[test_idx].tolist(),
        "y_pred": np.full(len(test_idx), np.nan, dtype=float).tolist(),
        "groups": groups_arr[test_idx].tolist(),
        "test_idx": test_idx.tolist(),
        "best_params_rec": None,
    }

def create_stratified_cv_by_binned_targets(y: np.ndarray, n_splits: Optional[int] = None, n_bins: Optional[int] = None, random_state: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
    if random_state is None:
        config = ensure_config(config)
        random_state = int(get_config_value(config, "project.random_state", 42))
    
    constants = _load_decoding_constants(config)
    if n_splits is None:
        n_splits = constants["default_n_splits"]
    if n_bins is None:
        n_bins = constants["default_n_bins"]
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    if pd.isna(y_binned).all():
        y_binned = np.zeros_like(y, dtype=int)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), y_binned

def create_block_aware_cv_for_within_subject(blocks: np.ndarray, n_splits: Optional[int] = None, random_state: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
    constants = _load_decoding_constants(config)
    if n_splits is None:
        n_splits = constants["default_n_splits"]
    unique_blocks = np.unique(blocks[~pd.isna(blocks)])
    n_unique_blocks = len(unique_blocks)
    
    if n_unique_blocks < 2:
        logger.warning(
            f"Block-aware CV: insufficient unique blocks ({n_unique_blocks}). "
            f"GroupKFold requires at least 2 unique groups. Returning None to skip subject."
        )
        return None, 0
    
    effective_n_splits = min(n_splits, n_unique_blocks)
    if effective_n_splits < n_splits:
        logger.info(
            f"Block-aware CV: reduced splits from {n_splits} to {effective_n_splits} "
            f"due to {n_unique_blocks} unique blocks (values: {unique_blocks.tolist()})"
        )
    
    cv = GroupKFold(n_splits=effective_n_splits)
    return cv, effective_n_splits

def compute_within_subject_trial_index(groups: np.ndarray) -> np.ndarray:
    groups_arr = np.asarray(groups)
    trial_idx = np.zeros(len(groups_arr), dtype=int)
    counters = {}
    
    for i, subject in enumerate(groups_arr):
        trial_idx[i] = counters.get(subject, 0)
        counters[subject] = counters.get(subject, 0) + 1
    
    return trial_idx

def log_cv_adjacency_info(indices: np.ndarray, fold_name: str = "") -> None:
    if len(indices) < 2:
        return
    s = np.sort(indices)
    total_pairs = len(s) - 1
    if total_pairs <= 0:
        return
    consec = np.sum(np.diff(s) == 1)
    ratio = consec / total_pairs
    if ratio > 0.5:
        logger.info(f"{fold_name}: High temporal adjacency detected ({consec}/{total_pairs}, {ratio:.2f}). Consider block-aware CV.")
    elif consec > 0:
        logger.debug(f"{fold_name}: Some temporal adjacency ({consec}/{total_pairs}, {ratio:.2f})")
    rng = s[-1] - s[0] + 1
    density = len(indices) / rng if rng > 0 else 1.0
    if density < 0.5:
        logger.debug(f"{fold_name}: Sparse index distribution (density={density:.2f})")

def blocks_constant_per_subject(blocks: Optional[np.ndarray], groups: np.ndarray) -> bool:
    if blocks is None:
        return True
    b = np.asarray(blocks)
    if len(b) != len(groups):
        return True
    g = np.asarray(groups)
    any_valid = False
    for s in np.unique(g):
        vals = b[g == s]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        any_valid = True
        unique_vals = np.unique(vals)
        logger.debug(f"blocks_constant_per_subject: subject {s} has {len(unique_vals)} unique blocks: {unique_vals.tolist()}")
        if unique_vals.size > 1:
            return False
    result = not any_valid
    logger.debug(f"blocks_constant_per_subject: returning {result} (any_valid={any_valid})")
    return result


###################################################################
# Statistical Helpers
###################################################################

def safe_pearsonr(x: np.ndarray, y: np.ndarray, config: Optional[dict] = None, min_variance: Optional[float] = None) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        return np.nan, np.nan
    
    valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid_mask.sum() < 2:
        return np.nan, np.nan
    
    x_valid = x_arr[valid_mask]
    y_valid = y_arr[valid_mask]
    
    if len(x_valid) < 2 or len(y_valid) < 2:
        return np.nan, np.nan
    
    if min_variance is None:
        if config is None:
            try:
                constants = _load_decoding_constants(None)
                min_variance = constants["min_variance_for_correlation"]
            except (ValueError, KeyError):
                min_variance = 1e-10
        else:
            try:
                constants = _load_decoding_constants(config)
                min_variance = constants["min_variance_for_correlation"]
            except (ValueError, KeyError):
                min_variance = 1e-10
    
    if np.var(x_valid, ddof=1) < min_variance or np.var(y_valid, ddof=1) < min_variance:
        return np.nan, np.nan
    
    correlation, p_value = pearsonr(x_valid, y_valid)
    
    if not (np.isfinite(correlation) and np.isfinite(p_value)):
        return np.nan, np.nan
    
    return float(np.clip(correlation, -1.0, 1.0)), float(p_value)

def partial_corr_xy_given_z(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    
    if z_arr.ndim == 1:
        z_arr = z_arr[:, None]
    
    valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.all(np.isfinite(z_arr), axis=1)
    x_valid = x_arr[valid_mask]
    y_valid = y_arr[valid_mask]
    z_valid = z_arr[valid_mask]
    
    min_samples = max(3, z_valid.shape[1] + 2)
    if len(x_valid) < min_samples:
        return np.nan
    
    design_matrix = np.c_[np.ones(len(z_valid)), z_valid]
    condition_number = np.linalg.cond(design_matrix)
    if condition_number > 1e8:
        return np.nan
    
    if np.linalg.matrix_rank(design_matrix, tol=1e-10) < design_matrix.shape[1]:
        return np.nan
    
    x_coef, _, x_rank, _ = np.linalg.lstsq(design_matrix, x_valid, rcond=1e-10)
    y_coef, _, y_rank, _ = np.linalg.lstsq(design_matrix, y_valid, rcond=1e-10)
    
    if x_rank < design_matrix.shape[1] or y_rank < design_matrix.shape[1]:
        return np.nan
    
    x_residuals = x_valid - design_matrix @ x_coef
    y_residuals = y_valid - design_matrix @ y_coef
    
    x_var = np.var(x_residuals, ddof=1) if len(x_residuals) > 1 else 0.0
    y_var = np.var(y_residuals, ddof=1) if len(y_residuals) > 1 else 0.0
    variance_tolerance = 1e-12 * max(np.var(x_valid), np.var(y_valid), 1.0)
    
    if x_var < variance_tolerance or y_var < variance_tolerance:
        return np.nan
    
    correlation, _ = safe_pearsonr(x_residuals, y_residuals)
    return float(correlation)

def cluster_bootstrap_subjects(
    df: pd.DataFrame, 
    subject_col: str, 
    n_boot: Optional[int] = None, 
    seed: Optional[int] = None, 
    func=None, 
    config: Optional[Any] = None
) -> Tuple[float, Tuple[float, float]]:
    if n_boot is None:
        config = ensure_config(config)
        n_boot = int(get_config_value(config, "decoding.analysis.bootstrap_n", 1000))
    
    if seed is None:
        config = ensure_config(config)
        seed = int(get_config_value(config, "project.random_state", 42))
    
    rng = np.random.default_rng(seed)
    theta0 = float(func(df))
    subs = df[subject_col].unique().tolist()
    thetas = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        parts = [df[df[subject_col] == s] for s in pick]
        bs = pd.concat(parts, axis=0, ignore_index=True)
        thetas[b] = float(func(bs))
    
    thetas = thetas[np.isfinite(thetas)]
    if len(thetas) == 0:
        return theta0, (float("nan"), float("nan"))
    
    constants = _load_decoding_constants(config)
    alpha_low = constants["bootstrap_ci_alpha_low"]
    alpha_high = constants["bootstrap_ci_alpha_high"]
    
    lo_p, hi_p = np.percentile(thetas, [alpha_low * 100, alpha_high * 100]).tolist()
    
    jk_vals = []
    for s in subs:
        d_jk = df[df[subject_col] != s]
        if len(d_jk) == 0:
            continue
        v = float(func(d_jk))
        if np.isfinite(v):
            jk_vals.append(v)
    
    jk_vals = np.asarray(jk_vals, float)
    if len(jk_vals) >= 3 and np.isfinite(theta0):
        tdot = float(np.mean(jk_vals))
        num = np.sum((tdot - jk_vals) ** 3)
        den = 6.0 * (np.sum((tdot - jk_vals) ** 2) ** 1.5 + 1e-12)
        a = float(num / den) if np.isfinite(num) and np.isfinite(den) and den != 0 else 0.0
        
        z0 = float(norm.ppf((np.sum(thetas < theta0) + 1e-12) / (len(thetas) + 2e-12))) if np.isfinite(theta0) else 0.0
        
        def _bca_quant(alpha: float) -> float:
            zalpha = float(norm.ppf(alpha))
            adj = z0 + (z0 + zalpha) / max(1e-12, (1 - a * (z0 + zalpha)))
            return float(norm.cdf(adj))
        
        a1 = _bca_quant(alpha_low)
        a2 = _bca_quant(alpha_high)
        lo_bca, hi_bca = np.percentile(thetas, [100 * a1, 100 * a2]).tolist()
    else:
        lo_bca, hi_bca = lo_p, hi_p
    
    return theta0, (float(lo_bca), float(hi_bca))


###################################################################
# Metrics Computation
###################################################################

def compute_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r_calibration": np.nan, "n_samples": 0}
    slope, intercept, r_cal, p_val, std_err = scipy.stats.linregress(yp, yt)
    return {"slope": float(slope), "intercept": float(intercept), "r_calibration": float(r_cal), "p_value": float(p_val), "std_err": float(std_err), "n_samples": int(len(yt))}

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Tuple[dict, pd.DataFrame]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    groups = np.asarray(groups)[mask]
    
    if len(y_true) < 2:
        pooled = {
            "pearson_r": np.nan,
            "p_value": np.nan,
            "r2": np.nan,
            "explained_variance": np.nan,
            "avg_subject_r_fisher_z": np.nan,
        }
        per_subject = pd.DataFrame(columns=["group", "pearson_r", "p_value", "r2", "explained_variance", "n_trials"])
        return pooled, per_subject
    
    r, p = safe_pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    pooled = {"pearson_r": float(r), "p_value": float(p), "r2": float(r2), "explained_variance": float(evs)}

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    rows = []
    r_list = []
    
    for g, d in df.groupby("group"):
        if len(d) < 2:
            continue
        
        yt, yp = d["y_true"].values, d["y_pred"].values
        mask_subj = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[mask_subj], yp[mask_subj]
        
        if len(yt) < 2:
            continue
            
        rg, pg = safe_pearsonr(yt, yp)
        r2g = r2_score(yt, yp)
        evsg = explained_variance_score(yt, yp)
        rows.append({"group": g, "pearson_r": float(rg), "p_value": float(pg), "r2": float(r2g), "explained_variance": float(evsg), "n_trials": int(len(yt))})
        r_list.append(float(rg))
    
    per_subject = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    
    r_values = []
    n_trials_list = []
    for _, row in per_subject.iterrows():
        r_val = row["pearson_r"]
        n_trials = row["n_trials"]
        if np.isfinite(r_val) and -0.999 < r_val < 0.999:
            r_values.append(float(r_val))
            n_trials_list.append(int(n_trials))
    
    if len(r_values) > 0:
        r_vals = np.asarray(r_values)
        n_trials_vals = np.asarray(n_trials_list)
        r_vals = np.clip(r_vals, -0.999999, 0.999999)
        z_vals = np.arctanh(r_vals)
        weights = np.maximum(n_trials_vals - 3, 1.0)
        weights = weights / weights.sum()
        avg_r_fisher_z = float(np.tanh(np.average(z_vals, weights=weights)))
    else:
        avg_r_fisher_z = np.nan
    
    pooled["avg_subject_r_fisher_z"] = float(avg_r_fisher_z)
    return pooled, per_subject

def bootstrap_pooled_metrics_by_subject(pred_df: pd.DataFrame, n_boot: Optional[int] = None, seed: Optional[int] = None, config_dict: Optional[dict] = None, config: Optional[Dict[str, Any]] = None) -> dict:
    if seed is None:
        config = ensure_config(config)
        seed = int(get_config_value(config, "project.random_state", 42))
    rng = np.random.default_rng(seed)
    if n_boot is None:
        if config_dict is not None:
            analysis_config = config_dict.get("analysis")
            if analysis_config is None:
                raise ValueError("analysis not found in config_dict")
            n_boot = analysis_config.get("bootstrap_n")
            if n_boot is None:
                raise ValueError("analysis.bootstrap_n not found in config_dict")
            n_boot = int(n_boot)
        else:
            constants = _load_decoding_constants(config)
            n_boot = constants["bootstrap_n"]
    
    r_point, _ = safe_pearsonr(pred_df["y_true"].values, pred_df["y_pred"].values)
    r2_point = r2_score(pred_df["y_true"].values, pred_df["y_pred"].values)
    subs = pred_df["subject_id"].astype(str).unique()
    
    if len(subs) < 2:
        return {"r_point": float(r_point), "r2_point": float(r2_point), "r_ci": [np.nan, np.nan], "r2_ci": [np.nan, np.nan], "n_bootstrap": 0}
    
    r_vals = []
    r2_vals = []
    for _ in range(n_boot):
        boot_subs = rng.choice(subs, size=len(subs), replace=True)
        boot_blocks = [pred_df[pred_df["subject_id"].astype(str) == s] for s in boot_subs]
        boot_df = pd.concat(boot_blocks, axis=0, ignore_index=True)
        y_t = boot_df["y_true"].values
        y_p = boot_df["y_pred"].values
        r, _ = safe_pearsonr(y_t, y_p)
        r2 = r2_score(y_t, y_p) if len(y_t) > 1 else np.nan
        r_vals.append(float(r))
        r2_vals.append(float(r2))
    
    _r_finite = np.asarray([v for v in r_vals if np.isfinite(v)], dtype=float)
    _r2_finite = np.asarray([v for v in r2_vals if np.isfinite(v)], dtype=float)

    constants = _load_decoding_constants(config)
    ci_alpha_low = constants["bootstrap_ci_alpha_low"]
    ci_alpha_high = constants["bootstrap_ci_alpha_high"]
    
    r_ci_pct = [float(np.percentile(_r_finite, ci_alpha_low * 100)), float(np.percentile(_r_finite, ci_alpha_high * 100))] if _r_finite.size > 0 else [np.nan, np.nan]
    r2_ci_pct = [float(np.percentile(_r2_finite, ci_alpha_low * 100)), float(np.percentile(_r2_finite, ci_alpha_high * 100))] if _r2_finite.size > 0 else [np.nan, np.nan]

    def _bca_ci(thetas: np.ndarray, theta0: float, jk_vals: np.ndarray, alpha_low: Optional[float] = None, alpha_high: Optional[float] = None) -> Tuple[float, float]:
        if alpha_low is None:
            alpha_low = constants["bootstrap_ci_alpha_low"]
        if alpha_high is None:
            alpha_high = constants["bootstrap_ci_alpha_high"]
        thetas = np.asarray(thetas, float)
        thetas = thetas[np.isfinite(thetas)]
        jk_vals = np.asarray(jk_vals, float)
        jk_vals = jk_vals[np.isfinite(jk_vals)]
        min_samples = constants["bca_ci_min_samples"]
        min_jackknife = constants["bca_ci_min_jackknife"]
        if thetas.size < min_samples or jk_vals.size < min_jackknife or not np.isfinite(theta0):
            return float("nan"), float("nan")
        
        tdot = float(np.mean(jk_vals))
        
        # Check for degenerate case where jackknife values have no variance
        jk_var = np.var(jk_vals)
        if jk_var < 1e-15:
            # Fall back to percentile method when BCa acceleration is undefined
            lo, hi = np.percentile(thetas, [alpha_low * 100, alpha_high * 100]).tolist()
            return float(lo), float(hi)
        
        num = np.sum((tdot - jk_vals) ** 3)
        den = 6.0 * (np.sum((tdot - jk_vals) ** 2) ** 1.5 + 1e-12)
        a = float(num / den) if np.isfinite(num) and np.isfinite(den) and den != 0 else 0.0
        
        # Bound acceleration parameter to prevent extreme adjustments
        a = float(np.clip(a, -0.5, 0.5))
        
        z0 = float(norm.ppf((np.sum(thetas < theta0) + 1e-12) / (len(thetas) + 2e-12)))
        
        def _adj(alpha: float) -> float:
            zalpha = float(norm.ppf(alpha))
            denom = 1 - a * (z0 + zalpha)
            # Prevent division by zero or extreme values
            if abs(denom) < 1e-6:
                return alpha
            adj = z0 + (z0 + zalpha) / denom
            return float(norm.cdf(adj))
        
        a1 = _adj(alpha_low)
        a2 = _adj(alpha_high)
        q_low = 100 * np.clip(a1, 0.0, 1.0)
        q_high = 100 * np.clip(a2, 0.0, 1.0)
        lo, hi = np.percentile(thetas, [q_low, q_high]).tolist()
        return float(lo), float(hi)

    r_jk, r2_jk = [], []
    for s in subs:
        d_jk = pred_df[pred_df["subject_id"].astype(str) != str(s)]
        if len(d_jk) < 2:
            continue
        rj, _ = safe_pearsonr(d_jk["y_true"].values, d_jk["y_pred"].values)
        r2j = r2_score(d_jk["y_true"].values, d_jk["y_pred"].values) if len(d_jk) > 1 else np.nan
        r_jk.append(float(rj))
        r2_jk.append(float(r2j))
    r_jk = np.asarray(r_jk, float)
    r2_jk = np.asarray(r2_jk, float)

    r_ci_bca = _bca_ci(_r_finite, float(r_point), r_jk)
    r2_ci_bca = _bca_ci(_r2_finite, float(r2_point), r2_jk)
    r_ci = [float(r_ci_bca[0]) if np.isfinite(r_ci_bca[0]) else r_ci_pct[0],
            float(r_ci_bca[1]) if np.isfinite(r_ci_bca[1]) else r_ci_pct[1]]
    r2_ci = [float(r2_ci_bca[0]) if np.isfinite(r2_ci_bca[0]) else r2_ci_pct[0],
             float(r2_ci_bca[1]) if np.isfinite(r2_ci_bca[1]) else r2_ci_pct[1]]
    
    return {
        "r_point": float(r_point),
        "r2_point": float(r2_point),
        "r_ci": r_ci,
        "r2_ci": r2_ci,
        "n_bootstrap": int(n_boot),
        "n_subjects": int(len(subs)),
    }

def per_subject_pearson_and_spearman(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, d in pred_df.groupby("subject_id"):
        if len(d) < 2:
            continue
        r, _ = safe_pearsonr(d["y_true"].values, d["y_pred"].values)
        try:
            rho, _ = spearmanr(d["y_true"].values, d["y_pred"].values)
        except (ValueError, RuntimeError):
            rho = np.nan
        rows.append({"subject_id": sid, "pearson_r": float(r), "spearman_rho": float(rho)})
    return pd.DataFrame(rows)


###################################################################
# Shared Utilities for Decoding Modules
###################################################################

def _save_best_params(best_param_records: List[dict], best_params_log_path: Optional[Path]) -> None:
    if best_params_log_path is None:
        return
    best_params_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_log_path, "a", encoding="utf-8") as f:
        for rec in best_param_records:
            f.write(json.dumps(rec) + "\n")


def _fit_with_inner_cv(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    config_dict: Optional[dict],
    fold: int,
    n_jobs: int,
    logger: logging.Logger,
    random_state: int,
    inner_splits: Optional[int] = None,
) -> Tuple[Pipeline, Optional[pd.DataFrame]]:
    config_local = load_settings()
    n_unique = len(np.unique(groups_train))
    min_subjects_for_loso = config_local.get("analysis.min_subjects_for_group", 2)
    
    if n_unique >= min_subjects_for_loso:
        if inner_splits is not None:
            n_splits_inner = inner_splits
        else:
            n_splits_inner = get_inner_cv_splits(config_dict, n_unique, config=config_local)
        inner_cv = create_inner_cv(groups_train, n_splits_inner)
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
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=refit_metric,
        )
        gs = grid_search_with_warning_logging(
            gs, X_train, y_train,
            fold_info=f"fold {fold}",
            logger=logger,
            groups=groups_train
        )
        cv_results = pd.DataFrame(gs.cv_results_)
        return gs.best_estimator_, cv_results
    else:
        return _fit_default_pipeline(pipe, X_train, y_train, fold, random_state), None


def _fit_default_pipeline(pipe: Pipeline, X_train: np.ndarray, y_train: np.ndarray, fold: int, random_state: Optional[int] = None) -> Pipeline:
    logger.warning(f"Fold {fold}: Insufficient subjects for inner CV, using default pipeline")
    fitted = clone(pipe)
    if random_state is not None:
        regressor_step = fitted.named_steps.get("regressor")
        if regressor_step is not None and hasattr(regressor_step, "regressor") and hasattr(regressor_step.regressor, "random_state"):
            regressor_step.regressor.random_state = random_state
    fitted.fit(X_train, y_train)
    return fitted


def _predict_and_log(
    estimator: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    logger: logging.Logger,
) -> np.ndarray:
    y_pred = estimator.predict(X_test)
    r = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else np.nan
    logger.info(f"Fold {fold}: test r={r:.3f}, n={len(y_test)}")
    return y_pred


def _create_riemann_pipeline() -> Pipeline:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    return Pipeline([("cov", Covariances()), ("ts", TangentSpace()), ("ridge", Ridge())])


def _get_riemann_param_grid() -> dict:
    return {"cov__estimator": ["oas", "lwf"], "ridge__alpha": [1e-2, 1e-1, 1, 10]}
