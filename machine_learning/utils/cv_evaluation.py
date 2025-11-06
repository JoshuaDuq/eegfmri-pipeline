from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr

from .cv_utils import (
    _build_groupkfold_with_temperature_coverage,
    compute_cv_splits,
    enforce_temperature_coverage,
    fold_group_label,
    make_inner_cv,
    make_outer_splitter,
    prepare_temperature_metadata,
)
from .metrics import compute_metrics

###################################################################
# Manual Grid Search with External Validation
###################################################################

@dataclass
class _ManualSearchResult:
    best_estimator_: Pipeline
    best_params_: Dict[str, Any]
    best_score_: float
    cv_results_: Dict[str, Any]
    refit_validation_indices_: Optional[np.ndarray]


def _pipeline_supports_external_validation(pipeline: Pipeline) -> Tuple[bool, str, Any]:
    if not isinstance(pipeline, Pipeline) or not pipeline.steps:
        return False, "", None
    final_name, final_estimator = pipeline.steps[-1]
    if getattr(final_estimator, "supports_external_validation", False):
        return True, final_name, final_estimator
    return False, final_name, final_estimator


def _safe_index_rows(data: Any, indices: np.ndarray) -> Any:
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    if isinstance(data, np.ndarray):
        return data[indices]
    if isinstance(data, list):
        return [data[int(i)] for i in indices]
    return np.asarray(data)[indices]


def _subset_value_for_indices(value: Any, indices: np.ndarray, total_length: int) -> Any:
    if hasattr(value, "iloc"):
        subset = value.iloc[indices]
        return subset.reset_index(drop=True)
    try:
        length = len(value)
    except TypeError:
        return value
    if length != total_length:
        return value
    if isinstance(value, np.ndarray):
        return value[indices]
    if isinstance(value, list):
        return [value[int(i)] for i in indices]
    if isinstance(value, tuple):
        return tuple(value[int(i)] for i in indices)
    try:
        return value[indices]
    except Exception:
        return value


def _subset_fit_params_for_step(
    fit_params: Dict[str, Any],
    step_name: str,
    indices: np.ndarray,
    total_length: int,
) -> Dict[str, Any]:
    subset: Dict[str, Any] = {}
    prefix = f"{step_name}__"
    for key, value in fit_params.items():
        if not key.startswith(prefix):
            continue
        param_key = key[len(prefix) :]
        subset[param_key] = _subset_value_for_indices(value, indices, total_length)
    return subset


def _manual_grid_search_with_external_validation(
    base_pipeline: Pipeline,
    param_grid: Dict[str, Sequence],
    inner_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fit_params: Dict[str, Any],
    logger: Optional[Any] = None,
) -> Optional[_ManualSearchResult]:
    supports_external, final_step_name, _ = _pipeline_supports_external_validation(base_pipeline)
    if not supports_external:
        return None

    all_param_sets = list(ParameterGrid(param_grid)) or [{}]
    results: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Optional[Dict[str, Any]] = None
    best_pipeline: Optional[Pipeline] = None
    total_length = len(X_train)
    best_refit_validation_indices: Optional[np.ndarray] = None
    log = logger

    for param_set in all_param_sets:
        param_set = dict(param_set)
        candidate_pipeline = clone(base_pipeline)
        if param_set:
            candidate_pipeline.set_params(**param_set)
        preprocessor_template = candidate_pipeline[:-1]
        final_estimator_template = candidate_pipeline.steps[-1][1]

        split_scores: List[float] = []
        split_best_indices: Optional[np.ndarray] = None
        candidate_failed = False
        for inner_train_idx, inner_val_idx in inner_splits:
            inner_train_idx = np.asarray(inner_train_idx, dtype=int)
            inner_val_idx = np.asarray(inner_val_idx, dtype=int)

            X_inner_train = _safe_index_rows(X_train, inner_train_idx)
            y_inner_train = _safe_index_rows(y_train, inner_train_idx)

            preprocessor = clone(preprocessor_template)
            Xt_inner_train = preprocessor.fit_transform(X_inner_train, y_inner_train)

            X_inner_val = _safe_index_rows(X_train, inner_val_idx)
            y_inner_val = _safe_index_rows(y_train, inner_val_idx)
            Xt_inner_val = preprocessor.transform(X_inner_val)

            final_estimator = clone(final_estimator_template)
            train_fit_params = _subset_fit_params_for_step(
                fit_params,
                final_step_name,
                inner_train_idx,
                total_length,
            )
            train_fit_params = dict(train_fit_params)

            try:
                final_estimator.fit(Xt_inner_train, y_inner_train, **train_fit_params)
            except ValueError as exc:
                if log is not None:
                    log.warning(
                        "Skipping parameter set %s due to estimator fit failure on inner fold "
                        "(train size=%d, val size=%d): %s",
                        param_set,
                        len(inner_train_idx),
                        len(inner_val_idx),
                        exc,
                    )
                candidate_failed = True
                break
            preds = final_estimator.predict(Xt_inner_val)
            split_scores.append(r2_score(np.asarray(y_inner_val), preds))
            if split_best_indices is None or split_scores[-1] >= max(split_scores[:-1] or [-np.inf]):
                split_best_indices = np.unique(np.asarray(inner_val_idx, dtype=int))

        if candidate_failed:
            results.append({"params": dict(param_set), "mean_test_score": float("-inf")})
            continue

        mean_score = float(np.mean(split_scores)) if split_scores else float("-inf")
        results.append({"params": dict(param_set), "mean_test_score": mean_score})

        if best_pipeline is None or mean_score > best_score:
            best_score = mean_score
            best_params = dict(param_set)
            best_pipeline = clone(base_pipeline)
            if param_set:
                best_pipeline.set_params(**param_set)
            refit_fit_params = dict(fit_params)
            if split_best_indices is not None and split_best_indices.size:
                refit_fit_params[f"{final_step_name}__validation_indices"] = np.asarray(
                    split_best_indices, dtype=int
                )
                best_refit_validation_indices = np.asarray(split_best_indices, dtype=int)
            else:
                best_refit_validation_indices = None
            try:
                if refit_fit_params:
                    best_pipeline.fit(X_train, y_train, **refit_fit_params)
                else:
                    best_pipeline.fit(X_train, y_train)
            except ValueError as exc:
                if log is not None:
                    log.warning(
                        "Refit failed for parameter set %s on full training data; skipping as candidate. "
                        "Error: %s",
                        param_set,
                        exc,
                    )
                best_pipeline = None
                best_params = None
                best_refit_validation_indices = None
                best_score = -np.inf
                continue
            else:
                if best_refit_validation_indices is not None:
                    estimator = best_pipeline.named_steps.get(final_step_name)
                    if estimator is not None and hasattr(estimator, "validation_indices_"):
                        setattr(
                            estimator,
                            "validation_indices_",
                            np.asarray(best_refit_validation_indices, dtype=int),
                        )

    if best_pipeline is None or best_params is None:
        return None

    cv_results = {
        "params": [dict(entry["params"]) for entry in results],
        "mean_test_score": np.array([entry["mean_test_score"] for entry in results], dtype=float),
    }

    return _ManualSearchResult(
        best_pipeline,
        best_params,
        float(best_score),
        cv_results,
        None if best_refit_validation_indices is None else np.asarray(best_refit_validation_indices, dtype=int),
    )


###################################################################
# Nested CV Evaluation
###################################################################

def nested_cv_evaluate(
    model_name: str,
    builder,
    param_grid: Dict[str, Sequence],
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    meta: pd.DataFrame,
    outer_groups: np.ndarray,
    run_groups: np.ndarray,
    random_state: int,
    n_jobs: int,
    logger: Any,
    log_progress: bool = True,
    fit_params_fn: Optional[Callable[[pd.DataFrame], Dict[str, Any]]] = None,
):
    predictions = np.zeros(len(y), dtype=float)
    fold_assignments = np.full(len(y), -1, dtype=int)
    fold_details: List[Dict[str, object]] = []

    temp_labels_all: Optional[np.ndarray]
    temp_levels: np.ndarray
    if "temp_celsius" in meta.columns:
        temp_labels_all, temp_levels = prepare_temperature_metadata(meta["temp_celsius"])
    else:
        temp_labels_all = None
        temp_levels = np.array([], dtype=str)

    outer_splitter, outer_groups_used, outer_desc = make_outer_splitter(
        outer_groups,
        random_state,
        stratify_labels=temp_labels_all,
        temperature_levels=temp_levels,
        logger=logger,
    )
    outer_splits = compute_cv_splits(
        outer_splitter,
        X,
        y,
        groups=outer_groups_used,
        stratify_labels=temp_labels_all,
    )
    if temp_labels_all is not None and len(temp_levels) > 1:
        enforce_temperature_coverage(
            temp_labels_all,
            outer_splits,
            temp_levels,
            context=f"Model {model_name} outer CV",
            check_train=True,
        )
    if log_progress:
        logger.info("Model %s | outer CV strategy: %s", model_name, outer_desc)
    inner_desc_record: Optional[str] = None

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if "subject" in meta.columns:
            test_subjects = meta.iloc[test_idx]["subject"].unique()
            train_subjects = meta.iloc[train_idx]["subject"].unique()
            leakage_subjects = np.intersect1d(test_subjects, train_subjects)
            if leakage_subjects.size:
                raise ValueError(
                    f"Outer CV leakage detected for model {model_name}: training fold contains held-out subjects {leakage_subjects}."
                )

        inner_temp_labels: Optional[np.ndarray]
        inner_temp_levels: np.ndarray
        if temp_labels_all is not None:
            inner_temp_labels = temp_labels_all[train_idx]
            inner_temp_levels = np.array(
                sorted(
                    {label for label in inner_temp_labels if not pd.isna(label)},
                    key=lambda v: str(v),
                ),
                dtype=object,
            )
        else:
            inner_temp_labels = None
            inner_temp_levels = np.array([], dtype=object)

        inner_cv, inner_groups, inner_desc = make_inner_cv(
            run_groups[train_idx],
            random_state,
            stratify_labels=inner_temp_labels,
            temperature_levels=inner_temp_levels,
            logger=logger,
        )
        inner_splits = compute_cv_splits(
            inner_cv,
            X_train,
            y_train,
            groups=inner_groups,
            stratify_labels=inner_temp_labels,
        )
        if inner_temp_labels is not None and len(inner_temp_levels) > 1:
            try:
                enforce_temperature_coverage(
                    inner_temp_labels,
                    inner_splits,
                    inner_temp_levels,
                    context=f"Model {model_name} inner CV (outer fold {fold_idx})",
                    check_train=True,
                    check_validation=True,
                    logger=logger,
                )
            except ValueError as exc:
                if logger is not None:
                    logger.warning(
                        "%s; attempting temperature-balanced GroupKFold fallback.",
                        exc,
                    )
                if inner_groups is None:
                    raise ValueError(
                        f"Model {model_name} inner CV (outer fold {fold_idx}): cannot build GroupKFold fallback "
                        "because group labels are unavailable."
                    ) from exc
                inner_cv, inner_splits, inner_desc = _build_groupkfold_with_temperature_coverage(
                    X_train,
                    y_train,
                    np.asarray(inner_groups),
                    inner_temp_labels,
                    inner_temp_levels,
                    random_state=random_state,
                    context=f"Model {model_name} inner CV (outer fold {fold_idx})",
                    logger=logger,
                )

        if inner_desc_record is None or inner_desc_record != inner_desc:
            inner_desc_record = inner_desc
            if log_progress:
                logger.info("Model %s | inner CV strategy: %s", model_name, inner_desc_record)

        if "subject" in meta.columns:
            test_subjects = meta.iloc[test_idx]["subject"].unique()
            for _, inner_val_idx in inner_splits:
                inner_val_subjects = meta.iloc[train_idx].iloc[inner_val_idx]["subject"].unique()
                leakage_subjects = np.intersect1d(test_subjects, inner_val_subjects)
                if leakage_subjects.size:
                    raise ValueError(
                        f"Inner CV leakage detected for model {model_name}: validation fold references outer test subjects {leakage_subjects}."
                    )

        estimator = builder(random_state, n_jobs)

        fit_params: Dict[str, Any] = {}
        if fit_params_fn is not None:
            raw_params = fit_params_fn(meta.iloc[train_idx])
            if raw_params:
                fit_params = {key: value for key, value in raw_params.items() if value is not None}
                for key, value in fit_params.items():
                    if hasattr(value, "__len__") and len(value) != len(X_train):
                        raise ValueError(
                            f"Fit parameter '{key}' has length {len(value)}, expected {len(X_train)}."
                        )

        manual_search = _manual_grid_search_with_external_validation(
            estimator,
            param_grid,
            inner_splits,
            X_train,
            y_train,
            fit_params,
            logger,
        )

        if manual_search is not None:
            search = manual_search
        else:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="r2",
                cv=inner_splits,
                n_jobs=n_jobs,
                refit=True,
            )

            if fit_params:
                search.fit(X_train, y_train, **fit_params)
            else:
                search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(X_test)

        predictions[test_idx] = y_pred
        fold_assignments[test_idx] = fold_idx

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        try:
            pear = pearsonr(y_test, y_pred)[0]
        except Exception:
            pear = float("nan")
        try:
            spear = spearmanr(y_test, y_pred)[0]
        except Exception:
            spear = float("nan")

        temp_counts = (
            meta.iloc[test_idx]["temp_celsius"].value_counts().sort_index().to_dict()
            if "temp_celsius" in meta.columns
            else {}
        )

        fold_info = {
            "name": model_name,
            "fold": fold_idx,
            "outer_group": fold_group_label(outer_groups_used, test_idx),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "test_r2": float(r2),
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_explained_variance": float(evs),
            "test_pearson_r": float(pear),
            "test_spearman_r": float(spear),
            "best_params": search.best_params_,
            "test_temp_counts": temp_counts,
        }
        fold_details.append(fold_info)
        if log_progress:
            logger.info(
                "Model %s | fold %d | group %s | R2=%.3f | Pearson=%.3f",
                model_name,
                fold_idx,
                fold_info["outer_group"],
                r2,
                pear,
            )
            if temp_counts:
                logger.info(
                    "Model %s | fold %d temperature counts: %s",
                    model_name,
                    fold_idx,
                    temp_counts,
                )

    metrics = compute_metrics(y.to_numpy(), predictions)
    logger.info(
        "Model %s | aggregated outer CV metrics: R2=%.3f | MAE=%.3f | RMSE=%.3f | Pearson=%.3f",
        model_name,
        metrics["r2"],
        metrics["mae"],
        metrics["rmse"],
        metrics.get("pearson_r", float("nan")),
    )

    return {
        "name": model_name,
        "predictions": predictions,
        "fold_assignments": fold_assignments,
        "fold_details": fold_details,
        "summary_metrics": metrics,
        "feature_names": list(feature_names),
        "outer_cv_desc": outer_desc,
        "inner_cv_desc": inner_desc_record or "Not determined",
    }


###################################################################
# Final Estimator Fitting
###################################################################

def fit_final_estimator(
    model_name: str,
    builder,
    param_grid: Dict[str, Sequence],
    X: pd.DataFrame,
    y: pd.Series,
    run_groups: np.ndarray,
    random_state: int,
    n_jobs: int,
    fit_params: Optional[Dict[str, Any]] = None,
    stratify_labels: Optional[Sequence[Any]] = None,
    logger: Optional[Any] = None,
):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    estimator = builder(random_state, n_jobs)

    stratify_array: Optional[np.ndarray]
    stratify_levels: np.ndarray
    if stratify_labels is not None:
        stratify_array, stratify_levels = prepare_temperature_metadata(stratify_labels)
    else:
        stratify_array = None
        stratify_levels = np.array([], dtype=str)

    effective_fit_params: Dict[str, Any] = {}
    if fit_params:
        effective_fit_params = {key: value for key, value in fit_params.items() if value is not None}
        for key, value in effective_fit_params.items():
            if hasattr(value, "__len__") and len(value) != len(X):
                raise ValueError(
                    f"Fit parameter '{key}' has length {len(value)}, expected {len(X)}."
                )

    unique_runs = np.unique(run_groups)
    splits: List[Tuple[np.ndarray, np.ndarray]]
    cv_desc: str

    if len(unique_runs) < 2:
        raise ValueError(
            f"{model_name} final refit requires at least two distinct run groups to validate hyper-parameters without leakage."
        )

    n_splits = min(5, len(unique_runs))
    if n_splits < 2:
        n_splits = 2
    splits = []
    cv_desc = ""
    if stratify_array is not None and len(stratify_levels) > 1:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits)
            splits = compute_cv_splits(
                splitter,
                X,
                y,
                groups=run_groups,
                stratify_labels=stratify_array,
            )
            enforce_temperature_coverage(
                stratify_array,
                splits,
                stratify_levels,
                context=f"{model_name} final refit",
                check_train=True,
                check_validation=True,
                logger=logger,
            )
            cv_desc = f"StratifiedGroupKFold(n_splits={n_splits}) on run labels"
        except ValueError as exc:
            if logger is not None:
                logger.warning(
                    "StratifiedGroupKFold unavailable for final refit due to run/temperature distribution: %s. Attempting GroupKFold fallback.",
                    exc,
                )
    if not splits:
        if stratify_array is not None and len(stratify_levels) > 1:
            _, splits, cv_desc = _build_groupkfold_with_temperature_coverage(
                X,
                y,
                run_groups,
                stratify_array,
                stratify_levels,
                random_state=random_state,
                context=f"{model_name} final refit",
                logger=logger,
            )
        else:
            splitter = GroupKFold(n_splits=n_splits)
            splits = compute_cv_splits(splitter, X, y, groups=run_groups)
            cv_desc = f"GroupKFold(n_splits={n_splits}) on run labels"

    manual_search = _manual_grid_search_with_external_validation(
        estimator,
        param_grid,
        splits,
        X,
        y,
        effective_fit_params,
        logger,
    )
    if manual_search is not None:
        return (
            manual_search.best_estimator_,
            manual_search.best_params_,
            float(manual_search.best_score_),
            cv_desc,
        )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="r2",
        cv=splits,
        n_jobs=n_jobs,
        refit=True,
    )

    if effective_fit_params:
        search.fit(X, y, **effective_fit_params)
    else:
        search.fit(X, y)

    return search.best_estimator_, search.best_params_, float(search.best_score_), cv_desc

