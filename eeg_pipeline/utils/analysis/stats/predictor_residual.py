"""
Predictor Residual (Subject-Level)
===================================

Defines a subject-level predictor residual:

    predictor_residual = outcome - f(predictor)

where f(·) is a flexible dose-response curve fitted to outcome ~ predictor.
This targets "response beyond stimulus intensity" for downstream feature
associations. Only valid when the predictor is continuous and ordinal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import get_config_value as _get_config_value
from .validation import assert_continuous_predictor


def _prepare_data(
    predictor: pd.Series,
    outcome: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """Align and validate predictor and outcome series."""
    predictor_values = pd.to_numeric(predictor, errors="coerce")
    outcome_values = pd.to_numeric(outcome, errors="coerce")
    common_index = predictor_values.index.intersection(outcome_values.index)
    predictor_aligned = predictor_values.loc[common_index]
    outcome_aligned = outcome_values.loc[common_index]
    return predictor_aligned, outcome_aligned, common_index


def _compute_residuals(
    outcome_values: pd.Series,
    predicted_values: pd.Series,
) -> pd.Series:
    """Compute residuals as outcome - predicted."""
    return outcome_values - predicted_values


def _fit_spline_model(
    predictor_values: pd.Series,
    outcome_values: pd.Series,
    config: Optional[Any],
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Fit the best spline model and return the fitted estimator."""
    import statsmodels.formula.api as smf

    degrees_of_freedom_candidates = _get_config_value(
        config, "behavior_analysis.predictor_residual.spline_df_candidates", [3, 4, 5]
    )

    model_data = pd.DataFrame(
        {
            "pred": predictor_values.to_numpy(dtype=float),
            "outcome": outcome_values.to_numpy(dtype=float),
        }
    )

    best_model = None
    best_metadata = None
    best_aic = np.inf

    for candidate_df in degrees_of_freedom_candidates:
        try:
            degrees_of_freedom = int(candidate_df)
        except (ValueError, TypeError):
            continue

        if degrees_of_freedom < 3:
            continue

        formula = f"outcome ~ bs(pred, df={degrees_of_freedom}, degree=3)"
        try:
            model = smf.ols(formula, data=model_data).fit()
        except (ValueError, TypeError, AttributeError):
            continue

        if not np.isfinite(model.aic):
            continue

        if model.aic < best_aic:
            best_model = model
            best_aic = model.aic
            best_metadata = {
                "df": degrees_of_freedom,
                "aic": float(model.aic),
                "bic": float(model.bic),
                "formula": formula,
            }

    if best_model is None:
        return None

    metadata = {
        "model": "spline",
        "status": "ok",
    }
    if best_metadata:
        metadata.update(best_metadata)
    try:
        metadata["r2"] = float(best_model.rsquared)
    except (AttributeError, ValueError, TypeError):
        pass
    return best_model, metadata


def _predict_spline_model(
    model: Any,
    predictor_values: pd.Series,
) -> pd.Series:
    """Predict outcomes from a fitted spline model."""
    prediction_data = pd.DataFrame({"pred": predictor_values.to_numpy(dtype=float)})
    predicted_values = model.predict(prediction_data)
    return pd.Series(
        np.asarray(predicted_values, dtype=float),
        index=predictor_values.index,
        dtype=float,
    )


def _fit_polynomial_model(
    predictor_values: pd.Series,
    outcome_values: pd.Series,
    config: Optional[Any],
) -> Tuple[np.poly1d, Dict[str, Any]]:
    """Fit a polynomial model and return the polynomial estimator."""
    degree = int(_get_config_value(config, "behavior_analysis.predictor_residual.poly_degree", 2))
    degree = max(1, min(degree, 5))

    try:
        coefficients = np.polyfit(
            predictor_values.to_numpy(dtype=float),
            outcome_values.to_numpy(dtype=float),
            deg=degree,
        )
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        raise ValueError("Predictor residual polynomial fit failed.") from error

    metadata = {
        "model": "poly",
        "status": "ok",
        "poly_degree": int(degree),
    }
    return np.poly1d(coefficients), metadata


def _predict_polynomial_model(
    polynomial: np.poly1d,
    predictor_values: pd.Series,
) -> pd.Series:
    """Predict outcomes from a fitted polynomial model."""
    predicted_values = polynomial(predictor_values.to_numpy(dtype=float))
    return pd.Series(
        np.asarray(predicted_values, dtype=float),
        index=predictor_values.index,
        dtype=float,
    )


def _resolve_predictor_residual_method(
    config: Optional[Any],
    *,
    key: str = "behavior_analysis.predictor_residual.method",
    default: str = "spline",
) -> str:
    """Resolve the configured predictor-residual model family."""
    method = str(_get_config_value(config, key, default)).strip().lower()
    if method not in {"spline", "poly"}:
        raise ValueError(
            f"Unsupported predictor_residual method {method!r}. Expected 'spline' or 'poly'."
        )
    return method


def _resolve_crossfit_method(
    config: Optional[Any],
    method: Optional[str],
) -> str:
    """Resolve cross-fit method, inheriting the main method by default."""
    if method is not None:
        resolved_method = str(method).strip().lower()
    else:
        configured_crossfit_method = _get_config_value(
            config,
            "behavior_analysis.predictor_residual.crossfit.method",
            None,
        )
        if configured_crossfit_method not in (None, ""):
            resolved_method = str(configured_crossfit_method).strip().lower()
        else:
            return _resolve_predictor_residual_method(config)

    if resolved_method not in {"spline", "poly"}:
        raise ValueError(
            f"Unsupported predictor_residual crossfit method {resolved_method!r}. "
            "Expected 'spline' or 'poly'."
        )
    return resolved_method


def _fit_predictor_outcome_model(
    predictor_values: pd.Series,
    outcome_values: pd.Series,
    *,
    config: Optional[Any] = None,
    method: Optional[str] = None,
) -> Tuple[str, Any, Dict[str, Any]]:
    """Fit the configured predictor→outcome model and return its estimator."""
    resolved_method = (
        str(method).strip().lower()
        if method is not None
        else _resolve_predictor_residual_method(config)
    )
    if resolved_method == "spline":
        spline_result = _fit_spline_model(predictor_values, outcome_values, config)
        if spline_result is None:
            raise ValueError(
                "Predictor residual spline fit failed for all configured spline_df_candidates."
            )
        estimator, metadata = spline_result
        return resolved_method, estimator, metadata
    if resolved_method == "poly":
        estimator, metadata = _fit_polynomial_model(predictor_values, outcome_values, config)
        return resolved_method, estimator, metadata
    raise ValueError(
        f"Unsupported predictor_residual method {resolved_method!r}. Expected 'spline' or 'poly'."
    )


def _predict_predictor_outcome_model(
    method: str,
    estimator: Any,
    predictor_values: pd.Series,
) -> pd.Series:
    """Predict outcomes from the fitted estimator."""
    if method == "spline":
        return _predict_spline_model(estimator, predictor_values)
    if method == "poly":
        return _predict_polynomial_model(estimator, predictor_values)
    raise ValueError(
        f"Unsupported predictor_residual method {method!r}. Expected 'spline' or 'poly'."
    )


def _build_group_folds(
    group_values: np.ndarray,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build deterministic group-wise folds with balanced sample counts."""
    groups = pd.Series(group_values, dtype=object)
    counts = groups.value_counts(dropna=False, sort=False)
    ordered_groups = list(counts.index)
    first_seen = {group: idx for idx, group in enumerate(ordered_groups)}
    ordered_groups.sort(key=lambda group: (-int(counts[group]), first_seen[group]))

    fold_groups: List[List[Any]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for group in ordered_groups:
        fold_index = int(np.argmin(fold_sizes))
        fold_groups[fold_index].append(group)
        fold_sizes[fold_index] += int(counts[group])

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for groups_in_fold in fold_groups:
        test_mask = groups.isin(groups_in_fold).to_numpy()
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def crossfit_predictor_outcome_curve(
    predictor: pd.Series,
    outcome: pd.Series,
    groups: Optional[pd.Series],
    *,
    config: Optional[Any] = None,
    method: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Cross-fit the predictor→outcome model using group-held-out folds."""
    assert_continuous_predictor(predictor, config, context="predictor_residual.crossfit")
    predictor_aligned, outcome_aligned, common_index = _prepare_data(predictor, outcome)
    group_series = None
    if groups is not None:
        group_index = getattr(groups, "index", common_index)
        group_series = pd.Series(groups, index=group_index).reindex(common_index)
    resolved_method = _resolve_crossfit_method(config, method)

    metadata: Dict[str, Any] = {
        "status": "init",
        "n_total": int(len(common_index)),
        "method": resolved_method,
    }

    predicted = pd.Series(np.nan, index=common_index, dtype=float)
    residual = pd.Series(np.nan, index=common_index, dtype=float)
    if group_series is None:
        metadata["status"] = "skipped_missing_groups"
        return predicted, residual, metadata

    valid_mask = predictor_aligned.notna() & outcome_aligned.notna() & group_series.notna()
    n_valid = int(valid_mask.sum())
    metadata["n_valid"] = n_valid

    min_samples = int(_get_config_value(config, "behavior_analysis.predictor_residual.min_samples", 10))
    if n_valid < min_samples:
        metadata["status"] = "skipped_insufficient_samples"
        return predicted, residual, metadata

    predictor_valid = predictor_aligned.loc[valid_mask]
    outcome_valid = outcome_aligned.loc[valid_mask]
    groups_valid = group_series.loc[valid_mask]
    unique_groups = pd.unique(groups_valid.to_numpy(dtype=object))
    n_groups = int(len(unique_groups))
    metadata["n_groups"] = n_groups
    if n_groups < 2:
        metadata["status"] = "skipped_insufficient_groups"
        return predicted, residual, metadata

    n_splits_requested = int(
        _get_config_value(config, "behavior_analysis.predictor_residual.crossfit.n_splits", 5)
    )
    n_splits = max(2, min(n_splits_requested, n_groups))
    splits = _build_group_folds(groups_valid.to_numpy(dtype=object), n_splits)
    if len(splits) < 2:
        metadata["status"] = "skipped_insufficient_groups"
        return predicted, residual, metadata

    cv_predictions = np.full(len(predictor_valid), np.nan, dtype=float)
    fold_metadata: List[Dict[str, Any]] = []
    for train_idx, test_idx in splits:
        if int(len(train_idx)) < min_samples:
            raise ValueError(
                "Predictor residual crossfit requires each training fold to meet "
                f"min_samples={min_samples}."
            )
        resolved_method, estimator, model_metadata = _fit_predictor_outcome_model(
            predictor_valid.iloc[train_idx],
            outcome_valid.iloc[train_idx],
            config=config,
            method=metadata["method"],
        )
        fold_predictions = _predict_predictor_outcome_model(
            resolved_method,
            estimator,
            predictor_valid.iloc[test_idx],
        )
        cv_predictions[test_idx] = fold_predictions.to_numpy(dtype=float)
        fold_metadata.append(model_metadata)

    if not np.isfinite(cv_predictions).all():
        raise ValueError("Predictor residual crossfit failed to produce finite predictions for all held-out groups.")

    predicted.loc[predictor_valid.index] = cv_predictions
    residual.loc[outcome_valid.index] = outcome_valid.to_numpy(dtype=float) - cv_predictions
    metadata["status"] = "ok"
    metadata["n_splits"] = int(len(splits))
    metadata["fold_models"] = fold_metadata
    return predicted, residual, metadata


def fit_predictor_outcome_curve(
    predictor: pd.Series,
    outcome: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Fit outcome ~ f(predictor) and return (predicted, residual, metadata)."""
    assert_continuous_predictor(predictor, config, context="predictor_residual")
    predictor_aligned, outcome_aligned, common_index = _prepare_data(predictor, outcome)

    metadata: Dict[str, Any] = {
        "model": None,
        "status": "init",
        "n_total": int(len(common_index)),
    }

    valid_mask = predictor_aligned.notna() & outcome_aligned.notna()
    n_valid = int(valid_mask.sum())
    metadata["n_valid"] = n_valid

    min_samples = int(_get_config_value(config, "behavior_analysis.predictor_residual.min_samples", 10))
    if n_valid < min_samples:
        metadata["status"] = "skipped_insufficient_samples"
        predicted = pd.Series(np.nan, index=common_index, dtype=float)
        residual = pd.Series(np.nan, index=common_index, dtype=float)
        return predicted, residual, metadata

    predictor_valid = predictor_aligned[valid_mask]
    outcome_valid = outcome_aligned[valid_mask]

    method, estimator, model_metadata = _fit_predictor_outcome_model(
        predictor_valid,
        outcome_valid,
        config=config,
    )
    predicted_valid = _predict_predictor_outcome_model(
        method,
        estimator,
        predictor_valid,
    )

    predicted_full = pd.Series(np.nan, index=common_index, dtype=float)
    predicted_full.loc[predictor_valid.index] = predicted_valid
    residual_valid = _compute_residuals(outcome_valid, predicted_valid)
    residual_full = pd.Series(np.nan, index=common_index, dtype=float)
    residual_full.loc[outcome_valid.index] = residual_valid
    metadata.update(model_metadata)
    return predicted_full, residual_full, metadata


__all__ = [
    "crossfit_predictor_outcome_curve",
    "fit_predictor_outcome_curve",
]
