"""Fold-contained nuisance residualization for continuous ML targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value


@dataclass(frozen=True)
class FoldNuisanceFit:
    train_target: np.ndarray
    test_target: np.ndarray
    train_prediction: np.ndarray
    test_prediction: np.ndarray
    train_residual: np.ndarray
    test_residual: np.ndarray
    details: dict[str, Any]


def configured_target_residualization_columns(config: Any) -> tuple[str, ...]:
    enabled = bool(get_config_value(config, "machine_learning.target_residualization.enabled", False))
    if not enabled:
        return tuple()

    raw_columns = get_config_value(config, "machine_learning.target_residualization.columns", [])
    if not isinstance(raw_columns, (list, tuple)):
        raise ValueError("machine_learning.target_residualization.columns must be a list of column names.")
    columns = tuple(str(column).strip() for column in raw_columns if str(column).strip())
    if not columns:
        raise ValueError(
            "machine_learning.target_residualization.columns must be non-empty when enabled."
        )
    return columns


def residualize_targets_for_fold(
    *,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    nuisance_fit = fit_nuisance_model_for_fold(
        y=y,
        meta=meta,
        train_idx=train_idx,
        test_idx=test_idx,
        columns=columns,
    )
    return (
        nuisance_fit.train_residual.astype(float),
        nuisance_fit.test_residual.astype(float),
        nuisance_fit.details,
    )


def fit_nuisance_model_for_fold(
    *,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    columns: Sequence[str],
) -> FoldNuisanceFit:
    column_names = tuple(str(column).strip() for column in columns if str(column).strip())
    if not column_names:
        raise ValueError("Target residualization requires at least one nuisance column.")

    y_values = np.asarray(y, dtype=float)
    train_indices = np.asarray(train_idx, dtype=int)
    test_indices = np.asarray(test_idx, dtype=int)
    _validate_indices(y_values, train_indices, test_indices)

    design_train = _design_matrix(
        meta.iloc[train_indices],
        column_names,
        check_rank=True,
    )
    design_test = _design_matrix(
        meta.iloc[test_indices],
        column_names,
        check_rank=False,
    )
    y_train = y_values[train_indices]
    y_test = y_values[test_indices]
    if not np.all(np.isfinite(y_train)) or not np.all(np.isfinite(y_test)):
        raise ValueError("Target residualization requires finite train and test target values.")
    if len(y_train) <= design_train.shape[1]:
        raise ValueError(
            "Target residualization requires more training rows than nuisance parameters: "
            f"rows={len(y_train)}, parameters={design_train.shape[1]}."
        )

    coefficients, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    train_prediction = design_train @ coefficients
    test_prediction = design_test @ coefficients
    train_residual = y_train - train_prediction
    test_residual = y_test - test_prediction
    details = {
        "columns": list(column_names),
        "n_parameters": int(design_train.shape[1]),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
    }
    return FoldNuisanceFit(
        train_target=y_train.astype(float),
        test_target=y_test.astype(float),
        train_prediction=train_prediction.astype(float),
        test_prediction=test_prediction.astype(float),
        train_residual=train_residual.astype(float),
        test_residual=test_residual.astype(float),
        details=details,
    )


def _validate_indices(y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if y.ndim != 1:
        raise ValueError(f"Target residualization expects a 1D target vector, got shape {y.shape}.")
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Target residualization requires non-empty train and test indices.")
    max_index = len(y) - 1
    if np.any(train_idx < 0) or np.any(test_idx < 0):
        raise ValueError("Target residualization indices must be non-negative.")
    if np.any(train_idx > max_index) or np.any(test_idx > max_index):
        raise ValueError("Target residualization indices exceed target length.")


def _design_matrix(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    check_rank: bool,
) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Target residualization nuisance columns are missing from trial metadata: "
            f"{missing}."
        )

    design_columns: list[np.ndarray] = [np.ones(len(frame), dtype=float)]
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Target residualization column '{column}' contains non-finite values.")
        design_columns.append(values)

    design = np.column_stack(design_columns)
    if check_rank:
        _validate_training_nuisance_rank(design[:, 1:], columns)
    return design


def _validate_training_nuisance_rank(
    nuisance_design: np.ndarray,
    columns: tuple[str, ...],
    *,
    tolerance: float = 1e-10,
) -> None:
    centered = nuisance_design - np.mean(nuisance_design, axis=0, keepdims=True)
    column_norms = np.linalg.norm(centered, axis=0)
    if np.any(column_norms <= 0.0):
        raise ValueError(
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)} contain constant training-fold nuisance terms."
        )

    scaled = centered / column_norms
    singular_values = np.linalg.svd(scaled, compute_uv=False)
    if singular_values.size < len(columns):
        raise ValueError(
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)}, rank={singular_values.size}, parameters={len(columns)}."
        )

    max_singular_value = float(singular_values[0])
    if max_singular_value <= 0.0:
        raise ValueError(
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)} have zero training-fold variance."
        )
    singular_ratios = singular_values / max_singular_value
    if np.any(singular_ratios < float(tolerance)):
        rank = int(np.sum(singular_ratios >= float(tolerance)))
        raise ValueError(
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)}, rank={rank}, parameters={len(columns)}, "
            f"tolerance={float(tolerance)}."
        )


__all__ = [
    "FoldNuisanceFit",
    "configured_target_residualization_columns",
    "fit_nuisance_model_for_fold",
    "residualize_targets_for_fold",
]
