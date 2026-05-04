"""Fold-contained nuisance residualization for continuous ML targets."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value


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
    column_names = tuple(str(column).strip() for column in columns if str(column).strip())
    if not column_names:
        raise ValueError("Target residualization requires at least one nuisance column.")

    y_values = np.asarray(y, dtype=float)
    train_indices = np.asarray(train_idx, dtype=int)
    test_indices = np.asarray(test_idx, dtype=int)
    _validate_indices(y_values, train_indices, test_indices)

    design_train = _design_matrix(meta.iloc[train_indices], column_names)
    design_test = _design_matrix(meta.iloc[test_indices], column_names)
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
    train_residual = y_train - design_train @ coefficients
    test_residual = y_test - design_test @ coefficients
    details = {
        "columns": list(column_names),
        "n_parameters": int(design_train.shape[1]),
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
    }
    return train_residual.astype(float), test_residual.astype(float), details


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


def _design_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
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
    rank = int(np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        raise ValueError(
            "Target residualization design is rank deficient. "
            f"Columns={list(columns)}, rank={rank}, parameters={design.shape[1]}."
        )
    return design


__all__ = [
    "configured_target_residualization_columns",
    "residualize_targets_for_fold",
]
