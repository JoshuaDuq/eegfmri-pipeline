"""Training loop for Study 1 deep regression."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from eeg_pipeline.analysis.machine_learning.target_residualization import residualize_targets_for_fold
from studies.pain_study.study1.deep_regression.model import build_band_regressor
from studies.pain_study.study1.targets import nuisance_columns, nuisance_regression_enabled


def _import_torch():
    try:
        import torch  # type: ignore[import-untyped]
        import torch.nn as nn  # type: ignore[import-untyped]
        import torch.optim as optim  # type: ignore[import-untyped]
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-untyped]
    except Exception as exc:
        raise ImportError(
            "Study 1 deep regression requires PyTorch. Install torch and retry."
        ) from exc
    return torch, nn, optim, DataLoader, TensorDataset


@dataclass(frozen=True)
class DeepRegressionResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: dict[str, Any]


def _safe_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: float) -> float:
    if len(y_true) < 2:
        return np.nan
    try:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_train_mean) ** 2)
        if ss_tot < 1e-12:
            return np.nan
        return float(1.0 - (ss_res / ss_tot))
    except ValueError:
        return np.nan


def _validation_indices(groups_train: np.ndarray, *, seed: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if fraction <= 0:
        return np.arange(len(groups_train), dtype=int), np.asarray([], dtype=int)

    unique_groups = np.unique(groups_train.astype(str))
    if len(unique_groups) < 2:
        return np.arange(len(groups_train), dtype=int), np.asarray([], dtype=int)

    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(unique_groups, dtype=object)[rng.permutation(len(unique_groups))]
    n_val_groups = int(round(float(fraction) * len(shuffled)))
    n_val_groups = max(1, min(len(shuffled) - 1, n_val_groups))
    validation_groups = set(str(group) for group in shuffled[:n_val_groups].tolist())
    val_mask = np.array([str(group) in validation_groups for group in groups_train], dtype=bool)
    train_mask = ~val_mask
    return np.flatnonzero(train_mask).astype(int), np.flatnonzero(val_mask).astype(int)


def _standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 3), keepdims=True)
    std = X_train.std(axis=(0, 3), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _standardize_train_targets(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return ((y_train - mean) / std).astype(float), mean, std


def _apply_target_standardization(
    values: np.ndarray,
    *,
    mean: float,
    std: float,
) -> np.ndarray:
    return ((values - mean) / std).astype(float)


def _invert_target_standardization(
    values: np.ndarray,
    *,
    mean: float,
    std: float,
) -> np.ndarray:
    return (values * std + mean).astype(float)


def _fit_regressor(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    config: Any,
    seed: int,
) -> np.ndarray:
    torch, nn, optim, DataLoader, TensorDataset = _import_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))

    X_train_n, X_test_n = _standardize_train_test(X_train, X_test)
    train_idx, val_idx = _validation_indices(
        groups_train,
        seed=seed,
        fraction=float(config.get("study1.deep_regression.validation_fraction", 0.2)),
    )
    X_fit = X_train_n[train_idx]
    y_fit = y_train[train_idx]
    X_val = X_train_n[val_idx]
    y_val = y_train[val_idx]
    y_fit_n, y_mean, y_std = _standardize_train_targets(y_fit)
    y_val_n = _apply_target_standardization(y_val, mean=y_mean, std=y_std)

    model = build_band_regressor(
        nn=nn,
        input_shape=(int(X_train.shape[1]), int(X_train.shape[2]), int(X_train.shape[3])),
        config=config,
    )
    use_cuda = bool(config.get("study1.deep_regression.use_cuda", False))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.get("study1.deep_regression.learning_rate", 1e-3)),
        weight_decay=float(config.get("study1.deep_regression.weight_decay", 1e-4)),
    )
    loss_fn = nn.MSELoss()
    batch_size = int(config.get("study1.deep_regression.batch_size", 32))
    n_epochs = int(config.get("study1.deep_regression.n_epochs", 25))
    patience = int(config.get("study1.deep_regression.patience", 5))

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_fit, dtype=torch.float32),
            torch.tensor(y_fit_n, dtype=torch.float32),
        ),
        batch_size=max(1, min(batch_size, len(X_fit))),
        shuffle=True,
        drop_last=False,
    )

    best_state = None
    best_val = np.inf
    no_improve = 0
    has_validation = len(val_idx) > 0
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device) if has_validation else None
    y_val_tensor = torch.tensor(y_val_n, dtype=torch.float32, device=device) if has_validation else None

    for _epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        if not has_validation or X_val_tensor is None or y_val_tensor is None:
            continue

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_tensor), y_val_tensor).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds_n = model(torch.tensor(X_test_n, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return _invert_target_standardization(preds_n, mean=y_mean, std=y_std)


def run_loso_deep_regression(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    target_name: str,
    preset_name: str,
    bands: list[str],
    config: Any,
    logger: logging.Logger | None = None,
) -> DeepRegressionResult:
    if logger is None:
        logger = logging.getLogger(__name__)

    if X.ndim != 4:
        raise ValueError(f"Deep regression expects 4D input tensors, got shape {X.shape}.")
    if len(X) != len(y) or len(X) != len(groups) or len(X) != len(meta):
        raise ValueError("Deep regression input arrays must have matching first dimensions.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Deep regression targets must be finite.")

    logo = LeaveOneGroupOut()
    predictions = np.full(len(y), np.nan, dtype=float)
    y_eval = np.full(len(y), np.nan, dtype=float)
    fold_records: list[dict[str, Any]] = []
    residual_columns = nuisance_columns(config) if nuisance_regression_enabled(config) else tuple()
    residualization_summary: dict[str, Any] = {
        "enabled": bool(residual_columns),
        "columns": list(residual_columns),
    }
    for fold_id, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        train_groups = groups[train_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        if residual_columns:
            y_train, y_test, residualization_summary = residualize_targets_for_fold(
                y=y,
                meta=meta,
                train_idx=train_idx,
                test_idx=test_idx,
                columns=residual_columns,
            )
            residualization_summary = {
                "enabled": True,
                **residualization_summary,
            }
        fold_pred = _fit_regressor(
            X_train=X[train_idx],
            y_train=y_train,
            groups_train=train_groups,
            X_test=X[test_idx],
            config=config,
            seed=int(config.get("project.random_state", 42)) + fold_id,
        )
        predictions[test_idx] = fold_pred
        y_eval[test_idx] = y_test
        y_true_fold = y_test
        fold_records.append(
            {
                "fold_id": fold_id,
                "test_subject": str(groups[test_idx[0]]),
                "r": _safe_r(y_true_fold, fold_pred),
                "mae": float(mean_absolute_error(y_true_fold, fold_pred)),
                "r2": _safe_r2(y_true_fold, fold_pred, y_train_mean=float(np.mean(y_train))),
                "n_trials": int(len(test_idx)),
            }
        )

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Deep regression failed to produce predictions for every trial.")
    if not np.all(np.isfinite(y_eval)):
        raise RuntimeError("Deep regression failed to produce evaluation targets for every trial.")

    pred_df = meta.copy()
    pred_df["y_raw"] = y
    pred_df["y_true"] = y_eval
    pred_df["y_pred"] = predictions
    fold_df = pd.DataFrame(fold_records)
    summary = {
        "model_name": "band_temporal_regressor",
        "target": target_name,
        "preset": preset_name,
        "bands": list(bands),
        "mean_r": float(pd.to_numeric(fold_df["r"], errors="coerce").mean()),
        "mean_mae": float(pd.to_numeric(fold_df["mae"], errors="coerce").mean()),
        "mean_r2": float(pd.to_numeric(fold_df["r2"], errors="coerce").mean()),
        "n_folds": int(len(fold_df)),
        "n_trials": int(len(pred_df)),
        "target_residualization": residualization_summary,
    }
    logger.info(
        "Deep regression complete for %s/%s: mean_r2=%.4f",
        target_name,
        preset_name,
        summary["mean_r2"],
    )
    return DeepRegressionResult(predictions=pred_df, fold_metrics=fold_df, summary=summary)


__all__ = ["DeepRegressionResult", "run_loso_deep_regression"]
