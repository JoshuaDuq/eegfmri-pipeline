"""
CNN classifiers for EEG trial-level decoding.

Implements an EEGNet-style binary classifier with subject-aware CV support.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

from eeg_pipeline.analysis.machine_learning.classification import ClassificationResult
from eeg_pipeline.analysis.machine_learning.config import get_ml_config


def _import_torch():
    try:
        import torch  # type: ignore[import-untyped]
        import torch.nn as nn  # type: ignore[import-untyped]
        import torch.optim as optim  # type: ignore[import-untyped]
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-untyped]
    except Exception as exc:
        raise ImportError(
            "CNN model requires PyTorch. Install torch and retry classification with model='cnn'."
        ) from exc
    return torch, nn, optim, DataLoader, TensorDataset


def _split_train_val_indices(
    groups_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y_train)
    if n < 6:
        idx = np.arange(n)
        return idx, idx

    unique_groups = np.unique(groups_train.astype(str))
    if len(unique_groups) >= 2:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=float(val_fraction), random_state=seed)
            train_idx, val_idx = next(gss.split(np.zeros((n, 1)), y_train, groups=groups_train))
            if len(train_idx) > 0 and len(val_idx) > 0:
                return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)
        except Exception:
            pass

    idx_all = np.arange(n)
    stratify = y_train if len(np.unique(y_train)) > 1 else None
    try:
        train_idx, val_idx = train_test_split(
            idx_all,
            test_size=float(val_fraction),
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except Exception:
        train_idx, val_idx = idx_all, idx_all
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _channelwise_standardize(
    X_train: np.ndarray,
    X_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_other - mean) / std


def fit_predict_cnn_binary_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    config: Any,
    logger: Any = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train EEGNet-style CNN on train split and predict labels/probabilities on test split.

    Expected X shapes: (n_trials, n_channels, n_timepoints).
    """
    torch, nn, optim, DataLoader, TensorDataset = _import_torch()
    cfg = get_ml_config(config)

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).astype(int)
    groups_train = np.asarray(groups_train)

    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(f"CNN expects 3D input [trials, channels, time], got {X_train.shape} and {X_test.shape}")
    if len(np.unique(y_train)) < 2:
        raise ValueError("CNN training fold has only one class.")

    # Reproducibility
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))

    val_fraction = float(cfg.get("cnn_val_fraction", 0.2))
    train_idx, val_idx = _split_train_val_indices(groups_train, y_train, seed=seed, val_fraction=val_fraction)

    X_fit = X_train[train_idx]
    y_fit = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    X_fit_n, X_val_n = _channelwise_standardize(X_fit, X_val)
    _, X_test_n = _channelwise_standardize(X_fit, X_test)

    X_fit_t = torch.tensor(X_fit_n[:, None, :, :], dtype=torch.float32)
    y_fit_t = torch.tensor(y_fit, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_n[:, None, :, :], dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_n[:, None, :, :], dtype=torch.float32)

    n_channels = int(X_train.shape[1])
    n_times = int(X_train.shape[2])

    temporal_filters = int(cfg.get("cnn_temporal_filters", 8))
    depth_multiplier = int(cfg.get("cnn_depth_multiplier", 2))
    pointwise_filters = int(cfg.get("cnn_pointwise_filters", 16))
    dropout = float(cfg.get("cnn_dropout", 0.5))
    kernel_len = int(cfg.get("cnn_kernel_length", 64))
    separable_kernel_len = int(cfg.get("cnn_separable_kernel_length", 16))
    kernel_len = max(3, kernel_len + (kernel_len % 2 == 0))
    separable_kernel_len = max(3, separable_kernel_len + (separable_kernel_len % 2 == 0))

    class EEGNetBinary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, temporal_filters, kernel_size=(1, kernel_len), padding=(0, kernel_len // 2), bias=False),
                nn.BatchNorm2d(temporal_filters),
                nn.Conv2d(
                    temporal_filters,
                    temporal_filters * depth_multiplier,
                    kernel_size=(n_channels, 1),
                    groups=temporal_filters,
                    bias=False,
                ),
                nn.BatchNorm2d(temporal_filters * depth_multiplier),
                nn.ELU(inplace=True),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(
                    temporal_filters * depth_multiplier,
                    temporal_filters * depth_multiplier,
                    kernel_size=(1, separable_kernel_len),
                    padding=(0, separable_kernel_len // 2),
                    groups=temporal_filters * depth_multiplier,
                    bias=False,
                ),
                nn.Conv2d(temporal_filters * depth_multiplier, pointwise_filters, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(pointwise_filters),
                nn.ELU(inplace=True),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(pointwise_filters, 1)

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.block1(x)
            x = self.block2(x)
            x = self.head(x)
            x = x.flatten(start_dim=1)
            return self.classifier(x).squeeze(-1)

    use_cuda = bool(cfg.get("cnn_use_cuda", False))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = EEGNetBinary().to(device)

    n_pos = int(np.sum(y_fit == 1))
    n_neg = int(np.sum(y_fit == 0))
    pos_weight = None
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([float(n_neg / max(n_pos, 1))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("cnn_learning_rate", 1e-3)),
        weight_decay=float(cfg.get("cnn_weight_decay", 1e-3)),
    )

    batch_size = int(cfg.get("cnn_batch_size", 64))
    max_epochs = int(cfg.get("cnn_max_epochs", 75))
    patience = int(cfg.get("cnn_patience", 10))
    grad_clip = float(cfg.get("cnn_gradient_clip_norm", 1.0))

    train_loader = DataLoader(
        TensorDataset(X_fit_t, y_fit_t),
        batch_size=max(4, min(batch_size, len(X_fit_t))),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=max(4, min(batch_size, len(X_val_t))),
        shuffle=False,
        drop_last=False,
    )

    best_state = None
    best_val_loss = np.inf
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_losses.append(float(criterion(logits, yb).detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else np.inf

        if val_loss < (best_val_loss - 1e-6):
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_prob = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


def nested_loso_cnn_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int = 42,
    config: Any = None,
    logger: Any = None,
) -> Tuple[ClassificationResult, pd.DataFrame]:
    """Nested-like LOSO for CNN (inner split used for early stopping, not hyperparameter search)."""
    import logging

    log = logger or logging.getLogger(__name__)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    if X.ndim != 3:
        raise ValueError(f"CNN LOSO expects X with shape (trials, channels, time), got {X.shape}")

    outer_cv = LeaveOneGroupOut()
    outer_splits = list(outer_cv.split(X, y, groups))

    y_pred_all = np.zeros(len(y), dtype=int)
    y_prob_all = np.full(len(y), np.nan, dtype=float)
    failed_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]

        if len(np.unique(y_train)) < 2:
            maj = int(np.median(y_train)) if len(y_train) > 0 else 0
            y_pred_all[test_idx] = maj
            y_prob_all[test_idx] = float(maj)
            failed_folds += 1
            continue

        try:
            y_pred, y_prob = fit_predict_cnn_binary_classifier(
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                X_test=X_test,
                seed=seed + int(fold_idx),
                config=config,
                logger=log,
            )
            y_pred_all[test_idx] = y_pred
            y_prob_all[test_idx] = y_prob
        except Exception as exc:
            log.warning("CNN fold %d failed (%s); falling back to majority prediction.", int(fold_idx), exc)
            maj = int(np.median(y_train)) if len(y_train) > 0 else 0
            y_pred_all[test_idx] = maj
            y_prob_all[test_idx] = float(maj)
            failed_folds += 1

    result = ClassificationResult(
        y_true=y,
        y_pred=y_pred_all,
        y_prob=y_prob_all if np.any(np.isfinite(y_prob_all)) else None,
        groups=groups,
        failed_fold_count=int(failed_folds),
        n_folds_total=int(len(outer_splits)),
    )
    log.info(result.summary())
    return result, pd.DataFrame()


__all__ = [
    "fit_predict_cnn_binary_classifier",
    "nested_loso_cnn_classification",
]
