import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for CNN-Transformer model. Install torch before running this script."
    ) from exc

from ..models.cnn import FiniteValueChecker, ZeroVarianceDropper
from ..utils import (
    config_loader,
    cv_evaluation,
    data_loading,
    feature_utils,
    io_utils,
    metrics,
    permutation,
    target_signatures,
)

###################################################################
# Constants
###################################################################

MODEL_NAME = "cnn_transformer"


###################################################################
# CUDA Determinism
###################################################################

def _ensure_cuda_determinism() -> None:
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


###################################################################
# PyTorch CNN-Transformer Model
###################################################################

class _CNNTransformerModule(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        conv_channels: int,
        kernel_size: int,
        transformer_dim: int,
        num_heads: int,
        num_layers: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_kernel = min(kernel_size, n_features)
        if effective_kernel % 2 == 0:
            effective_kernel = max(1, effective_kernel - 1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, effective_kernel, padding=effective_kernel // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.project = nn.Sequential(
            nn.Linear(conv_channels, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim * ff_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim * ff_multiplier, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        x = x.transpose(1, 2)
        x = self.project(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.regressor(pooled).squeeze(-1)


class HybridCNNTransformerRegressor(BaseEstimator, RegressorMixin):
    supports_external_validation: bool = True

    def __init__(
        self,
        *,
        conv_channels: int = 32,
        kernel_size: int = 5,
        transformer_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 25,
        grad_clip: Optional[float] = None,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.transformer_dim = transformer_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_multiplier = ff_multiplier
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        request = str(self.device).lower()
        if request == "cpu":
            return torch.device("cpu")
        if request == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            _ensure_cuda_determinism()
            return torch.device("cuda")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            _ensure_cuda_determinism()
        return device

    def _build_model(self, n_features: int) -> nn.Module:
        if self.transformer_dim % self.num_heads != 0:
            raise ValueError("transformer_dim must be divisible by num_heads.")
        return _CNNTransformerModule(
            n_features=n_features,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            transformer_dim=self.transformer_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_multiplier=self.ff_multiplier,
            dropout=self.dropout,
        )

    @staticmethod
    def _coerce_groups(
        groups: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any]]],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        if groups is None:
            return None
        if isinstance(groups, pd.DataFrame):
            if groups.empty:
                return None
            if groups.shape[1] == 1:
                coerced = groups.iloc[:, 0].to_numpy()
            else:
                coerced = groups.astype("object").apply(lambda row: tuple(row), axis=1).to_numpy()
        elif isinstance(groups, pd.Series):
            coerced = groups.to_numpy()
        else:
            coerced = np.asarray(groups)
            if coerced.ndim > 1:
                coerced = np.array([tuple(row) for row in coerced])

        coerced = np.atleast_1d(np.asarray(coerced, dtype=object))
        if coerced.shape[0] != n_samples:
            raise ValueError(
                f"groups must have length {n_samples}, received {coerced.shape[0]}."
            )
        return coerced

    @staticmethod
    def _coerce_stratify(
        stratify: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any]]],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        if stratify is None:
            return None
        if isinstance(stratify, pd.DataFrame):
            if stratify.empty:
                return None
            if stratify.shape[1] == 1:
                coerced = stratify.iloc[:, 0].to_numpy()
            else:
                coerced = stratify.astype("object").apply(lambda row: tuple(row), axis=1).to_numpy()
        elif isinstance(stratify, pd.Series):
            coerced = stratify.to_numpy()
        else:
            coerced = np.asarray(stratify)
            if coerced.ndim > 1:
                coerced = np.array([tuple(row) for row in coerced])

        coerced = np.atleast_1d(np.asarray(coerced, dtype=object))
        if coerced.shape[0] != n_samples:
            raise ValueError(
                f"stratify must have length {n_samples}, received {coerced.shape[0]}."
            )
        return coerced

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any]]] = None,
        stratify: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any]]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        validation_indices: Optional[Sequence[int]] = None,
        **_: Any,
    ) -> "HybridCNNTransformerRegressor":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("Expected 2D array for X.")
        n_samples, n_features = X_arr.shape
        if n_samples < 2:
            raise ValueError("At least two samples are required to train the hybrid regressor.")
        if self.conv_channels <= 0 or self.kernel_size <= 0:
            raise ValueError("conv_channels and kernel_size must be positive.")
        if self.transformer_dim <= 0 or self.num_heads <= 0:
            raise ValueError("Transformer dimensions and heads must be positive.")
        if self.ff_multiplier <= 0:
            raise ValueError("ff_multiplier must be positive.")
        if self.lr <= 0 or self.batch_size <= 0:
            raise ValueError("Learning rate and batch size must be positive.")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must lie in [0, 1).")
        if validation_data is not None and validation_indices is not None:
            raise ValueError("Provide either validation_data or validation_indices, not both.")

        groups_arr = self._coerce_groups(groups, n_samples) if groups is not None else None
        stratify_arr = (
            self._coerce_stratify(stratify, n_samples) if stratify is not None else None
        )
        stratify_has_diversity = False
        if stratify_arr is not None:
            unique_vals, counts = np.unique(stratify_arr, return_counts=True)
            stratify_has_diversity = unique_vals.size >= 2 and np.all(counts >= 2)
        required_levels = (
            {val for val in stratify_arr if not pd.isna(val)} if stratify_arr is not None else set()
        )
        min_groups_required = 0
        if groups_arr is not None:
            unique_groups = np.unique(groups_arr)
            if unique_groups.size >= 2:
                min_groups_required = min(2, unique_groups.size)
            elif self.patience > 0:
                raise ValueError(
                    "HybridCNNTransformerRegressor requires at least two distinct groups for early stopping. "
                    "Provide additional grouping structure or disable early stopping (patience=0)."
                )
        min_val_required = max(5, len(required_levels), min_groups_required)
        min_val_required = min(min_val_required, max(1, n_samples - 1))

        def covers_levels(indices: np.ndarray) -> bool:
            if stratify_arr is None or not required_levels:
                return True
            return required_levels.issubset(
                {val for val in stratify_arr[indices] if not pd.isna(val)}
            )

        def covers_groups(indices: np.ndarray) -> bool:
            if groups_arr is None or min_groups_required == 0:
                return True
            return np.unique(groups_arr[indices]).size >= min_groups_required

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        device = self._resolve_device()

        train_idx = np.arange(n_samples, dtype=int)
        val_idx: Optional[np.ndarray] = None
        val_loader: Optional[DataLoader] = None
        train_X = X_arr
        train_y = y_arr

        if validation_indices is not None:
            try:
                val_idx = np.asarray(validation_indices, dtype=int)
            except Exception as exc:
                raise ValueError("validation_indices must be an iterable of integers.") from exc
            if val_idx.ndim != 1:
                raise ValueError("validation_indices must be one-dimensional.")
            if val_idx.size == 0:
                val_idx = None
            else:
                if (val_idx < 0).any() or (val_idx >= n_samples).any():
                    raise ValueError("validation_indices contain out-of-range values.")
                val_idx = np.unique(val_idx)
                if val_idx.size < min_val_required:
                    raise ValueError(
                        f"validation_indices must contain at least {min_val_required} samples."
                    )
                if not covers_levels(val_idx) or not covers_groups(val_idx):
                    raise ValueError(
                        "validation_indices do not satisfy required temperature/group coverage."
                    )
                train_idx = np.setdiff1d(train_idx, val_idx)
                if train_idx.size == 0:
                    raise ValueError("validation_indices remove all training samples.")
                train_X = X_arr[train_idx]
                train_y = y_arr[train_idx]
                val_X = X_arr[val_idx]
                val_y = y_arr[val_idx]
                val_dataset = TensorDataset(
                    torch.from_numpy(val_X.astype(np.float32)),
                    torch.from_numpy(val_y.astype(np.float32)),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=min(self.batch_size, len(val_dataset))
                )

        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValueError("validation_data must be a tuple of (X_val, y_val).")
            val_X_raw, val_y_raw = validation_data
            val_X_arr = np.asarray(val_X_raw, dtype=np.float32)
            val_y_arr = np.asarray(val_y_raw, dtype=np.float32).reshape(-1)
            if val_X_arr.ndim != 2:
                raise ValueError("validation_data[0] must be a 2D array.")
            if val_X_arr.shape[0] == 0:
                raise ValueError("validation_data contains zero samples.")
            if val_X_arr.shape[1] != n_features:
                raise ValueError(
                    "validation_data feature dimension %d does not match training features %d."
                    % (val_X_arr.shape[1], n_features)
                )
            if val_X_arr.shape[0] != val_y_arr.shape[0]:
                raise ValueError(
                    "validation_data feature count %d does not match target count %d."
                    % (val_X_arr.shape[0], val_y_arr.shape[0])
                )
            if not np.isfinite(val_X_arr).all():
                raise ValueError("Non-finite values detected in validation features.")
            if not np.isfinite(val_y_arr).all():
                raise ValueError("Non-finite values detected in validation targets.")
            if val_X_arr.shape[0] < min_val_required:
                raise ValueError(
                    f"validation_data must contain at least {min_val_required} samples."
                )
            val_dataset = TensorDataset(
                torch.from_numpy(val_X_arr.astype(np.float32)),
                torch.from_numpy(val_y_arr.astype(np.float32)),
            )
            val_loader = DataLoader(
                val_dataset, batch_size=min(self.batch_size, len(val_dataset))
            )

        if (
            val_loader is None
            and validation_data is None
            and validation_indices is None
            and n_samples > 8
            and self.patience > 0
        ):
            val_fraction = min(0.2, max(1.0 / n_samples, 0.1))
            split_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None
            min_val_samples = max(min_val_required, int(round(val_fraction * n_samples)))
            min_val_samples = min(min_val_samples, max(1, n_samples - 1))
            rng = np.random.default_rng(self.random_state)

            if groups_arr is not None and np.unique(groups_arr).size > 1:
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=val_fraction, random_state=self.random_state
                )
                try:
                    split_indices = next(splitter.split(X_arr, y_arr, groups_arr))
                except ValueError:
                    split_indices = None
                if split_indices is None:
                    unique_groups = rng.permutation(np.unique(groups_arr))
                    collected: List[int] = []
                    for group in unique_groups:
                        collected.extend(np.where(groups_arr == group)[0].tolist())
                        if len(collected) >= min_val_samples:
                            candidate = np.array(sorted(set(collected)), dtype=int)
                            if 0 < candidate.size < n_samples:
                                if covers_levels(candidate) and covers_groups(candidate):
                                    train_idx = np.array(
                                        sorted(set(range(n_samples)) - set(candidate)), dtype=int
                                    )
                                    split_indices = (train_idx, candidate)
                                    break

            if split_indices is None and groups_arr is None and stratify_has_diversity:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=val_fraction, random_state=self.random_state
                )
                try:
                    split_indices = next(splitter.split(X_arr, stratify_arr))
                except ValueError:
                    split_indices = None

            if split_indices is not None:
                train_idx, val_idx = split_indices
            elif groups_arr is None:
                all_indices = np.arange(n_samples, dtype=int)
                train_idx, val_idx = train_test_split(
                    all_indices,
                    test_size=val_fraction,
                    random_state=self.random_state,
                    stratify=stratify_arr if stratify_has_diversity else None,
                )
                val_idx = np.asarray(val_idx, dtype=int)
                if val_idx.size < min_val_samples or not covers_levels(val_idx):
                    val_idx = None
                    train_idx = np.arange(n_samples, dtype=int)
            else:
                val_idx = None

            if val_idx is not None:
                if val_idx.size < min_val_samples or not covers_levels(val_idx) or not covers_groups(val_idx):
                    val_idx = None
                    train_idx = np.arange(n_samples, dtype=int)
                else:
                    train_idx = np.asarray(train_idx, dtype=int)
                train_X, train_y = X_arr[train_idx], y_arr[train_idx]
                val_X, val_y = X_arr[val_idx], y_arr[val_idx]
                val_dataset = TensorDataset(
                    torch.from_numpy(val_X.astype(np.float32)),
                    torch.from_numpy(val_y.astype(np.float32)),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=min(self.batch_size, len(val_dataset))
                )

        needs_validation = self.patience > 0 and (groups_arr is not None or stratify_has_diversity)
        if (
            val_loader is None
            and validation_data is None
            and validation_indices is None
            and needs_validation
        ):
            raise ValueError(
                "Unable to form a validation split that satisfies grouping/stratification requirements."
            )

        self.validation_indices_ = None if val_idx is None else np.asarray(val_idx, dtype=int)

        train_dataset = TensorDataset(
            torch.from_numpy(train_X.astype(np.float32)),
            torch.from_numpy(train_y.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
        )

        model = self._build_model(n_features).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        history: List[Dict[str, float]] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            model.train()
            epoch_losses: List[float] = []
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device).unsqueeze(1)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            if val_loader is not None:
                model.eval()
                val_losses: List[float] = []
                with torch.no_grad():
                    for val_X_batch, val_y_batch in val_loader:
                        val_X_batch = val_X_batch.to(device).unsqueeze(1)
                        val_y_batch = val_y_batch.to(device)
                        val_pred = model(val_X_batch)
                        val_loss = criterion(val_pred, val_y_batch)
                        val_losses.append(float(val_loss.detach().cpu().item()))
                monitored_loss = float(np.mean(val_losses)) if val_losses else train_loss
            else:
                monitored_loss = train_loss

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "monitor_loss": monitored_loss})
            improved = monitored_loss < (best_loss - 1e-6)
            if improved:
                best_loss = monitored_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbose:
                print(
                    f"Epoch {epoch + 1:03d} | train_loss={train_loss:.4f} | monitor={monitored_loss:.4f}"
                )

            if self.patience > 0 and epochs_without_improvement >= self.patience:
                break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model = model.to(torch.device("cpu"))
        model.eval()

        self.model_ = model
        self.history_ = history
        self.best_loss_ = best_loss
        self.input_features_ = n_features
        self.device_ = torch.device("cpu")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model has not been fitted yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != getattr(self, "input_features_", X_arr.shape[1]):
            raise ValueError("Input feature dimension does not match the fitted data.")
        dataset = TensorDataset(torch.from_numpy(X_arr.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=min(len(dataset), 512))
        preds: List[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for (batch_X,) in loader:
                outputs = self.model_(batch_X.unsqueeze(1))
                preds.append(outputs.numpy())
        return np.concatenate(preds, axis=0)


###################################################################
# Helper Functions
###################################################################

def resolve_device_request(requested: str, logger) -> str:
    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device option '{requested}'.")

    if normalized == "auto":
        if torch.cuda.is_available():
            logger.info("CUDA available; selecting GPU device.")
            return "cuda"
        logger.info("CUDA unavailable; falling back to CPU device.")
        return "cpu"

    if normalized == "cuda":
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available. Use --device cpu or ensure GPU drivers are configured.")
            raise SystemExit(1)
        return "cuda"

    return "cpu"


def normalize_grid(values: Sequence[Union[int, float]], *, value_type: str) -> List[Union[int, float]]:
    unique_values = []
    for val in values:
        cast_val = float(val)
        if value_type == "int":
            cast_val = int(round(cast_val))
        if cast_val not in unique_values:
            unique_values.append(cast_val)
    if not unique_values:
        raise ValueError("Provided grid must contain at least one value.")
    return unique_values


def make_cnn_transformer_builder(device: str, verbose: bool) -> Callable:
    def builder(random_state: int, _: int) -> Pipeline:
        regressor = HybridCNNTransformerRegressor(
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", ZeroVarianceDropper()),
                ("scaler", StandardScaler()),
                ("finite_check", FiniteValueChecker()),
                ("hybrid", regressor),
            ]
        )

    return builder


def build_cnn_transformer_fit_params(meta_subset: pd.DataFrame) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    group_cols = [col for col in ("subject", "run") if col in meta_subset.columns]
    if group_cols:
        params["hybrid__groups"] = meta_subset[group_cols].reset_index(drop=True)
    if "temp_celsius" in meta_subset.columns:
        params["hybrid__stratify"] = meta_subset["temp_celsius"].to_numpy()
    return params


###################################################################
# Argument Parsing
###################################################################

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NPS beta responses from EEG oscillatory power with a CNN-Transformer hybrid."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to machine_learning/outputs/eeg_to_signature_cnn_transformer_<timestamp>.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Subject identifiers (with or without sub- prefix). Defaults to all available.",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=list(feature_utils.DEFAULT_BANDS),
        help="EEG frequency bands to include (subset of %s)." % (feature_utils.SUPPORTED_BANDS,),
    )
    parser.add_argument(
        "--target-signature",
        type=str,
        default=target_signatures.DEFAULT_TARGET_KEY,
        choices=sorted(target_signatures.TARGET_SIGNATURES),
        help="fMRI signature to decode (default: %(default)s).",
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Include stimulus temperature as an additional predictor.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for grid searches (Torch models force 1 regardless).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for cross-validation reproducibility.",
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=None,
        help="Random seed for label permutations (defaults to --random-state when omitted).",
    )
    parser.add_argument(
        "--permutation-count",
        type=int,
        default=0,
        help="Number of label permutations for R^2 significance testing (0 disables).",
    )
    parser.add_argument(
        "--conv-channels-grid",
        type=int,
        nargs="*",
        default=[32, 48],
        help="Grid of convolution channel counts to explore.",
    )
    parser.add_argument(
        "--kernel-size-grid",
        type=int,
        nargs="*",
        default=[3, 5],
        help="Grid of convolution kernel sizes (positive integers).",
    )
    parser.add_argument(
        "--transformer-dim-grid",
        type=int,
        nargs="*",
        default=[64, 96],
        help="Grid of Transformer embedding dimensions (must be divisible by selected num_heads).",
    )
    parser.add_argument(
        "--num-heads-grid",
        type=int,
        nargs="*",
        default=[4, 8],
        help="Grid of Transformer attention head counts.",
    )
    parser.add_argument(
        "--num-layers-grid",
        type=int,
        nargs="*",
        default=[1, 2],
        help="Grid of Transformer encoder layer counts.",
    )
    parser.add_argument(
        "--ff-multiplier-grid",
        type=int,
        nargs="*",
        default=[2, 4],
        help="Grid of feed-forward multipliers (multiplied by transformer_dim).",
    )
    parser.add_argument(
        "--dropout-grid",
        type=float,
        nargs="*",
        default=[0.1, 0.3],
        help="Grid of dropout probabilities (between 0 and 1).",
    )
    parser.add_argument(
        "--learning-rate-grid",
        type=float,
        nargs="*",
        default=[1e-3],
        help="Grid of learning rates.",
    )
    parser.add_argument(
        "--weight-decay-grid",
        type=float,
        nargs="*",
        default=[1e-4, 5e-4],
        help="Grid of weight decay (L2 regularisation) values.",
    )
    parser.add_argument(
        "--batch-size-grid",
        type=int,
        nargs="*",
        default=[32],
        help="Grid of batch sizes.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum training epochs per fit call.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Gradient clipping norm (disabled if not provided or <= 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device to use for training (auto selects CUDA if available).",
    )
    parser.add_argument(
        "--verbose-transformer",
        action="store_true",
        help="Print per-epoch training progress for the hybrid estimator.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. Defaults to utils/ml_config.yaml if not specified.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Disable loading configuration from YAML file.",
    )
    args = parser.parse_args(argv)
    
    if not args.no_config:
        config = config_loader.load_config(args.config)
        config_loader.apply_config_to_args(config, args)
        
        cnn_transformer_config = config_loader.get_model_config(config, "cnn_transformer")
        if cnn_transformer_config:
            if not args.conv_channels_grid:
                args.conv_channels_grid = cnn_transformer_config.get("conv_channels_grid", [32, 64])
            if not args.kernel_size_grid:
                args.kernel_size_grid = cnn_transformer_config.get("kernel_size_grid", [3, 5, 7])
            if not args.transformer_dim_grid:
                args.transformer_dim_grid = cnn_transformer_config.get("transformer_dim_grid", [64, 128])
            if not args.num_heads_grid:
                args.num_heads_grid = cnn_transformer_config.get("num_heads_grid", [4, 8])
            if not args.num_layers_grid:
                args.num_layers_grid = cnn_transformer_config.get("num_layers_grid", [2, 3])
            if not args.ff_multiplier_grid:
                args.ff_multiplier_grid = cnn_transformer_config.get("ff_multiplier_grid", [2, 4])
            if not args.dropout_grid:
                args.dropout_grid = cnn_transformer_config.get("dropout_grid", [0.1, 0.2, 0.3])
            if not args.learning_rate_grid:
                args.learning_rate_grid = cnn_transformer_config.get("learning_rate_grid", [1e-4, 5e-4, 1e-3])
            if not args.weight_decay_grid:
                args.weight_decay_grid = cnn_transformer_config.get("weight_decay_grid", [1e-5, 1e-4])
            if not args.batch_size_grid:
                args.batch_size_grid = cnn_transformer_config.get("batch_size_grid", [16, 32])
            if not hasattr(args, "max_epochs") or args.max_epochs is None:
                args.max_epochs = cnn_transformer_config.get("max_epochs", 200)
            if not hasattr(args, "patience") or args.patience is None:
                args.patience = cnn_transformer_config.get("patience", 25)
            if not hasattr(args, "grad_clip") or args.grad_clip is None:
                args.grad_clip = cnn_transformer_config.get("grad_clip")
            if not hasattr(args, "verbose_transformer"):
                args.verbose_transformer = cnn_transformer_config.get("verbose", False)
    
    return args


###################################################################
# Main Function
###################################################################

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    target = target_signatures.get_target_signature(args.target_signature)

    bands = tuple(dict.fromkeys(band.lower() for band in args.bands))
    if not bands:
        raise ValueError("No bands specified.")
    for band in bands:
        if band not in feature_utils.SUPPORTED_BANDS:
            raise ValueError(f"Unsupported band '{band}'. Supported bands: {feature_utils.SUPPORTED_BANDS}.")

    conv_grid = [int(v) for v in normalize_grid(args.conv_channels_grid, value_type="int") if int(v) > 0]
    if not conv_grid:
        raise ValueError("conv_channels grid must contain positive integers.")
    kernel_grid = [int(v) for v in normalize_grid(args.kernel_size_grid, value_type="int") if int(v) > 0]
    if not kernel_grid:
        raise ValueError("kernel_size grid must contain positive integers.")
    transformer_dim_grid = [
        int(v) for v in normalize_grid(args.transformer_dim_grid, value_type="int") if int(v) > 0
    ]
    if not transformer_dim_grid:
        raise ValueError("transformer_dim grid must contain positive integers.")
    num_heads_grid = [int(v) for v in normalize_grid(args.num_heads_grid, value_type="int") if int(v) > 0]
    if not num_heads_grid:
        raise ValueError("num_heads grid must contain positive integers.")
    num_layers_grid = [int(v) for v in normalize_grid(args.num_layers_grid, value_type="int") if int(v) > 0]
    if not num_layers_grid:
        raise ValueError("num_layers grid must contain positive integers.")
    ff_multiplier_grid = [
        int(v) for v in normalize_grid(args.ff_multiplier_grid, value_type="int") if int(v) > 0
    ]
    if not ff_multiplier_grid:
        raise ValueError("ff_multiplier grid must contain positive integers.")
    dropout_grid = [
        float(v)
        for v in normalize_grid(args.dropout_grid, value_type="float")
        if 0 <= float(v) < 1
    ]
    if not dropout_grid:
        raise ValueError("dropout grid must contain probabilities in [0, 1).")
    lr_grid = [float(v) for v in normalize_grid(args.learning_rate_grid, value_type="float") if float(v) > 0]
    if not lr_grid:
        raise ValueError("learning-rate grid must contain positive floats.")
    weight_decay_grid = [
        float(v) for v in normalize_grid(args.weight_decay_grid, value_type="float") if float(v) >= 0
    ]
    if not weight_decay_grid:
        raise ValueError("weight-decay grid must contain non-negative floats.")
    batch_grid = [int(v) for v in normalize_grid(args.batch_size_grid, value_type="int") if int(v) > 0]
    if not batch_grid:
        raise ValueError("batch-size grid must contain positive integers.")

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        Path(__file__).resolve().parent.parent / "outputs" / f"eeg_to_{target.key}_cnn_transformer_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = io_utils.setup_logging(output_dir, logger_name=f"eeg_to_{target.key}_cnn_transformer")
    logger.info("EEG -> %s CNN-Transformer training pipeline started.", target.display_name)
    logger.info("Using bands: %s", ", ".join(bands))
    logger.info(
        "Hyperparameter grid | conv_channels: %s | kernel: %s | transformer_dim: %s | heads: %s | layers: %s | ff_mult: %s | dropout: %s | lr: %s | weight_decay: %s | batch: %s",
        conv_grid,
        kernel_grid,
        transformer_dim_grid,
        num_heads_grid,
        num_layers_grid,
        ff_multiplier_grid,
        dropout_grid,
        lr_grid,
        weight_decay_grid,
        batch_grid,
    )
    logger.info("Max epochs: %d | Patience: %d | Grad clip: %s", args.max_epochs, args.patience, args.grad_clip)

    device = resolve_device_request(args.device, logger)
    logger.info("Effective training device: %s", device)

    eeg_deriv_root = repo_root / "eeg_pipeline" / "bids_output" / "derivatives"
    try:
        fmri_outputs_root = target_signatures.resolve_fmri_outputs_root(repo_root, target)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
    logger.info("Using fMRI outputs root: %s", fmri_outputs_root)

    if not eeg_deriv_root.exists():
        logger.error("EEG derivatives directory not found: %s", eeg_deriv_root)
        raise SystemExit(1)

    available_subjects = data_loading.discover_subjects(eeg_deriv_root, fmri_outputs_root, target)
    if not available_subjects:
        logger.error(
            "No subjects with both EEG features and fMRI %s scores were found.",
            target.short_name,
        )
        raise SystemExit(1)
    logger.info("Available subjects with aligned data: %s", ", ".join(available_subjects))

    if args.subjects:
        requested = [s if s.startswith("sub-") else f"sub-{s}" for s in args.subjects]
        invalid = sorted(set(requested) - set(available_subjects))
        if invalid:
            logger.error("Requested subjects missing required data: %s", ", ".join(invalid))
            raise SystemExit(1)
        subjects = sorted(requested)
    else:
        subjects = available_subjects

    subject_results = []
    drops_summary: Dict[str, List[Dict[str, float]]] = {}

    for subject in subjects:
        result = data_loading.load_subject_dataset(subject, eeg_deriv_root, fmri_outputs_root, bands, target, logger)
        subject_results.append(result)
        if result.dropped_trials:
            drops_summary[subject] = result.dropped_trials

    if not subject_results:
        logger.error("No feature columns detected.")
        raise SystemExit(1)

    target_column = subject_results[0].target_column

    all_feature_sets = [set(res.feature_columns) for res in subject_results]
    if not all_feature_sets:
        logger.error("No feature columns available after loading subjects.")
        raise SystemExit(1)
    master_features = sorted(set().union(*all_feature_sets))
    if not master_features:
        logger.error("Feature union across subjects is empty.")
        raise SystemExit(1)

    metadata_cols = [
        "subject",
        "run",
        "trial_idx_run",
        "trial_idx_global",
        "temp_celsius",
        "vas_rating",
        "pain_binary",
        target_column,
    ]
    if target_column != "br_score":
        metadata_cols.append("br_score")

    for subj_idx, res in enumerate(subject_results):
        missing = sorted(set(master_features) - set(res.feature_columns))
        if missing:
            preview = ", ".join(missing[:10])
            if len(missing) > 10:
                preview += ", ..."
            logger.info(
                "  %s: filling %d missing feature(s) with NaN: %s",
                subjects[subj_idx],
                len(missing),
                preview,
            )
        res.data = res.data.reindex(columns=metadata_cols + master_features)
        res.feature_columns = list(master_features)

    data = pd.concat([res.data for res in subject_results], ignore_index=True)
    feature_columns = list(master_features)
    if args.include_temperature and "temp_celsius" not in feature_columns:
        feature_columns.append("temp_celsius")

    feature_columns = feature_utils.filter_zero_variance_features(data, feature_columns, logger=logger)

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    if y.isna().any():
        n_na = y.isna().sum()
        logger.error("Target variable (%s) contains %d NaN values; cannot proceed.", target_column, n_na)
        raise SystemExit(1)
    if np.isinf(y).any():
        n_inf = np.isinf(y).sum()
        logger.error("Target variable (%s) contains %d infinite values; cannot proceed.", target_column, n_inf)
        raise SystemExit(1)

    logger.info(
        "Target (%s) statistics: min=%.3f, max=%.3f, mean=%.3f, std=%.3f",
        target_column,
        y.min(),
        y.max(),
        y.mean(),
        y.std(),
    )

    if len(subjects) > 1:
        outer_groups = data["subject"].to_numpy()
        outer_group_level = "subject"
    else:
        outer_groups = data["run"].to_numpy()
        outer_group_level = "run"
    composite_run_ids = (
        data["subject"].astype(str).str.strip()
        + "__run__"
        + data["run"].astype(str).str.strip()
    ).to_numpy()
    run_groups = composite_run_ids
    logger.info(
        "Inner CV grouping will use subject+run composites (%d unique groups).",
        len(np.unique(run_groups)),
    )

    logger.info(
        "Assembled dataset: %d trials, %d subjects, %d features.",
        len(data),
        len(subjects),
        len(feature_columns),
    )

    feature_ratio = len(feature_columns) / max(len(data), 1)
    logger.info(
        "Feature-to-sample ratio: %.2f (%d features / %d trials)",
        feature_ratio,
        len(feature_columns),
        len(data),
    )
    if feature_ratio > 0.1:
        logger.warning(
            "Feature-to-sample ratio %.2f exceeds 0.10; consider stronger regularisation, feature selection, or dimensionality reduction.",
            feature_ratio,
        )

    if "temp_celsius" in data.columns:
        temp_counts = data["temp_celsius"].value_counts().sort_index()
        logger.info("Temperature distribution (counts per condition):\n%s", temp_counts.to_string())
    else:
        temp_counts = pd.Series(dtype=int)

    logger.info(
        "%s betas reflect the delayed (~5-7 s) hemodynamic response; EEG features are interpreted with this lag in mind.",
        target.display_name,
    )

    plateau_window = None
    eeg_config_path = repo_root / "eeg_pipeline" / "utils" / "eeg_config.yaml"
    if eeg_config_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(eeg_config_path.read_text())
            plateau_window = cfg.get("time_frequency_analysis", {}).get("plateau_window")
            if plateau_window is not None:
                logger.info("EEG plateau window from config: %s seconds", plateau_window)
                if not isinstance(plateau_window, (int, float)) or float(plateau_window) <= 0:
                    raise ValueError(
                        "Invalid plateau_window in eeg_config.yaml; expected positive number of seconds."
                    )
                plateau_window = float(plateau_window)
                if plateau_window < 3 or plateau_window > 10:
                    logger.warning(
                        "Configured plateau window %.2f s is outside the typical 3-10 s range for HRF-aligned EEG features.",
                        plateau_window,
                    )
        except Exception as exc:
            logger.warning("Could not read EEG plateau window from %s: %s", eeg_config_path, exc)
    else:
        logger.warning("EEG config file not found at %s; plateau window not verified.", eeg_config_path)

    if plateau_window is None:
        raise ValueError(
            "Unable to determine EEG plateau window from configuration; verify EEG/fMRI alignment before training."
        )

    temperature_baseline_metrics: Optional[Dict[str, float]] = None
    temperature_baseline_desc: Optional[str] = None
    if args.include_temperature and "temp_celsius" in data.columns:
        temperature_baseline_metrics, temperature_baseline_desc = metrics.compute_temperature_baseline_cv(
            temp=data["temp_celsius"],
            target=y,
            outer_groups=outer_groups,
            random_state=args.random_state,
            logger=logger,
        )

    builder = make_cnn_transformer_builder(device=device, verbose=args.verbose_transformer)

    temperature_predictor_note = (
        "Temperature covariate included as predictor alongside EEG features."
        if args.include_temperature and "temp_celsius" in feature_columns
        else "Temperature covariate excluded from predictors; per-temperature diagnostics reflect distributional balance only."
    )
    logger.info(temperature_predictor_note)
    param_grid = {
        "hybrid__conv_channels": conv_grid,
        "hybrid__kernel_size": kernel_grid,
        "hybrid__transformer_dim": transformer_dim_grid,
        "hybrid__num_heads": num_heads_grid,
        "hybrid__num_layers": num_layers_grid,
        "hybrid__ff_multiplier": ff_multiplier_grid,
        "hybrid__dropout": dropout_grid,
        "hybrid__lr": lr_grid,
        "hybrid__weight_decay": weight_decay_grid,
        "hybrid__batch_size": batch_grid,
        "hybrid__max_epochs": [int(args.max_epochs)],
        "hybrid__patience": [int(args.patience)],
    }
    if args.grad_clip is not None and args.grad_clip > 0:
        param_grid["hybrid__grad_clip"] = [float(args.grad_clip)]

    grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    if grid_size * 5 > len(X):
        logger.warning(
            "CNN-Transformer grid (%d combinations) is large relative to available trials (%d); consider trimming the search space.",
            grid_size,
            len(X),
        )

    effective_n_jobs = 1
    if args.n_jobs not in (None, 1):
        logger.warning("Hybrid pipeline forcing n_jobs=1 to ensure deterministic Torch training.")

    result = cv_evaluation.nested_cv_evaluate(
        model_name=MODEL_NAME,
        builder=builder,
        param_grid=param_grid,
        X=X,
        y=y,
        feature_names=feature_columns,
        meta=data,
        outer_groups=outer_groups,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=effective_n_jobs,
        logger=logger,
        fit_params_fn=build_cnn_transformer_fit_params,
    )

    summary_metrics = result["summary_metrics"]
    headline_r2 = summary_metrics.get("r2")
    if headline_r2 is not None:
        logger.info("Hybrid CNN-Transformer nested CV R2 (out-of-fold)=%.3f", headline_r2)

    pred_df = metrics.build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=result["predictions"],
        model_name=MODEL_NAME,
        target_column=target_column,
        target_key=target.key,
        fold_assignments=result["fold_assignments"],
    )
    pred_path = output_dir / f"predictions_{MODEL_NAME}.tsv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, sep="\t", index=False)

    subj_metrics = metrics.compute_group_metrics(pred_df, ["subject"])
    subj_metrics_path = output_dir / f"per_subject_metrics_{MODEL_NAME}.tsv"
    subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    subj_metrics.to_csv(subj_metrics_path, sep="\t", index=False)

    temp_metrics = metrics.compute_group_metrics(pred_df, ["temp_celsius"])
    temp_metrics_path = output_dir / f"per_temperature_metrics_{MODEL_NAME}.tsv"
    temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temp_metrics.to_csv(temp_metrics_path, sep="\t", index=False)

    fold_path = None
    fold_df = pd.DataFrame(result["fold_details"])
    if not fold_df.empty:
        fold_df["best_params"] = fold_df["best_params"].apply(lambda d: json.dumps(d))
        if "test_temp_counts" in fold_df.columns:
            fold_df["test_temp_counts"] = fold_df["test_temp_counts"].apply(lambda d: json.dumps(d))
        fold_path = output_dir / f"cv_folds_{MODEL_NAME}.tsv"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(fold_path, sep="\t", index=False)

    metrics_path = output_dir / f"metrics_{MODEL_NAME}.json"
    io_utils.write_json(metrics_path, summary_metrics)

    best_params_path = output_dir / f"best_params_{MODEL_NAME}.json"
    io_utils.write_json(best_params_path, [fold["best_params"] for fold in result["fold_details"]])

    r2_values = [fold["test_r2"] for fold in result["fold_details"] if fold.get("test_r2") is not None]
    fold_mean_r2 = float(np.mean(r2_values)) if r2_values else None
    if len(r2_values) > 1:
        fold_std_r2 = float(np.std(r2_values, ddof=1))
        fold_sem_r2 = fold_std_r2 / math.sqrt(len(r2_values))
        margin = 1.96 * fold_sem_r2
        ref_value = headline_r2 if headline_r2 is not None else fold_mean_r2
        fold_ci95 = {
            "lower": float(ref_value - margin) if ref_value is not None else None,
            "upper": float(ref_value + margin) if ref_value is not None else None,
        }
    elif r2_values:
        fold_std_r2 = 0.0
        fold_sem_r2 = None
        fold_ci95 = None
    else:
        fold_std_r2 = None
        fold_sem_r2 = None
        fold_ci95 = None
    model_entry: Dict[str, Any] = {
        "name": MODEL_NAME,
        "metrics": summary_metrics,
        "fold_mean_r2": fold_mean_r2,
        "fold_std_r2": fold_std_r2,
        "prediction_file": pred_path.name,
        "per_subject_metrics_file": subj_metrics_path.name,
        "per_temperature_metrics_file": temp_metrics_path.name,
        "fold_details_file": fold_path.name if fold_path else None,
        "metrics_file": metrics_path.name,
        "best_params_file": best_params_path.name,
        "outer_cv": result.get("outer_cv_desc"),
        "inner_cv": result.get("inner_cv_desc"),
        "param_grid": param_grid,
    }

    if args.permutation_count > 0:
        perm_summary, perm_null = permutation.permutation_test_r2(
            model_name=MODEL_NAME,
            builder=builder,
            param_grid=param_grid,
            X=X,
            y=y,
            feature_names=feature_columns,
            meta=data,
            outer_groups=outer_groups,
            run_groups=run_groups,
            n_permutations=args.permutation_count,
            true_r2=summary_metrics["r2"],
            random_state=
                args.permutation_seed if args.permutation_seed is not None else args.random_state,
            n_jobs=effective_n_jobs,
            fit_params_fn=build_cnn_transformer_fit_params,
            logger=logger,
        )
        perm_json_path = output_dir / f"permutation_{MODEL_NAME}.json"
        io_utils.write_json(perm_json_path, perm_summary)
        null_path = output_dir / f"permutation_{MODEL_NAME}_null.npy"
        np.save(null_path, perm_null)
        model_entry["permutation_test"] = {
            "p_value": perm_summary["p_value"],
            "true_r2": perm_summary["true_r2"],
            "null_mean": perm_summary["null_mean"],
            "null_std": perm_summary["null_std"],
            "null_quantiles": perm_summary["null_quantiles"],
            "result_file": perm_json_path.name,
            "null_distribution_file": null_path.name,
        }

    final_fit_params = build_cnn_transformer_fit_params(data)
    final_stratify = data["temp_celsius"] if "temp_celsius" in data.columns else None
    final_estimator, final_best_params, final_cv_score, final_cv_desc = cv_evaluation.fit_final_estimator(
        model_name=MODEL_NAME,
        builder=builder,
        param_grid=param_grid,
        X=X,
        y=y,
        run_groups=run_groups,
        random_state=args.random_state,
        n_jobs=effective_n_jobs,
        fit_params=final_fit_params,
        stratify_labels=final_stratify,
        logger=logger,
    )
    logger.info("CNN-Transformer refit using %s", final_cv_desc)
    final_predictions = final_estimator.predict(X)
    final_metrics = metrics.compute_metrics(y.to_numpy(), final_predictions)

    final_pred_df = metrics.build_prediction_frame(
        data=data,
        y_true=y,
        y_pred=final_predictions,
        model_name=MODEL_NAME,
        target_column=target_column,
        target_key=target.key,
    )
    final_pred_path = output_dir / f"final_model_predictions_{MODEL_NAME}.tsv"
    final_pred_path.parent.mkdir(parents=True, exist_ok=True)
    final_pred_df.to_csv(final_pred_path, sep="\t", index=False)

    final_subj_metrics = metrics.compute_group_metrics(final_pred_df, ["subject"])
    final_subj_metrics_path = output_dir / f"final_per_subject_metrics_{MODEL_NAME}.tsv"
    final_subj_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_subj_metrics.to_csv(final_subj_metrics_path, sep="\t", index=False)

    final_temp_metrics = metrics.compute_group_metrics(final_pred_df, ["temp_celsius"])
    final_temp_metrics_path = output_dir / f"final_per_temperature_metrics_{MODEL_NAME}.tsv"
    final_temp_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_temp_metrics.to_csv(final_temp_metrics_path, sep="\t", index=False)

    model_payload = {
        "model": final_estimator,
        "feature_names": feature_columns,
        "bands": list(bands),
        "subjects": subjects,
        "target": {
            "key": target.key,
            "display_name": target.display_name,
            "column": target_column,
            "description": target.description,
        },
        "trained_on": {
            "n_trials": int(len(data)),
            "outer_group_level": outer_group_level,
        },
    }
    joblib.dump(model_payload, output_dir / f"final_model_{MODEL_NAME}.joblib")

    temperature_distribution = temp_counts.to_dict() if isinstance(temp_counts, pd.Series) else {}
    summary = {
        "target": {
            "key": target.key,
            "display_name": target.display_name,
            "column": target_column,
            "description": target.description,
        },
        "bands": list(bands),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_trials": int(len(data)),
        "feature_count": len(feature_columns),
        "feature_to_sample_ratio": feature_ratio,
        "temperature_distribution": temperature_distribution,
        "eeg_plateau_window": plateau_window,
        "temperature_only_r2": temperature_baseline_metrics["r2"] if temperature_baseline_metrics else None,
        "temperature_only_baseline": {
            "metrics": temperature_baseline_metrics,
            "cv_strategy": temperature_baseline_desc,
        }
        if temperature_baseline_metrics
        else None,
        "outer_cv_level": outer_group_level,
        "outer_cv_strategy": result.get("outer_cv_desc"),
        "inner_cv_strategy": result.get("inner_cv_desc"),
        "include_temperature": bool(args.include_temperature),
        "models": [model_entry],
        "best_model": {
            "name": MODEL_NAME,
            "cv_best_score": final_cv_score,
            "final_best_params": final_best_params,
            "final_metrics": final_metrics,
            "prediction_file": final_pred_path.name,
            "per_subject_metrics_file": final_subj_metrics_path.name,
            "per_temperature_metrics_file": final_temp_metrics_path.name,
            "model_artifact_file": f"final_model_{MODEL_NAME}.joblib",
            "refit_cv_strategy": final_cv_desc,
            "permutation_test": model_entry.get("permutation_test"),
        },
        "notes": [
            f"Target signature: {target.display_name} (column '{target_column}').",
            "EEG features derive from the stimulation plateau window and are aligned to delayed hemodynamic responses.",
        ],
    }
    summary["notes"].append(temperature_predictor_note)
    if temperature_baseline_metrics:
        summary["notes"].append(
            "Temperature-only baseline R² = %.3f (%s)."
            % (temperature_baseline_metrics["r2"], temperature_baseline_desc)
        )
    if drops_summary:
        summary["dropped_trials"] = drops_summary

    if headline_r2 is not None:
        primary_metric: Dict[str, Any] = {
            "name": "nested_cv_r2_oof",
            "value": headline_r2,
            "fold_mean": fold_mean_r2,
            "fold_std": fold_std_r2,
            "fold_count": len(r2_values),
        }
        if fold_sem_r2 is not None:
            primary_metric["fold_sem"] = fold_sem_r2
        if fold_ci95 is not None and all(v is not None for v in fold_ci95.values()):
            primary_metric["ci95"] = fold_ci95
        summary["primary_metric"] = primary_metric

    io_utils.write_json(output_dir / "summary.json", summary)
    logger.info("Hybrid CNN-Transformer in-sample R2 (refit on all data)=%.3f", final_metrics["r2"])
    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()

