import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.nn.utils import clip_grad_norm_
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for graph neural network model. Install torch before running this script."
    ) from exc

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError as exc:
    raise ImportError(
        "The torch-geometric package is required for the GNN pipeline. Install torch-geometric before running this script."
    ) from exc

try:
    import mne
except ImportError as exc:
    raise ImportError(
        "mne is required to obtain canonical sensor coordinates for graph construction."
    ) from exc

from ..models.cnn import FiniteValueChecker
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

MODEL_NAME = "gnn"


###################################################################
# Graph Construction Helpers
###################################################################

def _canonicalize_name(name: str) -> str:
    return name.upper().replace(" ", "").replace("EEG", "").replace("-REF", "").replace("REF", "")


def _resolve_montage_positions(channel_names: Sequence[str], logger) -> Dict[str, np.ndarray]:
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()["ch_pos"]
    resolved: Dict[str, np.ndarray] = {}
    missing_channels: List[str] = []
    for ch in channel_names:
        canon = _canonicalize_name(ch)
        if canon in ch_pos:
            resolved[ch] = np.asarray(ch_pos[canon], dtype=float)
            continue
        if ch in ch_pos:
            resolved[ch] = np.asarray(ch_pos[ch], dtype=float)
            continue
        missing_channels.append(ch)
    if missing_channels:
        raise ValueError(
            "The following channels are missing from the standard_1020 montage (check naming/casing): "
            + ", ".join(sorted(missing_channels))
        )
    return resolved


def _build_distance_edges(
    channel_names: Sequence[str],
    channel_positions: Dict[str, np.ndarray],
    *,
    radius: float,
    k_nearest: int,
    sigma: float,
) -> Dict[Tuple[int, int], float]:
    positions = np.stack([channel_positions[ch] for ch in channel_names], axis=0)
    dist_matrix = cdist(positions, positions, metric="euclidean")
    n_channels = len(channel_names)
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(n_channels):
        sorted_idx = np.argsort(dist_matrix[i])
        neighbours = set(sorted_idx[1 : 1 + max(k_nearest, 0)]) if k_nearest > 0 else set()
        for j in range(n_channels):
            if i == j:
                continue
            dist = dist_matrix[i, j]
            if radius > 0 and dist <= radius:
                pass
            elif k_nearest > 0 and j in neighbours:
                pass
            else:
                continue
            if sigma > 0:
                weight = float(np.exp(-dist**2 / (2 * sigma**2)))
            else:
                weight = 1.0
            edges[(i, j)] = max(edges.get((i, j), 0.0), weight)
    return edges


def _build_functional_edges(
    data_matrix: np.ndarray,
    channel_order: Sequence[str],
    channel_indices: Dict[str, List[int]],
    bands: Sequence[str],
    threshold: float,
) -> Dict[Tuple[int, int], float]:
    if threshold <= 0:
        return {}
    n_samples = data_matrix.shape[0]
    channel_signals = np.zeros((n_samples, len(channel_order)), dtype=float)
    for idx, ch in enumerate(channel_order):
        if ch not in channel_indices:
            raise KeyError(f"Channel '{ch}' missing from channel_indices mapping.")
        band_cols = [channel_indices[ch][band_idx] for band_idx in range(len(bands))]
        channel_signals[:, idx] = data_matrix[:, band_cols].mean(axis=1)
    corr = np.corrcoef(channel_signals, rowvar=False)
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(len(channel_order)):
        for j in range(i + 1, len(channel_order)):
            value = corr[i, j]
            if np.isnan(value):
                continue
            if abs(value) >= threshold:
                weight = float(value)
                edges[(i, j)] = weight
                edges[(j, i)] = weight
    return edges


def _merge_edges(
    base_edges: Dict[Tuple[int, int], float],
    functional_edges: Dict[Tuple[int, int], float],
) -> Tuple[np.ndarray, np.ndarray]:
    combined: Dict[Tuple[int, int], float] = dict(base_edges)
    for key, value in functional_edges.items():
        current = combined.get(key)
        if current is None or abs(value) >= abs(current):
            combined[key] = value
    if not combined:
        raise ValueError("No edges were created; adjust radius/k_nearest/functional_threshold parameters.")
    edge_pairs = sorted(combined.items())
    edge_index = np.array([[i, j] for (i, j), _ in edge_pairs], dtype=np.int64).T
    edge_weight = np.array([w for _, w in edge_pairs], dtype=np.float32)
    return edge_index, edge_weight


###################################################################
# PyTorch Geometric Model
###################################################################

class EEGGNNModel(nn.Module):
    def __init__(
        self,
        *,
        node_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        global_dim: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        in_dim = node_feature_dim
        for _ in range(num_layers):
            conv = GCNConv(in_dim, hidden_dim)
            self.convs.append(conv)
            in_dim = hidden_dim
        self.dropout = dropout
        self.global_dim = global_dim
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch):
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_weight
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        pooled = global_mean_pool(x, batch.batch)
        if self.global_dim > 0 and hasattr(batch, "global_features"):
            pooled = torch.cat([pooled, batch.global_features], dim=1)
        return self.readout(pooled).squeeze(-1)


class EEGGraphRegressor(BaseEstimator, RegressorMixin):
    supports_external_validation: bool = True

    def __init__(
        self,
        *,
        feature_names: Optional[List[str]] = None,
        channel_names: Optional[List[str]] = None,
        band_names: Optional[List[str]] = None,
        base_edges: Optional[Dict[Tuple[int, int], float]] = None,
        functional_threshold: float = 0.0,
        global_feature_names: Optional[List[str]] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        max_epochs: int = 300,
        patience: int = 30,
        grad_clip: Optional[float] = None,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.feature_names = feature_names
        self.channel_names = channel_names
        self.band_names = band_names
        self.base_edges = base_edges
        self.functional_threshold = functional_threshold
        self.global_feature_names = global_feature_names or []
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
                raise RuntimeError("CUDA device requested but not available.")
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_mappings(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names must be provided to EEGGraphRegressor.")
        if not self.channel_names:
            raise ValueError("channel_names must be provided to EEGGraphRegressor.")
        if not self.band_names:
            raise ValueError("band_names must be provided to EEGGraphRegressor.")
        self.band_names_ = list(self.band_names)
        self.channel_names_ = list(self.channel_names)
        self.global_feature_names_ = list(self.global_feature_names)
        band_index = {band: idx for idx, band in enumerate(self.band_names_)}
        feature_map: Dict[Tuple[str, str], int] = {}
        for idx, name in enumerate(self.feature_names):
            if not name.startswith("pow_"):
                continue
            parts = name.split("_", 2)
            if len(parts) != 3:
                continue
            _, band, channel = parts
            if band not in band_index:
                continue
            feature_map[(channel, band)] = idx
        missing = [
            (channel, band)
            for channel in self.channel_names_
            for band in self.band_names_
            if (channel, band) not in feature_map
        ]
        if missing:
            raise ValueError(
                "Missing channel-band features for combinations: %s" % (", ".join([f"{c}-{b}" for c, b in missing]),)
            )
        self.feature_map_ = feature_map
        self.channel_band_indices_ = {
            channel: [feature_map[(channel, band)] for band in self.band_names_]
            for channel in self.channel_names_
        }
        self.global_feature_indices_ = [
            self.feature_names.index(name) for name in self.global_feature_names_ if name in self.feature_names
        ]
        node_dim = len(self.band_names_)
        self.node_feature_dim_ = node_dim

    @staticmethod
    def _coerce_groups(
        groups: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any], np.ndarray]],
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
            raise ValueError(f"groups must have length {n_samples}, received {coerced.shape[0]}.")
        if coerced.size <= 1:
            return None
        return coerced

    @staticmethod
    def _coerce_stratify(
        stratify: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any], np.ndarray]],
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
            raise ValueError(f"stratify must have length {n_samples}, received {coerced.shape[0]}.")
        if coerced.size <= 1:
            return None
        return coerced

    def _build_graphs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> List[Data]:
        graphs: List[Data] = []
        n_channels = len(self.channel_names_)
        for row_idx, row in enumerate(X):
            node_features = np.zeros((n_channels, len(self.band_names_)), dtype=np.float32)
            for ch_idx, channel in enumerate(self.channel_names_):
                cols = self.channel_band_indices_[channel]
                node_features[ch_idx, :] = row[cols]
            data_kwargs = {
                "x": torch.from_numpy(node_features),
                "edge_index": self.edge_index_tensor_,
                "edge_weight": self.edge_weight_tensor_,
            }
            if self.global_feature_indices_:
                data_kwargs["global_features"] = torch.from_numpy(
                    row[self.global_feature_indices_].astype(np.float32)
                )
            if y is not None:
                target = float(y[row_idx])
                data_kwargs["y"] = torch.tensor([target], dtype=torch.float32)
            graphs.append(Data(**data_kwargs))
        return graphs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any], np.ndarray]] = None,
        stratify: Optional[Union[pd.Series, pd.DataFrame, Sequence[Any], np.ndarray]] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        validation_indices: Optional[Sequence[int]] = None,
    ) -> "EEGGraphRegressor":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("Expected 2D array for X.")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must contain the same number of samples.")
        if X_arr.shape[0] < 2:
            raise ValueError("At least two samples are required to train the GNN regressor.")

        self._prepare_mappings()
        if self.base_edges is None:
            raise ValueError("base_edges must be provided to EEGGraphRegressor.")
        base_edges = dict(self.base_edges)
        self.base_edges_ = base_edges

        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        n_samples = X_arr.shape[0]
        groups_arr = self._coerce_groups(groups, n_samples)
        stratify_arr = self._coerce_stratify(stratify, n_samples)
        required_levels = (
            {label for label in stratify_arr if not pd.isna(label)} if stratify_arr is not None else set()
        )
        min_groups_required = 0
        if groups_arr is not None:
            unique_groups = np.unique(groups_arr)
            if unique_groups.size >= 2:
                min_groups_required = min(2, unique_groups.size)
            elif self.patience > 0:
                raise ValueError(
                    "EEGGraphRegressor requires at least two distinct groups for validation-driven early stopping. "
                    "Provide additional grouping structure or disable early stopping (patience=0)."
                )
        min_val_required = max(5, len(required_levels), min_groups_required)
        min_val_required = min(min_val_required, max(1, n_samples - 1))

        def covers_levels(indices: np.ndarray) -> bool:
            if stratify_arr is None or not required_levels:
                return True
            return required_levels.issubset(
                {label for label in stratify_arr[indices] if not pd.isna(label)}
            )

        def covers_groups(indices: np.ndarray) -> bool:
            if groups_arr is None or min_groups_required == 0:
                return True
            return np.unique(groups_arr[indices]).size >= min_groups_required

        if validation_data is not None and validation_indices is not None:
            raise ValueError("Provide either validation_data or validation_indices, not both.")
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
            if val_X_arr.shape[1] != X_arr.shape[1]:
                raise ValueError(
                    "validation_data feature dimension %d does not match training features %d."
                    % (val_X_arr.shape[1], X_arr.shape[1])
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

        val_idx: Optional[np.ndarray] = None
        if validation_indices is not None:
            try:
                candidate_idx = np.asarray(validation_indices, dtype=int)
            except Exception as exc:
                raise ValueError("validation_indices must be an iterable of integers.") from exc
            if candidate_idx.ndim != 1:
                raise ValueError("validation_indices must be one-dimensional.")
            if candidate_idx.size == 0:
                val_idx = None
            else:
                if (candidate_idx < 0).any() or (candidate_idx >= n_samples).any():
                    raise ValueError("validation_indices contain out-of-range values.")
                candidate_idx = np.unique(candidate_idx)
                if candidate_idx.size < min_val_required:
                    raise ValueError(
                        f"validation_indices must contain at least {min_val_required} samples."
                    )
                if not covers_levels(candidate_idx) or not covers_groups(candidate_idx):
                    raise ValueError(
                        "validation_indices do not satisfy required temperature/group coverage."
                    )
                val_idx = candidate_idx

        if n_samples > 8 and self.patience > 0:
            val_fraction = min(0.2, max(1.0 / n_samples, 0.1))
            target_val = max(1, int(round(n_samples * val_fraction)))
            target_val = min(target_val, n_samples - 1)
            target_val = max(target_val, min_val_required)
            if val_idx is None and validation_data is None and groups_arr is not None and np.unique(groups_arr).size > 1:
                test_size = max(val_fraction, target_val / n_samples)
                test_size = min(0.5, max(test_size, 1.0 / np.unique(groups_arr).size))
                base_state = self.random_state if isinstance(self.random_state, (int, np.integer)) else None
                for attempt in range(10):
                    random_state = None if base_state is None else base_state + attempt
                    splitter = GroupShuffleSplit(
                        n_splits=1,
                        test_size=test_size,
                        random_state=random_state,
                    )
                    try:
                        _, candidate = next(
                            splitter.split(
                                np.zeros((n_samples, 1)),
                                groups=groups_arr,
                            )
                        )
                    except ValueError:
                        candidate = np.array([], dtype=int)
                    candidate = np.asarray(candidate, dtype=int)
                    if candidate.size < target_val or candidate.size >= n_samples:
                        continue
                    if not covers_levels(candidate) or not covers_groups(candidate):
                        continue
                    val_idx = np.sort(candidate)
                    break
                if val_idx is None:
                    unique_groups = rng.permutation(np.unique(groups_arr))
                    selected: List[int] = []
                    for group in unique_groups:
                        selected.extend(np.where(groups_arr == group)[0].tolist())
                        if len(selected) >= target_val:
                            candidate = np.array(sorted(set(selected)), dtype=int)
                            if candidate.size < n_samples:
                                if not covers_levels(candidate) or not covers_groups(candidate):
                                    continue
                                val_idx = candidate
                                break
            if (
                val_idx is None
                and validation_data is None
                and groups_arr is None
                and stratify_arr is not None
                and len(required_levels) > 0
            ):
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=max(val_fraction, target_val / n_samples),
                    random_state=self.random_state,
                )
                try:
                    _, candidate = next(
                        splitter.split(
                            np.zeros((n_samples, 1)),
                            stratify_arr,
                        )
                    )
                except ValueError:
                    candidate = np.array([], dtype=int)
                candidate = np.asarray(candidate, dtype=int)
                if 0 < candidate.size < n_samples:
                    if covers_levels(candidate):
                        val_idx = np.sort(candidate)
            if (
                val_idx is None
                and validation_data is None
                and groups_arr is None
                and len(required_levels) == 0
                and n_samples > target_val
            ):
                indices = rng.permutation(n_samples)
                val_idx = np.sort(indices[:target_val])
        needs_validation = self.patience > 0 and (groups_arr is not None or len(required_levels) > 1)
        if val_idx is None and validation_data is None and needs_validation:
            raise ValueError(
                "Unable to construct a validation split that preserves group/temperature coverage."
            )

        self.validation_indices_ = None if val_idx is None else np.asarray(val_idx, dtype=int)

        if val_idx is not None:
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
        else:
            train_idx = np.arange(n_samples, dtype=int)

        if train_idx.size == 0:
            raise ValueError("Internal training split is empty; cannot compute functional connectivity edges.")
        train_subset = X_arr[train_idx]

        min_edge_samples = max(10, 2 * len(self.band_names_))
        if train_subset.shape[0] < min_edge_samples:
            if self.verbose:
                print(
                    f"Skipping functional connectivity edges (train samples={train_subset.shape[0]} < {min_edge_samples})."
                )
            functional_edges: Dict[Tuple[int, int], float] = {}
        else:
            functional_edges = _build_functional_edges(
                train_subset,
                self.channel_names_,
                self.channel_band_indices_,
                self.band_names_,
                threshold=self.functional_threshold,
            )
        edge_index, edge_weight = _merge_edges(base_edges, functional_edges)
        self.edge_index_ = edge_index
        self.edge_weight_ = edge_weight
        self.functional_edges_ = functional_edges
        self.functional_threshold_ = self.functional_threshold
        self.edge_index_tensor_ = torch.from_numpy(edge_index.astype(np.int64))
        self.edge_weight_tensor_ = torch.from_numpy(edge_weight.astype(np.float32))

        graphs = self._build_graphs(X_arr, y_arr)

        if val_idx is not None:
            val_graphs = [graphs[i] for i in val_idx]
            train_graphs = [graphs[i] for i in train_idx]
        else:
            train_graphs = graphs
            val_graphs = []

        device = self._resolve_device()

        train_loader = DataLoader(
            train_graphs,
            batch_size=min(self.batch_size, len(train_graphs)),
            shuffle=True,
        )
        val_loader = (
            DataLoader(val_graphs, batch_size=min(self.batch_size, len(val_graphs))) if val_graphs else None
        )

        global_dim = len(self.global_feature_indices_)
        model = EEGGNNModel(
            node_feature_dim=self.node_feature_dim_,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            global_dim=global_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        history: List[Dict[str, float]] = []
        best_state = None
        best_loss = float("inf")
        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(self.max_epochs):
            model.train()
            epoch_losses: List[float] = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                preds = model(batch)
                loss = criterion(preds, batch.y)
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
                    for batch in val_loader:
                        batch = batch.to(device)
                        preds = model(batch)
                        loss = criterion(preds, batch.y)
                        val_losses.append(float(loss.detach().cpu().item()))
                monitored_loss = float(np.mean(val_losses)) if val_losses else train_loss
            else:
                monitored_loss = train_loss

            history.append({"epoch": epoch + 1, "train_loss": train_loss, "monitor_loss": monitored_loss})
            improved = monitored_loss < (best_loss - 1e-6)
            if improved:
                best_loss = monitored_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
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
            best_epoch = len(history) if history else 1
        elif best_epoch <= 0:
            best_epoch = len(history) if history else 1

        model.load_state_dict(best_state)
        model = model.to(torch.device("cpu"))
        model.eval()

        self.model_ = model
        self.history_ = history
        self.best_loss_ = best_loss
        self.best_epoch_ = int(best_epoch)
        self.input_features_ = X_arr.shape[1]
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

        graphs = self._build_graphs(X_arr, y=None)
        loader = DataLoader(graphs, batch_size=min(len(graphs), 256))
        preds: List[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(torch.device("cpu"))
                outputs = self.model_(batch)
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


def make_gnn_builder(
    *,
    device: str,
    verbose: bool,
    feature_names: List[str],
    channel_names: List[str],
    band_names: List[str],
    base_edges: Dict[Tuple[int, int], float],
    functional_threshold: float,
    global_feature_names: List[str],
) -> Callable:
    def builder(random_state: int, _: int) -> Pipeline:
        regressor = EEGGraphRegressor(
            feature_names=feature_names,
            channel_names=channel_names,
            band_names=band_names,
            base_edges=dict(base_edges),
            functional_threshold=functional_threshold,
            global_feature_names=global_feature_names,
            random_state=random_state,
            device=device,
            verbose=verbose,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("finite_check", FiniteValueChecker()),
                ("gnn", regressor),
            ]
        )

    return builder


def build_gnn_fit_params(meta_subset: pd.DataFrame) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    group_cols = [col for col in ("subject", "run") if col in meta_subset.columns]
    if group_cols:
        params["gnn__groups"] = meta_subset[group_cols].reset_index(drop=True)
    if "temp_celsius" in meta_subset.columns:
        params["gnn__stratify"] = meta_subset["temp_celsius"].to_numpy()
    return params


###################################################################
# Argument Parsing
###################################################################

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NPS beta responses from EEG oscillatory power with a graph neural network."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs. Defaults to machine_learning/outputs/eeg_to_signature_gnn_<timestamp>.",
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
        help="Include stimulus temperature as a global graph covariate.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for grid searches (values <=0 fallback to 1).",
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
        "--hidden-dim-grid",
        type=int,
        nargs="*",
        default=[64, 96],
        help="Grid of hidden dimensions for graph convolutions.",
    )
    parser.add_argument(
        "--num-layers-grid",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Grid of graph convolution layer counts.",
    )
    parser.add_argument(
        "--dropout-grid",
        type=float,
        nargs="*",
        default=[0.2, 0.4],
        help="Grid of dropout probabilities.",
    )
    parser.add_argument(
        "--learning-rate-grid",
        type=float,
        nargs="*",
        default=[5e-4, 1e-3],
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
        default=[16, 32],
        help="Grid of batch sizes.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Maximum training epochs per fit call.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
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
        "--verbose-gnn",
        action="store_true",
        help="Print per-epoch training progress for the GNN estimator.",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=0.08,
        help="Maximum Euclidean distance (meters) between electrodes for spatial edges (0 disables).",
    )
    parser.add_argument(
        "--edge-k-nearest",
        type=int,
        default=6,
        help="Number of nearest neighbours to connect when radius is insufficient (0 disables).",
    )
    parser.add_argument(
        "--distance-sigma",
        type=float,
        default=0.05,
        help="Gaussian kernel sigma for distance-based edge weights (<=0 uses unweighted edges).",
    )
    parser.add_argument(
        "--functional-threshold",
        type=float,
        default=0.5,
        help="Absolute correlation threshold for functional connectivity edges (<=0 disables).",
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
        
        graph_config = config_loader.get_model_config(config, "graph")
        if graph_config:
            graph_construction = graph_config.get("graph_construction", {})
            if not hasattr(args, "radius") or args.radius is None:
                args.radius = graph_construction.get("radius", 0.0)
            if not hasattr(args, "k_nearest") or args.k_nearest is None:
                args.k_nearest = graph_construction.get("k_nearest", 5)
            if not hasattr(args, "sigma") or args.sigma is None:
                args.sigma = graph_construction.get("sigma", 0.1)
            if not hasattr(args, "functional_threshold") or args.functional_threshold is None:
                args.functional_threshold = graph_construction.get("functional_threshold", 0.3)
            
            if not args.hidden_dim_grid:
                args.hidden_dim_grid = graph_config.get("hidden_dim_grid", [64, 96])
            if not args.num_layers_grid:
                args.num_layers_grid = graph_config.get("num_layers_grid", [2, 3])
            if not args.dropout_grid:
                args.dropout_grid = graph_config.get("dropout_grid", [0.2, 0.4])
            if not args.global_dim_grid:
                args.global_dim_grid = graph_config.get("global_dim_grid", [16, 32])
            if not args.learning_rate_grid:
                args.learning_rate_grid = graph_config.get("learning_rate_grid", [5e-4, 1e-3])
            if not args.weight_decay_grid:
                args.weight_decay_grid = graph_config.get("weight_decay_grid", [1e-4, 5e-4])
            if not args.batch_size_grid:
                args.batch_size_grid = graph_config.get("batch_size_grid", [16, 32])
            if not hasattr(args, "max_epochs") or args.max_epochs is None:
                args.max_epochs = graph_config.get("max_epochs", 200)
            if not hasattr(args, "patience") or args.patience is None:
                args.patience = graph_config.get("patience", 25)
            if not hasattr(args, "grad_clip") or args.grad_clip is None:
                args.grad_clip = graph_config.get("grad_clip")
            if not hasattr(args, "verbose_gnn"):
                args.verbose_gnn = graph_config.get("verbose", False)
    
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

    hidden_grid = [int(v) for v in normalize_grid(args.hidden_dim_grid, value_type="int") if int(v) > 0]
    if not hidden_grid:
        raise ValueError("hidden_dim grid must contain positive integers.")
    layers_grid = [int(v) for v in normalize_grid(args.num_layers_grid, value_type="int") if int(v) > 0]
    if not layers_grid:
        raise ValueError("num_layers grid must contain positive integers.")
    dropout_grid = [float(v) for v in normalize_grid(args.dropout_grid, value_type="float") if 0 <= float(v) < 1]
    if not dropout_grid:
        raise ValueError("dropout grid must contain probabilities in [0, 1).")
    lr_grid = [float(v) for v in normalize_grid(args.learning_rate_grid, value_type="float") if float(v) > 0]
    if not lr_grid:
        raise ValueError("learning-rate grid must contain positive floats.")
    weight_decay_grid = [float(v) for v in normalize_grid(args.weight_decay_grid, value_type="float") if float(v) >= 0]
    if not weight_decay_grid:
        raise ValueError("weight-decay grid must contain non-negative floats.")
    batch_grid = [int(v) for v in normalize_grid(args.batch_size_grid, value_type="int") if int(v) > 0]
    if not batch_grid:
        raise ValueError("batch-size grid must contain positive integers.")

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        Path(__file__).resolve().parent.parent / "outputs" / f"eeg_to_{target.key}_gnn_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = io_utils.setup_logging(output_dir, logger_name=f"eeg_to_{target.key}_gnn")
    logger.info("EEG -> %s GNN training pipeline started.", target.display_name)
    logger.info("Using bands: %s", ", ".join(bands))
    logger.info(
        "Hyperparameter grid | hidden: %s | layers: %s | dropout: %s | lr: %s | weight_decay: %s | batch: %s",
        hidden_grid,
        layers_grid,
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

    channel_feature_columns = feature_utils.select_direct_power_columns(master_features, bands)
    if not channel_feature_columns:
        raise ValueError("No direct sensor power columns were found for the requested bands.")

    global_feature_columns: List[str] = []
    if args.include_temperature:
        global_feature_columns.append("temp_celsius")

    feature_columns = channel_feature_columns + global_feature_columns

    data = pd.concat([res.data for res in subject_results], ignore_index=True)
    missing_global = [col for col in global_feature_columns if col not in data.columns]
    if missing_global:
        raise ValueError(f"Global feature columns missing from dataset: {missing_global}")

    feature_columns = feature_utils.filter_zero_variance_features(data, feature_columns, logger=logger)
    channel_feature_columns = [col for col in channel_feature_columns if col in feature_columns]
    global_feature_columns = [col for col in global_feature_columns if col in feature_columns]
    if not channel_feature_columns:
        raise ValueError("All channel features were removed due to zero variance; cannot build graphs.")
    channel_names = sorted({name.split("_", 2)[-1] for name in channel_feature_columns})

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
        "Assembled dataset: %d trials, %d subjects, %d features (channels=%d, global=%d).",
        len(data),
        len(subjects),
        len(feature_columns),
        len(channel_feature_columns),
        len(global_feature_columns),
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
        "Graph construction: nodes=%d electrodes, node features=bands (%s), spatial radius=%.3f m, k-nearest=%d, sigma=%.3f, functional threshold=%.2f",
        len(channel_names),
        ", ".join(bands),
        args.edge_radius,
        args.edge_k_nearest,
        args.distance_sigma,
        args.functional_threshold,
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
            "Unable to determine EEG plateau window from configuration; verify EEG/NPS alignment before training."
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

    temperature_predictor_note = (
        "Temperature covariate included as predictor alongside EEG-derived graph features."
        if args.include_temperature and "temp_celsius" in feature_columns
        else "Temperature covariate excluded from predictors; per-temperature diagnostics reflect distributional balance only."
    )
    logger.info(temperature_predictor_note)

    channel_positions = _resolve_montage_positions(channel_names, logger)
    base_edges = _build_distance_edges(
        channel_names,
        channel_positions,
        radius=args.edge_radius,
        k_nearest=args.edge_k_nearest,
        sigma=args.distance_sigma,
    )

    builder = make_gnn_builder(
        device=device,
        verbose=args.verbose_gnn,
        feature_names=feature_columns,
        channel_names=channel_names,
        band_names=list(bands),
        base_edges=base_edges,
        functional_threshold=args.functional_threshold,
        global_feature_names=global_feature_columns,
    )

    param_grid = {
        "gnn__hidden_dim": hidden_grid,
        "gnn__num_layers": layers_grid,
        "gnn__dropout": dropout_grid,
        "gnn__lr": lr_grid,
        "gnn__weight_decay": weight_decay_grid,
        "gnn__batch_size": batch_grid,
        "gnn__max_epochs": [int(args.max_epochs)],
        "gnn__patience": [int(args.patience)],
    }
    if args.grad_clip is not None and args.grad_clip > 0:
        param_grid["gnn__grad_clip"] = [float(args.grad_clip)]

    grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    if grid_size * 5 > len(X):
        logger.warning(
            "GNN grid (%d combinations) is large relative to available trials (%d); consider trimming the search space.",
            grid_size,
            len(X),
        )

    effective_n_jobs = 1
    if args.n_jobs not in (None, 1):
        logger.warning("GNN pipeline forcing n_jobs=%d to ensure deterministic Torch training.", effective_n_jobs)

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
        fit_params_fn=build_gnn_fit_params,
    )

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
    io_utils.write_json(metrics_path, result["summary_metrics"])

    best_params_path = output_dir / f"best_params_{MODEL_NAME}.json"
    io_utils.write_json(best_params_path, [fold["best_params"] for fold in result["fold_details"]])

    r2_values = [fold["test_r2"] for fold in result["fold_details"]]
    graph_overview = {
        "channels": channel_names,
        "bands": list(bands),
        "edge_radius": args.edge_radius,
        "edge_k_nearest": args.edge_k_nearest,
        "distance_sigma": args.distance_sigma,
        "functional_threshold": args.functional_threshold,
        "base_edge_count": int(len(base_edges)),
        "functional_edges_recomputed_per_fold": True,
        "internal_validation_group_aware": True,
    }
    model_entry: Dict[str, Any] = {
        "name": MODEL_NAME,
        "metrics": result["summary_metrics"],
        "fold_mean_r2": float(np.mean(r2_values)) if r2_values else None,
        "fold_std_r2": float(np.std(r2_values)) if r2_values else None,
        "prediction_file": pred_path.name,
        "per_subject_metrics_file": subj_metrics_path.name,
        "per_temperature_metrics_file": temp_metrics_path.name,
        "fold_details_file": fold_path.name if fold_path else None,
        "metrics_file": metrics_path.name,
        "best_params_file": best_params_path.name,
        "outer_cv": result.get("outer_cv_desc"),
        "inner_cv": result.get("inner_cv_desc"),
        "param_grid": param_grid,
        "graph": graph_overview,
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
            true_r2=result["summary_metrics"]["r2"],
            random_state=
                args.permutation_seed if args.permutation_seed is not None else args.random_state,
            n_jobs=effective_n_jobs,
            fit_params_fn=build_gnn_fit_params,
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

    final_stratify = data["temp_celsius"] if "temp_celsius" in data.columns else None
    final_fit_params = build_gnn_fit_params(data)
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
    logger.info("GNN refit using %s", final_cv_desc)
    final_predictions = final_estimator.predict(X)
    final_metrics = metrics.compute_metrics(y.to_numpy(), final_predictions)
    final_gnn: EEGGraphRegressor = final_estimator.named_steps["gnn"]
    final_edge_index = getattr(final_gnn, "edge_index_", None)
    final_edge_weight = getattr(final_gnn, "edge_weight_", None)
    if final_edge_index is None or final_edge_weight is None:
        raise RuntimeError("Fitted EEGGraphRegressor is missing edge attributes; ensure fit() was called.")
    final_graph_details = dict(graph_overview)
    final_graph_details.update(
        {
            "edge_index": final_edge_index.tolist(),
            "edge_weight": final_edge_weight.tolist(),
            "total_edge_count": int(final_edge_index.shape[1]),
            "functional_edge_count": int(len(getattr(final_gnn, "functional_edges_", {}))),
            "global_features": global_feature_columns,
        }
    )

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
        "graph": final_graph_details,
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
            "graph": final_graph_details,
        },
        "notes": [
            f"Target signature: {target.display_name} (column '{target_column}').",
            "EEG graph edges combine spatial proximity (standard_1020 montage) and functional correlations above the requested threshold.",
            "Node features contain log-power for each requested band, preserving per-electrode spectral structure.",
            "Functional connectivity edges are re-estimated inside each training fold to avoid information leakage.",
            "Early stopping validation splits are run-aware and derived solely from training folds.",
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

    io_utils.write_json(output_dir / "summary.json", summary)
    logger.info("GNN final R2=%.3f", final_metrics["r2"])
    logger.info("All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()

