"""Directional-consistency gates for Study 2 source maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from studies.pain_study.study2.statistics import pearson_r
from studies.pain_study.study2.validation import require_config_float


@dataclass(frozen=True)
class DirectionalConsistencyQC:
    spatial_r: float
    same_sign_fraction: float
    passed: bool
    failed_gates: tuple[str, ...]


def evaluate_directional_consistency(
    *,
    prediction_map: np.ndarray,
    target_map: np.ndarray,
    cluster_mask: np.ndarray,
    config: Any,
) -> DirectionalConsistencyQC:
    prediction = np.asarray(prediction_map, dtype=float)
    target = np.asarray(target_map, dtype=float)
    mask = np.asarray(cluster_mask, dtype=bool)
    _validate_inputs(prediction, target, mask)

    min_spatial_r = require_config_float(
        config,
        "study2.directional_consistency.min_true_target_spatial_r",
    )
    min_same_sign = require_config_float(
        config,
        "study2.directional_consistency.min_cluster_same_sign_fraction",
    )
    spatial_r = pearson_r(prediction, target, name="directional consistency")
    same_sign_fraction = float(np.mean(np.sign(prediction[mask]) == np.sign(target[mask])))

    failed: list[str] = []
    if spatial_r < min_spatial_r:
        failed.append("true_target_spatial_r")
    if same_sign_fraction < min_same_sign:
        failed.append("cluster_same_sign_fraction")
    return DirectionalConsistencyQC(
        spatial_r=spatial_r,
        same_sign_fraction=same_sign_fraction,
        passed=not failed,
        failed_gates=tuple(failed),
    )


def _validate_inputs(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
    if prediction.ndim != 1 or target.ndim != 1 or mask.ndim != 1:
        raise ValueError("Study 2 directional-consistency inputs must be 1D.")
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("Study 2 directional-consistency inputs must have the same shape.")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Study 2 prediction_map contains non-finite values.")
    if not np.all(np.isfinite(target)):
        raise ValueError("Study 2 target_map contains non-finite values.")
    if not np.any(mask):
        raise ValueError("Study 2 cluster_mask must contain at least one vertex.")
    if np.any(prediction[mask] == 0.0) or np.any(target[mask] == 0.0):
        raise ValueError("Study 2 cluster_mask contains zero-valued map entries.")


__all__ = ["DirectionalConsistencyQC", "evaluate_directional_consistency"]
