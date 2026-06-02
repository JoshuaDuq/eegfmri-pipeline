"""Study 2 source-power association maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourcePowerAssociationMap:
    partial_r: np.ndarray
    fisher_z: np.ndarray
    valid_vertices: np.ndarray


def compute_source_power_association_map(
    *,
    source_power: np.ndarray,
    score: np.ndarray,
    design: np.ndarray,
    variance_tolerance: float = 1e-12,
) -> SourcePowerAssociationMap:
    """Compute subject-level source-power partial correlations."""
    source_arr = np.asarray(source_power, dtype=float)
    score_arr = np.asarray(score, dtype=float)
    design_arr = np.asarray(design, dtype=float)
    _validate_inputs(source_arr, score_arr, design_arr)

    score_residual = _residualize_vector(score_arr, design_arr)
    score_norm = float(np.linalg.norm(score_residual - np.mean(score_residual)))
    if score_norm <= 0.0:
        raise ValueError("Study 2 source-stage score has zero variance after residualization.")

    source_residual = _residualize_matrix(source_arr, design_arr)
    source_centered = source_residual - np.mean(source_residual, axis=0, keepdims=True)
    score_centered = score_residual - float(np.mean(score_residual))
    source_norms = np.linalg.norm(source_centered, axis=0)
    valid_vertices = source_norms > float(variance_tolerance)

    partial_r = np.full(source_arr.shape[1], np.nan, dtype=float)
    partial_r[valid_vertices] = (
        score_centered @ source_centered[:, valid_vertices]
    ) / (score_norm * source_norms[valid_vertices])
    partial_r[valid_vertices] = np.clip(partial_r[valid_vertices], -0.999999, 0.999999)

    fisher_z = np.full(source_arr.shape[1], np.nan, dtype=float)
    fisher_z[valid_vertices] = np.arctanh(partial_r[valid_vertices])
    return SourcePowerAssociationMap(
        partial_r=partial_r,
        fisher_z=fisher_z,
        valid_vertices=valid_vertices,
    )


def _validate_inputs(source_power: np.ndarray, score: np.ndarray, design: np.ndarray) -> None:
    if source_power.ndim != 2:
        raise ValueError(
            f"Study 2 source_power must be 2D, got shape {source_power.shape}."
        )
    if score.ndim != 1:
        raise ValueError(f"Study 2 source-stage score must be 1D, got shape {score.shape}.")
    if design.ndim != 2:
        raise ValueError(f"Study 2 source-stage design must be 2D, got shape {design.shape}.")
    if source_power.shape[0] != len(score) or design.shape[0] != len(score):
        raise ValueError("source_power, score, and design must have the same number of trials.")
    if source_power.shape[0] <= design.shape[1] + 1:
        raise ValueError(
            "Study 2 source-stage association requires more trials than design parameters."
        )
    if not np.all(np.isfinite(source_power)):
        raise ValueError("Study 2 source_power contains non-finite values.")
    if not np.all(np.isfinite(score)):
        raise ValueError("Study 2 source-stage score contains non-finite values.")
    if not np.all(np.isfinite(design)):
        raise ValueError("Study 2 source-stage design contains non-finite values.")
    _validate_design_full_rank(design)


def _validate_design_full_rank(design: np.ndarray) -> None:
    predictors = _scaled_predictors(design)
    rank = int(np.linalg.matrix_rank(predictors))
    if rank < predictors.shape[1]:
        raise ValueError("Study 2 source-stage design must have full column rank.")


def _residualize_vector(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    predictors = _scaled_predictors(design)
    coefficients, *_ = np.linalg.lstsq(predictors, values, rcond=None)
    return values - predictors @ coefficients


def _residualize_matrix(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    predictors = _scaled_predictors(design)
    coefficients, *_ = np.linalg.lstsq(predictors, values, rcond=None)
    return values - predictors @ coefficients


def _scaled_predictors(design: np.ndarray) -> np.ndarray:
    return _with_intercept(_scale_design_columns(design))


def _scale_design_columns(design: np.ndarray) -> np.ndarray:
    centered = design - np.mean(design, axis=0, keepdims=True)
    column_norms = np.linalg.norm(centered, axis=0)
    if np.any(column_norms <= 0.0):
        raise ValueError("Study 2 source-stage design must have full column rank.")
    return centered / column_norms


def _with_intercept(design: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(design.shape[0], dtype=float), design])


__all__ = ["SourcePowerAssociationMap", "compute_source_power_association_map"]
