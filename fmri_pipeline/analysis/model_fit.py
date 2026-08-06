"""Persist exact voxelwise series retained by a fitted first-level model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ModelFitImages:
    """Unwhitened model-response residual and predicted series, ordered by run."""

    residuals: Tuple[Any, ...]
    predicted: Tuple[Any, ...]


@dataclass(frozen=True)
class ModelFitPaths:
    """Written model-fit series, ordered by run."""

    residuals: Tuple[Path, ...]
    predicted: Tuple[Path, ...]


def extract_model_fit_images(flm: Any) -> ModelFitImages:
    """Reconstruct retained-frame ``X beta`` and ``Y - X beta`` images.

    Nilearn's public ``predicted_`` property uses each regression result's whitened
    design while ``residuals_`` subtracts that quantity from the unwhitened response.
    Reconstructing both quantities from the original fitted design keeps predictions
    and residuals in one scientifically coherent model-response space.
    """
    designs = tuple(getattr(flm, "design_matrices_"))
    labels_by_run = tuple(getattr(flm, "labels_"))
    results_by_run = tuple(getattr(flm, "results_"))
    masker = getattr(flm, "masker_")

    run_count = len(designs)
    if run_count == 0 or len(labels_by_run) != run_count or len(results_by_run) != run_count:
        raise ValueError(
            "The fitted model must expose one label and regression-result set per design."
        )

    residuals = []
    predicted = []
    for run_index, (design, labels, results) in enumerate(
        zip(designs, labels_by_run, results_by_run), start=1
    ):
        residual, prediction = _reconstruct_run_images(
            design=design,
            labels=labels,
            results=results,
            masker=masker,
            run_index=run_index,
        )
        _validate_run_images(
            residual,
            prediction,
            design_rows=len(design),
            run_index=run_index,
        )
        residuals.append(residual)
        predicted.append(prediction)

    return ModelFitImages(residuals=tuple(residuals), predicted=tuple(predicted))


def _reconstruct_run_images(
    *,
    design: Any,
    labels: Any,
    results: Any,
    masker: Any,
    run_index: int,
) -> Tuple[Any, Any]:
    """Reconstruct one run in the unwhitened fitted-response space."""
    design_values = np.asarray(design, dtype=float)
    label_values = np.asarray(labels)
    if design_values.ndim != 2:
        raise ValueError(f"Run {run_index} fitted design must be two-dimensional.")
    if label_values.ndim != 1 or label_values.size == 0:
        raise ValueError(f"Run {run_index} fitted labels must be a nonempty vector.")
    if not isinstance(results, Mapping) or not results:
        raise ValueError(f"Run {run_index} fitted results must be a nonempty mapping.")

    response = np.empty((design_values.shape[0], label_values.size), dtype=float)
    prediction = np.empty_like(response)
    assigned_voxels = np.zeros(label_values.size, dtype=bool)

    for label, result in results.items():
        label_mask = label_values == label
        voxel_count = int(label_mask.sum())
        if voxel_count == 0:
            raise ValueError(f"Run {run_index} contains a regression result without voxels.")

        coefficients = np.asarray(getattr(result, "theta"), dtype=float)
        observed = np.asarray(getattr(result, "Y"), dtype=float)
        expected_coefficient_shape = (design_values.shape[1], voxel_count)
        expected_response_shape = (design_values.shape[0], voxel_count)
        if coefficients.shape != expected_coefficient_shape:
            raise ValueError(
                f"Run {run_index} coefficients have shape {coefficients.shape}; "
                f"expected {expected_coefficient_shape}."
            )
        if observed.shape != expected_response_shape:
            raise ValueError(
                f"Run {run_index} response has shape {observed.shape}; "
                f"expected {expected_response_shape}."
            )

        response[:, label_mask] = observed
        prediction[:, label_mask] = np.einsum(
            "ij,jk->ik",
            design_values,
            coefficients,
        )
        assigned_voxels |= label_mask

    if not assigned_voxels.all():
        raise ValueError(f"Run {run_index} has voxels without regression results.")

    residual = response - prediction
    return masker.inverse_transform(residual), masker.inverse_transform(prediction)


def _validate_run_images(
    residual: Any,
    prediction: Any,
    *,
    design_rows: int,
    run_index: int,
) -> None:
    """Require one reconstructed pair to preserve geometry and retained frames."""
    residual_shape = tuple(residual.shape)
    prediction_shape = tuple(prediction.shape)
    if len(residual_shape) != 4 or residual_shape != prediction_shape:
        raise ValueError(f"Run {run_index} residual and predicted series require matching shapes.")
    if not np.allclose(residual.affine, prediction.affine):
        raise ValueError(f"Run {run_index} residual and predicted series require matching affines.")
    if residual_shape[3] != design_rows:
        raise ValueError(f"Run {run_index} model-fit timepoints must equal the fitted design rows.")


def write_model_fit_images(
    images: ModelFitImages,
    *,
    out_dir: Path,
    stem: str,
    cfg_hash: str,
    run_labels: Sequence[str],
) -> ModelFitPaths:
    """Write one explicit residual and predicted 4D image per run."""
    labels = tuple(str(label) for label in run_labels)
    run_count = len(images.residuals)
    if len(images.predicted) != run_count:
        raise ValueError(
            "Model-fit residuals and predicted images must have equal lengths, got "
            f"{run_count} and {len(images.predicted)}."
        )
    if len(labels) != run_count:
        raise ValueError(
            f"run_labels must align with model-fit images, got {len(labels)} and {run_count}."
        )

    import nibabel as nib

    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    residual_paths = []
    predicted_paths = []
    for run_label, residual, prediction in zip(labels, images.residuals, images.predicted):
        residual_path = output_directory / (
            f"{stem}_{run_label}_desc-modelResponseResidual_bold_{cfg_hash}.nii.gz"
        )
        predicted_path = output_directory / (
            f"{stem}_{run_label}_desc-modelResponsePredicted_bold_{cfg_hash}.nii.gz"
        )
        nib.save(residual, str(residual_path))
        nib.save(prediction, str(predicted_path))
        residual_paths.append(residual_path)
        predicted_paths.append(predicted_path)

    return ModelFitPaths(
        residuals=tuple(residual_paths),
        predicted=tuple(predicted_paths),
    )


__all__ = [
    "ModelFitImages",
    "ModelFitPaths",
    "extract_model_fit_images",
    "write_model_fit_images",
]
