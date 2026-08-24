"""Exact run-level measurements from persisted fitted-model series."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from fmri_pipeline.analysis.report.figures._validation import validated_binary_mask

Quartiles = Tuple[float, float, float]

MODEL_FIT_CHUNK_FRAMES = 16


@dataclass(frozen=True)
class ResidualStandardDeviationMap:
    """Pooled temporal residual standard deviation on the fitted-mask grid."""

    image: object
    run_count: int
    retained_frames: int
    voxel_count: int


@dataclass(frozen=True)
class RSquaredMap:
    """Pooled voxelwise model R² on the fitted-mask grid."""

    image: object
    run_count: int
    retained_frames: int
    voxel_count: int


def pooled_r_squared(
    *,
    residual_paths: Sequence[Path],
    predicted_paths: Sequence[Path],
    mask_path: Path,
) -> RSquaredMap:
    """Compute whole-model R² pooled across runs without mixing run means."""
    import nibabel as nib

    residuals = tuple(Path(path) for path in residual_paths)
    predictions = tuple(Path(path) for path in predicted_paths)
    if not residuals or len(residuals) != len(predictions):
        raise ValueError("The R² map requires equal non-empty residual and prediction series.")

    mask_image = nib.load(str(mask_path))
    mask = validated_binary_mask(mask_image)
    voxel_count = int(mask.sum())

    residual_sum_squares = np.zeros(voxel_count, dtype=np.float64)
    total_sum_squares = np.zeros(voxel_count, dtype=np.float64)
    retained_frames = 0
    for run_index, (residual_path, predicted_path) in enumerate(
        zip(residuals, predictions), start=1
    ):
        residual_image = nib.load(str(residual_path))
        predicted_image = nib.load(str(predicted_path))
        _validate_run_geometry(
            label=f"Run {run_index}",
            residual_image=residual_image,
            predicted_image=predicted_image,
            mask_image=mask_image,
        )
        if residual_image.shape[3] < 2:
            raise ValueError(f"Run {run_index} requires at least two retained frames.")

        run_mean = np.zeros(voxel_count, dtype=np.float64)
        run_sum_squared_deviations = np.zeros(voxel_count, dtype=np.float64)
        run_frames = 0
        for frame_start in range(0, residual_image.shape[3], MODEL_FIT_CHUNK_FRAMES):
            frame_stop = min(
                frame_start + MODEL_FIT_CHUNK_FRAMES,
                residual_image.shape[3],
            )
            frame_slice = (..., slice(frame_start, frame_stop))
            residual = np.asarray(
                residual_image.dataobj[frame_slice],
                dtype=np.float32,
            )[mask]
            predicted = np.asarray(
                predicted_image.dataobj[frame_slice],
                dtype=np.float32,
            )[mask]
            if not np.isfinite(residual).all() or not np.isfinite(predicted).all():
                raise ValueError(f"Run {run_index} model-fit series contain non-finite values.")

            observed = predicted + residual
            chunk_frames = int(observed.shape[1])
            chunk_mean = observed.mean(axis=1, dtype=np.float64)
            chunk_centered = observed - chunk_mean[:, np.newaxis]
            chunk_sum_squared_deviations = np.einsum(
                "ij,ij->i",
                chunk_centered,
                chunk_centered,
                dtype=np.float64,
            )
            combined_frames = run_frames + chunk_frames
            mean_difference = chunk_mean - run_mean
            run_sum_squared_deviations += chunk_sum_squared_deviations
            run_sum_squared_deviations += (
                mean_difference**2 * run_frames * chunk_frames / combined_frames
            )
            run_mean += mean_difference * chunk_frames / combined_frames
            run_frames = combined_frames

            residual_sum_squares += np.einsum(
                "ij,ij->i",
                residual,
                residual,
                dtype=np.float64,
            )

        total_sum_squares += run_sum_squared_deviations
        retained_frames += run_frames

    if np.any(total_sum_squares <= 0):
        raise ValueError("The R² map contains voxels with undefined total variation.")
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    if not np.isfinite(r_squared).all():
        raise ValueError("The R² map contains non-finite fitted-mask values.")

    volume = np.zeros(mask_image.shape, dtype=np.float32)
    volume[mask] = r_squared
    header = mask_image.header.copy()
    header.set_data_dtype(np.float32)
    return RSquaredMap(
        image=nib.Nifti1Image(volume, mask_image.affine, header),
        run_count=len(residuals),
        retained_frames=retained_frames,
        voxel_count=voxel_count,
    )


def pooled_residual_standard_deviation(
    *,
    residual_paths: Sequence[Path],
    mask_path: Path,
) -> ResidualStandardDeviationMap:
    """Compute population SD across every retained residual sample."""
    import nibabel as nib

    paths = tuple(Path(path) for path in residual_paths)
    if not paths:
        raise ValueError("The residual SD map requires at least one residual series.")

    mask_image = nib.load(str(mask_path))
    if len(mask_image.shape) != 3:
        raise ValueError(f"The fitted analysis mask must be 3D, got {mask_image.shape}.")
    mask = np.asanyarray(mask_image.dataobj).astype(bool)
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        raise ValueError("The fitted analysis mask contains no voxels.")

    pooled_mean = np.zeros(voxel_count, dtype=np.float64)
    pooled_sum_squared_deviations = np.zeros(voxel_count, dtype=np.float64)
    retained_frames = 0
    for run_index, path in enumerate(paths, start=1):
        residual_image = nib.load(str(path))
        if len(residual_image.shape) != 4 or residual_image.shape[:3] != mask_image.shape:
            raise ValueError(
                f"Run {run_index} residual series do not match the fitted analysis mask."
            )
        if not np.allclose(residual_image.affine, mask_image.affine):
            raise ValueError(f"Run {run_index} residual series do not share the mask affine.")

        run_frames = int(residual_image.shape[3])
        if run_frames < 1:
            raise ValueError(f"Run {run_index} residual series contain no retained frames.")
        residual = np.asarray(residual_image.dataobj, dtype=np.float32)[mask]
        if not np.isfinite(residual).all():
            raise ValueError(f"Run {run_index} residual series contain non-finite values.")

        run_mean = residual.mean(axis=1, dtype=np.float64)
        run_sum_squared_deviations = residual.var(axis=1, dtype=np.float64) * run_frames
        combined_frames = retained_frames + run_frames
        mean_difference = run_mean - pooled_mean
        pooled_sum_squared_deviations += run_sum_squared_deviations
        pooled_sum_squared_deviations += (
            mean_difference**2 * retained_frames * run_frames / combined_frames
        )
        pooled_mean += mean_difference * run_frames / combined_frames
        retained_frames = combined_frames

    if retained_frames < 2:
        raise ValueError("The residual SD map requires at least two retained frames.")
    residual_variance = pooled_sum_squared_deviations / retained_frames
    if not np.isfinite(residual_variance).all() or np.any(residual_variance <= 0):
        raise ValueError("The residual SD map contains undefined fitted-mask voxels.")

    residual_standard_deviation = np.sqrt(residual_variance)
    volume = np.zeros(mask_image.shape, dtype=np.float32)
    volume[mask] = residual_standard_deviation
    header = mask_image.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(volume, mask_image.affine, header)
    return ResidualStandardDeviationMap(
        image=image,
        run_count=len(paths),
        retained_frames=retained_frames,
        voxel_count=voxel_count,
    )


@dataclass(frozen=True)
class ResidualCarpet:
    """Retained residuals placed on the acquired-frame axis."""

    values: np.ndarray
    tissue_codes: np.ndarray | None
    run_boundaries: Tuple[int, ...]
    not_retained: np.ndarray
    total_voxels: int


def collect_residual_carpet(
    *,
    residual_paths: Sequence[Path],
    retained_frame_indices: Sequence[Sequence[int]],
    acquired_frame_counts: Sequence[int],
    mask_path: Path,
    tissue_codes: np.ndarray | None,
    max_rows: int = 6000,
) -> ResidualCarpet:
    """Place standardized fitted residuals at their exact acquired-frame indices."""
    import nibabel as nib

    paths = tuple(Path(path) for path in residual_paths)
    retained_by_run = tuple(tuple(indices) for indices in retained_frame_indices)
    acquired_by_run = tuple(int(count) for count in acquired_frame_counts)
    run_count = len(paths)
    if run_count == 0:
        raise ValueError("The residual carpet requires at least one run.")
    if len(retained_by_run) != run_count or len(acquired_by_run) != run_count:
        raise ValueError(
            "Residual paths, retained frame indices, and acquired frame counts must align."
        )

    mask_image = nib.load(str(mask_path))
    if len(mask_image.shape) != 3:
        raise ValueError(f"The fitted analysis mask must be 3D, got {mask_image.shape}.")
    mask = np.asanyarray(mask_image.dataobj).astype(bool)
    total_voxels = int(mask.sum())
    if total_voxels == 0:
        raise ValueError("The fitted analysis mask contains no voxels.")

    codes = None if tissue_codes is None else np.asarray(tissue_codes)
    if codes is not None and (codes.ndim != 1 or codes.size != total_voxels):
        raise ValueError("Residual-carpet tissue codes must align with fitted-mask voxels.")
    row_indices = _residual_carpet_row_indices(
        total_voxels,
        codes,
        max_rows=max_rows,
    )
    sampled_codes = None if codes is None else codes[row_indices]

    runs = []
    not_retained_runs = []
    run_boundaries = []
    elapsed_frames = 0
    for run_index, (path, retained, acquired_frames) in enumerate(
        zip(paths, retained_by_run, acquired_by_run),
        start=1,
    ):
        image = nib.load(str(path))
        if len(image.shape) != 4 or image.shape[:3] != mask_image.shape:
            raise ValueError(
                f"Run {run_index} residual series do not match the fitted analysis mask."
            )
        if not np.allclose(image.affine, mask_image.affine):
            raise ValueError(f"Run {run_index} residual series do not share the mask affine.")

        indices = np.asarray(retained, dtype=int)
        _validate_retained_indices(
            indices,
            residual_frames=int(image.shape[3]),
            acquired_frames=acquired_frames,
            run_index=run_index,
        )
        residual = np.asarray(image.dataobj, dtype=np.float32)[mask][row_indices]
        if not np.isfinite(residual).all():
            raise ValueError(f"Run {run_index} residual series contain non-finite values.")
        standardized = _standardize_residual_rows(residual, run_index=run_index)

        expanded = np.full(
            (row_indices.size, acquired_frames),
            np.nan,
            dtype=np.float32,
        )
        expanded[:, indices] = standardized
        runs.append(expanded)

        not_retained = np.ones(acquired_frames, dtype=bool)
        not_retained[indices] = False
        not_retained_runs.append(not_retained)
        elapsed_frames += acquired_frames
        if run_index < run_count:
            run_boundaries.append(elapsed_frames)

    return ResidualCarpet(
        values=np.concatenate(runs, axis=1),
        tissue_codes=sampled_codes,
        run_boundaries=tuple(run_boundaries),
        not_retained=np.concatenate(not_retained_runs),
        total_voxels=total_voxels,
    )


def _residual_carpet_row_indices(
    total_voxels: int,
    tissue_codes: np.ndarray | None,
    *,
    max_rows: int,
    min_rows_per_class: int = 200,
) -> np.ndarray:
    """Select deterministic fitted-mask rows before concatenating runs."""
    if max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}.")
    if total_voxels <= max_rows:
        return np.arange(total_voxels, dtype=int)
    if tissue_codes is None:
        return np.linspace(0, total_voxels - 1, max_rows).astype(int)

    selected = []
    for code in np.unique(tissue_codes):
        positions = np.flatnonzero(tissue_codes == code)
        proportional_share = int(round(max_rows * positions.size / total_voxels))
        take = min(positions.size, max(proportional_share, min_rows_per_class))
        stride = np.linspace(0, positions.size - 1, take).astype(int)
        selected.append(positions[stride])
    return np.sort(np.concatenate(selected))


def _validate_retained_indices(
    indices: np.ndarray,
    *,
    residual_frames: int,
    acquired_frames: int,
    run_index: int,
) -> None:
    """Require one retained-frame vector to define an exact temporal mapping."""
    if acquired_frames < 1:
        raise ValueError(f"Run {run_index} acquired frame count must be positive.")
    if indices.ndim != 1 or indices.size != residual_frames:
        raise ValueError(f"Run {run_index} retained indices do not match residual timepoints.")
    if indices.size < 2:
        raise ValueError(f"Run {run_index} residual carpet requires two retained frames.")
    if np.any(np.diff(indices) <= 0):
        raise ValueError(f"Run {run_index} retained frame indices must increase strictly.")
    if indices[0] < 0 or indices[-1] >= acquired_frames:
        raise ValueError(f"Run {run_index} retained frame indices exceed acquired frames.")


def _standardize_residual_rows(
    residual: np.ndarray,
    *,
    run_index: int,
) -> np.ndarray:
    """Standardize each sampled residual voxel within one fitted run."""
    mean = residual.mean(axis=1, keepdims=True, dtype=np.float64)
    standard_deviation = residual.std(axis=1, keepdims=True, dtype=np.float64)
    if np.any(standard_deviation <= 0):
        raise ValueError(
            f"Run {run_index} contains residual voxels with undefined standardization."
        )
    return np.asarray((residual - mean) / standard_deviation, dtype=np.float32)


@dataclass(frozen=True)
class RunModelFitMeasurements:
    """Voxelwise model-fit measurements summarized inside the fitted mask."""

    label: str
    retained_frames: int
    r_squared: Quartiles
    residual_standard_deviation: Quartiles
    residual_autocorrelation_lag_1: Quartiles


def summarize_model_fit(
    *,
    run_labels: Sequence[str],
    residual_paths: Sequence[Path],
    predicted_paths: Sequence[Path],
    mask_path: Path,
) -> Tuple[RunModelFitMeasurements, ...]:
    """Measure each run in the recorded unwhitened model-response space."""
    import nibabel as nib

    labels = tuple(str(label) for label in run_labels)
    residuals = tuple(Path(path) for path in residual_paths)
    predictions = tuple(Path(path) for path in predicted_paths)
    if not labels:
        raise ValueError("Model-fit measurements require at least one run.")
    if len(residuals) != len(labels) or len(predictions) != len(labels):
        raise ValueError(
            "Model-fit run labels, residuals, and predictions must have equal lengths."
        )

    mask_image = nib.load(str(mask_path))
    if len(mask_image.shape) != 3:
        raise ValueError(f"The fitted analysis mask must be 3D, got {mask_image.shape}.")
    mask = np.asanyarray(mask_image.dataobj).astype(bool)
    if not mask.any():
        raise ValueError("The fitted analysis mask contains no voxels.")

    measurements = []
    for label, residual_path, predicted_path in zip(labels, residuals, predictions):
        residual_image = nib.load(str(residual_path))
        predicted_image = nib.load(str(predicted_path))
        _validate_run_geometry(
            label=label,
            residual_image=residual_image,
            predicted_image=predicted_image,
            mask_image=mask_image,
        )
        residual = np.asarray(residual_image.dataobj, dtype=np.float32)[mask]
        predicted = np.asarray(predicted_image.dataobj, dtype=np.float32)[mask]
        measurements.append(_measure_run(label=label, residual=residual, predicted=predicted))
    return tuple(measurements)


def _validate_run_geometry(
    *,
    label: str,
    residual_image: object,
    predicted_image: object,
    mask_image: object,
) -> None:
    """Require one residual-prediction pair to share the fitted mask grid."""
    residual_shape = tuple(residual_image.shape)
    predicted_shape = tuple(predicted_image.shape)
    if len(residual_shape) != 4 or residual_shape != predicted_shape:
        raise ValueError(f"{label} residual and predicted series require matching 4D shapes.")
    if residual_shape[:3] != tuple(mask_image.shape):
        raise ValueError(f"{label} model-fit series do not match the fitted analysis mask.")
    if not np.allclose(residual_image.affine, predicted_image.affine) or not np.allclose(
        residual_image.affine, mask_image.affine
    ):
        raise ValueError(f"{label} model-fit series do not share the fitted mask affine.")


def _measure_run(
    *,
    label: str,
    residual: np.ndarray,
    predicted: np.ndarray,
) -> RunModelFitMeasurements:
    """Compute direct voxelwise measurements for one retained-frame series."""
    if residual.shape != predicted.shape or residual.ndim != 2:
        raise ValueError(f"{label} masked residual and predicted arrays must match.")
    if residual.shape[1] < 2:
        raise ValueError(f"{label} requires at least two retained frames.")
    if not np.isfinite(residual).all() or not np.isfinite(predicted).all():
        raise ValueError(f"{label} model-fit series contain non-finite values.")

    observed = predicted + residual
    observed_centered = observed - observed.mean(axis=1, keepdims=True, dtype=np.float64)
    total_sum_of_squares = np.sum(
        observed_centered * observed_centered,
        axis=1,
        dtype=np.float64,
    )
    if np.any(total_sum_of_squares <= 0):
        raise ValueError(f"{label} contains voxels with undefined voxelwise R².")
    residual_sum_of_squares = np.sum(
        residual * residual,
        axis=1,
        dtype=np.float64,
    )
    r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares

    residual_standard_deviation = residual.std(axis=1, dtype=np.float64)
    residual_centered = residual - residual.mean(
        axis=1,
        keepdims=True,
        dtype=np.float64,
    )
    autocorrelation_denominator = np.sum(
        residual_centered * residual_centered,
        axis=1,
        dtype=np.float64,
    )
    if np.any(autocorrelation_denominator <= 0):
        raise ValueError(f"{label} contains voxels with undefined residual autocorrelation.")
    autocorrelation_lag_1 = (
        np.sum(
            residual_centered[:, :-1] * residual_centered[:, 1:],
            axis=1,
            dtype=np.float64,
        )
        / autocorrelation_denominator
    )

    return RunModelFitMeasurements(
        label=label,
        retained_frames=int(residual.shape[1]),
        r_squared=_quartiles(r_squared),
        residual_standard_deviation=_quartiles(residual_standard_deviation),
        residual_autocorrelation_lag_1=_quartiles(autocorrelation_lag_1),
    )


def _quartiles(values: np.ndarray) -> Quartiles:
    """Return the 25th percentile, median, and 75th percentile."""
    quartiles = np.percentile(np.asarray(values, dtype=float), [25.0, 50.0, 75.0])
    if not np.isfinite(quartiles).all():
        raise ValueError("Model-fit measurement quartiles must be finite.")
    return tuple(float(value) for value in quartiles)


def model_fit_table(
    measurements: Sequence[RunModelFitMeasurements],
    *,
    response_units: str,
) -> Tuple[str, list[str]]:
    """Return the run measurements as an HTML table and identical TSV rows."""
    if not measurements:
        raise ValueError("The model-fit table requires at least one run.")
    headers = [
        "Run",
        "Retained frames",
        "Median R²",
        "R² IQR",
        f"Residual SD ({response_units})",
        f"Residual SD IQR ({response_units})",
        "Median residual ACF(1)",
        "Residual ACF(1) IQR",
    ]
    rows = []
    for run in measurements:
        r_squared_q1, r_squared_median, r_squared_q3 = run.r_squared
        residual_sd_q1, residual_sd_median, residual_sd_q3 = run.residual_standard_deviation
        residual_acf_q1, residual_acf_median, residual_acf_q3 = run.residual_autocorrelation_lag_1
        rows.append(
            [
                run.label,
                f"{run.retained_frames:,}",
                _number(r_squared_median),
                _interval(r_squared_q1, r_squared_q3),
                _number(residual_sd_median),
                _interval(residual_sd_q1, residual_sd_q3),
                _number(residual_acf_median),
                _interval(residual_acf_q1, residual_acf_q3),
            ]
        )

    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    table_html = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    tsv_rows = ["\t".join(headers), *("\t".join(row) for row in rows)]
    return table_html, tsv_rows


def _number(value: float) -> str:
    """Format one dimensionless or model-response measurement for exact readout."""
    return f"{float(value):.6f}"


def _interval(lower: float, upper: float) -> str:
    """Format a 25th-to-75th percentile interval."""
    return f"{float(lower):.6f}–{float(upper):.6f}"


def write_model_fit_tsv(
    measurements: Sequence[RunModelFitMeasurements],
    *,
    path: Path,
    response_units: str,
) -> Path:
    """Write the model-fit measurement table beside the report."""
    _html, rows = model_fit_table(measurements, response_units=response_units)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "RSquaredMap",
    "ResidualStandardDeviationMap",
    "ResidualCarpet",
    "RunModelFitMeasurements",
    "collect_residual_carpet",
    "model_fit_table",
    "pooled_residual_standard_deviation",
    "pooled_r_squared",
    "summarize_model_fit",
    "write_model_fit_tsv",
]
