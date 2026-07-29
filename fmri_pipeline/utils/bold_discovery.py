"""Shared BOLD discovery and GLM helper utilities."""

from __future__ import annotations

import inspect
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fmri_pipeline.analysis.confounds_selection import select_fmriprep_confounds_columns


def _subject_label(subject: str) -> str:
    return subject if str(subject).startswith("sub-") else f"sub-{subject}"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_positive_float_attr(
    cfg: Any, attr_name: str, default: Any = None
) -> Optional[float]:
    raw_value = getattr(cfg, attr_name, default)
    if raw_value is None:
        return None
    coerced = _coerce_float(raw_value)
    if coerced is None:
        raise ValueError(f"{attr_name} must be numeric or null, got {raw_value!r}.")
    if not math.isfinite(coerced):
        raise ValueError(f"{attr_name} must be finite when provided, got {raw_value!r}.")
    if coerced <= 0:
        return None
    return coerced


def _read_repetition_time(sidecar: Path) -> Optional[float]:
    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BOLD sidecar JSON at {sidecar}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read BOLD sidecar JSON at {sidecar}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"BOLD sidecar {sidecar} must contain a JSON object.")
    return _coerce_float(meta.get("RepetitionTime"))


def _read_header_tr(bold_path: Path) -> Optional[float]:
    import nibabel as nib  # type: ignore

    img = nib.load(str(bold_path))
    zooms = img.header.get_zooms()
    if len(zooms) >= 4:
        return float(zooms[3])
    return None


def discover_fmriprep_preproc_bold(
    bids_derivatives: Path,
    subject: str,
    task: str,
    run_num: int,
    *,
    space: Optional[str] = "T1w",
) -> Optional[Path]:
    """Discover an fMRIPrep preprocessed BOLD file for one run."""
    sub_label = _subject_label(subject)
    search_dirs = [
        bids_derivatives / "preprocessed" / "fmri" / sub_label / "func",
        bids_derivatives / "preprocessed" / "fmri" / "fmriprep" / sub_label / "func",
        bids_derivatives / "fmriprep" / sub_label / "func",
    ]

    run_tokens = [f"run-{run_num:02d}", f"run-{run_num}"]
    patterns: List[str] = []
    for run_tok in run_tokens:
        if space:
            patterns.append(
                f"{sub_label}_task-{task}_{run_tok}_space-{space}_desc-preproc_bold.nii.gz"
            )
        patterns.append(f"{sub_label}_task-{task}_{run_tok}_desc-preproc_bold.nii.gz")

    for func_dir in search_dirs:
        if not func_dir.exists():
            continue
        for pattern in patterns:
            candidate = func_dir / pattern
            if candidate.exists():
                return candidate
    return None


def discover_runless_fmriprep_preproc_bold(
    bids_derivatives: Path,
    subject: str,
    task: str,
    *,
    space: Optional[str] = "T1w",
) -> Optional[Path]:
    """Discover an explicitly runless fMRIPrep preprocessed BOLD file."""
    sub_label = _subject_label(subject)
    search_dirs = [
        bids_derivatives / "preprocessed" / "fmri" / sub_label / "func",
        bids_derivatives / "preprocessed" / "fmri" / "fmriprep" / sub_label / "func",
        bids_derivatives / "fmriprep" / sub_label / "func",
    ]

    patterns: List[str] = []
    if space:
        patterns.append(f"{sub_label}_task-{task}_space-{space}_desc-preproc_bold.nii.gz")
    patterns.append(f"{sub_label}_task-{task}_desc-preproc_bold.nii.gz")

    for func_dir in search_dirs:
        if not func_dir.exists():
            continue
        for pattern in patterns:
            candidate = func_dir / pattern
            if candidate.exists():
                return candidate
    return None


def discover_single_runless_bids_pair(
    *,
    func_dir: Path,
    sub_label: str,
    task: str,
) -> Optional[Tuple[Path, Path]]:
    """Return the only runless BIDS events/BOLD pair, or fail if ambiguous."""
    events_candidates = [
        path
        for path in sorted(func_dir.glob(f"{sub_label}_task-{task}*_events.tsv"))
        if "_run-" not in path.name and not path.name.endswith("_bold_events.tsv")
    ]
    if len(events_candidates) > 1:
        raise FileNotFoundError(
            "Multiple BOLD/events files were found without explicit run entities. "
            f"Add BIDS run labels or request specific runs explicitly: {[path.name for path in events_candidates]}."
        )
    if not events_candidates:
        return None

    events_path = events_candidates[0]
    bold_name = events_path.name.replace("_events.tsv", "_bold.nii.gz")
    bold_path = events_path.with_name(bold_name)
    if not bold_path.exists():
        return None
    return events_path, bold_path


def select_consistent_run_source(
    *,
    run_numbers: Sequence[int],
    discover_preproc_bold: Callable[[int], Optional[Path]],
    require_fmriprep: bool,
) -> Tuple[str, Dict[int, Optional[Path]]]:
    """
    Resolve one BOLD source for all requested runs.

    Returns ``("fmriprep", paths_by_run)`` when every run has a preprocessed file.
    Returns ``("bids_raw", paths_by_run)`` only when no run has a preprocessed file and
    raw BIDS is therefore the sole consistent source.
    Raises when preprocessed availability is mixed across runs.
    """
    preproc_by_run = {int(run_num): discover_preproc_bold(int(run_num)) for run_num in run_numbers}
    found_runs = [run_num for run_num, path in preproc_by_run.items() if path is not None]
    missing_runs = [run_num for run_num, path in preproc_by_run.items() if path is None]

    if not missing_runs:
        _validate_unique_preprocessed_paths(preproc_by_run)
        return "fmriprep", preproc_by_run

    if found_runs:
        raise FileNotFoundError(
            "fMRIPrep availability is inconsistent across runs. "
            f"Preprocessed BOLD exists for runs {sorted(found_runs)} but is missing for runs {sorted(missing_runs)}. "
            "Use one consistent source for all runs."
        )

    if require_fmriprep:
        raise FileNotFoundError(
            "Requested fMRIPrep input, but no preprocessed BOLD files were found for "
            f"runs {sorted(int(run_num) for run_num in run_numbers)}."
        )

    return "bids_raw", preproc_by_run


def _validate_unique_preprocessed_paths(
    preproc_by_run: Dict[int, Optional[Path]],
) -> None:
    resolved_by_run = {
        run_num: path.resolve() for run_num, path in preproc_by_run.items() if path is not None
    }
    unique_paths = set(resolved_by_run.values())
    if len(unique_paths) == len(resolved_by_run):
        return

    path_to_runs: Dict[Path, List[int]] = {}
    for run_num, path in resolved_by_run.items():
        path_to_runs.setdefault(path, []).append(run_num)
    duplicates = {str(path): sorted(runs) for path, runs in path_to_runs.items() if len(runs) > 1}
    raise FileNotFoundError(
        "The same fMRIPrep preprocessed BOLD file resolved for multiple runs: " f"{duplicates}."
    )


def discover_brain_mask_for_bold(bold_path: Path) -> Optional[Path]:
    """Discover a matching fMRIPrep brain mask for a BOLD image."""
    name = bold_path.name
    suffix = "_desc-preproc_bold.nii.gz"
    if name.endswith(suffix):
        candidate = bold_path.with_name(name.replace(suffix, "_desc-brain_mask.nii.gz"))
        if candidate.exists():
            return candidate
    return None


def get_tr_from_bold(bold_path: Path) -> float:
    """Extract TR from BOLD sidecar JSON or NIfTI header."""
    sidecar = bold_path.with_suffix("").with_suffix(".json")
    if sidecar.exists():
        repetition_time = _read_repetition_time(sidecar)
        if repetition_time is not None:
            try:
                header_tr = _read_header_tr(bold_path)
            except Exception as exc:
                raise ValueError(
                    f"Could not validate TR from NIfTI header for {bold_path}: {exc}"
                ) from exc
            if (
                header_tr is not None
                and math.isfinite(header_tr)
                and not math.isclose(repetition_time, header_tr, rel_tol=0.0, abs_tol=1e-6)
            ):
                raise ValueError(
                    f"TR mismatch for {bold_path}: sidecar RepetitionTime={repetition_time}, "
                    f"NIfTI header zooms[3]={header_tr}."
                )
            return repetition_time
        raise ValueError(f"BOLD sidecar {sidecar} is missing a valid RepetitionTime.")

    header_tr = _read_header_tr(bold_path)
    if header_tr is not None:
        return header_tr
    raise ValueError(f"Could not determine TR for {bold_path}")


#: What nilearn's ``signal_scaling`` axis codes mean, named.
#:
#: Written into the manifest and read back by the report to put units on an effect
#: map. A bare ``0`` there would be a number no reader of the JSON could interpret,
#: and the three modes do not produce the same quantity.
_SIGNAL_SCALING_MODES = {
    0: "voxel-mean",
    1: "timepoint-mean",
    (0, 1): "grand-mean",
}


def fitted_signal_scaling_mode(model: Any) -> Optional[str]:
    """How a fitted model scaled its signal, or None if it scaled none.

    Read off the model rather than off the config, because the config has no such
    setting: :func:`build_first_level_model` passes ``signal_scaling=0``
    unconditionally. A manifest that asked the config recorded "no signal scaling" for
    every contrast this pipeline has ever produced, and the report then labelled effect
    maps that are in percent signal change "arbitrary BOLD units" -- refusing, on the
    strength of a value nobody had set, to put units on the one map that has them.

    ``voxel-mean`` is the mode that yields percent signal change: each voxel is divided
    by its own temporal mean, so an effect is a percentage of that voxel's baseline.
    The other two are percentages of something else and are named separately rather
    than collapsed onto the same label.

    Returns ``None`` for a mode nilearn accepts but this vocabulary does not name, so an
    unrecognised setting costs the units line rather than mislabelling it.
    """
    scaling = getattr(model, "signal_scaling", False)
    if scaling is False or scaling is None:
        return None
    key = tuple(scaling) if isinstance(scaling, (tuple, list)) else scaling
    try:
        return _SIGNAL_SCALING_MODES.get(key)
    except TypeError:  # pragma: no cover - an unhashable setting nilearn cannot take
        return None


def build_first_level_model(
    *,
    tr: float,
    cfg: Any,
    mask_img: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Create a nilearn FirstLevelModel with compatibility guards."""
    from nilearn.glm.first_level import FirstLevelModel  # type: ignore

    low_pass = _parse_optional_positive_float_attr(cfg, "low_pass_hz")
    high_pass = _parse_optional_positive_float_attr(cfg, "high_pass_hz", 0.0)

    kwargs: dict[str, Any] = dict(
        t_r=float(tr),
        hrf_model=getattr(cfg, "hrf_model", "spm"),
        drift_model=getattr(cfg, "drift_model", None),
        high_pass=high_pass,
        noise_model="ar1",
        standardize=False,
        signal_scaling=0,
        minimize_memory=False,
    )

    from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm

    smoothing_fwhm = normalize_smoothing_fwhm(getattr(cfg, "smoothing_fwhm", None))
    if smoothing_fwhm is not None:
        kwargs["smoothing_fwhm"] = smoothing_fwhm

    sig = inspect.signature(FirstLevelModel)
    if "low_pass" in sig.parameters:
        kwargs["low_pass"] = low_pass
    elif low_pass is not None:
        raise ValueError(
            "fmri_contrast.low_pass_hz is set, but the installed "
            "nilearn.glm.first_level.FirstLevelModel does not support low_pass."
        )
    if mask_img is not None and "mask_img" in sig.parameters:
        kwargs["mask_img"] = mask_img

    return FirstLevelModel(**kwargs)


def validate_design_matrices(
    flm: Any,
    *,
    context: str,
    min_residual_dof: int = 1,
    max_condition_number: Optional[float] = None,
    target_columns: Sequence[str] = (),
    min_target_efficiency: Optional[float] = None,
) -> None:
    """Fail fast when nilearn produced a rank-deficient or overfit design."""
    design_mats = getattr(flm, "design_matrices_", None)
    if not isinstance(design_mats, list) or not design_mats:
        raise ValueError(f"{context}: nilearn did not expose any fitted design matrices.")

    min_residual_dof = max(int(min_residual_dof), 0)
    if max_condition_number is not None:
        max_condition_number = float(max_condition_number)
        if not math.isfinite(max_condition_number) or max_condition_number <= 0:
            raise ValueError("max_condition_number must be finite and positive when provided.")
    if min_target_efficiency is not None:
        min_target_efficiency = float(min_target_efficiency)
        if not math.isfinite(min_target_efficiency) or min_target_efficiency <= 0:
            raise ValueError("min_target_efficiency must be finite and positive when provided.")
        if not target_columns:
            raise ValueError("target_columns must be provided when min_target_efficiency is set.")

    for run_idx, design_matrix in enumerate(design_mats, start=1):
        values = np.asarray(design_matrix, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError(f"{context}: run {run_idx} design matrix is empty.")
        if not np.isfinite(values).all():
            raise ValueError(f"{context}: run {run_idx} design matrix contains non-finite values.")

        n_frames, n_regressors = values.shape
        columns = [str(column) for column in getattr(design_matrix, "columns", [])]
        if columns:
            duplicates = sorted({column for column in columns if columns.count(column) > 1})
            if duplicates:
                raise ValueError(
                    f"{context}: run {run_idx} design matrix contains duplicate columns: "
                    f"{duplicates}."
                )
        rank = int(np.linalg.matrix_rank(values))
        residual_dof = int(n_frames - rank)

        if rank < n_regressors:
            raise ValueError(
                f"{context}: run {run_idx} design matrix is rank-deficient "
                f"(rank={rank}, regressors={n_regressors}, frames={n_frames}). "
                "Reduce overlapping regressors, trial count, or nuisance regressors."
            )
        if residual_dof < min_residual_dof:
            raise ValueError(
                f"{context}: run {run_idx} has insufficient residual degrees of freedom "
                f"(frames={n_frames}, rank={rank}, residual_dof={residual_dof}). "
                "Reduce nuisance regressors or modeled events."
            )
        if max_condition_number is not None:
            condition_number = float(np.linalg.cond(values))
            if not math.isfinite(condition_number) or condition_number > max_condition_number:
                raise ValueError(
                    f"{context}: run {run_idx} design matrix condition number "
                    f"{condition_number:.6g} exceeds {max_condition_number:.6g}."
                )
        if min_target_efficiency is not None:
            if not columns:
                raise ValueError(
                    f"{context}: run {run_idx} design matrix columns are required "
                    "for target design-efficiency validation."
                )
            index_by_column = {column: idx for idx, column in enumerate(columns)}
            missing_targets = [
                str(column) for column in target_columns if str(column) not in index_by_column
            ]
            if missing_targets:
                raise ValueError(
                    f"{context}: run {run_idx} missing target design column(s): "
                    f"{missing_targets}."
                )
            xtx_inv = np.linalg.pinv(values.T @ values)
            for target_column in target_columns:
                contrast = np.zeros(n_regressors, dtype=float)
                contrast[int(index_by_column[str(target_column)])] = 1.0
                variance_factor = float(contrast @ xtx_inv @ contrast)
                efficiency = math.inf if variance_factor <= 0 else 1.0 / variance_factor
                if not math.isfinite(efficiency) or efficiency < min_target_efficiency:
                    raise ValueError(
                        f"{context}: run {run_idx} target '{target_column}' design efficiency "
                        f"{efficiency:.6g} is below {min_target_efficiency:.6g}."
                    )


def coerce_condition_value(value: Any, series: Any) -> Any:
    """Coerce a condition value to match a pandas Series dtype (best effort)."""
    is_integer = pd.api.types.is_integer_dtype(series)
    is_float = pd.api.types.is_float_dtype(series)
    is_bool = pd.api.types.is_bool_dtype(series)

    if is_integer:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if is_float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if is_bool:
        return str(value).strip().lower() in ("true", "1", "yes")
    return value


def _is_censor_column(column: str) -> bool:
    return (
        column.startswith("motion_outlier")
        or column.startswith("non_steady_state_outlier")
        or column.startswith("outlier")
    )


def _finite_numeric_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    nonfinite_columns = [
        column
        for column in numeric.columns
        if not np.isfinite(numeric[column].to_numpy(dtype=float)).all()
    ]
    if nonfinite_columns:
        raise ValueError(f"{context} must be finite numeric values in columns {nonfinite_columns}.")
    return numeric


def _sample_mask_from_censor_columns(censor_frame: pd.DataFrame) -> Optional[np.ndarray]:
    if censor_frame.empty:
        return None

    numeric = _finite_numeric_frame(censor_frame, context="Censor confound columns")
    values = numeric.to_numpy(dtype=float)
    censor_rows = np.any(values > 0.0, axis=1)
    sample_mask = np.flatnonzero(~censor_rows).astype(int)
    if sample_mask.size == values.shape[0]:
        return None
    if sample_mask.size == 0:
        raise ValueError("Censor confound columns remove every BOLD volume.")
    return sample_mask


def _exclude_initial_nonfinite_volume(
    *,
    sample_mask: Optional[np.ndarray],
    nonfinite_values: np.ndarray,
) -> Optional[np.ndarray]:
    if nonfinite_values.shape[0] == 0 or not np.any(nonfinite_values[0, :]):
        return sample_mask

    retained = np.ones(nonfinite_values.shape[0], dtype=bool)
    if sample_mask is not None:
        retained[:] = False
        retained[np.asarray(sample_mask, dtype=int)] = True
    retained[0] = False
    updated = np.flatnonzero(retained).astype(int)
    if updated.size == 0:
        raise ValueError("Initial-volume confound censoring removes every BOLD volume.")
    return updated


def select_confounds_for_glm(
    confounds_df: pd.DataFrame,
    strategy: str,
    *,
    auto_compcor_n: int = 5,
) -> Tuple[Optional[pd.DataFrame], List[str], Optional[np.ndarray]]:
    """Select confounds and explicit sample mask for nilearn first-level GLMs."""
    cols = select_fmriprep_confounds_columns(
        list(confounds_df.columns),
        strategy=str(strategy or "auto"),
        auto_compcor_n=int(auto_compcor_n),
    )
    if not cols:
        return None, [], None

    censor_cols = [column for column in cols if _is_censor_column(column)]
    nuisance_cols = [column for column in cols if column not in censor_cols]
    if not nuisance_cols:
        return None, [], _sample_mask_from_censor_columns(confounds_df[censor_cols].copy())

    nuisance = confounds_df[nuisance_cols].copy()
    numeric = nuisance.apply(pd.to_numeric, errors="coerce")
    nonfinite_values = ~np.isfinite(numeric.to_numpy(dtype=float))
    if not nonfinite_values.any():
        sample_mask = _sample_mask_from_censor_columns(confounds_df[censor_cols].copy())
        return numeric, list(numeric.columns), sample_mask

    sample_mask = _sample_mask_from_censor_columns(confounds_df[censor_cols].copy())
    sample_mask = _exclude_initial_nonfinite_volume(
        sample_mask=sample_mask,
        nonfinite_values=nonfinite_values,
    )
    retained = np.zeros(len(numeric), dtype=bool)
    if sample_mask is None:
        retained[:] = True
    else:
        retained[sample_mask] = True
    bad_rows = np.flatnonzero(np.any(nonfinite_values, axis=1) & retained)
    if bad_rows.size:
        bad_columns = [
            str(numeric.columns[column_idx])
            for column_idx in np.flatnonzero(nonfinite_values[bad_rows].any(axis=0))
        ]
        raise ValueError(
            "Selected fMRIPrep confounds contain non-finite values in rows "
            f"{bad_rows.tolist()} that are not marked by censor columns. "
            f"Columns: {bad_columns}."
        )

    return numeric, list(numeric.columns), sample_mask


def select_confounds_for_glm_from_path(
    confounds_path: Optional[Path],
    strategy: str,
    *,
    auto_compcor_n: int = 5,
) -> Tuple[Optional[pd.DataFrame], List[str], Optional[np.ndarray]]:
    """Read fMRIPrep confounds and select GLM regressors plus censor mask."""
    if confounds_path is None or not confounds_path.exists():
        return None, [], None

    confounds_df = pd.read_csv(confounds_path, sep="\t")
    return select_confounds_for_glm(
        confounds_df,
        strategy,
        auto_compcor_n=auto_compcor_n,
    )


def prepare_confounds_for_first_level_model(
    confounds: Optional[pd.DataFrame],
    sample_mask: Optional[np.ndarray],
) -> Optional[pd.DataFrame]:
    """Standardize confounds over retained volumes for a nilearn first-level GLM.

    Non-finite entries are permitted only in volumes that ``sample_mask`` censors;
    nilearn requires a finite array, so those censored entries are filled with the
    retained-volume mean of their own column. That fill never reaches the fit,
    because nilearn drops the censored rows via ``sample_masks``.
    """
    if confounds is None:
        return None

    values = confounds.to_numpy(dtype=float)
    nonfinite = ~np.isfinite(values)
    if not nonfinite.any():
        return confounds

    retained = np.zeros(values.shape[0], dtype=bool)
    if sample_mask is None:
        retained[:] = True
    else:
        retained[np.asarray(sample_mask, dtype=int)] = True
    if np.any(nonfinite & retained[:, None]):
        raise ValueError("First-level confounds contain non-finite values in retained volumes.")

    prepared = confounds.copy()
    for column in prepared.columns:
        column_values = prepared[column].to_numpy(dtype=float).copy()
        missing_rows = ~np.isfinite(column_values)
        if not missing_rows.any():
            continue
        retained_values = column_values[retained]
        finite_retained = retained_values[np.isfinite(retained_values)]
        if finite_retained.size == 0:
            raise ValueError(
                "First-level confounds cannot be prepared because retained volumes contain "
                f"no finite values for {column!r}."
            )
        column_values[missing_rows] = float(finite_retained.mean())
        prepared[column] = column_values

    values = prepared.to_numpy(dtype=float)
    retained_values = values[retained, :]
    means = retained_values.mean(axis=0)
    scales = retained_values.std(axis=0)
    invalid_scales = [
        str(prepared.columns[index])
        for index, scale in enumerate(scales)
        if not math.isfinite(float(scale)) or float(scale) <= 0.0
    ]
    if invalid_scales:
        raise ValueError(
            "First-level confounds must vary across retained volumes before GLM fitting. "
            f"Constant columns: {invalid_scales}."
        )

    scaled = (values - means) / scales
    return type(prepared)(scaled, columns=prepared.columns, index=prepared.index)


def select_confound_columns(
    confounds_df: pd.DataFrame,
    strategy: str,
    *,
    auto_compcor_n: int = 5,
) -> Optional[pd.DataFrame]:
    """Select and sanitize confound columns for nilearn GLM fitting."""
    cols = select_fmriprep_confounds_columns(
        list(confounds_df.columns),
        strategy=str(strategy or "auto"),
        auto_compcor_n=int(auto_compcor_n),
    )
    if not cols:
        return None

    selected = confounds_df[cols].copy()
    missing_cols = [col for col in selected.columns if selected[col].isna().any()]
    if missing_cols:
        raise ValueError(
            "Selected fMRIPrep confounds contain missing values in columns "
            f"{missing_cols}. Do not replace missing nuisance regressors with zero; "
            "fix the confounds file or use an explicit censoring policy."
        )

    numeric = selected.apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = [
        col
        for col in numeric.columns
        if numeric[col].isna().any() or not np.isfinite(numeric[col].to_numpy(dtype=float)).all()
    ]
    if non_numeric_cols:
        raise ValueError(
            "Selected fMRIPrep confounds must be finite numeric values in columns "
            f"{non_numeric_cols}."
        )

    return numeric


def select_confounds(
    confounds_path: Optional[Path],
    strategy: str,
    *,
    auto_compcor_n: int = 5,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Read/select confounds from TSV path."""
    if confounds_path is None or not confounds_path.exists():
        return None, []

    confounds_df = pd.read_csv(confounds_path, sep="\t")
    selected = select_confound_columns(
        confounds_df,
        strategy,
        auto_compcor_n=auto_compcor_n,
    )
    if selected is None:
        return None, []
    return selected, list(selected.columns)
