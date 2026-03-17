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


def _parse_optional_positive_float_attr(cfg: Any, attr_name: str, default: Any = None) -> Optional[float]:
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
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    return _coerce_float(meta.get("RepetitionTime"))


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
    preproc_by_run = {
        int(run_num): discover_preproc_bold(int(run_num))
        for run_num in run_numbers
    }
    found_runs = [run_num for run_num, path in preproc_by_run.items() if path is not None]
    missing_runs = [run_num for run_num, path in preproc_by_run.items() if path is None]

    if not missing_runs:
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
            return repetition_time

    import nibabel as nib  # type: ignore

    img = nib.load(str(bold_path))
    zooms = img.header.get_zooms()
    if len(zooms) >= 4:
        return float(zooms[3])
    raise ValueError(f"Could not determine TR for {bold_path}")


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
    if mask_img is not None and "mask_img" in sig.parameters:
        kwargs["mask_img"] = mask_img

    return FirstLevelModel(**kwargs)


def validate_design_matrices(
    flm: Any,
    *,
    context: str,
    min_residual_dof: int = 1,
) -> None:
    """Fail fast when nilearn produced a rank-deficient or overfit design."""
    design_mats = getattr(flm, "design_matrices_", None)
    if not isinstance(design_mats, list) or not design_mats:
        raise ValueError(f"{context}: nilearn did not expose any fitted design matrices.")

    min_residual_dof = max(int(min_residual_dof), 0)

    for run_idx, design_matrix in enumerate(design_mats, start=1):
        values = np.asarray(design_matrix, dtype=float)
        if values.ndim != 2 or values.size == 0:
            raise ValueError(f"{context}: run {run_idx} design matrix is empty.")
        if not np.isfinite(values).all():
            raise ValueError(f"{context}: run {run_idx} design matrix contains non-finite values.")

        n_frames, n_regressors = values.shape
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


def select_confound_columns(
    confounds_df: pd.DataFrame,
    strategy: str,
) -> Optional[pd.DataFrame]:
    """Select and sanitize confound columns for nilearn GLM fitting."""
    cols = select_fmriprep_confounds_columns(
        list(confounds_df.columns),
        strategy=str(strategy or "auto"),
    )
    if not cols:
        return None
    return confounds_df[cols].copy().fillna(0)


def select_confounds(
    confounds_path: Optional[Path],
    strategy: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Read/select confounds from TSV path."""
    if confounds_path is None or not confounds_path.exists():
        return None, []

    confounds_df = pd.read_csv(confounds_path, sep="\t")
    selected = select_confound_columns(confounds_df, strategy)
    if selected is None:
        return None, []
    return selected, list(selected.columns)
