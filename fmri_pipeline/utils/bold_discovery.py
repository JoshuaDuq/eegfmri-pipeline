"""Shared BOLD discovery and GLM helper utilities."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

from fmri_pipeline.analysis.confounds_selection import select_fmriprep_confounds_columns


_LOGGER = logging.getLogger(__name__)


def _subject_label(subject: str) -> str:
    return subject if str(subject).startswith("sub-") else f"sub-{subject}"


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
        try:
            meta = json.loads(sidecar.read_text())
            if "RepetitionTime" in meta:
                return float(meta["RepetitionTime"])
        except Exception:
            pass

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

    low_pass = None
    try:
        low_pass_hz = getattr(cfg, "low_pass_hz", None)
        if low_pass_hz is not None:
            lp = float(low_pass_hz)
            if lp > 0:
                low_pass = lp
    except Exception:
        low_pass = None

    high_pass = None
    try:
        high_pass_hz = float(getattr(cfg, "high_pass_hz", 0.0))
        if high_pass_hz > 0:
            high_pass = high_pass_hz
    except Exception:
        high_pass = None

    kwargs: dict[str, Any] = dict(
        t_r=float(tr),
        hrf_model=getattr(cfg, "hrf_model", "spm"),
        drift_model=getattr(cfg, "drift_model", None),
        high_pass=high_pass,
        noise_model="ar1",
        standardize=True,
        signal_scaling=0,
        minimize_memory=False,
    )

    try:
        from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm

        smoothing_fwhm = normalize_smoothing_fwhm(getattr(cfg, "smoothing_fwhm", None))
        if smoothing_fwhm is not None:
            kwargs["smoothing_fwhm"] = smoothing_fwhm
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to normalize smoothing_fwhm; continuing without smoothing: %s", exc)

    sig = inspect.signature(FirstLevelModel)
    if "low_pass" in sig.parameters:
        kwargs["low_pass"] = low_pass
    if mask_img is not None and "mask_img" in sig.parameters:
        kwargs["mask_img"] = mask_img

    return FirstLevelModel(**kwargs)


def coerce_condition_value(value: Any, series: Any) -> Any:
    """Coerce a condition value to match a pandas Series dtype (best effort)."""
    try:
        if pd.api.types.is_integer_dtype(series):
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if pd.api.types.is_float_dtype(series):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if pd.api.types.is_bool_dtype(series):
            return str(value).strip().lower() in ("true", "1", "yes")
    except Exception:
        pass
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

    use_logger = logger or _LOGGER
    try:
        confounds_df = pd.read_csv(confounds_path, sep="\t")
        selected = select_confound_columns(confounds_df, strategy)
        if selected is None:
            return None, []
        return selected, list(selected.columns)
    except Exception as exc:
        use_logger.warning("Failed to read/select confounds from %s (%s)", confounds_path, exc)
        return None, []
