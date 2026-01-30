"""
fMRI Contrast Builder
=====================

Builds fMRI statistical maps from BOLD data using nilearn's GLM.
Produces z-score/t-stat maps for fMRI-informed EEG source localization.

Requirements:
- BOLD NIfTI files in BIDS format
- Events TSV files with onset, duration, trial_type columns
- FreeSurfer subject space (T1.mgz) for resampling
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fmri_pipeline.analysis.events_selection import normalize_trial_type_list

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContrastBuilderConfig:
    enabled: bool
    input_source: str  # "fmriprep" (preferred) or "bids_raw"
    fmriprep_space: Optional[str]  # e.g., "T1w", "MNI152NLin6Asym"
    require_fmriprep: bool
    contrast_type: str
    condition1: Optional[str]
    condition2: Optional[str]
    condition_a_column: Optional[str]
    condition_a_value: Optional[str]
    condition_b_column: Optional[str]
    condition_b_value: Optional[str]
    formula: Optional[str]
    name: str
    runs: Optional[List[int]]
    hrf_model: str
    drift_model: Optional[str]
    high_pass_hz: float
    low_pass_hz: Optional[float]
    cluster_correction: bool
    cluster_p_threshold: float
    output_type: str
    resample_to_freesurfer: bool
    # Which trial_type rows are eligible for condition remapping.
    # If None, defaults to ["stimulation"] when conditioning on a non-trial_type column.
    condition_scope_trial_types: Optional[List[str]] = None
    # Confounds / QC (optional)
    confounds_strategy: str = "auto"  # none|motion6|motion12|motion24|motion24+wmcsf|motion24+wmcsf+fd|auto
    write_design_matrix: bool = False
    smoothing_fwhm: Optional[float] = None
    # Optional: restrict which trial_type rows are passed to nilearn for GLM.
    # If None, all rows are modeled (default/current behavior).
    # Recommended for multi-phase tasks: ["stimulation", "pain_question", "vas_rating"].
    events_to_model: Optional[List[str]] = None


def _safe_slug(value: str) -> str:
    value = (value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "contrast"


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _get_contrast_hash(contrast_cfg: ContrastBuilderConfig) -> str:
    """Generate hash from contrast config parameters to detect changes."""
    key_parts = [
        contrast_cfg.name,
        contrast_cfg.input_source,
        str(contrast_cfg.fmriprep_space or ""),
        str(bool(contrast_cfg.require_fmriprep)),
        contrast_cfg.contrast_type,
        contrast_cfg.condition_a_column or "",
        contrast_cfg.condition_a_value or "",
        contrast_cfg.condition_b_column or "",
        contrast_cfg.condition_b_value or "",
        contrast_cfg.formula or "",
        str(contrast_cfg.high_pass_hz),
        str(contrast_cfg.low_pass_hz or ""),
        str(contrast_cfg.drift_model or ""),
        str(contrast_cfg.hrf_model),
        str(getattr(contrast_cfg, "confounds_strategy", "auto")),
        str(bool(getattr(contrast_cfg, "write_design_matrix", False))),
        str(getattr(contrast_cfg, "smoothing_fwhm", None) or ""),
        str(getattr(contrast_cfg, "events_to_model", None) or ""),
    ]
    key = "_".join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def load_contrast_config(config: Any) -> ContrastBuilderConfig:
    """Load contrast builder configuration from pipeline config."""
    src_cfg = _cfg_get(config, "feature_engineering.sourcelocalization", {}) or {}
    fmri_cfg = {}
    contrast_cfg = {}

    if isinstance(src_cfg, dict):
        fmri_cfg = src_cfg.get("fmri", {}) or {}
        contrast_cfg = fmri_cfg.get("contrast", {}) or {}
    elif hasattr(config, "get"):
        fmri_cfg = config.get("feature_engineering.sourcelocalization.fmri", {}) or {}
        contrast_cfg = fmri_cfg.get("contrast", {}) or {}

    runs_raw = contrast_cfg.get("runs")
    runs = None
    if runs_raw is not None:
        if isinstance(runs_raw, list):
            runs = [int(r) for r in runs_raw]
        elif isinstance(runs_raw, str):
            runs = [int(r.strip()) for r in runs_raw.split(",") if r.strip()]
        
        # If explicitly empty list/string, treat as None (auto-detect) per comment in yaml
        if runs is not None and len(runs) == 0:
            runs = None

    drift = contrast_cfg.get("drift_model", "cosine")
    if drift == "none":
        drift = None

    input_source = str(contrast_cfg.get("input_source", "fmriprep")).strip().lower()
    if input_source not in {"fmriprep", "bids_raw"}:
        input_source = "fmriprep"

    fmriprep_space = contrast_cfg.get("fmriprep_space", "T1w")
    if fmriprep_space is not None:
        fmriprep_space = str(fmriprep_space).strip()
        if not fmriprep_space:
            fmriprep_space = None

    # Extract condition A/B column + value from nested config
    cond_a_cfg = contrast_cfg.get("condition_a", {}) or {}
    cond_b_cfg = contrast_cfg.get("condition_b", {}) or {}

    scope_raw = contrast_cfg.get("condition_scope_trial_types")
    condition_scope_trial_types = None
    if scope_raw is not None:
        if isinstance(scope_raw, list):
            condition_scope_trial_types = [str(v).strip() for v in scope_raw if str(v).strip()]
        else:
            # Allow comma-separated string
            condition_scope_trial_types = [
                part.strip()
                for part in str(scope_raw).split(",")
                if part.strip()
            ] or None

    # Confounds/QC (optional; safe defaults keep behavior stable)
    confounds_strategy = str(contrast_cfg.get("confounds_strategy", "auto")).strip().lower()
    if confounds_strategy == "":
        confounds_strategy = "auto"
    write_design_matrix = bool(contrast_cfg.get("write_design_matrix", False))
    try:
        from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm

        smoothing_fwhm = normalize_smoothing_fwhm(contrast_cfg.get("smoothing_fwhm"))
    except Exception:
        smoothing_fwhm = None

    events_to_model = normalize_trial_type_list(contrast_cfg.get("events_to_model"))

    return ContrastBuilderConfig(
        enabled=bool(contrast_cfg.get("enabled", False)),
        input_source=input_source,
        fmriprep_space=fmriprep_space,
        require_fmriprep=bool(contrast_cfg.get("require_fmriprep", True)),
        contrast_type=str(contrast_cfg.get("type", "t-test")),
        # Legacy condition1/condition2 for backward compatibility
        condition1=contrast_cfg.get("condition1"),
        condition2=contrast_cfg.get("condition2"),
        # New condition_a/condition_b column + value pairs from TUI
        condition_a_column=cond_a_cfg.get("column"),
        condition_a_value=cond_a_cfg.get("value"),
        condition_b_column=cond_b_cfg.get("column"),
        condition_b_value=cond_b_cfg.get("value"),
        condition_scope_trial_types=condition_scope_trial_types,
        formula=contrast_cfg.get("formula"),
        name=str(contrast_cfg.get("name", "contrast")),
        runs=runs,
        hrf_model=str(contrast_cfg.get("hrf_model", "spm")),
        drift_model=drift,
        high_pass_hz=float(contrast_cfg.get("high_pass_hz", 0.008)),
        low_pass_hz=contrast_cfg.get("low_pass_hz"),
        cluster_correction=bool(contrast_cfg.get("cluster_correction", True)),
        cluster_p_threshold=float(contrast_cfg.get("cluster_p_threshold", 0.001)),
        output_type=str(contrast_cfg.get("output_type", "z_score")),
        resample_to_freesurfer=bool(contrast_cfg.get("resample_to_freesurfer", True)),
        confounds_strategy=confounds_strategy,
        write_design_matrix=write_design_matrix,
        smoothing_fwhm=smoothing_fwhm,
        events_to_model=events_to_model,
    )


###################################################################
# BIDS Discovery
###################################################################


def discover_available_conditions(
    bids_fmri_root: Path,
    subject: str,
    task: str = "pain",
) -> List[str]:
    """
    Discover available trial_type conditions from BIDS events files.

    Scans all events TSV files for the subject/task and returns unique
    trial_type values sorted alphabetically.
    """
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    func_dir = bids_fmri_root / sub_label / "func"

    if not func_dir.exists():
        return []

    all_conditions = set()

    candidates = [
        p
        for p in sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_events.tsv"))
        if not p.name.endswith("_bold_events.tsv")
    ]
    if not candidates:
        # Backward-compat: older outputs used a non-BIDS `_bold_events.tsv` suffix.
        candidates = sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_bold_events.tsv"))

    for events_file in candidates:
        try:
            events_df = pd.read_csv(events_file, sep="\t")
            if "trial_type" in events_df.columns:
                conditions = events_df["trial_type"].dropna().unique().tolist()
                all_conditions.update(str(c) for c in conditions)
        except Exception as e:
            logger.warning("Failed to read events file %s: %s", events_file, e)
            continue

    return sorted(all_conditions)


def discover_bold_runs(
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    subject: str,
    task: str = "pain",
    runs: Optional[List[int]] = None,
    *,
    cfg: Optional[ContrastBuilderConfig] = None,
) -> List[Tuple[Path, Path, int]]:
    """
    Discover BOLD runs and their events files from BIDS fMRI directory.

    Returns list of (bold_path, events_path, run_number) tuples.
    """
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    func_dir = bids_fmri_root / sub_label / "func"

    if not func_dir.exists():
        raise FileNotFoundError(f"fMRI func directory not found: {func_dir}")

    discovered: List[Tuple[Path, Path, int]] = []

    # Discover run numbers from events first (most robust).
    run_nums: List[int] = []
    events_glob = [
        p
        for p in sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_events.tsv"))
        if not p.name.endswith("_bold_events.tsv")
    ]
    if not events_glob:
        events_glob = sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_bold_events.tsv"))

    for events_file in events_glob:
        try:
            run_str = events_file.name.split("_run-")[1].split("_")[0]
            run_nums.append(int(run_str))
        except Exception:
            continue

    if not run_nums:
        # Fall back to discovering from raw BOLD file names.
        for bold_file in sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_bold.nii.gz")):
            try:
                run_str = bold_file.name.split("_run-")[1].split("_")[0]
                run_nums.append(int(run_str))
            except Exception:
                continue

    run_nums = sorted(set(run_nums))
    if runs is not None:
        run_nums = [r for r in run_nums if r in {int(x) for x in runs}]

    if not run_nums:
        raise FileNotFoundError(
            f"No runs found for subject {subject}, task {task} in {func_dir}"
        )

    # Resolve events + BOLD for each run.
    for run_num in run_nums:
        events_patterns = [
            f"{sub_label}_task-{task}_run-{run_num:02d}_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num}_events.tsv",
            f"{sub_label}_task-{task}_run-0{run_num}_events.tsv",
            # Backward-compat: legacy non-BIDS naming
            f"{sub_label}_task-{task}_run-{run_num:02d}_bold_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num}_bold_events.tsv",
            f"{sub_label}_task-{task}_run-0{run_num}_bold_events.tsv",
        ]
        events_file = None
        for pattern in events_patterns:
            candidate = func_dir / pattern
            if candidate.exists():
                events_file = candidate
                break
        if events_file is None:
            logger.warning("No events file found for run %d, skipping", run_num)
            continue

        bold_path = None
        if bids_derivatives is not None and cfg is not None and cfg.input_source == "fmriprep":
            bold_path = _discover_fmriprep_preproc_bold(
                bids_derivatives=bids_derivatives,
                subject=subject,
                task=task,
                run_num=run_num,
                space=cfg.fmriprep_space,
            )

        if bold_path is None:
            raw_patterns = [
                f"{sub_label}_task-{task}_run-{run_num:02d}_bold.nii.gz",
                f"{sub_label}_task-{task}_run-{run_num}_bold.nii.gz",
            ]
            for pattern in raw_patterns:
                candidate = func_dir / pattern
                if candidate.exists():
                    bold_path = candidate
                    break

        if bold_path is None:
            logger.warning("No BOLD file found for run %d, skipping", run_num)
            continue

        discovered.append((bold_path, events_file, int(run_num)))

    if not discovered:
        raise FileNotFoundError(
            f"No BOLD runs found for subject {subject}, task {task} in {func_dir}"
        )

    logger.info("Discovered %d BOLD runs for subject %s", len(discovered), subject)
    return discovered


def discover_confounds(
    bids_derivatives: Path,
    subject: str,
    task: str,
    run_num: int,
) -> Optional[Path]:
    """
    Discover confounds file from fMRIPrep derivatives if available.

    Searches (aligned with _discover_fmriprep_preproc_bold):
      - derivatives/preprocessed/fmri/sub-*/func/
      - derivatives/preprocessed/fmri/fmriprep/sub-*/func/
      - derivatives/fmriprep/sub-*/func/
      - derivatives/sub-*/func/

    Returns path to confounds TSV or None if not found.
    """
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"

    search_dirs = [
        bids_derivatives / "preprocessed" / "fmri" / sub_label / "func",
        bids_derivatives / "preprocessed" / "fmri" / "fmriprep" / sub_label / "func",
        bids_derivatives / "fmriprep" / sub_label / "func",
        bids_derivatives / sub_label / "func",
    ]

    patterns = [
        f"{sub_label}_task-{task}_run-{run_num}_desc-confounds_timeseries.tsv",
        f"{sub_label}_task-{task}_run-{run_num:02d}_desc-confounds_timeseries.tsv",
        f"{sub_label}_task-{task}_run-{run_num}_desc-confounds_regressors.tsv",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            candidate = search_dir / pattern
            if candidate.exists():
                return candidate

    return None


###################################################################
# GLM Fitting
###################################################################

def _discover_fmriprep_preproc_bold(
    bids_derivatives: Path,
    subject: str,
    task: str,
    run_num: int,
    *,
    space: Optional[str] = "T1w",
) -> Optional[Path]:
    """
    Discover fMRIPrep preprocessed BOLD for a given run.

    Searches multiple possible fMRIPrep output locations:
      1. derivatives/fmriprep/sub-*/func/ (standard fMRIPrep layout)
      2. derivatives/preprocessed/fmri/sub-*/func/ (fmri_preprocessing pipeline default)

    We prefer `space-{space}` outputs if provided, but fall back to any
    `desc-preproc_bold` if needed.
    """
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"

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


def _discover_brain_mask_for_bold(bold_path: Path) -> Optional[Path]:
    """
    Discover a matching brain mask for a BOLD image (fMRIPrep convention).

    For fMRIPrep outputs, a `*_desc-brain_mask.nii.gz` typically exists alongside:
      `*_desc-preproc_bold.nii.gz`
    """
    name = bold_path.name
    suffix = "_desc-preproc_bold.nii.gz"
    if name.endswith(suffix):
        candidate = bold_path.with_name(name.replace(suffix, "_desc-brain_mask.nii.gz"))
        if candidate.exists():
            return candidate
    return None


def _get_tr_from_bold(bold_path: Path) -> float:
    """Extract TR from BOLD sidecar JSON or NIfTI header."""
    json_path = bold_path.with_suffix("").with_suffix(".json")
    if json_path.exists():
        import json
        with open(json_path) as f:
            meta = json.load(f)
            if "RepetitionTime" in meta:
                return float(meta["RepetitionTime"])

    import nibabel as nib
    img = nib.load(str(bold_path))
    if len(img.header.get_zooms()) >= 4:
        return float(img.header.get_zooms()[3])

    raise ValueError(f"Could not determine TR for {bold_path}")


def _build_first_level_model(
    *,
    tr: float,
    cfg: "ContrastBuilderConfig",
    mask_img: Optional[Any] = None,
) -> Any:
    """Create a nilearn FirstLevelModel with best-effort compatibility across versions."""
    from nilearn.glm.first_level import FirstLevelModel
    import inspect

    low_pass = None
    try:
        if cfg.low_pass_hz is not None:
            lp = float(cfg.low_pass_hz)
            if lp > 0:
                low_pass = lp
    except Exception:
        low_pass = None

    high_pass = cfg.high_pass_hz if float(cfg.high_pass_hz) > 0 else None

    kwargs: dict[str, Any] = dict(
        t_r=tr,
        hrf_model=cfg.hrf_model,
        drift_model=cfg.drift_model,
        high_pass=high_pass,
        noise_model="ar1",
        standardize=True,
        signal_scaling=0,
        minimize_memory=False,
    )

    # Optional spatial smoothing at first-level (mm FWHM).
    try:
        from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm

        smoothing_fwhm = normalize_smoothing_fwhm(getattr(cfg, "smoothing_fwhm", None))
        if smoothing_fwhm is not None:
            kwargs["smoothing_fwhm"] = smoothing_fwhm
    except Exception:
        pass

    # Some nilearn versions support low_pass; add only if present to avoid TypeError.
    sig = inspect.signature(FirstLevelModel)
    if "low_pass" in sig.parameters:
        kwargs["low_pass"] = low_pass

    if mask_img is not None:
        kwargs["mask_img"] = mask_img

    return FirstLevelModel(**kwargs)


def _select_confound_columns(
    confounds_df: pd.DataFrame,
    cfg: ContrastBuilderConfig,
) -> Optional[pd.DataFrame]:
    """Select confound columns for GLM (fMRIPrep confounds convention).

    Notes
    -----
    - Defaults are designed to be robust across datasets and nilearn versions.
    - This is *first-level* nuisance regression; for group-level inference you
      should also apply appropriate second-level modeling and multiple
      comparisons correction.
    """
    from fmri_pipeline.analysis.confounds_selection import select_fmriprep_confounds_columns

    cols = select_fmriprep_confounds_columns(
        list(confounds_df.columns),
        strategy=str(getattr(cfg, "confounds_strategy", "auto") or "auto"),
    )
    if not cols:
        return None

    selected = confounds_df[cols].copy()
    # fMRIPrep FD has NaN for the first frame; safest is to set to 0.
    selected = selected.fillna(0)
    return selected


def _run_label_from_bold_path(bold_path: Path, fallback_idx: int) -> str:
    """Extract a BIDS-style run label (e.g., 'run-01') from a BOLD filename."""
    for token in bold_path.name.split("_"):
        if token.startswith("run-") and len(token) >= 5:
            return token
    return f"run-{fallback_idx + 1:02d}"


def _write_design_matrices(
    flm: "FirstLevelModel",
    included_bold_paths: List[Path],
    output_dir: Path,
    *,
    prefix: str,
) -> Dict[str, Any]:
    """
    Write per-run design matrices to disk for QC/debugging.

    - Always writes TSVs when possible.
    - Attempts to write PNGs when nilearn plotting is available.
    """
    design_mats = getattr(flm, "design_matrices_", None)
    if not isinstance(design_mats, list) or len(design_mats) == 0:
        return {}

    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    tsv_paths: List[str] = []
    png_paths: List[str] = []

    for idx, dm in enumerate(design_mats):
        run_label = (
            _run_label_from_bold_path(included_bold_paths[idx], idx)
            if idx < len(included_bold_paths)
            else f"run-{idx + 1:02d}"
        )

        tsv_path = qc_dir / f"{prefix}_{run_label}_design_matrix.tsv"
        try:
            dm.to_csv(tsv_path, sep="\t", index=True, index_label="frame")
            tsv_paths.append(str(tsv_path))
        except Exception as exc:
            logger.warning("Failed to write design matrix TSV for %s (%s)", run_label, exc)

        # Optional PNG output (best-effort).
        try:
            from nilearn.plotting import plot_design_matrix

            ax = plot_design_matrix(dm)
            fig = getattr(ax, "figure", None) or getattr(ax, "get_figure", lambda: None)()
            if fig is not None:
                png_path = qc_dir / f"{prefix}_{run_label}_design_matrix.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                png_paths.append(str(png_path))
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:
                    pass
        except Exception:
            pass

    out: Dict[str, Any] = {}
    if tsv_paths:
        out["design_matrix_tsv_paths"] = tsv_paths
    if png_paths:
        out["design_matrix_png_paths"] = png_paths
    return out


@dataclass
class ConditionRemapResult:
    """Result of condition remapping for a single run."""

    events_df: pd.DataFrame
    synthetic_labels: List[str]
    cond_a_found: bool
    cond_b_found: bool
    cond_a_count: int = 0
    cond_b_count: int = 0
    missing_cond_a_msg: str = ""
    missing_cond_b_msg: str = ""


def _remap_events_by_condition_columns(
    events_df: pd.DataFrame,
    cfg: ContrastBuilderConfig,
    *,
    strict: bool = False,
) -> ConditionRemapResult:
    """
    Remap events trial_type based on condition column/value pairs.

    When condition_a_column is specified, creates synthetic trial_type labels
    (e.g., 'cond_a', 'cond_b') by filtering on the specified column values.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with at least 'onset', 'duration', 'trial_type'.
    cfg : ContrastBuilderConfig
        Configuration with condition column/value pairs.
    strict : bool
        If True, raise ValueError when conditions are missing (legacy behavior).
        If False, return result with found=False flags for missing conditions.

    Returns
    -------
    ConditionRemapResult
        Contains modified events_df, synthetic labels, and status flags.
    """
    if not cfg.condition_a_column:
        return ConditionRemapResult(
            events_df=events_df,
            synthetic_labels=[],
            cond_a_found=True,
            cond_b_found=True,
        )

    events_out = events_df.copy()
    synthetic_labels = []

    col_a = cfg.condition_a_column
    val_a = cfg.condition_a_value
    col_b = cfg.condition_b_column
    val_b = cfg.condition_b_value
    scope_trial_types = cfg.condition_scope_trial_types

    if scope_trial_types is None and col_a and col_a != "trial_type" and "trial_type" in events_df.columns:
        # Most BIDS task designs include multiple phases (fixation/question/rating) with the
        # per-trial condition columns repeated across all phases. If we remap all rows, we'd
        # incorrectly label non-stimulation phases as cond_a/cond_b, harming GLM validity.
        trial_types = set(events_df["trial_type"].dropna().astype(str).tolist())
        if "stimulation" in trial_types:
            scope_trial_types = ["stimulation"]

    scope_mask = pd.Series(True, index=events_df.index)
    if scope_trial_types:
        scope_mask = events_df.get("trial_type", "").astype(str).isin(scope_trial_types)

    # Validate column A exists (this is always required)
    if col_a not in events_df.columns:
        raise ValueError(
            f"Condition A column '{col_a}' not found in events. "
            f"Available columns: {list(events_df.columns)}"
        )

    # Check condition A value
    val_a_typed = _coerce_condition_value(val_a, events_df[col_a])
    mask_a = scope_mask & (events_df[col_a] == val_a_typed)
    cond_a_found = mask_a.any()
    cond_a_count = int(mask_a.sum())
    missing_cond_a_msg = ""

    if not cond_a_found:
        missing_cond_a_msg = (
            f"Condition A value '{val_a}' not found in column '{col_a}'. "
            f"Available values: {sorted(events_df[col_a].dropna().unique().tolist())}"
        )
        if strict:
            raise ValueError(missing_cond_a_msg)

    label_a = f"cond_a_{cfg.name}"
    if cond_a_found:
        events_out.loc[mask_a, "trial_type"] = label_a
        synthetic_labels.append(label_a)
        logger.info(
            "Mapped %d events where %s=%s to '%s'",
            cond_a_count, col_a, val_a, label_a
        )

    # Check condition B if specified
    cond_b_found = True
    cond_b_count = 0
    missing_cond_b_msg = ""

    if col_b and val_b is not None and str(val_b).strip() != "":
        if col_b not in events_df.columns:
            raise ValueError(
                f"Condition B column '{col_b}' not found in events. "
                f"Available columns: {list(events_df.columns)}"
            )

        val_b_typed = _coerce_condition_value(val_b, events_df[col_b])
        mask_b = scope_mask & (events_df[col_b] == val_b_typed)
        cond_b_found = mask_b.any()
        cond_b_count = int(mask_b.sum())

        if not cond_b_found:
            missing_cond_b_msg = (
                f"Condition B value '{val_b}' not found in column '{col_b}'. "
                f"Available values: {sorted(events_df[col_b].dropna().unique().tolist())}"
            )
            if strict:
                raise ValueError(missing_cond_b_msg)

        label_b = f"cond_b_{cfg.name}"
        if cond_b_found:
            events_out.loc[mask_b, "trial_type"] = label_b
            synthetic_labels.append(label_b)
            logger.info(
                "Mapped %d events where %s=%s to '%s'",
                cond_b_count, col_b, val_b, label_b
            )

    return ConditionRemapResult(
        events_df=events_out,
        synthetic_labels=synthetic_labels,
        cond_a_found=cond_a_found,
        cond_b_found=cond_b_found,
        cond_a_count=cond_a_count,
        cond_b_count=cond_b_count,
        missing_cond_a_msg=missing_cond_a_msg,
        missing_cond_b_msg=missing_cond_b_msg,
    )


def _coerce_condition_value(value: Any, series: pd.Series) -> Any:
    """
    Coerce string value to match the dtype of the series.

    Handles cases where CLI passes '1' but column contains integers.
    """
    if pd.api.types.is_integer_dtype(series):
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
    if pd.api.types.is_float_dtype(series):
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    if pd.api.types.is_bool_dtype(series):
        return str(value).strip().lower() in ("true", "1", "yes")
    return value


def fit_first_level_glm(
    bold_path: Path,
    events_path: Path,
    confounds_path: Optional[Path],
    cfg: ContrastBuilderConfig,
) -> Tuple["FirstLevelModel", List[str]]:
    """
    Fit a first-level GLM to a single BOLD run.

    Returns (fitted FirstLevelModel, synthetic_condition_labels).
    """
    from nilearn.glm.first_level import FirstLevelModel

    tr = _get_tr_from_bold(bold_path)
    logger.info("Fitting GLM for %s (TR=%.2fs)", bold_path.name, tr)

    events_df = pd.read_csv(events_path, sep="\t")
    required_cols = {"onset", "duration", "trial_type"}
    if not required_cols.issubset(events_df.columns):
        raise ValueError(
            f"Events file missing required columns. "
            f"Found: {list(events_df.columns)}, need: {required_cols}"
        )

    if getattr(cfg, "events_to_model", None):
        allow = set(str(x) for x in (cfg.events_to_model or []) if str(x).strip())
        if allow:
            events_df = events_df[events_df["trial_type"].astype(str).isin(allow)].copy()
            if events_df.empty:
                raise ValueError(
                    f"After applying events_to_model={sorted(allow)}, no events remained in {events_path}."
                )

    # Use strict=True for single-run: cannot skip the only run
    remap_result = _remap_events_by_condition_columns(events_df, cfg, strict=True)
    events_df = remap_result.events_df[["onset", "duration", "trial_type"]].copy()
    synthetic_labels = remap_result.synthetic_labels

    confounds = None
    if confounds_path is not None and confounds_path.exists():
        confounds_df = pd.read_csv(confounds_path, sep="\t")
        confounds = _select_confound_columns(confounds_df, cfg)
        if confounds is not None:
            logger.info("Using %d confound regressors", confounds.shape[1])

    # Use a matching brain mask when available (best practice with fMRIPrep outputs).
    mask_img = None
    try:
        import nibabel as nib

        mask_path = _discover_brain_mask_for_bold(bold_path)
        if mask_path is not None and mask_path.exists():
            mask_img = nib.load(str(mask_path))
    except Exception:
        mask_img = None

    flm = _build_first_level_model(tr=tr, cfg=cfg, mask_img=mask_img)

    flm.fit(bold_path, events=events_df, confounds=confounds)

    return flm, synthetic_labels

@dataclass
class MultiRunGLMResult:
    """Result from multi-run GLM fitting with run inclusion details."""

    flm: Any  # FirstLevelModel
    synthetic_labels: List[str]
    all_conditions: List[str]
    confound_columns: List[str]
    included_bold_paths: List[Path]
    included_events_paths: List[Path]
    included_confounds_paths: List[Optional[Path]]
    skipped_runs: List[Tuple[int, str]]  # (run_idx, reason)
    total_cond_a_events: int
    total_cond_b_events: int


def fit_first_level_glm_multi_run(
    bold_paths: List[Path],
    events_paths: List[Path],
    confounds_paths: List[Optional[Path]],
    cfg: ContrastBuilderConfig,
) -> MultiRunGLMResult:
    """
    Fit a first-level GLM across runs using nilearn's multi-run support.

    This avoids the scientifically invalid approach of averaging per-run z/t maps.

    Runs missing one or both requested conditions are excluded from the GLM
    with a warning. The GLM only fails if NO runs contain both conditions.

    Returns MultiRunGLMResult with fitted model and run inclusion details.
    """
    from nilearn.glm.first_level import FirstLevelModel

    if not bold_paths:
        raise ValueError("No BOLD runs provided.")
    if len(bold_paths) != len(events_paths) or len(bold_paths) != len(confounds_paths):
        raise ValueError("Mismatch between number of BOLD, events, and confounds inputs.")

    tr = _get_tr_from_bold(bold_paths[0])
    logger.info("Fitting multi-run GLM (%d runs, TR=%.2fs)", len(bold_paths), tr)

    required_cols = {"onset", "duration", "trial_type"}

    # Track which runs are valid for inclusion
    valid_bold_paths: List[Path] = []
    valid_events_list: List[pd.DataFrame] = []
    valid_events_paths: List[Path] = []
    valid_confounds_list: List[Optional[pd.DataFrame]] = []
    valid_confounds_paths: List[Optional[Path]] = []
    skipped_runs: List[Tuple[int, str]] = []

    all_conditions: set[str] = set()
    synthetic_labels: List[str] = []
    confound_columns: set[str] = set()

    total_cond_a_events = 0
    total_cond_b_events = 0

    for run_idx, (bold_path, events_path, confounds_path) in enumerate(
        zip(bold_paths, events_paths, confounds_paths), start=1
    ):
        events_df = pd.read_csv(events_path, sep="\t")
        if not required_cols.issubset(events_df.columns):
            raise ValueError(
                f"Events file missing required columns: {events_path}. "
                f"Found: {list(events_df.columns)}, need: {required_cols}"
            )

        if getattr(cfg, "events_to_model", None):
            allow = set(str(x) for x in (cfg.events_to_model or []) if str(x).strip())
            if allow:
                events_df = events_df[events_df["trial_type"].astype(str).isin(allow)].copy()
                if events_df.empty:
                    skipped_runs.append((run_idx, f"events_to_model removed all events for {events_path.name}"))
                    logger.warning(
                        "Run %d excluded from GLM: events_to_model removed all events (%s)",
                        run_idx,
                        events_path.name,
                    )
                    continue

        # Remap conditions, allowing missing values (strict=False)
        remap_result = _remap_events_by_condition_columns(events_df, cfg, strict=False)

        # Check if this run should be skipped due to missing conditions
        needs_both = cfg.condition_b_column and cfg.condition_b_value
        run_valid = True
        skip_reason = ""

        if cfg.condition_a_column:
            if not remap_result.cond_a_found:
                run_valid = False
                skip_reason = remap_result.missing_cond_a_msg
            elif needs_both and not remap_result.cond_b_found:
                run_valid = False
                skip_reason = remap_result.missing_cond_b_msg

        if not run_valid:
            skipped_runs.append((run_idx, skip_reason))
            logger.warning(
                "Run %d excluded from GLM: %s",
                run_idx, skip_reason
            )
            continue

        # Run is valid - include it
        total_cond_a_events += remap_result.cond_a_count
        total_cond_b_events += remap_result.cond_b_count

        if remap_result.synthetic_labels and not synthetic_labels:
            synthetic_labels = remap_result.synthetic_labels

        events_out = remap_result.events_df[["onset", "duration", "trial_type"]].copy()
        valid_bold_paths.append(bold_path)
        valid_events_paths.append(events_path)
        valid_events_list.append(events_out)
        all_conditions.update(str(c) for c in events_out["trial_type"].dropna().unique().tolist())

        # Handle confounds
        confounds = None
        if confounds_path is not None and confounds_path.exists():
            confounds_df = pd.read_csv(confounds_path, sep="\t")
            confounds = _select_confound_columns(confounds_df, cfg)
            if confounds is not None:
                confound_columns.update(list(confounds.columns))
                logger.info("Run %d: using %d confound regressors", run_idx, confounds.shape[1])
        valid_confounds_list.append(confounds)
        valid_confounds_paths.append(confounds_path)

    # Validate we have at least one run with both conditions
    if not valid_bold_paths:
        skip_summary = "; ".join(f"Run {r}: {msg}" for r, msg in skipped_runs)
        raise ValueError(
            f"No runs contain both requested condition values. "
            f"Skipped all {len(bold_paths)} runs. Details: {skip_summary}"
        )

    if skipped_runs:
        logger.warning(
            "Excluded %d/%d runs due to missing conditions. "
            "Proceeding with %d runs (cond_a: %d events, cond_b: %d events)",
            len(skipped_runs), len(bold_paths), len(valid_bold_paths),
            total_cond_a_events, total_cond_b_events
        )

    # If fMRIPrep brain masks exist, use their intersection to stabilize masking.
    mask_img = None
    try:
        import nibabel as nib
        from nilearn.masking import intersect_masks

        mask_paths = [
            p for p in (_discover_brain_mask_for_bold(bp) for bp in valid_bold_paths)
            if p is not None
        ]
        if len(mask_paths) == len(valid_bold_paths) and mask_paths:
            mask_imgs = [nib.load(str(p)) for p in mask_paths]
            mask_img = intersect_masks(mask_imgs, threshold=1.0)
    except Exception:
        mask_img = None

    flm = _build_first_level_model(tr=tr, cfg=cfg, mask_img=mask_img)

    strategy = str(getattr(cfg, "confounds_strategy", "auto") or "auto").strip().lower()
    if strategy in {"", "default"}:
        strategy = "auto"

    all_none = all(c is None for c in valid_confounds_list)
    any_none = any(c is None for c in valid_confounds_list)

    if all_none:
        if strategy in {"none", "no", "off"}:
            confounds_arg: Optional[List[pd.DataFrame]] = None
        else:
            raise ValueError(
                "No confounds files found for any run. Under deriv-root, looked for "
                "*desc-confounds_timeseries.tsv or *desc-confounds_regressors.tsv in: "
                "preprocessed/fmri/sub-<id>/func, preprocessed/fmri/fmriprep/sub-<id>/func, "
                "fmriprep/sub-<id>/func, sub-<id>/func. "
                "Use --confounds-strategy none to run without confounds, or ensure fMRIPrep "
                "confounds exist."
            )
    elif any_none:
        raise ValueError(
            "Confounds file missing for some runs but present for others. "
            "Supply confounds for all runs, or use --confounds-strategy none when no "
            "confounds are available."
        )
    else:
        confounds_arg = valid_confounds_list

    flm.fit(valid_bold_paths, events=valid_events_list, confounds=confounds_arg)

    return MultiRunGLMResult(
        flm=flm,
        synthetic_labels=synthetic_labels,
        all_conditions=sorted(all_conditions),
        confound_columns=sorted(confound_columns),
        included_bold_paths=valid_bold_paths,
        included_events_paths=valid_events_paths,
        included_confounds_paths=valid_confounds_paths,
        skipped_runs=skipped_runs,
        total_cond_a_events=total_cond_a_events,
        total_cond_b_events=total_cond_b_events,
    )


def compute_contrast_map(
    flm: "FirstLevelModel",
    cfg: ContrastBuilderConfig,
    available_conditions: List[str],
    synthetic_labels: Optional[List[str]] = None,
) -> Tuple["Nifti1Image", str, str]:
    """
    Compute contrast from fitted GLM.

    Returns (contrast_map, contrast_def, output_type).
    """
    output_map = {
        "z-score": "z_score",
        "z_score": "z_score",
        "t-stat": "stat",
        "t_stat": "stat",
        "cope": "effect_size",
        "beta": "effect_size",
    }
    output_type = output_map.get(cfg.output_type, "z_score")

    if cfg.contrast_type == "custom" and cfg.formula:
        contrast_def = cfg.formula
    elif synthetic_labels:
        if len(synthetic_labels) == 2:
            contrast_def = f"{synthetic_labels[0]} - {synthetic_labels[1]}"
        elif len(synthetic_labels) == 1:
            contrast_def = synthetic_labels[0]
        else:
            raise ValueError(
                f"Unexpected number of synthetic labels: {synthetic_labels}"
            )
    else:
        cond_a = cfg.condition_a_value or cfg.condition1
        cond_b = cfg.condition_b_value or cfg.condition2

        if cond_a and cond_b:
            if cond_a not in available_conditions:
                raise ValueError(
                    f"Condition A value '{cond_a}' not found in events. "
                    f"Available: {available_conditions}"
                )
            if cond_b not in available_conditions:
                raise ValueError(
                    f"Condition B value '{cond_b}' not found in events. "
                    f"Available: {available_conditions}"
                )
            contrast_def = f"{cond_a} - {cond_b}"
        elif cond_a:
            if cond_a not in available_conditions:
                raise ValueError(
                    f"Condition A value '{cond_a}' not found in events. "
                    f"Available: {available_conditions}"
                )
            contrast_def = cond_a
        else:
            raise ValueError(
                "Must specify condition_a_value (or legacy condition1), "
                "optionally condition_b_value (or legacy condition2), or a custom formula"
            )

    logger.info("Computing contrast: %s (output: %s)", contrast_def, output_type)

    # For multi-run models, nilearn warns if a single contrast definition is passed.
    # Provide one contrast per run to avoid noisy warnings and be explicit.
    contrast_arg: Any = contrast_def
    try:
        n_runs = len(getattr(flm, "design_matrices_", []) or [])
        if n_runs > 1 and not isinstance(contrast_def, (list, tuple, dict)):
            contrast_arg = [contrast_def] * n_runs
    except Exception:
        contrast_arg = contrast_def

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"One contrast given, assuming it for all.*",
            category=UserWarning,
        )
        contrast_map = flm.compute_contrast(contrast_arg, output_type=output_type)

    return contrast_map, str(contrast_def), str(output_type)


###################################################################
# Multi-run Processing
###################################################################


def build_contrast_from_runs(
    bids_fmri_root: Path,
    bids_derivatives: Path,
    subject: str,
    task: str,
    cfg: ContrastBuilderConfig,
    *,
    output_dir: Optional[Path] = None,
) -> Tuple["Nifti1Image", Dict[str, Any]]:
    contrast_map, meta, _glm_result, _contrast_def, _output_type = build_contrast_from_runs_detailed(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        cfg=cfg,
        output_dir=output_dir,
    )
    return contrast_map, meta


def build_contrast_from_runs_detailed(
    bids_fmri_root: Path,
    bids_derivatives: Path,
    subject: str,
    task: str,
    cfg: ContrastBuilderConfig,
    *,
    output_dir: Optional[Path] = None,
) -> Tuple["Nifti1Image", Dict[str, Any], "MultiRunGLMResult", str, str]:
    """
    Build contrast map from multiple runs using a single multi-run GLM.

    This is the statistically valid approach for combining runs at the
    first level (fixed-effects across runs).
    """
    runs_data = discover_bold_runs(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        runs=cfg.runs,
        cfg=cfg,
    )

    if cfg.require_fmriprep and cfg.input_source == "fmriprep":
        sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
        raw_prefix = str((bids_fmri_root / sub_label / "func").resolve())
        raw_runs = [p.name for p, _, _ in runs_data if str(p.resolve()).startswith(raw_prefix)]
        if raw_runs:
            raise FileNotFoundError(
                "Contrast builder requires fMRIPrep preprocessed BOLD, "
                f"but these runs resolved to raw BOLD in {raw_prefix}: {raw_runs}. "
                "Set feature_engineering.sourcelocalization.fmri.contrast.require_fmriprep=false "
                "or ensure fMRIPrep outputs exist under derivatives/fmriprep."
            )

    bold_paths: List[Path] = []
    events_paths: List[Path] = []
    confounds_paths: List[Optional[Path]] = []

    for bold_path, events_path, run_num in runs_data:
        bold_paths.append(bold_path)
        events_paths.append(events_path)
        confounds_paths.append(discover_confounds(bids_derivatives, subject, task, run_num))

    glm_result = fit_first_level_glm_multi_run(
        bold_paths=bold_paths,
        events_paths=events_paths,
        confounds_paths=confounds_paths,
        cfg=cfg,
    )

    qc_meta: Dict[str, Any] = {}
    if bool(getattr(cfg, "write_design_matrix", False)) and output_dir is not None:
        try:
            sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
            prefix = f"{sub_label}_task-{task}_contrast-{_safe_slug(str(getattr(cfg, 'name', 'contrast')))}"
            qc_meta = _write_design_matrices(
                glm_result.flm,
                glm_result.included_bold_paths,
                Path(output_dir).expanduser().resolve(),
                prefix=prefix,
            )
        except Exception as exc:
            logger.warning("Failed to write design-matrix QC outputs (%s); continuing.", exc)

    contrast_map, contrast_def, output_type = compute_contrast_map(
        glm_result.flm,
        cfg,
        glm_result.all_conditions,
        synthetic_labels=glm_result.synthetic_labels or None,
    )

    # Build comprehensive metadata with run provenance
    meta: Dict[str, Any] = {
        # Discovered runs (before filtering)
        "n_runs_discovered": len(bold_paths),
        "discovered_bold_paths": [str(p) for p in bold_paths],
        "discovered_events_paths": [str(p) for p in events_paths],
        # Included runs (after filtering for conditions)
        "n_runs_included": len(glm_result.included_bold_paths),
        "included_bold_paths": [str(p) for p in glm_result.included_bold_paths],
        "included_events_paths": [str(p) for p in glm_result.included_events_paths],
        "included_confounds_paths": [
            str(p) if p is not None else None
            for p in glm_result.included_confounds_paths
        ],
        # Skipped runs with reasons
        "n_runs_skipped": len(glm_result.skipped_runs),
        "skipped_runs": [
            {"run_index": idx, "reason": reason}
            for idx, reason in glm_result.skipped_runs
        ],
        # Event counts
        "total_cond_a_events": glm_result.total_cond_a_events,
        "total_cond_b_events": glm_result.total_cond_b_events,
        # GLM details
        "confound_columns": glm_result.confound_columns,
        "confounds_strategy": str(getattr(cfg, "confounds_strategy", "auto")),
        "contrast_def": contrast_def,
        "output_type": output_type,
    }
    meta.update(qc_meta)
    return contrast_map, meta, glm_result, str(contrast_def), str(output_type)


###################################################################
# Resampling to FreeSurfer Space
###################################################################


def resample_to_freesurfer(
    contrast_map: "Nifti1Image",
    freesurfer_subject_dir: Path,
    interpolation: str = "continuous",
) -> "Nifti1Image":
    """
    Resample contrast map to FreeSurfer subject space.

    The output will be aligned with T1.mgz or orig.mgz.
    """
    from nilearn.image import resample_to_img
    import nibabel as nib

    mri_dir = freesurfer_subject_dir / "mri"

    target_candidates = ["T1.mgz", "orig.mgz", "brain.mgz"]
    target_img = None

    for candidate in target_candidates:
        target_path = mri_dir / candidate
        if target_path.exists():
            target_img = nib.load(str(target_path))
            logger.info("Resampling to FreeSurfer space using %s", candidate)
            break

    if target_img is None:
        raise FileNotFoundError(
            f"No FreeSurfer MRI found in {mri_dir}. "
            f"Looked for: {target_candidates}"
        )

    resampled = resample_to_img(
        contrast_map,
        target_img,
        interpolation=interpolation,
        copy_header=False,
    )

    return resampled


def _cluster_extent_mask_from_z_map(
    z_img: "Nifti1Image",
    *,
    p_threshold: float,
    tail: str,
    min_cluster_voxels: int,
) -> "Nifti1Image":
    """Create a cluster-extent thresholded mask from a z-stat image.

    Notes
    -----
    This performs (1) voxelwise z-thresholding derived from `p_threshold`,
    followed by (2) connected-component filtering by minimum cluster size.

    This is a pragmatic heuristic for generating fMRI constraint masks and
    should not be interpreted as cluster-level FWE correction.
    """
    import nibabel as nib
    from scipy.ndimage import generate_binary_structure, label as cc_label
    from scipy import stats as sps

    tail = str(tail).strip().lower()
    if tail not in {"pos", "abs"}:
        tail = "pos"

    try:
        p_threshold = float(p_threshold)
    except Exception:
        p_threshold = 0.001
    if not np.isfinite(p_threshold) or p_threshold <= 0 or p_threshold >= 1:
        p_threshold = 0.001

    p_use = p_threshold / 2.0 if tail == "abs" else p_threshold
    z_thr = float(sps.norm.isf(p_use))

    img = nib.load(str(z_img)) if isinstance(z_img, (str, Path)) else z_img
    data = np.asarray(img.get_fdata(dtype=np.float32))
    finite = np.isfinite(data)
    if tail == "abs":
        seed = finite & (np.abs(data) >= z_thr)
    else:
        seed = finite & (data >= z_thr)

    if not np.any(seed):
        mask = np.zeros_like(seed, dtype=np.uint8)
        return nib.Nifti1Image(mask, img.affine, img.header)

    structure = generate_binary_structure(3, 1)
    labeled, n_clusters = cc_label(seed.astype(np.uint8), structure=structure)

    keep = np.zeros_like(seed, dtype=bool)
    for cluster_id in range(1, int(n_clusters) + 1):
        vox = (labeled == cluster_id)
        if int(np.sum(vox)) >= int(min_cluster_voxels):
            keep |= vox

    mask_u8 = keep.astype(np.uint8)
    return nib.Nifti1Image(mask_u8, img.affine, img.header)


###################################################################
# Main Entry Point
###################################################################


def build_fmri_contrast(
    bids_fmri_root: Path,
    bids_derivatives: Path,
    freesurfer_subjects_dir: Path,
    subject: str,
    config: Any,
    task: str = "pain",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Build fMRI contrast map from BOLD data.

    Main entry point for the contrast builder. Orchestrates:
    1. BIDS discovery
    2. GLM fitting per run
    3. Contrast computation
    4. Resampling to FreeSurfer space
    5. Saving output

    Returns path to saved contrast map, or None if contrast building is disabled.
    """
    import nibabel as nib

    cfg = load_contrast_config(config)

    if not cfg.enabled:
        logger.debug("Contrast builder disabled in config")
        return None

    logger.info(
        "Building fMRI contrast for subject %s: %s (%s)",
        subject,
        cfg.name,
        cfg.contrast_type,
    )

    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    if output_dir is None:
        output_dir = bids_derivatives / sub_label / "fmri_contrasts"

    contrast_map, run_meta = build_contrast_from_runs(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        cfg=cfg,
        output_dir=output_dir,
    )

    fs_subject_dir = freesurfer_subjects_dir / sub_label

    if cfg.resample_to_freesurfer and fs_subject_dir.exists():
        contrast_map = resample_to_freesurfer(contrast_map, fs_subject_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    contrast_hash = _get_contrast_hash(cfg)
    output_name = f"{sub_label}_{cfg.name}_{cfg.output_type}_{contrast_hash}.nii.gz"
    output_path = output_dir / output_name

    nib.save(contrast_map, str(output_path))
    logger.info("Saved contrast map to %s", output_path)

    # Optional: create a cluster-extent thresholded mask for downstream constraints.
    if cfg.cluster_correction and str(run_meta.get("output_type")) == "z_score":
        try:
            src_cfg = _cfg_get(config, "feature_engineering.sourcelocalization", {}) or {}
            fmri_cfg = src_cfg.get("fmri", {}) if isinstance(src_cfg, dict) else {}
            tail = str(fmri_cfg.get("tail", "pos")).strip().lower()
            min_cluster_voxels = int(fmri_cfg.get("cluster_min_voxels", 50))
        except Exception:
            tail = "pos"
            min_cluster_voxels = 50

        try:
            mask_img = _cluster_extent_mask_from_z_map(
                contrast_map,
                p_threshold=cfg.cluster_p_threshold,
                tail=tail,
                min_cluster_voxels=min_cluster_voxels,
            )
            mask_name = f"{sub_label}_{cfg.name}_mask_cluster_{contrast_hash}.nii.gz"
            mask_path = output_dir / mask_name
            nib.save(mask_img, str(mask_path))
            run_meta["cluster_mask_path"] = str(mask_path)
        except Exception as exc:
            logger.warning("Failed to create cluster mask (%s); continuing without it.", exc)

    # Write a lightweight sidecar for reproducibility/debugging.
    try:
        import json

        sidecar_path = output_path.with_suffix("").with_suffix(".json")
        payload = {
            "subject": sub_label,
            "task": task,
            "contrast_name": cfg.name,
            "contrast_type": cfg.contrast_type,
            "input_source": cfg.input_source,
            "fmriprep_space": cfg.fmriprep_space,
            "require_fmriprep": bool(cfg.require_fmriprep),
            "hrf_model": cfg.hrf_model,
            "drift_model": cfg.drift_model,
            "high_pass_hz": cfg.high_pass_hz,
            "output_type_requested": cfg.output_type,
            "output_type_actual": run_meta.get("output_type"),
            "contrast_def": run_meta.get("contrast_def"),
            "runs": cfg.runs,
            "run_inputs": run_meta,
        }
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        logger.debug("Failed to write contrast sidecar JSON (%s)", exc)

    return output_path


def ensure_fmri_stats_map(
    config: Any,
    bids_fmri_root: Optional[Path],
    bids_derivatives: Path,
    freesurfer_subjects_dir: Path,
    subject: str,
    task: str = "pain",
) -> Optional[Path]:
    """
    Ensure fMRI stats map exists, building it if necessary.

    This is the main integration point with source_localization.py.
    Returns path to stats map (either pre-existing or newly built).

    Caches the contrast map to avoid rebuilding across multiple time windows.
    """
    src_cfg = _cfg_get(config, "feature_engineering.sourcelocalization", {}) or {}
    fmri_cfg = {}
    if isinstance(src_cfg, dict):
        fmri_cfg = src_cfg.get("fmri", {}) or {}

    # Check for explicitly configured stats map path
    existing_path = fmri_cfg.get("stats_map_path") or fmri_cfg.get("stats_map")
    if existing_path:
        existing_path = Path(str(existing_path)).expanduser()
        if existing_path.exists():
            logger.info("Using existing fMRI stats map: %s", existing_path)
            return existing_path

    contrast_cfg = load_contrast_config(config)
    if not contrast_cfg.enabled:
        return None

    if bids_fmri_root is None:
        logger.warning(
            "Contrast builder enabled but no BIDS fMRI root provided. "
            "Set paths.bids_fmri_root in config."
        )
        return None

    # Check if the contrast map already exists at the expected output path
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    output_dir = bids_derivatives / sub_label / "fmri_contrasts"
    contrast_hash = _get_contrast_hash(contrast_cfg)
    output_name = f"{sub_label}_{contrast_cfg.name}_{contrast_cfg.output_type}_{contrast_hash}.nii.gz"
    cached_path = output_dir / output_name

    if cached_path.exists():
        logger.info("Using cached fMRI contrast map: %s", cached_path)
        return cached_path

    return build_fmri_contrast(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        freesurfer_subjects_dir=freesurfer_subjects_dir,
        subject=subject,
        config=config,
        task=task,
        output_dir=output_dir,
    )
