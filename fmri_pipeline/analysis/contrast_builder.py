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

import copy
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fmri_pipeline.analysis.events_selection import normalize_trial_type_list
from fmri_pipeline.analysis.constraint_masking import build_thresholded_constraint_mask
from fmri_pipeline.utils.bold_discovery import (
    build_first_level_model as _build_first_level_model,
    coerce_condition_value as _coerce_condition_value,
    discover_brain_mask_for_bold as _discover_brain_mask_for_bold,
    discover_fmriprep_preproc_bold as _discover_fmriprep_preproc_bold,
    get_tr_from_bold as _get_tr_from_bold,
    select_consistent_run_source,
    select_confound_columns as _select_confound_columns,
    validate_design_matrices as _validate_design_matrices,
)
from fmri_pipeline.utils.text import safe_slug as _safe_slug
from eeg_pipeline.utils.config.loader import get_config_value

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image
    from nilearn.glm.first_level import FirstLevelModel


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
    output_type: str
    resample_to_freesurfer: bool
    # Which events rows are eligible for condition remapping.
    # Values are matched against `condition_scope_column`.
    # If None, no condition-scope filtering is applied.
    condition_scope_trial_types: Optional[List[str]] = None
    # Events column used for `condition_scope_trial_types`.
    condition_scope_column: str = ""
    # Confounds / QC (optional)
    confounds_strategy: str = "auto"  # none|motion6|motion12|motion24|motion24+wmcsf|motion24+wmcsf+fd|auto
    write_design_matrix: bool = False
    smoothing_fwhm: Optional[float] = None
    # Optional: restrict which trial_type rows are passed to nilearn for GLM.
    # If None, all rows are modeled (default/current behavior).
    # Useful for multi-phase tasks to limit modeled rows.
    events_to_model: Optional[List[str]] = None
    # Events column used for `events_to_model`.
    events_to_model_column: str = ""
    # Optional: restrict which stimulation sub-phases are modeled when events.tsv includes 'stim_phase'.
    # If None, no stim_phase scoping is applied. Use ["all"] to disable phase scoping.
    stim_phases_to_model: Optional[List[str]] = None
    # Events column that stores phase labels used by `stim_phases_to_model`.
    phase_column: str = ""
    # Events column used to scope phase filtering to specific rows.
    phase_scope_column: str = ""
    # Optional value in `phase_scope_column` to restrict where phase filtering is applied.
    # If None, phase filtering applies to all rows.
    phase_scope_value: Optional[str] = None


SUPPORTED_CONTRAST_TYPES = frozenset({"custom", "t-test"})
SUPPORTED_INPUT_SOURCES = frozenset({"bids_raw", "fmriprep"})
OUTPUT_TYPE_MAP = {
    "z-score": "z_score",
    "z_score": "z_score",
    "t-stat": "stat",
    "t_stat": "stat",
    "cope": "effect_size",
    "beta": "effect_size",
}
LEGACY_CONDITION_KEY_MAP = {
    "cond_a": "condition_a",
    "cond_b": "condition_b",
}
REMOVED_CONTRAST_KEYS = frozenset({"cluster_correction", "cluster_p_threshold"})


def _value_is_specified(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _normalize_contrast_type(raw_value: Any) -> str:
    contrast_type = str(raw_value or "t-test").strip().lower()
    if contrast_type not in SUPPORTED_CONTRAST_TYPES:
        supported = ", ".join(sorted(SUPPORTED_CONTRAST_TYPES))
        raise ValueError(
            f"Unsupported fmri_contrast.type {raw_value!r}. "
            f"Supported values: {supported}."
        )
    return contrast_type


def _normalize_input_source(raw_value: Any) -> str:
    input_source = str(raw_value or "fmriprep").strip().lower()
    if input_source not in SUPPORTED_INPUT_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SOURCES))
        raise ValueError(
            f"Unsupported fmri input_source {raw_value!r}. "
            f"Supported values: {supported}."
        )
    return input_source


def _normalize_requested_output_type(raw_value: Any) -> str:
    output_type = str(raw_value or "z-score").strip().lower()
    if output_type not in OUTPUT_TYPE_MAP:
        supported = ", ".join(sorted(OUTPUT_TYPE_MAP))
        raise ValueError(
            f"Unsupported fmri output_type {raw_value!r}. "
            f"Supported values: {supported}."
        )
    return output_type


def _assert_constraint_mask_requires_z_output(
    *,
    output_type: Any,
    constraint_spec: Optional[Dict[str, Any]],
) -> None:
    if constraint_spec is None:
        return
    normalized_output = OUTPUT_TYPE_MAP[_normalize_requested_output_type(output_type)]
    if normalized_output != "z_score":
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri thresholding requires "
            "fmri_contrast.output_type='z-score' so the constraint mask is derived from z statistics."
        )


def validate_contrast_config_section(contrast_cfg: Dict[str, Any]) -> None:
    """Reject misleading or unsupported contrast-section keys."""
    legacy_keys = [key for key in LEGACY_CONDITION_KEY_MAP if key in contrast_cfg]
    if legacy_keys:
        key_map = ", ".join(f"{key}->{LEGACY_CONDITION_KEY_MAP[key]}" for key in sorted(legacy_keys))
        raise ValueError(
            "Use fmri_contrast.condition_a / fmri_contrast.condition_b keys. "
            f"Legacy keys are unsupported: {key_map}."
        )

    removed_keys = [key for key in sorted(REMOVED_CONTRAST_KEYS) if key in contrast_cfg]
    if removed_keys:
        keys = ", ".join(removed_keys)
        raise ValueError(
            f"Contrast-level thresholding keys are unsupported: {keys}. "
            "Thresholding belongs in plotting settings for visualization or in "
            "feature_engineering.sourcelocalization.fmri for EEG constraint masks."
        )


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
        str(getattr(contrast_cfg, "condition_scope_trial_types", None) or ""),
        str(getattr(contrast_cfg, "condition_scope_column", "") or ""),
        str(getattr(contrast_cfg, "events_to_model", None) or ""),
        str(getattr(contrast_cfg, "stim_phases_to_model", None) or ""),
        str(getattr(contrast_cfg, "phase_column", "") or ""),
        str(getattr(contrast_cfg, "phase_scope_column", "") or ""),
        str(getattr(contrast_cfg, "phase_scope_value", "") or ""),
    ]
    key = "_".join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def load_contrast_config_section(config: Any) -> Dict[str, Any]:
    """Resolve the configured contrast section with nested source-localization precedence."""
    top_level = get_config_value(config, "fmri_contrast", {}) or {}
    src_cfg = get_config_value(config, "feature_engineering.sourcelocalization", {}) or {}

    if hasattr(config, "get"):
        top_level = config.get("fmri_contrast", top_level) or top_level
        src_cfg = config.get("feature_engineering.sourcelocalization", src_cfg) or src_cfg

    top_level = top_level if isinstance(top_level, dict) else {}
    nested_fmri = src_cfg.get("fmri", {}) if isinstance(src_cfg, dict) else {}
    nested = nested_fmri.get("contrast", {}) if isinstance(nested_fmri, dict) else {}
    nested = nested if isinstance(nested, dict) else {}

    return copy.deepcopy(nested or top_level)


def load_contrast_config(config: Any) -> ContrastBuilderConfig:
    """Load contrast builder configuration from pipeline config."""
    contrast_cfg = load_contrast_config_section(config)
    validate_contrast_config_section(contrast_cfg)

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

    input_source = _normalize_input_source(contrast_cfg.get("input_source", "fmriprep"))

    contrast_type = _normalize_contrast_type(contrast_cfg.get("type", "t-test"))

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
    condition_scope_column = str(contrast_cfg.get("condition_scope_column", "") or "").strip()

    # Confounds/QC (optional; safe defaults keep behavior stable)
    confounds_strategy = str(contrast_cfg.get("confounds_strategy", "auto")).strip().lower()
    if confounds_strategy == "":
        confounds_strategy = "auto"
    write_design_matrix = bool(contrast_cfg.get("write_design_matrix", False))
    from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm

    smoothing_fwhm = normalize_smoothing_fwhm(contrast_cfg.get("smoothing_fwhm"))

    events_to_model = normalize_trial_type_list(contrast_cfg.get("events_to_model"))
    events_to_model_column = str(
        contrast_cfg.get("events_to_model_column", "") or ""
    ).strip()
    stim_phases_to_model = normalize_trial_type_list(contrast_cfg.get("stim_phases_to_model"))
    phase_column = str(contrast_cfg.get("phase_column", "") or "").strip()
    phase_scope_column = str(contrast_cfg.get("phase_scope_column", "") or "").strip()
    phase_scope_value = contrast_cfg.get("phase_scope_value")
    if phase_scope_value is not None:
        phase_scope_value = str(phase_scope_value).strip() or None
    if condition_scope_trial_types and not condition_scope_column:
        raise ValueError(
            "fmri_contrast.condition_scope_trial_types requires "
            "fmri_contrast.condition_scope_column."
        )
    if events_to_model and not events_to_model_column:
        raise ValueError(
            "fmri_contrast.events_to_model requires "
            "fmri_contrast.events_to_model_column."
        )
    if stim_phases_to_model and not phase_column:
        raise ValueError(
            "fmri_contrast.stim_phases_to_model requires fmri_contrast.phase_column."
        )
    if phase_scope_value and not phase_scope_column:
        raise ValueError(
            "fmri_contrast.phase_scope_value requires fmri_contrast.phase_scope_column."
        )

    return ContrastBuilderConfig(
        enabled=bool(contrast_cfg.get("enabled", False)),
        input_source=input_source,
        fmriprep_space=fmriprep_space,
        require_fmriprep=bool(contrast_cfg.get("require_fmriprep", True)),
        contrast_type=contrast_type,
        # Legacy condition1/condition2 for backward compatibility
        condition1=contrast_cfg.get("condition1"),
        condition2=contrast_cfg.get("condition2"),
        # New condition_a/condition_b column + value pairs from TUI
        condition_a_column=cond_a_cfg.get("column"),
        condition_a_value=cond_a_cfg.get("value"),
        condition_b_column=cond_b_cfg.get("column"),
        condition_b_value=cond_b_cfg.get("value"),
        condition_scope_trial_types=condition_scope_trial_types,
        condition_scope_column=condition_scope_column,
        formula=contrast_cfg.get("formula"),
        name=str(contrast_cfg.get("name", "contrast")),
        runs=runs,
        hrf_model=str(contrast_cfg.get("hrf_model", "spm")),
        drift_model=drift,
        high_pass_hz=float(contrast_cfg.get("high_pass_hz", 0.008)),
        low_pass_hz=contrast_cfg.get("low_pass_hz"),
        output_type=_normalize_requested_output_type(contrast_cfg.get("output_type", "z-score")),
        resample_to_freesurfer=bool(contrast_cfg.get("resample_to_freesurfer", True)),
        confounds_strategy=confounds_strategy,
        write_design_matrix=write_design_matrix,
        smoothing_fwhm=smoothing_fwhm,
        events_to_model=events_to_model,
        events_to_model_column=events_to_model_column,
        stim_phases_to_model=stim_phases_to_model,
        phase_column=phase_column,
        phase_scope_column=phase_scope_column,
        phase_scope_value=phase_scope_value,
    )


def _apply_trial_phase_scoping(
    events_df: pd.DataFrame,
    *,
    allowed_phases: Optional[List[str]],
    phase_column: str = "",
    phase_scope_column: str = "",
    phase_scope_value: Optional[str] = None,
) -> pd.DataFrame:
    """
    Restrict events to specific phase values, optionally scoped to one row subset.

    Scoping is only applied when ``allowed_phases`` is explicitly provided.
    When ``phase_scope_value`` is set, phase filtering only applies to rows
    where ``phase_scope_column`` matches that value; all other rows are kept as-is.
    When ``phase_scope_value`` is None, phase filtering applies to all rows.
    """
    keep = _build_trial_phase_model_mask(
        events_df,
        allowed_phases=allowed_phases,
        phase_column=phase_column,
        phase_scope_column=phase_scope_column,
        phase_scope_value=phase_scope_value,
    )
    if bool(keep.all()):
        return events_df
    return events_df.loc[keep].copy()


def _build_events_to_model_mask(
    events_df: pd.DataFrame,
    *,
    allowed_values: Optional[List[str]],
    events_column: str = "",
) -> pd.Series:
    if not allowed_values:
        return pd.Series(True, index=events_df.index, dtype=bool)

    column_name = str(events_column or "").strip()
    if not column_name:
        column_name = "trial_type"
    if column_name not in events_df.columns:
        raise ValueError(
            "events_to_model is set but events file has no "
            f"'{column_name}' column."
        )

    allow = {str(value).strip() for value in allowed_values if str(value).strip()}
    if not allow:
        return pd.Series(True, index=events_df.index, dtype=bool)

    return events_df[column_name].astype(str).isin(sorted(allow))


def _build_trial_phase_model_mask(
    events_df: pd.DataFrame,
    *,
    allowed_phases: Optional[List[str]],
    phase_column: str = "",
    phase_scope_column: str = "",
    phase_scope_value: Optional[str] = None,
) -> pd.Series:
    phase_column_name = str(phase_column or "").strip()
    phase_scope_column_name = str(phase_scope_column or "").strip()

    raw = allowed_phases
    if raw is None:
        return pd.Series(True, index=events_df.index, dtype=bool)

    allow_norm = [str(v).strip().lower() for v in raw if str(v).strip()]
    if not allow_norm:
        return pd.Series(True, index=events_df.index, dtype=bool)
    if "all" in set(allow_norm):
        return pd.Series(True, index=events_df.index, dtype=bool)

    if not phase_column_name:
        raise ValueError(
            "stim_phases_to_model is set but no phase_column was configured."
        )
    if phase_column_name not in events_df.columns:
        raise ValueError(
            "stim_phases_to_model is set but events file has no "
            f"'{phase_column_name}' column. Available columns: {list(events_df.columns)}"
        )

    if phase_scope_value is not None and not phase_scope_column_name:
        raise ValueError(
            "phase_scope_value is set but no phase_scope_column was configured."
        )
    if phase_scope_value is not None and phase_scope_column_name not in events_df.columns:
        raise ValueError(
            "phase_scope_value is set but events file has no "
            f"'{phase_scope_column_name}' column. Available columns: {list(events_df.columns)}"
        )

    allow = set(allow_norm)
    phase_norm = events_df[phase_column_name].fillna("").astype(str).str.strip().str.lower()
    if phase_scope_value is not None and phase_scope_column_name in events_df.columns:
        scope_match = events_df[phase_scope_column_name].astype(str).str.strip().str.lower().eq(
            str(phase_scope_value).strip().lower()
        )
        return (~scope_match) | phase_norm.isin(sorted(allow))
    return phase_norm.isin(sorted(allow))


def _build_nuisance_trial_type(
    row: pd.Series,
    *,
    phase_column: str,
) -> str:
    trial_type = str(row.get("trial_type", "event")).strip()
    if trial_type.startswith("nuis_"):
        return trial_type

    tokens = [
        "nuis",
        _safe_slug(trial_type, default="event"),
    ]
    phase_value = row.get(phase_column)
    if phase_value is not None and str(phase_value).strip():
        tokens.append(_safe_slug(str(phase_value), default="phase"))
    return "_".join(tokens)


def _prepare_events_for_glm(
    events_df: pd.DataFrame,
    cfg: ContrastBuilderConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    phase_column_name = str(getattr(cfg, "phase_column", "") or "").strip()
    trial_type_mask = _build_events_to_model_mask(
        events_df,
        allowed_values=getattr(cfg, "events_to_model", None),
        events_column=str(getattr(cfg, "events_to_model_column", "") or "").strip(),
    )
    phase_mask = _build_trial_phase_model_mask(
        events_df,
        allowed_phases=getattr(cfg, "stim_phases_to_model", None),
        phase_column=phase_column_name,
        phase_scope_column=str(getattr(cfg, "phase_scope_column", "") or "").strip(),
        phase_scope_value=getattr(cfg, "phase_scope_value", None),
    )
    eligible_mask = trial_type_mask & phase_mask
    if bool(eligible_mask.all()):
        return events_df, eligible_mask

    events_out = events_df.copy()
    excluded_rows = events_out.loc[~eligible_mask]
    events_out.loc[~eligible_mask, "trial_type"] = [
        _build_nuisance_trial_type(row, phase_column=phase_column_name)
        for _idx, row in excluded_rows.iterrows()
    ]
    return events_out, eligible_mask


def _get_bold_run_duration_seconds(bold_path: Path) -> float:
    import nibabel as nib

    img = nib.load(str(bold_path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D BOLD image for {bold_path}, got shape={img.shape}.")

    n_scans = int(img.shape[3])
    if n_scans <= 0:
        raise ValueError(f"BOLD image has no timepoints: {bold_path}")

    tr = float(_get_tr_from_bold(bold_path))
    return float(n_scans) * tr


def _validate_events_against_bold_run(
    events_df: pd.DataFrame,
    *,
    bold_path: Path,
    context: str,
) -> None:
    onset = pd.to_numeric(events_df["onset"], errors="coerce")
    duration = pd.to_numeric(events_df["duration"], errors="coerce")

    if onset.isna().any():
        bad_rows = onset.index[onset.isna()].tolist()
        raise ValueError(f"{context}: onset contains non-numeric or missing values at rows {bad_rows}.")
    if duration.isna().any():
        bad_rows = duration.index[duration.isna()].tolist()
        raise ValueError(f"{context}: duration contains non-numeric or missing values at rows {bad_rows}.")
    if not np.isfinite(onset.to_numpy(dtype=float)).all():
        raise ValueError(f"{context}: onset contains non-finite values.")
    if not np.isfinite(duration.to_numpy(dtype=float)).all():
        raise ValueError(f"{context}: duration contains non-finite values.")
    if (onset < 0).any():
        bad_rows = onset.index[onset < 0].tolist()
        raise ValueError(f"{context}: onset must be >= 0, found negative values at rows {bad_rows}.")
    if (duration < 0).any():
        bad_rows = duration.index[duration < 0].tolist()
        raise ValueError(f"{context}: duration must be >= 0, found negative values at rows {bad_rows}.")

    run_duration = _get_bold_run_duration_seconds(bold_path)
    tr = float(_get_tr_from_bold(bold_path))
    tolerance = max(tr * 0.5, 1e-6)
    if (onset > (run_duration + tolerance)).any():
        bad_rows = onset.index[onset > (run_duration + tolerance)].tolist()
        raise ValueError(
            f"{context}: onset exceeds run duration ({run_duration:.6f}s) beyond tolerance "
            f"({tolerance:.6f}s) at rows {bad_rows}."
        )

    event_end = onset + duration
    if (event_end > (run_duration + tolerance)).any():
        bad_rows = event_end.index[event_end > (run_duration + tolerance)].tolist()
        raise ValueError(
            f"{context}: onset + duration exceeds run duration ({run_duration:.6f}s) beyond tolerance "
            f"({tolerance:.6f}s) at rows {bad_rows}."
        )


def _load_matching_brain_mask_for_bold(bold_path: Path) -> Optional[Any]:
    import nibabel as nib

    mask_path = _discover_brain_mask_for_bold(bold_path)
    if mask_path is None:
        return None
    return nib.load(str(mask_path))


def _build_intersection_brain_mask(bold_paths: Sequence[Path]) -> Optional[Any]:
    import nibabel as nib
    from nilearn.masking import intersect_masks

    mask_paths: list[Path] = []
    for bold_path in bold_paths:
        mask_path = _discover_brain_mask_for_bold(bold_path)
        if mask_path is None:
            return None
        mask_paths.append(mask_path)

    if not mask_paths:
        return None

    mask_imgs = [nib.load(str(path)) for path in mask_paths]
    return intersect_masks(mask_imgs, threshold=1.0)


###################################################################
# BIDS Discovery
###################################################################


def discover_available_conditions(
    bids_fmri_root: Path,
    subject: str,
    task: str,
    *,
    condition_column: str,
) -> List[str]:
    """
    Discover available condition values from a configured BIDS events column.

    Scans all events TSV files for the subject/task and returns unique
    values from ``condition_column`` sorted alphabetically.
    """
    selected_column = str(condition_column or "").strip()
    if not selected_column:
        raise ValueError("condition_column is required to discover available fMRI conditions.")

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
            if selected_column in events_df.columns:
                conditions = events_df[selected_column].dropna().unique().tolist()
                all_conditions.update(str(c) for c in conditions)
        except Exception as e:
            logger.warning("Failed to read events file %s: %s", events_file, e)
            continue

    return sorted(all_conditions)


def discover_bold_runs(
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    subject: str,
    task: str,
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
        if runs is not None:
            requested = sorted({int(x) for x in runs})
            raise FileNotFoundError(
                f"None of the requested runs were found for subject {subject}, task {task}: {requested}"
            )
        raise FileNotFoundError(
            f"No runs found for subject {subject}, task {task} in {func_dir}"
        )

    selected_input_source = _normalize_input_source(getattr(cfg, "input_source", "bids_raw"))
    preproc_by_run: Dict[int, Optional[Path]] = {}
    if bids_derivatives is not None and cfg is not None and selected_input_source == "fmriprep":
        selected_input_source, preproc_by_run = select_consistent_run_source(
            run_numbers=run_nums,
            discover_preproc_bold=lambda run_num: _discover_fmriprep_preproc_bold(
                bids_derivatives=bids_derivatives,
                subject=subject,
                task=task,
                run_num=run_num,
                space=cfg.fmriprep_space,
            ),
            require_fmriprep=bool(getattr(cfg, "require_fmriprep", False)),
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
        if selected_input_source == "fmriprep":
            bold_path = preproc_by_run.get(int(run_num))
        else:
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

    if runs is not None:
        requested_runs = {int(run_num) for run_num in runs}
        discovered_runs = {int(run_num) for _bold_path, _events_path, run_num in discovered}
        missing_runs = sorted(requested_runs - discovered_runs)
        if missing_runs:
            raise FileNotFoundError(
                "Some requested runs could not be resolved to matching BOLD + events inputs: "
                f"{missing_runs}."
            )

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
            dm.to_csv(tsv_path, sep="\t", index=True, index_label="frame", encoding="utf-8")
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
                except Exception as exc:
                    logger.debug("Failed to close design matrix figure for %s: %s", run_label, exc)
        except Exception as exc:
            logger.warning("Failed to render design matrix PNG for %s: %s", run_label, exc)

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
    eligible_mask: Optional[pd.Series] = None,
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
    scope_column = str(getattr(cfg, "condition_scope_column", "") or "").strip()

    scope_mask = pd.Series(True, index=events_df.index, dtype=bool)
    if eligible_mask is not None:
        if len(eligible_mask) != len(events_df):
            raise ValueError("eligible_mask must match events_df length.")
        scope_mask &= eligible_mask.astype(bool)
    if scope_trial_types:
        if not scope_column:
            raise ValueError(
                "condition_scope_trial_types is set but no condition_scope_column was configured."
            )
        normalized_scope = [str(v).strip() for v in scope_trial_types if str(v).strip()]
        if normalized_scope and not any(v.lower() in {"all", "*", "@all"} for v in normalized_scope):
            if scope_column not in events_df.columns:
                raise ValueError(
                    f"Condition scope column '{scope_column}' not found in events. "
                    f"Available columns: {list(events_df.columns)}"
                )
            scope_mask &= events_df[scope_column].astype(str).isin(normalized_scope)

    excluded_mask = ~scope_mask
    if bool(excluded_mask.any()):
        phase_column_name = str(getattr(cfg, "phase_column", "") or "").strip()
        events_out.loc[excluded_mask, "trial_type"] = [
            _build_nuisance_trial_type(row, phase_column=phase_column_name)
            for _idx, row in events_out.loc[excluded_mask].iterrows()
        ]

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
        overlap_mask = mask_a & mask_b
        overlap_count = int(overlap_mask.sum())
        if overlap_count > 0:
            raise ValueError(
                f"Condition definitions overlap for {overlap_count} event(s): "
                f"{col_a}={val_a!r} and {col_b}={val_b!r} select the same rows. "
                "Condition A and B must be mutually exclusive."
            )
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

    tr = _get_tr_from_bold(bold_path)
    logger.info("Fitting GLM for %s (TR=%.2fs)", bold_path.name, tr)

    events_df = pd.read_csv(events_path, sep="\t")
    required_cols = {"onset", "duration", "trial_type"}
    if not required_cols.issubset(events_df.columns):
        raise ValueError(
            f"Events file missing required columns. "
            f"Found: {list(events_df.columns)}, need: {required_cols}"
        )
    _validate_events_against_bold_run(
        events_df,
        bold_path=bold_path,
        context=f"Events validation ({events_path.name})",
    )

    events_df, eligible_mask = _prepare_events_for_glm(events_df, cfg)

    # Use strict=True for single-run: cannot skip the only run
    remap_result = _remap_events_by_condition_columns(
        events_df,
        cfg,
        strict=True,
        eligible_mask=eligible_mask,
    )
    events_df = remap_result.events_df[["onset", "duration", "trial_type"]].copy()
    synthetic_labels = remap_result.synthetic_labels

    confounds = None
    confounds_strategy = str(getattr(cfg, "confounds_strategy", "auto") or "auto").strip().lower()
    if confounds_strategy in {"", "default"}:
        confounds_strategy = "auto"
    if confounds_strategy not in {"none", "no", "off"}:
        if confounds_path is None or not confounds_path.exists():
            raise ValueError(
                "Single-run first-level GLMs require confounds unless confounds_strategy is 'none'. "
                f"Missing confounds for {bold_path.name}."
            )
        confounds_df = pd.read_csv(confounds_path, sep="\t")
        confounds = _select_confound_columns(confounds_df, confounds_strategy)
        if confounds is None:
            raise ValueError(
                "No confound regressors were selected for the single-run first-level GLM. "
                f"Strategy={confounds_strategy!r}, file={confounds_path}."
            )
        logger.info("Using %d confound regressors", confounds.shape[1])

    # Use a matching brain mask when available (best practice with fMRIPrep outputs).
    mask_img = _load_matching_brain_mask_for_bold(bold_path)

    flm = _build_first_level_model(tr=tr, cfg=cfg, mask_img=mask_img)

    flm.fit(bold_path, events=events_df, confounds=confounds)
    _validate_design_matrices(
        flm,
        context=f"First-level GLM ({bold_path.name})",
        min_residual_dof=1,
    )

    return flm, synthetic_labels

@dataclass
class MultiRunGLMResult:
    """Result from multi-run GLM fitting with run inclusion details."""

    flm: Any  # FirstLevelModel
    mask_img: Optional[Any]
    synthetic_labels: List[str]
    all_conditions: List[str]
    confound_columns: List[str]
    included_bold_paths: List[Path]
    included_events_paths: List[Path]
    included_confounds_paths: List[Optional[Path]]
    skipped_runs: List[Tuple[int, str]]  # (run_idx, reason)
    total_cond_a_events: int
    total_cond_b_events: int


def _validate_consistent_trs(bold_paths: List[Path]) -> float:
    if not bold_paths:
        raise ValueError("Cannot validate TR consistency without any BOLD runs.")

    trs = [_get_tr_from_bold(path) for path in bold_paths]
    reference_tr = float(trs[0])
    mismatches = [
        f"{path.name}={tr:.6f}s"
        for path, tr in zip(bold_paths, trs)
        if not np.isclose(float(tr), reference_tr, rtol=0.0, atol=1e-6)
    ]
    if mismatches:
        formatted = ", ".join(mismatches)
        raise ValueError(
            "All runs included in one multi-run first-level GLM must share the same TR. "
            f"Reference TR={reference_tr:.6f}s; mismatched runs: {formatted}."
        )
    return reference_tr


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

    if not bold_paths:
        raise ValueError("No BOLD runs provided.")
    if len(bold_paths) != len(events_paths) or len(bold_paths) != len(confounds_paths):
        raise ValueError("Mismatch between number of BOLD, events, and confounds inputs.")

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
        _validate_events_against_bold_run(
            events_df,
            bold_path=bold_path,
            context=f"Events validation ({events_path.name})",
        )

        events_df, eligible_mask = _prepare_events_for_glm(events_df, cfg)

        # Remap conditions, allowing missing values (strict=False)
        remap_result = _remap_events_by_condition_columns(
            events_df,
            cfg,
            strict=False,
            eligible_mask=eligible_mask,
        )

        # Check if this run should be skipped due to missing conditions
        needs_both = bool(cfg.condition_b_column) and _value_is_specified(cfg.condition_b_value)
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
            confounds = _select_confound_columns(confounds_df, getattr(cfg, "confounds_strategy", "auto"))
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

    tr = _validate_consistent_trs(valid_bold_paths)
    logger.info("Fitting multi-run GLM (%d runs, TR=%.2fs)", len(valid_bold_paths), tr)

    # If fMRIPrep brain masks exist, use their intersection to stabilize masking.
    mask_img = _build_intersection_brain_mask(valid_bold_paths)

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
    _validate_design_matrices(
        flm,
        context="Multi-run first-level GLM",
        min_residual_dof=1,
    )

    return MultiRunGLMResult(
        flm=flm,
        mask_img=mask_img,
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
    requested_output_type = _normalize_requested_output_type(cfg.output_type)
    output_type = OUTPUT_TYPE_MAP[requested_output_type]

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
        cond_a = cfg.condition_a_value if _value_is_specified(cfg.condition_a_value) else cfg.condition1
        cond_b = cfg.condition_b_value if _value_is_specified(cfg.condition_b_value) else cfg.condition2

        if _value_is_specified(cond_a) and _value_is_specified(cond_b):
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
        elif _value_is_specified(cond_a):
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
        sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
        prefix = (
            f"{sub_label}_task-{task}_contrast-"
            f"{_safe_slug(str(getattr(cfg, 'name', 'contrast')), default='contrast')}"
        )
        qc_meta = _write_design_matrices(
            glm_result.flm,
            glm_result.included_bold_paths,
            Path(output_dir).expanduser().resolve(),
            prefix=prefix,
        )

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
        force_resample=True,
        copy_header=True,
    )

    return resampled


def _contrast_sidecar_path(contrast_path: Path) -> Path:
    return contrast_path.with_suffix("").with_suffix(".json")


def _load_constraint_mask_spec(config: Any) -> Optional[Dict[str, Any]]:
    src_cfg = get_config_value(config, "feature_engineering.sourcelocalization", {}) or {}
    if not isinstance(src_cfg, dict):
        return None

    fmri_cfg = src_cfg.get("fmri", {}) or {}
    if not isinstance(fmri_cfg, dict) or not bool(fmri_cfg.get("enabled", False)):
        return None

    threshold = float(fmri_cfg.get("threshold", 3.1))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.threshold must be a finite float > 0."
        )

    tail = str(fmri_cfg.get("tail", "pos")).strip().lower()
    if tail not in {"pos", "abs"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.tail must be one of {'pos','abs'}."
        )

    thresholding_cfg = fmri_cfg.get("thresholding", {}) or {}
    threshold_mode = str(
        thresholding_cfg.get("mode", fmri_cfg.get("threshold_mode", "z"))
    ).strip().lower()
    if threshold_mode not in {"z", "fdr"}:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.thresholding.mode must be one of {'z','fdr'}."
        )

    fdr_q = float(thresholding_cfg.get("fdr_q", 0.05))
    if not np.isfinite(fdr_q) or not (0 < fdr_q < 1):
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.thresholding.fdr_q must be in (0, 1)."
        )

    cluster_min_voxels = int(fmri_cfg.get("cluster_min_voxels", 50))
    if cluster_min_voxels < 1:
        raise ValueError(
            "feature_engineering.sourcelocalization.fmri.cluster_min_voxels must be >= 1."
        )

    cluster_min_volume_mm3 = fmri_cfg.get("cluster_min_volume_mm3")
    if cluster_min_volume_mm3 is not None:
        cluster_min_volume_mm3 = float(cluster_min_volume_mm3)
        if not np.isfinite(cluster_min_volume_mm3) or cluster_min_volume_mm3 <= 0:
            raise ValueError(
                "feature_engineering.sourcelocalization.fmri.cluster_min_volume_mm3 must be > 0 when provided."
            )

    return {
        "threshold": threshold,
        "tail": tail,
        "threshold_mode": threshold_mode,
        "fdr_q": fdr_q,
        "cluster_min_voxels": cluster_min_voxels,
        "cluster_min_volume_mm3": cluster_min_volume_mm3,
    }


def _get_constraint_mask_hash(spec: Dict[str, Any], *, resample_to_freesurfer: bool) -> str:
    cluster_volume = spec["cluster_min_volume_mm3"]
    cluster_volume_token = "" if cluster_volume is None else f"{float(cluster_volume):.8g}"
    key = (
        f"{spec['threshold_mode']}|{spec['threshold']:.8g}|{spec['fdr_q']:.8g}|"
        f"{spec['tail']}|{int(spec['cluster_min_voxels'])}|"
        f"{cluster_volume_token}|"
        f"{int(bool(resample_to_freesurfer))}"
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def _constraint_mask_output_path(
    *,
    output_dir: Path,
    sub_label: str,
    contrast_name: str,
    contrast_hash: str,
    constraint_hash: str,
) -> Path:
    return output_dir / (
        f"{sub_label}_{contrast_name}_constraint-mask_{contrast_hash}_{constraint_hash}.nii.gz"
    )


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import json

        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_json_dict(path: Path, payload: Dict[str, Any]) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_constraint_mask_path(contrast_path: Path, config: Any) -> Optional[Path]:
    spec = _load_constraint_mask_spec(config)
    if spec is None:
        return None

    sidecar = _read_json_dict(_contrast_sidecar_path(contrast_path))
    if sidecar is None:
        return None

    constraint_info = sidecar.get("constraint_mask")
    if not isinstance(constraint_info, dict):
        return None

    contrast_cfg = load_contrast_config(config)
    expected_hash = _get_constraint_mask_hash(
        spec,
        resample_to_freesurfer=bool(contrast_cfg.resample_to_freesurfer),
    )
    if str(constraint_info.get("hash", "")) != expected_hash:
        return None

    path_value = constraint_info.get("path")
    if not path_value:
        return None

    candidate = Path(str(path_value)).expanduser()
    if not candidate.exists():
        return None
    return candidate


def _resolve_fmri_stats_artifact(contrast_path: Path, config: Any) -> Optional[Path]:
    constraint_path = _resolve_constraint_mask_path(contrast_path, config)
    if constraint_path is not None:
        return constraint_path
    if _load_constraint_mask_spec(config) is not None:
        return None
    if contrast_path.exists():
        return contrast_path
    return None


###################################################################
# Main Entry Point
###################################################################


def build_fmri_contrast(
    bids_fmri_root: Path,
    bids_derivatives: Path,
    freesurfer_subjects_dir: Path,
    subject: str,
    config: Any,
    task: str = "task",
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

    constraint_spec = _load_constraint_mask_spec(config)
    _assert_constraint_mask_requires_z_output(
        output_type=cfg.output_type,
        constraint_spec=constraint_spec,
    )

    contrast_map, run_meta, glm_result, _contrast_def, _output_type = build_contrast_from_runs_detailed(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        cfg=cfg,
        output_dir=output_dir,
    )

    fs_subject_dir = freesurfer_subjects_dir / sub_label
    native_contrast_map = contrast_map
    constraint_mask_img = None
    constraint_mask_path = None
    constraint_mask_meta = None
    contrast_hash = _get_contrast_hash(cfg)

    if constraint_spec is not None and str(run_meta.get("output_type")) == "z_score":
        constraint_mask_img = build_thresholded_constraint_mask(
            native_contrast_map,
            threshold_mode=str(constraint_spec["threshold_mode"]),
            z_threshold=float(constraint_spec["threshold"]),
            fdr_q=float(constraint_spec["fdr_q"]),
            tail=str(constraint_spec["tail"]),
            analysis_mask_img=glm_result.mask_img,
            min_cluster_voxels=int(constraint_spec["cluster_min_voxels"]),
            min_cluster_volume_mm3=constraint_spec["cluster_min_volume_mm3"],
        )
        if not np.any(np.asarray(constraint_mask_img.get_fdata(), dtype=np.uint8) > 0):
            raise ValueError(
                "fMRI constraint mask thresholding produced an empty mask in native GLM space. "
                "Adjust feature_engineering.sourcelocalization.fmri threshold settings."
            )

    if cfg.resample_to_freesurfer and fs_subject_dir.exists():
        contrast_map = resample_to_freesurfer(contrast_map, fs_subject_dir)
        if constraint_mask_img is not None:
            constraint_mask_img = resample_to_freesurfer(
                constraint_mask_img,
                fs_subject_dir,
                interpolation="nearest",
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{sub_label}_{cfg.name}_{cfg.output_type}_{contrast_hash}.nii.gz"
    output_path = output_dir / output_name

    nib.save(contrast_map, str(output_path))
    logger.info("Saved contrast map to %s", output_path)

    if constraint_mask_img is not None and constraint_spec is not None:
        constraint_hash = _get_constraint_mask_hash(
            constraint_spec,
            resample_to_freesurfer=bool(cfg.resample_to_freesurfer and fs_subject_dir.exists()),
        )
        constraint_mask_path = _constraint_mask_output_path(
            output_dir=output_dir,
            sub_label=sub_label,
            contrast_name=cfg.name,
            contrast_hash=contrast_hash,
            constraint_hash=constraint_hash,
        )
        nib.save(constraint_mask_img, str(constraint_mask_path))
        constraint_mask_meta = {
            "artifact_type": "fmri_constraint_mask",
            "hash": constraint_hash,
            "path": str(constraint_mask_path),
            "source_contrast_path": str(output_path),
            "threshold_mode": str(constraint_spec["threshold_mode"]),
            "threshold": float(constraint_spec["threshold"]),
            "fdr_q": float(constraint_spec["fdr_q"]),
            "tail": str(constraint_spec["tail"]),
            "cluster_min_voxels": int(constraint_spec["cluster_min_voxels"]),
            "cluster_min_volume_mm3": constraint_spec["cluster_min_volume_mm3"],
            "resample_to_freesurfer": bool(cfg.resample_to_freesurfer and fs_subject_dir.exists()),
        }
        _write_json_dict(_contrast_sidecar_path(constraint_mask_path), constraint_mask_meta)
        run_meta["constraint_mask_path"] = str(constraint_mask_path)

    # Write a lightweight sidecar for reproducibility/debugging.
    sidecar_path = _contrast_sidecar_path(output_path)
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
        "constraint_mask": constraint_mask_meta,
        "run_inputs": run_meta,
    }
    _write_json_dict(sidecar_path, payload)

    return output_path


def ensure_fmri_stats_map(
    config: Any,
    bids_fmri_root: Optional[Path],
    bids_derivatives: Path,
    freesurfer_subjects_dir: Path,
    subject: str,
    task: str = "task",
) -> Optional[Path]:
    """
    Ensure fMRI stats map exists, building it if necessary.

    This is the main integration point with source_localization.py.
    Returns path to stats map (either pre-existing or newly built).

    Caches the contrast map to avoid rebuilding across multiple time windows.
    """
    src_cfg = get_config_value(config, "feature_engineering.sourcelocalization", {}) or {}
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
        resolved_path = _resolve_fmri_stats_artifact(cached_path, config)
        if resolved_path is not None:
            logger.info("Using cached fMRI stats artifact: %s", resolved_path)
            return resolved_path
        logger.info(
            "Cached contrast exists but its constraint-mask artifact is missing or stale. Rebuilding %s.",
            cached_path,
        )

    built_path = build_fmri_contrast(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        freesurfer_subjects_dir=freesurfer_subjects_dir,
        subject=subject,
        config=config,
        task=task,
        output_dir=output_dir,
    )
    resolved_path = _resolve_fmri_stats_artifact(built_path, config)
    if resolved_path is not None:
        return resolved_path
    return built_path
