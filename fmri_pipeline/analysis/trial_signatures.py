from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from fmri_pipeline.analysis.multivariate_signatures import compute_signature_expression
from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm
from fmri_pipeline.utils.bold_discovery import (
    build_first_level_model as _build_first_level_model,
    coerce_condition_value as _coerce_condition_value,
    discover_brain_mask_for_bold as _discover_brain_mask_for_bold,
    discover_fmriprep_preproc_bold as _discover_fmriprep_preproc_bold,
    get_tr_from_bold as _get_tr_from_bold,
    select_confounds as _select_confounds,
    select_consistent_run_source,
    validate_design_matrices as _validate_design_matrices,
)
from fmri_pipeline.utils.text import safe_slug as _safe_slug

logger = logging.getLogger(__name__)


def _normalize_input_source(input_source: str) -> str:
    normalized = str(input_source or "fmriprep").strip().lower()
    if normalized not in {"fmriprep", "bids_raw"}:
        raise ValueError(
            f"input_source must be 'fmriprep' or 'bids_raw', got {input_source!r}."
        )
    return normalized


def _normalize_optional_frequency(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric or null, got {value!r}.") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite when provided, got {value!r}.")
    if numeric_value <= 0:
        return None
    return numeric_value


@dataclass(frozen=True)
class TrialSignatureExtractionConfig:
    # Data selection
    input_source: str  # "fmriprep" | "bids_raw"
    fmriprep_space: str  # e.g. "MNI152NLin2009cAsym" or "T1w"
    require_fmriprep: bool
    runs: Optional[List[int]]
    task: str

    # Trial selection
    name: str
    condition_a_column: str
    condition_a_value: str
    condition_b_column: str
    condition_b_value: str

    # GLM
    hrf_model: str
    drift_model: Optional[str]
    high_pass_hz: float
    low_pass_hz: Optional[float]
    smoothing_fwhm: Optional[float]
    confounds_strategy: str

    # Signature extraction
    method: str  # "beta-series" | "lss"
    include_other_events: bool = True
    lss_other_regressors: str = "per_condition"  # "per_condition" | "all"
    # Events column used for `condition_scope_trial_types`.
    condition_scope_trial_type_column: str = ""
    # Events column used for `condition_scope_stim_phases`.
    condition_scope_phase_column: str = ""
    # Optional: scope which events.tsv rows are eligible for trial selection (prevents mixing phases).
    # Use ("all",) to disable scoping.
    condition_scope_trial_types: Optional[Tuple[str, ...]] = None
    # Optional: further restrict to phase values (only when `condition_scope_phase_column` exists).
    # Use ("all",) to disable scoping.
    condition_scope_stim_phases: Optional[Tuple[str, ...]] = None
    max_trials_per_run: Optional[int] = None
    fixed_effects_weighting: str = "variance"  # "variance" | "mean"
    signatures: Optional[Tuple[str, ...]] = None  # default: all discovered
    signature_group_column: Optional[str] = None  # e.g., "temperature"
    signature_group_values: Optional[Tuple[str, ...]] = None  # values to include (e.g., ("44.3","45.3"))
    signature_group_scope: str = "across_runs"  # "across_runs" | "per_run"

    # Outputs
    write_trial_betas: bool = False
    write_trial_variances: bool = False
    write_condition_betas: bool = True

    def normalized(self) -> "TrialSignatureExtractionConfig":
        input_source = _normalize_input_source(self.input_source)

        drift_model = (self.drift_model or "").strip().lower() or None
        if drift_model == "none":
            drift_model = None

        low_pass_hz = _normalize_optional_frequency(
            self.low_pass_hz,
            field_name="low_pass_hz",
        )

        smoothing_fwhm = normalize_smoothing_fwhm(self.smoothing_fwhm)

        method = (self.method or "beta-series").strip().lower()
        if method not in {"beta-series", "lss"}:
            raise ValueError(
                f"method must be 'beta-series' or 'lss', got {self.method!r}."
            )

        lss_other = (self.lss_other_regressors or "per_condition").strip().lower().replace("-", "_")
        if lss_other not in {"per_condition", "all"}:
            raise ValueError(
                "lss_other_regressors must be 'per_condition' or 'all', "
                f"got {self.lss_other_regressors!r}."
            )

        weight = (self.fixed_effects_weighting or "variance").strip().lower()
        if weight not in {"variance", "mean"}:
            raise ValueError(
                "fixed_effects_weighting must be 'variance' or 'mean', "
                f"got {self.fixed_effects_weighting!r}."
            )

        def _norm_scope(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
            if not values:
                return None
            raw: List[str] = []
            for item in values:
                if item is None:
                    continue
                s = str(item).replace(";", ",").strip()
                if not s:
                    continue
                raw.extend([p.strip() for p in s.split(",") if p.strip()])
            if not raw:
                return None
            if any(s.strip().lower() in {"all", "*", "@all"} for s in raw):
                # Explicitly disable scoping. Represent as empty tuple so we can distinguish
                # "user disabled" from "user did not specify" (None).
                return tuple()
            out: List[str] = []
            seen = set()
            for s in raw:
                key = s.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s.strip())
            return tuple(out) if out else None

        scope_trial_types = _norm_scope(self.condition_scope_trial_types)
        scope_stim_phases = _norm_scope(self.condition_scope_stim_phases)

        group_col = (self.signature_group_column or "").strip() or None
        group_vals = None
        if self.signature_group_values:
            raw_vals: List[str] = []
            for item in self.signature_group_values:
                if item is None:
                    continue
                s = str(item).replace(";", ",").strip()
                if not s:
                    continue
                raw_vals.extend([p.strip() for p in s.split(",") if p.strip()])
            dedup_vals: List[str] = []
            seen_vals = set()
            for v in raw_vals:
                k = v.strip()
                if not k:
                    continue
                if k in seen_vals:
                    continue
                seen_vals.add(k)
                dedup_vals.append(k)
            group_vals = tuple(dedup_vals) if dedup_vals else None

        group_scope = (self.signature_group_scope or "across_runs").strip().lower().replace("-", "_")
        if group_scope not in {"across_runs", "per_run"}:
            raise ValueError(
                "signature_group_scope must be 'across_runs' or 'per_run', "
                f"got {self.signature_group_scope!r}."
            )

        scope_trial_type_column = (self.condition_scope_trial_type_column or "").strip()
        scope_phase_column = (self.condition_scope_phase_column or "").strip()
        if scope_trial_types and not scope_trial_type_column:
            raise ValueError(
                "condition_scope_trial_types requires condition_scope_trial_type_column."
            )
        if scope_stim_phases and not scope_phase_column:
            raise ValueError(
                "condition_scope_stim_phases requires condition_scope_phase_column."
            )

        return TrialSignatureExtractionConfig(
            **{
                **asdict(self),
                "input_source": input_source,
                "drift_model": drift_model,
                "low_pass_hz": low_pass_hz,
                "smoothing_fwhm": smoothing_fwhm,
                "method": method,
                "lss_other_regressors": lss_other,
                "condition_scope_trial_types": scope_trial_types,
                "condition_scope_stim_phases": scope_stim_phases,
                "condition_scope_trial_type_column": scope_trial_type_column,
                "condition_scope_phase_column": scope_phase_column,
                "fixed_effects_weighting": weight,
                "signature_group_column": group_col,
                "signature_group_values": group_vals,
                "signature_group_scope": group_scope,
            }
        )


@dataclass(frozen=True)
class TrialInfo:
    run: int
    run_label: str
    trial_index: int
    condition: str  # "A" | "B" | signature group label
    regressor: str
    onset: float
    duration: float
    original_trial_type: str
    source_events_path: Path
    source_row: int
    extra: Dict[str, str]


def _resample_mask_to_target(mask_img: Any, target_img: Any) -> Any:
    import numpy as np  # type: ignore

    try:
        from nilearn import image as nilearn_image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Mask resampling requires nilearn to align masks to the target image grid.") from exc

    # Best-effort: if already aligned, avoid resampling. (nilearn will still handle it if needed.)
    try:
        same_shape = tuple(getattr(mask_img, "shape", ())) == tuple(getattr(target_img, "shape", ()))
        same_affine = np.allclose(np.asanyarray(mask_img.affine), np.asanyarray(target_img.affine))
        if same_shape and same_affine:
            return mask_img
    except Exception as exc:
        logger.debug("Mask/target alignment pre-check failed; falling back to resampling: %s", exc)
    return nilearn_image.resample_to_img(
        mask_img, target_img, interpolation="nearest",
        force_resample=True, copy_header=True,
    )


def _union_masks_to_target(mask_imgs: Sequence[Any], target_img: Any) -> Any:
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore

    if not mask_imgs:
        raise ValueError("No masks provided.")
    resampled = [_resample_mask_to_target(m, target_img) for m in mask_imgs]
    datas = [np.asanyarray(m.get_fdata()) for m in resampled]
    m0 = datas[0]
    union = np.isfinite(m0) & (m0 > 0)
    for d in datas[1:]:
        union = union | (np.isfinite(d) & (d > 0))
    ref = resampled[0]
    return nib.Nifti1Image(union.astype(np.uint8), ref.affine, ref.header)


def _normalize_confounds_strategy(strategy: str) -> str:
    normalized = str(strategy or "auto").strip().lower()
    if normalized in {"", "default"}:
        return "auto"
    return normalized


def _discover_confounds(
    *,
    bids_derivatives: Path,
    sub_label: str,
    task: str,
    run_num: int,
) -> Optional[Path]:
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
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            p = d / pat
            if p.exists():
                return p
    return None


def _discover_runs(
    *,
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    subject: str,
    task: str,
    runs: Optional[List[int]],
    input_source: str,
    fmriprep_space: Optional[str],
    require_fmriprep: bool,
) -> List[Tuple[int, Path, Path, Optional[Path]]]:
    """
    Return list of (run_num, bold_path, events_path, confounds_path).
    """
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    func_dir = bids_fmri_root / sub_label / "func"
    if not func_dir.exists():
        raise FileNotFoundError(f"fMRI func directory not found: {func_dir}")

    # Discover run numbers from events files first.
    events_glob = [
        p for p in sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_events.tsv")) if not p.name.endswith("_bold_events.tsv")
    ]
    if not events_glob:
        events_glob = sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_bold_events.tsv"))
    run_nums: List[int] = []
    for p in events_glob:
        try:
            run_str = p.name.split("_run-")[1].split("_")[0]
            run_nums.append(int(run_str))
        except Exception:
            continue

    if runs is not None:
        want = {int(r) for r in runs}
        run_nums = [r for r in sorted(set(run_nums)) if r in want]
    else:
        run_nums = sorted(set(run_nums))

    if not run_nums:
        if runs is not None:
            requested = sorted({int(r) for r in runs})
            raise FileNotFoundError(
                f"None of the requested runs were found for subject {subject}, task {task}: {requested}"
            )
        raise FileNotFoundError(f"No runs found for subject {subject}, task {task} in {func_dir}")

    selected_input_source = _normalize_input_source(input_source)
    preproc_by_run: Dict[int, Optional[Path]] = {}
    if selected_input_source == "fmriprep" and bids_derivatives is not None:
        selected_input_source, preproc_by_run = select_consistent_run_source(
            run_numbers=run_nums,
            discover_preproc_bold=lambda run_num: _discover_fmriprep_preproc_bold(
                bids_derivatives=bids_derivatives,
                subject=sub_label,
                task=task,
                run_num=run_num,
                space=fmriprep_space,
            ),
            require_fmriprep=require_fmriprep,
        )

    out: List[Tuple[int, Path, Path, Optional[Path]]] = []
    for run_num in run_nums:
        events_patterns = [
            f"{sub_label}_task-{task}_run-{run_num:02d}_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num}_events.tsv",
            f"{sub_label}_task-{task}_run-0{run_num}_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num:02d}_bold_events.tsv",
            f"{sub_label}_task-{task}_run-{run_num}_bold_events.tsv",
            f"{sub_label}_task-{task}_run-0{run_num}_bold_events.tsv",
        ]
        events_path = next((func_dir / n for n in events_patterns if (func_dir / n).exists()), None)
        if events_path is None:
            continue

        bold_path: Optional[Path] = None
        confounds_path: Optional[Path] = None

        if selected_input_source == "fmriprep":
            bold_path = preproc_by_run.get(int(run_num))
            confounds_path = _discover_confounds(
                bids_derivatives=bids_derivatives,
                sub_label=sub_label,
                task=task,
                run_num=run_num,
            )
        else:
            raw_patterns = [
                f"{sub_label}_task-{task}_run-{run_num:02d}_bold.nii.gz",
                f"{sub_label}_task-{task}_run-{run_num}_bold.nii.gz",
            ]
            bold_path = next((func_dir / n for n in raw_patterns if (func_dir / n).exists()), None)

        if bold_path is None:
            continue

        out.append((int(run_num), bold_path, events_path, confounds_path))

    if runs is not None:
        requested_runs = {int(run_num) for run_num in runs}
        discovered_runs = {
            int(run_num)
            for run_num, _bold_path, _events_path, _confounds_path in out
        }
        missing_runs = sorted(requested_runs - discovered_runs)
        if missing_runs:
            raise FileNotFoundError(
                "Some requested runs could not be resolved to matching BOLD + events inputs: "
                f"{missing_runs}."
            )

    if not out:
        raise FileNotFoundError(f"No usable runs found for subject {subject}, task {task}")
    return out


def _make_trial_regressor(run_label: str, trial_index: int, condition: str) -> str:
    return f"trial_{_safe_slug(run_label)}_{trial_index:03d}_{_safe_slug(condition).lower()}"


def _extract_trials_for_run(
    *,
    events_df: Any,
    cfg: TrialSignatureExtractionConfig,
    events_path: Path,
    run_num: int,
) -> Tuple[List[TrialInfo], Any]:
    """
    Returns (trial_infos, modeled_events_df).
    """
    import pandas as pd  # type: ignore

    required = {"onset", "duration"}
    if not required.issubset(set(events_df.columns)):
        raise ValueError(f"Events file missing required columns {required}: {events_path}")

    col_a = str(cfg.condition_a_column).strip()
    col_b = str(cfg.condition_b_column).strip()
    group_col = (cfg.signature_group_column or "").strip()
    group_vals = tuple(cfg.signature_group_values or ())

    group_mode = bool(group_col and group_vals)
    mask_a = None
    sel_mask = None

    scope_mask = pd.Series([True] * int(len(events_df)), index=events_df.index)
    scope_trial_type_column = str(cfg.condition_scope_trial_type_column or "").strip()
    scope_phase_column = str(cfg.condition_scope_phase_column or "").strip()

    scope_trial_types = tuple(cfg.condition_scope_trial_types or ())
    if scope_trial_types:
        if not scope_trial_type_column:
            raise ValueError(
                "condition_scope_trial_types is set but no condition_scope_trial_type_column was configured."
            )
        if scope_trial_type_column not in events_df.columns:
            raise ValueError(
                "condition_scope_trial_types is set but events file has no "
                f"'{scope_trial_type_column}' column: {events_path}"
            )
        allow = {str(v).strip() for v in scope_trial_types if str(v).strip()}
        scope_mask &= events_df[scope_trial_type_column].astype(str).str.strip().isin(list(allow))

    scope_stim_phases = tuple(cfg.condition_scope_stim_phases or ())
    if scope_stim_phases:
        if not scope_phase_column:
            raise ValueError(
                "condition_scope_stim_phases is set but no condition_scope_phase_column was configured."
            )
        if scope_phase_column not in events_df.columns:
            raise ValueError(
                "condition_scope_stim_phases is set but events file has no "
                f"'{scope_phase_column}' column: {events_path}"
            )
        allow = {str(v).strip() for v in scope_stim_phases if str(v).strip()}
        scope_mask &= events_df[scope_phase_column].astype(str).str.strip().isin(list(allow))

    if group_mode:
        if group_col not in events_df.columns:
            raise ValueError(f"Events file missing signature group column: {group_col} in {events_path}")
        allowed = {str(v).strip() for v in group_vals if str(v).strip()}
        col_str = events_df[group_col].astype(str).str.strip()
        sel_mask = scope_mask & col_str.isin(list(allowed))
        selected = events_df.loc[sel_mask].copy()
        if selected.empty:
            raise ValueError(
                f"No trials matched signature group selection in {events_path} (column={group_col}, values={sorted(allowed)})"
            )
        selected = selected.sort_values("onset").reset_index(drop=False)
    else:
        if col_a not in events_df.columns or col_b not in events_df.columns:
            raise ValueError(f"Events file missing selection columns: {col_a}, {col_b} in {events_path}")

        val_a = _coerce_condition_value(cfg.condition_a_value, events_df[col_a])
        val_b = _coerce_condition_value(cfg.condition_b_value, events_df[col_b])

        mask_a = scope_mask & (events_df[col_a] == val_a)
        mask_b = scope_mask & (events_df[col_b] == val_b)
        overlap_mask = mask_a & mask_b
        overlap_count = int(overlap_mask.sum())
        if overlap_count > 0:
            raise ValueError(
                f"Condition definitions overlap for {overlap_count} event(s): "
                f"{col_a}={cfg.condition_a_value!r} and {col_b}={cfg.condition_b_value!r} "
                "select the same rows. Trial conditions must be mutually exclusive."
            )
        sel_mask = mask_a | mask_b

        selected = events_df.loc[sel_mask].copy()
        if selected.empty:
            raise ValueError(f"No trials matched cond A/B selection in {events_path}")

        selected = selected.sort_values("onset").reset_index(drop=False)

    run_label = f"run-{run_num:02d}"
    trials: List[TrialInfo] = []

    max_trials = cfg.max_trials_per_run
    if max_trials is not None and int(max_trials) > 0:
        selected = selected.iloc[: int(max_trials)].copy()

    modeled_rows: List[Dict[str, Any]] = []

    for idx, row in selected.iterrows():
        if group_mode:
            condition = str(row.get(group_col, "")).strip() or "group"
        else:
            condition = "A" if bool(mask_a.loc[row["index"]]) else "B"
        reg = _make_trial_regressor(run_label, int(idx) + 1, condition)

        onset = float(row["onset"])
        duration = float(row["duration"])
        orig_tt = str(row.get("trial_type", "n/a"))

        extra: Dict[str, str] = {}
        for k, v in row.items():
            if k in {"onset", "duration", "trial_type", "index"}:
                continue
            if v is None:
                continue
            extra[str(k)] = str(v)

        trials.append(
            TrialInfo(
                run=int(run_num),
                run_label=run_label,
                trial_index=int(idx) + 1,
                condition=condition,
                regressor=reg,
                onset=onset,
                duration=duration,
                original_trial_type=orig_tt,
                source_events_path=events_path,
                source_row=int(row["index"]),
                extra=extra,
            )
        )
        modeled_rows.append({"onset": onset, "duration": duration, "trial_type": reg})

    if cfg.include_other_events:
        others = events_df.loc[~sel_mask].copy()
        if not others.empty:
            # Group non-selected events by their original trial_type (best-effort).
            for _idx, row in others.iterrows():
                tt = str(row.get("trial_type", "other"))
                modeled_rows.append(
                    {
                        "onset": float(row["onset"]),
                        "duration": float(row["duration"]),
                        "trial_type": f"nuis_{_safe_slug(tt)}",
                    }
                )

    modeled_events = pd.DataFrame(modeled_rows, columns=["onset", "duration", "trial_type"])
    return trials, modeled_events


def _build_lss_events(
    *,
    trial: TrialInfo,
    all_trials: Sequence[TrialInfo],
    original_events_df: Any,
    cfg: TrialSignatureExtractionConfig,
) -> Any:
    import pandas as pd  # type: ignore

    rows: List[Dict[str, Any]] = []
    # Target trial regressor
    rows.append({"onset": trial.onset, "duration": trial.duration, "trial_type": "target"})
    grouping_enabled = bool(cfg.signature_group_column and cfg.signature_group_values)

    # Other selected trials
    others = [t for t in all_trials if not (t.run == trial.run and t.trial_index == trial.trial_index)]
    if cfg.lss_other_regressors == "all":
        for t in others:
            rows.append({"onset": t.onset, "duration": t.duration, "trial_type": "other_trials"})
    else:
        for t in others:
            if grouping_enabled:
                other_label = f"other_group_{_safe_slug(str(t.condition)).lower()}"
            else:
                other_label = "other_cond_a" if t.condition == "A" else "other_cond_b"
            rows.append(
                {
                    "onset": t.onset,
                    "duration": t.duration,
                    "trial_type": other_label,
                }
            )

    if cfg.include_other_events:
        # Add non-selected events (those not represented in all_trials) as nuisance regressors.
        represented_rows = {int(t.source_row) for t in all_trials}
        for idx, row in original_events_df.iterrows():
            try:
                onset = float(row["onset"])
                duration = float(row["duration"])
            except Exception:
                continue
            row_id = None
            try:
                row_id = int(idx)
            except Exception:
                row_id = None
            if row_id is not None and row_id in represented_rows:
                continue
            tt = str(row.get("trial_type", "other"))
            rows.append({"onset": onset, "duration": duration, "trial_type": f"nuis_{_safe_slug(tt)}"})

    return pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])


def _contrast_vector_for_column(columns: Sequence[str], col: str) -> Any:
    # Nilearn's FirstLevelModel.compute_contrast treats Python lists as "one contrast per run".
    # For single-run models, passing a list of floats (i.e., a 1D contrast vector) is ambiguous
    # and can be misinterpreted as multiple per-run contrasts. Use a numpy array to force the
    # intended 1D vector semantics.
    import numpy as np  # type: ignore

    try:
        idx = list(columns).index(col)
    except Exception as exc:
        raise KeyError(f"Column not found in design matrix: {col}") from exc
    vec = np.zeros(len(columns), dtype=float)
    vec[idx] = 1.0
    return vec


def _contrast_vector_for_columns(columns: Sequence[str], cols: Sequence[str]) -> Any:
    import numpy as np  # type: ignore

    if not cols:
        raise ValueError("At least one design-matrix column is required.")

    index_by_name = {str(name): idx for idx, name in enumerate(columns)}
    vec = np.zeros(len(columns), dtype=float)
    for col in cols:
        key = str(col)
        if key not in index_by_name:
            raise KeyError(f"Column not found in design matrix: {col}")
        vec[int(index_by_name[key])] = 1.0
    vec /= float(len(cols))
    return vec


def _combine_effect_images(
    *,
    effects: Sequence[Any],
    variances: Optional[Sequence[Any]],
    method: str,
) -> Any:
    """
    Combine a set of effect images using either a simple mean or inverse-variance weighting.

    The caller is responsible for ensuring these inputs are suitable to combine.
    """
    import numpy as np  # type: ignore
    import nibabel as nib  # type: ignore

    if not effects:
        raise ValueError("No effect images provided.")

    ref = effects[0]
    ref_img = nib.load(str(ref)) if isinstance(ref, (str, Path)) else ref
    eff_imgs = [nib.load(str(e)) if isinstance(e, (str, Path)) else e for e in effects]
    eff = np.stack([np.asanyarray(img.dataobj) for img in eff_imgs], axis=0)

    method = (method or "variance").strip().lower()
    if method == "mean" or not variances:
        out = np.nanmean(eff, axis=0)
        return nib.Nifti1Image(out, ref_img.affine, ref_img.header)

    var_imgs = [nib.load(str(v)) if isinstance(v, (str, Path)) else v for v in variances]
    var = np.stack([np.asanyarray(img.dataobj) for img in var_imgs], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / var
        w[~np.isfinite(w)] = 0.0
        num = np.sum(w * eff, axis=0)
        den = np.sum(w, axis=0)
        out = np.full_like(num, np.nan, dtype=float)
        m = den > 0
        out[m] = num[m] / den[m]
    return nib.Nifti1Image(out, ref_img.affine, ref_img.header)


def _write_tsv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def run_trial_signature_extraction_for_subject(
    *,
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    deriv_root: Path,
    subject: str,
    cfg: TrialSignatureExtractionConfig,
    signature_root: Optional[Path],
    signature_specs: Optional[List[Any]] = None,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Compute trial-wise beta maps (beta-series or LSS) and multivariate signature expression.

        Outputs (per subject) under:
          <deriv_root>/sub-XX/fmri/<beta_series|lss>/task-<task>/contrast-<name>/
            - trials.tsv
            - signatures/trial_signature_expression.tsv
            - condition_betas/*.nii.gz (optional)
            - signatures/condition_signature_expression.tsv
            - provenance.json
            - trial_betas/**/*.nii.gz (optional)
        """
    cfg = cfg.normalized()
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    if signature_root is not None and signature_specs:
        # Signature weight maps must be in the same space as the beta images.
        # Most shared signature maps are distributed in MNI space.
        space = str(cfg.fmriprep_space or "").strip().lower()
        if "mni" not in space:
            raise ValueError(
                "Trial-wise signature extraction requires images in the same space as the "
                "signature weight maps (typically MNI space). "
                "Re-run with --input-source fmriprep --fmriprep-space MNI152NLin2009cAsym "
                "(or provide MNI-space inputs)."
            )
    if cfg.method == "beta-series":
        root = deriv_root / sub_label / "fmri" / "beta_series" / f"task-{cfg.task}" / f"contrast-{_safe_slug(cfg.name)}"
    else:
        root = deriv_root / sub_label / "fmri" / "lss" / f"task-{cfg.task}" / f"contrast-{_safe_slug(cfg.name)}"
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else root
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=cfg.task,
        runs=cfg.runs,
        input_source=cfg.input_source,
        fmriprep_space=cfg.fmriprep_space,
        require_fmriprep=cfg.require_fmriprep,
    )

    condition_map_inference = (
        "run_level_fixed_effects" if cfg.method == "beta-series" else "descriptive_trial_summary"
    )
    provenance = {
        "subject": sub_label,
        "task": cfg.task,
        "method": cfg.method,
        "cfg": asdict(cfg),
        "signature_root": str(signature_root) if signature_root else None,
        "n_runs": len(runs),
        "condition_map_inference": condition_map_inference,
    }
    phase_column_name = str(cfg.condition_scope_phase_column or "").strip()

    trial_infos: List[TrialInfo] = []
    trial_rows_out: List[Dict[str, Any]] = []
    trial_sig_rows: List[Dict[str, Any]] = []
    run_brain_masks: List[Any] = []
    confounds_strategy = _normalize_confounds_strategy(cfg.confounds_strategy)

    grouping_enabled = bool(cfg.signature_group_column and cfg.signature_group_values)

    group_effects: Dict[str, List[Any]] = {}
    group_vars: Dict[str, List[Any]] = {}
    group_effects_by_run: Dict[int, Dict[str, List[Any]]] = {}
    group_vars_by_run: Dict[int, Dict[str, List[Any]]] = {}

    def _append_group(container: Dict[str, List[Any]], key: str, value: Any) -> None:
        container.setdefault(str(key), []).append(value)

    def _append_group_by_run(container: Dict[int, Dict[str, List[Any]]], run_num: int, key: str, value: Any) -> None:
        container.setdefault(int(run_num), {}).setdefault(str(key), []).append(value)

    for run_num, bold_path, events_path, confounds_path in runs:
        import pandas as pd  # type: ignore
        import nibabel as nib  # type: ignore

        events_df = pd.read_csv(events_path, sep="\t")

        mask_img = None
        mask_path = _discover_brain_mask_for_bold(bold_path)
        if mask_path is not None and mask_path.exists():
            try:
                mask_img = nib.load(str(mask_path))
            except Exception:
                mask_img = None
        if mask_img is not None:
            run_brain_masks.append(mask_img)

        tr = _get_tr_from_bold(bold_path)

        # Determine trial list and build run-level modeled events.
        trials, modeled_events = _extract_trials_for_run(
            events_df=events_df,
            cfg=cfg,
            events_path=events_path,
            run_num=run_num,
        )
        if not trials:
            continue

        confounds = None
        conf_cols: List[str] = []
        if confounds_strategy not in {"none", "no", "off"}:
            if confounds_path is None or not confounds_path.exists():
                raise ValueError(
                    "Trial-wise fMRI GLMs require confounds for every included run unless "
                    "confounds_strategy is 'none'. "
                    f"Missing confounds for {bold_path.name}."
                )
            confounds, conf_cols = _select_confounds(
                confounds_path,
                confounds_strategy,
                logger=logger,
            )
            if confounds is None or not conf_cols:
                raise ValueError(
                    "Trial-wise fMRI GLMs require at least one confound regressor for every "
                    f"included run. Strategy={confounds_strategy!r}, file={confounds_path}."
                )

        if cfg.method == "beta-series":
            flm = _build_first_level_model(tr=tr, cfg=cfg, mask_img=mask_img, logger=logger)
            flm.fit(bold_path, events=modeled_events, confounds=confounds)
            _validate_design_matrices(
                flm,
                context=f"Trial-wise beta-series GLM ({bold_path.name})",
                min_residual_dof=1,
            )

            dm = flm.design_matrices_[0]
            dm_cols = list(getattr(dm, "columns", []))
            run_regressors_by_condition: Dict[str, List[str]] = {}

            for t in trials:
                if progress_callback is not None:
                    progress_callback(
                        f"beta-series {sub_label} {t.run_label} trial-{t.trial_index:03d}"
                    )
                trial_infos.append(t)
                trial_rows_out.append(
                    {
                        "subject": sub_label,
                        "task": cfg.task,
                        "run": t.run_label,
                        "trial_index": t.trial_index,
                        "condition": t.condition,
                        "regressor": t.regressor,
                        "onset": f"{t.onset:.6f}",
                        "duration": f"{t.duration:.6f}",
                        "original_trial_type": t.original_trial_type,
                        "events_file": str(t.source_events_path),
                        "events_row": t.source_row,
                        **{f"events_{k}": v for k, v in sorted(t.extra.items())},
                    }
                )

                con = _contrast_vector_for_column(dm_cols, t.regressor)
                beta_img = flm.compute_contrast(con, output_type="effect_size")
                var_img = None
                try:
                    var_img = flm.compute_contrast(con, output_type="effect_variance")
                except Exception:
                    var_img = None

                run_regressors_by_condition.setdefault(str(t.condition), []).append(str(t.regressor))

                if cfg.write_trial_betas:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_{t.regressor}_beta.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(beta_img, str(p))
                if cfg.write_trial_variances and var_img is not None:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_{t.regressor}_var.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(var_img, str(p))

                if signature_root is not None and signature_specs:
                    sigs = compute_signature_expression(
                        stat_or_effect_img=beta_img,
                        signature_root=signature_root,
                        signature_specs=signature_specs,
                        mask_img=mask_img,
                        signatures=cfg.signatures,
                    )
                    for s in sigs:
                        trial_sig_rows.append(
                            {
                                "subject": sub_label,
                                "task": cfg.task,
                                "method": cfg.method,
                                "run": t.run_label,
                                "run_num": t.run,
                                "trial_index": t.trial_index,
                                "condition": t.condition,
                                "regressor": t.regressor,
                                "original_trial_type": t.original_trial_type,
                                "phase_column": phase_column_name,
                                "phase_value": str(t.extra.get(phase_column_name, "")),
                                "onset": f"{t.onset:.6f}",
                                "duration": f"{t.duration:.6f}",
                                "signature": s.name,
                                "dot": f"{s.dot:.8g}",
                                "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                "n_voxels": s.n_voxels,
                                "weights": str(s.weight_path),
                            }
                        )

            for condition, regressors in sorted(run_regressors_by_condition.items()):
                run_contrast = _contrast_vector_for_columns(dm_cols, regressors)
                run_effect_img = flm.compute_contrast(run_contrast, output_type="effect_size")
                _append_group(group_effects, condition, run_effect_img)
                _append_group_by_run(group_effects_by_run, run_num, condition, run_effect_img)

                try:
                    run_var_img = flm.compute_contrast(run_contrast, output_type="effect_variance")
                except Exception:
                    run_var_img = None
                if run_var_img is not None:
                    _append_group(group_vars, condition, run_var_img)
                    _append_group_by_run(group_vars_by_run, run_num, condition, run_var_img)

        else:
            # LSS: one model per trial within each run
            # Build the run-level trial list once, then fit per-trial models.
            for t in trials:
                if progress_callback is not None:
                    progress_callback(
                        f"lss {sub_label} {t.run_label} trial-{t.trial_index:03d}"
                    )
                trial_infos.append(t)
                trial_rows_out.append(
                    {
                        "subject": sub_label,
                        "task": cfg.task,
                        "run": t.run_label,
                        "trial_index": t.trial_index,
                        "condition": t.condition,
                        "regressor": "target",
                        "onset": f"{t.onset:.6f}",
                        "duration": f"{t.duration:.6f}",
                        "original_trial_type": t.original_trial_type,
                        "events_file": str(t.source_events_path),
                        "events_row": t.source_row,
                        **{f"events_{k}": v for k, v in sorted(t.extra.items())},
                    }
                )

                lss_events = _build_lss_events(trial=t, all_trials=trials, original_events_df=events_df, cfg=cfg)
                flm = _build_first_level_model(tr=tr, cfg=cfg, mask_img=mask_img, logger=logger)
                flm.fit(bold_path, events=lss_events, confounds=confounds)
                _validate_design_matrices(
                    flm,
                    context=(
                        f"Trial-wise LSS GLM ({bold_path.name}, {t.run_label}, "
                        f"trial {t.trial_index:03d})"
                    ),
                    min_residual_dof=1,
                )

                dm = flm.design_matrices_[0]
                dm_cols = list(getattr(dm, "columns", []))
                con = _contrast_vector_for_column(dm_cols, "target")
                beta_img = flm.compute_contrast(con, output_type="effect_size")
                var_img = None
                try:
                    var_img = flm.compute_contrast(con, output_type="effect_variance")
                except Exception:
                    var_img = None

                _append_group(group_effects, t.condition, beta_img)
                _append_group_by_run(group_effects_by_run, run_num, t.condition, beta_img)
                if var_img is not None:
                    _append_group(group_vars, t.condition, var_img)
                    _append_group_by_run(group_vars_by_run, run_num, t.condition, var_img)

                if cfg.write_trial_betas:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_trial-{t.trial_index:03d}_beta.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(beta_img, str(p))
                if cfg.write_trial_variances and var_img is not None:
                    p = out_dir / "trial_betas" / t.run_label / f"{sub_label}_task-{cfg.task}_{t.run_label}_trial-{t.trial_index:03d}_var.nii.gz"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(var_img, str(p))

                if signature_root is not None and signature_specs:
                    sigs = compute_signature_expression(
                        stat_or_effect_img=beta_img,
                        signature_root=signature_root,
                        signature_specs=signature_specs,
                        mask_img=mask_img,
                        signatures=cfg.signatures,
                    )
                    for s in sigs:
                        trial_sig_rows.append(
                            {
                                "subject": sub_label,
                                "task": cfg.task,
                                "method": cfg.method,
                                "run": t.run_label,
                                "run_num": t.run,
                                "trial_index": t.trial_index,
                                "condition": t.condition,
                                "regressor": "target",
                                "original_trial_type": t.original_trial_type,
                                "phase_column": phase_column_name,
                                "phase_value": str(t.extra.get(phase_column_name, "")),
                                "signature": s.name,
                                "dot": f"{s.dot:.8g}",
                                "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                "n_voxels": s.n_voxels,
                                "weights": str(s.weight_path),
                                "onset": f"{t.onset:.6f}",
                                "duration": f"{t.duration:.6f}",
                            }
                        )

    # Write metadata tables
    _write_tsv(out_dir / "trials.tsv", trial_rows_out)
    trial_signature_path = out_dir / "signatures" / "trial_signature_expression.tsv"
    if signature_root is not None and signature_specs:
        _write_tsv(
            trial_signature_path,
            trial_sig_rows,
        )
    elif trial_signature_path.exists():
        trial_signature_path.unlink()

    # Condition/group summary maps + signatures
    cond_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    if (group_effects) and (cfg.write_condition_betas or signature_root is not None):
        import nibabel as nib  # type: ignore

        cond_dir = out_dir / "condition_betas"
        if cfg.write_condition_betas:
            cond_dir.mkdir(parents=True, exist_ok=True)

        if not grouping_enabled:
            cond_a_effects = list(group_effects.get("A", []))
            cond_a_vars = list(group_vars.get("A", []))
            cond_b_effects = list(group_effects.get("B", []))
            cond_b_vars = list(group_vars.get("B", []))

            a_img = None
            b_img = None
            if cond_a_effects:
                a_img = _combine_effect_images(
                    effects=cond_a_effects,
                    variances=cond_a_vars if cond_a_vars else None,
                    method=cfg.fixed_effects_weighting,
                )
                if cfg.write_condition_betas:
                    nib.save(a_img, str(cond_dir / f"{sub_label}_task-{cfg.task}_cond-a_beta.nii.gz"))
            if cond_b_effects:
                b_img = _combine_effect_images(
                    effects=cond_b_effects,
                    variances=cond_b_vars if cond_b_vars else None,
                    method=cfg.fixed_effects_weighting,
                )
                if cfg.write_condition_betas:
                    nib.save(b_img, str(cond_dir / f"{sub_label}_task-{cfg.task}_cond-b_beta.nii.gz"))

            diff_img = None
            if a_img is not None and b_img is not None:
                import numpy as np  # type: ignore

                a = np.asanyarray(a_img.dataobj)
                b = np.asanyarray(b_img.dataobj)
                diff_img = nib.Nifti1Image(a - b, a_img.affine, a_img.header)
                if cfg.write_condition_betas:
                    nib.save(diff_img, str(cond_dir / f"{sub_label}_task-{cfg.task}_cond-a_minus_b_beta.nii.gz"))

            if signature_root is not None and signature_specs:
                for label, img in [
                    ("cond_a", a_img),
                    ("cond_b", b_img),
                    ("cond_a_minus_b", diff_img),
                ]:
                    if img is None:
                        continue
                    brain_union = None
                    if run_brain_masks:
                        try:
                            brain_union = _union_masks_to_target(run_brain_masks, img)
                        except Exception:
                            brain_union = None
                    sigs = compute_signature_expression(
                        stat_or_effect_img=img,
                        signature_root=signature_root,
                        signature_specs=signature_specs,
                        mask_img=brain_union,
                        signatures=cfg.signatures,
                    )
                    for s in sigs:
                        cond_rows.append(
                            {
                                "subject": sub_label,
                                "task": cfg.task,
                                "method": cfg.method,
                                "map_inference": condition_map_inference,
                                "map": label,
                                "signature": s.name,
                                "dot": f"{s.dot:.8g}",
                                "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                "n_voxels": s.n_voxels,
                                "weights": str(s.weight_path),
                            }
                        )
                _write_tsv(out_dir / "signatures" / "condition_signature_expression.tsv", cond_rows)

        else:
            def _count_trials_for_group(run_label: Optional[str], group: str) -> int:
                n = 0
                for t in trial_infos:
                    if str(t.condition) != str(group):
                        continue
                    if run_label is not None and str(t.run_label) != str(run_label):
                        continue
                    n += 1
                return n

            def _iter_group_sets():
                scope = str(cfg.signature_group_scope or "across_runs").strip().lower().replace("-", "_")
                if scope == "per_run":
                    for r in sorted(group_effects_by_run.keys()):
                        by_group = group_effects_by_run.get(int(r), {})
                        for g in sorted(by_group.keys()):
                            yield (
                                "per_run",
                                f"run-{int(r):02d}",
                                int(r),
                                str(g),
                                by_group.get(str(g), []),
                                group_vars_by_run.get(int(r), {}).get(str(g), []),
                            )
                else:
                    for g in sorted(group_effects.keys()):
                        yield (
                            "across_runs",
                            "",
                            None,
                            str(g),
                            group_effects.get(str(g), []),
                            group_vars.get(str(g), []),
                        )

            for scope, run_label, run_num, group, effects, variances in _iter_group_sets():
                if not effects:
                    continue
                img = _combine_effect_images(
                    effects=effects,
                    variances=variances if variances else None,
                    method=cfg.fixed_effects_weighting,
                )

                if cfg.write_condition_betas:
                    parts = [f"{sub_label}_task-{cfg.task}"]
                    if run_label:
                        parts.append(run_label)
                    parts.append(f"group-{_safe_slug(group)}")
                    nib.save(img, str(cond_dir / ("_".join(parts) + "_beta.nii.gz")))

                n_trials = _count_trials_for_group(run_label or None, group)

                if signature_root is not None and signature_specs:
                    brain_union = None
                    if run_brain_masks:
                        try:
                            brain_union = _union_masks_to_target(run_brain_masks, img)
                        except Exception:
                            brain_union = None
                    sigs = compute_signature_expression(
                        stat_or_effect_img=img,
                        signature_root=signature_root,
                        signature_specs=signature_specs,
                        mask_img=brain_union,
                        signatures=cfg.signatures,
                    )
                    for s in sigs:
                        group_rows.append(
                            {
                                "subject": sub_label,
                                "task": cfg.task,
                                "method": cfg.method,
                                "map_inference": condition_map_inference,
                                "scope": scope,
                                "run": run_label,
                                "run_num": "" if run_num is None else int(run_num),
                                "group_column": str(cfg.signature_group_column or ""),
                                "group_value": str(group),
                                "n_trials": n_trials,
                                "signature": s.name,
                                "dot": f"{s.dot:.8g}",
                                "cosine": "" if s.cosine is None else f"{s.cosine:.8g}",
                                "pearson_r": "" if s.pearson_r is None else f"{s.pearson_r:.8g}",
                                "n_voxels": s.n_voxels,
                                "weights": str(s.weight_path),
                            }
                        )

            _write_tsv(out_dir / "signatures" / "group_signature_expression.tsv", group_rows)

    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return {
        "output_dir": str(out_dir),
        "n_trials": len(trial_infos),
        "n_trial_signature_rows": len(trial_sig_rows),
        "n_condition_signature_rows": len(cond_rows),
    }
