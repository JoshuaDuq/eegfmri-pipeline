"""Mono-trial fMRI diagnostics for EEG-BOLD coupling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from eeg_pipeline.utils.config.loader import get_config_value
from fmri_pipeline.analysis.trial_signatures import (
    TrialSignatureExtractionConfig,
    _build_lss_events,
    _discover_runs,
    _extract_trials_for_run,
)
from fmri_pipeline.utils.bold_discovery import get_tr_from_bold, select_confounds


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


@dataclass(frozen=True)
class FMRIDesignQCConfig:
    max_abs_target_other_correlation: Optional[float]
    min_target_efficiency: Optional[float]
    max_target_variance: Optional[float]


@dataclass(frozen=True)
class FMRIRoiQCConfig:
    outlier_mad_threshold: float
    max_outlier_proportion: Optional[float]


@dataclass(frozen=True)
class FMRIQCConfig:
    enabled: bool
    design: FMRIDesignQCConfig
    roi: FMRIRoiQCConfig

    @classmethod
    def from_config(cls, config: Any) -> "FMRIQCConfig":
        raw = _require_mapping(
            get_config_value(config, "eeg_bold_coupling.fmri_qc", {}),
            path="eeg_bold_coupling.fmri_qc",
        )
        design_raw = _require_mapping(
            raw.get("design", {}),
            path="eeg_bold_coupling.fmri_qc.design",
        )
        roi_raw = _require_mapping(
            raw.get("roi", {}),
            path="eeg_bold_coupling.fmri_qc.roi",
        )
        return cls(
            enabled=bool(raw.get("enabled", True)),
            design=FMRIDesignQCConfig(
                max_abs_target_other_correlation=(
                    None
                    if design_raw.get("max_abs_target_other_correlation", None) in {None, ""}
                    else float(design_raw.get("max_abs_target_other_correlation"))
                ),
                min_target_efficiency=(
                    None
                    if design_raw.get("min_target_efficiency", None) in {None, ""}
                    else float(design_raw.get("min_target_efficiency"))
                ),
                max_target_variance=(
                    None
                    if design_raw.get("max_target_variance", None) in {None, ""}
                    else float(design_raw.get("max_target_variance"))
                ),
            ),
            roi=FMRIRoiQCConfig(
                outlier_mad_threshold=float(
                    roi_raw.get("outlier_mad_threshold", 5.0)
                ),
                max_outlier_proportion=(
                    None
                    if roi_raw.get("max_outlier_proportion", None) in {None, ""}
                    else float(roi_raw.get("max_outlier_proportion"))
                ),
            ),
        )


@dataclass(frozen=True)
class SubjectFMRIQC:
    design_table: pd.DataFrame
    roi_table: pd.DataFrame
    summary: Dict[str, Any]
    global_failure: bool
    non_interpretable_rois: Set[str]


def _build_confounds_matrix(
    *,
    confounds_path: Optional[Path],
    strategy: str,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    confounds_df, confound_columns = select_confounds(confounds_path, strategy)
    if confounds_df is None:
        return None, None
    values = confounds_df.to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("Confounds matrix must be 2-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Confounds matrix contains non-finite values: {confounds_path}")
    return values, confound_columns


def _other_trial_columns(columns: Sequence[str]) -> List[str]:
    return [
        str(column)
        for column in columns
        if str(column).startswith("other_") or str(column) == "other_trials"
    ]


def _max_abs_correlation(target: np.ndarray, others: np.ndarray) -> float:
    if others.ndim != 2 or others.shape[1] == 0:
        return 0.0
    values: List[float] = []
    for idx in range(others.shape[1]):
        other = others[:, idx]
        if np.std(target, ddof=0) <= 0 or np.std(other, ddof=0) <= 0:
            values.append(0.0)
            continue
        corr = np.corrcoef(target, other)[0, 1]
        values.append(abs(float(corr)))
    return float(max(values)) if values else 0.0


def _design_qc_table(
    *,
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    subject: str,
    task: str,
    lss_cfg: TrialSignatureExtractionConfig,
) -> pd.DataFrame:
    lss_cfg = lss_cfg.normalized()
    runs = _discover_runs(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        runs=lss_cfg.runs,
        input_source=lss_cfg.input_source,
        fmriprep_space=lss_cfg.fmriprep_space,
        require_fmriprep=lss_cfg.require_fmriprep,
    )

    rows: List[Dict[str, Any]] = []
    for run_num, bold_path, events_path, confounds_path in runs:
        events_df = pd.read_csv(events_path, sep="\t")
        trials, _modeled_events = _extract_trials_for_run(
            events_df=events_df,
            cfg=lss_cfg,
            events_path=events_path,
            run_num=run_num,
        )
        if not trials:
            continue

        confounds_matrix, confound_columns = _build_confounds_matrix(
            confounds_path=confounds_path,
            strategy=lss_cfg.confounds_strategy,
        )
        tr = float(get_tr_from_bold(bold_path))
        if confounds_matrix is not None:
            n_scans = int(confounds_matrix.shape[0])
        else:
            import nibabel as nib

            n_scans = int(nib.load(str(bold_path)).shape[3])
        frame_times = np.arange(n_scans, dtype=float) * tr

        for trial in trials:
            lss_events = _build_lss_events(
                trial=trial,
                all_trials=trials,
                original_events_df=events_df,
                cfg=lss_cfg,
            )
            design = make_first_level_design_matrix(
                frame_times=frame_times,
                events=lss_events,
                hrf_model=lss_cfg.hrf_model,
                drift_model=lss_cfg.drift_model,
                high_pass=lss_cfg.high_pass_hz,
                add_regs=confounds_matrix,
                add_reg_names=confound_columns,
            )
            if "target" not in design.columns:
                raise ValueError(
                    f"LSS design matrix is missing the target regressor for {events_path}."
                )
            values = design.to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Design matrix contains non-finite values for {events_path}.")
            target = design["target"].to_numpy(dtype=float)
            other_columns = _other_trial_columns(list(design.columns))
            other_matrix = design[other_columns].to_numpy(dtype=float) if other_columns else np.empty((len(design), 0), dtype=float)
            rank = int(np.linalg.matrix_rank(values))
            residual_dof = int(values.shape[0] - rank)
            rank_deficient = bool(rank < values.shape[1])
            target_variance = np.nan
            target_efficiency = np.nan
            if not rank_deficient:
                target_idx = list(design.columns).index("target")
                xtx_inv = np.linalg.inv(values.T @ values)
                target_variance = float(xtx_inv[target_idx, target_idx])
                if target_variance > 0:
                    target_efficiency = float(1.0 / target_variance)
            rows.append(
                {
                    "run_num": int(run_num),
                    "trial_index": int(trial.trial_index),
                    "trial_key": f"{int(run_num)}|{float(trial.onset):.3f}|{float(trial.duration):.3f}",
                    "target_other_corr_max_abs": _max_abs_correlation(target, other_matrix),
                    "target_variance": target_variance,
                    "target_efficiency": target_efficiency,
                    "rank": rank,
                    "n_regressors": int(values.shape[1]),
                    "n_frames": int(values.shape[0]),
                    "residual_dof": residual_dof,
                    "rank_deficient": rank_deficient,
                }
            )
    return pd.DataFrame(rows)


def _roi_outlier_table(
    *,
    bold_table: pd.DataFrame,
    cfg: FMRIQCConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    roi_columns = [
        str(column)
        for column in bold_table.columns
        if str(column).startswith("bold_")
    ]
    for bold_column in roi_columns:
        roi = bold_column.replace("bold_", "", 1)
        values = pd.to_numeric(bold_table[bold_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"ROI BOLD values contain non-finite values for {roi}.")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            outlier_mask = np.zeros(values.shape[0], dtype=bool)
        else:
            cutoff = cfg.roi.outlier_mad_threshold * scale
            outlier_mask = np.abs(values - median) > cutoff
        rows.append(
            {
                "roi": roi,
                "median_beta": median,
                "mad_scale": scale,
                "n_trials": int(values.size),
                "n_outliers": int(outlier_mask.sum()),
                "outlier_proportion": float(outlier_mask.mean()) if values.size > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _summarize_failures(
    *,
    design_table: pd.DataFrame,
    roi_table: pd.DataFrame,
    cfg: FMRIQCConfig,
) -> Tuple[bool, Set[str], Dict[str, Any]]:
    global_failure = False
    non_interpretable_rois: Set[str] = set()

    if not design_table.empty:
        if bool(design_table["rank_deficient"].any()):
            global_failure = True
        if cfg.design.max_abs_target_other_correlation is not None:
            if bool(
                (
                    pd.to_numeric(design_table["target_other_corr_max_abs"], errors="coerce")
                    > float(cfg.design.max_abs_target_other_correlation)
                ).any()
            ):
                global_failure = True
        if cfg.design.min_target_efficiency is not None:
            if bool(
                (
                    pd.to_numeric(design_table["target_efficiency"], errors="coerce")
                    < float(cfg.design.min_target_efficiency)
                ).any()
            ):
                global_failure = True
        if cfg.design.max_target_variance is not None:
            if bool(
                (
                    pd.to_numeric(design_table["target_variance"], errors="coerce")
                    > float(cfg.design.max_target_variance)
                ).any()
            ):
                global_failure = True

    if not roi_table.empty and cfg.roi.max_outlier_proportion is not None:
        flagged = roi_table.loc[
            pd.to_numeric(roi_table["outlier_proportion"], errors="coerce")
            > float(cfg.roi.max_outlier_proportion),
            "roi",
        ]
        non_interpretable_rois.update(str(value) for value in flagged.tolist())

    summary = {
        "fmri_qc_config": asdict(cfg),
        "global_failure": bool(global_failure),
        "non_interpretable_rois": sorted(non_interpretable_rois),
        "n_design_rows": int(len(design_table)),
        "n_roi_rows": int(len(roi_table)),
    }
    if not design_table.empty:
        summary["max_abs_target_other_corr"] = float(
            pd.to_numeric(
                design_table["target_other_corr_max_abs"],
                errors="coerce",
            ).max()
        )
        summary["min_target_efficiency"] = float(
            pd.to_numeric(design_table["target_efficiency"], errors="coerce").min()
        )
        summary["max_target_variance"] = float(
            pd.to_numeric(design_table["target_variance"], errors="coerce").max()
        )
    return global_failure, non_interpretable_rois, summary


def run_subject_fmri_qc(
    *,
    subject: str,
    task: str,
    deriv_root: Path,
    config: Any,
    lss_cfg: TrialSignatureExtractionConfig,
    bold_table: pd.DataFrame,
) -> SubjectFMRIQC:
    cfg = FMRIQCConfig.from_config(config)
    if not cfg.enabled:
        return SubjectFMRIQC(
            design_table=pd.DataFrame(),
            roi_table=pd.DataFrame(),
            summary={"fmri_qc_config": asdict(cfg), "enabled": False},
            global_failure=False,
            non_interpretable_rois=set(),
        )

    bids_fmri_root_raw = str(get_config_value(config, "paths.bids_fmri_root", "")).strip()
    if not bids_fmri_root_raw:
        raise ValueError("paths.bids_fmri_root is required for fMRI QC.")
    bids_fmri_root = Path(bids_fmri_root_raw).expanduser().resolve()
    bids_derivatives = deriv_root
    design_table = _design_qc_table(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        lss_cfg=lss_cfg,
    )
    roi_table = _roi_outlier_table(
        bold_table=bold_table,
        cfg=cfg,
    )
    global_failure, non_interpretable_rois, summary = _summarize_failures(
        design_table=design_table,
        roi_table=roi_table,
        cfg=cfg,
    )
    summary["subject"] = subject
    summary["task"] = task
    return SubjectFMRIQC(
        design_table=design_table,
        roi_table=roi_table,
        summary=summary,
        global_failure=global_failure,
        non_interpretable_rois=non_interpretable_rois,
    )


def apply_fmri_qc_to_subject_results(
    *,
    subject_results: pd.DataFrame,
    qc_result: SubjectFMRIQC,
) -> pd.DataFrame:
    out = subject_results.copy()
    if out.empty:
        return out
    if qc_result.global_failure:
        out["status"] = "fmri_qc_failed"
        out["r"] = np.nan
        out["p_value"] = np.nan
        return out
    if not qc_result.non_interpretable_rois:
        return out
    mask = out["roi"].astype(str).isin(sorted(qc_result.non_interpretable_rois))
    out.loc[mask, "status"] = "fmri_qc_failed"
    out.loc[mask, "r"] = np.nan
    out.loc[mask, "p_value"] = np.nan
    return out


def write_subject_fmri_qc(
    *,
    output_dir: Path,
    qc_result: SubjectFMRIQC,
) -> None:
    qc_result.design_table.to_csv(output_dir / "fmri_design_qc.tsv", sep="\t", index=False)
    qc_result.roi_table.to_csv(output_dir / "fmri_roi_qc.tsv", sep="\t", index=False)
    (output_dir / "fmri_qc_summary.json").write_text(
        json.dumps(qc_result.summary, indent=2, default=str),
        encoding="utf-8",
    )


__all__ = [
    "apply_fmri_qc_to_subject_results",
    "run_subject_fmri_qc",
    "write_subject_fmri_qc",
]
