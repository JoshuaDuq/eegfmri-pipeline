from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from fmri_pipeline.analysis.contrast_builder import discover_confounds
from fmri_pipeline.analysis.smoothing import normalize_smoothing_fwhm
from fmri_pipeline.utils.bold_discovery import (
    discover_fmriprep_preproc_bold,
    get_tr_from_bold,
    select_confounds,
    select_consistent_run_source,
)
from fmri_pipeline.utils.text import safe_slug


def load_resting_state_config_section(config: Any) -> dict[str, Any]:
    section = config.get("fmri_resting_state", {})
    if isinstance(section, dict):
        return section
    return {}


@dataclass(frozen=True)
class RestingStateAnalysisConfig:
    input_source: str = "fmriprep"
    fmriprep_space: str = "MNI152NLin2009cAsym"
    require_fmriprep: bool = True
    runs: Optional[list[int]] = None
    confounds_strategy: str = "auto"
    high_pass_hz: Optional[float] = 0.008
    low_pass_hz: Optional[float] = 0.1
    smoothing_fwhm: Optional[float] = None
    atlas_labels_img: Path | str | None = None
    atlas_labels_tsv: Path | str | None = None
    connectivity_kind: str = "correlation"
    standardize: bool = True
    detrend: bool = True

    def normalized(self) -> "RestingStateAnalysisConfig":
        input_source = str(self.input_source or "fmriprep").strip().lower()
        if input_source not in {"fmriprep", "bids_raw"}:
            raise ValueError(
                f"input_source must be 'fmriprep' or 'bids_raw', got {input_source!r}."
            )

        confounds_strategy = str(self.confounds_strategy or "auto").strip().lower()
        if not confounds_strategy:
            confounds_strategy = "auto"

        connectivity_kind = str(self.connectivity_kind or "correlation").strip().lower()
        if connectivity_kind not in {"correlation"}:
            raise ValueError(
                f"connectivity_kind must be 'correlation', got {connectivity_kind!r}."
            )

        atlas_labels_img = _normalize_required_path(
            self.atlas_labels_img,
            key_name="fmri_resting_state.atlas_labels_img",
        )
        atlas_labels_tsv = _normalize_optional_path(self.atlas_labels_tsv)

        high_pass_hz = _normalize_optional_frequency(self.high_pass_hz, "high_pass_hz")
        low_pass_hz = _normalize_optional_frequency(self.low_pass_hz, "low_pass_hz")
        if (
            high_pass_hz is not None
            and low_pass_hz is not None
            and low_pass_hz <= high_pass_hz
        ):
            raise ValueError(
                "low_pass_hz must be greater than high_pass_hz when both are provided."
            )

        runs = None if self.runs is None else [int(run_num) for run_num in self.runs]

        return RestingStateAnalysisConfig(
            input_source=input_source,
            fmriprep_space=str(self.fmriprep_space or "MNI152NLin2009cAsym").strip(),
            require_fmriprep=bool(self.require_fmriprep),
            runs=runs,
            confounds_strategy=confounds_strategy,
            high_pass_hz=high_pass_hz,
            low_pass_hz=low_pass_hz,
            smoothing_fwhm=normalize_smoothing_fwhm(self.smoothing_fwhm),
            atlas_labels_img=atlas_labels_img,
            atlas_labels_tsv=atlas_labels_tsv,
            connectivity_kind=connectivity_kind,
            standardize=bool(self.standardize),
            detrend=bool(self.detrend),
        )


def atlas_output_name(atlas_labels_img: Path | str) -> str:
    atlas_path = Path(atlas_labels_img)
    name = atlas_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = atlas_path.stem
    return safe_slug(name, default="atlas")


def _prepare_confounds_and_sample_mask(
    confounds_df: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[np.ndarray], list[str]]:
    if confounds_df is None or confounds_df.empty:
        return None, None, []

    scrub_columns = [
        column
        for column in confounds_df.columns
        if column.startswith("motion_outlier")
        or column.startswith("non_steady_state_outlier")
        or column.startswith("outlier")
    ]

    sample_mask: Optional[np.ndarray] = None
    if scrub_columns:
        scrub_values = (
            confounds_df[scrub_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        retained_mask = ~(scrub_values > 0).any(axis=1)
        retained_indices = np.flatnonzero(retained_mask)
        if retained_indices.size == 0:
            sample_mask = retained_indices.astype(int, copy=False)
        elif retained_indices.size < confounds_df.shape[0]:
            sample_mask = retained_indices.astype(int, copy=False)

    cleaned_confounds = confounds_df.drop(columns=scrub_columns) if scrub_columns else confounds_df.copy()
    if cleaned_confounds.shape[1] == 0:
        cleaned_confounds = None

    return cleaned_confounds, sample_mask, scrub_columns


def _validate_filter_settings(
    repetition_time: float,
    cfg: RestingStateAnalysisConfig,
) -> None:
    tr = float(repetition_time)
    if not math.isfinite(tr) or tr <= 0:
        raise ValueError(f"TR must be positive and finite, got {repetition_time!r}.")

    nyquist_hz = 0.5 / tr
    for field_name, value in (
        ("high_pass_hz", cfg.high_pass_hz),
        ("low_pass_hz", cfg.low_pass_hz),
    ):
        if value is None:
            continue
        if value >= nyquist_hz:
            raise ValueError(
                f"{field_name}={value} Hz is invalid for TR={tr:.6f}s; it must be below the Nyquist frequency {nyquist_hz:.6f} Hz."
            )


def _validate_roi_timeseries(
    timeseries: np.ndarray,
    roi_labels: list[str],
    *,
    subject: str,
    run_num: int,
) -> None:
    if timeseries.ndim != 2 or timeseries.shape[1] == 0:
        raise ValueError(f"No ROI timeseries were extracted for {subject}, run {run_num}.")
    if timeseries.shape[0] < 2:
        raise ValueError(
            f"Not enough retained frames for {subject}, run {run_num} after confound censoring."
        )
    if not np.isfinite(timeseries).all():
        invalid = np.flatnonzero(~np.isfinite(timeseries).all(axis=0))
        invalid_labels = [roi_labels[index] for index in invalid[:5]]
        raise ValueError(
            f"Non-finite ROI timeseries detected for {subject}, run {run_num}: {invalid_labels}."
        )

    roi_std = np.std(timeseries, axis=0)
    degenerate = np.flatnonzero(~np.isfinite(roi_std) | (roi_std <= 1e-12))
    if degenerate.size > 0:
        degenerate_labels = [roi_labels[index] for index in degenerate[:5]]
        raise ValueError(
            f"Degenerate ROI timeseries detected for {subject}, run {run_num}: {degenerate_labels}. Check atlas/BOLD overlap and censoring."
        )


def _aggregate_connectivity_matrices(
    run_matrices: list[np.ndarray],
    run_weights: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not run_matrices:
        raise ValueError("run_matrices must contain at least one connectivity matrix.")
    if len(run_matrices) != len(run_weights):
        raise ValueError("run_matrices and run_weights must have the same length.")

    weights = np.asarray(run_weights, dtype=float)
    if weights.ndim != 1 or weights.size != len(run_matrices):
        raise ValueError("run_weights must define exactly one weight per connectivity matrix.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("run_weights must be finite and strictly positive.")

    if len(run_matrices) == 1:
        matrix = np.asarray(run_matrices[0], dtype=float)
        return matrix, _fisher_z_matrix(matrix)

    fisher_stack = np.stack([_fisher_z_matrix(matrix) for matrix in run_matrices], axis=0)
    fisher_mean = np.average(fisher_stack, axis=0, weights=weights)
    connectivity = np.tanh(fisher_mean)
    np.fill_diagonal(connectivity, 1.0)
    return connectivity, fisher_mean


def run_resting_state_analysis_for_subject(
    *,
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    deriv_root: Path,
    subject: str,
    task: str,
    cfg: RestingStateAnalysisConfig,
    output_dir: Path,
) -> dict[str, Any]:
    from nilearn.maskers import NiftiLabelsMasker

    normalized_cfg = cfg.normalized()
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_dir = output_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)

    discovered_runs = _discover_rest_runs(
        bids_fmri_root=bids_fmri_root,
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        cfg=normalized_cfg,
    )

    roi_labels: Optional[list[str]] = None
    run_summaries: list[dict[str, Any]] = []
    concatenated_timeseries: list[np.ndarray] = []
    run_connectivity_matrices: list[np.ndarray] = []
    run_connectivity_weights: list[int] = []

    for bold_path, run_num in discovered_runs:
        confounds_df, confound_columns = _load_rest_confounds(
            bids_derivatives=bids_derivatives,
            subject=subject,
            task=task,
            run_num=run_num,
            input_source=normalized_cfg.input_source,
            strategy=normalized_cfg.confounds_strategy,
        )
        repetition_time = get_tr_from_bold(bold_path)
        _validate_filter_settings(repetition_time, normalized_cfg)
        masker_confounds, sample_mask, scrub_columns = _prepare_confounds_and_sample_mask(confounds_df)
        original_n_volumes = int(confounds_df.shape[0]) if confounds_df is not None else None
        if sample_mask is not None and sample_mask.size == 0:
            raise ValueError(
                f"All frames were censored for {sub_label}, run {run_num}."
            )
        masker = NiftiLabelsMasker(
            labels_img=str(normalized_cfg.atlas_labels_img),
            smoothing_fwhm=normalized_cfg.smoothing_fwhm,
            standardize=normalized_cfg.standardize,
            detrend=normalized_cfg.detrend,
            low_pass=normalized_cfg.low_pass_hz,
            high_pass=normalized_cfg.high_pass_hz,
            t_r=float(repetition_time),
        )
        timeseries = masker.fit_transform(
            str(bold_path),
            confounds=masker_confounds,
            sample_mask=sample_mask,
        )
        if roi_labels is None:
            roi_labels = _resolve_roi_labels(
                normalized_cfg.atlas_labels_tsv,
                expected_count=timeseries.shape[1],
            )
        if len(roi_labels) != timeseries.shape[1]:
            raise ValueError(
                "Atlas label count does not match extracted ROI time series count."
            )
        _validate_roi_timeseries(timeseries, roi_labels, subject=sub_label, run_num=run_num)

        frame_df = pd.DataFrame(timeseries, columns=roi_labels)
        frame_df.index.name = "frame"
        frame_df.to_csv(
            timeseries_dir / f"{sub_label}_task-{task}_run-{run_num:02d}_timeseries.tsv",
            sep="\t",
            encoding="utf-8",
        )
        concatenated_timeseries.append(timeseries)
        run_connectivity_matrices.append(_compute_connectivity_matrix(timeseries))
        run_connectivity_weights.append(int(timeseries.shape[0]))
        retained_n_volumes = int(timeseries.shape[0])
        scrubbed_n_volumes = (
            int(original_n_volumes - retained_n_volumes)
            if original_n_volumes is not None
            else 0
        )
        run_summaries.append(
            {
                "run": int(run_num),
                "bold_path": str(bold_path),
                "n_volumes": retained_n_volumes,
                "n_original_volumes": original_n_volumes or retained_n_volumes,
                "n_scrubbed_volumes": scrubbed_n_volumes,
                "tr": float(repetition_time),
                "confounds_columns": confound_columns,
                "scrubbing_columns": scrub_columns,
            }
        )

    if not concatenated_timeseries or roi_labels is None:
        raise ValueError(f"No usable resting-state runs were found for {sub_label}.")

    concatenated = np.vstack(concatenated_timeseries)
    concatenated_df = pd.DataFrame(concatenated, columns=roi_labels)
    concatenated_df.index.name = "frame"
    concatenated_path = timeseries_dir / f"{sub_label}_task-{task}_timeseries_concat.tsv"
    concatenated_df.to_csv(concatenated_path, sep="\t", encoding="utf-8")

    connectivity_matrix, fisher_z_matrix = _aggregate_connectivity_matrices(
        run_connectivity_matrices,
        run_connectivity_weights,
    )
    connectivity_df = pd.DataFrame(
        connectivity_matrix,
        index=roi_labels,
        columns=roi_labels,
    )
    connectivity_path = (
        output_dir / f"{sub_label}_task-{task}_{normalized_cfg.connectivity_kind}_connectivity.tsv"
    )
    connectivity_df.to_csv(connectivity_path, sep="\t", encoding="utf-8")

    fisher_z_path: Optional[Path] = None
    fisher_z_df: Optional[pd.DataFrame] = None
    if normalized_cfg.connectivity_kind == "correlation":
        fisher_z_df = pd.DataFrame(
            fisher_z_matrix,
            index=roi_labels,
            columns=roi_labels,
        )
        fisher_z_path = (
            output_dir / f"{sub_label}_task-{task}_{normalized_cfg.connectivity_kind}_connectivity_fisher_z.tsv"
        )
        fisher_z_df.to_csv(fisher_z_path, sep="\t", encoding="utf-8")

    labels_df = pd.DataFrame(
        {"index": np.arange(1, len(roi_labels) + 1), "label": roi_labels}
    )
    labels_path = output_dir / "roi_labels.tsv"
    labels_df.to_csv(labels_path, sep="\t", index=False, encoding="utf-8")

    payload = {
        "subject": sub_label,
        "task": task,
        "config": _config_payload(normalized_cfg),
        "atlas_output_name": atlas_output_name(normalized_cfg.atlas_labels_img),
        "runs": run_summaries,
        "roi_count": len(roi_labels),
        "n_timepoints": int(concatenated.shape[0]),
        "timeseries_concat_path": str(concatenated_path),
        "connectivity_path": str(connectivity_path),
        "connectivity_fisher_z_path": str(fisher_z_path) if fisher_z_path else None,
        "labels_path": str(labels_path),
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "timeseries_concat_path": str(concatenated_path),
        "connectivity_path": str(connectivity_path),
        "connectivity_fisher_z_path": str(fisher_z_path) if fisher_z_path else None,
        "labels_path": str(labels_path),
        "provenance_path": str(provenance_path),
        "n_runs": len(run_summaries),
        "roi_count": len(roi_labels),
        "n_timepoints": int(concatenated.shape[0]),
    }


def _normalize_optional_frequency(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    frequency = float(value)
    if not math.isfinite(frequency):
        raise ValueError(f"{field_name} must be finite when provided.")
    if frequency <= 0:
        return None
    return frequency


def _discover_rest_runs(
    *,
    bids_fmri_root: Path,
    bids_derivatives: Optional[Path],
    subject: str,
    task: str,
    cfg: RestingStateAnalysisConfig,
) -> list[tuple[Path, int]]:
    sub_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    func_dir = bids_fmri_root / sub_label / "func"
    if not func_dir.exists():
        raise FileNotFoundError(f"fMRI func directory not found: {func_dir}")

    runless_bold_path: Optional[Path] = None
    run_nums: list[int] = []
    for bold_file in sorted(func_dir.glob(f"{sub_label}_task-{task}_run-*_bold.nii.gz")):
        try:
            run_str = bold_file.name.split("_run-")[1].split("_")[0]
            run_nums.append(int(run_str))
        except Exception:
            continue

    if not run_nums:
        runless_candidates = sorted(func_dir.glob(f"{sub_label}_task-{task}_*_bold.nii.gz"))
        runless_candidates += sorted(func_dir.glob(f"{sub_label}_task-{task}_bold.nii.gz"))
        runless_candidates = [
            candidate
            for candidate in runless_candidates
            if "_run-" not in candidate.name
        ]
        runless_candidates = list(dict.fromkeys(runless_candidates))
        if len(runless_candidates) > 1:
            raise FileNotFoundError(
                "Multiple resting-state BOLD files were found without explicit run entities. "
                f"Add BIDS run labels or request specific runs explicitly: {[path.name for path in runless_candidates]}."
            )
        if runless_candidates:
            runless_bold_path = runless_candidates[0]
            run_nums = [1]

    run_nums = sorted(set(run_nums))
    if cfg.runs is not None:
        requested_runs = {int(run_num) for run_num in cfg.runs}
        run_nums = [run_num for run_num in run_nums if run_num in requested_runs]

    if not run_nums:
        raise FileNotFoundError(
            f"No resting-state BOLD runs found for subject {subject}, task {task} in {func_dir}"
        )

    selected_input_source = str(cfg.input_source or "bids_raw").strip().lower()
    preproc_by_run: dict[int, Optional[Path]] = {}
    if bids_derivatives is not None and selected_input_source == "fmriprep":
        selected_input_source, preproc_by_run = select_consistent_run_source(
            run_numbers=run_nums,
            discover_preproc_bold=lambda run_num: discover_fmriprep_preproc_bold(
                bids_derivatives=bids_derivatives,
                subject=subject,
                task=task,
                run_num=run_num,
                space=cfg.fmriprep_space,
            ),
            require_fmriprep=bool(cfg.require_fmriprep),
        )

    discovered: list[tuple[Path, int]] = []
    for run_num in run_nums:
        if selected_input_source == "fmriprep":
            bold_path = preproc_by_run.get(int(run_num))
        else:
            bold_path = None
            if runless_bold_path is not None and int(run_num) == 1:
                bold_path = runless_bold_path
            for pattern in (
                f"{sub_label}_task-{task}_run-{run_num:02d}_bold.nii.gz",
                f"{sub_label}_task-{task}_run-{run_num}_bold.nii.gz",
            ):
                if bold_path is not None:
                    break
                candidate = func_dir / pattern
                if candidate.exists():
                    bold_path = candidate
                    break
        if bold_path is None:
            continue
        discovered.append((bold_path, int(run_num)))

    if cfg.runs is not None:
        requested_runs = {int(run_num) for run_num in cfg.runs}
        discovered_runs = {int(run_num) for _bold_path, run_num in discovered}
        missing_runs = sorted(requested_runs - discovered_runs)
        if missing_runs:
            raise FileNotFoundError(
                "Some requested resting-state runs could not be resolved to BOLD inputs: "
                f"{missing_runs}."
            )

    if not discovered:
        raise FileNotFoundError(
            f"No usable resting-state BOLD runs found for subject {subject}, task {task}."
        )
    return discovered


def _normalize_optional_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalize_required_path(value: Any, *, key_name: str) -> Path:
    path = _normalize_optional_path(value)
    if path is None:
        raise ValueError(f"Missing required config value: {key_name}")
    return path


def _load_rest_confounds(
    *,
    bids_derivatives: Optional[Path],
    subject: str,
    task: str,
    run_num: int,
    input_source: str,
    strategy: str,
) -> tuple[Optional[pd.DataFrame], list[str]]:
    if strategy == "none":
        return None, []
    if bids_derivatives is None:
        raise ValueError(
            "Resting-state confound regression requires derivatives inputs or confounds_strategy='none'."
        )
    confounds_path = discover_confounds(
        bids_derivatives=bids_derivatives,
        subject=subject,
        task=task,
        run_num=run_num,
    )
    if confounds_path is None:
        if input_source == "fmriprep":
            raise FileNotFoundError(
                f"Missing fMRIPrep confounds file for resting-state run {run_num}."
            )
        raise ValueError(
            "Resting-state confound regression requires fMRIPrep confounds. Use confounds_strategy='none' when working from raw BIDS inputs."
        )
    confounds_df, columns = select_confounds(confounds_path, strategy)
    return confounds_df, columns


def _resolve_roi_labels(
    labels_tsv: Optional[Path],
    *,
    expected_count: int,
) -> list[str]:
    if labels_tsv is None:
        return [f"roi_{index:03d}" for index in range(1, expected_count + 1)]

    labels_df = pd.read_csv(labels_tsv, sep="\t")
    name_column = None
    for candidate in ("name", "label", "region", "roi"):
        if candidate in labels_df.columns:
            name_column = candidate
            break
    if name_column is None:
        raise ValueError(
            f"Atlas labels TSV must contain one of: name, label, region, roi ({labels_tsv})."
        )

    if "index" in labels_df.columns:
        ordered = labels_df.copy()
        ordered = ordered[pd.to_numeric(ordered["index"], errors="coerce").notna()]
        ordered["index"] = ordered["index"].astype(int)
        ordered = ordered[ordered["index"] > 0].sort_values("index")
        labels = [str(value).strip() for value in ordered[name_column].tolist() if str(value).strip()]
    else:
        labels = [str(value).strip() for value in labels_df[name_column].tolist() if str(value).strip()]

    if len(labels) != expected_count:
        raise ValueError(
            f"Atlas labels TSV defines {len(labels)} ROI labels but extracted data has {expected_count} columns."
        )
    return labels


def _compute_connectivity_matrix(timeseries: np.ndarray) -> np.ndarray:
    if timeseries.ndim != 2 or timeseries.shape[1] == 0:
        raise ValueError("timeseries must be a 2D array with at least one ROI column.")
    if timeseries.shape[0] < 2:
        raise ValueError("timeseries must contain at least two retained frames.")
    if not np.isfinite(timeseries).all():
        raise ValueError("timeseries contains non-finite values.")
    if timeseries.shape[1] == 1:
        return np.array([[1.0]], dtype=float)
    matrix = np.corrcoef(timeseries, rowvar=False)
    if not np.isfinite(matrix).all():
        raise ValueError("connectivity matrix contains non-finite values.")
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _fisher_z_matrix(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(matrix, dtype=float), -0.999999, 0.999999)
    fisher = np.arctanh(clipped)
    np.fill_diagonal(fisher, 0.0)
    return fisher


def _config_payload(cfg: RestingStateAnalysisConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["atlas_labels_img"] = str(cfg.atlas_labels_img)
    payload["atlas_labels_tsv"] = str(cfg.atlas_labels_tsv) if cfg.atlas_labels_tsv else None
    return payload
