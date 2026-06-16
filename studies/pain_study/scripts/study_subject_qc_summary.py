#!/usr/bin/env python
"""Write subject-level QC summaries for Pain Study 1 and Study 2 outputs."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEMPORAL_WINDOWS = (
    "prestimulus_wide",
    "immediate_prestimulus",
    "ramp_up",
    "early_plateau",
    "mid_plateau",
    "late_plateau",
)
TEMPORAL_MODELS = ("elasticnet", "ridge")
SOURCE_BANDS = ("alpha", "beta", "gamma")
EXPECTED_STUDY1_RUNS = tuple(range(1, 7))
EXPECTED_STUDY1_TRIALS_PER_RUN = 11


@dataclass(frozen=True)
class SourcePowerShape:
    rows: int
    vertices: int


@dataclass(frozen=True)
class SubjectQcInputs:
    subjects: tuple[str, ...]
    study1_targets: pd.DataFrame
    primary_model: pd.DataFrame
    gamma_model: pd.DataFrame
    temporal_models: dict[tuple[str, str], pd.DataFrame]
    study2_source_qc: dict[str, pd.DataFrame]
    study2_source_input: pd.DataFrame
    source_power_shapes: dict[str, SourcePowerShape]
    anatomy_status: dict[str, str]


@dataclass(frozen=True)
class SubjectQcSummary:
    subject_rows: list[dict[str, object]]
    temporal_rows: list[dict[str, object]]
    mask_rows: list[dict[str, object]]
    completeness_rows: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-file", required=True)
    parser.add_argument("--study1-root", required=True)
    parser.add_argument("--study2-root", required=True)
    parser.add_argument("--subjects-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_subject_qc(load_inputs(args))
    write_qc_outputs(summary, Path(args.output_dir))


def load_inputs(args: argparse.Namespace) -> SubjectQcInputs:
    study1_root = Path(args.study1_root)
    study2_root = Path(args.study2_root)
    subjects = load_subjects(Path(args.subjects_file))
    return SubjectQcInputs(
        subjects=tuple(subjects),
        study1_targets=read_parquet(study1_root / "targets" / "primary_targets.parquet"),
        primary_model=read_tsv(
            study1_root
            / "feature_benchmark"
            / "primary"
            / "NPS"
            / "alpha_beta_gamma"
            / "model_comparison"
            / "model_comparison.tsv"
        ),
        gamma_model=read_tsv(
            study1_root
            / "feature_benchmark"
            / "primary"
            / "NPS"
            / "gamma"
            / "model_comparison"
            / "model_comparison.tsv"
        ),
        temporal_models=read_temporal_models(study1_root),
        study2_source_qc=read_source_qc(study2_root),
        study2_source_input=read_tsv(study2_root / "source_stage" / "source_stage_input.tsv"),
        source_power_shapes=read_source_power_shapes(study2_root),
        anatomy_status=read_anatomy_status(Path(args.subjects_dir), subjects),
    )


def load_subjects(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Subject list not found: {path}")
    subjects = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not subjects:
        raise ValueError(f"Subject list is empty: {path}")
    return subjects


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required parquet file not found: {path}")
    return pd.read_parquet(path)


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required TSV file not found: {path}")
    return pd.read_csv(path, sep="\t")


def read_temporal_models(study1_root: Path) -> dict[tuple[str, str], pd.DataFrame]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for model in TEMPORAL_MODELS:
        for window in TEMPORAL_WINDOWS:
            path = (
                study1_root
                / "feature_benchmark"
                / "temporal_control"
                / "NPS"
                / f"temporal_{window}"
                / "model_comparison"
                / "model_comparison.tsv"
            )
            frame = read_tsv(path)
            require_columns(
                frame,
                {"model", "test_subject", "delta_r2"},
                table_name=f"Temporal {model} {window} table",
            )
            frames[(model, window)] = frame.loc[frame["model"] == model].copy()
    return frames


def read_source_qc(study2_root: Path) -> dict[str, pd.DataFrame]:
    return {
        band: read_tsv(study2_root / "source_stage" / f"qc_{band}.tsv")
        for band in SOURCE_BANDS
    }


def read_source_power_shapes(study2_root: Path) -> dict[str, SourcePowerShape]:
    shapes: dict[str, SourcePowerShape] = {}
    for subject_dir in sorted(study2_root.glob("sub-*/eeg/source")):
        subject_id = subject_dir.parts[-3]
        band_shapes = [
            np.load(subject_dir / f"source_power_{band}.npy", mmap_mode="r").shape
            for band in SOURCE_BANDS
        ]
        if len(set(band_shapes)) != 1:
            raise ValueError(f"Source-power shapes differ across bands for {subject_id}: {band_shapes}")
        rows, vertices = band_shapes[0]
        shapes[subject_id] = SourcePowerShape(rows=int(rows), vertices=int(vertices))
    return shapes


def read_anatomy_status(subjects_dir: Path, subjects: Iterable[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for subject_id in subjects:
        trans = subjects_dir / subject_id / "bem" / f"{subject_id}-trans.fif"
        bem = subjects_dir / subject_id / "bem" / f"{subject_id}-5120-5120-5120-bem-sol.fif"
        status[subject_id] = _anatomy_label(trans.exists(), bem.exists())
    return status


def _anatomy_label(has_trans: bool, has_bem: bool) -> str:
    if has_trans and has_bem:
        return "trans+BEM"
    if has_trans:
        return "trans_only"
    if has_bem:
        return "BEM_only"
    return "missing"


def build_subject_qc(inputs: SubjectQcInputs) -> SubjectQcSummary:
    validate_inputs(inputs)
    subject_rows = [
        build_subject_row(subject_id, inputs)
        for subject_id in inputs.subjects
    ]
    temporal_rows = build_temporal_rows(inputs)
    mask_rows = build_mask_rows(inputs.study1_targets)
    completeness_rows = build_completeness_rows(inputs)
    return SubjectQcSummary(
        subject_rows=subject_rows,
        temporal_rows=temporal_rows,
        mask_rows=mask_rows,
        completeness_rows=completeness_rows,
    )


def validate_inputs(inputs: SubjectQcInputs) -> None:
    require_columns(
        inputs.study1_targets,
        {
            "subject_id",
            "block",
            "acquisition_run",
            "onset",
            "NPS",
            "SIIPS1",
            "NPS_fmri_n_voxels",
            "NPS_fmri_scoring_mask_sha256",
            "SIIPS1_fmri_n_voxels",
            "SIIPS1_fmri_scoring_mask_sha256",
            "hrf_weighted_framewise_displacement",
            "hrf_weighted_std_dvars",
            "hrf_weighted_fp1_fp2_high_frequency_power",
            "residual_ecg_coupling",
            "stimulus_temp",
            "selected_surface",
        },
        table_name="Study 1 target table",
    )
    require_model_columns(inputs.primary_model, table_name="Primary model table")
    require_model_columns(inputs.gamma_model, table_name="Gamma model table")
    require_columns(
        inputs.study2_source_input,
        {"subject_id", "block", "acquisition_run"},
        table_name="Study 2 source-stage input",
    )
    for band, frame in inputs.study2_source_qc.items():
        require_columns(
            frame,
            {
                "subject_id",
                "eligible",
                "retained_trials",
                "valid_blocks",
                "design_rank",
                "residual_degrees_of_freedom",
                "condition_number",
                "reason",
            },
            table_name=f"Study 2 {band} QC table",
        )
    for (model, window), frame in inputs.temporal_models.items():
        require_columns(
            frame,
            {"model", "test_subject", "delta_r2"},
            table_name=f"Temporal {model} {window} table",
        )


def require_model_columns(frame: pd.DataFrame, *, table_name: str) -> None:
    require_columns(
        frame,
        {
            "model",
            "test_subject",
            "r2",
            "r2_nuisance",
            "delta_r2",
            "mae",
            "mae_nuisance",
        },
        table_name=table_name,
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    table_name: str = "table",
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required column(s): {missing}")


def build_subject_row(subject_id: str, inputs: SubjectQcInputs) -> dict[str, object]:
    target_rows = inputs.study1_targets.loc[inputs.study1_targets["subject_id"] == subject_id]
    source_input = inputs.study2_source_input.loc[
        inputs.study2_source_input["subject_id"] == subject_id
    ]
    source_qc_rows = source_qc_rows_for(subject_id, inputs.study2_source_qc)
    source_shape = inputs.source_power_shapes.get(subject_id)

    row = {
        "subject_id": subject_id,
        **study1_metrics(subject_id, target_rows, inputs),
        **study2_metrics(subject_id, source_input, source_qc_rows, source_shape, inputs),
    }
    row["overall_flag"] = combine_flags(
        row["study1_flag"],
        row["temporal_flag"],
        row["study2_flag"],
    )
    return row


def study1_metrics(
    subject_id: str,
    target_rows: pd.DataFrame,
    inputs: SubjectQcInputs,
) -> dict[str, object]:
    if target_rows.empty:
        return {
            "study1_flag": "FAIL",
            "study1_note": "No retained Study 1 target rows.",
            "study1_retained_trials": 0,
            "study1_runs": 0,
            "study1_blocks": 0,
            "study1_trials_by_run": "",
            "study1_trials_by_block": "",
            "study1_missing_runs": ",".join(str(run) for run in EXPECTED_STUDY1_RUNS),
            "study1_incomplete_runs": "",
            "study1_stimulus_temperatures": 0,
            "study1_selected_surfaces": 0,
            "mean_fd": "",
            "mean_std_dvars": "",
            "mean_residual_ecg_coupling": "",
            "nps_temp_r": "",
            "siips1_temp_r": "",
            "primary_elasticnet_delta_r2": "",
            "primary_elasticnet_r2": "",
            "gamma_elasticnet_delta_r2": "",
            "temporal_flag": "FAIL",
            "temporal_note": "No retained Study 1 target rows.",
        }

    primary = model_subject_row(inputs.primary_model, subject_id, "elasticnet")
    gamma = model_subject_row(inputs.gamma_model, subject_id, "elasticnet")
    nps_temp_r = correlation(target_rows["NPS"], target_rows["stimulus_temp"])
    siips1_temp_r = correlation(target_rows["SIIPS1"], target_rows["stimulus_temp"])
    missing_runs = missing_run_summary(target_rows["acquisition_run"])
    incomplete_runs = incomplete_run_summary(target_rows["acquisition_run"])
    study1_flag = study1_flag_for(target_rows, primary, nps_temp_r)
    if (missing_runs or incomplete_runs) and study1_flag == "PASS":
        study1_flag = "WARNING"
    temporal_flag, temporal_note = temporal_flag_for(subject_id, inputs.temporal_models)

    return {
        "study1_flag": study1_flag,
        "study1_note": study1_note_for(
            study1_flag,
            primary,
            nps_temp_r,
            missing_runs,
            incomplete_runs,
        ),
        "study1_retained_trials": len(target_rows),
        "study1_runs": target_rows["acquisition_run"].nunique(),
        "study1_blocks": target_rows["block"].nunique(),
        "study1_trials_by_run": count_summary(target_rows["acquisition_run"]),
        "study1_trials_by_block": count_summary(target_rows["block"]),
        "study1_missing_runs": missing_runs,
        "study1_incomplete_runs": incomplete_runs,
        "study1_stimulus_temperatures": target_rows["stimulus_temp"].nunique(),
        "study1_selected_surfaces": target_rows["selected_surface"].nunique(),
        "mean_fd": rounded_mean(target_rows["hrf_weighted_framewise_displacement"]),
        "mean_std_dvars": rounded_mean(target_rows["hrf_weighted_std_dvars"]),
        "mean_residual_ecg_coupling": rounded_mean(target_rows["residual_ecg_coupling"]),
        "nps_temp_r": rounded(nps_temp_r),
        "siips1_temp_r": rounded(siips1_temp_r),
        "primary_elasticnet_delta_r2": rounded(primary["delta_r2"]),
        "primary_elasticnet_r2": rounded(primary["r2"]),
        "gamma_elasticnet_delta_r2": rounded(gamma["delta_r2"]),
        "temporal_flag": temporal_flag,
        "temporal_note": temporal_note,
    }


def study1_flag_for(
    target_rows: pd.DataFrame,
    primary: pd.Series,
    nps_temp_r: float,
) -> str:
    if len(target_rows) < 25 or target_rows["block"].nunique() < 3:
        return "FAIL"
    if target_rows["NPS_fmri_scoring_mask_sha256"].nunique() != 1:
        return "FAIL"
    if target_rows["SIIPS1_fmri_scoring_mask_sha256"].nunique() != 1:
        return "FAIL"
    if not math.isfinite(nps_temp_r) or nps_temp_r <= 0:
        return "WARNING"
    if float(primary["delta_r2"]) < 0:
        return "WARNING"
    return "PASS"


def study1_note_for(
    flag: str,
    primary: pd.Series,
    nps_temp_r: float,
    missing_runs: str,
    incomplete_runs: str,
) -> str:
    if flag == "PASS":
        return "Retained target rows, fixed masks, positive NPS-temperature relation."
    if missing_runs:
        return f"Missing retained Study 1 run(s): {missing_runs}."
    if incomplete_runs:
        return f"Incomplete retained Study 1 run(s): {incomplete_runs}."
    if float(primary["delta_r2"]) < 0:
        return "Primary EEG model does not improve over nuisance for held-out subject."
    if not math.isfinite(nps_temp_r) or nps_temp_r <= 0:
        return "NPS-temperature relation is absent or negative."
    return "Study 1 target validity warning."


def study2_metrics(
    subject_id: str,
    source_input: pd.DataFrame,
    source_qc_rows: list[pd.Series],
    source_shape: SourcePowerShape | None,
    inputs: SubjectQcInputs,
) -> dict[str, object]:
    if not source_qc_rows:
        return {
            "study2_flag": "FAIL",
            "study2_note": "No source-stage QC row.",
            "study2_eligible": "",
            "study2_retained_trials": 0,
            "study2_valid_blocks": 0,
            "study2_design_rank": "",
            "study2_residual_df": "",
            "study2_condition_number": "",
            "source_input_rows": len(source_input),
            "source_power_rows": source_shape.rows if source_shape else "",
            "source_vertices": source_shape.vertices if source_shape else "",
            "source_anatomy": inputs.anatomy_status.get(subject_id, "missing"),
        }

    if source_qc_bands_disagree(source_qc_rows):
        source_qc = source_qc_rows[0]
        return {
            "study2_flag": "FAIL",
            "study2_note": "Source-stage QC differs across bands.",
            "study2_eligible": "",
            "study2_retained_trials": int(source_qc["retained_trials"]),
            "study2_valid_blocks": int(source_qc["valid_blocks"]),
            "study2_design_rank": int(source_qc["design_rank"]),
            "study2_residual_df": int(source_qc["residual_degrees_of_freedom"]),
            "study2_condition_number": source_qc["condition_number"],
            "source_input_rows": len(source_input),
            "source_power_rows": source_shape.rows if source_shape else "",
            "source_vertices": source_shape.vertices if source_shape else "",
            "source_anatomy": inputs.anatomy_status.get(subject_id, "missing"),
        }

    source_qc = source_qc_rows[0]
    eligible = bool(source_qc["eligible"])
    anatomy = inputs.anatomy_status.get(subject_id, "missing")
    input_rows = len(source_input)
    source_rows = source_shape.rows if source_shape else None
    study2_flag = study2_flag_for(
        eligible=eligible,
        anatomy=anatomy,
        input_rows=input_rows,
        source_rows=source_rows,
    )
    return {
        "study2_flag": study2_flag,
        "study2_note": study2_note_for(study2_flag, source_qc, input_rows, source_rows),
        "study2_eligible": eligible,
        "study2_retained_trials": int(source_qc["retained_trials"]),
        "study2_valid_blocks": int(source_qc["valid_blocks"]),
        "study2_design_rank": int(source_qc["design_rank"]),
        "study2_residual_df": int(source_qc["residual_degrees_of_freedom"]),
        "study2_condition_number": source_qc["condition_number"],
        "source_input_rows": input_rows,
        "source_power_rows": source_rows if source_rows is not None else "",
        "source_vertices": source_shape.vertices if source_shape else "",
        "source_anatomy": anatomy,
    }


def study2_flag_for(
    *,
    eligible: bool,
    anatomy: str,
    input_rows: int,
    source_rows: int | None,
) -> str:
    if not eligible:
        return "FAIL"
    if anatomy != "trans+BEM" or source_rows is None:
        return "FAIL"
    if source_rows != input_rows:
        return "WARNING"
    return "PASS"


def study2_note_for(
    flag: str,
    source_qc: pd.Series,
    input_rows: int,
    source_rows: int | None,
) -> str:
    raw_reason = source_qc.get("reason", "")
    reason = "" if pd.isna(raw_reason) else str(raw_reason).strip()
    if flag == "PASS":
        return "Source-stage QC passed; source input and source arrays match."
    if source_rows != input_rows:
        return "Source-power rows do not match retained source-stage input rows."
    if reason:
        return reason
    return "Study 2 source validity failure."


def source_qc_rows_for(
    subject_id: str,
    qc_by_band: dict[str, pd.DataFrame],
) -> list[pd.Series]:
    matches: list[pd.Series] = []
    for band in SOURCE_BANDS:
        frame = qc_by_band.get(band)
        if frame is None:
            continue
        rows = frame.loc[frame["subject_id"] == subject_id]
        if not rows.empty:
            matches.append(rows.iloc[0])
    return matches


def source_qc_bands_disagree(rows: list[pd.Series]) -> bool:
    if len(rows) <= 1:
        return False
    fields = (
        "eligible",
        "retained_trials",
        "valid_blocks",
        "design_rank",
        "residual_degrees_of_freedom",
        "condition_number",
    )
    reference = tuple(rows[0][field] for field in fields)
    return any(tuple(row[field] for field in fields) != reference for row in rows[1:])


def build_mask_rows(targets: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for signature in ("NPS", "SIIPS1"):
        hashes = sorted(str(value) for value in targets[f"{signature}_fmri_scoring_mask_sha256"].dropna().unique())
        voxels = sorted(targets[f"{signature}_fmri_n_voxels"].dropna().unique())
        rows.append(
            {
                "signature": signature,
                "mask_hash_count": len(hashes),
                "mask_sha256": ",".join(hashes),
                "mask_voxel_count": ",".join(str(int(value)) for value in voxels),
                "flag": "PASS" if len(hashes) == 1 and len(voxels) == 1 else "FAIL",
            }
        )
    return rows


def build_temporal_rows(inputs: SubjectQcInputs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject_id in inputs.subjects:
        for model in TEMPORAL_MODELS:
            row: dict[str, object] = {"subject_id": subject_id, "model": model}
            for window in TEMPORAL_WINDOWS:
                frame = inputs.temporal_models.get((model, window))
                row[window] = temporal_delta(frame, subject_id)
            row["temporal_flag"], row["temporal_note"] = temporal_flag_from_row(row)
            rows.append(row)
    return rows


def build_completeness_rows(inputs: SubjectQcInputs) -> list[dict[str, object]]:
    study1_columns = set(inputs.study1_targets.columns)
    return [
        {
            "qc_category": "study1_retained_trial_counts",
            "availability": "RECORDED",
            "source": "targets/primary_targets.parquet",
            "note": "Retained counts by subject, run, and block are available.",
        },
        {
            "qc_category": "study1_trial_attrition_reasons",
            "availability": availability_for_columns(
                study1_columns,
                {"drop_reason", "exclusion_reason", "retention_reason"},
            ),
            "source": "targets/primary_targets.parquet",
            "note": "Dropped-trial reasons are not recorded in the current target table.",
        },
        {
            "qc_category": "eeg_fmri_alignment_residuals",
            "availability": availability_for_columns(
                study1_columns,
                {"alignment_error_s", "eeg_fmri_offset_s", "timing_residual_s"},
            ),
            "source": "targets/primary_targets.parquet",
            "note": "Explicit EEG-fMRI timing residual columns are not recorded.",
        },
        {
            "qc_category": "study2_source_stage_qc",
            "availability": "RECORDED",
            "source": "source_stage/qc_alpha.tsv; qc_beta.tsv; qc_gamma.tsv",
            "note": "Source-stage rank, residual df, condition number, and eligibility are available.",
        },
    ]


def availability_for_columns(available: set[str], candidates: set[str]) -> str:
    if available.intersection(candidates):
        return "RECORDED"
    return "NOT_RECORDED"


def temporal_flag_for(
    subject_id: str,
    temporal_models: dict[tuple[str, str], pd.DataFrame],
) -> tuple[str, str]:
    row: dict[str, object] = {"subject_id": subject_id, "model": "elasticnet"}
    for window in TEMPORAL_WINDOWS:
        row[window] = temporal_delta(temporal_models.get(("elasticnet", window)), subject_id)
    return temporal_flag_from_row(row)


def temporal_delta(frame: pd.DataFrame | None, subject_id: str) -> float | str:
    if frame is None:
        return ""
    rows = frame.loc[frame["test_subject"] == subject_id]
    if rows.empty:
        return ""
    return rounded(float(rows.iloc[0]["delta_r2"]))


def temporal_flag_from_row(row: dict[str, object]) -> tuple[str, str]:
    controls = [
        numeric_or_none(row.get("prestimulus_wide")),
        numeric_or_none(row.get("immediate_prestimulus")),
        numeric_or_none(row.get("ramp_up")),
    ]
    plateaus = [
        numeric_or_none(row.get("early_plateau")),
        numeric_or_none(row.get("mid_plateau")),
        numeric_or_none(row.get("late_plateau")),
    ]
    controls = [value for value in controls if value is not None]
    plateaus = [value for value in plateaus if value is not None]
    if not controls or not plateaus:
        return "FAIL", "Temporal-control rows are incomplete."
    max_control = max(controls)
    max_plateau = max(plateaus)
    if max_control > 0.2:
        return "WARNING", "Positive pre-stimulus or ramp-up control reduces timing specificity."
    if max_plateau <= 0:
        return "WARNING", "No positive plateau-window delta R2."
    return "PASS", "Plateau windows exceed negative-control concern threshold."


def model_subject_row(frame: pd.DataFrame, subject_id: str, model: str) -> pd.Series:
    rows = frame.loc[(frame["test_subject"] == subject_id) & (frame["model"] == model)]
    if rows.empty:
        raise ValueError(f"No {model} model row found for {subject_id}.")
    return rows.iloc[0]


def combine_flags(*flags: object) -> str:
    if "FAIL" in flags:
        return "FAIL"
    if "WARNING" in flags:
        return "WARNING"
    return "PASS"


def count_summary(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    return ", ".join(f"{int(index)}:{int(count)}" for index, count in counts.items())


def missing_run_summary(values: pd.Series) -> str:
    observed = {int(value) for value in values.dropna().unique()}
    missing = [run for run in EXPECTED_STUDY1_RUNS if run not in observed]
    return ",".join(str(run) for run in missing)


def incomplete_run_summary(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    if counts.empty:
        return ""
    incomplete = [
        f"{int(run)}:{int(count)}"
        for run, count in counts.items()
        if int(count) < EXPECTED_STUDY1_TRIALS_PER_RUN
    ]
    return ", ".join(incomplete)


def correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2:
        return float("nan")
    value = left.astype(float).corr(right.astype(float))
    return float(value)


def rounded_mean(values: pd.Series) -> float:
    return rounded(float(values.astype(float).mean()))


def rounded(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(value, 3)


def numeric_or_none(value: object) -> float | None:
    if value == "" or value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def write_qc_outputs(summary: SubjectQcSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_frame = pd.DataFrame(summary.subject_rows)
    temporal_frame = pd.DataFrame(summary.temporal_rows)
    mask_frame = pd.DataFrame(summary.mask_rows)
    completeness_frame = pd.DataFrame(summary.completeness_rows)
    subject_frame.to_csv(output_dir / "subject_qc_summary.tsv", sep="\t", index=False)
    temporal_frame.to_csv(output_dir / "subject_temporal_qc.tsv", sep="\t", index=False)
    mask_frame.to_csv(output_dir / "signature_mask_qc.tsv", sep="\t", index=False)
    completeness_frame.to_csv(output_dir / "qc_completeness.tsv", sep="\t", index=False)
    (output_dir / "subject_qc_summary.md").write_text(
        render_markdown(subject_frame, temporal_frame, mask_frame, completeness_frame),
        encoding="utf-8",
    )


def render_markdown(
    subject_frame: pd.DataFrame,
    temporal_frame: pd.DataFrame,
    mask_frame: pd.DataFrame,
    completeness_frame: pd.DataFrame,
) -> str:
    return "\n\n".join(
        [
            "# Pain Study Subject QC Summary",
            "This file is generated from completed Study 1 and Study 2 outputs. "
            "It reports subject-level QC only, not group inference.",
            "## Subject QC",
            markdown_table(subject_frame),
            "## Temporal QC",
            markdown_table(temporal_frame),
            "## Signature Mask QC",
            markdown_table(mask_frame),
            "## QC Completeness",
            markdown_table(completeness_frame),
            "",
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(markdown_cell(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *body])


def markdown_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
