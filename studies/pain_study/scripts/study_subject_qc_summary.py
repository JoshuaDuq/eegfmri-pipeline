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
TIMING_ALIGNMENT_REQUIRED_COLUMNS = {
    "subject_id",
    "n_target_trials",
    "n_fmri_plateau_events",
    "n_lss_plateau_trials",
    "n_unmatched_target_trials",
    "n_missing_fmri_plateau_events",
    "n_invalid_fmri_plateau_events",
    "n_missing_lss_plateau_trials",
    "n_invalid_lss_plateau_trials",
    "n_missing_temporal_feature_rows",
    "max_abs_fmri_plateau_start_delta_s",
    "max_abs_lss_plateau_start_delta_s",
}


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
    timing_alignment: pd.DataFrame
    study2_source_qc: dict[str, pd.DataFrame]
    study2_source_input: pd.DataFrame
    source_power_shapes: dict[str, SourcePowerShape]
    anatomy_status: dict[str, str]


@dataclass(frozen=True)
class SubjectQcSummary:
    subject_rows: list[dict[str, object]]
    temporal_rows: list[dict[str, object]]
    timing_alignment_rows: list[dict[str, object]]
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
        timing_alignment=read_tsv(
            study1_root / "qc" / "timing_audit" / "study1_timing_audit_summary.tsv"
        ),
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
    labels: dict[str, str] = {}
    for subject_id in subjects:
        trans = subjects_dir / subject_id / "bem" / f"{subject_id}-trans.fif"
        bem = subjects_dir / subject_id / "bem" / f"{subject_id}-5120-5120-5120-bem-sol.fif"
        labels[subject_id] = _anatomy_label(trans.exists(), bem.exists())
    return labels


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
    timing_alignment_rows = build_timing_alignment_rows(inputs)
    mask_rows = build_mask_rows(inputs.study1_targets)
    completeness_rows = build_completeness_rows(inputs)
    return SubjectQcSummary(
        subject_rows=subject_rows,
        temporal_rows=temporal_rows,
        timing_alignment_rows=timing_alignment_rows,
        mask_rows=mask_rows,
        completeness_rows=completeness_rows,
    )


def validate_inputs(inputs: SubjectQcInputs) -> None:
    require_columns(
        inputs.study1_targets,
        {
            "subject_id",
            "run",
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
        {"subject_id", "run"},
        table_name="Study 2 source-stage input",
    )
    for band, frame in inputs.study2_source_qc.items():
        require_columns(
            frame,
            {
                "subject_id",
                "source_stage_criteria_met",
                "retained_trials",
                "valid_runs",
                "design_rank",
                "residual_degrees_of_freedom",
                "condition_number",
                "unmet_criteria",
            },
            table_name=f"Study 2 {band} QC table",
        )
    for (model, window), frame in inputs.temporal_models.items():
        require_columns(
            frame,
            {"model", "test_subject", "delta_r2"},
            table_name=f"Temporal {model} {window} table",
        )
    require_columns(
        inputs.timing_alignment,
        TIMING_ALIGNMENT_REQUIRED_COLUMNS,
        table_name="Study 1 timing-alignment audit",
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
    return row


def study1_metrics(
    subject_id: str,
    target_rows: pd.DataFrame,
    inputs: SubjectQcInputs,
) -> dict[str, object]:
    if target_rows.empty:
        return {
            "study1_retained_trials": 0,
            "study1_runs": 0,
            "study1_trials_by_run": "",
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
        }

    primary = model_subject_row(inputs.primary_model, subject_id, "elasticnet")
    gamma = model_subject_row(inputs.gamma_model, subject_id, "elasticnet")
    nps_temp_r = correlation(target_rows["NPS"], target_rows["stimulus_temp"])
    siips1_temp_r = correlation(target_rows["SIIPS1"], target_rows["stimulus_temp"])
    missing_runs = missing_run_summary(target_rows["run"])
    incomplete_runs = incomplete_run_summary(target_rows["run"])

    return {
        "study1_retained_trials": len(target_rows),
        "study1_runs": target_rows["run"].nunique(),
        "study1_trials_by_run": count_summary(target_rows["run"]),
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
    }


def study2_metrics(
    subject_id: str,
    source_input: pd.DataFrame,
    source_qc_rows: list[pd.Series],
    source_shape: SourcePowerShape | None,
    inputs: SubjectQcInputs,
) -> dict[str, object]:
    if not source_qc_rows:
        return {
            "study2_qc_bands_consistent": "",
            "study2_source_stage_criteria_met": "",
            "study2_retained_trials": 0,
            "study2_valid_runs": 0,
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
            "study2_qc_bands_consistent": False,
            "study2_source_stage_criteria_met": "",
            "study2_retained_trials": int(source_qc["retained_trials"]),
            "study2_valid_runs": int(source_qc["valid_runs"]),
            "study2_design_rank": int(source_qc["design_rank"]),
            "study2_residual_df": int(source_qc["residual_degrees_of_freedom"]),
            "study2_condition_number": source_qc["condition_number"],
            "source_input_rows": len(source_input),
            "source_power_rows": source_shape.rows if source_shape else "",
            "source_vertices": source_shape.vertices if source_shape else "",
            "source_anatomy": inputs.anatomy_status.get(subject_id, "missing"),
        }

    source_qc = source_qc_rows[0]
    source_stage_criteria_met = bool(source_qc["source_stage_criteria_met"])
    anatomy = inputs.anatomy_status.get(subject_id, "missing")
    input_rows = len(source_input)
    source_rows = source_shape.rows if source_shape else None
    return {
        "study2_qc_bands_consistent": True,
        "study2_source_stage_criteria_met": source_stage_criteria_met,
        "study2_retained_trials": int(source_qc["retained_trials"]),
        "study2_valid_runs": int(source_qc["valid_runs"]),
        "study2_design_rank": int(source_qc["design_rank"]),
        "study2_residual_df": int(source_qc["residual_degrees_of_freedom"]),
        "study2_condition_number": source_qc["condition_number"],
        "source_input_rows": input_rows,
        "source_power_rows": source_rows if source_rows is not None else "",
        "source_vertices": source_shape.vertices if source_shape else "",
        "source_anatomy": anatomy,
    }


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
        "source_stage_criteria_met",
        "retained_trials",
        "valid_runs",
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
            rows.append(row)
    return rows


def build_timing_alignment_rows(inputs: SubjectQcInputs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject_id in inputs.subjects:
        timing_row = timing_alignment_subject_row(inputs.timing_alignment, subject_id)
        if timing_row is None:
            rows.append(
                {
                    "subject_id": subject_id,
                    "target_trials": "",
                    "fmri_plateau_events": "",
                    "lss_plateau_trials": "",
                    "missing_target_linked_rows": "",
                    "max_plateau_start_delta_ms": "",
                }
            )
            continue
        rows.append(timing_alignment_qc_row(timing_row))
    return rows


def timing_alignment_qc_row(row: pd.Series) -> dict[str, object]:
    return {
        "subject_id": str(row["subject_id"]),
        "target_trials": int(row["n_target_trials"]),
        "fmri_plateau_events": int(row["n_fmri_plateau_events"]),
        "lss_plateau_trials": int(row["n_lss_plateau_trials"]),
        "missing_target_linked_rows": timing_alignment_missing_count(row),
        "max_plateau_start_delta_ms": timing_alignment_max_delta_ms(row),
    }


def timing_alignment_subject_row(
    timing_alignment: pd.DataFrame,
    subject_id: str,
) -> pd.Series | None:
    rows = timing_alignment.loc[timing_alignment["subject_id"] == subject_id]
    if rows.empty:
        return None
    if len(rows) > 1:
        raise ValueError(f"Timing-alignment audit has duplicate rows for {subject_id}.")
    return rows.iloc[0]


def timing_alignment_missing_count(row: pd.Series) -> int:
    count_columns = (
        "n_unmatched_target_trials",
        "n_missing_fmri_plateau_events",
        "n_invalid_fmri_plateau_events",
        "n_missing_lss_plateau_trials",
        "n_invalid_lss_plateau_trials",
        "n_missing_temporal_feature_rows",
    )
    return int(sum(int(row[column]) for column in count_columns))


def timing_alignment_max_delta_ms(row: pd.Series) -> float:
    deltas = [
        numeric_or_none(row["max_abs_fmri_plateau_start_delta_s"]),
        numeric_or_none(row["max_abs_lss_plateau_start_delta_s"]),
    ]
    finite = [value for value in deltas if value is not None]
    if not finite:
        return float("nan")
    return rounded(1000.0 * max(finite))


def build_completeness_rows(inputs: SubjectQcInputs) -> list[dict[str, object]]:
    study1_columns = set(inputs.study1_targets.columns)
    return [
        {
            "qc_category": "study1_retained_trial_counts",
            "availability": "recorded",
            "source": "targets/primary_targets.parquet",
        },
        {
            "qc_category": "study1_trial_attrition_reasons",
            "availability": availability_for_columns(
                study1_columns,
                {"drop_reason", "exclusion_reason", "retention_reason"},
            ),
            "source": "targets/primary_targets.parquet",
        },
        {
            "qc_category": "eeg_fmri_alignment_residuals",
            "availability": "recorded",
            "source": (
                "qc/timing_audit/study1_timing_audit_summary.tsv; "
                "reports/subject_qc/subject_timing_alignment_qc.tsv"
            ),
        },
        {
            "qc_category": "study2_source_stage_qc",
            "availability": "recorded",
            "source": "source_stage/qc_alpha.tsv; qc_beta.tsv; qc_gamma.tsv",
        },
    ]


def availability_for_columns(available: set[str], candidates: set[str]) -> str:
    if available.intersection(candidates):
        return "recorded"
    return "not_recorded"


def temporal_delta(frame: pd.DataFrame | None, subject_id: str) -> float | str:
    if frame is None:
        return ""
    rows = frame.loc[frame["test_subject"] == subject_id]
    if rows.empty:
        return ""
    return rounded(float(rows.iloc[0]["delta_r2"]))


def model_subject_row(frame: pd.DataFrame, subject_id: str, model: str) -> pd.Series:
    rows = frame.loc[(frame["test_subject"] == subject_id) & (frame["model"] == model)]
    if rows.empty:
        raise ValueError(f"No {model} model row found for {subject_id}.")
    return rows.iloc[0]


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
    timing_alignment_frame = pd.DataFrame(summary.timing_alignment_rows)
    mask_frame = pd.DataFrame(summary.mask_rows)
    completeness_frame = pd.DataFrame(summary.completeness_rows)
    subject_frame.to_csv(output_dir / "subject_qc_summary.tsv", sep="\t", index=False)
    temporal_frame.to_csv(output_dir / "subject_temporal_qc.tsv", sep="\t", index=False)
    timing_alignment_frame.to_csv(
        output_dir / "subject_timing_alignment_qc.tsv",
        sep="\t",
        index=False,
    )
    mask_frame.to_csv(output_dir / "signature_mask_qc.tsv", sep="\t", index=False)
    completeness_frame.to_csv(output_dir / "qc_completeness.tsv", sep="\t", index=False)
    (output_dir / "subject_qc_summary.md").write_text(
        render_markdown(
            subject_frame,
            temporal_frame,
            timing_alignment_frame,
            mask_frame,
            completeness_frame,
        ),
        encoding="utf-8",
    )


def render_markdown(
    subject_frame: pd.DataFrame,
    temporal_frame: pd.DataFrame,
    timing_alignment_frame: pd.DataFrame,
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
            "## Timing Alignment QC",
            markdown_table(timing_alignment_frame),
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
