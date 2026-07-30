from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _target_frame() -> pd.DataFrame:
    rows = []
    for subject_id in ("sub-0001", "sub-0002"):
        runs = (1, 2, 3, 4, 5, 6) if subject_id == "sub-0001" else (1, 2, 3, 5, 6)
        for run in runs:
            for trial in range(1, 12):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "run": run,
                        "within_run_trial": trial,
                        "onset": 10.0 * run + trial,
                        "NPS": float(run + trial),
                        "SIIPS1": float(10 * run + trial),
                        "NPS_fmri_n_voxels": 10,
                        "NPS_fmri_scoring_mask_sha256": "nps-mask",
                        "SIIPS1_fmri_n_voxels": 20,
                        "SIIPS1_fmri_scoring_mask_sha256": "siips-mask",
                        "hrf_weighted_framewise_displacement": 0.05,
                        "hrf_weighted_std_dvars": 1.0,
                        "hrf_weighted_fp1_fp2_high_frequency_power": 0.1,
                        "residual_ecg_coupling": 0.2,
                        "stimulus_temp": 44.3 + ((trial - 1) % 5) + 1,
                        "selected_surface": ((trial - 1) % 5) + 1,
                    }
                )
    return pd.DataFrame(rows)


def _model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "elasticnet",
                "test_subject": "sub-0001",
                "r2": 0.10,
                "r2_nuisance": 0.02,
                "delta_r2": 0.08,
                "mae": 1.0,
                "mae_nuisance": 1.2,
            },
            {
                "model": "elasticnet",
                "test_subject": "sub-0002",
                "r2": -0.20,
                "r2_nuisance": -0.10,
                "delta_r2": -0.10,
                "mae": 2.0,
                "mae_nuisance": 1.5,
            },
        ]
    )


def _temporal_frames() -> dict[tuple[str, str], pd.DataFrame]:
    frames = {}
    for model in ("elasticnet", "ridge"):
        frames[(model, "prestimulus_wide")] = pd.DataFrame(
            [
                {"model": model, "test_subject": "sub-0001", "delta_r2": 0.01},
                {"model": model, "test_subject": "sub-0002", "delta_r2": 0.40},
            ]
        )
        frames[(model, "ramp_up")] = pd.DataFrame(
            [
                {"model": model, "test_subject": "sub-0001", "delta_r2": -0.05},
                {"model": model, "test_subject": "sub-0002", "delta_r2": 0.20},
            ]
        )
        frames[(model, "mid_plateau")] = pd.DataFrame(
            [
                {"model": model, "test_subject": "sub-0001", "delta_r2": 0.30},
                {"model": model, "test_subject": "sub-0002", "delta_r2": 0.10},
            ]
        )
    return frames


def _source_qc_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "sub-0001",
                "band": "combined",
                "source_stage_criteria_met": True,
                "retained_trials": 6,
                "valid_runs": 3,
                "design_rank": 4,
                "residual_degrees_of_freedom": 2,
                "condition_number": 10.0,
                "unmet_criteria": "",
            },
            {
                "subject_id": "sub-0002",
                "band": "combined",
                "source_stage_criteria_met": False,
                "retained_trials": 6,
                "valid_runs": 3,
                "design_rank": 2,
                "residual_degrees_of_freedom": 0,
                "condition_number": float("inf"),
                "unmet_criteria": "source_stage_design_rank",
            },
        ]
    )


def _timing_alignment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "sub-0001",
                "n_target_trials": 66,
                "n_fmri_plateau_events": 66,
                "n_lss_plateau_trials": 66,
                "n_unmatched_target_trials": 0,
                "n_missing_fmri_plateau_events": 0,
                "n_invalid_fmri_plateau_events": 0,
                "n_missing_lss_plateau_trials": 0,
                "n_invalid_lss_plateau_trials": 0,
                "n_missing_temporal_feature_rows": 0,
                "max_abs_fmri_plateau_start_delta_s": 0.003,
                "max_abs_lss_plateau_start_delta_s": 0.003,
            },
            {
                "subject_id": "sub-0002",
                "n_target_trials": 55,
                "n_fmri_plateau_events": 55,
                "n_lss_plateau_trials": 54,
                "n_unmatched_target_trials": 0,
                "n_missing_fmri_plateau_events": 0,
                "n_invalid_fmri_plateau_events": 0,
                "n_missing_lss_plateau_trials": 1,
                "n_invalid_lss_plateau_trials": 1,
                "n_missing_temporal_feature_rows": 0,
                "max_abs_fmri_plateau_start_delta_s": 0.004,
                "max_abs_lss_plateau_start_delta_s": 0.004,
            },
        ]
    )


def test_subject_qc_summary_writes_machine_and_human_readable_outputs(tmp_path: Path) -> None:
    from studies.pain_study.scripts.study_support.study_subject_qc_summary import (
        SourcePowerShape,
        SubjectQcInputs,
        build_subject_qc,
        write_qc_outputs,
    )

    inputs = SubjectQcInputs(
        subjects=("sub-0001", "sub-0002", "sub-0003"),
        study1_targets=_target_frame(),
        primary_model=_model_frame(),
        gamma_model=_model_frame(),
        temporal_models=_temporal_frames(),
        timing_alignment=_timing_alignment_frame(),
        study2_source_qc={"alpha": _source_qc_frame()},
        study2_source_input=pd.DataFrame(
            [
                {"subject_id": "sub-0001", "run": 1},
                {"subject_id": "sub-0001", "run": 2},
                {"subject_id": "sub-0001", "run": 3},
                {"subject_id": "sub-0001", "run": 1},
                {"subject_id": "sub-0001", "run": 2},
                {"subject_id": "sub-0001", "run": 3},
                {"subject_id": "sub-0002", "run": 1},
            ]
        ),
        source_power_shapes={
            "sub-0001": SourcePowerShape(rows=6, vertices=8196),
            "sub-0002": SourcePowerShape(rows=7, vertices=8196),
        },
        anatomy_status={"sub-0001": "trans+BEM", "sub-0002": "missing"},
    )

    summary = build_subject_qc(inputs)

    rows = {row["subject_id"]: row for row in summary.subject_rows}
    assert rows["sub-0001"]["study1_missing_runs"] == ""
    assert rows["sub-0001"]["study1_incomplete_runs"] == ""
    assert rows["sub-0001"]["study1_stimulus_temperatures"] == 5
    assert rows["sub-0001"]["study1_selected_surfaces"] == 5
    assert rows["sub-0002"]["study1_missing_runs"] == "4"
    assert not {
        "overall_classification",
        "study1_classification",
        "study1_note",
        "timing_alignment_classification",
        "timing_alignment_note",
        "temporal_classification",
        "temporal_note",
        "study2_classification",
        "study2_note",
    }.intersection(rows["sub-0001"])
    assert len(summary.temporal_rows) == 6
    assert len(summary.timing_alignment_rows) == 3
    assert len(summary.completeness_rows) == 4
    assert any(
        row["qc_category"] == "eeg_fmri_alignment_residuals"
        and row["availability"] == "recorded"
        for row in summary.completeness_rows
    )

    write_qc_outputs(summary, tmp_path)

    subject_tsv = tmp_path / "subject_qc_summary.tsv"
    temporal_tsv = tmp_path / "subject_temporal_qc.tsv"
    timing_tsv = tmp_path / "subject_timing_alignment_qc.tsv"
    completeness_tsv = tmp_path / "qc_completeness.tsv"
    markdown = tmp_path / "subject_qc_summary.md"
    assert subject_tsv.exists()
    assert temporal_tsv.exists()
    assert timing_tsv.exists()
    assert completeness_tsv.exists()
    assert markdown.exists()
    markdown_text = markdown.read_text()
    assert "sub-0001" in markdown_text
    assert "Timing Alignment QC" in markdown_text
    assert "classification" not in markdown_text


def test_subject_qc_summary_rejects_missing_required_columns() -> None:
    from studies.pain_study.scripts.study_support.study_subject_qc_summary import (
        require_columns,
    )

    with pytest.raises(ValueError, match="missing required column"):
        require_columns(pd.DataFrame({"subject_id": ["sub-0001"]}), {"subject_id", "run"})


def test_subject_qc_summary_warns_when_source_qc_bands_disagree() -> None:
    from studies.pain_study.scripts.study_support.study_subject_qc_summary import (
        SourcePowerShape,
        SubjectQcInputs,
        build_subject_qc,
    )

    beta_qc = _source_qc_frame().copy()
    beta_qc.loc[beta_qc["subject_id"] == "sub-0001", "source_stage_criteria_met"] = False
    beta_qc.loc[beta_qc["subject_id"] == "sub-0001", "unmet_criteria"] = "beta_criterion"

    inputs = SubjectQcInputs(
        subjects=("sub-0001",),
        study1_targets=_target_frame(),
        primary_model=_model_frame(),
        gamma_model=_model_frame(),
        temporal_models=_temporal_frames(),
        timing_alignment=_timing_alignment_frame(),
        study2_source_qc={"alpha": _source_qc_frame(), "beta": beta_qc},
        study2_source_input=pd.DataFrame(
            [
                {"subject_id": "sub-0001", "run": 1},
                {"subject_id": "sub-0001", "run": 2},
                {"subject_id": "sub-0001", "run": 3},
                {"subject_id": "sub-0001", "run": 1},
                {"subject_id": "sub-0001", "run": 2},
                {"subject_id": "sub-0001", "run": 3},
            ]
        ),
        source_power_shapes={"sub-0001": SourcePowerShape(rows=6, vertices=8196)},
        anatomy_status={"sub-0001": "trans+BEM"},
    )

    summary = build_subject_qc(inputs)

    row = summary.subject_rows[0]
    assert "study2_classification" not in row
    assert "study2_note" not in row
    assert row["study2_qc_bands_consistent"] is False


def test_incomplete_run_summary_uses_design_trial_count() -> None:
    from studies.pain_study.scripts.study_support.study_subject_qc_summary import (
        incomplete_run_summary,
    )

    uniformly_truncated_runs = pd.Series([run for run in range(1, 7) for _ in range(5)])

    assert incomplete_run_summary(uniformly_truncated_runs) == (
        "1:5, 2:5, 3:5, 4:5, 5:5, 6:5"
    )
