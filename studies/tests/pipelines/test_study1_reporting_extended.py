"""Tests for studies.pain_study.study1.reporting aggregation and multiplicity correction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from studies.tests.test_support import DotConfig


def _config(root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": str(root / "derivatives")},
            "study1": {
                "outputs": {"root_name": "study1"},
                "targets": {"names": ["NPS", "SIIPS1"]},
                "feature_benchmark": {
                    "n_perm": 5000,
                    "max_invalid_permutation_fraction": 0.20,
                },
                "deep_regression": {
                    "presets": {
                        "alpha": ["alpha"],
                        "beta": ["beta"],
                        "alpha_beta": ["alpha", "beta"],
                    }
                },
            },
        }
    )


def _study1_root(config: DotConfig) -> Path:
    return Path(config.get("paths.deriv_root")) / "group" / "multimodal" / "study1"


def _write_feature_summary(
    root: Path,
    target: str,
    feature_spec: str,
    *,
    partition: str = "primary",
    p_value_r2: float | None = 0.05,
    p_value_delta_r2: float | None = 0.01,
    write_fold_table: bool = True,
    diagnostics: dict[str, object] | None = None,
) -> None:
    feature_summary = (
        root / "feature_benchmark" / partition / target / feature_spec
        / "model_comparison" / "metrics"
    )
    feature_summary.mkdir(parents=True, exist_ok=True)
    payload = {
        "elasticnet": {
            "mean_r2": 0.4,
            "std_r2": 0.01,
            "ci_low_r2": 0.35,
            "ci_high_r2": 0.45,
            "mean_delta_r2": 0.12,
            "std_delta_r2": 0.02,
            "ci_low_delta_r2": 0.08,
            "ci_high_delta_r2": 0.16,
            "overall_r2": 0.38,
            "p_value_r2": p_value_r2,
            "p_value_delta_r2": p_value_delta_r2,
            "mean_mae": 0.2,
            "std_mae": 0.01,
            "ci_low_mae": 0.18,
            "ci_high_mae": 0.22,
            "n_perm": 5000,
            "n_perm_requested": 5000,
            "n_perm_completed": 5000,
            "n_perm_attempted": 5100,
            "n_invalid_permutations": 100,
            "n_folds": 4,
        },
        "ridge": {
            "mean_r2": 0.3,
            "std_r2": 0.01,
            "ci_low_r2": 0.25,
            "ci_high_r2": 0.35,
            "mean_delta_r2": 0.08,
            "std_delta_r2": 0.02,
            "ci_low_delta_r2": 0.04,
            "ci_high_delta_r2": 0.12,
            "overall_r2": 0.29,
            "p_value_r2": p_value_r2,
            "p_value_delta_r2": p_value_delta_r2,
            "mean_mae": 0.3,
            "std_mae": 0.01,
            "ci_low_mae": 0.28,
            "ci_high_mae": 0.32,
            "n_perm": 5000,
            "n_perm_requested": 5000,
            "n_perm_completed": 5000,
            "n_perm_attempted": 5050,
            "n_invalid_permutations": 50,
            "n_folds": 4,
        },
        "subject_selection": {
            "n_requested": 4,
            "n_included": 4,
            "n_excluded": 0,
            "excluded_fraction": 0.0,
            "max_excluded_subject_fraction": 1.0,
        },
    }
    if diagnostics:
        for model_name in ("elasticnet", "ridge"):
            payload[model_name].update(diagnostics)

    with open(feature_summary / "model_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    if write_fold_table:
        pd.DataFrame(
            [
                {
                    "model": "elasticnet",
                    "fold": 0,
                    "test_subject": "sub-0001",
                    "best_params": "{'model__alpha': 0.1}",
                },
                {
                    "model": "ridge",
                    "fold": 0,
                    "test_subject": "sub-0001",
                    "best_params": "{'model__alpha': 1.0}",
                },
            ]
        ).to_csv(feature_summary / "model_comparison.tsv", sep="\t", index=False)


def _write_deep_summary(root: Path, target: str, preset: str) -> None:
    deep_summary = root / "deep_regression" / target / preset
    deep_summary.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "band_temporal_regressor",
        "mean_r2": 0.5,
        "mean_mae": 0.1,
        "n_folds": 4,
    }
    with open(deep_summary / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _write_complete_outputs(root: Path, config: DotConfig) -> None:
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(root, target, feature_spec)
            _write_deep_summary(root, target, feature_spec)


###################################################################
# write_study1_report
###################################################################


def test_report_writes_three_output_formats(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)

    assert report_path.name == "study1_report.tsv"
    report_dir = report_path.parent
    assert (report_dir / "study1_report.parquet").exists()
    assert (report_dir / "study1_report.json").exists()


def test_report_json_contains_summary_metadata(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    json_path = report_path.parent / "study1_report.json"
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert summary["task"] == "pain"
    assert summary["n_records"] > 0
    assert "feature_benchmark" in summary["lanes"]
    assert "deep_regression" in summary["lanes"]
    assert "primary_gate" in summary["claim_tiers"]
    assert "secondary_confirmatory" in summary["claim_tiers"]
    assert set(summary["targets"]) == {"NPS", "SIIPS1"}


def test_report_rejects_empty_results(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)

    with pytest.raises(FileNotFoundError, match="No Study 1"):
        write_study1_report(task="pain", config=cfg)


###################################################################
# Holm multiplicity correction
###################################################################


def test_report_holm_corrected_p_values_are_present(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    assert "p_value_r2_holm" in report.columns
    assert "p_value_delta_r2_holm" in report.columns

    primary_feature = report.loc[
        (report["lane"] == "feature_benchmark")
        & (report["analysis_partition"] == "primary")
    ]
    assert primary_feature["p_value_r2_holm"].notna().any()
    assert primary_feature["p_value_delta_r2_holm"].notna().any()


def test_report_separates_primary_gate_from_secondary_confirmatory_grid(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    primary_gate = report.loc[report["claim_tier"] == "primary_gate"]
    secondary = report.loc[report["claim_tier"] == "secondary_confirmatory"]

    assert len(primary_gate) == 1
    assert len(secondary) == 11
    assert primary_gate.iloc[0]["target"] == "NPS"
    assert primary_gate.iloc[0]["feature_spec"] == "alpha_beta"
    assert primary_gate.iloc[0]["model"] == "elasticnet"
    assert primary_gate.iloc[0]["p_value_delta_r2_holm"] == pytest.approx(
        primary_gate.iloc[0]["p_value_delta_r2"]
    )
    assert (
        pd.to_numeric(secondary["p_value_delta_r2_holm"], errors="coerce")
        >= pd.to_numeric(secondary["p_value_delta_r2"], errors="coerce")
    ).all()


def test_report_rejects_missing_primary_delta_p_values(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(root, target, feature_spec, p_value_delta_r2=None)

    with pytest.raises(ValueError, match="p_value_delta_r2"):
        write_study1_report(task="pain", config=cfg)


def test_report_rejects_missing_primary_subject_selection_counts(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)
    summary_path = (
        root
        / "feature_benchmark"
        / "primary"
        / "NPS"
        / "alpha"
        / "model_comparison"
        / "metrics"
        / "model_comparison_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload.pop("subject_selection")
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="n_subjects_requested"):
        write_study1_report(task="pain", config=cfg)


def test_report_rejects_invalid_permutation_budget_violation(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)
    summary_path = (
        root
        / "feature_benchmark"
        / "primary"
        / "NPS"
        / "alpha_beta"
        / "model_comparison"
        / "metrics"
        / "model_comparison_summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["elasticnet"]["n_perm_attempted"] = 6251
    payload["elasticnet"]["n_invalid_permutations"] = 1251
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid permutation budget"):
        write_study1_report(task="pain", config=cfg)


def test_report_includes_protocol_audit_fields(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary = report.loc[
        (report["lane"] == "feature_benchmark")
        & (report["analysis_partition"] == "primary")
        & (report["model"] == "elasticnet")
    ].iloc[0]

    assert primary["ci_low_delta_r2"] == pytest.approx(0.08)
    assert primary["ci_high_delta_r2"] == pytest.approx(0.16)
    assert primary["n_perm_completed"] == 5000
    assert primary["n_perm_attempted"] == 5100
    assert primary["n_invalid_permutations"] == 100
    assert primary["n_subjects_included"] == 4
    assert primary["n_subjects_excluded"] == 0
    assert "{'model__alpha': 0.1}" in primary["best_params_by_fold"]


def test_report_marks_failed_diagnostics_without_invalidating_primary_prediction(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    limiting_diagnostics = {
        "target_split_half_reliability": 0.31,
        "target_reliability_n_trials": 28,
        "precision_flag_passed": False,
        "level2_mean_delta_r2": 0.001,
        "within_subject_centered_delta_r2": -0.002,
        "temporal_negative_controls_passed": False,
        "artifact_censoring_robustness_passed": False,
    }
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(
                root,
                target,
                feature_spec,
                diagnostics=limiting_diagnostics,
            )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert primary_gate["analysis_validity_status"] == "analysis_valid"
    assert primary_gate["primary_prediction_status"] == "primary_prediction_positive"
    assert primary_gate["study2_source_entry_status"] == "source_interpretation_exploratory"
    flags = set(str(primary_gate["interpretation_flags"]).split(";"))
    assert "target_reliability_limited" in flags
    assert "precision_limited" in flags
    assert "level2_convergence_limited" in flags
    assert "within_subject_tracking_limited" in flags
    assert "temporal_specificity_limited" in flags
    assert "artifact_robustness_limited" in flags


def test_report_requires_reliability_trial_count_for_source_entry(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    diagnostics = {
        "target_split_half_reliability": 0.62,
        "target_reliability_n_trials": 29,
        "level2_mean_delta_r2": 0.006,
        "within_subject_centered_delta_r2": 0.002,
        "temporal_negative_controls_passed": True,
        "artifact_censoring_robustness_passed": True,
    }
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(root, target, feature_spec, diagnostics=diagnostics)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert primary_gate["study2_source_entry_status"] == "source_interpretation_exploratory"
    flags = set(str(primary_gate["interpretation_flags"]).split(";"))
    assert "target_reliability_limited" in flags


def test_report_marks_missing_diagnostics_as_not_evaluated_not_invalid(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert primary_gate["analysis_validity_status"] == "analysis_valid"
    assert primary_gate["study2_source_entry_status"] == "source_entry_not_evaluated"
    missing = set(str(primary_gate["missing_interpretation_diagnostics"]).split(";"))
    assert "target_split_half_reliability" in missing
    assert "temporal_negative_controls_passed" in missing


def test_report_holm_correction_inflates_p_values(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    primary = report.loc[
        (report["lane"] == "feature_benchmark")
        & (report["analysis_partition"] == "primary")
    ]
    raw = pd.to_numeric(primary["p_value_r2"], errors="coerce").dropna()
    adjusted = pd.to_numeric(primary["p_value_r2_holm"], errors="coerce").dropna()

    assert len(adjusted) == len(raw)
    assert (adjusted >= raw.values[: len(adjusted)]).all()


###################################################################
# _validate_complete_primary_outputs
###################################################################


def test_report_does_not_require_deep_regression_outputs(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)

    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "alpha_beta"):
            _write_feature_summary(root, target, feature_spec)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    assert set(report["lane"]) == {"feature_benchmark"}
    assert len(report) == 12


###################################################################
# Exploratory records included alongside primary
###################################################################


def test_report_includes_exploratory_records(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    _write_feature_summary(
        root, "NPS", "spectral", partition="exploratory"
    )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    exploratory = report.loc[report["analysis_partition"] == "exploratory"]
    assert len(exploratory) >= 2
    assert set(exploratory["feature_spec"]) == {"spectral"}
