"""Tests for studies.pain_study.study1.reporting aggregation and multiplicity correction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from studies.tests.test_support import DotConfig, validity_figure_test_config


def _config(root: Path) -> DotConfig:
    deriv_root = str(root / "derivatives")
    return DotConfig(
        {
            "deriv_root": deriv_root,
            "paths": {"deriv_root": deriv_root},
            "time_frequency_analysis": {"active_window": [3.0, 10.5]},
            "study1": {
                "cohort": {"min_subjects": 2},
                "outputs": {"root_name": "study1"},
                "figures": validity_figure_test_config((45.3, 49.3)),
                "targets": {"names": ["NPS", "SIIPS1"]},
                "feature_benchmark": {
                    "n_perm": 5000,
                    "max_invalid_permutation_fraction": 0.20,
                },
                "temporal_negative_controls": {
                    "feature_transform": "raw_log_power",
                    "feature_baseline_window": None,
                    "windows": {"prestimulus_wide": [-5.0, -0.01]},
                    "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                    "plateau_windows": {"early_plateau": [3.0, 5.5]},
                },
                "deep_regression": {
                    "presets": {
                        "alpha": ["alpha"],
                        "beta": ["beta"],
                        "alpha_beta": ["alpha", "beta"],
                        "gamma": [
                            "gamma_low_clean",
                            "gamma_mid_clean",
                            "gamma_high_clean",
                        ],
                        "alpha_beta_gamma": [
                            "alpha",
                            "beta",
                            "gamma_low_clean",
                            "gamma_mid_clean",
                            "gamma_high_clean",
                        ],
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
    _write_article_inputs(root)
    feature_summary = (
        root
        / "feature_benchmark"
        / partition
        / target
        / feature_spec
        / "model_comparison"
        / "metrics"
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


def _write_article_inputs(root: Path) -> None:
    target_dir = root / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_rows: list[dict[str, object]] = []
    event_rows_by_subject: dict[str, list[dict[str, object]]] = {
        "sub-0001": [],
        "sub-0002": [],
    }
    pain_pattern = (0, 0, 1, 1)
    intensity_pattern = (10.0, 30.0, 30.0, 10.0)
    nuisance_pattern = (-1.0, 1.0, -1.0, 1.0)
    for subject_index, subject_id in enumerate(("sub-0001", "sub-0002")):
        for run, (pain, intensity, nuisance) in enumerate(
            zip(pain_pattern, intensity_pattern, nuisance_pattern, strict=True),
            start=1,
        ):
            for trial_number, (temperature, target_base) in enumerate(
                ((45.3, 1.1), (49.3, 5.1)),
                start=1,
            ):
                nps = (
                    subject_index
                    + target_base
                    + 0.4 * (pain - 0.5)
                    + 0.01 * (intensity - 20.0)
                    + 0.2 * nuisance
                )
                siips1 = 100.0 * nps + 5.0 * nuisance
                target_rows.append(
                    _target_row(subject_id, run, trial_number, temperature, nps, siips1)
                )
                rating = intensity + 100.0 if pain else intensity
                event_rows_by_subject[subject_id].append(
                    _event_row(run, trial_number, temperature, pain, rating)
                )
    pd.DataFrame(target_rows).to_parquet(
        target_dir / "primary_targets.parquet",
        index=False,
    )

    deriv_root = root.parents[2]
    for subject_id in ("sub-0001", "sub-0002"):
        event_dir = deriv_root / "preprocessed" / "eeg" / subject_id / "eeg"
        event_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(event_rows_by_subject[subject_id]).to_csv(
            event_dir / f"{subject_id}_task-pain_proc-clean_events.tsv",
            sep="\t",
            index=False,
        )


def _target_row(
    subject_id: str,
    run: int,
    within_run_trial: int,
    stimulus_temp: float,
    nps: float,
    siips1: float,
) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "task": "pain",
        "run": run,
        "trial_index": within_run_trial,
        "within_run_trial": within_run_trial,
        "onset": float(within_run_trial),
        "duration": 1.0,
        "NPS": nps,
        "SIIPS1": siips1,
        "hrf_weighted_framewise_displacement": 0.01,
        "hrf_weighted_std_dvars": 0.02,
        "hrf_weighted_fp1_fp2_high_frequency_power": 0.03,
        "residual_ecg_coupling": 0.04,
        "stimulus_temp": stimulus_temp,
        "selected_surface": 1,
    }


def _event_row(
    run: int,
    trial_number: int,
    stimulus_temp: float,
    pain_binary: int,
    rating: float,
) -> dict[str, object]:
    return {
        "run_id": run,
        "trial_number": trial_number,
        "stimulus_temp": stimulus_temp,
        "selected_surface": 1,
        "pain_binary_coded": pain_binary,
        "vas_final_coded_rating": rating,
        "residual_ecg_coupling": 0.04,
        "fp1_fp2_high_frequency_power": 0.05,
    }


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
        for feature_spec in ("alpha", "beta", "gamma", "alpha_beta", "alpha_beta_gamma"):
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
        (report["lane"] == "feature_benchmark") & (report["analysis_partition"] == "primary")
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
    assert len(secondary) == 19
    assert primary_gate.iloc[0]["target"] == "NPS"
    assert primary_gate.iloc[0]["feature_spec"] == "alpha_beta_gamma"
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
        for feature_spec in ("alpha", "beta", "gamma", "alpha_beta", "alpha_beta_gamma"):
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


def test_report_keeps_diagnostic_metrics_without_auto_interpretation(tmp_path) -> None:
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
        for feature_spec in ("alpha", "beta", "gamma", "alpha_beta", "alpha_beta_gamma"):
            _write_feature_summary(
                root,
                target,
                feature_spec,
                diagnostics=limiting_diagnostics,
            )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert primary_gate["target_split_half_reliability"] == 0.31
    assert primary_gate["target_reliability_n_trials"] == 28
    assert bool(primary_gate["precision_flag_passed"]) is False
    assert pd.isna(primary_gate["temporal_negative_controls_passed"])
    assert bool(primary_gate["artifact_censoring_robustness_passed"]) is False
    assert not {
        "analysis_validity_classification",
        "primary_prediction_interpretation",
        "study2_source_entry_interpretation",
        "interpretation_limitations",
    }.intersection(report.columns)


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
        for feature_spec in ("alpha", "beta", "gamma", "alpha_beta", "alpha_beta_gamma"):
            _write_feature_summary(root, target, feature_spec, diagnostics=diagnostics)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert primary_gate["target_reliability_n_trials"] == 29
    assert pd.isna(primary_gate["temporal_negative_controls_passed"])
    assert "study2_source_entry_interpretation" not in report.columns
    assert "interpretation_limitations" not in report.columns


def test_report_marks_missing_diagnostics_as_not_evaluated_not_invalid(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert pd.isna(primary_gate["target_split_half_reliability"])
    assert pd.isna(primary_gate["target_reliability_n_trials"])
    assert "analysis_validity_classification" not in report.columns
    assert "study2_source_entry_interpretation" not in report.columns
    assert "missing_interpretation_diagnostics" not in report.columns


def test_report_holm_correction_inflates_p_values(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    primary = report.loc[
        (report["lane"] == "feature_benchmark") & (report["analysis_partition"] == "primary")
    ]
    raw = pd.to_numeric(primary["p_value_r2"], errors="coerce").dropna()
    adjusted = pd.to_numeric(primary["p_value_r2_holm"], errors="coerce").dropna()

    assert len(adjusted) == len(raw)
    assert (adjusted >= raw.values[: len(adjusted)]).all()


###################################################################
# _validate_complete_primary_outputs
###################################################################


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean_delta_r2", float("inf")),
        ("ci_low_delta_r2", -float("inf")),
        ("p_value_delta_r2", -0.01),
        ("p_value_delta_r2", 1.01),
        ("n_perm_completed", 5000.9),
        ("n_perm_attempted", 5100.9),
        ("n_invalid_permutations", 100.9),
        ("n_folds", 4.5),
        ("n_subjects_included", -1),
        ("subject_excluded_fraction", 1.1),
    ],
)
def test_primary_report_rejects_invalid_numeric_values(tmp_path, field, value):
    from studies.pain_study.study1.reporting import (
        _feature_records,
        _validate_complete_primary_outputs,
    )

    cfg = _config(tmp_path)
    _write_complete_outputs(_study1_root(cfg), cfg)
    records = _feature_records(cfg)
    records[0][field] = value
    with pytest.raises(ValueError, match=field):
        _validate_complete_primary_outputs(records=records, config=cfg)


@pytest.mark.parametrize("invalid_p", [-0.01, 1.01, float("inf"), "invalid", None])
def test_holm_rejects_invalid_or_partial_p_value_families(invalid_p):
    from studies.pain_study.study1.reporting import _append_feature_multiplicity

    frame = pd.DataFrame(
        {
            "lane": ["feature_benchmark"] * 2,
            "analysis_partition": ["exploratory"] * 2,
            "claim_tier": ["exploratory"] * 2,
            "p_value_delta_r2": [0.01, invalid_p],
        }
    )
    with pytest.raises(ValueError, match="p_value_delta_r2"):
        _append_feature_multiplicity(frame)


def test_holm_preserves_absent_optional_family_and_probability_boundaries():
    from studies.pain_study.study1.reporting import _append_feature_multiplicity

    frame = pd.DataFrame(
        {
            "lane": ["feature_benchmark"] * 2,
            "analysis_partition": ["exploratory"] * 2,
            "claim_tier": ["exploratory"] * 2,
            "p_value_r2": [None, None],
            "p_value_delta_r2": [0.0, 1.0],
        }
    )
    result = _append_feature_multiplicity(frame)
    assert result["p_value_r2_holm"].isna().all()
    assert result["p_value_delta_r2_holm"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "target,feature,model",
    [("ROI", "alpha", "ridge"), ("NPS", "broadband", "ridge"), ("NPS", "alpha", "rf")],
)
def test_primary_claim_tier_rejects_unprespecified_cells(target, feature, model):
    from studies.pain_study.study1.reporting import _claim_tier

    with pytest.raises(ValueError, match="prespecified"):
        _claim_tier(
            lane="feature_benchmark",
            partition="primary",
            target_name=target,
            feature_spec=feature,
            model_name=model,
        )


def test_report_does_not_require_deep_regression_outputs(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)

    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "gamma", "alpha_beta", "alpha_beta_gamma"):
            _write_feature_summary(root, target, feature_spec)

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    assert set(report["lane"]) == {"feature_benchmark"}
    assert len(report) == 20


###################################################################
# Exploratory records included alongside primary
###################################################################


def test_report_includes_exploratory_records(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)

    _write_feature_summary(root, "NPS", "spectral", partition="exploratory")

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    exploratory = report.loc[report["analysis_partition"] == "exploratory"]
    assert len(exploratory) >= 2
    assert set(exploratory["feature_spec"]) == {"spectral"}
    assert exploratory["p_value_delta_r2_holm"].tolist() == pytest.approx([0.02, 0.02])
    assert exploratory["p_value_r2_holm"].tolist() == pytest.approx([0.10, 0.10])


def test_report_includes_temporal_control_metadata_and_holm_values(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)
    for target in ("NPS", "SIIPS1"):
        for feature_spec in (
            "temporal_prestimulus_wide",
            "temporal_ramp_up",
            "temporal_early_plateau",
        ):
            _write_feature_summary(
                root,
                target,
                feature_spec,
                partition="temporal_control",
                p_value_delta_r2=0.02,
            )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    temporal = report.loc[report["analysis_partition"] == "temporal_control"]

    assert not temporal.empty
    assert set(temporal["claim_tier"]) == {"exploratory"}
    assert set(temporal["temporal_control_window"]) == {
        "prestimulus_wide",
        "ramp_up",
        "early_plateau",
    }
    assert set(temporal["temporal_control_kind"]) == {
        "prestimulus",
        "wrong_lag",
        "plateau_sensitivity",
    }
    assert pd.to_numeric(temporal["p_value_delta_r2_holm"], errors="coerce").notna().all()


def test_report_does_not_infer_equivalence_from_nonsignificant_controls(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("temporal_prestimulus_wide", "temporal_ramp_up"):
            _write_feature_summary(
                root,
                target,
                feature_spec,
                partition="temporal_control",
                p_value_delta_r2=0.9,
            )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert pd.isna(primary_gate["temporal_negative_controls_passed"])
    assert "missing_interpretation_diagnostics" not in report.columns


def test_report_does_not_treat_anticipatory_prediction_as_null_control_failure(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("temporal_prestimulus_wide", "temporal_ramp_up"):
            _write_feature_summary(
                root,
                target,
                feature_spec,
                partition="temporal_control",
                p_value_delta_r2=0.0001,
            )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert pd.isna(primary_gate["temporal_negative_controls_passed"])
    assert "interpretation_limitations" not in report.columns
    assert (
        report.loc[report["target"] == "SIIPS1", "temporal_negative_controls_passed"].isna().all()
    )


def test_report_keeps_temporal_criterion_unassessed_with_plateau_sensitivity(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_complete_outputs(root, cfg)
    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("temporal_prestimulus_wide", "temporal_ramp_up"):
            _write_feature_summary(
                root,
                target,
                feature_spec,
                partition="temporal_control",
                p_value_delta_r2=0.9,
            )
        _write_feature_summary(
            root,
            target,
            "temporal_early_plateau",
            partition="temporal_control",
            p_value_delta_r2=0.0001,
        )

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"].iloc[0]

    assert pd.isna(primary_gate["temporal_negative_controls_passed"])
    assert "interpretation_limitations" not in report.columns
