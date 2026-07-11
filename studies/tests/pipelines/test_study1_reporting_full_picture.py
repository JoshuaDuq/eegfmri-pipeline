from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from studies.tests.test_support import DotConfig, validity_figure_test_config


def _config(root: Path, *, sensitivity_outputs: list[dict[str, str]] | None = None) -> DotConfig:
    reporting = {}
    if sensitivity_outputs is not None:
        reporting["sensitivity_output_roots"] = sensitivity_outputs

    deriv_root = str(root / "derivatives")
    return DotConfig(
        {
            "deriv_root": deriv_root,
            "paths": {"deriv_root": deriv_root},
            "study1": {
                "cohort": {"min_subjects": 2},
                "outputs": {"root_name": "study1"},
                "targets": {"names": ["NPS", "SIIPS1"]},
                "feature_benchmark": {
                    "n_perm": 10,
                    "max_invalid_permutation_fraction": 0.20,
                },
                "figures": validity_figure_test_config((45.3, 49.3)),
                "reporting": reporting,
            },
        }
    )


def _study1_root(config: DotConfig) -> Path:
    return Path(config.get("paths.deriv_root")) / "group" / "multimodal" / "study1"


def _write_primary_target_table(root: Path) -> None:
    rows: list[dict[str, object]] = []
    pain_pattern = (0, 0, 1, 1)
    intensity_pattern = (10.0, 30.0, 30.0, 10.0)
    nuisance_pattern = (-1.0, 1.0, -1.0, 1.0)
    for subject_index, subject_id in enumerate(("sub-0001", "sub-0002")):
        for run, (pain, intensity, nuisance) in enumerate(
            zip(pain_pattern, intensity_pattern, nuisance_pattern, strict=True),
            start=1,
        ):
            for trial_number, (temperature, nps_base, siips1_base) in enumerate(
                ((45.3, 1.1, 107.5), (49.3, 5.1, 507.5)),
                start=1,
            ):
                nps = (
                    subject_index
                    + nps_base
                    + 0.4 * (pain - 0.5)
                    + 0.01 * (intensity - 20.0)
                    + 0.2 * nuisance
                )
                siips1 = (
                    100.0 * nps
                    + siips1_base
                    - 100.0 * nps_base
                    + 10.0 * (intensity - 20.0)
                )
                rows.append(
                    _target_row(subject_id, run, trial_number, temperature, nps, siips1)
                )
    targets = pd.DataFrame(rows)
    target_dir = root / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(target_dir / "primary_targets.parquet", index=False)
    targets.to_csv(target_dir / "primary_targets.tsv", sep="\t", index=False)


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
        "NPS_fmri_n_voxels": 10,
        "NPS_fmri_scoring_mask_sha256": "nps-mask",
        "SIIPS1_fmri_n_voxels": 10,
        "SIIPS1_fmri_scoring_mask_sha256": "siips-mask",
        "hrf_weighted_framewise_displacement": 0.01,
        "hrf_weighted_std_dvars": 0.02,
        "hrf_weighted_fp1_fp2_high_frequency_power": 0.03,
        "residual_ecg_coupling": 0.04,
        "stimulus_temp": stimulus_temp,
        "selected_surface": 1,
    }


def _write_clean_events(config: DotConfig) -> None:
    deriv_root = Path(config.get("paths.deriv_root"))
    pain_pattern = (0, 0, 1, 1)
    intensity_pattern = (10.0, 30.0, 30.0, 10.0)
    for subject_id in ("sub-0001", "sub-0002"):
        event_dir = deriv_root / "preprocessed" / "eeg" / subject_id / "eeg"
        event_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for run, (pain, intensity) in enumerate(
            zip(pain_pattern, intensity_pattern, strict=True),
            start=1,
        ):
            rating = intensity + 100.0 if pain else intensity
            rows.extend(
                (
                    _event_row(run, 1, 45.3, pain, rating),
                    _event_row(run, 2, 49.3, pain, rating),
                )
            )
        events = pd.DataFrame(rows)
        events.to_csv(
            event_dir / f"{subject_id}_task-pain_proc-clean_events.tsv",
            sep="\t",
            index=False,
        )


def _event_row(
    run: int,
    trial_number: int,
    stimulus_temp: float,
    pain_binary: int,
    rating: float,
) -> dict[str, object]:
    return {
        "run": run,
        "trial_number": trial_number,
        "stimulus_temp": stimulus_temp,
        "selected_surface": 1,
        "pain_binary_coded": pain_binary,
        "vas_final_coded_rating": rating,
        "residual_ecg_coupling": 0.04,
        "fp1_fp2_high_frequency_power": 0.05,
    }


def _write_feature_summary(
    root: Path,
    target: str,
    feature_spec: str,
    *,
    partition: str = "primary",
    include_incremental_metrics: bool = True,
) -> None:
    metrics_dir = (
        root
        / "feature_benchmark"
        / partition
        / target
        / feature_spec
        / "model_comparison"
        / "metrics"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "elasticnet": _model_metrics(0.20, include_incremental_metrics=include_incremental_metrics),
        "ridge": _model_metrics(0.10, include_incremental_metrics=include_incremental_metrics),
        "subject_selection": {
            "n_requested": 2,
            "n_included": 2,
            "n_excluded": 0,
            "excluded_fraction": 0.0,
        },
    }
    (metrics_dir / "model_comparison_summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
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
    ).to_csv(metrics_dir / "model_comparison.tsv", sep="\t", index=False)


def _model_metrics(
    mean_r2: float,
    *,
    include_incremental_metrics: bool,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "mean_r2": mean_r2,
        "std_r2": 0.01,
        "ci_low_r2": mean_r2 - 0.02,
        "ci_high_r2": mean_r2 + 0.02,
        "overall_r2": mean_r2,
        "p_value_r2": 0.1,
        "mean_mae": 0.2,
        "std_mae": 0.01,
        "ci_low_mae": 0.18,
        "ci_high_mae": 0.22,
        "n_perm": 10,
        "n_perm_requested": 10,
        "n_perm_completed": 10,
        "n_perm_attempted": 11,
        "n_invalid_permutations": 1,
        "n_folds": 2,
    }
    if include_incremental_metrics:
        metrics.update(
            {
                "mean_nuisance_r2": 0.05,
                "std_nuisance_r2": 0.01,
                "mean_delta_r2": mean_r2 - 0.05,
                "std_delta_r2": 0.01,
                "ci_low_delta_r2": mean_r2 - 0.07,
                "ci_high_delta_r2": mean_r2 - 0.03,
                "p_value_delta_r2": 0.1,
            }
        )
    return metrics


def test_report_writes_full_picture_bundle(tmp_path, monkeypatch) -> None:
    from studies.pain_study.study1 import reporting

    cfg = _config(tmp_path)
    root = _study1_root(cfg)
    _write_primary_target_table(root)
    _write_clean_events(cfg)
    for target in ("NPS", "SIIPS1"):
        _write_feature_summary(root, target, "alpha")

    monkeypatch.setattr(reporting, "PRIMARY_BAND_PRESETS", {"alpha": ["alpha"]})

    report_path = reporting.write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")
    full_picture_root = report_path.parent / "full_picture"

    assert "temporal_control_interpretation" not in report.columns

    assert (full_picture_root / "full_picture_manifest.json").exists()
    assert (full_picture_root / "primary_feature_model_summary.tsv").exists()
    assert (full_picture_root / "model_leaderboard_by_mean_r2.tsv").exists()
    assert (full_picture_root / "model_leaderboard_by_delta_r2.tsv").exists()
    assert (full_picture_root / "target_by_stimulus_temp.tsv").exists()
    assert (full_picture_root / "target_by_subject_and_stimulus_temp.tsv").exists()
    assert (full_picture_root / "target_qc_metrics.tsv").exists()
    assert (full_picture_root / "behavior_signature_validity_by_subject.tsv").exists()
    assert (full_picture_root / "behavior_signature_validity_summary.tsv").exists()

    figure_root = report_path.parent / "figures" / "supplementary" / "validity"
    expected_figures = {
        "behavioral_dose_response": figure_root / "behavioral_dose_response.svg",
        "nps_behavioral_validity": figure_root / "nps_behavioral_validity.svg",
        "nps_dose_response": figure_root / "nps_dose_response.svg",
        "siips1_behavioral_validity": figure_root / "siips1_behavioral_validity.svg",
        "siips1_dose_response": figure_root / "siips1_dose_response.svg",
    }
    assert sorted(figure_root.iterdir()) == sorted(expected_figures.values())
    manifest = json.loads((full_picture_root / "full_picture_manifest.json").read_text())
    assert manifest["supplementary_figures"] == {
        name: str(path) for name, path in expected_figures.items()
    }

    by_temp = pd.read_csv(full_picture_root / "target_by_stimulus_temp.tsv", sep="\t")
    assert by_temp["stimulus_temp"].tolist() == [45.3, 49.3]
    assert by_temp["mean_NPS"].tolist() == [1.6, 5.6]
    assert by_temp["mean_SIIPS1"].tolist() == [157.5, 557.5]

    participant_validity = pd.read_csv(
        full_picture_root / "behavior_signature_validity_by_subject.tsv",
        sep="\t",
    )
    assert set(participant_validity["target"]) == {"NPS", "SIIPS1"}
    assert participant_validity["estimable"].all()
    assert set(participant_validity["non_estimability_reason"]) == {"none"}
    assert participant_validity[
        ["painful_report_beta", "within_scale_intensity_beta"]
    ].notna().all().all()

    cohort_validity = pd.read_csv(
        full_picture_root / "behavior_signature_validity_summary.tsv",
        sep="\t",
    )
    assert cohort_validity[["target", "term"]].values.tolist() == [
        ["NPS", "painful_report"],
        ["NPS", "within_scale_intensity"],
        ["SIIPS1", "painful_report"],
        ["SIIPS1", "within_scale_intensity"],
    ]
    assert cohort_validity["n_subjects"].tolist() == [2, 2, 2, 2]
    assert cohort_validity[["mean", "ci_low", "ci_high"]].notna().all().all()

    target_qc = pd.read_csv(full_picture_root / "target_qc_metrics.tsv", sep="\t")
    assert set(target_qc["target"]) == {"NPS", "SIIPS1"}
    assert "target_interpretation" not in target_qc.columns
    assert "validity_limitations" not in target_qc.columns
    assert "expected_construct_relation" not in target_qc.columns
    assert "scope_sensitivity" not in target_qc.columns
    assert "within_scale_intensity_r" in target_qc.columns
    assert "vas_rating_r" not in target_qc.columns
    assert "siips1_rating_beyond_temperature_nps_r" not in target_qc.columns
    siips1 = target_qc.loc[target_qc["target"] == "SIIPS1"].iloc[0]
    assert siips1["siips1_intensity_beyond_temperature_nps_r"] > 0.0


def test_report_compares_configured_sensitivity_roots(tmp_path, monkeypatch) -> None:
    from studies.pain_study.study1 import reporting

    cfg = _config(
        tmp_path,
        sensitivity_outputs=[{"label": "raw_target", "root_name": "study1_raw"}],
    )
    root = _study1_root(cfg)
    raw_root = Path(cfg.get("paths.deriv_root")) / "group" / "multimodal" / "study1_raw"
    _write_primary_target_table(root)
    _write_clean_events(cfg)
    for target in ("NPS", "SIIPS1"):
        _write_feature_summary(root, target, "alpha")
        _write_feature_summary(
            raw_root,
            target,
            "alpha",
            include_incremental_metrics=False,
        )

    monkeypatch.setattr(reporting, "PRIMARY_BAND_PRESETS", {"alpha": ["alpha"]})

    report_path = reporting.write_study1_report(task="pain", config=cfg)
    full_picture_root = report_path.parent / "full_picture"
    sensitivity = pd.read_csv(
        full_picture_root / "configured_sensitivity_model_summary.tsv",
        sep="\t",
    )
    pivot = pd.read_csv(full_picture_root / "primary_sensitivity_comparison.tsv", sep="\t")

    assert set(sensitivity["analysis_label"]) == {"study1", "raw_target"}
    assert set(sensitivity["target"]) == {"NPS", "SIIPS1"}
    raw_rows = sensitivity.loc[sensitivity["analysis_label"] == "raw_target"]
    assert raw_rows["mean_delta_r2"].isna().all()
    assert "mean_r2_raw_target" in pivot.columns
    assert "mean_delta_r2_study1" in pivot.columns
