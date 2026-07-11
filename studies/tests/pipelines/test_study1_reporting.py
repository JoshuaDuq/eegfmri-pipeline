from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from studies.tests.test_support import DotConfig, validity_figure_test_config


def _config(root: Path) -> DotConfig:
    deriv_root = str(root / "derivatives")
    return DotConfig(
        {
            "deriv_root": deriv_root,
            "paths": {"deriv_root": deriv_root},
            "study1": {
                "cohort": {"min_subjects": 2},
                "outputs": {"root_name": "study1"},
                "figures": validity_figure_test_config((45.3, 49.3)),
                "targets": {"names": ["NPS", "SIIPS1"]},
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


def _write_feature_summary(
    root: Path,
    target: str,
    feature_spec: str,
    *,
    partition: str = "primary",
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
            "ci_low_r2": 0.35,
            "ci_high_r2": 0.45,
            "mean_delta_r2": 0.12,
            "ci_low_delta_r2": 0.08,
            "ci_high_delta_r2": 0.16,
            "overall_r2": 0.38,
            "p_value_delta_r2": 0.01,
            "mean_mae": 0.2,
            "n_perm": 5000,
            "n_perm_completed": 5000,
            "n_perm_attempted": 5025,
            "n_invalid_permutations": 25,
            "n_folds": 4,
        },
        "ridge": {
            "mean_r2": 0.3,
            "ci_low_r2": 0.25,
            "ci_high_r2": 0.35,
            "mean_delta_r2": 0.08,
            "ci_low_delta_r2": 0.04,
            "ci_high_delta_r2": 0.12,
            "overall_r2": 0.29,
            "p_value_delta_r2": 0.03,
            "mean_mae": 0.3,
            "n_perm": 5000,
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
        },
    }
    with open(feature_summary / "model_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
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
        "run": run,
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


def test_write_study1_report_aggregates_feature_and_deep_summaries(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = Path(cfg.get("paths.deriv_root")) / "group" / "multimodal" / "study1"

    for target in ("NPS", "SIIPS1"):
        for feature_spec in ("alpha", "beta", "gamma", "alpha_beta", "alpha_beta_gamma"):
            _write_feature_summary(root, target, feature_spec)
            _write_deep_summary(root, target, feature_spec)

    _write_feature_summary(root, "NPS", "spectral", partition="exploratory")

    report_path = write_study1_report(task="pain", config=cfg)
    report = pd.read_csv(report_path, sep="\t")

    assert len(report) == 32
    assert set(report["lane"]) == {"feature_benchmark", "deep_regression"}
    assert set(report["analysis_partition"]) == {"primary", "exploratory"}
    assert set(report["claim_tier"]) == {
        "primary_gate",
        "secondary_confirmatory",
        "exploratory",
    }
    assert set(report["feature_spec"]) == {"alpha", "beta", "gamma", "alpha_beta", "alpha_beta_gamma", "spectral"}
    assert "mean_delta_r2" in report.columns
    assert "p_value_delta_r2" in report.columns
    primary_gate = report.loc[report["claim_tier"] == "primary_gate"]
    assert len(primary_gate) == 1
    assert primary_gate.iloc[0]["target"] == "NPS"
    assert primary_gate.iloc[0]["feature_spec"] == "alpha_beta_gamma"
    assert primary_gate.iloc[0]["model"] == "elasticnet"
    primary_feature_models = set(
        report.loc[
            (report["lane"] == "feature_benchmark")
            & (report["analysis_partition"] == "primary"),
            "model",
        ]
    )
    assert primary_feature_models == {"elasticnet", "ridge"}


def test_write_study1_report_rejects_incomplete_primary_outputs(tmp_path) -> None:
    from studies.pain_study.study1.reporting import write_study1_report

    cfg = _config(tmp_path)
    root = Path(cfg.get("paths.deriv_root")) / "group" / "multimodal" / "study1"
    _write_feature_summary(root, "NPS", "alpha")

    try:
        write_study1_report(task="pain", config=cfg)
    except FileNotFoundError as exc:
        assert "missing prespecified Study 1 outputs" in str(exc)
    else:
        raise AssertionError("Expected incomplete Study 1 outputs to be rejected.")
