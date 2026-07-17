from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eeg_pipeline.utils.config.loader import ConfigDict
from studies.pain_study.study1.feature_benchmark import PRIMARY_BAND_PRESETS
from studies.pain_study.study1.reporting import FEATURE_MODELS, write_study1_report
from studies.pain_study.study1.targets import PRIMARY_SIGNATURES
from studies.tests.test_support import validity_figure_test_config


TRIAL_TEMPERATURES = (
    44.0,
    44.0,
    44.0,
    46.0,
    46.0,
    46.0,
    48.0,
    48.0,
    48.0,
    50.0,
    50.0,
    50.0,
)


def test_write_study1_report_writes_article_tables(tmp_path: Path) -> None:
    config = ConfigDict(
        {
            "paths": {"deriv_root": str(tmp_path / "derivatives")},
            "study1": {
                "outputs": {"root_name": "study1"},
                "cohort": {"min_subjects": 2},
                "figures": validity_figure_test_config((44.0, 46.0, 48.0, 50.0)),
                "feature_benchmark": {
                    "n_perm": 10,
                    "max_invalid_permutation_fraction": 0.2,
                },
            },
        }
    )
    study_root = tmp_path / "derivatives" / "group" / "multimodal" / "study1"
    subjects = ["sub-0001", "sub-0002"]

    _write_target_table(study_root, subjects)
    _write_clean_events(tmp_path / "derivatives", subjects)
    _write_feature_benchmark_outputs(study_root, subjects)

    report_path = write_study1_report(task="thermalactive", config=config)

    article_root = report_path.parent / "article_tables"
    model_path = article_root / "article_model_results.tsv"
    cohort_path = article_root / "article_cohort_summary.tsv"
    diagnostics_path = article_root / "article_target_diagnostics.tsv"
    manifest_path = article_root / "article_table_manifest.json"

    for path in (model_path, cohort_path, diagnostics_path, manifest_path):
        assert path.exists()

    model_table = pd.read_csv(model_path, sep="\t")
    assert {
        "target",
        "feature_spec",
        "model",
        "mean_delta_r2",
        "p_value_delta_r2_holm",
    }.issubset(model_table.columns)
    assert not {
        "primary_prediction_interpretation",
        "temporal_control_interpretation",
        "interpretation_limitations",
        "study2_source_entry_interpretation",
    }.intersection(model_table.columns)
    assert len(model_table) == len(PRIMARY_SIGNATURES) * len(PRIMARY_BAND_PRESETS) * len(
        FEATURE_MODELS
    )

    cohort_table = pd.read_csv(cohort_path, sep="\t")
    assert cohort_table["subject_id"].tolist() == subjects
    assert {
        "n_target_trials",
        "n_clean_event_trials",
        "n_event_without_target",
        "vas_temp_r",
        "NPS_temp_r",
        "SIIPS1_high_minus_low_temp",
    }.issubset(cohort_table.columns)
    assert cohort_table["n_event_without_target"].tolist() == [0, 0]

    diagnostics_table = pd.read_csv(diagnostics_path, sep="\t")
    assert diagnostics_table["target"].tolist() == list(PRIMARY_SIGNATURES)
    assert {
        "n_trials",
        "stimulus_temp_r",
        "within_scale_intensity_r",
        "stimulus_surface_in_sample_r2",
        "official_nuisance_in_sample_r2",
        "residual_target_variance_fraction",
        "split_half_subject_temperature_r",
    }.issubset(diagnostics_table.columns)
    assert "vas_rating_r" not in diagnostics_table.columns
    residual_variance = diagnostics_table["residual_target_variance_fraction"]
    assert ((residual_variance >= 0.0) & (residual_variance <= 1.0)).all()

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["n_subjects"] == len(subjects)
    assert manifest["n_trials"] == len(subjects) * len(TRIAL_TEMPERATURES)
    assert manifest["included_subjects"] == subjects
    assert set(manifest["tables"]) == {
        "model_results",
        "cohort_summary",
        "target_diagnostics",
    }


def _write_target_table(study_root: Path, subjects: list[str]) -> None:
    rows = []
    for subject_idx, subject in enumerate(subjects):
        for trial_idx, temp in enumerate(TRIAL_TEMPERATURES, start=1):
            rows.append(
                {
                    "subject_id": subject,
                    "task": "thermalactive",
                    "run": (trial_idx - 1) // 3 + 1,
                    "trial_index": trial_idx,
                    "within_run_trial": (trial_idx - 1) % 3 + 1,
                    "onset": float(trial_idx * 10),
                    "duration": 0.001,
                    "NPS": float(subject_idx + temp / 10.0 + ((trial_idx * 7) % 5) * 0.13),
                    "SIIPS1": float(
                        subject_idx * 100 + temp * 20.0 + ((trial_idx * 11) % 7) * 0.19
                    ),
                    "NPS_fmri_n_voxels": 1000,
                    "SIIPS1_fmri_n_voxels": 2000,
                    "hrf_weighted_framewise_displacement": 0.01 * trial_idx,
                    "hrf_weighted_std_dvars": 0.1 * trial_idx,
                    "hrf_weighted_fp1_fp2_high_frequency_power": 1e-15 * trial_idx,
                    "residual_ecg_coupling": 0.001 * trial_idx,
                    "stimulus_temp": temp,
                    "selected_surface": float(1 if trial_idx % 2 else 2),
                    "stimulus_temp_level_46_0": 1.0 if temp == 46.0 else 0.0,
                    "stimulus_temp_level_48_0": 1.0 if temp == 48.0 else 0.0,
                    "stimulus_temp_level_50_0": 1.0 if temp == 50.0 else 0.0,
                    "selected_surface_level_2_0": 1.0 if trial_idx % 2 == 0 else 0.0,
                }
            )
    target_dir = study_root / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target_dir / "primary_targets.parquet", index=False)


def _write_clean_events(deriv_root: Path, subjects: list[str]) -> None:
    for subject_idx, subject in enumerate(subjects):
        rows = []
        for trial_idx, temp in enumerate(TRIAL_TEMPERATURES, start=1):
            within_temperature_trial = (trial_idx - 1) % 3
            pain_report = within_temperature_trial == 1
            within_scale_intensity = subject_idx + temp + 2.0 * within_temperature_trial
            rows.append(
                {
                    "trial_id": trial_idx,
                    "onset": float(trial_idx * 10),
                    "duration": 0.001,
                    "run_id": (trial_idx - 1) // 3 + 1,
                    "trial_number": within_temperature_trial + 1,
                    "stimulus_temp": temp,
                    "selected_surface": float(1 if trial_idx % 2 else 2),
                    "pain_binary_coded": float(pain_report),
                    "vas_final_coded_rating": float(
                        within_scale_intensity + (100.0 if pain_report else 0.0)
                    ),
                    "residual_ecg_coupling": 0.001 * trial_idx,
                    "fp1_fp2_high_frequency_power": 1e-12 * trial_idx,
                }
            )
        event_dir = deriv_root / "preprocessed" / "eeg" / subject / "eeg"
        event_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            event_dir / f"{subject}_task-thermalactive_proc-clean_events.tsv",
            sep="\t",
            index=False,
        )


def _write_feature_benchmark_outputs(study_root: Path, subjects: list[str]) -> None:
    for target in PRIMARY_SIGNATURES:
        for feature_spec in PRIMARY_BAND_PRESETS:
            metrics_dir = (
                study_root
                / "feature_benchmark"
                / "primary"
                / target
                / feature_spec
                / "model_comparison"
                / "metrics"
            )
            metrics_dir.mkdir(parents=True, exist_ok=True)
            summary: dict[str, object] = {
                "subject_selection": {
                    "n_requested": len(subjects),
                    "n_included": len(subjects),
                    "n_excluded": 0,
                    "excluded_fraction": 0.0,
                }
            }
            fold_rows = []
            for model_idx, model in enumerate(FEATURE_MODELS):
                summary[model] = {
                    "mean_r2": 0.1 + model_idx,
                    "std_r2": 0.01,
                    "ci_low_r2": 0.0,
                    "ci_high_r2": 0.2,
                    "mean_nuisance_r2": 0.05,
                    "std_nuisance_r2": 0.01,
                    "mean_delta_r2": 0.05 + model_idx,
                    "std_delta_r2": 0.01,
                    "ci_low_delta_r2": 0.01,
                    "ci_high_delta_r2": 0.09,
                    "overall_r2": 0.08,
                    "mean_mae": 1.0,
                    "std_mae": 0.1,
                    "ci_low_mae": 0.8,
                    "ci_high_mae": 1.2,
                    "mean_nuisance_mae": 1.1,
                    "p_value_r2": 0.1,
                    "p_value_delta_r2": 0.1,
                    "n_perm": 10,
                    "n_perm_requested": 10,
                    "n_perm_completed": 10,
                    "n_perm_attempted": 10,
                    "n_invalid_permutations": 0,
                    "n_folds": len(subjects),
                }
                for fold, subject in enumerate(subjects):
                    fold_rows.append(
                        {
                            "model": model,
                            "fold": fold,
                            "test_subject": subject,
                            "r2": 0.1,
                            "r2_nuisance": 0.05,
                            "delta_r2": 0.05,
                            "mae": 1.0,
                            "best_params": "{'alpha': 1.0}",
                        }
                    )
            with open(
                metrics_dir / "model_comparison_summary.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(summary, handle)
            pd.DataFrame(fold_rows).to_csv(
                metrics_dir / "model_comparison.tsv",
                sep="\t",
                index=False,
            )
