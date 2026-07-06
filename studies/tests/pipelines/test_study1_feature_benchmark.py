from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from studies.tests.test_support import DotConfig

EXPLORATORY_FAMILIES = ["spectral", "erds"]


def _config(root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": str(root / "derivatives")},
            "project": {"random_state": 7},
            "time_frequency_analysis": {"active_window": [3.0, 10.5]},
            "study1": {
                "outputs": {"root_name": "study1"},
                "cohort": {"min_subjects": 2},
                "targets": {
                    "names": ["NPS", "SIIPS1"],
                    "method": "lss",
                    "contrast_name": "contrast",
                    "metric": "dot",
                    "normalization": "none",
                    "round_decimals": 3,
                },
                "feature_benchmark": {
                    "n_perm": 10,
                    "inner_splits": 3,
                    "outer_jobs": 1,
                    "feature_harmonization": "intersection",
                    "permutation_scheme": "circular_shift_within_run",
                    "max_invalid_permutation_fraction": 0.20,
                    "circular_shift": {
                        "min_valid_runs_per_subject": 3,
                        "min_retained_trials_per_subject": 25,
                    },
                },
                "features": {"exploratory_feature_families": list(EXPLORATORY_FAMILIES)},
            },
        }
    )


def _write_primary_targets(config: DotConfig) -> Path:
    target_path = (
        Path(config.get("paths.deriv_root"))
        / "group"
        / "multimodal"
        / "study1"
        / "targets"
        / "primary_targets.parquet"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "subject_id": ["sub-0001", "sub-0001", "sub-0002", "sub-0002"],
            "task": ["pain", "pain", "pain", "pain"],
            "run": [1, 1, 1, 1],
            "trial_index": [1, 2, 1, 2],
            "within_run_trial": [1, 2, 1, 2],
            "onset": [1.0, 2.0, 1.0, 2.0],
            "duration": [0.5, 0.5, 0.5, 0.5],
            "pain_binary_coded": [0, 1, 0, 1],
            "NPS": [1.0, 1.1, 1.2, 1.3],
            "SIIPS1": [2.0, 2.1, 2.2, 2.3],
        }
    ).to_parquet(target_path, index=False)
    return target_path


def _write_prepared_power_features(
    config: DotConfig,
    subject_id: str,
    *,
    primary_erp_subtraction: str | None = None,
    power_subtract_evoked: bool = False,
) -> None:
    _write_prepared_feature_rows(
        config,
        subject_id,
        "power",
        n_trials=2,
        primary_erp_subtraction=primary_erp_subtraction,
        power_subtract_evoked=power_subtract_evoked,
    )


def _write_prepared_feature(config: DotConfig, subject_id: str, family: str) -> None:
    _write_prepared_feature_rows(config, subject_id, family, n_trials=2)


def _write_prepared_temporal_power_features(
    config: DotConfig,
    subject_id: str,
    *,
    n_trials: int = 2,
) -> None:
    feature_dir = (
        Path(config.get("paths.deriv_root"))
        / "group"
        / "multimodal"
        / "study1"
        / "features_temporal_controls"
        / subject_id
        / "eeg"
        / "features"
        / "power"
    )
    metadata_dir = feature_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    trial_ids = np.arange(1, n_trials + 1, dtype=int)
    feature_values = trial_ids.astype(float)
    pd.DataFrame(
        {
            "trial_id": trial_ids,
            "power_prestimulus_wide_alpha_ch_Cz_log10raw": feature_values,
            "power_prestimulus_wide_beta_ch_Cz_log10raw": feature_values + 0.1,
            "power_prestimulus_wide_gamma_ch_Cz_log10raw": feature_values + 0.2,
            "power_ramp_up_alpha_ch_Cz_log10raw": feature_values + 0.3,
            "power_ramp_up_beta_ch_Cz_log10raw": feature_values + 0.4,
            "power_ramp_up_gamma_ch_Cz_log10raw": feature_values + 0.5,
        }
    ).to_parquet(feature_dir / "features_power.parquet", index=False)
    metadata = {
        "analysis_mode": "trial_ml_safe",
        "power_subtract_evoked": False,
        "precomputed_subtract_evoked": False,
        "aperiodic_subtract_evoked": False,
        "bands_use_iaf": False,
        "bursts_threshold_reference": "trial",
    }
    (metadata_dir / "extraction_config.json").write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )


def _write_prepared_feature_rows(
    config: DotConfig,
    subject_id: str,
    family: str,
    *,
    n_trials: int,
    primary_erp_subtraction: str | None = None,
    power_subtract_evoked: bool = False,
) -> None:
    feature_dir = (
        Path(config.get("paths.deriv_root"))
        / "group"
        / "multimodal"
        / "study1"
        / "features_trial_ml_safe"
        / subject_id
        / "eeg"
        / "features"
        / family
    )
    metadata_dir = feature_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    subject_numeric = int(subject_id.rsplit("-", maxsplit=1)[-1])
    trial_ids = np.arange(1, n_trials + 1, dtype=int)
    feature_values = trial_ids.astype(float) + subject_numeric * 0.1
    pd.DataFrame(
        {
            "trial_id": trial_ids,
            f"{family}_baseline_alpha_ch_Fp1_mean": feature_values,
            f"{family}_baseline_alpha_ch_Fp2_mean": feature_values[::-1],
            f"{family}_baseline_alpha_ch_Cz_mean": feature_values + 0.25,
            f"{family}_active_alpha_ch_Fp1_logratio": feature_values,
            f"{family}_active_alpha_ch_Fp2_logratio": feature_values[::-1],
            f"{family}_active_alpha_ch_Cz_logratio": feature_values + 0.25,
            f"{family}_active_alpha_roi_frontal_logratio_mean": feature_values + 0.50,
            f"{family}_active_alpha_global_logratio_mean": feature_values + 0.75,
            f"{family}_active_alpha_ch_Cz_db": (feature_values + 0.25) * 10.0,
            f"{family}_baseline_beta_ch_Fp1_mean": feature_values * 0.5,
            f"{family}_baseline_beta_ch_Cz_mean": feature_values * 0.75,
            f"{family}_active_beta_ch_Fp1_logratio": feature_values * 0.5,
            f"{family}_active_beta_ch_Cz_logratio": feature_values * 0.75,
        }
    ).to_parquet(
        feature_dir / f"features_{family}.parquet",
        index=False,
    )
    metadata = {
        "analysis_mode": "trial_ml_safe",
        "power_subtract_evoked": power_subtract_evoked,
        "precomputed_subtract_evoked": False,
        "aperiodic_subtract_evoked": False,
        "bands_use_iaf": False,
        "bursts_threshold_reference": "trial",
    }
    if primary_erp_subtraction is not None:
        metadata["primary_erp_subtraction"] = primary_erp_subtraction
    (metadata_dir / "extraction_config.json").write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )


def _write_clean_events(
    config: DotConfig,
    subject_id: str,
    *,
    task: str,
    n_trials: int,
) -> None:
    events_dir = Path(config.get("paths.deriv_root")) / "preprocessed" / "eeg" / subject_id / "eeg"
    events_dir.mkdir(parents=True, exist_ok=True)
    trial_ids = np.arange(1, n_trials + 1, dtype=int)
    pd.DataFrame(
        {
            "trial_id": trial_ids,
            "trial_number": trial_ids,
            "run": ((trial_ids - 1) // 3) + 1,
            "onset": trial_ids.astype(float) * 10.0,
            "duration": np.ones(n_trials, dtype=float),
        }
    ).to_csv(
        events_dir / f"{subject_id}_task-{task}_proc-clean_events.tsv",
        sep="\t",
        index=False,
    )


def _write_four_subject_targets(config: DotConfig, *, task: str, n_trials: int) -> list[str]:
    subjects = [f"sub-000{i}" for i in range(1, 5)]
    rows: list[dict[str, float | int | str]] = []
    for subject_number, subject_id in enumerate(subjects, start=1):
        for trial_id in range(1, n_trials + 1):
            rows.append(
                {
                    "subject_id": subject_id,
                    "task": task,
                    "run": ((trial_id - 1) // 3) + 1,
                    "trial_index": trial_id,
                    "within_run_trial": ((trial_id - 1) % 3) + 1,
                    "onset": float(trial_id * 10),
                    "duration": 1.0,
                    "NPS": float(subject_number + trial_id * 0.25),
                    "SIIPS1": float(subject_number * 2 + trial_id * 0.5),
                }
            )

    target_path = (
        Path(config.get("paths.deriv_root"))
        / "group"
        / "multimodal"
        / "study1"
        / "targets"
        / "primary_targets.parquet"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target_path, index=False)
    return subjects


def test_run_feature_benchmark_requires_prepared_features(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

    cfg = _config(tmp_path)
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.feature_benchmark.run_model_comparison_ml"
    ) as run_model_comparison:
        try:
            run_feature_benchmark(
                subjects=["0002", "0001"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except FileNotFoundError as exc:
            assert "prepare-features" in str(exc)
        else:
            raise AssertionError("Expected missing prepared features to raise FileNotFoundError.")

    run_model_comparison.assert_not_called()


def test_study1_prepared_power_features_must_not_claim_primary_erp_subtraction(
    tmp_path,
) -> None:
    from studies.pain_study.study1.prepare_features import require_prepared_study1_features

    cfg = _config(tmp_path)
    _write_prepared_power_features(
        cfg,
        "sub-0001",
        primary_erp_subtraction="fold_level_training_grand_average",
    )

    with pytest.raises(ValueError, match="primary_erp_subtraction"):
        require_prepared_study1_features(
            subjects=["sub-0001"],
            config=cfg,
            feature_families=["power"],
        )


def test_run_feature_benchmark_uses_study1_prepared_feature_root(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import (
        EXPLORATORY_BAND_PRESETS,
        PRIMARY_BAND_PRESETS,
        run_feature_benchmark,
    )

    cfg = _config(tmp_path)
    _write_primary_targets(cfg)
    _write_prepared_power_features(cfg, "sub-0001")
    _write_prepared_power_features(cfg, "sub-0002")
    _write_prepared_feature(cfg, "sub-0001", "spectral")
    _write_prepared_feature(cfg, "sub-0002", "spectral")
    _write_prepared_feature(cfg, "sub-0001", "erds")
    _write_prepared_feature(cfg, "sub-0002", "erds")
    captured_calls: list[dict] = []

    def _capture(**kwargs):
        captured_calls.append(kwargs)
        return Path(kwargs["results_root"]) / "model_comparison"

    with patch(
        "studies.pain_study.study1.feature_benchmark.run_model_comparison_ml",
        side_effect=_capture,
    ):
        outputs = run_feature_benchmark(
            subjects=["0002", "0001"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    expected_calls = 2 * (
        len(PRIMARY_BAND_PRESETS) + len(EXPLORATORY_BAND_PRESETS) + len(EXPLORATORY_FAMILIES)
    )
    assert len(outputs) == expected_calls
    assert len(captured_calls) == expected_calls
    first_call = captured_calls[0]
    assert first_call["subjects"] == ["sub-0001", "sub-0002"]
    assert first_call["target"] == "fmri_signature"
    assert first_call["feature_families"] == ["power"]
    assert first_call["feature_segments"] == ["active"]
    assert first_call["feature_scopes"] == ["ch"]
    assert first_call["feature_stats"] == ["logratio"]
    assert first_call["feature_input_root"] == (
        tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "features_trial_ml_safe"
    )
    assert first_call["config"].get("machine_learning.data.require_trial_ml_safe") is True
    assert first_call["config"].get("feature_engineering.analysis_mode") == "trial_ml_safe"
    assert (
        first_call["config"].get("machine_learning.cv.permutation_scheme")
        == "circular_shift_within_run"
    )
    assert first_call["model_names"] == ["elasticnet", "ridge"]
    assert first_call["config"].get("machine_learning.fmri_signature.target_table_path") == str(
        tmp_path
        / "derivatives"
        / "group"
        / "multimodal"
        / "study1"
        / "targets"
        / "primary_targets.parquet"
    )
    assert first_call["config"].get("machine_learning.fmri_signature.target_column") == "NPS"
    assert first_call["config"].get("machine_learning.target_residualization.enabled") is False
    assert first_call["results_root"].parts[-4:] == (
        "feature_benchmark",
        "primary",
        "NPS",
        "alpha",
    )
    exploratory_call = next(
        call
        for call in captured_calls
        if call["results_root"].parts[-4:]
        == ("feature_benchmark", "exploratory", "NPS", "spectral")
    )
    assert exploratory_call["feature_families"] == ["spectral"]
    assert exploratory_call["feature_bands"] is None
    assert exploratory_call["model_names"] == ["elasticnet", "ridge"]
    exploratory_delta_call = next(
        call
        for call in captured_calls
        if call["results_root"].parts[-4:]
        == ("feature_benchmark", "exploratory", "NPS", "delta")
    )
    assert exploratory_delta_call["feature_families"] == ["power"]
    assert exploratory_delta_call["feature_bands"] == ["delta"]


def test_feature_benchmark_config_requires_permutation_scheme(tmp_path) -> None:
    from eeg_pipeline.utils.config.loader import ConfigError
    from studies.pain_study.study1.feature_benchmark import feature_benchmark_config

    cfg = _config(tmp_path)
    cfg["study1"]["feature_benchmark"].pop("permutation_scheme")

    with pytest.raises(ConfigError, match="permutation_scheme"):
        feature_benchmark_config(cfg, target_name="NPS")


def test_run_feature_benchmark_passes_foldwise_nuisance_residualization(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import (
        EXPLORATORY_BAND_PRESETS,
        PRIMARY_BAND_PRESETS,
        run_feature_benchmark,
    )

    cfg = _config(tmp_path)
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    cfg["study1"]["targets"]["nuisance_regression"] = {
        "enabled": True,
        "continuous_columns": ["pain_binary_coded", "run", "onset"],
        "categorical_columns": [],
    }
    _write_primary_targets(cfg)
    _write_prepared_power_features(cfg, "sub-0001")
    _write_prepared_power_features(cfg, "sub-0002")
    captured_calls: list[dict] = []

    def _capture(**kwargs):
        captured_calls.append(kwargs)
        return Path(kwargs["results_root"]) / "model_comparison"

    with patch(
        "studies.pain_study.study1.feature_benchmark.run_model_comparison_ml",
        side_effect=_capture,
    ):
        run_feature_benchmark(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    expected_calls = 2 * (len(PRIMARY_BAND_PRESETS) + len(EXPLORATORY_BAND_PRESETS))
    assert len(captured_calls) == expected_calls
    assert captured_calls[0]["config"].get("machine_learning.fmri_signature.target_column") == "NPS"
    assert (
        captured_calls[0]["config"].get("machine_learning.target_residualization.enabled") is True
    )
    assert (
        captured_calls[0]["config"].get("machine_learning.target_residualization.strategy")
        == "staged_residual_learning"
    )
    assert captured_calls[0]["config"].get("machine_learning.target_residualization.columns") == [
        "pain_binary_coded",
        "run",
        "onset",
    ]


def test_run_feature_benchmark_adds_temporal_control_windows(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

    cfg = _config(tmp_path)
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    cfg["study1"]["temporal_negative_controls"] = {
        "feature_transform": "raw_log_power",
        "feature_baseline_window": None,
        "windows": {"prestimulus_wide": [-5.0, -0.01]},
        "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
        "plateau_windows": {"early_plateau": [3.0, 5.5]},
    }
    _write_primary_targets(cfg)
    for subject_id in ("sub-0001", "sub-0002"):
        _write_prepared_power_features(cfg, subject_id)
        _write_prepared_temporal_power_features(cfg, subject_id)
    captured_calls: list[dict] = []

    def _capture(**kwargs):
        captured_calls.append(kwargs)
        return Path(kwargs["results_root"]) / "model_comparison"

    with patch(
        "studies.pain_study.study1.feature_benchmark.run_model_comparison_ml",
        side_effect=_capture,
    ):
        run_feature_benchmark(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    prestimulus_call = next(
        call
        for call in captured_calls
        if call["results_root"].parts[-4:]
        == ("feature_benchmark", "temporal_control", "NPS", "temporal_prestimulus_wide")
    )
    assert prestimulus_call["feature_input_root"] == (
        tmp_path / "derivatives" / "group" / "multimodal" / "study1"
        / "features_temporal_controls"
    )
    assert prestimulus_call["feature_families"] == ["power"]
    assert prestimulus_call["feature_bands"] == [
        "alpha",
        "beta",
        "gamma_low_clean",
        "gamma_mid_clean",
        "gamma_high_clean",
    ]
    assert prestimulus_call["feature_segments"] == ["prestimulus_wide"]
    assert prestimulus_call["feature_scopes"] == ["ch"]
    assert prestimulus_call["feature_stats"] == ["log10raw"]
    assert prestimulus_call["model_names"] == ["elasticnet", "ridge"]
    plateau_call = next(
        call
        for call in captured_calls
        if call["results_root"].parts[-4:]
        == ("feature_benchmark", "temporal_control", "NPS", "temporal_early_plateau")
    )
    assert plateau_call["feature_segments"] == ["early_plateau"]


def test_run_feature_benchmark_uses_grouped_inner_cv_with_four_subjects(tmp_path) -> None:
    from studies.pain_study.study1 import feature_benchmark

    cfg = _config(tmp_path)
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    cfg["study1"]["feature_benchmark"]["n_perm"] = 1
    cfg["study1"]["feature_benchmark"]["inner_splits"] = 3
    cfg["study1"]["feature_benchmark"]["permutation_scheme"] = "within_subject"
    cfg["machine_learning"] = {
        "evaluation": {"bootstrap_iterations": 10},
        "models": {
            "elasticnet": {
                "alpha_grid": [0.1],
                "l1_ratio_grid": [0.5],
                "max_iter": 1000,
            },
            "ridge": {"alpha_grid": [0.1, 1.0]},
        },
        "preprocessing": {"variance_threshold_grid": [0.0]},
    }
    task = "pain"
    n_trials = 6
    subjects = _write_four_subject_targets(cfg, task=task, n_trials=n_trials)
    for subject_id in subjects:
        _write_clean_events(cfg, subject_id, task=task, n_trials=n_trials)
        _write_prepared_feature_rows(
            cfg,
            subject_id,
            "power",
            n_trials=n_trials,
        )

    with (
        patch.object(feature_benchmark, "PRIMARY_SIGNATURES", ("NPS",)),
        patch.object(feature_benchmark, "PRIMARY_BAND_PRESETS", {"alpha": ["alpha"]}),
        patch.object(feature_benchmark, "EXPLORATORY_BAND_PRESETS", {}),
    ):
        outputs = feature_benchmark.run_feature_benchmark(
            subjects=subjects,
            task=task,
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    assert len(outputs) == 1
    metrics_path = outputs[0] / "metrics" / "model_comparison.tsv"
    metrics = pd.read_csv(metrics_path, sep="\t")
    assert set(metrics["test_subject"]) == set(subjects)
    assert metrics.groupby("model")["fold"].nunique().to_dict() == {
        "elasticnet": 4,
        "ridge": 4,
    }
    assert not (metrics["best_params"].astype(str) == "{}").any()


def test_run_feature_benchmark_requires_permutation_inference(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

    cfg = _config(tmp_path)
    cfg["study1"]["feature_benchmark"]["n_perm"] = 0
    _write_primary_targets(cfg)
    _write_prepared_power_features(cfg, "sub-0001")
    _write_prepared_power_features(cfg, "sub-0002")

    with patch("studies.pain_study.study1.feature_benchmark.run_model_comparison_ml"):
        try:
            run_feature_benchmark(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except ValueError as exc:
            assert "n_perm" in str(exc)
        else:
            raise AssertionError("Expected Study 1 feature benchmark to require n_perm > 0.")


def test_run_feature_benchmark_requires_prepared_exploratory_features(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

    cfg = _config(tmp_path)
    _write_primary_targets(cfg)
    _write_prepared_power_features(cfg, "sub-0001")
    _write_prepared_power_features(cfg, "sub-0002")

    with patch("studies.pain_study.study1.feature_benchmark.run_model_comparison_ml"):
        try:
            run_feature_benchmark(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except FileNotFoundError as exc:
            assert "spectral" in str(exc)
        else:
            raise AssertionError("Expected missing exploratory feature families to be rejected.")


def test_run_feature_benchmark_cleans_appledouble_sidecars(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

    cfg = _config(tmp_path)
    _write_primary_targets(cfg)
    for subject_id in ("sub-0001", "sub-0002"):
        _write_prepared_power_features(cfg, subject_id)
        _write_prepared_feature(cfg, subject_id, "spectral")
        _write_prepared_feature(cfg, subject_id, "erds")

    def _capture(**kwargs):
        results_root = Path(kwargs["results_root"]) / "model_comparison"
        metrics_dir = results_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "model_comparison_summary.json").write_text(
            json.dumps({"elasticnet": {"mean_r2": -1.0, "mean_mae": 1.0, "n_folds": 2}}) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "._junk").write_text("", encoding="utf-8")
        return results_root

    with patch(
        "studies.pain_study.study1.feature_benchmark.run_model_comparison_ml",
        side_effect=_capture,
    ):
        run_feature_benchmark(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    benchmark_root = (
        tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "feature_benchmark"
    )
    assert not any(path.name.startswith("._") for path in benchmark_root.rglob("._*"))


def test_model_comparison_permutation_refits_full_pipeline_for_subject_mean_r2() -> None:
    from eeg_pipeline.analysis.machine_learning import orchestration

    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}).to_numpy(dtype=float)
    y = pd.Series([1.0, 2.0, 3.0, 4.0]).to_numpy(dtype=float)
    groups = pd.Series(["sub-0001", "sub-0001", "sub-0002", "sub-0002"]).to_numpy(dtype=object)
    meta = pd.DataFrame({"run": [1, 1, 1, 1]})
    outer_folds = [
        (pd.Series([2, 3]).to_numpy(dtype=int), pd.Series([0, 1]).to_numpy(dtype=int)),
        (pd.Series([0, 1]).to_numpy(dtype=int), pd.Series([2, 3]).to_numpy(dtype=int)),
    ]
    permuted = [
        pd.Series([2.0, 1.0, 4.0, 3.0]).to_numpy(dtype=float),
        pd.Series([1.5, 2.5, 3.5, 4.5]).to_numpy(dtype=float),
    ]
    refit_targets: list[list[float]] = []

    def _capture_cv(**kwargs):
        refit_targets.append(list(kwargs["y"]))
        return kwargs["y"].copy(), kwargs["y"].copy(), [{"r2": 1.0}]

    with (
        patch(
            "eeg_pipeline.analysis.machine_learning.orchestration._generate_effective_permutation",
            side_effect=[
                (permuted[0], True, 1.0, "within_subject"),
                (permuted[1], True, 1.0, "within_subject"),
            ],
        ),
        patch(
            "eeg_pipeline.analysis.machine_learning.orchestration.model_comparison_cv_predictions",
            side_effect=_capture_cv,
        ),
    ):
        p_value = orchestration._model_comparison_permutation_p_value(
            observed_mean_r2=1.0,
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            outer_folds=outer_folds,
            model_name="ridge",
            pipe=object(),
            param_grid={},
            inner_splits=2,
            outer_jobs=1,
            config=DotConfig(
                {"machine_learning": {"cv": {"permutation_scheme": "within_subject"}}}
            ),
            harmonization_mode="intersection",
            covariates=None,
            target_residualization_columns=tuple(),
            rng=np.random.default_rng(123),
            n_perm=2,
        )

    assert refit_targets == [list(permuted[0]), list(permuted[1])]
    assert p_value.p_value == 1.0
    assert p_value.n_perm_completed == 2
    assert p_value.n_perm_attempted == 2
    assert p_value.n_invalid_permutations == 0


def test_model_comparison_permutation_resamples_until_requested_valid_draws() -> None:
    from eeg_pipeline.analysis.machine_learning import orchestration

    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}).to_numpy(dtype=float)
    y = pd.Series([1.0, 2.0, 3.0, 4.0]).to_numpy(dtype=float)
    groups = pd.Series(["sub-0001", "sub-0001", "sub-0002", "sub-0002"]).to_numpy(dtype=object)
    meta = pd.DataFrame({"run": [1, 1, 1, 1]})
    outer_folds = [
        (pd.Series([2, 3]).to_numpy(dtype=int), pd.Series([0, 1]).to_numpy(dtype=int)),
        (pd.Series([0, 1]).to_numpy(dtype=int), pd.Series([2, 3]).to_numpy(dtype=int)),
    ]
    null_scores = [0.0, 2.0]

    def _capture_cv(**kwargs):
        return kwargs["y"].copy(), kwargs["y"].copy(), [{"r2": null_scores.pop(0)}]

    with (
        patch(
            "eeg_pipeline.analysis.machine_learning.orchestration._generate_effective_permutation",
            side_effect=[
                (y.copy(), False, 0.0, "within_subject"),
                (y + 10.0, True, 1.0, "within_subject"),
                (y + 20.0, True, 1.0, "within_subject"),
            ],
        ) as generate_permutation,
        patch(
            "eeg_pipeline.analysis.machine_learning.orchestration.model_comparison_cv_predictions",
            side_effect=_capture_cv,
        ) as refit_cv,
    ):
        p_value = orchestration._model_comparison_permutation_p_value(
            observed_mean_r2=1.0,
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            outer_folds=outer_folds,
            model_name="ridge",
            pipe=object(),
            param_grid={},
            inner_splits=2,
            outer_jobs=1,
            config=DotConfig(
                {"machine_learning": {"cv": {"permutation_scheme": "within_subject"}}}
            ),
            harmonization_mode="intersection",
            covariates=None,
            target_residualization_columns=tuple(),
            rng=np.random.default_rng(123),
            n_perm=2,
        )

    assert generate_permutation.call_count == 3
    assert refit_cv.call_count == 2
    assert p_value.p_value == pytest.approx(2 / 3)
    assert p_value.n_perm_completed == 2
    assert p_value.n_perm_attempted == 3
    assert p_value.n_invalid_permutations == 1


def test_model_comparison_staged_residual_learning_scores_raw_incremental_prediction() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        model_comparison_cv_predictions,
    )

    nuisance = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float), 2)
    eeg_signal = np.tile(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=float), 2)
    y = 10.0 + 2.0 * nuisance + 5.0 * eeg_signal
    groups = np.array(["sub-0001"] * 5 + ["sub-0002"] * 5, dtype=object)
    meta = pd.DataFrame({"nuisance": nuisance})
    outer_folds = [
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])),
        (np.array([5, 6, 7, 8, 9]), np.array([0, 1, 2, 3, 4])),
    ]

    y_true, y_pred, records = model_comparison_cv_predictions(
        model_name="linear",
        pipe=LinearRegression(),
        param_grid={},
        X=eeg_signal.reshape(-1, 1),
        y=y,
        groups=groups,
        meta=meta,
        outer_folds=outer_folds,
        inner_splits=2,
        outer_jobs=1,
        config=DotConfig(
            {
                "machine_learning": {
                    "target_residualization": {
                        "strategy": "staged_residual_learning",
                    }
                }
            }
        ),
        harmonization_mode="none",
        covariates=None,
        target_residualization_columns=("nuisance",),
        collect_records=True,
    )

    assert np.allclose(y_true, y)
    assert np.all(np.isfinite(y_pred))
    for record, (_train_idx, test_idx) in zip(records, outer_folds):
        expected_mae = float(np.mean(np.abs(y[test_idx] - y_pred[test_idx])))
        assert record["mae"] == pytest.approx(expected_mae)
    assert all(record["delta_r2"] > 0.0 for record in records)
    assert all(record["r2_nuisance"] < record["r2"] for record in records)


def test_model_comparison_fixed_params_skips_inner_cv_and_refits_frozen_params(monkeypatch) -> None:
    from sklearn.base import BaseEstimator, RegressorMixin

    from eeg_pipeline.analysis.machine_learning import orchestration
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        model_comparison_cv_predictions,
    )

    class _ConstantRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, alpha: float = 0.0) -> None:
            self.alpha = alpha

        def fit(self, X, y):
            self.fitted_alpha_ = self.alpha
            return self

        def predict(self, X):
            return np.full(len(X), self.alpha, dtype=float)

    def _fail_inner_cv(**_kwargs):
        raise AssertionError("inner CV must be skipped when fixed_params is provided")

    monkeypatch.setattr(orchestration, "_fit_subject_weighted_inner_cv_estimator", _fail_inner_cv)

    y = np.arange(6, dtype=float)
    groups = np.array(["sub-0001"] * 3 + ["sub-0002"] * 3, dtype=object)
    meta = pd.DataFrame({"nuisance": y})
    outer_folds = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]

    _y_true, y_pred, records = model_comparison_cv_predictions(
        model_name="constant",
        pipe=_ConstantRegressor(),
        param_grid={},
        X=np.zeros((6, 1), dtype=float),
        y=y,
        groups=groups,
        meta=meta,
        outer_folds=outer_folds,
        inner_splits=2,
        outer_jobs=1,
        config=DotConfig({"machine_learning": {}}),
        harmonization_mode="none",
        covariates=None,
        target_residualization_columns=(),
        collect_records=True,
        fixed_params={"alpha": 0.5},
    )

    assert records[0]["best_params"] == str({"alpha": 0.5})
    assert np.allclose(y_pred[outer_folds[0][1]], 0.5)


def test_model_comparison_staged_residual_learning_scores_raw_nuisance_model() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        model_comparison_cv_predictions,
    )

    nuisance = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
    y = np.exp(1.0 + 0.55 * nuisance) + np.array(
        [0.0, 0.2, -0.1, 0.1, 0.6, -0.3, 0.4, -0.2],
        dtype=float,
    )
    groups = np.array(["sub-0001"] * 4 + ["sub-0002"] * 4, dtype=object)
    meta = pd.DataFrame({"nuisance": nuisance})
    outer_folds = [
        (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
        (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3])),
    ]

    _y_true, _y_pred, records = model_comparison_cv_predictions(
        model_name="linear",
        pipe=LinearRegression(),
        param_grid={},
        X=np.zeros((len(y), 1), dtype=float),
        y=y,
        groups=groups,
        meta=meta,
        outer_folds=outer_folds,
        inner_splits=2,
        outer_jobs=1,
        config=DotConfig(
            {
                "machine_learning": {
                    "target_residualization": {
                        "strategy": "staged_residual_learning",
                    }
                }
            }
        ),
        harmonization_mode="none",
        covariates=None,
        target_residualization_columns=("nuisance",),
        collect_records=True,
    )

    for record, (train_idx, test_idx) in zip(records, outer_folds):
        design_train = np.column_stack([np.ones(len(train_idx), dtype=float), nuisance[train_idx]])
        design_test = np.column_stack([np.ones(len(test_idx), dtype=float), nuisance[test_idx]])
        coefficients, *_ = np.linalg.lstsq(design_train, y[train_idx], rcond=None)
        raw_nuisance_prediction = design_test @ coefficients
        ss_res = np.sum((y[test_idx] - raw_nuisance_prediction) ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[train_idx])) ** 2)
        expected_r2 = 1.0 - ss_res / ss_tot

        assert record["r2_nuisance"] == pytest.approx(expected_r2)


def test_model_comparison_summary_reports_staged_incremental_delta_r2(tmp_path) -> None:
    from eeg_pipeline.analysis.machine_learning import orchestration

    groups = np.array(["sub-0001", "sub-0001", "sub-0002", "sub-0002"], dtype=object)
    meta = pd.DataFrame({"nuisance": [0.0, 1.0, 0.0, 1.0]})
    records = [
        {
            "model": "ridge",
            "fold": 0,
            "test_subject": "sub-0001",
            "r2": 0.50,
            "r2_nuisance": 0.20,
            "delta_r2": 0.30,
            "mae": 1.0,
            "mae_nuisance": 1.4,
            "best_params": "{}",
        },
        {
            "model": "ridge",
            "fold": 1,
            "test_subject": "sub-0002",
            "r2": 0.70,
            "r2_nuisance": 0.10,
            "delta_r2": 0.60,
            "mae": 0.8,
            "mae_nuisance": 1.5,
            "best_params": "{}",
        },
    ]

    with (
        patch.object(
            orchestration,
            "load_active_matrix",
            return_value=(
                np.zeros((4, 1), dtype=float),
                np.arange(4, dtype=float),
                groups,
                ["feature"],
                meta,
            ),
        ),
        patch.object(orchestration, "export_subject_selection_report", return_value={}),
        patch.object(
            orchestration,
            "model_comparison_cv_predictions",
            return_value=(
                np.arange(4, dtype=float),
                np.arange(4, dtype=float),
                records,
            ),
        ),
        patch.object(orchestration, "write_reproducibility_info"),
        patch.object(orchestration, "_maybe_generate_mode_plots"),
    ):
        out_dir = orchestration.run_model_comparison_ml(
            subjects=["0001", "0002"],
            task="pain",
            deriv_root=tmp_path,
            config=DotConfig(
                {
                    "machine_learning": {
                        "target_residualization": {
                            "enabled": True,
                            "columns": ["nuisance"],
                            "strategy": "staged_residual_learning",
                        },
                        "evaluation": {"bootstrap_iterations": 10},
                    }
                }
            ),
            n_perm=0,
            inner_splits=2,
            outer_jobs=1,
            rng_seed=7,
            results_root=tmp_path,
            logger=logging.getLogger(__name__),
            model_names=["ridge"],
        )

    summary = json.loads(
        (out_dir / "metrics" / "model_comparison_summary.json").read_text(encoding="utf-8")
    )
    assert summary["ridge"]["mean_r2"] == pytest.approx(0.60)
    assert summary["ridge"]["mean_nuisance_r2"] == pytest.approx(0.15)
    assert summary["ridge"]["mean_delta_r2"] == pytest.approx(0.45)


def test_staged_permutation_reconstructs_raw_targets_from_shifted_residuals() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        reconstruct_staged_permutation_target_for_fold,
    )
    from eeg_pipeline.analysis.machine_learning.target_residualization import FoldNuisanceFit

    y = np.zeros(22, dtype=float)
    groups = np.array(["sub-0001"] * 11 + ["sub-0002"] * 11, dtype=object)
    runs = np.ones(22, dtype=float)
    trial_indices = np.tile(np.arange(1, 12, dtype=int), 2)
    meta = pd.DataFrame({"nuisance": np.arange(22, dtype=float)})
    train_idx = np.arange(0, 11, dtype=int)
    test_idx = np.arange(11, 22, dtype=int)
    nuisance_fit = FoldNuisanceFit(
        train_target=np.zeros(11, dtype=float),
        test_target=np.zeros(11, dtype=float),
        train_prediction=np.full(11, 100.0, dtype=float),
        test_prediction=np.full(11, 200.0, dtype=float),
        train_residual=np.arange(11, dtype=float),
        test_residual=np.arange(11, 22, dtype=float),
        details={"columns": ["nuisance"]},
    )

    def _shift_residuals(residuals, groups_arg, *, runs, trial_indices, rng, scheme):
        np.testing.assert_array_equal(groups_arg, groups)
        np.testing.assert_array_equal(residuals[train_idx], nuisance_fit.train_residual)
        np.testing.assert_array_equal(residuals[test_idx], nuisance_fit.test_residual)
        return residuals + 1.0

    with (
        patch(
            "eeg_pipeline.analysis.machine_learning.orchestration.fit_nuisance_model_for_fold",
            return_value=nuisance_fit,
        ),
        patch(
            "eeg_pipeline.analysis.machine_learning.orchestration._permute_labels_by_scheme",
            side_effect=_shift_residuals,
        ),
    ):
        y_perm = reconstruct_staged_permutation_target_for_fold(
            y=y,
            groups=groups,
            meta=meta,
            train_idx=train_idx,
            test_idx=test_idx,
            columns=("nuisance",),
            runs=runs,
            trial_indices=trial_indices,
            rng=np.random.default_rng(7),
            scheme="circular_shift_within_run",
        )

    np.testing.assert_array_equal(
        y_perm[train_idx],
        nuisance_fit.train_prediction + nuisance_fit.train_residual + 1.0,
    )
    np.testing.assert_array_equal(
        y_perm[test_idx],
        nuisance_fit.test_prediction + nuisance_fit.test_residual + 1.0,
    )


def test_model_comparison_permutation_reconstructs_staged_targets_per_outer_fold() -> None:
    from eeg_pipeline.analysis.machine_learning import orchestration

    X = np.zeros((22, 1), dtype=float)
    y = np.arange(22, dtype=float)
    groups = np.array(["sub-0001"] * 11 + ["sub-0002"] * 11, dtype=object)
    meta = pd.DataFrame(
        {
            "run": np.ones(22, dtype=int),
            "trial_index": np.tile(np.arange(1, 12, dtype=int), 2),
            "nuisance": np.arange(22, dtype=float),
        }
    )
    outer_folds = [
        (np.arange(0, 11, dtype=int), np.arange(11, 22, dtype=int)),
        (np.arange(11, 22, dtype=int), np.arange(0, 11, dtype=int)),
    ]
    reconstructed_targets = [y + 10.0, y + 20.0]
    captured_refit_targets: list[np.ndarray] = []

    def _capture_cv(**kwargs):
        captured_refit_targets.append(np.asarray(kwargs["y"], dtype=float).copy())
        return kwargs["y"].copy(), kwargs["y"].copy(), [{"delta_r2": 0.5}]

    with (
        patch.object(
            orchestration,
            "reconstruct_staged_permutation_target_for_fold",
            side_effect=reconstructed_targets,
        ) as reconstruct,
        patch.object(
            orchestration,
            "model_comparison_cv_predictions",
            side_effect=_capture_cv,
        ),
    ):
        p_value = orchestration._model_comparison_permutation_p_value(
            observed_mean_r2=0.5,
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            outer_folds=outer_folds,
            model_name="ridge",
            pipe=object(),
            param_grid={},
            inner_splits=2,
            outer_jobs=1,
            config=DotConfig(
                {
                    "machine_learning": {
                        "cv": {"permutation_scheme": "circular_shift_within_run"},
                        "target_residualization": {
                            "strategy": "staged_residual_learning",
                        },
                    }
                }
            ),
            harmonization_mode="intersection",
            covariates=None,
            target_residualization_columns=("nuisance",),
            rng=np.random.default_rng(123),
            n_perm=1,
            score_column="delta_r2",
        )

    assert reconstruct.call_count == 2
    np.testing.assert_array_equal(captured_refit_targets[0], y + 10.0)
    np.testing.assert_array_equal(captured_refit_targets[1], y + 20.0)
    assert p_value.p_value == 1.0
    assert p_value.n_perm_completed == 1
    assert p_value.n_perm_attempted == 1


def test_circular_shift_within_run_requires_runs() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _permute_labels_by_scheme

    y = np.arange(11, dtype=float)
    groups = np.array(["sub-0001"] * len(y), dtype=object)

    with pytest.raises(ValueError, match="circular_shift_within_run.*requires run labels"):
        _permute_labels_by_scheme(
            y,
            groups,
            runs=None,
            rng=np.random.default_rng(1),
            scheme="circular_shift_within_run",
        )


def test_circular_shift_within_run_requires_trial_indices() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _permute_labels_by_scheme

    y = np.arange(11, dtype=float)
    groups = np.array(["sub-0001"] * len(y), dtype=object)
    runs = np.ones(len(y), dtype=float)

    with pytest.raises(ValueError, match="circular_shift_within_run.*trial indices"):
        _permute_labels_by_scheme(
            y,
            groups,
            runs=runs,
            rng=np.random.default_rng(1),
            scheme="circular_shift_within_run",
        )


def test_admissible_circular_shifts_follow_original_trial_distance_rule() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _admissible_circular_shifts

    complete_run = np.arange(1, 12, dtype=int)
    assert _admissible_circular_shifts(complete_run) == (5, 6, 7, 8, 9, 10)
    assert _admissible_circular_shifts(np.arange(1, 8, dtype=int)) == tuple()


def test_circular_shift_within_run_preserves_subject_run_label_sets() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _permute_labels_by_scheme

    y = np.arange(22, dtype=float)
    groups = np.array(["sub-0001"] * len(y), dtype=object)
    runs = np.repeat([1.0, 2.0], 11)
    trial_indices = np.tile(np.arange(1, 12), 2)

    y_perm = _permute_labels_by_scheme(
        y,
        groups,
        runs=runs,
        trial_indices=trial_indices,
        rng=np.random.default_rng(1),
        scheme="circular_shift_within_run",
    )

    first_run = runs == 1.0
    second_run = runs == 2.0
    assert set(y_perm[first_run]) == set(y[first_run])
    assert set(y_perm[second_run]) == set(y[second_run])
    assert not np.array_equal(y_perm, y)


def test_circular_shift_trial_structure_filter_excludes_invalid_runs_and_subjects() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        filter_circular_shift_permutation_rows,
    )

    rows: list[dict[str, object]] = []
    for subject, run_lengths in (
        ("sub-0001", [9, 9, 9, 7]),
        ("sub-0002", [9, 9, 7]),
    ):
        for run, run_length in enumerate(run_lengths, start=1):
            for trial_index in range(1, run_length + 1):
                rows.append(
                    {
                        "subject_id": subject,
                        "run": run,
                        "trial_index": trial_index,
                    }
                )

    frame = pd.DataFrame(rows)
    groups = frame["subject_id"].to_numpy(dtype=object)
    X = np.arange(len(frame), dtype=float).reshape(-1, 1)
    y = np.arange(len(frame), dtype=float)
    meta = frame[["run", "trial_index"]].copy()

    X_filtered, y_filtered, groups_filtered, meta_filtered = (
        filter_circular_shift_permutation_rows(
            X=X,
            y=y,
            groups=groups,
            meta=meta,
            config=DotConfig(
                {
                    "machine_learning": {
                        "cv": {
                            "permutation_scheme": "circular_shift_within_run",
                            "circular_shift": {
                                "min_valid_runs_per_subject": 3,
                                "min_retained_trials_per_subject": 25,
                            },
                        }
                    }
                }
            ),
            logger=logging.getLogger(__name__),
        )
    )

    assert len(X_filtered) == 27
    assert len(y_filtered) == 27
    assert set(groups_filtered) == {"sub-0001"}
    assert set(meta_filtered["run"]) == {1, 2, 3}
    excluded_subjects = meta_filtered.attrs["excluded_subjects"]
    assert excluded_subjects == [
        {
            "subject_id": "sub-0002",
            "reason": (
                "Excluded by circular-shift permutation structure: "
                "valid_runs=2, retained_trials=18."
            ),
        }
    ]
