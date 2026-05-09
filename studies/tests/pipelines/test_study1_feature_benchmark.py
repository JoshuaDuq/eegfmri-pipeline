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
            "block": [1, 1, 1, 1],
            "trial_index": [1, 2, 1, 2],
            "onset": [1.0, 2.0, 1.0, 2.0],
            "duration": [0.5, 0.5, 0.5, 0.5],
            "pain_binary_coded": [0, 1, 0, 1],
            "NPS": [1.0, 1.1, 1.2, 1.3],
            "SIIPS1": [2.0, 2.1, 2.2, 2.3],
        }
    ).to_parquet(target_path, index=False)
    return target_path


def _write_prepared_power_features(config: DotConfig, subject_id: str) -> None:
    _write_prepared_feature(config, subject_id, "power")


def _write_prepared_feature(config: DotConfig, subject_id: str, family: str) -> None:
    _write_prepared_feature_rows(config, subject_id, family, n_trials=2)


def _write_prepared_feature_rows(
    config: DotConfig,
    subject_id: str,
    family: str,
    *,
    n_trials: int,
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
            f"{family}_baseline_beta_ch_Fp1_mean": feature_values * 0.5,
        }
    ).to_parquet(
        feature_dir / f"features_{family}.parquet",
        index=False,
    )
    (metadata_dir / "extraction_config.json").write_text(
        json.dumps(
            {
                "analysis_mode": "trial_ml_safe",
                "power_subtract_evoked": False,
                "precomputed_subtract_evoked": False,
                "aperiodic_subtract_evoked": False,
                "bands_use_iaf": False,
                "bursts_threshold_reference": "trial",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_clean_events(
    config: DotConfig,
    subject_id: str,
    *,
    task: str,
    n_trials: int,
) -> None:
    events_dir = (
        Path(config.get("paths.deriv_root"))
        / "preprocessed"
        / "eeg"
        / subject_id
        / "eeg"
    )
    events_dir.mkdir(parents=True, exist_ok=True)
    trial_ids = np.arange(1, n_trials + 1, dtype=int)
    pd.DataFrame(
        {
            "trial_id": trial_ids,
            "trial_number": trial_ids,
            "block": ((trial_ids - 1) // 3) + 1,
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
                    "block": ((trial_id - 1) // 3) + 1,
                    "trial_index": trial_id,
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


def test_run_feature_benchmark_uses_study1_prepared_feature_root(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

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

    assert len(outputs) == 10
    assert len(captured_calls) == 10
    first_call = captured_calls[0]
    assert first_call["subjects"] == ["sub-0001", "sub-0002"]
    assert first_call["target"] == "fmri_signature"
    assert first_call["feature_families"] == ["power"]
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


def test_run_feature_benchmark_passes_foldwise_nuisance_residualization(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

    cfg = _config(tmp_path)
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    cfg["study1"]["targets"]["nuisance_regression"] = {
        "enabled": True,
        "continuous_columns": ["pain_binary_coded", "block", "onset"],
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

    assert len(captured_calls) == 6
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
        "block",
        "onset",
    ]


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
        _write_prepared_feature_rows(cfg, subject_id, "power", n_trials=n_trials)

    with (
        patch.object(feature_benchmark, "PRIMARY_SIGNATURES", ("NPS",)),
        patch.object(feature_benchmark, "PRIMARY_BAND_PRESETS", {"alpha": ["alpha"]}),
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
    meta = pd.DataFrame({"block": [1, 1, 1, 1]})
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
            "eeg_pipeline.analysis.machine_learning.orchestration._model_comparison_cv_predictions",
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
    assert p_value == 1.0


def test_model_comparison_permutation_resamples_until_requested_valid_draws() -> None:
    from eeg_pipeline.analysis.machine_learning import orchestration

    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}).to_numpy(dtype=float)
    y = pd.Series([1.0, 2.0, 3.0, 4.0]).to_numpy(dtype=float)
    groups = pd.Series(["sub-0001", "sub-0001", "sub-0002", "sub-0002"]).to_numpy(dtype=object)
    meta = pd.DataFrame({"block": [1, 1, 1, 1]})
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
            "eeg_pipeline.analysis.machine_learning.orchestration._model_comparison_cv_predictions",
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
    assert p_value == pytest.approx(2 / 3)


def test_model_comparison_staged_residual_learning_scores_raw_incremental_prediction() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        _model_comparison_cv_predictions,
    )

    nuisance = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], dtype=float)
    eeg_signal = np.array([1.0, -2.0, 1.0, 2.0, -4.0, 2.0], dtype=float)
    y = 10.0 + 2.0 * nuisance + 5.0 * eeg_signal
    groups = np.array(["sub-0001"] * 3 + ["sub-0002"] * 3, dtype=object)
    meta = pd.DataFrame({"nuisance": nuisance})
    outer_folds = [
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
    ]

    y_true, y_pred, records = _model_comparison_cv_predictions(
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
            "_model_comparison_cv_predictions",
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
        _reconstruct_staged_permutation_target_for_fold,
    )
    from eeg_pipeline.analysis.machine_learning.target_residualization import FoldNuisanceFit

    y = np.zeros(22, dtype=float)
    groups = np.array(["sub-0001"] * 11 + ["sub-0002"] * 11, dtype=object)
    blocks = np.ones(22, dtype=float)
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

    def _shift_residuals(residuals, groups_arg, *, blocks, trial_indices, rng, scheme):
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
        y_perm = _reconstruct_staged_permutation_target_for_fold(
            y=y,
            groups=groups,
            meta=meta,
            train_idx=train_idx,
            test_idx=test_idx,
            columns=("nuisance",),
            blocks=blocks,
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
            "block": np.ones(22, dtype=int),
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
            "_reconstruct_staged_permutation_target_for_fold",
            side_effect=reconstructed_targets,
        ) as reconstruct,
        patch.object(
            orchestration,
            "_model_comparison_cv_predictions",
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
    assert p_value == 1.0


def test_circular_shift_within_run_requires_blocks() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _permute_labels_by_scheme

    y = np.arange(11, dtype=float)
    groups = np.array(["sub-0001"] * len(y), dtype=object)

    with pytest.raises(ValueError, match="circular_shift_within_run.*requires block labels"):
        _permute_labels_by_scheme(
            y,
            groups,
            blocks=None,
            rng=np.random.default_rng(1),
            scheme="circular_shift_within_run",
        )


def test_circular_shift_within_run_requires_trial_indices() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _permute_labels_by_scheme

    y = np.arange(11, dtype=float)
    groups = np.array(["sub-0001"] * len(y), dtype=object)
    blocks = np.ones(len(y), dtype=float)

    with pytest.raises(ValueError, match="circular_shift_within_run.*trial indices"):
        _permute_labels_by_scheme(
            y,
            groups,
            blocks=blocks,
            rng=np.random.default_rng(1),
            scheme="circular_shift_within_run",
        )


def test_admissible_circular_shifts_follow_original_trial_distance_rule() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _admissible_circular_shifts

    complete_block = np.arange(1, 12, dtype=int)
    assert _admissible_circular_shifts(complete_block) == (5, 6, 7, 8, 9, 10)
    assert _admissible_circular_shifts(np.arange(1, 8, dtype=int)) == tuple()


def test_circular_shift_within_run_preserves_subject_block_label_sets() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import _permute_labels_by_scheme

    y = np.arange(22, dtype=float)
    groups = np.array(["sub-0001"] * len(y), dtype=object)
    blocks = np.repeat([1.0, 2.0], 11)
    trial_indices = np.tile(np.arange(1, 12), 2)

    y_perm = _permute_labels_by_scheme(
        y,
        groups,
        blocks=blocks,
        trial_indices=trial_indices,
        rng=np.random.default_rng(1),
        scheme="circular_shift_within_run",
    )

    first_block = blocks == 1.0
    second_block = blocks == 2.0
    assert set(y_perm[first_block]) == set(y[first_block])
    assert set(y_perm[second_block]) == set(y[second_block])
    assert not np.array_equal(y_perm, y)


def test_circular_shift_trial_structure_filter_excludes_invalid_blocks_and_subjects() -> None:
    from eeg_pipeline.analysis.machine_learning.orchestration import (
        _filter_circular_shift_permutation_rows,
    )

    rows: list[dict[str, object]] = []
    for subject, block_lengths in (
        ("sub-0001", [9, 9, 9, 7]),
        ("sub-0002", [9, 9, 7]),
    ):
        for block_id, block_length in enumerate(block_lengths, start=1):
            for trial_index in range(1, block_length + 1):
                rows.append(
                    {
                        "subject_id": subject,
                        "block": block_id,
                        "trial_index": trial_index,
                    }
                )

    frame = pd.DataFrame(rows)
    groups = frame["subject_id"].to_numpy(dtype=object)
    X = np.arange(len(frame), dtype=float).reshape(-1, 1)
    y = np.arange(len(frame), dtype=float)
    meta = frame[["block", "trial_index"]].copy()

    X_filtered, y_filtered, groups_filtered, meta_filtered = (
        _filter_circular_shift_permutation_rows(
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
                                "min_valid_blocks_per_subject": 3,
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
    assert set(meta_filtered["block"]) == {1, 2, 3}
    excluded_subjects = meta_filtered.attrs["excluded_subjects"]
    assert excluded_subjects == [
        {
            "subject_id": "sub-0002",
            "reason": (
                "Excluded by circular-shift permutation structure: "
                "valid_blocks=2, retained_trials=18."
            ),
        }
    ]
