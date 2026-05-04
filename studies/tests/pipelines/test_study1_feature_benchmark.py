from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd

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
            "NPS": [1.0, 1.1, 1.2, 1.3],
            "SIIPS1": [2.0, 2.1, 2.2, 2.3],
        }
    ).to_parquet(target_path, index=False)
    return target_path


def _write_prepared_power_features(config: DotConfig, subject_id: str) -> None:
    _write_prepared_feature(config, subject_id, "power")


def _write_prepared_feature(config: DotConfig, subject_id: str, family: str) -> None:
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
    pd.DataFrame({f"{family}_analysis_feature": [1.0, 1.1]}).to_parquet(
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

    assert len(outputs) == 12
    assert len(captured_calls) == 12
    first_call = captured_calls[0]
    assert first_call["subjects"] == ["sub-0001", "sub-0002"]
    assert first_call["target"] == "fmri_signature"
    assert first_call["feature_families"] == ["power"]
    assert first_call["feature_input_root"] == (
        tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "features_trial_ml_safe"
    )
    assert first_call["config"].get("machine_learning.data.require_trial_ml_safe") is True
    assert first_call["config"].get("feature_engineering.analysis_mode") == "trial_ml_safe"
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
        if call["results_root"].parts[-4:] == ("feature_benchmark", "exploratory", "NPS", "spectral")
    )
    assert exploratory_call["feature_families"] == ["spectral"]
    assert exploratory_call["feature_bands"] is None


def test_run_feature_benchmark_passes_foldwise_nuisance_residualization(tmp_path) -> None:
    from studies.pain_study.study1.feature_benchmark import run_feature_benchmark

    cfg = _config(tmp_path)
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    cfg["study1"]["targets"]["nuisance_regression"] = {
        "enabled": True,
        "columns": ["pain_binary_coded", "block", "onset"],
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

    assert len(captured_calls) == 8
    assert captured_calls[0]["config"].get("machine_learning.fmri_signature.target_column") == "NPS"
    assert captured_calls[0]["config"].get("machine_learning.target_residualization.enabled") is True
    assert captured_calls[0]["config"].get("machine_learning.target_residualization.columns") == [
        "pain_binary_coded",
        "block",
        "onset",
    ]


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
