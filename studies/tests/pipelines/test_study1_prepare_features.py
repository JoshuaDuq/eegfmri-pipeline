from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from studies.pain_study.study1.runner import SignaturePredictionRunner
from studies.tests.test_support import DotConfig

EXPLORATORY_FAMILIES = [
    "spectral",
    "aperiodic",
    "erds",
    "ratios",
    "asymmetry",
    "complexity",
    "bursts",
]
ALL_FAMILIES = ["power", *EXPLORATORY_FAMILIES]


def _config(deriv_root: Path) -> DotConfig:
    return DotConfig(
        {
            "paths": {"deriv_root": str(deriv_root)},
            "feature_engineering": {
                "power": {"subtract_evoked": True},
                "precomputed": {"subtract_evoked": None},
                "aperiodic": {"subtract_evoked": True},
                "bands": {"use_iaf": True},
                "bursts": {"threshold_reference": "subject"},
            },
            "time_frequency_analysis": {
                "baseline_window": [-5.0, -0.01],
                "active_window": [3.0, 10.5],
            },
            "study1": {
                "outputs": {"root_name": "study1"},
                "cohort": {"min_subjects": 2},
                "targets": {"names": ["NPS", "SIIPS1"]},
                "features": {
                    "exploratory_feature_families": list(EXPLORATORY_FAMILIES),
                },
                "temporal_negative_controls": {
                    "feature_transform": "raw_log_power",
                    "feature_baseline_window": None,
                    "windows": {
                        "prestimulus_wide": [-5.0, 0.0],
                        "immediate_prestimulus": [-0.2, 0.0],
                    },
                    "wrong_lag_windows": {
                        "ramp_up": [0.0, 3.0],
                        "late_ramp_down": [10.5, 15.0],
                    },
                },
            },
        }
    )


def _write_primary_targets(config: DotConfig) -> None:
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
            "subject_id": ["sub-0001", "sub-0002"],
            "task": ["pain", "pain"],
            "block": [1, 1],
            "trial_index": [1, 1],
            "within_block_trial": [1, 1],
            "onset": [0.0, 0.0],
            "duration": [1.0, 1.0],
            "NPS": [1.0, 1.1],
            "SIIPS1": [2.0, 2.1],
        }
    ).to_parquet(target_path, index=False)


def test_study1_feature_root_is_under_group_multimodal(tmp_path) -> None:
    from studies.pain_study.study1.cohort import study1_feature_root

    root = study1_feature_root(_config(tmp_path / "derivatives"))
    assert (
        root
        == tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "features_trial_ml_safe"
    )


def _write_feature_output(
    feature_root: Path,
    subject_id: str,
    family: str,
    *,
    analysis_mode: str = "trial_ml_safe",
    power_subtract_evoked: bool = False,
    precomputed_subtract_evoked: bool = False,
    aperiodic_subtract_evoked: bool = False,
    bands_use_iaf: bool = False,
    bursts_threshold_reference: str = "trial",
    primary_erp_subtraction: str | None = None,
) -> None:
    family_dir = feature_root / subject_id / "eeg" / "features" / family
    metadata_dir = family_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"{family}_feature": [1.0]}).to_parquet(
        family_dir / f"features_{family}.parquet",
        index=False,
    )
    payload = {
        "analysis_mode": analysis_mode,
        "power_subtract_evoked": power_subtract_evoked,
        "precomputed_subtract_evoked": precomputed_subtract_evoked,
        "aperiodic_subtract_evoked": aperiodic_subtract_evoked,
        "bands_use_iaf": bands_use_iaf,
        "bursts_threshold_reference": bursts_threshold_reference,
    }
    if family == "power" and primary_erp_subtraction is not None:
        payload["primary_erp_subtraction"] = primary_erp_subtraction
    (metadata_dir / "extraction_config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_power_active_feature_output(
    feature_root: Path,
    subject_id: str,
) -> None:
    _write_feature_output(feature_root, subject_id, "power")
    family_dir = feature_root / subject_id / "eeg" / "features" / "power"
    pd.DataFrame({"trial_id": [1], "power_active_alpha_ch_Cz_logratio": [1.0]}).to_parquet(
        family_dir / "features_power.parquet",
        index=False,
    )


def _write_temporal_power_feature_output(
    feature_root: Path,
    subject_id: str,
) -> None:
    _write_feature_output(feature_root, subject_id, "power")
    family_dir = feature_root / subject_id / "eeg" / "features" / "power"
    pd.DataFrame(
        {"trial_id": [1], "power_prestimulus_wide_alpha_ch_Cz_log10raw": [1.0]}
    ).to_parquet(
        family_dir / "features_power.parquet",
        index=False,
    )


def _write_all_prepared_outputs(
    feature_root: Path, subjects: list[str], families: list[str]
) -> None:
    for subject_id in subjects:
        for family in families:
            _write_feature_output(feature_root, subject_id, family)


def _write_windowed_duplicate_outputs(
    feature_root: Path,
    subject_id: str,
    family: str,
) -> None:
    family_dir = feature_root / subject_id / "eeg" / "features" / family
    metadata_dir = family_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({f"{family}_active_feature": [1.0]})
    frame.to_parquet(family_dir / f"features_{family}.parquet", index=False)
    frame.to_parquet(family_dir / f"features_{family}_active.parquet", index=False)
    payload = {
        "analysis_mode": "trial_ml_safe",
        "power_subtract_evoked": False,
        "precomputed_subtract_evoked": False,
        "aperiodic_subtract_evoked": False,
        "bands_use_iaf": False,
        "bursts_threshold_reference": "trial",
    }
    for name in (
        "extraction_config.json",
        "extraction_config_active.json",
        "extraction_config_baseline.json",
        f"features_{family}.json",
        f"features_{family}_active.json",
    ):
        (metadata_dir / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (family_dir / "._sidecar").write_text("", encoding="utf-8")
    (metadata_dir / "._metadata_sidecar").write_text("", encoding="utf-8")


def test_prepare_study1_features_uses_study1_feature_root_and_full_family_set(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            _write_all_prepared_outputs(
                feature_root,
                list(kwargs["subjects"]),
                list(kwargs["feature_categories"]),
            )
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        prepare_study1_features(
            subjects=["0002", "0001"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    assert feature_pipeline.run_batch.call_count == 2
    call = feature_pipeline.run_batch.call_args_list[0]
    temporal_call = feature_pipeline.run_batch.call_args_list[1]

    assert call.kwargs == {
        "subjects": ["sub-0001", "sub-0002"],
        "task": "pain",
        "fail_fast": True,
        "analysis_mode": "trial_ml_safe",
        "feature_categories": ALL_FAMILIES,
        "feature_output_root": (
            tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "features_trial_ml_safe"
        ),
        "save_canonical_trial_table": False,
        "time_ranges": [
            {"name": "baseline", "tmin": -5.0, "tmax": -0.01},
            {"name": "active", "tmin": 3.0, "tmax": 10.5},
        ],
    }
    assert temporal_call.kwargs["feature_categories"] == ["power"]
    assert temporal_call.kwargs["feature_output_root"] == (
        tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "features_temporal_controls"
    )
    assert temporal_call.kwargs["time_ranges"] == [
        {"name": "prestimulus_wide", "tmin": -5.0, "tmax": 0.0},
        {"name": "immediate_prestimulus", "tmin": -0.2, "tmax": 0.0},
        {"name": "ramp_up", "tmin": 0.0, "tmax": 3.0},
        {"name": "late_ramp_down", "tmin": 10.5, "tmax": 15.0},
    ]


def test_prepare_study1_features_extracts_primary_power_on_active_window(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            for subject_id in kwargs["subjects"]:
                _write_power_active_feature_output(feature_root, subject_id)
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        prepare_study1_features(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    assert feature_pipeline.run_batch.call_count == 2
    primary_call = feature_pipeline.run_batch.call_args_list[0]
    assert primary_call.kwargs["feature_categories"] == ["power"]
    assert primary_call.kwargs["time_ranges"] == [
        {"name": "baseline", "tmin": -5.0, "tmax": -0.01},
        {"name": "active", "tmin": 3.0, "tmax": 10.5},
    ]


def test_prepare_study1_features_extracts_temporal_control_raw_power_windows(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        primary_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            for subject_id in kwargs["subjects"]:
                if feature_root.name == "features_trial_ml_safe":
                    _write_power_active_feature_output(feature_root, subject_id)
                else:
                    _write_temporal_power_feature_output(feature_root, subject_id)
            return []

        primary_pipeline.run_batch.side_effect = _write_outputs

        prepare_study1_features(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    assert primary_pipeline.run_batch.call_count == 2
    temporal_call = primary_pipeline.run_batch.call_args_list[1]
    assert temporal_call.kwargs["feature_categories"] == ["power"]
    assert temporal_call.kwargs["feature_output_root"] == (
        tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "features_temporal_controls"
    )
    assert temporal_call.kwargs["time_ranges"] == [
        {"name": "prestimulus_wide", "tmin": -5.0, "tmax": 0.0},
        {"name": "immediate_prestimulus", "tmin": -0.2, "tmax": 0.0},
        {"name": "ramp_up", "tmin": 0.0, "tmax": 3.0},
        {"name": "late_ramp_down", "tmin": 10.5, "tmax": 15.0},
    ]
    temporal_config = feature_pipeline_cls.call_args.kwargs["config"]
    assert temporal_config.get("feature_engineering.power.require_baseline") is False


def test_signature_prediction_runner_dispatches_prepare_features(tmp_path) -> None:
    cfg = _config(tmp_path / "derivatives")
    runner = SignaturePredictionRunner(config=cfg)

    with patch("studies.pain_study.study1.runner.prepare_study1_features") as prepare_features:
        runner.run(mode="prepare-features", subjects=["0001"], task="pain")

    prepare_features.assert_called_once_with(
        subjects=["0001"],
        task="pain",
        config=cfg,
        logger=runner.logger,
    )


def test_prepare_study1_features_pins_trial_safe_family_overrides(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            _write_all_prepared_outputs(
                feature_root,
                list(kwargs["subjects"]),
                list(kwargs["feature_categories"]),
            )
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        prepare_study1_features(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    pipeline_config = feature_pipeline_cls.call_args.kwargs["config"]
    assert pipeline_config.get("feature_engineering.power.subtract_evoked") is False
    assert pipeline_config.get("feature_engineering.precomputed.subtract_evoked") is False
    assert pipeline_config.get("feature_engineering.aperiodic.subtract_evoked") is False
    assert pipeline_config.get("feature_engineering.bands.use_iaf") is False
    assert pipeline_config.get("feature_engineering.bursts.threshold_reference") == "trial"
    assert cfg.get("feature_engineering.power.subtract_evoked") is True
    assert cfg.get("feature_engineering.aperiodic.subtract_evoked") is True
    assert cfg.get("feature_engineering.bands.use_iaf") is True
    assert cfg.get("feature_engineering.bursts.threshold_reference") == "subject"


def test_prepare_study1_features_rejects_power_evoked_subtraction_metadata(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["features"]["exploratory_feature_families"] = []
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            for subject_id in kwargs["subjects"]:
                _write_feature_output(
                    feature_root,
                    subject_id,
                    "power",
                    power_subtract_evoked=True,
                )
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        try:
            prepare_study1_features(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except ValueError as exc:
            assert "power.subtract_evoked" in str(exc)
        else:
            raise AssertionError("Expected evoked-subtracted power metadata to raise ValueError.")


def test_prepare_study1_features_requires_all_configured_family_outputs(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            _write_all_prepared_outputs(
                feature_root, list(kwargs["subjects"]), ["power", "spectral"]
            )
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        try:
            prepare_study1_features(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except FileNotFoundError as exc:
            assert "aperiodic" in str(exc)
        else:
            raise AssertionError("Expected missing exploratory outputs to raise FileNotFoundError.")


def test_prepare_study1_features_prunes_windowed_duplicates_and_sidecars(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["features"]["exploratory_feature_families"] = ["erds", "bursts"]
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            for subject_id in kwargs["subjects"]:
                for family in kwargs["feature_categories"]:
                    if family == "power":
                        _write_feature_output(feature_root, subject_id, family)
                    else:
                        _write_windowed_duplicate_outputs(feature_root, subject_id, family)
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        prepare_study1_features(
            subjects=["0001", "0002"],
            task="pain",
            config=cfg,
            logger=logging.getLogger(__name__),
        )

    feature_root = (
        tmp_path / "derivatives" / "group" / "multimodal" / "study1" / "features_trial_ml_safe"
    )
    erds_dir = feature_root / "sub-0001" / "eeg" / "features" / "erds"
    bursts_dir = feature_root / "sub-0001" / "eeg" / "features" / "bursts"
    assert (erds_dir / "features_erds.parquet").exists()
    assert not (erds_dir / "features_erds_active.parquet").exists()
    assert not (erds_dir / "metadata" / "features_erds_active.json").exists()
    assert not (erds_dir / "metadata" / "extraction_config_active.json").exists()
    assert not (erds_dir / "metadata" / "extraction_config_baseline.json").exists()
    assert not any(path.name.startswith("._") for path in feature_root.rglob("._*"))
    assert (bursts_dir / "features_bursts.parquet").exists()
    assert not (bursts_dir / "features_bursts_active.parquet").exists()


def test_prepare_study1_features_rejects_invalid_baseline_window_for_windowed_families(
    tmp_path,
) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["time_frequency_analysis"]["baseline_window"] = [0.0, -1.0]
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        try:
            prepare_study1_features(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except ValueError as exc:
            assert "time_frequency_analysis.baseline_window" in str(exc)
        else:
            raise AssertionError("Expected invalid baseline window to raise ValueError.")

    feature_pipeline_cls.assert_not_called()


def test_prepare_study1_features_rejects_invalid_aperiodic_metadata(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["features"]["exploratory_feature_families"] = ["aperiodic"]
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            for subject_id in kwargs["subjects"]:
                _write_feature_output(feature_root, subject_id, "power")
                _write_feature_output(
                    feature_root,
                    subject_id,
                    "aperiodic",
                    aperiodic_subtract_evoked=True,
                )
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        try:
            prepare_study1_features(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except ValueError as exc:
            assert "aperiodic.subtract_evoked" in str(exc)
        else:
            raise AssertionError("Expected invalid aperiodic metadata to raise ValueError.")


def test_prepare_study1_features_rejects_iaf_enabled_metadata(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["features"]["exploratory_feature_families"] = ["spectral"]
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            for subject_id in kwargs["subjects"]:
                _write_feature_output(feature_root, subject_id, "power")
                _write_feature_output(
                    feature_root,
                    subject_id,
                    "spectral",
                    bands_use_iaf=True,
                )
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        try:
            prepare_study1_features(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except ValueError as exc:
            assert "bands.use_iaf" in str(exc)
        else:
            raise AssertionError("Expected IAF-enabled metadata to raise ValueError.")


def test_prepare_study1_features_rejects_non_trial_burst_threshold_metadata(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import prepare_study1_features

    cfg = _config(tmp_path / "derivatives")
    cfg["study1"]["features"]["exploratory_feature_families"] = ["bursts"]
    _write_primary_targets(cfg)

    with patch(
        "studies.pain_study.study1.prepare_features.FeaturePipeline"
    ) as feature_pipeline_cls:
        feature_pipeline = feature_pipeline_cls.return_value

        def _write_outputs(**kwargs) -> list[dict]:
            feature_root = Path(kwargs["feature_output_root"])
            for subject_id in kwargs["subjects"]:
                _write_feature_output(feature_root, subject_id, "power")
                _write_feature_output(
                    feature_root,
                    subject_id,
                    "bursts",
                    bursts_threshold_reference="subject",
                )
            return []

        feature_pipeline.run_batch.side_effect = _write_outputs

        try:
            prepare_study1_features(
                subjects=["0001", "0002"],
                task="pain",
                config=cfg,
                logger=logging.getLogger(__name__),
            )
        except ValueError as exc:
            assert "bursts.threshold_reference" in str(exc)
        else:
            raise AssertionError("Expected non-trial burst threshold metadata to raise ValueError.")


def test_clear_subject_feature_outputs_ignores_transient_missing_entries(tmp_path) -> None:
    from studies.pain_study.study1.prepare_features import _clear_subject_feature_outputs

    cfg = _config(tmp_path / "derivatives")
    subject_root = (
        tmp_path
        / "derivatives"
        / "group"
        / "multimodal"
        / "study1"
        / "features_trial_ml_safe"
        / "sub-0001"
    )
    subject_root.mkdir(parents=True, exist_ok=True)

    with patch(
        "studies.pain_study.study1.prepare_features.shutil.rmtree",
        side_effect=FileNotFoundError("._features"),
    ):
        _clear_subject_feature_outputs(subjects=["sub-0001"], config=cfg)
