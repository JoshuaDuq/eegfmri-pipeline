from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from studies.pain_study.study1.config.loader import load_study1_config
from studies.pain_study.study1.figures.sensor_cluster_inference import (
    SensorClusterResult,
    SensorMapResult,
)
from studies.pain_study.study1.figures.sensor_topography_data import SensorMontage
from studies.pain_study.study1.figures.sensor_topography_estimands import ParticipantEffects

CORE_SUFFIXES = (
    ".svg",
    ".png",
    "_by_subject.tsv",
    "_by_subject.parquet",
    "_sensors.tsv",
    "_sensors.parquet",
    "_clusters.tsv",
    "_family.tsv",
    "_caption.txt",
    "_manifest.json",
)
SENSITIVITY_SUFFIXES = (
    "_sensitivity_by_subject.tsv",
    "_sensitivity_by_subject.parquet",
    "_sensitivity_summary.tsv",
    "_sensitivity_summary.parquet",
)


def test_output_paths_are_immutable_exact_families(tmp_path: Path) -> None:
    from studies.pain_study.study1.figures.sensor_topography_outputs import (
        ConstructSensorTopographyPaths,
        SensorTopographyPaths,
        sensor_topography_paths,
    )

    svg = tmp_path / "sensor_power_topographies.svg"
    construct = sensor_topography_paths(svg, include_sensitivity=True)
    signature = sensor_topography_paths(
        tmp_path / "signature_power_topographies.svg",
        include_sensitivity=False,
    )

    assert isinstance(construct, ConstructSensorTopographyPaths)
    assert isinstance(signature, SensorTopographyPaths)
    assert tuple(path.name for path in construct.all_files) == tuple(
        f"sensor_power_topographies{suffix}"
        for suffix in (*CORE_SUFFIXES[:-1], *SENSITIVITY_SUFFIXES, CORE_SUFFIXES[-1])
    )
    assert tuple(path.name for path in signature.all_files) == tuple(
        f"signature_power_topographies{suffix}" for suffix in CORE_SUFFIXES
    )
    with pytest.raises(AttributeError):
        signature.svg = tmp_path / "changed.svg"  # type: ignore[misc]

    for invalid in ("figure.png", "figure.SVG", "figure"):
        with pytest.raises(ValueError, match="must be an SVG"):
            sensor_topography_paths(tmp_path / invalid, include_sensitivity=False)


def test_publish_construct_family_writes_exact_schema_caption_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures import sensor_topography_outputs as module

    config = _config()
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    output = tmp_path / "sensor_power_topographies.svg"
    effects = _effects(family="construct", sensitivity=True)
    inference = _inference(effects)
    montage = _montage(effects.sensor_order)
    figure = plt.figure()
    monkeypatch.setattr(module, "_save_png", _fake_save)
    monkeypatch.setattr(module, "_save_svg", _fake_save)

    paths = module.publish_sensor_topography_family(
        figure=figure,
        effects=effects,
        inference=inference,
        montage=montage,
        config=config,
        source_paths=(source,),
        output_path=output,
        family="construct",
    )

    expected = {
        f"sensor_power_topographies{suffix}" for suffix in (*CORE_SUFFIXES, *SENSITIVITY_SUFFIXES)
    }
    assert {path.name for path in tmp_path.iterdir()} == expected | {"source.parquet"}
    assert {path.name for path in paths.all_files} == expected
    pd.testing.assert_frame_equal(pd.read_parquet(paths.by_subject_parquet), effects.effects)
    pd.testing.assert_frame_equal(pd.read_parquet(paths.sensors_parquet), inference.sensor_frame())
    pd.testing.assert_frame_equal(
        pd.read_parquet(paths.sensitivity_by_subject_parquet),
        effects.sensitivity_effects,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(paths.sensitivity_summary_parquet),
        effects.sensitivity_summary,
    )
    caption = paths.caption.read_text(encoding="utf-8")
    for phrase in (
        "two rows",
        "five frequency bands",
        "unthresholded",
        "dark rings",
        "joint family-wise error",
        "sensor space",
        "not source localization",
    ):
        assert phrase in caption

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["participant_count"] == len(effects.participant_order)
    assert manifest["participant_order"] == list(effects.participant_order)
    assert manifest["participant_exclusions"] == []
    assert manifest["channel_order"] == list(effects.sensor_order)
    assert manifest["montage"]["name"] == "standard_1020"
    assert manifest["adjacency"]["method"] == "Delaunay top-view triangulation"
    assert manifest["inference"]["requested_sign_count"] == 32
    assert manifest["inference"]["actual_sign_count"] == 32
    assert manifest["inference"]["observed_label_included"] is True
    assert set(manifest["software"]) >= {
        "python",
        "matplotlib",
        "mne",
        "numpy",
        "pandas",
        "scipy",
    }
    assert manifest["source_sha256"] == {str(source.resolve()): module._sha256(source)}
    assert set(manifest["output_sha256"]) == {
        path.name for path in paths.all_files if path != paths.manifest
    }
    assert paths.manifest.name not in manifest["output_sha256"]


def test_signature_caption_identifies_univariate_association(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures import sensor_topography_outputs as module

    source = tmp_path / "source.tsv"
    source.write_text("source\n", encoding="utf-8")
    effects = _effects(family="signature", sensitivity=False)
    monkeypatch.setattr(module, "_save_png", _fake_save)
    monkeypatch.setattr(module, "_save_svg", _fake_save)
    paths = module.publish_sensor_topography_family(
        figure=plt.figure(),
        effects=effects,
        inference=_inference(effects),
        montage=_montage(effects.sensor_order),
        config=_config(),
        source_paths=(source,),
        output_path=tmp_path / "signature_power_topographies.svg",
        family="signature",
    )

    caption = paths.caption.read_text(encoding="utf-8")
    assert "univariate association" in caption
    assert "not model importance" in caption


def test_publication_is_byte_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures import sensor_topography_outputs as module

    source = tmp_path / "source.tsv"
    source.write_text("source\n", encoding="utf-8")
    effects = _effects(family="signature", sensitivity=False)
    output = tmp_path / "signature_power_topographies.svg"
    monkeypatch.setattr(module, "_save_png", _fake_save)
    monkeypatch.setattr(module, "_save_svg", _fake_save)
    arguments = {
        "effects": effects,
        "inference": _inference(effects),
        "montage": _montage(effects.sensor_order),
        "config": _config(),
        "source_paths": (source,),
        "output_path": output,
        "family": "signature",
    }

    first = module.publish_sensor_topography_family(figure=plt.figure(), **arguments)
    first_bytes = {path.name: path.read_bytes() for path in first.all_files}
    second = module.publish_sensor_topography_family(figure=plt.figure(), **arguments)

    assert {path.name: path.read_bytes() for path in second.all_files} == first_bytes


def test_staging_failure_preserves_existing_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures import sensor_topography_outputs as module

    output = tmp_path / "signature_power_topographies.svg"
    paths = module.sensor_topography_paths(output, include_sensitivity=False)
    original = {path: f"old:{path.name}".encode() for path in paths.all_files}
    for path, content in original.items():
        path.write_bytes(content)
    source = tmp_path / "source.tsv"
    source.write_text("source\n", encoding="utf-8")
    effects = _effects(family="signature", sensitivity=False)
    monkeypatch.setattr(module, "_save_png", _fake_save)
    monkeypatch.setattr(module, "_save_svg", _fake_save)

    def fail_text(path: Path, content: str) -> None:
        if path.name.endswith("_caption.txt"):
            raise OSError("injected staging failure")
        module._write_text_unchecked(path, content)

    monkeypatch.setattr(module, "_write_text", fail_text)
    with pytest.raises(OSError, match="injected staging failure"):
        module.publish_sensor_topography_family(
            figure=plt.figure(),
            effects=effects,
            inference=_inference(effects),
            montage=_montage(effects.sensor_order),
            config=_config(),
            source_paths=(source,),
            output_path=output,
            family="signature",
        )

    assert {path: path.read_bytes() for path in paths.all_files} == original
    assert not tuple(tmp_path.glob(".signature_power_topographies-*"))


def test_promotion_failure_restores_every_prior_file_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studies.pain_study.study1.figures import sensor_topography_outputs as module

    output = tmp_path / "signature_power_topographies.svg"
    paths = module.sensor_topography_paths(output, include_sensitivity=False)
    original = {path: f"old:{path.name}".encode() for path in paths.all_files}
    for path, content in original.items():
        path.write_bytes(content)
    source = tmp_path / "source.tsv"
    source.write_text("source\n", encoding="utf-8")
    effects = _effects(family="signature", sensitivity=False)
    monkeypatch.setattr(module, "_save_png", _fake_save)
    monkeypatch.setattr(module, "_save_svg", _fake_save)
    real_replace = module._replace_path
    promotions = 0

    def fail_third_promotion(source_path: Path, destination_path: Path) -> None:
        nonlocal promotions
        if source_path.parent.name.startswith(".signature_power_topographies-"):
            promotions += 1
            if promotions == 3:
                raise OSError("injected promotion failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(module, "_replace_path", fail_third_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        module.publish_sensor_topography_family(
            figure=plt.figure(),
            effects=effects,
            inference=_inference(effects),
            montage=_montage(effects.sensor_order),
            config=_config(),
            source_paths=(source,),
            output_path=output,
            family="signature",
        )

    assert {path: path.read_bytes() for path in paths.all_files} == original
    assert not tuple(tmp_path.glob(".signature_power_topographies-*"))


@pytest.mark.parametrize(
    ("module_name", "writer_name", "filename"),
    [
        (
            "plot_sensor_power_topographies",
            "write_sensor_power_topographies",
            "construct.svg",
        ),
        (
            "plot_signature_power_topographies",
            "write_signature_power_topographies",
            "signature.svg",
        ),
    ],
)
def test_cli_parses_required_inputs_and_prints_svg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module_name: str,
    writer_name: str,
    filename: str,
) -> None:
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(f"studies.pain_study.study1.figures.{module_name}")
    output = tmp_path / filename
    captured: dict[str, object] = {}

    def fake_writer(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(svg=output)

    monkeypatch.setattr(module, writer_name, fake_writer)
    monkeypatch.setattr(module, "load_config", lambda path: {"loaded": Path(path)})
    monkeypatch.setattr(module, "apply_study1_config_defaults", lambda config, path: None)
    module.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--study1-config",
            str(tmp_path / "study1.yaml"),
            "--deriv-root",
            str(tmp_path / "derivatives"),
            "--task",
            "thermal",
            "--output",
            str(output),
        ]
    )

    assert captured["task"] == "thermal"
    assert captured["output_path"] == output
    assert captured["config"]["paths.deriv_root"] == str((tmp_path / "derivatives").resolve())
    assert capsys.readouterr().out == f"{output}\n"


def test_sensor_power_cli_runs_end_to_end_on_synthetic_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from studies.pain_study.study1.figures.plot_sensor_power_topographies import main

    config_path, study1_path, derivative_root = _write_cli_inputs(tmp_path)
    output = tmp_path / "publication" / "sensor_power_topographies.svg"

    paths = main(
        [
            "--config",
            str(config_path),
            "--study1-config",
            str(study1_path),
            "--deriv-root",
            str(derivative_root),
            "--task",
            "thermal",
            "--output",
            str(output),
        ]
    )

    assert paths.svg == output
    assert all(path.is_file() for path in paths.all_files)
    assert capsys.readouterr().out.endswith(f"{output}\n")


def test_signature_power_cli_runs_end_to_end_on_synthetic_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from studies.pain_study.study1.figures.plot_signature_power_topographies import main

    config_path, study1_path, derivative_root = _write_cli_inputs(tmp_path)
    output = tmp_path / "publication" / "signature_power_topographies.svg"

    paths = main(
        [
            "--config",
            str(config_path),
            "--study1-config",
            str(study1_path),
            "--deriv-root",
            str(derivative_root),
            "--task",
            "thermal",
            "--output",
            str(output),
        ]
    )

    assert paths.svg == output
    assert all(path.is_file() for path in paths.all_files)
    assert capsys.readouterr().out.endswith(f"{output}\n")


def _fake_save(figure: object, path: Path, config: object, **kwargs: object) -> Path:
    path.write_bytes(path.suffix.encode())
    return path


def _config() -> dict[str, object]:
    config = load_study1_config()
    config["eeg"] = {"montage": "standard_1020"}
    return config


def _write_cli_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    derivative_root = tmp_path / "derivatives"
    config_path = tmp_path / "eeg_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"deriv_root": str(derivative_root)},
                "eeg": {"montage": "standard_1020"},
            }
        ),
        encoding="utf-8",
    )
    default_path = (
        Path(__file__).parents[2] / "pain_study" / "study1" / "config" / "study1_config.yaml"
    )
    study1_config = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    study1_config["study1"]["targets"]["nuisance_regression"] = {
        "enabled": True,
        "continuous_columns": ["run", "within_run_trial", "residual_ecg_coupling"],
        "categorical_columns": [],
    }
    circular = study1_config["study1"]["feature_benchmark"]["circular_shift"]
    circular["min_retained_trials_per_subject"] = 12
    circular["min_valid_runs_per_subject"] = 3
    study1_config["study1"]["figures"] = {
        "power_construct_validity": {
            "rating_model": {
                "minimum_trials": 12,
                "minimum_runs": 3,
                "max_condition_number": 100.0,
            }
        },
        "sensor_topographies": {
            "inference": {
                "cluster_forming_p": 0.01,
                "family_alpha": 0.05,
                "max_null_draws": 32,
                "seed": 20260715,
            }
        },
    }
    study1_path = tmp_path / "study1_config.yaml"
    study1_path.write_text(yaml.safe_dump(study1_config), encoding="utf-8")
    _write_synthetic_artifacts(derivative_root)
    return config_path, study1_path, derivative_root


def _write_synthetic_artifacts(derivative_root: Path) -> None:
    subjects = tuple(f"sub-{index:04d}" for index in range(1, 7))
    target_frames = []
    for subject_index, subject in enumerate(subjects):
        targets, events, features = _synthetic_subject(subject, subject_index)
        target_frames.append(targets)
        event_path = (
            derivative_root / subject / "eeg" / f"{subject}_task-thermal_proc-clean_events.tsv"
        )
        event_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(event_path, sep="\t", index=False)
        feature_path = (
            derivative_root
            / "group"
            / "multimodal"
            / "study1"
            / "features_trial_ml_safe"
            / subject
            / "eeg"
            / "features"
            / "power"
            / "features_power.parquet"
        )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(feature_path, index=False)
    target_path = (
        derivative_root / "group" / "multimodal" / "study1" / "targets" / "primary_targets.parquet"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(target_frames, ignore_index=True).to_parquet(target_path, index=False)


def _synthetic_subject(
    subject: str,
    subject_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bands = ("alpha", "beta", "gamma_low_clean", "gamma_mid_clean", "gamma_high_clean")
    channels = ("Fp1", "Fp2", "F3", "Cz", "F4")
    temperatures = (44.3, 45.3, 46.3, 47.3, 48.3, 49.3)
    rng = np.random.default_rng(20260715 + subject_index)
    targets = []
    events = []
    features = []
    for trial_index in range(30):
        run = trial_index % 3 + 1
        within_run_trial = ((trial_index // 3) * 7) % 10 + 1
        temperature = temperatures[trial_index // 5]
        selected_surface = trial_index % 5 + 1
        ecg = rng.normal()
        artifact = rng.normal()
        intensity_latent = rng.normal()
        nps_latent = rng.normal()
        siips_latent = rng.normal()
        intensity = 45.0 + 2.0 * (temperature - temperatures[0]) + intensity_latent
        nps = 0.2 * ecg + nps_latent
        siips1 = 0.4 * nps + siips_latent
        target = {
            "subject_id": subject,
            "task": "thermal",
            "run": run,
            "trial_index": trial_index,
            "within_run_trial": within_run_trial,
            "onset": float(trial_index * 12),
            "duration": 10.0,
            "NPS": nps,
            "SIIPS1": siips1,
            "stimulus_temp": temperature,
            "selected_surface": selected_surface,
            "residual_ecg_coupling": ecg,
        }
        targets.append(target)
        events.append(
            {
                "trial_id": trial_index + 1,
                "run_id": run,
                "trial_number": within_run_trial,
                "stimulus_temp": temperature,
                "selected_surface": selected_surface,
                "pain_binary_coded": 1,
                "vas_final_coded_rating": 100.0 + intensity,
                "residual_ecg_coupling": ecg,
                "fp1_fp2_high_frequency_power": artifact,
            }
        )
        feature = {"trial_id": trial_index + 1}
        for band_index, band in enumerate(bands):
            for channel_index, channel in enumerate(channels):
                noise = rng.normal(scale=0.05)
                power_db = (
                    (0.18 + 0.01 * channel_index) * temperature
                    + 0.45 * intensity_latent
                    + 0.55 * nps_latent
                    + 0.45 * siips_latent
                    + 0.08 * ecg
                    - 0.06 * artifact
                    + 0.02 * band_index
                    + noise
                )
                feature[f"power_baseline_{band}_ch_{channel}_mean"] = 2.0
                feature[f"power_active_{band}_ch_{channel}_logratio"] = power_db / 10.0
        features.append(feature)
    return (
        pd.DataFrame.from_records(targets),
        pd.DataFrame.from_records(events),
        pd.DataFrame.from_records(features),
    )


def _effects(*, family: str, sensitivity: bool) -> ParticipantEffects:
    participants = tuple(f"sub-{index:04d}" for index in range(1, 7))
    sensors = ("F3", "Cz", "F4")
    estimands = ("temperature", "intensity") if family == "construct" else ("NPS", "SIIPS1")
    bands = ("alpha", "beta", "gamma_low_clean", "gamma_mid_clean", "gamma_high_clean")
    maps = tuple((estimand, band) for estimand in estimands for band in bands)
    records = []
    for subject_index, subject in enumerate(participants):
        for estimand, band in maps:
            for sensor_index, sensor in enumerate(sensors):
                value = 0.2 + 0.01 * subject_index + 0.001 * sensor_index
                records.append(
                    {
                        "effect_value": value,
                        "partial_r": np.nan if estimand == "temperature" else np.tanh(value),
                        "fisher_z": np.nan if estimand == "temperature" else value,
                        "inference_value": value,
                        "n_trials": 30,
                        "n_runs": 3,
                        "subject_id": subject,
                        "estimand": estimand,
                        "band": band,
                        "channel": sensor,
                        "channel_scope": (
                            "primary" if family == "construct" else "feature_benchmark"
                        ),
                        "include_fp1_fp2": family == "construct",
                        **(
                            {
                                "design_rank": np.nan,
                                "design_parameters": np.nan,
                                "residual_degrees_of_freedom": np.nan,
                                "condition_number": np.nan,
                            }
                            if family == "construct"
                            else {
                                "resolved_nuisance_columns": "run,nuisance",
                                "omitted_constant_nuisance_columns": "",
                                "retained_nuisance_columns": "run,nuisance",
                                "design_parameters": 3,
                                "residual_degrees_of_freedom": 27,
                                "minimum_singular_value_ratio": 0.5,
                                "condition_number": 2.0,
                                "minimum_standardized_nuisance_norm": 1.0,
                                "maximum_standardized_nuisance_norm": 1.0,
                            }
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    columns = module_columns(family)
    frame = frame.loc[:, columns]
    summary = _summary(frame, maps, sensors)
    sensitivity_effects = frame.copy() if sensitivity else None
    sensitivity_summary = summary.copy() if sensitivity else None
    return ParticipantEffects(
        effects=frame,
        summary=summary,
        exclusions=pd.DataFrame(columns=("subject_id", "reason", "estimand", "band", "channel")),
        sensitivity_effects=sensitivity_effects,
        sensitivity_summary=sensitivity_summary,
        map_order=maps,
        sensor_order=sensors,
        participant_order=participants,
    )


def _summary(
    effects: pd.DataFrame,
    maps: tuple[tuple[str, str], ...],
    sensors: tuple[str, ...],
) -> pd.DataFrame:
    records = []
    for estimand, band in maps:
        for sensor in sensors:
            rows = effects.loc[
                effects["estimand"].eq(estimand)
                & effects["band"].eq(band)
                & effects["channel"].eq(sensor)
            ]
            mean = float(rows["inference_value"].mean())
            records.append(
                {
                    "estimand": estimand,
                    "band": band,
                    "channel": sensor,
                    "mean_inference_value": mean,
                    "display_value": mean if estimand == "temperature" else np.tanh(mean),
                    "n_participants": len(rows),
                }
            )
    return pd.DataFrame.from_records(records)


def _inference(effects: ParticipantEffects) -> SensorClusterResult:
    maps = tuple(
        SensorMapResult(
            estimand=estimand,
            band=band,
            cohort_values=tuple(
                effects.summary.loc[
                    effects.summary["estimand"].eq(estimand) & effects.summary["band"].eq(band),
                    "display_value",
                ]
            ),
            t_statistics=(2.0, 2.1, 2.2),
            clusters=(),
            significant_sensors=(),
        )
        for estimand, band in effects.map_order
    )
    adjacency = (
        (True, True, True),
        (True, True, True),
        (True, True, True),
    )
    return SensorClusterResult(
        participant_order=effects.participant_order,
        map_order=effects.map_order,
        sensor_order=effects.sensor_order,
        adjacency=adjacency,
        map_results=maps,
        null_max_cluster_masses=tuple(float(index) for index in range(32)),
        sign_patterns=tuple((1, 1, 1, 1, 1, 1) for _ in range(32)),
        cluster_forming_p=0.01,
        family_alpha=0.05,
        positive_threshold=4.0,
        negative_threshold=-4.0,
        requested_max_null_draws=32,
        sampled_null_draws=32,
        total_exact_patterns=32,
        observed_included=True,
        exact_enumeration=True,
        seed=20260715,
    )


def _montage(sensors: tuple[str, ...]) -> SensorMontage:
    positions_3d = ((-0.4, 0.5, 0.7), (0.0, 0.0, 1.0), (0.4, 0.5, 0.7))
    return SensorMontage(
        name="standard_1020",
        channels=sensors,
        positions_3d=positions_3d,
        positions_xy=tuple(position[:2] for position in positions_3d),
    )


def module_columns(family: str) -> tuple[str, ...]:
    common_start = (
        "effect_value",
        "partial_r",
        "fisher_z",
        "inference_value",
        "n_trials",
        "n_runs",
    )
    common_end = (
        "subject_id",
        "estimand",
        "band",
        "channel",
        "channel_scope",
        "include_fp1_fp2",
    )
    if family == "construct":
        return (
            *common_start,
            *common_end,
            "design_rank",
            "design_parameters",
            "residual_degrees_of_freedom",
            "condition_number",
        )
    return (
        *common_start,
        "resolved_nuisance_columns",
        "omitted_constant_nuisance_columns",
        "retained_nuisance_columns",
        "design_parameters",
        "residual_degrees_of_freedom",
        "minimum_singular_value_ratio",
        "condition_number",
        "minimum_standardized_nuisance_norm",
        "maximum_standardized_nuisance_norm",
        *common_end,
    )
