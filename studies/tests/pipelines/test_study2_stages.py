"""Behavior tests for Study 2 orchestration stages."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from studies.pain_study.study2 import paths, stages
from studies.pain_study.study2.config import load_study2_config
from studies.pain_study.study2.runner import Study2StageContext
from studies.pain_study.study2.stages import (
    artifact_controls_required_inputs,
    behavioral_convergence_required_inputs,
    band_unique_inference_required_inputs,
    band_unique_stage_required_inputs,
    directional_consistency_required_inputs,
    haufe_required_inputs,
    inference_required_inputs,
    point_spread_required_inputs,
    run_gate,
    run_artifact_controls,
    run_behavioral_convergence,
    run_band_unique_inference,
    run_band_unique_stage,
    run_directional_consistency,
    run_haufe,
    run_inference,
    run_point_spread,
    run_robustness,
    run_source_model_qc,
    run_source_power,
    run_source_stage,
    run_target_permutations,
    source_power_required_inputs,
    source_model_qc_required_inputs,
    spatial_correspondence_required_inputs,
    target_permutations_required_inputs,
    robustness_required_inputs,
    run_spatial_correspondence,
    run_spatial_surrogates,
    spatial_surrogates_required_inputs,
)
from studies.pain_study.study2.source_vertex_manifest import ensure_common_source_vertices
from studies.pain_study.study2.study1_context import study1_model_comparison_path
from studies.tests.pipelines.test_study2_source_family import (
    _chain_adjacency,
    _clustered_subject_maps,
)
from studies.tests.pipelines.test_study2_source_maps import (
    _cohort_source_stage_frame,
    _source_power_by_subject,
)


def _config(tmp_path: Path) -> dict:
    config = load_study2_config()
    config["paths"] = {"deriv_root": str(tmp_path)}
    return config


def _context(config: dict, subjects: tuple[str, ...]) -> Study2StageContext:
    return Study2StageContext(
        config=config,
        subjects=subjects,
        task="pain",
        logger=logging.getLogger("test"),
    )


def test_study2_paths_require_configured_output_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    del config["study2"]["outputs"]["root_name"]

    with pytest.raises(ValueError, match="study2.outputs.root_name"):
        paths.study2_output_root(config)


def test_study2_paths_require_configured_study1_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    del config["study2"]["inputs"]["study1_root_name"]

    with pytest.raises(ValueError, match="study2.inputs.study1_root_name"):
        paths.study1_report_path(config)


def test_primary_source_figure_paths_share_one_output_family(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_dir = paths.figures_dir(config)

    assert (
        paths.primary_source_figure_path(config) == output_dir / "primary_source_associations.svg"
    )
    assert paths.primary_source_png_path(config) == output_dir / "primary_source_associations.png"
    assert paths.primary_source_vertices_path(config) == (
        output_dir / "primary_source_associations_vertices.tsv"
    )
    assert paths.primary_source_clusters_path(config) == (
        output_dir / "primary_source_associations_clusters.tsv"
    )
    assert paths.primary_source_summary_path(config) == (
        output_dir / "primary_source_associations_summary.tsv"
    )
    assert paths.primary_source_caption_path(config) == (
        output_dir / "primary_source_associations_caption.txt"
    )
    assert paths.primary_source_manifest_path(config) == (
        output_dir / "primary_source_associations_manifest.json"
    )


def _write_study1_report(config: dict, *, overrides: dict[str, object]) -> None:
    row = {
        "analysis_partition": "primary",
        "target": "NPS",
        "feature_spec": "alpha_beta_gamma",
        "model": "elasticnet",
        "mean_delta_r2": 0.031,
        "p_value_delta_r2_holm": 0.018,
        "ci_low_delta_r2": 0.006,
        "level2_mean_delta_r2": 0.006,
        "target_split_half_reliability": 0.51,
        "target_reliability_n_trials": 31,
        "within_subject_centered_delta_r2": 0.004,
        "temporal_negative_controls_passed": True,
        "artifact_censoring_robustness_passed": True,
    }
    row.update(overrides)
    report_path = paths.study1_report_path(config)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(report_path, sep="\t", index=False)


def test_run_gate_writes_confirmatory_criteria(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_study1_report(config, overrides={})

    run_gate(_context(config, subjects=()))

    payload = json.loads(paths.gate_qc_path(config).read_text())
    assert set(payload) == {"confirmatory_criteria_met", "unmet_criteria"}
    assert payload["confirmatory_criteria_met"] is True
    assert payload["unmet_criteria"] == []


def test_run_gate_writes_unmet_confirmatory_criteria(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_study1_report(config, overrides={"p_value_delta_r2_holm": 0.5})

    run_gate(_context(config, subjects=()))

    payload = json.loads(paths.gate_qc_path(config).read_text())
    assert payload["confirmatory_criteria_met"] is False
    assert payload["unmet_criteria"] == ["significant_positive_delta_r2"]


def test_study1_capable_config_uses_study2_study1_root_name(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["study2"]["inputs"] = {"study1_root_name": "study1_custom"}

    merged = stages._study1_capable_config(config)

    assert merged["study1"]["outputs"]["root_name"] == "study1_custom"
    assert (
        tmp_path / "group" / "multimodal" / "study1_custom"
        in study1_model_comparison_path(merged).parents
    )


def test_target_permutations_required_inputs_use_study2_study1_root_name(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["study2"]["inputs"] = {"study1_root_name": "study1_custom"}

    required = target_permutations_required_inputs(_context(config, subjects=()))

    assert (
        tmp_path / "group" / "multimodal" / "study1_custom" / "feature_benchmark"
        in required[1].parents
    )


def _with_anatomy(config: dict, subjects_dir: Path) -> dict:
    config["study2"]["source_modeling"]["anatomy"] = {
        "subjects_dir": str(subjects_dir),
        "trans_path_template": "{subjects_dir}/{subject}-trans.fif",
        "bem_path_template": "{subjects_dir}/{subject}-bem.fif",
    }
    return config


def test_source_power_required_inputs_lists_anatomy_files(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "freesurfer"
    config = _with_anatomy(_config(tmp_path), subjects_dir)

    required = source_power_required_inputs(_context(config, subjects=("sub-0000",)))

    assert subjects_dir in required
    assert subjects_dir / "sub-0000-trans.fif" in required
    assert subjects_dir / "sub-0000-bem.fif" in required


def test_source_power_required_inputs_fails_when_anatomy_unconfigured(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="anatomy"):
        source_power_required_inputs(_context(config, subjects=("sub-0000",)))


def test_band_frequency_ranges_keep_gamma_scanner_clean(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert stages._band_frequency_ranges(config) == {
        "alpha": ((8.0, 12.9),),
        "beta": ((13.0, 30.0),),
        "gamma": ((30.1, 38.0), (43.0, 56.0), (67.0, 77.0)),
    }


def test_run_source_power_writes_per_band_logratio_power(tmp_path: Path, monkeypatch) -> None:
    config = _with_anatomy(_config(tmp_path), tmp_path / "freesurfer")
    times = np.linspace(-5.0, 10.5, 32)
    rng = np.random.default_rng(0)
    stcs = [SimpleNamespace(data=rng.normal(size=(3, times.size))) for _ in range(4)]
    filtered_ranges = []

    def fake_filter(low, high, **kwargs):
        filtered_ranges.append((low, high))
        return object()

    fake_epochs = SimpleNamespace(
        info={},
        times=times,
        copy=lambda: SimpleNamespace(filter=fake_filter),
    )

    monkeypatch.setattr(stages, "_load_subject_epochs", lambda *a, **k: fake_epochs)
    monkeypatch.setattr(stages, "build_surface_forward_model", lambda *a, **k: object())
    monkeypatch.setattr(stages, "compute_baseline_noise_covariance", lambda *a, **k: object())
    monkeypatch.setattr(stages, "make_sloreta_inverse_operator", lambda **k: object())
    monkeypatch.setattr(stages, "apply_sloreta_inverse", lambda **k: stcs)
    monkeypatch.setattr(stages, "make_surface_source_morph", lambda **k: object())

    def fake_morphed_power(*, stcs, times, baseline_window_s, active_window_s, morph):
        return SimpleNamespace(
            power_logratio=np.stack([np.mean(stc.data, axis=1) for stc in stcs], axis=0),
            baseline_window_s=baseline_window_s,
            active_window_s=active_window_s,
            n_trials=len(stcs),
            n_vertices=stcs[0].data.shape[0],
            vertices=(np.array([0, 1], dtype=int), np.array([2], dtype=int)),
        )

    monkeypatch.setattr(
        stages,
        "compute_morphed_sloreta_hilbert_logratio_power",
        fake_morphed_power,
    )

    run_source_power(_context(config, subjects=("sub-0000",)))

    for band in ("alpha", "beta", "gamma"):
        power = np.load(paths.subject_source_power_path(config, subject_id="sub-0000", band=band))
        assert power.shape == (4, 3)
        assert np.all(np.isfinite(power))
        metadata = json.loads(
            paths.subject_source_power_metadata_path(
                config,
                subject_id="sub-0000",
                band=band,
            ).read_text()
        )
        assert metadata["subject_id"] == "sub-0000"
        assert metadata["band"] == band
        assert metadata["frequency_hz"] == [
            list(frequency_range) for frequency_range in stages._band_frequency_ranges(config)[band]
        ]
        if band == "gamma":
            assert metadata["frequency_aggregation"] == "bandwidth_weighted_logratio_mean"
        else:
            assert metadata["frequency_aggregation"] == "single_contiguous_band"
        assert metadata["baseline_window_s"] == [-5.0, -0.01]
        assert metadata["active_window_s"] == [3.0, 10.5]
        assert metadata["common_subject"] == "fsaverage"
        assert metadata["common_source_space_spacing"] == "oct6"
        assert metadata["n_trials"] == 4
        assert metadata["n_vertices"] == 3
    manifest = np.load(paths.source_vertex_manifest_path(config))
    np.testing.assert_array_equal(manifest["lh_vertices"], [0, 1])
    np.testing.assert_array_equal(manifest["rh_vertices"], [2])
    vertex_metadata = json.loads(paths.source_vertex_metadata_path(config).read_text())
    assert vertex_metadata["common_subject"] == "fsaverage"
    assert vertex_metadata["spacing"] == "oct6"
    assert vertex_metadata["n_vertices"] == 3
    assert filtered_ranges == [
        (8.0, 12.9),
        (13.0, 30.0),
        (30.1, 38.0),
        (43.0, 56.0),
        (67.0, 77.0),
    ]


def test_run_source_stage_writes_band_maps_and_qc(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = _cohort_source_stage_frame()
    paths.source_stage_dir(config).mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.source_stage_frame_path(config), sep="\t", index=False)

    subjects = ("sub-0001", "sub-0002", "sub-0003")
    source_power = _source_power_by_subject(frame, column="eta_combined_z")
    for band in ("alpha", "beta", "gamma"):
        for subject_id in subjects:
            power_path = paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            power_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(power_path, source_power[subject_id])

    run_source_stage(_context(config, subjects=subjects))

    source_stage_dir = paths.source_stage_dir(config)
    for band in ("alpha", "beta", "gamma"):
        fisher = np.load(source_stage_dir / f"fisher_z_{band}.npy")
        partial = np.load(source_stage_dir / f"partial_r_{band}.npy")
        qc = pd.read_csv(source_stage_dir / f"qc_{band}.tsv", sep="\t")
        assert fisher.shape == (2, 3)
        assert partial.shape == (2, 3)
        assert qc["source_stage_criteria_met"].tolist() == [True, True, False]


def test_target_permutations_required_inputs_lists_frame_power_and_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["study2"]["inputs"] = {"study1_root_name": "study1"}
    subjects = ("sub-0001", "sub-0002")

    required = target_permutations_required_inputs(_context(config, subjects=subjects))

    assert paths.source_stage_frame_path(config) in required
    assert study1_model_comparison_path(config) in required
    for band in ("alpha", "beta", "gamma"):
        for subject_id in subjects:
            assert (
                paths.subject_source_power_path(config, subject_id=subject_id, band=band)
                in required
            )


def test_run_target_permutations_writes_null_maps_per_band(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    config["study2"]["inputs"] = {"study1_root_name": "study1"}
    frame = _cohort_source_stage_frame()
    paths.source_stage_dir(config).mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.source_stage_frame_path(config), sep="\t", index=False)

    subjects = ("sub-0001", "sub-0002", "sub-0003")
    source_power = _source_power_by_subject(frame, column="eta_combined_z")
    for band in ("alpha", "beta", "gamma"):
        for subject_id in subjects:
            power_path = paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            power_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(power_path, source_power[subject_id])

    captured: dict = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        null = {band: np.zeros((5, 2, 3), dtype=float) for band in kwargs["bands"]}
        return null, SimpleNamespace(n_valid_draws=5, n_invalid_draws=1)

    monkeypatch.setattr(stages, "load_study1_model_context", lambda **k: SimpleNamespace())
    monkeypatch.setattr(
        stages,
        "_observed_source_stage_subject_ids",
        lambda *a, **k: ("sub-0001", "sub-0002"),
    )
    monkeypatch.setattr(stages, "build_target_retrained_null_maps", fake_build)

    run_target_permutations(_context(config, subjects=subjects))

    for band in ("alpha", "beta", "gamma"):
        null = np.load(paths.null_source_maps_path(config, band=band))
        assert null.shape == (5, 2, 3)
    assert captured["bands"] == ("alpha", "beta", "gamma")
    assert captured["expected_subject_ids"] == ("sub-0001", "sub-0002")
    assert (
        captured["n_valid_draws"]
        == config["study2"]["permutations"]["target_retrained_valid_draws"]
    )


def test_inference_required_inputs_lists_maps_and_adjacency(tmp_path: Path) -> None:
    config = _config(tmp_path)

    required = inference_required_inputs(_context(config, subjects=()))

    assert paths.source_adjacency_path(config) in required
    for band in ("alpha", "beta", "gamma"):
        assert paths.source_stage_fisher_z_path(config, band=band) in required
        assert paths.null_source_maps_path(config, band=band) in required


def test_run_inference_writes_holm_corrected_source_family_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths.source_stage_dir(config).mkdir(parents=True, exist_ok=True)
    paths.inference_dir(config).mkdir(parents=True, exist_ok=True)
    np.save(paths.source_adjacency_path(config), _chain_adjacency(5))

    observed = {
        "alpha": _clustered_subject_maps(),
        "beta": np.zeros((4, 5), dtype=float),
        "gamma": np.zeros((4, 5), dtype=float),
    }
    for band, maps in observed.items():
        np.save(paths.source_stage_fisher_z_path(config, band=band), maps)
        np.save(paths.null_source_maps_path(config, band=band), np.zeros((99, 4, 5), dtype=float))

    run_inference(_context(config, subjects=("sub-0001",)))

    summary = pd.read_csv(paths.source_family_summary_path(config), sep="\t")
    assert summary["band"].tolist() == ["alpha", "beta", "gamma"]
    alpha_row = summary.loc[summary["band"] == "alpha"].iloc[0]
    assert bool(alpha_row["significant"]) is True
    assert alpha_row["inference_tier"] == "below_feasibility"
    assert bool(summary.loc[summary["band"] == "beta", "significant"].iloc[0]) is False


def test_run_haufe_writes_sensor_pattern_outputs(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    model_path = study1_model_comparison_path(config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model": ["elasticnet"]}).to_csv(model_path, sep="\t", index=False)
    report_path = paths.study1_report_path(config)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("target\nNPS\n", encoding="utf-8")
    summary = SimpleNamespace(
        fold_patterns=pd.DataFrame({"target": ["NPS"], "fold": [0]}),
        aggregate_patterns=pd.DataFrame({"target": ["NPS"], "band": ["alpha"]}),
        stability=pd.DataFrame({"target": ["NPS"], "spatial_correlation": [0.8]}),
    )
    monkeypatch.setattr(stages, "load_study1_model_context", lambda **kwargs: object())
    monkeypatch.setattr(stages, "compute_sensor_pattern_summary", lambda **kwargs: summary)

    assert haufe_required_inputs(_context(config, subjects=("sub-0001",))) == (
        report_path,
        model_path,
    )

    run_haufe(_context(config, subjects=("sub-0001",)))

    assert pd.read_csv(paths.haufe_fold_patterns_path(config), sep="\t").shape == (1, 2)
    assert pd.read_parquet(
        paths.haufe_aggregate_patterns_path(config).with_suffix(".parquet")
    ).shape == (1, 2)
    assert pd.read_csv(paths.haufe_stability_path(config), sep="\t").shape == (1, 2)


def test_run_source_model_qc_writes_subject_criteria(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths.source_model_dir(config).mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "subject_id": "sub-0001",
                "freesurfer_visual_qc_passed": True,
                "bem_succeeded": True,
                "has_subject_specific_electrode_positions": True,
                "uses_template_electrode_coordinates": False,
                "valid_eeg_channel_location_fraction": 0.95,
                "mean_coregistration_error_mm": 2.0,
                "max_coregistration_error_mm": 6.0,
                "forward_solution_valid": True,
                "forward_solution_rank_deficient_channels": False,
                "morph_to_fsaverage_succeeded": True,
            },
            {
                "subject_id": "sub-0002",
                "freesurfer_visual_qc_passed": True,
                "bem_succeeded": True,
                "has_subject_specific_electrode_positions": True,
                "uses_template_electrode_coordinates": False,
                "valid_eeg_channel_location_fraction": 0.5,
                "mean_coregistration_error_mm": 2.0,
                "max_coregistration_error_mm": 6.0,
                "forward_solution_valid": True,
                "forward_solution_rank_deficient_channels": False,
                "morph_to_fsaverage_succeeded": True,
            },
        ]
    ).to_csv(paths.source_model_metrics_path(config), sep="\t", index=False)

    assert source_model_qc_required_inputs(_context(config, subjects=())) == (
        paths.source_model_metrics_path(config),
    )

    run_source_model_qc(_context(config, subjects=()))

    qc = pd.read_csv(paths.source_model_qc_path(config), sep="\t")
    assert qc.columns.tolist() == [
        "subject_id",
        "source_model_criteria_met",
        "valid_eeg_channel_location_fraction",
        "mean_coregistration_error_mm",
        "max_coregistration_error_mm",
        "unmet_criteria",
    ]
    assert qc["subject_id"].tolist() == ["sub-0001", "sub-0002"]
    assert qc["source_model_criteria_met"].tolist() == [True, False]
    assert "valid_eeg_channel_locations" in qc.loc[1, "unmet_criteria"]


def test_run_point_spread_writes_resolution_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths.source_model_dir(config).mkdir(parents=True)
    resolution = np.asarray([[1.0, 0.6], [0.4, 1.0]], dtype=float)
    distances = np.asarray([[0.0, 8.0], [8.0, 0.0]], dtype=float)
    np.save(paths.point_spread_resolution_matrix_path(config), resolution)
    np.save(paths.point_spread_distances_path(config), distances)

    assert point_spread_required_inputs(_context(config, subjects=())) == (
        paths.point_spread_resolution_matrix_path(config),
        paths.point_spread_distances_path(config),
    )

    run_point_spread(_context(config, subjects=()))

    vertex_fwhm = np.load(paths.point_spread_vertex_fwhm_path(config))
    summary = pd.read_csv(paths.point_spread_summary_path(config), sep="\t")
    assert vertex_fwhm.tolist() == [0.0, 8.0]
    assert summary.loc[0, "n_vertices"] == 2
    assert summary.loc[0, "resolution_matrix_axis"] == "columns_are_psfs"


def test_run_directional_consistency_writes_band_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths.diagnostics_dir(config).mkdir(parents=True)
    for band in ("alpha", "beta", "gamma"):
        np.save(
            paths.directional_prediction_map_path(config, band=band), np.asarray([1.0, 2.0, 3.0])
        )
        np.save(paths.directional_target_map_path(config, band=band), np.asarray([1.1, 2.1, 3.1]))
        np.save(
            paths.directional_cluster_mask_path(config, band=band), np.asarray([True, True, False])
        )

    required = directional_consistency_required_inputs(_context(config, subjects=()))
    assert paths.directional_prediction_map_path(config, band="alpha") in required

    run_directional_consistency(_context(config, subjects=()))

    summary = pd.read_csv(paths.directional_consistency_summary_path(config), sep="\t")
    assert summary.columns.tolist() == [
        "band",
        "spatial_r",
        "same_sign_fraction",
        "directional_criteria_met",
        "unmet_criteria",
    ]
    assert summary["band"].tolist() == ["alpha", "beta", "gamma"]
    assert summary["directional_criteria_met"].tolist() == [True, True, True]


def test_run_artifact_controls_writes_criteria_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths.diagnostics_dir(config).mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "band": "gamma",
                "metric": "scanner",
                "sensor_template_abs_r": 0.10,
                "source_artifact_map_abs_r": 0.60,
                "expression_p_value": 0.20,
            },
            {
                "band": "gamma",
                "metric": "dvars",
                "sensor_template_abs_r": 0.10,
                "source_artifact_map_abs_r": 0.10,
                "expression_p_value": 0.01,
            },
        ]
    ).to_csv(paths.artifact_metrics_path(config), sep="\t", index=False)

    assert artifact_controls_required_inputs(_context(config, subjects=())) == (
        paths.artifact_metrics_path(config),
    )

    run_artifact_controls(_context(config, subjects=()))

    summary = pd.read_csv(paths.artifact_controls_summary_path(config), sep="\t")
    assert summary.columns.tolist() == [
        "band",
        "artifact_control_criteria_met",
        "unmet_criteria",
        "expression_adjusted_p_values",
        "missing_controls",
    ]
    assert summary.loc[0, "band"] == "gamma"
    assert bool(summary.loc[0, "artifact_control_criteria_met"]) is False
    assert "source_artifact_template" in summary.loc[0, "unmet_criteria"]
    assert "artifact_expression" in summary.loc[0, "unmet_criteria"]


def test_run_robustness_writes_criteria_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths.diagnostics_dir(config).mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "band": "alpha",
                "significance_retained": True,
                "sign_retained": True,
                "unthresholded_spatial_r": 0.70,
                "cluster_dice": 0.60,
                "centroid_displacement_mm": 5.0,
            }
        ]
    ).to_csv(paths.robustness_metrics_path(config), sep="\t", index=False)

    assert robustness_required_inputs(_context(config, subjects=())) == (
        paths.robustness_metrics_path(config),
    )

    run_robustness(_context(config, subjects=()))

    summary = pd.read_csv(paths.robustness_summary_path(config), sep="\t")
    assert summary.columns.tolist() == [
        "band",
        "robustness_criteria_met",
        "unmet_criteria",
    ]
    assert summary.loc[0, "band"] == "alpha"
    assert bool(summary.loc[0, "robustness_criteria_met"]) is True


def test_run_spatial_correspondence_writes_band_results(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["study2"]["spatial_comparison"]["brainsmash_surrogates"] = 2
    paths.spatial_dir(config).mkdir(parents=True)
    np.save(paths.spatial_mask_path(config), np.asarray([True, True, True]))
    paths.spatial_surrogate_metadata_path(config).write_text(
        json.dumps(
            {
                "method": "BrainSMASH Base",
                "n_surrogates": 2,
                "bands": {band: {} for band in ("alpha", "beta", "gamma")},
            }
        ),
        encoding="utf-8",
    )
    for band in ("alpha", "beta", "gamma"):
        np.save(paths.spatial_eeg_map_path(config, band=band), np.asarray([1.0, 2.0, 3.0]))
        np.save(paths.spatial_fmri_map_path(config, band=band), np.asarray([1.0, 2.0, 3.0]))
        np.save(
            paths.spatial_surrogate_maps_path(config, band=band),
            np.asarray([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]], dtype=float),
        )

    required = spatial_correspondence_required_inputs(_context(config, subjects=()))
    assert paths.spatial_mask_path(config) in required
    assert paths.spatial_eeg_map_path(config, band="alpha") in required

    run_spatial_correspondence(_context(config, subjects=()))

    summary = pd.read_csv(paths.spatial_correspondence_summary_path(config), sep="\t")
    assert summary["band"].tolist() == ["alpha", "beta", "gamma"]
    assert summary["meaningful"].tolist() == [True, True, True]
    assert summary["holm_adjusted_p_value"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert summary["holm_significant"].tolist() == [False, False, False]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_spatial_surrogate_stage_inputs(
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mask: np.ndarray,
) -> None:
    ensure_common_source_vertices(
        array_path=paths.source_vertex_manifest_path(config),
        metadata_path=paths.source_vertex_metadata_path(config),
        vertices=(np.asarray([0, 1]), np.asarray([2])),
        common_subject="fsaverage",
        spacing="oct6",
    )
    paths.spatial_dir(config).mkdir(parents=True)
    np.save(paths.spatial_mask_path(config), mask)
    np.save(paths.spatial_distance_matrix_path(config), np.zeros((3, 3), dtype=float))
    for band in ("alpha", "beta", "gamma"):
        np.save(paths.spatial_fmri_map_path(config, band=band), np.arange(3, dtype=float))
    monkeypatch.setattr(
        stages,
        "generate_brainsmash_surrogates",
        lambda **kwargs: np.tile(kwargs["target_map"], (kwargs["n_surrogates"], 1)),
    )


def test_run_spatial_surrogates_records_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config["study2"]["spatial_comparison"]["brainsmash_surrogates"] = 2
    _write_spatial_surrogate_stage_inputs(
        config,
        monkeypatch,
        mask=np.ones(3, dtype=np.bool_),
    )

    run_spatial_surrogates(_context(config, subjects=()))

    metadata = json.loads(paths.spatial_surrogate_metadata_path(config).read_text(encoding="utf-8"))
    bands = ("alpha", "beta", "gamma")
    assert metadata == {
        "schema_version": 1,
        "method": "BrainSMASH Base",
        "n_surrogates": 2,
        "common_subject": "fsaverage",
        "common_source_space_spacing": "oct6",
        "source_vertex_manifest_sha256": _sha256(paths.source_vertex_manifest_path(config)),
        "analysis_mask_sha256": _sha256(paths.spatial_mask_path(config)),
        "bands": {
            band: {
                "hemispheres": {
                    "left": {
                        "seed": 42 + 2 * band_index,
                        "masked_vertices": 2,
                    },
                    "right": {
                        "seed": 43 + 2 * band_index,
                        "masked_vertices": 1,
                    },
                },
                "fmri_map_sha256": _sha256(paths.spatial_fmri_map_path(config, band=band)),
                "surrogate_maps_sha256": _sha256(
                    paths.spatial_surrogate_maps_path(config, band=band)
                ),
            }
            for band_index, band in enumerate(bands)
        },
    }
    required = spatial_surrogates_required_inputs(_context(config, subjects=()))
    assert required == (
        paths.source_vertex_manifest_path(config),
        paths.source_vertex_metadata_path(config),
        paths.spatial_mask_path(config),
        paths.spatial_distance_matrix_path(config),
        *(paths.spatial_fmri_map_path(config, band=band) for band in bands),
    )


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_run_spatial_surrogates_rejects_nonboolean_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
) -> None:
    config = _config(tmp_path)
    config["study2"]["spatial_comparison"]["brainsmash_surrogates"] = 2
    _write_spatial_surrogate_stage_inputs(
        config,
        monkeypatch,
        mask=np.ones(3, dtype=dtype),
    )

    with pytest.raises(
        ValueError,
        match=r"Study 2 spatial analysis mask must have boolean dtype\.",
    ):
        run_spatial_surrogates(_context(config, subjects=()))


def test_run_spatial_surrogates_rejects_complex_fmri_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config["study2"]["spatial_comparison"]["brainsmash_surrogates"] = 2
    _write_spatial_surrogate_stage_inputs(
        config,
        monkeypatch,
        mask=np.ones(3, dtype=np.bool_),
    )
    fmri_path = paths.spatial_fmri_map_path(config, band="alpha")
    fmri_map = np.arange(3, dtype=np.complex128)
    fmri_map[0] = 1.0j
    np.save(fmri_path, fmri_map)

    with pytest.raises(
        ValueError,
        match=r"Study 2 spatial fMRI map must be real-valued: alpha\.",
    ):
        run_spatial_surrogates(_context(config, subjects=()))


def test_run_behavioral_convergence_writes_summary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["study2"]["behavioral_convergence"]["expression_column"] = "expression"
    config["study2"]["behavioral_convergence"]["rating_column"] = "rating"
    config["study2"]["behavioral_convergence"]["design_columns"] = ["nuisance"]
    config["study2"]["behavioral_convergence"]["min_rated_trials"] = 8
    paths.behavioral_dir(config).mkdir(parents=True)
    rows = []
    for subject_id in ("sub-0001", "sub-0002"):
        for trial in range(10):
            rows.append(
                {
                    "subject_id": subject_id,
                    "run": (trial % 3) + 1,
                    "trial_id": trial,
                    "expression": float(trial),
                    "rating": float(trial) + 0.1,
                    "nuisance": float(trial % 2),
                }
            )
    pd.DataFrame(rows).to_csv(
        paths.behavioral_convergence_input_path(config), sep="\t", index=False
    )

    assert behavioral_convergence_required_inputs(_context(config, subjects=())) == (
        paths.behavioral_convergence_input_path(config),
    )

    run_behavioral_convergence(_context(config, subjects=()))

    summary = pd.read_csv(paths.behavioral_convergence_summary_path(config), sep="\t")
    assert summary.loc[0, "n_subjects"] == 2
    assert summary.loc[0, "mean_beta"] > 0.0


def test_required_config_string_tuple_rejects_empty_column_names() -> None:
    config = load_study2_config()
    config["study2"]["behavioral_convergence"]["design_columns"] = ["onset", ""]

    with pytest.raises(ValueError, match="empty column names"):
        stages._required_config_string_tuple(
            config,
            "study2.behavioral_convergence.design_columns",
        )


def test_run_band_unique_stage_and_inference_write_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    frame = _cohort_source_stage_frame()
    paths.source_stage_dir(config).mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.source_stage_frame_path(config), sep="\t", index=False)

    subjects = ("sub-0001", "sub-0002", "sub-0003")
    rng = np.random.default_rng(123)
    source_power = {
        str(subject_id): rng.normal(size=(len(subject_frame), 3))
        for subject_id, subject_frame in frame.groupby("subject_id", sort=True)
    }
    for band in ("alpha", "beta", "gamma"):
        for subject_id in subjects:
            power_path = paths.subject_source_power_path(config, subject_id=subject_id, band=band)
            power_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(power_path, source_power[subject_id])

    required = band_unique_stage_required_inputs(_context(config, subjects=subjects))
    assert paths.source_stage_frame_path(config) in required

    run_band_unique_stage(_context(config, subjects=subjects))

    paths.inference_dir(config).mkdir(parents=True, exist_ok=True)
    np.save(paths.source_adjacency_path(config), _chain_adjacency(3))
    for band in ("alpha", "beta", "gamma"):
        maps = np.load(paths.band_unique_fisher_z_path(config, band=band))
        assert maps.shape == (2, 3)
        np.save(
            paths.band_unique_null_maps_path(config, band=band), np.zeros((9, 2, 3), dtype=float)
        )

    required = band_unique_inference_required_inputs(_context(config, subjects=()))
    assert paths.source_adjacency_path(config) in required

    run_band_unique_inference(_context(config, subjects=()))

    summary = pd.read_csv(paths.band_unique_family_summary_path(config), sep="\t")
    assert summary["band"].tolist() == ["alpha", "beta", "gamma"]
