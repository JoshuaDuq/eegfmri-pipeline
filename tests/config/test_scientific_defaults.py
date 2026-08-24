from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(relative_path: str) -> dict:
    with open(REPO_ROOT / relative_path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def test_pac_default_uses_surrogate_null_model() -> None:
    config = _load_yaml("eeg_pipeline/utils/config/eeg_config.yaml")

    pac_config = config["feature_engineering"]["pac"]

    assert pac_config["n_surrogates"] >= 200


def test_decomb_manifest_is_disabled_by_default() -> None:
    config = _load_yaml("eeg_pipeline/utils/config/eeg_config.yaml")

    assert config["paths"]["decomb_manifest"] is None


def test_fmri_config_does_not_duplicate_source_constraint_defaults() -> None:
    config = _load_yaml("fmri_pipeline/utils/config/fmri_config.yaml")

    assert "fmri_constraint" not in config


def test_fmri_preprocessing_defaults_pin_a_reproducible_lts_command() -> None:
    config = _load_yaml("fmri_pipeline/utils/config/fmri_config.yaml")
    fmriprep = config["fmri_preprocessing"]["fmriprep"]

    assert fmriprep["image"] == "nipreps/fmriprep:25.2.5"
    assert fmriprep["omp_nthreads"] == 1
    assert fmriprep["skull_strip_fixed_seed"] is True
    assert fmriprep["random_seed"] == 42
    assert fmriprep["bold2anat_init"] == "t1w"
    assert fmriprep["output_layout"] == "bids"


def test_fmri_nuisance_defaults_are_fixed_not_derivative_adaptive() -> None:
    config = _load_yaml("fmri_pipeline/utils/config/fmri_config.yaml")

    assert config["fmri_contrast"]["confounds_strategy"] != "auto"
    assert config["fmri_resting_state"]["confounds_strategy"] != "auto"


def test_fmri_report_settings_are_explicit_in_yaml() -> None:
    config = _load_yaml("fmri_pipeline/utils/config/fmri_config.yaml")
    report = config["fmri_report"]

    assert report["enabled"] is True
    assert report["html_report"] is True
    assert report["include_carpet_qc"] is True
    assert report["include_design_qc"] is True


def test_fmri_statistical_artifacts_are_explicit_and_not_plot_triggered() -> None:
    config = _load_yaml("fmri_pipeline/utils/config/fmri_config.yaml")
    stats = config["fmri_stats"]

    assert stats["space"] == "native"
    assert stats["include_effect_size"] is True
    assert stats["include_standard_error"] is True
    assert stats["threshold_mode"] == "z"
    assert stats["z_threshold"] > 0


def test_fmri_cohort_report_and_threshold_are_explicit_in_yaml() -> None:
    config = _load_yaml("fmri_pipeline/utils/config/fmri_config.yaml")
    group = config["fmri_group_level"]

    assert group["report"]["enabled"] is True
    assert group["report"]["html_report"] is True
    assert group["report"]["embed_images"] is True
    assert group["report"]["include_design_correlation"] is True
    assert group["threshold"]["height_control"] == "fdr"
    assert group["threshold"]["alpha"] == 0.05
    assert group["threshold"]["cluster_min_voxels"] == 0
    assert group["threshold"]["two_sided"] is True
    assert group["permutation"]["random_state"] == 42


def test_ml_covariates_are_strict_by_default() -> None:
    config = _load_yaml("eeg_pipeline/utils/config/eeg_config.yaml")

    assert config["machine_learning"]["data"]["covariates_strict"] is True
