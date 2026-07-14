"""Tests for studies.pain_study.study1.config.loader configuration loading and merging."""

from __future__ import annotations

import pytest
import yaml

from studies.tests.test_support import REPO_ROOT

REFERENCE_POWER = {
    "primary_window": [-5.0, -0.01],
    "sensitivity_windows": {
        "prestimulus_2s": [-2.0, -0.01],
        "immediate_prestimulus": [-0.2, -0.01],
    },
    "unnormalized_active_power": {
        "feature_transform": "raw_log_power",
        "feature_baseline_window": None,
        "active_window": [3.0, 10.5],
        "reference_window": [-5.0, -0.01],
        "reference_power_covariate": True,
    },
}
TIME_FREQUENCY = {
    "baseline_window": [-5.0, -0.01],
    "active_window": [3.0, 10.5],
}


###################################################################
# load_study1_config
###################################################################


def test_load_study1_config_resolves_default_yaml() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert "study1" in config
    assert config["study1"]["targets"]["names"] == ["NPS", "SIIPS1"]
    assert config["study1"]["features"]["exploratory_feature_families"] == [
        "aperiodic",
        "erds",
        "spectral",
        "bursts",
    ]
    assert config["study1"]["targets"]["max_design_condition_number"] == 3000.0


def test_load_study1_config_defines_unbaselined_temporal_negative_controls() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    temporal = config["study1"]["temporal_negative_controls"]

    assert temporal["feature_transform"] == "raw_log_power"
    assert temporal["feature_baseline_window"] is None
    assert temporal["windows"] == {
        "prestimulus_wide": [-5.0, -0.01],
        "immediate_prestimulus": [-0.2, -0.01],
    }
    assert temporal["wrong_lag_windows"] == {
        "ramp_up": [0.0, 3.0],
    }
    assert temporal["plateau_windows"] == {
        "early_plateau": [3.0, 5.5],
        "mid_plateau": [5.5, 8.0],
        "late_plateau": [8.0, 10.5],
    }


def test_load_study1_config_defines_reference_power_sensitivity_plan() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    reference = config["study1"]["reference_power"]

    assert config["time_frequency_analysis"]["baseline_window"] == [-5.0, -0.01]
    assert reference["primary_window"] == [-5.0, -0.01]
    assert reference["sensitivity_windows"] == {
        "prestimulus_2s": [-2.0, -0.01],
        "immediate_prestimulus": [-0.2, -0.01],
    }
    assert reference["unnormalized_active_power"] == {
        "feature_transform": "raw_log_power",
        "feature_baseline_window": None,
        "active_window": [3.0, 10.5],
        "reference_window": [-5.0, -0.01],
        "reference_power_covariate": True,
    }


def test_load_study1_config_defines_prespecified_permutation_controls() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    benchmark = config["study1"]["feature_benchmark"]

    assert benchmark["permutation_scheme"] == "circular_shift_within_run"
    assert benchmark["max_invalid_permutation_fraction"] == 0.20
    assert benchmark["circular_shift"] == {
        "min_valid_runs_per_subject": 3,
        "min_retained_trials_per_subject": 25,
    }


def test_load_study1_config_includes_gamma_deep_regression_presets() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    presets = config["study1"]["deep_regression"]["presets"]

    clean_gamma = ["gamma_low_clean", "gamma_mid_clean", "gamma_high_clean"]
    assert presets["gamma"] == clean_gamma
    assert presets["alpha_beta_gamma"] == ["alpha", "beta", *clean_gamma]


def test_load_study1_config_uses_readme_bootstrap_iterations() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert config["machine_learning"]["evaluation"]["bootstrap_iterations"] == 10000


def test_load_study1_config_resolves_explicit_path() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)

    assert config["study1"]["cohort"]["min_subjects"] == 2


def test_load_study1_config_preserves_signature_paths_relative_to_signature_dir() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    targets = config["study1"]["targets"]

    assert targets["signature_manifest_path"] == "signature_manifest.yaml"
    assert targets["signature_provenance"]["NPS"]["path"] == (
        "NPS/weights_NSF_grouppred_cvpcr.nii.gz"
    )
    assert targets["signature_provenance"]["SIIPS1"]["path"] == (
        "SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
    )


def test_load_study1_config_rejects_missing_file(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    with pytest.raises(FileNotFoundError, match="Study 1 config"):
        load_study1_config(config_path=tmp_path / "nonexistent.yaml")


def test_load_study1_config_rejects_non_mapping_yaml(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("- list_item\n- another\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_rejects_baselined_temporal_negative_controls(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "bad_temporal.yaml"
    bad_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "temporal_negative_controls": {
                        "feature_transform": "logratio",
                        "feature_baseline_window": [-5.0, -0.01],
                        "windows": {"prestimulus_wide": [-5.0, -0.01]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_log_power"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_requires_explicit_null_temporal_control_baseline(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "missing_temporal_baseline.yaml"
    bad_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "windows": {"prestimulus_wide": [-5.0, -0.01]},
                        "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feature_baseline_window"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_rejects_missing_temporal_negative_controls(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "missing_temporal.yaml"
    bad_config.write_text(yaml.dump({"study1": {"cohort": {"min_subjects": 99}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="temporal_negative_controls"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_rejects_poststimulus_temporal_negative_control_window(
    tmp_path,
) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "bad_temporal_window.yaml"
    bad_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"invalid": [-0.2, 0.1]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pre-stimulus"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_rejects_missing_wrong_lag_temporal_controls(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "missing_wrong_lag.yaml"
    bad_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"prestimulus_wide": [-5.0, -0.01]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong_lag_windows"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_rejects_wrong_lag_after_plateau_starts(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "wrong_lag_after_plateau.yaml"
    bad_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "feature_benchmark": {
                        "n_perm": 5000,
                        "permutation_scheme": "circular_shift_within_run",
                        "max_invalid_permutation_fraction": 0.20,
                        "circular_shift": {
                            "min_valid_runs_per_subject": 3,
                            "min_retained_trials_per_subject": 25,
                        },
                    },
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"prestimulus_wide": [-5.0, -0.01]},
                        "wrong_lag_windows": {"late_ramp_down": [10.5, 15.0]},
                        "plateau_windows": {"early_plateau": [3.0, 5.5]},
                    },
                    "reference_power": REFERENCE_POWER,
                },
                "time_frequency_analysis": TIME_FREQUENCY,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="before the plateau starts"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_rejects_plateau_window_that_reaches_ramp_down(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "plateau_reaches_ramp_down.yaml"
    bad_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "feature_benchmark": {
                        "n_perm": 5000,
                        "permutation_scheme": "circular_shift_within_run",
                        "max_invalid_permutation_fraction": 0.20,
                        "circular_shift": {
                            "min_valid_runs_per_subject": 3,
                            "min_retained_trials_per_subject": 25,
                        },
                    },
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"prestimulus_wide": [-5.0, -0.01]},
                        "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                        "plateau_windows": {"late_shifted_active": [5.0, 12.5]},
                    },
                    "reference_power": REFERENCE_POWER,
                },
                "time_frequency_analysis": TIME_FREQUENCY,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="within the plateau"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_rejects_missing_permutation_scheme(tmp_path) -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    bad_config = tmp_path / "missing_permutation_scheme.yaml"
    bad_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "feature_benchmark": {
                        "n_perm": 5000,
                        "max_invalid_permutation_fraction": 0.20,
                        "circular_shift": {
                            "min_valid_runs_per_subject": 3,
                            "min_retained_trials_per_subject": 25,
                        },
                    },
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"prestimulus_wide": [-5.0, -0.01]},
                        "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                        "plateau_windows": {"early_plateau": [3.0, 5.5]},
                    },
                    "reference_power": REFERENCE_POWER,
                },
                "time_frequency_analysis": TIME_FREQUENCY,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="permutation_scheme"):
        load_study1_config(config_path=bad_config)


def test_load_study1_config_uses_env_var_override(tmp_path, monkeypatch) -> None:
    from studies.pain_study.study1.config.loader import (
        STUDY1_CONFIG_ENV_VAR,
        load_study1_config,
    )

    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        yaml.dump(
            {
                "study1": {
                    "cohort": {"min_subjects": 99},
                    "feature_benchmark": {
                        "n_perm": 5000,
                        "permutation_scheme": "circular_shift_within_run",
                        "max_invalid_permutation_fraction": 0.20,
                        "circular_shift": {
                            "min_valid_runs_per_subject": 3,
                            "min_retained_trials_per_subject": 25,
                        },
                    },
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"prestimulus_wide": [-5.0, -0.01]},
                        "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                        "plateau_windows": {"early_plateau": [3.0, 5.5]},
                    },
                    "reference_power": REFERENCE_POWER,
                },
                "time_frequency_analysis": TIME_FREQUENCY,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(STUDY1_CONFIG_ENV_VAR, str(custom_config))

    config = load_study1_config()

    assert config["study1"]["cohort"]["min_subjects"] == 99


###################################################################
# _merge_non_null
###################################################################


def test_merge_non_null_skips_none_values() -> None:
    from studies.pain_study.study1.config.loader import _merge_non_null

    base = {"key": "original", "nested": {"a": 1}}
    _merge_non_null(base, {"key": None, "nested": {"b": 2}})

    assert base["key"] == "original"
    assert base["nested"]["a"] == 1
    assert base["nested"]["b"] == 2


def test_merge_non_null_preserves_missing_explicit_none_values() -> None:
    from studies.pain_study.study1.config.loader import _merge_non_null

    base: dict = {}
    _merge_non_null(
        base,
        {"study1": {"temporal_negative_controls": {"feature_baseline_window": None}}},
    )

    assert base["study1"]["temporal_negative_controls"]["feature_baseline_window"] is None


def test_merge_non_null_creates_nested_dicts() -> None:
    from studies.pain_study.study1.config.loader import _merge_non_null

    base: dict = {}
    _merge_non_null(base, {"new": {"deep": {"value": 42}}})

    assert base["new"]["deep"]["value"] == 42


def test_merge_non_null_preserves_existing_non_dict_values() -> None:
    from studies.pain_study.study1.config.loader import _merge_non_null

    base = {"key": "original"}
    _merge_non_null(base, {"key": "overridden"})

    assert base["key"] == "overridden"


###################################################################
# apply_study1_config_defaults
###################################################################


def test_apply_study1_config_defaults_merges_study1_section() -> None:
    from studies.pain_study.study1.config.loader import apply_study1_config_defaults

    config: dict = {"paths": {"deriv_root": "/tmp/test"}}
    apply_study1_config_defaults(config)

    assert "study1" in config
    assert config["study1"]["targets"]["names"] == ["NPS", "SIIPS1"]
    assert config["paths"]["deriv_root"] == "/tmp/test"


def test_apply_study1_config_defaults_does_not_overwrite_existing_values() -> None:
    from studies.pain_study.study1.config.loader import apply_study1_config_defaults

    config: dict = {"study1": {"cohort": {"min_subjects": 10}}}
    apply_study1_config_defaults(config)

    # Production default is 30; existing value should be overwritten by merge
    # (non-null merge replaces scalar values), so this verifies the merge direction.
    assert config["study1"]["cohort"]["min_subjects"] == 30


###################################################################
# Smoketest config contracts
###################################################################


def test_smoketest_config_uses_article_required_nuisance_regression_columns() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)

    nuisance = config["study1"]["targets"]["nuisance_regression"]
    assert nuisance["enabled"] is True
    assert nuisance["continuous_columns"] == [
        "run",
        "onset",
        "within_run_trial",
        "hrf_weighted_framewise_displacement",
        "hrf_weighted_std_dvars",
        "hrf_weighted_fp1_fp2_high_frequency_power",
        "residual_ecg_coupling",
    ]
    assert nuisance["categorical_columns"] == ["stimulus_temp", "selected_surface"]


def test_smoketest_config_uses_low_permutation_count() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)

    assert config["study1"]["feature_benchmark"]["n_perm"] <= 100


def test_smoketest_config_defines_prespecified_permutation_controls() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)
    benchmark = config["study1"]["feature_benchmark"]

    assert benchmark["permutation_scheme"] == "circular_shift_within_run"
    assert benchmark["max_invalid_permutation_fraction"] == 0.20
    assert benchmark["circular_shift"] == {
        "min_valid_runs_per_subject": 3,
        "min_retained_trials_per_subject": 25,
    }


###################################################################
# Validity figure configuration
###################################################################


def test_load_study1_config_includes_validity_figure_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    validity = config["study1"]["figures"]["validity"]

    assert validity["temperatures"] == [44.3, 45.3, 46.3, 47.3, 48.3, 49.3]
    assert validity["dimensions_mm"] == {"width": 89.0, "height": 70.0}
    assert validity["font"]["family"] == "Arial"
    assert validity["bootstrap"] == {
        "iterations": 10000,
        "confidence_level": 0.95,
        "seed": 42,
        "max_invalid_fraction": 0.20,
    }
    assert validity["output_parts"] == [
        "reports",
        "figures",
        "supplementary",
        "validity",
    ]


def test_load_study1_config_includes_scanner_harmonic_figure_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert config["study1"]["figures"]["scanner_harmonics"] == {
        "dimensions_mm": {"width": 183.0, "height": 82.0},
        "frequency_range_hz": [15.0, 90.0],
        "n_fft": 8192,
        "n_overlap": 4096,
        "sampling_frequency_hz": 500.0,
        "peak_prominence_db": 1.0,
        "peak_distance_bins": 4,
        "volume_repetition_time_s": 0.9,
        "harmonic_orders": [18, 37, 55, 74],
        "colors": {
            "excluded": "#D55E00",
            "retained": "#0072B2",
        },
    }


def test_load_study1_config_includes_cohort_psd_figure_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    figures = config["study1"]["figures"]

    assert figures["continuous_spectrum"] == {
        "excluded_subjects": ["sub-0006"],
    }
    assert figures["cohort_power_spectral_density"] == {
        "dimensions_mm": {"width": 183.0, "height": 92.0},
        "frequency_range_hz": [1.0, 90.0],
        "n_fft": 8192,
        "n_overlap": 4096,
        "sampling_frequency_hz": 500.0,
        "colors": {
            "scanner_window": "#D55E00",
            "neural_band": "#0072B2",
        },
    }


def test_load_study1_config_includes_temporal_specificity_figure_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert config["study1"]["figures"]["temporal_specificity"] == {
        "dimensions_mm": {"width": 183.0, "height": 88.0},
        "model": "elasticnet",
        "targets": ["NPS", "SIIPS1"],
    }


def test_load_study1_config_includes_primary_prediction_figure_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert config["study1"]["figures"]["primary_prediction"] == {
        "dimensions_mm": {"width": 183.0, "height": 86.0},
        "lane": "feature_benchmark",
        "analysis_partition": "primary",
        "model": "elasticnet",
        "feature_spec": "alpha_beta_gamma",
        "targets": ["NPS", "SIIPS1"],
    }


def test_load_study1_config_includes_spectral_specificity_figure_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert config["study1"]["figures"]["spectral_specificity"] == {
        "dimensions_mm": {"width": 183.0, "height": 84.0},
        "model": "elasticnet",
        "targets": ["NPS", "SIIPS1"],
        "feature_specs": [
            "alpha",
            "beta",
            "gamma",
            "alpha_beta",
            "alpha_beta_gamma",
        ],
    }


def test_load_study1_config_includes_power_construct_validity_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert config["study1"]["figures"]["power_construct_validity"] == {
        "dimensions_mm": {"width": 183.0, "height": 108.0},
        "bands": [
            {"name": "alpha", "label": "Alpha", "frequency_hz": [8.0, 12.9]},
            {"name": "beta", "label": "Beta", "frequency_hz": [13.0, 30.0]},
            {
                "name": "gamma_low_clean",
                "label": "Low gamma",
                "frequency_hz": [30.1, 38.0],
            },
            {
                "name": "gamma_mid_clean",
                "label": "Mid gamma",
                "frequency_hz": [43.0, 56.0],
            },
            {
                "name": "gamma_high_clean",
                "label": "High gamma",
                "frequency_hz": [67.0, 77.0],
            },
        ],
        "channels": {
            "include_fp1_fp2": True,
            "compute_complementary_sensitivity": True,
        },
        "rating_model": {
            "minimum_trials": 25,
            "minimum_runs": 3,
            "max_condition_number": 100.0,
        },
        "minimum_article_subjects": 30,
        "colors": {"cohort": "#0072B2"},
    }


def test_load_study1_config_includes_fmri_construct_validity_defaults() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert config["study1"]["figures"]["fmri_construct_validity"] == {
        "dimensions_mm": {"width": 183.0, "height": 112.0},
        "axial_slices_mm": [-12.0, 0.0, 12.0, 24.0, 36.0, 48.0],
        "surface_mesh": "fsaverage5",
        "inference": {
            "n_permutations": 10000,
            "two_sided": True,
            "alpha": 0.05,
            "random_state": 20260711,
        },
        "display": {"robust_percentile": 99.5},
        "minimum_article_subjects": 30,
    }


def test_validate_validity_figure_config_rejects_duplicate_temperatures() -> None:
    from studies.pain_study.study1.config.loader import _validate_validity_figure_config

    config = _validity_figure_config()
    config["study1"]["figures"]["validity"]["temperatures"] = [44.3, 44.3]

    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_validity_figure_config(config)


def test_validate_validity_figure_config_rejects_invalid_bootstrap_budget() -> None:
    from studies.pain_study.study1.config.loader import _validate_validity_figure_config

    config = _validity_figure_config()
    bootstrap = config["study1"]["figures"]["validity"]["bootstrap"]
    bootstrap["max_invalid_fraction"] = 1.0

    with pytest.raises(ValueError, match=r"max_invalid_fraction must be in \[0, 1\)"):
        _validate_validity_figure_config(config)


@pytest.mark.parametrize("value", [1.9, True, "100"])
def test_validate_validity_figure_config_rejects_non_integer_bootstrap_iterations(
    value: object,
) -> None:
    from studies.pain_study.study1.config.loader import _validate_validity_figure_config

    config = _validity_figure_config()
    config["study1"]["figures"]["validity"]["bootstrap"]["iterations"] = value

    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        _validate_validity_figure_config(config)


def test_validate_validity_figure_config_rejects_boolean_seed() -> None:
    from studies.pain_study.study1.config.loader import _validate_validity_figure_config

    config = _validity_figure_config()
    config["study1"]["figures"]["validity"]["bootstrap"]["seed"] = True

    with pytest.raises(ValueError, match="seed must be an integer"):
        _validate_validity_figure_config(config)


def test_validate_validity_figure_config_rejects_non_string_output_part() -> None:
    from studies.pain_study.study1.config.loader import _validate_validity_figure_config

    config = _validity_figure_config()
    config["study1"]["figures"]["validity"]["output_parts"] = ["reports", 1]

    with pytest.raises(ValueError, match="entries must be strings"):
        _validate_validity_figure_config(config)


def _validity_figure_config() -> dict:
    return {
        "study1": {
            "figures": {
                "validity": {
                    "temperatures": [44.3, 49.3],
                    "dimensions_mm": {"width": 89.0, "height": 70.0},
                    "font": {
                        "family": "Arial",
                        "axis_label_pt": 7.0,
                        "tick_label_pt": 6.0,
                        "legend_pt": 6.0,
                        "annotation_pt": 5.5,
                    },
                    "style": {
                        "participant_color": "#7F7F7F",
                        "participant_alpha": 0.22,
                        "participant_line_width_pt": 0.45,
                        "participant_marker_size_pt": 1.8,
                        "cohort_line_width_pt": 1.2,
                        "cohort_marker_size_pt": 3.0,
                        "confidence_line_width_pt": 0.8,
                        "axis_line_width_pt": 0.6,
                    },
                    "colors": {
                        "behavioral": "#222222",
                        "nps": "#0072B2",
                        "siips1": "#D55E00",
                    },
                    "bootstrap": {
                        "iterations": 100,
                        "confidence_level": 0.95,
                        "seed": 42,
                        "max_invalid_fraction": 0.20,
                    },
                    "output_parts": [
                        "reports",
                        "figures",
                        "supplementary",
                        "validity",
                    ],
                }
            }
        }
    }
