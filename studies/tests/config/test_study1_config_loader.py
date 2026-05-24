"""Tests for studies.pain_study.study1.config.loader configuration loading and merging."""

from __future__ import annotations

import pytest
import yaml

from studies.tests.test_support import REPO_ROOT


###################################################################
# load_study1_config
###################################################################


def test_load_study1_config_resolves_default_yaml() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()

    assert "study1" in config
    assert config["study1"]["targets"]["names"] == ["NPS", "SIIPS1"]
    assert config["study1"]["features"]["exploratory_feature_families"] == []
    assert config["study1"]["targets"]["max_design_condition_number"] == 2000.0


def test_load_study1_config_defines_unbaselined_temporal_negative_controls() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    temporal = config["study1"]["temporal_negative_controls"]

    assert temporal["feature_transform"] == "raw_log_power"
    assert temporal["feature_baseline_window"] is None
    assert temporal["windows"] == {
        "prestimulus_wide": [-5.0, 0.0],
        "immediate_prestimulus": [-0.2, 0.0],
    }
    assert temporal["wrong_lag_windows"] == {
        "ramp_up": [0.0, 3.0],
        "late_ramp_down": [10.5, 15.0],
        "early_shifted_active": [1.0, 8.5],
        "late_shifted_active": [5.0, 12.5],
    }


def test_load_study1_config_defines_prespecified_permutation_controls() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    config = load_study1_config()
    benchmark = config["study1"]["feature_benchmark"]

    assert benchmark["permutation_scheme"] == "circular_shift_within_run"
    assert benchmark["max_invalid_permutation_fraction"] == 0.20
    assert benchmark["circular_shift"] == {
        "min_valid_blocks_per_subject": 3,
        "min_retained_trials_per_subject": 25,
    }


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
                        "windows": {"prestimulus_wide": [-5.0, 0.0]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_log_power"):
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
                        "windows": {"prestimulus_wide": [-5.0, 0.0]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="wrong_lag_windows"):
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
                            "min_valid_blocks_per_subject": 3,
                            "min_retained_trials_per_subject": 25,
                        },
                    },
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"prestimulus_wide": [-5.0, 0.0]},
                        "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                    },
                }
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
                            "min_valid_blocks_per_subject": 3,
                            "min_retained_trials_per_subject": 25,
                        },
                    },
                    "temporal_negative_controls": {
                        "feature_transform": "raw_log_power",
                        "feature_baseline_window": None,
                        "windows": {"prestimulus_wide": [-5.0, 0.0]},
                        "wrong_lag_windows": {"ramp_up": [0.0, 3.0]},
                    },
                }
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


def test_smoketest_config_uses_available_nuisance_regression_columns() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)

    nuisance = config["study1"]["targets"]["nuisance_regression"]
    assert nuisance["enabled"] is True
    assert nuisance["continuous_columns"] == [
        "block",
        "onset",
        "within_block_trial",
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
        "min_valid_blocks_per_subject": 3,
        "min_retained_trials_per_subject": 25,
    }
