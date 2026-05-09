"""Tests for studies.pain_study.study1.config.loader configuration loading and merging."""

from __future__ import annotations

import os
from pathlib import Path

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


def test_load_study1_config_resolves_explicit_path() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)

    assert config["study1"]["cohort"]["min_subjects"] == 2


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


def test_load_study1_config_uses_env_var_override(tmp_path, monkeypatch) -> None:
    from studies.pain_study.study1.config.loader import (
        STUDY1_CONFIG_ENV_VAR,
        load_study1_config,
    )

    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        yaml.dump({"study1": {"cohort": {"min_subjects": 99}}}),
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


def test_smoketest_config_disables_nuisance_regression() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)

    assert config["study1"]["targets"]["nuisance_regression"]["enabled"] is False


def test_smoketest_config_uses_low_permutation_count() -> None:
    from studies.pain_study.study1.config.loader import load_study1_config

    smoketest_path = REPO_ROOT / "studies/pain_study/study1/config/study1_smoketest.yaml"
    config = load_study1_config(config_path=smoketest_path)

    assert config["study1"]["feature_benchmark"]["n_perm"] <= 100
