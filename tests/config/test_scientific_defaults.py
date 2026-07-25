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


def test_fmri_config_does_not_duplicate_source_constraint_defaults() -> None:
    config = _load_yaml("fmri_pipeline/utils/config/fmri_config.yaml")

    assert "fmri_constraint" not in config


def test_ml_covariates_are_strict_by_default() -> None:
    config = _load_yaml("eeg_pipeline/utils/config/eeg_config.yaml")

    assert config["machine_learning"]["data"]["covariates_strict"] is True
