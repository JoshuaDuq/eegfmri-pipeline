from __future__ import annotations

import tomllib

from studies.tests.test_support import REPO_ROOT


EXPECTED_INIT_FILES = (
    "studies/__init__.py",
    "studies/pain_study/__init__.py",
    "studies/pain_study/analysis/__init__.py",
    "studies/pain_study/cli/__init__.py",
    "studies/pain_study/config/__init__.py",
    "studies/pain_study/pipelines/__init__.py",
    "studies/pain_study/scripts/__init__.py",
    "studies/pain_study/study1/__init__.py",
    "studies/pain_study/study1/config/__init__.py",
    "studies/pain_study/study1/deep_regression/__init__.py",
    "studies/pain_study/study2/__init__.py",
    "studies/pain_study/study2/config/__init__.py",
    "studies/pain_study/eeg_coupling/__init__.py",
    "studies/pain_study/eeg_coupling/analysis/__init__.py",
    "studies/pain_study/eeg_coupling/cli/__init__.py",
    "studies/pain_study/eeg_coupling/config/__init__.py",
    "studies/pain_study/eeg_coupling/pipelines/__init__.py",
)


def test_pain_study_package_layout_and_metadata() -> None:
    for rel_path in EXPECTED_INIT_FILES:
        assert (REPO_ROOT / rel_path).is_file(), rel_path

    config = tomllib.loads((REPO_ROOT / "studies" / "private_package.toml").read_text(encoding="utf-8"))
    package_find = config["tool"]["setuptools"]["packages"]["find"]
    package_data = config["tool"]["setuptools"]["package-data"]
    entry_points = config["project"]["entry-points"]["eeg_pipeline.cli_commands"]

    assert "studies*" in package_find["include"]
    assert entry_points["coupling"] == "studies.pain_study.cli.command_registry:coupling_command"
    assert (
        entry_points["signature-prediction"]
        == "studies.pain_study.cli.command_registry:signature_prediction_command"
    )
    assert package_data["studies.pain_study"] == ["scripts/config/*.yaml"]
    assert package_data["studies.pain_study.study1"] == [
        "config/*.yaml",
        "README.md",
    ]
    assert package_data["studies.pain_study.study2"] == [
        "config/*.yaml",
        "README.md",
    ]
    assert package_data["studies.pain_study.eeg_coupling"] == [
        "analysis/*.R",
        "config/*.yaml",
        "config/**/*.json",
        "config/**/*.label",
        "README.md",
    ]
