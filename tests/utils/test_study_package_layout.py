from __future__ import annotations

import tomllib

from tests import REPO_ROOT


EXPECTED_INIT_FILES = (
    "studies/__init__.py",
    "studies/pain_study/__init__.py",
    "studies/pain_study/analysis/__init__.py",
    "studies/pain_study/cli/__init__.py",
    "studies/pain_study/config/__init__.py",
    "studies/pain_study/pipelines/__init__.py",
    "studies/pain_study/scripts/__init__.py",
)

def test_pain_study_package_layout_and_metadata() -> None:
    for rel_path in EXPECTED_INIT_FILES:
        assert (REPO_ROOT / rel_path).is_file(), rel_path

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = config["tool"]["setuptools"]
    package_find = setuptools["packages"]["find"]
    package_data = setuptools["package-data"]

    assert "studies*" in package_find["include"]
    assert package_data["studies.pain_study"] == [
        "analysis/*.R",
        "config/*.yaml",
        "config/**/*.json",
        "config/**/*.label",
        "scripts/config/*.yaml",
    ]
