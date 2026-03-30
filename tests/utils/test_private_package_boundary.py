from __future__ import annotations

import tomllib

from tests import REPO_ROOT


def test_public_package_does_not_include_private_studies() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_find = config["tool"]["setuptools"]["packages"]["find"]
    package_data = config["tool"]["setuptools"]["package-data"]

    assert "studies*" not in package_find["include"]
    assert "studies.pain_study" not in package_data
    assert "studies.pain_study.study1" not in package_data
    assert "studies.pain_study.study2" not in package_data


def test_gitignore_keeps_private_studies_out_of_public_repo() -> None:
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    ignored_entries = {line.strip() for line in gitignore_text.splitlines() if line.strip()}

    assert "studies/" in ignored_entries
